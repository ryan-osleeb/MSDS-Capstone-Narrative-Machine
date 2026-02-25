"""
Narrative Pipeline Extensions
==============================
Three new capabilities that integrate with the existing NarrativePipeline:

1. BERTopic Integration
   - Automatic topic discovery with built-in labels
   - Topics over time (dynamic topic modeling)
   - Topic hierarchy and merging

2. Sentiment / Stance Layer
   - Per-article sentiment scoring
   - Narrative stance tracking over time
   - Stance drift detection

3. Spike Detection
   - Z-score based narrative spike detection
   - Auto-flagged events with top articles
   - Annotated prevalence plots

Usage:
    from narrative_pipeline import NarrativePipeline
    from narrative_config import AITECH_CONFIG
    from narrative_extensions import NarrativeExtensions

    pipeline = NarrativePipeline(AITECH_CONFIG)
    pipeline.load_data('unified_tech.csv')
    pipeline.run_full_analysis(cache_file='tech_embeddings.pkl')

    ext = NarrativeExtensions(pipeline)
    ext.run_all()   # runs all three extensions
    ext.generate_extension_report('output/')

Dependencies:
    pip install bertopic python-louvain textblob hdbscan --break-system-packages
    python -m textblob.download_corpora
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import pickle
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings('ignore')


# =============================================================================
# EXTENSION CLASS
# =============================================================================

class NarrativeExtensions:
    """
    Extensions for the NarrativePipeline: BERTopic, Sentiment, Spike Detection.

    Takes an existing NarrativePipeline object (already analyzed) and adds
    new layers of analysis. Stores results back into pipeline.results so
    everything stays integrated.

    Parameters
    ----------
    pipeline : NarrativePipeline
        Must have already run load_data() and compute_embeddings() at minimum.
    sentiment_method : str
        'textblob' (fast, no GPU) or 'transformer' (more accurate, needs GPU/CPU time)
    """

    def __init__(self, pipeline, sentiment_method: str = 'textblob'):
        self.pipeline = pipeline
        self.df = pipeline.results.df
        self.embeddings = pipeline.results.embeddings
        self.config = pipeline.config
        self.sentiment_method = sentiment_method

        # Validate prerequisites
        if self.df is None or self.embeddings is None:
            raise RuntimeError(
                "Pipeline must have data and embeddings loaded. "
                "Run pipeline.load_data() and pipeline.compute_embeddings() first."
            )

        # Extension results storage
        if not hasattr(pipeline.results, 'extensions'):
            pipeline.results.extensions = {}

    # =========================================================================
    # 1. BERTOPIC INTEGRATION
    # =========================================================================

    def run_bertopic(self, nr_topics: str = 'auto', min_topic_size: int = 15,
                     text_col: str = 'text', top_n_words: int = 10,
                     reduce_outliers: bool = True,
                     cache_file: Optional[str] = None) -> 'BERTopic':
        """
        Run BERTopic on the corpus using pre-computed embeddings.

        BERTopic combines your existing embeddings with HDBSCAN clustering
        and c-TF-IDF to produce auto-labeled topics that are more
        semantically coherent than plain KMeans + TF-IDF.

        Parameters
        ----------
        nr_topics : str or int
            'auto' to let BERTopic decide, or an int to merge down to N topics.
        min_topic_size : int
            Minimum documents per topic. Lower = more granular topics.
        text_col : str
            Column to use for c-TF-IDF label generation.
        top_n_words : int
            Number of words per topic label.
        reduce_outliers : bool
            If True, reassign outlier documents to nearest topic.
        cache_file : str, optional
            Path to save/load the BERTopic model.

        Returns
        -------
        BERTopic model
        """
        try:
            from bertopic import BERTopic
        except ImportError:
            raise ImportError(
                "BERTopic required: pip install bertopic --break-system-packages"
            )

        print(f"\n{'='*60}")
        print("EXTENSION: BERTopic Topic Modeling")
        print('='*60)

        df = self.df
        embeddings = self.embeddings
        texts = df[text_col].fillna('').astype(str).tolist()

        # Try loading cached model
        if cache_file and Path(cache_file).exists():
            print(f"  Loading BERTopic model from {cache_file}...")
            topic_model = BERTopic.load(cache_file)
            topics = topic_model.topics_

            # Auto-detect stale cache (stopword-only labels)
            _ti = topic_model.get_topic_info()
            _real = _ti[_ti['Topic'] != -1]
            _stale = False
            if len(_real) > 0:
                _first = _real['Name'].iloc[0].lower()
                _stale = any(f'_{w}_' in _first for w in ['the', 'and', 'to', 'of', 'in', 'is'])

            if len(topics) == len(df) and not _stale:
                print(f"  ✓ Loaded cached model with {len(set(topics)) - 1} topics")
                df['bertopic_id'] = topics
                self._store_bertopic_results(topic_model, topics, df)
                return topic_model
            else:
                reason = "stopword labels — needs vectorizer fix" if _stale else "size mismatch"
                print(f"  ⚠ Cache invalid ({reason}), recomputing...")

        # =================================================================
        # Configure sub-models
        # =================================================================
        from sklearn.feature_extraction.text import CountVectorizer

        n_docs = len(df)

        # --- Dimensionality reduction: UMAP preferred, PCA fallback ---
        try:
            from umap import UMAP
            dim_model = UMAP(
                n_neighbors=min(15, n_docs - 1),
                n_components=min(5, n_docs - 2),
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            print(f"  Using UMAP for dimensionality reduction")
        except ImportError:
            from sklearn.decomposition import PCA
            n_components = min(20, embeddings.shape[1], n_docs - 1)
            dim_model = PCA(n_components=n_components, random_state=42)
            print(f"  Using PCA({n_components}) for dimensionality reduction (pip install umap-learn for better results)")

        # --- Clustering: auto-scale to corpus size ---
        from hdbscan import HDBSCAN
        # For small corpora (<500), use smaller clusters to get more topics
        auto_min_cluster = max(5, min(min_topic_size, n_docs // 20))
        hdbscan_model = HDBSCAN(
            min_cluster_size=auto_min_cluster,
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        print(f"  HDBSCAN: min_cluster_size={auto_min_cluster} (corpus={n_docs} docs)")

        # --- Vectorizer: stopword filtering + safe min_df ---
        _base_stops = list(CountVectorizer(stop_words='english').get_stop_words())
        _extra_stops = [
            'said', 'says', 'new', 'year', 'years', 'time', 'just', 'like',
            'people', 'would', 'could', 'also', 'one', 'two', 'first', 'last',
            'according', 'report', 'reported', 'going', 'really', 'think',
            'know', 'want', 'need', 'good', 'way', 'day', 'week', 'today',
            'told', 'make', 'use', 'used', 'using', 'thing', 'things',
            'come', 'came', 'got', 'right', 'big', 'lot', 'let',
        ]
        vectorizer_model = CountVectorizer(
            stop_words=_base_stops + _extra_stops,
            ngram_range=(1, 2),
            min_df=1,           # safe for small topic counts
            max_df=1.0,         # no upper cap — let c-TF-IDF handle weighting
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b',
        )

        # =================================================================
        # Fit BERTopic
        # =================================================================
        print(f"  Fitting BERTopic...")
        topic_model = BERTopic(
            hdbscan_model=hdbscan_model,
            umap_model=dim_model,
            vectorizer_model=vectorizer_model,
            nr_topics=nr_topics,
            top_n_words=top_n_words,
            verbose=True,
            calculate_probabilities=True
        )

        topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)

        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        n_outliers = (np.array(topics) == -1).sum()
        print(f"  Discovered {n_topics} topics ({n_outliers} outlier docs)")

        # Reduce outliers by assigning to nearest topic
        if reduce_outliers and n_outliers > 0:
            print(f"  Reducing outliers...")
            new_topics = topic_model.reduce_outliers(
                texts, topics, strategy='embeddings', embeddings=embeddings
            )
            # update_topics recalculates c-TF-IDF labels — use same vectorizer
            try:
                topic_model.update_topics(texts, topics=new_topics,
                                           vectorizer_model=vectorizer_model)
            except ValueError:
                # Fallback: even more permissive vectorizer for very few topics
                print(f"    ⚠ Vectorizer too strict for {len(set(new_topics))} topics, using permissive settings...")
                _fallback_vec = CountVectorizer(
                    stop_words=_base_stops + _extra_stops,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=1.0,
                    token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b',
                )
                topic_model.update_topics(texts, topics=new_topics,
                                           vectorizer_model=_fallback_vec)
            topics = new_topics
            n_reassigned = n_outliers - (np.array(topics) == -1).sum()
            print(f"    Reassigned {n_reassigned} outlier docs to nearest topic")

        df['bertopic_id'] = topics

        # Save cache
        if cache_file:
            print(f"  Saving model to {cache_file}...")
            topic_model.save(cache_file, serialization='pickle')

        self._store_bertopic_results(topic_model, topics, df)
        return topic_model

    def _store_bertopic_results(self, topic_model, topics, df):
        """Store BERTopic results and print summary."""
        topic_info = topic_model.get_topic_info()
        topic_labels = {}
        for _, row in topic_info.iterrows():
            tid = row['Topic']
            if tid == -1:
                topic_labels[tid] = 'Outlier'
            else:
                topic_labels[tid] = row.get('Name', f'Topic_{tid}')

        # Map topic labels to df
        df['bertopic_label'] = df['bertopic_id'].map(topic_labels).fillna('Unknown')

        # Cross-reference with seeded narratives
        crossref = pd.crosstab(
            df['bertopic_label'],
            df['dominant_narrative'],
            normalize='index'
        )

        # Print summary
        print(f"\n  BERTopic Results:")
        print(f"  {'Topic':<40s} {'Count':>6s} {'Top Narrative':>25s}")
        print("  " + "-" * 75)

        for _, row in topic_info.iterrows():
            tid = row['Topic']
            if tid == -1:
                continue
            label = topic_labels.get(tid, f'Topic_{tid}')
            count = row['Count']
            # Find dominant seeded narrative for this topic
            if label in crossref.index:
                top_narr = crossref.loc[label].idxmax()
            else:
                top_narr = 'N/A'
            print(f"  {label[:40]:<40s} {count:>6d} {top_narr:>25s}")

        # Store results
        self.pipeline.results.extensions['bertopic'] = {
            'model': topic_model,
            'topic_info': topic_info,
            'topic_labels': topic_labels,
            'crossref': crossref,
        }

    def bertopic_topics_over_time(self, time_col: str = 'datetime',
                                    text_col: str = 'text',
                                    nr_bins: int = None) -> pd.DataFrame:
        """
        Track BERTopic topics over time using dynamic topic modeling.

        Returns a DataFrame with topic prevalence per time bin, which
        can be plotted with plot_bertopic_over_time().

        Parameters
        ----------
        time_col : str
            Datetime column in the DataFrame.
        text_col : str
            Text column for topic word recalculation.
        nr_bins : int, optional
            Number of time bins. None = auto (one per quarter or month).
        """
        if 'bertopic' not in self.pipeline.results.extensions:
            raise RuntimeError("Run run_bertopic() first.")

        print(f"\n  Computing BERTopic topics over time...")

        topic_model = self.pipeline.results.extensions['bertopic']['model']
        df = self.df
        texts = df[text_col].fillna('').astype(str).tolist()

        # Prepare timestamps
        timestamps = pd.to_datetime(df[time_col], errors='coerce')

        topics_over_time = topic_model.topics_over_time(
            texts,
            timestamps=timestamps.tolist(),
            nr_bins=nr_bins,
            datetime_format='%Y-%m'
        )

        self.pipeline.results.extensions['bertopic']['topics_over_time'] = topics_over_time

        print(f"  ✓ Computed topic dynamics across {topics_over_time['Timestamp'].nunique()} periods")

        return topics_over_time

    def bertopic_hierarchy(self, text_col: str = 'text') -> Dict[str, Any]:
        """
        Compute topic hierarchy showing which topics merge at different levels.

        Returns
        -------
        Dict with 'hierarchy_df' and 'tree' keys.
        """
        if 'bertopic' not in self.pipeline.results.extensions:
            raise RuntimeError("Run run_bertopic() first.")

        print(f"\n  Computing topic hierarchy...")

        topic_model = self.pipeline.results.extensions['bertopic']['model']
        texts = self.df[text_col].fillna('').astype(str).tolist()

        hierarchical_topics = topic_model.hierarchical_topics(texts)

        self.pipeline.results.extensions['bertopic']['hierarchy'] = hierarchical_topics

        print(f"  ✓ Built hierarchy with {len(hierarchical_topics)} merge steps")

        return {
            'hierarchy_df': hierarchical_topics,
        }

    # =========================================================================
    # 2. SENTIMENT / STANCE LAYER
    # =========================================================================

    def compute_sentiment(self, text_col: str = 'text',
                          batch_size: int = 64,
                          cache_file: Optional[str] = None) -> pd.DataFrame:
        """
        Compute per-article sentiment using either TextBlob (fast) or a
        transformer model (more accurate).

        Adds columns to df:
        - sentiment_score : float (-1 to 1, negative to positive)
        - sentiment_label : str ('Negative', 'Neutral', 'Positive')
        - sentiment_magnitude : float (0 to 1, strength of sentiment)

        Parameters
        ----------
        text_col : str
            Column to analyze. Tip: use 'title' for speed, 'text' for accuracy.
        batch_size : int
            Batch size for transformer method.
        cache_file : str, optional
            Path to cache sentiment scores.
        """
        print(f"\n{'='*60}")
        print(f"EXTENSION: Sentiment Analysis ({self.sentiment_method})")
        print('='*60)

        df = self.df

        # Try loading cache
        if cache_file and Path(cache_file).exists():
            print(f"  Loading cached sentiments from {cache_file}...")
            cached = pd.read_pickle(cache_file)
            if len(cached) == len(df):
                for col in ['sentiment_score', 'sentiment_label', 'sentiment_magnitude']:
                    if col in cached.columns:
                        df[col] = cached[col].values
                print(f"  ✓ Loaded {len(df)} cached sentiment scores")
                self._compute_narrative_stance()
                return df
            else:
                print(f"  ⚠ Cache mismatch, recomputing...")

        texts = df[text_col].fillna('').astype(str).tolist()

        if self.sentiment_method == 'textblob':
            scores, magnitudes = self._sentiment_textblob(texts)
        elif self.sentiment_method == 'transformer':
            scores, magnitudes = self._sentiment_transformer(texts, batch_size)
        else:
            raise ValueError(f"Unknown method: {self.sentiment_method}")

        df['sentiment_score'] = scores
        df['sentiment_magnitude'] = magnitudes
        df['sentiment_label'] = pd.cut(
            df['sentiment_score'],
            bins=[-1.01, -0.15, 0.15, 1.01],
            labels=['Negative', 'Neutral', 'Positive']
        )

        # Print summary
        print(f"\n  Sentiment Distribution:")
        print(f"    Positive:  {(df['sentiment_label'] == 'Positive').sum():>6d} "
              f"({(df['sentiment_label'] == 'Positive').mean() * 100:.1f}%)")
        print(f"    Neutral:   {(df['sentiment_label'] == 'Neutral').sum():>6d} "
              f"({(df['sentiment_label'] == 'Neutral').mean() * 100:.1f}%)")
        print(f"    Negative:  {(df['sentiment_label'] == 'Negative').sum():>6d} "
              f"({(df['sentiment_label'] == 'Negative').mean() * 100:.1f}%)")
        print(f"    Mean score: {df['sentiment_score'].mean():.3f}")

        # Cache
        if cache_file:
            df[['sentiment_score', 'sentiment_label', 'sentiment_magnitude']].to_pickle(
                cache_file
            )
            print(f"  Saved cache: {cache_file}")

        # Compute narrative-level stance
        self._compute_narrative_stance()

        return df

    def _sentiment_textblob(self, texts: List[str]) -> Tuple[List[float], List[float]]:
        """Fast sentiment via TextBlob (polarity-based)."""
        try:
            from textblob import TextBlob
        except ImportError:
            raise ImportError(
                "TextBlob required: pip install textblob --break-system-packages\n"
                "Then run: python -m textblob.download_corpora"
            )

        print(f"  Scoring {len(texts):,} texts with TextBlob...")
        scores = []
        magnitudes = []

        for i, text in enumerate(texts):
            if i % 5000 == 0 and i > 0:
                print(f"    {i:,}/{len(texts):,}...")
            try:
                # Use first 1000 chars for speed
                blob = TextBlob(text[:1000])
                polarity = blob.sentiment.polarity       # -1 to 1
                subjectivity = blob.sentiment.subjectivity  # 0 to 1
                scores.append(polarity)
                magnitudes.append(subjectivity)
            except Exception:
                scores.append(0.0)
                magnitudes.append(0.0)

        print(f"  ✓ Scored {len(texts):,} texts")
        return scores, magnitudes

    def _sentiment_transformer(self, texts: List[str],
                                batch_size: int = 64) -> Tuple[List[float], List[float]]:
        """More accurate sentiment via a pretrained transformer model."""
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError(
                "Transformers required: pip install transformers --break-system-packages"
            )

        print(f"  Loading transformer sentiment model...")
        classifier = hf_pipeline(
            'sentiment-analysis',
            model='cardiffnlp/twitter-roberta-base-sentiment-latest',
            tokenizer='cardiffnlp/twitter-roberta-base-sentiment-latest',
            top_k=None,
            truncation=True,
            max_length=512,
            device=-1   # CPU; change to 0 for GPU
        )

        # Map model labels to scores
        label_score_map = {
            'positive': 1.0,
            'neutral': 0.0,
            'negative': -1.0,
        }

        print(f"  Scoring {len(texts):,} texts in batches of {batch_size}...")
        scores = []
        magnitudes = []

        for i in range(0, len(texts), batch_size):
            batch = [t[:512] for t in texts[i:i + batch_size]]
            if i % (batch_size * 20) == 0 and i > 0:
                print(f"    {i:,}/{len(texts):,}...")

            try:
                results = classifier(batch)
            except Exception:
                # Fallback for problematic texts
                results = [None] * len(batch)

            for result in results:
                if result is None:
                    scores.append(0.0)
                    magnitudes.append(0.0)
                    continue

                # Compute weighted score: sum(label_score * confidence)
                weighted_score = 0.0
                max_conf = 0.0
                for item in result:
                    label = item['label'].lower()
                    conf = item['score']
                    weighted_score += label_score_map.get(label, 0.0) * conf
                    max_conf = max(max_conf, conf)

                scores.append(weighted_score)
                magnitudes.append(max_conf)

        print(f"  ✓ Scored {len(texts):,} texts")
        return scores, magnitudes

    def _compute_narrative_stance(self):
        """
        Compute per-narrative sentiment aggregates over time.

        Creates a stance DataFrame: for each (period, narrative), compute
        mean sentiment, % positive, % negative, and magnitude.
        """
        df = self.df

        if 'sentiment_score' not in df.columns:
            return

        if 'datetime' not in df.columns:
            return

        print(f"\n  Computing narrative stance over time...")

        # Use year as default time unit (consistent with existing pipeline)
        if 'year' not in df.columns:
            df['year'] = pd.to_datetime(df['datetime'], errors='coerce').dt.year

        periods = sorted(df['year'].dropna().unique())
        narratives = sorted(df['dominant_narrative'].dropna().unique())

        stance_data = []
        for period in periods:
            period_df = df[df['year'] == period]

            for narr in narratives:
                narr_df = period_df[period_df['dominant_narrative'] == narr]
                n = len(narr_df)

                if n < 3:
                    continue

                sent_scores = narr_df['sentiment_score']
                stance_data.append({
                    'period': period,
                    'narrative': narr,
                    'n_articles': n,
                    'mean_sentiment': sent_scores.mean(),
                    'median_sentiment': sent_scores.median(),
                    'std_sentiment': sent_scores.std(),
                    'pct_positive': (narr_df['sentiment_label'] == 'Positive').mean() * 100,
                    'pct_negative': (narr_df['sentiment_label'] == 'Negative').mean() * 100,
                    'pct_neutral': (narr_df['sentiment_label'] == 'Neutral').mean() * 100,
                    'mean_magnitude': narr_df['sentiment_magnitude'].mean(),
                })

        stance_df = pd.DataFrame(stance_data)

        self.pipeline.results.temporal['stance'] = stance_df
        self.pipeline.results.extensions['sentiment'] = {
            'stance_df': stance_df,
            'method': self.sentiment_method,
        }

        # Print stance summary
        print(f"\n  Narrative Stance Summary (overall):")
        print(f"  {'Narrative':<30s} {'Mean Sent':>10s} {'% Pos':>7s} {'% Neg':>7s}")
        print("  " + "-" * 56)

        overall = stance_df.groupby('narrative').agg({
            'mean_sentiment': 'mean',
            'pct_positive': 'mean',
            'pct_negative': 'mean',
        })
        for narr in overall.index:
            row = overall.loc[narr]
            print(f"  {narr:<30s} {row['mean_sentiment']:>10.3f} "
                  f"{row['pct_positive']:>6.1f}% {row['pct_negative']:>6.1f}%")

        return stance_df

    # =========================================================================
    # 3. SPIKE DETECTION
    # =========================================================================

    def detect_narrative_spikes(self, z_threshold: float = 2.0,
                                 rolling_window: int = 4,
                                 time_col: str = 'year',
                                 min_articles: int = 5,
                                 top_articles_per_spike: int = 5) -> Dict[str, Any]:
        """
        Detect periods where individual narratives spiked well above
        their rolling baseline using z-score analysis.

        Unlike detect_shift_periods() (which finds when the overall
        distribution changes), this finds when a SINGLE narrative surges.

        Parameters
        ----------
        z_threshold : float
            Z-score above which a period is flagged as a spike.
            2.0 = ~95th percentile. 1.5 = more sensitive.
        rolling_window : int
            Window for computing rolling mean/std baseline.
        time_col : str
            Column for time periods.
        min_articles : int
            Minimum articles in a period to consider.
        top_articles_per_spike : int
            Number of representative articles to extract per spike.

        Returns
        -------
        Dict with:
          - spike_df : DataFrame of all spike events
          - by_narrative : Dict[str, DataFrame] of spikes per narrative
          - z_scores_df : Full z-score matrix (periods x narratives)
        """
        print(f"\n{'='*60}")
        print("EXTENSION: Narrative Spike Detection")
        print('='*60)

        df = self.df

        # Compute per-period per-narrative counts
        periods = sorted(df[time_col].dropna().unique())
        narratives = sorted(df['dominant_narrative'].dropna().unique())

        prevalence = []
        for period in periods:
            period_df = df[df[time_col] == period]
            total = len(period_df)
            if total < min_articles:
                continue
            row = {'period': period, 'total': total}
            for narr in narratives:
                count = (period_df['dominant_narrative'] == narr).sum()
                row[f'{narr}_count'] = count
                row[f'{narr}_pct'] = 100 * count / total
            prevalence.append(row)

        prev_df = pd.DataFrame(prevalence).sort_values('period').reset_index(drop=True)

        # Compute z-scores against rolling baseline
        all_spikes = []
        z_score_records = []

        for narr in narratives:
            pct_col = f'{narr}_pct'
            if pct_col not in prev_df.columns:
                continue

            series = prev_df[pct_col].values
            periods_arr = prev_df['period'].values

            # Rolling baseline
            rolling_mean = pd.Series(series).rolling(
                window=rolling_window, min_periods=1, center=True
            ).mean().values

            rolling_std = pd.Series(series).rolling(
                window=rolling_window, min_periods=2, center=True
            ).std().fillna(series.std()).values

            # Prevent division by zero
            rolling_std = np.maximum(rolling_std, 0.5)

            z_scores = (series - rolling_mean) / rolling_std

            for i in range(len(series)):
                z_score_records.append({
                    'period': periods_arr[i],
                    'narrative': narr,
                    'pct': series[i],
                    'baseline_mean': rolling_mean[i],
                    'baseline_std': rolling_std[i],
                    'z_score': z_scores[i],
                    'is_spike': z_scores[i] >= z_threshold,
                })

                if z_scores[i] >= z_threshold:
                    # Get representative articles from this spike
                    spike_period = periods_arr[i]
                    spike_mask = (
                        (df[time_col] == spike_period) &
                        (df['dominant_narrative'] == narr)
                    )
                    spike_articles = df[spike_mask]

                    # Sort by narrative score if available
                    narr_id = self.config.name_to_id(narr) if hasattr(self.config, 'name_to_id') else None
                    score_col = f'{narr_id}_score' if narr_id else None

                    if score_col and score_col in spike_articles.columns:
                        top_articles = spike_articles.nlargest(
                            top_articles_per_spike, score_col
                        )
                    else:
                        top_articles = spike_articles.head(top_articles_per_spike)

                    if 'title' in top_articles.columns:
                        article_titles = top_articles['title'].tolist()
                    else:
                        article_titles = []

                    all_spikes.append({
                        'period': spike_period,
                        'narrative': narr,
                        'pct': series[i],
                        'baseline': rolling_mean[i],
                        'z_score': z_scores[i],
                        'n_articles': int(spike_mask.sum()),
                        'top_articles': article_titles,
                    })

        # Build DataFrames with explicit columns to avoid KeyError when empty
        spike_columns = ['period', 'narrative', 'pct', 'baseline', 'z_score',
                         'n_articles', 'top_articles']
        z_columns = ['period', 'narrative', 'pct', 'baseline_mean',
                     'baseline_std', 'z_score', 'is_spike']

        if len(all_spikes) > 0:
            spike_df = pd.DataFrame(all_spikes, columns=spike_columns)
        else:
            spike_df = pd.DataFrame(columns=spike_columns)

        if len(z_score_records) > 0:
            z_scores_df = pd.DataFrame(z_score_records, columns=z_columns)
        else:
            z_scores_df = pd.DataFrame(columns=z_columns)

        # Group by narrative
        by_narrative = {}
        if len(spike_df) > 0:
            for narr in narratives:
                narr_spikes = spike_df[spike_df['narrative'] == narr]
                if len(narr_spikes) > 0:
                    by_narrative[narr] = narr_spikes

        # Store results
        self.pipeline.results.temporal['spikes'] = {
            'spike_df': spike_df,
            'by_narrative': by_narrative,
            'z_scores_df': z_scores_df,
            'z_threshold': z_threshold,
            'rolling_window': rolling_window,
        }
        self.pipeline.results.extensions['spikes'] = {
            'spike_df': spike_df,
            'by_narrative': by_narrative,
            'z_scores_df': z_scores_df,
        }

        # Print summary
        print(f"\n  Spike Detection Results (z ≥ {z_threshold}):")
        print(f"  Total spikes found: {len(spike_df)}")

        if len(spike_df) > 0:
            print(f"\n  {'Period':<12s} {'Narrative':<30s} {'%':>6s} {'Base':>6s} "
                  f"{'Z':>6s} {'Articles':>8s}")
            print("  " + "-" * 72)
            for _, row in spike_df.sort_values('z_score', ascending=False).head(15).iterrows():
                print(f"  {str(row['period']):<12s} {row['narrative']:<30s} "
                      f"{row['pct']:>5.1f}% {row['baseline']:>5.1f}% "
                      f"{row['z_score']:>5.1f} {row['n_articles']:>8d}")
                if row.get('top_articles'):
                    for title in row['top_articles'][:2]:
                        print(f"    → {str(title)[:75]}")
        else:
            print("  No spikes detected. Try lowering z_threshold.")

        return {
            'spike_df': spike_df,
            'by_narrative': by_narrative,
            'z_scores_df': z_scores_df,
        }

    # =========================================================================
    # RUN ALL
    # =========================================================================

    def run_all(self, text_col: str = 'text',
                bertopic_min_topic_size: int = 15,
                bertopic_cache: Optional[str] = None,
                sentiment_cache: Optional[str] = None,
                spike_z_threshold: float = 2.0,
                spike_window: int = 4):
        """
        Run all three extensions in sequence.

        Parameters
        ----------
        text_col : str
            Text column for BERTopic and sentiment.
        bertopic_min_topic_size : int
            Minimum docs per BERTopic topic.
        bertopic_cache : str, optional
            Cache path for BERTopic model.
        sentiment_cache : str, optional
            Cache path for sentiment scores.
        spike_z_threshold : float
            Z-score threshold for spike detection.
        spike_window : int
            Rolling window for spike baseline.
        """
        print(f"\n{'#'*60}")
        print(f"# RUNNING ALL EXTENSIONS")
        print(f"{'#'*60}")

        # 1. BERTopic
        self.run_bertopic(
            text_col=text_col,
            min_topic_size=bertopic_min_topic_size,
            cache_file=bertopic_cache
        )
        if 'datetime' in self.df.columns:
            self.bertopic_topics_over_time(text_col=text_col)

        # 2. Sentiment
        # Use title for speed (full text can be slow with textblob on large corpora)
        sent_col = 'title' if 'title' in self.df.columns else text_col
        self.compute_sentiment(
            text_col=sent_col,
            cache_file=sentiment_cache
        )

        # 3. Spike Detection
        self.detect_narrative_spikes(
            z_threshold=spike_z_threshold,
            rolling_window=spike_window
        )

        print(f"\n{'#'*60}")
        print(f"# ALL EXTENSIONS COMPLETE")
        print(f"{'#'*60}")

    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    def generate_extension_report(self, output_dir: str) -> Dict[str, str]:
        """
        Generate visualizations and export data for all extensions.

        Parameters
        ----------
        output_dir : str
            Directory to save outputs.

        Returns
        -------
        Dict mapping output names to file paths.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        outputs = {}

        print(f"\nGenerating extension report in {output_dir}...")

        # BERTopic outputs
        if 'bertopic' in self.pipeline.results.extensions:
            fig = plot_bertopic_overview(self.pipeline)
            path = output_path / 'bertopic_overview.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            outputs['bertopic_overview'] = str(path)
            print(f"  ✓ {path.name}")

            if 'topics_over_time' in self.pipeline.results.extensions['bertopic']:
                fig = plot_bertopic_over_time(self.pipeline)
                path = output_path / 'bertopic_over_time.png'
                fig.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                outputs['bertopic_over_time'] = str(path)
                print(f"  ✓ {path.name}")

        # Sentiment outputs
        if 'sentiment' in self.pipeline.results.extensions:
            fig = plot_sentiment_by_narrative(self.pipeline)
            path = output_path / 'sentiment_by_narrative.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            outputs['sentiment_by_narrative'] = str(path)
            print(f"  ✓ {path.name}")

            if 'stance' in self.pipeline.results.temporal:
                fig = plot_stance_over_time(self.pipeline)
                path = output_path / 'stance_over_time.png'
                fig.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                outputs['stance_over_time'] = str(path)
                print(f"  ✓ {path.name}")

                fig = plot_stance_heatmap(self.pipeline)
                path = output_path / 'stance_heatmap.png'
                fig.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                outputs['stance_heatmap'] = str(path)
                print(f"  ✓ {path.name}")

        # Spike outputs
        if 'spikes' in self.pipeline.results.extensions:
            fig = plot_prevalence_with_spikes(self.pipeline)
            path = output_path / 'prevalence_with_spikes.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            outputs['prevalence_with_spikes'] = str(path)
            print(f"  ✓ {path.name}")

            fig = plot_spike_z_scores(self.pipeline)
            path = output_path / 'spike_z_scores.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            outputs['spike_z_scores'] = str(path)
            print(f"  ✓ {path.name}")

            # Export spike report as markdown
            report = generate_spike_report(self.pipeline)
            path = output_path / 'spike_report.md'
            with open(path, 'w') as f:
                f.write(report)
            outputs['spike_report'] = str(path)
            print(f"  ✓ {path.name}")

        print(f"  ✓ Extension report complete ({len(outputs)} files)")
        return outputs


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def _get_config_colors(pipeline, narratives):
    """Get color mapping from pipeline config, with fallback."""
    colors = {}
    for narr in narratives:
        try:
            colors[narr] = pipeline.config.get_color(narr)
        except (KeyError, AttributeError):
            # Fallback to a hash-based color
            hue = hash(narr) % 360
            colors[narr] = f'#{(hash(narr) * 2654435761) % (1 << 24):06x}'
    return colors


# ---- BERTopic Visualizations ----

def plot_bertopic_overview(pipeline, figsize=(18, 10), top_n=15, save_path=None):
    """
    Stacked horizontal bar chart showing each BERTopic topic's narrative
    composition, colored by seeded narrative colors. Much more readable
    than the old heatmap approach.
    """
    if 'bertopic' not in pipeline.results.extensions:
        raise RuntimeError("Run NarrativeExtensions.run_bertopic() first.")

    topic_info = pipeline.results.extensions['bertopic']['topic_info']
    topic_labels = pipeline.results.extensions['bertopic']['topic_labels']
    crossref = pipeline.results.extensions['bertopic']['crossref']
    df = pipeline.results.df

    # Filter out outlier topic
    real_topics = topic_info[topic_info['Topic'] != -1].head(top_n)

    # Get narrative colors from config
    narr_colors = {}
    for name in pipeline.config.display_names:
        try:
            narr_colors[name] = pipeline.config.get_color(name)
        except (KeyError, AttributeError):
            narr_colors[name] = '#888888'

    narratives = sorted(narr_colors.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                     gridspec_kw={'width_ratios': [2, 1]})

    # ── Left: Stacked bars (topic composition by narrative) ──
    topic_names = []
    y_positions = []

    for idx, (_, row) in enumerate(real_topics.iterrows()):
        tid = row['Topic']
        label = topic_labels.get(tid, f'Topic_{tid}')
        # Clean up BERTopic's default naming
        clean = label.replace('_', ' ').strip()
        if clean and clean[0].isdigit():
            clean = ' '.join(clean.split(' ')[1:])  # drop leading number
        topic_names.append((clean or f'Topic {tid}')[:35])
        y_positions.append(idx)

        # Get narrative proportions for this topic
        if label in crossref.index:
            props = crossref.loc[label]
        else:
            props = pd.Series(0.0, index=narratives)

        # Draw stacked bar segments
        left = 0
        for narr in narratives:
            width = props.get(narr, 0) * row['Count']
            if width > 0:
                ax1.barh(idx, width, left=left, height=0.7,
                        color=narr_colors.get(narr, '#888'),
                        edgecolor='white', linewidth=0.3)
                # Label segments > 15% with narrative abbreviation
                if props.get(narr, 0) > 0.15:
                    abbrev = narr[:12] + '..' if len(narr) > 14 else narr
                    ax1.text(left + width / 2, idx, f'{abbrev}\n{props.get(narr, 0):.0%}',
                            ha='center', va='center', fontsize=6.5,
                            fontweight='bold', color='white')
                left += width

        # Add total count at end
        ax1.text(left + 2, idx, f'n={row["Count"]}', va='center', fontsize=8,
                color='#555555')

    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(topic_names, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel('Number of Articles', fontsize=11)
    ax1.set_title('BERTopic Topics → Seeded Narrative Composition', fontsize=13, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── Right: Narrative legend with colors ──
    ax2.axis('off')
    ax2.set_title('Seeded Narratives', fontsize=12, fontweight='bold')

    for i, narr in enumerate(narratives):
        color = narr_colors.get(narr, '#888')
        # Count articles with this dominant narrative
        n = (df['dominant_narrative'] == narr).sum()
        ax2.add_patch(plt.Rectangle((0.05, 0.92 - i * 0.09), 0.08, 0.06,
                                     facecolor=color, edgecolor='white',
                                     transform=ax2.transAxes))
        ax2.text(0.16, 0.95 - i * 0.09, f'{narr}  ({n})',
                fontsize=10, va='center', transform=ax2.transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_bertopic_over_time(pipeline, top_n_topics=8, figsize=(16, 8), save_path=None):
    """
    Line chart showing BERTopic topic prevalence over time.
    Uses BERTopic's built-in topics_over_time results.
    """
    if 'bertopic' not in pipeline.results.extensions:
        raise RuntimeError("Run BERTopic extensions first.")

    ext = pipeline.results.extensions['bertopic']
    if 'topics_over_time' not in ext:
        raise RuntimeError("Run bertopic_topics_over_time() first.")

    tot = ext['topics_over_time']
    topic_labels = ext['topic_labels']

    # Get top topics by total frequency
    topic_totals = tot.groupby('Topic')['Frequency'].sum().sort_values(ascending=False)
    top_topics = topic_totals.head(top_n_topics).index.tolist()

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(top_topics)))

    for i, topic_id in enumerate(top_topics):
        topic_data = tot[tot['Topic'] == topic_id].sort_values('Timestamp')
        label = topic_labels.get(topic_id, f'Topic {topic_id}')
        label_short = label[:40]
        ax.plot(topic_data['Timestamp'], topic_data['Frequency'],
               label=label_short, color=colors[i], linewidth=2, marker='o', markersize=4)

    ax.set_xlabel('Time')
    ax.set_ylabel('Topic Frequency')
    ax.set_title(f'{pipeline.config.name}: BERTopic Topics Over Time',
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


# ---- Sentiment Visualizations ----

def plot_sentiment_by_narrative(pipeline, figsize=(14, 8), save_path=None):
    """
    Violin + strip plot showing sentiment distribution per seeded narrative.
    Reveals which narratives are framed positively vs negatively.
    """
    df = pipeline.results.df

    if 'sentiment_score' not in df.columns:
        raise RuntimeError("Run compute_sentiment() first.")

    narratives = sorted(df['dominant_narrative'].dropna().unique())
    colors = _get_config_colors(pipeline, narratives)

    fig, ax = plt.subplots(figsize=figsize)

    # Sort narratives by mean sentiment
    narr_order = (df.groupby('dominant_narrative')['sentiment_score']
                    .mean().sort_values().index.tolist())

    # Violin plot
    parts = ax.violinplot(
        [df[df['dominant_narrative'] == n]['sentiment_score'].dropna().values
         for n in narr_order],
        positions=range(len(narr_order)),
        showmeans=True,
        showmedians=True,
        widths=0.7
    )

    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        narr = narr_order[i]
        color = colors.get(narr, '#888888')
        pc.set_facecolor(color)
        pc.set_alpha(0.5)

    # Add mean markers
    means = [df[df['dominant_narrative'] == n]['sentiment_score'].mean()
             for n in narr_order]
    ax.scatter(range(len(narr_order)), means, color='black', s=50, zorder=5, marker='D')

    # Zero line
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xticks(range(len(narr_order)))
    ax.set_xticklabels(narr_order, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Sentiment Score', fontsize=12)
    ax.set_title(f'{pipeline.config.name}: Sentiment by Narrative',
                fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_stance_over_time(pipeline, figsize=(16, 8), save_path=None):
    """
    Line chart showing mean sentiment per narrative over time.
    Reveals narrative stance drift (e.g., "AI coverage becoming more negative").
    """
    if 'stance' not in pipeline.results.temporal:
        raise RuntimeError("Run compute_sentiment() first.")

    stance_df = pipeline.results.temporal['stance']
    narratives = sorted(stance_df['narrative'].unique())
    colors = _get_config_colors(pipeline, narratives)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1],
                                     sharex=True)

    # Top: mean sentiment lines
    for narr in narratives:
        data = stance_df[stance_df['narrative'] == narr].sort_values('period')
        if len(data) > 1:
            color = colors.get(narr, '#888888')
            ax1.plot(data['period'], data['mean_sentiment'],
                    label=narr, color=color, linewidth=2, marker='o', markersize=5)

            # Shade std
            ax1.fill_between(
                data['period'],
                data['mean_sentiment'] - data['std_sentiment'],
                data['mean_sentiment'] + data['std_sentiment'],
                alpha=0.1, color=color
            )

    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Mean Sentiment Score', fontsize=11)
    ax1.set_title(f'{pipeline.config.name}: Narrative Stance Over Time',
                 fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Bottom: % negative as area
    for narr in narratives:
        data = stance_df[stance_df['narrative'] == narr].sort_values('period')
        if len(data) > 1:
            color = colors.get(narr, '#888888')
            ax2.plot(data['period'], data['pct_negative'],
                    color=color, linewidth=1.5, alpha=0.7)

    ax2.set_ylabel('% Negative', fontsize=11)
    ax2.set_xlabel('Period', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_stance_heatmap(pipeline, figsize=(14, 8), save_path=None):
    """
    Heatmap: period × narrative, colored by mean sentiment.
    Red = negative framing, blue = positive framing.
    """
    if 'stance' not in pipeline.results.temporal:
        raise RuntimeError("Run compute_sentiment() first.")

    stance_df = pipeline.results.temporal['stance']

    pivot = stance_df.pivot_table(
        index='narrative', columns='period',
        values='mean_sentiment', aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu',
               center=0, linewidths=0.5, ax=ax,
               cbar_kws={'label': 'Mean Sentiment'})

    ax.set_title(f'{pipeline.config.name}: Narrative Stance Heatmap',
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Narrative')
    ax.set_xlabel('Period')
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ---- Spike Visualizations ----

def plot_prevalence_with_spikes(pipeline, metric='pct', figsize=(18, 10), save_path=None):
    """
    Enhanced version of plot_temporal_prevalence that overlays spike
    annotations as vertical markers with labels.
    """
    df = pipeline.results.df
    spikes_data = pipeline.results.extensions.get('spikes', {})
    spike_df = spikes_data.get('spike_df', pd.DataFrame())

    if 'prevalence' not in pipeline.results.temporal:
        raise RuntimeError("Run compute_temporal_prevalence() first.")

    prevalence_df = pipeline.results.temporal['prevalence']
    narratives = pipeline.config.display_names
    colors = {}
    for name in narratives:
        colors[name] = pipeline.config.get_color(name)

    fig, ax = plt.subplots(figsize=figsize)

    suffix = '_pct' if metric == 'pct' else '_avg_score'

    # Plot prevalence lines
    for display_name in narratives:
        col = f'{display_name}{suffix}'
        if col in prevalence_df.columns:
            color = colors[display_name]
            periods = prevalence_df['period'].astype(str)
            values = prevalence_df[col]
            ax.plot(periods, values, label=display_name, color=color,
                   linewidth=2, marker='o', markersize=4)

    # Overlay spike markers
    if len(spike_df) > 0:
        period_strs = prevalence_df['period'].astype(str).tolist()

        for _, spike in spike_df.iterrows():
            spike_period = str(spike['period'])
            narr = spike['narrative']
            z = spike['z_score']
            pct = spike['pct']

            color = colors.get(narr, '#FF0000')

            # Draw a diamond marker at the spike point
            if spike_period in period_strs:
                ax.plot(spike_period, pct, marker='D', markersize=12,
                       color=color, markeredgecolor='black', markeredgewidth=1.5,
                       zorder=5)

                # Add z-score annotation
                ax.annotate(
                    f'z={z:.1f}',
                    xy=(spike_period, pct),
                    xytext=(0, 15),
                    textcoords='offset points',
                    fontsize=7,
                    fontweight='bold',
                    color=color,
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                             edgecolor=color, alpha=0.8)
                )

    ax.set_xlabel('Time Period', fontsize=12)
    ylabel = 'Prevalence (%)' if metric == 'pct' else 'Average Similarity Score'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{pipeline.config.name}: Narrative Prevalence with Spike Detection',
                fontsize=14, fontweight='bold')

    # Legend with spike marker explanation
    handles, labels = ax.get_legend_handles_labels()
    spike_marker = Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                          markeredgecolor='black', markersize=10, label='Spike (z ≥ threshold)')
    handles.append(spike_marker)
    labels.append('Spike')
    ax.legend(handles=handles, labels=labels,
             bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_spike_z_scores(pipeline, figsize=(16, 8), save_path=None):
    """
    Heatmap showing z-scores for each narrative × period.
    Highlights spikes in hot colors, dips in cool colors.
    """
    spikes_data = pipeline.results.extensions.get('spikes', {})
    z_scores_df = spikes_data.get('z_scores_df', pd.DataFrame())
    z_threshold = pipeline.results.temporal.get('spikes', {}).get('z_threshold', 2.0)

    if len(z_scores_df) == 0:
        raise RuntimeError("Run detect_narrative_spikes() first.")

    pivot = z_scores_df.pivot_table(
        index='narrative', columns='period',
        values='z_score', aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdBu_r',
               center=0, linewidths=0.5, ax=ax,
               vmin=-3, vmax=3,
               cbar_kws={'label': 'Z-Score'})

    ax.set_title(f'{pipeline.config.name}: Narrative Spike Z-Scores '
                f'(threshold = {z_threshold})',
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Narrative')
    ax.set_xlabel('Period')
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ---- Spike Markdown Report ----

def generate_spike_report(pipeline) -> str:
    """Generate markdown report of detected narrative spikes."""
    spikes_data = pipeline.results.extensions.get('spikes', {})
    spike_df = spikes_data.get('spike_df', pd.DataFrame())
    z_threshold = pipeline.results.temporal.get('spikes', {}).get('z_threshold', 2.0)

    lines = [
        f"# {pipeline.config.name}: Narrative Spike Report",
        "",
        f"**Z-score threshold:** {z_threshold}",
        f"**Total spikes detected:** {len(spike_df)}",
        "",
        "---",
        "",
    ]

    if len(spike_df) == 0:
        lines.append("No spikes detected.")
        return "\n".join(lines)

    # Group by narrative
    for narr in sorted(spike_df['narrative'].unique()):
        narr_spikes = spike_df[spike_df['narrative'] == narr].sort_values(
            'z_score', ascending=False
        )
        lines.extend([
            f"## {narr}",
            "",
            f"**Spikes:** {len(narr_spikes)}",
            "",
        ])

        for _, row in narr_spikes.iterrows():
            lines.extend([
                f"### {row['period']} — z={row['z_score']:.1f}",
                "",
                f"- Prevalence: {row['pct']:.1f}% (baseline: {row['baseline']:.1f}%)",
                f"- Articles in spike: {row['n_articles']}",
                "",
            ])

            if row.get('top_articles'):
                lines.append("**Representative articles:**")
                lines.append("")
                for title in row['top_articles'][:5]:
                    lines.append(f"- {title}")
                lines.append("")

        lines.extend(["---", ""])

    return "\n".join(lines)