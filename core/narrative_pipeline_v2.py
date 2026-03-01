"""
Narrative Analysis Pipeline
===========================
Unified workflow for narrative analysis across domains.

Steps:
1. Ingest and clean canonical dataset
2. Generate/load embeddings
3. Seeded narrative detection (prototype-based)
4. Discovered narrative clustering
5. Narrative co-occurrence networks
6. Temporal analysis and change detection

Usage:
    from narrative_pipeline import NarrativePipeline
    from narrative_config import EV_CONFIG
    
    pipeline = NarrativePipeline(EV_CONFIG)
    pipeline.load_data('unified_news.csv')
    pipeline.run_full_analysis()
    pipeline.generate_report('output/')
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import seaborn as sns
import pickle
import json
import warnings
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# ML/NLP imports
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import networkx as nx

# Import config
from narrative_config import NarrativeConfig, get_config

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

@dataclass
class AnalysisResults:
    """Container for analysis results."""
    config: NarrativeConfig
    df: pd.DataFrame = None
    embeddings: np.ndarray = None
    prototype_embeddings: Dict[str, np.ndarray] = None
    narrative_scores: pd.DataFrame = None
    narrative_binary: pd.DataFrame = None
    clusters: Dict[str, Any] = field(default_factory=dict)
    networks: Dict[str, Any] = field(default_factory=dict)
    temporal: Dict[str, Any] = field(default_factory=dict)
    projections: Dict[str, np.ndarray] = field(default_factory=dict)


class NarrativePipeline:
    """
    Main pipeline for narrative analysis.
    
    Attributes:
        config: NarrativeConfig defining the narrative domain
        model: SentenceTransformer model for embeddings
        results: AnalysisResults container
    """
    
    def __init__(self, config: NarrativeConfig, model_name: str = 'all-mpnet-base-v2'):
        """
        Initialize pipeline with narrative configuration.
        
        Args:
            config: NarrativeConfig or domain name string
            model_name: Sentence transformer model name
        """
        if isinstance(config, str):
            config = get_config(config)
        
        self.config = config
        self.model_name = model_name
        self.model = None  # Lazy loading
        self.results = AnalysisResults(config=config)
        
        # Plot settings
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['figure.dpi'] = 100
    
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print("  ✓ Model loaded")
        return self.model
    
    def _build_topic_keywords(self) -> set:
        """
        Build a set of topic-relevant keywords from config for relevance filtering.
        
        Extracts keywords from:
        1. Config name (e.g., "Electric Vehicles" → {'electric', 'vehicles', 'ev'})
        2. Narrative display names (e.g., "Battery Tech" → {'battery', 'tech'})
        3. Key terms from prototype sentences
        
        Returns a set of lowercase keywords. An article's title must contain
        at least one to be considered on-topic.
        """
        keywords = set()
        
        # From config name
        for word in self.config.name.lower().split():
            if len(word) > 2:
                keywords.add(word)
        
        # Common abbreviations / variants based on domain
        name_lower = self.config.name.lower()
        if 'electric' in name_lower and 'vehicle' in name_lower:
            keywords.update(['ev', 'evs', 'electric', 'vehicle', 'vehicles',
                           'tesla', 'charging', 'battery', 'lithium', 'hybrid',
                           'plug-in', 'supercharger', 'range', 'emissions',
                           'automotive', 'car', 'cars', 'truck', 'suv',
                           'ford', 'gm', 'rivian', 'lucid', 'nio', 'byd',
                           'volkswagen', 'hyundai', 'kia', 'bmw', 'audi',
                           'clean energy', 'climate', 'grid', 'solar',
                           'renewable', 'motor', 'drivetrain', 'powertrain'])
        elif 'ai' in name_lower or 'tech' in name_lower:
            keywords.update(['ai', 'artificial', 'intelligence', 'machine',
                           'learning', 'deep', 'neural', 'algorithm',
                           'gpt', 'chatgpt', 'openai', 'google', 'microsoft',
                           'meta', 'nvidia', 'chip', 'semiconductor',
                           'technology', 'tech', 'software', 'data',
                           'robot', 'automation', 'llm', 'model',
                           'compute', 'computing', 'cloud', 'digital',
                           'startup', 'silicon', 'cyber', 'quantum',
                           'apple', 'amazon', 'anthropic', 'deepmind',
                           'regulation', 'privacy', 'surveillance',
                           'autonomous', 'self-driving', 'crypto', 'blockchain',
                           'trump', 'tariff', 'china', 'export'])
        elif 'retail' in name_lower or 'invest' in name_lower:
            keywords.update([
                # Retail-specific platforms and culture
                'investor', 'investors', 'retail', 'robinhood', 'wallstreetbets',
                'reddit', 'gamestop', 'meme', 'squeeze',
                # Investment vehicles retail investors use
                'etf', 'portfolio', 'stock', 'stocks', 'index fund', 'passive',
                # Crypto (major retail narrative)
                'crypto', 'bitcoin', 'cryptocurrency', 'ethereum',
                # Market conditions retail investors track
                'market crash', 'bear market', 'bull market', 'recession',
                'inflation', 'interest rate', 'federal reserve',
                # Trading behavior
                'trading', 'day trading', 'options',
            ])
        
        # From narrative display names
        for name in self.config.display_names:
            for word in name.lower().split():
                if len(word) > 2 and word not in {'and', 'the', 'for'}:
                    keywords.add(word)
        
        # From prototype sentences: extract the most distinctive nouns
        # (words that appear in prototypes but aren't generic English)
        generic = {'the', 'and', 'for', 'are', 'was', 'has', 'have', 'been',
                   'will', 'with', 'that', 'this', 'from', 'they', 'their',
                   'about', 'more', 'into', 'over', 'such', 'than', 'very',
                   'could', 'would', 'should', 'being', 'some', 'other',
                   'which', 'when', 'what', 'how', 'who', 'can', 'may',
                   'new', 'not', 'but', 'all', 'also', 'its'}
        
        for narr_id, prototypes in self.config.prototypes.items():
            for proto in prototypes:
                for word in proto.lower().split():
                    word = word.strip('.,!?;:()[]"\'')
                    if len(word) > 3 and word not in generic:
                        keywords.add(word)
        
        return keywords
    
    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    
    def load_data(self, filepath: str, text_col: str = 'full_text', 
                  title_col: str = 'title', date_col: str = 'published_at',
                  min_text_len: int = 100) -> pd.DataFrame:
        """
        Load and preprocess data from canonical dataset.
        
        Args:
            filepath: Path to CSV/Parquet file
            text_col: Column containing article text
            title_col: Column containing article title
            date_col: Column containing publication date
            min_text_len: Minimum text length to include
            
        Returns:
            Preprocessed DataFrame
        """
        print(f"\n{'='*60}")
        print("STEP 1: Loading Data")
        print('='*60)
        
        # Load file
        filepath = Path(filepath)
        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)
        
        print(f"  Loaded {len(df):,} records from {filepath.name}")
        
        # Handle column mapping
        col_map = {}
        if text_col in df.columns:
            col_map[text_col] = 'text'
        elif 'text' not in df.columns:
            # Try to find a text column
            text_candidates = ['full_text', 'content', 'body', 'article_text']
            for col in text_candidates:
                if col in df.columns:
                    col_map[col] = 'text'
                    break
        
        if title_col in df.columns and title_col != 'title':
            col_map[title_col] = 'title'
        
        if date_col in df.columns and date_col != 'date':
            col_map[date_col] = 'date'
        
        if col_map:
            df = df.rename(columns=col_map)
        
        # Ensure text column exists
        if 'text' not in df.columns:
            raise ValueError(f"Could not find text column. Available: {list(df.columns)}")
        
        # Filter by text length
        df['text'] = df['text'].fillna('')
        df['text_len'] = df['text'].str.len()
        initial_count = len(df)
        df = df[df['text_len'] >= min_text_len].copy()
        print(f"  Filtered to {len(df):,} records (min_text_len={min_text_len})")
        
        # ── Relevance filter: drop articles that are clearly off-topic ──
        # Uses topic keywords derived from config name + narrative prototypes.
        # This catches food articles, lifestyle content, etc. that leak into 
        # GDELT scrapes. Only filters on title (short, clean, reliable).
        if hasattr(self, 'config') and self.config is not None:
            topic_keywords = self._build_topic_keywords()
            if topic_keywords:
                before = len(df)
                # Check if title contains at least one topic keyword
                title_lower = df['title'].fillna('').str.lower()
                has_keyword = title_lower.apply(
                    lambda t: any(kw in t for kw in topic_keywords)
                )
                df = df[has_keyword].copy()
                n_dropped = before - len(df)
                if n_dropped > 0:
                    print(f"  Relevance filter: dropped {n_dropped} off-topic articles "
                          f"({len(df):,} remaining)")
        
        # Parse dates
        if 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
            df['year'] = df['datetime'].dt.year
            df['year_month'] = df['datetime'].dt.to_period('M')
            df = df.sort_values('datetime')
            
            date_range = df['datetime'].dropna()
            if len(date_range) > 0:
                print(f"  Date range: {date_range.min().date()} to {date_range.max().date()}")
        
        # Create combined text for embedding
        if 'title' in df.columns:
            df['embed_text'] = df['title'].fillna('') + ' ' + df['text']
        else:
            df['embed_text'] = df['text']
        
        self.results.df = df.reset_index(drop=True)
        print(f"  ✓ Data loaded: {len(df):,} articles")
        
        return df
    
    def load_dataframe(self, df: pd.DataFrame, text_col: str = 'text',
                       title_col: str = 'title', date_col: str = 'date') -> pd.DataFrame:
        """Load from existing DataFrame."""
        # Create a copy and call load_data logic
        temp_path = '/tmp/temp_narrative_data.csv'
        df.to_csv(temp_path, index=False)
        return self.load_data(temp_path, text_col, title_col, date_col)
    
    # =========================================================================
    # STEP 2: EMBEDDINGS
    # =========================================================================
    
    def compute_embeddings(self, cache_file: Optional[str] = None, 
                          batch_size: int = 32) -> np.ndarray:
        """
        Compute or load article embeddings.
        
        Args:
            cache_file: Path to cache embeddings (None = no caching)
            batch_size: Batch size for encoding
            
        Returns:
            Array of embeddings (n_articles, embedding_dim)
        """
        print(f"\n{'='*60}")
        print("STEP 2: Computing Embeddings")
        print('='*60)
        
        df = self.results.df
        
        # Try loading from cache
        if cache_file and Path(cache_file).exists():
            print(f"  Loading embeddings from {cache_file}...")
            with open(cache_file, 'rb') as f:
                embeddings = pickle.load(f)
            
            if len(embeddings) == len(df):
                print(f"  ✓ Loaded {len(embeddings):,} embeddings from cache")
                self.results.embeddings = embeddings
                return embeddings
            else:
                print(f"  ⚠ Cache size mismatch ({len(embeddings)} vs {len(df)}), recomputing...")
        
        # Compute embeddings
        model = self._load_model()
        
        print(f"  Computing embeddings for {len(df):,} articles...")
        embeddings = model.encode(
            df['embed_text'].tolist(),
            show_progress_bar=True,
            batch_size=batch_size,
            convert_to_numpy=True
        )
        
        # Save to cache
        if cache_file:
            print(f"  Saving embeddings to {cache_file}...")
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
        
        self.results.embeddings = embeddings
        print(f"  ✓ Computed embeddings: shape {embeddings.shape}")
        
        return embeddings
    
    def compute_prototype_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Compute embeddings for narrative prototypes.
        
        Returns:
            Dict mapping narrative_id to centroid embedding
        """
        print(f"\n  Computing prototype embeddings...")
        
        model = self._load_model()
        prototype_embeddings = {}
        
        for narrative_id, sentences in self.config.prototypes.items():
            sent_embeddings = model.encode(sentences, convert_to_numpy=True)
            centroid = sent_embeddings.mean(axis=0)
            prototype_embeddings[narrative_id] = centroid
            
            display_name = self.config.id_to_name(narrative_id)
            print(f"    {display_name}: {len(sentences)} prototypes")
        
        self.results.prototype_embeddings = prototype_embeddings
        return prototype_embeddings
    
    # =========================================================================
    # STEP 3A: SEEDED NARRATIVE DETECTION
    # =========================================================================
    
    def detect_seeded_narratives(self, threshold: Optional[float] = None,
                                 calibrate: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect narratives using prototype-based similarity.
        
        Args:
            threshold: Similarity threshold (None = use config default)
            calibrate: Whether to compute calibration metrics
            
        Returns:
            Tuple of (scores_df, binary_df)
        """
        print(f"\n{'='*60}")
        print("STEP 3A: Seeded Narrative Detection")
        print('='*60)
        
        if threshold is None:
            threshold = self.config.default_threshold
        
        embeddings = self.results.embeddings
        
        # Compute prototype embeddings if needed
        if self.results.prototype_embeddings is None:
            self.compute_prototype_embeddings()
        
        prototype_embeddings = self.results.prototype_embeddings
        
        print(f"  Detecting narratives (threshold={threshold})...")
        
        # Compute similarities
        scores = {}
        for narrative_id, prototype_emb in prototype_embeddings.items():
            similarities = cosine_similarity(embeddings, prototype_emb.reshape(1, -1))
            scores[narrative_id] = similarities.flatten()
        
        # Create DataFrames
        narrative_scores = pd.DataFrame(scores)
        narrative_binary = (narrative_scores > threshold).astype(int)
        
        # Add to main DataFrame
        df = self.results.df
        for narrative_id in self.config.narrative_ids:
            df[f'{narrative_id}_score'] = narrative_scores[narrative_id].values
            df[narrative_id] = narrative_binary[narrative_id].values
        
        # Compute dominant narrative
        df['dominant_narrative'] = narrative_scores.idxmax(axis=1).apply(
            self.config.id_to_name
        )
        df['max_narrative_score'] = narrative_scores.max(axis=1)
        
        # Print statistics
        print(f"\n  Narrative Detection Results:")
        print(f"  {'Narrative':30s} {'Count':>6s} {'%':>7s} {'Avg Score':>10s}")
        print("  " + "-" * 55)
        
        for display_name, narrative_id in self.config.narratives.items():
            count = int(df[narrative_id].sum())
            pct = 100 * count / len(df)
            avg_score = df[f'{narrative_id}_score'].mean()
            print(f"  {display_name:30s} {count:6d} {pct:6.1f}% {avg_score:10.3f}")
        
        print(f"\n  Dominant Narrative Distribution:")
        print(df['dominant_narrative'].value_counts().to_string())
        
        self.results.narrative_scores = narrative_scores
        self.results.narrative_binary = narrative_binary
        
        return narrative_scores, narrative_binary
    
    # =========================================================================
    # STEP 3B: DISCOVERED NARRATIVE CLUSTERING
    # =========================================================================
    
    def discover_clusters(self, n_clusters: int = None, method: str = 'kmeans',
                         min_cluster_size: int = 10, 
                         time_window: str = None) -> Dict[str, Any]:
        """
        Discover narrative clusters from embeddings.
        
        Args:
            n_clusters: Number of clusters (None = auto)
            method: 'kmeans', 'dbscan', or 'hierarchical'
            min_cluster_size: Minimum articles per cluster
            time_window: Optional time filtering ('year', 'quarter', etc.)
            
        Returns:
            Dict with cluster info
        """
        print(f"\n{'='*60}")
        print("STEP 3B: Discovered Narrative Clustering")
        print('='*60)
        
        df = self.results.df
        embeddings = self.results.embeddings
        
        # Auto-select n_clusters if not specified
        if n_clusters is None:
            n_clusters = len(self.config.narratives)
        
        print(f"  Method: {method}, n_clusters={n_clusters}")
        
        # Run clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(embeddings)
            cluster_centers = clusterer.cluster_centers_
            
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=min_cluster_size)
            cluster_labels = clusterer.fit_predict(embeddings)
            cluster_centers = None
            
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(embeddings)
            # Compute centers manually
            cluster_centers = np.array([
                embeddings[cluster_labels == i].mean(axis=0)
                for i in range(n_clusters)
            ])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        df['cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_info = {}
        for i in range(max(cluster_labels) + 1):
            mask = df['cluster'] == i
            cluster_embeddings = embeddings[mask]
            cluster_texts = df[mask]['text'].tolist()
            
            if len(cluster_texts) < min_cluster_size:
                continue
            
            # Get cluster keywords
            keywords = self._extract_cluster_keywords(cluster_texts)
            
            # Get exemplar articles
            if cluster_centers is not None:
                center = cluster_centers[i]
                similarities = cosine_similarity(cluster_embeddings, center.reshape(1, -1)).flatten()
                exemplar_indices = similarities.argsort()[-5:][::-1]
                exemplars = df[mask].iloc[exemplar_indices]['title'].tolist() if 'title' in df.columns else []
            else:
                exemplars = df[mask].head(5)['title'].tolist() if 'title' in df.columns else []
            
            # Match to nearest seeded narrative
            if cluster_centers is not None and self.results.prototype_embeddings:
                narrative_match = self._match_cluster_to_narrative(cluster_centers[i])
            else:
                narrative_match = None
            
            cluster_info[i] = {
                'size': mask.sum(),
                'keywords': keywords,
                'exemplars': exemplars,
                'narrative_match': narrative_match,
                'center': cluster_centers[i] if cluster_centers is not None else None,
            }
            
            print(f"\n  Cluster {i} ({mask.sum()} articles)")
            print(f"    Keywords: {keywords}")
            if narrative_match:
                print(f"    Matches: {narrative_match['narrative']} (sim={narrative_match['similarity']:.3f})")
            if exemplars:
                print(f"    Exemplar: {exemplars[0][:80]}...")
        
        self.results.clusters = {
            'labels': cluster_labels,
            'centers': cluster_centers,
            'info': cluster_info,
            'method': method,
            'n_clusters': n_clusters,
        }
        
        return cluster_info
    
    def _extract_cluster_keywords(self, texts: List[str], n_keywords: int = 5) -> str:
        """Extract top TF-IDF keywords from cluster texts."""
        if len(texts) < 3:
            return ""
        
        custom_stopwords = list(ENGLISH_STOP_WORDS) + [
            'said', 'says', 'new', 'year', 'years', 'time', 'like', 'just',
            'people', 'would', 'could', 'also', 'one', 'two', 'first', 'last',
            'percent', 'million', 'billion', 'according', 'report', 'reported',
            'company', 'companies', 'market', 'business', 'work', 'working',
        ]
        
        try:
            vec = TfidfVectorizer(
                max_features=100,
                stop_words=custom_stopwords,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b'
            )
            tfidf = vec.fit_transform(texts)
            scores = np.array(tfidf.mean(axis=0)).flatten()
            top_idx = scores.argsort()[-n_keywords:][::-1]
            keywords = [vec.get_feature_names_out()[i] for i in top_idx]
            return ', '.join(keywords)
        except:
            return ""
    
    def _match_cluster_to_narrative(self, cluster_center: np.ndarray) -> Dict[str, Any]:
        """Match cluster center to nearest seeded narrative."""
        similarities = {}
        for narrative_id, proto_emb in self.results.prototype_embeddings.items():
            sim = cosine_similarity(
                cluster_center.reshape(1, -1),
                proto_emb.reshape(1, -1)
            )[0, 0]
            similarities[narrative_id] = sim
        
        best_id = max(similarities, key=similarities.get)
        return {
            'narrative_id': best_id,
            'narrative': self.config.id_to_name(best_id),
            'similarity': similarities[best_id],
            'all_similarities': {
                self.config.id_to_name(k): v 
                for k, v in similarities.items()
            }
        }
    
    # =========================================================================
    # STEP 4: NARRATIVE NETWORKS
    # =========================================================================
    
    def build_cooccurrence_network(self, threshold_quantile: float = 0.7,
                                   time_period: str = None) -> nx.Graph:
        """
        Build narrative co-occurrence network.
        
        Args:
            threshold_quantile: Quantile for "high" co-occurrence
            time_period: Filter to specific time period (e.g., '2023')
            
        Returns:
            NetworkX graph
        """
        print(f"\n{'='*60}")
        print("STEP 4: Narrative Co-occurrence Network")
        print('='*60)
        
        df = self.results.df
        narrative_scores = self.results.narrative_scores
        
        # Filter by time if specified
        if time_period:
            if 'year' in df.columns:
                mask = df['year'].astype(str).str.contains(str(time_period))
                df_period = df[mask]
                scores_period = narrative_scores[mask]
            else:
                df_period = df
                scores_period = narrative_scores
        else:
            df_period = df
            scores_period = narrative_scores
        
        print(f"  Articles in period: {len(df_period):,}")
        
        # Compute co-occurrence matrix
        narrative_ids = self.config.narrative_ids
        n_narratives = len(narrative_ids)
        
        # Define "high" threshold per narrative using quantile
        thresholds = {}
        for nid in narrative_ids:
            thresholds[nid] = scores_period[nid].quantile(threshold_quantile)
        
        # Build co-occurrence matrix
        cooccurrence = np.zeros((n_narratives, n_narratives))
        
        for idx in range(len(df_period)):
            # Find which narratives are "high" for this article
            high_narratives = []
            for i, nid in enumerate(narrative_ids):
                if scores_period.iloc[idx][nid] >= thresholds[nid]:
                    high_narratives.append(i)
            
            # Increment co-occurrence for all pairs
            for i in high_narratives:
                for j in high_narratives:
                    cooccurrence[i, j] += 1
        
        # Normalize by total
        cooccurrence_norm = cooccurrence / len(df_period)
        
        # Build graph
        G = nx.Graph()
        
        # Add nodes
        for nid in narrative_ids:
            display_name = self.config.id_to_name(nid)
            count = (df_period[nid] == 1).sum() if nid in df_period.columns else 0
            G.add_node(display_name, 
                      narrative_id=nid,
                      count=count,
                      color=self.config.get_color(nid))
        
        # Add edges (only above diagonal, exclude self-loops)
        for i in range(n_narratives):
            for j in range(i + 1, n_narratives):
                weight = cooccurrence_norm[i, j]
                if weight > 0.01:  # Minimum edge weight
                    G.add_edge(
                        self.config.id_to_name(narrative_ids[i]),
                        self.config.id_to_name(narrative_ids[j]),
                        weight=weight,
                        raw_count=cooccurrence[i, j]
                    )
        
        # Compute centrality
        if len(G.edges) > 0:
            centrality = nx.degree_centrality(G)
            nx.set_node_attributes(G, centrality, 'centrality')
        
        print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Store results
        self.results.networks[time_period or 'all'] = {
            'graph': G,
            'cooccurrence_matrix': cooccurrence,
            'cooccurrence_normalized': cooccurrence_norm,
            'thresholds': thresholds,
        }
        
        return G
    
    def compare_networks(self, period1: str, period2: str) -> Dict[str, Any]:
        """
        Compare network structures between two periods.
        
        Args:
            period1: First time period identifier
            period2: Second time period identifier
            
        Returns:
            Dict with comparison metrics
        """
        print(f"\n  Comparing networks: {period1} vs {period2}")
        
        # Build networks if needed
        if period1 not in self.results.networks:
            self.build_cooccurrence_network(time_period=period1)
        if period2 not in self.results.networks:
            self.build_cooccurrence_network(time_period=period2)
        
        G1 = self.results.networks[period1]['graph']
        G2 = self.results.networks[period2]['graph']
        
        # Edge changes
        edges1 = set(G1.edges())
        edges2 = set(G2.edges())
        
        new_edges = edges2 - edges1
        lost_edges = edges1 - edges2
        
        # Centrality changes
        cent1 = nx.degree_centrality(G1)
        cent2 = nx.degree_centrality(G2)
        
        centrality_changes = {}
        for node in set(cent1.keys()) | set(cent2.keys()):
            c1 = cent1.get(node, 0)
            c2 = cent2.get(node, 0)
            centrality_changes[node] = c2 - c1
        
        comparison = {
            'period1': period1,
            'period2': period2,
            'new_edges': list(new_edges),
            'lost_edges': list(lost_edges),
            'centrality_changes': centrality_changes,
            'top_gainers': sorted(centrality_changes.items(), key=lambda x: x[1], reverse=True)[:3],
            'top_losers': sorted(centrality_changes.items(), key=lambda x: x[1])[:3],
        }
        
        print(f"    New edges: {len(new_edges)}")
        print(f"    Lost edges: {len(lost_edges)}")
        print(f"    Top centrality gainers: {comparison['top_gainers']}")
        
        return comparison
    
    # =========================================================================
    # STEP 5: TEMPORAL ANALYSIS
    # =========================================================================
    
    def compute_temporal_prevalence(self, time_unit: str = 'month') -> pd.DataFrame:
        """
        Compute narrative prevalence over time.
        
        Args:
            time_unit: 'month', 'quarter', or 'year'
            
        Returns:
            DataFrame with prevalence by time period
        """
        print(f"\n{'='*60}")
        print("STEP 5: Temporal Analysis")
        print('='*60)
        
        df = self.results.df
        
        # Create time grouping
        if time_unit == 'month':
            df['time_group'] = df['datetime'].dt.to_period('M')
        elif time_unit == 'quarter':
            df['time_group'] = df['datetime'].dt.to_period('Q')
        elif time_unit == 'year':
            df['time_group'] = df['datetime'].dt.to_period('Y')
        else:
            raise ValueError(f"Unknown time_unit: {time_unit}")
        
        # Calculate prevalence per period
        prevalence_data = []
        
        for period, group in df.groupby('time_group'):
            row = {'period': period, 'article_count': len(group)}
            
            for narrative_id in self.config.narrative_ids:
                display_name = self.config.id_to_name(narrative_id)
                
                # Count-based prevalence
                if narrative_id in group.columns:
                    count = group[narrative_id].sum()
                    row[f'{display_name}_count'] = count
                    row[f'{display_name}_pct'] = 100 * count / len(group)
                
                # Average score
                score_col = f'{narrative_id}_score'
                if score_col in group.columns:
                    row[f'{display_name}_avg_score'] = group[score_col].mean()
            
            prevalence_data.append(row)
        
        prevalence_df = pd.DataFrame(prevalence_data)
        prevalence_df = prevalence_df.sort_values('period')
        
        self.results.temporal['prevalence'] = prevalence_df
        
        print(f"  Computed prevalence for {len(prevalence_df)} time periods")
        
        return prevalence_df
    
    def detect_shift_periods(self, window: int = 3) -> List[Dict[str, Any]]:
        """
        Detect periods with significant narrative distribution shifts.
        
        Uses Jensen-Shannon divergence between consecutive periods.
        
        Args:
            window: Smoothing window size
            
        Returns:
            List of shift periods with details
        """
        print(f"\n  Detecting narrative shift periods...")
        
        if 'prevalence' not in self.results.temporal:
            self.compute_temporal_prevalence()
        
        prevalence_df = self.results.temporal['prevalence']
        
        # Get percentage columns
        pct_cols = [c for c in prevalence_df.columns if c.endswith('_pct')]
        
        if len(pct_cols) == 0:
            print("    No prevalence data found")
            return []
        
        # Compute JS divergence between consecutive periods
        divergences = []
        
        for i in range(1, len(prevalence_df)):
            prev_dist = prevalence_df.iloc[i-1][pct_cols].values.astype(float) / 100
            curr_dist = prevalence_df.iloc[i][pct_cols].values.astype(float) / 100
            
            # Avoid zeros and NaN
            prev_dist = np.nan_to_num(prev_dist, nan=0.0)
            curr_dist = np.nan_to_num(curr_dist, nan=0.0)
            prev_dist = np.clip(prev_dist, 1e-10, 1)
            curr_dist = np.clip(curr_dist, 1e-10, 1)
            
            # Normalize
            prev_dist = prev_dist / prev_dist.sum()
            curr_dist = curr_dist / curr_dist.sum()
            
            # Jensen-Shannon divergence
            m = 0.5 * (prev_dist + curr_dist)
            js_div = 0.5 * (entropy(prev_dist, m) + entropy(curr_dist, m))
            
            divergences.append({
                'period': prevalence_df.iloc[i]['period'],
                'js_divergence': js_div,
                'prev_period': prevalence_df.iloc[i-1]['period'],
            })
        
        div_df = pd.DataFrame(divergences)
        
        # Find shift periods (above 75th percentile)
        threshold = div_df['js_divergence'].quantile(0.75)
        shifts = div_df[div_df['js_divergence'] > threshold].copy()
        
        # Add details about what changed
        shift_details = []
        for _, row in shifts.iterrows():
            period = row['period']
            prev_period = row['prev_period']
            
            prev_row = prevalence_df[prevalence_df['period'] == prev_period].iloc[0]
            curr_row = prevalence_df[prevalence_df['period'] == period].iloc[0]
            
            changes = {}
            for col in pct_cols:
                name = col.replace('_pct', '')
                change = curr_row[col] - prev_row[col]
                if abs(change) > 5:  # Significant change
                    changes[name] = change
            
            shift_details.append({
                'period': str(period),
                'prev_period': str(prev_period),
                'js_divergence': row['js_divergence'],
                'major_changes': changes,
            })
        
        self.results.temporal['shifts'] = shift_details
        
        print(f"    Found {len(shift_details)} shift periods")
        for shift in shift_details[:3]:
            print(f"      {shift['period']}: JS={shift['js_divergence']:.3f}")
            for name, change in list(shift['major_changes'].items())[:2]:
                direction = "↑" if change > 0 else "↓"
                print(f"        {name}: {direction}{abs(change):.1f}%")
        
        return shift_details
    
    def compute_narrative_drift(self, periods: List[Tuple[int, int, str]] = None) -> Dict[str, Any]:
        """
        Compute semantic drift of narratives across time periods.
        
        Args:
            periods: List of (start_year, end_year, label) tuples
            
        Returns:
            Dict with drift metrics
        """
        print(f"\n  Computing narrative semantic drift...")
        
        df = self.results.df
        embeddings = self.results.embeddings
        
        # Default periods if not specified
        if periods is None:
            years = df['year'].dropna().unique()
            if len(years) >= 4:
                mid = (min(years) + max(years)) // 2
                periods = [
                    (min(years), mid - 1, 'Early'),
                    (mid, max(years), 'Recent'),
                ]
            else:
                periods = [(min(years), max(years), 'All')]
        
        drift_results = {}
        
        for narrative_id in self.config.narrative_ids:
            display_name = self.config.id_to_name(narrative_id)
            
            period_centroids = []
            period_labels = []
            
            for start_year, end_year, label in periods:
                mask = (
                    (df['year'] >= start_year) & 
                    (df['year'] <= end_year) &
                    (df[narrative_id] == 1 if narrative_id in df.columns else True)
                )
                
                if mask.sum() > 10:
                    period_embs = embeddings[mask]
                    centroid = period_embs.mean(axis=0)
                    period_centroids.append(centroid)
                    period_labels.append(label)
            
            if len(period_centroids) >= 2:
                # Compute drift between consecutive periods
                drifts = []
                for i in range(len(period_centroids) - 1):
                    sim = cosine_similarity(
                        period_centroids[i].reshape(1, -1),
                        period_centroids[i+1].reshape(1, -1)
                    )[0, 0]
                    drifts.append({
                        'from': period_labels[i],
                        'to': period_labels[i+1],
                        'similarity': sim,
                        'drift': 1 - sim,
                    })
                
                drift_results[display_name] = {
                    'period_labels': period_labels,
                    'drifts': drifts,
                    'total_drift': sum(d['drift'] for d in drifts),
                }
        
        self.results.temporal['drift'] = drift_results
        
        # Print summary
        print(f"\n    Narrative Drift Summary:")
        for name, data in drift_results.items():
            if data['drifts']:
                total = data['total_drift']
                print(f"      {name}: total drift = {total:.4f}")
        
        return drift_results
    
    # =========================================================================
    # PROJECTIONS (PCA, t-SNE)
    # =========================================================================
    
    def compute_projections(self, methods: List[str] = ['pca', 'tsne']) -> Dict[str, np.ndarray]:
        """
        Compute 2D projections for visualization.
        
        Args:
            methods: List of methods ('pca', 'tsne')
            
        Returns:
            Dict mapping method name to 2D coordinates
        """
        print(f"\n  Computing projections...")
        
        embeddings = self.results.embeddings
        df = self.results.df
        
        projections = {}
        
        if 'pca' in methods:
            print("    Running PCA...")
            pca = PCA(n_components=2, random_state=42)
            pca_results = pca.fit_transform(embeddings)
            df['pca_x'] = pca_results[:, 0]
            df['pca_y'] = pca_results[:, 1]
            projections['pca'] = pca_results
            projections['pca_variance'] = pca.explained_variance_ratio_
        
        if 'tsne' in methods:
            print("    Running t-SNE...")
            # Use PCA to reduce dimensionality first (like working code)
            if embeddings.shape[1] > 50:
                pca_pre = PCA(n_components=50, random_state=123)
                embeddings_reduced = pca_pre.fit_transform(embeddings)
            else:
                embeddings_reduced = embeddings
            
            # Adjust perplexity for dataset size (like working code)
            n_samples = len(embeddings_reduced)
            perplexity = min(30, n_samples // 4)
            perplexity = max(5, perplexity)  # Ensure minimum
            
            tsne = TSNE(n_components=2, random_state=123, perplexity=perplexity, max_iter=500, verbose=1)
            tsne_results = tsne.fit_transform(embeddings_reduced)
            df['tsne_x'] = tsne_results[:, 0]
            df['tsne_y'] = tsne_results[:, 1]
            projections['tsne'] = tsne_results
        
        self.results.projections = projections
        
        return projections
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def plot_embedding_scatter(self, method: str = 'tsne', 
                               color_by: str = 'dominant_narrative',
                               figsize: Tuple[int, int] = (14, 10),
                               save_path: str = None) -> plt.Figure:
        """
        Plot embedding scatter colored by narrative.
        
        Args:
            method: 'pca' or 'tsne'
            color_by: Column to color by
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        df = self.results.df
        
        x_col = f'{method}_x'
        y_col = f'{method}_y'
        
        if x_col not in df.columns:
            self.compute_projections([method])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for narrative in df[color_by].unique():
            mask = df[color_by] == narrative
            color = self.config.get_color(narrative)
            ax.scatter(
                df[mask][x_col],
                df[mask][y_col],
                alpha=0.7,
                s=60,
                c=color,
                label=f"{narrative} ({mask.sum()})",
                edgecolors='black',
                linewidth=0.5
            )
        
        ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
        ax.set_title(f'{self.config.name}: Article Embeddings by {color_by}', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        return fig
    
    def plot_narrative_network(self, period: str = None,
                               figsize: Tuple[int, int] = (14, 12),
                               save_path: str = None) -> plt.Figure:
        """
        Plot narrative co-occurrence network.
        
        Args:
            period: Time period (None = all data)
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        period_key = period or 'all'
        
        if period_key not in self.results.networks:
            self.build_cooccurrence_network(time_period=period)
        
        G = self.results.networks[period_key]['graph']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Node colors and sizes
        node_colors = [self.config.get_color(n) for n in G.nodes()]
        node_sizes = [G.nodes[n].get('count', 100) * 10 + 500 for n in G.nodes()]
        
        # Edge weights
        edge_weights = [G[u][v]['weight'] * 10 for u, v in G.edges()]
        
        # Draw
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, 
                               width=edge_weights, edge_color='gray')
        nx.draw_networkx_nodes(G, pos, ax=ax, 
                               node_color=node_colors,
                               node_size=node_sizes,
                               edgecolors='black',
                               linewidths=2)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
        
        period_label = f" ({period})" if period else ""
        ax.set_title(f'{self.config.name}: Narrative Co-occurrence Network{period_label}',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        return fig
    
    def plot_temporal_prevalence(self, metric: str = 'pct',
                                 figsize: Tuple[int, int] = (16, 8),
                                 save_path: str = None) -> plt.Figure:
        """
        Plot narrative prevalence over time.
        
        Args:
            metric: 'pct' or 'avg_score'
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        if 'prevalence' not in self.results.temporal:
            self.compute_temporal_prevalence()
        
        prevalence_df = self.results.temporal['prevalence']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        suffix = '_pct' if metric == 'pct' else '_avg_score'
        
        for display_name in self.config.display_names:
            col = f'{display_name}{suffix}'
            if col in prevalence_df.columns:
                color = self.config.get_color(display_name)
                ax.plot(prevalence_df['period'].astype(str), 
                       prevalence_df[col],
                       label=display_name,
                       color=color,
                       linewidth=2,
                       marker='o',
                       markersize=4)
        
        ax.set_xlabel('Time Period', fontsize=12)
        ylabel = 'Prevalence (%)' if metric == 'pct' else 'Average Similarity Score'
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{self.config.name}: Narrative Prevalence Over Time',
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Rotate x labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        return fig
    
    def plot_score_distributions(self, figsize: Tuple[int, int] = (16, 10),
                                 save_path: str = None) -> plt.Figure:
        """
        Plot similarity score distributions for each narrative.
        
        Args:
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        df = self.results.df
        n_narratives = len(self.config.narratives)
        
        n_cols = 3
        n_rows = (n_narratives + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (display_name, narrative_id) in enumerate(self.config.narratives.items()):
            ax = axes[idx]
            score_col = f'{narrative_id}_score'
            
            if score_col in df.columns:
                scores = df[score_col]
                color = self.config.get_color(display_name)
                
                ax.hist(scores, bins=50, alpha=0.7, color=color, edgecolor='black')
                ax.axvline(self.config.default_threshold, color='red', 
                          linestyle='--', linewidth=2, label=f'Threshold ({self.config.default_threshold})')
                ax.set_xlabel('Similarity Score')
                ax.set_ylabel('Frequency')
                ax.set_title(display_name, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(len(self.config.narratives), len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f'{self.config.name}: Narrative Score Distributions',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        return fig
    
    def plot_prototype_similarity_matrix(self, figsize: Tuple[int, int] = (12, 10),
                                         save_path: str = None) -> plt.Figure:
        """
        Plot heatmap of similarity between narrative prototypes.
        
        Args:
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        if self.results.prototype_embeddings is None:
            self.compute_prototype_embeddings()
        
        narrative_ids = self.config.narrative_ids
        n = len(narrative_ids)
        
        # Compute similarity matrix
        sim_matrix = np.zeros((n, n))
        for i, id1 in enumerate(narrative_ids):
            for j, id2 in enumerate(narrative_ids):
                emb1 = self.results.prototype_embeddings[id1].reshape(1, -1)
                emb2 = self.results.prototype_embeddings[id2].reshape(1, -1)
                sim_matrix[i, j] = cosine_similarity(emb1, emb2)[0, 0]
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        labels = [self.config.id_to_name(nid) for nid in narrative_ids]
        
        sns.heatmap(sim_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                   xticklabels=labels, yticklabels=labels,
                   square=True, linewidths=1,
                   cbar_kws={'label': 'Cosine Similarity'},
                   ax=ax)
        
        ax.set_title(f'{self.config.name}: Narrative Prototype Similarity',
                    fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        return fig
    
    # =========================================================================
    # FULL ANALYSIS
    # =========================================================================
    
    def run_full_analysis(self, cache_file: str = None,
                         compute_clusters: bool = True,
                         compute_networks: bool = True,
                         compute_temporal: bool = True,
                         compute_tsne: bool = True) -> AnalysisResults:
        """
        Run complete analysis pipeline.
        
        Args:
            cache_file: Path to cache embeddings
            compute_clusters: Whether to run discovered clustering
            compute_networks: Whether to build co-occurrence networks
            compute_temporal: Whether to compute temporal analysis
            
        Returns:
            AnalysisResults object
        """
        print(f"\n{'='*60}")
        print(f"RUNNING FULL ANALYSIS: {self.config.name}")
        print('='*60)
        
        # Step 2: Embeddings
        self.compute_embeddings(cache_file=cache_file)
        
        # Step 3A: Seeded narratives
        self.detect_seeded_narratives()
        
        # Step 3B: Discovered clusters
        if compute_clusters:
            self.discover_clusters()
        
        # Projections for visualization
        if compute_tsne:
            self.compute_projections(methods=['pca', 'tsne'])
        else:
            self.compute_projections(methods=['pca'])
        
        # Step 4: Networks
        if compute_networks:
            self.build_cooccurrence_network()
        
        # Step 5: Temporal
        if compute_temporal and 'datetime' in self.results.df.columns:
            self.compute_temporal_prevalence()
            self.detect_shift_periods()
            self.compute_narrative_drift()
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print('='*60)
        
        return self.results
    
    def generate_report(self, output_dir: str, show_plots: bool = True) -> Dict[str, str]:
        """
        Generate comprehensive analysis report with visualizations.
        
        Args:
            output_dir: Directory to save outputs
            show_plots: Whether to display plots
            
        Returns:
            Dict mapping output names to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        outputs = {}
        
        print(f"\nGenerating report in {output_dir}...")
        
        # Visualizations
        fig = self.plot_embedding_scatter('tsne')
        save_path = output_path / 'embedding_scatter_tsne.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        outputs['embedding_scatter'] = str(save_path)
        if show_plots:
            plt.show()
        plt.close()
        
        fig = self.plot_score_distributions()
        save_path = output_path / 'score_distributions.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        outputs['score_distributions'] = str(save_path)
        if show_plots:
            plt.show()
        plt.close()
        
        fig = self.plot_prototype_similarity_matrix()
        save_path = output_path / 'prototype_similarity.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        outputs['prototype_similarity'] = str(save_path)
        if show_plots:
            plt.show()
        plt.close()
        
        if self.results.networks:
            fig = self.plot_narrative_network()
            save_path = output_path / 'narrative_network.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            outputs['narrative_network'] = str(save_path)
            if show_plots:
                plt.show()
            plt.close()
        
        if 'prevalence' in self.results.temporal:
            fig = self.plot_temporal_prevalence()
            save_path = output_path / 'temporal_prevalence.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            outputs['temporal_prevalence'] = str(save_path)
            if show_plots:
                plt.show()
            plt.close()
        
        # Export data
        self.results.df.to_csv(output_path / 'articles_with_narratives.csv', index=False)
        outputs['articles_csv'] = str(output_path / 'articles_with_narratives.csv')
        
        if 'prevalence' in self.results.temporal:
            self.results.temporal['prevalence'].to_csv(
                output_path / 'temporal_prevalence.csv', index=False
            )
            outputs['prevalence_csv'] = str(output_path / 'temporal_prevalence.csv')
        
        # Summary JSON
        summary = {
            'domain': self.config.name,
            'total_articles': len(self.results.df),
            'narrative_counts': {
                self.config.id_to_name(nid): int(self.results.df[nid].sum())
                for nid in self.config.narrative_ids
                if nid in self.results.df.columns
            },
            'dominant_narrative_distribution': self.results.df['dominant_narrative'].value_counts().to_dict(),
        }
        
        if 'shifts' in self.results.temporal:
            summary['shift_periods'] = self.results.temporal['shifts']
        
        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        outputs['summary_json'] = str(output_path / 'summary.json')
        
        print(f"  ✓ Report generated with {len(outputs)} files")
        
        return outputs


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def analyze_narratives(data_path: str, domain: str, 
                       cache_dir: str = './cache',
                       output_dir: str = './output') -> NarrativePipeline:
    """
    Quick function to run full narrative analysis.
    
    Args:
        data_path: Path to data file
        domain: Domain name ('ev', 'aitech', 'retail')
        cache_dir: Directory for embedding cache
        output_dir: Directory for outputs
        
    Returns:
        NarrativePipeline with completed analysis
    """
    config = get_config(domain)
    
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = Path(cache_dir) / f"{domain}_embeddings.pkl"
    
    pipeline = NarrativePipeline(config)
    pipeline.load_data(data_path)
    pipeline.run_full_analysis(cache_file=str(cache_file))
    pipeline.generate_report(output_dir)
    
    return pipeline


if __name__ == '__main__':
    print("Narrative Analysis Pipeline")
    print("=" * 50)
    print("\nUsage:")
    print("  from narrative_pipeline import NarrativePipeline")
    print("  from narrative_config import EV_CONFIG")
    print("")
    print("  pipeline = NarrativePipeline(EV_CONFIG)")
    print("  pipeline.load_data('unified_news.csv')")
    print("  pipeline.run_full_analysis()")
    print("  pipeline.generate_report('output/')")