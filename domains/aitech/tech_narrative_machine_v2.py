"""
Tech Narrative Machine — Full Pipeline + Extensions
=====================================================
Complete orchestrator showing the base narrative pipeline plus all extensions:
  1. BERTopic integration (auto topic discovery + temporal tracking)
  2. Sentiment / Stance layer (framing analysis per narrative over time)
  3. Spike detection (z-score flagging of narrative surges)
  4. Improved network visualization (reference-image aesthetic)

Usage:
    python tech_narrative_machine.py

Prerequisites:
    pip install sentence-transformers bertopic hdbscan textblob python-louvain --break-system-packages
    python -m textblob.download_corpora
"""

import matplotlib
matplotlib.use('Agg')      # non-interactive backend for server/batch runs
import matplotlib.pyplot as plt
from pathlib import Path

# ── Base pipeline imports ────────────────────────────────────────────────────
from canonical_news_scraper import CanonicalDatasetBuilder
from narrative_pipeline_v2 import NarrativePipeline
from narrative_config import AITECH_CONFIG
from narrative_visualizations import (
    plot_alluvial_diagram,
    plot_semantic_network,
    plot_semantic_strength_over_time,
    plot_expanded_network_v2,
    generate_cluster_narrative_report,
    plot_dense_cluster_network,
    plot_tsne_centroids_timeline,
)

# ── Extension imports ────────────────────────────────────────────────────────
from narrative_extensions import (
    NarrativeExtensions,
    # BERTopic visualizations
    plot_bertopic_overview,
    plot_bertopic_over_time,
    # Sentiment visualizations
    plot_sentiment_by_narrative,
    plot_stance_over_time,
    plot_stance_heatmap,
    # Spike visualizations
    plot_prevalence_with_spikes,
    plot_spike_z_scores,
    generate_spike_report,
)

# ── Improved network visualization ──────────────────────────────────────────
from narrative_network_improved import (
    plot_narrative_network_v3,
    plot_narrative_network_louvain,
)


# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)

# Choose sentiment method:
#   'textblob'    — fast, no GPU, good for titles/short text
#   'transformer' — slower, more accurate, better for full text
SENTIMENT_METHOD = 'textblob'


# =============================================================================
# STEP 1: DATA INGESTION
# =============================================================================

print("\n" + "="*60)
print("STEP 1: Data Ingestion & Canonicalization")
print("="*60)

builder = CanonicalDatasetBuilder()
builder.load_nyt_csv('nyt_aitech_articles.csv', topic_label='tech')
builder.load_gdelt_csv('historical_news_tech.csv', topic_label='tech')
unified = builder.build_unified_dataset()
builder.export_unified('unified_tech.csv')


# =============================================================================
# STEPS 2-5: BASE NARRATIVE ANALYSIS
# =============================================================================

pipeline = NarrativePipeline(AITECH_CONFIG)
pipeline.load_data('unified_tech.csv')
pipeline.run_full_analysis(cache_file='tech_embeddings.pkl', compute_tsne=False)


# =============================================================================
# STEP 6: BASE VISUALIZATIONS
# =============================================================================

print("\n" + "="*60)
print("STEP 6: Base Visualizations")
print("="*60)

# 6a. Alluvial diagram — narrative flow over time
plot_alluvial_diagram(pipeline, save_path=str(OUTPUT_DIR / 'tech_alluvial.png'))

# 6b. Semantic network with convex hulls (overall + per-year)
plot_semantic_network(pipeline, n_subclusters=15,
                      save_path=str(OUTPUT_DIR / 'tech_semantic_network.png'))

for year in [2020, 2021, 2022, 2024, 2025]:
    plot_semantic_network(pipeline, time_period=year,
                          save_path=str(OUTPUT_DIR / f'tech_semantic_{year}.png'))

# 6c. Semantic strength over time
plot_semantic_strength_over_time(pipeline,
                                 save_path=str(OUTPUT_DIR / 'tech_strength.png'))

# 6d. Expanded narrative network (seeded + sub-clusters)
plot_expanded_network_v2(pipeline, n_subclusters=15,
                         save_path=str(OUTPUT_DIR / 'tech_network_expanded.png'))

# 6e. t-SNE centroids timeline
plot_tsne_centroids_timeline(pipeline,
                              save_path=str(OUTPUT_DIR / 'tsne_timeline.png'))

# 6f. Cluster-narrative documentation report
report = generate_cluster_narrative_report(pipeline, text_col="title",
                                           save_path=str(OUTPUT_DIR / 'cluster_report.md'))


# =============================================================================
# STEP 7: IMPROVED NETWORK VISUALIZATION (reference-image style)
# =============================================================================

print("\n" + "="*60)
print("STEP 7: Improved Network Visualization")
print("="*60)

# 7a. KMeans variant — full corpus
plot_narrative_network_v3(
    pipeline,
    n_clusters=18,               # adjust for your corpus size
    text_col='title',
    knn_k=8,                     # nearest neighbors for edge construction
    similarity_threshold=0.28,   # lower = more edges, denser graph
    show_narrative=True,         # show dominant narrative per cluster
    save_path=str(OUTPUT_DIR / 'tech_network_v3.png'),
    dpi=200,
)

# 7b. Louvain variant — full corpus
try:
    plot_narrative_network_louvain(
        pipeline,
        text_col='title',
        knn_k=10,
        similarity_threshold=0.25,
        resolution=1.2,          # higher = more communities
        show_narrative=True,     # show dominant narrative per community
        save_path=str(OUTPUT_DIR / 'tech_network_louvain.png'),
        dpi=200,
    )
except ImportError:
    print("  ⚠ Skipping Louvain network (pip install python-louvain)")

# 7c. Yearly Louvain networks — narrative evolution over time
print("\n  Generating yearly Louvain networks...")
years = sorted(pipeline.results.df['year'].dropna().unique())
for yr in years:
    yr_count = (pipeline.results.df['year'] == yr).sum()
    if yr_count < 10:  # skip years with too few articles
        print(f"    Skipping {yr} ({yr_count} articles)")
        continue
    try:
        plot_narrative_network_louvain(
            pipeline,
            text_col='title',
            knn_k=8,
            similarity_threshold=0.25,
            resolution=1.0,
            show_narrative=True,
            time_period=yr,
            time_col='year',
            save_path=str(OUTPUT_DIR / f'tech_network_louvain_{int(yr)}.png'),
            dpi=200,
        )
    except Exception as e:
        print(f"    ⚠ Skipping {yr}: {e}")


# =============================================================================
# STEP 8: EXTENSIONS — BERTopic + Sentiment + Spikes
# =============================================================================

print("\n" + "="*60)
print("STEP 8: Running Pipeline Extensions")
print("="*60)

ext = NarrativeExtensions(pipeline, sentiment_method=SENTIMENT_METHOD)

# ── 8a. BERTopic ─────────────────────────────────────────────────────────────
# Discovers topics automatically from embeddings, labels them with c-TF-IDF,
# and cross-references against your seeded narratives.
ext.run_bertopic(
    text_col='text',             # full text for better topic quality
    min_topic_size=15,           # minimum docs per topic
    reduce_outliers=True,        # reassign noise docs to nearest topic
    cache_file='tech_bertopic_model',
)

# Track how BERTopic topics evolve over time
ext.bertopic_topics_over_time(text_col='text')

# Optional: build topic hierarchy
ext.bertopic_hierarchy(text_col='text')


# ── 8b. Sentiment / Stance ──────────────────────────────────────────────────
# Scores every article's sentiment, then aggregates per narrative per period
# to reveal stance drift (e.g., "AI coverage getting more negative over time").
ext.compute_sentiment(
    text_col='title',            # titles are faster; use 'text' for more accuracy
    cache_file='tech_sentiment_cache.pkl',
)


# ── 8c. Spike Detection ─────────────────────────────────────────────────────
# Finds periods where a single narrative surged above its rolling baseline.
# Returns the top articles from each spike for "what happened?" context.
ext.detect_narrative_spikes(
    z_threshold=1.5,             # 1.5 = sensitive; 2.0 = conservative
    rolling_window=4,            # baseline window (in time_col units)
    time_col='year',             # or 'year_month' for finer granularity
    top_articles_per_spike=5,
)


# =============================================================================
# STEP 9: EXTENSION VISUALIZATIONS
# =============================================================================

print("\n" + "="*60)
print("STEP 9: Extension Visualizations")
print("="*60)

# 9a. BERTopic overview (topic sizes + mapping to seeded narratives)
fig = plot_bertopic_overview(pipeline)
fig.savefig(OUTPUT_DIR / 'bertopic_overview.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ bertopic_overview.png")

# 9b. BERTopic topics over time
fig = plot_bertopic_over_time(pipeline, top_n_topics=8)
fig.savefig(OUTPUT_DIR / 'bertopic_over_time.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ bertopic_over_time.png")

# 9c. Sentiment distribution per narrative
fig = plot_sentiment_by_narrative(pipeline)
fig.savefig(OUTPUT_DIR / 'sentiment_by_narrative.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ sentiment_by_narrative.png")

# 9d. Stance drift over time (mean sentiment per narrative per period)
fig = plot_stance_over_time(pipeline)
fig.savefig(OUTPUT_DIR / 'stance_over_time.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ stance_over_time.png")

# 9e. Stance heatmap (period × narrative, red=negative, blue=positive)
fig = plot_stance_heatmap(pipeline)
fig.savefig(OUTPUT_DIR / 'stance_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ stance_heatmap.png")

# 9f. Prevalence plot with spike annotations
fig = plot_prevalence_with_spikes(pipeline)
fig.savefig(OUTPUT_DIR / 'prevalence_with_spikes.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ prevalence_with_spikes.png")

# 9g. Spike z-score heatmap
fig = plot_spike_z_scores(pipeline)
fig.savefig(OUTPUT_DIR / 'spike_z_scores.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ spike_z_scores.png")

# 9h. Spike report (markdown with representative articles)
spike_report = generate_spike_report(pipeline)
with open(OUTPUT_DIR / 'spike_report.md', 'w') as f:
    f.write(spike_report)
print(f"  ✓ spike_report.md")


# =============================================================================
# STEP 10: FULL EXTENSION REPORT (all-in-one)
# =============================================================================

# This is an alternative to steps 9a-9h — it generates everything at once:
# ext.generate_extension_report(str(OUTPUT_DIR))


# =============================================================================
# DONE
# =============================================================================

print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")
print(f"\nGenerated files:")
for f in sorted(OUTPUT_DIR.glob('*')):
    print(f"  {f.name}")