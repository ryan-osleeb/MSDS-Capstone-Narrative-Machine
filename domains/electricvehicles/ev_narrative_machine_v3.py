"""
EV Narrative Machine — Full Pipeline + Extensions
===================================================
Complete orchestrator for Electric Vehicles domain.
Mirrors tech_narrative_machine_v2.py structure.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Base pipeline imports ────────────────────────────────────────────────────
from canonical_news_scraper import CanonicalDatasetBuilder
from narrative_pipeline_v2 import NarrativePipeline
from narrative_config import EV_CONFIG
from narrative_visualizations import (
    plot_alluvial_diagram,
    plot_semantic_network,
    plot_semantic_strength_over_time,
    plot_expanded_network_v2,
    generate_cluster_narrative_report,
    plot_tsne_centroids_timeline,
)

# ── Extension imports ────────────────────────────────────────────────────────
from narrative_extensions import (
    NarrativeExtensions,
    plot_bertopic_overview,
    plot_bertopic_over_time,
    plot_sentiment_by_narrative,
    plot_stance_over_time,
    plot_stance_heatmap,
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
SENTIMENT_METHOD = 'textblob'


# =============================================================================
# STEP 1: DATA INGESTION
# =============================================================================

print("\n" + "="*60)
print("STEP 1: Data Ingestion & Canonicalization")
print("="*60)

builder = CanonicalDatasetBuilder()
builder.load_nyt_csv('nyt_ev_articles.csv', topic_label='EV')
builder.load_gdelt_csv('historical_news_evs.csv', topic_label='EV')
unified = builder.build_unified_dataset()
builder.export_unified('unified_ev.csv')


# =============================================================================
# STEPS 2-5: BASE NARRATIVE ANALYSIS
# =============================================================================

pipeline = NarrativePipeline(EV_CONFIG)
pipeline.load_data('unified_ev.csv')
pipeline.run_full_analysis(cache_file='ev_embeddings.pkl', compute_tsne=False)


# =============================================================================
# STEP 6: BASE VISUALIZATIONS
# =============================================================================

print("\n" + "="*60)
print("STEP 6: Base Visualizations")
print("="*60)

plot_alluvial_diagram(pipeline, save_path=str(OUTPUT_DIR / 'ev_alluvial.png'))

plot_semantic_network(pipeline, n_subclusters=15,
                      save_path=str(OUTPUT_DIR / 'ev_semantic_network.png'))

for year in [2020, 2022, 2024]:
    plot_semantic_network(pipeline, time_period=year,
                          save_path=str(OUTPUT_DIR / f'ev_semantic_{year}.png'))

plot_semantic_strength_over_time(pipeline,
                                 save_path=str(OUTPUT_DIR / 'ev_strength.png'))

plot_expanded_network_v2(pipeline, n_subclusters=15,
                         save_path=str(OUTPUT_DIR / 'ev_network_expanded.png'))

plot_tsne_centroids_timeline(pipeline,
                              save_path=str(OUTPUT_DIR / 'tsne_timeline.png'))

report = generate_cluster_narrative_report(pipeline, text_col="title",
                                           save_path=str(OUTPUT_DIR / 'cluster_report.md'))


# =============================================================================
# STEP 7: IMPROVED NETWORK VISUALIZATION
# =============================================================================

print("\n" + "="*60)
print("STEP 7: Improved Network Visualization")
print("="*60)

# 7a. KMeans — full corpus
plot_narrative_network_v3(
    pipeline,
    n_clusters=18,
    text_col='title',
    knn_k=8,
    similarity_threshold=0.28,
    show_narrative=True,
    save_path=str(OUTPUT_DIR / 'ev_network_v3.png'),
    dpi=200,
)

# 7b. Louvain — full corpus
try:
    plot_narrative_network_louvain(
        pipeline,
        text_col='title',
        knn_k=10,
        similarity_threshold=0.25,
        resolution=1.2,
        show_narrative=True,
        save_path=str(OUTPUT_DIR / 'ev_network_louvain.png'),
        dpi=200,
    )
except ImportError:
    print("  ⚠ Skipping Louvain network (pip install python-louvain)")

# 7c. Yearly Louvain networks
print("\n  Generating yearly Louvain networks...")
years = sorted(pipeline.results.df['year'].dropna().unique())
for yr in years:
    yr_count = (pipeline.results.df['year'] == yr).sum()
    if yr_count < 10:
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
            save_path=str(OUTPUT_DIR / f'ev_network_louvain_{int(yr)}.png'),
            dpi=200,
        )
    except Exception as e:
        print(f"    ⚠ Skipping {yr}: {e}")


# =============================================================================
# STEP 8: EXTENSIONS
# =============================================================================

print("\n" + "="*60)
print("STEP 8: Running Pipeline Extensions")
print("="*60)

ext = NarrativeExtensions(pipeline, sentiment_method=SENTIMENT_METHOD)

# 8a. BERTopic
ext.run_bertopic(
    text_col='text',
    min_topic_size=15,
    reduce_outliers=True,
    cache_file='ev_bertopic_model',
)
ext.bertopic_topics_over_time(text_col='text')

# 8b. Sentiment
ext.compute_sentiment(
    text_col='title',
    cache_file='ev_sentiment_cache.pkl',
)

# 8c. Spike Detection
ext.detect_narrative_spikes(
    z_threshold=1.5,
    rolling_window=4,
    time_col='year',
    top_articles_per_spike=5,
)


# =============================================================================
# STEP 9: EXTENSION VISUALIZATIONS
# =============================================================================

print("\n" + "="*60)
print("STEP 9: Extension Visualizations")
print("="*60)

fig = plot_bertopic_overview(pipeline)
fig.savefig(OUTPUT_DIR / 'bertopic_overview.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ bertopic_overview.png")

fig = plot_bertopic_over_time(pipeline, top_n_topics=8)
fig.savefig(OUTPUT_DIR / 'bertopic_over_time.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ bertopic_over_time.png")

fig = plot_sentiment_by_narrative(pipeline)
fig.savefig(OUTPUT_DIR / 'sentiment_by_narrative.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ sentiment_by_narrative.png")

fig = plot_stance_over_time(pipeline)
fig.savefig(OUTPUT_DIR / 'stance_over_time.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ stance_over_time.png")

fig = plot_stance_heatmap(pipeline)
fig.savefig(OUTPUT_DIR / 'stance_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ stance_heatmap.png")

fig = plot_prevalence_with_spikes(pipeline)
fig.savefig(OUTPUT_DIR / 'prevalence_with_spikes.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ prevalence_with_spikes.png")

fig = plot_spike_z_scores(pipeline)
fig.savefig(OUTPUT_DIR / 'spike_z_scores.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ spike_z_scores.png")

spike_report = generate_spike_report(pipeline)
with open(OUTPUT_DIR / 'spike_report.md', 'w') as f:
    f.write(spike_report)
print(f"  ✓ spike_report.md")


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