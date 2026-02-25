#!/usr/bin/env python3
"""
run_domain.py — Unified entry point for the Narrative Machine.
================================================================

Replaces ev_narrative_machine_v3.py and tech_narrative_machine_v2.py
with a single script driven by domain manifests.

Usage:
    # Run everything for a domain
    python run_domain.py --domain aitech

    # Run specific steps
    python run_domain.py --domain electricvehicles --steps ingest,analyze,viz

    # List available domains
    python run_domain.py --list

    # Dry-run (show what would execute)
    python run_domain.py --domain aitech --dry-run

Steps:
    ingest      — Build unified CSV from NYT + GDELT sources
    analyze     — Embeddings, narrative detection, clustering, temporal
    viz         — Base visualizations (alluvial, networks, t-SNE, etc.)
    network     — Improved network viz (KMeans v3, Louvain, yearly)
    extensions  — BERTopic, sentiment, spike detection
    ext_viz     — Extension visualizations + spike report
    all         — Run all steps in order (default)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import sys
from pathlib import Path
from time import time

# ── Project imports ──────────────────────────────────────────────────────────
from core import (
    # Config
    NarrativeConfig, get_config,
    # Data
    CanonicalDatasetBuilder,
    # Pipeline
    NarrativePipeline,
    # Extensions
    NarrativeExtensions,
    # Base visualizations
    plot_alluvial_diagram,
    plot_semantic_network,
    plot_semantic_strength_over_time,
    plot_expanded_network_v2,
    generate_cluster_narrative_report,
    plot_tsne_centroids_timeline,
    # Improved network
    plot_narrative_network_v3,
    plot_narrative_network_louvain,
    # Extension visualizations
    plot_bertopic_overview,
    plot_bertopic_over_time,
    plot_sentiment_by_narrative,
    plot_stance_over_time,
    plot_stance_heatmap,
    plot_prevalence_with_spikes,
    plot_spike_z_scores,
    generate_spike_report,
)
from domains import get_domain_manifest, list_available_domains


# =============================================================================
# STEP DEFINITIONS
# =============================================================================

ALL_STEPS = ['ingest', 'analyze', 'viz', 'network', 'extensions', 'ext_viz']


def banner(text: str):
    print(f"\n{'='*60}")
    print(text)
    print('='*60)


# ── Step 1: Data Ingestion ──────────────────────────────────────────────────

def step_ingest(manifest: dict, output_dir: Path, domain_folder: str) -> None:
    banner("STEP 1: Data Ingestion & Canonicalization")

    builder = CanonicalDatasetBuilder()
    nyt_path = manifest['nyt_csv']
    gdelt_path = manifest['gdelt_csv']
    topic = manifest['topic_label']

    if Path(nyt_path).exists():
        builder.load_nyt_csv(nyt_path, topic_label=topic)
    else:
        print(f"  ⚠ NYT CSV not found: {nyt_path}")

    if Path(gdelt_path).exists():
        builder.load_gdelt_csv(gdelt_path, topic_label=topic)
    else:
        print(f"  ⚠ GDELT CSV not found: {gdelt_path}")

    unified = builder.build_unified_dataset()
    builder.export_unified(manifest['unified_csv'])
    print(f"  ✓ Unified dataset: {len(unified)} articles → {manifest['unified_csv']}")

    # Persist to database
    from db import upsert_articles
    n = upsert_articles(unified, domain_folder)
    print(f"  ✓ Saved {n} articles to database (domain: {domain_folder})")


# ── Steps 2–5: Base Analysis ────────────────────────────────────────────────

def step_analyze(manifest: dict, config: NarrativeConfig,
                 output_dir: Path, domain_folder: str) -> NarrativePipeline:
    banner("STEPS 2-5: Base Narrative Analysis")

    pipeline = NarrativePipeline(config)
    _load_pipeline_data(pipeline, manifest, domain_folder)

    cache = str(output_dir / f"{manifest['cache_prefix']}_embeddings.pkl")
    pipeline.run_full_analysis(cache_file=cache, compute_tsne=False)
    return pipeline


def _load_pipeline_data(pipeline: NarrativePipeline, manifest: dict,
                        domain_folder: str) -> None:
    """
    Load article data into the pipeline, preferring the database when available.
    Falls back to the unified CSV if the domain has no rows in the database yet.
    """
    from db import get_articles_df
    df = get_articles_df(domain_folder)
    if df is not None:
        print(f"  ✓ Pulled {len(df)} articles from database (domain: {domain_folder})")
        # Refresh the unified CSV from the DB so it stays in sync
        df.to_csv(manifest['unified_csv'], index=False)
        pipeline.load_data(manifest['unified_csv'])
    else:
        print(f"  ⚠ No DB data found for '{domain_folder}', falling back to CSV")
        pipeline.load_data(manifest['unified_csv'])


# ── Step 6: Base Visualizations ──────────────────────────────────────────────

def step_viz(pipeline: NarrativePipeline, manifest: dict,
             output_dir: Path) -> None:
    banner("STEP 6: Base Visualizations")
    p = manifest['output_prefix']

    plot_alluvial_diagram(
        pipeline,
        save_path=str(output_dir / f'{p}_alluvial.png'),
    )

    plot_semantic_network(
        pipeline, n_subclusters=15,
        save_path=str(output_dir / f'{p}_semantic_network.png'),
    )

    # Yearly semantic networks — auto-detect years with enough data
    years = sorted(pipeline.results.df['year'].dropna().unique())
    for yr in years:
        yr_count = (pipeline.results.df['year'] == yr).sum()
        if yr_count < 20:
            continue
        plot_semantic_network(
            pipeline, time_period=yr,
            save_path=str(output_dir / f'{p}_semantic_{int(yr)}.png'),
        )

    plot_semantic_strength_over_time(
        pipeline,
        save_path=str(output_dir / f'{p}_strength.png'),
    )

    plot_expanded_network_v2(
        pipeline, n_subclusters=15,
        save_path=str(output_dir / f'{p}_network_expanded.png'),
    )

    plot_tsne_centroids_timeline(
        pipeline,
        save_path=str(output_dir / f'{p}_tsne_timeline.png'),
    )

    report = generate_cluster_narrative_report(
        pipeline, text_col="title",
        save_path=str(output_dir / f'{p}_cluster_report.md'),
    )
    print(f"  ✓ Cluster report → {p}_cluster_report.md")


# ── Step 7: Improved Network Visualization ───────────────────────────────────

def step_network(pipeline: NarrativePipeline, manifest: dict,
                 output_dir: Path) -> None:
    banner("STEP 7: Improved Network Visualization")
    p = manifest['output_prefix']

    # KMeans v3 — full corpus
    plot_narrative_network_v3(
        pipeline,
        n_clusters=18,
        text_col='title',
        knn_k=8,
        similarity_threshold=0.28,
        show_narrative=True,
        save_path=str(output_dir / f'{p}_network_v3.png'),
        dpi=200,
    )

    # Louvain — full corpus
    try:
        plot_narrative_network_louvain(
            pipeline,
            text_col='title',
            knn_k=10,
            similarity_threshold=0.25,
            resolution=1.2,
            show_narrative=True,
            save_path=str(output_dir / f'{p}_network_louvain.png'),
            dpi=200,
        )
    except ImportError:
        print("  ⚠ Skipping Louvain (pip install python-louvain)")

    # Yearly Louvain networks
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
                save_path=str(output_dir / f'{p}_network_louvain_{int(yr)}.png'),
                dpi=200,
            )
        except Exception as e:
            print(f"    ⚠ Skipping {yr}: {e}")


# ── Step 8: Extensions ───────────────────────────────────────────────────────

def step_extensions(pipeline: NarrativePipeline, manifest: dict,
                    output_dir: Path) -> NarrativeExtensions:
    banner("STEP 8: Pipeline Extensions")
    cp = manifest['cache_prefix']

    ext = NarrativeExtensions(
        pipeline,
        sentiment_method=manifest.get('sentiment_method', 'textblob'),
    )

    # BERTopic
    ext.run_bertopic(
        text_col='text',
        min_topic_size=15,
        reduce_outliers=True,
        cache_file=str(output_dir / f'{cp}_bertopic_model'),
    )
    ext.bertopic_topics_over_time(text_col='text')

    try:
        ext.bertopic_hierarchy(text_col='text')
    except Exception as e:
        print(f"  ⚠ BERTopic hierarchy skipped: {e}")

    # Sentiment
    ext.compute_sentiment(
        text_col='title',
        cache_file=str(output_dir / f'{cp}_sentiment_cache.pkl'),
    )

    # Spike detection
    ext.detect_narrative_spikes(
        z_threshold=1.5,
        rolling_window=4,
        time_col='year',
        top_articles_per_spike=5,
    )

    return ext


# ── Step 9: Extension Visualizations ─────────────────────────────────────────

def step_ext_viz(pipeline: NarrativePipeline, manifest: dict,
                 output_dir: Path) -> None:
    banner("STEP 9: Extension Visualizations")
    p = manifest['output_prefix']

    viz_specs = [
        ('bertopic_overview',       lambda: plot_bertopic_overview(pipeline)),
        ('bertopic_over_time',      lambda: plot_bertopic_over_time(pipeline, top_n_topics=8)),
        ('sentiment_by_narrative',  lambda: plot_sentiment_by_narrative(pipeline)),
        ('stance_over_time',        lambda: plot_stance_over_time(pipeline)),
        ('stance_heatmap',          lambda: plot_stance_heatmap(pipeline)),
        ('prevalence_with_spikes',  lambda: plot_prevalence_with_spikes(pipeline)),
        ('spike_z_scores',          lambda: plot_spike_z_scores(pipeline)),
    ]

    for name, fn in viz_specs:
        try:
            fig = fn()
            fig.savefig(output_dir / f'{p}_{name}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ {p}_{name}.png")
        except Exception as e:
            print(f"  ⚠ {name}: {e}")

    # Spike report (markdown)
    try:
        spike_report = generate_spike_report(pipeline)
        report_path = output_dir / f'{p}_spike_report.md'
        with open(report_path, 'w') as f:
            f.write(spike_report)
        print(f"  ✓ {p}_spike_report.md")
    except Exception as e:
        print(f"  ⚠ spike_report: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Narrative Machine — unified domain runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--domain', '-d',
        help='Domain to analyze (folder name under domains/)',
    )
    parser.add_argument(
        '--steps', '-s',
        default='all',
        help=f'Comma-separated steps to run. Options: {", ".join(ALL_STEPS)}, all (default: all)',
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output directory (default: output/<domain>/)',
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available domains and exit',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without running',
    )

    args = parser.parse_args()

    # ── List mode ────────────────────────────────────────────────────────
    if args.list:
        domains = list_available_domains()
        print("Available domains:")
        for d in domains:
            print(f"  • {d}")
        if not domains:
            print("  (none found — check domains/ directory)")
        sys.exit(0)

    # ── Validate ─────────────────────────────────────────────────────────
    if not args.domain:
        parser.error("--domain is required (use --list to see options)")

    manifest = get_domain_manifest(args.domain)
    config = get_config(manifest['config_key'])

    # Parse steps
    if args.steps.lower() == 'all':
        steps = ALL_STEPS
    else:
        steps = [s.strip().lower() for s in args.steps.split(',')]
        invalid = [s for s in steps if s not in ALL_STEPS]
        if invalid:
            parser.error(f"Unknown steps: {invalid}. Valid: {ALL_STEPS}")

    # Output directory
    output_dir = Path(args.output) if args.output else Path('output') / args.domain
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dry run ──────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\nDomain:     {args.domain}")
        print(f"Config:     {config.name} ({len(config.narratives)} narratives)")
        print(f"Steps:      {steps}")
        print(f"Output:     {output_dir.resolve()}")
        print(f"\nManifest:")
        for k, v in manifest.items():
            print(f"  {k}: {v}")
        sys.exit(0)

    # ── Run ──────────────────────────────────────────────────────────────
    banner(f"NARRATIVE MACHINE — {config.name.upper()}")
    print(f"Domain:  {args.domain}")
    print(f"Steps:   {steps}")
    print(f"Output:  {output_dir.resolve()}")

    t0 = time()
    pipeline = None
    ext = None

    if 'ingest' in steps:
        step_ingest(manifest, output_dir, args.domain)

    if 'analyze' in steps:
        pipeline = step_analyze(manifest, config, output_dir, args.domain)

    # For viz-only steps, we need a pipeline — re-load if not already created
    if any(s in steps for s in ('viz', 'network', 'extensions', 'ext_viz')):
        if pipeline is None:
            banner("Loading pipeline for visualization steps...")
            pipeline = NarrativePipeline(config)
            _load_pipeline_data(pipeline, manifest, args.domain)
            cache = str(output_dir / f"{manifest['cache_prefix']}_embeddings.pkl")
            pipeline.run_full_analysis(cache_file=cache, compute_tsne=False)

    if 'viz' in steps:
        step_viz(pipeline, manifest, output_dir)

    if 'network' in steps:
        step_network(pipeline, manifest, output_dir)

    if 'extensions' in steps:
        ext = step_extensions(pipeline, manifest, output_dir)

    if 'ext_viz' in steps:
        # If extensions weren't run this session, ext_viz may still work
        # if extension data was cached from a prior run
        step_ext_viz(pipeline, manifest, output_dir)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time() - t0
    banner("PIPELINE COMPLETE")
    print(f"\nDomain:  {config.name}")
    print(f"Time:    {elapsed:.1f}s")
    print(f"Output:  {output_dir.resolve()}")
    print(f"\nGenerated files:")
    for f in sorted(output_dir.glob('*')):
        size = f.stat().st_size
        label = f"{size/1024:.0f}KB" if size > 1024 else f"{size}B"
        print(f"  {f.name:<45} {label}")


if __name__ == '__main__':
    main()