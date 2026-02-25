"""
core — Shared narrative analysis infrastructure.

Adds this directory to sys.path so that existing cross-imports inside core
(e.g. narrative_pipeline_v2 → from narrative_config import ...) keep working
without rewriting every import statement.

Public API re-exports for convenience:
    from core import NarrativePipeline, NarrativeConfig, NarrativeExtensions
"""

import sys
from pathlib import Path

# Allow intra-core imports to work unmodified (e.g. `from narrative_config import ...`)
_CORE_DIR = str(Path(__file__).resolve().parent)
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

# ── Re-exports ───────────────────────────────────────────────────────────────
from narrative_config import (          # noqa: E402
    NarrativeConfig,
    get_config,
    list_domains,
    CONFIG_REGISTRY,
    EV_CONFIG,
    AITECH_CONFIG,
    RETAIL_CONFIG,
)
# Adjust these import names to match your actual filenames in core/.
# Your files may be named with or without "_v2" suffix.
try:
    from narrative_pipeline_v2 import (     # noqa: E402
        NarrativePipeline,
        AnalysisResults,
    )
except ImportError:
    from narrative_pipeline import (        # noqa: E402
        NarrativePipeline,
        AnalysisResults,
    )
from canonical_news_scraper import (    # noqa: E402
    CanonicalDatasetBuilder,
)
from nytimes_scraper import NYTScraper  # noqa: E402
from gdelt_scraper import GDELTScraper  # noqa: E402
from narrative_extensions import (      # noqa: E402
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
# Your orchestrators import "narrative_visualizations" but the uploaded file
# was "narrative_visualizations_v2.py". Adjust the name below to match
# whichever filename you have in core/.
try:
    from narrative_visualizations import (      # noqa: E402
        plot_alluvial_diagram,
        plot_semantic_network,
        plot_semantic_strength_over_time,
        plot_expanded_network_v2,
        generate_cluster_narrative_report,
        plot_tsne_centroids_timeline,
    )
except ImportError:
    from narrative_visualizations_v2 import (   # noqa: E402
        plot_alluvial_diagram,
        plot_semantic_network,
        plot_semantic_strength_over_time,
        plot_expanded_network_v2,
        generate_cluster_narrative_report,
        plot_tsne_centroids_timeline,
    )
from narrative_network_improved import (   # noqa: E402
    plot_narrative_network_v3,
    plot_narrative_network_louvain,
)