# Narrative Machine

**MSDS Capstone: Narrative Prevalence and Emergent Narrative Discovery Using Embedding-Based Measurements and Network Analysis**

Narrative Machine is a text analytics platform for detecting, analyzing, and visualizing competing narratives in news coverage. It combines web scraping, sentence embeddings, clustering, and network analysis to track how narratives evolve over time across different domains.

---

## Domains

| Domain | Articles | Date Range | Sources |
|--------|----------|------------|---------|
| Electric Vehicles | 9,928 | 2015–2024 | NYT, GDELT |
| AI & Technology | 2,322 | 2020–2024 | NYT, GDELT |

---

## Setup

**Requirements:** Python 3.10, the project virtualenv at `narrative_machine/`

```bash
# Clone the repo
git clone git@github.com:ryan-osleeb/MSDS-Capstone-Narrative-Machine.git
cd MSDS-Capstone-Narrative-Machine

# Create and activate the virtual environment
python3.10 -m venv narrative_machine
source narrative_machine/bin/activate

# Install dependencies
pip install sentence-transformers bertopic scikit-learn pandas numpy scipy networkx
pip install matplotlib seaborn requests beautifulsoup4 textblob streamlit
pip install python-louvain hdbscan python-dotenv
python -m textblob.download_corpora
```

**API Key:** Create a `.env` file in the project root:
```
NYT_API_KEY=your_nyt_api_key_here
```
Get a free key at [developer.nytimes.com](https://developer.nytimes.com/).

---

## Usage

```bash
# List available domains
./narrative_machine/bin/python run_domain.py --list

# Run the full pipeline for a domain
./narrative_machine/bin/python run_domain.py --domain aitech
./narrative_machine/bin/python run_domain.py --domain electricvehicles

# Run specific steps only (e.g., re-run visualizations without re-embedding)
./narrative_machine/bin/python run_domain.py --domain aitech --steps viz,ext_viz

# Fetch new articles incrementally (only pulls what's new)
./narrative_machine/bin/python update_data.py
./narrative_machine/bin/python update_data.py --domain aitech

# Fetch new data and re-run the full pipeline
./narrative_machine/bin/python update_data.py --run-pipeline

# Launch the Streamlit dashboard
./narrative_machine/bin/streamlit run dashboard.py
```

---

## Pipeline

The pipeline runs in six steps, executed in order:

| Step | Description |
|------|-------------|
| `ingest` | Merges NYT + GDELT CSVs into a unified schema and writes to SQLite |
| `analyze` | Computes sentence embeddings, detects narratives via cosine similarity, runs KMeans clustering |
| `viz` | Generates alluvial diagram, semantic networks, t-SNE timeline, cluster report |
| `network` | KMeans v3 network graph, Louvain community detection, yearly network snapshots |
| `extensions` | BERTopic topic modeling, sentiment scoring, spike detection |
| `ext_viz` | BERTopic plots, sentiment charts, stance heatmap, spike annotations and report |

---

## Project Structure

```
narrative_machine_v3/
├── core/                    # Shared NLP infrastructure
│   ├── narrative_config.py          # Domain configs: narratives, prototypes, colors
│   ├── narrative_pipeline_v2.py     # Embeddings, clustering, narrative detection
│   ├── narrative_extensions.py      # BERTopic, sentiment, spike detection
│   ├── narrative_visualizations_v2.py  # Base visualizations
│   ├── narrative_network_improved.py   # KMeans + Louvain network graphs
│   ├── canonical_news_scraper.py    # Merges NYT + GDELT into unified schema
│   ├── nytimes_scraper.py           # NYT Article Search API wrapper
│   └── gdelt_scraper.py             # GDELT DocSearch + article text fetcher
├── db/
│   └── store.py                     # SQLite schema and query functions
├── domains/
│   ├── electricvehicles/            # Domain manifest + raw and unified CSVs
│   ├── aitech/                      # Domain manifest + raw and unified CSVs
│   └── template/                    # Starter template for new domains
├── output/                          # Generated PNGs, reports, caches (not committed)
├── run_domain.py                    # Main pipeline entry point
├── dashboard.py                     # Streamlit web dashboard
└── update_data.py                   # Incremental data fetcher
```

---

## Adding a New Domain

1. Copy `domains/template/` to `domains/<name>/` and fill in the `DOMAIN_MANIFEST`.
2. Add a `NarrativeConfig` (narratives, prototype sentences, colors) to `core/narrative_config.py` and register it in `CONFIG_REGISTRY`.
3. Add NYT and GDELT query configs to `DOMAIN_SCRAPER_CONFIG` in `update_data.py`.
4. Fetch data and run the pipeline:
   ```bash
   ./narrative_machine/bin/python update_data.py --domain <name>
   ./narrative_machine/bin/python run_domain.py --domain <name>
   ```

---

## Output

Each domain produces the following in `output/<domain>/`:

- **Visualizations:** alluvial diagram, semantic networks (full corpus + yearly), Louvain community networks, BERTopic topic plots, sentiment charts, stance heatmap, spike annotations
- **Reports:** `*_cluster_report.md` (cluster breakdowns and top articles), `*_spike_report.md` (detected narrative spikes)
- **Caches:** sentence embeddings (`.pkl`), BERTopic model (~114MB) — regenerated automatically if deleted
