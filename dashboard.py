#!/usr/bin/env python3
"""
dashboard.py — Streamlit dashboard for Narrative Machine v3.

Launch:
    streamlit run dashboard.py
"""

import re
import sys
import time
import queue
import datetime
import subprocess
import threading
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path so `db` can be imported
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"

PIPELINE_STEPS = ["ingest", "analyze", "viz", "network", "extensions", "ext_viz"]

STEP_DESCRIPTIONS = {
    "ingest": "Build unified CSV from NYT + GDELT sources",
    "analyze": "Embeddings, narrative detection, clustering, temporal analysis",
    "viz": "Base visualizations (alluvial, semantic networks, t-SNE, etc.)",
    "network": "Improved networks (KMeans v3, Louvain, yearly variants)",
    "extensions": "BERTopic, sentiment analysis, spike detection",
    "ext_viz": "Extension visualizations (BERTopic overview, sentiment, spikes)",
}

# Ordered display categories → list of substrings to match in stem (prefix removed)
CATEGORY_PATTERNS: list[tuple[str, list[str]]] = [
    ("Temporal",  ["alluvial", "tsne", "strength", "stance_over_time", "prevalence"]),
    ("Networks",  ["semantic", "network"]),
    ("Sentiment", ["sentiment", "stance_heatmap", "spike"]),
    ("Topics",    ["bertopic"]),
]

YEAR_RE = re.compile(r"_(\d{4})\.png$")


# ─── Domain / output discovery ────────────────────────────────────────────────

def discover_domains() -> list[str]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(d.name for d in OUTPUT_DIR.iterdir() if d.is_dir())


def detect_prefix(pngs: list[Path]) -> str:
    """Infer the domain prefix (e.g. 'ev_', 'tech_') from the PNG filenames."""
    if not pngs:
        return ""
    parts = pngs[0].stem.split("_", 1)
    return f"{parts[0]}_" if len(parts) > 1 else ""


def categorize_pngs(pngs: list[Path], prefix: str) -> dict[str, list[Path]]:
    """Sort PNGs into display categories, preserving order within each category."""
    result: dict[str, list[Path]] = {}

    def _cat(stem_no_prefix: str) -> str:
        s = stem_no_prefix.lower()
        for cat_name, patterns in CATEGORY_PATTERNS:
            if any(p in s for p in patterns):
                return cat_name
        return "Other"

    for png in pngs:
        stem = png.stem[len(prefix):] if prefix and png.stem.startswith(prefix) else png.stem
        cat = _cat(stem)
        result.setdefault(cat, []).append(png)

    # Return in canonical order
    ordered: dict[str, list[Path]] = {}
    for cat_name, _ in CATEGORY_PATTERNS:
        if cat_name in result:
            ordered[cat_name] = result[cat_name]
    if "Other" in result:
        ordered["Other"] = result["Other"]
    return ordered


def get_domain_outputs(domain: str) -> tuple[dict[str, list[Path]], list[Path]]:
    domain_dir = OUTPUT_DIR / domain
    pngs = sorted(domain_dir.glob("*.png"))
    reports = sorted(domain_dir.glob("*.md"))
    prefix = detect_prefix(pngs)
    return categorize_pngs(pngs, prefix), reports, prefix


# ─── Image loading with mtime-keyed cache ─────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_image(path_str: str, _mtime: float) -> bytes:
    """Cache image bytes; auto-invalidates when the file is regenerated."""
    return Path(path_str).read_bytes()


def show_image(path: Path):
    mtime = path.stat().st_mtime if path.exists() else 0.0
    st.image(_load_image(str(path), mtime), use_container_width=True)


# ─── Label helpers ────────────────────────────────────────────────────────────

def pretty_name(path: Path, prefix: str) -> str:
    stem = path.stem
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix):]
    return stem.replace("_", " ").title()


# ─── Pipeline runner (background thread) ──────────────────────────────────────

def _pipeline_thread(domain: str, steps: list[str], log_q: "queue.Queue[str | None]"):
    cmd = [
        str(PROJECT_ROOT / "narrative_machine" / "bin" / "python"),
        "run_domain.py",
        "--domain", domain,
        "--steps", ",".join(steps),
    ]
    import os
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        for line in iter(proc.stdout.readline, ""):
            log_q.put(line.rstrip())
        proc.stdout.close()
        proc.wait()
    except Exception as exc:
        log_q.put(f"[ERROR] {exc}")
    finally:
        log_q.put(None)  # sentinel — signals completion


# ─── Session state init ───────────────────────────────────────────────────────

def init_state():
    defaults = {"running": False, "log_lines": [], "log_q": None}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Rendering helpers ────────────────────────────────────────────────────────

def render_png_section(cat_name: str, pngs: list[Path], prefix: str, domain: str):
    """Full-corpus PNGs shown directly; yearly variants grouped under a year slider."""
    full_pngs = [p for p in pngs if not YEAR_RE.search(p.name)]
    yearly_pngs = [p for p in pngs if YEAR_RE.search(p.name)]

    for p in full_pngs:
        st.subheader(pretty_name(p, prefix))
        show_image(p)

    if yearly_pngs:
        years = sorted({YEAR_RE.search(p.name).group(1) for p in yearly_pngs})
        label = f"Yearly views — {years[0]}–{years[-1]}" if len(years) > 1 else f"Year {years[0]}"
        with st.expander(label, expanded=False):
            if len(years) > 1:
                year_sel = st.select_slider(
                    "Select year",
                    options=years,
                    key=f"yr_{domain}_{cat_name}",
                )
            else:
                year_sel = years[0]

            for p in yearly_pngs:
                if YEAR_RE.search(p.name).group(1) == year_sel:
                    st.subheader(pretty_name(p, prefix))
                    show_image(p)


def render_reports(reports: list[Path]):
    for r in reports:
        label = r.stem.replace("_", " ").title()
        with st.expander(label, expanded=True):
            st.markdown(r.read_text(encoding="utf-8"))


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def _render_db_stats(domain: str) -> None:
    """Show a compact database summary for the selected domain in the sidebar."""
    try:
        from db import get_db_stats
        stats = get_db_stats()
    except Exception:
        return  # DB module not yet available or DB doesn't exist

    if not stats:
        st.caption("Database: not yet populated")
        return

    d = stats.get(domain)
    if d is None:
        st.caption("Database: no data for this domain")
        return

    st.divider()
    st.subheader("Database")
    col1, col2 = st.columns(2)
    col1.metric("Articles", f"{d['total']:,}")
    col2.metric("Extracted", f"{d['extracted']:,}")

    sources = d.get("sources", {})
    if sources:
        parts = " · ".join(f"{k.upper()}: {v:,}" for k, v in sorted(sources.items()))
        st.caption(parts)

    if d.get("earliest") and d.get("latest"):
        earliest = d["earliest"][:10]
        latest = d["latest"][:10]
        st.caption(f"{earliest} → {latest}")


def render_sidebar(domains: list[str]) -> str:
    with st.sidebar:
        st.title("Narrative Machine")
        st.caption("v3 · Analysis Dashboard")
        st.divider()

        domain = st.selectbox(
            "Domain",
            domains,
            format_func=lambda d: d.replace("_", " ").title(),
        )

        # ── Last-updated timestamp ────────────────────────────────────────────
        domain_dir = OUTPUT_DIR / domain
        mtimes = (
            [p.stat().st_mtime for p in domain_dir.glob("*.png")]
            if domain_dir.exists()
            else []
        )
        if mtimes:
            ts = datetime.datetime.fromtimestamp(max(mtimes))
            st.caption(f"Last output: {ts:%Y-%m-%d %H:%M}")

        # ── Database stats ────────────────────────────────────────────────────
        _render_db_stats(domain)

        st.divider()

        # ── Pipeline controls ─────────────────────────────────────────────────
        st.subheader("Run Pipeline")

        selected_steps = st.multiselect(
            "Steps",
            PIPELINE_STEPS,
            default=PIPELINE_STEPS,
            format_func=lambda s: s,
            help="\n".join(f"**{s}** — {d}" for s, d in STEP_DESCRIPTIONS.items()),
        )

        run_disabled = st.session_state.running or not selected_steps
        if st.button("Run", disabled=run_disabled, use_container_width=True):
            st.session_state.log_lines = []
            st.session_state.log_q = queue.Queue()
            st.session_state.running = True
            threading.Thread(
                target=_pipeline_thread,
                args=(domain, selected_steps, st.session_state.log_q),
                daemon=True,
            ).start()

        # ── Drain log queue ───────────────────────────────────────────────────
        if st.session_state.running and st.session_state.log_q is not None:
            q: queue.Queue = st.session_state.log_q
            while not q.empty():
                line = q.get_nowait()
                if line is None:
                    st.session_state.running = False
                    break
                st.session_state.log_lines.append(line)

        if st.session_state.log_lines or st.session_state.running:
            status = "Running..." if st.session_state.running else "Complete"
            st.caption(status)
            log_text = "\n".join(st.session_state.log_lines[-200:])
            st.code(log_text, language=None)
            if st.session_state.running:
                time.sleep(0.4)
                st.rerun()

        st.divider()
        if st.button("Refresh outputs", use_container_width=True, help="Reload all visualizations from disk"):
            st.rerun()

    return domain


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Narrative Machine Dashboard",
        layout="wide",
        page_icon="\U0001f4f0",  # newspaper emoji
    )
    init_state()

    domains = discover_domains()
    if not domains:
        st.error(
            f"No output directories found under `{OUTPUT_DIR}`.\n\n"
            "Run the pipeline first:\n\n"
            "```\npython run_domain.py --domain electricvehicles\n```"
        )
        return

    domain = render_sidebar(domains)
    pngs_by_cat, reports, prefix = get_domain_outputs(domain)

    n_pngs = sum(len(v) for v in pngs_by_cat.values())
    st.header(domain.replace("_", " ").title())
    st.caption(f"{n_pngs} visualization{'s' if n_pngs != 1 else ''} · {len(reports)} report{'s' if len(reports) != 1 else ''}")

    if not n_pngs and not reports:
        st.warning("No outputs found for this domain. Run the pipeline to generate them.")
        return

    tab_names = list(pngs_by_cat.keys()) + (["Reports"] if reports else [])
    tabs = st.tabs(tab_names)

    for tab, cat_name in zip(tabs, pngs_by_cat.keys()):
        with tab:
            render_png_section(cat_name, pngs_by_cat[cat_name], prefix, domain)

    if reports:
        with tabs[len(pngs_by_cat)]:
            render_reports(reports)


if __name__ == "__main__":
    main()
