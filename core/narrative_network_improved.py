"""
Improved Narrative Network Visualization
==========================================
Produces a dense article-level network matching the reference aesthetic:
- Organic cluster emergence via force-directed layout
- Labels positioned OUTSIDE the graph perimeter with leader-line arrows
- Descriptive TF-IDF labels with corpus percentage
- Small uniform nodes, thin intra-cluster edges
- Clean white background, no grid
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# COLOR PALETTE (expanded, vibrant, distinct)
# =============================================================================

NETWORK_COLORS = [
    '#e6194b',  # red
    '#3cb44b',  # green
    '#ffe119',  # yellow
    '#4363d8',  # blue
    '#f58231',  # orange
    '#911eb4',  # purple
    '#42d4f4',  # cyan
    '#f032e6',  # magenta
    '#bfef45',  # lime
    '#fabed4',  # pink
    '#469990',  # teal
    '#dcbeff',  # lavender
    '#9A6324',  # brown
    '#800000',  # maroon
    '#aaffc3',  # mint
    '#808000',  # olive
    '#ffd8b1',  # apricot
    '#000075',  # navy
    '#a9a9a9',  # grey
    '#e6beff',  # light purple
    '#aa6e28',  # tan
    '#00CED1',  # dark turquoise
    '#DC143C',  # crimson
    '#FF6B6B',  # coral
]


def _get_cluster_keywords(texts, n_terms=4, ngram_range=(1, 3)):
    """Extract descriptive TF-IDF keywords from a cluster's texts.
    
    Uses bigrams/trigrams for more descriptive labels, custom stopwords
    to filter generic news terms, and returns a formatted label string.
    """
    if len(texts) < 2:
        return "Miscellaneous"
    
    custom_stops = list(ENGLISH_STOP_WORDS) + [
        'said', 'says', 'new', 'year', 'years', 'time', 'like', 'just',
        'people', 'would', 'could', 'also', 'one', 'two', 'first', 'last',
        'percent', 'million', 'billion', 'according', 'report', 'reported',
        'company', 'companies', 'market', 'business', 'work', 'working',
        'week', 'day', 'today', 'make', 'way', 'told', 'going', 'think',
        'know', 'want', 'need', 'good', 'right', 'thing', 'things', 'lot',
        'big', 'long', 'high', 'low', 'real', 'come', 'came', 'got',
        'use', 'used', 'using', 'news', 'article', 'story', 'world',
    ]
    
    try:
        tfidf = TfidfVectorizer(
            max_features=200,
            stop_words=custom_stops,
            ngram_range=ngram_range,
            min_df=max(2, len(texts) // 20),
            max_df=0.85,
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b'
        )
        tfidf_matrix = tfidf.fit_transform(texts)
        mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        top_idx = mean_scores.argsort()[-n_terms:][::-1]
        features = tfidf.get_feature_names_out()
        terms = [features[i].title() for i in top_idx]
        return ', '.join(terms)
    except Exception:
        return "Miscellaneous"


def _compute_label_positions(centroids, graph_center, graph_radius, 
                              min_distance=0.25):
    """Position labels outside the graph perimeter.
    
    Places labels radially outward from each cluster centroid,
    with collision avoidance to prevent overlapping text.
    """
    label_positions = {}
    used_angles = []
    
    for cluster_id, centroid in centroids.items():
        # Direction from graph center to centroid
        dx = centroid[0] - graph_center[0]
        dy = centroid[1] - graph_center[1]
        angle = np.arctan2(dy, dx)
        
        # Avoid angular collision with existing labels
        for used_angle in used_angles:
            if abs(angle - used_angle) < 0.15:  # ~8.5 degrees
                angle += 0.18  # nudge
        used_angles.append(angle)
        
        # Place label outside graph boundary
        label_r = graph_radius * 1.25
        label_x = graph_center[0] + label_r * np.cos(angle)
        label_y = graph_center[1] + label_r * np.sin(angle)
        
        label_positions[cluster_id] = (label_x, label_y)
    
    return label_positions


def plot_narrative_network_v3(
    data, 
    embeddings=None,
    n_clusters=15,
    text_col='title',
    title=None,
    figsize=(20, 18),
    # Network construction
    knn_k=8,                    # k-nearest neighbors for edge construction
    similarity_threshold=0.30,  # minimum cosine sim for an edge
    # Layout
    layout_iterations=200,
    layout_k=None,              # spring constant (None=auto)
    layout_scale=2.0,
    # Appearance
    node_size=120,
    node_alpha=0.85,
    edge_alpha=0.12,
    edge_width=0.5,
    label_fontsize=13,
    show_edge_within_only=True, # only show intra-cluster edges
    min_cluster_pct=1.0,        # hide clusters smaller than this %
    zoom_percentile=88,         # crop view to this percentile of node positions
    label_radius_factor=1.18,   # how far out labels sit (1.0 = at boundary)
    # Narrative mapping
    show_narrative=True,        # show dominant seeded narrative per cluster
    # Time filtering
    time_period=None,           # filter to specific year/period (e.g., 2023)
    time_col='year',            # column used for time filtering
    # Output
    save_path=None,
    dpi=200,
):
    """
    Dense article-level narrative network matching the reference aesthetic.
    
    Key differences from plot_dense_cluster_network:
    1. KNN-based edge construction (faster, sparser, more organic)
    2. Labels positioned OUTSIDE the graph with leader-line arrows
    3. Better TF-IDF labeling with bigrams/trigrams
    4. Cluster-seeded spring layout for organic separation
    5. Optional intra-cluster-only edges for cleaner visuals
    6. No grid, no convex hulls — clusters emerge from topology
    
    Parameters
    ----------
    data : DataFrame or NarrativePipeline
    embeddings : np.ndarray, optional (required if data is DataFrame)
    n_clusters : int
        Number of clusters to discover
    text_col : str
        Column used for TF-IDF label extraction
    knn_k : int
        Number of nearest neighbors for edge construction.
        Higher = denser graph, more edges. 5-10 is typical.
    similarity_threshold : float
        Minimum cosine similarity to keep a KNN edge.
        Lower = more edges. 0.25-0.35 typical.
    show_edge_within_only : bool
        If True, only draw edges between nodes in the same cluster.
        Produces cleaner visual with distinct cluster blobs.
    """
    # -------------------------------------------------------------------------
    # Extract data
    # -------------------------------------------------------------------------
    if hasattr(data, 'results'):
        df = data.results.df.copy()
        embeddings = data.results.embeddings
        if title is None:
            title = f"{data.config.name}: Narrative Network"
    else:
        df = data.copy()
        if embeddings is None:
            raise ValueError("embeddings required when passing DataFrame")

    # Time-period filter
    if time_period is not None and time_col in df.columns:
        mask = df[time_col] == time_period
        df = df[mask].reset_index(drop=True)
        embeddings = embeddings[mask.values]
        if title and str(time_period) not in title:
            title = f"{title} ({time_period})"
        print(f"  Filtered to {time_period}: {len(df)} articles")
    
    if len(df) < 5:
        print(f"  ⚠ Too few articles ({len(df)}) for network visualization, skipping.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Too few articles for {time_period}', 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        return fig, ax, nx.Graph()
    
    n_articles = len(df)
    # Auto-scale clusters for smaller filtered datasets
    n_clusters = min(n_clusters, max(3, n_articles // 8))
    print(f"Building narrative network for {n_articles:,} articles ({n_clusters} clusters)...")
    
    # -------------------------------------------------------------------------
    # Step 1: Cluster articles
    # -------------------------------------------------------------------------
    print(f"  Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    df['_cluster'] = cluster_labels
    
    cluster_colors = {i: NETWORK_COLORS[i % len(NETWORK_COLORS)] 
                      for i in range(n_clusters)}
    
    # -------------------------------------------------------------------------
    # Step 2: Build KNN graph (much faster than all-pairs cosine)
    # -------------------------------------------------------------------------
    print(f"  Building KNN graph (k={knn_k})...")
    nn = NearestNeighbors(n_neighbors=min(knn_k + 1, n_articles), 
                          metric='cosine', algorithm='brute')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    
    G = nx.Graph()
    for i in range(n_articles):
        G.add_node(i, cluster=cluster_labels[i])
    
    edge_count = 0
    for i in range(n_articles):
        for j_idx in range(1, indices.shape[1]):  # skip self (index 0)
            j = indices[i, j_idx]
            sim = 1.0 - distances[i, j_idx]  # cosine distance -> similarity
            
            if sim < similarity_threshold:
                continue
            
            if show_edge_within_only and cluster_labels[i] != cluster_labels[j]:
                continue
            
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=sim)
                edge_count += 1
    
    # Remove isolated nodes — they add nothing visual and hurt layout
    isolated = list(nx.isolates(G))
    if isolated:
        G.remove_nodes_from(isolated)
        print(f"  Removed {len(isolated)} isolated nodes")
    
    print(f"  Network: {G.number_of_nodes()} nodes, {edge_count:,} edges")
    
    # -------------------------------------------------------------------------
    # Step 3: Cluster-seeded spring layout (tighter initialization)
    # -------------------------------------------------------------------------
    print(f"  Computing force-directed layout ({layout_iterations} iterations)...")
    
    init_pos = {}
    cluster_angles = {c: 2 * np.pi * c / n_clusters for c in range(n_clusters)}
    
    for node in G.nodes():
        cluster = G.nodes[node]['cluster']
        angle = cluster_angles[cluster]
        r = 0.3 + np.random.uniform(0, 0.3)
        jitter_angle = angle + np.random.uniform(-np.pi / (n_clusters + 1),
                                                  np.pi / (n_clusters + 1))
        init_pos[node] = (r * np.cos(jitter_angle), r * np.sin(jitter_angle))
    
    if layout_k is None:
        layout_k = 4.0 / np.sqrt(G.number_of_nodes())
    
    pos = nx.spring_layout(
        G, pos=init_pos,
        k=layout_k,
        iterations=layout_iterations,
        seed=42,
        scale=layout_scale
    )
    
    # -------------------------------------------------------------------------
    # Step 4: Percentile-based zoom + cluster centroids
    # -------------------------------------------------------------------------
    all_xy = np.array(list(pos.values()))
    
    lo = (100 - zoom_percentile) / 2
    hi = 100 - lo
    x_lo, x_hi = np.percentile(all_xy[:, 0], [lo, hi])
    y_lo, y_hi = np.percentile(all_xy[:, 1], [lo, hi])
    
    x_pad = (x_hi - x_lo) * 0.15
    y_pad = (y_hi - y_lo) * 0.15
    view_xmin, view_xmax = x_lo - x_pad, x_hi + x_pad
    view_ymin, view_ymax = y_lo - y_pad, y_hi + y_pad
    
    view_cx = (view_xmin + view_xmax) / 2
    view_cy = (view_ymin + view_ymax) / 2
    view_rx = (view_xmax - view_xmin) / 2
    view_ry = (view_ymax - view_ymin) / 2
    
    cluster_centroids = {}
    cluster_sizes = {}
    for c in range(n_clusters):
        c_nodes = [n for n in G.nodes() if G.nodes[n]['cluster'] == c]
        cluster_sizes[c] = len(c_nodes)
        if c_nodes:
            positions_arr = np.array([pos[n] for n in c_nodes])
            cluster_centroids[c] = positions_arr.mean(axis=0)
    
    total_nodes = G.number_of_nodes()
    
    # -------------------------------------------------------------------------
    # Step 5: Generate cluster labels (skip tiny clusters)
    # -------------------------------------------------------------------------
    print("  Generating cluster labels...")
    cluster_label_text = {}
    for c in range(n_clusters):
        pct = cluster_sizes.get(c, 0) / total_nodes * 100 if total_nodes > 0 else 0
        if pct < min_cluster_pct:
            continue
        
        c_df = df[df['_cluster'] == c]
        if text_col in c_df.columns and len(c_df) > 0:
            texts = c_df[text_col].fillna('').astype(str).tolist()
            keywords = _get_cluster_keywords(texts, n_terms=3)
        else:
            keywords = f"Cluster {c}"
        
        # Add dominant seeded narrative if available
        if show_narrative and 'dominant_narrative' in c_df.columns and len(c_df) > 0:
            top_narr = c_df['dominant_narrative'].value_counts().index[0]
            narr_pct = c_df['dominant_narrative'].value_counts().iloc[0] / len(c_df)
            cluster_label_text[c] = f"{top_narr} ({narr_pct:.0%})\n{keywords}\n({pct:.1f}%)"
        else:
            cluster_label_text[c] = f"{keywords}\n({pct:.1f}%)"
    
    # Position labels around the visible boundary
    label_positions = {}
    used_angles = []
    for c in sorted(cluster_label_text.keys()):
        if c not in cluster_centroids:
            continue
        centroid = cluster_centroids[c]
        dx = centroid[0] - view_cx
        dy = centroid[1] - view_cy
        angle = np.arctan2(dy, dx)
        
        for ua in used_angles:
            if abs(angle - ua) < 0.20:
                angle += 0.22
        used_angles.append(angle)
        
        lx = view_cx + view_rx * label_radius_factor * np.cos(angle)
        ly = view_cy + view_ry * label_radius_factor * np.sin(angle)
        label_positions[c] = (lx, ly)
    
    # -------------------------------------------------------------------------
    # Step 6: Draw
    # -------------------------------------------------------------------------
    print("  Drawing...")
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Edges (colored by cluster for structure visibility)
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        cluster_c = G.nodes[u].get('cluster', 0)
        ax.plot([x0, x1], [y0, y1], 
                color=cluster_colors.get(cluster_c, '#666666'),
                alpha=edge_alpha, linewidth=edge_width,
                solid_capstyle='round')
    
    # Nodes (draw per-cluster for consistent coloring)
    for c in range(n_clusters):
        c_nodes = [n for n in G.nodes() if G.nodes[n]['cluster'] == c]
        if not c_nodes:
            continue
        node_positions = np.array([pos[n] for n in c_nodes])
        ax.scatter(
            node_positions[:, 0], node_positions[:, 1],
            c=cluster_colors[c],
            s=node_size,
            alpha=node_alpha,
            edgecolors='white', linewidths=0.4,
            zorder=2
        )
    
    # Labels with leader-line arrows
    for c in cluster_label_text:
        if c not in cluster_centroids or c not in label_positions:
            continue
        
        centroid = cluster_centroids[c]
        lp = label_positions[c]
        color = cluster_colors[c]
        ha = 'left' if lp[0] > view_cx else 'right'
        
        ax.annotate(
            cluster_label_text[c],
            xy=centroid,
            xytext=lp,
            fontsize=label_fontsize,
            fontweight='bold',
            color=color,
            ha=ha,
            va='center',
            linespacing=1.3,
            arrowprops=dict(
                arrowstyle='->', 
                color=color,
                alpha=0.8,
                lw=1.2,
                connectionstyle='arc3,rad=0.05'
            )
        )
    
    # Apply the zoom crop
    ax.set_xlim(view_xmin, view_xmax)
    ax.set_ylim(view_ymin, view_ymax)
    
    ax.axis('off')
    ax.set_aspect('equal')
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='#333333')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"  Saved: {save_path}")
    
    # Clean up temp column
    df.drop(columns=['_cluster'], inplace=True, errors='ignore')
    
    return fig, ax, G


# =============================================================================
# VARIANT: Community detection (Louvain) instead of KMeans
# =============================================================================

def plot_narrative_network_louvain(
    data,
    embeddings=None,
    text_col='title',
    title=None,
    figsize=(20, 18),
    knn_k=10,
    similarity_threshold=0.28,
    resolution=1.0,            # Louvain resolution (higher = more clusters)
    layout_iterations=250,
    node_size=120,             # large visible dots
    edge_alpha=0.15,           # clearly visible edges
    edge_width=0.7,            # thick edges
    label_fontsize=13,         # large readable labels
    min_community_pct=1.0,     # hide communities smaller than this % of corpus
    zoom_percentile=88,        # aggressive crop to dense core
    label_radius_factor=1.18,  # how far out labels sit
    show_narrative=True,       # show dominant seeded narrative per community
    time_period=None,          # filter to specific year/period (e.g., 2023)
    time_col='year',           # column used for time filtering
    save_path=None,
    dpi=200,
):
    """
    Dense article-level network using Louvain community detection.
    
    Key features:
    - Auto-zooms to the dense region (ignores outlier nodes for axis limits)
    - Removes isolated nodes before layout for tighter clustering
    - Labels include dominant seeded narrative + TF-IDF keywords
    - Time filtering to show network for specific years
    - Labels placed relative to the *visible* bounding box
    
    Parameters
    ----------
    show_narrative : bool
        If True and 'dominant_narrative' column exists, labels show the 
        dominant seeded narrative per community.
    time_period : int or str, optional
        Filter data to a specific year or time period before building network.
    time_col : str
        Column to filter on when time_period is specified.
    resolution : float
        Louvain resolution. Higher = more (smaller) communities. Try 0.8–2.0.
    """
    try:
        import community as community_louvain
    except ImportError:
        raise ImportError(
            "Louvain community detection requires: pip install python-louvain"
        )
    
    # Extract data
    if hasattr(data, 'results'):
        df = data.results.df.copy()
        embeddings = data.results.embeddings
        if title is None:
            title = f"{data.config.name}: Narrative Network (Louvain)"
    else:
        df = data.copy()
        if embeddings is None:
            raise ValueError("embeddings required when passing DataFrame")

    # Time-period filter
    if time_period is not None and time_col in df.columns:
        mask = df[time_col] == time_period
        df = df[mask].reset_index(drop=True)
        embeddings = embeddings[mask.values]
        if title and str(time_period) not in title:
            title = f"{title} ({time_period})"
        print(f"  Filtered to {time_period}: {len(df)} articles")
    
    if len(df) < 5:
        print(f"  ⚠ Too few articles ({len(df)}) for network visualization, skipping.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Too few articles for {time_period}', 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        return fig, ax, nx.Graph()
    
    n_articles = len(df)
    print(f"Building Louvain network for {n_articles:,} articles...")
    
    # -------------------------------------------------------------------------
    # Build KNN graph
    # -------------------------------------------------------------------------
    print(f"  Building KNN graph (k={knn_k})...")
    nn = NearestNeighbors(n_neighbors=min(knn_k + 1, n_articles),
                          metric='cosine', algorithm='brute')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    
    G = nx.Graph()
    for i in range(n_articles):
        G.add_node(i)
    
    for i in range(n_articles):
        for j_idx in range(1, indices.shape[1]):
            j = indices[i, j_idx]
            sim = 1.0 - distances[i, j_idx]
            if sim >= similarity_threshold and not G.has_edge(i, j):
                G.add_edge(i, j, weight=sim)
    
    # Remove isolated nodes (no edges) — they add nothing visual and hurt layout
    isolated = list(nx.isolates(G))
    if isolated:
        G.remove_nodes_from(isolated)
        print(f"  Removed {len(isolated)} isolated nodes")
    
    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges():,} edges")
    
    # -------------------------------------------------------------------------
    # Louvain community detection
    # -------------------------------------------------------------------------
    print(f"  Running Louvain (resolution={resolution})...")
    partition = community_louvain.best_partition(G, resolution=resolution, 
                                                  random_state=42)
    
    n_clusters = max(partition.values()) + 1
    print(f"  Discovered {n_clusters} communities")
    
    nx.set_node_attributes(G, partition, 'cluster')
    
    cluster_colors = {i: NETWORK_COLORS[i % len(NETWORK_COLORS)]
                      for i in range(n_clusters)}
    
    # -------------------------------------------------------------------------
    # Cluster-seeded layout with tighter initialization
    # -------------------------------------------------------------------------
    print(f"  Computing layout...")
    
    # Tighter initial clustering: smaller radius, less jitter
    init_pos = {}
    cluster_angles = {c: 2 * np.pi * c / n_clusters for c in range(n_clusters)}
    for node in G.nodes():
        cluster = partition[node]
        angle = cluster_angles[cluster]
        r = 0.3 + np.random.uniform(0, 0.3)
        jitter = angle + np.random.uniform(-np.pi / (n_clusters + 1), 
                                            np.pi / (n_clusters + 1))
        init_pos[node] = (r * np.cos(jitter), r * np.sin(jitter))
    
    # Stronger spring constant = tighter clusters
    k = 4.0 / np.sqrt(G.number_of_nodes())
    pos = nx.spring_layout(G, pos=init_pos, k=k,
                           iterations=layout_iterations, seed=42, scale=2.0)
    
    # -------------------------------------------------------------------------
    # Compute visible bounding box (percentile-based zoom)
    # -------------------------------------------------------------------------
    all_xy = np.array(list(pos.values()))
    
    lo = (100 - zoom_percentile) / 2
    hi = 100 - lo
    x_lo, x_hi = np.percentile(all_xy[:, 0], [lo, hi])
    y_lo, y_hi = np.percentile(all_xy[:, 1], [lo, hi])
    
    # Add padding (15% of range on each side)
    x_pad = (x_hi - x_lo) * 0.15
    y_pad = (y_hi - y_lo) * 0.15
    view_xmin, view_xmax = x_lo - x_pad, x_hi + x_pad
    view_ymin, view_ymax = y_lo - y_pad, y_hi + y_pad
    
    view_cx = (view_xmin + view_xmax) / 2
    view_cy = (view_ymin + view_ymax) / 2
    view_rx = (view_xmax - view_xmin) / 2
    view_ry = (view_ymax - view_ymin) / 2
    
    # -------------------------------------------------------------------------
    # Cluster centroids + labels (only for communities above min_community_pct)
    # -------------------------------------------------------------------------
    cluster_centroids = {}
    cluster_sizes = {}
    for c in range(n_clusters):
        c_nodes = [n for n in G.nodes() if partition.get(n, -1) == c]
        cluster_sizes[c] = len(c_nodes)
        if c_nodes:
            positions = np.array([pos[n] for n in c_nodes])
            cluster_centroids[c] = positions.mean(axis=0)
    
    total_nodes = G.number_of_nodes()
    
    # Generate labels only for clusters above size threshold
    cluster_label_text = {}
    df_temp = df.copy()
    # Map cluster labels for nodes still in graph
    cluster_map = {i: partition.get(i, -1) for i in range(n_articles)}
    df_temp['_cluster'] = [cluster_map.get(i, -1) for i in range(n_articles)]
    
    for c in range(n_clusters):
        pct = cluster_sizes.get(c, 0) / total_nodes * 100 if total_nodes > 0 else 0
        if pct < min_community_pct:
            continue
        
        c_df = df_temp[df_temp['_cluster'] == c]
        if text_col in c_df.columns and len(c_df) > 0:
            texts = c_df[text_col].fillna('').astype(str).tolist()
            keywords = _get_cluster_keywords(texts, n_terms=3)
        else:
            keywords = f"Community {c}"
        
        # Add dominant seeded narrative if available
        if show_narrative and 'dominant_narrative' in c_df.columns and len(c_df) > 0:
            top_narr = c_df['dominant_narrative'].value_counts().index[0]
            narr_pct = c_df['dominant_narrative'].value_counts().iloc[0] / len(c_df)
            cluster_label_text[c] = f"{top_narr} ({narr_pct:.0%})\n{keywords}\n({pct:.1f}%)"
        else:
            cluster_label_text[c] = f"{keywords}\n({pct:.1f}%)"
    
    # Position labels around the visible boundary
    label_positions = {}
    used_angles = []
    for c in sorted(cluster_label_text.keys()):
        if c not in cluster_centroids:
            continue
        centroid = cluster_centroids[c]
        
        # Angle from view center to centroid
        dx = centroid[0] - view_cx
        dy = centroid[1] - view_cy
        angle = np.arctan2(dy, dx)
        
        # Nudge if too close to an existing label
        for ua in used_angles:
            if abs(angle - ua) < 0.20:
                angle += 0.22
        used_angles.append(angle)
        
        # Place label at the boundary of the visible area
        lx = view_cx + view_rx * label_radius_factor * np.cos(angle)
        ly = view_cy + view_ry * label_radius_factor * np.sin(angle)
        label_positions[c] = (lx, ly)
    
    # -------------------------------------------------------------------------
    # Draw
    # -------------------------------------------------------------------------
    print("  Drawing...")
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Edges (only within-cluster, colored by cluster for structure visibility)
    for u, v in G.edges():
        if partition.get(u) == partition.get(v):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            cluster_c = partition.get(u, 0)
            ax.plot([x0, x1], [y0, y1],
                    color=cluster_colors.get(cluster_c, '#666666'),
                    alpha=edge_alpha, linewidth=edge_width,
                    solid_capstyle='round')
    
    # Nodes
    for c in range(n_clusters):
        c_nodes = [n for n in G.nodes() if partition.get(n, -1) == c]
        if not c_nodes:
            continue
        node_positions = np.array([pos[n] for n in c_nodes])
        ax.scatter(node_positions[:, 0], node_positions[:, 1],
                   c=cluster_colors[c], s=node_size, alpha=0.85,
                   edgecolors='white', linewidths=0.4, zorder=2)
    
    # Labels with leader-line arrows
    for c in cluster_label_text:
        if c not in cluster_centroids or c not in label_positions:
            continue
        centroid = cluster_centroids[c]
        lp = label_positions[c]
        ha = 'left' if lp[0] > view_cx else 'right'
        ax.annotate(
            cluster_label_text[c], xy=centroid, xytext=lp,
            fontsize=label_fontsize, fontweight='bold',
            color=cluster_colors[c], ha=ha, va='center', linespacing=1.3,
            arrowprops=dict(arrowstyle='->', color=cluster_colors[c],
                           alpha=0.8, lw=1.2, connectionstyle='arc3,rad=0.05')
        )
    
    # Apply the zoom crop
    ax.set_xlim(view_xmin, view_xmax)
    ax.set_ylim(view_ymin, view_ymax)
    
    ax.axis('off')
    # Note: no set_aspect('equal') — allows the layout to fill the figure
    # naturally without compressing when cluster distribution is uneven
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='#333333')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  Saved: {save_path}")
    
    df.drop(columns=['_cluster'], inplace=True, errors='ignore')
    return fig, ax, G, n_clusters