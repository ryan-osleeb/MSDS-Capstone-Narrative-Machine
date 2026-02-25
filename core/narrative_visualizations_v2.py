"""
Narrative Visualizations Module
===============================
Complete visualization suite for narrative analysis including:
- Alluvial diagrams (narrative flow over time)
- Semantic networks with convex hulls
- Dense cluster networks (article-level)
- Semantic strength over time
- Expanded narrative networks
- Cluster documentation reports

Supports both:
1. Direct DataFrame + embeddings input
2. NarrativePipeline object input
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import networkx as nx
from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# COLOR UTILITIES
# =============================================================================

def get_narrative_colors(narratives):
    """Generate consistent color mapping for narratives."""
    colors = [
        '#00CED1',  # Cyan - Geopolitics
        '#FF6B6B',  # Coral/Red - Grid Impact  
        '#FFA500',  # Orange - Battery Tech
        '#32CD32',  # Green - Climate/Environment
        '#9370DB',  # Purple - Mainstream Adoption
        '#DC143C',  # Crimson - Barriers to Adoption
        '#4169E1',  # Royal Blue - Performance
        '#FFD700',  # Gold
        '#FF69B4',  # Hot Pink
        '#20B2AA',  # Light Sea Green
        '#8B4513',  # Saddle Brown
        '#4B0082',  # Indigo
    ]
    
    if hasattr(narratives, 'unique'):
        unique_narratives = sorted(narratives.unique())
    else:
        unique_narratives = sorted(set(narratives))
    
    return {narr: colors[i % len(colors)] for i, narr in enumerate(unique_narratives)}


def get_cluster_colors(n_clusters):
    """Generate colors for discovered clusters."""
    colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
        '#000000', '#dcbeff', '#469990', '#aa6e28', '#a9a9a9'
    ]
    return {i: colors[i % len(colors)] for i in range(n_clusters)}


def _extract_from_pipeline(pipeline):
    """Extract df and embeddings from NarrativePipeline object."""
    df = pipeline.results.df
    embeddings = pipeline.results.embeddings
    return df, embeddings


# =============================================================================
# 1. ALLUVIAL DIAGRAM
# =============================================================================

def plot_alluvial_diagram(data, time_col='year', narrative_col='dominant_narrative', 
                          title=None, figsize=(16, 10), min_flow=1, save_path=None):
    """
    Create an alluvial/Sankey-style diagram showing narrative flows over time.
    
    Parameters:
    -----------
    data : DataFrame or NarrativePipeline
    time_col : column name for time periods
    narrative_col : column name for narrative assignments
    title : plot title
    figsize : figure size tuple
    min_flow : minimum articles to show a flow
    save_path : path to save figure
    """
    # Handle pipeline input
    if hasattr(data, 'results'):
        df = data.results.df
        if title is None:
            title = f"{data.config.name}: Narrative Flows Over Time"
    else:
        df = data
    
    # Get time periods and narratives
    periods = sorted(df[time_col].dropna().unique())
    narratives = sorted(df[narrative_col].dropna().unique())
    color_map = get_narrative_colors(pd.Series(narratives))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate positions
    n_periods = len(periods)
    x_positions = np.linspace(0.1, 0.9, n_periods)
    
    # Calculate counts per period per narrative
    period_counts = {}
    for period in periods:
        counts = df[df[time_col] == period][narrative_col].value_counts()
        period_counts[period] = counts
    
    # Calculate y positions for each narrative in each period
    y_positions = {}
    for i, period in enumerate(periods):
        counts = period_counts[period]
        total = counts.sum()
        
        y_pos = {}
        current_y = 0.05
        available_height = 0.9
        
        for narr in narratives:
            count = counts.get(narr, 0)
            height = (count / total) * available_height if total > 0 else 0
            y_pos[narr] = (current_y, current_y + height)
            current_y += height + 0.01
        
        y_positions[period] = y_pos
    
    # Draw the flows between periods
    for i in range(len(periods) - 1):
        period1, period2 = periods[i], periods[i + 1]
        x1, x2 = x_positions[i], x_positions[i + 1]
        
        for narr in narratives:
            y1_start, y1_end = y_positions[period1].get(narr, (0, 0))
            y2_start, y2_end = y_positions[period2].get(narr, (0, 0))
            
            if y1_end - y1_start > 0.001 and y2_end - y2_start > 0.001:
                color = color_map[narr]
                
                n_points = 50
                t = np.linspace(0, 1, n_points)
                
                top_x = x1 + (x2 - x1) * t
                top_y = y1_end + (y2_end - y1_end) * (3*t**2 - 2*t**3)
                
                bot_x = x1 + (x2 - x1) * t
                bot_y = y1_start + (y2_start - y1_start) * (3*t**2 - 2*t**3)
                
                ax.fill_between(top_x, bot_y, top_y, alpha=0.4, color=color, 
                               edgecolor='none')
    
    # Draw the bars at each period
    bar_width = 0.03
    for i, period in enumerate(periods):
        x = x_positions[i]
        for narr in narratives:
            y_start, y_end = y_positions[period].get(narr, (0, 0))
            if y_end - y_start > 0.001:
                color = color_map[narr]
                rect = plt.Rectangle((x - bar_width/2, y_start), bar_width, 
                                     y_end - y_start, color=color, 
                                     edgecolor='white', linewidth=1)
                ax.add_patch(rect)
    
    # Add period labels
    for i, period in enumerate(periods):
        ax.text(x_positions[i], -0.02, str(int(period) if isinstance(period, float) else period), 
               ha='center', va='top', fontsize=12, fontweight='bold')
        total = period_counts[period].sum()
        ax.text(x_positions[i], -0.06, f'(n={total})', ha='center', va='top',
               fontsize=10, color='gray')
    
    # Create legend
    legend_elements = [mpatches.Patch(facecolor=color_map[narr], label=narr, alpha=0.7)
                      for narr in narratives if any(period_counts[p].get(narr, 0) > 0 
                                                    for p in periods)]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
             fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1)
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig, ax


# =============================================================================
# 2. SEMANTIC NETWORK WITH CONVEX HULLS
# =============================================================================

def plot_semantic_network(data, embeddings=None, time_period=None,
                          n_subclusters=15, narrative_col='dominant_narrative',
                          title=None, figsize=(16, 14), save_path=None):
    """
    Create semantic network visualization with convex hulls around narrative clusters.
    
    Parameters:
    -----------
    data : DataFrame or NarrativePipeline
    embeddings : numpy array (required if data is DataFrame)
    time_period : filter to specific time period (e.g., 2020)
    n_subclusters : number of sub-clusters to discover within narratives
    narrative_col : column for narrative assignments
    title : plot title
    figsize : figure size
    save_path : path to save figure
    """
    # Handle pipeline input
    if hasattr(data, 'results'):
        df = data.results.df.copy()
        embeddings = data.results.embeddings
        if title is None:
            title = f"{data.config.name}: Semantic Network"
    else:
        df = data.copy()
        if embeddings is None:
            raise ValueError("embeddings required when passing DataFrame")
    
    # Filter by time period if specified
    if time_period is not None:
        if 'year' in df.columns:
            mask = df['year'] == time_period
            df = df[mask].reset_index(drop=True)
            embeddings = embeddings[mask.values]
            if title:
                title = f"{title} ({time_period})"
        else:
            print("Warning: 'year' column not found, ignoring time_period filter")
    
    if len(df) < 3:
        print(f"Not enough data points ({len(df)})")
        return None, None
    
    # Compute t-SNE using the working pattern
    print("Computing t-SNE projection...")
    n_samples, n_features = embeddings.shape

    # Need at least 2 samples for PCA->tSNE to make sense
    if n_samples < 2:
        print(f"Skipping semantic network for {time_period}: only {n_samples} samples", flush=True)
        return
    
    # First reduce with PCA (like working code)
    if embeddings.shape[1] > 50:
        from sklearn.decomposition import PCA
        n_pca = min(50, n_samples - 1, n_features)
        pca = PCA(n_components=n_pca, random_state=123)
        embeddings_pca = pca.fit_transform(embeddings)
    else:
        embeddings_pca = embeddings
    
    # t-SNE with parameters from working code
    perplexity = min(30, max(2, (n_samples - 1) // 3))
    perplexity = min(perplexity, n_samples - 1)
    
    tsne = TSNE(n_components=2, random_state=123, perplexity=perplexity, max_iter=500, verbose=1)
    coords = tsne.fit_transform(embeddings_pca)
    
    df['x'] = coords[:, 0]
    df['y'] = coords[:, 1]
    
    narratives = sorted(df[narrative_col].dropna().unique())
    color_map = get_narrative_colors(pd.Series(narratives))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw convex hulls for each narrative
    for narr in narratives:
        narr_df = df[df[narrative_col] == narr]
        if len(narr_df) >= 3:
            points = narr_df[['x', 'y']].values
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])
                
                ax.fill(hull_points[:, 0], hull_points[:, 1],
                       alpha=0.15, color=color_map[narr])
                ax.plot(hull_points[:, 0], hull_points[:, 1],
                       color=color_map[narr], alpha=0.5, linewidth=2)
            except Exception as e:
                pass
    
    # Draw article points
    for narr in narratives:
        narr_df = df[df[narrative_col] == narr]
        ax.scatter(narr_df['x'], narr_df['y'], 
                  c=color_map[narr], label=f'{narr} ({len(narr_df)})',
                  s=50, alpha=0.7, edgecolors='white', linewidths=0.5)
    
    # Add narrative labels at centroids
    for narr in narratives:
        narr_df = df[df[narrative_col] == narr]
        if len(narr_df) > 0:
            centroid_x = narr_df['x'].mean()
            centroid_y = narr_df['y'].mean()
            ax.annotate(narr, (centroid_x, centroid_y),
                       fontsize=9, fontweight='bold', ha='center',
                       color=color_map[narr],
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', alpha=0.7))
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig, ax


# =============================================================================
# 3. SEMANTIC STRENGTH OVER TIME
# =============================================================================

def plot_semantic_strength_over_time(data, embeddings=None, time_col='year',
                                      narrative_col='dominant_narrative',
                                      title=None, figsize=(14, 8), save_path=None):
    """
    Plot semantic coherence/strength of each narrative over time.
    Measures how tightly clustered articles are within each narrative per time period.
    
    Parameters:
    -----------
    data : DataFrame or NarrativePipeline
    embeddings : numpy array (required if data is DataFrame)
    time_col : column for time periods
    narrative_col : column for narrative assignments
    title : plot title
    figsize : figure size
    save_path : path to save figure
    """
    # Handle pipeline input
    if hasattr(data, 'results'):
        df = data.results.df.copy()
        embeddings = data.results.embeddings
        if title is None:
            title = f"{data.config.name}: Narrative Semantic Strength Over Time"
    else:
        df = data.copy()
        if embeddings is None:
            raise ValueError("embeddings required when passing DataFrame")
    
    periods = sorted(df[time_col].dropna().unique())
    narratives = sorted(df[narrative_col].dropna().unique())
    color_map = get_narrative_colors(pd.Series(narratives))
    
    # Calculate semantic strength (avg pairwise similarity) per narrative per period
    strength_data = []
    
    for period in periods:
        period_mask = df[time_col] == period
        
        for narr in narratives:
            narr_mask = df[narrative_col] == narr
            combined_mask = period_mask & narr_mask
            
            indices = df[combined_mask].index.tolist()
            n_articles = len(indices)
            
            if n_articles >= 2:
                narr_embeddings = embeddings[indices]
                sim_matrix = cosine_similarity(narr_embeddings)
                upper_tri = sim_matrix[np.triu_indices(n_articles, k=1)]
                avg_similarity = upper_tri.mean()
            else:
                avg_similarity = np.nan
            
            strength_data.append({
                'period': period,
                'narrative': narr,
                'strength': avg_similarity,
                'n_articles': n_articles
            })
    
    strength_df = pd.DataFrame(strength_data)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for narr in narratives:
        narr_data = strength_df[strength_df['narrative'] == narr]
        ax.plot(narr_data['period'], narr_data['strength'], 
               marker='o', label=narr, color=color_map[narr],
               linewidth=2, markersize=8)
    
    ax.set_xlabel('Time Period', fontsize=12)
    ax.set_ylabel('Semantic Strength (Avg Cosine Similarity)', fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig, ax


# =============================================================================
# 3b. t-SNE CENTROIDS TIMELINE (Per-period t-SNE like working code)
# =============================================================================

def plot_tsne_centroids_timeline(data, embeddings=None, time_col='year',
                                  narrative_col='dominant_narrative',
                                  title=None, figsize=(18, 12),
                                  save_path=None):
    """
    Create t-SNE visualization for each time period showing narrative centroids.
    Computes t-SNE separately per period (not globally) for better local structure.
    
    Based on working code pattern.
    """
    # Handle pipeline input
    if hasattr(data, 'results'):
        df = data.results.df.copy()
        embeddings = data.results.embeddings
        if title is None:
            title = f"{data.config.name}: Semantic Space Over Time"
    else:
        df = data.copy()
        if embeddings is None:
            raise ValueError("embeddings required when passing DataFrame")
    
    # Get color map
    narratives = sorted(df[narrative_col].dropna().unique())
    color_map = get_narrative_colors(pd.Series(narratives))
    
    # Get periods
    periods = sorted(df[time_col].dropna().unique())
    
    # Pre-compute PCA on all embeddings (like working code)
    print("Running PCA dimensionality reduction...")
    pca = PCA(n_components=50, random_state=123)
    embeddings_pca = pca.fit_transform(embeddings)
    
    # Setup subplots
    n_periods = len(periods)
    n_cols = min(3, n_periods)
    n_rows = (n_periods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_periods == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, period in enumerate(periods):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        period_mask = df[time_col] == period
        period_count = period_mask.sum()
        
        if period_count < 10:
            ax.text(0.5, 0.5, f'{period}\nOnly {period_count} articles',
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue
        
        print(f"  {period}: {period_count} articles - Running t-SNE...")
        
        # Get period data
        period_df = df[period_mask].copy()
        period_embeddings = embeddings_pca[period_mask.values]
        
        # t-SNE for this period (like working code)
        perplexity = min(30, period_count // 4)
        perplexity = max(5, perplexity)
        
        tsne = TSNE(n_components=2, random_state=123, perplexity=perplexity, max_iter=500, verbose=1)
        period_coords = tsne.fit_transform(period_embeddings)
        
        period_df['tsne_x'] = period_coords[:, 0]
        period_df['tsne_y'] = period_coords[:, 1]
        
        # Calculate narrative centroids
        narrative_centroids = {}
        for narrative_name in narratives:
            narrative_mask = period_df[narrative_col] == narrative_name
            if narrative_mask.sum() >= 3:
                centroid_x = period_df.loc[narrative_mask, 'tsne_x'].mean()
                centroid_y = period_df.loc[narrative_mask, 'tsne_y'].mean()
                narrative_centroids[narrative_name] = (centroid_x, centroid_y)
        
        # Plot articles
        for narrative_name in narratives:
            mask = period_df[narrative_col] == narrative_name
            if mask.sum() > 0:
                ax.scatter(
                    period_df.loc[mask, 'tsne_x'],
                    period_df.loc[mask, 'tsne_y'],
                    c=color_map[narrative_name],
                    label=f"{narrative_name} ({mask.sum()})",
                    alpha=0.5,
                    s=40,
                    edgecolors='black',
                    linewidth=0.3
                )
        
        # Plot centroids
        for narrative_name, (cx, cy) in narrative_centroids.items():
            ax.scatter(cx, cy, 
                      c=color_map[narrative_name],
                      s=300, 
                      marker='*',
                      edgecolors='black',
                      linewidth=1.5,
                      zorder=10)
            
            # Add label
            ax.annotate(narrative_name[:15],
                       xy=(cx, cy),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=7,
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor=color_map[narrative_name],
                                alpha=0.7,
                                edgecolor='black'))
        
        ax.set_title(f'{period} ({period_count} articles)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide empty subplots
    for idx in range(n_periods, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    # Add legend to last visible subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=8)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig, axes


# =============================================================================
# 4. EXPANDED NETWORK (Sub-clusters as satellite nodes)
# =============================================================================

def plot_expanded_network_v2(data, embeddings=None, n_subclusters=15,
                              narrative_col='dominant_narrative',
                              text_col='headline', title=None, 
                              figsize=(18, 16), save_path=None):
    """
    Create expanded network showing seeded narratives at center with
    discovered sub-clusters as satellite nodes.
    
    Parameters:
    -----------
    data : DataFrame or NarrativePipeline
    embeddings : numpy array (required if data is DataFrame)
    n_subclusters : number of sub-clusters to discover
    narrative_col : column for narrative assignments
    text_col : column for text (used for sub-cluster labeling)
    title : plot title
    figsize : figure size
    save_path : path to save figure
    """
    # Handle pipeline input
    if hasattr(data, 'results'):
        df = data.results.df.copy()
        embeddings = data.results.embeddings
        if title is None:
            title = f"{data.config.name}: Expanded Narrative Network"
    else:
        df = data.copy()
        if embeddings is None:
            raise ValueError("embeddings required when passing DataFrame")
    
    narratives = sorted(df[narrative_col].dropna().unique())
    color_map = get_narrative_colors(pd.Series(narratives))
    
    # Discover sub-clusters
    print(f"Discovering {n_subclusters} sub-clusters...")
    kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
    df['subcluster'] = kmeans.fit_predict(embeddings)
    
    # Get sub-cluster labels using TF-IDF
    subcluster_labels = {}
    for sc_id in range(n_subclusters):
        sc_df = df[df['subcluster'] == sc_id]
        if len(sc_df) > 0 and text_col in df.columns:
            texts = sc_df[text_col].fillna('').astype(str).tolist()
            try:
                tfidf = TfidfVectorizer(max_features=50, stop_words='english',
                                       ngram_range=(1, 2), min_df=1)
                tfidf_matrix = tfidf.fit_transform(texts)
                mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
                top_idx = mean_tfidf.argsort()[-2:][::-1]
                features = tfidf.get_feature_names_out()
                label = ', '.join([features[i].title() for i in top_idx])
            except:
                label = f"SC{sc_id}"
        else:
            label = f"SC{sc_id}"
        subcluster_labels[sc_id] = label
    
    # Build network
    G = nx.Graph()
    
    # Add seeded narrative nodes (central)
    for narr in narratives:
        count = (df[narrative_col] == narr).sum()
        G.add_node(narr, node_type='narrative', count=count)
    
    # Add sub-cluster nodes
    for sc_id in range(n_subclusters):
        count = (df['subcluster'] == sc_id).sum()
        sc_df = df[df['subcluster'] == sc_id]
        if len(sc_df) > 0:
            dominant_narr = sc_df[narrative_col].mode().iloc[0]
        else:
            dominant_narr = narratives[0]
        
        G.add_node(f"SC{sc_id}", node_type='subcluster', 
                   count=count, label=subcluster_labels[sc_id],
                   dominant_narrative=dominant_narr)
    
    # Add edges between narratives and their sub-clusters
    for sc_id in range(n_subclusters):
        sc_df = df[df['subcluster'] == sc_id]
        for narr in narratives:
            overlap = (sc_df[narrative_col] == narr).sum()
            if overlap > 0:
                G.add_edge(narr, f"SC{sc_id}", weight=overlap)
    
    # Layout: narratives in center circle, sub-clusters around
    pos = {}
    
    n_narr = len(narratives)
    for i, narr in enumerate(narratives):
        angle = 2 * np.pi * i / n_narr
        pos[narr] = (0.4 * np.cos(angle), 0.4 * np.sin(angle))
    
    for sc_id in range(n_subclusters):
        sc_node = f"SC{sc_id}"
        dominant = G.nodes[sc_node]['dominant_narrative']
        base_x, base_y = pos[dominant]
        
        angle = np.arctan2(base_y, base_x)
        angle += np.random.uniform(-0.5, 0.5)
        r = 0.8 + np.random.uniform(0, 0.3)
        pos[sc_node] = (r * np.cos(angle), r * np.sin(angle))
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw edges
    for u, v, edge_data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = edge_data.get('weight', 1)
        ax.plot([x0, x1], [y0, y1], 'b--', alpha=0.3, 
               linewidth=0.5 + weight/20)
    
    # Draw narrative nodes
    for narr in narratives:
        x, y = pos[narr]
        count = G.nodes[narr]['count']
        size = 800 + count * 5
        ax.scatter(x, y, s=size, c=color_map[narr], 
                  edgecolors='white', linewidths=2, zorder=3)
        ax.annotate(narr, (x, y), fontsize=8, ha='center', va='center',
                   fontweight='bold', color='white')
    
    # Draw sub-cluster nodes
    for sc_id in range(n_subclusters):
        sc_node = f"SC{sc_id}"
        x, y = pos[sc_node]
        count = G.nodes[sc_node]['count']
        ax.scatter(x, y, s=200 + count * 3, c='#9370DB', marker='s',
                  edgecolors='blue', linewidths=1, zorder=2, alpha=0.8)
        ax.annotate(sc_node, (x, y), fontsize=7, ha='center', va='center',
                   color='white', fontweight='bold')
    
    legend_elements = [
        mpatches.Patch(facecolor='gray', edgecolor='white', label='Seeded Narratives'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#9370DB',
               markersize=10, label='Discovered Sub-clusters')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')
    
    if title:
        ax.set_title(f"{title}\n({len(narratives)} narratives + {n_subclusters} sub-clusters)",
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig, ax, G


# =============================================================================
# 5. CLUSTER-NARRATIVE DOCUMENTATION REPORT
# =============================================================================

def generate_cluster_narrative_report(data, embeddings=None, n_clusters=15,
                                       narrative_col='dominant_narrative',
                                       text_col='headline', 
                                       save_path=None) -> str:
    """
    Generate a markdown report documenting discovered clusters and their
    relationship to seeded narratives.
    """
    # Handle pipeline input
    if hasattr(data, 'results'):
        df = data.results.df.copy()
        embeddings = data.results.embeddings
        domain_name = data.config.name
    else:
        df = data.copy()
        domain_name = "Analysis"
        if embeddings is None:
            raise ValueError("embeddings required when passing DataFrame")
    
    print(f"Discovering {n_clusters} clusters for report...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(embeddings)
    
    narratives = sorted(df[narrative_col].dropna().unique())
    
    report_lines = [
        f"# {domain_name}: Cluster-Narrative Analysis Report",
        "",
        f"**Total Articles:** {len(df):,}",
        f"**Seeded Narratives:** {len(narratives)}",
        f"**Discovered Clusters:** {n_clusters}",
        "",
        "---",
        "",
        "## Narrative Distribution",
        "",
    ]
    
    for narr in narratives:
        count = (df[narrative_col] == narr).sum()
        pct = count / len(df) * 100
        report_lines.append(f"- **{narr}**: {count:,} articles ({pct:.1f}%)")
    
    report_lines.extend(["", "---", "", "## Discovered Clusters", ""])
    
    for cluster_id in range(n_clusters):
        cluster_df = df[df['cluster'] == cluster_id]
        n_articles = len(cluster_df)
        pct = n_articles / len(df) * 100
        
        if text_col in df.columns:
            texts = cluster_df[text_col].fillna('').astype(str).tolist()
            try:
                tfidf = TfidfVectorizer(max_features=100, stop_words='english',
                                       ngram_range=(1, 2), min_df=1)
                tfidf_matrix = tfidf.fit_transform(texts)
                mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
                top_idx = mean_tfidf.argsort()[-5:][::-1]
                features = tfidf.get_feature_names_out()
                top_terms = [features[i] for i in top_idx]
            except:
                top_terms = ["N/A"]
        else:
            top_terms = ["N/A"]
        
        narr_counts = cluster_df[narrative_col].value_counts()
        
        report_lines.extend([
            f"### Cluster {cluster_id}: {', '.join(top_terms[:3]).title()}",
            "",
            f"**Articles:** {n_articles:,} ({pct:.1f}%)",
            "",
            f"**Top Terms:** {', '.join(top_terms)}",
            "",
            "**Narrative Composition:**",
            ""
        ])
        
        for narr in narr_counts.head(5).index:
            count = narr_counts[narr]
            narr_pct = count / n_articles * 100
            report_lines.append(f"  - {narr}: {count} ({narr_pct:.0f}%)")
        
        if text_col in df.columns:
            report_lines.extend(["", "**Sample Articles:**", ""])
            for _, row in cluster_df.head(3).iterrows():
                headline = str(row.get(text_col, ''))[:100]
                report_lines.append(f"  - {headline}")
        
        report_lines.extend(["", "---", ""])
    
    report = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"  Saved: {save_path}")
    
    return report


# =============================================================================
# 6. DENSE CLUSTER NETWORK (Article-level, like reference image)
# =============================================================================

def plot_dense_cluster_network(data, embeddings=None, n_clusters=15,
                                title=None, figsize=(18, 16),
                                similarity_threshold=0.25,
                                edge_sample_rate=0.3,
                                node_size=20,
                                text_col='headline',
                                save_path=None):
    """
    Create a dense article network where clusters emerge from the data.
    Each node is an article, edges connect similar articles, clusters are
    discovered and labeled with top terms.
    """
    # Handle pipeline input
    if hasattr(data, 'results'):
        df = data.results.df.copy()
        embeddings = data.results.embeddings
        if title is None:
            title = f"{data.config.name}: Article Network (Discovered Clusters)"
    else:
        df = data.copy()
        if embeddings is None:
            raise ValueError("embeddings required when passing DataFrame")
    
    print(f"Clustering {len(df)} articles into {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    df['cluster'] = cluster_labels
    
    cluster_colors = get_cluster_colors(n_clusters)
    
    print("Building network...")
    n_articles = len(df)
    similarities = cosine_similarity(embeddings)
    
    G = nx.Graph()
    for i in range(n_articles):
        G.add_node(i, cluster=cluster_labels[i])
    
    print("Adding edges...")
    edge_list = []
    for i in range(n_articles):
        for j in range(i + 1, n_articles):
            if similarities[i, j] >= similarity_threshold:
                edge_list.append((i, j, similarities[i, j]))
    
    if len(edge_list) > 50000:
        np.random.shuffle(edge_list)
        edge_list = edge_list[:int(len(edge_list) * edge_sample_rate)]
    
    for i, j, w in edge_list:
        G.add_edge(i, j, weight=w)
    
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    print("Computing layout...")
    init_pos = {}
    cluster_angles = {c: 2 * np.pi * c / n_clusters for c in range(n_clusters)}
    for node in G.nodes():
        cluster = G.nodes[node]['cluster']
        angle = cluster_angles[cluster]
        r = 0.3 + np.random.uniform(0, 0.4)
        init_pos[node] = (r * np.cos(angle) + np.random.uniform(-0.15, 0.15),
                         r * np.sin(angle) + np.random.uniform(-0.15, 0.15))
    
    pos = nx.spring_layout(G, pos=init_pos, 
                          k=1.5/np.sqrt(G.number_of_nodes()),
                          iterations=150, seed=42, scale=2.0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    print("Drawing...")
    edge_alpha = max(0.02, 0.15 - len(edge_list) / 100000)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=edge_alpha, 
                          width=0.3, edge_color='#888888')
    
    for cluster_id in range(n_clusters):
        cluster_nodes = [n for n in G.nodes() if G.nodes[n]['cluster'] == cluster_id]
        if cluster_nodes:
            node_positions = np.array([pos[n] for n in cluster_nodes])
            ax.scatter(node_positions[:, 0], node_positions[:, 1],
                      c=cluster_colors[cluster_id], s=node_size,
                      alpha=0.8, edgecolors='white', linewidths=0.2)
    
    if text_col in df.columns:
        print("Adding labels...")
        for cluster_id in range(n_clusters):
            cluster_df = df[df['cluster'] == cluster_id]
            cluster_pct = len(cluster_df) / len(df) * 100
            
            if len(cluster_df) > 0:
                texts = cluster_df[text_col].fillna('').astype(str).tolist()
                try:
                    tfidf = TfidfVectorizer(max_features=100, stop_words='english',
                                           ngram_range=(1, 2), min_df=1)
                    tfidf_matrix = tfidf.fit_transform(texts)
                    mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
                    top_idx = mean_tfidf.argsort()[-3:][::-1]
                    features = tfidf.get_feature_names_out()
                    top_terms = [features[i].title() for i in top_idx]
                    label = ', '.join(top_terms)
                except:
                    label = f"Cluster {cluster_id}"
                
                label = f"{label} ({cluster_pct:.1f}%)"
                
                cluster_nodes = [n for n in G.nodes() if G.nodes[n]['cluster'] == cluster_id]
                if cluster_nodes:
                    positions = np.array([pos[n] for n in cluster_nodes])
                    centroid = positions.mean(axis=0)
                    
                    direction = centroid / (np.linalg.norm(centroid) + 0.001)
                    label_pos = centroid + direction * 0.15
                    
                    ax.annotate(label, xy=centroid, xytext=label_pos,
                               fontsize=8, fontweight='bold',
                               color=cluster_colors[cluster_id],
                               ha='center', va='center',
                               arrowprops=dict(arrowstyle='->', 
                                              color=cluster_colors[cluster_id],
                                              alpha=0.7, lw=1))
    
    ax.axis('off')
    ax.set_aspect('equal')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig, ax, G, df


# =============================================================================
# DEMO FUNCTION
# =============================================================================

def demo_visualizations():
    """Generate demo visualizations with synthetic data."""
    
    np.random.seed(42)
    
    n_articles = 300
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    narratives = ['Geopolitics', 'Grid Impact', 'Battery Tech', 
                  'Climate/Environment', 'Mainstream Adoption', 
                  'Barriers to Adoption', 'Performance']
    
    topics = ['Tesla', 'Ford', 'GM', 'Rivian', 'BYD', 'VW', 'charging', 'battery', 
              'sales', 'market', 'policy', 'climate', 'range', 'price', 'tariff']
    
    headlines = []
    for i in range(n_articles):
        t1, t2 = np.random.choice(topics, 2, replace=False)
        headlines.append(f'{t1} {t2} {np.random.choice(["news", "update", "report"])}')
    
    data = {
        'year': np.random.choice(years, n_articles),
        'dominant_narrative': np.random.choice(narratives, n_articles),
        'headline': headlines
    }
    df = pd.DataFrame(data)
    
    n_topics = 15
    topic_centroids = {i: np.random.randn(384) for i in range(n_topics)}
    
    embeddings = []
    for i, row in df.iterrows():
        topic1 = hash(row['headline'].split()[0]) % n_topics
        topic2 = hash(row['headline'].split()[1]) % n_topics
        base = 0.6 * topic_centroids[topic1] + 0.4 * topic_centroids[topic2]
        noise = np.random.randn(384) * 0.5
        embeddings.append(base + noise)
    
    embeddings = np.array(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    print("Generating demo visualizations...")
    output_dir = Path('/home/claude')
    
    fig1, _ = plot_alluvial_diagram(df, title='Demo: Narrative Flows')
    fig1.savefig(output_dir / 'demo_alluvial.png', dpi=150, bbox_inches='tight')
    print("Saved: demo_alluvial.png")
    
    fig2, _ = plot_semantic_network(df, embeddings, title='Demo: Semantic Network')
    fig2.savefig(output_dir / 'demo_semantic_network.png', dpi=150, bbox_inches='tight')
    print("Saved: demo_semantic_network.png")
    
    fig3, _ = plot_semantic_strength_over_time(df, embeddings, title='Demo: Semantic Strength')
    fig3.savefig(output_dir / 'demo_strength.png', dpi=150, bbox_inches='tight')
    print("Saved: demo_strength.png")
    
    fig4, _, _ = plot_expanded_network_v2(df, embeddings, n_subclusters=12, title='Demo: Expanded Network')
    fig4.savefig(output_dir / 'demo_expanded.png', dpi=150, bbox_inches='tight')
    print("Saved: demo_expanded.png")
    
    fig5, _, _, _ = plot_dense_cluster_network(df, embeddings, n_clusters=15,
                                                title='Demo: Dense Cluster Network',
                                                similarity_threshold=0.20)
    fig5.savefig(output_dir / 'demo_dense_cluster.png', dpi=150, bbox_inches='tight')
    print("Saved: demo_dense_cluster.png")
    
    report = generate_cluster_narrative_report(df, embeddings, n_clusters=10,
                                               save_path=output_dir / 'demo_report.md')
    print("Saved: demo_report.md")
    
    plt.close('all')
    print("\nDemo complete!")
    
    return df, embeddings


if __name__ == '__main__':
    demo_visualizations()