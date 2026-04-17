"""
Pokémon Data Showdown - Streamlit Dashboard
============================================
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pokémon Data Showdown",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

    .stApp {
        background: radial-gradient(ellipse at top left, #0f0c29, #302b63, #24243e);
        color: #f0f0f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * { color: #d0d8ff !important; }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 18px 22px;
        backdrop-filter: blur(12px);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #a78bfa; }
    .metric-card .label { font-size: 0.82rem; color: #94a3b8; margin-top: 4px; }

    /* Section headers */
    .section-header {
        font-size: 1.7rem; font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #38bdf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    /* Pokemon badge */
    .type-badge {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600; margin: 2px;
        background: rgba(167,139,250,0.25); color: #c4b5fd;
        border: 1px solid rgba(167,139,250,0.4);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] { color: #94a3b8 !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #a78bfa !important; border-bottom: 2px solid #a78bfa;
    }

    /* Plotly chart backgrounds */
    .js-plotly-plot { border-radius: 12px; }

    /* Team card */
    .team-card {
        background: linear-gradient(135deg, rgba(167,139,250,0.15), rgba(56,189,248,0.1));
        border: 1px solid rgba(167,139,250,0.3);
        border-radius: 14px; padding: 16px; margin: 6px 0;
        transition: all 0.2s;
    }
    .team-card:hover { background: rgba(167,139,250,0.25); transform: scale(1.01); }
    .pokemon-name { font-size: 1.1rem; font-weight: 700; color: #e2e8ff; }
    .pokemon-total { font-size: 1.5rem; font-weight: 900; color: #fbbf24; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Type Colours ───────────────────────────────────────────────────────────────
TYPE_COLORS = {
    "Normal": "#A8A878", "Fire": "#F08030", "Water": "#6890F0",
    "Electric": "#F8D030", "Grass": "#78C850", "Ice": "#98D8D8",
    "Fighting": "#C03028", "Poison": "#A040A0", "Ground": "#E0C068",
    "Flying": "#A890F0", "Psychic": "#F85888", "Bug": "#A8B820",
    "Rock": "#B8A038", "Ghost": "#705898", "Dragon": "#7038F8",
    "Dark": "#705848", "Steel": "#B8B8D0", "Fairy": "#EE99AC",
}

# ── Data loader ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Pokemon.csv")
    df["Type1"] = df["Type1"].str.strip()
    df["Type2"] = df["Type2"].str.strip().replace("", np.nan)
    df["Form"] = df["Form"].str.strip()
    base = df[df["Form"] == ""].copy()
    return df, base


@st.cache_data
def run_clustering(df, k=5):
    stat_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
    X = df[stat_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df = df.copy()
    df["Cluster"] = labels
    df["PCA1"] = X_pca[:, 0]
    df["PCA2"] = X_pca[:, 1]
    # Elbow & silhouette
    inertias, silhouettes = [], []
    for ki in range(2, 11):
        km_i = KMeans(n_clusters=ki, random_state=42, n_init=10)
        li = km_i.fit_predict(X_scaled)
        inertias.append(km_i.inertia_)
        silhouettes.append(silhouette_score(X_scaled, li))
    return df, inertias, silhouettes, pca, km, scaler


@st.cache_data
def build_team(df, max_total=2700, max_same_type=2, team_size=6):
    stat_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
    cands = df.sort_values("Total", ascending=False).copy()
    team, type_count, running = [], {}, 0
    for _, row in cands.iterrows():
        if len(team) == team_size:
            break
        t1 = row["Type1"]
        t2 = row["Type2"] if pd.notna(row.get("Type2")) else None
        if running + row["Total"] > max_total:
            continue
        if type_count.get(t1, 0) >= max_same_type:
            continue
        if t2 and type_count.get(t2, 0) >= max_same_type:
            continue
        team.append(row)
        running += row["Total"]
        type_count[t1] = type_count.get(t1, 0) + 1
        if t2:
            type_count[t2] = type_count.get(t2, 0) + 1
    return pd.DataFrame(team), running


df_all, df_base = load_data()
STAT_COLS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

# ── Default clustering (K=5) ──────────────────────────────────────────────────
df_clustered, inertias, silhouettes, pca_model, km_model, scaler_model = run_clustering(df_base, k=5)
CLUSTER_NAMES = {
    0: "Balanced Defenders",
    1: "Speed Sweepers",
    2: "Sp. Atk Nukers",
    3: "Powerhouses",
    4: "Fragile Supporters",
}
CLUSTER_COLORS = ["#EF476F", "#06D6A0", "#118AB2", "#FFD166", "#A78BFA"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("plots/pokemon_main_img.webp", use_container_width=True)
    st.markdown("## Filters")

    gen_options = sorted(df_base["Generation"].unique())
    sel_gens = st.multiselect("Generations", gen_options, default=gen_options)

    type_options = sorted(df_base["Type1"].unique())
    sel_types = st.multiselect("Primary Types", type_options, default=type_options)

    total_range = st.slider(
        "Total Stat Range",
        int(df_base["Total"].min()),
        int(df_base["Total"].max()),
        (int(df_base["Total"].min()), int(df_base["Total"].max())),
    )

    st.markdown("---")
    st.markdown("## Clustering")
    k_clusters = st.slider("Number of Clusters (K)", 2, 10, 5)

    st.markdown("---")
    st.markdown("## Team Builder")
    tb_max_total = st.slider("Max Combined Total", 1800, 3000, 2700, step=50)
    tb_max_type = st.slider("Max Same-Type", 1, 3, 2)

# ── Filter base dataframe ─────────────────────────────────────────────────────
mask = (
    df_base["Generation"].isin(sel_gens)
    & df_base["Type1"].isin(sel_types)
    & df_base["Total"].between(total_range[0], total_range[1])
)
df_filt = df_base[mask].copy()

# Re-cluster on filtered data if k changed
df_cl_filt, iner_filt, sil_filt, _, _, _ = run_clustering(df_filt, k=k_clusters)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center; font-size:2.8rem; font-weight:900;
    background:linear-gradient(90deg,#a78bfa,#38bdf8,#fb923c);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:4px;'>
     Pokémon Data Showdown
    </h1>
    <p style='text-align:center; color:#94a3b8; margin-bottom:28px;'>
    Exploring structure in Pokémon battle attributes
    </p>
    """,
    unsafe_allow_html=True,
)

# ── KPI row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (len(df_filt), "Pokémon Selected"),
    (df_filt["Type1"].nunique(), "Unique Types"),
    (int(df_filt["Total"].max()), "Highest Total"),
    (round(df_filt["Total"].mean(), 1), "Avg Total"),
    (df_filt["Generation"].nunique(), "Generations"),
]
for col, (val, label) in zip([k1, k2, k3, k4, k5], kpis):
    col.markdown(
        f"""<div class="metric-card">
              <div class="value">{val}</div>
              <div class="label">{label}</div>
            </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tabs = st.tabs(
    ["EDA", "Modeling", "Team Builder", "Pokedex Explorer"]
)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 - EDA
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<p class="section-header"> Exploratory Data Analysis</p>', unsafe_allow_html=True)

    # Row 1: Distribution + Avg Stats
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig_hist = px.histogram(
            df_filt, x="Total", nbins=35,
            title="Distribution of Total Base Stats",
            color_discrete_sequence=["#a78bfa"],
            template="plotly_dark",
        )
        fig_hist.add_vline(
            x=df_filt["Total"].mean(), line_dash="dash", line_color="#fbbf24",
            annotation_text=f"Mean: {df_filt['Total'].mean():.0f}",
            annotation_position="top right",
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d0d8ff"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with r1c2:
        avg_stats = df_filt[STAT_COLS].mean().reset_index()
        avg_stats.columns = ["Stat", "Average"]
        stat_colors = ["#EF476F", "#F78C6B", "#FFD166", "#06D6A0", "#118AB2", "#073B4C"]
        fig_bar = px.bar(
            avg_stats, x="Average", y="Stat", orientation="h",
            title="Average Stat Values",
            color="Stat", color_discrete_sequence=stat_colors,
            template="plotly_dark",
        )
        fig_bar.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d0d8ff"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Row 2: Type Count + Avg Total by Type
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        tc = df_filt["Type1"].value_counts().reset_index()
        tc.columns = ["Type", "Count"]
        tc["Color"] = tc["Type"].map(lambda x: TYPE_COLORS.get(x, "#888"))
        fig_type = px.bar(
            tc, x="Type", y="Count",
            title="Pokémon Count by Primary Type",
            color="Type",
            color_discrete_map=TYPE_COLORS,
            template="plotly_dark",
        )
        fig_type.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d0d8ff"),
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with r2c2:
        avg_by_type = (
            df_filt.groupby("Type1")["Total"].mean()
            .reset_index()
            .sort_values("Total", ascending=False)
        )
        avg_by_type.columns = ["Type", "Avg Total"]
        fig_avg = px.bar(
            avg_by_type, x="Type", y="Avg Total",
            title="Avg Total by Primary Type",
            color="Type",
            color_discrete_map=TYPE_COLORS,
            template="plotly_dark",
        )
        fig_avg.add_hline(
            y=df_filt["Total"].mean(), line_dash="dash",
            line_color="cyan", annotation_text="Overall avg",
        )
        fig_avg.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d0d8ff"),
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_avg, use_container_width=True)

    # Row 3: Correlation Heatmap + Boxplot by Generation
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        corr = df_filt[STAT_COLS + ["Total"]].corr().round(2)
        fig_hm = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Stat Correlation Matrix",
            template="plotly_dark",
        )
        fig_hm.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d0d8ff"),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    with r3c2:
        fig_box = px.box(
            df_filt, x="Generation", y="Total",
            title="Total Stats by Generation",
            color="Generation",
            template="plotly_dark",
        )
        fig_box.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d0d8ff"),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Row 4: Radar charts for top type profiles
    st.markdown("###  Stat Profiles - Top 6 Primary Types (Radar)")
    top6 = df_filt["Type1"].value_counts().head(6).index.tolist()
    radar_data = df_filt[df_filt["Type1"].isin(top6)].groupby("Type1")[STAT_COLS].mean()

    radar_cols = st.columns(3)
    for i, t in enumerate(top6):
        vals = radar_data.loc[t].values.tolist()
        vals_closed = vals + [vals[0]]
        cats_closed = STAT_COLS + [STAT_COLS[0]]
        fig_radar = go.Figure(
            go.Scatterpolar(
                r=vals_closed, theta=cats_closed,
                fill="toself",
                fillcolor=f"rgba({int(TYPE_COLORS.get(t,'#888')[1:3],16)},"
                           f"{int(TYPE_COLORS.get(t,'#888')[3:5],16)},"
                           f"{int(TYPE_COLORS.get(t,'#888')[5:7],16)},0.2)",
                line_color=TYPE_COLORS.get(t, "#888"),
                name=t,
            )
        )
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 130], showticklabels=True, tickfont=dict(size=8)),
                bgcolor="rgba(0,0,0,0)",
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            title=dict(text=f"Type: {t}", font=dict(color=TYPE_COLORS.get(t, "#fff"), size=13)),
            font=dict(color="#d0d8ff"),
            margin=dict(l=40, r=40, t=50, b=40),
            height=280,
        )
        radar_cols[i % 3].plotly_chart(fig_radar, use_container_width=True)

    # Row 5: Scatter - Attack vs Sp.Atk coloured by type
    st.markdown("###  Physical vs Special Attackers")
    fig_scatter = px.scatter(
        df_filt, x="Attack", y="Sp. Atk",
        color="Type1", color_discrete_map=TYPE_COLORS,
        hover_name="Name", hover_data=["Total", "Generation"],
        size="Total", size_max=22,
        title="Attack vs Sp. Atk (size = Total)",
        template="plotly_dark",
        opacity=0.75,
    )
    fig_scatter.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d0d8ff"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 - MODELING
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="section-header"> K-Means Clustering</p>', unsafe_allow_html=True)
    st.caption(f"Filtered dataset: **{len(df_filt)}** Pokémon | K = **{k_clusters}**")

    # Elbow + Silhouette
    k_range = list(range(2, 11))
    erow1, erow2 = st.columns(2)
    with erow1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(
            go.Scatter(x=k_range, y=iner_filt, mode="lines+markers",
                       marker=dict(size=8, color="#a78bfa"), line=dict(color="#a78bfa"),
                       name="Inertia")
        )
        fig_elbow.update_layout(
            title="Elbow Method", xaxis_title="K", yaxis_title="Inertia",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d0d8ff"),
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with erow2:
        fig_sil = go.Figure()
        fig_sil.add_trace(
            go.Scatter(x=k_range, y=sil_filt, mode="lines+markers",
                       marker=dict(size=8, color="#06D6A0"), line=dict(color="#06D6A0"),
                       name="Silhouette")
        )
        fig_sil.add_vline(
            x=k_range[int(np.argmax(sil_filt))],
            line_dash="dash", line_color="#FFD166",
            annotation_text=f"Best K={k_range[int(np.argmax(sil_filt))]}",
            annotation_position="top right",
        )
        fig_sil.update_layout(
            title="Silhouette Score vs K", xaxis_title="K", yaxis_title="Score",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d0d8ff"),
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    # PCA 2D scatter
    st.markdown("###  PCA Cluster Map")
    cluster_name_map = {i: f"Cluster {i}" for i in range(k_clusters)}
    df_cl_filt["Cluster_label"] = df_cl_filt["Cluster"].map(cluster_name_map)

    clr_seq = px.colors.qualitative.Bold[:k_clusters]
    fig_pca = px.scatter(
        df_cl_filt, x="PCA1", y="PCA2",
        color="Cluster_label",
        hover_name="Name",
        hover_data=["Type1", "Total"],
        color_discrete_sequence=clr_seq,
        opacity=0.7,
        title="2-D PCA Projection of Clusters",
        template="plotly_dark",
    )
    fig_pca.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d0d8ff"),
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    # Cluster stat profiles
    st.markdown("###  Cluster Mean Stat Profiles")
    cl_means = df_cl_filt.groupby("Cluster")[STAT_COLS].mean().reset_index()
    cl_means_long = cl_means.melt(id_vars="Cluster", var_name="Stat", value_name="Average")
    cl_means_long["Cluster_label"] = cl_means_long["Cluster"].map(cluster_name_map)

    fig_profile = px.bar(
        cl_means_long, x="Stat", y="Average",
        color="Cluster_label", barmode="group",
        color_discrete_sequence=clr_seq,
        title="Mean Stats Per Cluster",
        template="plotly_dark",
    )
    fig_profile.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d0d8ff"),
    )
    st.plotly_chart(fig_profile, use_container_width=True)

    # Cluster type composition
    st.markdown("###  Type Composition by Cluster")
    comp = df_cl_filt.groupby(["Cluster_label", "Type1"]).size().reset_index(name="Count")
    fig_comp = px.bar(
        comp, x="Cluster_label", y="Count", color="Type1",
        color_discrete_map=TYPE_COLORS,
        barmode="stack", title="Type Composition per Cluster",
        template="plotly_dark",
    )
    fig_comp.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d0d8ff"),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # 3D PCA
    if len(df_filt) >= 3:
        st.markdown("###  3-D PCA Projection")
        scaler3d = StandardScaler()
        X3 = scaler3d.fit_transform(df_cl_filt[STAT_COLS])
        pca3 = PCA(n_components=3, random_state=42)
        coords3 = pca3.fit_transform(X3)
        df_3d = df_cl_filt.copy()
        df_3d["PC1"], df_3d["PC2"], df_3d["PC3"] = coords3[:, 0], coords3[:, 1], coords3[:, 2]
        fig_3d = px.scatter_3d(
            df_3d, x="PC1", y="PC2", z="PC3",
            color="Cluster_label", hover_name="Name",
            color_discrete_sequence=clr_seq,
            opacity=0.65, template="plotly_dark",
            title="3-D PCA Cluster View",
        )
        fig_3d.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d0d8ff"),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 - TEAM BUILDER
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="section-header"> Optimal Team Builder</p>', unsafe_allow_html=True)

    st.info(
        f"**Constraints** - Combined Total <= **{tb_max_total}** | "
        f"No more than **{tb_max_type}** Pokemon of same type"
    )

    team_df, team_total = build_team(df_filt, max_total=tb_max_total, max_same_type=tb_max_type)

    if team_df.empty:
        st.error("No valid team could be built with the current filters & constraints. Relax filters.")
    else:
        tc1, tc2, tc3 = st.columns(3)
        tc1.markdown(
            f"""<div class="metric-card"><div class="value">{len(team_df)}</div>
            <div class="label">Team Size</div></div>""",
            unsafe_allow_html=True,
        )
        tc2.markdown(
            f"""<div class="metric-card"><div class="value">{team_total}</div>
            <div class="label">Combined Total</div></div>""",
            unsafe_allow_html=True,
        )
        remaining = tb_max_total - team_total
        tc3.markdown(
            f"""<div class="metric-card"><div class="value">{remaining}</div>
            <div class="label">Remaining Budget</div></div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Team cards
        st.markdown("###  Your Team")
        tm_cols = st.columns(3)
        for i, (_, row) in enumerate(team_df.iterrows()):
            t1 = row["Type1"]
            t2 = row["Type2"] if pd.notna(row.get("Type2")) else ""
            badge_color = TYPE_COLORS.get(t1, "#888")

            # Build type badges as plain strings   no nesting inside f-string
            t1_badge = (
                '<span class="type-badge" style="background:' + badge_color + '22;'
                'color:' + badge_color + ';border-color:' + badge_color + '55;">'
                + t1 + '</span>'
            )
            t2_badge = (
                '<span class="type-badge">' + t2 + '</span>'
                if t2 else ""
            )

            card_html = (
                '<div class="team-card">'
                  '<div class="pokemon-name">#' + str(i + 1) + " " + str(row["Name"]) + "</div>"
                  + t1_badge + t2_badge
                  + '<div style="margin-top:8px;display:flex;justify-content:space-between;align-items:center;">'
                    '<span style="color:#94a3b8;font-size:0.8rem;">Total</span>'
                    '<span class="pokemon-total">' + str(int(row["Total"])) + "</span>"
                  "</div>"
                  '<div style="margin-top:8px;font-size:0.75rem;color:#94a3b8;">'
                    "HP " + str(int(row["HP"])) +
                    " · Atk " + str(int(row["Attack"])) +
                    " · Def " + str(int(row["Defense"])) + "<br>"
                    "SpA " + str(int(row["Sp. Atk"])) +
                    " · SpD " + str(int(row["Sp. Def"])) +
                    " · Spd " + str(int(row["Speed"]))
                  + "</div>"
                "</div>"
            )
            tm_cols[i % 3].markdown(card_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Team visualizations
        tv1, tv2 = st.columns(2)
        with tv1:
            fig_team_bar = px.bar(
                team_df, y="Name", x="Total", orientation="h",
                color="Type1", color_discrete_map=TYPE_COLORS,
                title="Team Members - Total Stats",
                template="plotly_dark",
            )
            fig_team_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#d0d8ff"), showlegend=False,
            )
            st.plotly_chart(fig_team_bar, use_container_width=True)

        with tv2:
            team_long = team_df.melt(
                id_vars=["Name"], value_vars=STAT_COLS,
                var_name="Stat", value_name="Value",
            )
            fig_team_stack = px.bar(
                team_long, y="Name", x="Value", color="Stat",
                orientation="h", barmode="stack",
                title="Stat Breakdown",
                color_discrete_sequence=["#EF476F","#F78C6B","#FFD166","#06D6A0","#118AB2","#073B4C"],
                template="plotly_dark",
            )
            fig_team_stack.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#d0d8ff"),
            )
            st.plotly_chart(fig_team_stack, use_container_width=True)

        # Radar for team aggregate
        st.markdown("###  Team Aggregate Radar")
        team_avg = team_df[STAT_COLS].mean().values.tolist()
        team_avg_closed = team_avg + [team_avg[0]]
        fig_team_radar = go.Figure(
            go.Scatterpolar(
                r=team_avg_closed,
                theta=STAT_COLS + [STAT_COLS[0]],
                fill="toself",
                fillcolor="rgba(167,139,250,0.2)",
                line_color="#a78bfa",
                name="Team Avg",
            )
        )
        # Also plot overall avg
        overall_avg = df_filt[STAT_COLS].mean().values.tolist()
        overall_avg_closed = overall_avg + [overall_avg[0]]
        fig_team_radar.add_trace(
            go.Scatterpolar(
                r=overall_avg_closed,
                theta=STAT_COLS + [STAT_COLS[0]],
                fill="toself",
                fillcolor="rgba(56,189,248,0.1)",
                line_color="#38bdf8",
                line_dash="dash",
                name="Dataset Avg",
            )
        )
        fig_team_radar.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 160], showticklabels=True),
                bgcolor="rgba(0,0,0,0)",
            ),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d0d8ff"),
            legend=dict(font=dict(color="#d0d8ff")),
            title=dict(text="Team Avg vs Dataset Avg", font=dict(color="#d0d8ff")),
        )
        st.plotly_chart(fig_team_radar, use_container_width=True)

        # Download
        st.markdown("###  Export Team")
        csv = team_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            " Download Team CSV", csv, "optimal_team.csv", "text/csv",
        )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 - POKÉDEX EXPLORER
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="section-header"> Pokédex Explorer</p>', unsafe_allow_html=True)

    search = st.text_input(" Search Pokémon by name", "")
    results = df_filt[df_filt["Name"].str.contains(search, case=False, na=False)]

    st.markdown(f"Showing **{len(results)}** Pokémon")

    # Colour Total
    def colour_total(val):
        if val >= 600:
            return "color: #fbbf24; font-weight:700"
        elif val >= 500:
            return "color: #a78bfa"
        elif val >= 400:
            return "color: #38bdf8"
        return "color: #94a3b8"

    display_df = results[["Name", "Type1", "Type2", "Total"] + STAT_COLS + ["Generation"]].reset_index(drop=True)
    display_df["Type2"] = display_df["Type2"].fillna(" ")

    st.dataframe(
        display_df.style.applymap(colour_total, subset=["Total"]),
        use_container_width=True, height=400,
    )

    # Individual lookup
    st.markdown("---")
    st.markdown("###  Pokémon Detail View")
    pokemon_names = sorted(df_filt["Name"].unique())
    sel_poke = st.selectbox("Choose a Pokémon", pokemon_names)
    poke_row = df_filt[df_filt["Name"] == sel_poke].iloc[0]

    dc1, dc2 = st.columns([1, 2])
    with dc1:
        t1 = poke_row["Type1"]
        t2 = poke_row.get("Type2")
        t2_display = t2 if pd.notna(t2) else ""
        # Pre-compute type2 badge to avoid nested f-string issues
        t2_badge_html = (
            f'<span class="type-badge">{t2_display}</span>'
            if t2_display
            else ""
        )
        t1_color = TYPE_COLORS.get(t1, "#888")
        st.markdown(
            f"""<div class="team-card" style="padding:24px;">
              <div style="font-size:1.8rem; font-weight:900; color:#e2e8ff;">{poke_row['Name']}</div>
              <div style="margin:8px 0;">
                <span class="type-badge" style="background:{t1_color}33;
                color:{t1_color};border-color:{t1_color}55;">{t1}</span>
                {t2_badge_html}
              </div>
              <div style="margin-top:12px;">
                <span style="color:#94a3b8;">Gen</span>
                <b style="color:#fbbf24">{int(poke_row['Generation'])}</b>
                &nbsp;|&nbsp;
                <span style="color:#94a3b8;">Total</span>
                <b style="color:#a78bfa">{int(poke_row['Total'])}</b>
              </div>
              <div style="margin-top:16px; font-size:0.85rem; color:#94a3b8; line-height:1.9;">
                HP: <b style="color:#EF476F">{int(poke_row['HP'])}</b><br>
                Attack: <b style="color:#F78C6B">{int(poke_row['Attack'])}</b><br>
                Defense: <b style="color:#FFD166">{int(poke_row['Defense'])}</b><br>
                Sp. Atk: <b style="color:#06D6A0">{int(poke_row['Sp. Atk'])}</b><br>
                Sp. Def: <b style="color:#118AB2">{int(poke_row['Sp. Def'])}</b><br>
                Speed: <b style="color:#a78bfa">{int(poke_row['Speed'])}</b>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with dc2:
        vals = [poke_row[s] for s in STAT_COLS]
        vals_closed = vals + [vals[0]]
        fig_poke_radar = go.Figure(
            go.Scatterpolar(
                r=vals_closed, theta=STAT_COLS + [STAT_COLS[0]],
                fill="toself",
                fillcolor=f"rgba({int(TYPE_COLORS.get(t1,'#888')[1:3],16)},"
                           f"{int(TYPE_COLORS.get(t1,'#888')[3:5],16)},"
                           f"{int(TYPE_COLORS.get(t1,'#888')[5:7],16)},0.25)",
                line_color=TYPE_COLORS.get(t1, "#a78bfa"),
                name=poke_row["Name"],
            )
        )
        fig_poke_radar.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 200], showticklabels=True),
                bgcolor="rgba(0,0,0,0)",
            ),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d0d8ff"),
            title=dict(
                text=f"{poke_row['Name']} - Stat Radar",
                font=dict(color="#e2e8ff", size=15),
            ),
            height=380,
        )
        st.plotly_chart(fig_poke_radar, use_container_width=True)

    # Compare two Pokémon
    st.markdown("---")
    st.markdown("###  Compare Two Pokémon")
    cmp1, cmp2 = st.columns(2)
    pA = cmp1.selectbox("Pokémon A", pokemon_names, index=0, key="cmpA")
    pB = cmp2.selectbox("Pokémon B", pokemon_names, index=min(1, len(pokemon_names)-1), key="cmpB")
    rowA = df_filt[df_filt["Name"] == pA].iloc[0]
    rowB = df_filt[df_filt["Name"] == pB].iloc[0]

    fig_cmp = go.Figure()
    fig_cmp.add_trace(
        go.Bar(name=pA, x=STAT_COLS, y=[rowA[s] for s in STAT_COLS],
               marker_color="#a78bfa", opacity=0.85)
    )
    fig_cmp.add_trace(
        go.Bar(name=pB, x=STAT_COLS, y=[rowB[s] for s in STAT_COLS],
               marker_color="#38bdf8", opacity=0.85)
    )
    fig_cmp.update_layout(
        barmode="group", template="plotly_dark",
        title=f"{pA} vs {pB}",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d0d8ff"),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr style='border-color: rgba(255,255,255,0.1); margin-top:40px;'>
    <p style='text-align:center; color:#475569; font-size:0.8rem;'>
     Pokémon Data Showdown · Built with Streamlit & Plotly
    </p>
    """,
    unsafe_allow_html=True,
)
