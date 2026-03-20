import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import Counter
from datetime import datetime, timedelta

st.set_page_config(page_title="Bout Comparison Analysis", layout="wide")

# ── large-font CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-size: 22px; }
p, div, span, label { font-size: 22px !important; }
h1 { font-size: 3.0rem !important; font-weight: 700 !important; }
h2 { font-size: 2.2rem !important; font-weight: 600 !important; margin-top: 1.8rem !important; }
h3 { font-size: 1.6rem !important; font-weight: 600 !important; }
.stButton button         { font-size: 22px !important; padding: 0.85rem 1.75rem !important; font-weight: 500 !important; }
.stDownloadButton button { font-size: 22px !important; padding: 0.85rem 1.75rem !important; }
.stSelectbox label       { font-size: 22px !important; font-weight: 600 !important; }
.stTextInput label       { font-size: 22px !important; font-weight: 600 !important; }
.stNumberInput label     { font-size: 22px !important; font-weight: 600 !important; }
.stDateInput label       { font-size: 22px !important; font-weight: 600 !important; }
.stRadio label           { font-size: 22px !important; }
.stRadio > div > label   { font-size: 22px !important; }
.stMultiSelect label     { font-size: 22px !important; font-weight: 600 !important; }
.stCheckbox label        { font-size: 22px !important; }
.stFileUploader label    { font-size: 22px !important; font-weight: 600 !important; }
.stFileUploader div      { font-size: 22px !important; }
.streamlit-expanderHeader{ font-size: 22px !important; font-weight: 500 !important; }
.stAlert > div           { font-size: 22px !important; }
.stMarkdown p            { font-size: 22px !important; }
.stSelectbox div[data-baseweb="select"] > div { font-size: 22px !important; }
[role="option"]          { font-size: 22px !important; }
[data-testid="stMetricValue"] { font-size: 2.8rem !important; font-weight: 600 !important; }
[data-testid="stMetricLabel"] { font-size: 1.5rem !important; font-weight: 500 !important; }
.stTabs [data-baseweb="tab"] { font-size: 22px !important; font-weight: 500 !important; padding: 12px 20px !important; }
.dataframe th { font-size: 20px !important; font-weight: 600 !important; padding: 12px !important; }
.dataframe td { font-size: 20px !important; padding: 10px !important; }
/* Streamlit dataframe / AgGrid newer selectors */
[data-testid="stDataFrame"] * { font-size: 18px !important; }
[data-testid="stDataFrame"] th { font-size: 18px !important; font-weight: 600 !important; }
[data-testid="stDataFrame"] td { font-size: 18px !important; }
.dvn-scroller * { font-size: 18px !important; }
.stDataFrameContainer * { font-size: 18px !important; }
[class*="glideDataEditor"] * { font-size: 18px !important; }
/* force iframe content font where possible */
[data-testid="stDataFrame"] canvas { font-size: 18px !important; }
[data-testid="stDataFrame"] > div { font-size: 18px !important; }
div[data-testid="stSidebarContent"] p,
div[data-testid="stSidebarContent"] label,
div[data-testid="stSidebarContent"] div,
div[data-testid="stSidebarContent"] span { font-size: 18px !important; }
div[data-testid="stSidebarContent"] h2 { font-size: 1.6rem !important; }
div[data-testid="stSidebarContent"] .stSelectbox label,
div[data-testid="stSidebarContent"] .stMultiSelect label,
div[data-testid="stSidebarContent"] .stNumberInput label { font-size: 18px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🔬 Bout Comparison Analysis</h1>", unsafe_allow_html=True)
st.markdown("Compare bout recordings from two observers, instruments, or annotation sources.")

# ── helpers ───────────────────────────────────────────────────────────────────

def load_csv(source, sep):
    if hasattr(source, "read"):
        raw = source.read().decode()
    else:
        raw = source
    from io import StringIO
    df = pd.read_csv(StringIO(raw), sep=sep, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df


def prepare_df(df, group_col, start_col, end_col, extra_cols):
    out = df[[group_col, start_col, end_col] + extra_cols].copy()
    out = out.rename(columns={group_col: "_group",
                               start_col: "_start_raw",
                               end_col:   "_end_raw"})
    out["_start_dt"] = pd.to_datetime(out["_start_raw"], dayfirst=True)
    out["_end_dt"]   = pd.to_datetime(out["_end_raw"],   dayfirst=True)
    out["_start_s"]  = out["_start_dt"].astype(np.int64) // 10**9
    out["_end_s"]    = out["_end_dt"].astype(np.int64)   // 10**9
    out["_dur_s"]    = out["_end_s"] - out["_start_s"]
    out = out[out["_dur_s"] > 0].reset_index(drop=True)
    out["_bout_id"]  = out["_group"].astype(str) + "_" + (out.index + 1).astype(str)
    return out


# grp2_name is used inside categorize; set as a module-level var after UI resolves
_grp2_label = ""

def categorize(bout, others):
    """Returns (category_str, matched_bout_ids_str)."""
    s1, e1 = bout["_start_s"], bout["_end_s"]
    overlapping = [o for o in others
                   if min(e1, o["_end_s"]) - max(s1, o["_start_s"]) > 0]
    if len(overlapping) == 0:
        return "No overlap", ""
    ids = ", ".join(str(o.get("_bout_id", "?")) for o in overlapping)
    if len(overlapping) > 1:
        return "Spans multiple", ids
    o = overlapping[0]
    s2, e2 = o["_start_s"], o["_end_s"]
    if s1 == s2 and e1 == e2:
        return "Identical", ids
    if s1 >= s2 and e1 <= e2:
        return f"Within {_grp2_label} bout", ids
    if s1 <= s2 and e1 >= e2:
        return f"Contains {_grp2_label} bout", ids
    return "Partial overlap", ids


def build_categories(primary_df, reference_df):
    ref = reference_df.to_dict("records")
    results = [categorize(row, ref) for _, row in primary_df.iterrows()]
    cats = [r[0] for r in results]
    match_ids = [r[1] for r in results]
    return cats, match_ids


def second_sets(df):
    s = set()
    for _, r in df.iterrows():
        s.update(range(int(r["_start_s"]), int(r["_end_s"])))
    return s


BASE_CAT_COLORS = {
    "Identical":       "#1d9e75",
    "Partial overlap": "#ef9f27",
    "Spans multiple":  "#ba7517",
    "No overlap":      "#e24b4a",
}

def cat_color(cat):
    if cat in BASE_CAT_COLORS:
        return BASE_CAT_COLORS[cat]
    if "Within" in cat:
        return "#378add"
    if "Contains" in cat:
        return "#5863c4"
    return "#888888"


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h2>Data & settings</h2>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV file", type=["csv", "txt"])
    pasted   = st.text_area(
        "…or paste CSV text", height=150,
        placeholder="observer;start_time;end_time;animal\n"
                    "ERKU;01/10/2025 09:59:56;01/10/2025 10:00:20;3061"
    )
    sep = st.selectbox(
        "Column separator",
        options=[";", ",", "\t", "|"],
        format_func=lambda x: {";": "Semicolon (;)", ",": "Comma (,)",
                                "\t": "Tab", "|": "Pipe (|)"}[x]
    )

    data_ready = False
    df_raw     = None
    df_prep    = None

    if uploaded or pasted.strip():
        try:
            df_raw = load_csv(uploaded if uploaded else pasted, sep)
            st.success(f"Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns.")
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if df_raw is not None:
        st.markdown("---")
        st.markdown("**Column mapping**")
        cols = df_raw.columns.tolist()

        group_col = st.selectbox("Group column (observer / instrument / …)", cols, index=0)

        default_start = next((i for i, c in enumerate(cols) if "start" in c.lower()), 1)
        default_end   = next((i for i, c in enumerate(cols) if "end"   in c.lower()), 2)
        start_col = st.selectbox("Start time column", cols, index=default_start)
        end_col   = st.selectbox("End time column",   cols, index=default_end)

        extra_options = [c for c in cols if c not in [group_col, start_col, end_col]]
        extra_cols = st.multiselect("Extra columns to keep (optional)", extra_options,
                                     default=extra_options[:2] if extra_options else [])

        st.markdown("---")
        st.markdown("**Select groups to compare**")

        try:
            df_prep = prepare_df(df_raw, group_col, start_col, end_col, extra_cols)
            groups  = sorted(df_prep["_group"].unique())

            grp1 = st.selectbox("Group 1", groups, index=0)
            grp2 = st.selectbox("Group 2", groups, index=1 if len(groups) > 1 else 0)

            if grp1 == grp2:
                st.warning("Please select two different groups.")
            else:
                st.markdown("---")
                st.markdown("**Animal / subject column**")
                animal_none = "(none — compare all bouts together)"
                animal_col_options = [animal_none] + [c for c in extra_cols]
                animal_col = st.selectbox(
                    "Compare within each animal separately",
                    animal_col_options,
                    help="If set, bouts are only compared to bouts of the same animal in the other group."
                )
                use_animal = animal_col != animal_none
                data_ready = True
        except Exception as e:
            st.error(f"Could not parse timestamps: {e}")
    else:
        st.info("Upload or paste a CSV to get started.")

# ── guard ─────────────────────────────────────────────────────────────────────

if not data_ready:
    st.markdown("""
    ### How to use
    1. Upload a CSV or paste text in the sidebar
    2. Map the group, start-time and end-time columns
    3. Pick the two groups to compare (observer, instrument, annotation method, …)
    4. Explore the **Analysis** and **Timeline** tabs

    **Expected format:** any semicolon- or comma-separated file with at minimum a group
    column and start/end timestamp columns (`DD/MM/YYYY HH:MM:SS`).
    """)
    st.stop()

# resolve global label used inside categorize()
_grp2_label = grp2

df1 = df_prep[df_prep["_group"] == grp1].reset_index(drop=True)
df2 = df_prep[df_prep["_group"] == grp2].reset_index(drop=True)

def dur_shared(a1, a2):
    """
    Compute shared / only1 / only2 seconds by direct interval intersection,
    no deduplication needed — bouts within the same observer+animal do not
    overlap by construction.
    """
    # collect intervals as sorted list of (start, end) in epoch-seconds
    def intervals(df):
        return sorted(zip(df["_start_s"].astype(int), df["_end_s"].astype(int)))

    def total_dur(ivs):
        return sum(e - s for s, e in ivs)

    def intersect(ivs1, ivs2):
        """Sum of pairwise overlapping seconds between two interval lists."""
        total = 0
        j = 0
        for s1, e1 in ivs1:
            while j < len(ivs2) and ivs2[j][1] <= s1:
                j += 1
            k = j
            while k < len(ivs2) and ivs2[k][0] < e1:
                total += min(e1, ivs2[k][1]) - max(s1, ivs2[k][0])
                k += 1
        return total

    iv1 = intervals(a1)
    iv2 = intervals(a2)
    a_shared = intersect(iv1, iv2)
    a_total1 = total_dur(iv1)
    a_total2 = total_dur(iv2)
    return a_shared, a_total1 - a_shared, a_total2 - a_shared, a_total1, a_total2


if use_animal and animal_col in df1.columns:
    # ── per-animal comparison ──────────────────────────────────────────────
    animals = sorted(set(df1[animal_col].dropna().astype(str).unique()) |
                     set(df2[animal_col].dropna().astype(str).unique()))

    # use index-keyed dicts so categories are assigned back to the correct rows
    # regardless of the order animals are iterated
    cats1_by_idx  = {}
    match1_by_idx = {}
    cats2_by_idx  = {}
    match2_by_idx = {}
    shared, only1, only2, total1, total2 = 0, 0, 0, 0, 0
    per_animal_results = {}

    for animal in animals:
        a1 = df1[df1[animal_col].astype(str) == animal]
        a2 = df2[df2[animal_col].astype(str) == animal]

        _grp2_label = grp2
        ac1, am1 = build_categories(a1, a2)
        for idx, cat, mid in zip(a1.index, ac1, am1):
            cats1_by_idx[idx]  = cat
            match1_by_idx[idx] = mid

        _grp2_label = grp1
        ac2, am2 = build_categories(a2, a1)
        for idx, cat, mid in zip(a2.index, ac2, am2):
            cats2_by_idx[idx]  = cat
            match2_by_idx[idx] = mid
        _grp2_label = grp2

        a_shared, a_only1, a_only2, a_total1, a_total2 = dur_shared(a1, a2)
        shared += a_shared;  only1 += a_only1;  only2 += a_only2
        total1 += a_total1;  total2 += a_total2

        per_animal_results[animal] = {
            "cats1": ac1, "cats2": ac2,
            "n1": len(a1), "n2": len(a2),
            "shared": a_shared, "only1": a_only1, "only2": a_only2,
            "total1": a_total1, "total2": a_total2,
        }

    # reconstruct in original df row order
    cats1   = [cats1_by_idx[i]  for i in df1.index]
    match1  = [match1_by_idx[i] for i in df1.index]
    cats2   = [cats2_by_idx[i]  for i in df2.index]
    match2  = [match2_by_idx[i] for i in df2.index]

else:
    # ── no animal column: compare all bouts directly ───────────────────────
    _grp2_label = grp2
    cats1, match1 = build_categories(df1, df2)
    _grp2_label = grp1
    cats2, match2 = build_categories(df2, df1)
    _grp2_label = grp2

    shared, only1, only2, total1, total2 = dur_shared(df1, df2)

df1 = df1.copy(); df1["_cat"] = cats1; df1["_match_ids"] = match1
df2 = df2.copy(); df2["_cat"] = cats2; df2["_match_ids"] = match2

all_cats_seen = sorted(set(cats1) | set(cats2))
CAT_ORDER_DYNAMIC = (
    ["Identical"]
    + [c for c in all_cats_seen if "Within"   in c]
    + [c for c in all_cats_seen if "Contains" in c]
    + ["Partial overlap", "Spans multiple", "No overlap"]
)
CAT_ORDER_DYNAMIC = list(dict.fromkeys(CAT_ORDER_DYNAMIC))

tab_analysis, tab_timeline = st.tabs(["📊 Analysis", "🔍 Timeline"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 – ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab_analysis:

    if use_animal and animal_col in df1.columns:
        n_animals = len(set(df1[animal_col].dropna().astype(str)) |
                        set(df2[animal_col].dropna().astype(str)))
        st.info(f"🐾 Comparing bouts **within each animal** (column: `{animal_col}`, "
                f"{n_animals} animals found). Each bout is only matched against bouts "
                f"of the same animal in the other group.")
    else:
        st.warning("⚠️ Comparing **all bouts regardless of animal**. "
                   "Set an animal column in the sidebar to compare within animals.")

    st.markdown("<h2>Bout categorisation</h2>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    def counts_table(cats, total):
        c = Counter(cats)
        return pd.DataFrame([
            {"Category": cat, "Bouts": c.get(cat, 0),
             "%": f"{c.get(cat,0)/total*100:.0f}%"}
            for cat in CAT_ORDER_DYNAMIC
        ])

    def color_cat(val):
        return f"color: {cat_color(val)}"

    def render_counts_table(cats, total, label_a, label_b):
        """Render category counts as an HTML table so font CSS applies."""
        rows_html = ""
        c = Counter(cats)
        for cat in CAT_ORDER_DYNAMIC:
            n = c.get(cat, 0)
            pct = f"{n/total*100:.0f}%" if total else "—"
            color = cat_color(cat)
            rows_html += (
                f"<tr>"
                f"<td style='color:{color};font-size:20px;padding:8px 12px;'>{cat}</td>"
                f"<td style='font-size:20px;padding:8px 12px;text-align:right;font-weight:600;'>{n}</td>"
                f"<td style='font-size:20px;padding:8px 12px;text-align:right;color:#888;'>{pct}</td>"
                f"</tr>"
            )
        html = (
            f"<table style='width:100%;border-collapse:collapse;'>"
            f"<thead><tr>"
            f"<th style='font-size:20px;padding:8px 12px;text-align:left;border-bottom:2px solid #ddd;'>Category</th>"
            f"<th style='font-size:20px;padding:8px 12px;text-align:right;border-bottom:2px solid #ddd;'>Bouts</th>"
            f"<th style='font-size:20px;padding:8px 12px;text-align:right;border-bottom:2px solid #ddd;'>%</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table>"
        )
        return html

    with col_a:
        st.markdown(f"**{grp1}** bouts vs {grp2} &nbsp; *(n = {len(df1)})*")
        st.markdown(render_counts_table(cats1, len(df1), grp1, grp2), unsafe_allow_html=True)

    with col_b:
        st.markdown(f"**{grp2}** bouts vs {grp1} &nbsp; *(n = {len(df2)})*")
        st.markdown(render_counts_table(cats2, len(df2), grp2, grp1), unsafe_allow_html=True)

    # ── duration overlap ──────────────────────────────────────────────────
    st.markdown("<h2>Duration overlap (seconds)</h2>", unsafe_allow_html=True)
    st.markdown(
        "Computed by direct interval intersection. "
        f"Shared + unique = total bout seconds per group: "
        f"**{grp1}**: {shared:,} + {only1:,} = {total1:,}s &nbsp;|&nbsp; "
        f"**{grp2}**: {shared:,} + {only2:,} = {total2:,}s"
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Shared (s)",           f"{shared:,}")
    m2.metric(f"Unique {grp1} (s)",   f"{only1:,}")
    m3.metric(f"Unique {grp2} (s)",   f"{only2:,}")
    m4.metric(f"{grp1} total (s)",    f"{total1:,}")
    m5.metric(f"{grp2} total (s)",    f"{total2:,}")

    def dur_bar(label, val_shared, val_unique, col_unique, total):
        fig = go.Figure(go.Bar(
            x=[val_shared, val_unique],
            y=[label, label],
            orientation="h",
            marker_color=["#1d9e75", col_unique],
            text=[
                f"{val_shared:,}s ({val_shared/total*100:.0f}%)" if total else "0s",
                f"{val_unique:,}s ({val_unique/total*100:.0f}%)" if total else "0s",
            ],
            textposition="inside",
        ))
        fig.update_layout(
            title=dict(text=f"{label} recording time", font=dict(size=18)),
            height=150,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
            xaxis=dict(title="seconds", title_font=dict(size=16),
                       tickfont=dict(size=15)),
            barmode="stack",
            yaxis=dict(showticklabels=False),
            font=dict(size=16),
        )
        return fig

    cb1, cb2 = st.columns(2)
    with cb1:
        st.plotly_chart(dur_bar(grp1, shared, only1, "#378add", total1),
                        use_container_width=True)
    with cb2:
        st.plotly_chart(dur_bar(grp2, shared, only2, "#ef9f27", total2),
                        use_container_width=True)

    # ── bout-level detail table ───────────────────────────────────────────
    st.markdown("<h2>Bout-level details</h2>", unsafe_allow_html=True)

    keep = ["_bout_id", "_group", "_start_raw", "_end_raw", "_dur_s",
            "_cat", "_match_ids"] + extra_cols
    keep = [c for c in keep if c in df1.columns]
    combined = pd.concat([df1[keep].copy(), df2[keep].copy()])
    combined = combined.sort_values("_start_raw").reset_index(drop=True)
    combined = combined.rename(columns={
        "_bout_id": "Bout ID", "_group": "Group",
        "_start_raw": "Start", "_end_raw": "End",
        "_dur_s": "Duration (s)", "_cat": "Category",
        "_match_ids": "Overlaps with"
    })

    st.dataframe(
        combined.style.map(color_cat, subset=["Category"]),
        use_container_width=True, hide_index=True, height=420
    )

    # ── per-animal breakdown ──────────────────────────────────────────────
    if use_animal and animal_col in df1.columns and per_animal_results:
        st.markdown("<h2>Per-animal breakdown</h2>", unsafe_allow_html=True)

        # summary table across all animals
        summary_rows = []
        for animal, r in per_animal_results.items():
            c1 = Counter(r["cats1"])
            c2 = Counter(r["cats2"])
            row = {"Animal": animal,
                   f"{grp1} bouts": r["n1"], f"{grp2} bouts": r["n2"],
                   "Shared (s)": r["shared"],
                   f"Unique {grp1} (s)": r["only1"],
                   f"Unique {grp2} (s)": r["only2"],
                   f"{grp1} total (s)": r["total1"],
                   f"{grp2} total (s)": r["total2"],
                   f"% shared {grp1}": f"{r['shared']/r['total1']*100:.0f}%" if r["total1"] else "—",
                   f"% shared {grp2}": f"{r['shared']/r['total2']*100:.0f}%" if r["total2"] else "—",
                   }
            for cat in CAT_ORDER_DYNAMIC:
                row[f"{grp1}: {cat}"] = c1.get(cat, 0)
                row[f"{grp2}: {cat}"] = c2.get(cat, 0)
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True, height=400)

        csv_summary = summary_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download per-animal summary CSV",
                           data=csv_summary,
                           file_name="per_animal_summary.csv",
                           mime="text/csv")

        # expandable detail per animal
        st.markdown("**Detailed view per animal**")
        for animal, r in per_animal_results.items():
            with st.expander(f"Animal {animal}  —  {grp1}: {r['n1']} bouts, {grp2}: {r['n2']} bouts"):
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    st.markdown(f"**{grp1}** vs {grp2}")
                    if r["n1"] > 0:
                        st.markdown(render_counts_table(r["cats1"], r["n1"], grp1, grp2),
                                    unsafe_allow_html=True)
                    else:
                        st.caption(f"No bouts for {grp1}")
                with dcol2:
                    st.markdown(f"**{grp2}** vs {grp1}")
                    if r["n2"] > 0:
                        st.markdown(render_counts_table(r["cats2"], r["n2"], grp2, grp1),
                                    unsafe_allow_html=True)
                    else:
                        st.caption(f"No bouts for {grp2}")

                dm1, dm2, dm3, dm4, dm5 = st.columns(5)
                dm1.metric("Shared (s)",          f"{r['shared']:,}")
                dm2.metric(f"Unique {grp1} (s)",  f"{r['only1']:,}")
                dm3.metric(f"Unique {grp2} (s)",  f"{r['only2']:,}")
                dm4.metric(f"{grp1} total (s)",   f"{r['total1']:,}")
                dm5.metric(f"{grp2} total (s)",   f"{r['total2']:,}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 – TIMELINE
# ═════════════════════════════════════════════════════════════════════════════
with tab_timeline:
    st.markdown("<h2>Bout timeline</h2>", unsafe_allow_html=True)

    all_starts = pd.concat([df1["_start_dt"], df2["_start_dt"]])
    all_ends   = pd.concat([df1["_end_dt"],   df2["_end_dt"]])
    t_min = all_starts.min()
    t_max = all_ends.max()

    # ── zoom controls: date picker + hour/minute number inputs ──────────
    with st.expander("🔎 Zoom & display options", expanded=True):
        zcol1, zcol2 = st.columns(2)
        with zcol1:
            st.markdown("**View from**")
            vd_start = st.date_input("Start date", value=t_min.date(),
                                      min_value=t_min.date(), max_value=t_max.date(),
                                      key="vd_start")
            tc1, tc2 = st.columns(2)
            with tc1:
                vh_start = st.number_input("Start hour", min_value=0, max_value=23,
                                            value=t_min.hour, step=1, key="vh_start")
            with tc2:
                vm_start = st.number_input("Start minute", min_value=0, max_value=59,
                                            value=t_min.minute, step=1, key="vm_start")
        with zcol2:
            st.markdown("**View until**")
            vd_end = st.date_input("End date", value=t_max.date(),
                                    min_value=t_min.date(), max_value=t_max.date(),
                                    key="vd_end")
            tc3, tc4 = st.columns(2)
            with tc3:
                vh_end = st.number_input("End hour", min_value=0, max_value=23,
                                          value=t_max.hour, step=1, key="vh_end")
            with tc4:
                vm_end = st.number_input("End minute", min_value=0, max_value=59,
                                          value=t_max.minute, step=1, key="vm_end")

        # colour-by options: always Category + Group, plus any extra cols
        colour_options = ["Category", "Group"] + extra_cols
        color_by = st.radio("Colour bouts by", colour_options, horizontal=True)

    from datetime import time as dtime
    view_start = pd.Timestamp(datetime.combine(vd_start, dtime(int(vh_start), int(vm_start))))
    view_end   = pd.Timestamp(datetime.combine(vd_end,   dtime(int(vh_end),   int(vm_end))))

    if view_start >= view_end:
        st.warning("'View from' must be earlier than 'View until'.")
        st.stop()

    d1v = df1[(df1["_end_dt"] > view_start) & (df1["_start_dt"] < view_end)].copy()
    d2v = df2[(df2["_end_dt"] > view_start) & (df2["_start_dt"] < view_end)].copy()

    # ── figure ────────────────────────────────────────────────────────────
    fig = go.Figure()

    LANE_Y     = {grp1: 1.0, grp2: 0.0}
    LANE_H     = 0.28
    GRP_COLORS = {grp1: "#378add", grp2: "#d85a30"}
    PALETTE    = ["#378add","#1d9e75","#d85a30","#7f77dd",
                  "#ef9f27","#e24b4a","#0f6e56","#ba7517"]

    def extra_col_color(val, col_name):
        vals = sorted(df_prep[col_name].dropna().astype(str).unique())
        idx  = vals.index(str(val)) % len(PALETTE) if str(val) in vals else 0
        return PALETTE[idx]

    legend_added = set()

    for obs_df, obs_name in [(d1v, grp1), (d2v, grp2)]:
        y_center = LANE_Y[obs_name]
        for _, row in obs_df.iterrows():
            s = max(row["_start_dt"], view_start)
            e = min(row["_end_dt"],   view_end)
            cat = row["_cat"]

            if color_by == "Category":
                fill = cat_color(cat)
                legend_key = cat
            elif color_by == "Group":
                fill = GRP_COLORS[obs_name]
                legend_key = obs_name
            else:
                # color by named extra column
                val = str(row.get(color_by, "")) if color_by in row.index else ""
                fill = extra_col_color(val, color_by) if color_by in df_prep.columns else "#888"
                legend_key = f"{color_by}={val}"

            show_legend = legend_key not in legend_added
            if show_legend:
                legend_added.add(legend_key)

            hover_parts = [
                f"<b>{obs_name}</b>",
                f"Start: {row['_start_raw']}",
                f"End:   {row['_end_raw']}",
                f"Duration: {int(row['_dur_s'])}s",
                f"Category: {cat}",
            ] + [f"{ec}: {row[ec]}" for ec in extra_cols if ec in row.index]

            # filled polygon = visible rect + hover
            fig.add_trace(go.Scatter(
                x=[s, s, e, e, s],
                y=[y_center - LANE_H/2, y_center + LANE_H/2,
                   y_center + LANE_H/2, y_center - LANE_H/2,
                   y_center - LANE_H/2],
                fill="toself",
                fillcolor=fill,
                opacity=0.85,
                mode="lines",
                line=dict(color="white", width=0.8),
                hovertemplate="<br>".join(hover_parts) + "<extra></extra>",
                name=legend_key,
                showlegend=show_legend,
                legendgroup=legend_key,
            ))

    # lane labels
    for obs_name in [grp1, grp2]:
        fig.add_annotation(
            x=view_start, y=LANE_Y[obs_name],
            text=f"<b>{obs_name}</b>",
            showarrow=False, xanchor="right",
            font=dict(size=17), xref="x", yref="y"
        )

    # separator between lanes
    fig.add_shape(type="line",
                  x0=view_start, x1=view_end, y0=0.5, y1=0.5,
                  line=dict(color="#cccccc", width=1, dash="dot"))

    span_s   = (view_end - view_start).total_seconds()
    tick_fmt = "%H:%M:%S" if span_s < 6 * 3600 else "%d %b %H:%M"

    fig.update_layout(
        height=320,
        margin=dict(l=120, r=20, t=20, b=100),
        xaxis=dict(range=[view_start, view_end], title="Time",
                   tickformat=tick_fmt, showgrid=True, gridcolor="#eeeeee",
                   tickfont=dict(size=16), title_font=dict(size=18)),
        yaxis=dict(range=[-0.22, 1.22], showticklabels=False,
                   showgrid=False, zeroline=False),
        plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.38, x=0,
                    font=dict(size=16), itemsizing="constant"),
        hoverlabel=dict(font_size=16),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Showing **{len(d1v)}** bouts for {grp1} and **{len(d2v)}** bouts for {grp2} "
        f"between {view_start.strftime('%d %b %Y %H:%M:%S')} "
        f"and {view_end.strftime('%d %b %Y %H:%M:%S')}"
    )
