"""Streamlit dashboard for the hierarchical demand forecasting platform.

Reads the compact extract in data/dashboard (versioned in git), so it runs on
Streamlit Community Cloud without the modelling stack:

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data" / "dashboard"
REPO_URL = "https://github.com/advaithvemurisai/demand-forecast-platform"

# Reference palette, fixed slot order (validated for CVD separation on the light surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQUENTIAL = [[0, "#eaf2fc"], [1, "#1d5fae"]]
LEVELS = ["total", "state", "store", "category", "department", "item"]
LEVEL_LABELS = {"total": "Total", "state": "State", "store": "Store", "category": "Category × store", "department": "Department × store", "item": "Item × store"}
METHOD_LABELS = {
    "base": "SARIMA base (incoherent)", "bottom_up": "Bottom-up", "top_down": "Top-down",
    "mint_diagonal": "MinT, in-sample weights", "mint_oos": "MinT, out-of-sample weights",
}
MODEL_LABELS = {"naive": "Naive", "seasonal_naive": "Seasonal naive", "lgbm_local": "LightGBM local", "lgbm_global": "LightGBM global", "sarima": "SARIMA", "prophet": "Prophet"}

st.set_page_config(page_title="Demand Forecasting Platform", page_icon="📦", layout="wide")


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{name}.parquet")


@st.cache_data
def summary() -> dict:
    return json.loads((DATA / "run_summary.json").read_text())


def style(fig: go.Figure, height: int = 380, y_title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=48, b=8), plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        font=dict(color=MUTED, size=13), hoverlabel=dict(bgcolor="white", font_color=INK),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, title=None),
    )
    fig.update_xaxes(showgrid=False, linecolor=MUTED, ticks="outside")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=y_title)
    return fig


def rgba(hex_colour: str, alpha: float) -> str:
    red, green, blue = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def node_label(node_id: str) -> str:
    level, _, rest = node_id.partition(":")
    return "All CA stores" if rest == "total" else rest.replace("_id=", ": ").replace("|", " · ")


run = summary()
served = run["served_method"]
# Colour follows the method everywhere: the served method takes slot 1, the rest keep their fixed order.
METHOD_COLORS = dict(zip([served, *(m for m in METHOD_LABELS if m != served)], SERIES))
recon = load("reconciliation_metrics")
holdout = recon[recon["fold"] == "holdout"]

# --------------------------------------------------------------------------- header
st.title("Hierarchical demand forecasting and allocation")
st.markdown(
    "**The problem.** A retailer has to decide how much of each product to stock and which stores get it. "
    "Forecasts made separately for items, stores and the whole region disagree, and when the warehouse is short "
    "someone has to choose which stores go without. Too little stock loses sales; too much ties up cash. "
    "This platform turns Walmart store sales into forecasts that add up at every level, a confidence range for each, "
    "and a weekly allocation that maximises revenue when supply is short."
)
st.caption(
    f"M5 (Walmart) California: {run['n_series']:,} item-store series across 4 stores, forecast at 6 hierarchy levels. "
    f"Three rolling backtests plus a holdout on the M5 validation window ({run['origins']['holdout']} → {run['origins']['production']}). "
    f"[Code and methodology]({REPO_URL})"
)

alloc_bt = load("allocation_backtest")
revenue = alloc_bt.groupby("policy")["revenue_fulfilled"].sum()
weeks = alloc_bt.pivot_table(index=["fold", "week"], columns="policy", values="revenue_fulfilled")
weeks_won = int((weeks["lp_scenario"] > weeks["pro_rata"]).sum())
coverage = load("interval_coverage")
item_cov = coverage[(coverage["level"] == "item") & (coverage["nominal"] == 0.95)]["coverage"].mean()
served_category = holdout[(holdout["method"] == served) & (holdout["level"] == "category")]["wmape"].iloc[0]
bottom_up_category = holdout[(holdout["method"] == "bottom_up") & (holdout["level"] == "category")]["wmape"].iloc[0]
stockouts = alloc_bt.groupby("policy")["stockout_nodes"].sum()
models = load("model_metrics")
item_models = models[(models["fold"] == "holdout") & (models["level"] == "item")].set_index("model")["wmape"]
drift = load("drift")

kpis = st.columns(4)
kpis[0].metric("Holdout WRMSSE (served method)", f"{run['wrmsse_holdout'][served]:.3f}",
               help="Overall forecast error across all 6 levels, from single items up to the state; lower is better. "
                    "Below 1 means better than simply repeating last period's sales. Official M5 metric, but over 6 CA levels, "
                    "so not comparable to the 12-level M5 leaderboard.")
kpis[1].metric("Category WMAPE, holdout", f"{served_category:.1%}", f"{(served_category - bottom_up_category) * 100:+.1f} pts vs bottom-up", delta_color="inverse",
               help="Average % miss on category sales per store, the level buyers plan at. Bottom-up (just adding up the item "
                    "forecasts) is the naive way to get numbers that add up; reconciliation beats it.")
kpis[2].metric("95% interval coverage, items", f"{item_cov:.1%}", help="How often actual sales landed inside the 95% range. Close to 95% means the ranges can be trusted to set "
                    "safety stock. Out-of-sample: each fold is calibrated only on earlier folds.")
kpis[3].metric("LP revenue vs pro-rata", f"{revenue['lp_scenario'] / revenue['pro_rata'] - 1:+.2%}", f"{weeks_won} of {len(weeks)} weeks won",
               help="When the warehouse holds only 90% of forecast demand, the optimiser earns this much more revenue than splitting "
                    "stock in proportion to each store's forecast. Scored on what actually sold, across 28 store × department nodes.")

with st.expander("What each step solves for the business", expanded=True):
    st.markdown(f"""
| Step | Business question | Result |
|---|---|---|
| **Forecast** (LightGBM) | How much will each item sell in each store over the next 28 days? | {1 - item_models['lgbm_global'] / item_models['seasonal_naive']:.0%} lower item error than repeating last week's sales |
| **Reconcile** (MinT) | Do the item, store and state plans agree, so every team works from the same numbers? | Forecasts add up at every level; category error {(bottom_up_category - served_category) / bottom_up_category:.0%} lower than adding up item forecasts |
| **Intervals** (conformal) | How sure are we, and how bad could it get? | 95% ranges contain {item_cov:.1%} of actual item sales |
| **Safety stock** | How much buffer does each item need to be in stock 95% of the time? | An order-up-to level for all {run['n_series']:,} items |
| **Allocate** (scenario LP) | When stock is short, which stores and departments get it? | {revenue['lp_scenario'] / revenue['pro_rata'] - 1:+.2%} revenue and {1 - stockouts['lp_scenario'] / stockouts['pro_rata']:.0%} fewer stockouts than a proportional split |
| **Monitor** (drift) | Has demand shifted enough that the model needs retraining? | {'Retraining flagged' if drift['retrain'].iloc[0] else 'Demand is stable; no retraining needed'} |
""")

tabs = st.tabs(["Forecast explorer", "Accuracy", "Intervals", "Allocation", "Drift", "Method"])

# --------------------------------------------------------------------------- forecast explorer
with tabs[0]:
    st.caption("**Business question:** how much will sell, anywhere in the hierarchy, and how wide is the range of outcomes?")
    production = load("production_forecast")
    backtest = load("backtest_forecasts")
    left, right = st.columns([1, 3])
    with left:
        level = st.selectbox("Level", LEVELS, format_func=LEVEL_LABELS.get, index=2)
        nodes = production.loc[production["level"] == level, "series_id"].drop_duplicates().sort_values()
        if level == "item":
            store = st.selectbox("Store", sorted(nodes.str.extract(r"store_id=(.+)$")[0].unique()))
            nodes = nodes[nodes.str.endswith(f"store_id={store}")]
        node = st.selectbox("Node", nodes.tolist(), format_func=node_label)
        methods = []
        if level != "item":
            methods = st.multiselect(
                "Backtest methods", list(METHOD_LABELS), default=[served, "base"], format_func=METHOD_LABELS.get,
                help="Forecasts made at each fold origin, compared with what actually sold.",
            )
        show_80 = st.toggle("Show 80% interval", value=True)

    with right:
        fig = go.Figure()
        future = production[production["series_id"] == node]
        if level != "item":
            history = backtest[backtest["series_id"] == node].sort_values("date")
            actual = history[history["method"] == "base"]
            fig.add_scatter(x=actual["date"], y=actual["actual"], name="Actual", line=dict(color=INK, width=2),
                            hovertemplate="%{y:,.0f}")
            for method, colour in METHOD_COLORS.items():
                if method in methods:
                    rows = history[history["method"] == method]
                    for fold, part in rows.groupby("fold"):
                        fig.add_scatter(x=part["date"], y=part["forecast"], name=METHOD_LABELS[method], legendgroup=method,
                                        showlegend=fold == rows["fold"].iloc[0], line=dict(color=colour, width=1.6),
                                        hovertemplate="%{y:,.0f}")
        fig.add_scatter(x=future["date"], y=future["upper_95"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=future["date"], y=future["lower_95"], fill="tonexty", fillcolor=rgba(METHOD_COLORS[served], 0.14),
                        line=dict(width=0), name="95% interval", hoverinfo="skip")
        if show_80:
            fig.add_scatter(x=future["date"], y=future["upper_80"], line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=future["date"], y=future["lower_80"], fill="tonexty", fillcolor=rgba(METHOD_COLORS[served], 0.30),
                            line=dict(width=0), name="80% interval", hoverinfo="skip")
        fig.add_scatter(x=future["date"], y=future["forecast"], name=f"Production forecast ({METHOD_LABELS[served]})",
                        line=dict(color=METHOD_COLORS[served], width=2.4),
                        customdata=future[["lower_95", "upper_95"]],
                        hovertemplate="%{y:,.1f}<br>95%: %{customdata[0]:,.1f} – %{customdata[1]:,.1f}")
        fig.add_vline(x=pd.Timestamp(run["origins"]["production"]).timestamp() * 1000, line=dict(color=MUTED, dash="dot", width=1))
        fig.update_layout(hovermode="x unified", title=dict(text=node_label(node), font=dict(color=INK)))
        st.plotly_chart(style(fig, 440, "Units per day"), width="stretch")
        if level == "item":
            st.caption("Item-level backtest lines are omitted to keep the hosted extract small; the production forecast and intervals are shown for every item.")
        else:
            st.caption("Left of the dotted line: forecasts made at each fold origin vs actual sales. Right: the served 28-day production forecast.")

# --------------------------------------------------------------------------- accuracy
with tabs[1]:
    st.caption("**Business question:** which way of making the forecasts add up is the most accurate, and how good is the underlying model?")
    c1, c2 = st.columns(2)
    fold = c1.radio("Evaluation window", ["holdout", "backtest mean"], horizontal=True,
                    help="The holdout was never used for any modelling decision.")
    metric = c2.radio("Metric", ["wmape", "wrmsse"], horizontal=True, format_func=str.upper)
    frame = holdout if fold == "holdout" else recon[recon["fold"].str.startswith("backtest")]
    grid = frame.pivot_table(index="level", columns="method", values=metric, aggfunc="mean").loc[LEVELS, list(METHOD_LABELS)]
    # Colour each cell by its gap to the best method in the same row; item-level errors
    # are ~10x the aggregate ones and would otherwise wash out every other row.
    relative = grid.div(grid.min(axis=1), axis=0) - 1
    heat = go.Figure(go.Heatmap(
        z=relative.to_numpy(), zmin=0, customdata=grid.to_numpy(), x=[METHOD_LABELS[m] for m in grid.columns], y=[LEVEL_LABELS[l] for l in grid.index],
        colorscale=SEQUENTIAL, text=grid.map(lambda v: f"{v:.3f}").to_numpy(), texttemplate="%{text}",
        hovertemplate="%{y}<br>%{x}<br>" + metric.upper() + " %{customdata:.4f}<br>%{z:+.0%} vs best in row<extra></extra>", showscale=False, xgap=2, ygap=2,
    ))
    heat.update_yaxes(autorange="reversed", gridcolor=SURFACE)
    heat.update_layout(title=dict(text=f"Reconciliation {metric.upper()} by level (lower is better; darker = further behind the best in its row)", font=dict(color=INK)))
    st.plotly_chart(style(heat, 360), width="stretch")
    st.markdown(
        f"**Served:** {METHOD_LABELS[served]}, chosen by the lowest backtest WRMSSE among coherent methods. "
        "The SARIMA base is the most accurate at the aggregate levels but *incoherent*: stores don't add up to the state, "
        "so a planner can't use it. MinT makes the numbers add up by moving the least reliable forecasts the most; "
        "see the Method tab."
    )

    model_level = st.selectbox("Model comparison level", LEVELS, format_func=LEVEL_LABELS.get, index=5)
    frame = models if fold == "backtest mean" else models[models["fold"] == "holdout"]
    if fold == "backtest mean":
        frame = frame[frame["fold"].str.startswith("backtest")]
    bars = frame[frame["level"] == model_level].groupby("model")[metric].mean().sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=bars.values, y=[MODEL_LABELS.get(m, m) for m in bars.index], orientation="h", marker_color=SERIES[0],
        text=[f"{v:.3f}" for v in bars.values], textposition="outside", hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(title=dict(text=f"Base models at {LEVEL_LABELS[model_level].lower()} level", font=dict(color=INK)), bargap=0.45)
    fig.update_xaxes(title=metric.upper(), gridcolor=GRID, showgrid=True)
    st.plotly_chart(style(fig, 320), width="stretch")

# --------------------------------------------------------------------------- intervals
with tabs[2]:
    st.caption("**Business question:** can the forecast ranges be trusted to set safety stock? If a 95% range misses more than 5% of the time, stores run out more often than planned.")
    mean_cov = coverage.groupby(["level", "nominal"], as_index=False)[["coverage", "mean_width"]].mean()
    fig = go.Figure()
    for colour, nominal in zip(SERIES, (0.8, 0.95)):
        rows = mean_cov[mean_cov["nominal"] == nominal].set_index("level").loc[LEVELS].reset_index()
        fig.add_scatter(x=[LEVEL_LABELS[l] for l in rows["level"]], y=rows["coverage"], mode="markers", name=f"{nominal:.0%} interval",
                        marker=dict(color=colour, size=13, line=dict(color=SURFACE, width=2)),
                        hovertemplate="%{x}<br>coverage %{y:.1%}<extra></extra>")
        fig.add_hline(y=nominal, line=dict(color=colour, dash="dot", width=1))
    fig.update_yaxes(tickformat=".0%", range=[0.6, 1.02])
    fig.update_layout(title=dict(text="Empirical coverage vs nominal (dotted lines)", font=dict(color=INK)))
    st.plotly_chart(style(fig, 380, "Share of actuals inside the interval"), width="stretch")
    st.markdown(
        "Split-conformal intervals with signed, scale-normalised scores. Each fold is calibrated only on **earlier** folds, "
        "so this coverage is out-of-sample. Items are well calibrated; aggregate levels run a few points narrow because they "
        "calibrate on few nodes × 28 days."
    )
    table = mean_cov.pivot_table(index="level", columns="nominal", values=["coverage", "mean_width"]).loc[LEVELS]
    table.columns = [f"{'Coverage' if name == 'coverage' else 'Mean width (units)'} {nominal:.0%}" for name, nominal in table.columns]
    table.index = [LEVEL_LABELS[l] for l in table.index]
    st.dataframe(table.style.format({c: "{:.1%}" for c in table.columns if c.startswith("Coverage")} | {c: "{:,.1f}" for c in table.columns if c.startswith("Mean width")}), width="stretch")

# --------------------------------------------------------------------------- allocation
with tabs[3]:
    st.caption("**Business question:** when the warehouse can't cover every store's demand, who gets the stock?")
    allocation = load("allocation")
    st.markdown(
        f"**Week of {pd.Timestamp(allocation['week_start'].iloc[0]):%d %b %Y}.** The DC holds {allocation['supply'].iloc[0]:,.0f} units "
        "(90% of forecast demand). The scenario LP maximises expected fulfilled revenue over conformal demand scenarios; "
        "pro-rata splits supply in proportion to the point forecast."
    )
    stores = sorted(allocation["store_id"].unique())
    store = st.segmented_control("Store", stores, default=stores[0], required=True)
    rows = allocation[allocation["store_id"] == store].sort_values("dept_id")
    fig = go.Figure()
    fig.add_bar(x=rows["dept_id"], y=rows["allocated_quantity"], name="Scenario LP", marker_color=SERIES[0],
                customdata=rows[["stockout_risk", "unit_value"]], hovertemplate="LP: %{y:,.0f} units<br>stockout risk %{customdata[0]:.0%}<br>$%{customdata[1]:.2f}/unit<extra></extra>")
    fig.add_bar(x=rows["dept_id"], y=rows["pro_rata_quantity"], name="Pro-rata", marker_color=SERIES[1],
                hovertemplate="Pro-rata: %{y:,.0f} units<extra></extra>")
    fig.add_scatter(x=rows["dept_id"], y=rows["forecast"], mode="markers", name="Forecast demand",
                    marker=dict(color=INK, size=10, symbol="line-ew-open", line=dict(width=2)), hovertemplate="forecast %{y:,.0f}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08, title=dict(text=f"Allocation by department, {store}", font=dict(color=INK)))
    st.plotly_chart(style(fig, 380, "Units for the week"), width="stretch")
    st.caption(
        "The LP is revenue-weighted with no minimum-fill floor, so under a 10% shortfall it can give a low-value department "
        "(here HOBBIES_2) nothing. `allocate_inventory(..., min_fill=...)` adds a service floor when that is unacceptable."
    )

    weekly = alloc_bt.pivot_table(index=["fold", "week"], columns="policy", values="revenue_fulfilled").reset_index()
    weekly["uplift"] = weekly["lp_scenario"] / weekly["pro_rata"] - 1
    weekly["label"] = weekly["fold"].str.replace("backtest_", "backtest ") + " · wk " + weekly["week"].astype(str)
    fig = go.Figure(go.Bar(x=weekly["label"], y=weekly["uplift"], marker_color=SERIES[0],
                           hovertemplate="%{x}<br>LP revenue %{y:+.2%} vs pro-rata<extra></extra>"))
    fig.update_yaxes(tickformat="+.1%")
    fig.update_layout(title=dict(text="Backtest: LP revenue fulfilled vs pro-rata, per week (realised demand)", font=dict(color=INK)))
    st.plotly_chart(style(fig, 300), width="stretch")

    with st.expander("Item-level safety stock (week 1, 95% cycle service level)"):
        safety = load("safety_stock")
        query = st.text_input("Filter items", placeholder="e.g. FOODS_3_090")
        view = safety[safety["series_id"].str.contains(query, case=False, regex=False)] if query else safety
        st.dataframe(
            view[["item_id", "store_id", "forecast", "safety_stock", "order_up_to", "stockout_risk"]].sort_values("forecast", ascending=False).head(500),
            width="stretch", hide_index=True,
            column_config={"forecast": st.column_config.NumberColumn("Week forecast", format="%.1f"),
                           "safety_stock": st.column_config.NumberColumn(format="%.1f"),
                           "order_up_to": st.column_config.NumberColumn("Order-up-to", format="%.1f"),
                           "stockout_risk": st.column_config.NumberColumn("Stockout risk", format="percent")},
        )

# --------------------------------------------------------------------------- drift
with tabs[4]:
    st.caption("**Business question:** is the model still accurate, or has customer demand shifted enough to retrain it?")
    view = drift.assign(
        status=drift["drift"].map({True: "⚠️ Drift", False: "✅ Stable"}),
        store=drift["node_id"].str.replace("store:store_id=", ""),
    )[["store", "status", "demand_psi", "residual_psi", "reference_mean_daily", "current_mean_daily"]]
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "store": st.column_config.TextColumn("Store"),
        "status": st.column_config.TextColumn("Status"),
        "demand_psi": st.column_config.NumberColumn("Demand-mix PSI", format="%.3f"),
        "residual_psi": st.column_config.NumberColumn("Residual PSI", format="%.3f"),
        "reference_mean_daily": st.column_config.NumberColumn("Prior-year daily units", format="%,.0f"),
        "current_mean_daily": st.column_config.NumberColumn("Last 28 days daily units", format="%,.0f"),
    })
    flag = drift["retrain"].iloc[0]
    (st.warning if flag else st.success)(f"Retraining flag: **{'on' if flag else 'off'}** ({drift['retrain_reasons'].iloc[0]}). "
                                         "Alerts fire at PSI ≥ 0.2 or a >10% rise in store-level WMAPE.")

# --------------------------------------------------------------------------- method
with tabs[5]:
    st.markdown(f"""
**The business problem.** A retailer plans stock at several levels at once: buyers plan categories, store managers plan their
store, and the supply chain plans the region. If each level is forecast separately, the numbers disagree: the four stores
might add up to 16,500 units while the state forecast says 17,200, and nobody knows which plan to order against. On top of
that, supply is often short, so someone has to decide which stores go without. Getting it wrong means empty shelves in one
store and excess stock in another.

**What MinT reconciliation does, in plain terms.** It takes the forecasts from every level and finds the closest set of numbers
that add up exactly. Forecasts that have been reliable barely move; noisy ones absorb most of the correction. So in the example
above, if the state forecast has a good track record and one store's hasn't, most of the 700-unit gap is closed by adjusting that
store. The name comes from the maths: of all the ways to make the numbers add up, it picks the one with the smallest total
forecast-error variance (the *trace* of the error covariance matrix).

**What the allocation adds.** A forecast says what *would* sell; the allocation decides what each store *gets*. The LP uses the
forecast ranges to weigh the chance each unit actually sells against its price, so scarce stock goes where it earns the most.

**Pipeline.** M5 raw → silver Parquet → leakage-safe features (every lag ≥ the 28-day horizon) → naive floors, LightGBM global/local,
SARIMA and Prophet → bottom-up, top-down and MinT reconciliation → split-conformal intervals and safety stock → scenario LP allocation.
Runs are tracked in MLflow; forecasts are served by FastAPI and exported for Tableau.

**Evaluation.** Three 28-day rolling-origin backtests, then a holdout on the M5 validation window scored against real actuals.
The bottom-level model (**{MODEL_LABELS[run['selected_bottom_model']]}**) and the served reconciliation method
(**{METHOD_LABELS[served]}**) were chosen on backtest folds only.

**MinT at SKU scale.** A diagonal-covariance MinT is solved with the Woodbury identity, which inverts a 46×46 matrix instead of
{run['n_series']:,}×{run['n_series']:,}. The out-of-sample variant weights each node by its base model's error in the previous 28-day window.

**Honest caveats.** The un-reconciled SARIMA aggregates beat every coherent method at the aggregate levels, because the LightGBM bottom
level under-forecasts by about 7%. WRMSSE covers 6 California levels and is not comparable to the M5 leaderboard.

Full write-up and code: [{REPO_URL}]({REPO_URL})
""")
