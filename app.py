"""
Dynamic Parking Pricing — Streamlit Dashboard
------------------------------------------------
Drop this file into the root of your urban_parking_project
(same folder as main.py), next to utils/ and models/, then run:

    pip install streamlit plotly
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.haversine import compute_nearby_lots
from models.model1_baseline import baseline_linear_model
from models.model2_demand import demand_based_model
from models.model3_competitive import competitive_model
from utils.preprocess import load_and_preprocess

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Urban Parking — Dynamic Pricing Dashboard",
    page_icon="🅿️",
    layout="wide",
)

# ----------------------------------------------------------------------
# Data / model pipeline (cached so it doesn't re-run on every interaction)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and preprocessing dataset...")
def load_data(path: str) -> pd.DataFrame:
    return load_and_preprocess(path)


@st.cache_data(show_spinner="Running pricing models...")
def run_pipeline(
    df: pd.DataFrame,
    alpha_baseline: float,
    threshold_occupancy: float,
    lambda_scale_demand: float,
    alpha_demand: float,
    radius_km: float,
    mu: float,
    lambda_scale_comp: float,
):
    df1 = baseline_linear_model(
        df.copy(), alpha=alpha_baseline, threshold_occupancy=threshold_occupancy
    )
    df2 = demand_based_model(
        df1.copy(), lambda_scale=lambda_scale_demand, alpha=alpha_demand
    )

    df_meta = df2[["SystemCodeNumber", "Latitude", "Longitude"]].drop_duplicates()
    nearby_map = compute_nearby_lots(df_meta, radius_km=radius_km)

    df3 = competitive_model(
        df2.copy(), nearby_map, mu=mu, lambda_scale=lambda_scale_comp
    )
    return df1, df2, df3, nearby_map


# ----------------------------------------------------------------------
# Sidebar — data + model controls
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

data_path = st.sidebar.text_input("Dataset path", value="dataset.csv")

st.sidebar.subheader("Model parameters")
alpha_baseline = st.sidebar.slider("Baseline: alpha", 0.1, 3.0, 1.1, 0.1)
threshold_occupancy = st.sidebar.slider("Baseline: occupancy threshold", 0.0, 1.0, 0.6, 0.05)
lambda_scale_demand = st.sidebar.slider("Demand: lambda_scale", 0.1, 2.0, 0.8, 0.1)
alpha_demand = st.sidebar.slider("Demand: alpha", 0.5, 10.0, 3.0, 0.5)
radius_km = st.sidebar.slider("Competitive: nearby radius (km)", 0.1, 5.0, 1.0, 0.1)
mu = st.sidebar.slider("Competitive: mu", 0.0, 1.0, 0.05, 0.01)
lambda_scale_comp = st.sidebar.slider("Competitive: lambda_scale", 0.0, 1.0, 0.2, 0.01)

try:
    df_raw = load_data(data_path)
except Exception as e:
    st.error(f"Couldn't load '{data_path}': {e}")
    st.stop()

df1, df2, df3, nearby_map = run_pipeline(
    df_raw,
    alpha_baseline,
    threshold_occupancy,
    lambda_scale_demand,
    alpha_demand,
    radius_km,
    mu,
    lambda_scale_comp,
)

all_lots = sorted(df3["SystemCodeNumber"].unique())

st.sidebar.subheader("Lot selection")
selected_lot = st.sidebar.selectbox("Focus lot", all_lots)

# ----------------------------------------------------------------------
# Sidebar — feature filters (vehicle type, traffic, special day, queue)
# ----------------------------------------------------------------------
st.sidebar.subheader("Filters")

vehicle_options = sorted(df3["VehicleType"].dropna().unique()) if "VehicleType" in df3.columns else []
traffic_options = sorted(df3["TrafficConditionNearby"].dropna().unique()) if "TrafficConditionNearby" in df3.columns else []

selected_vehicle_types = (
    st.sidebar.multiselect("Vehicle type", vehicle_options, default=vehicle_options)
    if vehicle_options else []
)
selected_traffic = (
    st.sidebar.multiselect("Traffic condition nearby", traffic_options, default=traffic_options)
    if traffic_options else []
)
special_day_filter = (
    st.sidebar.selectbox("Special day", ["All", "Special day only", "Regular day only"])
    if "IsSpecialDay" in df3.columns else "All"
)
queue_range = (
    st.sidebar.slider(
        "Queue length range",
        int(df3["QueueLength"].min()), int(df3["QueueLength"].max()),
        (int(df3["QueueLength"].min()), int(df3["QueueLength"].max())),
    )
    if "QueueLength" in df3.columns else None
)

# Apply filters to build the working dataframe used across all tabs
df3_filtered = df3.copy()
if vehicle_options:
    df3_filtered = df3_filtered[df3_filtered["VehicleType"].isin(selected_vehicle_types)]
if traffic_options:
    df3_filtered = df3_filtered[df3_filtered["TrafficConditionNearby"].isin(selected_traffic)]
if "IsSpecialDay" in df3.columns and special_day_filter != "All":
    want = 1 if special_day_filter == "Special day only" else 0
    df3_filtered = df3_filtered[df3_filtered["IsSpecialDay"] == want]
if queue_range is not None:
    df3_filtered = df3_filtered[
        df3_filtered["QueueLength"].between(queue_range[0], queue_range[1])
    ]

if df3_filtered.empty:
    st.warning("No rows match the current filters — showing unfiltered data instead.")
    df3_filtered = df3.copy()

# ----------------------------------------------------------------------
# Header + KPIs
# ----------------------------------------------------------------------
st.title("🅿️ Urban Parking — Dynamic Pricing Dashboard")
st.caption("Baseline → Demand-based → Competitive-adjusted pricing models")

latest = df3_filtered.sort_values("Timestamp").groupby("SystemCodeNumber").tail(1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total lots", len(all_lots))
col2.metric("Avg current price", f"${latest['AdjustedPrice'].mean():.2f}")
col3.metric("Max current price", f"${latest['AdjustedPrice'].max():.2f}")
col4.metric("Min current price", f"${latest['AdjustedPrice'].min():.2f}")

st.divider()

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Lot Pricing", "🌍 All Lots Overview", "📊 Raw Data",
     "🔍 Model Comparison", "🧩 Feature Insights"]
)

# --- Tab 1: single-lot deep dive -----------------------------------
with tab1:
    st.subheader(f"Pricing detail — {selected_lot}")

    df_sample = df3_filtered[df3_filtered["SystemCodeNumber"] == selected_lot].sort_values("Timestamp")

    if df_sample.empty:
        st.info("No data for this lot under the current filters.")
        st.stop()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sample["Timestamp"], y=df_sample["Price"],
        name="Baseline/Demand Price", line=dict(color="orange")
    ))
    fig.add_trace(go.Scatter(
        x=df_sample["Timestamp"], y=df_sample["AdjustedPrice"],
        name="Competitive Adjusted Price", line=dict(color="purple")
    ))
    fig.update_layout(
        title=f"Price Over Time — {selected_lot}",
        xaxis_title="Time", yaxis_title="Price ($)",
        legend=dict(orientation="h", y=1.1),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_sample["Timestamp"], y=df_sample["Price"],
        name="Price ($)", yaxis="y1", line=dict(color="purple")
    ))
    fig2.add_trace(go.Scatter(
        x=df_sample["Timestamp"], y=df_sample["SmoothedDemand"],
        name="Smoothed Demand", yaxis="y2", line=dict(color="steelblue")
    ))
    fig2.update_layout(
    title=f"Price vs Demand — {selected_lot}",

    xaxis=dict(
        title=dict(
            text="Demand",
            font=dict(size=16)
        )
    ),

    yaxis=dict(
        title=dict(
            text="Price",
            font=dict(size=16)
        )
    ),

    height=420
)
    st.plotly_chart(fig2, use_container_width=True)

# --- Tab 2: all-lots overview ---------------------------------------
with tab2:
    st.subheader("All lots — current adjusted price")

    if {"Latitude", "Longitude"}.issubset(latest.columns):
        fig_map = px.scatter_map(
            latest,
            lat="Latitude", lon="Longitude",
            color="AdjustedPrice", size="AdjustedPrice",
            hover_name="SystemCodeNumber",
            color_continuous_scale="Turbo",
            zoom=10, height=520,
            mapbox_style="open-street-map",
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No Latitude/Longitude columns found — showing bar chart instead.")

    fig_bar = px.bar(
        latest.sort_values("AdjustedPrice", ascending=False),
        x="SystemCodeNumber", y="AdjustedPrice",
        title="Current Adjusted Price by Lot",
        color="AdjustedPrice", color_continuous_scale="Turbo",
    )
    fig_bar.update_layout(height=420, xaxis_title="Lot", yaxis_title="Price ($)")
    st.plotly_chart(fig_bar, use_container_width=True)

# --- Tab 3: raw data --------------------------------------------------
with tab3:
    st.subheader("Raw / processed data")
    st.caption("Reflects the sidebar filters (vehicle type, traffic, special day, queue length).")
    lot_filter = st.multiselect("Filter by lot", all_lots, default=[selected_lot])
    df_view = df3_filtered[df3_filtered["SystemCodeNumber"].isin(lot_filter)] if lot_filter else df3_filtered
    st.dataframe(df_view, use_container_width=True, height=500)
    st.download_button(
        "Download filtered data as CSV",
        df_view.to_csv(index=False).encode("utf-8"),
        file_name="filtered_pricing_data.csv",
        mime="text/csv",
    )

# --- Tab 4: model comparison ------------------------------------------
with tab4:
    st.subheader(f"Model 2 (Demand-Based) vs Model 3 (Competitive) — {selected_lot}")
    df_sample = df3_filtered[df3_filtered["SystemCodeNumber"] == selected_lot].sort_values("Timestamp")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_sample["Timestamp"], y=df_sample["Price"],
        name="Model 2: Demand-Based", line=dict(color="orange")
    ))
    fig3.add_trace(go.Scatter(
        x=df_sample["Timestamp"], y=df_sample["AdjustedPrice"],
        name="Model 3: Competitive Adjusted", line=dict(color="purple")
    ))
    fig3.update_layout(
        title=f"Pricing Comparison — {selected_lot}",
        xaxis_title="Time", yaxis_title="Price ($)",
        legend=dict(orientation="h", y=1.1),
        height=450,
    )
    st.plotly_chart(fig3, use_container_width=True)

    if selected_lot in nearby_map:
        st.markdown(f"**Nearby competing lots (within {radius_km} km):**")
        st.write(nearby_map[selected_lot])
    else:
        st.info("No nearby lots found within the selected radius.")

# --- Tab 5: feature insights -------------------------------------------
with tab5:
    st.subheader("How pricing relates to vehicle type, traffic, special days & queues")
    st.caption("Uses all lots under the current sidebar filters.")

    fcol1, fcol2 = st.columns(2)

    with fcol1:
        if "VehicleType" in df3_filtered.columns:
            fig_vt = px.box(
                df3_filtered, x="VehicleType", y="AdjustedPrice", color="VehicleType",
                title="Adjusted Price by Vehicle Type",
            )
            fig_vt.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_vt, use_container_width=True)
        else:
            st.info("No VehicleType column found.")

    with fcol2:
        if "TrafficConditionNearby" in df3_filtered.columns:
            fig_tc = px.box(
                df3_filtered, x="TrafficConditionNearby", y="AdjustedPrice",
                color="TrafficConditionNearby",
                title="Adjusted Price by Traffic Condition Nearby",
            )
            fig_tc.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_tc, use_container_width=True)
        else:
            st.info("No TrafficConditionNearby column found.")

    fcol3, fcol4 = st.columns(2)

    with fcol3:
        if "IsSpecialDay" in df3_filtered.columns:
            df_sd = df3_filtered.copy()
            df_sd["Special Day"] = df_sd["IsSpecialDay"].map({1: "Special day", 0: "Regular day"})
            fig_sd = px.box(
                df_sd, x="Special Day", y="AdjustedPrice", color="Special Day",
                title="Adjusted Price: Special Day vs Regular Day",
            )
            fig_sd.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_sd, use_container_width=True)
        else:
            st.info("No IsSpecialDay column found.")

    with fcol4:
        if "QueueLength" in df3_filtered.columns:
            fig_ql = px.scatter(
                df3_filtered, x="QueueLength", y="AdjustedPrice",
                color="TrafficConditionNearby" if "TrafficConditionNearby" in df3_filtered.columns else None,
                title="Queue Length vs Adjusted Price",
                opacity=0.5,
            )
            fig_ql.update_layout(height=380)
            st.plotly_chart(fig_ql, use_container_width=True)
        else:
            st.info("No QueueLength column found.")

    if "Occupancy" in df3_filtered.columns and "Capacity" in df3_filtered.columns:
        df3_filtered["OccupancyRate"] = df3_filtered["Occupancy"] / df3_filtered["Capacity"]
        fig_occ = px.scatter(
            df3_filtered, x="OccupancyRate", y="AdjustedPrice",
            color="VehicleType" if "VehicleType" in df3_filtered.columns else None,
            title="Occupancy Rate vs Adjusted Price",
            opacity=0.5,
        )
        fig_occ.update_layout(height=400)
        st.plotly_chart(fig_occ, use_container_width=True)

st.divider()
st.caption("Built for the Dynamic Pricing for Urban Parking Lots project.")
