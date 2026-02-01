# main_app.py

import streamlit as st
import pandas as pd
import geopy 

from geo_utils import geocode_canada
from map_picker import pick_location_map
from openmeteo_client import fetch_historical_weather

import plot_utils
from plot_utils import (row_line_violin, row_bar_violin, row_windrose_violin)



st.write("plot_utils loaded from:", plot_utils.__file__)
st.write("has violin_fig:", hasattr(plot_utils, "violin_fig"))


st.set_page_config(page_title="Canada Historic Weather", layout="wide")
st.title("Canada historic weather - Quick Analysis")

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "map_center_lat" not in st.session_state:
    st.session_state.map_center_lat = 56.1304
    st.session_state.map_center_lon = -106.3468
    st.session_state.map_zoom = 4

if "selected_lat" not in st.session_state:
    st.session_state.selected_lat = None
    st.session_state.selected_lon = None

if "weather_df" not in st.session_state:
    st.session_state.weather_df = None

# ---------------------------------------------------------
# SEARCH → PAN MAP
# ---------------------------------------------------------
st.subheader("Find a location and refine by clicking on the map")

search_place = st.text_input(
    "Search a place (city, address, campus, etc.)",
    value="Toronto, ON"
)

colA, colB = st.columns([1, 5])

with colA:
    if st.button("Locate on map"):
        with st.spinner("Searching location…"):
            geo = geocode_canada(search_place.strip())

        if geo is None:
            st.error("Location not found. Try adding province (e.g., 'Calgary, AB').")
        else:
            st.session_state.map_center_lat = geo["latitude"]
            st.session_state.map_center_lon = geo["longitude"]
            st.session_state.map_zoom = 13
            st.success(f"Centered near: {geo['display_name']}")

# ---------------------------------------------------------
# MAP CLICK → LAT/LON
# ---------------------------------------------------------
with colB:
    lat, lon = pick_location_map(
        center_lat=st.session_state.map_center_lat,
        center_lon=st.session_state.map_center_lon,
        zoom=st.session_state.map_zoom
    )

if lat is not None and lon is not None:
    st.session_state.selected_lat = lat
    st.session_state.selected_lon = lon

sel_lat = st.session_state.selected_lat
sel_lon = st.session_state.selected_lon

if sel_lat is not None and sel_lon is not None:
    st.success(f"Selected point: lat={sel_lat:.6f}, lon={sel_lon:.6f}")
else:
    st.info("Click on the map to select latitude & longitude.")

st.divider()

# ---------------------------------------------------------
# DATE / TIMEZONE CONTROLS
# ---------------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))

with c2:
    end_date = st.date_input("End date", value=pd.to_datetime("2020-01-07"))

with c3:
    timezone = st.selectbox(
        "Timezone",
        ["America/Toronto", "America/Vancouver", "America/Edmonton", "America/Winnipeg", "America/Halifax", "UTC"],
        index=0,
    )

hourly_variables = st.multiselect(
    "Hourly variables",
    [
        "temperature_2m",
        "relative_humidity_2m",
        "rain",
        "snowfall",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
        "wind_gusts_10m",
        "surface_pressure",
        "cloud_cover",
        "dew_point_2m",
    ],
    default=[
        "temperature_2m",
        "relative_humidity_2m",
        "rain",
        "snowfall",
        "wind_speed_10m",
        "wind_direction_10m",
    ],
)

# ---------------------------------------------------------
# FETCH DATA
# ---------------------------------------------------------
if st.button("Fetch data"):
    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    if sel_lat is None or sel_lon is None:
        st.error("Please select a point on the map first.")
        st.stop()

    with st.spinner("Downloading historical data…"):
        df = fetch_historical_weather(
            sel_lat,
            sel_lon,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            timezone,
            hourly_variables,
        )

    if df.empty:
        st.warning("No data returned for this selection.")
        st.session_state.weather_df = None
        st.stop()

    st.session_state.weather_df = df
    st.success("Data fetched successfully.")

# ---------------------------------------------------------
# DISPLAY + SEASON FILTER + ROW DASHBOARD
# ---------------------------------------------------------
df = st.session_state.weather_df

if df is None or df.empty:
    st.info("Click **Fetch data** to load and analyze weather data.")
    st.stop()

# Ensure datetime
df_plot = df.copy()
df_plot["time"] = pd.to_datetime(df_plot["time"])

# Season filter
WINTER_MONTHS = {11, 12, 1, 2, 3}       # Nov–Mar
SUMMER_MONTHS = {4, 5, 6, 7, 8, 9, 10}  # Apr–Oct

season_choice = st.selectbox("Season", ["All", "Winter (Nov–Mar)", "Summer (Apr–Oct)"], index=0)

if season_choice.startswith("Winter"):
    df_plot = df_plot[df_plot["time"].dt.month.isin(WINTER_MONTHS)]
elif season_choice.startswith("Summer"):
    df_plot = df_plot[df_plot["time"].dt.month.isin(SUMMER_MONTHS)]

if df_plot.empty:
    st.warning("No data available for this season selection.")
    st.stop()

st.caption(f"Season: {season_choice} | Rows: {len(df_plot):,} | Range: {start_date} → {end_date}")

st.subheader("Preview")
st.dataframe(df_plot, use_container_width=True)

st.subheader("Dashboard")

st.subheader("Dashboard")

# 1) Temperature: line + violin
fig = row_line_violin(df_plot, "temperature_2m", "Temperature (2m)", "°C")
if fig is None:
    st.info("temperature_2m not available (add it in hourly variables).")
else:
    st.pyplot(fig, use_container_width=True, clear_figure=True)

# 2) Relative Humidity: line + violin
fig = row_line_violin(df_plot, "relative_humidity_2m", "Relative Humidity (2m)", "%")
if fig is None:
    st.info("relative_humidity_2m not available (add it in hourly variables).")
else:
    st.pyplot(fig, use_container_width=True, clear_figure=True)

# 3) Rain: line + violin
fig = row_line_violin(df_plot, "rain", "Rain", "mm")
if fig is None:
    st.info("rain not available (add it in hourly variables).")
else:
    st.pyplot(fig, use_container_width=True, clear_figure=True)

# 4) Snowfall: bar + violin
fig = row_bar_violin(df_plot, "snowfall", "Snowfall", "cm")
if fig is None:
    st.info("snowfall not available (add it in hourly variables).")
else:
    st.pyplot(fig, use_container_width=True, clear_figure=True)

# 5) Wind: wind rose + wind speed violin
fig = row_windrose_violin(
    df_plot,
    speed_col="wind_speed_10m",
    dir_col="wind_direction_10m",
    n_sectors=16,
    violin_col="wind_speed_10m",
    violin_label="m/s",
)
if fig is None:
    st.info("Wind plots need wind_speed_10m and wind_direction_10m.")
else:
    st.pyplot(fig, use_container_width=True, clear_figure=True)
