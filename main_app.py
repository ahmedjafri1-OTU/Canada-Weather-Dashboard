# main_app.py
import streamlit as st
import pandas as pd

# MUST be first Streamlit call
st.set_page_config(page_title="Canada Weather Dashboard", layout="wide")

# ---------------------------------------------------------
# SAFE IMPORTS
# ---------------------------------------------------------
try:
    from geo_utils import geocode_canada
except Exception as e:
    st.error(f"Failed to import geo_utils.geocode_canada: {e}")
    st.stop()

try:
    from map_picker import pick_location_map
except Exception as e:
    st.error(f"Failed to import map_picker.pick_location_map: {e}")
    st.stop()

try:
    from openmeteo_client import fetch_historical_weather
except Exception as e:
    st.error(f"Failed to import openmeteo_client.fetch_historical_weather: {e}")
    st.stop()

try:
    import plot_utils
except Exception as e:
    st.error(f"Failed to import plot_utils.py: {e}")
    st.stop()

st.title("Canada Weather Dashboard")
st.caption("Location A + Location B. Compare plots on the same graph (A vs B).")
st.write("plot_utils loaded from:", getattr(plot_utils, "__file__", "unknown"))

WINTER_MONTHS = {11, 12, 1, 2, 3}
SUMMER_MONTHS = {4, 5, 6, 7, 8, 9, 10}


# ---------------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------------
def init_state():
    defaults = {
        "map_center_lat_a": 56.1304,
        "map_center_lon_a": -106.3468,
        "map_zoom_a": 4,
        "selected_lat_a": None,
        "selected_lon_a": None,
        "weather_df_a": None,
        "label_a": "Location A",

        "map_center_lat_b": 56.1304,
        "map_center_lon_b": -106.3468,
        "map_zoom_b": 4,
        "selected_lat_b": None,
        "selected_lon_b": None,
        "weather_df_b": None,
        "label_b": "Location B",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ---------------------------------------------------------
# SHARED CONTROLS
# ---------------------------------------------------------
st.subheader("1) Date range, timezone, and variables")

c1, c2, c3 = st.columns(3)
with c1:
    start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01").date())
with c2:
    end_date = st.date_input("End date", value=pd.to_datetime("2020-01-07").date())
with c3:
    timezone = st.selectbox(
        "Timezone",
        ["America/Toronto", "America/Vancouver", "America/Edmonton", "America/Winnipeg", "America/Halifax", "UTC"],
        index=0,
    )

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

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
        "precipitation",
    ],
)

st.divider()


# ---------------------------------------------------------
# LOCATION PICKER UI
# ---------------------------------------------------------
def location_picker(prefix: str, default_search: str):
    label_key = f"label_{prefix}"
    search_key = f"search_{prefix}"
    locate_btn = f"locate_{prefix}"
    fetch_btn = f"fetch_{prefix}"

    col1, col2 = st.columns([1, 2])

    with col1:
        st.session_state[label_key] = st.text_input(
            f"Label for Location {prefix.upper()}",
            value=st.session_state[label_key],
            key=f"label_in_{prefix}",
        )

        search_place = st.text_input(
            f"Search Location {prefix.upper()}",
            value=default_search,
            key=search_key,
        )

        if st.button(f"Locate {prefix.upper()} on map", key=locate_btn):
            with st.spinner(f"Geocoding Location {prefix.upper()}…"):
                geo = geocode_canada(search_place.strip())
            if not geo:
                st.error(f"Location {prefix.upper()} not found.")
            else:
                st.session_state[f"map_center_lat_{prefix}"] = float(geo["latitude"])
                st.session_state[f"map_center_lon_{prefix}"] = float(geo["longitude"])
                st.session_state[f"map_zoom_{prefix}"] = 13
                st.success(f"{prefix.upper()} centered near: {geo.get('display_name','(unknown)')}")

    with col2:
        lat, lon = pick_location_map(
            center_lat=st.session_state[f"map_center_lat_{prefix}"],
            center_lon=st.session_state[f"map_center_lon_{prefix}"],
            zoom=st.session_state[f"map_zoom_{prefix}"],
        )
        if lat is not None and lon is not None:
            st.session_state[f"selected_lat_{prefix}"] = lat
            st.session_state[f"selected_lon_{prefix}"] = lon

        slat = st.session_state[f"selected_lat_{prefix}"]
        slon = st.session_state[f"selected_lon_{prefix}"]

        if slat is not None and slon is not None:
            st.success(f"{prefix.upper()} selected: lat={slat:.6f}, lon={slon:.6f}")
        else:
            st.info(f"Click the map to select Location {prefix.upper()} coordinates.")

    if st.button(f"Fetch data for Location {prefix.upper()}", key=fetch_btn):
        slat = st.session_state[f"selected_lat_{prefix}"]
        slon = st.session_state[f"selected_lon_{prefix}"]
        if slat is None or slon is None:
            st.error(f"Select Location {prefix.upper()} on the map first.")
            st.stop()

        with st.spinner(f"Downloading Location {prefix.upper()} historical data…"):
            df = fetch_historical_weather(
                slat,
                slon,
                pd.to_datetime(start_date).strftime("%Y-%m-%d"),
                pd.to_datetime(end_date).strftime("%Y-%m-%d"),
                timezone,
                hourly_variables,
            )

        if df is None or df.empty:
            st.warning(f"No data returned for Location {prefix.upper()}.")
            st.session_state[f"weather_df_{prefix}"] = None
            return

        if "time" not in df.columns:
            st.error(f"Location {prefix.upper()} dataframe has no 'time' column.")
            st.write("Columns:", list(df.columns))
            st.session_state[f"weather_df_{prefix}"] = None
            return

        st.session_state[f"weather_df_{prefix}"] = df
        st.success(f"Location {prefix.upper()} data fetched.")


# ---------------------------------------------------------
# LOCATION A + B
# ---------------------------------------------------------
st.subheader("2) Location A")
location_picker("a", "Toronto, ON")
st.divider()

st.subheader("3) Location B")
location_picker("b", "Ottawa, ON")
st.divider()

df_a = st.session_state.weather_df_a
df_b = st.session_state.weather_df_b


# ---------------------------------------------------------
# SEASON FILTER
# ---------------------------------------------------------
def apply_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d = d[d["time"].notna()]
    if d.empty:
        return d
    if season == "Winter":
        return d[d["time"].dt.month.isin(WINTER_MONTHS)]
    if season == "Summer":
        return d[d["time"].dt.month.isin(SUMMER_MONTHS)]
    return d


# ---------------------------------------------------------
# SINGLE-LOCATION DASHBOARD (A or B)
# ---------------------------------------------------------
st.subheader("4) Single-location dashboard (existing plots)")

available = []
if df_a is not None and not df_a.empty:
    available.append(st.session_state.label_a)
if df_b is not None and not df_b.empty:
    available.append(st.session_state.label_b)

if not available:
    st.info("Fetch Location A or Location B first.")
else:
    who = st.selectbox("Show dashboard for", available, index=0, key="dash_who")
    base = df_a if who == st.session_state.label_a else df_b

    season = st.selectbox("Season", ["All", "Winter", "Summer"], index=0, key="dash_season")
    d = apply_season(base, season)

    if d is None or d.empty:
        st.warning("No data after season filter.")
    else:
        # Temperature
        if "temperature_2m" in d.columns:
            fig = plot_utils.row_line_violin(d, "temperature_2m", "Temperature (2m)", "°C")
            st.pyplot(fig, use_container_width=True, clear_figure=True)

        # RH
        if "relative_humidity_2m" in d.columns:
            fig = plot_utils.row_line_violin(d, "relative_humidity_2m", "Relative Humidity (2m)", "%")
            st.pyplot(fig, use_container_width=True, clear_figure=True)

        # Rain
        if "rain" in d.columns:
            fig = plot_utils.row_line_violin(d, "rain", "Rain", "mm")
            st.pyplot(fig, use_container_width=True, clear_figure=True)

        # Snow
        if "snowfall" in d.columns:
            fig = plot_utils.row_bar_violin(d, "snowfall", "Snowfall", "cm")
            st.pyplot(fig, use_container_width=True, clear_figure=True)

        # Windrose
        if "wind_speed_10m" in d.columns and "wind_direction_10m" in d.columns:
            fig = plot_utils.row_windrose_violin(
                d,
                speed_col="wind_speed_10m",
                dir_col="wind_direction_10m",
                n_sectors=16,
                violin_col="wind_speed_10m",
                violin_label="m/s",
            )
            st.pyplot(fig, use_container_width=True, clear_figure=True)

st.divider()


# ---------------------------------------------------------
# COMPARE A vs B ON SAME GRAPH (your requirement)
# ---------------------------------------------------------
st.subheader("5) Compare Location A vs Location B (same graph)")

if df_a is None or df_a.empty or df_b is None or df_b.empty:
    st.info("Fetch BOTH Location A and Location B to compare.")
else:
    season_cmp = st.selectbox("Season for A vs B comparison", ["All", "Winter", "Summer"], index=0, key="ab_season")
    da = apply_season(df_a, season_cmp)
    db = apply_season(df_b, season_cmp)

    agg = st.selectbox("Aggregation", ["none", "daily"], index=0, key="ab_agg")

    label_a = st.session_state.label_a
    label_b = st.session_state.label_b

    pairs = [
        ("temperature_2m", "Temperature (°C)", "Temperature (2m)"),
        ("rain", "Rain (mm)", "Rain"),
        ("snowfall", "Snowfall (cm)", "Snowfall"),
        ("wind_speed_10m", "Wind speed (m/s)", "Wind speed (10m)"),
        ("precipitation", "Precipitation (mm)", "Precipitation"),
    ]

    for col, ylabel, title in pairs:
        if col in da.columns and col in db.columns:
            fig = plot_utils.compare_timeseries_ab(
                da, db,
                y=col,
                label_a=label_a,
                label_b=label_b,
                title=f"{title}: {label_a} vs {label_b} ({season_cmp})",
                ylabel=ylabel,
                agg=agg,
            )
            if fig is not None:
                st.pyplot(fig, use_container_width=True, clear_figure=True)
        else:
            st.info(f"Missing column '{col}' in one of the locations (check Hourly variables).")

    st.subheader("Windrose (A vs B)")
    if ("wind_speed_10m" in da.columns and "wind_direction_10m" in da.columns
            and "wind_speed_10m" in db.columns and "wind_direction_10m" in db.columns):
        fig_wr = plot_utils.compare_windrose_ab(
            da, db,
            label_a=label_a,
            label_b=label_b,
            speed_col="wind_speed_10m",
            dir_col="wind_direction_10m",
            n_sectors=16,
        )
        if fig_wr is not None:
            st.pyplot(fig_wr, use_container_width=True, clear_figure=True)
    else:
        st.info("Windrose needs wind_speed_10m and wind_direction_10m for both locations.")
st.divider()
st.header("Statistics / Distribution Fitting")

if df_a is None or df_a.empty:
    st.warning("Location A not loaded yet.")
if df_b is None or df_b.empty:
    st.warning("Location B not loaded yet.")

available_stats = []
if df_a is not None and not df_a.empty:
    available_stats.append(st.session_state.label_a)
if df_b is not None and not df_b.empty:
    available_stats.append(st.session_state.label_b)

# ---------------------------------------------------------
# 1) WEIBULL PARAMETERS (choose dataset + season)
# ---------------------------------------------------------
st.subheader("Weibull parameters (wind_speed_10m)")

if not available_stats:
    st.info("Fetch at least one location to compute Weibull parameters.")
else:
    who_w = st.selectbox("Dataset", available_stats, index=0, key="stats_weibull_ds")
    base_w = df_a if who_w == st.session_state.label_a else df_b

    season_w = st.selectbox("Season", ["All", "Winter", "Summer"], index=0, key="stats_weibull_season")
    df_w = apply_season(base_w, season_w)

    if df_w is None or df_w.empty or "wind_speed_10m" not in df_w.columns:
        st.warning("No wind_speed_10m available for Weibull.")
    else:
        model = st.radio(
            "Weibull model",
            ["Weibull 2-parameter (loc=0)", "Weibull 3-parameter (loc free)"],
            horizontal=True,
            key="stats_weibull_model",
        )
        force_loc0 = model.startswith("Weibull 2")

        try:
            params = plot_utils.weibull_fit_params(df_w["wind_speed_10m"], force_loc0=force_loc0)
        except Exception as e:
            params = None
            st.error(f"Weibull fit failed: {e}")

        if params is None:
            st.warning("Not enough wind data to fit Weibull.")
        else:
            st.write(
                f"**n={params['n']:,}**, "
                f"**shape k={params['shape_k']:.4f}**, "
                f"**scale λ={params['scale_lambda']:.4f}**, "
                f"**loc={params['loc']:.4f}**"
            )
            st.dataframe(pd.DataFrame([params]), use_container_width=True)

st.divider()

# ---------------------------------------------------------
# 2) WIND FIT REPORT (PDF → CDF → Tail) for one dataset
# ---------------------------------------------------------
st.subheader("Wind fit report (PDF → CDF → Tail)")

if not available_stats:
    st.info("Fetch at least one location first.")
else:
    who_r = st.selectbox("Dataset for report", available_stats, index=0, key="stats_report_ds")
    base_r = df_a if who_r == st.session_state.label_a else df_b

    season_r = st.selectbox("Season for report", ["All", "Winter", "Summer"], index=0, key="stats_report_season")
    months_r = None
    if season_r == "Winter":
        months_r = WINTER_MONTHS
    elif season_r == "Summer":
        months_r = SUMMER_MONTHS

    c1, c2, c3 = st.columns(3)
    with c1:
        bin_width = float(st.number_input("Histogram bin width (m/s)", 0.2, 5.0, 0.5, 0.1, key="stats_report_bin"))
    with c2:
        xmax_user = float(st.number_input("Max wind speed (0=auto)", 0.0, 200.0, 0.0, 1.0, key="stats_report_xmax"))
    with c3:
        grid_pts = int(st.number_input("Curve grid points", 200, 2000, 600, 100, key="stats_report_grid"))

    xmax_val = None if xmax_user == 0.0 else xmax_user

    try:
        fig = plot_utils.wind_fit_report_3panel(
            base_r,
            speed_col="wind_speed_10m",
            months_set=months_r,
            season_label=f"{who_r} - {season_r}",
            bin_width=bin_width,
            x_min=0.0,
            x_max=xmax_val,
            n_grid=grid_pts,
        )
    except Exception as e:
        fig = None
        st.error(f"wind_fit_report_3panel failed: {e}")

    if fig is None:
        st.warning("Not enough wind data (or fit failed).")
    else:
        st.pyplot(fig, use_container_width=True, clear_figure=True)

st.divider()

# ---------------------------------------------------------
# 3) WINTER vs SUMMER (PDF + CDF) for one dataset
# ---------------------------------------------------------
st.subheader("Winter vs Summer fits (PDF + CDF, ranked by peak PDF)")

if not available_stats:
    st.info("Fetch at least one location first.")
else:
    who_s = st.selectbox("Dataset for winter vs summer", available_stats, index=0, key="stats_seasonal_ds")
    base_s = df_a if who_s == st.session_state.label_a else df_b

    dist_list = st.multiselect(
        "Distributions",
        ["weibull2", "weibull3", "champernowne", "rayleigh", "rice", "gamma", "lognorm"],
        default=["weibull2", "champernowne", "rayleigh", "rice", "weibull3"],
        key="stats_seasonal_dists",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        min_n = int(st.number_input("Min points per season", 10, 500, 30, 5, key="stats_seasonal_min_n"))
    with c2:
        xmax_user2 = float(st.number_input("Max wind speed (0=auto)", 0.0, 200.0, 0.0, 1.0, key="stats_seasonal_xmax"))
    with c3:
        grid_pts2 = int(st.number_input("Grid points", 200, 2000, 500, 100, key="stats_seasonal_grid"))

    xmax_val2 = None if xmax_user2 == 0.0 else xmax_user2

    try:
        fig = plot_utils.seasonal_pdf_cdf_comparison(
            base_s,
            speed_col="wind_speed_10m",
            winter_months=WINTER_MONTHS,
            summer_months=SUMMER_MONTHS,
            dist_names=dist_list,
            x_min=0.0,
            x_max=xmax_val2,
            n_grid=grid_pts2,
            min_n=min_n,
        )
    except Exception as e:
        fig = None
        st.error(f"seasonal_pdf_cdf_comparison failed: {e}")

    if fig is None:
        st.warning("Not enough data / fits failed.")
    else:
        st.pyplot(fig, use_container_width=True, clear_figure=True)

st.divider()

# ---------------------------------------------------------
# 4) A vs B COMPARE WIND SPEED PDF/CDF
# ---------------------------------------------------------
st.subheader("Compare Location A vs B: wind speed PDF + CDF")

if df_a is None or df_a.empty or df_b is None or df_b.empty:
    st.info("Fetch BOTH locations first.")
else:
    season_cmp = st.selectbox("Season for PDF/CDF compare", ["All", "Winter", "Summer"], index=0, key="stats_cmp_season")
    if season_cmp == "Winter":
        months_cmp = WINTER_MONTHS
    elif season_cmp == "Summer":
        months_cmp = SUMMER_MONTHS
    else:
        months_cmp = None

    c1, c2 = st.columns(2)
    with c1:
        bin_w = float(st.number_input("Histogram bin width (m/s)", 0.2, 5.0, 0.5, 0.1, key="stats_cmp_bin"))
    with c2:
        xmax_user3 = float(st.number_input("Max wind speed (0=auto)", 0.0, 200.0, 0.0, 1.0, key="stats_cmp_xmax"))
    xmax_val3 = None if xmax_user3 == 0.0 else xmax_user3

    fig = plot_utils.compare_two_locations_pdf_cdf(
        df_a, df_b,
        label_a=st.session_state.label_a,
        label_b=st.session_state.label_b,
        speed_col="wind_speed_10m",
        months_set=months_cmp,
        bin_width=bin_w,
        x_min=0.0,
        x_max=xmax_val3,
    )

    if fig is None:
        st.warning("Not enough wind data to compare A vs B for this season.")
    else:
        st.pyplot(fig, use_container_width=True, clear_figure=True)

