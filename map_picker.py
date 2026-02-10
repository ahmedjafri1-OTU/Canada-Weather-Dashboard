# map_picker.py
import streamlit as st
import folium
from streamlit_folium import st_folium

def pick_location_map(center_lat: float, center_lon: float, zoom: int, key: str):
    """
    Shows an interactive map and returns (lat, lon) when user clicks.
    IMPORTANT: `key` must be unique per map instance (map_a, map_b)
    """

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)

    st.caption("Tip: click on the map to select coordinates.")

    out = st_folium(
        m,
        width=750,
        height=450,
        key=key,  
    )

    lat = lon = None
    if out and out.get("last_clicked"):
        lat = out["last_clicked"]["lat"]
        lon = out["last_clicked"]["lng"]

    return lat, lon
