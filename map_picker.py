# map_picker.py
import folium
from streamlit_folium import st_folium

def pick_location_map(
    center_lat=56.1304,   # Canada center-ish
    center_lon=-106.3468,
    zoom=4
):
    """
    Returns (lat, lon) if user clicked on the map, else (None, None).
    Uses OpenStreetMap tiles via Folium (Leaflet).
    """

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="OpenStreetMap"
    )

    # Optional: show lat/lon under mouse
    folium.LatLngPopup().add_to(m)

    # Render and capture interactions
    out = st_folium(m, height=520, width=None)

    # When user clicks, streamlit-folium returns "last_clicked"
    last = out.get("last_clicked")
    if last:
        return float(last["lat"]), float(last["lng"])

    return None, None
