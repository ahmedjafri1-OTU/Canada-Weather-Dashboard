# map_picker.py

import folium
from streamlit_folium import st_folium


def pick_location_map(
    center_lat=56.1304,
    center_lon=-106.3468,
    zoom=5
):
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="OpenStreetMap"
    )

    folium.LatLngPopup().add_to(m)

    out = st_folium(m, height=520, width=None)

    last = out.get("last_clicked")
    if last:
        return float(last["lat"]), float(last["lng"])

    return None, None
