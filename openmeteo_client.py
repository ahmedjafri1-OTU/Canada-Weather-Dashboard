# openmeteo_client.py

import requests
import pandas as pd


def fetch_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str,
    hourly_variables: list[str]
) -> pd.DataFrame:

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_variables),
        "timezone": timezone
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    hourly = data.get("hourly")

    if hourly is None or "time" not in hourly:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])

    return df
if __name__ == "__main__":

    df = fetch_historical_weather(
        43.6532,
        -79.3832,
        "2020-01-01",
        "2020-01-02",
        "America/Toronto",
        ["temperature_2m", "wind_speed_10m"]
    )

    print(df.head())
