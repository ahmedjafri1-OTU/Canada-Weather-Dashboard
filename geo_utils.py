from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

def get_geocoder():
    geolocator=Nominatim(
        user_agent="Canada_weather_dashboard"

    )
    return RateLimiter(geolocator.geocode, min_delay_seconds=1)

def geocode_canada(place_name: str):
    geocode = get_geocoder()

    location = geocode(
        place_name,
        country_codes="ca",
        addressdetails=True
    )

    if location is None:
        return None

    return {
        "display_name": location.address,
        "latitude": float(location.latitude),
        "longitude": float(location.longitude)
    }

if __name__ == "__main__":
    result = geocode_canada("Toronto, ON")
    print(result)

