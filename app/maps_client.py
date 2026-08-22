"""Small async clients for Places API (New) and Routes API."""

from __future__ import annotations

import json
from dataclasses import dataclass

import aiohttp

from app.stroll_planner import Place

PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
ROUTE_MATRIX_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
GOOGLE_MAPS_SOLUTION_ID = "gmp_git_agentskills_v1"
CITY_PLACE_TYPES = {
    "locality",
    "postal_town",
    "administrative_area_level_1",
    "administrative_area_level_2",
    "administrative_area_level_3",
}


def build_request_headers(api_key: str, field_mask: str) -> dict[str, str]:
    """Build the required headers shared by Maps Platform POST requests."""

    return {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": field_mask,
        "X-Goog-Maps-Solution-ID": GOOGLE_MAPS_SOLUTION_ID,
    }


class MapsApiError(RuntimeError):
    """Raised when a Google Maps Platform request fails."""


@dataclass(frozen=True)
class SearchSpec:
    category: str
    query: str
    preference_key: str
    required: bool = False


@dataclass(frozen=True)
class CityResolution:
    place_id: str
    name: str
    formatted_address: str
    latitude: float
    longitude: float
    viewport_low: tuple[float, float]
    viewport_high: tuple[float, float]


@dataclass(frozen=True)
class AnchorResolution:
    place_id: str
    name: str
    formatted_address: str
    latitude: float
    longitude: float
    google_maps_uri: str

    def as_place(self) -> Place:
        """Return the planner representation without treating the anchor as a stop."""

        return Place(
            place_id=self.place_id,
            name=self.name,
            address=self.formatted_address,
            latitude=self.latitude,
            longitude=self.longitude,
            google_maps_uri=self.google_maps_uri,
            categories={"anchor"},
        )


def location_in_city_viewport(
    latitude: float, longitude: float, city: CityResolution
) -> bool:
    """Return whether coordinates fall inside the resolved city rectangle."""

    latitude_inside = city.viewport_low[0] <= latitude <= city.viewport_high[0]
    low_longitude = city.viewport_low[1]
    high_longitude = city.viewport_high[1]
    if low_longitude <= high_longitude:
        longitude_inside = low_longitude <= longitude <= high_longitude
    else:
        longitude_inside = longitude >= low_longitude or longitude <= high_longitude
    return latitude_inside and longitude_inside


class GoogleMapsClient:
    def __init__(self, api_key: str, timeout_seconds: int = 20) -> None:
        if not api_key:
            raise ValueError("A Google Maps API key is required")
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def resolve_city(self, city: str) -> CityResolution:
        """Resolve a user-supplied city to a canonical locality and viewport."""

        headers = build_request_headers(
            self._api_key,
            (
                "places.id,places.displayName,places.formattedAddress,"
                "places.location,places.viewport,places.types"
            ),
        )
        payload = {
            "textQuery": city,
            "pageSize": 5,
            "languageCode": "en",
        }
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(
                PLACES_SEARCH_URL, headers=headers, json=payload
            ) as response:
                body = await response.text()
                if response.status != 200:
                    raise MapsApiError(
                        f"City resolution returned HTTP {response.status}: {body[:500]}"
                    )

        for item in json.loads(body).get("places", []):
            if not CITY_PLACE_TYPES.intersection(item.get("types", [])):
                continue
            location = item.get("location") or {}
            viewport = item.get("viewport") or {}
            low = viewport.get("low") or {}
            high = viewport.get("high") or {}
            required_values = (
                item.get("id"),
                (item.get("displayName") or {}).get("text"),
                location.get("latitude"),
                location.get("longitude"),
                low.get("latitude"),
                low.get("longitude"),
                high.get("latitude"),
                high.get("longitude"),
            )
            if any(value is None for value in required_values):
                continue
            return CityResolution(
                place_id=str(required_values[0]),
                name=str(required_values[1]),
                formatted_address=item.get("formattedAddress", ""),
                latitude=float(required_values[2]),
                longitude=float(required_values[3]),
                viewport_low=(float(required_values[4]), float(required_values[5])),
                viewport_high=(float(required_values[6]), float(required_values[7])),
            )
        raise MapsApiError(
            "No city or administrative area with a usable viewport was found. "
            "Add a country or region."
        )

    async def resolve_anchor(
        self, anchor: str, city: CityResolution
    ) -> AnchorResolution:
        """Resolve an anchor and reject results outside the requested city."""

        headers = build_request_headers(
            self._api_key,
            (
                "places.id,places.displayName,places.formattedAddress,"
                "places.location,places.googleMapsUri,places.businessStatus"
            ),
        )
        payload = {
            "textQuery": f"{anchor} in {city.name}",
            "pageSize": 5,
            "languageCode": "en",
            "locationBias": {
                "rectangle": {
                    "low": {
                        "latitude": city.viewport_low[0],
                        "longitude": city.viewport_low[1],
                    },
                    "high": {
                        "latitude": city.viewport_high[0],
                        "longitude": city.viewport_high[1],
                    },
                }
            },
        }
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(
                PLACES_SEARCH_URL, headers=headers, json=payload
            ) as response:
                body = await response.text()
                if response.status != 200:
                    raise MapsApiError(
                        f"Anchor resolution returned HTTP {response.status}: "
                        f"{body[:500]}"
                    )

        for item in json.loads(body).get("places", []):
            if item.get("businessStatus") == "CLOSED_PERMANENTLY":
                continue
            location = item.get("location") or {}
            latitude = location.get("latitude")
            longitude = location.get("longitude")
            required_values = (
                item.get("id"),
                (item.get("displayName") or {}).get("text"),
                latitude,
                longitude,
                item.get("googleMapsUri"),
            )
            if any(value is None for value in required_values):
                continue
            if not location_in_city_viewport(float(latitude), float(longitude), city):
                continue
            return AnchorResolution(
                place_id=str(required_values[0]),
                name=str(required_values[1]),
                formatted_address=item.get("formattedAddress", ""),
                latitude=float(latitude),
                longitude=float(longitude),
                google_maps_uri=str(required_values[4]),
            )
        raise MapsApiError(
            f"The anchor '{anchor}' could not be resolved inside {city.name}."
        )

    async def search_text(self, spec: SearchSpec, city: CityResolution) -> list[Place]:
        headers = build_request_headers(
            self._api_key,
            (
                "places.id,places.displayName,places.formattedAddress,"
                "places.location,places.googleMapsUri,places.rating,"
                "places.userRatingCount,places.businessStatus"
            ),
        )
        payload = {
            "textQuery": f"{spec.query} in {city.name}",
            "pageSize": 15,
            "languageCode": "en",
            "locationRestriction": {
                "rectangle": {
                    "low": {
                        "latitude": city.viewport_low[0],
                        "longitude": city.viewport_low[1],
                    },
                    "high": {
                        "latitude": city.viewport_high[0],
                        "longitude": city.viewport_high[1],
                    },
                }
            },
        }
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(
                PLACES_SEARCH_URL, headers=headers, json=payload
            ) as response:
                body = await response.text()
                if response.status != 200:
                    raise MapsApiError(
                        f"Places API returned HTTP {response.status}: {body[:500]}"
                    )
        data = json.loads(body)
        results: list[Place] = []
        for search_rank, item in enumerate(data.get("places", [])):
            if item.get("businessStatus") == "CLOSED_PERMANENTLY":
                continue
            location = item.get("location") or {}
            place_id = item.get("id")
            name = (item.get("displayName") or {}).get("text")
            latitude = location.get("latitude")
            longitude = location.get("longitude")
            maps_uri = item.get("googleMapsUri")
            if (
                not all((place_id, name, maps_uri))
                or latitude is None
                or longitude is None
            ):
                continue
            place = Place(
                place_id=place_id,
                name=name,
                address=item.get("formattedAddress", ""),
                latitude=float(latitude),
                longitude=float(longitude),
                google_maps_uri=maps_uri,
                rating=float(item["rating"]) if "rating" in item else None,
                user_rating_count=int(item.get("userRatingCount", 0)),
                search_rank=search_rank,
            )
            place.merge_match(
                spec.category,
                spec.preference_key,
                required=spec.required,
            )
            results.append(place)
        return results

    async def route_matrix(self, places: tuple[Place, ...]) -> list[list[int]]:
        if not places:
            raise ValueError("At least one route-matrix waypoint is required")
        if len(places) ** 2 > 625:
            raise ValueError("Route matrix exceeds the 625-element walking limit")
        waypoints = [
            {
                "waypoint": {
                    "location": {
                        "latLng": {
                            "latitude": place.latitude,
                            "longitude": place.longitude,
                        }
                    }
                }
            }
            for place in places
        ]
        headers = build_request_headers(
            self._api_key,
            ("originIndex,destinationIndex,distanceMeters,duration,status,condition"),
        )
        payload = {
            "origins": waypoints,
            "destinations": waypoints,
            "travelMode": "WALK",
        }
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(
                ROUTE_MATRIX_URL, headers=headers, json=payload
            ) as response:
                body = await response.text()
                if response.status != 200:
                    raise MapsApiError(
                        f"Routes API returned HTTP {response.status}: {body[:500]}"
                    )
        try:
            elements = json.loads(body)
        except json.JSONDecodeError:
            elements = [json.loads(line) for line in body.splitlines() if line.strip()]
        if isinstance(elements, dict):
            elements = elements.get("routeMatrix", [])

        size = len(places)
        matrix = [[-1 for _ in range(size)] for _ in range(size)]
        for index in range(size):
            matrix[index][index] = 0
        for element in elements:
            origin = element.get("originIndex")
            destination = element.get("destinationIndex")
            distance = element.get("distanceMeters")
            if (
                isinstance(origin, int)
                and isinstance(destination, int)
                and isinstance(distance, int)
                and element.get("condition", "ROUTE_EXISTS") == "ROUTE_EXISTS"
            ):
                matrix[origin][destination] = distance
        return matrix
