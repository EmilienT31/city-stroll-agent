"""Live City Stroll vertical slice exposed as an ADK tool."""

from __future__ import annotations

import asyncio
import os

from app.maps_client import GoogleMapsClient, MapsApiError, SearchSpec
from app.stroll_planner import (
    Place,
    build_neighborhood_candidates,
    build_route_url,
    optimize_path_with_route_matrix,
)

DEFAULT_SEARCHES = (
    SearchSpec("shopping", "independent fashion menswear and design shops"),
    SearchSpec("food", "ramen shops"),
    SearchSpec("drink", "matcha tea cafes"),
    SearchSpec("interest", "cultural attractions museums temples and architecture"),
    SearchSpec("drink", "traditional coffee shops and kissaten"),
)


def _merge_places(search_results: list[list[Place]]) -> list[Place]:
    merged: dict[str, Place] = {}
    for result in search_results:
        for place in result:
            existing = merged.get(place.place_id)
            if existing is None:
                merged[place.place_id] = place
                continue
            existing.categories.update(place.categories)
            existing.matched_queries.update(place.matched_queries)
            existing.search_rank = min(existing.search_rank, place.search_rank)
    return list(merged.values())


def _taste_explanation(place: Place) -> str:
    labels = {
        "shopping": "shopping match",
        "food": "food match",
        "drink": "tea or coffee match",
        "interest": "place-of-interest match",
        "taste": "direct request match",
    }
    return ", ".join(labels.get(category, category) for category in sorted(place.categories))


async def generate_paths(city: str, tastes: str, avoidances: str = "") -> dict:
    """Generate three compact, real-venue walking strolls for a city.

    Args:
        city: City plus country or region when the name could be ambiguous.
        tastes: Temporary trip tastes such as shopping, cuisine, drinks, and culture.
        avoidances: Optional things the traveler wants to avoid for this trip.

    Returns:
        A structured result with three alternatives when venue coverage is sufficient.
    """

    if not city.strip():
        return {"status": "insufficient_data", "reason": "A city is required."}
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        return {
            "status": "configuration_error",
            "reason": "GOOGLE_MAPS_API_KEY is not set in the server environment.",
        }

    searches = list(DEFAULT_SEARCHES)
    if tastes.strip():
        searches.append(SearchSpec("taste", tastes.strip()))
    client = GoogleMapsClient(api_key)
    try:
        results = await asyncio.gather(
            *(client.search_text(search, city.strip()) for search in searches)
        )
        places = _merge_places(results)
        candidates = build_neighborhood_candidates(places)
        if len(candidates) < 3:
            return {
                "status": "insufficient_data",
                "reason": "Fewer than three compact, category-balanced areas were found.",
                "candidateVenueCount": len(places),
                "pathCount": len(candidates),
            }

        paths = []
        for candidate in candidates[:3]:
            matrix = await client.route_matrix(candidate.places)
            ordered, distance_meters = optimize_path_with_route_matrix(
                candidate.places, matrix
            )
            caveats = [
                "Verify opening hours, dietary needs, accessibility, and current conditions directly with venues."
            ]
            if distance_meters < 2000 or distance_meters > 4000:
                caveats.append(
                    "This live walking estimate falls outside the preferred 2-4 km range."
                )
            if avoidances.strip():
                caveats.append(
                    "Avoidance request was used as context but is not guaranteed: "
                    f"{avoidances.strip()}"
                )
            paths.append(
                {
                    "neighborhood": candidate.name,
                    "rationale": (
                        "A compact mix of shopping, food or drink, and a place of interest, "
                        f"matched against: {tastes.strip() or 'the default prototype profile'}."
                    ),
                    "walkingDistanceMeters": distance_meters,
                    "routeUrl": build_route_url(ordered),
                    "stops": [
                        {
                            "order": index + 1,
                            "name": place.name,
                            "category": sorted(place.categories),
                            "tasteMatch": _taste_explanation(place),
                            "placeId": place.place_id,
                            "address": place.address,
                            "googleMapsUri": place.google_maps_uri,
                        }
                        for index, place in enumerate(ordered)
                    ],
                    "caveats": caveats,
                }
            )
        return {
            "status": "ok",
            "city": city.strip(),
            "paths": paths,
            "source": "Google Places API (New) and Routes API",
        }
    except MapsApiError as error:
        return {"status": "maps_api_error", "reason": str(error)}
