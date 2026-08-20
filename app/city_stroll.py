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

CATEGORY_DEFAULTS = {
    "shopping": "independent shops",
    "food": "local restaurants",
    "drink": "cafes and tea shops",
    "interest": "cultural attractions",
}
MAX_PREFERENCES_PER_CATEGORY = 2
MAX_REQUIRED_PREFERENCES = 3


def _preference_key(value: str) -> str:
    return " ".join(value.casefold().split())


def _clean_preferences(values: list[str], limit: int) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        phrase = " ".join(value.split())
        key = _preference_key(phrase)
        if not phrase or key in seen:
            continue
        cleaned.append(phrase)
        seen.add(key)
        if len(cleaned) == limit:
            break
    return cleaned


def compile_searches(
    shopping_preferences: list[str],
    food_preferences: list[str],
    drink_preferences: list[str],
    interest_preferences: list[str],
    required_preferences: list[str],
) -> tuple[list[SearchSpec], frozenset[str]]:
    """Compile structured traveler preferences into bounded Places searches."""

    required = _clean_preferences(required_preferences, MAX_REQUIRED_PREFERENCES)
    required_by_key = {_preference_key(value): value for value in required}
    categorized = {
        "shopping": shopping_preferences,
        "food": food_preferences,
        "drink": drink_preferences,
        "interest": interest_preferences,
    }
    searches: list[SearchSpec] = []
    seen: set[str] = set()
    for category, values in categorized.items():
        phrases = _clean_preferences(values, MAX_PREFERENCES_PER_CATEGORY)
        if not phrases:
            phrases = [CATEGORY_DEFAULTS[category]]
        elif (
            len(phrases) == 1
            and category in {"shopping", "interest"}
            and _preference_key(phrases[0])
            != _preference_key(CATEGORY_DEFAULTS[category])
        ):
            # These two categories are hard gates for every route. A generic,
            # city-neutral discovery query prevents one narrow search's top results
            # from concentrating every viable route in the same neighborhood.
            phrases.append(CATEGORY_DEFAULTS[category])
        for phrase in phrases:
            key = _preference_key(phrase)
            searches.append(
                SearchSpec(
                    category=category,
                    query=phrase,
                    preference_key=key,
                    required=key in required_by_key,
                )
            )
            seen.add(key)

    for key, phrase in required_by_key.items():
        if key not in seen:
            searches.append(
                SearchSpec(
                    category="preference",
                    query=phrase,
                    preference_key=key,
                    required=True,
                )
            )
    return searches, frozenset(required_by_key)


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
            existing.required_preferences.update(place.required_preferences)
            existing.search_rank = min(existing.search_rank, place.search_rank)
    return list(merged.values())


def _taste_explanation(place: Place) -> str:
    labels = {
        "shopping": "shopping match",
        "food": "food match",
        "drink": "tea or coffee match",
        "interest": "place-of-interest match",
        "preference": "direct request match",
    }
    return ", ".join(labels.get(category, category) for category in sorted(place.categories))


async def generate_paths(
    city: str,
    shopping_preferences: list[str] = [],
    food_preferences: list[str] = [],
    drink_preferences: list[str] = [],
    interest_preferences: list[str] = [],
    required_preferences: list[str] = [],
    avoidances: list[str] = [],
) -> dict:
    """Generate three compact, real-venue walking strolls for a city.

    Args:
        city: City plus country or region when the name could be ambiguous.
        shopping_preferences: Desired shop types, products, or styles.
        food_preferences: Desired cuisines, dishes, or restaurant styles.
        drink_preferences: Desired drinks or cafe styles.
        interest_preferences: Desired culture, architecture, parks, or attractions.
        required_preferences: At most three preferences that every path must cover.
            Repeat each item in its appropriate category list when possible.
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

    searches, required = compile_searches(
        shopping_preferences,
        food_preferences,
        drink_preferences,
        interest_preferences,
        required_preferences,
    )
    preference_summary = [
        spec.query
        for spec in searches
        if spec.query not in CATEGORY_DEFAULTS.values()
    ]
    client = GoogleMapsClient(api_key)
    try:
        resolved_city = await client.resolve_city(city.strip())
        results = await asyncio.gather(
            *(client.search_text(search, resolved_city) for search in searches)
        )
        places = _merge_places(results)
        candidates = build_neighborhood_candidates(
            places,
            required_preferences=required,
            city_name=resolved_city.name,
        )
        if len(candidates) < 3:
            return {
                "status": "insufficient_data",
                "reason": "Fewer than three compact, category-balanced areas were found.",
                "resolvedCity": resolved_city.formatted_address,
                "candidateVenueCount": len(places),
                "pathCount": len(candidates),
                "requiredPreferences": sorted(required),
            }

        paths = []
        for candidate in candidates[:3]:
            matrix = await client.route_matrix(candidate.places)
            ordered, distance_meters = optimize_path_with_route_matrix(
                candidate.places, matrix, required_preferences=required
            )
            caveats = [
                "Verify opening hours, dietary needs, accessibility, and current conditions directly with venues."
            ]
            if distance_meters < 2000 or distance_meters > 4000:
                caveats.append(
                    "This live walking estimate falls outside the preferred 2-4 km range."
                )
            if avoidances:
                caveats.append(
                    "Avoidance requests are not guaranteed by text search and must be "
                    f"verified: {', '.join(_clean_preferences(avoidances, 8))}"
                )
            paths.append(
                {
                    "neighborhood": candidate.name,
                    "rationale": (
                        "A compact mix of shopping, food or drink, and a place of interest, "
                        "matched against: "
                        f"{', '.join(preference_summary) or 'city-neutral defaults'}."
                    ),
                    "walkingDistanceMeters": distance_meters,
                    "routeUrl": build_route_url(ordered),
                    "stops": [
                        {
                            "order": index + 1,
                            "name": place.name,
                            "category": sorted(place.categories),
                            "tasteMatch": _taste_explanation(place),
                            "matchedPreferences": sorted(place.matched_queries),
                            "requiredPreferences": sorted(place.required_preferences),
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
            "cityQuery": city.strip(),
            "resolvedCity": {
                "name": resolved_city.name,
                "formattedAddress": resolved_city.formatted_address,
                "placeId": resolved_city.place_id,
            },
            "requiredPreferences": sorted(required),
            "paths": paths,
            "source": "Google Places API (New) and Routes API",
        }
    except MapsApiError as error:
        return {"status": "maps_api_error", "reason": str(error)}
