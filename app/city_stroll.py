"""Live City Stroll vertical slice exposed as an ADK tool."""

from __future__ import annotations

import asyncio
import os

from app.maps_client import AnchorResolution, GoogleMapsClient, MapsApiError, SearchSpec
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
MAX_CANDIDATE_AREAS = 12


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
    return ", ".join(
        labels.get(category, category) for category in sorted(place.categories)
    )


def _serialize_anchor(anchor: AnchorResolution | None) -> dict | None:
    if anchor is None:
        return None
    return {
        "name": anchor.name,
        "placeId": anchor.place_id,
        "address": anchor.formatted_address,
        "latitude": anchor.latitude,
        "longitude": anchor.longitude,
        "googleMapsUri": anchor.google_maps_uri,
    }


async def generate_paths(
    city: str,
    shopping_preferences: list[str] = [],
    food_preferences: list[str] = [],
    drink_preferences: list[str] = [],
    interest_preferences: list[str] = [],
    required_preferences: list[str] = [],
    avoidances: list[str] = [],
    start_anchor: str = "",
    end_anchor: str = "",
    round_trip: bool = False,
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
        start_anchor: Optional hotel, station, address, or landmark to start from.
        end_anchor: Optional hotel, station, address, or landmark to finish at.
        round_trip: Return to start_anchor after visiting the selected venues.

    Returns:
        A structured result with three alternatives when venue coverage is sufficient.
    """

    if not city.strip():
        return {"status": "insufficient_data", "reason": "A city is required."}
    start_query = " ".join(start_anchor.split())
    end_query = " ".join(end_anchor.split())
    if round_trip and not start_query:
        return {
            "status": "insufficient_data",
            "reason": "A start anchor is required for a round trip.",
        }
    if (
        round_trip
        and end_query
        and _preference_key(end_query) != _preference_key(start_query)
    ):
        return {
            "status": "insufficient_data",
            "reason": "A round trip cannot use a different end anchor.",
        }
    if round_trip:
        end_query = start_query
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
        spec.query for spec in searches if spec.query not in CATEGORY_DEFAULTS.values()
    ]
    client = GoogleMapsClient(api_key)
    try:
        resolved_city = await client.resolve_city(city.strip())
        resolved_start = (
            await client.resolve_anchor(start_query, resolved_city)
            if start_query
            else None
        )
        if end_query and (
            resolved_start is None
            or _preference_key(end_query) != _preference_key(start_query)
        ):
            resolved_end = await client.resolve_anchor(end_query, resolved_city)
        else:
            resolved_end = resolved_start if end_query else None

        raw_results = await asyncio.gather(
            *(client.search_text(search, resolved_city) for search in searches),
            return_exceptions=True,
        )
        results: list[list[Place]] = []
        search_failure_count = 0
        for result in raw_results:
            if isinstance(result, MapsApiError):
                search_failure_count += 1
            elif isinstance(result, BaseException):
                raise result
            else:
                results.append(result)
        if not results:
            raise MapsApiError("All Places searches failed.")
        places = _merge_places(results)
        anchor_ids = {
            anchor.place_id
            for anchor in (resolved_start, resolved_end)
            if anchor is not None
        }
        places = [place for place in places if place.place_id not in anchor_ids]
        anchor_places_by_id = {
            anchor.place_id: anchor.as_place()
            for anchor in (resolved_start, resolved_end)
            if anchor is not None
        }
        anchor_places = tuple(anchor_places_by_id.values())
        candidates = build_neighborhood_candidates(
            places,
            required_preferences=required,
            city_name=resolved_city.name,
            anchor_points=anchor_places,
            max_candidates=MAX_CANDIDATE_AREAS,
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
        route_failure_count = 0
        attempted_route_distances: list[int] = []
        used_stop_ids: set[str] = set()
        candidate_queue = list(candidates)
        while candidate_queue and len(paths) < 3:
            candidate = candidate_queue.pop(0)
            route_nodes = list(candidate.places)
            start_anchor_place = resolved_start.as_place() if resolved_start else None
            end_anchor_place = resolved_end.as_place() if resolved_end else None
            start_anchor_index = None
            end_anchor_index = None
            if start_anchor_place is not None:
                start_anchor_index = len(route_nodes)
                route_nodes.append(start_anchor_place)
            if end_anchor_place is not None:
                if (
                    start_anchor_place is not None
                    and end_anchor_place.place_id == start_anchor_place.place_id
                ):
                    end_anchor_index = start_anchor_index
                else:
                    end_anchor_index = len(route_nodes)
                    route_nodes.append(end_anchor_place)
            try:
                matrix = await client.route_matrix(tuple(route_nodes))
            except MapsApiError:
                route_failure_count += 1
                continue
            ordered, distance_meters = optimize_path_with_route_matrix(
                candidate.places,
                matrix,
                required_preferences=required,
                start_anchor_index=start_anchor_index,
                end_anchor_index=end_anchor_index,
                excluded_place_ids=frozenset(used_stop_ids),
            )
            if distance_meters >= 0:
                attempted_route_distances.append(distance_meters)
            if not ordered or not 2000 <= distance_meters <= 4000:
                continue
            caveats = [
                "Verify opening hours, dietary needs, accessibility, and current conditions directly with venues."
            ]
            if avoidances:
                caveats.append(
                    "Avoidance requests are not guaranteed by text search and must be "
                    f"verified: {', '.join(_clean_preferences(avoidances, 8))}"
                )
            if search_failure_count:
                caveats.append(
                    f"{search_failure_count} venue search request(s) failed; "
                    "the route uses the remaining live results."
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
                    "routeUrl": build_route_url(
                        ordered,
                        start_anchor=start_anchor_place,
                        end_anchor=end_anchor_place,
                    ),
                    "anchors": {
                        "start": _serialize_anchor(resolved_start),
                        "end": _serialize_anchor(resolved_end),
                        "roundTrip": round_trip,
                    },
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
            used_stop_ids.update(place.place_id for place in ordered)
            if anchor_places and len(paths) < 3:
                remaining_places = [
                    place for place in places if place.place_id not in used_stop_ids
                ]
                candidate_queue = build_neighborhood_candidates(
                    remaining_places,
                    required_preferences=required,
                    city_name=resolved_city.name,
                    anchor_points=anchor_places,
                    max_candidates=MAX_CANDIDATE_AREAS,
                )
        if len(paths) < 3:
            return {
                "status": "insufficient_data",
                "reason": (
                    "Fewer than three routes satisfied every preference, anchor, "
                    "and the 2-4 km walking target."
                ),
                "resolvedCity": resolved_city.formatted_address,
                "candidateVenueCount": len(places),
                "pathCount": len(paths),
                "attemptedRouteDistancesMeters": attempted_route_distances,
                "searchFailureCount": search_failure_count,
                "routeMatrixFailureCount": route_failure_count,
                "requiredPreferences": sorted(required),
                "anchors": {
                    "start": _serialize_anchor(resolved_start),
                    "end": _serialize_anchor(resolved_end),
                    "roundTrip": round_trip,
                },
            }
        return {
            "status": "ok",
            "cityQuery": city.strip(),
            "resolvedCity": {
                "name": resolved_city.name,
                "formattedAddress": resolved_city.formatted_address,
                "placeId": resolved_city.place_id,
            },
            "requiredPreferences": sorted(required),
            "anchors": {
                "start": _serialize_anchor(resolved_start),
                "end": _serialize_anchor(resolved_end),
                "roundTrip": round_trip,
            },
            "paths": paths,
            "source": "Google Places API (New) and Routes API",
            "attribution": "Google Maps",
        }
    except MapsApiError as error:
        return {"status": "maps_api_error", "reason": str(error)}
