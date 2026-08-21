# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio

from app.city_stroll import CATEGORY_DEFAULTS, compile_searches, generate_paths
from app.maps_client import (
    GOOGLE_MAPS_SOLUTION_ID,
    CityResolution,
    build_request_headers,
    location_in_city_viewport,
)
from app.stroll_planner import (
    Place,
    build_neighborhood_candidates,
    build_route_url,
    haversine_km,
    optimize_path_with_route_matrix,
    order_with_route_matrix,
)
from tests.eval.response_quality import validate_success_result


def test_maps_request_headers_include_solution_attribution() -> None:
    headers = build_request_headers("test-key", "places.id")
    assert headers == {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": "test-key",
        "X-Goog-FieldMask": "places.id",
        "X-Goog-Maps-Solution-ID": GOOGLE_MAPS_SOLUTION_ID,
    }


def _place(place_id: str, latitude: float, longitude: float, category: str) -> Place:
    return Place(
        place_id=place_id,
        name=f"Venue{place_id.replace('-', '')}Brand",
        address="Shibuya City, Tokyo, Japan",
        latitude=latitude,
        longitude=longitude,
        google_maps_uri=f"https://maps.google.com/?cid={place_id}",
        rating=4.5,
        user_rating_count=100,
        categories={category},
        matched_queries={category},
    )


def test_haversine_distance_is_symmetric() -> None:
    first = _place("a", 35.6595, 139.7005, "shopping")
    second = _place("b", 35.6654, 139.7120, "food")
    assert haversine_km(first, second) == haversine_km(second, first)
    assert 1.0 < haversine_km(first, second) < 1.5


def test_build_route_url_contains_ordered_waypoints() -> None:
    places = tuple(
        _place(str(index), 35.65 + index * 0.001, 139.70 + index * 0.001, "food")
        for index in range(4)
    )
    url = build_route_url(places)
    assert "travelmode=walking" in url
    assert "waypoints=" in url
    assert "|" in url


def test_build_route_url_honors_round_trip_anchor() -> None:
    anchor = _place("anchor", 35.6580, 139.7016, "anchor")
    places = tuple(
        _place(str(index), 35.66 + index * 0.001, 139.70, "food") for index in range(3)
    )
    url = build_route_url(places, start_anchor=anchor, end_anchor=anchor)
    assert "origin=35.658000,139.701600" in url
    assert "destination=35.658000,139.701600" in url


def test_route_matrix_controls_order() -> None:
    places = tuple(
        _place(str(index), 35.65, 139.70 + index * 0.001, "food") for index in range(3)
    )
    matrix = [[0, 50, 10], [50, 0, 10], [10, 10, 0]]
    ordered = order_with_route_matrix(places, matrix)
    assert [place.place_id for place in ordered] == ["0", "2", "1"]


def test_route_optimizer_drops_detour_but_keeps_category_balance() -> None:
    categories = ("shopping", "food", "drink", "interest", "shopping", "drink")
    places = tuple(
        _place(str(index), 35.65, 139.70 + index * 0.001, category)
        for index, category in enumerate(categories)
    )
    matrix = [[abs(first - second) * 500 for second in range(6)] for first in range(6)]
    for index in range(5):
        matrix[index][5] = 3000
        matrix[5][index] = 3000
    ordered, distance = optimize_path_with_route_matrix(places, matrix)
    assert len(ordered) == 5
    assert "5" not in {place.place_id for place in ordered}
    assert 2000 <= distance <= 4000


def test_route_optimizer_honors_fixed_start_anchor() -> None:
    categories = ("shopping", "food", "drink", "interest", "shopping")
    places = tuple(
        _place(str(index), 35.65, 139.70 + index * 0.001, category)
        for index, category in enumerate(categories)
    )
    matrix = [[500 for _ in range(6)] for _ in range(6)]
    for index in range(6):
        matrix[index][index] = 0
    matrix[5] = [100, 3000, 3000, 3000, 3000, 0]
    ordered, distance = optimize_path_with_route_matrix(
        places, matrix, start_anchor_index=5
    )
    assert ordered[0].place_id == "0"
    assert distance == 2100


def test_route_optimizer_rejects_unreachable_complete_path() -> None:
    categories = ("shopping", "food", "drink", "interest", "shopping")
    places = tuple(
        _place(str(index), 35.65, 139.70 + index * 0.001, category)
        for index, category in enumerate(categories)
    )
    matrix = [[-1 for _ in range(5)] for _ in range(5)]
    for index in range(5):
        matrix[index][index] = 0
    assert optimize_path_with_route_matrix(places, matrix) == ((), -1)


def test_neighborhood_candidates_are_distinct_and_balanced() -> None:
    categories = ("shopping", "food", "drink", "interest", "shopping", "food")
    places = []
    for cluster_index, longitude in enumerate((139.70, 139.74, 139.78)):
        for index, category in enumerate(categories):
            places.append(
                _place(
                    f"{cluster_index}-{index}",
                    35.65 + index * 0.001,
                    longitude + index * 0.001,
                    category,
                )
            )
    candidates = build_neighborhood_candidates(places)
    assert len(candidates) == 3
    candidate_ids = [
        {place.place_id for place in candidate.places} for candidate in candidates
    ]
    assert all(
        not candidate_ids[first] & candidate_ids[second]
        for first in range(len(candidate_ids))
        for second in range(first + 1, len(candidate_ids))
    )
    assert all(5 <= len(candidate.places) <= 7 for candidate in candidates)
    assert all(
        {"shopping", "interest"}.issubset(
            set().union(*(place.categories for place in candidate.places))
        )
        for candidate in candidates
    )


def test_searches_are_compiled_from_structured_preferences() -> None:
    searches, required = compile_searches(
        shopping_preferences=["ceramics", "vintage clothing", "ignored third item"],
        food_preferences=["vegetarian bistros"],
        drink_preferences=[],
        interest_preferences=["modern architecture"],
        required_preferences=["ceramics"],
    )
    assert required == frozenset({"ceramics"})
    assert [search.query for search in searches if search.category == "shopping"] == [
        "ceramics",
        "vintage clothing",
    ]
    assert any(
        search.preference_key == "ceramics" and search.required for search in searches
    )
    assert any(search.query == CATEGORY_DEFAULTS["drink"] for search in searches)
    assert all("tokyo" not in search.query.casefold() for search in searches)


def test_uncategorized_requirement_gets_its_own_search() -> None:
    searches, required = compile_searches([], [], [], [], ["record stores"])
    assert required == frozenset({"record stores"})
    assert any(
        search.category == "preference"
        and search.query == "record stores"
        and search.required
        for search in searches
    )


def test_single_shopping_and_interest_tastes_get_discovery_fallbacks() -> None:
    searches, _ = compile_searches(
        ["bookstores"], ["bistros"], ["coffee"], ["architecture"], []
    )
    assert [search.query for search in searches if search.category == "shopping"] == [
        "bookstores",
        CATEGORY_DEFAULTS["shopping"],
    ]
    assert [search.query for search in searches if search.category == "interest"] == [
        "architecture",
        CATEGORY_DEFAULTS["interest"],
    ]


def test_route_optimizer_preserves_required_preference() -> None:
    categories = ("shopping", "food", "drink", "interest", "shopping", "drink")
    places = tuple(
        _place(str(index), 48.85, 2.34 + index * 0.001, category)
        for index, category in enumerate(categories)
    )
    places[5].required_preferences.add("specialty tea")
    matrix = [[abs(first - second) * 500 for second in range(6)] for first in range(6)]
    for index in range(5):
        matrix[index][5] = 3000
        matrix[5][index] = 3000
    ordered, _ = optimize_path_with_route_matrix(
        places, matrix, required_preferences=frozenset({"specialty tea"})
    )
    assert "5" in {place.place_id for place in ordered}


def test_adaptive_radius_supports_more_spread_out_venues() -> None:
    categories = ("shopping", "food", "drink", "interest", "shopping", "food")
    places = [
        _place(str(index), 34.05, -118.25 + index * 0.004, category)
        for index, category in enumerate(categories)
    ]
    candidates = build_neighborhood_candidates(places, city_name="Los Angeles")
    assert candidates
    assert candidates[0].radius_km > 0.9


def test_city_viewport_rejects_out_of_city_anchor() -> None:
    city = CityResolution(
        place_id="paris",
        name="Paris",
        formatted_address="Paris, France",
        latitude=48.8566,
        longitude=2.3522,
        viewport_low=(48.80, 2.20),
        viewport_high=(48.92, 2.48),
    )
    assert location_in_city_viewport(48.86, 2.35, city)
    assert not location_in_city_viewport(51.50, -0.12, city)


def test_round_trip_requires_start_anchor() -> None:
    result = asyncio.run(generate_paths(city="Tokyo, Japan", round_trip=True))
    assert result == {
        "status": "insufficient_data",
        "reason": "A start anchor is required for a round trip.",
    }


def test_round_trip_rejects_different_end_anchor() -> None:
    result = asyncio.run(
        generate_paths(
            city="Tokyo, Japan",
            start_anchor="Shibuya Station",
            end_anchor="Tokyo Station",
            round_trip=True,
        )
    )
    assert result == {
        "status": "insufficient_data",
        "reason": "A round trip cannot use a different end anchor.",
    }


def test_eval_gate_checks_required_preferences_and_route_shape() -> None:
    result = {
        "status": "ok",
        "attribution": "Google Maps",
        "anchors": {"start": None, "end": None, "roundTrip": False},
        "requiredPreferences": ["ceramics"],
        "paths": [
            {
                "walkingDistanceMeters": 3000,
                "routeUrl": "https://www.google.com/maps/dir/",
                "stops": [
                    {
                        "placeId": f"{path_index}-{stop_index}",
                        "googleMapsUri": "https://maps.google.com/place",
                        "requiredPreferences": ["ceramics"] if stop_index == 0 else [],
                    }
                    for stop_index in range(5)
                ],
            }
            for path_index in range(3)
        ],
    }
    valid, _ = validate_success_result(result)
    assert valid
    result["paths"][2]["stops"][0]["requiredPreferences"] = []
    valid, reason = validate_success_result(result)
    assert not valid
    assert "required preference" in reason


def test_eval_gate_checks_anchor_route_coordinates() -> None:
    anchor = {
        "placeId": "anchor",
        "latitude": 35.658,
        "longitude": 139.7016,
    }
    result = {
        "status": "ok",
        "attribution": "Google Maps",
        "anchors": {"start": anchor, "end": None, "roundTrip": False},
        "requiredPreferences": [],
        "paths": [
            {
                "walkingDistanceMeters": 3000,
                "routeUrl": (
                    "https://www.google.com/maps/dir/?api=1&travelmode=walking"
                    "&origin=35.658000,139.701600&destination=35.680000,139.720000"
                ),
                "anchors": {"start": anchor, "end": None, "roundTrip": False},
                "stops": [
                    {
                        "placeId": f"{path_index}-{stop_index}",
                        "googleMapsUri": "https://maps.google.com/place",
                        "requiredPreferences": [],
                    }
                    for stop_index in range(5)
                ],
            }
            for path_index in range(3)
        ],
    }
    valid, _ = validate_success_result(result)
    assert valid
    result["paths"][0]["routeUrl"] = result["paths"][0]["routeUrl"].replace(
        "35.658000", "35.659000"
    )
    valid, reason = validate_success_result(result)
    assert not valid
    assert "start anchor" in reason
