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
from app.stroll_planner import (
    Place,
    build_neighborhood_candidates,
    build_route_url,
    haversine_km,
    optimize_path_with_route_matrix,
    order_with_route_matrix,
)


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


def test_route_matrix_controls_order() -> None:
    places = tuple(
        _place(str(index), 35.65, 139.70 + index * 0.001, "food")
        for index in range(3)
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
