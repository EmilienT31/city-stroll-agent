"""Deterministic venue selection and walking-path planning."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from itertools import combinations, pairwise, permutations
from urllib.parse import urlencode

EARTH_RADIUS_KM = 6371.0088
TARGET_STOPS = 6
NEIGHBORHOOD_RADII_KM = (0.9, 1.3, 1.8, 2.4)


@dataclass
class Place:
    """A Google Places result plus local planner annotations."""

    place_id: str
    name: str
    address: str
    latitude: float
    longitude: float
    google_maps_uri: str
    rating: float | None = None
    user_rating_count: int = 0
    categories: set[str] = field(default_factory=set)
    matched_queries: set[str] = field(default_factory=set)
    required_preferences: set[str] = field(default_factory=set)
    search_rank: int = 999

    def merge_match(self, category: str, query: str, *, required: bool = False) -> None:
        self.categories.add(category)
        self.matched_queries.add(query)
        if required:
            self.required_preferences.add(query)


@dataclass(frozen=True)
class NeighborhoodCandidate:
    name: str
    center_latitude: float
    center_longitude: float
    places: tuple[Place, ...]
    approximate_distance_meters: int
    score: float
    radius_km: float


def haversine_km(first: Place, second: Place) -> float:
    """Return the great-circle distance between two places."""

    lat1 = math.radians(first.latitude)
    lat2 = math.radians(second.latitude)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(second.longitude - first.longitude)
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(value))


def _popularity(place: Place) -> float:
    rating = place.rating or 0.0
    return rating * 8 + math.log1p(place.user_rating_count)


def _nearest_neighbor_order(
    places: tuple[Place, ...], distances: list[list[int]] | None = None
) -> tuple[Place, ...]:
    if not places:
        return ()

    start = min(
        range(len(places)),
        key=lambda index: (places[index].longitude, places[index].latitude),
    )
    remaining = set(range(len(places)))
    remaining.remove(start)
    order = [start]

    def distance(first_index: int, second_index: int) -> float:
        if distances is not None:
            value = distances[first_index][second_index]
            if value >= 0:
                return float(value)
        return haversine_km(places[first_index], places[second_index]) * 1000

    while remaining:
        current = order[-1]
        next_index = min(
            remaining,
            key=lambda index: (distance(current, index), places[index].place_id),
        )
        order.append(next_index)
        remaining.remove(next_index)

    return tuple(places[index] for index in order)


def path_distance_meters(
    ordered_places: tuple[Place, ...], distances: list[list[int]] | None = None
) -> int:
    if len(ordered_places) < 2:
        return 0
    if distances is None:
        return round(
            sum(
                haversine_km(first, second)
                for first, second in pairwise(ordered_places)
            )
            * 1000
        )
    return sum(
        distances[index][index + 1]
        for index in range(len(ordered_places) - 1)
        if distances[index][index + 1] >= 0
    )


def _local_score(place: Place, seed: Place) -> float:
    relevance = max(0, 20 - place.search_rank) * 3
    distance_penalty = haversine_km(seed, place) * 18
    return _popularity(place) + relevance - distance_penalty


def _brand_key(place: Place) -> str:
    tokens = re.findall(r"[a-z0-9]+", place.name.casefold())
    return tokens[0] if tokens else place.name.casefold()[:16]


def _select_balanced_places(
    pool: list[Place], seed: Place, required_preferences: frozenset[str]
) -> tuple[Place, ...]:
    selected: list[Place] = []
    selected_ids: set[str] = set()
    for preference in sorted(required_preferences):
        matches = [place for place in pool if preference in place.required_preferences]
        if not matches:
            return ()
        choice = max(
            matches, key=lambda place: (_local_score(place, seed), place.place_id)
        )
        if choice.place_id not in selected_ids:
            selected.append(choice)
            selected_ids.add(choice.place_id)

    for group in ({"shopping"}, {"food", "drink"}, {"interest"}):
        matches = [place for place in pool if place.categories & group]
        if not matches:
            return ()
        choice = max(
            matches, key=lambda place: (_local_score(place, seed), place.place_id)
        )
        if choice.place_id not in selected_ids:
            selected.append(choice)
            selected_ids.add(choice.place_id)

    while len(selected) < min(TARGET_STOPS, len(pool)):
        selected_brands = {_brand_key(place) for place in selected}
        remaining = [
            place
            for place in pool
            if place.place_id not in selected_ids
            and _brand_key(place) not in selected_brands
        ]
        if not remaining:
            break
        covered = set().union(*(place.categories for place in selected))
        choice = max(
            remaining,
            key=lambda place: (
                len(place.categories - covered) * 20 + _local_score(place, seed),
                place.place_id,
            ),
        )
        selected.append(choice)
        selected_ids.add(choice.place_id)

    return tuple(selected) if len(selected) >= 5 else ()


def _neighborhood_name(places: tuple[Place, ...], city_name: str) -> str:
    labels: dict[str, int] = {}
    for place in places:
        for match in re.findall(
            r"\b([^,]{2,40}(?:City|Ward|Borough|District|Arrondissement))\b",
            place.address,
            flags=re.IGNORECASE,
        ):
            label = match.strip()
            if label.casefold() == city_name.casefold():
                continue
            labels[label] = labels.get(label, 0) + 1
    if labels:
        return max(labels, key=lambda label: (labels[label], -len(label), label))
    return f"Around {places[0].name}"


def build_neighborhood_candidates(
    places: list[Place],
    required_preferences: frozenset[str] = frozenset(),
    city_name: str = "",
    anchor_points: tuple[Place, ...] = (),
    max_candidates: int = 3,
) -> list[NeighborhoodCandidate]:
    """Build compact, category-balanced candidate paths around venue seeds."""

    candidates: list[NeighborhoodCandidate] = []
    for seed in sorted(places, key=lambda place: place.place_id):
        balanced: tuple[Place, ...] = ()
        selected_radius = NEIGHBORHOOD_RADII_KM[-1]
        for radius_km in NEIGHBORHOOD_RADII_KM:
            pool = [place for place in places if haversine_km(seed, place) <= radius_km]
            balanced = _select_balanced_places(pool, seed, required_preferences)
            if balanced:
                selected_radius = radius_km
                break
        if not balanced:
            continue
        ordered = _nearest_neighbor_order(balanced)
        distance_meters = path_distance_meters(ordered)
        if distance_meters > 5000:
            continue
        category_count = len(set().union(*(place.categories for place in ordered)))
        target_penalty = abs(distance_meters - 3000) / 1000
        anchor_penalty = sum(haversine_km(seed, anchor) for anchor in anchor_points)
        popularity = sum(_popularity(place) for place in ordered) / len(ordered)
        score = (
            category_count * 15 + popularity - target_penalty * 6 - anchor_penalty * 12
        )
        candidates.append(
            NeighborhoodCandidate(
                name=_neighborhood_name(ordered, city_name),
                center_latitude=seed.latitude,
                center_longitude=seed.longitude,
                places=ordered,
                approximate_distance_meters=distance_meters,
                score=score,
                radius_km=selected_radius,
            )
        )

    candidates.sort(key=lambda candidate: (-candidate.score, candidate.name))
    if anchor_points:
        selected: list[NeighborhoodCandidate] = []
        seen_place_sets: set[frozenset[str]] = set()
        for candidate in candidates:
            candidate_ids = frozenset(place.place_id for place in candidate.places)
            if candidate_ids in seen_place_sets:
                continue
            seen_place_sets.add(candidate_ids)
            selected.append(candidate)
            if len(selected) == max_candidates:
                break
        return selected

    selected: list[NeighborhoodCandidate] = []
    for candidate in candidates:
        candidate_ids = {place.place_id for place in candidate.places}
        distinct = True
        for existing in selected:
            existing_ids = {place.place_id for place in existing.places}
            overlap = len(candidate_ids & existing_ids) / len(
                candidate_ids | existing_ids
            )
            center_a = Place(
                "", "", "", candidate.center_latitude, candidate.center_longitude, ""
            )
            center_b = Place(
                "", "", "", existing.center_latitude, existing.center_longitude, ""
            )
            minimum_separation = max(1.2, min(candidate.radius_km, existing.radius_km))
            if overlap > 0 or haversine_km(center_a, center_b) < minimum_separation:
                distinct = False
                break
        if distinct:
            selected.append(candidate)
        if len(selected) == max_candidates:
            break
    return selected


def order_with_route_matrix(
    places: tuple[Place, ...], distances: list[list[int]]
) -> tuple[Place, ...]:
    """Order places using the Routes API matrix with geographic fallback."""

    if len(distances) != len(places) or any(
        len(row) != len(places) for row in distances
    ):
        raise ValueError("Route matrix dimensions do not match places")
    return _nearest_neighbor_order(places, distances)


def optimize_path_with_route_matrix(
    places: tuple[Place, ...],
    distances: list[list[int]],
    required_preferences: frozenset[str] = frozenset(),
    start_anchor_index: int | None = None,
    end_anchor_index: int | None = None,
    excluded_place_ids: frozenset[str] = frozenset(),
) -> tuple[tuple[Place, ...], int]:
    """Choose and order venues, honoring optional fixed route endpoints."""

    matrix_size = len(distances)
    if matrix_size < len(places) or any(len(row) != matrix_size for row in distances):
        raise ValueError("Route matrix dimensions do not match route nodes")
    for anchor_index in (start_anchor_index, end_anchor_index):
        if anchor_index is not None and not 0 <= anchor_index < matrix_size:
            raise ValueError("Anchor index is outside the route matrix")

    best: tuple[tuple[int, int, int], tuple[Place, ...], int] | None = None
    for size in range(min(TARGET_STOPS, len(places)), 4, -1):
        for indices in combinations(range(len(places)), size):
            subset = tuple(places[index] for index in indices)
            if any(place.place_id in excluded_place_ids for place in subset):
                continue
            covered = set().union(*(place.categories for place in subset))
            if not {"shopping", "interest"}.issubset(covered) or not (
                {"food", "drink"} & covered
            ):
                continue
            covered_requirements = set().union(
                *(place.required_preferences for place in subset)
            )
            if not required_preferences.issubset(covered_requirements):
                continue
            for order in permutations(indices):
                route_indices = list(order)
                if start_anchor_index is not None:
                    route_indices.insert(0, start_anchor_index)
                if end_anchor_index is not None:
                    route_indices.append(end_anchor_index)
                legs = [
                    distances[first][second]
                    for first, second in pairwise(route_indices)
                ]
                if any(distance < 0 for distance in legs):
                    continue
                distance = sum(legs)
                outside_target = 0 if 2000 <= distance <= 4000 else 1
                score = (outside_target, abs(distance - 3000), -size)
                ordered = tuple(places[index] for index in order)
                if best is None or score < best[0]:
                    best = (score, ordered, distance)
    if best is None:
        return (), -1
    return best[1], best[2]


def build_route_url(
    places: tuple[Place, ...],
    start_anchor: Place | None = None,
    end_anchor: Place | None = None,
) -> str:
    """Build a cross-platform Google Maps walking directions URL."""

    route_points = list(places)
    if start_anchor is not None:
        route_points.insert(0, start_anchor)
    if end_anchor is not None:
        route_points.append(end_anchor)
    if len(route_points) < 2:
        return route_points[0].google_maps_uri if route_points else ""
    coordinates = [
        f"{place.latitude:.6f},{place.longitude:.6f}" for place in route_points
    ]
    parameters = {
        "api": "1",
        "travelmode": "walking",
        "origin": coordinates[0],
        "destination": coordinates[-1],
    }
    if len(coordinates) > 2:
        parameters["waypoints"] = "|".join(coordinates[1:-1])
    return f"https://www.google.com/maps/dir/?{urlencode(parameters, safe='|,')}"
