"""Deterministic venue selection and walking-path planning."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from itertools import combinations, pairwise
from urllib.parse import urlencode

EARTH_RADIUS_KM = 6371.0088
TARGET_STOPS = 6
NEIGHBORHOOD_RADIUS_KM = 1.6


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
    search_rank: int = 999

    def merge_match(self, category: str, query: str) -> None:
        self.categories.add(category)
        self.matched_queries.add(query)


@dataclass(frozen=True)
class NeighborhoodCandidate:
    name: str
    center_latitude: float
    center_longitude: float
    places: tuple[Place, ...]
    approximate_distance_meters: int
    score: float


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


def _select_balanced_places(pool: list[Place], seed: Place) -> tuple[Place, ...]:
    selected: list[Place] = []
    selected_ids: set[str] = set()
    for group in ({"shopping"}, {"food", "drink"}, {"interest"}):
        matches = [place for place in pool if place.categories & group]
        if not matches:
            return ()
        choice = max(matches, key=lambda place: (_local_score(place, seed), place.place_id))
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


def _neighborhood_name(places: tuple[Place, ...]) -> str:
    labels: dict[str, int] = {}
    for place in places:
        for match in re.findall(
            r"\b([A-Za-z][A-Za-z .'-]{1,35}(?:City|Ward))\b", place.address
        ):
            label = match.strip()
            labels[label] = labels.get(label, 0) + 1
    if labels:
        return max(labels, key=lambda label: (labels[label], -len(label), label))
    return f"Around {places[0].name}"


def build_neighborhood_candidates(places: list[Place]) -> list[NeighborhoodCandidate]:
    """Build compact, category-balanced candidate paths around venue seeds."""

    candidates: list[NeighborhoodCandidate] = []
    for seed in sorted(places, key=lambda place: place.place_id):
        pool = [
            place
            for place in places
            if haversine_km(seed, place) <= NEIGHBORHOOD_RADIUS_KM
        ]
        balanced = _select_balanced_places(pool, seed)
        if not balanced:
            continue
        ordered = _nearest_neighbor_order(balanced)
        distance_meters = path_distance_meters(ordered)
        if distance_meters > 5000:
            continue
        category_count = len(set().union(*(place.categories for place in ordered)))
        target_penalty = abs(distance_meters - 3000) / 1000
        popularity = sum(_popularity(place) for place in ordered) / len(ordered)
        score = category_count * 15 + popularity - target_penalty * 6
        candidates.append(
            NeighborhoodCandidate(
                name=_neighborhood_name(ordered),
                center_latitude=seed.latitude,
                center_longitude=seed.longitude,
                places=ordered,
                approximate_distance_meters=distance_meters,
                score=score,
            )
        )

    candidates.sort(key=lambda candidate: (-candidate.score, candidate.name))
    selected: list[NeighborhoodCandidate] = []
    for candidate in candidates:
        candidate_ids = {place.place_id for place in candidate.places}
        distinct = True
        for existing in selected:
            existing_ids = {place.place_id for place in existing.places}
            overlap = len(candidate_ids & existing_ids) / len(candidate_ids | existing_ids)
            center_a = Place(
                "", "", "", candidate.center_latitude, candidate.center_longitude, ""
            )
            center_b = Place(
                "", "", "", existing.center_latitude, existing.center_longitude, ""
            )
            if overlap > 0 or haversine_km(center_a, center_b) < 1.8:
                distinct = False
                break
        if distinct:
            selected.append(candidate)
        if len(selected) == 3:
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
    places: tuple[Place, ...], distances: list[list[int]]
) -> tuple[tuple[Place, ...], int]:
    """Choose a balanced five- or six-stop subset closest to the 3 km target."""

    if len(distances) != len(places) or any(
        len(row) != len(places) for row in distances
    ):
        raise ValueError("Route matrix dimensions do not match places")

    best: tuple[tuple[int, int, int], tuple[Place, ...], int] | None = None
    for size in range(min(TARGET_STOPS, len(places)), 4, -1):
        for indices in combinations(range(len(places)), size):
            subset = tuple(places[index] for index in indices)
            covered = set().union(*(place.categories for place in subset))
            if not {"shopping", "interest"}.issubset(covered) or not (
                {"food", "drink"} & covered
            ):
                continue
            subset_matrix = [
                [distances[first][second] for second in indices] for first in indices
            ]
            ordered = order_with_route_matrix(subset, subset_matrix)
            original_index = {place.place_id: index for index, place in enumerate(places)}
            distance = sum(
                distances[original_index[first.place_id]][original_index[second.place_id]]
                for first, second in pairwise(ordered)
                if distances[original_index[first.place_id]][original_index[second.place_id]]
                >= 0
            )
            outside_target = 0 if 2000 <= distance <= 4000 else 1
            score = (outside_target, abs(distance - 3000), -size)
            if best is None or score < best[0]:
                best = (score, ordered, distance)
    if best is None:
        ordered = order_with_route_matrix(places, distances)
        original_index = {place.place_id: index for index, place in enumerate(places)}
        distance = sum(
            distances[original_index[first.place_id]][original_index[second.place_id]]
            for first, second in pairwise(ordered)
            if distances[original_index[first.place_id]][original_index[second.place_id]] >= 0
        )
        return ordered, distance
    return best[1], best[2]


def build_route_url(places: tuple[Place, ...]) -> str:
    """Build a cross-platform Google Maps walking directions URL."""

    if len(places) < 2:
        return places[0].google_maps_uri if places else ""
    coordinates = [f"{place.latitude:.6f},{place.longitude:.6f}" for place in places]
    parameters = {
        "api": "1",
        "travelmode": "walking",
        "origin": coordinates[0],
        "destination": coordinates[-1],
    }
    if len(coordinates) > 2:
        parameters["waypoints"] = "|".join(coordinates[1:-1])
    return f"https://www.google.com/maps/dir/?{urlencode(parameters, safe='|,')}"
