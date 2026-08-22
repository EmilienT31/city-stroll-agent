"""Run the City Stroll tool without invoking the language model."""

from __future__ import annotations

import argparse
import asyncio
import json

from app.city_stroll import generate_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--shopping", action="append", default=[])
    parser.add_argument("--food", action="append", default=[])
    parser.add_argument("--drink", action="append", default=[])
    parser.add_argument("--interest", action="append", default=[])
    parser.add_argument("--required", action="append", default=[])
    parser.add_argument("--avoid", action="append", default=[])
    parser.add_argument("--start-anchor", default="")
    parser.add_argument("--end-anchor", default="")
    parser.add_argument("--round-trip", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(
        generate_paths(
            city=args.city,
            shopping_preferences=args.shopping,
            food_preferences=args.food,
            drink_preferences=args.drink,
            interest_preferences=args.interest,
            required_preferences=args.required,
            avoidances=args.avoid,
            start_anchor=args.start_anchor,
            end_anchor=args.end_anchor,
            round_trip=args.round_trip,
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
