"""Run the City Stroll tool without invoking the language model."""

from __future__ import annotations

import argparse
import asyncio
import json

from app.city_stroll import generate_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--tastes", required=True)
    parser.add_argument("--avoidances", default="")
    args = parser.parse_args()
    result = asyncio.run(generate_paths(args.city, args.tastes, args.avoidances))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
