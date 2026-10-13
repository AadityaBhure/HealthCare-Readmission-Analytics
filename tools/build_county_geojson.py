"""One-off helper that produced data/geo/california_counties.geojson.

Source: Plotly open datasets (MIT licence),
https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json
which is derived from US Census Bureau cartographic boundary files (public domain).
Keeps California (STATE FIPS 06) only, rounds coordinates to 4 decimals (~10 m) to keep
the file small, and stores the county name in properties.NAME.

Usage:  python tools/build_county_geojson.py <path-to-downloaded-geojson>
"""
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "geo" / "california_counties.geojson"


def _round(coords):
    if isinstance(coords[0], (int, float)):
        return [round(coords[0], 4), round(coords[1], 4)]
    return [_round(c) for c in coords]


def main(src: str) -> None:
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    feats = []
    for f in data["features"]:
        if f["properties"].get("STATE") != "06":
            continue
        feats.append({"type": "Feature", "id": f["id"],
                      "properties": {"NAME": f["properties"]["NAME"], "FIPS": f["id"]},
                      "geometry": {"type": f["geometry"]["type"], "coordinates": _round(f["geometry"]["coordinates"])}})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")),
                   encoding="utf-8")
    print(f"wrote {len(feats)} California counties to {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv[1])
