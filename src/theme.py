"""Design tokens shared by the dashboard (dark navy) and report figures (light).

Categorical palettes were checked with a colour-vision-deficiency validator (OKLab
distance under protan/deutan simulation, lightness band, chroma floor and contrast):
* dark palette on surface #101B31 - all checks pass; first three slots pass all-pairs;
* light palette on white - passes; amber is below 3:1 contrast, so it is always paired
  with direct labels or a table.
Colours are assigned to *entities* (a model, a measure, a coding system) and never by rank.
"""

# ---------------- dark dashboard ----------------
DARK = {
    "page": "#0B1324",
    "surface": "#101B31",
    "surface_2": "#15233F",
    "border": "rgba(148, 163, 184, 0.16)",
    "grid": "#1D2A44",
    "axis": "#2A3A5A",
    "text": "#E6EDF7",
    "text_2": "#A9B6CC",
    "muted": "#7D8BA3",
    "accent": "#22D3EE",
}
SERIES_DARK = ["#119fbf", "#d06a1e", "#8a7ff0", "#c98500", "#d55181", "#3987e5"]
SEQUENTIAL_DARK = ["#13294b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"]
DIVERGING_DARK = {"neg": "#3987e5", "mid": "#3a4660", "pos": "#d06a1e"}

# ---------------- light report ----------------
LIGHT = {"surface": "#ffffff", "grid": "#e4e7ec", "axis": "#c3c8d2", "text": "#0f172a",
         "text_2": "#475569", "muted": "#8a94a6", "navy": "#0b2545", "accent": "#0e7fa0"}
SERIES_LIGHT = ["#0e7fa0", "#d9661f", "#5b4fc7", "#c98500", "#c2416f", "#2a78d6"]

# ---------------- status (fixed meaning, always shown with a text label) ----------------
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b",
          "neutral": "#8A97AD"}
RATING_COLORS = {
    "Better": STATUS["good"], "Worse": STATUS["critical"],
    "Average": STATUS["neutral"], "As Expected": STATUS["neutral"], "Not Applicable": "#4B5872",
}

# ---------------- entity -> colour (same entity, same colour, everywhere) ----------------
MODEL_COLORS_DARK = {
    "linear_regression": SERIES_DARK[0], "random_forest": SERIES_DARK[1],
    "persistence": "#A9B6CC", "mean_baseline": "#5B6880", "actual": "#E6EDF7",
}
MODEL_COLORS_LIGHT = {
    "linear_regression": SERIES_LIGHT[0], "random_forest": SERIES_LIGHT[1],
    "persistence": "#475569", "mean_baseline": "#94a3b8", "actual": "#0f172a",
}
MODEL_SYMBOLS = {"linear_regression": "diamond", "random_forest": "square", "persistence": "circle-open",
                 "mean_baseline": "x"}
CODING_COLORS_DARK = {"ICD-9": SERIES_DARK[2], "ICD-10": SERIES_DARK[0]}
CODING_COLORS_LIGHT = {"ICD-9": SERIES_LIGHT[2], "ICD-10": SERIES_LIGHT[0]}
MEASURE_COLORS_DARK = {"30-day Readmission": SERIES_DARK[0], "30-day Mortality": SERIES_DARK[1]}
MEASURE_COLORS_LIGHT = {"30-day Readmission": SERIES_LIGHT[0], "30-day Mortality": SERIES_LIGHT[1]}

FONT_FAMILY = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"
