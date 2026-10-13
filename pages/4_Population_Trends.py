"""Page 4 - All-cause readmission trends by county and population group."""
import streamlit as st

from src import config, dashboard_data as dd, ui, visualizations as viz
from src.theme import CODING_COLORS_DARK, SERIES_DARK

state = dd.artifact_csv("population/statewide.csv")
county = dd.artifact_csv("population/county.csv")
strata = dd.artifact_csv("population/strata.csv")
pop = dd.artifact_json("population/summary.json")

ui.page_header(
    "Population-level outcomes",
    "Population & County Trends",
    "All-cause unplanned 30-day readmissions for California residents, 2011-2024. County figures are available for "
    "all patients combined; age, sex, race/ethnicity and expected-payer breakdowns are published at state level only. "
    "Differences between groups are descriptive associations and do not establish causes.",
    chips=[("Counties", str(pop["n_counties"])), ("ICD-9 years", f"{pop['icd9_years'][0]}-{pop['icd9_years'][-1]}"),
           ("ICD-10 years", f"{pop['icd10_years'][0]}-{pop['icd10_years'][-1]}")])

years = sorted(state["year"].unique())
with ui.filter_bar("pop"):
    c = st.columns([1, 1.1, 1.2, 2.2])
    year = c[0].selectbox("Year", years, index=len(years) - 1)
    min_admits = c[1].number_input("Min. admissions to rank a county", 0, 50000, config.MIN_COUNTY_ADMITS_DEFAULT, 250,
                                   help="Small counties (e.g. Alpine, ~10 admissions/year) produce unstable rates.")
    category = c[2].selectbox("Population group", ["Expected Payer", "Age", "Race-Ethnicity", "Sex"])
    big = county[(county["year"] == year) & (county["admits"] >= 20000)].sort_values("admits", ascending=False)
    picks = c[3].multiselect("Counties for the trend chart (max 6)", sorted(county["county"].unique()),
                             default=big["county"].head(3).tolist(), max_selections=6)

# Stable colour per county: a county keeps its slot while it stays selected (colour follows the entity).
slots = st.session_state.setdefault("county_color_slots", {})
for cty in list(slots):
    if cty not in picks:
        slots.pop(cty)
for cty in picks:
    if cty not in slots:
        free = [i for i in range(6) if i not in slots.values()]
        slots[cty] = free[0]
color_map = {cty: SERIES_DARK[i] for cty, i in slots.items()}

s_row = state[state["year"] == year].iloc[0]
prev = state[state["year"] == year - 1]
same_era = not prev.empty and prev.iloc[0]["coding_system"] == s_row["coding_system"]
cy = county[county["year"] == year]
ranked = cy[(cy["admits"] >= min_admits) & cy["rate"].notna()]

k = st.columns(5)
with k[0]:
    d = s_row["rate"] - prev.iloc[0]["rate"] if same_era else None
    ui.kpi(f"California readmission rate, {year}", f"{s_row['rate']:.2f}", "%",
           f"{d:+.2f} pp vs {year - 1}" if d is not None else None,
           ("up" if d > 0 else "down") if d is not None else "flat",
           f"{s_row['coding_system']}" + (" - partial year" if s_row["coding_transition_year"] else "")
           + ("" if same_era or prev.empty else " - previous year used a different coding system"),
           CODING_COLORS_DARK[s_row["coding_system"]])
with k[1]:
    ui.kpi("Index admissions", f"{s_row['admits'] / 1e6:.2f}", "M", None, "flat", f"Statewide, {year}", "#7D8BA3")
with k[2]:
    ui.kpi("30-day readmissions", f"{s_row['readmits'] / 1e3:,.0f}", "k", None, "flat", f"Statewide, {year}", SERIES_DARK[0])
with k[3]:
    ui.kpi("Counties ranked", f"{len(ranked)}", f" of {cy['county'].nunique()}", None, "flat",
           f"With at least {min_admits:,} admissions and a published rate", "#7D8BA3")
with k[4]:
    if not ranked.empty:
        hi = ranked.loc[ranked["rate"].idxmax()]
        ui.kpi("Highest county rate", f"{hi['rate']:.2f}", "%", None, "flat", f"{hi['county']} - {hi['admits']:,.0f} admissions",
               SERIES_DARK[3])

st.write("")
left, right = st.columns([1.15, 1])
with left:
    with ui.card("pop-map"):
        ui.section(f"30-day readmission rate by county, {year}",
                   f"Grey counties have fewer than {min_admits:,} admissions (or a suppressed rate) and are not shaded. "
                   "County boundaries: US Census cartographic files.")
        ui.plot(viz.county_choropleth(county, year, min_admits), key="pop_map")
with right:
    with ui.card("pop-rank"):
        ui.section(f"County ranking, {year}", "Amber = counties selected for the trend chart. Dotted line = California.")
        if ranked.empty:
            st.info("No county meets the admissions threshold.")
        else:
            ui.plot(viz.county_bars(ranked, "rate", "county", "30-day readmission rate (%)", float(s_row["rate"]), "California",
                                    set(picks), height=max(420, 60 + 15 * len(ranked)),
                                    hover_extra="<br>%{customdata[0]:,.0f} admissions"), key="pop_rank")

with ui.card("pop-trends"):
    ui.section("County trends vs California",
               "White = California. The dotted vertical line marks the ICD-9 to ICD-10 coding change; compare trends "
               "within an era rather than across it.")
    if picks:
        ui.plot(viz.county_trends(county, state, picks, color_map), key="pop_ctrend")
    else:
        st.info("Select one or more counties above.")

group = strata[strata["strata"] == category]
left, right = st.columns(2)
with left:
    with ui.card("pop-strata"):
        ui.section(f"Readmission rate by {category.lower()}, {year}", "Statewide; dotted line = all patients.")
        ui.plot(viz.strata_bars(group, year, float(s_row["rate"])), key="pop_strata")
with right:
    with ui.card("pop-counts"):
        ui.section(f"Admissions and readmissions by {category.lower()}, {year}",
                   "Each bar is the number of index admissions; the cyan part was readmitted within 30 days.")
        ui.plot(viz.readmitted_share_bars(group, year), key="pop_counts")

with ui.card("pop-small"):
    ui.section(f"{category} trends, 2011-2024",
               "Small multiples share one y-axis. Violet = ICD-9 era, cyan = ICD-10 era; 2015 is a partial year.")
    ui.plot(viz.strata_small_multiples(group, height=260 if group["strata_name"].nunique() <= 4 else 420), key="pop_small")

with ui.card("pop-notes"):
    ui.section("Data notes and limitations")
    notes = [
        f"<b>Coding systems.</b> {pop['icd9_years'][0]}-{pop['transition_year']} use ICD-9 and "
        f"{pop['icd10_first_year']}-{pop['latest_year']} use ICD-10. Diagnosis coding changes can shift case definitions, "
        "so rates across the two eras are shown as separate series and not compared directly.",
        f"<b>{pop['transition_year']} is a partial year.</b> Statewide admissions fall by about a quarter versus the prior year, "
        "consistent with ICD-9 coding ending on 1 October 2015; no ICD-10 rows exist for that year in this file.",
        f"<b>2020.</b> Statewide admissions fell {abs(pop.get('admits_change_2019_2020_pct', 0)):.1f}% from 2019, coinciding "
        "with the COVID-19 pandemic; the rate itself changed less.",
        "<b>Small counties.</b> Some counties have only a few dozen admissions per year, so a handful of readmissions "
        "moves the rate by several points; they are excluded from rankings by the threshold above.",
        "<b>Missing values.</b> Suppressed values (e.g. Alpine 2017 readmissions) remain missing - they are never set to zero.",
        "<b>Categories.</b> 'Multiracial' appears only from 2019; 'Unknown County' appears only in 2019.",
        "<b>Interpretation.</b> Group differences describe who is readmitted; they do not show that a characteristic causes readmission.",
    ]
    st.markdown("".join(f"<div class='section-caption' style='font-size:0.86rem'>&bull; {n}</div>" for n in notes),
                unsafe_allow_html=True)
ui.footer()
