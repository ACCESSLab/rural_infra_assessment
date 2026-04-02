#!/usr/bin/env python3
"""Compile readiness evaluation artifacts into figures, summaries, and a PDF.

This module is the reporting endpoint for the readiness pipeline. It loads
frame-level evaluation JSON files, derives segment summaries and issue labels,
creates visualizations, and renders a research-style PDF report that can be
shared outside the codebase.
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import markdown2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from openai import OpenAI
from weasyprint import CSS, HTML

# ===============================
# Configuration
# ===============================

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
EVAL_FOLDER = RESULTS_DIR / "evaluated"
FIGURE_FOLDER = RESULTS_DIR / "figures"
IMAGE_CANDIDATE_DIRS = [
    RESULTS_DIR / "readiness" / "keypoint_overlays",
    RESULTS_DIR / "lane" / "images",
]
SEGMENT_SIZE_MILES = 1.0

FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)


def _configure_results_paths(results_dir: Path) -> None:
    """Retarget all report inputs/outputs to a caller-provided results directory."""
    global RESULTS_DIR, EVAL_FOLDER, FIGURE_FOLDER, IMAGE_CANDIDATE_DIRS
    RESULTS_DIR = results_dir
    EVAL_FOLDER = RESULTS_DIR / "evaluated"
    FIGURE_FOLDER = RESULTS_DIR / "figures"
    IMAGE_CANDIDATE_DIRS = [
        RESULTS_DIR / "readiness" / "keypoint_overlays",
        RESULTS_DIR / "lane" / "images",
    ]
    FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)


# ===============================
# Utilities
# ===============================

def _safe_float(v: Any, default: float = 0.0) -> float:
    """Convert a loosely typed value to ``float`` while preserving a safe default."""
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in miles between two GPS coordinates."""
    # Earth's radius in miles.
    r = 3958.7613
    import math

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(d_lambda / 2.0) ** 2)
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return r * c



def _resolve_image_path(image_name: Optional[str]) -> Optional[str]:
    """Resolve an evidence image name against the known readiness image folders."""
    if not image_name:
        return None
    for image_dir in IMAGE_CANDIDATE_DIRS:
        p = image_dir / image_name
        if p.exists():
            return str(p)
    return None


def _state_name_from_fips(state_fips: Optional[str]) -> str:
    """Translate a two-digit state FIPS code into a human-readable state name."""
    if not state_fips:
        return "Unknown"
    names = {
        "01": "Alabama",
        "02": "Alaska",
        "04": "Arizona",
        "05": "Arkansas",
        "06": "California",
        "08": "Colorado",
        "09": "Connecticut",
        "10": "Delaware",
        "11": "District of Columbia",
        "12": "Florida",
        "13": "Georgia",
        "15": "Hawaii",
        "16": "Idaho",
        "17": "Illinois",
        "18": "Indiana",
        "19": "Iowa",
        "20": "Kansas",
        "21": "Kentucky",
        "22": "Louisiana",
        "23": "Maine",
        "24": "Maryland",
        "25": "Massachusetts",
        "26": "Michigan",
        "27": "Minnesota",
        "28": "Mississippi",
        "29": "Missouri",
        "30": "Montana",
        "31": "Nebraska",
        "32": "Nevada",
        "33": "New Hampshire",
        "34": "New Jersey",
        "35": "New Mexico",
        "36": "New York",
        "37": "North Carolina",
        "38": "North Dakota",
        "39": "Ohio",
        "40": "Oklahoma",
        "41": "Oregon",
        "42": "Pennsylvania",
        "44": "Rhode Island",
        "45": "South Carolina",
        "46": "South Dakota",
        "47": "Tennessee",
        "48": "Texas",
        "49": "Utah",
        "50": "Vermont",
        "51": "Virginia",
        "53": "Washington",
        "54": "West Virginia",
        "55": "Wisconsin",
        "56": "Wyoming",
    }
    return names.get(str(state_fips).zfill(2), "Unknown")


def _infer_location(df: pd.DataFrame) -> Dict[str, Any]:
    """Infer county/state context for the route using the GPS trace footprint."""
    out: Dict[str, Any] = {
        "county": "Unknown",
        "state": "Unknown",
        "state_fips": None,
        "county_geoid": None,
        "county_row": None,
        "counties": [],
        "county_geoids": [],
        "county_rows": [],
        "state_fips_list": [],
    }
    if df.empty:
        return out

    coords = df[["lat", "lon"]].dropna()
    if coords.empty:
        return out

    center_lat = float(coords["lat"].mean())
    center_lon = float(coords["lon"].mean())

    county_zip = REPO_ROOT / "connectivity" / "data_cache" / "counties" / "tl_2025_us_county.zip"
    if not county_zip.exists():
        return out

    try:
        import geopandas as gp
        from shapely.geometry import Point
    except Exception:
        return out

    try:
        counties = gp.read_file(county_zip.as_posix()).to_crs(4326)
        sample_step = max(1, len(coords) // 500)
        sampled = coords.iloc[::sample_step].copy()
        if not sampled.tail(1).equals(coords.tail(1)):
            sampled = pd.concat([sampled, coords.tail(1)], ignore_index=True)
        point_gdf = gp.GeoDataFrame(
            sampled,
            geometry=gp.points_from_xy(sampled["lon"].astype(float), sampled["lat"].astype(float)),
            crs=4326,
        )
        joined = gp.sjoin(
            point_gdf,
            counties[["GEOID", "NAMELSAD", "NAME", "STATEFP", "geometry"]],
            how="left",
            predicate="intersects",
        )
        count_by_geoid = joined["GEOID"].dropna().astype(str).value_counts()
        touched_rows = counties[counties["GEOID"].astype(str).isin(count_by_geoid.index)].copy()
        if touched_rows.empty:
            pt = Point(center_lon, center_lat)
            hit = counties[counties.geometry.contains(pt)]
            if hit.empty:
                nearest_idx = counties.geometry.distance(pt).idxmin()
                hit = counties.loc[[nearest_idx]]
            touched_rows = hit.copy()
            count_by_geoid = pd.Series({str(touched_rows.iloc[0].get("GEOID") or ""): 1})
        touched_rows["__count"] = touched_rows["GEOID"].astype(str).map(count_by_geoid).fillna(0)
        touched_rows = touched_rows.sort_values(["STATEFP", "__count", "NAMELSAD", "NAME"], ascending=[True, False, True, True])
        row = touched_rows.iloc[0]
        county_name = str(row.get("NAMELSAD") or row.get("NAME") or "Unknown")
        state_name = _state_name_from_fips(str(row.get("STATEFP") or ""))
        out["county"] = county_name
        out["state"] = state_name
        out["state_fips"] = str(row.get("STATEFP") or "").zfill(2)
        out["county_geoid"] = str(row.get("GEOID") or "")
        out["county_row"] = row
        out["counties"] = [
            {
                "county": str(r.get("NAMELSAD") or r.get("NAME") or "Unknown"),
                "state": _state_name_from_fips(str(r.get("STATEFP") or "")),
                "state_fips": str(r.get("STATEFP") or "").zfill(2),
                "county_geoid": str(r.get("GEOID") or ""),
            }
            for _, r in touched_rows.iterrows()
        ]
        out["county_geoids"] = [str(r.get("GEOID") or "") for _, r in touched_rows.iterrows()]
        out["county_rows"] = [r for _, r in touched_rows.iterrows()]
        out["state_fips_list"] = sorted({str(r.get("STATEFP") or "").zfill(2) for _, r in touched_rows.iterrows() if str(r.get("STATEFP") or "").strip()})
        return out
    except Exception:
        return out


def _generate_location_figures(df: pd.DataFrame, location_info: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Create contextual map figures used by the report's location section."""
    result: Dict[str, Optional[str]] = {"context_map": None, "zoom_map": None}
    if df.empty:
        return result

    pts = df[["lat", "lon", "connectivity_score", "overall_score"]].dropna(subset=["lat", "lon"]).copy()
    if pts.empty:
        return result

    lats = pts["lat"].astype(float)
    lons = pts["lon"].astype(float)

    context_path = FIGURE_FOLDER / "location_context_map.png"
    zoom_path = FIGURE_FOLDER / "location_zoom_map.png"

    # Context map: state/county view focused on the route footprint.
    fig1, ax1 = plt.subplots(figsize=(6.6, 3.6))
    rendered_state_context = False
    county_zip = REPO_ROOT / "connectivity" / "data_cache" / "counties" / "tl_2025_us_county.zip"
    state_fips = str(location_info.get("state_fips") or "")
    state_fips_list = [str(x) for x in (location_info.get("state_fips_list") or []) if str(x).strip()]
    county_geoid = str(location_info.get("county_geoid") or "")
    county_geoids = [str(x) for x in (location_info.get("county_geoids") or []) if str(x).strip()]
    if county_zip.exists():
        try:
            import geopandas as gp
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch

            counties = gp.read_file(county_zip.as_posix()).to_crs(4326)
            state_targets = state_fips_list or ([state_fips] if state_fips else [])
            state_counties = counties[counties["STATEFP"].astype(str).isin(state_targets)].copy() if state_targets else counties.iloc[0:0].copy()
            county_targets = county_geoids or ([county_geoid] if county_geoid else [])
            route_counties = counties[counties["GEOID"].astype(str).isin(county_targets)].copy() if county_targets else counties.iloc[0:0].copy()

            if not state_counties.empty:
                state_counties.plot(ax=ax1, color="#edf2f7", edgecolor="#94a3b8", linewidth=0.5)
                if not route_counties.empty:
                    route_counties.plot(ax=ax1, color="#ef9a9a", edgecolor="#b71c1c", linewidth=1.2)
                ax1.plot(lons, lats, color="#235789", linewidth=1.3, alpha=0.95)

                minx, miny, maxx, maxy = state_counties.total_bounds
                lon_pad = max(0.01, (maxx - minx) * 0.06)
                lat_pad = max(0.01, (maxy - miny) * 0.06)
                ax1.set_xlim(minx - lon_pad, maxx + lon_pad)
                ax1.set_ylim(miny - lat_pad, maxy + lat_pad)

                handles = [
                    Patch(facecolor="#edf2f7", edgecolor="#94a3b8", label="State counties"),
                    Patch(facecolor="#ef9a9a", edgecolor="#b71c1c", label="Route counties"),
                    Line2D([0], [0], color="#235789", linewidth=1.3, label="Route trace"),
                ]
                ax1.legend(handles=handles, loc="lower left", fontsize=7, frameon=True)
                rendered_state_context = True
        except Exception:
            rendered_state_context = False

    if not rendered_state_context:
        # Fallback: county boundary (if available) + route.
        county_rows = location_info.get("county_rows") or ([location_info.get("county_row")] if location_info.get("county_row") is not None else [])
        if county_rows:
            try:
                import geopandas as gp

                gp.GeoSeries([row.geometry for row in county_rows], crs=4326).plot(
                    ax=ax1, color="#ecf3fb", edgecolor="#7aa6d8", linewidth=1.2
                )
            except Exception:
                pass
        ax1.plot(lons, lats, color="#235789", linewidth=1.4, alpha=0.9)

    ax1.set_title("State/County Context")
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_aspect("equal", adjustable="box")
    fig1.tight_layout()
    fig1.savefig(context_path, dpi=260)
    plt.close(fig1)
    result["context_map"] = str(context_path)

    # Zoom map: use dynamic orientation and tight bounds so route fills the panel.
    lat_span = max(1e-9, float(lats.max()) - float(lats.min()))
    lon_span = max(1e-9, float(lons.max()) - float(lons.min()))
    is_tall_route = lat_span > (1.35 * lon_span)
    zoom_figsize = (4.6, 6.8) if is_tall_route else (6.8, 4.2)
    fig2, ax2 = plt.subplots(figsize=zoom_figsize)
    ax2.plot(lons, lats, color="#1f4e79", linewidth=2.0, alpha=0.95)
    low = pts[pts["overall_score"] < 0.6]
    if not low.empty:
        ax2.scatter(low["lon"], low["lat"], color="#c62828", s=18, alpha=0.95, label="Low score points")
    ax2.scatter(lons.iloc[0], lats.iloc[0], color="#2e7d32", s=24, label="Start")
    ax2.scatter(lons.iloc[-1], lats.iloc[-1], color="#6a1b9a", s=24, label="End")
    lon_pad = max(0.00025, lon_span * 0.03)
    lat_pad = max(0.00025, lat_span * 0.03)
    ax2.set_xlim(float(lons.min()) - lon_pad, float(lons.max()) + lon_pad)
    ax2.set_ylim(float(lats.min()) - lat_pad, float(lats.max()) + lat_pad)
    ax2.set_title("Zoomed Route Snapshot")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.grid(alpha=0.25, linewidth=0.4)
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.margins(x=0.0, y=0.0)
    ax2.legend(loc="best", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(zoom_path, dpi=260)
    plt.close(fig2)
    result["zoom_map"] = str(zoom_path)

    return result


def _build_location_markdown(df: pd.DataFrame) -> str:
    """Build the markdown fragment that describes where the route was collected."""
    info = _infer_location(df)
    maps = _generate_location_figures(df, info)
    counties = info.get("counties") or []
    if counties:
        county_names = [str(item.get("county") or "Unknown") for item in counties]
        states = [str(item.get("state") or "Unknown") for item in counties]
        unique_states = sorted({s for s in states if s and s != "Unknown"})
        if len(county_names) == 1:
            sentence = f"This experiment is conducted in {county_names[0]}, {states[0] or 'Unknown'}."
        elif len(unique_states) == 1:
            if len(county_names) == 2:
                county_text = f"{county_names[0]} and {county_names[1]}"
            else:
                county_text = ", ".join(county_names[:-1]) + f", and {county_names[-1]}"
            sentence = f"This experiment is conducted across {county_text} in {unique_states[0]}."
        else:
            county_state_pairs = [f"{c}, {s}" for c, s in zip(county_names, states)]
            if len(county_state_pairs) == 2:
                pair_text = f"{county_state_pairs[0]} and {county_state_pairs[1]}"
            else:
                pair_text = ", ".join(county_state_pairs[:-1]) + f", and {county_state_pairs[-1]}"
            sentence = f"This experiment is conducted across {pair_text}."
    else:
        county = info.get("county") or "Unknown"
        state = info.get("state") or "Unknown"
        sentence = f"This experiment is conducted in {county}, {state}."

    context_img = maps.get("context_map")
    zoom_img = maps.get("zoom_map")
    if not context_img and not zoom_img:
        return sentence

    items: List[str] = ['<div class="location-grid">']
    if context_img:
        items.append(
            "<div class=\"location-item\">"
            "<div class=\"location-caption\">U.S. State/County Context</div>"
            f"<img src=\"{context_img}\" alt=\"Location context map\" />"
            "</div>"
        )
    if zoom_img:
        items.append(
            "<div class=\"location-item\">"
            "<div class=\"location-caption\">Zoomed Route Snapshot</div>"
            f"<img src=\"{zoom_img}\" alt=\"Zoomed route map\" />"
            "</div>"
        )
    items.append("</div>")
    return sentence + "\n\n" + "\n".join(items)


def _build_weights_markdown(df: pd.DataFrame) -> str:
    """Build a markdown table describing the evaluation weights used for this run."""
    weight_keys = ["w1", "w2", "k1", "k2", "k3", "m1", "m2", "m3"]
    available = [key for key in weight_keys if key in df.columns and not df[key].dropna().empty]
    if not available:
        return "Evaluation weights were not available in the evaluated records for this run."

    first = df[available].dropna(how="all").head(1)
    if first.empty:
        return "Evaluation weights were not available in the evaluated records for this run."

    row = first.iloc[0].to_dict()
    descriptions = {
        "w1": "Physical infrastructure weight",
        "w2": "Digital infrastructure weight",
        "k1": "Lane-marking weight within physical infrastructure",
        "k2": "Reserved physical component weight",
        "k3": "Reserved physical component weight",
        "m1": "Connectivity weight within digital infrastructure",
        "m2": "GPS weight within digital infrastructure",
        "m3": "HD-map weight within digital infrastructure",
    }
    lines = [
        "These weights were read from the evaluated records used for this report. If the run was re-weighted, the values below reflect that updated configuration.",
        "",
        "| Weight | Value | Meaning |",
        "|---|---:|---|",
    ]
    for key in available:
        lines.append(f"| {key} | {float(row.get(key, 0.0)):.4f} | {descriptions.get(key, '')} |")
    return "\n".join(lines)


# ===============================
# Data Loader
# ===============================


def load_dataframe() -> pd.DataFrame:
    """Load evaluated readiness JSON files into a normalized pandas dataframe."""
    records: List[Dict[str, Any]] = []

    if not EVAL_FOLDER.exists():
        return pd.DataFrame(records)

    for file in sorted(EVAL_FOLDER.iterdir()):
        if file.suffix != ".json":
            continue

        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        evaluation = data.get("evaluation") or {}
        lane_eval = evaluation.get("lane") or {}
        gps_eval = evaluation.get("gps") or {}
        conn_eval = evaluation.get("connectivity") or {}
        details = evaluation.get("overall_score_details") or {}
        component_breakdown = details.get("components") or {}
        lane_best = lane_eval.get("best_threshold_detail") or {}
        lane_length_check = lane_best.get("length_check") or {}
        lane_curvature_check = lane_best.get("curvature_check") or {}
        lane_separation_check = lane_best.get("separation_check") or {}

        gps_raw = data.get("gps") or {}
        lat = _safe_float(gps_raw.get("lat"), default=float("nan"))
        lon = _safe_float(gps_raw.get("lon"), default=float("nan"))

        connectivity = data.get("connectivity") or {}
        weights = evaluation.get("weights") or {}
        records.append(
            {
                "source_json": str(file),
                "image": data.get("image"),
                "image_path": _resolve_image_path(data.get("image")),
                "frame_index": _safe_float(data.get("frame_index"), default=float("nan")),
                "lat": lat,
                "lon": lon,
                "lane_score": _safe_float(lane_eval.get("score")),
                "gps_score": _safe_float(gps_eval.get("score")),
                "connectivity_score": _safe_float(conn_eval.get("score")),
                "hd_maps_score": _safe_float(component_breakdown.get("hd_maps_score")),
                "overall_score": _safe_float(evaluation.get("overall_score")),
                "physical_score": _safe_float(component_breakdown.get("physical_score")),
                "digital_score": _safe_float(component_breakdown.get("digital_score")),
                "physical_infra_score": _safe_float(component_breakdown.get("physical_infra_score")),
                "digital_infra_score": _safe_float(component_breakdown.get("digital_infra_score")),
                "lane_curves_with_geometry": _safe_float(lane_best.get("curves_with_geometry")),
                "lane_valid_curves_before_separation": _safe_float(lane_best.get("valid_curves_before_separation")),
                "lane_pairs_evaluated": _safe_float(lane_best.get("pairs_evaluated")),
                "lane_pairs_meeting_separation": _safe_float(lane_best.get("pairs_meeting_separation")),
                "lane_length_pass_rate": _safe_float(lane_length_check.get("pass_rate")),
                "lane_curvature_pass_rate": _safe_float(lane_curvature_check.get("pass_rate")),
                "lane_separation_pass_rate": _safe_float(lane_separation_check.get("pass_rate")),
                # Add bandwidth and latency fields
                "avg_d_mbps": _safe_float(connectivity.get("avg_d_mbps"), default=float("nan")),
                "avg_u_mbps": _safe_float(connectivity.get("avg_u_mbps"), default=float("nan")),
                "avg_download_latency_ms": _safe_float(connectivity.get("avg_download_latency_ms"), default=float("nan")),
                "avg_upload_latency_ms": _safe_float(connectivity.get("avg_upload_latency_ms"), default=float("nan")),
                "w1": _safe_float(weights.get("w1"), default=float("nan")),
                "w2": _safe_float(weights.get("w2"), default=float("nan")),
                "k1": _safe_float(weights.get("k1"), default=float("nan")),
                "k2": _safe_float(weights.get("k2"), default=float("nan")),
                "k3": _safe_float(weights.get("k3"), default=float("nan")),
                "m1": _safe_float(weights.get("m1"), default=float("nan")),
                "m2": _safe_float(weights.get("m2"), default=float("nan")),
                "m3": _safe_float(weights.get("m3"), default=float("nan")),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    if "frame_index" in df.columns:
        df = df.sort_values(by="frame_index", na_position="last").reset_index(drop=True)

    # Build cumulative route distance so later charts can aggregate by mile bucket.
    cumulative: List[float] = []
    total = 0.0
    prev_lat = None
    prev_lon = None
    for _, row in df.iterrows():
        lat = row.get("lat")
        lon = row.get("lon")
        if pd.notna(lat) and pd.notna(lon) and prev_lat is not None and prev_lon is not None:
            total += _haversine_miles(prev_lat, prev_lon, float(lat), float(lon))
        cumulative.append(total)
        if pd.notna(lat) and pd.notna(lon):
            prev_lat = float(lat)
            prev_lon = float(lon)

    df["cumulative_miles"] = cumulative
    df["segment_id"] = (df["cumulative_miles"] / SEGMENT_SIZE_MILES).astype(int)

    return df


# ===============================
# Issue Detection
# ===============================


def classify_issues(row: pd.Series) -> List[str]:
    """Assign high-level issue labels to a frame based on component thresholds."""
    issues: List[str] = []

    if row.get("lane_score", 0.0) < 0.5:
        issues.append("lane quality low")
    if row.get("lane_curves_with_geometry", 0.0) > 0 and row.get("lane_length_pass_rate", 1.0) < 0.5:
        issues.append("lane length check weak")
    if row.get("lane_curves_with_geometry", 0.0) > 0 and row.get("lane_curvature_pass_rate", 1.0) < 0.5:
        issues.append("lane curvature check weak")
    if row.get("lane_pairs_evaluated", 0.0) > 0 and row.get("lane_separation_pass_rate", 1.0) < 0.5:
        issues.append("lane separation check weak")
    if row.get("gps_score", 0.0) < 0.5:
        issues.append("gps quality low")
    if row.get("connectivity_score", 0.0) < 0.5:
        issues.append("connectivity low")
    if row.get("hd_maps_score", 0.0) < 0.5:
        issues.append("hd-map coverage low")
    if row.get("digital_score", 0.0) < 0.5:
        issues.append("digital score low")
    if row.get("overall_score", 0.0) < 0.6:
        issues.append("overall readiness low")

    return issues


def build_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate frame-level records into segment-level readiness summaries."""
    rows: List[Dict[str, Any]] = []

    for segment_id, seg in df.groupby("segment_id", dropna=False):
        issue_counter: Counter = Counter()
        for _, row in seg.iterrows():
            issue_counter.update(classify_issues(row))

        rows.append(
            {
                "segment_id": int(segment_id),
                "start_mile": float(seg["cumulative_miles"].min()),
                "end_mile": float(seg["cumulative_miles"].max()),
                "sample_count": int(len(seg)),
                "overall_score_avg": float(seg["overall_score"].mean()),
                "lane_score_avg": float(seg["lane_score"].mean()),
                "lane_length_pass_rate_avg": float(seg["lane_length_pass_rate"].mean()),
                "lane_curvature_pass_rate_avg": float(seg["lane_curvature_pass_rate"].mean()),
                "lane_separation_pass_rate_avg": float(seg["lane_separation_pass_rate"].mean()),
                "gps_score_avg": float(seg["gps_score"].mean()),
                "connectivity_score_avg": float(seg["connectivity_score"].mean()),
                "issues": dict(issue_counter),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by="segment_id").reset_index(drop=True)



def build_issue_markdown(segment_df: pd.DataFrame) -> str:
    """Render the segment summary into markdown focused on dominant issues."""
    if segment_df.empty:
        return "No segment-level issue data available."

    lines: List[str] = []
    for _, row in segment_df.iterrows():
        issues: Dict[str, int] = row.get("issues") or {}
        top_issues = sorted(issues.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_issues:
            issue_text = ", ".join([f"`{k}` ({v})" for k, v in top_issues])
        else:
            issue_text = "no dominant issue"

        lines.append(
            "- Segment "
            f"{int(row['segment_id'])} "
            f"(mile {row['start_mile']:.2f}-{row['end_mile']:.2f}, n={int(row['sample_count'])}): "
            f"avg overall={row['overall_score_avg']:.3f}, avg lane={row['lane_score_avg']:.3f}, "
            f"lane checks L/C/S=({row['lane_length_pass_rate_avg']:.2f}/"
            f"{row['lane_curvature_pass_rate_avg']:.2f}/{row['lane_separation_pass_rate_avg']:.2f}), "
            f"avg gps={row['gps_score_avg']:.3f}, avg connectivity={row['connectivity_score_avg']:.3f}; "
            f"issues: {issue_text}."
        )

    return "\n".join(lines)



def pick_issue_images(df: pd.DataFrame, top_k_segments: int = 3, images_per_segment: int = 2) -> List[Dict[str, Any]]:
    """Select representative evidence images from the lowest-performing segments."""
    if df.empty:
        return []

    selected: List[Dict[str, Any]] = []

    # Focus on the lowest-performing segments.
    segment_rank = (
        df.groupby("segment_id")["overall_score"]
        .mean()
        .sort_values(ascending=True)
        .head(top_k_segments)
        .index.tolist()
    )

    for segment_id in segment_rank:
        seg = df[df["segment_id"] == segment_id].copy()
        seg = seg.sort_values(by="overall_score", ascending=True)
        added = 0
        for _, row in seg.iterrows():
            image_path = row.get("image_path")
            if not image_path:
                continue
            selected.append(
                {
                    "segment_id": int(segment_id),
                    "image": row.get("image"),
                    "frame_index": int(row.get("frame_index")) if pd.notna(row.get("frame_index")) else None,
                    "overall_score": float(row.get("overall_score", 0.0)),
                    "lane_score": float(row.get("lane_score", 0.0)),
                    "gps_score": float(row.get("gps_score", 0.0)),
                    "connectivity_score": float(row.get("connectivity_score", 0.0)),
                    "lane_length_pass_rate": float(row.get("lane_length_pass_rate", 0.0)),
                    "lane_curvature_pass_rate": float(row.get("lane_curvature_pass_rate", 0.0)),
                    "lane_separation_pass_rate": float(row.get("lane_separation_pass_rate", 0.0)),
                    "issues": classify_issues(row),
                    "image_path": image_path,
                }
            )
            added += 1
            if added >= images_per_segment:
                break

    return selected


# ===============================
# Plot Generator
# ===============================


def generate_figures(df: pd.DataFrame, segment_df: pd.DataFrame) -> List[str]:
    """Generate the chart set embedded in the compiled readiness PDF report."""
    figures: List[str] = []

    metrics = [
        "lane_score",
        "gps_score",
        "connectivity_score",
        "overall_score",
        "lane_length_pass_rate",
        "lane_curvature_pass_rate",
        # "lane_separation_pass_rate",
    ]

    for metric in metrics:
        if metric not in df.columns:
            continue

        plt.figure(figsize=(7, 4))
        df[metric].hist(bins=30)
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.title(f"{metric} Distribution")

        fig_path = FIGURE_FOLDER / f"{metric}.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=400)
        plt.close()
        figures.append(str(fig_path))

    # if not df.empty:
    #     plt.figure(figsize=(8, 4))
    #     sns.scatterplot(data=df, x="frame_index", y="overall_score", hue="segment_id", legend=False, s=18)
    #     plt.xlabel("frame_index")
    #     plt.ylabel("overall_score")
    #     plt.title("Overall Score Along Route")
    #     plt.tight_layout()
    #     route_fig = FIGURE_FOLDER / "overall_vs_frame.png"
    #     plt.savefig(route_fig, dpi=200)
    #     plt.close()
    #     figures.append(str(route_fig))

    if not segment_df.empty:
        plt.figure(figsize=(7, 4))
        sns.barplot(data=segment_df, x="segment_id", y="overall_score_avg", color="#1972D2")
        plt.xlabel("Road segment (mile bucket)")
        plt.ylabel("Average overall_score")
        plt.title("Segment-wise Readiness")
        plt.tight_layout()
        seg_fig = FIGURE_FOLDER / "segment_overall_score.png"
        plt.savefig(seg_fig, dpi=400)
        plt.close()
        figures.append(str(seg_fig))

    summary_values = {
        "Lane": _safe_float(df.get("lane_score", pd.Series(dtype=float)).mean(), default=float("nan")),
        "Physical": _safe_float(df.get("physical_score", pd.Series(dtype=float)).mean(), default=float("nan")),
        "Connectivity": _safe_float(df.get("connectivity_score", pd.Series(dtype=float)).mean(), default=float("nan")),
        "GPS": _safe_float(df.get("gps_score", pd.Series(dtype=float)).mean(), default=float("nan")),
        "HD-Map": _safe_float(df.get("hd_maps_score", pd.Series(dtype=float)).mean(), default=float("nan")),
        "Digital": _safe_float(df.get("digital_score", pd.Series(dtype=float)).mean(), default=float("nan")),
        "Overall": _safe_float(df.get("overall_score", pd.Series(dtype=float)).mean(), default=float("nan")),
    }
    if not np.isnan(summary_values["Lane"]) and np.isnan(summary_values["Physical"]):
        summary_values["Physical"] = summary_values["Lane"]
    if np.isnan(summary_values["Digital"]):
        digital_components = [
            value
            for key, value in (
                ("Connectivity", summary_values["Connectivity"]),
                ("GPS", summary_values["GPS"]),
                ("HD-Map", summary_values["HD-Map"]),
            )
            if not np.isnan(value)
        ]
        if digital_components:
            summary_values["Digital"] = float(np.mean(digital_components))

    physical_items = [("Lane", summary_values["Lane"]), ("Physical", summary_values["Physical"])]
    digital_items = [
        ("Connectivity", summary_values["Connectivity"]),
        ("GPS", summary_values["GPS"]),
        ("HD-Map", summary_values["HD-Map"]),
        ("Digital", summary_values["Digital"]),
    ]
    overall_items = [("Overall", summary_values["Overall"])]
    if any(not np.isnan(value) for _, value in physical_items + digital_items + overall_items):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(10.8, 4.6),
            sharey=True,
            gridspec_kw={"width_ratios": [2.2, 4.2, 1.8]},
        )
        summary_groups = [
            ("Physical Readiness", physical_items, "#1972D2"),
            ("Digital Readiness", digital_items, "#D2691E"),
            ("Overall Readiness", overall_items, "#2E8B57"),
        ]
        for ax, (title, items, color) in zip(axes, summary_groups):
            labels = [label for label, value in items if not np.isnan(value)]
            values = [value for _, value in items if not np.isnan(value)]
            if not values:
                ax.axis("off")
                continue
            bars = ax.bar(labels, values, color=color, alpha=0.9)
            ax.set_title(title, pad=14)
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            ax.tick_params(axis="x", rotation=20)
            for bar, value in zip(bars, values):
                text_x = bar.get_x() + bar.get_width() / 2.0
                if value >= 0.14:
                    text_y = max(value - 0.06, 0.03)
                    va = "top"
                    text_color = "white"
                    bbox = None
                else:
                    text_y = min(value + 0.03, 0.98)
                    va = "bottom"
                    text_color = color
                    bbox = dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor=color, alpha=0.95)
                ax.text(
                    text_x,
                    text_y,
                    f"{value:.3f}",
                    ha="center",
                    va=va,
                    fontsize=9,
                    fontweight="bold",
                    color=text_color,
                    bbox=bbox,
                )
        axes[0].set_ylabel("Average readiness score")
        fig.suptitle("Physical, Digital, and Overall Readiness Summary", fontsize=13, fontweight="bold")
        fig.tight_layout()
        summary_fig = FIGURE_FOLDER / "readiness_component_summary.png"
        fig.savefig(summary_fig, dpi=400)
        plt.close(fig)
        figures.append(str(summary_fig))

    # plt.figure(figsize=(7, 5))
    # sns.heatmap(df.corr(numeric_only=True), annot=False)
    # plt.title("Metric Correlation Matrix")
    # plt.tight_layout()
    # corr_path = FIGURE_FOLDER / "correlation_matrix.png"
    # plt.savefig(corr_path, dpi=200)
    # plt.close()
    # figures.append(str(corr_path))

    # Per-mile plots share the same route segmentation, which keeps connectivity,
    # lane, and GPS views aligned on the same mile buckets in the final report.
    per_mile_metrics = [
        ("lane_score", "Average Lane Score", "lane_score_per_mile.png"),
        ("gps_score", "Average GPS Score", "gps_score_per_mile.png"),
        ("avg_d_mbps", "Average Download Speed (Mbps)", "download_speed_per_mile.png"),
        ("avg_u_mbps", "Average Upload Speed (Mbps)", "upload_speed_per_mile.png"),
        ("avg_download_latency_ms", "Average Latency (ms)", "latency_per_mile.png"),
    ]
    line_plot_metrics = {"lane_score", "gps_score"}
    for col, ylabel, fname in per_mile_metrics:
        if col in df.columns and not df[col].isna().all():
            # Average per mile (segment)
            per_mile = df.groupby("segment_id")[col].mean().reset_index()
            avg_value = float(per_mile[col].mean())
            p10_value = float(per_mile[col].quantile(0.10))
            p90_value = float(per_mile[col].quantile(0.90))
            fig, ax = plt.subplots(figsize=(8, 4))
            if col in line_plot_metrics:
                # Lane/GPS use a hybrid view: bars for bucket magnitude plus a line
                # to emphasize the trend across consecutive route segments.
                sns.barplot(x=per_mile["segment_id"], y=per_mile[col], color="#1972D2", ax=ax)
                ax.plot(
                    range(len(per_mile)),
                    per_mile[col].to_numpy(),
                    color="#c23b22",
                    linewidth=2.4,
                    marker="o",
                    markersize=5.5,
                )
                ax.set_ylim(0.0, 1.0)
            else:
                sns.barplot(x=per_mile["segment_id"], y=per_mile[col], color="#1972D2", ax=ax)
            ax.axhline(avg_value, color="#1f8f4c", linestyle="--", linewidth=2.2)
            ax.text(
                0.99,
                avg_value,
                f"Avg {avg_value:.2f}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                color="#1f8f4c",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#1f8f4c", alpha=0.95),
            )
            if col not in line_plot_metrics and p90_value > p10_value:
                ax.axhline(p90_value, color="#c23b22", linestyle=":", linewidth=2.0)
                ax.axhline(p10_value, color="#c23b22", linestyle=":", linewidth=2.0)
                ax.text(
                    0.99,
                    p90_value,
                    f"P90 {p90_value:.2f}",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="bottom",
                    color="#c23b22",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#c23b22", alpha=0.95),
                )
                ax.text(
                    0.99,
                    p10_value,
                    f"P10 {p10_value:.2f}",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="top",
                    color="#c23b22",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#c23b22", alpha=0.95),
                )
            ax.set_xlabel("Road segment (mile bucket)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} by Mile Segment")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig_path = FIGURE_FOLDER / fname
            fig.savefig(fig_path, dpi=400)
            plt.close(fig)
            figures.append(str(fig_path))

    route_metrics = [
        ("lane_score", "Lane Marking Heatmap", "route_heatmap_lane_score.png"),
        ("connectivity_score", "Connectivity Heatmap", "route_heatmap_connectivity_score.png"),
        ("gps_score", "GPS Heatmap", "route_heatmap_gps_score.png"),
        ("hd_maps_score", "HD-Map Heatmap", "route_heatmap_hd_maps_score.png"),
        ("overall_score", "Overall Score Heatmap", "route_heatmap_overall_score.png"),
    ]
    route_df = (
        df[["lat", "lon", *[metric for metric, _, _ in route_metrics if metric in df.columns]]]
        .dropna(subset=["lat", "lon"])
        .copy()
    )
    if len(route_df) >= 2:
        lats = route_df["lat"].astype(float).to_numpy()
        lons = route_df["lon"].astype(float).to_numpy()
        point_pairs = np.column_stack([lons, lats]).reshape(-1, 1, 2)
        segments = np.concatenate([point_pairs[:-1], point_pairs[1:]], axis=1)
        for metric, title, fname in route_metrics:
            if metric not in route_df.columns:
                continue
            values = route_df[metric].astype(float).fillna(0.0).clip(0.0, 1.0).to_numpy()
            if len(values) < 2:
                continue
            seg_values = (values[:-1] + values[1:]) / 2.0
            fig, ax = plt.subplots(figsize=(7.2, 5.0))
            lc = LineCollection(segments, cmap="RdYlGn", norm=plt.Normalize(0.0, 1.0))
            lc.set_array(seg_values)
            lc.set_linewidth(4.5)
            ax.add_collection(lc)
            scatter = ax.scatter(lons, lats, c=values, cmap="RdYlGn", vmin=0.0, vmax=1.0, s=10, zorder=3)
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Score")
            ax.set_title(title)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(alpha=0.2, linewidth=0.4)
            ax.set_aspect("equal", adjustable="datalim")
            ax.margins(x=0.03, y=0.03)
            fig.tight_layout()
            fig_path = FIGURE_FOLDER / fname
            fig.savefig(fig_path, dpi=320)
            plt.close(fig)
            figures.append(str(fig_path))
    return figures


# ===============================
# OpenAI Summary Generator
# ===============================


def build_summary_payload(df: pd.DataFrame, segment_df: pd.DataFrame) -> str:
    """Create the structured statistics payload sent to the narrative summarizer."""
    stat_cols = [
        "lane_score",
        "gps_score",
        "connectivity_score",
        "hd_maps_score",
        "overall_score",
        "lane_length_pass_rate",
        "lane_curvature_pass_rate",
        "lane_separation_pass_rate",
    ]
    payload = {
        "num_samples": int(len(df)),
        "num_segments": int(segment_df["segment_id"].nunique()) if not segment_df.empty else 0,
        "overall_score_mean": _safe_float(df.get("overall_score", pd.Series(dtype=float)).mean(), 0.0),
        "statistics": df[stat_cols].describe().to_dict(),
        "segment_snapshot": segment_df.head(8).to_dict(orient="records") if not segment_df.empty else [],
    }
    return json.dumps(payload)



def summarize_results(df: pd.DataFrame, segment_df: pd.DataFrame) -> str:
    """Generate a concise markdown summary, using OpenAI when available."""
    payload = build_summary_payload(df, segment_df)

    prompt = f"""
Write a concise research-style evaluation report summary in Markdown.

The idea is about rural road infrastructure evaluation to help readiness of the infrastructure of for AV deployment. 
The evaluation includes lane detection, GPS quality (erro measurement), and 5G connectivity. 
The data includes frame-level scores and segment-level summaries, and overall scores.

Rules:
- Use headings and bullet lists when necessary.
- Keep it short and to the point.
- Keep it factual (no assumptions, use mean and standard deviation/variance).
- Acronyms must be defined when used first time.
- respond to the following titles (Under each title, you can include statistics and multiple bullet points interpreting the data)
    - Executive Summary, (first paragraph is the general summary in terms of readiness based on the overall score-no bullet points,
      followed by key facts about the data, the evalution of each component,... in a bullet list)
    - Overall Performance, 
    - Segment-level Issues
    - Conclusions, (based on the overall performance and segment-level issues)

Dataset summary
{payload}
"""

    try:
        client = OpenAI()
        response = client.responses.create(model="gpt-5-mini", input=prompt)
        text = (getattr(response, "output_text", "") or "").strip()
        if text:
            return text
    except Exception as exc:
        print(f"OpenAI summary generation failed: {exc}")
        return (
            "## Automated Summary (Fallback)\n\n"
            "OpenAI summary generation was unavailable in this run.\n\n"
            "- Reason: "
            f"`{type(exc).__name__}`\n"
            "- A deterministic summary is provided via segment/issue sections below."
        )

    return (
        "## Automated Summary (Fallback)\n\n"
        "OpenAI summary returned no text. A deterministic summary is provided below."
    )


# ===============================
# PDF Report Builder
# ===============================


def _build_issue_images_markdown(issue_images: List[Dict[str, Any]]) -> str:
    """Render selected issue images into a markdown section for the report."""
    if not issue_images:
        return "No representative issue images were found on disk."

    lines: List[str] = []
    for item in issue_images:
        issues = ", ".join([f"`{x}`" for x in (item.get("issues") or ["no_issue_label"])])
        caption = (
            f"Segment {item['segment_id']} | frame {item.get('frame_index')} | "
            f"overall={item['overall_score']:.3f}, lane={item['lane_score']:.3f}, "
            f"gps={item['gps_score']:.3f}, conn={item['connectivity_score']:.3f}, "
            f"lane Leng/Curv/Sepa=({item['lane_length_pass_rate']:.2f}/{item['lane_curvature_pass_rate']:.2f}/"
            f"{item['lane_separation_pass_rate']:.2f}) | issues: {issues}"
        )
        lines.append(f"### {caption}")
        lines.append(f"![{caption}]({item['image_path']})")

    return "\n\n".join(lines)


def _build_figure_markdown(figures: List[str]) -> str:
    """Render generated figure files into the markdown figure section."""
    if not figures:
        return "No figures generated."

    descriptions = {
        "lane_score.png": "Histogram of lane readiness score. Higher values indicate stronger lane detection confidence/quality.",
        "gps_score.png": "Histogram of GPS quality score derived from localization error.",
        "connectivity_score.png": "Histogram of connectivity score from throughput/latency metrics.",
        "overall_score.png": "Histogram of final combined readiness score.",
        "lane_length_pass_rate.png": "Distribution of pass-rate for lane minimum-length checks per frame.",
        "lane_curvature_pass_rate.png": "Distribution of pass-rate for lane curvature checks per frame.",
        "lane_separation_pass_rate.png": "Distribution of pass-rate for lane separation checks per frame.",
        "readiness_component_summary.png": "Summary chart comparing average physical readiness, digital readiness, and overall readiness, along with their component scores.",
        "segment_overall_score.png": "Average overall readiness by road segment (mile bucket), useful for hotspot identification.",
        "lane_score_per_mile.png": "Bar chart of average lane readiness score per mile segment with average, P10, and P90 reference lines.",
        "gps_score_per_mile.png": "Bar chart of average GPS readiness score per mile segment with average, P10, and P90 reference lines.",
        "download_speed_per_mile.png": "Bar chart of average download speed (Mbps) per mile segment with average, P10, and P90 reference lines.",
        "upload_speed_per_mile.png": "Bar chart of average upload speed (Mbps) per mile segment with average, P10, and P90 reference lines.",
        "latency_per_mile.png": "Bar chart of average latency (ms) per mile segment with average, P10, and P90 reference lines. Ookla provides a single latency measure, so upload/download latency are not shown separately.",
        "route_heatmap_lane_score.png": "Route heatmap for the lane-marking component score.",
        "route_heatmap_connectivity_score.png": "Route heatmap for the connectivity component score.",
        "route_heatmap_gps_score.png": "Route heatmap for the GPS component score.",
        "route_heatmap_hd_maps_score.png": "Route heatmap for the HD-map component score.",
        "route_heatmap_overall_score.png": "Route heatmap for the overall readiness score.",
    }

    blocks: List[str] = []
    for fig in figures:
        name = Path(fig).name
        desc = descriptions.get(name, "Metric visualization generated from evaluated records.")
        blocks.append(
            "\n".join(
                [
                    '<div class="figure-block">',
                    # f"<h2>{name}</h2>",
                    f"<p>{desc}</p>",
                    f'<img src="{fig}" alt="{name}" />',
                    "</div>",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_pdf_report(
    summary_markdown: str,
    df: pd.DataFrame,
    figures: List[str],
    segment_df: pd.DataFrame,
    issue_images: List[Dict[str, Any]],
) -> None:
    """Assemble report sections and render the final PDF into the results folder."""
    pdf_path = RESULTS_DIR / "research_report.pdf"

    figure_markdown = _build_figure_markdown(figures)
    segment_issue_markdown = build_issue_markdown(segment_df)
    issue_images_markdown = _build_issue_images_markdown(issue_images)
    location_markdown = _build_location_markdown(df)
    weights_markdown = _build_weights_markdown(df)

    segment_section = "" if "Segment-level Issues" in summary_markdown else f"## Segment-Level Issues\n\n{segment_issue_markdown}\n"
    
    full_markdown = f"""
<div class="header">
    Rural Road Infrastructure Evaluation Report
</div>

## Location

{location_markdown}

## Evaluation Weights

{weights_markdown}

<div class="page-break"></div>

{summary_markdown}

{segment_section}

## Figures
{figure_markdown}

## Issue Evidence Images

{issue_images_markdown}

## Evaluation Data Snapshot Sample (Top 40)

{df.head(40)[['frame_index','image','segment_id','lane_score','gps_score','connectivity_score','overall_score']].to_markdown(index=False)}
"""

    html_content = markdown2.markdown(full_markdown, extras=["tables", "code-friendly"])

    HTML(string=html_content, base_url=str(RESULTS_DIR)).write_pdf(
        str(pdf_path),
        stylesheets=[
            CSS(
                string="""
                @page {
                    size: A4 portrait;
                    margin: 14mm;
                }
                .header {
                    text-align: center;
                    font-size: 20px;
                    font-weight: 600;
                    color: black;
                }               
                
                body {
                    font-family: Arial, sans-serif;
                    font-size: 12px;
                    line-height: 1.35;
                }
                h1, h2, h3, h4 {
                    font-size: 13px;
                    font-weight: 600;
                    margin: 0.4em 0 0.25em 0;
                }
                h1, h2 {
                    color: #1565c0;
                }
                h3, h4 {
                    color: #111;
                }
                h2 {
                    text-align: center;
                    margin-top: 0.9em;
                }
                h2::after {
                    content: "";
                    display: block;
                    width: 40%;
                    margin: 6px auto 0 auto;
                    border-top: 1.5px solid #90a4ae;
                }
                .figure-block {
                    width: 80%;
                    margin: 0 auto 10mm auto;
                    text-align: center;
                    page-break-inside: avoid;
                }
                .location-grid {
                    display: block;
                    margin-top: 6px;
                    margin-bottom: 8px;
                }
                .location-item {
                    width: 82%;
                    margin: 0 auto 8mm auto;
                    text-align: center;
                }
                .location-caption {
                    font-size: 10px;
                    color: #333;
                    margin-bottom: 4px;
                }
                .page-break {
                    page-break-after: always;
                }
                .figure-block h3,
                .figure-block p {
                    text-align: center;
                    margin-left: auto;
                    margin-right: auto;
                }
                img {
                    max-width: 100%;
                    max-height: 220mm;
                    object-fit: contain;
                    display: block;
                    margin: 4mm auto;
                    border: 1px solid #ddd;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 10px;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 4px;
                }
                """
            )
        ],
    )

    print(f"PDF generated -> {pdf_path}")


# ===============================
# Main Pipeline
# ===============================


def main() -> None:
    """Command-line entry point for the readiness report compilation pipeline."""
    parser = argparse.ArgumentParser(description="Compile readiness PDF report from evaluated records.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results root containing evaluated/, figures/, readiness/, lane/.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (REPO_ROOT / results_dir).resolve()
    _configure_results_paths(results_dir)

    print("Loading dataset...")
    df = load_dataframe()

    if len(df) == 0:
        print("No evaluation data found.")
        return

    segment_df = build_segment_summary(df)
    issue_images = pick_issue_images(df)

    print("Generating figures...")
    figures = generate_figures(df, segment_df)

    print("Generating OpenAI summary...")
    summary = summarize_results(df, segment_df)

    print("Building PDF report...")
    build_pdf_report(summary, df, figures, segment_df, issue_images)

    print("Pipeline completed.")


if __name__ == "__main__":
    main()
