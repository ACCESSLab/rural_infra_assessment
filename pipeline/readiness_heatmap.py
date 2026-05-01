#!/usr/bin/env python3
"""Build readiness map, dashboard, and per-mile JSON artifacts from evaluated data."""

import argparse
import html
import json
import math
import os
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import cv2
import folium
import numpy as np
from branca.colormap import LinearColormap
import geopandas as gp
from shapely.geometry import Point

# Ensure top-level project imports work when invoked as:
#   python3 pipeline/readiness_heatmap.py
# (script dir would otherwise be only /.../pipeline).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from connectivity.aggregate_by_state import OoklaSpeedLookup


METRIC_LAYERS: List[Dict[str, str]] = [
    {"id": "lane", "label": "Lane Marking", "record_key": "lane_score", "mile_key": "lane_score_avg"},
    {"id": "connectivity", "label": "Connectivity", "record_key": "connectivity_score", "mile_key": "connectivity_score_avg"},
    {"id": "gps", "label": "GPS", "record_key": "gps_score", "mile_key": "gps_score_avg"},
    {"id": "hd_map", "label": "HD-Map", "record_key": "hd_maps_score", "mile_key": "hd_maps_score_avg"},
    {"id": "overall", "label": "Overall Score", "record_key": "overall_score", "mile_key": "readiness_index"},
]


def _safe_float(v: Any) -> Optional[float]:
    """Convert a value to float, returning ``None`` when conversion fails."""
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in miles between two coordinates."""
    r_miles = 3958.7613
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r_miles * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _readiness_label(score: float) -> Tuple[str, str]:
    """Map a normalized readiness score to its label and display color."""
    # Per provided rubric:
    # 0.0..0.4 Low (red), 0.4..0.8 Medium (yellow), 0.8..1.0 High (green).
    if score <= 0.4:
        return "Low Readiness", "red"
    if score <= 0.8:
        return "Medium Readiness", "yellow"
    return "High Readiness", "green"


def _score_color(score: float) -> str:
    """Return the display color associated with a readiness score."""
    _, color = _readiness_label(max(0.0, min(1.0, score)))
    return color


def _path_for_ui(path_like: Any) -> str:
    """Return a UI-safe POSIX path, preferring repo-relative form when possible."""
    raw = str(path_like or "").strip()
    if not raw:
        return ""
    try:
        p = Path(raw)
        if p.is_absolute():
            resolved = p.resolve()
            try:
                return resolved.relative_to(REPO_ROOT).as_posix()
            except Exception:
                return resolved.as_posix()
        return p.as_posix()
    except Exception:
        return raw.replace("\\", "/")


def _load_records(evaluated_dir: Path) -> List[Dict[str, Any]]:
    """Load evaluated JSON records and normalize the fields used by the dashboard."""
    records: List[Dict[str, Any]] = []
    for path in sorted(evaluated_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        gps = obj.get("gps") if isinstance(obj, dict) else None
        eval_obj = obj.get("evaluation") if isinstance(obj, dict) else None
        if not isinstance(gps, dict) or not isinstance(eval_obj, dict):
            continue

        lat = _safe_float(gps.get("lat"))
        lon = _safe_float(gps.get("lon"))
        if lat is None or lon is None:
            continue

        lane_score = _safe_float((eval_obj.get("lane") or {}).get("score")) or 0.0
        gps_score = _safe_float((eval_obj.get("gps") or {}).get("score")) or 0.0
        conn_score = _safe_float((eval_obj.get("connectivity") or {}).get("score")) or 0.0
        component_breakdown = ((eval_obj.get("overall_score_details") or {}).get("components")) or {}
        hd_maps_score = _safe_float(component_breakdown.get("hd_maps_score")) or 0.0
        overall_score = _safe_float(eval_obj.get("overall_score")) or 0.0
        lane_best = (eval_obj.get("lane") or {}).get("best_threshold_detail") or {}
        lane_length_pass_rate = _safe_float(((lane_best.get("length_check") or {}).get("pass_rate"))) or 0.0
        lane_curvature_pass_rate = _safe_float(((lane_best.get("curvature_check") or {}).get("pass_rate"))) or 0.0
        lane_curvature_soft_pass_rate = _safe_float(((lane_best.get("curvature_check") or {}).get("soft_pass_rate"))) or 0.0
        lane_separation_pass_rate = _safe_float(((lane_best.get("separation_check") or {}).get("pass_rate"))) or 0.0

        records.append(
            {
                "file": path.name,
                "image": str(obj.get("image", "")),
                "frame_index": int(obj.get("frame_index", 0)),
                "timestamp_ns": int(obj.get("timestamp_ns", 0)),
                "lat": lat,
                "lon": lon,
                "lane_score": lane_score,
                "lane_length_pass_rate": lane_length_pass_rate,
                "lane_curvature_pass_rate": lane_curvature_pass_rate,
                "lane_curvature_soft_pass_rate": lane_curvature_soft_pass_rate,
                "lane_separation_pass_rate": lane_separation_pass_rate,
                "gps_score": gps_score,
                "connectivity_score": conn_score,
                "hd_maps_score": hd_maps_score,
                "overall_score": overall_score,
                "evaluation": eval_obj,
                "lane_raw": obj.get("lane"),
                "gps_raw": gps,
                "connectivity_raw": obj.get("connectivity"),
            }
        )

    records.sort(key=lambda r: (r["frame_index"], r["timestamp_ns"], r["file"]))
    return records


def _aggregate_per_mile(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    """Aggregate frame-level records into one-mile readiness buckets."""
    if not records:
        return [], 0.0

    cumulative_miles = 0.0
    prev = records[0]
    prev["mile_marker"] = 0.0
    prev["mile_index"] = 0

    bins: Dict[int, Dict[str, Any]] = {}

    def ensure_bin(idx: int) -> Dict[str, Any]:
        if idx not in bins:
            bins[idx] = {
                "mile_index": idx,
                "start_mile": float(idx),
                "end_mile": float(idx + 1),
                "count": 0,
                "sum_lane": 0.0,
                "sum_gps": 0.0,
                "sum_connectivity": 0.0,
                "sum_hd_map": 0.0,
                "sum_overall": 0.0,
                "sum_lat": 0.0,
                "sum_lon": 0.0,
            }
        return bins[idx]

    for i, rec in enumerate(records):
        if i > 0:
            cumulative_miles += _haversine_miles(prev["lat"], prev["lon"], rec["lat"], rec["lon"])
        rec["mile_marker"] = cumulative_miles
        rec["mile_index"] = int(math.floor(cumulative_miles))
        prev = rec

        b = ensure_bin(rec["mile_index"])
        b["count"] += 1
        b["sum_lane"] += rec["lane_score"]
        b["sum_gps"] += rec["gps_score"]
        b["sum_connectivity"] += rec["connectivity_score"]
        b["sum_hd_map"] += rec["hd_maps_score"]
        b["sum_overall"] += rec["overall_score"]
        b["sum_lat"] += rec["lat"]
        b["sum_lon"] += rec["lon"]

    per_mile: List[Dict[str, Any]] = []
    for idx in sorted(bins.keys()):
        b = bins[idx]
        count = max(1, b["count"])
        lane_avg = b["sum_lane"] / count
        gps_avg = b["sum_gps"] / count
        conn_avg = b["sum_connectivity"] / count
        hd_map_avg = b["sum_hd_map"] / count
        overall_avg = b["sum_overall"] / count
        readiness, color = _readiness_label(overall_avg)

        per_mile.append(
            {
                "mile_index": idx,
                "start_mile": b["start_mile"],
                "end_mile": b["end_mile"],
                "count": b["count"],
                "center_lat": b["sum_lat"] / count,
                "center_lon": b["sum_lon"] / count,
                "lane_score_avg": lane_avg,
                "gps_score_avg": gps_avg,
                "connectivity_score_avg": conn_avg,
                "hd_maps_score_avg": hd_map_avg,
                "readiness_index": overall_avg,
                "readiness_level": readiness,
                "color": color,
            }
        )

    return per_mile, cumulative_miles


def _build_map(
    records: List[Dict[str, Any]],
    per_mile: List[Dict[str, Any]],
    key_points: List[Dict[str, Any]],
    output_html: Path,
    mapbox_token: Optional[str],
) -> None:
    """Render the interactive readiness heatmap with basemap and key-point controls."""
    center_lat = sum(r["lat"] for r in records) / len(records)
    center_lon = sum(r["lon"] for r in records) / len(records)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True, tiles=None)

    base_layers: Dict[str, folium.TileLayer] = {}

    def _register_base_layer(layer_id: str, **kwargs: Any) -> None:
        layer = folium.TileLayer(control=False, **kwargs)
        layer.add_to(m)
        base_layers[layer_id] = layer

    _register_base_layer(
        "positron",
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="CartoDB Positron",
        max_zoom=20,
        overlay=False,
        subdomains="abcd",
    )
    _register_base_layer(
        "voyager",
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="CartoDB Voyager",
        max_zoom=20,
        overlay=False,
        subdomains="abcd",
        show=False,
    )
    _register_base_layer(
        "gray",
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Esri World Gray Canvas",
        max_zoom=16,
        overlay=False,
        show=False,
    )
    _register_base_layer(
        "dark",
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="CartoDB Dark Matter",
        max_zoom=20,
        overlay=False,
        subdomains="abcd",
        show=False,
    )
    _register_base_layer(
        "osm",
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        show=False,
    )
    if mapbox_token:
        _register_base_layer(
            "mapbox_light",
            tiles=(
                "https://api.mapbox.com/styles/v1/mapbox/light-v11/tiles/"
                "{z}/{x}/{y}?access_token=" + mapbox_token
            ),
            attr="Mapbox",
            max_zoom=19,
            tile_size=512,
            zoom_offset=-1,
            name="Mapbox Light",
            overlay=False,
            show=False,
        )

    points = [(r["lat"], r["lon"]) for r in records]
    colormap = LinearColormap(colors=["red", "yellow", "green"], vmin=0.0, vmax=1.0)
    colormap.caption = "Infrastructure Score"
    colormap.add_to(m)
    metric_groups: Dict[str, folium.FeatureGroup] = {}
    metric_panes: Dict[str, str] = {}
    keypoint_panes: Dict[str, str] = {}
    for metric in METRIC_LAYERS:
        metric_pane = f"metricPane_{metric['id']}"
        keypoint_pane = f"keypointPane_{metric['id']}"
        folium.map.CustomPane(metric_pane, z_index=410).add_to(m)
        folium.map.CustomPane(keypoint_pane, z_index=430).add_to(m)
        metric_panes[metric["id"]] = metric_pane
        keypoint_panes[metric["id"]] = keypoint_pane
        group = folium.FeatureGroup(
            name=f"{metric['label']} Heatmap",
            show=True,
            control=False,
        )
        group.add_to(m)
        metric_groups[metric["id"]] = group

    for metric in METRIC_LAYERS:
        score_key = metric["record_key"]
        metric_group = metric_groups[metric["id"]]
        metric_values = [max(0.0, min(1.0, _safe_float(r.get(score_key)) or 0.0)) for r in records]
        for i in range(1, len(points)):
            seg_score = (metric_values[i - 1] + metric_values[i]) / 2.0
            folium.PolyLine(
                locations=[points[i - 1], points[i]],
                color=colormap(seg_score),
                weight=7,
                opacity=0.9,
                pane=metric_panes[metric["id"]],
            ).add_to(metric_group)

    # Optional county-level connectivity layer under readiness path.
    county_layer_var = None
    try:
        lookup = OoklaSpeedLookup(verbose=False)
        counties = lookup._get_all_counties()[["GEOID", "NAMELSAD", "STATEFP", "geometry"]].copy()
        point_rows = []
        for r in records:
            avg_d = _safe_float(((r.get("connectivity_raw") or {}).get("avg_d_mbps")))
            point_rows.append(
                {
                    "lat": r["lat"],
                    "lon": r["lon"],
                    "conn_score": r["connectivity_score"],
                    "avg_d_mbps": avg_d if avg_d is not None else np.nan,
                }
            )
        points_gdf = gp.GeoDataFrame(
            point_rows,
            geometry=[Point(xy[1], xy[0]) for xy in [(r["lat"], r["lon"]) for r in point_rows]],
            crs=4326,
        )
        joined = gp.sjoin(points_gdf, counties, how="left", predicate="within")
        agg = (
            joined.groupby("GEOID", dropna=True)
            .agg(
                conn_score=("conn_score", "mean"),
                avg_d_mbps=("avg_d_mbps", "mean"),
                sample_count=("conn_score", "count"),
            )
            .reset_index()
        )
        merged = counties.merge(agg, on="GEOID", how="inner")
        if not merged.empty:
            conn_cmap = LinearColormap(colors=["#f7fcf5", "#74c476", "#00441b"], vmin=0.0, vmax=1.0)
            conn_cmap.caption = "County Connectivity Coverage (score)"
            conn_cmap.add_to(m)
            folium.map.CustomPane("countyPane", z_index=350).add_to(m)

            def _county_style(feat: Dict[str, Any]) -> Dict[str, Any]:
                val = _safe_float((feat.get("properties") or {}).get("conn_score"))
                v = 0.0 if val is None else max(0.0, min(1.0, val))
                return {
                    "fillColor": conn_cmap(v),
                    "color": "#1b5e20",
                    "weight": 1,
                    "fillOpacity": 0.35,
                    "opacity": 0.4,
                }

            county_layer = folium.GeoJson(
                data=json.loads(merged.to_json()),
                name="County Connectivity Layer",
                style_function=_county_style,
                tooltip=folium.GeoJsonTooltip(
                    fields=["NAMELSAD", "conn_score", "avg_d_mbps", "sample_count"],
                    aliases=["County", "Connectivity Score", "Avg Download Mbps", "Samples"],
                    localize=True,
                    sticky=False,
                    labels=True,
                ),
                pane="countyPane",
            ).add_to(m)
            county_layer_var = county_layer.get_name()
    except Exception:
        county_layer_var = None

    folium.Marker(location=points[0], tooltip="Start", icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker(location=points[-1], tooltip="End", icon=folium.Icon(color="purple")).add_to(m)

    for row in per_mile:
        for metric in METRIC_LAYERS:
            metric_value = max(0.0, min(1.0, _safe_float(row.get(metric["mile_key"])) or 0.0))
            metric_level, metric_color = _readiness_label(metric_value)
            popup = (
                f"Mile {row['mile_index']}<br>"
                f"{metric['label']}: {metric_value:.3f} ({metric_level})<br>"
                f"Lane: {row['lane_score_avg']:.3f}, GPS: {row['gps_score_avg']:.3f}, "
                f"Connectivity: {row['connectivity_score_avg']:.3f}, HD-Map: {row['hd_maps_score_avg']:.3f}<br>"
                f"Overall: {row['readiness_index']:.3f} ({row['readiness_level']})<br>"
                f"Samples: {row['count']}"
            )
            folium.CircleMarker(
                location=[row["center_lat"], row["center_lon"]],
                radius=5,
                color=metric_color,
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(popup, max_width=380),
                tooltip=f"Mile {row['mile_index']} | {metric['label']}: {metric_level}",
                pane=metric_panes[metric["id"]],
            ).add_to(metric_groups[metric["id"]])

    keypoint_groups: Dict[str, folium.FeatureGroup] = {}
    for metric in METRIC_LAYERS:
        metric_id = metric["id"]
        group = folium.FeatureGroup(
            name=f"{metric['label']} Key Points",
            show=True,
            control=False,
        )
        group.add_to(m)
        keypoint_groups[metric_id] = group

    for kp in key_points:
        for metric in METRIC_LAYERS:
            metric_id = metric["id"]
            folium.CircleMarker(
                location=[kp["lat"], kp["lon"]],
                radius=7,
                color=(kp.get("color_by_metric") or {}).get(metric_id, kp["color"]),
                fill=True,
                fill_opacity=0.95,
                weight=2,
                tooltip=folium.Tooltip((kp.get("tooltip_html_by_metric") or {}).get(metric_id, kp["tooltip_html"]), sticky=True),
                popup=folium.Popup((kp.get("tooltip_html_by_metric") or {}).get(metric_id, kp["tooltip_html"]), max_width=460),
                pane=keypoint_panes[metric_id],
            ).add_to(keypoint_groups[metric_id])

    folium.LayerControl(position="topright", collapsed=True).add_to(m)
    # Allow parent dashboard to command map zoom/focus via postMessage.
    map_var = m.get_name()
    base_layer_vars = {layer_id: layer.get_name() for layer_id, layer in base_layers.items()}
    metric_group_vars = {metric_id: group.get_name() for metric_id, group in metric_groups.items()}
    keypoint_group_vars = {metric_id: group.get_name() for metric_id, group in keypoint_groups.items()}
    focus_script = f"""
<script>
(function() {{
  var mapName = {json.dumps(map_var)};
  var baseLayers = {json.dumps(base_layer_vars)};
  var basemapStorageKey = 'readiness.basemap';
  var activeBaseLayerId = 'positron';
  var metricLayers = {json.dumps(metric_group_vars)};
  var keypointLayers = {json.dumps(keypoint_group_vars)};
  var metricPanes = {json.dumps(metric_panes)};
  var keypointPanes = {json.dumps(keypoint_panes)};
  var activeMetricId = 'overall';
  var keypointsEnabled = true;
  var focusMarker = null;
  var pendingFocus = null;
  function getMap() {{
    return window[mapName] || null;
  }}
  function getLayer(name) {{
    return name ? (window[name] || null) : null;
  }}
  function getPane(name) {{
    var map = getMap();
    return (map && name) ? map.getPane(name) : null;
  }}
  function setControlVisibility(hidden) {{
    var panel = document.getElementById('map-controls-panel');
    var toggle = document.getElementById('map-controls-toggle');
    if (!panel || !toggle) return false;
    if (hidden) {{
      panel.style.display = 'none';
      toggle.textContent = 'Show Controls';
      toggle.setAttribute('aria-expanded', 'false');
    }} else {{
      panel.style.display = 'block';
      toggle.textContent = 'Hide Controls';
      toggle.setAttribute('aria-expanded', 'true');
    }}
    return true;
  }}
  function setBasemap(layerId) {{
    var map = getMap();
    if (!map) return false;
    activeBaseLayerId = layerId || 'positron';
    Object.keys(baseLayers).forEach(function(id) {{
      var layer = getLayer(baseLayers[id]);
      if (!layer) return;
      if (id === activeBaseLayerId) {{
        if (!map.hasLayer(layer)) map.addLayer(layer);
      }} else if (map.hasLayer(layer)) {{
        map.removeLayer(layer);
      }}
    }});
    var select = document.getElementById('basemap-select');
    if (select) select.value = activeBaseLayerId;
    try {{
      window.localStorage.setItem(basemapStorageKey, activeBaseLayerId);
    }} catch (e) {{}}
    return true;
  }}
  function getInitialBasemap() {{
    try {{
      var stored = window.localStorage.getItem(basemapStorageKey);
      if (stored && Object.prototype.hasOwnProperty.call(baseLayers, stored)) return stored;
    }} catch (e) {{}}
    return 'positron';
  }}
  function syncKeypointLayers() {{
    var map = getMap();
    if (!map) return false;
    Object.keys(keypointLayers).forEach(function(id) {{
      var layer = getLayer(keypointLayers[id]);
      if (!layer) return;
      var shouldShow = keypointsEnabled && id === activeMetricId;
      if (shouldShow) {{
        if (!map.hasLayer(layer)) map.addLayer(layer);
      }} else if (map.hasLayer(layer)) {{
        map.removeLayer(layer);
      }}
    }});
    var toggle = document.getElementById('keypoint-toggle');
    if (toggle) toggle.checked = keypointsEnabled;
    return true;
  }}
  function setMetric(metricId) {{
    var map = getMap();
    if (!map) return false;
    activeMetricId = metricId || 'overall';
    Object.keys(metricLayers).forEach(function(id) {{
      var layer = getLayer(metricLayers[id]);
      if (!layer) return;
      var shouldShow = id === activeMetricId;
      if (shouldShow) {{
        if (!map.hasLayer(layer)) map.addLayer(layer);
      }} else if (map.hasLayer(layer)) {{
        map.removeLayer(layer);
      }}
    }});
    syncKeypointLayers();
    var boxes = document.querySelectorAll('.metric-toggle');
    boxes.forEach(function(box) {{
      box.checked = box.value === activeMetricId;
    }});
    return true;
  }}
  function setKeypointsEnabled(enabled) {{
    keypointsEnabled = !!enabled;
    return syncKeypointLayers();
  }}
  function applyFocus(lat, lon, zoom, label) {{
    var map = getMap();
    if (!map) return false;
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return false;
    map.flyTo([lat, lon], zoom, {{duration: 0.8}});
    if (focusMarker) {{
      try {{ map.removeLayer(focusMarker); }} catch (e) {{}}
    }}
    focusMarker = L.circleMarker([lat, lon], {{
      radius: 9,
      color: '#0d47a1',
      weight: 2,
      fillColor: '#42a5f5',
      fillOpacity: 0.95
    }}).addTo(map);
    if (label) {{
      focusMarker.bindPopup(String(label), {{ maxWidth: 520, minWidth: 280 }}).openPopup();
    }}
    return true;
  }}
  function queueOrApply(lat, lon, zoom, label) {{
    if (!applyFocus(lat, lon, zoom, label)) {{
      pendingFocus = {{ lat: lat, lon: lon, zoom: zoom, label: label }};
    }}
  }}
  function flushPending() {{
    if (!pendingFocus) return;
    if (applyFocus(pendingFocus.lat, pendingFocus.lon, pendingFocus.zoom, pendingFocus.label)) {{
      pendingFocus = null;
    }}
  }}
  function parseHashFocus() {{
    var hash = String(window.location.hash || '');
    if (!hash.startsWith('#focus=')) return;
    var raw = hash.slice(7);
    var p1 = raw.indexOf(',');
    var p2 = raw.indexOf(',', p1 + 1);
    if (p1 < 0 || p2 < 0) return;
    var lat = Number(raw.slice(0, p1));
    var lon = Number(raw.slice(p1 + 1, p2));
    var label = '';
    try {{
      label = decodeURIComponent(raw.slice(p2 + 1));
    }} catch (e) {{
      label = raw.slice(p2 + 1);
    }}
    queueOrApply(lat, lon, 18, label);
  }}
  window.addEventListener('message', function(event) {{
    var data = event.data || {{}};
    if (!data || data.type !== 'focus_point') return;
    var lat = Number(data.lat), lon = Number(data.lon);
    var zoom = Number(data.zoom || 17);
    queueOrApply(lat, lon, zoom, data.label || '');
  }});
  window.addEventListener('hashchange', parseHashFocus);
  document.addEventListener('change', function(event) {{
    var target = event.target;
    if (!target) return;
    if (target.classList && target.classList.contains('metric-toggle')) {{
      setMetric(target.value || 'overall');
      return;
    }}
    if (target.id === 'basemap-select') {{
      setBasemap(target.value || 'positron');
      return;
    }}
    if (target.id === 'keypoint-toggle') {{
      setKeypointsEnabled(target.checked);
    }}
  }});
  document.addEventListener('click', function(event) {{
    var target = event.target;
    if (!target || target.id !== 'map-controls-toggle') return;
    var panel = document.getElementById('map-controls-panel');
    setControlVisibility(panel && panel.style.display !== 'none');
  }});
  setTimeout(parseHashFocus, 0);
  var readyTimer = setInterval(function() {{
    if (getMap()) {{
      setBasemap(getInitialBasemap());
      setMetric('overall');
      setKeypointsEnabled(true);
      setControlVisibility(false);
      flushPending();
      if (!pendingFocus) clearInterval(readyTimer);
    }}
  }}, 150);
}})();
</script>
"""
    m.get_root().html.add_child(folium.Element(focus_script))
    control_html = """
<div style="
  position:absolute;
  top:12px;
  left:12px;
  z-index:9999;
  font-family:Arial,sans-serif;
  font-size:13px;
  line-height:1.5;
">
  <button id="map-controls-toggle" type="button" aria-expanded="true" style="display:block;margin-bottom:8px;padding:6px 10px;border:1px solid #aab7c4;border-radius:8px;background:rgba(255,255,255,0.96);box-shadow:0 1px 6px rgba(0,0,0,0.18);font-weight:700;cursor:pointer;">Hide Controls</button>
  <div id="map-controls-panel" style="background:rgba(255,255,255,0.96);border:1px solid #c8c8c8;border-radius:8px;padding:10px 12px;box-shadow:0 1px 6px rgba(0,0,0,0.18);">
    <div style="font-weight:700;margin-bottom:6px;">Basemap</div>
    <select id="basemap-select" style="display:block;width:100%;margin-bottom:10px;padding:4px 6px;border:1px solid #c8c8c8;border-radius:6px;background:#fff;">
      <option value="positron">CartoDB Positron</option>
      <option value="gray">Esri Gray Canvas</option>
      <option value="voyager">CartoDB Voyager</option>
      <option value="dark">CartoDB Dark Matter</option>
      <option value="osm">OpenStreetMap</option>
      {('<option value="mapbox_light">Mapbox Light</option>' if mapbox_token else '')}
    </select>
    <div style="font-weight:700;margin-bottom:6px;">Heatmap Layer</div>
    <label style="display:block;"><input class="metric-toggle" name="metric-layer" type="radio" value="lane"> Lane Marking</label>
    <label style="display:block;"><input class="metric-toggle" name="metric-layer" type="radio" value="connectivity"> Connectivity</label>
    <label style="display:block;"><input class="metric-toggle" name="metric-layer" type="radio" value="gps"> GPS</label>
    <label style="display:block;"><input class="metric-toggle" name="metric-layer" type="radio" value="hd_map"> HD-Map</label>
    <label style="display:block;"><input class="metric-toggle" name="metric-layer" type="radio" value="overall" checked> Overall Score</label>
    <label style="display:block;margin-top:8px;border-top:1px solid #ddd;padding-top:8px;">
      <input id="keypoint-toggle" type="checkbox" checked> Show key point images
    </label>
    <div style="margin-top:6px;color:#666;font-size:11px;">Hide this panel for clean screenshots, then use Show Controls to bring it back.</div>
  </div>
</div>
"""
    m.get_root().html.add_child(folium.Element(control_html))
    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(output_html.as_posix())


def _pick_curves_for_selected_threshold(record: Dict[str, Any]) -> Tuple[Optional[float], List[List[float]]]:
    """Recover the lane curves corresponding to the selected scoring threshold."""
    lane_eval = (record.get("evaluation") or {}).get("lane") or {}
    selected_thr = _safe_float(lane_eval.get("selected_threshold"))
    best = lane_eval.get("best_threshold_detail") or {}
    curves = best.get("selected_curves")
    if isinstance(curves, list):
        return selected_thr, curves

    lane_raw = record.get("lane_raw")
    if not isinstance(lane_raw, dict):
        return selected_thr, []
    results = lane_raw.get("results")
    if not isinstance(results, list):
        return selected_thr, []

    if selected_thr is not None:
        for r in results:
            if not isinstance(r, dict):
                continue
            thr = _safe_float(r.get("thr"))
            if thr is None:
                continue
            if abs(thr - selected_thr) < 1e-9:
                fallback_curves = r.get("curves")
                if isinstance(fallback_curves, list):
                    return selected_thr, fallback_curves

    for r in sorted(
        [r for r in results if isinstance(r, dict)],
        key=lambda x: _safe_float(x.get("thr")) or 0.0,
        reverse=True,
    ):
        fallback_curves = r.get("curves")
        if isinstance(fallback_curves, list) and fallback_curves:
            return _safe_float(r.get("thr")), fallback_curves
    return selected_thr, []


def _overlay_lane_curves(
    image_path: Path,
    curves: List[List[float]],
    out_path: Path,
    max_width: int,
) -> bool:
    """Draw lane curves over an image and write the composited preview to disk."""
    img = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
    if img is None:
        return False

    overlay = img.copy()
    colors = [
        (0, 255, 0),
        (0, 255, 255),
        (0, 165, 255),
        (255, 255, 0),
        (255, 0, 255),
        (255, 0, 0),
    ]
    for idx, curve in enumerate(curves):
        if not isinstance(curve, list) or len(curve) < 4:
            continue
        pts = []
        for i in range(0, len(curve) - 1, 2):
            x = _safe_float(curve[i])
            y = _safe_float(curve[i + 1])
            if x is None or y is None:
                continue
            pts.append([int(round(x)), int(round(y))])
        if len(pts) < 2:
            continue
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [arr], isClosed=False, color=colors[idx % len(colors)], thickness=3)

    alpha = 0.9
    composed = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    h, w = composed.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        composed = cv2.resize(composed, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(out_path.as_posix(), composed))


def _key_point_indices(records: List[Dict[str, Any]], stride: int) -> List[int]:
    """Choose representative key-point indices for map popups and evidence cards."""
    n = len(records)
    if n == 0:
        return []
    idx = {0, n - 1}
    stride = max(1, stride)
    for i in range(0, n, stride):
        idx.add(i)
    for i in range(1, n):
        if records[i]["mile_index"] != records[i - 1]["mile_index"]:
            idx.add(i)
    return sorted(idx)


def _format_keypoint_tooltip_html(
    rec: Dict[str, Any],
    image_ref: Optional[str],
    metric_id: str,
    metric_label: str,
    metric_score: float,
) -> str:
    """Build the popup/tooltip HTML for one key point under a given metric layer."""
    lane_eval = (rec.get("evaluation") or {}).get("lane") or {}
    gps_eval = (rec.get("evaluation") or {}).get("gps") or {}
    conn_eval = (rec.get("evaluation") or {}).get("connectivity") or {}
    metric_readiness, _ = _readiness_label(metric_score)
    readiness, _ = _readiness_label(rec["overall_score"])

    er = _safe_float(gps_eval.get("er"))
    metrics = conn_eval.get("metrics") if isinstance(conn_eval.get("metrics"), dict) else {}

    lines = [
        f"<b>{rec['image']}</b>",
        f"Frame: {rec['frame_index']} | Mile: {rec['mile_marker']:.3f}",
        f"{metric_label}: {metric_score:.4f} ({metric_readiness})",
        f"Readiness: {rec['overall_score']:.4f} ({readiness})",
        f"Lane score: {rec['lane_score']:.4f}",
        f"GPS score: {rec['gps_score']:.4f} | er: {f'{er:.4f}' if er is not None else 'N/A'}",
        (
            "Connectivity score: "
            f"{rec['connectivity_score']:.4f} | d/u Mbps: "
            f"{_safe_float(metrics.get('avg_d_mbps'))} / {_safe_float(metrics.get('avg_u_mbps'))}"
        ),
        (
            "Latency dl/ul ms: "
            f"{_safe_float(metrics.get('avg_download_latency_ms'))} / "
            f"{_safe_float(metrics.get('avg_upload_latency_ms'))}"
        ),
    ]
    if metric_id == "hd_map":
        lines.append("HD-Map score is derived from the configured HD-map availability input.")
    tooltip_html = "<br>".join(lines)
    if image_ref:
        tooltip_html += (
            "<br><img src=\""
            + html.escape(image_ref, quote=True)
            + "\" style=\"width:360px;max-width:360px;border:1px solid #444;border-radius:4px;\"/>"
        )
    return tooltip_html


def _format_keypoint_score_lines(kp: Dict[str, Any]) -> str:
    """Format the compact score block shown in dashboard evidence cards."""
    return (
        f"Overall score: {float(kp.get('overall_score', 0.0)):.4f}<br>"
        f"Lane score: {float(kp.get('lane_score', 0.0)):.4f}<br>"
        f"GPS score: {float(kp.get('gps_score', 0.0)):.4f}<br>"
        f"Connectivity score: {float(kp.get('connectivity_score', 0.0)):.4f}"
    )


def _reduce_multi_run_key_points(key_points: List[Dict[str, Any]], max_points: int) -> List[Dict[str, Any]]:
    """Keep a representative subset of key points for the combined all-runs view."""
    if max_points <= 0 or len(key_points) <= max_points:
        return key_points

    chosen: set[int] = {0, len(key_points) - 1}
    scored = list(enumerate(key_points))

    # Preserve the weakest evidence first so low-readiness hotspots remain visible.
    worst_count = min(len(key_points), max(12, max_points // 3))
    for idx, _ in sorted(scored, key=lambda item: _safe_float(item[1].get("overall_score")) or 1.0)[:worst_count]:
        if len(chosen) >= max_points:
            break
        chosen.add(idx)

    # Fill the remaining budget with evenly spaced route coverage.
    if len(chosen) < max_points:
        remaining = max_points - len(chosen)
        step = max(1.0, len(key_points) / float(remaining))
        cursor = 0.0
        while len(chosen) < max_points and int(round(cursor)) < len(key_points):
            chosen.add(min(len(key_points) - 1, int(round(cursor))))
            cursor += step

    return [key_points[idx] for idx in sorted(chosen)]


def _resolve_keypoint_image_ref(
    kp: Dict[str, Any],
    overlay_dir: Path,
    lane_images_dir: Path,
    html_dir: Path,
) -> Optional[str]:
    """Resolve the best available key-point preview path relative to the generated HTML."""
    image_name = str(kp.get("image", "")).strip()
    if not image_name:
        return None
    candidates = [overlay_dir / image_name, lane_images_dir / image_name]
    for candidate in candidates:
        try:
            if candidate.exists():
                return os.path.relpath(candidate.as_posix(), html_dir.as_posix())
        except Exception:
            continue
    return None


def _downsample_route_points(points: List[Tuple[float, float]], max_points: int) -> List[Tuple[float, float]]:
    """Reduce route geometry density for browser rendering while preserving endpoints."""
    if max_points <= 0 or len(points) <= max_points:
        return points
    chosen: set[int] = {0, len(points) - 1}
    step = max(1.0, len(points) / float(max_points - 1))
    cursor = 0.0
    while len(chosen) < max_points and int(round(cursor)) < len(points):
        chosen.add(min(len(points) - 1, int(round(cursor))))
        cursor += step
    return [points[idx] for idx in sorted(chosen)]


def _metric_route_chunks(
    records: List[Dict[str, Any]],
    score_key: str,
    bucket_count: int = 12,
) -> List[Tuple[float, List[Tuple[float, float]]]]:
    """Group neighboring route segments into larger chunks for one metric heat layer."""
    if len(records) < 2:
        return []

    bucket_count = max(2, bucket_count)
    chunks: List[Tuple[float, List[Tuple[float, float]]]] = []
    current_bucket: Optional[int] = None
    current_score = 0.0
    current_points: List[Tuple[float, float]] = []

    for i in range(1, len(records)):
        prev = records[i - 1]
        cur = records[i]
        lat1 = _safe_float(prev.get("lat"))
        lon1 = _safe_float(prev.get("lon"))
        lat2 = _safe_float(cur.get("lat"))
        lon2 = _safe_float(cur.get("lon"))
        if None in (lat1, lon1, lat2, lon2):
            continue

        seg_score = (
            max(0.0, min(1.0, _safe_float(prev.get(score_key)) or 0.0))
            + max(0.0, min(1.0, _safe_float(cur.get(score_key)) or 0.0))
        ) / 2.0
        bucket = min(bucket_count - 1, int(seg_score * bucket_count))

        if current_bucket is None:
            current_bucket = bucket
            current_score = seg_score
            current_points = [(float(lat1), float(lon1)), (float(lat2), float(lon2))]
            continue

        if bucket == current_bucket:
            current_points.append((float(lat2), float(lon2)))
            current_score = seg_score
            continue

        if len(current_points) >= 2:
            chunks.append((current_score, current_points))
        current_bucket = bucket
        current_score = seg_score
        current_points = [(float(lat1), float(lon1)), (float(lat2), float(lon2))]

    if len(current_points) >= 2:
        chunks.append((current_score, current_points))
    return chunks


def _build_key_points(
    records: List[Dict[str, Any]],
    lane_images_dir: Path,
    overlay_dir: Path,
    keypoint_stride: int,
    max_preview_width: int,
    generate_missing_overlays: bool = True,
) -> List[Dict[str, Any]]:
    """Build key-point metadata and overlay previews for the map and dashboard."""
    key_points: List[Dict[str, Any]] = []
    for idx in _key_point_indices(records, stride=keypoint_stride):
        rec = records[idx]
        selected_thr, curves = _pick_curves_for_selected_threshold(rec)
        img_name = rec.get("image") or f"image_{rec['frame_index']}.jpg"
        img_path = lane_images_dir / img_name

        image_ref = None
        overlay_candidate = overlay_dir / img_name
        overlay_path: Optional[Path] = overlay_candidate if overlay_candidate.exists() else None
        if img_path.exists():
            if overlay_path is None and generate_missing_overlays:
                ok = _overlay_lane_curves(
                    image_path=img_path,
                    curves=curves,
                    out_path=overlay_candidate,
                    max_width=max_preview_width,
                )
                if ok and overlay_candidate.exists():
                    overlay_path = overlay_candidate
        if overlay_path is not None and overlay_path.exists():
            image_ref = os.path.relpath(overlay_path.as_posix(), overlay_dir.parent.as_posix())

        _, color = _readiness_label(rec["overall_score"])
        metric_tooltips = {}
        metric_colors = {}
        for metric in METRIC_LAYERS:
            score = max(0.0, min(1.0, _safe_float(rec.get(metric["record_key"])) or 0.0))
            metric_tooltips[metric["id"]] = _format_keypoint_tooltip_html(
                rec,
                image_ref,
                metric["id"],
                metric["label"],
                score,
            )
            metric_colors[metric["id"]] = _score_color(score)
        key_points.append(
            {
                "index": idx,
                "lat": rec["lat"],
                "lon": rec["lon"],
                "color": color,
                "frame_index": rec["frame_index"],
                "image": img_name,
                "mile_index": rec["mile_index"],
                "mile_marker": rec["mile_marker"],
                "selected_threshold": selected_thr,
                "lane_score": rec["lane_score"],
                "lane_length_pass_rate": rec.get("lane_length_pass_rate", 0.0),
                "lane_curvature_pass_rate": rec.get("lane_curvature_pass_rate", 0.0),
                "lane_curvature_soft_pass_rate": rec.get("lane_curvature_soft_pass_rate", 0.0),
                "lane_separation_pass_rate": rec.get("lane_separation_pass_rate", 0.0),
                "gps_score": rec["gps_score"],
                "connectivity_score": rec["connectivity_score"],
                "hd_maps_score": rec["hd_maps_score"],
                "overall_score": rec["overall_score"],
                "overlay_path": overlay_path.as_posix() if overlay_path is not None else None,
                "image_path": img_path.as_posix() if img_path.exists() else None,
                "tooltip_html": metric_tooltips.get("overall", ""),
                "tooltip_html_by_metric": metric_tooltips,
                "color_by_metric": metric_colors,
            }
        )
    return key_points


def _extract_eval_config(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract evaluation parameters to display in the dashboard when available."""
    if not records:
        return {}
    ev = records[0].get("evaluation") or {}
    lane_eval = ev.get("lane") or {}
    lane_params = lane_eval.get("params") or {}
    if not lane_params:
        best = lane_eval.get("best_threshold_detail") or {}
        length_check = best.get("length_check") or {}
        sep_check = best.get("separation_check") or {}
        curv_check = best.get("curvature_check") or {}
        lane_params = {
            "min_length_m": length_check.get("min_length_m"),
            "min_separation_m": sep_check.get("min_separation_m"),
            "max_curvature_1pm": curv_check.get("max_curvature_1pm"),
        }
    weights = ev.get("weights") or {}
    return {"lane_params": lane_params, "weights": weights}


def _build_dashboard_html(
    out_dashboard: Path,
    map_html_path: Path,
    readiness_json_path: Path,
    report_pdf_path: Path,
    total_miles: float,
    global_readiness: float,
    global_level: str,
    per_mile: List[Dict[str, Any]],
    key_points: List[Dict[str, Any]],
    eval_config: Dict[str, Any],
    pipeline_config: Dict[str, Any],
) -> None:
    """Write the split-screen readiness dashboard HTML that embeds the heatmap."""
    def _score_tone(score: float) -> Tuple[str, str, str]:
        level, _ = _readiness_label(score)
        if score <= 0.4:
            return level, "#b3261e", "#fde8e7"
        if score <= 0.8:
            return level, "#8a5a00", "#fff3d6"
        return level, "#17693a", "#e4f5e9"

    avg_lane = sum(x["lane_score_avg"] for x in per_mile) / max(1, len(per_mile))
    avg_gps = sum(x["gps_score_avg"] for x in per_mile) / max(1, len(per_mile))
    avg_conn = sum(x["connectivity_score_avg"] for x in per_mile) / max(1, len(per_mile))
    avg_hd_map = sum(x["hd_maps_score_avg"] for x in per_mile) / max(1, len(per_mile))

    reasons: List[str] = []
    if avg_lane < 0.6:
        reasons.append(f"Lane detection contribution is weak (avg {avg_lane:.3f} < 0.600).")
    if avg_gps < 0.6:
        reasons.append(f"GPS quality contribution is weak (avg {avg_gps:.3f} < 0.600).")
    if avg_conn < 0.6:
        reasons.append(f"Connectivity contribution is weak (avg {avg_conn:.3f} < 0.600).")
    if not reasons and global_readiness <= 0.8:
        reasons.append("Readiness is constrained by combined moderate scores across lane/GPS/connectivity.")

    low_miles = [m for m in per_mile if m["readiness_index"] <= 0.8]
    low_miles = sorted(low_miles, key=lambda x: x["readiness_index"])[:6]
    keypoint_by_mile: Dict[int, List[Dict[str, Any]]] = {}
    for kp in key_points:
        keypoint_by_mile.setdefault(int(kp["mile_index"]), []).append(kp)

    def _file_rel_for_html(path_str: Optional[str]) -> Optional[str]:
        if not path_str:
            return None
        try:
            p = Path(path_str)
            return os.path.relpath(p.as_posix(), out_dashboard.parent.as_posix())
        except Exception:
            return None

    table_rows = []
    for m in per_mile:
        tone_label, tone_fg, tone_bg = _score_tone(float(m["readiness_index"]))
        table_rows.append(
            f"<tr style=\"background:{tone_bg};\">"
            f"<td>{int(m['mile_index'])}</td>"
            f"<td>{m['readiness_index']:.3f}</td>"
            f"<td><span style=\"display:inline-block;padding:2px 8px;border-radius:999px;background:{tone_bg};color:{tone_fg};font-weight:700;\">{html.escape(tone_label)}</span></td>"
            f"<td>{m['lane_score_avg']:.3f}</td>"
            f"<td>{m['connectivity_score_avg']:.3f}</td>"
            f"<td>{m['gps_score_avg']:.3f}</td>"
            f"<td>{m['hd_maps_score_avg']:.3f}</td>"
            f"<td>{int(m['count'])}</td>"
            "</tr>"
        )

    map_rel = os.path.relpath(map_html_path.as_posix(), out_dashboard.parent.as_posix())
    readiness_rel = os.path.relpath(readiness_json_path.as_posix(), out_dashboard.parent.as_posix())
    report_rel = os.path.relpath(report_pdf_path.as_posix(), out_dashboard.parent.as_posix())
    report_results_dir = _path_for_ui(report_pdf_path.parent)
    report_regen_enabled = os.getenv("ENABLE_REPORT_REGEN", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
    report_action_html = (
        f' <span class="report-action"><button type="button" onclick=\'startReportRegen(this, {json.dumps(report_results_dir)}, {json.dumps("current report")})\' style="margin-left:6px;padding:4px 8px;border:1px solid rgba(255,255,255,0.35);border-radius:8px;background:rgba(255,255,255,0.14);color:#fff;cursor:pointer;">Re-run</button><span class="report-status" style="margin-left:6px;font-size:11px;opacity:0.9;"></span></span>'
        if report_regen_enabled
        else ' <span class="report-action"><button type="button" disabled title="Report regeneration is disabled in this view-only deployment." style="margin-left:6px;padding:4px 8px;border:1px solid #c9ced6;border-radius:8px;background:#e5e7eb;color:#7a828a;cursor:not-allowed;opacity:0.72;">Re-run</button><span class="report-status" style="margin-left:6px;font-size:11px;opacity:0.9;"></span></span>'
    )
    mile_summary_rows = []
    for m in per_mile:
        mile_summary_rows.append(
            "<tr>"
            f"<td>{int(m['mile_index'])}</td>"
            f"<td>{m['readiness_index']:.3f}</td>"
            f"<td>{html.escape(m['readiness_level'])}</td>"
            "</tr>"
        )

    overall_fg = "#b3261e" if global_readiness <= 0.4 else "#8a5a00" if global_readiness <= 0.8 else "#17693a"
    overall_bg = "#fde8e7" if global_readiness <= 0.4 else "#fff3d6" if global_readiness <= 0.8 else "#e4f5e9"

    component_cards = []
    for label, score, note in [
        ("Lane", avg_lane, "Visible lane structure quality"),
        ("Connectivity", avg_conn, "Digital network readiness"),
        ("GPS", avg_gps, "Localization confidence"),
        ("HD-Map", avg_hd_map, "Mapped-digital support"),
    ]:
        tone_label, tone_fg, tone_bg = _score_tone(score)
        component_cards.append(
            "<div class='score-card'>"
            f"<div class='score-card-title'>{html.escape(label)}</div>"
            f"<div class='score-card-value'>{score:.4f}</div>"
            f"<div class='score-pill' style='background:{tone_bg};color:{tone_fg};'>{html.escape(tone_label)}</div>"
            f"<div class='score-card-note'>{html.escape(note)}</div>"
            "</div>"
        )

    mile_strip_cells = []
    for m in per_mile:
        tone_label, tone_fg, tone_bg = _score_tone(float(m["readiness_index"]))
        mile_strip_cells.append(
            f"<div class='mile-strip-cell' title='Mile {int(m['mile_index'])}: {float(m['readiness_index']):.3f} ({html.escape(tone_label)})' "
            f"style='background:{tone_fg};'></div>"
        )

    low_mile_pills = []
    for mile in low_miles:
        tone_label, tone_fg, tone_bg = _score_tone(float(mile["readiness_index"]))
        low_mile_pills.append(
            "<span class='low-mile-pill' "
            f"style='background:{tone_bg};color:{tone_fg};border-color:{tone_fg};'>"
            f"Mile {int(mile['mile_index'])} · {float(mile['readiness_index']):.3f}"
            "</span>"
        )

    evidence_cards: List[str] = []
    for mile in low_miles:
        mile_idx = int(mile["mile_index"])
        mile_kps = keypoint_by_mile.get(mile_idx, [])
        items: List[str] = []
        max_imgs = 4 if mile["lane_score_avg"] < 0.6 else 2
        for kp in mile_kps[:max_imgs]:
            rel = _file_rel_for_html(kp.get("overlay_path")) or _file_rel_for_html(kp.get("image_path"))
            popup_image_html = (
                f'<img src="{html.escape(rel)}" '
                'loading="lazy" decoding="async" '
                'style="width:100%;max-width:340px;border:1px solid #bbb;border-radius:4px;margin:6px 0;" />'
                if rel
                else ""
            )
            score_lines_html = _format_keypoint_score_lines(kp)
            focus_popup_html = (
                f"<b>{html.escape(str(kp.get('image', '')))}</b><br>"
                f"frame {int(kp.get('frame_index', 0))}, mile {float(kp.get('mile_marker', 0.0)):.3f}<br>"
                f"{popup_image_html}"
                f"{score_lines_html}"
            )
            img = (
                (
                    f"<a href=\"{html.escape(map_rel)}#focus={float(kp.get('lat', 0.0))},{float(kp.get('lon', 0.0))},{quote(focus_popup_html, safe='')}\" "
                    "target=\"mapframe\" "
                    "class=\"evidence-link\" "
                    f"data-lat=\"{float(kp.get('lat', 0.0))}\" "
                    f"data-lon=\"{float(kp.get('lon', 0.0))}\" "
                    f"data-popup=\"{html.escape(focus_popup_html, quote=True)}\" "
                    "title=\"Click to zoom map and show evidence details\">"
                    f'<img src="{html.escape(rel)}" '
                    'loading="lazy" decoding="async" '
                    'style="width:100%;max-width:260px;border:1px solid #1e88e5;border-radius:6px;margin-top:6px;cursor:pointer;" />'
                    "</a>"
                )
                if rel
                else '<div style="font-size:12px;color:#777;margin-top:6px;">No preview image available.</div>'
            )
            items.append(
                "<div class='card' style='margin:8px 0;'>"
                f"<div><b>{html.escape(str(kp.get('image')))}</b> "
                f"(frame {int(kp.get('frame_index', 0))}, mile {float(kp.get('mile_marker', 0.0)):.3f})</div>"
                f"<div class='kv'>{score_lines_html}</div>"
                f"{img}"
                "</div>"
            )
        if not items:
            items.append("<div class='small'>No key-point evidence for this mile.</div>")
        evidence_cards.append(
            "<details class='card' open>"
            f"<summary><b>Mile {mile_idx}</b> readiness={mile['readiness_index']:.3f} ({mile['readiness_level']})</summary>"
            f"<div style='margin-top:8px'>{''.join(items)}</div>"
            "</details>"
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Road Readiness Dashboard</title>
  <style>
    html, body {{ margin: 0; height: 100%; font-family: Arial, sans-serif; }}
    .wrap {{ display: flex; height: 100vh; width: 100vw; }}
    .left {{ width: 38%; min-width: 360px; max-width: 620px; overflow: auto; padding: 14px; box-sizing: border-box; background: #f5f6f8; border-right: 1px solid #d9d9d9; }}
    .right {{ flex: 1; min-width: 0; }}
    .mapframe {{ width: 100%; height: 100%; border: none; }}
    h1 {{ font-size: 20px; margin: 0 0 10px 0; }}
    h2 {{ font-size: 15px; margin: 16px 0 8px 0; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 10px; }}
    .kv {{ font-size: 13px; line-height: 1.5; }}
    .hero-card {{ padding: 14px; border-radius: 14px; background: linear-gradient(135deg, #16344f 0%, #245f83 100%); color: #fff; margin-bottom: 12px; }}
    .hero-eyebrow {{ font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.78; }}
    .hero-score {{ display:flex; align-items:flex-end; gap:10px; margin-top:8px; }}
    .hero-value {{ font-size: 36px; line-height: 1; font-weight: 800; }}
    .hero-meta {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:12px; }}
    .hero-meta-card {{ background: rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.16); border-radius:10px; padding:8px 10px; }}
    .score-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:12px 0; }}
    .score-card {{ background:#fff; border:1px solid #d7dee5; border-radius:12px; padding:12px; }}
    .score-card-title {{ font-size:12px; color:#52606d; text-transform:uppercase; letter-spacing:0.06em; }}
    .score-card-value {{ font-size:24px; font-weight:800; color:#162535; margin-top:4px; }}
    .score-pill {{ display:inline-block; margin-top:8px; padding:3px 8px; border-radius:999px; font-size:11px; font-weight:700; }}
    .score-card-note {{ margin-top:8px; font-size:12px; color:#60707c; }}
    .mile-strip {{ margin-top:10px; }}
    .mile-strip-scroll {{ overflow-x:auto; overflow-y:hidden; padding-bottom:6px; }}
    .mile-strip-track {{ display:grid; grid-template-columns: repeat({max(1, len(per_mile))}, minmax(12px, 1fr)); gap:3px; min-width:max-content; }}
    .mile-strip-cell {{ height:18px; border-radius:4px; min-width:12px; }}
    .mile-strip-scale {{ display:flex; justify-content:space-between; gap:8px; margin-top:8px; font-size:11px; color:#60707c; }}
    .low-mile-pills {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }}
    .low-mile-pill {{ display:inline-block; padding:4px 8px; border-radius:999px; border:1px solid; font-size:11px; font-weight:700; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
    th {{ background: #fafafa; position: sticky; top: 0; }}
    .tag {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; color: #fff; background: {"#c62828" if global_readiness <= 0.4 else "#f9a825" if global_readiness <= 0.8 else "#2e7d32"}; }}
    .small {{ font-size: 12px; color: #666; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    details.card > summary {{ cursor: pointer; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="left">
      <h1>Road Readiness Dashboard</h1>
      <details class="card" open>
        <summary><b>1) Summary (Overall + Per-Mile Readiness)</b></summary>
        <div class="hero-card" style="margin-top:8px;">
          <div class="hero-eyebrow">Road Readiness Overview</div>
          <div class="hero-score">
            <div class="hero-value">{global_readiness:.4f}</div>
            <span class="tag" style="background:{overall_bg};color:{overall_fg};">{html.escape(global_level)}</span>
          </div>
          <div class="hero-meta">
            <div class="hero-meta-card"><b>Total Distance</b><br>{total_miles:.3f} miles</div>
            <div class="hero-meta-card"><b>Mile Bins</b><br>{len(per_mile)}</div>
            <div class="hero-meta-card"><b>Readiness JSON</b><br><span class="mono">{html.escape(readiness_rel)}</span></div>
            <div class="hero-meta-card"><b>PDF Report</b><br><a href="{html.escape(report_rel)}" target="_blank" rel="noopener noreferrer" style="color:#fff;">report.pdf</a>{report_action_html}</div>
          </div>
        </div>
        <div class="score-grid">
          {''.join(component_cards)}
        </div>
        <div class="card">
          <div style="font-weight:700;">Per-Mile Readiness Strip</div>
          <div class="small" style="margin-top:4px;">Compact scan of route readiness from start to end. Hover any block for the mile and score.</div>
          <div class="mile-strip">
            <div class="mile-strip-scroll">
              <div class="mile-strip-track">
                {''.join(mile_strip_cells)}
              </div>
            </div>
            <div class="mile-strip-scale">
              <span>Start</span>
              <span>{len(per_mile)} mile bins</span>
              <span>End</span>
            </div>
          </div>
        </div>
        <div style="max-height: 260px; overflow:auto; margin-top:8px;">
          <table>
            <thead><tr><th>Mile</th><th>Readiness</th><th>Level</th></tr></thead>
            <tbody>{''.join(mile_summary_rows)}</tbody>
          </table>
        </div>
      </details>

      <details class="card" open>
        <summary><b>2) Detailed Evaluation Per Mile</b></summary>
        <div style="max-height: 340px; overflow:auto; margin-top:8px;">
          <table>
            <thead>
              <tr><th>Mile</th><th>Overall</th><th>Status</th><th>Lane</th><th>Conn</th><th>GPS</th><th>HD</th><th>N</th></tr>
            </thead>
            <tbody>{''.join(table_rows)}</tbody>
          </table>
        </div>
      </details>

      <details class="card" open>
        <summary><b>3) Evidence For Low Scores</b></summary>
        <div class="card kv" style="margin-top:8px;">
          {'<br>'.join(html.escape(r) for r in reasons) if reasons else 'No dominant low-score driver detected.'}
        </div>
        <div class="low-mile-pills">
          {''.join(low_mile_pills) if low_mile_pills else '<span class="small">No low-score mile bins highlighted.</span>'}
        </div>
        {''.join(evidence_cards) if evidence_cards else '<div class="small">No low-score evidence found.</div>'}
      </details>

    </div>
    <div class="right">
      <iframe class="mapframe" name="mapframe" src="{html.escape(map_rel)}"></iframe>
    </div>
  </div>
<script>
function focusMapPoint(lat, lon, label) {{
  var frame = document.querySelector('.mapframe');
  if (!frame) return;
  var payload = {{
    type: 'focus_point',
    lat: Number(lat),
    lon: Number(lon),
    zoom: 18,
    label: label || ''
  }};
  var post = function() {{
    if (!frame.contentWindow) return false;
    frame.contentWindow.postMessage(payload, '*');
    return true;
  }};
  if (!post()) {{
    frame.addEventListener('load', function() {{ post(); }}, {{ once: true }});
  }}
}}

document.addEventListener('DOMContentLoaded', function() {{
  var links = document.querySelectorAll('.evidence-link');
  links.forEach(function(link) {{
    link.addEventListener('click', function(evt) {{
      evt.preventDefault();
      var lat = Number(link.getAttribute('data-lat'));
      var lon = Number(link.getAttribute('data-lon'));
      var popup = link.getAttribute('data-popup') || '';
      focusMapPoint(lat, lon, popup);
    }});
  }});
}});

let reportStatusTimer = null;

function normalizeReportPath(path) {{
  return String(path || '').replace(/\\\\/g, '/').replace(/\/+$/, '');
}}

function updateReportStatus(button, text, color) {{
  if (!button) return;
  const status = button.parentElement ? button.parentElement.querySelector('.report-status') : null;
  if (status) {{
    status.textContent = text || '';
    status.style.color = color || '';
    status.style.fontWeight = '700';
  }}
}}

async function watchReportRegen(button, resultsDir) {{
  if (reportStatusTimer) {{
    clearTimeout(reportStatusTimer);
    reportStatusTimer = null;
  }}
  const wanted = normalizeReportPath(resultsDir);
  try {{
    const res = await fetch('/api/status');
    const j = await res.json();
    const active = normalizeReportPath((j.params || {{}}).results_dir || '');
    const sameReport = wanted && active && wanted === active;
    if (j.running && j.run_kind === 'report' && sameReport) {{
      button.disabled = true;
      updateReportStatus(button, j.progress_text || 'Running...', '#ffd54f');
      reportStatusTimer = setTimeout(() => watchReportRegen(button, resultsDir), 1500);
      return;
    }}
    if (sameReport && j.stage === 'completed') {{
      button.disabled = false;
      updateReportStatus(button, 'Completed', '#ffd54f');
      return;
    }}
    if (sameReport && j.stage === 'failed') {{
      button.disabled = false;
      updateReportStatus(button, 'Failed', '#b3261e');
      return;
    }}
    if (sameReport && j.stage === 'stopped') {{
      button.disabled = false;
      updateReportStatus(button, 'Stopped', '#ffd54f');
      return;
    }}
    button.disabled = false;
  }} catch (err) {{
    button.disabled = false;
    updateReportStatus(button, 'Status unavailable', '#b3261e');
  }}
}}

async function startReportRegen(button, resultsDir, label) {{
  if (button) {{
    button.disabled = true;
    updateReportStatus(button, 'Starting...', '#ffd54f');
  }}
  try {{
    const res = await fetch('/api/report', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ results_dir: resultsDir }})
    }});
    let j = {{}};
    try {{
      j = await res.json();
    }} catch (err) {{
      j = {{}};
    }}
    if (!res.ok || !j.ok) {{
      if (button) {{
        button.disabled = false;
        updateReportStatus(
          button,
          res.status === 404 ? 'Restart UI server' : (j.error || 'Failed to start'),
          '#b3261e'
        );
      }}
      return;
    }}
    updateReportStatus(button, `Running ${{label || 'report'}}...`, '#ffd54f');
    watchReportRegen(button, resultsDir);
  }} catch (err) {{
    if (button) {{
      button.disabled = false;
      updateReportStatus(button, 'Request failed', '#b3261e');
    }}
  }}
}}
</script>
</body>
</html>
"""
    out_dashboard.parent.mkdir(parents=True, exist_ok=True)
    with out_dashboard.open("w", encoding="utf-8") as f:
        f.write(html_doc)




def _build_multi_run_map(
    run_payloads: List[Dict[str, Any]],
    output_html: Path,
    mapbox_token: Optional[str],
) -> None:
    """Render one readiness map containing all available evaluation runs."""
    all_points = [
        (float(rec["lat"]), float(rec["lon"]))
        for run in run_payloads
        for rec in run.get("records", [])
        if rec.get("lat") is not None and rec.get("lon") is not None
    ]
    if not all_points:
        all_points = [
            (float(row["center_lat"]), float(row["center_lon"]))
            for run in run_payloads
            for row in run.get("per_mile", [])
            if row.get("center_lat") is not None and row.get("center_lon") is not None
        ]
    if not all_points:
        raise FileNotFoundError("No evaluated readiness records available for the combined map.")

    center_lat = sum(p[0] for p in all_points) / len(all_points)
    center_lon = sum(p[1] for p in all_points) / len(all_points)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True, tiles=None)

    base_layers: Dict[str, folium.TileLayer] = {}

    def _register_base_layer(layer_id: str, **kwargs: Any) -> None:
        layer = folium.TileLayer(control=False, **kwargs)
        layer.add_to(m)
        base_layers[layer_id] = layer

    _register_base_layer(
        "positron",
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="CartoDB Positron",
        max_zoom=20,
        overlay=False,
        subdomains="abcd",
    )
    _register_base_layer(
        "voyager",
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="CartoDB Voyager",
        max_zoom=20,
        overlay=False,
        subdomains="abcd",
        show=False,
    )
    _register_base_layer(
        "gray",
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Esri World Gray Canvas",
        max_zoom=16,
        overlay=False,
        show=False,
    )
    _register_base_layer(
        "dark",
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="CartoDB Dark Matter",
        max_zoom=20,
        overlay=False,
        subdomains="abcd",
        show=False,
    )
    _register_base_layer(
        "osm",
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        show=False,
    )
    if mapbox_token:
        _register_base_layer(
            "mapbox_light",
            tiles=(
                "https://api.mapbox.com/styles/v1/mapbox/light-v11/tiles/"
                "{z}/{x}/{y}?access_token=" + mapbox_token
            ),
            attr="Mapbox",
            max_zoom=19,
            tile_size=512,
            zoom_offset=-1,
            name="Mapbox Light",
            overlay=False,
            show=False,
        )

    colormap = LinearColormap(colors=["red", "yellow", "green"], vmin=0.0, vmax=1.0)
    colormap.caption = "Infrastructure Score"
    colormap.add_to(m)

    route_group = folium.FeatureGroup(name="Run Routes", show=True, control=False)
    route_group.add_to(m)
    route_anchor_group = folium.FeatureGroup(name="Run Anchors", show=True, control=False)
    route_anchor_group.add_to(m)
    run_route_colors = [
        "#1565c0",
        "#2e7d32",
        "#ef6c00",
        "#6a1b9a",
        "#c62828",
        "#00838f",
        "#5d4037",
    ]

    metric_groups: Dict[str, folium.FeatureGroup] = {}
    keypoint_groups: Dict[str, folium.FeatureGroup] = {}
    metric_panes: Dict[str, str] = {}
    keypoint_panes: Dict[str, str] = {}
    for metric in METRIC_LAYERS:
        metric_pane = f"multiMetricPane_{metric['id']}"
        keypoint_pane = f"multiKeypointPane_{metric['id']}"
        folium.map.CustomPane(metric_pane, z_index=410).add_to(m)
        folium.map.CustomPane(keypoint_pane, z_index=430).add_to(m)
        metric_group = folium.FeatureGroup(
            name=f"{metric['label']} Heatmap",
            show=True,
            control=False,
        )
        metric_group.add_to(m)
        metric_groups[metric["id"]] = metric_group
        metric_panes[metric["id"]] = metric_pane

        keypoint_group = folium.FeatureGroup(
            name=f"{metric['label']} Key Points",
            show=True,
            control=False,
        )
        keypoint_group.add_to(m)
        keypoint_groups[metric["id"]] = keypoint_group
        keypoint_panes[metric["id"]] = keypoint_pane

    for run_idx, run in enumerate(run_payloads):
        records = run.get("records", [])
        per_mile = run.get("per_mile", [])
        route_points = [(r["lat"], r["lon"]) for r in records if r.get("lat") is not None and r.get("lon") is not None]
        if not route_points:
            route_points = [
                (float(row["center_lat"]), float(row["center_lon"]))
                for row in per_mile
                if row.get("center_lat") is not None and row.get("center_lon") is not None
            ]
        if not route_points:
            continue
        run_label = str(run.get("label", "Run"))
        route_color = run_route_colors[run_idx % len(run_route_colors)]
        run_bounds = run.get("bounds") or {}
        summary_html = (
            f"<b>{html.escape(run_label)}</b><br>"
            f"Readiness: {float(run.get('global_readiness', 0.0)):.4f} "
            f"({html.escape(str(run.get('global_level', 'Unknown')))})<br>"
            f"Distance: {float(run.get('total_miles', 0.0)):.3f} miles<br>"
            f"Points: {int(run.get('total_points', 0))}"
        )
        folium.PolyLine(
            locations=route_points,
            color=route_color,
            weight=3,
            opacity=0.28,
            tooltip=f"{run_label} route",
        ).add_to(route_group)
        start_label = summary_html + "<br>Route anchor: start"
        end_label = summary_html + "<br>Route anchor: end"
        folium.Marker(
            location=route_points[0],
            tooltip=f"{run_label} start",
            popup=folium.Popup(start_label, max_width=360),
            icon=folium.Icon(color="blue", icon="play"),
        ).add_to(route_anchor_group)
        folium.Marker(
            location=route_points[-1],
            tooltip=f"{run_label} end",
            popup=folium.Popup(end_label, max_width=360),
            icon=folium.Icon(color="purple", icon="stop"),
        ).add_to(route_anchor_group)

        if run_bounds:
            center_lat = (float(run_bounds.get("south", 0.0)) + float(run_bounds.get("north", 0.0))) / 2.0
            center_lon = (float(run_bounds.get("west", 0.0)) + float(run_bounds.get("east", 0.0))) / 2.0
            folium.CircleMarker(
                location=[center_lat, center_lon],
                radius=8,
                color="#0d47a1",
                weight=2,
                fill=True,
                fill_color="#90caf9",
                fill_opacity=0.85,
                tooltip=f"{run_label} route focus",
                popup=folium.Popup(summary_html + "<br>Route anchor: focus", max_width=360),
            ).add_to(route_anchor_group)

        for metric in METRIC_LAYERS:
            metric_group = metric_groups[metric["id"]]
            if records:
                for seg_score, seg_points in _metric_route_chunks(records, metric["record_key"]):
                    folium.PolyLine(
                        locations=seg_points,
                        color=colormap(seg_score),
                        weight=6,
                        opacity=0.9,
                        tooltip=f"{run_label} | {metric['label']}: {seg_score:.3f}",
                        pane=metric_panes[metric["id"]],
                    ).add_to(metric_group)

        for row in per_mile:
            mile_index = int(row["mile_index"])
            for metric in METRIC_LAYERS:
                metric_value = max(0.0, min(1.0, _safe_float(row.get(metric["mile_key"])) or 0.0))
                metric_level, metric_color = _readiness_label(metric_value)
                popup = (
                    f"<b>{html.escape(run_label)}</b><br>"
                    f"Mile {mile_index}<br>"
                    f"{metric['label']}: {metric_value:.3f} ({metric_level})<br>"
                    f"Lane: {row['lane_score_avg']:.3f}, GPS: {row['gps_score_avg']:.3f}, "
                    f"Connectivity: {row['connectivity_score_avg']:.3f}, HD-Map: {row['hd_maps_score_avg']:.3f}<br>"
                    f"Overall: {row['readiness_index']:.3f} ({row['readiness_level']})<br>"
                    f"Samples: {row['count']}"
                )
                folium.CircleMarker(
                    location=[row["center_lat"], row["center_lon"]],
                    radius=5,
                    color=metric_color,
                    fill=True,
                    fill_opacity=0.82,
                    popup=folium.Popup(popup, max_width=380),
                    tooltip=f"{run_label} | Mile {mile_index} | {metric['label']}: {metric_level}",
                    pane=metric_panes[metric["id"]],
                ).add_to(metric_groups[metric["id"]])

        for kp in run.get("key_points", []):
            for metric in METRIC_LAYERS:
                metric_id = metric["id"]
                tooltip_html = (kp.get("tooltip_html_by_metric") or {}).get(metric_id, kp.get("tooltip_html", ""))
                popup_html = (kp.get("popup_html_by_metric") or {}).get(metric_id, tooltip_html)
                folium.CircleMarker(
                    location=[kp["lat"], kp["lon"]],
                    radius=7,
                    color=(kp.get("color_by_metric") or {}).get(metric_id, kp["color"]),
                    fill=True,
                    fill_opacity=0.95,
                    weight=2,
                    tooltip=folium.Tooltip(tooltip_html, sticky=False),
                    popup=folium.Popup(popup_html, max_width=460),
                    pane=keypoint_panes[metric_id],
                ).add_to(keypoint_groups[metric_id])

    folium.LayerControl(position="topright", collapsed=True).add_to(m)
    map_var = m.get_name()
    base_layer_vars = {layer_id: layer.get_name() for layer_id, layer in base_layers.items()}
    metric_group_vars = {metric_id: group.get_name() for metric_id, group in metric_groups.items()}
    keypoint_group_vars = {metric_id: group.get_name() for metric_id, group in keypoint_groups.items()}
    focus_script = f"""
<script>
(function() {{
  var mapName = {json.dumps(map_var)};
  var baseLayers = {json.dumps(base_layer_vars)};
  var basemapStorageKey = 'readiness.basemap';
  var activeBaseLayerId = 'positron';
  var metricLayers = {json.dumps(metric_group_vars)};
  var keypointLayers = {json.dumps(keypoint_group_vars)};
  var metricPanes = {json.dumps(metric_panes)};
  var keypointPanes = {json.dumps(keypoint_panes)};
  var activeMetricId = 'overall';
  var keypointsEnabled = true;
  var focusMarker = null;
  var pendingAction = null;
  function getMap() {{
    return window[mapName] || null;
  }}
  function getLayer(name) {{
    return name ? (window[name] || null) : null;
  }}
  function getPane(name) {{
    var map = getMap();
    return (map && name) ? map.getPane(name) : null;
  }}
  function setControlVisibility(hidden) {{
    var panel = document.getElementById('map-controls-panel');
    var toggle = document.getElementById('map-controls-toggle');
    if (!panel || !toggle) return false;
    if (hidden) {{
      panel.style.display = 'none';
      toggle.textContent = 'Show Controls';
      toggle.setAttribute('aria-expanded', 'false');
    }} else {{
      panel.style.display = 'block';
      toggle.textContent = 'Hide Controls';
      toggle.setAttribute('aria-expanded', 'true');
    }}
    return true;
  }}
  function setBasemap(layerId) {{
    var map = getMap();
    if (!map) return false;
    activeBaseLayerId = layerId || 'positron';
    Object.keys(baseLayers).forEach(function(id) {{
      var layer = getLayer(baseLayers[id]);
      if (!layer) return;
      if (id === activeBaseLayerId) {{
        if (!map.hasLayer(layer)) map.addLayer(layer);
      }} else if (map.hasLayer(layer)) {{
        map.removeLayer(layer);
      }}
    }});
    var select = document.getElementById('basemap-select');
    if (select) select.value = activeBaseLayerId;
    try {{
      window.localStorage.setItem(basemapStorageKey, activeBaseLayerId);
    }} catch (e) {{}}
    return true;
  }}
  function getInitialBasemap() {{
    try {{
      var stored = window.localStorage.getItem(basemapStorageKey);
      if (stored && Object.prototype.hasOwnProperty.call(baseLayers, stored)) return stored;
    }} catch (e) {{}}
    return 'positron';
  }}
  function syncKeypointLayers() {{
    var map = getMap();
    if (!map) return false;
    Object.keys(keypointLayers).forEach(function(id) {{
      var layer = getLayer(keypointLayers[id]);
      if (!layer) return;
      var shouldShow = keypointsEnabled && id === activeMetricId;
      if (shouldShow) {{
        if (!map.hasLayer(layer)) map.addLayer(layer);
      }} else if (map.hasLayer(layer)) {{
        map.removeLayer(layer);
      }}
    }});
    var toggle = document.getElementById('keypoint-toggle');
    if (toggle) toggle.checked = keypointsEnabled;
    return true;
  }}
  function setMetric(metricId) {{
    var map = getMap();
    if (!map) return false;
    activeMetricId = metricId || 'overall';
    Object.keys(metricLayers).forEach(function(id) {{
      var layer = getLayer(metricLayers[id]);
      if (!layer) return;
      var shouldShow = id === activeMetricId;
      if (shouldShow) {{
        if (!map.hasLayer(layer)) map.addLayer(layer);
      }} else if (map.hasLayer(layer)) {{
        map.removeLayer(layer);
      }}
    }});
    syncKeypointLayers();
    var boxes = document.querySelectorAll('.metric-toggle');
    boxes.forEach(function(box) {{
      box.checked = box.value === activeMetricId;
    }});
    return true;
  }}
  function setKeypointsEnabled(enabled) {{
    keypointsEnabled = !!enabled;
    return syncKeypointLayers();
  }}
  function placeFocusMarker(lat, lon, label) {{
    var map = getMap();
    if (!map) return false;
    if (focusMarker) {{
      try {{ map.removeLayer(focusMarker); }} catch (e) {{}}
    }}
    focusMarker = L.circleMarker([lat, lon], {{
      radius: 9,
      color: '#0d47a1',
      weight: 2,
      fillColor: '#42a5f5',
      fillOpacity: 0.95
    }}).addTo(map);
    if (label) {{
      focusMarker.bindPopup(String(label), {{ maxWidth: 520, minWidth: 280 }}).openPopup();
    }}
    return true;
  }}
  function applyFocus(lat, lon, zoom, label) {{
    var map = getMap();
    if (!map) return false;
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return false;
    map.flyTo([lat, lon], zoom, {{duration: 0.8}});
    return placeFocusMarker(lat, lon, label);
  }}
  function applyBounds(south, west, north, east, label) {{
    var map = getMap();
    if (!map) return false;
    if (![south, west, north, east].every(Number.isFinite)) return false;
    var bounds = L.latLngBounds([[south, west], [north, east]]);
    map.flyToBounds(bounds, {{padding:[40,40], duration:0.8}});
    var center = bounds.getCenter();
    return placeFocusMarker(center.lat, center.lng, label);
  }}
  function queueAction(action) {{
    pendingAction = action;
    flushPending();
  }}
  function flushPending() {{
    if (!pendingAction) return;
    var ok = false;
    if (pendingAction.type === 'focus_point') {{
      ok = applyFocus(
        Number(pendingAction.lat),
        Number(pendingAction.lon),
        Number(pendingAction.zoom || 17),
        pendingAction.label || ''
      );
    }} else if (pendingAction.type === 'focus_bounds') {{
      ok = applyBounds(
        Number(pendingAction.south),
        Number(pendingAction.west),
        Number(pendingAction.north),
        Number(pendingAction.east),
        pendingAction.label || ''
      );
    }}
    if (ok) pendingAction = null;
  }}
  window.addEventListener('message', function(event) {{
    var data = event.data || {{}};
    if (!data || (data.type !== 'focus_point' && data.type !== 'focus_bounds')) return;
    queueAction(data);
  }});
  document.addEventListener('change', function(event) {{
    var target = event.target;
    if (!target) return;
    if (target.classList && target.classList.contains('metric-toggle')) {{
      setMetric(target.value || 'overall');
      return;
    }}
    if (target.id === 'basemap-select') {{
      setBasemap(target.value || 'positron');
      return;
    }}
    if (target.id === 'keypoint-toggle') {{
      setKeypointsEnabled(target.checked);
    }}
  }});
  document.addEventListener('click', function(event) {{
    var target = event.target;
    if (!target || target.id !== 'map-controls-toggle') return;
    var panel = document.getElementById('map-controls-panel');
    setControlVisibility(panel && panel.style.display !== 'none');
  }});
  var readyTimer = setInterval(function() {{
    if (getMap()) {{
      setBasemap(getInitialBasemap());
      setMetric('overall');
      setKeypointsEnabled(true);
      setControlVisibility(false);
      flushPending();
      if (!pendingAction) clearInterval(readyTimer);
    }}
  }}, 150);
}})();
</script>
"""
    m.get_root().html.add_child(folium.Element(focus_script))
    control_html = """
<div style="
  position:absolute;
  top:12px;
  left:12px;
  z-index:9999;
  font-family:Arial,sans-serif;
  font-size:13px;
  line-height:1.5;
">
  <button id="map-controls-toggle" type="button" aria-expanded="true" style="display:block;margin-bottom:8px;padding:6px 10px;border:1px solid #aab7c4;border-radius:8px;background:rgba(255,255,255,0.96);box-shadow:0 1px 6px rgba(0,0,0,0.18);font-weight:700;cursor:pointer;">Hide Controls</button>
  <div id="map-controls-panel" style="background:rgba(255,255,255,0.96);border:1px solid #c8c8c8;border-radius:8px;padding:10px 12px;box-shadow:0 1px 6px rgba(0,0,0,0.18);">
    <div style="font-weight:700;margin-bottom:6px;">Basemap</div>
    <select id="basemap-select" style="display:block;width:100%;margin-bottom:10px;padding:4px 6px;border:1px solid #c8c8c8;border-radius:6px;background:#fff;">
      <option value="positron">CartoDB Positron</option>
      <option value="gray">Esri Gray Canvas</option>
      <option value="voyager">CartoDB Voyager</option>
      <option value="dark">CartoDB Dark Matter</option>
      <option value="osm">OpenStreetMap</option>
      {('<option value="mapbox_light">Mapbox Light</option>' if mapbox_token else '')}
    </select>
    <div style="font-weight:700;margin-bottom:6px;">Heatmap Layer</div>
    <label style="display:block;"><input class="metric-toggle" name="metric-layer" type="radio" value="lane"> Lane Marking</label>
    <label style="display:block;"><input class="metric-toggle" name="metric-layer" type="radio" value="connectivity"> Connectivity</label>
    <label style="display:block;"><input class="metric-toggle" name="metric-layer" type="radio" value="gps"> GPS</label>
    <label style="display:block;"><input class="metric-toggle" name="metric-layer" type="radio" value="hd_map"> HD-Map</label>
    <label style="display:block;"><input class="metric-toggle" name="metric-layer" type="radio" value="overall" checked> Overall Score</label>
    <label style="display:block;margin-top:8px;border-top:1px solid #ddd;padding-top:8px;">
      <input id="keypoint-toggle" type="checkbox" checked> Show key point images
    </label>
    <div style="margin-top:6px;color:#666;font-size:11px;">All runs stay on the same map. Use the sidebar cards to jump to one run at a time.</div>
  </div>
</div>
"""
    m.get_root().html.add_child(folium.Element(control_html))
    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(output_html.as_posix())


def _build_multi_run_dashboard_html(
    out_dashboard: Path,
    map_html_path: Path,
    run_payloads: List[Dict[str, Any]],
) -> None:
    """Write the split-screen multi-run dashboard HTML."""
    map_rel = os.path.relpath(map_html_path.as_posix(), out_dashboard.parent.as_posix())
    total_runs = len(run_payloads)
    total_points = sum(int(run.get("total_points", 0)) for run in run_payloads)
    total_miles = sum(float(run.get("total_miles", 0.0)) for run in run_payloads)

    run_cards: List[str] = []
    for idx, run in enumerate(run_payloads):
        label = str(run.get("label", "Run"))
        score = float(run.get("global_readiness", 0.0))
        level = str(run.get("global_level", "Unknown"))
        level_fg = "#b3261e" if score <= 0.4 else "#8a5a00" if score <= 0.8 else "#17693a"
        level_bg = "#fde8e7" if score <= 0.4 else "#fff3d6" if score <= 0.8 else "#e4f5e9"
        bounds = run.get("bounds") or {}
        center = run.get("center") or {}
        report_path = run.get("report_pdf")
        report_rel = (
            os.path.relpath(Path(report_path).as_posix(), out_dashboard.parent.as_posix()) if report_path else None
        )
        report_results_dir = _path_for_ui(Path(report_path).parent) if report_path else None
        report_html = "Unavailable"
        report_regen_enabled = os.getenv("ENABLE_REPORT_REGEN", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
        if report_rel and report_results_dir:
            report_action_html = ""
            if report_regen_enabled:
                report_action_html = (
                    f'<span class="report-action"><button type="button" onclick=\'startReportRegen(this, {json.dumps(report_results_dir)}, {json.dumps(label + " report")})\' '
                    'style="margin-left:6px;padding:4px 8px;border:1px solid #c7d0d9;border-radius:8px;background:#fff;cursor:pointer;">'
                    'Re-run</button><span class="report-status" style="margin-left:6px;font-size:11px;"></span></span>'
                )
            else:
                report_action_html = (
                    '<span class="report-action"><button type="button" disabled title="Report regeneration is disabled in this view-only deployment." '
                    'style="margin-left:6px;padding:4px 8px;border:1px solid #c9ced6;border-radius:8px;background:#e5e7eb;color:#7a828a;cursor:not-allowed;opacity:0.72;">'
                    'Re-run</button><span class="report-status" style="margin-left:6px;font-size:11px;"></span></span>'
                )
            report_html = (
                f'<a href="{html.escape(report_rel)}" target="_blank" rel="noopener noreferrer">report.pdf</a> '
                f'{report_action_html}'
            )
        avg_lane = sum(x["lane_score_avg"] for x in run.get("per_mile", [])) / max(1, len(run.get("per_mile", [])))
        avg_conn = sum(x["connectivity_score_avg"] for x in run.get("per_mile", [])) / max(1, len(run.get("per_mile", [])))
        avg_gps = sum(x["gps_score_avg"] for x in run.get("per_mile", [])) / max(1, len(run.get("per_mile", [])))
        avg_hd = sum(x["hd_maps_score_avg"] for x in run.get("per_mile", [])) / max(1, len(run.get("per_mile", [])))
        low_miles = sorted(
            [mile for mile in run.get("per_mile", []) if float(mile.get("readiness_index", 0.0)) <= 0.8],
            key=lambda mile: float(mile.get("readiness_index", 0.0)),
        )[:6]
        mile_buttons = []
        for mile in run.get("per_mile", []):
            popup = (
                f"<b>{html.escape(label)}</b><br>"
                f"Mile {int(mile['mile_index'])}<br>"
                f"Overall: {float(mile['readiness_index']):.3f} ({html.escape(str(mile['readiness_level']))})<br>"
                f"Lane: {float(mile['lane_score_avg']):.3f}, GPS: {float(mile['gps_score_avg']):.3f}, "
                f"Connectivity: {float(mile['connectivity_score_avg']):.3f}, HD-Map: {float(mile['hd_maps_score_avg']):.3f}"
            )
            mile_buttons.append(
                "<button class='mile-chip' "
                f"data-lat='{float(mile['center_lat'])}' "
                f"data-lon='{float(mile['center_lon'])}' "
                f"data-label='{html.escape(popup, quote=True)}'>"
                f"Mile {int(mile['mile_index'])} · {float(mile['readiness_index']):.3f}"
                "</button>"
            )
        low_mile_pills = []
        for mile in low_miles:
            low_mile_pills.append(
                f"<span class='low-pill'>Mile {int(mile['mile_index'])} · {float(mile['readiness_index']):.3f}</span>"
            )
        table_rows = []
        for mile in run.get("per_mile", []):
            table_rows.append(
                "<tr>"
                f"<td>{int(mile['mile_index'])}</td>"
                f"<td>{float(mile['readiness_index']):.3f}</td>"
                f"<td>{float(mile['lane_score_avg']):.3f}</td>"
                f"<td>{float(mile['connectivity_score_avg']):.3f}</td>"
                f"<td>{float(mile['gps_score_avg']):.3f}</td>"
                f"<td>{float(mile['hd_maps_score_avg']):.3f}</td>"
                f"<td>{int(mile['count'])}</td>"
                "</tr>"
            )
        route_button_attrs = (
            f"data-south='{float(bounds.get('south', 0.0))}' "
            f"data-west='{float(bounds.get('west', 0.0))}' "
            f"data-north='{float(bounds.get('north', 0.0))}' "
            f"data-east='{float(bounds.get('east', 0.0))}' "
            f"data-label='{html.escape(f'<b>{label}</b><br>Readiness: {score:.4f} ({level})', quote=True)}'"
        )
        start_button_attrs = (
            f"data-lat='{float(run.get('start_lat', center.get('lat', 0.0)))}' "
            f"data-lon='{float(run.get('start_lon', center.get('lon', 0.0)))}' "
            f"data-label='{html.escape(f'<b>{label}</b><br>Route anchor: start', quote=True)}'"
        )
        end_button_attrs = (
            f"data-lat='{float(run.get('end_lat', center.get('lat', 0.0)))}' "
            f"data-lon='{float(run.get('end_lon', center.get('lon', 0.0)))}' "
            f"data-label='{html.escape(f'<b>{label}</b><br>Route anchor: end', quote=True)}'"
        )
        run_cards.append(
            f"""
<details class="card run-card" {'open' if idx == 0 else ''}>
  <summary>
    <span>{html.escape(label)}</span>
    <span class="run-pill" style="background:{level_bg};color:{level_fg};">{score:.4f} · {html.escape(level)}</span>
  </summary>
  <div class="run-meta">
    <div class="metric-grid">
      <div class="metric-card"><b>Distance</b><span>{float(run.get('total_miles', 0.0)):.3f} mi</span></div>
      <div class="metric-card"><b>Mile Bins</b><span>{len(run.get('per_mile', []))}</span></div>
      <div class="metric-card"><b>Lane</b><span>{avg_lane:.3f}</span></div>
      <div class="metric-card"><b>Connectivity</b><span>{avg_conn:.3f}</span></div>
      <div class="metric-card"><b>GPS</b><span>{avg_gps:.3f}</span></div>
      <div class="metric-card"><b>HD-Map</b><span>{avg_hd:.3f}</span></div>
    </div>
    <div class="focus-row">
      <button class="focus-route-btn" {route_button_attrs}>Focus Route</button>
      <button class="focus-point-btn" {start_button_attrs}>Start</button>
      <button class="focus-point-btn" {end_button_attrs}>End</button>
    </div>
    <div class="small">Each evaluation stays separate here. This view only places all routes on one shared map for comparison.</div>
    <div class="low-pills">{''.join(low_mile_pills) if low_mile_pills else '<span class="small">No low-score mile bins.</span>'}</div>
    <div class="mile-chip-row">{''.join(mile_buttons) if mile_buttons else '<span class="small">No per-mile summary available.</span>'}</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Mile</th><th>Overall</th><th>Lane</th><th>Conn</th><th>GPS</th><th>HD</th><th>N</th></tr></thead>
        <tbody>{''.join(table_rows)}</tbody>
      </table>
    </div>
    <div class="small">Report: {report_html}</div>
  </div>
</details>
"""
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>All Evaluated Results</title>
  <style>
    html, body {{ margin:0; height:100%; font-family:Arial,sans-serif; }}
    .wrap {{ display:flex; height:100vh; width:100vw; }}
    .left {{ width:38%; min-width:380px; max-width:640px; overflow:auto; padding:14px; box-sizing:border-box; background:#f5f6f8; border-right:1px solid #d9d9d9; }}
    .right {{ flex:1; min-width:0; }}
    .mapframe {{ width:100%; height:100%; border:none; }}
    .hero {{ background:linear-gradient(135deg, #16344f 0%, #245f83 100%); color:#fff; border-radius:14px; padding:16px; margin-bottom:12px; }}
    .hero h1 {{ margin:0; font-size:20px; }}
    .hero p {{ margin:8px 0 0 0; font-size:13px; line-height:1.5; opacity:0.92; }}
    .hero-grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; margin-top:12px; }}
    .hero-stat {{ background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.18); border-radius:10px; padding:10px; }}
    .hero-stat b {{ display:block; font-size:12px; text-transform:uppercase; letter-spacing:0.06em; opacity:0.82; }}
    .hero-stat span {{ display:block; margin-top:6px; font-size:18px; font-weight:800; }}
    .card {{ background:#fff; border:1px solid #ddd; border-radius:10px; padding:10px; margin-bottom:10px; }}
    details.card > summary {{ cursor:pointer; display:flex; align-items:center; justify-content:space-between; gap:12px; font-weight:700; }}
    .run-pill {{ display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:700; }}
    .run-meta {{ margin-top:10px; }}
    .metric-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
    .metric-card {{ border:1px solid #dde3ea; border-radius:10px; padding:10px; background:#fafbfd; }}
    .metric-card b {{ display:block; font-size:12px; color:#4f6271; text-transform:uppercase; letter-spacing:0.05em; }}
    .metric-card span {{ display:block; margin-top:6px; font-size:20px; font-weight:800; color:#173042; }}
    .focus-row {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }}
    button {{ padding:8px 10px; border:1px solid #c7d0d9; border-radius:8px; background:#fff; cursor:pointer; }}
    button:hover {{ background:#f4f7fb; }}
    .mile-chip-row {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }}
    .mile-chip {{ font-size:12px; }}
    .low-pills {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }}
    .low-pill {{ display:inline-block; padding:4px 8px; border-radius:999px; background:#fff3d6; color:#8a5a00; border:1px solid #c89b33; font-size:11px; font-weight:700; }}
    .table-wrap {{ margin-top:10px; max-height:240px; overflow:auto; }}
    table {{ border-collapse:collapse; width:100%; background:#fff; font-size:12px; }}
    th, td {{ border:1px solid #ddd; padding:6px; text-align:left; }}
    th {{ background:#fafafa; position:sticky; top:0; }}
    .small {{ font-size:12px; color:#60707c; margin-top:8px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="left">
      <div class="hero">
        <h1>All Evaluated Results</h1>
        <p>Every evaluation route is rendered on the same map, while each run keeps its own readiness summary, mile bins, and focus controls in separate collapsible cards.</p>
        <div class="hero-grid">
          <div class="hero-stat"><b>Runs</b><span>{total_runs}</span></div>
          <div class="hero-stat"><b>Total Points</b><span>{total_points}</span></div>
          <div class="hero-stat"><b>Total Distance</b><span>{total_miles:.3f} mi</span></div>
        </div>
      </div>
      {''.join(run_cards) if run_cards else '<div class="card">No evaluated runs available.</div>'}
    </div>
    <div class="right">
      <iframe class="mapframe" name="allRunsMapFrame" src="{html.escape(map_rel)}"></iframe>
    </div>
  </div>
<script>
function postToMap(payload) {{
  var frame = document.querySelector('.mapframe');
  if (!frame) return;
  var send = function() {{
    if (!frame.contentWindow) return false;
    frame.contentWindow.postMessage(payload, '*');
    return true;
  }};
  if (!send()) {{
    frame.addEventListener('load', function() {{ send(); }}, {{ once:true }});
  }}
}}

document.addEventListener('DOMContentLoaded', function() {{
  document.querySelectorAll('.focus-route-btn').forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      postToMap({{
        type: 'focus_bounds',
        south: Number(btn.getAttribute('data-south')),
        west: Number(btn.getAttribute('data-west')),
        north: Number(btn.getAttribute('data-north')),
        east: Number(btn.getAttribute('data-east')),
        label: btn.getAttribute('data-label') || ''
      }});
    }});
  }});
  document.querySelectorAll('.focus-point-btn, .mile-chip').forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      postToMap({{
        type: 'focus_point',
        lat: Number(btn.getAttribute('data-lat')),
        lon: Number(btn.getAttribute('data-lon')),
        zoom: 17,
        label: btn.getAttribute('data-label') || ''
      }});
    }});
  }});
}});

let reportStatusTimer = null;

function normalizeReportPath(path) {{
  return String(path || '').replace(/\\\\/g, '/').replace(/\/+$/, '');
}}

function updateReportStatus(button, text, color) {{
  if (!button) return;
  const status = button.parentElement ? button.parentElement.querySelector('.report-status') : null;
  if (status) {{
    status.textContent = text || '';
    status.style.color = color || '';
    status.style.fontWeight = '700';
  }}
}}

async function watchReportRegen(button, resultsDir) {{
  if (reportStatusTimer) {{
    clearTimeout(reportStatusTimer);
    reportStatusTimer = null;
  }}
  const wanted = normalizeReportPath(resultsDir);
  try {{
    const res = await fetch('/api/status');
    const j = await res.json();
    const active = normalizeReportPath((j.params || {{}}).results_dir || '');
    const sameReport = wanted && active && wanted === active;
    if (j.running && j.run_kind === 'report' && sameReport) {{
      button.disabled = true;
      updateReportStatus(button, j.progress_text || 'Running...', '#ffd54f');
      reportStatusTimer = setTimeout(() => watchReportRegen(button, resultsDir), 1500);
      return;
    }}
    if (sameReport && j.stage === 'completed') {{
      button.disabled = false;
      updateReportStatus(button, 'Completed', '#17693a');
      return;
    }}
    if (sameReport && j.stage === 'failed') {{
      button.disabled = false;
      updateReportStatus(button, 'Failed', '#b3261e');
      return;
    }}
    if (sameReport && j.stage === 'stopped') {{
      button.disabled = false;
      updateReportStatus(button, 'Stopped', '#ffd54f');
      return;
    }}
    button.disabled = false;
  }} catch (err) {{
    button.disabled = false;
    updateReportStatus(button, 'Status unavailable', '#b3261e');
  }}
}}

async function startReportRegen(button, resultsDir, label) {{
  if (button) {{
    button.disabled = true;
    updateReportStatus(button, 'Starting...', '#ffd54f');
  }}
  try {{
    const res = await fetch('/api/report', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ results_dir: resultsDir }})
    }});
    let j = {{}};
    try {{
      j = await res.json();
    }} catch (err) {{
      j = {{}};
    }}
    if (!res.ok || !j.ok) {{
      if (button) {{
        button.disabled = false;
        updateReportStatus(
          button,
          res.status === 404 ? 'Restart UI server' : (j.error || 'Failed to start'),
          '#b3261e'
        );
      }}
      return;
    }}
    updateReportStatus(button, `Running ${{label || 'report'}}...`, '#ffd54f');
    watchReportRegen(button, resultsDir);
  }} catch (err) {{
    if (button) {{
      button.disabled = false;
      updateReportStatus(button, 'Request failed', '#b3261e');
    }}
  }}
}}
</script>
</body>
</html>
"""
    out_dashboard.parent.mkdir(parents=True, exist_ok=True)
    with out_dashboard.open("w", encoding="utf-8") as f:
        f.write(html_doc)


def build_multi_run_dashboard_bundle(
    run_specs: List[Dict[str, Any]],
    out_map: Path,
    out_dashboard: Path,
    mapbox_token: Optional[str],
    default_keypoint_stride: int = 20,
    preview_width: int = 360,
    max_combined_key_points_per_run: int = 120,
) -> List[Dict[str, Any]]:
    """Generate one combined dashboard/map bundle from multiple evaluation runs."""
    run_payloads: List[Dict[str, Any]] = []
    for spec in run_specs:
        evaluated_dir = Path(str(spec.get("evaluated_dir", "")))
        lane_images_dir = Path(str(spec.get("lane_images_dir", "")))
        overlay_dir = Path(str(spec.get("overlay_dir", "")))
        if not evaluated_dir.exists():
            continue
        records = _load_records(evaluated_dir)
        if not records:
            continue
        per_mile, total_miles = _aggregate_per_mile(records)
        keypoint_stride = int(spec.get("keypoint_stride", default_keypoint_stride) or default_keypoint_stride)
        key_points = _build_key_points(
            records=records,
            lane_images_dir=lane_images_dir,
            overlay_dir=overlay_dir,
            keypoint_stride=keypoint_stride,
            max_preview_width=preview_width,
            generate_missing_overlays=False,
        )
        for kp in key_points:
            rec = records[int(kp.get("index", 0))]
            image_ref = None
            overlay_path = kp.get("overlay_path")
            image_path = kp.get("image_path")
            try:
                if overlay_path and Path(str(overlay_path)).exists():
                    image_ref = os.path.relpath(Path(str(overlay_path)).as_posix(), out_map.parent.as_posix())
                elif image_path and Path(str(image_path)).exists():
                    image_ref = os.path.relpath(Path(str(image_path)).as_posix(), out_map.parent.as_posix())
            except Exception:
                image_ref = None
            metric_tooltips: Dict[str, str] = {}
            for metric in METRIC_LAYERS:
                score = max(0.0, min(1.0, _safe_float(rec.get(metric["record_key"])) or 0.0))
                metric_tooltips[metric["id"]] = _format_keypoint_tooltip_html(
                    rec,
                    image_ref,
                    metric["id"],
                    metric["label"],
                    score,
                )
            kp["tooltip_html_by_metric"] = metric_tooltips
            kp["tooltip_html"] = metric_tooltips.get("overall", kp.get("tooltip_html", ""))
        key_points = _reduce_multi_run_key_points(key_points, max_combined_key_points_per_run)
        global_readiness = sum(r["overall_score"] for r in records) / len(records)
        level, _ = _readiness_label(global_readiness)
        lats = [r["lat"] for r in records]
        lons = [r["lon"] for r in records]
        run_payloads.append(
            {
                "id": str(spec.get("id", evaluated_dir.name)),
                "label": str(spec.get("label", evaluated_dir.name)),
                "records": records,
                "per_mile": per_mile,
                "key_points": key_points,
                "global_readiness": global_readiness,
                "global_level": level,
                "total_miles": total_miles,
                "total_points": len(records),
                "start_lat": float(records[0]["lat"]),
                "start_lon": float(records[0]["lon"]),
                "end_lat": float(records[-1]["lat"]),
                "end_lon": float(records[-1]["lon"]),
                "center": {
                    "lat": sum(lats) / len(lats),
                    "lon": sum(lons) / len(lons),
                },
                "bounds": {
                    "south": min(lats),
                    "west": min(lons),
                    "north": max(lats),
                    "east": max(lons),
                },
                "report_pdf": str(spec.get("report_pdf", "")) or None,
            }
        )

    if not run_payloads:
        raise FileNotFoundError("No evaluated runs with usable readiness data were found.")

    _build_multi_run_map(run_payloads=run_payloads, output_html=out_map, mapbox_token=mapbox_token)
    _build_multi_run_dashboard_html(out_dashboard=out_dashboard, map_html_path=out_map, run_payloads=run_payloads)
    return run_payloads


def build_multi_run_dashboard_bundle_from_readiness_specs(
    run_specs: List[Dict[str, Any]],
    out_map: Path,
    out_dashboard: Path,
    mapbox_token: Optional[str],
    max_combined_key_points_per_run: int = 120,
) -> List[Dict[str, Any]]:
    """Generate the combined dashboard/map bundle from lightweight readiness JSON files."""
    run_payloads: List[Dict[str, Any]] = []
    for spec in run_specs:
        readiness_json = Path(str(spec.get("readiness_json", "")))
        if not readiness_json.exists():
            continue
        with readiness_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        per_mile = payload.get("per_mile") or []
        if not per_mile:
            continue
        key_points_raw = payload.get("key_points") or []
        key_points: List[Dict[str, Any]] = []
        overlay_dir = Path(str(spec.get("overlay_dir", "")))
        lane_images_dir = Path(str(spec.get("lane_images_dir", "")))
        for kp in key_points_raw:
            overall_score = max(0.0, min(1.0, _safe_float(kp.get("overall_score")) or 0.0))
            image_ref = _resolve_keypoint_image_ref(
                kp=kp,
                overlay_dir=overlay_dir,
                lane_images_dir=lane_images_dir,
                html_dir=out_map.parent,
            )
            frame_index = int(kp.get("frame_index", 0) or 0)
            mile_marker = _safe_float(kp.get("mile_marker"))
            mile_label = f"{mile_marker:.3f}" if mile_marker is not None else "N/A"
            metric_tooltips: Dict[str, str] = {}
            metric_popups: Dict[str, str] = {}
            metric_colors: Dict[str, str] = {}
            for metric in METRIC_LAYERS:
                score = max(0.0, min(1.0, _safe_float(kp.get(metric["record_key"])) or 0.0))
                metric_tooltips[metric["id"]] = (
                    f"{html.escape(str(spec.get('label', 'Run')))} | "
                    f"{html.escape(str(kp.get('image', 'key point')))} | "
                    f"{metric['label']}: {score:.4f}"
                )
                metric_popups[metric["id"]] = (
                    f"<div><b>{html.escape(str(spec.get('label', 'Run')))}</b></div>"
                    f"<b>{html.escape(str(kp.get('image', 'key point')))}</b><br>"
                    f"Frame: {frame_index} | Mile: {mile_label}<br>"
                    f"{metric['label']}: {score:.4f}<br>"
                    f"Overall score: {overall_score:.4f}"
                    + (
                        "<br><img src=\""
                        + html.escape(image_ref, quote=True)
                        + "\" loading=\"lazy\" decoding=\"async\" "
                        + "style=\"width:280px;max-width:280px;border:1px solid #444;border-radius:4px;margin-top:8px;\"/>"
                        if image_ref
                        else ""
                    )
                )
                metric_colors[metric["id"]] = _score_color(score)
            key_points.append(
                {
                    "lat": float(kp.get("lat", 0.0)),
                    "lon": float(kp.get("lon", 0.0)),
                    "color": _score_color(overall_score),
                    "tooltip_html": metric_tooltips.get("overall", ""),
                    "tooltip_html_by_metric": metric_tooltips,
                    "popup_html_by_metric": metric_popups,
                    "color_by_metric": metric_colors,
                }
            )
        key_points = _reduce_multi_run_key_points(key_points, max_combined_key_points_per_run)

        center_points = [
            (float(row["center_lat"]), float(row["center_lon"]))
            for row in per_mile
            if row.get("center_lat") is not None and row.get("center_lon") is not None
        ]
        if not center_points:
            continue
        lats = [p[0] for p in center_points]
        lons = [p[1] for p in center_points]
        run_payloads.append(
            {
                "id": str(spec.get("id", readiness_json.parent.parent.name)),
                "label": str(spec.get("label", readiness_json.parent.parent.name)),
                "records": [],
                "per_mile": per_mile,
                "key_points": key_points,
                "global_readiness": float(payload.get("global_readiness_index", 0.0)),
                "global_level": str(payload.get("global_readiness_level", "Unknown")),
                "total_miles": float(payload.get("total_distance_miles", 0.0)),
                "total_points": int(payload.get("total_points", 0)),
                "start_lat": lats[0],
                "start_lon": lons[0],
                "end_lat": lats[-1],
                "end_lon": lons[-1],
                "center": {
                    "lat": sum(lats) / len(lats),
                    "lon": sum(lons) / len(lons),
                },
                "bounds": {
                    "south": min(lats),
                    "west": min(lons),
                    "north": max(lats),
                    "east": max(lons),
                },
                "report_pdf": str(spec.get("report_pdf", "")) or None,
            }
        )

    if not run_payloads:
        raise FileNotFoundError("No readiness summaries with usable data were found.")

    _build_multi_run_map(run_payloads=run_payloads, output_html=out_map, mapbox_token=mapbox_token)
    _build_multi_run_dashboard_html(out_dashboard=out_dashboard, map_html_path=out_map, run_payloads=run_payloads)
    return run_payloads


def run(
    input_dir: Path,
    out_json: Path,
    out_map: Path,
    report_pdf_path: Path,
    mapbox_token: Optional[str],
    lane_images_dir: Path,
    overlay_dir: Path,
    keypoint_stride: int,
    preview_width: int,
    out_dashboard: Path,
    open_browser: bool,
) -> None:
    """Generate readiness JSON, map HTML, and dashboard HTML from evaluated inputs."""
    records = _load_records(input_dir)
    if not records:
        raise FileNotFoundError(f"No usable evaluated JSON records with gps+evaluation found in: {input_dir}")

    per_mile, total_miles = _aggregate_per_mile(records)
    key_points = _build_key_points(
        records=records,
        lane_images_dir=lane_images_dir,
        overlay_dir=overlay_dir,
        keypoint_stride=keypoint_stride,
        max_preview_width=preview_width,
    )
    global_readiness = sum(r["overall_score"] for r in records) / len(records)
    level, color = _readiness_label(global_readiness)

    source_fused_dir = input_dir.parent / "fused"
    result = {
        "input_dir": input_dir.as_posix(),
        "total_points": len(records),
        "total_distance_miles": total_miles,
        "global_readiness_index": global_readiness,
        "global_readiness_level": level,
        "global_color": color,
        "per_mile": per_mile,
        "key_points_count": len(key_points),
        "key_points": [
            {
                "index": kp["index"],
                "frame_index": kp["frame_index"],
                "image": kp["image"],
                "lat": kp["lat"],
                "lon": kp["lon"],
                "color": kp["color"],
                "mile_index": kp["mile_index"],
                "mile_marker": kp["mile_marker"],
                "selected_threshold": kp["selected_threshold"],
                "lane_score": kp["lane_score"],
                "lane_length_pass_rate": kp.get("lane_length_pass_rate", 0.0),
                "lane_curvature_pass_rate": kp.get("lane_curvature_pass_rate", 0.0),
                "lane_curvature_soft_pass_rate": kp.get("lane_curvature_soft_pass_rate", 0.0),
                "lane_separation_pass_rate": kp.get("lane_separation_pass_rate", 0.0),
                "gps_score": kp["gps_score"],
                "connectivity_score": kp["connectivity_score"],
                "overall_score": kp["overall_score"],
            }
            for kp in key_points
        ],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    _build_map(
        records=records,
        per_mile=per_mile,
        key_points=key_points,
        output_html=out_map,
        mapbox_token=mapbox_token,
    )
    _build_dashboard_html(
        out_dashboard=out_dashboard,
        map_html_path=out_map,
        readiness_json_path=out_json,
        report_pdf_path=report_pdf_path,
        total_miles=total_miles,
        global_readiness=global_readiness,
        global_level=level,
        per_mile=per_mile,
        key_points=key_points,
        eval_config=_extract_eval_config(records),
        pipeline_config={
            "input_dir": _path_for_ui(input_dir),
            "source_fused_dir": _path_for_ui(source_fused_dir),
            "lane_images_dir": _path_for_ui(lane_images_dir),
            "source_lane_images_dir": _path_for_ui(lane_images_dir),
            "overlay_dir": _path_for_ui(overlay_dir),
            "keypoint_stride": keypoint_stride,
            "preview_width": preview_width,
            "mapbox_token_set": bool(mapbox_token),
        },
    )
    print(f"Wrote aggregated per-mile readiness JSON: {out_json}")
    print(f"Wrote readiness heat map HTML: {out_map}")
    print(f"Wrote split-screen dashboard HTML: {out_dashboard}")

    if open_browser:
        try:
            webbrowser.open(out_dashboard.resolve().as_uri(), new=2)
            print("Opened dashboard in browser.")
        except Exception as exc:
            print(f"Could not open browser automatically: {exc}")


def main() -> None:
    """CLI entry point for readiness heatmap and dashboard generation."""
    parser = argparse.ArgumentParser(
        description="Aggregate readiness score per mile from evaluated outputs and generate a map heat view."
    )
    parser.add_argument("--input-dir", default="results/evaluated", help="Directory with evaluated JSON files.")
    parser.add_argument(
        "--output-json",
        default="results/readiness/readiness_per_mile.json",
        help="Path to save aggregated per-mile readiness JSON.",
    )
    parser.add_argument(
        "--output-map",
        default="results/readiness/readiness_heatmap.html",
        help="Path to save interactive readiness map HTML.",
    )
    parser.add_argument(
        "--output-dashboard",
        default="results/readiness/readiness_dashboard.html",
        help="Path to save split-screen dashboard HTML.",
    )
    parser.add_argument(
        "--report-pdf",
        default="results/research_report.pdf",
        help="PDF report path used for dashboard link.",
    )
    parser.add_argument(
        "--mapbox-token-env",
        default="MAPBOX_TOKEN",
        help="Environment variable name that stores your Mapbox token.",
    )
    parser.add_argument(
        "--lane-images-dir",
        default="results/lane/images",
        help="Directory with lane images used for key-point visual overlays.",
    )
    parser.add_argument(
        "--overlay-dir",
        default="results/readiness/keypoint_overlays",
        help="Directory to save overlay previews used in key-point hover cards.",
    )
    parser.add_argument(
        "--keypoint-stride",
        type=int,
        default=20,
        help="Select one key point every N records (plus start/end and mile transitions).",
    )
    parser.add_argument(
        "--preview-width",
        type=int,
        default=360,
        help="Preview width in pixels for key-point hover images.",
    )
    parser.add_argument("--open-browser", action="store_true", help="Open the generated HTML map in a browser.")
    args = parser.parse_args()

    token = os.getenv(args.mapbox_token_env)
    run(
        input_dir=Path(args.input_dir),
        out_json=Path(args.output_json),
        out_map=Path(args.output_map),
        report_pdf_path=Path(args.report_pdf),
        mapbox_token=token,
        lane_images_dir=Path(args.lane_images_dir),
        overlay_dir=Path(args.overlay_dir),
        keypoint_stride=int(args.keypoint_stride),
        preview_width=int(args.preview_width),
        out_dashboard=Path(args.output_dashboard),
        open_browser=bool(args.open_browser),
    )


if __name__ == "__main__":
    main()
