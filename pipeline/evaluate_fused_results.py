#!/usr/bin/env python3
"""Score fused lane, GPS, and connectivity records for readiness evaluation."""

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple


Point = Tuple[float, float]
Mat3 = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]

DEFAULT_LANE_HOMOGRAPHY: Mat3 = (
    (-0.862608526825564, -4.439095599323822, 596.0347002043012),
    (0.0, -12.17826427554507, 1356.171517157721),
    (4.82495910821212e-19, -0.01387217397944601, 1.0),
)


def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning ``None`` when conversion fails."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_homography(value: str) -> Mat3:
    """Parse a serialized 3x3 homography matrix from CLI/config text."""
    cleaned = value.strip().replace("[", "").replace("]", "").replace("\n", " ")
    parts = [x.strip() for x in cleaned.replace(";", ",").split(",") if x.strip()]
    vals = [float(x) for x in parts]
    if len(vals) != 9:
        raise ValueError("lane_homography must contain exactly 9 numeric values")
    return (
        (vals[0], vals[1], vals[2]),
        (vals[3], vals[4], vals[5]),
        (vals[6], vals[7], vals[8]),
    )


def _apply_homography(h: Mat3, x: float, y: float) -> Optional[Point]:
    """Project one 2D point through a homography matrix."""
    d = (h[2][0] * x) + (h[2][1] * y) + h[2][2]
    if abs(d) < 1e-9:
        return None
    xw = ((h[0][0] * x) + (h[0][1] * y) + h[0][2]) / d
    yw = ((h[1][0] * x) + (h[1][1] * y) + h[1][2]) / d
    return (xw, yw)


def _polyline_points(curve: Sequence[float], px_to_m: float, homography: Optional[Mat3]) -> List[Point]:
    """Convert a flat lane-curve list into geometry points for scoring."""
    pts: List[Point] = []
    for i in range(0, len(curve) - 1, 2):
        x = _safe_float(curve[i])
        y = _safe_float(curve[i + 1])
        if x is None or y is None:
            continue
        if homography is not None:
            p = _apply_homography(homography, x, y)
            if p is None:
                continue
            pts.append((p[0], p[1]))
        else:
            pts.append((x * px_to_m, y * px_to_m))
    return pts


def _polyline_length_m(points: Sequence[Point]) -> float:
    """Compute polyline length in the current evaluation geometry space."""
    total = 0.0
    for i in range(1, len(points)):
        total += math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
    return total


def _polyline_curvature_1pm(points: Sequence[Point]) -> float:
    """Estimate curvature from point triplets and return a robust median value."""
    # Discrete curvature estimate from triplets. Median keeps it stable against outliers.
    values: List[float] = []
    for i in range(1, len(points) - 1):
        y1, x1 = points[i - 1]
        y2, x2 = points[i]
        y3, x3 = points[i + 1]
        a = math.hypot(x2 - x1, y2 - y1)
        b = math.hypot(x3 - x2, y3 - y2)
        c = math.hypot(x3 - x1, y3 - y1)
        if min(a, b, c) < 1e-9:
            continue
        area2 = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        kappa = 2.0 * area2 / (a * b * c)
        values.append(kappa)
    return float(median(values)) if values else 0.0


def _interp_x_at_y(points: Sequence[Point], y: float) -> Optional[float]:
    """Interpolate the lateral position of a lane curve at a given longitudinal coordinate."""
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        if (y1 <= y <= y2) or (y2 <= y <= y1):
            if abs(y2 - y1) < 1e-9:
                return x1
            t = (y - y1) / (y2 - y1)
            return x1 + t * (x2 - x1)
    return None


def _summary_stats(values: Sequence[float]) -> Dict[str, Any]:
    """Return basic summary stats used in diagnostic output."""
    vals = [float(v) for v in values]
    if not vals:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(vals),
        "min": min(vals),
        "median": float(median(vals)),
        "max": max(vals),
    }


def _pair_median_separation_m(points_a: Sequence[Point], points_b: Sequence[Point], y_min: float, y_max: float) -> Optional[float]:
    """Estimate median lane separation over the shared longitudinal span of two curves."""
    if y_max < y_min:
        return None
    if abs(y_max - y_min) < 1e-9:
        sample_ys = [y_max]
    else:
        sample_ys = [y_min + (y_max - y_min) * (k / 4.0) for k in range(5)]

    seps: List[float] = []
    for y in sample_ys:
        xa = _interp_x_at_y(points_a, y)
        xb = _interp_x_at_y(points_b, y)
        if xa is None or xb is None:
            continue
        seps.append(abs(xa - xb))
    if not seps:
        return None
    return float(median(seps))


def _lane_score(
    lane: Any,
    px_to_m: float,
    min_len_m: float,
    min_sep_m: float,
    max_curv_1pm: float,
    homography: Optional[Mat3],
) -> Dict[str, Any]:
    """Score lane detections by selecting the best detector threshold candidate."""
    # Soft margin keeps scoring stable when curvature is slightly noisy at a given threshold.
    soft_curv_1pm = max_curv_1pm * 1.8

    results = []
    if isinstance(lane, dict):
        maybe_results = lane.get("results")
        if isinstance(maybe_results, list):
            results = maybe_results

    def thr_value(item: Any) -> float:
        if not isinstance(item, dict):
            return 0.0
        return _safe_float(item.get("thr")) or 0.0

    ranked_results = sorted(results, key=thr_value, reverse=True)
    threshold_candidates: List[Dict[str, Any]] = []

    for res in ranked_results:
        curves = res.get("curves") if isinstance(res, dict) else None
        if not isinstance(curves, list):
            continue

        valid_soft: List[Dict[str, Any]] = []
        valid_strict: List[Dict[str, Any]] = []
        all_lengths_m: List[float] = []
        all_curvature_1pm: List[float] = []
        curves_with_geometry = 0
        curves_meeting_length = 0
        curves_meeting_curvature = 0
        curves_meeting_soft_curvature = 0

        for curve in curves:
            if not isinstance(curve, list):
                continue
            pts = _polyline_points(curve, px_to_m=px_to_m, homography=homography)
            if len(pts) < 3:
                continue

            length_m = _polyline_length_m(pts)
            curvature_1pm = _polyline_curvature_1pm(pts)
            curves_with_geometry += 1
            all_lengths_m.append(length_m)
            all_curvature_1pm.append(curvature_1pm)
            if length_m >= min_len_m:
                curves_meeting_length += 1
            if curvature_1pm <= max_curv_1pm:
                curves_meeting_curvature += 1
            if curvature_1pm <= soft_curv_1pm:
                curves_meeting_soft_curvature += 1

            if length_m >= min_len_m and curvature_1pm <= soft_curv_1pm:
                ys = [p[1] for p in pts]
                item = {
                    "points": pts,
                    "length_m": length_m,
                    "curvature_1pm": curvature_1pm,
                    "min_y": min(ys),
                    "max_y": max(ys),
                }
                valid_soft.append(item)
                if curvature_1pm <= max_curv_1pm:
                    valid_strict.append(item)

        def _lane_pair_stats(valid: List[Dict[str, Any]]) -> Tuple[int, int, int, List[float]]:
            if not valid:
                return 0, 0, 0, []
            if len(valid) == 1:
                return 1, 0, 0, []

            participating = set()
            pairwise_separations: List[float] = []
            pairs_evaluated_local = 0
            pairs_meeting_local = 0
            for i in range(len(valid)):
                i_min_y = float(valid[i]["min_y"])
                i_max_y = float(valid[i]["max_y"])
                for j in range(i + 1, len(valid)):
                    j_min_y = float(valid[j]["min_y"])
                    j_max_y = float(valid[j]["max_y"])
                    overlap_min = max(i_min_y, j_min_y)
                    overlap_max = min(i_max_y, j_max_y)
                    if overlap_min > overlap_max:
                        continue
                    sep_m = _pair_median_separation_m(valid[i]["points"], valid[j]["points"], overlap_min, overlap_max)
                    if sep_m is None:
                        continue
                    pairs_evaluated_local += 1
                    pairwise_separations.append(sep_m)
                    if sep_m >= min_sep_m:
                        pairs_meeting_local += 1
                        participating.add(i)
                        participating.add(j)
            return len(participating), pairs_evaluated_local, pairs_meeting_local, pairwise_separations

        valid_count, pairs_evaluated, pairs_meeting_separation, pairwise_separations_m = _lane_pair_stats(valid_soft)
        strict_valid_count, strict_pairs_evaluated, strict_pairs_meeting_separation, strict_pairwise_separations_m = _lane_pair_stats(valid_strict)

        if strict_valid_count >= 2:
            cand_score = 1.0
        elif valid_count >= 2 and strict_valid_count >= 1:
            cand_score = 1.0
        elif valid_count >= 2:
            cand_score = 0.75
        elif valid_count == 1:
            cand_score = 0.5
        else:
            cand_score = 0.0

        # Prefer lower threshold only as a tie-breaker and avoid over-detection bias.
        over_detection_penalty = max(0.0, (len(curves) - 4) * 0.01)
        cand_score = max(0.0, cand_score - over_detection_penalty)

        length_pass_rate = float(curves_meeting_length) / float(curves_with_geometry) if curves_with_geometry else 0.0
        curvature_pass_rate = (
            float(curves_meeting_curvature) / float(curves_with_geometry) if curves_with_geometry else 0.0
        )
        curvature_soft_pass_rate = (
            float(curves_meeting_soft_curvature) / float(curves_with_geometry) if curves_with_geometry else 0.0
        )
        separation_pass_rate = float(pairs_meeting_separation) / float(pairs_evaluated) if pairs_evaluated else 0.0
        strict_separation_pass_rate = (
            float(strict_pairs_meeting_separation) / float(strict_pairs_evaluated) if strict_pairs_evaluated else 0.0
        )

        threshold_candidates.append(
            {
                "threshold": _safe_float(res.get("thr")),
                "selected_curves": curves,
                "valid_lane_lines": valid_count,
                "strict_valid_lane_lines": strict_valid_count,
                "score": cand_score,
                "curves_with_geometry": curves_with_geometry,
                "valid_curves_before_separation": len(valid_soft),
                "strict_valid_curves_before_separation": len(valid_strict),
                "pairs_evaluated": pairs_evaluated,
                "pairs_meeting_separation": pairs_meeting_separation,
                "strict_pairs_evaluated": strict_pairs_evaluated,
                "strict_pairs_meeting_separation": strict_pairs_meeting_separation,
                "length_check": {
                    "min_length_m": min_len_m,
                    "pass_count": curves_meeting_length,
                    "total_count": curves_with_geometry,
                    "pass_rate": length_pass_rate,
                    "stats_m": _summary_stats(all_lengths_m),
                },
                "curvature_check": {
                    "max_curvature_1pm": max_curv_1pm,
                    "soft_max_curvature_1pm": soft_curv_1pm,
                    "pass_count": curves_meeting_curvature,
                    "soft_pass_count": curves_meeting_soft_curvature,
                    "total_count": curves_with_geometry,
                    "pass_rate": curvature_pass_rate,
                    "soft_pass_rate": curvature_soft_pass_rate,
                    "stats_1pm": _summary_stats(all_curvature_1pm),
                },
                "separation_check": {
                    "min_separation_m": min_sep_m,
                    "pairs_meeting_separation": pairs_meeting_separation,
                    "pairs_evaluated": pairs_evaluated,
                    "pass_rate": separation_pass_rate,
                    "stats_m": _summary_stats(pairwise_separations_m),
                },
                "strict_separation_check": {
                    "min_separation_m": min_sep_m,
                    "pairs_meeting_separation": strict_pairs_meeting_separation,
                    "pairs_evaluated": strict_pairs_evaluated,
                    "pass_rate": strict_separation_pass_rate,
                    "stats_m": _summary_stats(strict_pairwise_separations_m),
                },
                "curves_total": len(curves),
            }
        )

    threshold_ranking = sorted(
        threshold_candidates,
        key=lambda x: (
            float(x.get("score", 0.0)),
            int(x.get("valid_lane_lines", 0)),
            int(x.get("pairs_meeting_separation", 0)),
            int(x.get("valid_curves_before_separation", 0)),
            float(((x.get("separation_check") or {}).get("stats_m") or {}).get("median") or 0.0),
            float(((x.get("length_check") or {}).get("pass_rate") or 0.0)),
            float(((x.get("curvature_check") or {}).get("soft_pass_rate") or 0.0)),
            -float(x.get("threshold", 0.0)) if x.get("threshold") is not None else -1.0,
            -int(x.get("curves_total", 0)),
        ),
        reverse=True,
    )

    best = threshold_ranking[0] if threshold_ranking else {
        "threshold": None,
        "valid_lane_lines": 0,
        "strict_valid_lane_lines": 0,
        "score": 0.0,
        "curves_with_geometry": 0,
        "valid_curves_before_separation": 0,
        "strict_valid_curves_before_separation": 0,
        "pairs_evaluated": 0,
        "pairs_meeting_separation": 0,
        "strict_pairs_evaluated": 0,
        "strict_pairs_meeting_separation": 0,
        "length_check": {
            "min_length_m": min_len_m,
            "pass_count": 0,
            "total_count": 0,
            "pass_rate": 0.0,
            "stats_m": _summary_stats([]),
        },
        "curvature_check": {
            "max_curvature_1pm": max_curv_1pm,
            "soft_max_curvature_1pm": soft_curv_1pm,
            "pass_count": 0,
            "soft_pass_count": 0,
            "total_count": 0,
            "pass_rate": 0.0,
            "soft_pass_rate": 0.0,
            "stats_1pm": _summary_stats([]),
        },
        "separation_check": {
            "min_separation_m": min_sep_m,
            "pairs_meeting_separation": 0,
            "pairs_evaluated": 0,
            "pass_rate": 0.0,
            "stats_m": _summary_stats([]),
        },
        "strict_separation_check": {
            "min_separation_m": min_sep_m,
            "pairs_meeting_separation": 0,
            "pairs_evaluated": 0,
            "pass_rate": 0.0,
            "stats_m": _summary_stats([]),
        },
        "curves_total": 0,
    }
    best_valid_count = int(best.get("valid_lane_lines", 0))
    selected_thr = best.get("threshold")
    score = float(best.get("score", 0.0))

    return {
        "score": score,
        "valid_lane_lines": best_valid_count,
        "selected_threshold": selected_thr,
        "geometry_space": "warped_px" if homography is not None else "scaled_m",
        "best_threshold_detail": {
            "threshold": selected_thr,
            "selected_curves": best.get("selected_curves", []),
            "curves_with_geometry": int(best.get("curves_with_geometry", 0)),
            "valid_curves_before_separation": int(best.get("valid_curves_before_separation", 0)),
            "strict_valid_curves_before_separation": int(best.get("strict_valid_curves_before_separation", 0)),
            "pairs_evaluated": int(best.get("pairs_evaluated", 0)),
            "pairs_meeting_separation": int(best.get("pairs_meeting_separation", 0)),
            "strict_pairs_evaluated": int(best.get("strict_pairs_evaluated", 0)),
            "strict_pairs_meeting_separation": int(best.get("strict_pairs_meeting_separation", 0)),
            "strict_valid_lane_lines": int(best.get("strict_valid_lane_lines", 0)),
            "length_check": best.get("length_check", {}),
            "curvature_check": best.get("curvature_check", {}),
            "separation_check": best.get("separation_check", {}),
            "strict_separation_check": best.get("strict_separation_check", {}),
            "curves_total": int(best.get("curves_total", 0)),
        },
    }


def _gps_score(gps: Any) -> Dict[str, Any]:
    """Score GPS quality from the combined horizontal standard deviation."""
    if not isinstance(gps, dict):
        return {"score": 0.0, "er": None, "reason": "missing gps"}

    lat_stdev = _safe_float(gps.get("lat_stdev"))
    lon_stdev = _safe_float(gps.get("lon_stdev"))
    if lat_stdev is None or lon_stdev is None:
        return {"score": 0.0, "er": None, "reason": "missing lat_stdev/lon_stdev"}

    er = math.sqrt(lat_stdev * lat_stdev + lon_stdev * lon_stdev)

    if er <= 3.0:
        score = 1.0
    elif er <= 5.0:
        score = 0.75
    elif er <= 8.0:
        score = 0.5
    elif er <= 10.0:
        score = 0.25
    else:
        score = 0.0
    return {"score": score, "er": er, "lat_stdev": lat_stdev, "lon_stdev": lon_stdev}


def _score_download_mbps(v: Optional[float]) -> float:
    """Map download speed into the discrete readiness rubric."""
    if v is None:
        return 0.0
    if v > 100.0:
        return 1.0
    if v >= 75.0:
        return 0.75
    if v >= 50.0:
        return 0.5
    if v >= 25.0:
        return 0.25
    return 0.0


def _score_upload_mbps(v: Optional[float]) -> float:
    """Map upload speed into the discrete readiness rubric."""
    if v is None:
        return 0.0
    if v > 50.0:
        return 1.0
    if v >= 40.0:
        return 0.75
    if v >= 30.0:
        return 0.5
    if v >= 10.0:
        return 0.25
    return 0.0


def _score_download_latency(v: Optional[float]) -> float:
    """Map download latency into the discrete readiness rubric."""
    if v is None:
        return 0.0
    if v < 10.0:
        return 1.0
    if v <= 50.0:
        return 0.75
    if v <= 75.0:
        return 0.5
    if v <= 100.0:
        return 0.25
    return 0.0


def _score_upload_latency(v: Optional[float]) -> float:
    """Map upload latency into the discrete readiness rubric."""
    if v is None:
        return 0.0
    if v < 50.0:
        return 1.0
    if v <= 75.0:
        return 0.75
    if v <= 100.0:
        return 0.5
    if v <= 150.0:
        return 0.25
    return 0.0


def _connectivity_score(conn: Any) -> Dict[str, Any]:
    """Score connectivity using throughput and latency subcomponents."""
    if not isinstance(conn, dict):
        return {"score": 0.0, "subscores": {}, "reason": "missing connectivity"}

    d_mbps = _safe_float(conn.get("avg_d_mbps"))
    u_mbps = _safe_float(conn.get("avg_u_mbps"))
    dl_ms = _safe_float(conn.get("avg_download_latency_ms"))
    ul_ms = _safe_float(conn.get("avg_upload_latency_ms"))

    subscores = {
        "avg_d_mbps": _score_download_mbps(d_mbps),
        "avg_u_mbps": _score_upload_mbps(u_mbps),
        "avg_download_latency_ms": _score_download_latency(dl_ms),
        "avg_upload_latency_ms": _score_upload_latency(ul_ms),
    }
    overall = sum(subscores.values()) / len(subscores)
    return {
        "score": overall,
        "subscores": subscores,
        "metrics": {
            "avg_d_mbps": d_mbps,
            "avg_u_mbps": u_mbps,
            "avg_download_latency_ms": dl_ms,
            "avg_upload_latency_ms": ul_ms,
        },
    }


def evaluate_record(
    rec: Dict[str, Any],
    px_to_m: float,
    min_len_m: float,
    min_sep_m: float,
    max_curv_1pm: float,
    lane_homography: Optional[Mat3],
    w1: float,
    w2: float,
    k1: float,
    k2: float,
    k3: float,
    m1: float,
    m2: float,
    m3: float,
    pavement_score: float,
    traffic_sign_score: float,
    hd_maps_score: float,
) -> Dict[str, Any]:
    """Evaluate one fused record and return the full readiness breakdown."""
    lane_eval = _lane_score(
        rec.get("lane"),
        px_to_m=px_to_m,
        min_len_m=min_len_m,
        min_sep_m=min_sep_m,
        max_curv_1pm=max_curv_1pm,
        homography=lane_homography,
    )
    gps_eval = _gps_score(rec.get("gps"))
    conn_eval = _connectivity_score(rec.get("connectivity"))

    physical_score = (k1 * lane_eval["score"]) + (k2 * pavement_score) + (k3 * traffic_sign_score)
    digital_score = (m1 * conn_eval["score"]) + (m2 * gps_eval["score"]) + (m3 * hd_maps_score)
    physical_infra_score = w1 * physical_score
    digital_infra_score = w2 * digital_score
    overall = physical_infra_score + digital_infra_score

    return {
        "lane": lane_eval,
        "gps": gps_eval,
        "connectivity": conn_eval,
        "weights": {
            "w1": w1,
            "w2": w2,
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "m1": m1,
            "m2": m2,
            "m3": m3,
        },
        "overall_score": overall,
        "overall_score_details": {
            "formula": "overall_score = w1*physical_score + w2*digital_score",
            "physical_contribution": {
                "weight": w1,
                "score": physical_score,
                "weighted_score": physical_infra_score,
            },
            "digital_contribution": {
                "weight": w2,
                "score": digital_score,
                "weighted_score": digital_infra_score,
            },
            "components": {
                "physical_infra_score": physical_infra_score,
                "digital_infra_score": digital_infra_score,
                "physical_score": physical_score,
                "digital_score": digital_score,
                "lane_score": lane_eval["score"],
                "pavement_score": pavement_score,
                "traffic_sign_score": traffic_sign_score,
                "hd_maps_score": hd_maps_score,
                "connectivity_score": conn_eval["score"],
                "gps_score": gps_eval["score"],
            },
        },
    }


def run(
    input_dir: Path,
    output_dir: Path,
    px_to_m: float,
    min_len_m: float,
    min_sep_m: float,
    max_curv_1pm: float,
    lane_homography: Optional[Mat3],
    w1: float,
    w2: float,
    k1: float,
    k2: float,
    k3: float,
    m1: float,
    m2: float,
    m3: float,
    pavement_score: float,
    traffic_sign_score: float,
    hd_maps_score: float,
) -> None:
    """Evaluate all fused JSON records in a directory and write trimmed outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")

    count = 0
    for src in files:
        with src.open("r", encoding="utf-8") as f:
            rec = json.load(f)
        evaluation = evaluate_record(
            rec,
            px_to_m=px_to_m,
            min_len_m=min_len_m,
            min_sep_m=min_sep_m,
            max_curv_1pm=max_curv_1pm,
            lane_homography=lane_homography,
            w1=w1,
            w2=w2,
            k1=k1,
            k2=k2,
            k3=k3,
            m1=m1,
            m2=m2,
            m3=m3,
            pavement_score=pavement_score,
            traffic_sign_score=traffic_sign_score,
            hd_maps_score=hd_maps_score,
        )
        trimmed = {
            "image": rec.get("image"),
            "frame_index": rec.get("frame_index"),
            "timestamp_ns": rec.get("timestamp_ns"),
            "gps": rec.get("gps"),
            "connectivity": rec.get("connectivity"),
            "connectivity_error": rec.get("connectivity_error"),
            "evaluation": evaluation,
        }
        dst = output_dir / src.name
        with dst.open("w", encoding="utf-8") as f:
            json.dump(trimmed, f, indent=2)
        count += 1

    print(f"Evaluated {count} files from {input_dir} -> {output_dir}")


def main() -> None:
    """CLI entry point for readiness evaluation of fused JSON files."""
    parser = argparse.ArgumentParser(
        description="Evaluate fused lane+gps+connectivity JSON files and write scored JSON outputs."
    )
    parser.add_argument("--input-dir", default="results/fused", help="Directory containing fused JSON files.")
    parser.add_argument(
        "--output-dir", default="results/evaluated", help="Directory to write evaluated JSON files."
    )
    parser.add_argument(
        "--px-to-m",
        type=float,
        default=1.0,
        help="Pixel-to-meter conversion for lane geometric checks.",
    )
    parser.add_argument("--lane-min-length-m", type=float, default=180.0, help="Minimum lane segment length in pixels.")
    parser.add_argument(
        "--lane-min-separation-m",
        type=float,
        default=2.5,
        help="Minimum separation between lane lines in meters.",
    )
    parser.add_argument(
        "--lane-max-curvature",
        type=float,
        default=0.1,
        help="Maximum lane curvature in 1/m (per eval.txt: curvature < 1/10m).",
    )
    parser.add_argument(
        "--lane-homography",
        type=str,
        default="; ".join(
            [
                ", ".join(str(v) for v in DEFAULT_LANE_HOMOGRAPHY[0]),
                ", ".join(str(v) for v in DEFAULT_LANE_HOMOGRAPHY[1]),
                ", ".join(str(v) for v in DEFAULT_LANE_HOMOGRAPHY[2]),
            ]
        ),
        help=(
            "3x3 homography used to warp lane points to a BEV-like space before geometry checks. "
            "Format: 'h11,h12,h13; h21,h22,h23; h31,h32,h33'"
        ),
    )
    parser.add_argument(
        "--disable-lane-homography",
        action="store_true",
        help="Disable homography warp and evaluate lane geometry in scaled image coordinates.",
    )
    parser.add_argument("--w1", type=float, default=0.6, help="Weight for physical infrastructure score.")
    parser.add_argument("--w2", type=float, default=0.4, help="Weight for digital infrastructure score.")
    parser.add_argument("--k1", type=float, default=1.0, help="Lane weight in physical score.")
    parser.add_argument("--k2", type=float, default=0.0, help="Pavement weight in physical score.")
    parser.add_argument("--k3", type=float, default=0.0, help="Traffic-sign weight in physical score.")
    parser.add_argument("--m1", type=float, default=0.5, help="Connectivity weight in digital score.")
    parser.add_argument("--m2", type=float, default=0.4, help="GPS weight in digital score.")
    parser.add_argument("--m3", type=float, default=0.1, help="HD-maps weight in digital score.")
    parser.add_argument("--pavement-score", type=float, default=0.0, help="Pavement score placeholder.")
    parser.add_argument("--traffic-sign-score", type=float, default=0.0, help="Traffic-sign score placeholder.")
    parser.add_argument("--hd-maps-score", type=float, default=0.0, help="HD maps score.")
    args = parser.parse_args()

    lane_h = None if args.disable_lane_homography else _parse_homography(args.lane_homography)

    run(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        px_to_m=args.px_to_m,
        min_len_m=args.lane_min_length_m,
        min_sep_m=args.lane_min_separation_m,
        max_curv_1pm=args.lane_max_curvature,
        lane_homography=lane_h,
        w1=args.w1,
        w2=args.w2,
        k1=args.k1,
        k2=args.k2,
        k3=args.k3,
        m1=args.m1,
        m2=args.m2,
        m3=args.m3,
        pavement_score=args.pavement_score,
        traffic_sign_score=args.traffic_sign_score,
        hd_maps_score=args.hd_maps_score,
    )


if __name__ == "__main__":
    main()
