#!/usr/bin/env python3
"""Serve a Flask UI for running the readiness pipeline and browsing results."""

import json
import hmac
import os
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, request, send_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "defaults.json"
RESULTS_DIR = REPO_ROOT / "results"
RUNTIME_DIR = RESULTS_DIR / "ui_runtime"
READINESS_DIR = RESULTS_DIR / "readiness"
READINESS_ARCHIVE_ROOT = RESULTS_DIR / "archive" / "readiness_runs"
ALL_EVALUATIONS_DASHBOARD = READINESS_DIR / "all_evaluations_dashboard.html"
ALL_EVALUATIONS_MAP = READINESS_DIR / "all_evaluations_map.html"

STATE_LOCK = threading.Lock()
STATE: Dict[str, Any] = {
    "running": False,
    "stage": "idle",
    "last_error": None,
    "last_run_id": None,
    "last_started_at": None,
    "last_finished_at": None,
    "dashboard_path": "results/readiness/readiness_dashboard.html",
    "logs": [],
    "params": {},
    "progress_pct": 0,
    "progress_text": "Idle",
    "cancel_requested": False,
    "run_kind": None,
    "reload_dashboard_on_complete": False,
    "active_pid": None,
    "active_archive_path": None,
}

app = Flask(__name__)
ACTIVE_PROC_LOCK = threading.Lock()
ACTIVE_PROC: Optional[subprocess.Popen[str]] = None
ARCHIVE_REFRESH_LOCK = threading.Lock()
ARCHIVE_REFRESH_CACHE: Dict[str, Dict[str, Any]] = {}
ARCHIVE_REFRESH_EVENTS: Dict[str, threading.Event] = {}

MUTATING_ENDPOINTS = {"run", "reweight", "report", "stop", "fs_list"}
ARTIFACT_ROOT_PREFIXES = ("results",)


def _env_bool(name: str, default: bool) -> bool:
    """Parse common boolean environment-variable values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


PUBLIC_MODE = _env_bool("PUBLIC_MODE", False)
ENABLE_FULL_PIPELINE = _env_bool("ENABLE_FULL_PIPELINE", not PUBLIC_MODE)
ENABLE_REEVALUATION = _env_bool("ENABLE_REEVALUATION", True)
ENABLE_REPORT_REGEN = _env_bool("ENABLE_REPORT_REGEN", True)
ENABLE_FS_BROWSER = _env_bool("ENABLE_FS_BROWSER", False if PUBLIC_MODE else True)
APP_API_TOKEN = os.getenv("APP_API_TOKEN", "").strip()

_raw_fs_roots = os.getenv("FS_BROWSER_ROOTS", "").strip()
if _raw_fs_roots:
    FS_BROWSER_ROOTS = [Path(s).expanduser().resolve() for s in _raw_fs_roots.split(":") if s.strip()]
else:
    FS_BROWSER_ROOTS = [REPO_ROOT.resolve()] if PUBLIC_MODE else [Path.home().resolve(), REPO_ROOT.resolve()]


class PipelineInterrupted(RuntimeError):
    """Raised when a running pipeline is cancelled by the user."""
    pass


def _append_log(msg: str) -> None:
    """Append one line to the in-memory UI log buffer."""
    with STATE_LOCK:
        STATE["logs"].append(msg)
        if len(STATE["logs"]) > 2000:
            STATE["logs"] = STATE["logs"][-2000:]


def _set_state(**kwargs: Any) -> None:
    """Update the shared UI state atomically."""
    with STATE_LOCK:
        STATE.update(kwargs)


def _snapshot_state() -> Dict[str, Any]:
    """Return a copy of the current UI state plus a recent log tail."""
    with STATE_LOCK:
        out = dict(STATE)
        out["logs_tail"] = STATE["logs"][-200:]
    return out


def _set_stage(stage: str) -> None:
    """Update the pipeline stage and the user-facing progress message."""
    stage_progress = {
        "idle": (0, "Idle"),
        "preparing": (5, "Preparing runtime configuration"),
        "archiving_readiness": (10, "Archiving previous readiness outputs"),
        "process_bag_pipeline": (20, "Running process_bag_pipeline"),
        "evaluate_fused_results": (70, "Running evaluate_fused_results"),
        "readiness_heatmap": (90, "Running readiness_heatmap"),
        "compile_report": (95, "Compiling PDF report"),
        "reweight_evaluate": (70, "Reweight: evaluating fused results"),
        "reweight_heatmap": (90, "Reweight: generating readiness dashboard"),
        "reweight_report": (95, "Reweight: compiling PDF report"),
        "stopping": (100, "Stopping and restoring previous results"),
        "completed": (100, "Completed"),
        "stopped": (100, "Stopped"),
        "failed": (100, "Failed"),
    }
    pct, text = stage_progress.get(stage, (0, stage))
    _set_state(stage=stage, progress_pct=pct, progress_text=text)


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object from disk, returning an empty dict for non-dict content."""
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _parse_float(params: Dict[str, str], key: str, default: float) -> float:
    """Parse one float parameter from a request payload."""
    raw = str(params.get(key, "")).strip()
    return float(raw) if raw else float(default)


def _parse_int(params: Dict[str, str], key: str, default: int) -> int:
    """Parse one integer parameter from a request payload."""
    raw = str(params.get(key, "")).strip()
    return int(raw) if raw else int(default)


def _path_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` is equal to or contained in ``root``."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _extract_request_token() -> str:
    """Extract bearer/API-key token from request headers."""
    auth = str(request.headers.get("Authorization", "")).strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return str(request.headers.get("X-API-Key", "")).strip()


def _require_api_token(endpoint_name: str) -> Optional[Response]:
    """Enforce token auth for mutating endpoints when configured."""
    if endpoint_name not in MUTATING_ENDPOINTS:
        return None
    if not APP_API_TOKEN:
        return None
    provided = _extract_request_token()
    if provided and hmac.compare_digest(provided, APP_API_TOKEN):
        return None
    return jsonify({"ok": False, "error": "Unauthorized"}), 401


def _feature_disabled(msg: str) -> Response:
    """Build a consistent feature-disabled response body."""
    return jsonify({"ok": False, "error": msg}), 403


def _ensure_abs(path_str: str) -> str:
    """Resolve a repo-relative path string to an absolute POSIX path string."""
    p = Path(path_str)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p.as_posix()


def _build_run_config(run_dir: Path, params: Dict[str, str]) -> Path:
    """Build the runtime config file used by a pipeline execution thread."""
    cfg = _load_json(DEFAULT_CONFIG)
    bag_file = params.get("bag_file", "").strip()
    if bag_file:
        cfg["bag_file"] = bag_file
        for key in ("lane", "gps", "pipeline"):
            section = cfg.get(key, {})
            if isinstance(section, dict):
                section["bag_file"] = bag_file
                cfg[key] = section

    frame_stride = _parse_int(params, "frame_stride", int(((cfg.get("pipeline") or {}).get("frame_stride") or 10)))
    lane_sec = cfg.get("lane", {}) if isinstance(cfg.get("lane"), dict) else {}
    pipe_sec = cfg.get("pipeline", {}) if isinstance(cfg.get("pipeline"), dict) else {}
    lane_sec["frame_stride"] = frame_stride
    pipe_sec["frame_stride"] = frame_stride

    lane_sec["model_path"] = _ensure_abs(str(lane_sec.get("model_path", "lane/model/lane_seg_model_onnx.onnx")))
    lane_sec["output_dir"] = _ensure_abs(str(lane_sec.get("output_dir", "results/lane")))
    if lane_sec.get("bag_file"):
        lane_sec["bag_file"] = _ensure_abs(str(lane_sec.get("bag_file")))

    cfg["lane"] = lane_sec
    cfg["pipeline"] = pipe_sec

    cfg_path = run_dir / "runtime_config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return cfg_path


def _archive_existing_readiness() -> Optional[Path]:
    """Snapshot current results into the readiness archive before overwriting them."""
    # Snapshot current outputs that can be overwritten by a new run.
    # Include both directories and top-level artifacts produced by reporting.
    snapshot_targets = [
        RESULTS_DIR / "readiness",
        RESULTS_DIR / "evaluated",
        RESULTS_DIR / "fused",
        RESULTS_DIR / "lane",
        RESULTS_DIR / "gps",
        RESULTS_DIR / "connectivity",
        RESULTS_DIR / "figures",
        RESULTS_DIR / "research_report.pdf",
        RESULTS_DIR / "report.pdf",
        RESULTS_DIR / "report.html",
    ]

    existing: List[Path] = []
    for p in snapshot_targets:
        if not p.exists():
            continue
        if p.is_dir():
            try:
                if not any(p.iterdir()):
                    continue
            except Exception:
                continue
        elif p.is_file() and p.stat().st_size <= 0:
            continue
        existing.append(p)

    if not existing:
        return None

    READINESS_ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dst_root = READINESS_ARCHIVE_ROOT / f"run_{tag}"
    dst_root.mkdir(parents=True, exist_ok=True)

    archived_items: List[str] = []
    for src in existing:
        rel = src.relative_to(RESULTS_DIR)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        archived_items.append(rel.as_posix())

    # Include quick metadata for traceability.
    meta = {
        "archived_at": datetime.now().isoformat(),
        "archived_items": archived_items,
    }
    with (dst_root / "archive_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return dst_root


def _clear_current_outputs(preserve_dirs: Optional[List[Path]] = None) -> List[str]:
    """Clear current result outputs while optionally preserving selected directories."""
    preserve_abs = {p.resolve() for p in (preserve_dirs or []) if p is not None}
    clear_targets = [
        RESULTS_DIR / "readiness",
        RESULTS_DIR / "evaluated",
        RESULTS_DIR / "fused",
        RESULTS_DIR / "lane",
        RESULTS_DIR / "gps",
        RESULTS_DIR / "connectivity",
        RESULTS_DIR / "figures",
        RESULTS_DIR / "research_report.pdf",
        RESULTS_DIR / "report.pdf",
        RESULTS_DIR / "report.html",
    ]

    cleared: List[str] = []
    for p in clear_targets:
        try:
            rp = p.resolve()
            # Preserve when target is equal to keep, under keep, or an ancestor of keep.
            if any(
                rp == keep
                or str(rp).startswith(str(keep) + os.sep)
                or str(keep).startswith(str(rp) + os.sep)
                for keep in preserve_abs
            ):
                continue
        except Exception:
            pass

        if not p.exists():
            continue
        if p.is_dir():
            shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)
        else:
            p.unlink()
        cleared.append(p.relative_to(REPO_ROOT).as_posix())
    return cleared


def _restore_archived_outputs(archive_dir: Optional[Path]) -> List[str]:
    """Restore a previously archived result snapshot back into ``results/``."""
    if archive_dir is None or not archive_dir.exists() or not archive_dir.is_dir():
        return []
    restored: List[str] = []
    for src in archive_dir.iterdir():
        if src.name == "archive_meta.json":
            continue
        dst = RESULTS_DIR / src.name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        restored.append(dst.relative_to(REPO_ROOT).as_posix())
    return restored


def _set_active_process(proc: Optional[subprocess.Popen[str]]) -> None:
    """Track the currently running subprocess for stop/cleanup handling."""
    with ACTIVE_PROC_LOCK:
        global ACTIVE_PROC
        ACTIVE_PROC = proc
    pid = getattr(proc, "pid", None) if proc is not None else None
    _set_state(active_pid=pid)


def _get_active_process() -> Optional[subprocess.Popen[str]]:
    """Return the subprocess currently owned by the UI, if any."""
    with ACTIVE_PROC_LOCK:
        return ACTIVE_PROC


def _terminate_process(proc: subprocess.Popen[str]) -> bool:
    """Best-effort terminate a subprocess group and report whether it exited."""
    try:
        if proc.poll() is not None:
            return True
    except Exception:
        return True

    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass

    try:
        proc.wait(timeout=6)
        return True
    except Exception:
        pass

    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass
    return False


def _is_stop_requested() -> bool:
    """Return whether the user has requested cancellation of the active run."""
    with STATE_LOCK:
        return bool(STATE.get("cancel_requested"))


def _request_stop() -> Dict[str, Any]:
    """Mark the active run for cancellation and signal the current subprocess."""
    with STATE_LOCK:
        if not STATE.get("running"):
            return {"ok": False, "error": "No run is currently in progress."}
        STATE["cancel_requested"] = True
    _set_stage("stopping")
    _append_log("Stop requested by user.")
    proc = _get_active_process()
    if proc is None:
        return {"ok": True, "message": "Stop requested. Waiting for current stage to check cancellation."}
    _terminate_process(proc)
    return {"ok": True, "message": "Stop requested. Active process termination sent."}


def _run_cmd(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    """Run one pipeline command while streaming output into the UI log/state."""
    if _is_stop_requested():
        raise PipelineInterrupted("Run interrupted by user request.")
    _append_log("$ " + " ".join(cmd))
    p = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT.as_posix(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    _set_active_process(p)
    assert p.stdout is not None
    try:
        for line in p.stdout:
            s = line.rstrip("\n")
            _append_log(s)

            if "Running lane detector:" in s:
                _set_state(progress_pct=30, progress_text="Lane detector running")
            elif s.startswith("PIPE_PROGRESS "):
                parts: Dict[str, str] = {}
                for tok in s.replace("PIPE_PROGRESS", "").strip().split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        parts[k] = v
                try:
                    selected = int(parts.get("selected", "0"))
                except Exception:
                    selected = 0
                pct = min(65, 30 + min(35, selected // 5))
                _set_state(
                    progress_pct=pct,
                    progress_text=(
                        f"Processing bag/connectivity (selected={parts.get('selected','0')}, "
                        f"conn_hits={parts.get('connectivity_hits','0')})"
                    ),
                )
            elif s.startswith("Finished. Total image frames:"):
                _set_state(progress_pct=65, progress_text="Fused results generated")
            elif s.startswith("Evaluated ") and "files from" in s:
                _set_state(progress_pct=88, progress_text="Evaluation complete")
            elif "Wrote readiness heat map HTML:" in s:
                _set_state(progress_pct=96, progress_text="Heatmap generated")
            elif "Wrote split-screen dashboard HTML:" in s:
                _set_state(progress_pct=99, progress_text="Dashboard generated")
            elif s.startswith("PDF generated ->"):
                _set_state(progress_pct=99, progress_text="PDF report generated")

            if _is_stop_requested():
                _terminate_process(p)
                raise PipelineInterrupted("Run interrupted by user request.")
    finally:
        _set_active_process(None)

    code = p.wait()
    if _is_stop_requested():
        raise PipelineInterrupted("Run interrupted by user request.")
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(cmd)}")


def _run_eval_and_dashboard(
    params: Dict[str, str],
    input_fused_dir: Path,
    input_lane_images_dir: Path,
    stage_eval: str,
    stage_dash: str,
    stage_report: str,
) -> None:
    """Run evaluation, heatmap/dashboard generation, and PDF compilation stages."""
    _set_stage(stage_eval)
    eval_cmd = [
        "python3",
        "pipeline/evaluate_fused_results.py",
        "--input-dir",
        input_fused_dir.as_posix(),
        "--output-dir",
        "results/evaluated",
        "--w1",
        str(_parse_float(params, "w1", 0.6)),
        "--w2",
        str(_parse_float(params, "w2", 0.4)),
        "--m1",
        str(_parse_float(params, "m1", 0.5)),
        "--m2",
        str(_parse_float(params, "m2", 0.4)),
        "--m3",
        str(_parse_float(params, "m3", 0.1)),
    ]
    _run_cmd(eval_cmd)

    _set_stage(stage_dash)
    map_cmd = [
        "python3",
        "pipeline/readiness_heatmap.py",
        "--input-dir",
        "results/evaluated",
        "--output-json",
        "results/readiness/readiness_per_mile.json",
        "--output-map",
        "results/readiness/readiness_heatmap.html",
        "--output-dashboard",
        "results/readiness/readiness_dashboard.html",
        "--report-pdf",
        "results/research_report.pdf",
        "--lane-images-dir",
        input_lane_images_dir.as_posix(),
        "--overlay-dir",
        str(params.get("overlay_dir", "results/readiness/keypoint_overlays")),
        "--keypoint-stride",
        str(_parse_int(params, "keypoint_stride", 10)),
    ]
    _run_cmd(map_cmd, env=os.environ.copy())

    if ENABLE_REPORT_REGEN:
        _set_stage(stage_report)
        report_cmd = ["python3", "pipeline/compile_report.py"]
        _run_cmd(report_cmd, env=os.environ.copy())
    else:
        _append_log("Skipping report compilation (ENABLE_REPORT_REGEN is disabled).")


def _pipeline_worker(run_id: str, params: Dict[str, str]) -> None:
    """Background worker that executes the full rosbag-to-dashboard pipeline."""
    run_dir = RUNTIME_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _set_state(
        running=True,
        last_error=None,
        last_run_id=run_id,
        last_started_at=datetime.now().isoformat(),
        params=params,
        cancel_requested=False,
        run_kind="pipeline",
        reload_dashboard_on_complete=True,
        active_archive_path=None,
    )
    _append_log(f"Starting run_id={run_id}")
    archived: Optional[Path] = None
    try:
        _set_stage("archiving_readiness")
        archived = _archive_existing_readiness()
        _set_state(active_archive_path=archived.as_posix() if archived else None)
        if archived:
            _append_log(f"Archived existing readiness outputs to: {archived}")
        else:
            _append_log("No previous readiness output to archive.")
        cleared = _clear_current_outputs()
        _append_log(
            "Cleared current outputs after archive: "
            + (", ".join(cleared) if cleared else "none")
        )

        _set_stage("preparing")
        cfg_path = _build_run_config(run_dir, params)
        _append_log(f"Wrote runtime config: {cfg_path}")

        _set_stage("process_bag_pipeline")
        _run_cmd(["python3", "pipeline/process_bag_pipeline.py", "--config", cfg_path.as_posix()])

        _run_eval_and_dashboard(
            params=params,
            input_fused_dir=Path("results/fused"),
            input_lane_images_dir=Path("results/lane/images"),
            stage_eval="evaluate_fused_results",
            stage_dash="readiness_heatmap",
            stage_report="compile_report",
        )

        _set_state(
            running=False,
            last_finished_at=datetime.now().isoformat(),
            dashboard_path="results/readiness/readiness_dashboard.html",
        )
        _set_stage("completed")
        _append_log("Run completed successfully.")
    except PipelineInterrupted:
        _append_log("Stop acknowledged. Cleaning partial outputs.")
        cleared = _clear_current_outputs()
        _append_log("Cleared partial outputs: " + (", ".join(cleared) if cleared else "none"))
        restored = _restore_archived_outputs(archived)
        _append_log("Restored archived outputs: " + (", ".join(restored) if restored else "none"))
        _set_state(
            running=False,
            last_error=None,
            last_finished_at=datetime.now().isoformat(),
            dashboard_path="results/readiness/readiness_dashboard.html",
        )
        _set_stage("stopped")
    except Exception as exc:
        _append_log(f"ERROR: {exc}")
        _set_state(
            running=False,
            last_error=str(exc),
            last_finished_at=datetime.now().isoformat(),
        )
        _set_stage("failed")
    finally:
        _set_state(cancel_requested=False, run_kind=None, reload_dashboard_on_complete=False, active_archive_path=None, active_pid=None)


def _reweight_worker(run_id: str, params: Dict[str, str]) -> None:
    """Background worker that reruns only evaluation/reporting on existing fused data."""
    _set_state(
        running=True,
        last_error=None,
        last_run_id=run_id,
        last_started_at=datetime.now().isoformat(),
        params=params,
        cancel_requested=False,
        run_kind="reweight",
        reload_dashboard_on_complete=True,
        active_archive_path=None,
    )
    _append_log(f"Starting reweight run_id={run_id}")
    archived: Optional[Path] = None
    try:
        _set_stage("archiving_readiness")
        archived = _archive_existing_readiness()
        _set_state(active_archive_path=archived.as_posix() if archived else None)
        if archived:
            _append_log(f"Archived existing outputs to: {archived}")
        else:
            _append_log("No previous output to archive.")

        fused_dir = Path(str(params.get("source_fused_dir", "results/fused")))
        lane_images_dir = Path(str(params.get("source_lane_images_dir", "results/lane/images")))
        if not fused_dir.is_absolute():
            fused_dir = (REPO_ROOT / fused_dir).resolve()
        if not lane_images_dir.is_absolute():
            lane_images_dir = (REPO_ROOT / lane_images_dir).resolve()
        if not fused_dir.exists():
            raise FileNotFoundError(f"Source fused dir not found: {fused_dir}")
        if not lane_images_dir.exists():
            raise FileNotFoundError(f"Source lane images dir not found: {lane_images_dir}")
        if not any(lane_images_dir.iterdir()):
            raise FileNotFoundError(f"Source lane images dir is empty: {lane_images_dir}")

        preserve = [fused_dir, lane_images_dir]
        cleared = _clear_current_outputs(preserve_dirs=preserve)
        _append_log(
            "Cleared current outputs after archive (reweight): "
            + (", ".join(cleared) if cleared else "none")
        )

        _run_eval_and_dashboard(
            params=params,
            input_fused_dir=fused_dir,
            input_lane_images_dir=lane_images_dir,
            stage_eval="reweight_evaluate",
            stage_dash="reweight_heatmap",
            stage_report="reweight_report",
        )

        _set_state(
            running=False,
            last_finished_at=datetime.now().isoformat(),
            dashboard_path="results/readiness/readiness_dashboard.html",
        )
        _set_stage("completed")
        _append_log("Reweight evaluation completed successfully.")
    except PipelineInterrupted:
        _append_log("Stop acknowledged. Cleaning partial outputs.")
        cleared = _clear_current_outputs()
        _append_log("Cleared partial outputs: " + (", ".join(cleared) if cleared else "none"))
        restored = _restore_archived_outputs(archived)
        _append_log("Restored archived outputs: " + (", ".join(restored) if restored else "none"))
        _set_state(
            running=False,
            last_error=None,
            last_finished_at=datetime.now().isoformat(),
            dashboard_path="results/readiness/readiness_dashboard.html",
        )
        _set_stage("stopped")
    except Exception as exc:
        _append_log(f"ERROR: {exc}")
        _set_state(
            running=False,
            last_error=str(exc),
            last_finished_at=datetime.now().isoformat(),
        )
        _set_stage("failed")
    finally:
        _set_state(cancel_requested=False, run_kind=None, reload_dashboard_on_complete=False, active_archive_path=None, active_pid=None)


def _start_run(params: Dict[str, str]) -> Dict[str, Any]:
    """Validate parameters and launch a full pipeline run in a worker thread."""
    with STATE_LOCK:
        if STATE["running"]:
            return {"ok": False, "error": "A run is already in progress."}
        STATE["logs"] = []
        STATE["cancel_requested"] = False

    bag_file = str(params.get("bag_file", "")).strip()
    if not bag_file:
        return {"ok": False, "error": "bag_file is required."}
    if not Path(bag_file).exists():
        return {"ok": False, "error": f"bag_file does not exist: {bag_file}"}

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    t = threading.Thread(target=_pipeline_worker, args=(run_id, params), daemon=True)
    t.start()
    return {"ok": True, "run_id": run_id}


def _start_reweight(params: Dict[str, str]) -> Dict[str, Any]:
    """Validate parameters and launch a reweight-only run in a worker thread."""
    with STATE_LOCK:
        if STATE["running"]:
            return {"ok": False, "error": "A run is already in progress."}
        STATE["logs"] = []
        STATE["cancel_requested"] = False

    source_fused = str(params.get("source_fused_dir", "results/fused")).strip()
    p = Path(source_fused)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if not p.exists():
        return {"ok": False, "error": f"source fused dir does not exist: {p}"}

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    t = threading.Thread(target=_reweight_worker, args=(run_id, params), daemon=True)
    t.start()
    return {"ok": True, "run_id": run_id}


def _report_worker(run_id: str, results_dir: Path, dashboard_path: str) -> None:
    """Background worker that reruns only PDF report compilation."""
    _set_state(
        running=True,
        last_error=None,
        last_run_id=run_id,
        last_started_at=datetime.now().isoformat(),
        params={"results_dir": results_dir.as_posix()},
        cancel_requested=False,
        run_kind="report",
        reload_dashboard_on_complete=False,
        active_archive_path=None,
    )
    _append_log(f"Starting report regeneration run_id={run_id} results_dir={results_dir}")
    try:
        _set_stage("compile_report")
        report_cmd = ["python3", "pipeline/compile_report.py", "--results-dir", results_dir.as_posix()]
        _run_cmd(report_cmd, env=os.environ.copy())
        _set_state(
            running=False,
            last_finished_at=datetime.now().isoformat(),
            dashboard_path=dashboard_path,
        )
        _set_stage("completed")
        _append_log("Report regeneration completed successfully.")
    except PipelineInterrupted:
        _append_log("Report regeneration stop acknowledged.")
        _set_state(
            running=False,
            last_error=None,
            last_finished_at=datetime.now().isoformat(),
            dashboard_path=dashboard_path,
        )
        _set_stage("stopped")
    except Exception as exc:
        _append_log(f"ERROR: {exc}")
        _set_state(
            running=False,
            last_error=str(exc),
            last_finished_at=datetime.now().isoformat(),
            dashboard_path=dashboard_path,
        )
        _set_stage("failed")
    finally:
        _set_state(cancel_requested=False, run_kind=None, reload_dashboard_on_complete=False, active_archive_path=None, active_pid=None)


def _start_report_regen(params: Dict[str, str]) -> Dict[str, Any]:
    """Validate parameters and launch a report-only regeneration worker."""
    if not ENABLE_REPORT_REGEN:
        return {"ok": False, "error": "Report regeneration is disabled on this deployment."}

    with STATE_LOCK:
        if STATE["running"]:
            return {"ok": False, "error": "A run is already in progress."}
        STATE["logs"] = []
        STATE["cancel_requested"] = False

    raw_results_dir = str(params.get("results_dir", "results")).strip() or "results"
    results_dir = Path(raw_results_dir)
    if not results_dir.is_absolute():
        results_dir = (REPO_ROOT / results_dir).resolve()
    else:
        results_dir = results_dir.resolve()
    results_dir.relative_to(REPO_ROOT)

    evaluated_dir = results_dir / "evaluated"
    readiness_dir = results_dir / "readiness"
    lane_images_dir = results_dir / "lane" / "images"
    if not evaluated_dir.exists():
        return {"ok": False, "error": f"evaluated dir does not exist: {evaluated_dir}"}
    if not readiness_dir.exists():
        return {"ok": False, "error": f"readiness dir does not exist: {readiness_dir}"}
    if not lane_images_dir.exists():
        return {"ok": False, "error": f"lane images dir does not exist: {lane_images_dir}"}

    dashboard_path = (
        "results/readiness/readiness_dashboard.html"
        if results_dir == RESULTS_DIR
        else (results_dir / "readiness" / "readiness_dashboard.html").relative_to(REPO_ROOT).as_posix()
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    t = threading.Thread(target=_report_worker, args=(run_id, results_dir, dashboard_path), daemon=True)
    t.start()
    return {"ok": True, "run_id": run_id}


def _list_runs() -> List[Dict[str, str]]:
    """Return the current run plus archived runs available in the dashboard selector."""
    runs: List[Dict[str, str]] = []

    current_dash = READINESS_DIR / "readiness_dashboard.html"
    if (
        not current_dash.exists()
        and (RESULTS_DIR / "evaluated").exists()
        and (RESULTS_DIR / "lane" / "images").exists()
    ):
        _refresh_current_readiness_outputs()
        current_dash = READINESS_DIR / "readiness_dashboard.html"
    if current_dash.exists():
        runs.append(
            {
                "id": "current",
                "label": "Current (latest)",
                "dashboard_path": "results/readiness/readiness_dashboard.html",
                "map_path": "results/readiness/readiness_heatmap.html",
                "json_path": "results/readiness/readiness_per_mile.json",
                "source_fused_dir": "results/fused",
                "source_lane_images_dir": "results/lane/images",
            }
        )

    if READINESS_ARCHIVE_ROOT.exists():
        for d in sorted(READINESS_ARCHIVE_ROOT.iterdir(), key=lambda p: p.name, reverse=True):
            dash = d / "readiness" / "readiness_dashboard.html"
            if dash.exists():
                rel_dash = dash.relative_to(REPO_ROOT).as_posix()
                rel_map = (d / "readiness" / "readiness_heatmap.html").relative_to(REPO_ROOT).as_posix()
                rel_json = (d / "readiness" / "readiness_per_mile.json").relative_to(REPO_ROOT).as_posix()
                runs.append(
                    {
                        "id": d.name,
                        "label": d.name.replace("run_", "Archive "),
                        "dashboard_path": rel_dash,
                        "map_path": rel_map,
                        "json_path": rel_json,
                        "source_fused_dir": (d / "fused").relative_to(REPO_ROOT).as_posix(),
                        "source_lane_images_dir": (d / "lane" / "images").relative_to(REPO_ROOT).as_posix(),
                    }
                )

    return runs


def _list_all_evaluation_sources() -> List[Dict[str, Any]]:
    """Return current and archived evaluation directories that can feed the combined view."""
    sources: List[Dict[str, Any]] = []

    current_evaluated = RESULTS_DIR / "evaluated"
    current_lane_images = RESULTS_DIR / "lane" / "images"
    current_overlay = READINESS_DIR / "keypoint_overlays"
    current_readiness_json = READINESS_DIR / "readiness_per_mile.json"
    if current_readiness_json.exists():
        sources.append(
            {
                "id": "current",
                "label": "Current (latest)",
                "readiness_json": current_readiness_json.as_posix(),
                "evaluated_dir": current_evaluated.as_posix(),
                "lane_images_dir": current_lane_images.as_posix(),
                "overlay_dir": current_overlay.as_posix(),
                "report_pdf": (RESULTS_DIR / "research_report.pdf").as_posix(),
            }
        )

    if READINESS_ARCHIVE_ROOT.exists():
        for d in sorted(READINESS_ARCHIVE_ROOT.iterdir(), key=lambda p: p.name, reverse=True):
            evaluated_dir = d / "evaluated"
            lane_images_dir = d / "lane" / "images"
            overlay_dir = d / "readiness" / "keypoint_overlays"
            readiness_json = d / "readiness" / "readiness_per_mile.json"
            if not readiness_json.exists():
                continue
            sources.append(
                {
                    "id": d.name,
                    "label": d.name.replace("run_", "Archive "),
                    "readiness_json": readiness_json.as_posix(),
                    "evaluated_dir": evaluated_dir.as_posix(),
                    "lane_images_dir": lane_images_dir.as_posix(),
                    "overlay_dir": overlay_dir.as_posix(),
                    "report_pdf": (d / "research_report.pdf").as_posix(),
                }
            )
    return sources


def _refresh_all_evaluations_outputs() -> None:
    """Build the combined all-evaluations dashboard and map."""
    from pipeline import readiness_heatmap

    specs = _list_all_evaluation_sources()
    if not specs:
        return
    cache_key = "__all_evaluations__"
    source_script = REPO_ROOT / "pipeline" / "readiness_heatmap.py"

    def _path_sig(path_str: str) -> tuple:
        try:
            p = Path(path_str)
            if not p.exists():
                return (path_str, False, None)
            return (path_str, True, p.stat().st_mtime_ns)
        except Exception:
            return (path_str, False, None)

    try:
        spec_sig = tuple(
            (
                str(spec.get("id", "")),
                _path_sig(str(spec.get("readiness_json", ""))),
                _path_sig(str(spec.get("report_pdf", ""))),
            )
            for spec in specs
        )
        before_sig = (
            source_script.stat().st_mtime_ns,
            ALL_EVALUATIONS_DASHBOARD.exists(),
            ALL_EVALUATIONS_DASHBOARD.stat().st_mtime_ns if ALL_EVALUATIONS_DASHBOARD.exists() else None,
            ALL_EVALUATIONS_MAP.exists(),
            ALL_EVALUATIONS_MAP.stat().st_mtime_ns if ALL_EVALUATIONS_MAP.exists() else None,
            spec_sig,
        )
    except Exception:
        before_sig = None

    with ARCHIVE_REFRESH_LOCK:
        cached = ARCHIVE_REFRESH_CACHE.get(cache_key)
        if cached and cached.get("signature") == before_sig and not cached.get("needs_refresh", True):
            return

        needs_refresh = (
            not ALL_EVALUATIONS_DASHBOARD.exists()
            or not ALL_EVALUATIONS_MAP.exists()
            or ALL_EVALUATIONS_DASHBOARD.stat().st_size <= 0
            or ALL_EVALUATIONS_MAP.stat().st_size <= 0
        )
        try:
            if not needs_refresh:
                if ALL_EVALUATIONS_DASHBOARD.stat().st_mtime < source_script.stat().st_mtime:
                    needs_refresh = True
                elif ALL_EVALUATIONS_MAP.stat().st_mtime < source_script.stat().st_mtime:
                    needs_refresh = True
                else:
                    for spec in specs:
                        for key in ("readiness_json", "report_pdf"):
                            raw = str(spec.get(key, "")).strip()
                            if not raw:
                                continue
                            p = Path(raw)
                            if p.exists() and (
                                ALL_EVALUATIONS_DASHBOARD.stat().st_mtime < p.stat().st_mtime
                                or ALL_EVALUATIONS_MAP.stat().st_mtime < p.stat().st_mtime
                            ):
                                needs_refresh = True
                                break
                        if needs_refresh:
                            break
        except Exception:
            needs_refresh = True

        ARCHIVE_REFRESH_CACHE[cache_key] = {
            "signature": before_sig,
            "needs_refresh": needs_refresh,
        }
        if not needs_refresh:
            return
        in_progress = ARCHIVE_REFRESH_EVENTS.get(cache_key)
        if in_progress is not None:
            wait_event = in_progress
        else:
            wait_event = threading.Event()
            ARCHIVE_REFRESH_EVENTS[cache_key] = wait_event
            wait_event = None

    if wait_event is not None:
        wait_event.wait()
        return

    try:
        READINESS_DIR.mkdir(parents=True, exist_ok=True)
        readiness_heatmap.build_multi_run_dashboard_bundle(
            run_specs=specs,
            out_map=ALL_EVALUATIONS_MAP,
            out_dashboard=ALL_EVALUATIONS_DASHBOARD,
            mapbox_token=os.getenv("MAPBOX_TOKEN"),
        )
        with ARCHIVE_REFRESH_LOCK:
            try:
                after_sig = (
                    source_script.stat().st_mtime_ns,
                    ALL_EVALUATIONS_DASHBOARD.exists(),
                    ALL_EVALUATIONS_DASHBOARD.stat().st_mtime_ns if ALL_EVALUATIONS_DASHBOARD.exists() else None,
                    ALL_EVALUATIONS_MAP.exists(),
                    ALL_EVALUATIONS_MAP.stat().st_mtime_ns if ALL_EVALUATIONS_MAP.exists() else None,
                    tuple(
                        (
                            str(spec.get("id", "")),
                            _path_sig(str(spec.get("readiness_json", ""))),
                            _path_sig(str(spec.get("report_pdf", ""))),
                        )
                        for spec in specs
                    ),
                )
            except Exception:
                after_sig = None
            ARCHIVE_REFRESH_CACHE[cache_key] = {
                "signature": after_sig,
                "needs_refresh": False,
            }
    finally:
        with ARCHIVE_REFRESH_LOCK:
            done_event = ARCHIVE_REFRESH_EVENTS.pop(cache_key, None)
        if done_event is not None:
            done_event.set()


def _safe_repo_file(relpath: str) -> Path:
    """Resolve and validate an artifact path relative to the repository root."""
    normalized = relpath.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("Empty artifact path")
    parts = Path(normalized).parts
    if not parts:
        raise ValueError("Invalid artifact path")
    if parts[0] not in ARTIFACT_ROOT_PREFIXES:
        raise ValueError("Artifact path outside allowed root")

    p = (REPO_ROOT / normalized).resolve()
    p.relative_to(REPO_ROOT)
    allowed_roots = [(REPO_ROOT / prefix).resolve() for prefix in ARTIFACT_ROOT_PREFIXES]
    if not any(_path_within(p, root) for root in allowed_roots):
        raise ValueError("Artifact path outside allowed root")
    return p


def _archive_root_from_relpath(relpath: str) -> Optional[Path]:
    """Map an archived artifact relative path back to its archive run root."""
    normalized = relpath.strip().replace("\\", "/")
    parts = Path(normalized).parts
    expected_prefix = ("results", "archive", "readiness_runs")
    if len(parts) < 5 or tuple(parts[:3]) != expected_prefix:
        return None
    run_name = parts[3]
    if not run_name.startswith("run_"):
        return None
    return (READINESS_ARCHIVE_ROOT / run_name).resolve()


def _current_readiness_signature() -> Optional[tuple]:
    """Build a refresh signature for current readiness artifacts and source code."""
    dashboard = READINESS_DIR / "readiness_dashboard.html"
    heatmap = READINESS_DIR / "readiness_heatmap.html"
    source_script = REPO_ROOT / "pipeline" / "readiness_heatmap.py"
    try:
        return (
            source_script.stat().st_mtime_ns,
            dashboard.exists(),
            dashboard.stat().st_mtime_ns if dashboard.exists() else None,
            heatmap.exists(),
            heatmap.stat().st_mtime_ns if heatmap.exists() else None,
        )
    except Exception:
        return None


def _archive_refresh_signature(archive_root: Path) -> Optional[tuple]:
    """Build a refresh signature for one archived readiness snapshot."""
    dashboard = archive_root / "readiness" / "readiness_dashboard.html"
    heatmap = archive_root / "readiness" / "readiness_heatmap.html"
    source_script = REPO_ROOT / "pipeline" / "readiness_heatmap.py"
    try:
        return (
            source_script.stat().st_mtime_ns,
            dashboard.exists(),
            dashboard.stat().st_mtime_ns if dashboard.exists() else None,
            heatmap.exists(),
            heatmap.stat().st_mtime_ns if heatmap.exists() else None,
        )
    except Exception:
        return None


def _should_check_archived_refresh(relpath: str) -> bool:
    """Return whether an archived artifact request should trigger refresh checks."""
    normalized = relpath.strip().replace("\\", "/").lower()
    return normalized.endswith("/readiness/readiness_dashboard.html") or normalized.endswith(
        "/readiness/readiness_heatmap.html"
    )


def _should_check_current_refresh(relpath: str) -> bool:
    """Return whether a current artifact request should trigger refresh checks."""
    normalized = relpath.strip().replace("\\", "/").lower()
    return normalized in {
        "results/readiness/readiness_dashboard.html",
        "results/readiness/readiness_heatmap.html",
    }


def _needs_archived_refresh(archive_root: Path) -> bool:
    """Detect whether an archived readiness dashboard/heatmap should be regenerated."""
    dashboard = archive_root / "readiness" / "readiness_dashboard.html"
    heatmap = archive_root / "readiness" / "readiness_heatmap.html"
    source_script = REPO_ROOT / "pipeline" / "readiness_heatmap.py"
    if not dashboard.exists() or not heatmap.exists():
        return True
    try:
        if dashboard.stat().st_mtime < source_script.stat().st_mtime:
            return True
        if heatmap.stat().st_mtime < source_script.stat().st_mtime:
            return True
        snippet = heatmap.read_text(encoding="utf-8", errors="ignore")[:12000]
        return (
            "Heatmap Layer" not in snippet
            or "basemap-select" not in snippet
            or "readiness.basemap" not in snippet
            or "metric-toggle" not in snippet
            or "keypoint-toggle" not in snippet
            or "keypointsEnabled" not in snippet
        )
    except Exception:
        return True


def _needs_current_refresh() -> bool:
    """Detect whether current readiness artifacts should be regenerated."""
    dashboard = READINESS_DIR / "readiness_dashboard.html"
    heatmap = READINESS_DIR / "readiness_heatmap.html"
    source_script = REPO_ROOT / "pipeline" / "readiness_heatmap.py"
    if not dashboard.exists() or not heatmap.exists():
        return True
    try:
        if dashboard.stat().st_mtime < source_script.stat().st_mtime:
            return True
        if heatmap.stat().st_mtime < source_script.stat().st_mtime:
            return True
        snippet = heatmap.read_text(encoding="utf-8", errors="ignore")[:12000]
        return (
            "Heatmap Layer" not in snippet
            or "basemap-select" not in snippet
            or "readiness.basemap" not in snippet
            or "metric-toggle" not in snippet
            or "keypoint-toggle" not in snippet
            or "keypointsEnabled" not in snippet
        )
    except Exception:
        return True


def _refresh_current_readiness_outputs() -> None:
    """Regenerate current readiness outputs when they are missing or stale."""
    evaluated_dir = RESULTS_DIR / "evaluated"
    lane_images_dir = RESULTS_DIR / "lane" / "images"
    overlay_dir = READINESS_DIR / "keypoint_overlays"
    if not evaluated_dir.exists() or (not lane_images_dir.exists() and not overlay_dir.exists()):
        return

    cache_key = "__current__"
    with ARCHIVE_REFRESH_LOCK:
        before_sig = _current_readiness_signature()
        cached = ARCHIVE_REFRESH_CACHE.get(cache_key)
        if cached and cached.get("signature") == before_sig and not cached.get("needs_refresh", True):
            return

        needs_refresh = _needs_current_refresh()
        ARCHIVE_REFRESH_CACHE[cache_key] = {
            "signature": before_sig,
            "needs_refresh": needs_refresh,
        }
        if not needs_refresh:
            return
        in_progress = ARCHIVE_REFRESH_EVENTS.get(cache_key)
        if in_progress is not None:
            wait_event = in_progress
        else:
            wait_event = threading.Event()
            ARCHIVE_REFRESH_EVENTS[cache_key] = wait_event
            wait_event = None

    if wait_event is not None:
        wait_event.wait()
        return

    try:
        READINESS_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "pipeline/readiness_heatmap.py",
            "--input-dir",
            evaluated_dir.as_posix(),
            "--output-json",
            (READINESS_DIR / "readiness_per_mile.json").as_posix(),
            "--output-map",
            (READINESS_DIR / "readiness_heatmap.html").as_posix(),
            "--output-dashboard",
            (READINESS_DIR / "readiness_dashboard.html").as_posix(),
            "--report-pdf",
            (RESULTS_DIR / "research_report.pdf").as_posix(),
            "--lane-images-dir",
            lane_images_dir.as_posix(),
            "--overlay-dir",
            overlay_dir.as_posix(),
            "--keypoint-stride",
            "10",
        ]
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT.as_posix(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            _append_log("Current readiness refresh failed: " + (proc.stdout.strip() or "unknown error"))
            return

        with ARCHIVE_REFRESH_LOCK:
            ARCHIVE_REFRESH_CACHE[cache_key] = {
                "signature": _current_readiness_signature(),
                "needs_refresh": False,
            }
    finally:
        with ARCHIVE_REFRESH_LOCK:
            done_event = ARCHIVE_REFRESH_EVENTS.pop(cache_key, None)
        if done_event is not None:
            done_event.set()


def _refresh_archived_readiness_outputs(relpath: str) -> None:
    """Regenerate archived readiness outputs for the archive containing ``relpath``."""
    archive_root = _archive_root_from_relpath(relpath)
    if archive_root is None or not archive_root.exists():
        return

    cache_key = archive_root.as_posix()
    with ARCHIVE_REFRESH_LOCK:
        before_sig = _archive_refresh_signature(archive_root)
        cached = ARCHIVE_REFRESH_CACHE.get(cache_key)
        if cached and cached.get("signature") == before_sig and not cached.get("needs_refresh", True):
            return

        needs_refresh = _needs_archived_refresh(archive_root)
        ARCHIVE_REFRESH_CACHE[cache_key] = {
            "signature": before_sig,
            "needs_refresh": needs_refresh,
        }
        if not needs_refresh:
            return
        in_progress = ARCHIVE_REFRESH_EVENTS.get(cache_key)
        if in_progress is not None:
            wait_event = in_progress
        else:
            wait_event = threading.Event()
            ARCHIVE_REFRESH_EVENTS[cache_key] = wait_event
            wait_event = None

    if wait_event is not None:
        wait_event.wait()
        return

    try:
        evaluated_dir = archive_root / "evaluated"
        lane_images_dir = archive_root / "lane" / "images"
        overlay_dir = archive_root / "readiness" / "keypoint_overlays"
        if not evaluated_dir.exists() or (not lane_images_dir.exists() and not overlay_dir.exists()):
            return

        readiness_dir = archive_root / "readiness"
        readiness_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "pipeline/readiness_heatmap.py",
            "--input-dir",
            evaluated_dir.as_posix(),
            "--output-json",
            (readiness_dir / "readiness_per_mile.json").as_posix(),
            "--output-map",
            (readiness_dir / "readiness_heatmap.html").as_posix(),
            "--output-dashboard",
            (readiness_dir / "readiness_dashboard.html").as_posix(),
            "--report-pdf",
            (archive_root / "research_report.pdf").as_posix(),
            "--lane-images-dir",
            lane_images_dir.as_posix(),
            "--overlay-dir",
            overlay_dir.as_posix(),
            "--keypoint-stride",
            "10",
        ]
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT.as_posix(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            _append_log(
                "Archived readiness refresh failed for "
                f"{archive_root.name}: {proc.stdout.strip() or 'unknown error'}"
            )
            return

        with ARCHIVE_REFRESH_LOCK:
            ARCHIVE_REFRESH_CACHE[cache_key] = {
                "signature": _archive_refresh_signature(archive_root),
                "needs_refresh": False,
            }
    finally:
        with ARCHIVE_REFRESH_LOCK:
            done_event = ARCHIVE_REFRESH_EVENTS.pop(cache_key, None)
        if done_event is not None:
            done_event.set()


def _fallback_dashboard_html() -> str:
    """Return a minimal dashboard page for runs with no readiness data yet."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Road Readiness Dashboard</title>
  <style>
    html, body { margin: 0; height: 100%; font-family: Arial, sans-serif; }
    .wrap { display: flex; height: 100vh; width: 100vw; }
    .left { width: 38%; min-width: 360px; max-width: 620px; overflow: auto; padding: 14px; box-sizing: border-box; background: #f5f6f8; border-right: 1px solid #d9d9d9; }
    .right { flex: 1; min-width: 0; }
    .mapframe { width: 100%; height: 100%; border: none; }
    .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 10px; }
    .small { font-size: 12px; color: #666; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="left">
      <h1>Road Readiness Dashboard</h1>
      <div class="card">
        <b>No evaluation data available.</b>
        <div class="small" style="margin-top:6px;">
          Run the pipeline (or select an archived run) to populate readiness, key points, evidence images, and report.
        </div>
      </div>
    </div>
    <div class="right">
      <iframe class="mapframe" src="readiness_heatmap.html"></iframe>
    </div>
  </div>
</body>
</html>"""


def _fallback_heatmap_html() -> str:
    """Return a minimal base-map page for runs with no readiness heatmap yet."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Readiness Heatmap</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    html, body, #map { margin: 0; height: 100%; width: 100%; }
    .overlay {
      position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
      z-index: 900; background: rgba(255,255,255,0.95); border: 1px solid #ccc;
      border-radius: 8px; padding: 10px 14px; font-family: Arial, sans-serif; font-size: 13px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.2);
    }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="overlay"><b>No evaluation data to display.</b> Base map loaded.</div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const map = L.map('map').setView([36.17, -79.72], 12);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap'
    }).addTo(map);
  </script>
</body>
</html>"""


@app.get("/")
def ui_index() -> Response:
    """Serve the main Flask UI with pipeline controls and dashboard iframe."""
    default_bag = (_load_json(DEFAULT_CONFIG).get("bag_file") or "")
    requires_api_token = "true" if APP_API_TOKEN else "false"
    enable_full_pipeline = "true" if ENABLE_FULL_PIPELINE else "false"
    enable_reevaluation = "true" if ENABLE_REEVALUATION else "false"
    enable_report_regen = "true" if ENABLE_REPORT_REGEN else "false"
    enable_fs_browser = "true" if ENABLE_FS_BROWSER else "false"
    public_mode = "true" if PUBLIC_MODE else "false"
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Readiness Pipeline UI (Flask)</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background:#f4f5f7; }}
    .root {{ height: 100vh; display: flex; flex-direction: column; }}
    .status {{ padding:10px; border-bottom:1px solid #ddd; background:#fafafa; font-size:13px; }}
    .tabs {{ display:flex; border-bottom:1px solid #ddd; background:#fff; }}
    .tab-btn {{ padding:10px 14px; cursor:pointer; border:none; background:transparent; font-weight:600; }}
    .tab-btn.active {{ color:#0f4c81; border-bottom:2px solid #0f4c81; }}
    .main {{ flex:1; min-height:0; display:flex; flex-direction:column; }}
    .tab-panel {{ display:none; flex:1; min-height:0; overflow:auto; }}
    .tab-panel.active {{ display:block; }}
    .run-wrap {{ max-width:900px; padding:14px; }}
    .row {{ margin-bottom:10px; }}
    .inline {{ display:flex; gap:8px; }}
    label {{ font-size:12px; color:#444; display:block; margin-bottom:4px; }}
    input, select {{ width:100%; box-sizing:border-box; padding:7px; }}
    button {{ padding:8px 12px; }}
    .weights {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
    .hint {{ font-size:12px; color:#666; }}
    .notice {{ margin-bottom:12px; border-radius:8px; border:1px solid #d8dde4; background:#fff; padding:10px 12px; font-size:12px; color:#334; }}
    .dash-wrap {{ height:100%; display:flex; flex-direction:column; }}
    .dash-toolbar {{ padding:8px; border-bottom:1px solid #ddd; background:#fff; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
    .dash-view {{ position:relative; flex:1; min-height:0; }}
    iframe {{ flex:1; border:none; width:100%; height:100%; }}
    .dash-loading {{
      position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
      background:rgba(244,245,247,0.92); z-index:10; transition:opacity 0.18s ease;
    }}
    .dash-loading.hidden {{ opacity:0; pointer-events:none; }}
    .loading-card {{
      min-width:280px; max-width:420px; background:#fff; border:1px solid #d4d8de; border-radius:12px;
      box-shadow:0 12px 34px rgba(24,39,75,0.14); padding:18px 20px; text-align:center;
    }}
    .spinner {{
      width:34px; height:34px; margin:0 auto 12px auto; border-radius:999px;
      border:4px solid #d7e5f1; border-top-color:#0f4c81; animation:spin 0.9s linear infinite;
    }}
    .loading-title {{ font-size:15px; font-weight:700; color:#18364d; }}
    .loading-copy {{ font-size:12px; color:#5b6b77; margin-top:6px; line-height:1.45; }}
    .loading-note {{ font-size:11px; color:#7b8790; margin-top:10px; }}
    @keyframes spin {{ from {{ transform:rotate(0deg); }} to {{ transform:rotate(360deg); }} }}
    .log-head {{ display:flex; align-items:center; justify-content:space-between; background:#161616; color:#ddd; padding:6px 10px; font-size:12px; }}
    pre {{ background:#111; color:#d7ffd7; padding:10px; margin:0; height:220px; min-height:90px; max-height:50vh; overflow:auto; font-size:12px; resize:vertical; }}
    .hidden {{ display:none !important; }}
    .modal-backdrop {{ position:fixed; inset:0; background:rgba(0,0,0,0.35); display:none; align-items:center; justify-content:center; z-index:1000; }}
    .modal {{ width:700px; max-width:92vw; height:520px; background:#fff; border-radius:8px; border:1px solid #ccc; display:flex; flex-direction:column; }}
    .modal-head {{ padding:10px; border-bottom:1px solid #ddd; display:flex; gap:8px; align-items:center; }}
    .modal-body {{ flex:1; overflow:auto; padding:8px; }}
    .dir-item {{ padding:6px 8px; border-bottom:1px solid #eee; cursor:pointer; font-size:13px; }}
    .dir-item:hover {{ background:#f5f8ff; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; }}
    .badge {{ display:inline-block; padding:2px 8px; border:1px solid #cdd5de; border-radius:999px; background:#f6f8fb; font-size:11px; color:#4b5b6a; }}
  </style>
</head>
<body>
<div class="root">
  <div class="tabs">
    <button class="tab-btn" data-tab="runTab" onclick="switchTab('runTab', this)">Run Pipeline</button>
    <button class="tab-btn active" data-tab="dashTab" onclick="switchTab('dashTab', this)">Road Readiness Dashboard</button>
    <button class="tab-btn" data-tab="allDashTab" onclick="switchTab('allDashTab', this); ensureAllEvaluationsLoaded();">All Evaluated Results</button>
  </div>
  <div class="status">
    <span id="deploySummary">Deployment mode</span>
  </div>
  <div class="main">
    <div id="runTab" class="tab-panel">
      <div class="run-wrap">
        <h3 style="margin-top:0;">Run Pipeline</h3>
        <div id="pipelineDisabledNotice" class="notice hidden">
          Full rosbag processing is disabled in this deployment. You can still run re-evaluation from the dashboard tab.
        </div>
        <div class="row" id="tokenRow">
          <label>API Token (if required by deployment)</label>
          <input id="apiToken" type="password" placeholder="Paste token to run/stop/re-evaluate endpoints" />
          <div class="hint">Token is stored in this browser only.</div>
        </div>
        <div class="row">
          <label>Rosbag Path</label>
          <div class="inline">
            <input id="bag_file" value="{default_bag}" />
            <button id="browseBtn" type="button" onclick="openBrowser()">Browse...</button>
            <button id="localDialogBtn" type="button" onclick="openLocalDialog()">Local Dialog</button>
            <input id="bag_picker" type="file" webkitdirectory directory multiple class="hidden" onchange="handleLocalDialog(event)" />
          </div>
        </div>
        <div class="row"><label>Frame Stride</label><input id="frame_stride" value="10"/></div>
        <div id="browseHint" class="hint">Use <b>Browse...</b> for server-side filesystem path selection. Browser local dialog is kept for convenience but does not provide reliable absolute server path.</div>
        <hr/>
        <div class="weights">
          <div class="row"><label>w1 (Physical Infra Weight)</label><input id="w1" value="0.6"/></div>
          <div class="row"><label>w2 (Digital Infra Weight)</label><input id="w2" value="0.4"/></div>
          <div class="row"><label>m1 (Connectivity in Digital)</label><input id="m1" value="0.5"/></div>
          <div class="row"><label>m2 (GPS in Digital)</label><input id="m2" value="0.4"/></div>
          <div class="row"><label>m3 (HD Maps in Digital)</label><input id="m3" value="0.1"/></div>
          <div class="row"><label>keypoint_stride</label><input id="keypoint_stride" value="10"/></div>
        </div>
        <div class="row inline">
          <button id="startBtn" onclick="startRun()">Start Pipeline</button>
          <button id="stopBtn" onclick="stopRun()" disabled>Stop Pipeline</button>
        </div>
        <div class="status" style="margin-top:8px;">
          <div id="status">Status: idle</div>
          <div style="height:10px;background:#ddd;border-radius:999px;margin-top:6px;overflow:hidden;">
            <div id="progressBar" style="height:100%;width:0%;background:#2e7d32;"></div>
          </div>
          <div id="progressText" style="font-size:12px;color:#555;margin-top:4px;">Idle</div>
        </div>
        <div class="log-head" style="margin-top:10px;">
          <span>Pipeline Log</span>
          <button style="padding:4px 8px;" onclick="toggleLogs()">Minimize/Restore</button>
        </div>
        <pre id="logs"></pre>
      </div>
    </div>

    <div id="dashTab" class="tab-panel active">
      <div class="dash-wrap">
        <div class="dash-toolbar">
          <label style="margin:0;">Archived Run</label>
          <select id="runSelect" style="max-width:420px;"></select>
          <button onclick="loadSelectedRun()">Load</button>
          <button id="reweightBtn" onclick="startReweightFromSelected()">Re-evaluate Selected</button>
          <span class="badge" id="reportBadge">Report enabled</span>
        </div>
        <div class="dash-view">
          <div id="dashLoading" class="dash-loading">
            <div class="loading-card">
              <div class="spinner"></div>
              <div class="loading-title">Loading dashboard</div>
              <div id="dashLoadingText" class="loading-copy">Preparing the selected run and loading the split-screen readiness view.</div>
              <div class="loading-note">Large archived runs can take a bit longer while the browser loads evidence images.</div>
            </div>
          </div>
          <iframe id="dash"></iframe>
        </div>
      </div>
    </div>

    <div id="allDashTab" class="tab-panel">
      <div class="dash-wrap">
        <div class="dash-toolbar">
          <div style="font-weight:600;color:#18364d;">All evaluations in one map view</div>
          <button onclick="ensureAllEvaluationsLoaded(true)">Refresh</button>
        </div>
        <div class="dash-view">
          <div id="allDashLoading" class="dash-loading">
            <div class="loading-card">
              <div class="spinner"></div>
              <div class="loading-title">Loading all evaluated results</div>
              <div id="allDashLoadingText" class="loading-copy">Combining current and archived evaluations into one map while keeping each run summary separate.</div>
              <div class="loading-note">Archived runs without evaluated data are skipped automatically.</div>
            </div>
          </div>
          <iframe id="allDash"></iframe>
        </div>
      </div>
    </div>
  </div>
</div>

<div id="browserModal" class="modal-backdrop">
  <div class="modal">
    <div class="modal-head">
      <input id="browserPath" class="mono" style="flex:1;" />
      <button onclick="goBrowserPath()">Go</button>
      <button onclick="selectCurrentPath()">Select This Folder</button>
      <button onclick="closeBrowser()">Close</button>
    </div>
    <div class="modal-body" id="browserList"></div>
  </div>
</div>

<script>
const APP_CONFIG = {{
  requiresApiToken: {requires_api_token},
  enableFullPipeline: {enable_full_pipeline},
  enableReevaluation: {enable_reevaluation},
  enableReportRegen: {enable_report_regen},
  enableFsBrowser: {enable_fs_browser},
  publicMode: {public_mode}
}};

let currentBrowsePath = '';
let lastLoadedRunId = '';
let logsMin = false;
let dashLoadToken = 0;
let dashLoadTimer = null;
let allEvaluationsLoaded = false;
let runsByDashboardPath = {{}};

function switchTab(tabId, btn) {{
  for (const el of document.querySelectorAll('.tab-panel')) el.classList.remove('active');
  const target = document.getElementById(tabId);
  if (target) target.classList.add('active');
  for (const b of document.querySelectorAll('.tab-btn')) b.classList.remove('active');
  if (btn) btn.classList.add('active');
}}

function toggleLogs() {{
  logsMin = !logsMin;
  document.getElementById('logs').style.display = logsMin ? 'none' : 'block';
}}

function getApiToken() {{
  const input = document.getElementById('apiToken');
  const raw = (input ? input.value : '') || window.localStorage.getItem('readiness_api_token') || '';
  return raw.trim();
}}

function saveApiToken() {{
  const token = getApiToken();
  if (token) {{
    window.localStorage.setItem('readiness_api_token', token);
  }} else {{
    window.localStorage.removeItem('readiness_api_token');
  }}
}}

function requireTokenIfNeeded() {{
  if (!APP_CONFIG.requiresApiToken) return true;
  if (getApiToken()) return true;
  alert('This deployment requires an API token for this action.');
  return false;
}}

function jsonHeaders() {{
  const h = {{ 'Content-Type': 'application/json' }};
  const token = getApiToken();
  if (token) h['Authorization'] = 'Bearer ' + token;
  return h;
}}

function requestHeaders() {{
  const h = {{}};
  const token = getApiToken();
  if (token) h['Authorization'] = 'Bearer ' + token;
  return h;
}}

function applyDeploymentMode() {{
  const summary = document.getElementById('deploySummary');
  if (summary) {{
    const parts = [];
    parts.push(APP_CONFIG.publicMode ? 'Public mode' : 'Private mode');
    parts.push(APP_CONFIG.enableFullPipeline ? 'Full pipeline enabled' : 'Full pipeline disabled');
    parts.push(APP_CONFIG.enableReevaluation ? 'Re-evaluation enabled' : 'Re-evaluation disabled');
    parts.push(APP_CONFIG.enableReportRegen ? 'Report regeneration enabled' : 'Report regeneration disabled');
    summary.textContent = parts.join(' | ');
  }}

  const tokenInput = document.getElementById('apiToken');
  const storedToken = window.localStorage.getItem('readiness_api_token') || '';
  if (tokenInput && storedToken && !tokenInput.value) tokenInput.value = storedToken;
  if (tokenInput) tokenInput.addEventListener('change', saveApiToken);

  if (!APP_CONFIG.enableFullPipeline) {{
    const notice = document.getElementById('pipelineDisabledNotice');
    if (notice) notice.classList.remove('hidden');
    const bag = document.getElementById('bag_file');
    const frameStride = document.getElementById('frame_stride');
    const browseBtn = document.getElementById('browseBtn');
    const localDialogBtn = document.getElementById('localDialogBtn');
    if (bag) bag.disabled = true;
    if (frameStride) frameStride.disabled = true;
    if (browseBtn) browseBtn.disabled = true;
    if (localDialogBtn) localDialogBtn.disabled = true;
  }}

  if (!APP_CONFIG.enableFsBrowser) {{
    const browseBtn = document.getElementById('browseBtn');
    if (browseBtn) browseBtn.disabled = true;
    const browseHint = document.getElementById('browseHint');
    if (browseHint) browseHint.textContent = 'Server-side filesystem browsing is disabled in this deployment.';
  }}

  if (!APP_CONFIG.enableReevaluation) {{
    const rb = document.getElementById('reweightBtn');
    if (rb) rb.disabled = true;
  }}

  const reportBadge = document.getElementById('reportBadge');
  if (reportBadge) {{
    reportBadge.textContent = APP_CONFIG.enableReportRegen ? 'Report enabled' : 'Report disabled';
  }}
}}

async function fetchRuns() {{
  const res = await fetch('/api/runs');
  const j = await res.json();
  const sel = document.getElementById('runSelect');
  const prev = sel.value;
  sel.innerHTML = '';
  runsByDashboardPath = {{}};
  for (const r of (j.runs || [])) {{
    const opt = document.createElement('option');
    opt.value = r.dashboard_path;
    opt.textContent = r.label;
    sel.appendChild(opt);
    runsByDashboardPath[r.dashboard_path] = r;
  }}
  if (prev && runsByDashboardPath[prev]) {{
    sel.value = prev;
  }} else if (sel.options.length > 0) {{
    sel.selectedIndex = 0;
  }}
}}

function showDashLoading(message) {{
  const overlay = document.getElementById('dashLoading');
  const copy = document.getElementById('dashLoadingText');
  if (copy) copy.textContent = message || 'Loading dashboard...';
  if (overlay) overlay.classList.remove('hidden');
}}

function hideDashLoading(token, overlayId) {{
  if (token && token !== dashLoadToken) return;
  const overlay = document.getElementById(overlayId || 'dashLoading');
  if (overlay) overlay.classList.add('hidden');
  if (dashLoadTimer) {{
    clearTimeout(dashLoadTimer);
    dashLoadTimer = null;
  }}
}}

function loadDashboard(path, label) {{
  if (!path) return;
  dashLoadToken += 1;
  const token = dashLoadToken;
  const runLabel = label || 'selected run';
  showDashLoading(`Loading ${{runLabel}}. Large archived runs can take a bit while key point images render.`);
  const frame = document.getElementById('dash');
  if (!frame) return;
  frame.onload = () => hideDashLoading(token);
  frame.onerror = () => hideDashLoading(token);
  if (dashLoadTimer) clearTimeout(dashLoadTimer);
  dashLoadTimer = setTimeout(() => {{
    if (token !== dashLoadToken) return;
    showDashLoading(`Still loading ${{runLabel}}. The browser is working through the dashboard and embedded map.`);
  }}, 3500);
  frame.src = '/artifact/' + path.replace(/^\\//,'') + '?t=' + Date.now();
}}

function loadFrame(frameId, overlayId, textId, path, label, initialMessage, slowMessage) {{
  if (!path) return;
  dashLoadToken += 1;
  const token = dashLoadToken;
  const runLabel = label || 'selected view';
  const overlay = document.getElementById(overlayId);
  const copy = document.getElementById(textId);
  if (copy) copy.textContent = initialMessage || `Loading ${{runLabel}}...`;
  if (overlay) overlay.classList.remove('hidden');
  const frame = document.getElementById(frameId);
  if (!frame) return;
  frame.onload = () => hideDashLoading(token, overlayId);
  frame.onerror = () => hideDashLoading(token, overlayId);
  if (dashLoadTimer) clearTimeout(dashLoadTimer);
  dashLoadTimer = setTimeout(() => {{
    if (token !== dashLoadToken) return;
    if (copy) copy.textContent = slowMessage || `Still loading ${{runLabel}}...`;
    if (overlay) overlay.classList.remove('hidden');
  }}, 3500);
  frame.src = '/artifact/' + path.replace(/^\\//,'') + '?t=' + Date.now();
}}

function loadAllEvaluations() {{
  loadFrame(
    'allDash',
    'allDashLoading',
    'allDashLoadingText',
    'results/readiness/all_evaluations_dashboard.html',
    'all evaluations',
    'Combining current and archived evaluations into one map while keeping each run summary separate.',
    'Still combining evaluated results. Large archives can take a bit while the shared map and sidebar cards render.'
  );
}}

function ensureAllEvaluationsLoaded(force = false) {{
  if (force || !allEvaluationsLoaded) {{
    loadAllEvaluations();
    allEvaluationsLoaded = true;
  }}
}}

function buildReweightPayload(runMeta) {{
  return {{
    source_fused_dir: runMeta.source_fused_dir,
    source_lane_images_dir: runMeta.source_lane_images_dir,
    w1: document.getElementById('w1').value,
    w2: document.getElementById('w2').value,
    m1: document.getElementById('m1').value,
    m2: document.getElementById('m2').value,
    m3: document.getElementById('m3').value,
    keypoint_stride: document.getElementById('keypoint_stride').value
  }};
}}

function loadSelectedRun() {{
  const sel = document.getElementById('runSelect');
  const path = sel.value;
  if (!path) return;
  const label = sel.options[sel.selectedIndex] ? sel.options[sel.selectedIndex].textContent : 'selected run';
  loadDashboard(path, label);
  switchTab('dashTab', document.querySelector('.tab-btn[data-tab="dashTab"]'));
}}

async function startRun() {{
  if (!APP_CONFIG.enableFullPipeline) {{
    alert('Full rosbag pipeline is disabled in this deployment.');
    return;
  }}
  if (!requireTokenIfNeeded()) return;
  saveApiToken();
  const payload = {{
    bag_file: document.getElementById('bag_file').value,
    frame_stride: document.getElementById('frame_stride').value,
    w1: document.getElementById('w1').value,
    w2: document.getElementById('w2').value,
    m1: document.getElementById('m1').value,
    m2: document.getElementById('m2').value,
    m3: document.getElementById('m3').value,
    keypoint_stride: document.getElementById('keypoint_stride').value
  }};
  const res = await fetch('/api/run', {{ method:'POST', headers: jsonHeaders(), body: JSON.stringify(payload) }});
  const j = await res.json();
  if (!j.ok) {{
    alert(j.error || 'Failed to start run');
    return;
  }}
  lastLoadedRunId = '';
  switchTab('runTab', document.querySelector('.tab-btn[data-tab="runTab"]'));
}}

async function startReweightFromSelected() {{
  if (!APP_CONFIG.enableReevaluation) {{
    alert('Re-evaluation is disabled in this deployment.');
    return;
  }}
  if (!requireTokenIfNeeded()) return;
  saveApiToken();

  const sel = document.getElementById('runSelect');
  if (!sel || !sel.value) {{
    alert('No run selected for re-evaluation.');
    return;
  }}
  const runMeta = runsByDashboardPath[sel.value];
  if (!runMeta) {{
    alert('Selected run metadata is unavailable.');
    return;
  }}

  const payload = buildReweightPayload(runMeta);
  const res = await fetch('/api/reweight', {{ method:'POST', headers: jsonHeaders(), body: JSON.stringify(payload) }});
  const j = await res.json();
  if (!j.ok) {{
    alert(j.error || 'Failed to start re-evaluation');
    return;
  }}
  lastLoadedRunId = '';
  switchTab('runTab', document.querySelector('.tab-btn[data-tab="runTab"]'));
}}

async function stopRun() {{
  if (!requireTokenIfNeeded()) return;
  saveApiToken();
  const res = await fetch('/api/stop', {{ method:'POST', headers: requestHeaders() }});
  const j = await res.json();
  if (!j.ok) {{
    alert(j.error || 'Failed to stop run');
    return;
  }}
}}

async function poll() {{
  const res = await fetch('/api/status');
  const j = await res.json();
  document.getElementById('status').innerText = `Status: ${{j.stage}} | running=${{j.running}} | run_id=${{j.last_run_id || '-'}}`;
  const pct = Math.max(0, Math.min(100, Number(j.progress_pct || 0)));
  document.getElementById('progressBar').style.width = pct + '%';
  document.getElementById('progressBar').style.background = (j.stage === 'failed') ? '#c62828' : '#2e7d32';
  document.getElementById('progressText').innerText = `${{pct.toFixed(0)}}% - ${{j.progress_text || ''}}`;
  document.getElementById('logs').textContent = (j.logs_tail || []).join('\\n');
  const running = Boolean(j.running);
  document.getElementById('startBtn').disabled = running || !APP_CONFIG.enableFullPipeline;
  document.getElementById('stopBtn').disabled = !running;
  const reBtn = document.getElementById('reweightBtn');
  if (reBtn) reBtn.disabled = running || !APP_CONFIG.enableReevaluation;

  if (j.stage === 'completed' && j.reload_dashboard_on_complete && j.dashboard_path && j.last_run_id && j.last_run_id !== lastLoadedRunId) {{
    loadDashboard(j.dashboard_path, 'latest run');
    lastLoadedRunId = j.last_run_id;
    await fetchRuns();
  }} else if (j.stage === 'stopped' && j.reload_dashboard_on_complete && j.last_run_id && j.last_run_id !== lastLoadedRunId) {{
    await fetchRuns();
    const sel = document.getElementById('runSelect');
    if (sel && sel.options.length > 0) {{
      sel.selectedIndex = 0;
      loadSelectedRun();
    }}
    lastLoadedRunId = j.last_run_id;
  }}
}}

async function openBrowser() {{
  if (!APP_CONFIG.enableFsBrowser) {{
    alert('Filesystem browser is disabled in this deployment.');
    return;
  }}
  if (!requireTokenIfNeeded()) return;
  saveApiToken();
  document.getElementById('browserModal').style.display = 'flex';
  const seed = document.getElementById('bag_file').value || '/';
  await loadDir(seed);
}}

function closeBrowser() {{
  document.getElementById('browserModal').style.display = 'none';
}}

function openLocalDialog() {{
  document.getElementById('bag_picker').click();
}}

function handleLocalDialog(evt) {{
  const files = evt.target.files || [];
  if (!files.length) return;
  const rel = files[0].webkitRelativePath || '';
  const topDir = rel.split('/')[0] || '';
  alert('Selected in local dialog: ' + (topDir || '(folder)') + '. For absolute server path, use Browse...');
}}

async function goBrowserPath() {{
  await loadDir(document.getElementById('browserPath').value || '/');
}}

function selectCurrentPath() {{
  document.getElementById('bag_file').value = currentBrowsePath;
  closeBrowser();
}}

async function loadDir(path) {{
  const res = await fetch('/api/fs/list?path=' + encodeURIComponent(path || '/'), {{ headers: requestHeaders() }});
  const j = await res.json();
  if (!j.ok) {{
    alert(j.error || 'Failed to list directory');
    return;
  }}
  currentBrowsePath = j.path;
  document.getElementById('browserPath').value = j.path;
  const el = document.getElementById('browserList');
  const rows = [];
  if (j.parent) rows.push(`<div class="dir-item mono" onclick="loadDir('${{j.parent.replace(/'/g, "\\\\'")}}')">[..]</div>`);
  for (const d of (j.entries || [])) {{
    const p = String(d.path || '').replace(/'/g, "\\\\'");
    rows.push(`<div class="dir-item mono" onclick="loadDir('${{p}}')">[DIR] ${{d.name}}</div>`);
  }}
  if (!rows.length) rows.push('<div class="dir-item mono">No subdirectories</div>');
  el.innerHTML = rows.join('');
}}

applyDeploymentMode();
setInterval(poll, 2000);
poll();
fetchRuns();
loadDashboard('results/readiness/readiness_dashboard.html', 'latest run');
</script>
</body>
</html>"""
    return Response(html, mimetype="text/html")


@app.get("/api/status")
def api_status() -> Response:
    """Return current UI state and recent logs."""
    return jsonify(_snapshot_state())


@app.get("/api/runs")
def api_runs() -> Response:
    """Return the current and archived runs shown in the dashboard selector."""
    return jsonify({"runs": _list_runs()})


@app.post("/api/run")
def api_run() -> Response:
    """Start a full pipeline run from the posted UI parameters."""
    auth_err = _require_api_token("run")
    if auth_err is not None:
        return auth_err
    if not ENABLE_FULL_PIPELINE:
        return _feature_disabled("Full rosbag pipeline is disabled on this deployment.")

    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            raise ValueError("Invalid payload")
        params = {str(k): str(v) for k, v in payload.items()}
        return jsonify(_start_run(params))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/reweight")
def api_reweight() -> Response:
    """Start a reweight-only run from the posted UI parameters."""
    auth_err = _require_api_token("reweight")
    if auth_err is not None:
        return auth_err
    if not ENABLE_REEVALUATION:
        return _feature_disabled("Re-evaluation is disabled on this deployment.")

    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            raise ValueError("Invalid payload")
        params = {str(k): str(v) for k, v in payload.items()}
        return jsonify(_start_reweight(params))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/report")
def api_report() -> Response:
    """Start a report-only regeneration run."""
    auth_err = _require_api_token("report")
    if auth_err is not None:
        return auth_err
    if not ENABLE_REPORT_REGEN:
        return _feature_disabled("Report regeneration is disabled on this deployment.")

    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            raise ValueError("Invalid payload")
        params = {str(k): str(v) for k, v in payload.items()}
        return jsonify(_start_report_regen(params))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/stop")
def api_stop() -> Response:
    """Request cancellation of the currently running pipeline task."""
    auth_err = _require_api_token("stop")
    if auth_err is not None:
        return auth_err

    try:
        return jsonify(_request_stop())
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/api/fs/list")
def api_fs_list() -> Response:
    """List server-side directories for the bag-file browser dialog."""
    auth_err = _require_api_token("fs_list")
    if auth_err is not None:
        return auth_err
    if not ENABLE_FS_BROWSER:
        return _feature_disabled("Filesystem browser is disabled on this deployment.")

    try:
        raw = str(request.args.get("path", "/")).strip() or "/"
        p = Path(raw).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            raise ValueError(f"Not a directory: {p}")
        if not any(_path_within(p, root) for root in FS_BROWSER_ROOTS):
            raise PermissionError(f"Path is outside allowed roots: {p}")
        parent = str(p.parent) if p.parent != p else None
        if parent is not None:
            parent_path = Path(parent).resolve()
            if not any(_path_within(parent_path, root) for root in FS_BROWSER_ROOTS):
                parent = None
        entries = []
        for c in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if c.is_dir() and any(_path_within(c, root) for root in FS_BROWSER_ROOTS):
                entries.append({"name": c.name, "path": c.as_posix()})
        return jsonify({"ok": True, "path": p.as_posix(), "parent": parent, "entries": entries})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/artifact/<path:relpath>")
def artifact(relpath: str) -> Response:
    """Serve a generated artifact while performing refresh checks when needed."""
    try:
        normalized = relpath.strip().replace("\\", "/")
        if normalized in {
            "results/readiness/all_evaluations_dashboard.html",
            "results/readiness/all_evaluations_map.html",
        }:
            _refresh_all_evaluations_outputs()
        if _should_check_current_refresh(relpath):
            _refresh_current_readiness_outputs()
        if _should_check_archived_refresh(relpath):
            _refresh_archived_readiness_outputs(relpath)
        p = _safe_repo_file(relpath)
    except Exception:
        return jsonify({"error": "Invalid path"}), 403
    if not p.exists() or not p.is_file():
        normalized = relpath.strip().replace("\\\\", "/")
        if normalized == "results/readiness/readiness_dashboard.html":
            return Response(_fallback_dashboard_html(), mimetype="text/html")
        if normalized == "results/readiness/readiness_heatmap.html":
            return Response(_fallback_heatmap_html(), mimetype="text/html")
        return jsonify({"error": "Not found"}), 404
    return send_file(p)


def main() -> None:
    """CLI entry point for the Flask readiness UI server."""
    import argparse
    import webbrowser

    parser = argparse.ArgumentParser(description="Flask UI to run readiness pipeline and browse archived dashboards.")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"), help="Bind host")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8765")), help="Bind port")
    parser.add_argument("--reset-runtime", action="store_true", help="Clear previous ui runtime logs/state files.")
    parser.add_argument("--no-open-browser", action="store_true", help="Do not auto-open browser on startup.")
    args = parser.parse_args()

    if args.reset_runtime and RUNTIME_DIR.exists():
        shutil.rmtree(RUNTIME_DIR)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    READINESS_ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

    open_host = args.host
    if open_host in ("0.0.0.0", "::"):
        open_host = "127.0.0.1"
    url = f"http://{open_host}:{args.port}"

    print(f"Serving Flask UI at {url}")
    if not args.no_open_browser:
        threading.Timer(0.8, lambda: webbrowser.open_new_tab(url)).start()
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
