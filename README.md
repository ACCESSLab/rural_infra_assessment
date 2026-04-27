# Rural Infrastructure Assessment

This repository contains a deployment-ready road readiness assessment tool for
reviewing rural transportation infrastructure from vehicle sensor outputs. It
combines lane-marking quality, GPS quality, mobile broadband connectivity, and
HD-map availability into route-level readiness scores, then presents the results
through interactive maps, dashboards, and a generated research report.

The current repository is packaged as a lightweight public/demo subset: it
includes the Flask web UI, scoring and reporting scripts, Render deployment
configuration, and precomputed artifacts needed to browse and re-evaluate the
included route data.

## What This Project Does

- Scores road segments using physical and digital infrastructure signals.
- Aggregates frame-level evaluations into per-mile readiness summaries.
- Generates interactive Folium maps and HTML dashboards for route inspection.
- Provides a Flask UI for browsing outputs and adjusting score weights.
- Compiles evaluation results into a shareable PDF research report.
- Supports Render deployment with a locked Python runtime for geospatial
  dependencies.

## Readiness Signals

The readiness index is built from several components:

- **Lane marking quality**: evaluates detected lane geometry, continuity,
  curvature, and separation.
- **Connectivity**: looks up mobile broadband speed data using Ookla open data.
- **GPS quality**: scores positioning reliability from GPS metadata.
- **HD-map availability**: included as a configurable digital infrastructure
  component.

The default score combines physical and digital infrastructure using the
weights in the evaluation pipeline. These can be adjusted through the UI or
command-line options.

## Repository Structure

```text
.
├── config/                  # Default runtime and pipeline configuration
├── connectivity/            # Ookla and Census-based connectivity lookup tools
├── pipeline/                # Evaluation, map generation, report, and UI scripts
├── results/
│   ├── evaluated/           # Frame-level readiness evaluations
│   ├── fused/               # Fused lane/GPS/connectivity records
│   ├── lane/images/         # Extracted route imagery used by the dashboard
│   ├── readiness/           # HTML dashboards, maps, JSON summaries, overlays
│   └── research_report.pdf  # Generated report artifact
├── Procfile                 # Web process for hosted deployment
├── render.yaml              # Render service configuration
├── requirements.txt         # Python dependencies
└── README.md
```

Large historical archives and the original C++ lane-detection build tree are
not included in this deployment subset.

## Key Outputs

- `results/readiness/readiness_dashboard.html`: interactive dashboard for
  reviewing scored route frames and per-mile summaries.
- `results/readiness/readiness_heatmap.html`: route heatmap colored by readiness
  score.
- `results/readiness/readiness_per_mile.json`: machine-readable per-mile
  readiness aggregates.
- `results/readiness/keypoint_overlays/`: visual lane keypoint overlays used for
  inspection.
- `results/research_report.pdf`: compiled report summarizing the evaluation.

## Run Locally

This project is tested with Python 3.11. The runtime is pinned to `3.11.11` in
`.python-version` to avoid geospatial package build issues on newer runtimes.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 pipeline/readiness_ui_server.py
```

Then open the local URL printed by the server. By default, the UI binds to
`127.0.0.1:8765`.

## Regenerate Evaluation Artifacts

To re-score the included fused records:

```bash
python3 pipeline/evaluate_fused_results.py \
  --input-dir results/fused \
  --output-dir results/evaluated
```

To rebuild the readiness dashboard, map, and per-mile JSON:

```bash
python3 pipeline/readiness_heatmap.py --input-dir results/evaluated
```

To rebuild the PDF report:

```bash
python3 pipeline/compile_report.py
```

Report compilation can use the OpenAI API for narrative generation when
configured, so set `OPENAI_API_KEY` if you enable that workflow.

## Deployment

The repository includes a Render configuration in `render.yaml`. The hosted app
uses Gunicorn to serve `pipeline.readiness_ui_server:app`.

Important deployment settings:

- `PYTHON_VERSION=3.11.11`
- `PUBLIC_MODE=true`
- `ENABLE_FULL_PIPELINE=false`
- `ENABLE_REEVALUATION=true`
- `ENABLE_REPORT_REGEN=false`
- `ENABLE_FS_BROWSER=false`

These defaults make the public deployment suitable for browsing existing
artifacts and reweighting/re-evaluating the included results, while disabling
filesystem browsing and full raw-pipeline execution.

## Data Notes

Connectivity lookups use Ookla open data and Census county boundaries. The
lookup utilities cache downloaded source files under `connectivity/data_cache/`
when new data is requested.

This repository contains derived evaluation artifacts rather than the complete
raw sensor-processing workspace. To reproduce the full upstream pipeline, the
original bag files, lane detector executable, and omitted archive/build assets
would need to be restored.
