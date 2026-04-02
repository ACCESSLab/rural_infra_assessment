# rural_infra_assessment

Deployment-ready subset for Render.

Included:
- Flask UI + re-evaluation pipeline scripts
- Render deployment files (`requirements.txt`, `Procfile`, `render.yaml`)
- Current evaluation artifacts needed for viewing and re-evaluation:
  - `results/evaluated`
  - `results/readiness`
  - `results/fused`
  - `results/lane/images`
  - `results/research_report.pdf`

Not copied by default:
- `results/archive` (large historical payload)
- `lane/` C++ source/build tree
- `pipeline/process_bag_pipeline.py`

If you want historical archive dashboards too, copy `results/archive` into this repo before pushing.
