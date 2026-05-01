# Render Deployment Notes

This app can run on Render as a Python web service.

## What is supported

- Flask is supported on Render web services.
- Use Gunicorn as the production server.
- Your service must bind to `0.0.0.0` and the `PORT` Render provides.

## Files added for deployment

- `requirements.txt` for Python dependencies
- `Procfile` for process startup
- `render.yaml` for Render Blueprint-based setup

## Security defaults in this deployment profile

The UI server now supports deployment flags:

- `PUBLIC_MODE=true`
- `ENABLE_FULL_PIPELINE=false` (disables rosbag processing endpoint)
- `ENABLE_REEVALUATION=false` (disables re-evaluation endpoint)
- `ENABLE_REPORT_REGEN=false` (disables report regeneration endpoint)
- `ENABLE_FS_BROWSER=false` (disables server directory browsing endpoint)

Public-facing UX hardening now included:

- Run-pipeline, re-evaluation, report-regeneration, and filesystem-browse controls are disabled for public viewing.
- Server-side checks still block mutating endpoints when deployment flags are disabled.

Mutating endpoints disabled in the default Render profile:

- `POST /api/run`
- `POST /api/reweight`
- `POST /api/report`
- `POST /api/stop`
- `GET /api/fs/list`

Artifact serving is restricted to `results/**` paths only.

## Suggested Render setup flow

1. Push this branch to your remote repository.
2. Create a new Render Web Service from that repo.
3. If using Blueprint, let Render detect `render.yaml`.
4. Set `MAPBOX_TOKEN` in the Render dashboard if Mapbox basemap support is used.
5. Deploy and validate:
   - Dashboard loads
   - Archived/current results load
   - Public controls remain view-only

## Notes

- The Render deployment is intended for viewing precomputed artifacts only. Keep raw-pipeline execution, re-evaluation, report regeneration, and filesystem browsing disabled for the public app.
