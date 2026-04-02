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
- `ENABLE_REEVALUATION=true` (keeps re-evaluation endpoint enabled)
- `ENABLE_REPORT_REGEN=false` (skip report generation on re-evaluation)
- `ENABLE_FS_BROWSER=false` (disables server directory browsing endpoint)
- `APP_API_TOKEN` optional bearer token to protect mutating endpoints

Mutating endpoints protected by optional token:

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
4. Set secret env vars in Render dashboard:
   - `APP_API_TOKEN` (recommended)
   - `MAPBOX_TOKEN` (if Mapbox basemap is used)
5. Deploy and validate:
   - Dashboard loads
   - Archived/current results load
   - Re-evaluation works from the UI

## Notes

- If you want full rosbag pipeline execution in this deployment, set `ENABLE_FULL_PIPELINE=true` and ensure all pipeline/runtime dependencies and data access paths are available in the Render environment.
- Report generation can be re-enabled by setting `ENABLE_REPORT_REGEN=true`.
