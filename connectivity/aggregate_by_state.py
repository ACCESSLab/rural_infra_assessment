"""Reusable Ookla speed lookup utilities.

This module supports three common access patterns:
1) Coordinate lookup: get tile speed for a GPS point.
2) State summary: aggregate weighted county speeds for a state.
3) County lookup: return one county's aggregated speed.

Example
-------
>>> from connectivity.aggregate_by_state import OoklaSpeedLookup
>>> lookup = OoklaSpeedLookup(service_type="mobile")
>>> lookup.get_speed_at_coordinates(35.7796, -78.6382)
>>> lookup.get_county_speed(state_fips="37", county_name="Wake")
>>> lookup.get_state_county_stats(state_fips="37", min_tests=50).head()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import geopandas as gp
import numpy as np
import pandas as pd
from shapely.geometry import Point


DEFAULT_SERVICE_TYPE = "mobile"
DEFAULT_MIN_TILE_YEAR = 2019
DEFAULT_MIN_COUNTY_YEAR = 2020
DEFAULT_QUARTER = 4
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
DEFAULT_LOOKUP_ROUND_DECIMALS = 3


def _load_shared_config() -> Dict:
    """Load shared project configuration, returning an empty dict on failure."""
    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(os.getenv("PROJ2_CONFIG", repo_root / "config" / "defaults.json"))
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}


_SHARED_CONFIG = _load_shared_config()
_CONNECTIVITY_CFG = _SHARED_CONFIG.get("connectivity", {}) if isinstance(_SHARED_CONFIG.get("connectivity", {}), dict) else {}
if "cache_dir" in _CONNECTIVITY_CFG:
    _cache_dir = Path(_CONNECTIVITY_CFG["cache_dir"])
    if not _cache_dir.is_absolute():
        _cache_dir = Path(__file__).resolve().parents[1] / _cache_dir
    DEFAULT_CACHE_DIR = _cache_dir
if "service_type" in _CONNECTIVITY_CFG:
    DEFAULT_SERVICE_TYPE = str(_CONNECTIVITY_CFG["service_type"])
if "quarter" in _CONNECTIVITY_CFG:
    DEFAULT_QUARTER = int(_CONNECTIVITY_CFG["quarter"])
if "lookup_round_decimals" in _CONNECTIVITY_CFG:
    DEFAULT_LOOKUP_ROUND_DECIMALS = int(_CONNECTIVITY_CFG["lookup_round_decimals"])


def default_data_year() -> int:
    """Default target year: one year behind current year."""
    year_offset = int(_CONNECTIVITY_CFG.get("year_offset_from_current", 1))
    return datetime.now(timezone.utc).year - year_offset


DEFAULT_YEAR = default_data_year()

STATE_ABBR_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10",
    "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19",
    "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35",
    "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
}
STATE_NAME_TO_FIPS = {
    "alabama": "01", "alaska": "02", "arizona": "04", "arkansas": "05", "california": "06",
    "colorado": "08", "connecticut": "09", "delaware": "10", "district of columbia": "11",
    "florida": "12", "georgia": "13", "hawaii": "15", "idaho": "16", "illinois": "17", "indiana": "18",
    "iowa": "19", "kansas": "20", "kentucky": "21", "louisiana": "22", "maine": "23", "maryland": "24",
    "massachusetts": "25", "michigan": "26", "minnesota": "27", "mississippi": "28", "missouri": "29",
    "montana": "30", "nebraska": "31", "nevada": "32", "new hampshire": "33", "new jersey": "34",
    "new mexico": "35", "new york": "36", "north carolina": "37", "north dakota": "38", "ohio": "39",
    "oklahoma": "40", "oregon": "41", "pennsylvania": "42", "rhode island": "44", "south carolina": "45",
    "south dakota": "46", "tennessee": "47", "texas": "48", "utah": "49", "vermont": "50",
    "virginia": "51", "washington": "53", "west virginia": "54", "wisconsin": "55", "wyoming": "56",
}


def quarter_start(year: int, q: int) -> datetime:
    """Return first day of quarter for `year` and quarter index [1..4]."""
    if q not in (1, 2, 3, 4):
        raise ValueError("Quarter must be one of 1, 2, 3, 4")

    month = [1, 4, 7, 10]
    return datetime(year, month[q - 1], 1)


def get_tile_url(service_type: str, year: int, q: int) -> str:
    """Build Ookla tile URL for service/year/quarter."""
    dt = quarter_start(year, q)
    base_url = "https://ookla-open-data.s3-us-west-2.amazonaws.com/shapefiles/performance"
    return (
        f"{base_url}/type%3D{service_type}/year%3D{dt:%Y}/quarter%3D{q}/"
        f"{dt:%Y-%m-%d}_performance_{service_type}_tiles.zip"
    )


def get_county_url(year: int = 2020) -> str:
    """Build Census county boundary URL for a TIGER year."""
    return f"https://www2.census.gov/geo/tiger/TIGER{year}/COUNTY/tl_{year}_us_county.zip"


def _iter_periods_desc(start_year: int, start_quarter: int, min_year: int = DEFAULT_MIN_TILE_YEAR):
    """Yield candidate year/quarter pairs from newest to oldest."""
    year = int(start_year)
    quarter = int(start_quarter)
    while year >= int(min_year):
        q_start = quarter if year == start_year else 4
        for q in range(q_start, 0, -1):
            yield year, q
        year -= 1


def _download_file_if_needed(url: str, dst: Path, logger=None) -> Path:
    """Download a file once and reuse the cached copy on later calls."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if logger:
            logger(f"Using cached file: {dst}")
        return dst

    if logger:
        logger(f"Downloading {url}")
    tmp_dst = dst.with_suffix(dst.suffix + ".part")
    try:
        urlretrieve(url, tmp_dst.as_posix())
        tmp_dst.replace(dst)
    except Exception:
        if tmp_dst.exists():
            tmp_dst.unlink()
        raise
    return dst


def _normalize_tiles(tiles: gp.GeoDataFrame) -> gp.GeoDataFrame:
    """Ensure tile CRS and expected speed columns are available."""
    out = tiles.copy()

    if out.crs is None:
        out.set_crs(epsg=4326, inplace=True)
    else:
        out = out.to_crs(epsg=4326)

    if "avg_d_kbps" not in out.columns and "avg_d_mbps" in out.columns:
        out["avg_d_kbps"] = out["avg_d_mbps"] * 1000
    if "avg_u_kbps" not in out.columns and "avg_u_mbps" in out.columns:
        out["avg_u_kbps"] = out["avg_u_mbps"] * 1000

    if "tests" not in out.columns:
        out["tests"] = 0

    return out


def _pick_latency_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return best-available (overall, download, upload) latency column names."""
    overall_candidates = ["avg_lat_ms", "avg_latency_ms", "latency_ms", "avg_ping_ms"]
    down_candidates = [
        "avg_lat_d_ms",
        "avg_lat_dl_ms",
        "avg_download_lat_ms",
        "avg_download_latency_ms",
        "avg_d_lat_ms",
    ]
    up_candidates = [
        "avg_lat_u_ms",
        "avg_lat_ul_ms",
        "avg_upload_lat_ms",
        "avg_upload_latency_ms",
        "avg_u_lat_ms",
    ]

    def _first_present(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    overall_col = _first_present(overall_candidates)
    down_col = _first_present(down_candidates)
    up_col = _first_present(up_candidates)
    return overall_col, down_col, up_col


def load_tiles(tile_url: str) -> gp.GeoDataFrame:
    """Read Ookla tiles from URL or local file path."""
    return _normalize_tiles(gp.read_file(tile_url))


def load_counties(state_fips: str, year: int = 2020, source: Optional[str] = None) -> gp.GeoDataFrame:
    """Load county boundaries for a US state FIPS code."""
    counties = gp.read_file(source if source is not None else get_county_url(year=year))
    state_fips = str(state_fips).zfill(2)
    return counties.loc[counties["STATEFP"] == state_fips].to_crs(4326)


def load_all_counties(year: int = 2020, source: Optional[str] = None) -> gp.GeoDataFrame:
    """Load all US counties for a TIGER year."""
    return gp.read_file(source if source is not None else get_county_url(year=year)).to_crs(4326)


def parse_state_fips(state: Optional[str] = None, state_fips: Optional[str] = None) -> Optional[str]:
    """Normalize state input (FIPS, 2-letter code, or full name) to 2-digit FIPS."""
    raw = state_fips if state_fips is not None else state
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.isdigit():
        return s.zfill(2)
    upper = s.upper()
    if upper in STATE_ABBR_TO_FIPS:
        return STATE_ABBR_TO_FIPS[upper]
    lower = s.lower()
    if lower in STATE_NAME_TO_FIPS:
        return STATE_NAME_TO_FIPS[lower]
    return None


def join_tiles_to_counties(
    tiles: gp.GeoDataFrame,
    counties: gp.GeoDataFrame,
    how: str = "inner",
    predicate: str = "intersects",
) -> gp.GeoDataFrame:
    """Spatial-join Ookla tiles to county polygons."""
    return gp.sjoin(_normalize_tiles(tiles), counties.to_crs(4326), how=how, predicate=predicate)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Compute a weighted mean with a safe fallback when weights sum to zero."""
    weights = weights.fillna(0)
    if float(weights.sum()) <= 0:
        return float(values.mean())
    return float(np.average(values, weights=weights))


def compute_county_stats(tiles_in_counties: gp.GeoDataFrame) -> pd.DataFrame:
    """Compute tests-weighted county download/upload speeds in Mbps."""
    df = tiles_in_counties.copy()
    df["avg_d_mbps"] = df["avg_d_kbps"] / 1000.0
    df["avg_u_mbps"] = df["avg_u_kbps"] / 1000.0

    grouped = df.groupby(["STATEFP", "COUNTYFP", "GEOID", "NAMELSAD"], dropna=False)
    overall_lat_col, down_lat_col, up_lat_col = _pick_latency_columns(df)

    stats = grouped.apply(
        lambda x: pd.Series(
            {
                "avg_d_mbps_wt": _weighted_mean(x["avg_d_mbps"], x["tests"]),
                "avg_u_mbps_wt": _weighted_mean(x["avg_u_mbps"], x["tests"]),
                "avg_latency_ms_wt": _weighted_mean(x[overall_lat_col], x["tests"]) if overall_lat_col else np.nan,
                "avg_download_latency_ms_wt": (
                    _weighted_mean(x[down_lat_col], x["tests"])
                    if down_lat_col
                    else (_weighted_mean(x[overall_lat_col], x["tests"]) if overall_lat_col else np.nan)
                ),
                "avg_upload_latency_ms_wt": (
                    _weighted_mean(x[up_lat_col], x["tests"])
                    if up_lat_col
                    else (_weighted_mean(x[overall_lat_col], x["tests"]) if overall_lat_col else np.nan)
                ),
                "tests": int(x["tests"].fillna(0).sum()),
                "tile_count": int(len(x)),
            }
        )
    ).reset_index()

    return stats.sort_values(["avg_d_mbps_wt", "tests"], ascending=[False, False]).reset_index(drop=True)


def get_speed_at_point(lat: float, lon: float, tiles: gp.GeoDataFrame) -> Optional[Dict[str, float]]:
    """Backward-compatible point lookup helper."""
    return OoklaSpeedLookup(tiles=tiles).get_speed_at_coordinates(lat=lat, lon=lon)


def speed_for_location(
    lat: float,
    lon: float,
    service_type: str = DEFAULT_SERVICE_TYPE,
    year: Optional[int] = None,
    quarter: Optional[int] = None,
    state_fips: Optional[str] = None,
    state: Optional[str] = None,
    county_name: Optional[str] = None,
    min_tests: int = 0,
    verbose: bool = False,
) -> Dict:
    """Top-level convenience wrapper for location lookup workflows."""
    lookup = OoklaSpeedLookup(
        service_type=service_type,
        year=year,
        quarter=quarter,
        verbose=verbose,
    )
    return lookup.speed_for_location(
        lat=lat,
        lon=lon,
        state_fips=parse_state_fips(state=state, state_fips=state_fips),
        county_name=county_name,
        min_tests=min_tests,
    )


@dataclass
class OoklaSpeedLookup:
    """Lookup helper for coordinate/state/county speed queries."""

    service_type: str = DEFAULT_SERVICE_TYPE
    year: Optional[int] = None
    quarter: Optional[int] = None
    county_year: Optional[int] = None
    cache_dir: Path = DEFAULT_CACHE_DIR
    min_tile_year: int = DEFAULT_MIN_TILE_YEAR
    min_county_year: int = DEFAULT_MIN_COUNTY_YEAR
    tile_url: Optional[str] = None
    tiles: Optional[gp.GeoDataFrame] = None
    verbose: bool = False
    _counties_cache: Dict[str, gp.GeoDataFrame] = field(default_factory=dict)
    _all_counties: Optional[gp.GeoDataFrame] = None
    lookup_round_decimals: int = DEFAULT_LOOKUP_ROUND_DECIMALS
    max_lookup_cache_size: int = 20000
    _lookup_cache: Dict[Tuple[int, int], Optional[Dict]] = field(default_factory=dict)
    _lookup_cache_order: List[Tuple[int, int]] = field(default_factory=list)
    _tiles_4326: Optional[gp.GeoDataFrame] = None
    _tiles_3857: Optional[gp.GeoDataFrame] = None
    _latency_cols: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = None

    def __post_init__(self) -> None:
        """Normalize constructor inputs and eagerly sanitize any provided tiles."""
        self.year = int(self.year) if self.year is not None else default_data_year()
        self.quarter = int(self.quarter) if self.quarter is not None else DEFAULT_QUARTER
        self.county_year = int(self.county_year) if self.county_year is not None else self.year
        self.cache_dir = Path(self.cache_dir)
        if self.tiles is not None:
            self.tiles = _normalize_tiles(self.tiles)

    def _tile_cache_path(self, year: int, quarter: int) -> Path:
        """Return the cache location for a tile archive."""
        filename = f"{year}_q{quarter}_{self.service_type}_tiles.zip"
        return self.cache_dir / "tiles" / self.service_type / str(year) / filename

    def _county_cache_path(self, year: int) -> Path:
        """Return the cache location for a county boundary archive."""
        return self.cache_dir / "counties" / f"tl_{year}_us_county.zip"

    def _resolve_tiles_gdf(self) -> gp.GeoDataFrame:
        """Resolve and load the best available tile dataset for the requested period."""
        if self.tile_url is not None:
            if self.tile_url.startswith("http://") or self.tile_url.startswith("https://"):
                local_path = self._tile_cache_path(self.year, self.quarter)
                try:
                    local_path = _download_file_if_needed(self.tile_url, local_path, logger=self._log)
                    tiles = load_tiles(local_path.as_posix())
                    self._log(f"Loaded cached tile dataset for {self.year} Q{self.quarter}")
                    return tiles
                except Exception as exc:
                    self._log(f"Failed explicit tile URL download/read: {exc}")
                    raise
            return load_tiles(self.tile_url)

        last_exc = None
        for year, quarter in _iter_periods_desc(self.year, self.quarter, self.min_tile_year):
            url = get_tile_url(self.service_type, year, quarter)
            local_path = self._tile_cache_path(year, quarter)
            try:
                _download_file_if_needed(url, local_path, logger=self._log)
                tiles = load_tiles(local_path.as_posix())
                self.year = year
                self.quarter = quarter
                self.tile_url = url
                self._log(f"Using tile dataset {year} Q{quarter}")
                return tiles
            except Exception as exc:
                last_exc = exc
                self._log(f"Tile dataset unavailable for {year} Q{quarter}; trying older data")

        raise RuntimeError(
            f"Unable to find available tile data from {self.year} Q{self.quarter} "
            f"back to {self.min_tile_year} Q1"
        ) from last_exc

    def _get_tiles(self) -> gp.GeoDataFrame:
        """Load the tile dataset lazily and cache it on the instance."""
        if self.tiles is None:
            self._log(f"Resolving tile dataset from target {self.year} Q{self.quarter}")
            self.tiles = self._resolve_tiles_gdf()
            self._log(f"Loaded {len(self.tiles)} tile rows")
        return self.tiles

    def _get_tiles_4326(self) -> gp.GeoDataFrame:
        """Return the tile dataset reprojected to WGS84 coordinates."""
        if self._tiles_4326 is None:
            self._tiles_4326 = self._get_tiles().to_crs(4326)
        return self._tiles_4326

    def _get_tiles_3857(self) -> gp.GeoDataFrame:
        """Return the tile dataset reprojected to Web Mercator for distance math."""
        if self._tiles_3857 is None:
            self._tiles_3857 = self._get_tiles_4326().to_crs(epsg=3857)
        return self._tiles_3857

    def _get_latency_cols(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Cache and return the best latency column choices for the tile schema."""
        if self._latency_cols is None:
            self._latency_cols = _pick_latency_columns(self._get_tiles_4326())
        return self._latency_cols

    def _cache_key(self, lat: float, lon: float) -> Tuple[int, int]:
        """Quantize a coordinate pair into the lookup-cache key space."""
        scale = 10 ** int(self.lookup_round_decimals)
        return int(round(float(lat) * scale)), int(round(float(lon) * scale))

    def _cache_get(self, key: Tuple[int, int]) -> Tuple[bool, Optional[Dict]]:
        """Return a cached point lookup result when present."""
        if key in self._lookup_cache:
            return True, self._lookup_cache[key]
        return False, None

    def _cache_put(self, key: Tuple[int, int], value: Optional[Dict]) -> None:
        """Insert one point lookup result into the bounded FIFO cache."""
        self._lookup_cache[key] = value
        self._lookup_cache_order.append(key)
        if len(self._lookup_cache_order) > self.max_lookup_cache_size:
            old_key = self._lookup_cache_order.pop(0)
            self._lookup_cache.pop(old_key, None)

    def _get_counties(self, state_fips: str) -> gp.GeoDataFrame:
        """Load and cache county polygons for a single state."""
        state_fips = parse_state_fips(state_fips=state_fips)
        if state_fips is None:
            raise ValueError("Invalid state input. Provide FIPS, state code (e.g. NC), or full state name.")
        if state_fips not in self._counties_cache:
            self._log(f"Loading county boundaries for state FIPS {state_fips}")
            all_counties = self._get_all_counties()
            counties = all_counties.loc[all_counties["STATEFP"] == state_fips].copy()
            if counties.empty:
                raise ValueError(f"No counties found for state FIPS {state_fips}")
            self._counties_cache[state_fips] = counties
            self._log(f"Loaded {len(counties)} county polygons")
        return self._counties_cache[state_fips]

    def _get_all_counties(self) -> gp.GeoDataFrame:
        """Load and cache the nationwide county boundary dataset."""
        if self._all_counties is not None:
            return self._all_counties
        self._log("Loading nationwide county boundary layer")
        counties = None
        last_exc = None
        for year in range(self.county_year, self.min_county_year - 1, -1):
            url = get_county_url(year=year)
            local_path = self._county_cache_path(year)
            try:
                _download_file_if_needed(url, local_path, logger=self._log)
                counties = load_all_counties(year=year, source=local_path.as_posix())
                self._log(f"Using nationwide county boundary year {year}")
                break
            except Exception as exc:
                last_exc = exc
                self._log(f"County boundary year {year} unavailable; trying older year")
        if counties is None:
            raise RuntimeError(
                f"Unable to find county boundary data from {self.county_year} back to {self.min_county_year}"
            ) from last_exc
        self._all_counties = counties
        return self._all_counties

    def _log(self, message: str) -> None:
        """Emit a debug log line when verbose mode is enabled."""
        if self.verbose:
            print(f"[OoklaSpeedLookup] {message}")

    def _row_to_result(self, row: pd.Series, source: str) -> Dict[str, float]:
        """Convert one tile row into the normalized connectivity result schema."""
        overall_lat_col, down_lat_col, up_lat_col = self._get_latency_cols()
        avg_latency_ms = float(row.get(overall_lat_col, np.nan)) if overall_lat_col else np.nan
        avg_download_latency_ms = (
            float(row.get(down_lat_col, np.nan))
            if down_lat_col
            else avg_latency_ms
        )
        avg_upload_latency_ms = (
            float(row.get(up_lat_col, np.nan))
            if up_lat_col
            else avg_latency_ms
        )
        return {
            "avg_d_mbps": float(row.get("avg_d_kbps", np.nan)) / 1000.0,
            "avg_u_mbps": float(row.get("avg_u_kbps", np.nan)) / 1000.0,
            "avg_latency_ms": avg_latency_ms,
            "avg_download_latency_ms": avg_download_latency_ms,
            "avg_upload_latency_ms": avg_upload_latency_ms,
            "tests": int(row.get("tests", 0)) if not pd.isna(row.get("tests", None)) else 0,
            "source": source,
        }

    def _nearest_tile_row(self, pt: Point, tiles: gp.GeoDataFrame) -> Optional[pd.Series]:
        """Find the closest tile row to a point when no direct intersection exists."""
        if hasattr(tiles, "sindex"):
            try:
                nearest = list(tiles.sindex.nearest(pt, return_all=False))
                if len(nearest) >= 2 and len(nearest[1]) > 0:
                    return tiles.iloc[int(nearest[1][0])]
            except Exception:
                pass
        try:
            tiles_projected = self._get_tiles_3857()
            pt_projected = gp.GeoSeries([pt], crs=4326).to_crs(epsg=3857).iloc[0]
            idx = tiles_projected.geometry.distance(pt_projected).idxmin()
            return tiles.iloc[idx]
        except Exception:
            return None

    def get_speed_at_coordinates(self, lat: float, lon: float, nearest_if_missing: bool = True) -> Optional[Dict]:
        """Return tile speed for a latitude/longitude point.

        If no intersecting tile exists and `nearest_if_missing` is True,
        returns the closest tile value.
        """
        self._log(f"Looking up speed for lat={lat}, lon={lon}")
        key = self._cache_key(lat, lon)
        found, cached = self._cache_get(key)
        if found:
            return dict(cached) if isinstance(cached, dict) else None

        tiles = self._get_tiles_4326()
        pt = Point(float(lon), float(lat))

        # Spatial index pre-filter greatly reduces work vs scanning all tiles.
        if hasattr(tiles, "sindex"):
            try:
                candidate_idx = list(tiles.sindex.query(pt, predicate="intersects"))
            except Exception:
                candidate_idx = list(tiles.sindex.query(pt))
            candidates = tiles.iloc[candidate_idx] if candidate_idx else tiles.iloc[0:0]
            intersects = candidates.loc[candidates.geometry.intersects(pt)]
        else:
            intersects = tiles.loc[tiles.geometry.intersects(pt)]
        if not intersects.empty:
            row = intersects.iloc[0]
            source = "intersect"
            self._log("Found intersecting tile")
        elif nearest_if_missing:
            row = self._nearest_tile_row(pt, tiles)
            if row is None:
                self._cache_put(key, None)
                return None
            source = "nearest"
            self._log("No intersecting tile; using nearest tile fallback")
        else:
            self._cache_put(key, None)
            return None

        result = self._row_to_result(row, source)
        self._log(
            f"Result: download={result['avg_d_mbps']:.2f} Mbps, "
            f"upload={result['avg_u_mbps']:.2f} Mbps, "
            f"latency={result['avg_latency_ms']:.2f} ms, "
            f"dl_lat={result['avg_download_latency_ms']:.2f} ms, "
            f"ul_lat={result['avg_upload_latency_ms']:.2f} ms, "
            f"tests={result['tests']} ({source})"
        )
        self._cache_put(key, result)
        return result

    def get_state_county_stats(self, state_fips: str, min_tests: int = 0) -> pd.DataFrame:
        """Return county-level weighted speed stats for one state."""
        state_fips = parse_state_fips(state_fips=state_fips)
        if state_fips is None:
            raise ValueError("Invalid state input. Provide FIPS, state code (e.g. NC), or full state name.")
        self._log(f"Computing county stats for state FIPS {state_fips}")
        counties = self._get_counties(state_fips)
        self._log("Spatial-joining tiles to county polygons")
        joined = join_tiles_to_counties(self._get_tiles(), counties)
        stats = compute_county_stats(joined)
        if min_tests > 0:
            stats = stats.loc[stats["tests"] >= int(min_tests)].copy()
            self._log(f"Applied min_tests filter: {min_tests}; remaining counties={len(stats)}")
        else:
            self._log(f"Computed county stats for {len(stats)} counties")
        return stats.reset_index(drop=True)

    def get_county_speed(
        self,
        state_fips: str,
        county_name: Optional[str] = None,
        county_fips: Optional[str] = None,
        geoid: Optional[str] = None,
        min_tests: int = 0,
    ) -> Optional[Dict]:
        """Return one county's aggregated speed record.

        One of `geoid`, `county_fips`, or `county_name` must be provided.
        `county_name` supports values like "Wake" or "Wake County".
        """
        state_fips = parse_state_fips(state_fips=state_fips)
        if state_fips is None:
            raise ValueError("Invalid state input. Provide FIPS, state code (e.g. NC), or full state name.")
        stats = self.get_state_county_stats(state_fips=state_fips, min_tests=min_tests)

        row_df: pd.DataFrame
        if geoid:
            row_df = stats.loc[stats["GEOID"] == str(geoid)]
        elif county_fips:
            county_fips = str(county_fips).zfill(3)
            row_df = stats.loc[stats["COUNTYFP"] == county_fips]
        elif county_name:
            q = county_name.strip().lower()
            if not q.endswith(" county"):
                q_county = q + " county"
            else:
                q_county = q
            norm_names = stats["NAMELSAD"].astype(str).str.lower().str.strip()
            row_df = stats.loc[(norm_names == q) | (norm_names == q_county)]
        else:
            raise ValueError("Provide at least one of: geoid, county_fips, county_name")

        if row_df.empty:
            self._log("County not found for the provided selector")
            return None

        row = row_df.iloc[0]
        result = {
            "state_fips": str(row["STATEFP"]),
            "county_fips": str(row["COUNTYFP"]),
            "geoid": str(row["GEOID"]),
            "county": str(row["NAMELSAD"]),
            "avg_d_mbps_wt": float(row["avg_d_mbps_wt"]),
            "avg_u_mbps_wt": float(row["avg_u_mbps_wt"]),
            "avg_latency_ms_wt": float(row["avg_latency_ms_wt"]) if not pd.isna(row["avg_latency_ms_wt"]) else np.nan,
            "avg_download_latency_ms_wt": (
                float(row["avg_download_latency_ms_wt"]) if not pd.isna(row["avg_download_latency_ms_wt"]) else np.nan
            ),
            "avg_upload_latency_ms_wt": (
                float(row["avg_upload_latency_ms_wt"]) if not pd.isna(row["avg_upload_latency_ms_wt"]) else np.nan
            ),
            "tests": int(row["tests"]),
            "tile_count": int(row["tile_count"]),
            "service_type": self.service_type,
            "year": int(self.year),
            "quarter": int(self.quarter),
        }
        self._log(
            f"County match: {result['county']} ({result['geoid']}), "
            f"download={result['avg_d_mbps_wt']:.2f} Mbps, "
            f"upload={result['avg_u_mbps_wt']:.2f} Mbps, "
            f"latency={result['avg_latency_ms_wt']:.2f} ms, "
            f"dl_lat={result['avg_download_latency_ms_wt']:.2f} ms, "
            f"ul_lat={result['avg_upload_latency_ms_wt']:.2f} ms"
        )
        return result

    def speed_for_location(
        self,
        lat: float,
        lon: float,
        state_fips: Optional[str] = None,
        state: Optional[str] = None,
        county_name: Optional[str] = None,
        min_tests: int = 0,
    ) -> Dict:
        """Convenience wrapper for GPS speed lookup + optional county/state context."""
        self._log("Running combined speed_for_location workflow")
        point_speed = self.get_speed_at_coordinates(lat=lat, lon=lon)
        location = self.resolve_location(lat=lat, lon=lon)
        result: Dict = {"point_speed": point_speed, "location": location}

        resolved_state_fips = parse_state_fips(state=state, state_fips=state_fips)
        if resolved_state_fips is None and location is not None:
            resolved_state_fips = str(location["state_fips"])
        if county_name is None and location is not None:
            county_name = str(location["county"])

        if resolved_state_fips is not None:
            result["state_county_stats"] = self.get_state_county_stats(state_fips=resolved_state_fips, min_tests=min_tests)
            if county_name:
                result["county_speed"] = self.get_county_speed(
                    state_fips=resolved_state_fips,
                    county_name=county_name,
                    min_tests=min_tests,
                )
        return result

    def resolve_location(self, lat: float, lon: float) -> Optional[Dict]:
        """Resolve county/state FIPS for a point using the nationwide county layer."""
        pt = Point(float(lon), float(lat))
        counties = self._get_all_counties()
        contains = counties.loc[counties.geometry.contains(pt)]
        source = "intersect"
        if contains.empty:
            try:
                idx = counties.geometry.distance(pt).idxmin()
            except Exception:
                return None
            contains = counties.loc[[idx]]
            source = "nearest"
        row = contains.iloc[0]
        return {
            "state_fips": str(row.get("STATEFP", "")).zfill(2),
            "county_fips": str(row.get("COUNTYFP", "")).zfill(3),
            "geoid": str(row.get("GEOID", "")),
            "county": str(row.get("NAMELSAD", row.get("NAME", ""))),
            "source": source,
        }

    def plot_state_tiles(
        self,
        state_fips: Optional[str] = None,
        state: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        county_name: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        show: bool = True,
    ):
        """Plot state county boundaries + tile download speeds and highlight selected/current county."""
        import matplotlib.pyplot as plt

        resolved_state_fips = parse_state_fips(state=state, state_fips=state_fips)
        if resolved_state_fips is None and lat is not None and lon is not None:
            loc = self.resolve_location(lat=lat, lon=lon)
            if loc is not None:
                resolved_state_fips = str(loc["state_fips"])
                if county_name is None:
                    county_name = str(loc["county"])
                self._log(f"Auto-resolved state/county from point: {resolved_state_fips} / {county_name}")
        if resolved_state_fips is None:
            raise ValueError("Could not determine state. Provide state_fips, state, or lat/lon.")

        self._log(f"Preparing tile plot for state FIPS {resolved_state_fips}")
        counties = self._get_counties(resolved_state_fips).copy()
        joined = join_tiles_to_counties(self._get_tiles(), counties)
        joined = joined.copy()
        joined["avg_d_mbps"] = joined["avg_d_kbps"] / 1000.0

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        counties.boundary.plot(ax=ax, color="#666666", linewidth=0.8, alpha=0.8)
        joined.plot(
            ax=ax,
            column="avg_d_mbps",
            cmap="YlGnBu",
            alpha=0.6,
            linewidth=0.0,
            legend=True,
            legend_kwds={"label": "Avg Download Speed (Mbps)"},
        )

        highlighted = None
        if lat is not None and lon is not None:
            pt = Point(float(lon), float(lat))
            contains = counties.loc[counties.geometry.contains(pt)]
            if contains.empty:
                idx = counties.geometry.distance(pt).idxmin()
                contains = counties.loc[[idx]]
            highlighted = contains
            highlighted.boundary.plot(ax=ax, color="red", linewidth=2.2)
            gp.GeoSeries([pt], crs=4326).plot(ax=ax, color="red", markersize=35)
            self._log("Highlighted county based on provided coordinates")
        elif county_name:
            q = county_name.strip().lower()
            if not q.endswith(" county"):
                q = q + " county"
            mask = counties["NAMELSAD"].astype(str).str.lower().str.strip() == q
            contains = counties.loc[mask]
            if not contains.empty:
                highlighted = contains
                highlighted.boundary.plot(ax=ax, color="red", linewidth=2.2)
                self._log(f"Highlighted county by name: {county_name}")

        subtitle = f"{self.service_type.title()} | {self.year} Q{self.quarter}"
        if highlighted is not None and not highlighted.empty:
            county_label = str(highlighted.iloc[0]["NAMELSAD"])
            ax.set_title(f"Ookla Tile Download Speeds by County ({county_label})\n{subtitle}")
        else:
            ax.set_title(f"Ookla Tile Download Speeds by County\n{subtitle}")
        ax.set_axis_off()

        if show:
            plt.show()
        return fig, ax


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query Ookla tile speeds by coordinate, state, or county")
    parser.add_argument("--service-type", default=DEFAULT_SERVICE_TYPE, choices=["mobile", "fixed"])
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--quarter", type=int, default=DEFAULT_QUARTER)
    parser.add_argument("--state-fips", default=None, help="2-digit state FIPS (e.g. 37)")
    parser.add_argument("--state", default=None, help="State as 2-letter code/name/FIPS (e.g. NC, North Carolina, 37)")
    parser.add_argument("--county-name", default=None, help="County name (e.g. 'Wake' or 'Wake County')")
    parser.add_argument("--county-fips", default=None, help="3-digit county FIPS within state")
    parser.add_argument("--geoid", default=None, help="5-digit county GEOID")
    parser.add_argument("--min-tests", type=int, default=0)
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--verbose", action="store_true", help="Print progress logs")
    parser.add_argument("--plot", action="store_true", help="Render state tile plot (auto-resolves state from lat/lon if needed)")
    args = parser.parse_args()

    lookup = OoklaSpeedLookup(
        service_type=args.service_type,
        year=args.year,
        quarter=args.quarter,
        verbose=args.verbose,
    )

    if args.lat is not None and args.lon is not None:
        print(lookup.get_speed_at_coordinates(args.lat, args.lon))
        print({"location": lookup.resolve_location(args.lat, args.lon)})

    resolved_state_fips = parse_state_fips(state=args.state, state_fips=args.state_fips)

    if resolved_state_fips and not (args.county_name or args.county_fips or args.geoid):
        print(lookup.get_state_county_stats(resolved_state_fips, min_tests=args.min_tests).head())

    if resolved_state_fips and (args.county_name or args.county_fips or args.geoid):
        print(
            lookup.get_county_speed(
                state_fips=resolved_state_fips,
                county_name=args.county_name,
                county_fips=args.county_fips,
                geoid=args.geoid,
                min_tests=args.min_tests,
            )
        )

    if args.plot:
        lookup.plot_state_tiles(
            state_fips=resolved_state_fips,
            state=args.state,
            lat=args.lat,
            lon=args.lon,
            county_name=args.county_name,
            show=True,
        )
