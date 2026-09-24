"""GPS tracks: distance, speed and climbing from logged positions."""

from __future__ import annotations

from typing import Annotated, Any, NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely

from noodlelab import Param, Quantity, node

__all__ = ["track_statistics"]

Km = Quantity["km"]
M = Quantity["m"]
Hours = Quantity["h"]
KmH = Quantity["km/h"]


class Track(NamedTuple):
    points: pd.DataFrame
    line: gpd.GeoDataFrame
    distance: Km
    ascent: M
    descent: M
    moving_time: Hours
    total_time: Hours
    moving_speed: KmH
    summary: dict[str, Any]


def _col(table: pd.DataFrame, name: str) -> pd.Series:
    if name not in table.columns:
        raise KeyError(f"No column '{name}'; the table has {', '.join(map(str, table.columns))}")
    return table[name]


@node(category="Geo/Track", title="Track Statistics", sample=False)
def track_statistics(
    table: pd.DataFrame,
    time: Annotated[str, Param(options_from="table.columns")] = "time",
    lat: Annotated[str, Param(options_from="table.columns")] = "lat",
    lon: Annotated[str, Param(options_from="table.columns")] = "lon",
    elevation: Annotated[
        str,
        Param(
            options_from="table.columns",
            empty="none",
            description="Metres; none: no climbing figures",
        ),
    ] = "elevation",
    stopped_below: Annotated[
        float, Param(min=0.0, description="km/h: slower counts as stopped")
    ] = 1.0,
    smoothing: Annotated[
        int, Param(min=1, max=101, description="Points in the elevation smoothing window")
    ] = 5,
) -> Track:
    """Analyse a GPS log (one row per fix, in time order): the geodesic
    distance on the WGS84 ellipsoid, speed, grade, total ascent and descent
    (from lightly smoothed elevation, so GPS noise does not add up), and
    moving time.

    ``points``: the log with ``distance_km`` (cumulative), ``speed_kmh``,
    ``grade_pct`` and ``elapsed_h`` added. ``line``: the route as one line
    feature, for maps and GIS.
    """
    t = pd.to_datetime(_col(table, time))
    la = pd.to_numeric(_col(table, lat), errors="coerce").to_numpy(np.float64)
    lo = pd.to_numeric(_col(table, lon), errors="coerce").to_numpy(np.float64)
    if len(table) < 2:
        raise ValueError("A track needs at least two points")
    geod = pyproj.Geod(ellps="WGS84")
    _, _, step = geod.inv(lo[:-1], la[:-1], lo[1:], la[1:])
    step = np.concatenate([[0.0], np.nan_to_num(step)])
    seconds = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    dt = np.diff(seconds, prepend=seconds[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt * 3.6, 0.0)
    moving = speed >= stopped_below
    out = table.copy()
    out["distance_km"] = np.cumsum(step) / 1000
    out["speed_kmh"] = speed
    out["elapsed_h"] = seconds / 3600
    ascent = descent = 0.0
    has_elevation = bool(elevation.strip())
    if has_elevation:
        z = (
            pd.to_numeric(_col(table, elevation), errors="coerce")
            .interpolate()
            .to_numpy(np.float64)
        )
        zs = pd.Series(z).rolling(smoothing, center=True, min_periods=1).mean().to_numpy()
        dz = np.diff(zs, prepend=zs[0])
        ascent = float(dz[dz > 0].sum())
        descent = float(-dz[dz < 0].sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            out["grade_pct"] = np.where(step > 0.5, dz / step * 100, np.nan)
    distance_km = float(step.sum()) / 1000
    moving_h = float(dt[moving].sum()) / 3600
    total_h = float(seconds[-1]) / 3600
    line = gpd.GeoDataFrame(
        {"distance_km": [distance_km], "ascent_m": [ascent], "moving_h": [moving_h]},
        geometry=[shapely.LineString(np.column_stack([lo, la]))],
        crs="EPSG:4326",
    )
    avg = distance_km / moving_h if moving_h else float("nan")
    summary: dict[str, Any] = {
        "Start": t.iloc[0].strftime("%Y-%m-%d %H:%M"),
        "Distance": Quantity(distance_km, "km"),
        "Total time": _hm(total_h),
        "Moving time": _hm(moving_h),
        "Average moving speed": Quantity(avg, "km/h"),
        "Maximum speed": Quantity(float(np.nanmax(speed)), "km/h"),
    }
    if has_elevation:
        summary |= {
            "Total ascent": Quantity(ascent, "m"),
            "Total descent": Quantity(descent, "m"),
            "Highest point": Quantity(float(np.nanmax(z)), "m"),
            "Lowest point": Quantity(float(np.nanmin(z)), "m"),
        }
    summary["GPS fixes"] = len(table)
    return Track(
        out,
        line,
        Quantity(distance_km, "km"),
        Quantity(ascent, "m"),
        Quantity(descent, "m"),
        Quantity(moving_h, "h"),
        Quantity(total_h, "h"),
        Quantity(avg, "km/h"),
        summary,
    )


def _hm(hours: float) -> str:
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 60:
        h, m = h + 1, 0
    return f"{h} h {m:02d} min"


@track_statistics.check
def _check_track(time: str = "", lat: str = "", lon: str = ""):
    if not (time.strip() and lat.strip() and lon.strip()):
        return "Choose the time, latitude and longitude columns"
