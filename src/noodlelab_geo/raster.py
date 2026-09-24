"""Rasters: grids read from ESRI ASCII files, terrain analysis, indices,
classification, interpolation from points, and statistics per zone."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from numpy.typing import NDArray

from noodlelab import FileRef, Param, Probe, RunContext, node, warning
from noodlelab.core.meta import numeric_columns

from .types import X_NAMES, Y_NAMES, Raster, crs_label, crs_string, metric, same_crs

__all__ = [
    "classify",
    "reclassify",
    "hillshade",
    "idw_interpolation",
    "mask_raster",
    "normalized_difference",
    "point_density",
    "raster_math",
    "raster_statistics",
    "raster_values",
    "read_ascii_grid",
    "sample_raster",
    "save_ascii_grid",
    "slope_aspect",
    "zonal_statistics",
]

AsciiGrid = Annotated[FileRef, Param(accept=(".asc",))]

# --- ESRI ASCII grids --------------------------------------------------------------------------

_HEADER_KEYS = (
    "ncols",
    "nrows",
    "xllcorner",
    "yllcorner",
    "xllcenter",
    "yllcenter",
    "cellsize",
    "nodata_value",
)


def _parse_header(lines: list[str]) -> dict[str, float]:
    header: dict[str, float] = {}
    for line in lines:
        parts = line.split()
        if len(parts) != 2 or parts[0].lower() not in _HEADER_KEYS:
            break
        header[parts[0].lower()] = float(parts[1])
    missing = {"ncols", "nrows", "cellsize"} - set(header)
    if missing or not ({"xllcorner", "xllcenter"} & set(header)):
        raise ValueError(
            "Not an ESRI ASCII grid: the header needs ncols, nrows, xllcorner, yllcorner "
            "and cellsize"
        )
    return header


def _read_header(path: FileRef) -> dict[str, float]:
    with path.open("r") as f:
        return _parse_header([f.readline() for _ in range(6)])


def _prj_crs(path: FileRef) -> str | None:
    prj = path.with_suffix(".prj")
    if not prj.exists():
        return None
    return crs_string(pyproj.CRS.from_wkt(prj.read_text()))


@node(category="Geo/Raster", title="Read ASCII Grid", converter=True)
def read_ascii_grid(
    path: AsciiGrid,
    crs: Annotated[str, Param(description="EPSG code; empty: from the .prj file next to it")] = "",
    name: str = "",
) -> Raster:
    """Read an ESRI ASCII grid (.asc), the plain-text raster format most GIS
    programs export. The CRS comes from a .prj file with the same name, or is
    given here. No-data cells become NaN."""
    text = path.read_text()
    lines = text.splitlines()
    header = _parse_header(lines[:6])
    n_header = len(header)
    rows, cols, cell = int(header["nrows"]), int(header["ncols"]), header["cellsize"]
    data = np.loadtxt(io.StringIO("\n".join(lines[n_header:])), dtype=np.float64, ndmin=2)
    if data.shape != (rows, cols):
        raise ValueError(
            f"The header says {rows}×{cols} cells, the file holds {data.shape[0]}×{data.shape[1]}"
        )
    if "nodata_value" in header:
        data[data == header["nodata_value"]] = np.nan
    x0 = header.get("xllcorner", header.get("xllcenter", 0.0) - cell / 2)
    y_south = header.get("yllcorner", header.get("yllcenter", 0.0) - cell / 2)
    ref = crs_string(pyproj.CRS.from_user_input(crs)) if crs.strip() else _prj_crs(path)
    return Raster(
        data,
        float(x0),
        float(y_south + rows * cell),
        float(cell),
        ref,
        name or Path(path.name).stem,
    )


@read_ascii_grid.probe
def _probe_ascii(path: FileRef, crs: str = ""):
    h = _read_header(path)
    ref = crs or _prj_crs(path)
    return Probe(
        summary=f"grid {int(h['nrows'])}×{int(h['ncols'])} @ {h['cellsize']:g} · {crs_label(ref)}",
        meta={
            "shape": [int(h["nrows"]), int(h["ncols"])],
            "cell": h["cellsize"],
            "crs": crs_label(ref),
        },
    )


@read_ascii_grid.check
def _check_read_ascii(crs: str = ""):
    if crs.strip():
        try:
            pyproj.CRS.from_user_input(crs)
        except pyproj.exceptions.CRSError:
            return f"Unknown CRS '{crs}'"
    return None


@node(category="Output", title="Save ASCII Grid")
def save_ascii_grid(
    raster: Raster,
    ctx: RunContext,
    filename: str = "grid.asc",
    decimals: Annotated[int, Param(min=0, max=10)] = 3,
) -> Path:
    """Write a raster as an ESRI ASCII grid into this run's output folder, with
    a .prj file for its CRS."""
    if not filename.lower().endswith(".asc"):
        filename += ".asc"
    out = ctx.path(filename)
    nodata = -9999.0
    with out.open("w", encoding="utf-8") as f:
        f.write(f"ncols {raster.cols}\nnrows {raster.rows}\n")
        f.write(f"xllcorner {raster.x0!r}\nyllcorner {raster.bounds[1]!r}\n")
        f.write(f"cellsize {raster.cell!r}\nNODATA_value {nodata:g}\n")
        np.savetxt(f, np.where(np.isfinite(raster.data), raster.data, nodata), fmt=f"%.{decimals}f")
    if raster.crs:
        out.with_suffix(".prj").write_text(
            pyproj.CRS.from_user_input(raster.crs).to_wkt("WKT1_ESRI")
        )
    ctx.log(f"Wrote {out.name}")
    return out


@node(category="Geo/Raster", title="Raster Values", converter=True)
def raster_values(raster: Raster) -> NDArray[np.float64]:
    """The raster's valid (non-missing) values as a flat array, for histograms
    and statistics. Inserted when a raster is linked into an array input."""
    return raster.valid()


# --- arithmetic --------------------------------------------------------------------------------


def _aligned(a: Raster, b: Raster) -> None:
    if not a.same_grid(b):
        raise ValueError(
            f"The rasters are on different grids ({a.rows}×{a.cols} @ {a.cell:g}, "
            f"{crs_label(a.crs)} and {b.rows}×{b.cols} @ {b.cell:g}, {crs_label(b.crs)})"
        )


@node(category="Geo/Raster", title="Raster Math")
def raster_math(
    a: Raster,
    b: Raster | None = None,
    value: Annotated[float, Param(description="Used when B is not linked")] = 1.0,
    operation: Literal[
        "add",
        "subtract",
        "multiply",
        "divide",
        "power",
        "minimum",
        "maximum",
        "greater than",
        "less than",
        "equal",
    ] = "multiply",
    name: str = "",
) -> Raster:
    """Cell-by-cell arithmetic between two rasters on the same grid, or a
    raster and a number. Comparisons give 1 (true) and 0 (false)."""
    other: Any = value
    if b is not None:
        _aligned(a, b)
        other = b.data
    ops = {
        "add": np.add,
        "subtract": np.subtract,
        "multiply": np.multiply,
        "divide": np.divide,
        "power": np.power,
        "minimum": np.fmin,
        "maximum": np.fmax,
        "greater than": np.greater,
        "less than": np.less,
        "equal": np.equal,
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ops[operation](a.data, other).astype(np.float64)
    if operation in ("greater than", "less than", "equal"):
        out[~np.isfinite(a.data)] = np.nan
    return a.like(out, name or f"{a.name} {operation}")


@node(category="Geo/Raster", title="Normalized Difference")
def normalized_difference(a: Raster, b: Raster, name: str = "NDVI") -> Raster:
    """(A − B) / (A + B), between −1 and 1: with A = near infrared and B = red,
    the vegetation index NDVI; with green and NIR, the water index NDWI."""
    _aligned(a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        nd = (a.data - b.data) / (a.data + b.data)
    nd[~np.isfinite(nd)] = np.nan
    return a.like(np.clip(nd, -1, 1), name)


# --- terrain -----------------------------------------------------------------------------------


class Terrain(NamedTuple):
    slope: Raster
    aspect: Raster


def _gradients(dem: Raster, z_factor: float) -> tuple[np.ndarray, np.ndarray]:
    z = dem.data * z_factor
    # rows run north to south, so d/dy has the opposite sign of d/drow
    dz_drow, dz_dcol = np.gradient(z, dem.cell)
    return dz_dcol, -dz_drow


@node(category="Geo/Raster", title="Slope & Aspect")
def slope_aspect(
    dem: Raster,
    z_factor: Annotated[float, Param(description="Elevation units per horizontal unit")] = 1.0,
) -> Terrain:
    """Terrain slope in degrees (0 = flat) and aspect, the compass direction
    the slope faces (0 = north, 90 = east), from an elevation model in a
    projected CRS."""
    dzdx, dzdy = _gradients(dem, z_factor)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = (np.degrees(np.arctan2(-dzdx, -dzdy)) + 360) % 360
    aspect[slope == 0] = np.nan
    return Terrain(dem.like(slope, "slope (°)"), dem.like(aspect, "aspect (°)"))


@slope_aspect.check
def _check_slope(dem: Raster | None = None):
    if dem is not None and dem.crs and pyproj.CRS.from_user_input(dem.crs).is_geographic:
        return warning("The DEM is in degrees: slopes need a projected CRS (metres)")
    return None


@node(category="Geo/Raster")
def hillshade(
    dem: Raster,
    azimuth: Annotated[
        float, Param(min=0, max=360, description="Sun direction, degrees from north")
    ] = 315.0,
    altitude: Annotated[float, Param(min=0, max=90, description="Sun height, degrees")] = 45.0,
    z_factor: float = 1.0,
) -> Raster:
    """Shaded relief, 0 (dark) to 1 (lit), as seen with the sun at the given
    direction and height: the classic map backdrop for terrain."""
    dzdx, dzdy = _gradients(dem, z_factor)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdx, -dzdy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return dem.like(np.clip(shade, 0, 1), "hillshade")


# --- classification ----------------------------------------------------------------------------


class Classes(NamedTuple):
    classes: Raster
    areas: pd.DataFrame


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.replace(";", ",").split(",") if v.strip()]


@node(category="Geo/Raster")
def classify(
    raster: Raster,
    breaks: Annotated[
        str, Param(description="Class boundaries, ascending: 0.1, 0.3, 0.6")
    ] = "0.1, 0.3, 0.6",
    labels: Annotated[
        str, Param(description="One more label than breaks")
    ] = "water, bare, grassland, forest",
    area_unit: Literal["km²", "ha", "m²"] = "km²",
) -> Classes:
    """Group values into classes (e.g. NDVI into land cover), and the area of
    each class. A value equal to a break goes into the upper class."""
    edges = _floats(breaks)
    names = [s.strip() for s in labels.split(",") if s.strip()]
    if len(names) != len(edges) + 1:
        raise ValueError(
            f"{len(edges)} breaks make {len(edges) + 1} classes; {len(names)} labels given"
        )
    codes = np.digitize(raster.data, edges).astype(np.float64)
    codes[~np.isfinite(raster.data)] = np.nan
    div = {"km²": 1e6, "ha": 1e4, "m²": 1.0}[area_unit]
    counts = [int(np.sum(codes == i)) for i in range(len(names))]
    total = sum(counts) or 1
    bounds = [-np.inf, *edges, np.inf]
    suffix = {"km²": "km2", "ha": "ha", "m²": "m2"}[area_unit]
    areas = pd.DataFrame(
        {
            "class": names,
            "from": bounds[:-1],
            "to": bounds[1:],
            "cells": counts,
            f"area_{suffix}": [c * raster.cell_area / div for c in counts],
            "percent": [100 * c / total for c in counts],
        }
    )
    return Classes(raster.like(codes, raster.name + " classes", tuple(names)), areas)


@classify.check
def _check_classify(breaks: str = "", labels: str = ""):
    try:
        edges = _floats(breaks)
    except ValueError:
        return "Breaks must be numbers separated by commas"
    if edges != sorted(edges):
        return "Breaks must be in ascending order"
    n = len([s for s in labels.split(",") if s.strip()])
    if n != len(edges) + 1:
        return f"{len(edges)} breaks make {len(edges) + 1} classes, but there are {n} labels"
    return None


@node(category="Geo/Raster")
def reclassify(
    classes: Raster,
    values: Annotated[
        str, Param(description="One number per class, in class order: 1.0, 0.6, 0.35, 0.2")
    ] = "",
    name: str = "",
) -> Raster:
    """Give each class of a classified raster a number, such as a runoff
    coefficient or a roughness per land-cover class."""
    table = np.array(_floats(values), dtype=np.float64)
    if classes.categories and len(table) != len(classes.categories):
        raise ValueError(
            f"{len(classes.categories)} classes ({', '.join(classes.categories)}), "
            f"but {len(table)} values"
        )
    codes = classes.data
    ok = np.isfinite(codes) & (codes >= 0) & (codes < len(table))
    out = np.full(codes.shape, np.nan)
    out[ok] = table[codes[ok].astype(int)]
    return classes.like(out, name or "reclassified")


@reclassify.check
def _check_reclassify(values: str = ""):
    try:
        numbers = _floats(values)
    except ValueError:
        return "Values must be numbers separated by commas"
    if not numbers:
        return "Enter one value per class"
    return None


class RasterStats(NamedTuple):
    mean: float
    std: float
    minimum: float
    maximum: float
    valid_cells: int
    area_km2: float
    summary: dict[str, Any]


@node(category="Geo/Raster", title="Raster Statistics")
def raster_statistics(raster: Raster) -> RasterStats:
    """Summary statistics of the valid cells, and the area they cover (km²,
    for rasters in metres)."""
    v = raster.valid()
    area = v.size * raster.cell_area / 1e6
    stats = RasterStats(
        float(np.mean(v)) if v.size else float("nan"),
        float(np.std(v, ddof=1)) if v.size > 1 else float("nan"),
        float(np.min(v)) if v.size else float("nan"),
        float(np.max(v)) if v.size else float("nan"),
        int(v.size),
        area,
        {},
    )
    summary = {
        "Grid": f"{raster.rows} × {raster.cols} cells of {raster.cell:g}",
        "CRS": crs_label(raster.crs),
        "Valid cells": stats.valid_cells,
        "Area (km²)": area,
        "Mean": stats.mean,
        "Standard deviation": stats.std,
        "Minimum": stats.minimum,
        "Maximum": stats.maximum,
    }
    return stats._replace(summary=summary)


# --- vector and raster ----------------------------------------------------------------------


def _to_raster_crs(data: gpd.GeoDataFrame, raster: Raster) -> gpd.GeoDataFrame:
    if raster.crs and data.crs is not None and not same_crs(data.crs, raster.crs):
        return data.to_crs(raster.crs)
    return data


def _polygon_mask(raster: Raster, geometry: Any) -> np.ndarray:
    """True for cells whose centre lies in the geometry."""
    xs, ys = raster.centers()
    w, s, e, n = geometry.bounds
    inside = np.zeros(raster.data.shape, dtype=bool)
    box = (xs >= w) & (xs <= e) & (ys >= s) & (ys <= n)
    if box.any():
        inside[box] = shapely.contains_xy(geometry, xs[box], ys[box])
    return inside


@node(category="Geo/Raster", title="Sample Raster")
def sample_raster(
    points: gpd.GeoDataFrame,
    raster: Raster,
    name: str = "",
) -> gpd.GeoDataFrame:
    """Add the raster value under each point as a column (the cell the point
    falls in; NaN outside the grid)."""
    pts = _to_raster_crs(points, raster)
    col = ((pts.geometry.x.to_numpy() - raster.x0) / raster.cell).astype(int)
    row = ((raster.y0 - pts.geometry.y.to_numpy()) / raster.cell).astype(int)
    ok = (row >= 0) & (row < raster.rows) & (col >= 0) & (col < raster.cols)
    values = np.full(len(pts), np.nan)
    values[ok] = raster.data[row[ok], col[ok]]
    out = points.copy()
    out[name or raster.name or "value"] = values
    return out


@node(category="Geo/Raster", title="Mask Raster")
def mask_raster(raster: Raster, polygons: gpd.GeoDataFrame, invert: bool = False) -> Raster:
    """Keep the cells inside the polygons (or outside, inverted); the rest
    become missing."""
    geom = _to_raster_crs(polygons, raster).union_all()
    inside = _polygon_mask(raster, geom)
    keep = ~inside if invert else inside
    return raster.like(np.where(keep, raster.data, np.nan), categories=raster.categories)


def _grid_for(
    points: gpd.GeoDataFrame, cell: float, padding: float, template: Raster | None
) -> Raster:
    if template is not None:
        return template.like(np.zeros_like(template.data))
    w, s, e, n = points.total_bounds
    w, s, e, n = w - padding, s - padding, e + padding, n + padding
    cols = max(1, int(np.ceil((e - w) / cell)))
    rows = max(1, int(np.ceil((n - s) / cell)))
    if rows * cols > 25_000_000:
        raise ValueError(f"A {rows}×{cols} grid is too large: use a bigger cell size")
    return Raster(
        np.zeros((rows, cols)),
        float(w),
        float(s + rows * cell),
        float(cell),
        crs_string(points.crs),
    )


@node(category="Geo/Raster", title="IDW Interpolation", converter=True, cost=3.0)
def idw_interpolation(
    points: gpd.GeoDataFrame,
    value: Annotated[str, Param(options_from="points.numeric")] = "",
    template: Raster | None = None,
    cell: Annotated[
        float, Param(min=0.0, description="Cell size in metres, without a template")
    ] = 100.0,
    power: Annotated[float, Param(min=0.5, max=6.0, step=0.5)] = 2.0,
    neighbours: Annotated[int, Param(min=1, max=64)] = 12,
    padding: Annotated[float, Param(min=0.0, description="Metres added around the points")] = 0.0,
    name: str = "",
) -> Raster:
    """Interpolate point measurements onto a grid by inverse distance
    weighting: each cell is the average of its nearest points, weighted by
    1/distanceᵖ. Uses the template's grid if one is linked (to line up with
    other rasters), otherwise the points' extent. Points in longitude and
    latitude are projected to UTM first."""
    from scipy.spatial import cKDTree

    if value not in points.columns:
        raise KeyError(f"No column '{value}' in the points")
    pts = metric(points) if template is None else _to_raster_crs(points, template)
    v = pd.to_numeric(pts[value], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(v) & ~pts.geometry.is_empty.to_numpy()
    xy = np.column_stack([pts.geometry.x.to_numpy()[ok], pts.geometry.y.to_numpy()[ok]])
    v = v[ok]
    if len(v) == 0:
        if pts[value].notna().any():
            raise TypeError(
                f"Column '{value}' is not numeric; numeric columns: "
                + (", ".join(numeric_columns(points)) or "none")
            )
        raise ValueError("No points with a value")
    grid = _grid_for(pts, cell, padding, template)
    xs, ys = grid.centers()
    k = min(neighbours, len(v))
    dist, idx = cKDTree(xy).query(np.column_stack([xs.ravel(), ys.ravel()]), k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / np.power(dist, power)
        est = np.sum(w * v[idx], axis=1) / np.sum(w, axis=1)
    exact = dist[:, 0] == 0  # a cell centre on a point takes its value
    est[exact] = v[idx[exact, 0]]
    return grid.like(est.reshape(grid.data.shape), name or value)


@idw_interpolation.accepts
def _idw_fits(points: gpd.GeoDataFrame) -> dict[str, str] | None:
    """Points with a numeric column to interpolate: the first that is not a
    coordinate or an id."""
    coords = {*X_NAMES, *Y_NAMES}
    columns = [c for c in numeric_columns(points) if c.strip().lower() not in coords]
    if not columns:
        return None
    ids = [c for c in columns if c.strip().lower() == "id" or c.strip().lower().endswith("_id")]
    return {"value": next((c for c in columns if c not in ids), columns[0])}


@idw_interpolation.check
def _check_idw(value: str = ""):
    if not value.strip():
        return "Choose the column to interpolate"


@node(category="Geo/Raster", title="Point Density", converter=True, cost=4.0)
def point_density(
    points: gpd.GeoDataFrame,
    template: Raster | None = None,
    cell: Annotated[
        float, Param(min=0.0, description="Cell size in metres, without a template")
    ] = 1000.0,
    bandwidth: Annotated[float, Param(min=0.0, description="Smoothing radius in metres")] = 5000.0,
    padding: Annotated[
        float, Param(min=0.0, description="Metres added around the points")
    ] = 10000.0,
    weight: Annotated[
        str, Param(options_from="points.numeric", empty="none", description="Weight column")
    ] = "",
) -> Raster:
    """Kernel density: how many points (or how much weight) per km², smoothed
    with a Gaussian of the given bandwidth. Hot spots of events such as
    earthquakes or observations."""
    from scipy.ndimage import gaussian_filter

    pts = metric(points) if template is None else _to_raster_crs(points, template)
    grid = _grid_for(pts, cell, padding, template)
    w_vals = (
        pd.to_numeric(pts[weight], errors="coerce").fillna(0).to_numpy()
        if weight.strip()
        else np.ones(len(pts))
    )
    col = ((pts.geometry.x.to_numpy() - grid.x0) / grid.cell).astype(int)
    row = ((grid.y0 - pts.geometry.y.to_numpy()) / grid.cell).astype(int)
    ok = (row >= 0) & (row < grid.rows) & (col >= 0) & (col < grid.cols)
    counts = np.zeros(grid.data.shape)
    np.add.at(counts, (row[ok], col[ok]), w_vals[ok])
    smooth = gaussian_filter(counts, sigma=max(bandwidth / grid.cell, 0.01), mode="constant")
    per_km2 = smooth / (grid.cell_area / 1e6)
    return grid.like(per_km2, "density per km²")


# --- zonal statistics ----------------------------------------------------------------------------


@node(category="Geo/Raster", title="Zonal Statistics")
def zonal_statistics(
    zones: gpd.GeoDataFrame,
    raster: Raster,
    prefix: Annotated[str, Param(description="Column name prefix; empty: the raster's name")] = "",
    threshold: float = 0.0,
    share_above: Annotated[
        bool, Param(description="Add the % of each zone's area above the threshold")
    ] = False,
    categories: Annotated[
        bool, Param(description="Classified raster: the % of each zone in each class")
    ] = False,
) -> gpd.GeoDataFrame:
    """Summarise a raster inside each polygon: mean, min, max, standard
    deviation, and the area covered (km²); optionally the share above a
    threshold (e.g. of an interpolated concentration over a limit), or, for a
    classified raster, the share of each class. A cell belongs to a zone when
    its centre is inside."""
    z = _to_raster_crs(zones, raster).reset_index(drop=True)
    base = prefix.strip() or (raster.name or "value").split(" (")[0].replace(" ", "_")
    rows = []
    class_names = list(raster.categories)
    for geom in z.geometry:
        inside = (
            _polygon_mask(raster, geom) if geom is not None else np.zeros(raster.data.shape, bool)
        )
        v = raster.data[inside]
        v = v[np.isfinite(v)]
        row: dict[str, float] = {}
        if categories and class_names:
            for i, cls in enumerate(class_names):
                row[f"{cls}_pct"] = 100 * float(np.sum(v == i)) / v.size if v.size else np.nan
        else:
            row |= {
                f"{base}_mean": float(np.mean(v)) if v.size else np.nan,
                f"{base}_min": float(np.min(v)) if v.size else np.nan,
                f"{base}_max": float(np.max(v)) if v.size else np.nan,
                f"{base}_std": float(np.std(v, ddof=1)) if v.size > 1 else np.nan,
            }
            if share_above:
                row[f"{base}_above_pct"] = (
                    100 * float(np.sum(v > threshold)) / v.size if v.size else np.nan
                )
        row["cells"] = int(v.size)
        row["cell_area_km2"] = v.size * raster.cell_area / 1e6
        rows.append(row)
    stats = pd.DataFrame(rows, index=z.index)
    out = zones.reset_index(drop=True).copy()
    for c in stats.columns:
        out[c] = stats[c].to_numpy()
    return out


@zonal_statistics.check
def _check_zonal(categories: bool = False, share_above: bool = False):
    if categories and share_above:
        return warning("With categories on, the share above the threshold is not computed")
    return None
