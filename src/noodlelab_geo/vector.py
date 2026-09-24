"""Vector data: points, lines and polygons as GeoDataFrames."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Annotated, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj

from noodlelab import FileRef, Param, Probe, Quantity, RunContext, node, warning

from .types import coordinate_columns, crs_label, geometry_types, metric

__all__ = [
    "add_coordinates",
    "as_features",
    "buffer",
    "centroids",
    "clip",
    "count_in_polygons",
    "dissolve",
    "distance_to_nearest",
    "filter_features",
    "measure",
    "points_from_table",
    "read_vector",
    "reproject",
    "save_vector",
    "spatial_join",
]

VECTOR_FILES = (".geojson", ".json", ".gpkg", ".shp", ".fgb", ".zip")
VectorFile = Annotated[FileRef, Param(accept=VECTOR_FILES)]
Metres = Quantity["m"]
NO_LIMIT = Quantity(0.0, "m")

_UTM = ("utm", "auto", "utm (auto)")


def parse_crs(text: str) -> pyproj.CRS:
    try:
        return pyproj.CRS.from_user_input(text.strip())
    except pyproj.exceptions.CRSError:
        raise ValueError(
            f"Unknown CRS '{text}': use an EPSG code such as EPSG:4326 or EPSG:32632"
        ) from None


def _crs_problem(text: str, allow_utm: bool = False) -> str | None:
    if not text.strip():
        return "Enter a CRS, such as EPSG:4326"
    if allow_utm and text.strip().lower() in _UTM:
        return None
    try:
        pyproj.CRS.from_user_input(text.strip())
    except pyproj.exceptions.CRSError:
        return f"Unknown CRS '{text}': use an EPSG code such as EPSG:4326 or EPSG:32632"
    return None


def _column(source: str = "data", **kw) -> Param:
    return Param(options_from=f"{source}.columns", **kw)


# --- reading and writing --------------------------------------------------------------------


def _read(path: FileRef, layer: str, max_features: int | None = None) -> gpd.GeoDataFrame:
    kwargs = {"layer": layer or None, "max_features": max_features}
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return gpd.read_file(path.local_path(), engine="pyogrio", **kwargs)


@node(category="Geo/Vector", title="Read Vector File", converter=True)
def read_vector(
    path: VectorFile,
    layer: Annotated[str, Param(description="For GeoPackages with several layers")] = "",
) -> gpd.GeoDataFrame:
    """Read GeoJSON, GeoPackage, Shapefile or FlatGeobuf features, from the
    workspace or remote storage (downloaded once and cached)."""
    return _read(path, layer)


@read_vector.probe
def _probe_read_vector(path: FileRef, layer: str = ""):
    head = _read(path, layer, max_features=200)
    kinds = "/".join(geometry_types(head)) or "no geometry"
    return Probe(preview=head, summary=f"{kinds} · {crs_label(head.crs)}")


@node(category="Output", title="Save Vector File")
def save_vector(
    data: gpd.GeoDataFrame, ctx: RunContext, filename: str = "features.geojson"
) -> Path:
    """Write features into this run's output folder. The extension picks the
    format: .geojson, .gpkg (GeoPackage) or .fgb (FlatGeobuf)."""
    drivers = {".geojson": "GeoJSON", ".json": "GeoJSON", ".gpkg": "GPKG", ".fgb": "FlatGeobuf"}
    suffix = Path(filename).suffix.lower()
    if suffix not in drivers:
        filename, suffix = filename + ".geojson", ".geojson"
    out = ctx.path(filename)
    if out.exists():
        out.unlink()
    data.to_file(out, driver=drivers[suffix], engine="pyogrio")
    ctx.log(f"Wrote {len(data)} features to {out.name}")
    return out


@node(category="Geo/Vector", title="As Features", converter=True)
def as_features(
    table: pd.DataFrame,
    crs: Annotated[str, Param(description="Only for tables that lost their CRS")] = "",
) -> gpd.GeoDataFrame:
    """Features again after table nodes: the table nodes (Add Column, Filter
    Rows, Join Tables, ...) work on GeoDataFrames too, but say they return a
    plain table. This passes a GeoDataFrame through, or rebuilds one from a
    ``geometry`` column. Offered when such a table is linked into a
    features input."""
    if isinstance(table, gpd.GeoDataFrame):
        return table
    if "geometry" in table.columns:
        return gpd.GeoDataFrame(
            table, geometry="geometry", crs=parse_crs(crs) if crs.strip() else None
        )
    raise TypeError(
        "This table has no geometry: use Points From Table to make features from coordinates"
    )


@as_features.accepts
def _as_features_fits(table: pd.DataFrame) -> bool:
    return isinstance(table, gpd.GeoDataFrame) or "geometry" in table.columns


@node(category="Geo/Vector", title="Points From Table", converter=True)
def points_from_table(
    table: pd.DataFrame,
    x: Annotated[str, Param(options_from="table.columns", label="X / longitude")] = "lon",
    y: Annotated[str, Param(options_from="table.columns", label="Y / latitude")] = "lat",
    crs: Annotated[str, Param(description="The CRS of the coordinates")] = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Make point features from two coordinate columns, e.g. station or sample
    locations in a CSV. Rows without coordinates are dropped."""
    for c in (x, y):
        if c not in table.columns:
            raise KeyError(f"No column '{c}'; the table has {', '.join(map(str, table.columns))}")
    xs = pd.to_numeric(table[x], errors="coerce")
    ys = pd.to_numeric(table[y], errors="coerce")
    ok = (xs.notna() & ys.notna()).to_numpy()
    rows = table.loc[ok].reset_index(drop=True)
    return gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy(xs[ok], ys[ok]), crs=parse_crs(crs))


@points_from_table.accepts
def _points_fit(table: pd.DataFrame) -> dict[str, str] | None:
    """Tables with coordinate columns, and which they are. Coordinates outside
    ±180/±90 are projected: the CRS is left for the user to choose."""
    if isinstance(table, gpd.GeoDataFrame):
        return None
    found = coordinate_columns(table)
    if found is None:
        return None
    x, y = found
    xs = pd.to_numeric(table[x], errors="coerce").abs()
    ys = pd.to_numeric(table[y], errors="coerce").abs()
    geographic = bool((xs <= 180).all() and (ys <= 90).all())
    return {"x": x, "y": y} if geographic else {"x": x, "y": y, "crs": ""}


@points_from_table.check
def _check_points(x: str = "lon", y: str = "lat", crs: str = "EPSG:4326"):
    if not x.strip() or not y.strip():
        return "Choose the coordinate columns"
    return _crs_problem(crs)


# --- coordinate systems ----------------------------------------------------------------------


@node(category="Geo/Vector")
def reproject(
    data: gpd.GeoDataFrame,
    crs: Annotated[
        str, Param(description='An EPSG code, or "UTM" for the UTM zone of the data')
    ] = "UTM",
) -> gpd.GeoDataFrame:
    """Transform the coordinates to another CRS: UTM for metres, EPSG:4326 for
    longitude and latitude, or a national grid."""
    if data.crs is None:
        raise ValueError("The data has no CRS: set one where it is read or made")
    if crs.strip().lower() in _UTM:
        return data.to_crs(data.estimate_utm_crs())
    return data.to_crs(parse_crs(crs))


@reproject.check
def _check_reproject(crs: str = "UTM"):
    return _crs_problem(crs, allow_utm=True)


@node(category="Geo/Vector", title="Add Coordinates")
def add_coordinates(
    data: gpd.GeoDataFrame,
    crs: Annotated[str, Param(description="The coordinates' CRS; empty: the data's")] = "EPSG:4326",
    x_name: str = "lon",
    y_name: str = "lat",
) -> gpd.GeoDataFrame:
    """Add each feature's x and y (of its centroid, for lines and polygons) as
    columns, for tables in reports or plots."""
    target = data if not crs.strip() else data.to_crs(parse_crs(crs))
    pts = (
        target.geometry
        if (target.geom_type == "Point").all()
        else metric(target).centroid.to_crs(target.crs)
    )
    out = data.copy()
    out[x_name] = pts.x.to_numpy()
    out[y_name] = pts.y.to_numpy()
    return out


# --- selecting ---------------------------------------------------------------------------------


@node(category="Geo/Vector", title="Filter Features")
def filter_features(
    data: gpd.GeoDataFrame,
    condition: Annotated[
        str, Param(description='A pandas query on the columns: magnitude >= 3 and type == "well"')
    ] = "",
) -> gpd.GeoDataFrame:
    """Keep the features matching a condition on their attributes."""
    plain = re.sub(r"`[^`]*`", "x", condition)
    ast.parse(plain, mode="eval")
    return data.query(condition, engine="python").reset_index(drop=True)


@filter_features.check
def _check_filter(condition: str = ""):
    if not condition.strip():
        return "Enter a condition"
    try:
        ast.parse(re.sub(r"`[^`]*`", "x", condition), mode="eval")
    except SyntaxError as exc:
        return f"Not a valid condition: {exc.msg}"
    return None


@node(category="Geo/Vector")
def clip(data: gpd.GeoDataFrame, mask: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Cut features to the area of the mask polygons (points outside are dropped)."""
    return gpd.clip(data, mask.to_crs(data.crs)).reset_index(drop=True)


# --- geometry operations ----------------------------------------------------------------------


@node(category="Geo/Vector")
def buffer(
    data: gpd.GeoDataFrame,
    distance: Metres = Quantity(100.0, "m"),  # noqa: B008 - immutable
    merge: Annotated[bool, Param(description="Merge overlapping buffers into one area")] = False,
) -> gpd.GeoDataFrame:
    """The area within a distance of each feature, e.g. a protection zone
    around a river. Computed in metres whatever the CRS; the result is in the
    input's CRS."""
    m = metric(data)
    d = float(distance.m_as("m")) if hasattr(distance, "m_as") else float(distance)
    out = m.copy()
    out["geometry"] = m.buffer(d, resolution=16)
    if merge:
        out = gpd.GeoDataFrame({"buffer_m": [d]}, geometry=[out.union_all()], crs=m.crs)
    else:
        out["buffer_m"] = d
    return out.to_crs(data.crs) if data.crs is not None else out


@node(category="Geo/Vector")
def centroids(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The centre point of each feature (computed in metres, returned in the
    input's CRS)."""
    m = metric(data)
    out = m.copy()
    out["geometry"] = m.centroid
    return out.to_crs(data.crs) if data.crs is not None else out


@node(category="Geo/Vector")
def dissolve(
    data: gpd.GeoDataFrame,
    by: Annotated[
        str, _column(empty="all features", description="Merge features with the same value")
    ] = "",
    aggregate: Literal["first", "sum", "mean", "min", "max", "count"] = "sum",
) -> gpd.GeoDataFrame:
    """Merge features into one per group, aggregating their numeric columns."""
    if by and by not in data.columns:
        raise KeyError(f"No column '{by}'")
    numeric = data.select_dtypes("number").columns.tolist()
    agg = {c: aggregate for c in numeric if c != by}
    out = data.dissolve(by=by or None, aggfunc=agg if agg else "first")
    return out.reset_index()


@node(category="Geo/Vector")
def measure(
    data: gpd.GeoDataFrame,
    area_unit: Literal["km²", "ha", "m²"] = "km²",
    length_unit: Literal["km", "m"] = "km",
) -> gpd.GeoDataFrame:
    """Add each feature's area and perimeter (polygons) or length (lines) as
    columns. Longitude/latitude data is measured on the ellipsoid
    (geodesic), projected data in its own units (metres)."""
    out = data.copy()
    area_div = {"km²": 1e6, "ha": 1e4, "m²": 1.0}[area_unit]
    len_div = {"km": 1e3, "m": 1.0}[length_unit]
    if data.crs is not None and data.crs.is_geographic:
        geod = data.crs.get_geod() or pyproj.Geod(ellps="WGS84")
        pairs = [
            geod.geometry_area_perimeter(g) if g is not None else (np.nan, np.nan)
            for g in data.geometry
        ]
        area = np.abs(np.array([p[0] for p in pairs], dtype=float))
        length = np.array(
            [geod.geometry_length(g) if g is not None else np.nan for g in data.geometry]
        )
    else:
        area = data.geometry.area.to_numpy()
        length = data.geometry.length.to_numpy()
    polygons = data.geom_type.isin(["Polygon", "MultiPolygon"]).to_numpy()
    lines = data.geom_type.isin(["LineString", "MultiLineString", "LinearRing"]).to_numpy()
    suffix = {"km²": "km2", "ha": "ha", "m²": "m2"}[area_unit]
    if polygons.any():
        out[f"area_{suffix}"] = np.where(polygons, area / area_div, np.nan)
        out[f"perimeter_{length_unit}"] = np.where(polygons, length / len_div, np.nan)
    if lines.any():
        out[f"length_{length_unit}"] = np.where(lines, length / len_div, np.nan)
    return out


# --- relating layers ---------------------------------------------------------------------------


@node(category="Geo/Vector", title="Spatial Join")
def spatial_join(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    relation: Literal["within", "intersects", "contains", "nearest"] = "within",
    keep: Literal["all left features", "matches only"] = "all left features",
    max_distance: Annotated[Metres, Param(description="Nearest only; 0 m: no limit")] = NO_LIMIT,
) -> gpd.GeoDataFrame:
    """Attach the attributes of ``right`` to each feature of ``left`` by their
    spatial relation: the district each well lies within, or the nearest river
    (with the distance in metres in ``distance_m``)."""
    how = "left" if keep == "all left features" else "inner"
    r = right.to_crs(left.crs) if right.crs != left.crs else right
    r = r.drop(columns=[c for c in ("index_left", "index_right") if c in r.columns])
    if relation == "nearest":
        lm = metric(left)
        rm = r.to_crs(lm.crs)
        limit = (
            float(max_distance.m_as("m")) if hasattr(max_distance, "m_as") else float(max_distance)
        )
        joined = gpd.sjoin_nearest(
            lm, rm, how=how, max_distance=limit or None, distance_col="distance_m"
        )
        joined = joined.to_crs(left.crs)
    else:
        joined = gpd.sjoin(left, r, how=how, predicate=relation)
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined.drop(columns=["index_right"], errors="ignore").reset_index(drop=True)


@node(category="Geo/Vector", title="Distance To Nearest")
def distance_to_nearest(
    data: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    name: str = "distance_m",
) -> gpd.GeoDataFrame:
    """Add the distance in metres from each feature to the nearest target
    feature (e.g. wells to the river), computed in a metric CRS."""
    dm = metric(data)
    tm = targets.to_crs(dm.crs)
    union = tm.union_all()
    out = data.copy()
    out[name] = dm.distance(union).to_numpy()
    return out


@node(category="Geo/Vector", title="Count In Polygons")
def count_in_polygons(
    polygons: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    name: str = "count",
    value: Annotated[
        str,
        Param(
            options_from="points.columns", empty="none", description="Also aggregate this column"
        ),
    ] = "",
    aggregate: Literal["mean", "sum", "max", "min", "median"] = "mean",
    density: Annotated[bool, Param(description="Also add the count per km²")] = True,
) -> gpd.GeoDataFrame:
    """How many points fall in each polygon (earthquakes per region, samples
    per district), optionally with an aggregate of a point attribute and the
    density per km²."""
    pts = points.to_crs(polygons.crs) if points.crs != polygons.crs else points
    pts = pts.drop(columns=[c for c in ("index_left", "index_right") if c in pts.columns])
    poly = polygons.reset_index(drop=True)
    joined = gpd.sjoin(pts, poly[[poly.geometry.name]], how="inner", predicate="within")
    groups = joined.groupby("index_right")
    out = poly.copy()
    out[name] = groups.size().reindex(out.index, fill_value=0).astype(int).to_numpy()
    if value.strip():
        if value not in points.columns:
            raise KeyError(f"No column '{value}' in the points")
        stat = groups[value].agg(aggregate).reindex(out.index)
        out[f"{value}_{aggregate}"] = stat.to_numpy()
    if density:
        area_km2 = metric(poly).area.to_numpy() / 1e6
        out[f"{name}_per_km2"] = out[name].to_numpy() / area_km2
    return out


@spatial_join.check
def _check_join(relation: str = "within", max_distance: Quantity | None = None):
    if relation != "nearest" and max_distance is not None and max_distance.magnitude:
        return warning("The maximum distance only applies to 'nearest'")
    return None
