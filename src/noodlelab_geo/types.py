"""The geo pack's types: :class:`Raster`, and editor support for GeoDataFrame.

Both get a socket colour, a map thumbnail as preview, meta for the editor's
dropdowns, a sampler for previews while editing, and a checkpoint codec that
needs no pickle.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field, replace
from typing import IO, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from numpy.typing import NDArray

from noodlelab import (
    Preview,
    image_preview,
    register_codec,
    register_meta,
    register_preview,
    register_sampler,
    register_type,
)
from noodlelab.core.meta import numeric_columns

GDF = "geopandas.geodataframe.GeoDataFrame"


# --- rasters ---------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Raster:
    """A north-up grid: ``data[0, 0]`` is the north-west cell.

    ``x0``/``y0`` are the coordinates of the grid's west and north edges,
    ``cell`` the cell size in CRS units (metres for projected CRSs). Missing
    values are NaN. ``categories`` names the classes of a classified raster,
    whose values are then 0, 1, 2, ...
    """

    data: NDArray[np.float64]
    x0: float
    y0: float
    cell: float
    crs: str | None = None
    name: str = ""
    categories: tuple[str, ...] = field(default_factory=tuple)

    @property
    def rows(self) -> int:
        return int(self.data.shape[0])

    @property
    def cols(self) -> int:
        return int(self.data.shape[1])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(west, south, east, north)"""
        return (
            self.x0,
            self.y0 - self.rows * self.cell,
            self.x0 + self.cols * self.cell,
            self.y0,
        )

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(west, east, south, north), as Matplotlib's imshow takes it."""
        w, s, e, n = self.bounds
        return (w, e, s, n)

    @property
    def cell_area(self) -> float:
        return self.cell * self.cell

    def centers(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """x and y of every cell centre, each shaped like ``data``."""
        xs = self.x0 + (np.arange(self.cols) + 0.5) * self.cell
        ys = self.y0 - (np.arange(self.rows) + 0.5) * self.cell
        return np.meshgrid(xs, ys)

    def like(self, data: Any, name: str | None = None, categories: tuple[str, ...] = ()) -> Raster:
        """A raster on the same grid with other values."""
        arr = np.asarray(data, dtype=np.float64)
        if arr.shape != self.data.shape:
            raise ValueError(f"Expected a {self.data.shape} grid, got {arr.shape}")
        return replace(
            self, data=arr, name=self.name if name is None else name, categories=tuple(categories)
        )

    def same_grid(self, other: Raster) -> bool:
        return (
            self.data.shape == other.data.shape
            and np.isclose(self.x0, other.x0)
            and np.isclose(self.y0, other.y0)
            and np.isclose(self.cell, other.cell)
            and same_crs(self.crs, other.crs)
        )

    def valid(self) -> NDArray[np.float64]:
        """The values that are not missing, as a flat array."""
        v = self.data.ravel()
        return v[np.isfinite(v)]

    def __repr__(self) -> str:
        return (
            f"Raster({self.name or 'unnamed'}, {self.rows}×{self.cols}, cell {self.cell:g}, "
            f"crs {crs_label(self.crs)})"
        )


def same_crs(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return pyproj.CRS.from_user_input(a) == pyproj.CRS.from_user_input(b)
    except pyproj.exceptions.CRSError:
        return str(a) == str(b)


def crs_label(crs: Any) -> str:
    """ "EPSG:32632" when the CRS has an EPSG code, else its name."""
    if crs is None:
        return "no CRS"
    try:
        c = pyproj.CRS.from_user_input(crs)
    except pyproj.exceptions.CRSError:
        return str(crs)
    code = c.to_epsg()
    return f"EPSG:{code}" if code else c.name


def crs_string(crs: Any) -> str | None:
    """A compact, storable form of a CRS: "EPSG:xxxx" or WKT."""
    if crs is None:
        return None
    c = pyproj.CRS.from_user_input(crs)
    code = c.to_epsg()
    return f"EPSG:{code}" if code else c.to_wkt()


def is_geographic(crs: Any) -> bool:
    return crs is not None and pyproj.CRS.from_user_input(crs).is_geographic


X_NAMES = ("lon", "longitude", "long", "lng", "x", "easting", "east", "x_coord", "xcoord")
Y_NAMES = ("lat", "latitude", "y", "northing", "north", "y_coord", "ycoord")


def _numeric_share(values: Any) -> float:
    return float(pd.to_numeric(values, errors="coerce").notna().mean()) if len(values) else 0.0


def coordinate_columns(table: Any) -> tuple[str, str] | None:
    """The columns of a table that hold x and y (or longitude and latitude)
    coordinates, found by name ("lon", "Latitude", "easting"...) and numeric
    content; None when there are none."""
    by_name = {str(c).strip().lower().replace(" ", "_"): c for c in table.columns}

    def find(names: tuple[str, ...]) -> Any:
        for n in names:
            c = by_name.get(n)
            if c is not None and _numeric_share(table[c]) >= 0.5:
                return c
        return None

    x, y = find(X_NAMES), find(Y_NAMES)
    return None if x is None or y is None else (str(x), str(y))


def metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The data in a CRS with metres: unchanged when already projected, else
    in its UTM zone. Data without a CRS is assumed to be in metres."""
    if gdf.crs is None or not gdf.crs.is_geographic:
        return gdf
    return gdf.to_crs(gdf.estimate_utm_crs())


def _save_raster(r: Raster, fh: IO[bytes]) -> dict[str, Any]:
    np.save(fh, np.ascontiguousarray(r.data, dtype=np.float64), allow_pickle=False)
    return {
        "x0": r.x0,
        "y0": r.y0,
        "cell": r.cell,
        "crs": r.crs,
        "name": r.name,
        "categories": list(r.categories),
    }


def _load_raster(fh: IO[bytes], info: dict[str, Any]) -> Raster:
    data = np.load(fh, allow_pickle=False)
    return Raster(
        data,
        float(info["x0"]),
        float(info["y0"]),
        float(info["cell"]),
        info.get("crs"),
        info.get("name", ""),
        tuple(info.get("categories", ())),
    )


register_type(Raster, "RASTER", "#63c7a1", "A georeferenced grid of values")
register_codec(Raster, "raster-npy", save=_save_raster, load=_load_raster, suffix=".npy")


@register_meta(Raster)
def _raster_meta(r: Raster) -> dict[str, Any]:
    v = r.valid()
    return {
        "shape": [r.rows, r.cols],
        "cell": r.cell,
        "crs": crs_label(r.crs),
        "bounds": list(r.bounds),
        "min": float(v.min()) if v.size else None,
        "max": float(v.max()) if v.size else None,
        "categories": list(r.categories),
    }


@register_sampler(Raster)
def _raster_sample(r: Raster, size: int) -> Raster:
    step = max(1, int(np.ceil(max(r.rows, r.cols) / size)))
    if step == 1:
        return r
    return replace(r, data=r.data[::step, ::step], cell=r.cell * step)


def colorize(data: NDArray[np.float64], cmap: str = "viridis", categories: int = 0) -> np.ndarray:
    """RGBA bytes for a grid, missing values transparent. Continuous values are
    stretched between the 2nd and 98th percentile; categories get distinct colours."""
    from matplotlib import colormaps

    ok = np.isfinite(data)
    if categories:
        cm = colormaps["tab10" if categories <= 10 else "tab20"]
        rgba = cm(np.where(ok, data, 0).astype(int) % cm.N)
    else:
        v = data[ok]
        lo, hi = np.percentile(v, [2, 98]) if v.size else (0.0, 1.0)
        scaled = np.clip((data - lo) / ((hi - lo) or 1.0), 0, 1)
        rgba = colormaps[cmap](np.where(ok, scaled, 0))
    rgba[..., 3] = np.where(ok, 1.0, 0.0)
    return (rgba * 255).astype(np.uint8)


@register_preview(Raster)
def _raster_preview(r: Raster, ctx: Any) -> Preview:
    v = r.valid()
    rng = f", {v.min():.4g} … {v.max():.4g}" if v.size and not r.categories else ""
    kinds = f", {len(r.categories)} classes" if r.categories else ""
    return image_preview(
        colorize(r.data, categories=len(r.categories)),
        summary=f"raster {r.rows}×{r.cols} @ {r.cell:g}{rng}{kinds}",
    )


# --- GeoDataFrame ------------------------------------------------------------------------------


def geometry_types(gdf: gpd.GeoDataFrame) -> list[str]:
    return sorted({str(t) for t in gdf.geom_type.dropna().unique()})


def map_thumbnail(gdf: gpd.GeoDataFrame, size: float = 3.0) -> str:
    """A small map of the features as a PNG data URL."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(size, size), layout="constrained")
    ax = fig.add_subplot()
    if len(gdf):
        gdf.plot(
            ax=ax, color="#4c72b0", edgecolor="#1f3b63", linewidth=0.6, markersize=6, alpha=0.8
        )
    ax.set_axis_off()
    ax.set_aspect("equal" if gdf.crs is None or not gdf.crs.is_geographic else "auto")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@register_preview(GDF)
def _gdf_preview(gdf: gpd.GeoDataFrame, ctx: Any) -> Preview:
    kinds = "/".join(geometry_types(gdf)) or "empty"
    summary = f"{len(gdf)} features · {kinds} · {crs_label(gdf.crs)}"
    columns = [str(c) for c in gdf.columns if c != gdf.geometry.name]
    return Preview(
        kind="image",
        summary=summary,
        image=map_thumbnail(gdf),
        text=f"{summary}\ncolumns: {', '.join(columns)}",
    )


@register_meta(GDF)
def _gdf_meta(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    return {
        "columns": [str(c) for c in gdf.columns][:500],
        "rows": int(len(gdf)),
        "crs": crs_label(gdf.crs),
        "geometry": geometry_types(gdf),
        "dtypes": {str(c): str(t) for c, t in list(gdf.dtypes.items())[:500]},
        "numeric": numeric_columns(gdf),
    }


def _save_gdf(gdf: gpd.GeoDataFrame, fh: IO[bytes]) -> dict[str, Any]:
    if type(gdf) is not gpd.GeoDataFrame or gdf.attrs:
        raise TypeError("only plain GeoDataFrames are stored as GeoParquet")
    buf = io.BytesIO()
    gdf.to_parquet(buf)
    data = buf.getvalue()
    back = gpd.read_parquet(io.BytesIO(data))
    from geopandas.testing import assert_geodataframe_equal

    assert_geodataframe_equal(back, gdf, check_less_precise=False)
    fh.write(data)
    return {"crs": json.dumps(crs_string(gdf.crs))}


def _load_gdf(fh: IO[bytes], info: dict[str, Any]) -> gpd.GeoDataFrame:
    return gpd.read_parquet(io.BytesIO(fh.read()))


register_type(GDF, "GEODATAFRAME", "#4fb6d9", "Vector features with a CRS (GeoPandas)")
register_codec(GDF, "geoparquet", save=_save_gdf, load=_load_gdf, suffix=".parquet")
