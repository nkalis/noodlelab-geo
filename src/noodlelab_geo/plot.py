"""Maps: rasters and vector layers drawn together, with a scale bar and a north arrow."""

from __future__ import annotations

from typing import Annotated, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from noodlelab import Param, node

from .types import Raster, is_geographic, same_crs

__all__ = ["map_plot"]

RasterColormap = Literal[
    "viridis",
    "terrain",
    "gist_earth",
    "RdYlGn",
    "YlGnBu",
    "YlOrRd",
    "magma",
    "Blues",
    "coolwarm",
    "Greys",
]
PointColormap = Literal["viridis", "plasma", "YlOrRd", "RdYlBu_r", "coolwarm", "magma"]
Categorical = (
    "#3b7dd8",
    "#d9b26f",
    "#9bd46a",
    "#2e7d32",
    "#9e9e9e",
    "#c44e52",
    "#8172b3",
    "#64b5cd",
)


def _nice(length: float) -> float:
    exp = np.floor(np.log10(length))
    for f in (5, 2):
        if f * 10**exp <= length:
            return float(f * 10**exp)
    return float(10**exp)


def _scale_bar(ax, geographic: bool) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    metres_per_unit = 1.0
    if geographic:
        metres_per_unit = 111_320 * np.cos(np.radians((y0 + y1) / 2))
    width_m = (x1 - x0) * metres_per_unit
    length_m = _nice(width_m / 5)
    length = length_m / metres_per_unit
    bx, by = x0 + 0.05 * (x1 - x0), y0 + 0.05 * (y1 - y0)
    h = 0.012 * (y1 - y0)
    ax.fill_between([bx, bx + length / 2], by, by + h, color="black", zorder=10)
    ax.fill_between(
        [bx + length / 2, bx + length], by, by + h, color="white", edgecolor="black", zorder=10
    )
    ax.plot(
        [bx, bx + length, bx + length, bx, bx],
        [by, by, by + h, by + h, by],
        color="black",
        lw=0.8,
        zorder=11,
    )
    label = f"{length_m / 1000:g} km" if length_m >= 1000 else f"{length_m:g} m"
    ax.text(
        bx + length / 2,
        by + 2.2 * h,
        label,
        ha="center",
        va="bottom",
        fontsize=8,
        zorder=11,
        bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 1},
    )


def _north_arrow(ax) -> None:
    ax.annotate(
        "N",
        xy=(0.95, 0.93),
        xytext=(0.95, 0.80),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        arrowprops={"facecolor": "black", "width": 3, "headwidth": 9, "headlength": 8},
        zorder=12,
    )


def _in(gdf: gpd.GeoDataFrame | None, crs) -> gpd.GeoDataFrame | None:
    if gdf is None or crs is None or gdf.crs is None or same_crs(gdf.crs, crs):
        return gdf
    return gdf.to_crs(crs)


@node(category="Geo/Plot", title="Map Plot")
def map_plot(
    raster: Raster | None = None,
    hillshade: Annotated[
        Raster | None, Param(description="Grey relief drawn under the raster")
    ] = None,
    polygons: gpd.GeoDataFrame | None = None,
    lines: gpd.GeoDataFrame | None = None,
    points: gpd.GeoDataFrame | None = None,
    polygon_color: Annotated[
        str, Param(options_from="polygons.columns", empty="outlines", description="Fill by column")
    ] = "",
    polygon_labels: Annotated[
        str, Param(options_from="polygons.columns", empty="no labels", description="Label column")
    ] = "",
    point_color: Annotated[
        str,
        Param(options_from="points.columns", empty="one colour", description="Colour by column"),
    ] = "",
    point_labels: Annotated[
        str, Param(options_from="points.columns", empty="no labels", description="Label column")
    ] = "",
    point_size: Annotated[float, Param(min=1, max=200)] = 18.0,
    size_by: Annotated[
        str, Param(options_from="points.numeric", empty="same size", description="Scale by column")
    ] = "",
    raster_colormap: RasterColormap = "viridis",
    class_colors: Annotated[
        str,
        Param(
            description="Classified rasters: one colour per class, e.g. #2e7d32, #f2c14e, #d7301f"
        ),
    ] = "",
    point_colormap: PointColormap = "YlOrRd",
    raster_label: str = "",
    point_label: str = "",
    title: str = "",
    scale_bar: bool = True,
    north_arrow: bool = True,
    width: Annotated[float, Param(min=3, max=12, description="Inches")] = 6.5,
) -> Figure:
    """A map from any mix of layers, drawn bottom to top: hillshade, raster,
    polygons, lines, points. Layers are projected to the CRS of the first of
    raster, polygons, lines and points. Classified rasters get a legend, other
    rasters and numeric point colours a colour bar; the default class colours
    suit land cover (water, bare, grassland, forest), ``class_colors`` sets others."""
    crs = next(
        (
            c
            for c in (
                raster.crs if raster is not None else None,
                polygons.crs if polygons is not None else None,
                lines.crs if lines is not None else None,
                points.crs if points is not None else None,
            )
            if c is not None
        ),
        None,
    )
    polygons, lines, points = _in(polygons, crs), _in(lines, crs), _in(points, crs)
    extent_src = raster or hillshade
    bounds = []
    if extent_src is not None:
        bounds.append(extent_src.bounds)
    for layer in (polygons, lines, points):
        if layer is not None and len(layer):
            bounds.append(tuple(layer.total_bounds))
    if not bounds:
        raise ValueError("Link at least one layer")
    b = np.array(bounds)
    w, s, e, n = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    aspect = (n - s) / (e - w) if e > w else 1.0
    geographic = is_geographic(crs)
    if geographic:
        aspect /= np.cos(np.radians((n + s) / 2))
    fig = Figure(
        figsize=(width, max(3.0, min(width * aspect + 0.8, width * 1.6))), layout="constrained"
    )
    ax = fig.add_subplot()
    legend_handles = []
    if hillshade is not None:
        ax.imshow(
            hillshade.data,
            extent=hillshade.extent,
            cmap="Greys_r",
            vmin=0,
            vmax=1,
            interpolation="bilinear",
            zorder=1,
        )
    if raster is not None:
        alpha = (0.7 if raster.categories else 0.55) if hillshade is not None else 1.0
        if raster.categories:
            k = len(raster.categories)
            from matplotlib import colormaps
            from matplotlib.colors import ListedColormap

            chosen = [c.strip() for c in class_colors.split(",") if c.strip()]
            if len(chosen) >= k:
                cmap = ListedColormap(chosen[:k])
            elif k <= len(Categorical):
                cmap = ListedColormap(list(Categorical[:k]))
            else:
                cmap = colormaps["tab20"].resampled(k)
            ax.imshow(
                np.ma.masked_invalid(raster.data),
                extent=raster.extent,
                cmap=cmap,
                vmin=-0.5,
                vmax=k - 0.5,
                interpolation="nearest",
                alpha=alpha,
                zorder=2,
            )
            legend_handles += [
                Patch(color=cmap(i), label=c) for i, c in enumerate(raster.categories)
            ]
        else:
            valid = raster.valid()
            lo, hi = np.percentile(valid, [2, 98]) if valid.size else (0, 1)
            im = ax.imshow(
                np.ma.masked_invalid(raster.data),
                extent=raster.extent,
                cmap=raster_colormap,
                vmin=lo,
                vmax=hi,
                interpolation="bilinear",
                alpha=alpha,
                zorder=2,
            )
            fig.colorbar(im, ax=ax, shrink=0.75, label=raster_label or raster.name)
    if polygons is not None and len(polygons):
        if polygon_color.strip():
            col = polygons[polygon_color]
            numeric = pd.api.types.is_numeric_dtype(col)
            polygons.plot(
                ax=ax,
                column=polygon_color,
                cmap=raster_colormap if numeric else None,
                categorical=not numeric,
                legend=True,
                alpha=0.75,
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
                legend_kwds={"shrink": 0.75, "label": polygon_color}
                if numeric
                else {"fontsize": 8},
            )
        else:
            polygons.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=3)
        if polygon_labels.strip():
            reps = polygons.geometry.representative_point()
            for (x, y), text in zip(
                zip(reps.x, reps.y, strict=True), polygons[polygon_labels], strict=True
            ):
                ax.text(
                    x,
                    y,
                    str(text),
                    ha="center",
                    va="center",
                    fontsize=8,
                    zorder=6,
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
                )
    if lines is not None and len(lines):
        lines.plot(ax=ax, color="#1565c0", linewidth=1.8, zorder=4)
    if points is not None and len(points):
        sizes = point_size
        edge = 0.4 if point_size >= 10 else 0.0  # small dots: edges would hide the colours
        if size_by.strip():
            v = pd.to_numeric(points[size_by], errors="coerce").to_numpy(float)
            lo, hi = np.nanmin(v), np.nanmax(v)
            sizes = point_size * (0.3 + 2.7 * (v - lo) / ((hi - lo) or 1.0)) ** 1.5
        if point_color.strip():
            c = points[point_color]
            if pd.api.types.is_numeric_dtype(c):
                sc = ax.scatter(
                    points.geometry.x,
                    points.geometry.y,
                    c=c,
                    cmap=point_colormap,
                    s=sizes,
                    edgecolors="black",
                    linewidths=edge,
                    zorder=5,
                )
                fig.colorbar(sc, ax=ax, shrink=0.75, label=point_label or point_color)
            else:
                for i, key in enumerate(pd.unique(c)):
                    sel = (c == key).to_numpy()
                    handle = ax.scatter(
                        points.geometry.x[sel],
                        points.geometry.y[sel],
                        s=sizes[sel] if np.ndim(sizes) else sizes,
                        color=Categorical[(i + 3) % len(Categorical)],
                        edgecolors="black",
                        linewidths=edge,
                        zorder=5,
                        label=str(key),
                    )
                    legend_handles.append(handle)
        else:
            ax.scatter(
                points.geometry.x,
                points.geometry.y,
                s=sizes,
                color="#c44e52",
                edgecolors="black",
                linewidths=edge,
                zorder=5,
            )
    if points is not None and len(points) and point_labels.strip():
        dx = 0.012 * (e - w)
        for x, y, text in zip(
            points.geometry.x, points.geometry.y, points[point_labels], strict=True
        ):
            ax.text(
                x + dx,
                y,
                str(text),
                fontsize=8,
                va="center",
                zorder=6,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.2},
            )
    pad_x, pad_y = 0.02 * (e - w), 0.02 * (n - s)
    ax.set_xlim(w - pad_x, e + pad_x)
    ax.set_ylim(s - pad_y, n + pad_y)
    if geographic:
        ax.set_aspect(1 / np.cos(np.radians((n + s) / 2)))
        ax.set(xlabel="Longitude (°)", ylabel="Latitude (°)")
    else:
        ax.set_aspect("equal")
        ax.set(xlabel="Easting (m)", ylabel="Northing (m)")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(labelsize=7)
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=8, loc="lower right", framealpha=0.85)
    if scale_bar:
        _scale_bar(ax, geographic)
    if north_arrow:
        _north_arrow(ax)
    ax.set_title(title)
    return fig
