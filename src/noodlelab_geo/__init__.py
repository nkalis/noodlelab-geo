"""Geospatial nodes. Requires the ``geo`` extra: ``pip install "noodlelab[geo]"``.

Vector data are GeoPandas ``GeoDataFrame`` values: a table with a geometry
column and a coordinate reference system (CRS). A GeoDataFrame connects to
every table input too. Rasters are :class:`~.types.Raster` values: a NumPy grid
with its position, cell size and CRS, read from and written to ESRI ASCII
grids (``.asc``, with an optional ``.prj`` for the CRS).

Distances, buffers and areas are computed in metres: data in longitude and
latitude is projected to its UTM zone for the calculation, so no node asks
for a projected CRS first.

* :mod:`.vector`: reading, projecting, buffering, joining, measuring
* :mod:`.raster`: terrain, indices, classification, interpolation, zonal statistics
* :mod:`.track`: GPS tracks
* :mod:`.seismology`: earthquake catalogues
* :mod:`.plot`: maps
"""

from __future__ import annotations

from noodlelab.plugin import require

require("geo", "geopandas", "matplotlib", "pyproj", "scipy", "shapely")

from .plot import *  # noqa: F403
from .raster import *  # noqa: F403
from .seismology import *  # noqa: F403
from .track import *  # noqa: F403
from .types import Raster  # noqa: F401
from .vector import *  # noqa: F403
