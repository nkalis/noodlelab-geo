"""Earthquake catalogues: the magnitude–frequency (Gutenberg–Richter) relation."""

from __future__ import annotations

from typing import Annotated, Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from noodlelab import Param, node

__all__ = ["gutenberg_richter"]


class GutenbergRichter(NamedTuple):
    b_value: float
    b_error: float
    a_value: float
    completeness: float
    events: int
    table: pd.DataFrame
    plot: Figure
    summary: dict[str, Any]


def _max_curvature(m: np.ndarray, width: float) -> float:
    """The magnitude bin with the most events (Wiemer & Wyss 2000)."""
    bins = np.round(m / width) * width
    values, counts = np.unique(np.round(bins, 6), return_counts=True)
    return float(values[np.argmax(counts)])


@node(category="Geo/Seismology", title="Gutenberg-Richter", sample=False)
def gutenberg_richter(
    catalog: pd.DataFrame,
    magnitude: Annotated[str, Param(options_from="catalog.columns")] = "magnitude",
    bin_width: Annotated[float, Param(min=0.01, max=1.0, step=0.05, precision=2)] = 0.1,
    completeness: Literal["maximum curvature", "fixed"] = "maximum curvature",
    mc: Annotated[float, Param(description="The fixed magnitude of completeness")] = 2.0,
    correction: Annotated[
        float, Param(description="Added to the maximum-curvature estimate (often 0.2)")
    ] = 0.2,
) -> GutenbergRichter:
    """The magnitude–frequency distribution log₁₀ N(≥M) = a − b·M of a catalogue.

    Estimates the magnitude of completeness Mc (maximum curvature plus a
    correction, or a fixed value), the b-value by Aki's maximum likelihood
    formula with Utsu's bin correction, b = log₁₀e / (M̄ − (Mc − Δ/2)), its
    uncertainty by Shi & Bolt (1982), and the a-value. Also the table of
    counts per magnitude bin and the classic plot of both distributions.
    """
    if magnitude not in catalog.columns:
        raise KeyError(f"No column '{magnitude}'")
    m = pd.to_numeric(catalog[magnitude], errors="coerce").dropna().to_numpy(np.float64)
    if completeness == "maximum curvature":
        mc_value = _max_curvature(m, bin_width) + correction
    else:
        mc_value = mc
    mc_value = round(mc_value / bin_width) * bin_width
    above = m[m >= mc_value - 1e-9]
    n = len(above)
    if n < 10:
        raise ValueError(f"Only {n} events at or above Mc = {mc_value:.2f}: too few for a b-value")
    mean = float(np.mean(above))
    b = np.log10(np.e) / (mean - (mc_value - bin_width / 2))
    b_err = 2.3 * b**2 * float(np.sqrt(np.sum((above - mean) ** 2) / (n * (n - 1))))
    a = np.log10(n) + b * mc_value
    edges = np.arange(np.floor(m.min() / bin_width) * bin_width, m.max() + bin_width, bin_width)
    centres = np.round(edges, 6)
    binned = np.round(np.round(m / bin_width) * bin_width, 6)
    counts = np.array([int(np.sum(binned == c)) for c in centres])
    cumulative = np.array([int(np.sum(m >= c - bin_width / 2)) for c in centres])
    table = pd.DataFrame(
        {
            "magnitude": centres,
            "count": counts,
            "cumulative": cumulative,
            "model": 10 ** (a - b * centres),
        }
    )
    table = table[(table["count"] > 0) | (table["cumulative"] > 0)].reset_index(drop=True)

    fig = Figure(figsize=(6, 3.8), layout="constrained")
    ax = fig.add_subplot()
    ax.semilogy(
        table["magnitude"],
        table["cumulative"],
        "o",
        ms=4,
        color="#4c72b0",
        label="N(≥M), cumulative",
    )
    ax.semilogy(
        table["magnitude"],
        table["count"].where(table["count"] > 0),
        "s",
        ms=3,
        color="#8c8c8c",
        alpha=0.8,
        label="events per bin",
    )
    fit = table[table["magnitude"] >= mc_value - 1e-9]
    ax.semilogy(
        fit["magnitude"],
        fit["model"],
        "-",
        color="#c44e52",
        lw=2,
        label=f"log N = {a:.2f} − {b:.2f} M",
    )
    ax.axvline(mc_value, color="0.4", ls="--", lw=1, label=f"Mc = {mc_value:.1f}")
    ax.set(xlabel="Magnitude", ylabel="Number of events", title="Magnitude–frequency distribution")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    summary = {
        "Events in catalogue": len(m),
        "Magnitude of completeness Mc": round(mc_value, 2),
        "Events ≥ Mc": n,
        "b-value": f"{b:.3f} ± {b_err:.3f}",
        "a-value": a,
        "Largest magnitude": float(m.max()),
        "Method": "Aki–Utsu maximum likelihood, Shi–Bolt uncertainty",
    }
    return GutenbergRichter(
        float(b), float(b_err), float(a), float(mc_value), n, table, fig, summary
    )
