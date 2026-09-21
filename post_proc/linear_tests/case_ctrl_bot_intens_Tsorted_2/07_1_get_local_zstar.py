#!/usr/bin/env python3
"""Compute local z-star fields from interval reference density profiles.

For each 37-step interval, this script reads the interval-center rho_ref files
produced by 061_get_background_den_profile_87layers_v3.py, linearly
interpolates rho_ref in time, reads the matching rho.zarr interval, and maps
each rho(t, z, y, x) value to a vertical location by interpolating the local
column's (rho_ref, Z) curve.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm


ZARR_DIR = Path("/storage/ceoas-scratch/liux8/basin_modes/sulu_sea/post_proc/linear_tests/case_ctrl_bot_intens_Tsorted_2/zarr_output/")
REF_ROOT = Path("rhoref_zstar_linear_case_ctrl_v3")
OUTPUT_ROOT = REF_ROOT

CHUNK_LEN = 37
NT = 600

# I_MIN = 836
# I_MAX = 1579
# J_MIN = 508
# J_MAX = 1077


def interp_column_zstar(rho_profile: np.ndarray, rho_ref: np.ndarray, z_centers: np.ndarray) -> np.ndarray:
    """Interpolate rho_profile(time, z) onto z using one local rho_ref(z)."""
    valid = np.isfinite(rho_ref) & np.isfinite(z_centers)
    if np.count_nonzero(valid) < 2:
        return np.full_like(rho_profile, np.nan, dtype=np.float32)

    rho_valid = np.asarray(rho_ref[valid], dtype=np.float64)
    z_valid = np.asarray(z_centers[valid], dtype=np.float64)

    order = np.argsort(rho_valid)
    rho_sorted = rho_valid[order]
    z_sorted = z_valid[order]

    rho_unique, inverse = np.unique(rho_sorted, return_inverse=True)
    if rho_unique.size < 2:
        return np.full_like(rho_profile, np.nan, dtype=np.float32)

    z_unique = np.zeros_like(rho_unique, dtype=np.float64)
    counts = np.zeros_like(rho_unique, dtype=np.int64)
    np.add.at(z_unique, inverse, z_sorted)
    np.add.at(counts, inverse, 1)
    z_unique /= counts

    zstar = np.interp(
        np.asarray(rho_profile, dtype=np.float64),
        rho_unique,
        z_unique,
        left=z_unique[0],
        right=z_unique[-1],
    )
    zstar[~np.isfinite(rho_profile)] = np.nan
    return zstar.astype(np.float32)


def reference_center_times() -> list[int]:
    """Return the center time for each 37-step reference-density interval."""
    centers = []
    for k in range((NT - 1) // CHUNK_LEN + 1):
        it1 = k * CHUNK_LEN
        it2 = min(it1 + CHUNK_LEN, NT)
        centers.append((it1 + it2 - 1) // 2)
    return centers


def reference_pair(it: int, centers: list[int]) -> tuple[int, int, float]:
    """Return lower/upper reference times and interpolation weight for it."""
    if it <= centers[0]:
        return centers[0], centers[0], 0.0
    if it >= centers[-1]:
        return centers[-1], centers[-1], 0.0

    upper_idx = int(np.searchsorted(centers, it, side="left"))
    lower = centers[upper_idx - 1]
    upper = centers[upper_idx]
    weight = (it - lower) / (upper - lower)
    return lower, upper, float(weight)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python 071_get_local_zstar.py <k>")

    task_id = int(sys.argv[1])
    if task_id < 0:
        raise ValueError(f"k must be >= 0, got {task_id}.")

    it1 = task_id * CHUNK_LEN
    if it1 >= NT:
        max_k = (NT - 1) // CHUNK_LEN
        raise ValueError(f"k={task_id} is out of range for nt={NT}, chunk={CHUNK_LEN}. Max k is {max_k}.")
    it2 = min(it1 + CHUNK_LEN, NT)
    time_range = list(range(it1, it2))

    centers = reference_center_times()
    ref_pairs = {it: reference_pair(it, centers) for it in time_range}
    needed_ref_times = sorted({t for pair in ref_pairs.values() for t in pair[:2]})

    print(
        f"Loading reference densities {needed_ref_times} for interval {it1} to {it2 - 1}.",
        flush=True,
    )
    ref_arrays = {}
    z_centers = None
    for ref_time in needed_ref_times:
        ref_file = REF_ROOT / f"rhoref_t_{ref_time:03d}.nc"
        with xr.open_dataset(ref_file) as ref_ds:
            ref_arrays[ref_time] = ref_ds["rho_ref"].values
            if z_centers is None:
                z_centers = ref_ds["Z"].values
    if z_centers is None:
        raise RuntimeError("No reference density files were loaded.")

    print(f"Loading rho.zarr interval {it1} to {it2 - 1}.", flush=True)
    ds_rho = xr.open_zarr(ZARR_DIR / "rho.zarr", consolidated=False)
    rho_chunk = ds_rho["rho"].isel(
        {
            "time": slice(it1, it2),
            "z": slice(0, len(z_centers)),
            "y": slice(0, ds_rho.sizes["y"]),
            "x": slice(0, ds_rho.sizes["x"]),
            # "y": slice(I_MIN, I_MAX + 1),
            # "x": slice(J_MIN, J_MAX + 1),
        }
    )
    rho_chunk.load()

    ntime = len(time_range)
    nz = len(z_centers)
    ny = rho_chunk.sizes["y"]
    nx = rho_chunk.sizes["x"]

    zstar_all = np.full((ntime, nz, ny, nx), np.nan, dtype=np.float32)

    for jy in tqdm(range(ny), desc="Columns", file=sys.stdout):
        for ix in range(nx):
            rho_profile = rho_chunk.isel(x=ix, y=jy).transpose("time", "z").values
            for out_idx, it in enumerate(time_range):
                lower_ref_time, upper_ref_time, weight = ref_pairs[it]
                rho_ref_lower = ref_arrays[lower_ref_time][:, jy, ix]
                if lower_ref_time == upper_ref_time:
                    rho_ref_col = rho_ref_lower
                else:
                    rho_ref_upper = ref_arrays[upper_ref_time][:, jy, ix]
                    rho_ref_col = (1.0 - weight) * rho_ref_lower + weight * rho_ref_upper

                if np.count_nonzero(np.isfinite(rho_ref_col)) < 2:
                    continue

                zstar_all[out_idx, :, jy, ix] = interp_column_zstar(
                    rho_profile[out_idx, :],
                    rho_ref_col,
                    z_centers,
                )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    spatial_coords = {
        "Z": z_centers,
        "x": rho_chunk["x"].values if "x" in rho_chunk.coords else np.arange(nx),
        "y": rho_chunk["y"].values if "y" in rho_chunk.coords else np.arange(ny),
    }

    for out_idx, it in enumerate(time_range):
        lower_ref_time, upper_ref_time, weight = ref_pairs[it]
        zstar_ds = xr.Dataset(
            data_vars={"z_star": (("Z", "y", "x"), zstar_all[out_idx])},
            coords={
                **spatial_coords,
                "time": it,
                "rho_ref_time_lower": lower_ref_time,
                "rho_ref_time_upper": upper_ref_time,
                "rho_ref_interp_weight": weight,
            },
        )
        zstar_ds.to_netcdf(OUTPUT_ROOT / f"zstar_t_{it:03d}.nc", mode="w")

    del zstar_ds
    del ds_rho
    gc.collect()
    print(f"Saved zstar_t_*.nc files for interval {it1} to {it2 - 1} to {OUTPUT_ROOT}.", flush=True)


if __name__ == "__main__":
    main()
