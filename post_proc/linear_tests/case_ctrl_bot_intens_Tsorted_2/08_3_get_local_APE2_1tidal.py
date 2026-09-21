#!/usr/bin/env python3
"""Compute local APE2 fields from rho, z-star, and rho_ref profiles."""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from tqdm import tqdm
# from xmitgcm.utils import read_mds
import xmitgcm


ZARR_DIR = Path("/storage/ceoas-scratch/liux8/basin_modes/sulu_sea/post_proc/linear_tests/case_ctrl_bot_intens_Tsorted_2/zarr_output/")
REF_ROOT = Path("rhoref_zstar_linear_case_ctrl_v4_1tidal")  # Adjust this path as needed
ZSTAR_ROOT = REF_ROOT
CASE_DIR = Path("/home/ceoas/liux8/work/basin_modes/sulu_sea/linear_tests/case_ctrl_bot_intensify_Tsorted_2/")  # Adjust this path as needed
OUTPUT_ZARR = ZARR_DIR / "APE2_1tidal_integ.zarr"

CHUNK_LEN = 13
NT = 600

# I_MIN = 836
# I_MAX = 1579
# J_MIN = 508
# J_MAX = 1077

ds = xmitgcm.open_mdsdataset(f"{CASE_DIR}/", prefix=["outs_sn"], iters=[0])
drf = ds["drF"].values
zf = ds["Zl"].values


def compute_APE2_local(
    rho: np.ndarray,
    z: np.ndarray,
    z_ref: np.ndarray,
    rho_ref: np.ndarray,
    drf: np.ndarray,
    zf: np.ndarray,
    it: int,
    j: int,
    i: int,
    g: float = 9.81,
) -> np.ndarray:
    """Compute local APE2 = g * [rho*(z-z_ref) - int_zref^z rho_ref dz]."""
    rho = np.asarray(rho, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    z_ref = np.asarray(z_ref, dtype=np.float64)
    rho_ref = np.asarray(rho_ref, dtype=np.float64)
    zf = np.asarray(zf, dtype=np.float64)
    drf = np.asarray(drf, dtype=np.float64)

    valid_ref = np.isfinite(z) & np.isfinite(rho_ref)
    # if j==100:
    #     print(f"Valid reference count for it={it}, j={j}, i={i}: {np.count_nonzero(valid_ref)}", flush=True)
    if np.count_nonzero(valid_ref) < 2:
        return np.full_like(rho, np.nan, dtype=np.float32)

    z_valid = z[valid_ref]
    rho_valid = rho[valid_ref]
    z_ref_valid = z_ref[valid_ref]
    rho_ref_valid = rho_ref[valid_ref]
    drf_valid = -drf[valid_ref]

    order = np.argsort(z_valid)
    z_sorted = z_valid[order]
    rho_ref_sorted = rho_ref_valid[order]

    # z_unique, inverse = np.unique(z_sorted, return_inverse=True)
    # if z_unique.size < 2:
    #     return np.full_like(rho, np.nan, dtype=np.float32)

    # rho_ref_unique = np.zeros_like(z_unique, dtype=np.float64)
    # counts = np.zeros_like(z_unique, dtype=np.int64)
    # np.add.at(rho_ref_unique, inverse, rho_ref_sorted)
    # np.add.at(counts, inverse, 1)
    # rho_ref_unique /= counts

    # Keep a face-integral array for diagnostics and consistency with notebook logic.
    # z_faces = zf[0 : len(rho_ref_valid) + 1]
    # F_faces = np.zeros(z_faces.size, dtype=np.float64)
    # F_faces[1:] = np.cumsum(rho_ref_valid * drf_valid)

    # Piecewise-linear integral using all intermediate z-centers between bounds.
    def integrate_rho_between(z0: float, z1: float, z_grid: np.ndarray, rho_grid: np.ndarray) -> float:
        z_low = min(z0, z1)
        z_high = max(z0, z1)
        z_mid = z_grid[(z_grid > z_low) & (z_grid < z_high)]
        z_points = np.concatenate(([z0], z_mid, [z1]))
        rho_points = np.interp(z_points, z_grid, rho_grid)
        # if j==100:
        #     print(f"  Integrating rho between z0={z0} and z1={z1} for it={it}, j={j}, i={i}: z_points={z_points}, rho_points={rho_points}", flush=True)
        return float(np.trapz(rho_points, x=z_points))

    # # F(z) = int(rho_ref dz) from z_unique[0] to z.
    # def F_func(z_query: np.ndarray | float) -> np.ndarray | float:
    #     z_query_arr = np.atleast_1d(np.asarray(z_query, dtype=np.float64))
    #     out = np.empty_like(z_query_arr, dtype=np.float64)
    #     z0 = float(z_unique[0])
    #     for idx, zq in enumerate(z_query_arr):
    #         out[idx] = integrate_rho_between(z0, float(zq), z_unique, rho_ref_unique)
    #     if np.ndim(z_query) == 0:
    #         return float(out[0])
    #     return out
    ape2_t2 = np.zeros_like(rho, dtype=np.float64)
    # if j==100:
    #     print(f"p, zq, zrq for it={it}, j={j}, i={i}: z={z}, z_ref={z_ref}", flush=True)
    for icou, zq, zrq in zip(range(len(z_valid)), z_valid, z_ref_valid):
        # if j==100:
        #     print(f"Computing integral for it={it}, j={j}, i={i}, icou={icou}: zq={zq}, zrq={zrq}, z_sorted: {z_sorted}", flush=True)
        if not (np.isfinite(zrq) and np.isfinite(zq)):
            continue
        if zq < z_sorted[0] or zq > z_sorted[-1]:
            ape2_t2[icou] = np.nan
            continue
        if zrq < z_sorted[0] or zrq > z_sorted[-1]:
            ape2_t2[icou] = np.nan
            continue
        ape2_t2[icou] = integrate_rho_between(zrq, zq, z_sorted, rho_ref_sorted)
        # if j==100:
        #     print(f"  Integral result for it={it}, j={j}, i={i}, icou={icou}: {ape2_t2[icou]}", flush=True)
    # ape2 = g * (rho * (z - z_ref) - (F_func(z) - F_func(z_ref)))
    ape2 = g * (rho * (z - z_ref) - ape2_t2)
    # print(f'{ape2.shape}, {ape2_t2.shape}', flush=True)
    ape2[~(np.isfinite(z) & np.isfinite(rho_ref))] = np.nan
    # if j==100:
    #     print(f"APE2 debug at it={it}, j={j}, i={i}: ape2_t2={ape2_t2}, ape2={ape2}", flush=True)
    return ape2.astype(np.float32)


def init_ape2_store(
    path: Path,
    nt_full: int,
    nz: int,
    ny: int,
    nx: int,
    chunks: tuple[int, int, int, int],
    coords: dict[str, np.ndarray],
) -> None:
    """Create the APE2 zarr store once before region writes."""
    path_str = str(path)
    if os.path.exists(path_str):
        return

    os.makedirs(os.path.dirname(path_str), exist_ok=True)
    lock_path = f"{path_str}.init.lock"
    have_lock = False
    while not have_lock:
        try:
            os.mkdir(lock_path)
            have_lock = True
        except FileExistsError:
            if os.path.exists(path_str):
                return
            time.sleep(5)

    try:
        if os.path.exists(path_str):
            return

        try:
            root = zarr.open_group(path_str, mode="w", zarr_version=2)
        except TypeError:
            root = zarr.open_group(path_str, mode="w", zarr_format=2)

        root.create_array(
            "APE2",
            shape=(nt_full, nz, ny, nx),
            chunks=(
                min(nt_full, chunks[0]),
                min(nz, chunks[1]),
                min(ny, chunks[2]),
                min(nx, chunks[3]),
            ),
            dtype="float32",
            fill_value=None,
            overwrite=True,
        )
        for coord_name in ("time", "Z", "y", "x"):
            coord_values = np.asarray(coords[coord_name])
            root.create_array(
                coord_name,
                data=coord_values,
                chunks=(len(coord_values),),
                fill_value=None,
                overwrite=True,
            )

        root["APE2"].attrs["_ARRAY_DIMENSIONS"] = ["time", "Z", "y", "x"]
        root["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
        root["Z"].attrs["_ARRAY_DIMENSIONS"] = ["Z"]
        root["y"].attrs["_ARRAY_DIMENSIONS"] = ["y"]
        root["x"].attrs["_ARRAY_DIMENSIONS"] = ["x"]
    finally:
        if have_lock:
            os.rmdir(lock_path)


def reference_center_times() -> list[int]:
    centers = []
    for k in range((NT - 1) // CHUNK_LEN + 1):
        it1 = k * CHUNK_LEN
        it2 = min(it1 + CHUNK_LEN, NT)
        centers.append((it1 + it2 - 1) // 2)
    return centers


def reference_pair(it: int, centers: list[int]) -> tuple[int, int, float]:
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
        raise SystemExit("Usage: python 08_3_get_local_APE2_1tidal.py <k>")

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

    zf2 = np.append(zf, z_centers[-1])  # Add bottom face

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
    input_chunks = rho_chunk.encoding.get("preferred_chunks", {})
    data_chunks = (
        int(input_chunks.get("time", 1)),
        int(input_chunks.get("z", nz)),
        int(input_chunks.get("y", min(ny, 128))),
        int(input_chunks.get("x", min(nx, 128))),
    )

    print(f"Loading zstar files for interval {it1} to {it2 - 1}.", flush=True)
    zstar_all = np.full((ntime, nz, ny, nx), np.nan, dtype=np.float32)
    for out_idx, it in enumerate(time_range):
        with xr.open_dataset(ZSTAR_ROOT / f"zstar_t_{it:03d}.nc") as zstar_ds:
            zstar_all[out_idx] = zstar_ds["z_star"].values

    ape2_all = np.full((ntime, nz, ny, nx), np.nan, dtype=np.float32)

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
                # if jy ==100:
                #     print(f"Computing APE2 for it={it}, j={jy}, i={ix}.", flush=True)
                ape2_all[out_idx, :, jy, ix] = compute_APE2_local(
                    rho_profile[out_idx, :],
                    z_centers,
                    zstar_all[out_idx, :, jy, ix],
                    rho_ref_col,
                    drf,
                    zf2,
                    out_idx,
                    jy,
                    ix
                )

    coords_all = {
        "time": ds_rho["time"].values if "time" in ds_rho.coords else np.arange(NT, dtype=np.int32),
        "Z": z_centers,
        "x": rho_chunk["x"].values if "x" in rho_chunk.coords else np.arange(nx),
        "y": rho_chunk["y"].values if "y" in rho_chunk.coords else np.arange(ny),
    }
    init_ape2_store(OUTPUT_ZARR, NT, nz, ny, nx, data_chunks, coords_all)

    ape2_ds = xr.Dataset(
        data_vars={"APE2": (("time", "Z", "y", "x"), ape2_all)},
        coords={
            "time": coords_all["time"][it1:it2],
            "Z": coords_all["Z"],
            "y": coords_all["y"],
            "x": coords_all["x"],
        },
    )
    ape2_ds.to_zarr(
        OUTPUT_ZARR,
        mode="r+",
        region={
            "time": slice(it1, it2),
            "Z": slice(0, nz),
            "y": slice(0, ny),
            "x": slice(0, nx),
        },
        consolidated=False,
    )

    del ape2_ds
    del ds_rho
    gc.collect()
    print(f"Saved APE2 to {OUTPUT_ZARR} for interval {it1} to {it2 - 1}.", flush=True)


if __name__ == "__main__":
    main()
