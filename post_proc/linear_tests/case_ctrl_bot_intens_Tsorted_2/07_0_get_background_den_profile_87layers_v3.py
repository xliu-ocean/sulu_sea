#!/usr/bin/env python3
"""Compute background density profiles for the Sulu Sea region.

This script is a standalone version of the notebook section that loads the
Sulu Sea mask, extracts temperature profiles, converts them to density,
computes the reference background density profile, and writes one NetCDF file
per processed time step.

It is intended to be launched from SLURM array jobs, for example:

    python get_background_den_profile_87layers.py 0

or from a batch script that passes the SLURM array task id.
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import xmitgcm
from matplotlib.path import Path as MplPath
from tqdm import tqdm

rho0 = 1026.5
T0 = 5.0
alpha = 2e-4
g = 9.81

CASE_ID = "033"
CASE_DIR = Path("/home/ceoas/liux8/work/basin_modes/sulu_sea/linear_tests/case_ctrl_bot_intensify_Tsorted_2/")
ZARR_DIR = Path("/storage/ceoas-scratch/liux8/basin_modes/sulu_sea/post_proc/linear_tests/case_ctrl_bot_intens_Tsorted_2/zarr_output/")
OUTPUT_ROOT = Path("rhoref_zstar_linear_case_ctrl_v3")

CHUNK_LEN = 37

def compute_background_PE(rho, vol, mask, V_ref_k, z_centers):
    """
    Compute background potential energy Eb from dataset `ds`.

    Returns Eb (J) and reference profile (rho_ref, z_ref, V_ref).
    """
    # if rho.ndim == 2:
    #     rho_t = xr.DataArray(rho, dims=("Z", "YC"))
    # elif rho.ndim == 3:
    #     rho_t = xr.DataArray(rho, dims=("Z", "YC", "XC"))
    # else:
    #     raise ValueError(f"rho must be 2D or 3D, got shape {rho.shape}")

    dens = rho.ravel()

    vol = vol[mask]
    dens = dens[mask].astype(float)

    order = np.argsort(dens)
    dens_sorted = dens[order]
    vol_sorted = vol[order]

    dens_h = dens_sorted[::-1]
    vol_h = vol_sorted[::-1]

    K = len(V_ref_k)
    rho_ref = np.full(K, np.nan, dtype=float)

    idx = 0
    vol_left = vol_h[idx] if len(vol_h) > 0 else 0.0

    Vref_bottomup = V_ref_k[::-1]

    sumV_domain = 0
    for kk in range(K):
        V_target = Vref_bottomup[kk]
        sum_rhoV = 0.0
        sumV_filled = 0.0
        if V_target <= 0:
            continue

        while V_target - sumV_filled > 1:
            if idx >= len(dens_h):
                print(idx)
                print(kk)
                print(len(dens_h))
                print(V_target, sumV_filled)
                print("take dtype:", type(take))
                print("sum_V_filled dtype:", type(sumV_filled))
                print("V_target dtype:", type(V_target))
                raise RuntimeError(f"Ran out of parcels while filling reference layers.{idx}")

            take = min(vol_left, V_target - sumV_filled)
            sum_rhoV += dens_h[idx] * take

            sumV_filled += take
            sumV_domain += take
            vol_left -= take
            if vol_left <= 1e-16:
                idx += 1
                if idx < len(vol_h):
                    vol_left = vol_h[idx]

        rho_ref[K - 1 - kk] = sum_rhoV / V_target

    # Eb = g * np.nansum(rho_ref * z_centers * V_ref_k)
    return rho_ref


compute_background_pe = compute_background_PE


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python get_background_den_profile_87layers.py <k>")
    task_id = int(sys.argv[1])
    
    # ds_mask_3basins = xr.open_dataset('/home/ceoas/liux8/work/basin_modes/sulu_sea/post_proc/linear_tests/masks_3basins.nc')

    # i_min = 836
    # i_max = 1579
    # j_min = 508
    # j_max = 1077
    # print("Loading grid and mask fields...", flush=True)
    ds_mask = xmitgcm.open_mdsdataset(CASE_DIR, prefix=["outs_sn"],iters=[0])
    mask_deep = ds_mask.maskC.values[0,:,:]
    # mask_deep = np.zeros_like(ds_mask.maskC.values[0,:,:])
    # mask_sea = ds_mask.maskC.values[0, :,:,]
    # depth_sea = ds_mask.Depth.values
    # mask_deep[(mask_sea==1) & (depth_sea>200) & (ds_mask_3basins.mask_sulu.values==1)] = 1
    # i_min, i_max, j_min, j_max, mask_sulu_bbox = build_sulu_bbox(ds_mask)
    # del ds_mask

    # print("Loading case data...", flush=True)
    # ds_sn = xmitgcm.open_mdsdataset(CASE_DIR, prefix=["outs_sn"])

    nt = 600
    if task_id < 0:
        raise ValueError(f"k must be >= 0, got {task_id}.")

    it1 = task_id * CHUNK_LEN
    if it1 >= nt:
        max_k = (nt - 1) // CHUNK_LEN
        raise ValueError(f"k={task_id} is out of range for nt={nt}, chunk={CHUNK_LEN}. Max k is {max_k}.")
    it2 = min(it1 + CHUNK_LEN, nt)

    center_it = (it1 + it2 - 1) // 2

    print(
        f"Processing reference-density interval {it1} to {it2 - 1} of {nt}; "
        f"saving as center time {center_it}.",
        flush=True,
    )

    # rA = ds_mask["rA"].isel(YC=slice(i_min, i_max+1), XC=slice(j_min, j_max+1)) * mask_deep[i_min:i_max+1, j_min:j_max+1]
    rA = ds_mask["rA"]* mask_deep
    dr = ds_mask["drF"] # center
    # hFacC = ds_mask.hFacC.isel(YC=slice(i_min, i_max+1), XC=slice(j_min, j_max+1))
    hFacC = ds_mask.hFacC
    z_centers = ds_mask.Z.values
    drF = dr.values

    case_folder = OUTPUT_ROOT
    case_folder.mkdir(parents=True, exist_ok=True)

    print(f"Loading rho.zarr interval {it1} to {it2 - 1}...", flush=True)
    ds_rho = xr.open_zarr(ZARR_DIR / "rho.zarr", consolidated=False)
    rho_var = "rho" 
    rho_all = ds_rho[rho_var]
    time_dim = "time"
    z_dim = "z"
    y_dim = "y"
    x_dim = "x"
    nz = len(z_centers)
    ny = rho_all.sizes[y_dim]
    nx = rho_all.sizes[x_dim]
    rho_chunk = rho_all.isel(
        {
            time_dim: slice(it1, it2),
            z_dim: slice(0, nz),
            # y_dim: slice(i_min, i_max + 1),
            # x_dim: slice(j_min, j_max + 1),
            y_dim: slice(0, ny),
            x_dim: slice(0, nx),
        }
    )

    ntime = it2 - it1

    rho_chunk.load()  # Load the chunk into memory before processing

    rho_ref_all = np.full((nz, ny, nx), np.nan, dtype=float)

    rA_values = rA.values
    hFacC_values = hFacC.values

    for jy in tqdm(range(ny), desc="Columns", file=sys.stdout):
        for ix in range(nx):
            rA_col = rA_values[jy, ix]
            hFacC_col = hFacC_values[:, jy, ix]
            if rA_col <= 0 or not np.any(hFacC_col > 0):
                continue

            rho_profile_halo = rho_chunk.isel({x_dim: ix, y_dim: jy}).transpose(time_dim, z_dim).values

            rA_profile = np.full((ntime, nz), rA_col, dtype=np.longdouble)
            hFacC_profile = np.tile(hFacC_col, (ntime, 1))
            vol_profile = (
                rA_profile
                * drF.astype(np.longdouble)[None, :]
                * hFacC_profile.astype(np.longdouble)
            ).ravel()
            mask_profile = vol_profile > 0
            if not np.any(mask_profile):
                continue

            V_ref_k = np.zeros(nz, dtype=np.longdouble)
            for k in range(nz):
                V_ref_k[k] = np.sum(
                    rA_profile[:, k]
                    * drF.astype(np.longdouble)[k]
                    * hFacC_profile.astype(np.longdouble)[:, k]
                )

            rho_ref = compute_background_PE(
                rho_profile_halo,
                vol_profile,
                mask_profile,
                V_ref_k,
                z_centers,
            )

            rho_ref_all[:, jy, ix] = rho_ref

    spatial_coords = {
        "Z": z_centers,
        "x": rho_chunk[x_dim].values if x_dim in rho_chunk.coords else np.arange(nx),
        "y": rho_chunk[y_dim].values if y_dim in rho_chunk.coords else np.arange(ny),
    }

    theta_ref_ds = xr.Dataset(
        data_vars={
            "rho_ref": (("Z", "y", "x"), rho_ref_all),
        },
        coords={**spatial_coords, "time": center_it},
    )

    theta_ref_ds.to_netcdf(case_folder / f"rhoref_t_{center_it:03d}.nc", mode="w")

    del theta_ref_ds
    del ds_rho
    gc.collect()
    print(f"Saved interval rho_ref file for center time {center_it} to {case_folder}", flush=True)


if __name__ == "__main__":
    main()
