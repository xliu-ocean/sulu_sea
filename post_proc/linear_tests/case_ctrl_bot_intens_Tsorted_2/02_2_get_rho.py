import os
import sys

import numpy as np
import xarray as xr
import zarr
from xmitgcm.utils import read_mds

# Process one 50-step block using SLURM/job index k
k = int(sys.argv[1])

nt = 600
block = 50
it1 = k * block
it2 = min((k + 1) * block, nt)
nk = it2 - it1

print(f"Processing rho for time range {it1} to {it2 - 1}/{nt - 1}...", flush=True)

# Linear equation of state constants
rho0 = 1026.5
T0 = 5.0
alpha = 2e-4

case_dir = "/home/ceoas/liux8/work/basin_modes/sulu_sea/linear_tests/case_ctrl_bot_intensify_Tsorted_2/"
zarr_dir = "/storage/ceoas-scratch/liux8/basin_modes/sulu_sea/post_proc/linear_tests/case_ctrl_bot_intens_Tsorted_2/zarr_output"
theta_zarr = os.path.join(zarr_dir, "theta.zarr")
rho_zarr = os.path.join(zarr_dir, "rho.zarr")

# Get grid size from static fields
hfacC = read_mds(f"{case_dir}/hFacC")
hfacC_data = hfacC["hFacC"].compute()
nz, ny, nx = hfacC_data.shape

nt_full = nt
# Keep chunks comfortably below codec buffer limits.
data_chunks = (1, min(nz, 10), min(ny, 128), min(nx, 128))
coords_all = {
    "time": np.arange(nt_full, dtype=np.int32),
    "z": np.arange(nz, dtype=np.int32),
    "y": np.arange(ny, dtype=np.int32),
    "x": np.arange(nx, dtype=np.int32),
}


def _init_store(path, var_name, nt_full, data_chunks, coords_all):
    if os.path.exists(path):
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        root = zarr.open_group(path, mode="w", zarr_version=2)
    except TypeError:
        root = zarr.open_group(path, mode="w", zarr_format=2)

    root.create_array(
        var_name,
        shape=(nt_full, nz, ny, nx),
        chunks=(
            min(nt_full, data_chunks[0]),
            min(nz, data_chunks[1]),
            min(ny, data_chunks[2]),
            min(nx, data_chunks[3]),
        ),
        dtype="float32",
        fill_value=None,
        overwrite=True,
    )
    root.create_array(
        "time",
        data=np.asarray(coords_all["time"]),
        chunks=(len(coords_all["time"]),),
        fill_value=None,
        overwrite=True,
    )
    root.create_array(
        "z",
        data=np.asarray(coords_all["z"]),
        chunks=(len(coords_all["z"]),),
        fill_value=None,
        overwrite=True,
    )
    root.create_array(
        "y",
        data=np.asarray(coords_all["y"]),
        chunks=(len(coords_all["y"]),),
        fill_value=None,
        overwrite=True,
    )
    root.create_array(
        "x",
        data=np.asarray(coords_all["x"]),
        chunks=(len(coords_all["x"]),),
        fill_value=None,
        overwrite=True,
    )

    root[var_name].attrs["_ARRAY_DIMENSIONS"] = ["time", "z", "y", "x"]
    root["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    root["z"].attrs["_ARRAY_DIMENSIONS"] = ["z"]
    root["y"].attrs["_ARRAY_DIMENSIONS"] = ["y"]
    root["x"].attrs["_ARRAY_DIMENSIONS"] = ["x"]

_init_store(theta_zarr, "theta", nt_full, data_chunks, coords_all)
_init_store(rho_zarr, "rho", nt_full, data_chunks, coords_all)

theta_all = np.empty((nk, nz, ny, nx), dtype=np.float32)

print(f"Loading THETA from MDS for time range {it1} to {it2 - 1}...", flush=True)
for ii, it in enumerate(range(it1, it2)):
    iternum = it * 180
    ds = read_mds(f"{case_dir}/outs_sn.{iternum:010d}")
    theta_all[ii] = ds["THETA"].compute().astype(np.float32)

ds_theta = xr.Dataset(
    data_vars={
        "theta": (("time", "z", "y", "x"), theta_all),
    },
    coords={
        "time": coords_all["time"][it1:it2],
        "z": coords_all["z"],
        "y": coords_all["y"],
        "x": coords_all["x"],
    },
)

ds_theta.to_zarr(
    theta_zarr,
    mode="r+",
    region={
        "time": slice(it1, it2),
        "z": slice(0, nz),
        "y": slice(0, ny),
        "x": slice(0, nx),
    },
    consolidated=False,
)
del theta_all, ds_theta

print(f"Saved theta to {theta_zarr} for time from {it1} to {it2 - 1}.", flush=True)
# Load theta from zarr for this block and keep chunk/coord metadata
theta_zarr = os.path.join(zarr_dir, "theta.zarr")
with xr.open_zarr(theta_zarr, consolidated=False) as ds_theta:
    theta_da = ds_theta["theta"]
    theta_all = theta_da.isel(time=slice(it1, it2)).load().astype(np.float32)

    nt_full = theta_da.sizes["time"]
    input_chunks = theta_da.encoding.get("preferred_chunks", {})
    data_chunks = (
        int(input_chunks.get("time", nt_full)),
        int(input_chunks.get("z", nz)),
        int(input_chunks.get("y", ny)),
        int(input_chunks.get("x", nx)),
    )

    coords_all = {
        "time": ds_theta["time"].values if "time" in ds_theta.coords else np.arange(nt_full, dtype=np.int32),
        "z": ds_theta["z"].values if "z" in ds_theta.coords else np.arange(nz, dtype=np.int32),
        "y": ds_theta["y"].values if "y" in ds_theta.coords else np.arange(ny, dtype=np.int32),
        "x": ds_theta["x"].values if "x" in ds_theta.coords else np.arange(nx, dtype=np.int32),
    }
# rho_all = rho0 * (1.0 - alpha * (theta_all.values - T0))
rho_all = rho0 * ( - alpha * (theta_all.values - T0)) ### store the anomaly instead of the full density to save space

ds_rho = xr.Dataset(
    data_vars={
        "rho": (("time", "z", "y", "x"), rho_all),
    },
    coords={
        "time": coords_all["time"][it1:it2],
        "z": coords_all["z"],
        "y": coords_all["y"],
        "x": coords_all["x"],
    },
)

ds_rho.to_zarr(
    rho_zarr,
    mode="r+",
    region={
        "time": slice(it1, it2),
        "z": slice(0, nz),
        "y": slice(0, ny),
        "x": slice(0, nx),
    },
    consolidated=False,
)

print(f"Saved rho to {rho_zarr} for time from {it1} to {it2 - 1}.", flush=True)
