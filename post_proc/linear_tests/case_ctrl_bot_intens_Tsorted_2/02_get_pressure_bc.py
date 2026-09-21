import os
import sys

import numpy as np
import xarray as xr
import zarr
from tqdm import tqdm
from xmitgcm.utils import read_mds

# Process one 50-step block using SLURM/job index k
k = int(sys.argv[1])


nt = 600
print(f"Processing time range from {k * 50} to {k * 50 + 50}/{nt}...", flush=True)
it1 = k * 50
it2 = min((k + 1) * 50, nt)
nk = it2 - it1

g = 9.81

case_dir = "/home/ceoas/liux8/work/basin_modes/sulu_sea/linear_tests/case_ctrl_bot_intensify_Tsorted_2/"
zarr_dir = "/home/ceoas/liux8/work/basin_modes/sulu_sea/post_proc/linear_tests/case_ctrl_bot_intens_Tsorted_2/zarr_output"

phihyd_zarr = os.path.join(zarr_dir, "phihyd.zarr")
p_rho_bc_zarr = os.path.join(zarr_dir, "p_rho_bc.zarr")

# Static grid/metric fields from MDS
drF = read_mds(f"{case_dir}/DRF")
hfacC = read_mds(f"{case_dir}/hFacC")
dz_data = drF["DRF"].compute()
hfacC_data = hfacC["hFacC"].compute()

nz = len(drF["DRF"])
ny, nx = hfacC_data.shape[1], hfacC_data.shape[2]

# Load PHIHYD from zarr for this block and keep chunk/coord metadata
with xr.open_zarr(phihyd_zarr, consolidated=False) as ds_phi:
    phihyd_da = ds_phi["phihyd"]
    phihyd_all = phihyd_da.isel(time=slice(it1, it2)).load().astype(np.float32)

    nt_full = phihyd_da.sizes["time"]
    input_chunks = phihyd_da.encoding.get("preferred_chunks", {})
    data_chunks = (
        int(input_chunks.get("time", nt_full)),
        int(input_chunks.get("z", nz)),
        int(input_chunks.get("y", ny)),
        int(input_chunks.get("x", nx)),
    )

    coords_all = {
        "time": ds_phi["time"].values if "time" in ds_phi.coords else np.arange(nt_full, dtype=np.int32),
        "z": ds_phi["z"].values if "z" in ds_phi.coords else np.arange(nz, dtype=np.int32),
        "y": ds_phi["y"].values if "y" in ds_phi.coords else np.arange(ny, dtype=np.int32),
        "x": ds_phi["x"].values if "x" in ds_phi.coords else np.arange(nx, dtype=np.int32),
    }
p_rho_bc_all = np.zeros((nk, nz, ny, nx), dtype=np.float32)
p_rho_bt_all = np.zeros((nk, ny, nx), dtype=np.float32)
# p_bt_all = np.zeros((nk, ny, nx), dtype=np.float32)
eta_all = np.zeros((nk, ny, nx), dtype=np.float32)

# weight_3d = dz_data[:, np.newaxis, np.newaxis] * hfacC_data
# denom_2d = np.sum(weight_3d, axis=0)

# ETAN still comes from MDS
# for it in tqdm(range(it1, it2), file=sys.stdout):
print(f"Loading ETAN from MDS for time range {it1} to {it2}...", flush=True)
for it in range(it1, it2):
    data_2d = read_mds(f"{case_dir}/outs_sn_etan.{it * 180:010d}")
    eta_all[it - it1] = data_2d["ETAN"].astype(np.float32)

for ii in tqdm(range(nk), file=sys.stdout):
    phihyd = phihyd_all[ii]
    p_rho = eta_all[ii] * g + phihyd

    p_rho_bt_all[ii] = np.sum(phihyd * dz_data*hfacC_data, axis=0) / np.sum(dz_data * hfacC_data, axis=0)
    p_rho_bc_all[ii] = phihyd - p_rho_bt_all[ii]
    # del phihyd, p_rho
    # p_bt_all[ii] = np.sum(p_rho * weight_3d, axis=0) / denom_2d

# Initialize output zarr once, then write by time region
if not os.path.exists(p_rho_bc_zarr):
    try:
        root = zarr.open_group(p_rho_bc_zarr, mode="w", zarr_version=2)
    except TypeError:
        root = zarr.open_group(p_rho_bc_zarr, mode="w", zarr_format=2)

    root.create_array(
        "p_rho_bc",
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

    root["p_rho_bc"].attrs["_ARRAY_DIMENSIONS"] = ["time", "z", "y", "x"]
    root["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    root["z"].attrs["_ARRAY_DIMENSIONS"] = ["z"]
    root["y"].attrs["_ARRAY_DIMENSIONS"] = ["y"]
    root["x"].attrs["_ARRAY_DIMENSIONS"] = ["x"]

ds_bc = xr.Dataset(
    data_vars={
        "p_rho_bc": (("time", "z", "y", "x"), p_rho_bc_all),
    },
    coords={
        "time": coords_all["time"][it1:it2],
        "z": coords_all["z"],
        "y": coords_all["y"],
        "x": coords_all["x"],
    },
)

ds_bc.to_zarr(
    p_rho_bc_zarr,
    mode="r+",
    region={
        "time": slice(it1, it2),
        "z": slice(0, nz),
        "y": slice(0, ny),
        "x": slice(0, nx),
    },
    consolidated=False,
)

# time = np.arange(it1, it2)

# ds_bt = xr.Dataset(
#     data_vars={
#         "p_rho_bt_all": (("time", "y", "x"), p_rho_bt_all),
#         "p_bt_all": (("time", "y", "x"), p_bt_all),
#     },
#     coords={
#         "time": time,
#         "y": np.arange(ny),
#         "x": np.arange(nx),
#     },
# )

# ds_bt.to_netcdf(os.path.join(npy_dir, f"PRESSURE_bt_time{it1:03d}_{it2 - 1:03d}.nc"))

print(f"Saved p_rho_bc to {p_rho_bc_zarr} for time from {it1} to {it2 - 1}.", flush=True)
