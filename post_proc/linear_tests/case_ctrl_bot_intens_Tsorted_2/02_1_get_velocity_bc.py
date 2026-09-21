import os
import sys
import gc

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

u_zarr = os.path.join(zarr_dir, "u.zarr")
u_bc_zarr = os.path.join(zarr_dir, "u_bc.zarr")
v_zarr = os.path.join(zarr_dir, "v.zarr")
v_bc_zarr = os.path.join(zarr_dir, "v_bc.zarr")

# Static grid/metric fields from MDS
drF = read_mds(f"{case_dir}/DRF")
# hfacC = read_mds(f"{case_dir}/hFacC")
hfacW = read_mds(f"{case_dir}/hFacW")
hfacS = read_mds(f"{case_dir}/hFacS")
dz_data = drF["DRF"].compute()
# hfacC_data = hfacC["hFacC"].compute()
hfacW_data = hfacW["hFacW"].compute()
hfacS_data = hfacS["hFacS"].compute()

nz = len(drF["DRF"])
ny, nx = hfacW_data.shape[1], hfacW_data.shape[2]

denom_u = np.sum(dz_data * hfacW_data, axis=0)
denom_v = np.sum(dz_data * hfacS_data, axis=0)

def _init_bc_store(path, var_name, nt_full, data_chunks, coords_all):
    if os.path.exists(path):
        return

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


def _process_component(src_zarr, src_var, out_zarr, out_var, hfac_data, denom):
    with xr.open_zarr(src_zarr, consolidated=False) as ds_src:
        src_da = ds_src[src_var]
        src_all = src_da.isel(time=slice(it1, it2)).load().astype(np.float32).values

        nt_full = src_da.sizes["time"]
        input_chunks = src_da.encoding.get("preferred_chunks", {})
        data_chunks = (
            int(input_chunks.get("time", nt_full)),
            int(input_chunks.get("z", nz)),
            int(input_chunks.get("y", ny)),
            int(input_chunks.get("x", nx)),
        )

        coords_all = {
            "time": ds_src["time"].values if "time" in ds_src.coords else np.arange(nt_full, dtype=np.int32),
            "z": ds_src["z"].values if "z" in ds_src.coords else np.arange(nz, dtype=np.int32),
            "y": ds_src["y"].values if "y" in ds_src.coords else np.arange(ny, dtype=np.int32),
            "x": ds_src["x"].values if "x" in ds_src.coords else np.arange(nx, dtype=np.int32),
        }

    bc_all = np.zeros((nk, nz, ny, nx), dtype=np.float32)
    for ii in tqdm(range(nk), file=sys.stdout):
        vel = src_all[ii]
        bt = np.sum(vel * dz_data * hfac_data, axis=0) / denom
        bc_all[ii] = vel - bt

    _init_bc_store(out_zarr, out_var, nt_full, data_chunks, coords_all)

    ds_bc = xr.Dataset(
        data_vars={
            out_var: (("time", "z", "y", "x"), bc_all),
        },
        coords={
            "time": coords_all["time"][it1:it2],
            "z": coords_all["z"],
            "y": coords_all["y"],
            "x": coords_all["x"],
        },
    )

    ds_bc.to_zarr(
        out_zarr,
        mode="r+",
        region={
            "time": slice(it1, it2),
            "z": slice(0, nz),
            "y": slice(0, ny),
            "x": slice(0, nx),
        },
        consolidated=False,
    )

    del src_all
    del bc_all
    gc.collect()


print("Processing u first...", flush=True)
_process_component(u_zarr, "u", u_bc_zarr, "u_bc", hfacW_data, denom_u)
print(f"Saved u_bc to {u_bc_zarr} for time from {it1} to {it2 - 1}.", flush=True)

print("Processing v after u is written...", flush=True)
_process_component(v_zarr, "v", v_bc_zarr, "v_bc", hfacS_data, denom_v)
print(f"Saved v_bc to {v_bc_zarr} for time from {it1} to {it2 - 1}.", flush=True)
