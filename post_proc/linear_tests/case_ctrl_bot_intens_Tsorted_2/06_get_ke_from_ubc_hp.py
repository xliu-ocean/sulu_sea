import os
import sys
import gc

import numpy as np
import xarray as xr
import zarr
from tqdm import tqdm


# Process one 40-step block using SLURM/job index k
k = int(sys.argv[1])

nt = 600
chunk_len = 40
print(f"Processing time range from {k * chunk_len} to {k * chunk_len + chunk_len}/{nt}...", flush=True)
it1 = k * chunk_len
it2 = min((k + 1) * chunk_len, nt)
nk = it2 - it1

zarr_dir = "/home/ceoas/liux8/work/basin_modes/sulu_sea/post_proc/linear_tests/case_ctrl_bot_intens_Tsorted_2/zarr_output"

u_bc_zarr = os.path.join(zarr_dir, "u_bc_highpass.zarr")
v_bc_zarr = os.path.join(zarr_dir, "v_bc_highpass.zarr")
ke_zarr = os.path.join(zarr_dir, "ke_bc_highpass.zarr")


def _init_ke_store(path, var_name, nt_full, nz, ny, nx, data_chunks, coords_all):
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


with xr.open_zarr(u_bc_zarr, consolidated=False) as ds_u, xr.open_zarr(v_bc_zarr, consolidated=False) as ds_v:
    u_da = ds_u["u_bc_highpass"]
    v_da = ds_v["v_bc_highpass"]

    nt_full = u_da.sizes["time"]
    nz = u_da.sizes["z"]
    ny = u_da.sizes["y"]
    nx = u_da.sizes["x"]

    input_chunks = u_da.encoding.get("preferred_chunks", {})
    data_chunks = (
        int(input_chunks.get("time", nt_full)),
        int(input_chunks.get("z", nz)),
        int(input_chunks.get("y", ny)),
        int(input_chunks.get("x", nx)),
    )

    coords_all = {
        "time": ds_u["time"].values if "time" in ds_u.coords else np.arange(nt_full, dtype=np.int32),
        "z": ds_u["z"].values if "z" in ds_u.coords else np.arange(nz, dtype=np.int32),
        "y": ds_u["y"].values if "y" in ds_u.coords else np.arange(ny, dtype=np.int32),
        "x": ds_u["x"].values if "x" in ds_u.coords else np.arange(nx, dtype=np.int32),
    }

    u_all = u_da.isel(time=slice(it1, it2)).load().astype(np.float32).values
    v_all = v_da.isel(time=slice(it1, it2)).load().astype(np.float32).values

ke_all = np.zeros((nk, nz, ny, nx), dtype=np.float32)
for ii in tqdm(range(nk), file=sys.stdout):
    ke_all[ii] = 0.5 * (u_all[ii] ** 2 + v_all[ii] ** 2)

_init_ke_store(ke_zarr, "ke_bc_highpass", nt_full, nz, ny, nx, data_chunks, coords_all)

ds_ke = xr.Dataset(
    data_vars={
        "ke_bc_highpass": (("time", "z", "y", "x"), ke_all),
    },
    coords={
        "time": coords_all["time"][it1:it2],
        "z": coords_all["z"],
        "y": coords_all["y"],
        "x": coords_all["x"],
    },
)

ds_ke.to_zarr(
    ke_zarr,
    mode="r+",
    region={
        "time": slice(it1, it2),
        "z": slice(0, nz),
        "y": slice(0, ny),
        "x": slice(0, nx),
    },
    consolidated=False,
)

del u_all
del v_all
del ke_all
gc.collect()

print(f"Saved ke_bc_highpass to {ke_zarr} for time from {it1} to {it2 - 1}.", flush=True)
