"""
One-time initialization of the output Zarr stores for highpass-filtered velocity.
Run this ONCE before submitting the 03_1_run_highpass_velocity_zarr.sh array job.
It creates empty u_bc_highpass.zarr and v_bc_highpass.zarr with the
correct shape, chunks, coordinates, and dimension attributes so that array tasks
can write in parallel without any race on store creation.
"""
import os
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

DATA_DIR = "/home/ceoas/liux8/work/basin_modes/sulu_sea/post_proc/linear_tests/case_ctrl_bot_intens_Tsorted_2/zarr_output"

PAIRS = [
    ("u_bc", "u_bc_highpass_pl66"),
    ("v_bc", "v_bc_highpass_pl66"),
    ("p_rho_bc", "p_rho_bc_highpass_pl66")
]


def init_store(src_zarr: str, src_var: str, out_zarr: str, out_var: str) -> None:
    if os.path.exists(out_zarr):
        print(f"Already exists, skipping: {out_zarr}")
        return

    with xr.open_zarr(src_zarr, consolidated=False) as ds:
        da = ds[src_var]
        nt = da.sizes["time"]
        nz = da.sizes["z"]
        ny = da.sizes["y"]
        nx = da.sizes["x"]

        input_chunks = da.encoding.get("preferred_chunks") or {}
        data_chunks = (
            int(input_chunks.get("time", nt)),
            int(input_chunks.get("z", nz)),
            int(input_chunks.get("y", ny)),
            int(input_chunks.get("x", nx)),
        )

        coords = {
            "time": ds["time"].values if "time" in ds.coords else np.arange(nt, dtype=np.int32),
            "z":    ds["z"].values    if "z"    in ds.coords else np.arange(nz, dtype=np.int32),
            "y":    ds["y"].values    if "y"    in ds.coords else np.arange(ny, dtype=np.int32),
            "x":    ds["x"].values    if "x"    in ds.coords else np.arange(nx, dtype=np.int32),
        }

    Path(out_zarr).parent.mkdir(parents=True, exist_ok=True)
    try:
        root = zarr.open_group(out_zarr, mode="w", zarr_version=2)
    except TypeError:
        root = zarr.open_group(out_zarr, mode="w", zarr_format=2)

    root.create_array(
        out_var,
        shape=(nt, nz, ny, nx),
        chunks=(
            min(nt, data_chunks[0]),
            min(nz, data_chunks[1]),
            min(ny, data_chunks[2]),
            min(nx, data_chunks[3]),
        ),
        dtype="float32",
        fill_value=0.0,
        overwrite=True,
    )
    for name, data in coords.items():
        root.create_array(
            name,
            data=np.asarray(data),
            chunks=(len(data),),
            fill_value=None,
            overwrite=True,
        )
        root[name].attrs["_ARRAY_DIMENSIONS"] = [name]

    root[out_var].attrs["_ARRAY_DIMENSIONS"] = ["time", "z", "y", "x"]

    print(f"Initialized {out_zarr}  shape=({nt},{nz},{ny},{nx})  chunks={data_chunks}")


if __name__ == "__main__":
    for src_var, out_var in PAIRS:
        src_zarr = os.path.join(DATA_DIR, f"{src_var}.zarr")
        out_zarr = os.path.join(DATA_DIR, f"{out_var}.zarr")
        init_store(src_zarr, src_var, out_zarr, out_var)
    print("Done. You can now submit 03_1_run_highpass_velocity_zarr.sh safely.")
