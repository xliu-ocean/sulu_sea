import os
import shutil
import numpy as np
import zarr
from numcodecs import Blosc

out_dir = "/storage/ceoas-scratch/liux8/basin_modes/sulu_sea/post_proc/linear_tests/case_ctrl_bot_intens_Tsorted_2/zarr_output"

nt = 600
nz = 87
nx = 1520
ny = 1908

shape = (nt, nz, ny, nx)
chunk_shape = (20, 10, 128, 128)
dtype = "float32"

variables = {
    "u": "u.zarr",
    "v": "v.zarr",
    "phihyd": "phihyd.zarr",
    "theta": "theta.zarr",
}

compressor = Blosc(
    cname="zstd",
    clevel=3,
    shuffle=Blosc.BITSHUFFLE,
)

os.makedirs(out_dir, exist_ok=True)

for varname, zarr_name in variables.items():
    store_path = os.path.join(out_dir, zarr_name)

    if os.path.exists(store_path):
        shutil.rmtree(store_path)

    print(f"Initializing {store_path}")

    try:
        root = zarr.open_group(
            store_path,
            mode="w",
            zarr_version=2,
        )
    except TypeError:
        root = zarr.open_group(
            store_path,
            mode="w",
            zarr_format=2,
        )

    # Main data variable: use shape, no data
    root.create_array(
        varname,
        shape=shape,
        chunks=chunk_shape,
        dtype=dtype,
        fill_value=None,
        compressor=compressor,
        overwrite=True,
    )

    # Coordinate variables: use data, no shape
    root.create_array(
        "time",
        data=np.arange(nt, dtype="int32"),
        chunks=(nt,),
        fill_value=None,
        overwrite=True,
    )

    root.create_array(
        "z",
        data=np.arange(nz, dtype="int32"),
        chunks=(nz,),
        fill_value=None,
        overwrite=True,
    )

    root.create_array(
        "x",
        data=np.arange(nx, dtype="int32"),
        chunks=(nx,),
        fill_value=None,
        overwrite=True,
    )

    root.create_array(
        "y",
        data=np.arange(ny, dtype="int32"),
        chunks=(ny,),
        fill_value=None,
        overwrite=True,
    )

    root[varname].attrs["_ARRAY_DIMENSIONS"] = ["time", "z", "y", "x"]
    root["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    root["z"].attrs["_ARRAY_DIMENSIONS"] = ["z"]
    root["y"].attrs["_ARRAY_DIMENSIONS"] = ["y"]
    root["x"].attrs["_ARRAY_DIMENSIONS"] = ["x"]

print("Done.")