import shutil
import zarr
from numcodecs import Blosc

import numpy as np
import xarray as xr
import os
from xmitgcm.utils import read_mds

case_name = 'linear_tests/case_ctrl_bot_intensify_Tsorted_2'

drF = read_mds(f'/home/ceoas/liux8/work/basin_modes/sulu_sea/{case_name}/DRF')
hFacW = read_mds(f'/home/ceoas/liux8/work/basin_modes/sulu_sea/{case_name}/hFacW')
hFacS = read_mds(f'/home/ceoas/liux8/work/basin_modes/sulu_sea/{case_name}/hFacS')
XC = read_mds(f'/home/ceoas/liux8/work/basin_modes/sulu_sea/{case_name}/XC')
YC = read_mds(f'/home/ceoas/liux8/work/basin_modes/sulu_sea/{case_name}/YC')

lon = XC['XC'].compute()
lat = YC['YC'].compute()

nz = len(drF['DRF'])

Fx = xr.open_zarr('zarr_output/Fx.zarr/', consolidated=False)
# Fy = xr.open_zarr('zarr_output/Fy.zarr/', consolidated=False)

fx_ver_int = np.sum(Fx["Fx"] * drF["DRF"].compute() * hFacW["hFacW"].compute(), axis=1) / np.sum(
    drF["DRF"].compute() * hFacW["hFacW"].compute(), axis=0
)
# fy_ver_int = np.sum(Fy["Fy"] * drF["DRF"].compute() * hFacS["hFacS"].compute(), axis=1) / np.sum(
    # drF["DRF"].compute() * hFacS["hFacS"].compute(), axis=0
# )

zarr_dir_2 = "zarr_output"
os.makedirs(zarr_dir_2, exist_ok=True)

time_vals = np.asarray(Fx["time"].values)
y_vals = np.asarray(Fx["y"].values)
x_vals = np.asarray(Fx["x"].values)

nt = time_vals.shape[0]
ny = y_vals.shape[0]
nx = x_vals.shape[0]
chunk_shape = (20, 128, 128)
dtype = "float32"

compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

def write_ver_int(store_path, varname, data_array):
    if os.path.exists(store_path):
        shutil.rmtree(store_path)

    try:
        root = zarr.open_group(store_path, mode="w", zarr_version=2)
    except TypeError:
        root = zarr.open_group(store_path, mode="w", zarr_format=2)

    root.create_array(
        varname,
        shape=(nt, ny, nx),
        chunks=chunk_shape,
        dtype=dtype,
        fill_value=None,
        compressor=compressor,
        overwrite=True,
    )

    root.create_array("time", data=time_vals, chunks=(nt,), fill_value=None, overwrite=True)
    root.create_array("y", data=y_vals, chunks=(ny,), fill_value=None, overwrite=True)
    root.create_array("x", data=x_vals, chunks=(nx,), fill_value=None, overwrite=True)

    root[varname][:] = np.asarray(data_array.values, dtype=dtype)

    root[varname].attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x"]
    root["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    root["y"].attrs["_ARRAY_DIMENSIONS"] = ["y"]
    root["x"].attrs["_ARRAY_DIMENSIONS"] = ["x"]

write_ver_int(os.path.join(zarr_dir_2, "fx_ver_int_bp.zarr"), "fx_ver_int", fx_ver_int)
# write_ver_int(os.path.join(zarr_dir_2, "fy_ver_int_bp.zarr"), "fy_ver_int", fy_ver_int)

print("Wrote zarr_output/fx_ver_int.zarr and zarr_output/fy_ver_int.zarr")