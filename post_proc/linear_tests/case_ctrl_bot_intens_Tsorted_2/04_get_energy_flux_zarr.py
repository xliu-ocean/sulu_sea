import gc
import os
import sys

import numpy as np
import xarray as xr
import zarr


# Process one 50-step block using SLURM/job index k
k = int(sys.argv[1])

nt = 600
print(f"Processing time range from {k * 50} to {k * 50 + 50}/{nt}...", flush=True)
it1 = k * 50
it2 = min((k + 1) * 50, nt)
nk = it2 - it1

if nk <= 0:
    raise ValueError(f"Invalid time chunk for k={k}: it1={it1}, it2={it2}")

zarr_dir = "/home/ceoas/liux8/work/basin_modes/sulu_sea/post_proc/linear_tests/case_ctrl_bot_intens_Tsorted_2/zarr_output"
zarr_out_dir = zarr_dir

p_zarr = os.path.join(zarr_dir, "p_rho_bc_highpass.zarr")
u_zarr = os.path.join(zarr_dir, "u_bc_highpass.zarr")
v_zarr = os.path.join(zarr_dir, "v_bc_highpass.zarr")
fx_zarr = os.path.join(zarr_out_dir, "Fx.zarr")
fy_zarr = os.path.join(zarr_out_dir, "Fy.zarr")


def _get_preferred_chunks(da: xr.DataArray, nt_full: int, nz: int, ny: int, nx: int) -> tuple[int, int, int, int]:
    preferred_chunks = da.encoding.get("preferred_chunks") or {}
    return (
        int(preferred_chunks.get("time", nt_full)),
        int(preferred_chunks.get("z", nz)),
        int(preferred_chunks.get("y", ny)),
        int(preferred_chunks.get("x", nx)),
    )


def _init_output_store(path: str, var_name: str, nt_full: int, nz: int, ny: int, nx: int, chunks: tuple[int, int, int, int], coords: dict[str, np.ndarray]) -> None:
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
            min(nt_full, chunks[0]),
            min(nz, chunks[1]),
            min(ny, chunks[2]),
            min(nx, chunks[3]),
        ),
        dtype="float32",
        fill_value=None,
        overwrite=True,
    )
    root.create_array(
        "time",
        data=np.asarray(coords["time"]),
        chunks=(len(coords["time"]),),
        fill_value=None,
        overwrite=True,
    )
    root.create_array(
        "z",
        data=np.asarray(coords["z"]),
        chunks=(len(coords["z"]),),
        fill_value=None,
        overwrite=True,
    )
    root.create_array(
        "y",
        data=np.asarray(coords["y"]),
        chunks=(len(coords["y"]),),
        fill_value=None,
        overwrite=True,
    )
    root.create_array(
        "x",
        data=np.asarray(coords["x"]),
        chunks=(len(coords["x"]),),
        fill_value=None,
        overwrite=True,
    )

    root[var_name].attrs["_ARRAY_DIMENSIONS"] = ["time", "z", "y", "x"]
    root["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    root["z"].attrs["_ARRAY_DIMENSIONS"] = ["z"]
    root["y"].attrs["_ARRAY_DIMENSIONS"] = ["y"]
    root["x"].attrs["_ARRAY_DIMENSIONS"] = ["x"]


with xr.open_zarr(p_zarr, consolidated=False) as ds_p:
    p_da = ds_p["p_rho_bc_highpass"]
    p_all = p_da.isel(time=slice(it1, it2)).load().astype(np.float32).values

    nt_full = p_da.sizes["time"]
    nz = p_da.sizes["z"]
    ny = p_da.sizes["y"]
    nx = p_da.sizes["x"]
    data_chunks = _get_preferred_chunks(p_da, nt_full, nz, ny, nx)
    coords_all = {
        "time": ds_p["time"].values if "time" in ds_p.coords else np.arange(nt_full, dtype=np.int32),
        "z": ds_p["z"].values if "z" in ds_p.coords else np.arange(nz, dtype=np.int32),
        "y": ds_p["y"].values if "y" in ds_p.coords else np.arange(ny, dtype=np.int32),
        "x": ds_p["x"].values if "x" in ds_p.coords else np.arange(nx, dtype=np.int32),
    }


def _process_component(input_zarr: str, input_var: str, output_zarr: str, output_var: str) -> None:
    with xr.open_zarr(input_zarr, consolidated=False) as ds_in:
        vel_da = ds_in[input_var]
        vel_all = vel_da.isel(time=slice(it1, it2)).load().astype(np.float32).values

    if vel_all.shape != p_all.shape:
        raise ValueError(
            f"Shape mismatch for {input_var}: {vel_all.shape} vs pressure {p_all.shape}"
        )

    flux_all = (vel_all * p_all).astype(np.float32, copy=False)

    _init_output_store(output_zarr, output_var, nt_full, nz, ny, nx, data_chunks, coords_all)

    ds_flux = xr.Dataset(
        data_vars={
            output_var: (("time", "z", "y", "x"), flux_all),
        },
        coords={
            "time": coords_all["time"][it1:it2],
            "z": coords_all["z"],
            "y": coords_all["y"],
            "x": coords_all["x"],
        },
    )

    ds_flux.to_zarr(
        output_zarr,
        mode="r+",
        region={
            "time": slice(it1, it2),
            "z": slice(0, nz),
            "y": slice(0, ny),
            "x": slice(0, nx),
        },
        consolidated=False,
    )

    del vel_all
    del flux_all
    gc.collect()


print("Processing Fx from u_bc_highpass and p_rho_bc_highpass...", flush=True)
_process_component(u_zarr, "u_bc_highpass", fx_zarr, "Fx")
print("Processing Fy from v_bc_highpass and p_rho_bc_highpass...", flush=True)
_process_component(v_zarr, "v_bc_highpass", fy_zarr, "Fy")

print(f"Saved Fx to {fx_zarr} and Fy to {fy_zarr} for time from {it1} to {it2 - 1}.", flush=True)