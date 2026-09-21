import argparse
import gc
import os
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from scipy.signal import butter, filtfilt


TIME_DIM_CANDIDATES = ("time", "Time", "t")
DEPTH_DIM_CANDIDATES = ("z", "Z", "depth", "k", "layer")
Y_DIM_CANDIDATES = ("y", "Y", "YC", "j")
X_DIM_CANDIDATES = ("x", "X", "XC", "i")


# def get_dim_name(data_array: xr.DataArray, candidates: tuple[str, ...], label: str) -> str:
#     for candidate in candidates:
#         if candidate in data_array.dims:
#             return candidate
#     raise ValueError(
#         f"Could not find a {label} dimension in {data_array.dims}. "
#         f"Tried {candidates}."
#     )


def build_highpass_filter() -> tuple[np.ndarray, np.ndarray]:
    fs = 1 / 3600
    # cutoff1 = 1 / (14.5 * 3600)
    # cutoff2 = 1 / (11 * 3600)
    cutoff = 1 / (36.0 * 3600)
    wn = [cutoff / (0.5 * fs)]
    return butter(4, wn, btype="high")


def load_level_timeseries(
    input_zarr: str,
    variable_name: str,
    level_index: int,
    nt: int | None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, tuple[int, ...]]]:
    ds = xr.open_zarr(input_zarr, consolidated=False)
    try:
        if variable_name not in ds:
            raise KeyError(
                f"Variable {variable_name!r} not found in {input_zarr}. "
                f"Available variables: {list(ds.data_vars)}"
            )

        data = ds[variable_name]
        time_dim = "time"
        depth_dim = "z"
        y_dim = "y"
        x_dim = "x"
        input_chunks = data.encoding.get("preferred_chunks") or {}

        if level_index >= data.sizes[depth_dim]:
            raise IndexError(
                f"Depth index {level_index} is outside {depth_dim} size {data.sizes[depth_dim]}."
            )

        time_size = data.sizes[time_dim]
        if nt is None:
            nt = time_size
        elif nt > time_size:
            raise ValueError(
                f"Requested nt={nt}, but {input_zarr} only has {time_size} time steps."
            )

        level_data = data.isel({depth_dim: level_index, time_dim: slice(0, nt)})
        level_data = level_data.transpose(time_dim, y_dim, x_dim)

        array = level_data.load().astype(np.float32).values
        coords = {
            "time": ds[time_dim].isel({time_dim: slice(0, nt)}).values
            if time_dim in ds.coords
            else np.arange(nt, dtype=np.int32),
            "z": ds[depth_dim].values
            if depth_dim in ds.coords
            else np.arange(data.sizes[depth_dim], dtype=np.int32),
            "y": ds[y_dim].values if y_dim in ds.coords else np.arange(array.shape[1], dtype=np.int32),
            "x": ds[x_dim].values if x_dim in ds.coords else np.arange(array.shape[2], dtype=np.int32),
        }
        chunks = {
            "data": (
                min(nt, int(input_chunks.get(time_dim, nt))),
                min(len(coords["z"]), int(input_chunks.get(depth_dim, len(coords["z"])))),
                min(len(coords["y"]), int(input_chunks.get(y_dim, len(coords["y"])))),
                min(len(coords["x"]), int(input_chunks.get(x_dim, len(coords["x"])))),
            ),
            "time": (
                min(len(coords["time"]), int(ds[time_dim].encoding.get("preferred_chunks", {}).get(time_dim, len(coords["time"])))),
            ),
            "z": (
                min(len(coords["z"]), int(ds[depth_dim].encoding.get("preferred_chunks", {}).get(depth_dim, len(coords["z"])))),
            ),
            "y": (
                min(len(coords["y"]), int(ds[y_dim].encoding.get("preferred_chunks", {}).get(y_dim, len(coords["y"])))),
            ),
            "x": (
                min(len(coords["x"]), int(ds[x_dim].encoding.get("preferred_chunks", {}).get(x_dim, len(coords["x"])))),
            ),
        }
    finally:
        ds.close()

    return array, coords, chunks


def initialize_output_zarr(
    output_zarr: str,
    variable_name: str,
    coords: dict[str, np.ndarray],
    chunks: dict[str, tuple[int, ...]],
) -> None:
    if os.path.exists(output_zarr):
        return

    nt = len(coords["time"])
    nz = len(coords["z"])
    ny = len(coords["y"])
    nx = len(coords["x"])

    Path(output_zarr).parent.mkdir(parents=True, exist_ok=True)
    try:
        root = zarr.open_group(output_zarr, mode="w", zarr_version=2)
    except TypeError:
        root = zarr.open_group(output_zarr, mode="w", zarr_format=2)

    root.create_array(
        variable_name,
        shape=(nt, nz, ny, nx),
        chunks=chunks["data"],
        dtype="float32",
        fill_value=None,
        overwrite=True,
    )
    root.create_array(
        "time",
        data=np.asarray(coords["time"]),
        chunks=chunks["time"],
        fill_value=None,
        overwrite=True,
    )
    root.create_array(
        "z",
        data=np.asarray(coords["z"]),
        chunks=chunks["z"],
        fill_value=None,
        overwrite=True,
    )
    root.create_array(
        "y",
        data=np.asarray(coords["y"]),
        chunks=chunks["y"],
        fill_value=None,
        overwrite=True,
    )
    root.create_array(
        "x",
        data=np.asarray(coords["x"]),
        chunks=chunks["x"],
        fill_value=None,
        overwrite=True,
    )

    root[variable_name].attrs["_ARRAY_DIMENSIONS"] = ["time", "z", "y", "x"]
    root["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    root["z"].attrs["_ARRAY_DIMENSIONS"] = ["z"]
    root["y"].attrs["_ARRAY_DIMENSIONS"] = ["y"]
    root["x"].attrs["_ARRAY_DIMENSIONS"] = ["x"]


def write_level_region(
    output_zarr: str,
    variable_name: str,
    level_index: int,
    data: np.ndarray,
    coords: dict[str, np.ndarray],
) -> None:
    nt, ny, nx = data.shape
    ds_out = xr.Dataset(
        data_vars={
            variable_name: (("time", "z", "y", "x"), data[:, np.newaxis, :, :]),
        },
        coords={
            "time": coords["time"],
            "z": np.asarray([coords["z"][level_index]]),
            "y": coords["y"],
            "x": coords["x"],
        },
    )

    ds_out.to_zarr(
        output_zarr,
        mode="r+",
        region={
            "time": slice(0, nt),
            "z": slice(level_index, level_index + 1),
            "y": slice(0, ny),
            "x": slice(0, nx),
        },
        consolidated=False,
    )


def process_level(
    level_index: int,
    input_zarr: str,
    input_var: str,
    output_zarr: str,
    output_var: str,
    nt: int | None,
) -> None:
    print(f"Processing depth level {level_index} from {input_zarr}...", flush=True)
    data, coords, chunks = load_level_timeseries(
        input_zarr=input_zarr,
        variable_name=input_var,
        level_index=level_index,
        nt=nt,
    )

    b, a = build_highpass_filter()
    filtered = filtfilt(b, a, data, axis=0).astype(np.float32, copy=False)
    initialize_output_zarr(
        output_zarr=output_zarr,
        variable_name=output_var,
        coords=coords,
        chunks=chunks,
    )
    write_level_region(
        output_zarr=output_zarr,
        variable_name=output_var,
        level_index=level_index,
        data=filtered,
        coords=coords,
    )


def process_uv_level(level_index: int, data_dir: str, nt: int | None) -> None:
    """Process u_bc and v_bc successively for one depth level to minimise peak memory."""
    for in_var, out_var in (("u_bc", "u_bc_highpass"), ("v_bc", "v_bc_highpass")):
        input_zarr = os.path.join(data_dir, f"{in_var}.zarr")
        output_zarr = os.path.join(data_dir, f"{out_var}.zarr")
        print(f"Highpass filtering {in_var} at depth level {level_index}...", flush=True)
        process_level(
            level_index=level_index,
            input_zarr=input_zarr,
            input_var=in_var,
            output_zarr=output_zarr,
            output_var=out_var,
            nt=nt,
        )
        gc.collect()
    print(f"Done with depth level {level_index}.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Highpass-filter baroclinic velocity (u_bc, v_bc) for one depth level using Zarr I/O."
    )
    parser.add_argument("k", type=int, help="Depth index to process.")
    parser.add_argument(
        "--nt",
        type=int,
        default=None,
        help="Number of time steps to process. Defaults to all time steps in the input Zarr store.",
    )
    parser.add_argument(
        "--data-dir",
        default="/home/ceoas/liux8/work/basin_modes/sulu_sea/post_proc/linear_tests/case_ctrl_bot_intens_Tsorted_2/zarr_output",
        help="Base directory containing u_bc.zarr and v_bc.zarr.",
    )
    args = parser.parse_args()

    process_uv_level(
        level_index=args.k,
        data_dir=args.data_dir,
        nt=args.nt,
    )


if __name__ == "__main__":
    main()