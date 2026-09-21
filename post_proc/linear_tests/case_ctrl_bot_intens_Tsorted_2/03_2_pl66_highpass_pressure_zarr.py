import argparse
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from scipy import signal


TIME_DIM_CANDIDATES = ("time", "Time", "t")
DEPTH_DIM_CANDIDATES = ("z", "Z", "depth", "k", "layer")
Y_DIM_CANDIDATES = ("y", "Y", "YC", "j")
X_DIM_CANDIDATES = ("x", "X", "XC", "i")


@lru_cache(maxsize=None)
def build_pl66_lowpass_filter(dt_hours: float = 1.0, T_hours: float = 33.0) -> np.ndarray:
    """Return the PL66 low-pass filter weights from MATLAB pl66tn.m."""
    cutoff = T_hours / dt_hours
    fq = 1.0 / cutoff
    nw = int(round(2.0 * T_hours / dt_hours))
    if nw < 1:
        raise ValueError(f"Invalid PL66 filter width: nw={nw} for dt_hours={dt_hours}, T_hours={T_hours}")

    j = np.arange(1, nw + 1, dtype=np.float64)
    t = np.pi * j
    den = (fq * fq) * (t ** 3)
    wts = (2.0 * np.sin(2.0 * fq * t) - np.sin(fq * t) - np.sin(3.0 * fq * t)) / den
    wts = np.concatenate((wts[::-1], np.array([2.0 * fq]), wts))
    wts = wts / np.sum(wts)
    return wts


def _pl66_lowpass_valid_masked(
    x_valid: np.ndarray,
    dt_hours: float = 1.0,
    T_hours: float = 33.0,
) -> np.ndarray:
    """Apply PL66 low-pass to a 1D array with no NaNs/Infs in x_valid."""
    wts = build_pl66_lowpass_filter(dt_hours=dt_hours, T_hours=T_hours)
    nw = (len(wts) - 1) // 2
    nw2 = 2 * nw
    npts = x_valid.size
    if npts <= nw2:
        return np.full_like(x_valid, np.nan, dtype=np.float64)

    xdt = signal.detrend(x_valid, type="linear")
    trnd = x_valid - xdt
    cs = np.cos(np.pi * np.arange(1, nw + 1, dtype=np.float64) / nw2)
    y = np.concatenate(
        (
            cs[::-1] * xdt[:nw][::-1],
            xdt,
            cs * xdt[-nw:][::-1],
        )
    )
    yf = signal.lfilter(wts, 1.0, y)
    filtered = yf[nw2 : npts + nw2] + trnd
    return filtered


def pl66_lowpass_1d(x: np.ndarray, dt_hours: float = 1.0, T_hours: float = 33.0) -> np.ndarray:
    """Apply the MATLAB PL66 low-pass filter to a 1D time series, skipping non-finite points."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D input, got shape {x.shape}.")

    out = np.full_like(x, np.nan, dtype=np.float64)
    valid = np.isfinite(x)
    if not np.any(valid):
        return out

    x_valid = x[valid]
    lowpass_valid = _pl66_lowpass_valid_masked(x_valid, dt_hours=dt_hours, T_hours=T_hours)
    out[valid] = lowpass_valid
    return out


def pl66_highpass_1d(x: np.ndarray, dt_hours: float = 1.0, T_hours: float = 33.0) -> np.ndarray:
    """Return the high-pass series as x minus the PL66 low-pass series."""
    x = np.asarray(x, dtype=np.float64)
    lowpass = pl66_lowpass_1d(x, dt_hours=dt_hours, T_hours=T_hours)
    highpass = np.where(np.isfinite(x), x - lowpass, np.nan)
    return highpass.astype(np.float64, copy=False)


def pl66_highpass_data(data: np.ndarray, dt_hours: float = 1.0, T_hours: float = 33.0) -> np.ndarray:
    """Apply the PL66 high-pass filter along the time axis for a 3D field with NaN masking."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D time/y/x array, got shape {data.shape}.")

    nt, ny, nx = data.shape
    out = np.full(data.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(data)

    for j in range(ny):
        for i in range(nx):
            mask = valid[:, j, i]
            if not np.any(mask):
                continue
            series = data[mask, j, i]
            filtered = _pl66_lowpass_valid_masked(series, dt_hours=dt_hours, T_hours=T_hours)
            hp = data[:, j, i].copy()
            hp[mask] = data[mask, j, i] - filtered
            hp[~mask] = np.nan
            out[:, j, i] = hp.astype(np.float32, copy=False)
    return out


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
    dt_hours: float = 1.0,
    T_hours: float = 33.0,
) -> None:
    print(f"Processing depth level {level_index} from {input_zarr}...", flush=True)
    data, coords, chunks = load_level_timeseries(
        input_zarr=input_zarr,
        variable_name=input_var,
        level_index=level_index,
        nt=nt,
    )

    filtered = pl66_highpass_data(data, dt_hours=dt_hours, T_hours=T_hours).astype(np.float32, copy=False)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PL66 highpass-filter pressure anomaly time series for one depth level using Zarr I/O."
    )
    parser.add_argument("k", type=int, help="Depth index to process.")
    parser.add_argument(
        "--nt",
        type=int,
        default=None,
        help="Number of time steps to process. Defaults to all time steps in the input Zarr store.",
    )
    parser.add_argument(
        "--dt-hours",
        type=float,
        default=1.0,
        help="Sample interval in hours used by the PL66 filter.",
    )
    parser.add_argument(
        "--T-hours",
        type=float,
        default=33.0,
        help="PL66 filter half-amplitude period in hours.",
    )
    parser.add_argument(
        "--data-dir",
        default="/home/ceoas/liux8/work/basin_modes/sulu_sea/post_proc/linear_tests/case_ctrl_bot_intens_Tsorted_2/zarr_output",
        help="Base directory for default input/output paths.",
    )
    parser.add_argument(
        "--input-zarr",
        default=None,
        help="Path to the source Zarr store containing the 4D pressure anomaly field.",
    )
    parser.add_argument(
        "--input-var",
        default="p_rho_bc",
        help="Variable name to read from the source Zarr store.",
    )
    parser.add_argument(
        "--output-zarr",
        default=None,
        help="Path to the output Zarr store for the filtered level.",
    )
    parser.add_argument(
        "--output-var",
        default="p_rho_bc_highpass_pl66",
        help="Variable name to write to the output Zarr store.",
    )
    args = parser.parse_args()

    input_zarr = args.input_zarr or os.path.join(args.data_dir, "p_rho_bc.zarr")
    output_zarr = args.output_zarr or os.path.join(
        args.data_dir,
        "p_rho_bc_highpass_pl66.zarr",
    )

    process_level(
        level_index=args.k,
        input_zarr=input_zarr,
        input_var=args.input_var,
        output_zarr=output_zarr,
        output_var=args.output_var,
        nt=args.nt,
        dt_hours=args.dt_hours,
        T_hours=args.T_hours,
    )


if __name__ == "__main__":
    main()
