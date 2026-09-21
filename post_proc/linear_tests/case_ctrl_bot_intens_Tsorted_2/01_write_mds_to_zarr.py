import os
import argparse
import numpy as np
import xarray as xr
# from MITgcmutils import rdmds
from xmitgcm.utils import read_mds

# =========================
# User settings
# =========================

nt = 600
nz = 87
nx = 1520
ny = 1908

block_size = 50

mds_dir = "/home/ceoas/liux8/work/basin_modes/sulu_sea/linear_tests/case_ctrl_bot_intensify_Tsorted_2"
zarr_dir = "zarr_output"

# MITgcm variable prefixes
variables = {
    "u": {
        "mds_prefix": "UVEL",
        "zarr_path": "u.zarr",
    },
    "v": {
        "mds_prefix": "VVEL",
        "zarr_path": "v.zarr",
    },
    "phihyd": {
        "mds_prefix": "PHIHYD",
        "zarr_path": "phihyd.zarr",
    },
    "theta": {
        "mds_prefix": "THETA",
        "zarr_path": "theta.zarr",
    },
}

dtype = "float32"

# =========================
# Helper functions
# =========================

def read_one_time(ds, var_name):
    """
    Read one MITgCM MDS time record.

    Expected output shape: (z, x, y)
    """

    # var_prefix = 'outs_sn'
    # file_base = os.path.join(mds_dir, var_prefix)

    # arr = read_mds(f'{file_base}.{iternum:010d}')

    # arr = np.asarray(arr, dtype=dtype)
    arr = ds[var_name].compute().astype(dtype)

    if arr.shape != (nz, ny, nx):
        raise ValueError(
            f"Unexpected shape for {var_name}, ds={ds}: "
            f"{arr.shape}, expected {(nz, ny, nx)}"
        )
    return arr


def write_variable_block(varname, varinfo, t0, t1, iter_numbers):
    """
    Read MDS files from t0:t1 and write to zarr region.
    """

    var_prefix = 'outs_sn'
    var_name = varinfo["mds_prefix"]
    zarr_path = os.path.join(zarr_dir, varinfo["zarr_path"])

    ntime = t1 - t0

    print(f"Writing {varname}: time {t0}:{t1}")

    data_block = np.empty(
        (ntime, nz, ny, nx),
        dtype=dtype,
    )

    for ii, tt in enumerate(range(t0, t1)):
        iternum = iter_numbers[tt]*180
        print(f"  Reading {var_name}, time index={tt}, iter={iternum}")

        ds = read_mds(f'{mds_dir}/{var_prefix}.{iternum:010d}')
        data_block[ii, :, :, :] = read_one_time(ds, var_name)

    ds_block = xr.Dataset(
        {
            varname: (
                ("time", "z", "y", "x"),
                data_block,
            )
        },
        coords={
            "time": np.arange(t0, t1),
            "z": np.arange(nz),
            "y": np.arange(ny),
            "x": np.arange(nx),
        },
    )

    ds_block.to_zarr(
        zarr_path,
        mode="r+",
        region={
            "time": slice(t0, t1),
            "z": slice(0, nz),
            "y": slice(0, ny),
            "x": slice(0, nx),
        },
        consolidated=False,
    )

    print(f"Finished {varname}: time {t0}:{t1}")


# =========================
# Main
# =========================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-id",
        type=int,
        required=True,
        help="Slurm array task ID, starting from 0",
    )
    parser.add_argument(
        "--iter-start",
        type=int,
        default=0,
        help="First MITgCM iteration number",
    )
    parser.add_argument(
        "--iter-step",
        type=int,
        default=1,
        help="MITgCM iteration interval between saved outputs",
    )
    args = parser.parse_args()
    print(args)

    job_id = args.job_id

    t0 = job_id * block_size
    t1 = min(t0 + block_size, nt)

    if t0 >= nt:
        print(f"job-id {job_id} is outside nt={nt}. Nothing to do.")
        return

    iter_numbers = [
        args.iter_start + i * args.iter_step
        for i in range(nt)
    ]

    print(f"Job {job_id}: writing time {t0}:{t1}")

    for varname, varinfo in variables.items():
        write_variable_block(
            varname=varname,
            varinfo=varinfo,
            t0=t0,
            t1=t1,
            iter_numbers=iter_numbers,
        )

    print("Done.")


if __name__ == "__main__":
    main()
