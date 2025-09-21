import numpy as np
from numpy import cos, pi
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

Ho = 4000  # depth of ocean (m)
nx = 932    # gridpoints in x
ny = 1142    # gridpoints in y
xo = 112.5     # origin in x,y for ocean domain
yo = -3.3    # (i.e. southwestern corner of ocean domain)
dx = 1/60     # grid spacing in x (degrees longitude)
dy = 1/60     # grid spacing in y (degrees latitude)
xeast  = xo + (nx-2)*dx   # eastern extent of ocean domain
ynorth = yo + (ny-2)*dy   # northern extent of ocean domain

x = np.linspace(xo-dx, xeast, nx) + dx/2  #XC
y = np.linspace(yo-dy, ynorth, ny) + dy/2 #YC
# Flat bottom at z=-Ho
# h = -Ho * np.ones((ny, nx))
ds_topo = xr.open_dataset('topo_sulu_from_gebco.nc')

# Prepare the interpolator for the toposulu data
interp_func = RegularGridInterpolator(
    (ds_topo.lat.values, ds_topo.lon.values),
    ds_topo.elevation.values,
    bounds_error=False,
    fill_value=np.nan
)

# Create meshgrid for x and y
X, Y = np.meshgrid(x, y)
points = np.column_stack([Y.ravel(), X.ravel()])

# Interpolate toposulu onto the (x, y) grid
h = interp_func(points).reshape(Y.shape)

h[np.where(h>=0)]=np.nan
h[np.where(h<=-6000)]=-6000

# create a border ring of walls around edge of domain
# h[:, [0,-1]] = 0   # set ocean depth to zero at east and west walls
# h[[0,-1], :] = 0   # set ocean depth to zero at south and north walls
# save as single-precision (float32) with big-endian byte ordering
h.astype('>f4').tofile('bathy_h_fromgebco.bin')

# ocean domain extends from (xo,yo) to (xeast,ynorth)
# (i.e. the ocean spans nx-2, ny-2 grid cells)
# out-of-box-config: xo=0, yo=15, dx=dy=1 deg, ocean extent (0E,15N)-(60E,75N)
# model domain includes a land cell surrounding the ocean domain
# The full model domain cell centers are located at:
#    XC(:,1) = -0.5, +0.5, ..., +60.5 (degrees longitiude)
#    YC(1,:) = 14.5, 15.5, ..., 75.5 (degrees latitude)
# and full model domain cell corners are located at:
#    XG(:,1) = -1,  0, ..., 60 [, 61] (degrees longitiude)
#    YG(1,:) = 14, 15, ..., 75 [, 76] (degrees latitude)
# where the last value in brackets is not included 
# in the MITgcm grid variables XG,YG (but is in variables Xp1,Yp1)
# and reflects the eastern and northern edge of the model domain respectively.
# See section 2.11.4 of the MITgcm users manual.

# Zonal wind-stress
tauMax = 0.1
x = np.linspace(xo-dx, xeast, nx)
y = np.linspace(yo-dy, ynorth, ny) + dy/2
Y, X = np.meshgrid(y, x, indexing='ij')     # zonal wind-stress on (XG,YC) points
tau = -tauMax * cos(2*pi*((Y-yo)/(ny-2)/dy))  # ny-2 accounts for walls at N,S boundaries
# tau.astype('>f4').tofile('windx_cosy.bin')

# Restoring temperature (function of y only,
# from Tmax at southern edge to Tmin at northern edge)
Tmax = 30
Tmin = 0
Trest = (Tmax-Tmin)/(ny-2)/dy * (ynorth-Y) + Tmin # located and computed at YC points
# Trest.astype('>f4').tofile('SST_relax.bin')
