from dask_jobqueue import SLURMCluster
from dask.distributed import Client

cluster = SLURMCluster(cores=28,name='make_zarr',walltime='00:20:00',job_extra=['--constraint=BDW28','--exclusive','--nodes=1'],memory='40GB')
print(cluster.job_script())


cluster.scale(4)

c =Client()

print('###### startig computation #######')

import sys, glob
import numpy as np
import xarray as xr
import xscale.spectral.fft as xfft

sys.path.insert(0, "/home/aajayi/Lib/python/w_k_scripts/")
from SmallBox import smallbox
for rbox in smallbox:
    box = rbox

import Wavenum_freq_spec_func as wfs

# - Daily Dataset
Daily_data_dir = '/scratch/cnt0024/hmg2840/albert7a/NATL60/NATL60-CJM165-S/UVsurf/'
U_Daily_JFM_data_file = Daily_data_dir + 'NATL60-CJM165_y2013m0[1-3]*.1d_Usurf.nc' 
V_Daily_JFM_data_file = Daily_data_dir + 'NATL60-CJM165_y2013m0[1-3]*.1d_Vsurf.nc'

U_Daily_JAS_data_file = Daily_data_dir + 'NATL60-CJM165_y2013m0[7-9]*.1d_Usurf.nc' 
V_Daily_JAS_data_file = Daily_data_dir + 'NATL60-CJM165_y2013m0[7-9]*.1d_Vsurf.nc'

# - Hourly Dataset
Hourly_data_dir = '/scratch/cnt0024/hmg2840/albert7a/NATL60/NATL60-CJM165-S/1h/ALL/'
U_Hourly_JFM_data_file = Hourly_data_dir + 'NATL60-CJM165_y2013m0[1-3]d*.1h_gridU.nc'
V_Hourly_JFM_data_file = Hourly_data_dir + 'NATL60-CJM165_y2013m0[1-3]d*.1h_gridV.nc'

U_Hourly_JAS_data_file = Hourly_data_dir + 'NATL60-CJM165_y2013m0[7-9]d*.1h_gridU.nc'
V_Hourly_JAS_data_file = Hourly_data_dir + 'NATL60-CJM165_y2013m0[7-9]d*.1h_gridV.nc'

# - Save dataset to this folder
OutputFolder = '/scratch/cnt0024/hmg2840/albert7a/AJ/'

Ufile = U_Hourly_JFM_data_file
Vfile = V_Hourly_JFM_data_file
OutputFile = 'KE_Transfer_Flux_JFM_w_k_from_1h.nc'


################### Main computation  ######################
u = xr.open_mfdataset(Ufile,chunks={'x':100,'time_counter':1})['vozocrtx'][:,box.jmin:box.jmax,box.imin:box.imax]
v = xr.open_mfdataset(Vfile,chunks={'x':100,'time_counter':1})['vomecrty'][:,box.jmin:box.jmax,box.imin:box.imax]

# - remove NaN
u = u.interpolate_na(dim='y')
v = v.interpolate_na(dim='y')

# - get dx and dy
dx,dy = wfs.get_dx_dy(u[0])

#... Detrend data in all dimension ...
print('Detrend data in all dimension')
u = wfs.detrendn(u,axes=[0,1,2])
v = wfs.detrendn(v,axes=[0,1,2])

#... Apply hanning windowing ...') 
print('Apply hanning windowing')
u = wfs.apply_window(u, u.dims, window_type='hanning')
v = wfs.apply_window(v, v.dims, window_type='hanning')

# - get derivatives
derivatives = wfs.velocity_derivatives(u, v, xdim='x', ydim='y', dx={'x': dx, 'y': dy})
dudx = derivatives['u_x']; dudy = derivatives['u_y']
dvdx = derivatives['v_x']; dvdy = derivatives['v_y']

# - compute terms
phi1 = u*dudx + v*dudy
phi2 = u*dvdx + v*dvdy

uhat = xfft.fft(u, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)
vhat = xfft.fft(v, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)

phi1_hat = xfft.fft(phi1, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)
phi2_hat = xfft.fft(phi2, dim=('time_counter', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)

tm1 = (uhat.conj())*phi1_hat
tm2 = (vhat.conj())*phi2_hat

# - computer transfer
Nk,Nj,Ni = u.shape
transfer_2D = -1.0*(tm1 + tm2)/np.square(Ni*Nj)
transfer_term = transfer_2D.real

#... Get frequency and wavenumber ... 
print('Get frequency and wavenumber')
frequency,kx,ky = wfs.get_f_kx_ky(uhat)

#... Get istropic wavenumber ... 
print('Get istropic wavenumber')
wavenumber,kradial = wfs.get_wavnum_kradial(kx,ky)

#... Get numpy array ... 
print('Get numpy array')
var_psd_np = transfer_term.values

#... Get 2D frequency-wavenumber field ... 
print('Get transfer')
transfer = wfs.get_f_k_in_2D(kradial,wavenumber,var_psd_np) 

# - Get flux
print('Get flux')
flux = wfs.get_flux_in_1D(kradial,wavenumber,var_psd_np)

# Save to Netscdf file
# - build dataarray
print('Save to Netscdf file')
transfer_da = xr.DataArray(transfer,dims=['frequency','wavenumber'],name="transfer",coords=[frequency,wavenumber])
flux_da = xr.DataArray(flux,dims=['frequency','wavenumber'],name="flux",coords=[frequency,wavenumber])
transfer_da.attrs['Name'] = OutputFile

transfer_da.to_dataset().to_netcdf(path=OutputFolder+OutputFile,mode='w',engine='scipy')
flux_da.to_dataset().to_netcdf(path=OutputFolder+OutputFile,mode='a',engine='scipy')
