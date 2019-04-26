import sys
import numpy as np
from netCDF4 import Dataset as ncopen
gridfile = '/store/molines/NATL60/NATL60-I/NATL60_coordinates_v4.nc'


# - Define read data function
def read_datagrid(gridfile,latmin=None,latmax=None,lonmin=None,lonmax=None):
    """Return navlon,navlat."""
    ncfile = ncopen(gridfile,'r')
    # load navlon and navlat
    _navlon = ncfile.variables['nav_lon'][:,:]
    _navlat = ncfile.variables['nav_lat'][:,:]
    #-Define domain
    domain = (lonmin<_navlon) * (_navlon<lonmax) * (latmin<_navlat) * (_navlat<latmax)
    where = np.where(domain)
    vlats = _navlat[where]
    vlons = _navlon[where]
    #get indice
    jmin = where[0][vlats.argmin()]
    jmax = where[0][vlats.argmax()]
    imin = where[1][vlons.argmin()]
    imax = where[1][vlons.argmax()]
    #load arrays
    navlon = _navlon[jmin:jmax+1,imin:imax+1]
    navlat = _navlat[jmin:jmax+1,imin:imax+1]
    return navlon,navlat,jmin,jmax,imin,imax

# - Define box dimensions
latmin = 40.0; latmax = 45.0;
lonmin = -40.0; lonmax = -35.0;
box_name = 'SmallBox'

#- defining dictionaries for the boxes
class box: # empty container.
    def __init__(self,name=None):
        self.name = name
        return

dictboxes = {}
    
    # - Obtain navlon and Navlat
navlon,navlat,jmin,jmax,imin,imax = read_datagrid(gridfile,latmin=latmin,latmax=latmax,lonmin=lonmin,lonmax=lonmax)
    
# - save box parameter
abox = box(box_name)
abox.lonmin = navlon.min()
abox.lonmax = navlon.max()
abox.latmin = navlat.min()
abox.latmax = navlat.max()
abox.navlon = navlon
abox.navlat = navlat
abox.imin = imin
abox.imax = imax
abox.jmin = jmin
abox.jmax = jmax
dictboxes[box_name] = abox

smallbox = dictboxes.values()







