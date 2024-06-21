import numpy as np
import h5py

# Configurable variables
grp = 'g5760'
run = 'z4'
sim = f'{grp}/{run}'
snap = 188 # Snapshot number
zoom_dir = '/orcd/data/mvogelsb/004/Thesan-Zooms'
filename = f'{zoom_dir}/{sim}/postprocessing/proj/halo_proj_RHD_{snap:03d}.hdf5'

# Constants
mH = 1.6735327e-24         # Mass of hydrogen atom (g)
X  = 0.76                  # Primordial hydrogen mass fraction

# Read the projection file
with h5py.File(filename, 'r') as f:
    n_cameras = f['camera_directions'].shape[0] # Number of cameras
    halo_id = f['halo_id'][:] # Group IDs
    n_halos = len(halo_id) # Number of halos
    g = f['proj'] # Group containing the projections
    for i_halo in range(n_halos):
        p = g[f'{i_halo}'] # Group containing the projections of the halo
        rho = p['proj_rho_avg'][:] # Average density projection = sum(rho * dl) / sum(dl) [g/cm^3]
        x_HI = p['proj_xHI_mass'][:] # Mass-weighted neutral fraction projection = sum(x_HI * rho * dl) / sum(rho * dl)
        dl = p.attrs['proj_depth'] # Depth of the projection [cm]
        # halo_radius = p.attrs['halo_radius'] # Radius of the halo [cm]
        # image_radius = p.attrs['image_radius'] # Radius of the image [cm]
        # n_pixels = N_HI.shape[1] # Number of pixels
        # pixel_width = 2. * image_radius / n_pixels [cm]
        N_HI = x_HI * rho * (X * dl / mH) # Column density = sum(n_HI * dl) [cm^-2]
        # Note: N_HI has dimensions of [n_cameras, n_pixels, n_pixels]
