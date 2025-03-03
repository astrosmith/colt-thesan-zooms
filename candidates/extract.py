import numpy as np
import h5py, os, errno, platform

# Configurable global variables
if platform.system() == 'Darwin':
    sim, zoom_dir = 'g500531/z4', os.path.expandvars('$HOME/Engaging/Thesan-Zooms')
else:
    sim, zoom_dir = 'g5760/z4', '/orcd/data/mvogelsb/004/Thesan-Zooms'

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        sim = sys.argv[1]
    elif len(sys.argv) == 3:
        sim, zoom_dir = sys.argv[1], sys.argv[3]
    elif len(sys.argv) != 1:
        raise ValueError('Usage: python extract.py [sim] [zoom_dir]')

# Derived global variables
cand_dir = f'{zoom_dir}/{sim}/postprocessing/candidates'
colt_dir = f'{zoom_dir}-COLT/{sim}/ics'
os.makedirs(f'{colt_dir}_tree', exist_ok=True) # Ensure the new colt directory exists

# Overwrite for local testing
#dist_dir = '.'
#colt_dir = '.'

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

def progressbar(it, prefix="", size=100, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

f_vir = 2.  # Virial radius extraction factor
gas_fields = ['D', 'D_Si', 'SFR', 'T_dust', 'X', 'Y', 'Z', 'Z_C', 'Z_Fe', 'Z_Mg', 'Z_N', 'Z_Ne', 'Z_O', 'Z_S', 'Z_Si',
              'e_int', 'is_HR', 'r', 'rho', 'v', 'x_H2', 'x_HI', 'x_HeI', 'x_HeII', 'x_e', 'id', 'group_id', 'subhalo_id']
star_fields = ['Z_star', 'age_star', 'm_star', 'm_init_star', 'r_star', 'v_star', 'id_star', 'group_id_star', 'subhalo_id_star']
units = {'r': b'cm', 'v': b'cm/s', 'e_int': b'cm^2/s^2', 'T_dust': b'K', 'rho': b'g/cm^3', 'SFR': b'Msun/yr',
         'r_star': b'cm', 'v_star': b'cm/s', 'm_star': b'Msun', 'm_init_star': b'Msun', 'age_star': b'Gyr'}

print(f'Extracting {colt_dir} ...')
tree_file = f'/orcd/data/mvogelsb/004/Thesan-Zooms/analyze/trees/{sim}/tree.hdf5'
with h5py.File(tree_file, 'r') as f:
    snaps = f['Snapshots'][:] # Snapshots in the tree
    n_snaps = len(snaps)
    group_ids = f['Group']['GroupID'][:] # Group ID in the tree
    subhalo_ids = f['Subhalo']['SubhaloID'][:] # Subhalo ID in the tree

group_indices = np.empty(n_snaps, dtype=np.int32) # Group index in the candidates
subhalo_indices = np.empty(n_snaps, dtype=np.int32) # Subhalo index in the candidates
r_HRs = np.empty([n_snaps, 3]) # High-resolution center of mass positions [cm]
r_virs = np.empty([n_snaps, 3]) # Group positions [cm]
R_virs = np.empty(n_snaps) # Group virial radii [cm]
r_HRs.fill(np.nan) # Fill with NaNs
r_virs.fill(np.nan)
R_virs.fill(np.nan)
for i in progressbar(range(n_snaps)):
    snap = snaps[i]
    cand_file = f'{cand_dir}/candidates_{snap:03d}.hdf5'
    colt_file = f'{colt_dir}/colt_{snap:03d}.hdf5'
    new_file = f'{colt_dir}_tree/colt_{snap:03d}.hdf5'
    silentremove(new_file)
    with h5py.File(cand_file, 'r') as f:
        header = f['Header'].attrs
        a = header['Time']
        z = 1. / a - 1.
        BoxSize = header['BoxSize']
        h = header['HubbleParam']
        Omega0 = header['Omega0']
        OmegaBaryon = header['OmegaBaryon']
        UnitLength_in_cm = header['UnitLength_in_cm']
        UnitMass_in_g = header['UnitMass_in_g']
        UnitVelocity_in_cm_per_s = header['UnitVelocity_in_cm_per_s']
        length_to_cgs = a * UnitLength_in_cm / h
        cand_group_ids = f['Group']['GroupID'][:] # Group ID in the candidates
        cand_subhalo_ids = f['Subhalo']['SubhaloID'][:] # Subhalo ID in the candidates
        group_indices[i] = np.where(cand_group_ids == group_ids[i])[0][0] # Group index in the candidates
        subhalo_indices[i] = np.where(cand_subhalo_ids == subhalo_ids[i])[0][0] # Subhalo index in the candidates
        r_HRs[i] = length_to_cgs * header['PosHR'] # High-resolution center of mass position [cm]
        r_virs[i] = length_to_cgs * f['Group']['GroupPos'][group_indices[i]] - r_HRs[i] # Group position [cm]
        R_virs[i] = length_to_cgs * f['Group']['Group_R_Crit200'][group_indices[i]] # Virial radius [cm]

    with h5py.File(colt_file, 'r') as f, h5py.File(new_file, 'w') as g:
        # Read gas and star coordinates for the radial masks
        r = f['r'][:] - r_virs[i] # Gas position [cm]
        r_box = f_vir * R_virs[i] # Radial cut = 2 * virial radius
        gas_mask = (np.sum(r**2, axis=1) < r_box**2) # Sphere cut
        n_cells = np.int32(np.count_nonzero(gas_mask)) # Number of cells

        # Simulation properties
        g.attrs['n_cells'] = n_cells  # Number of cells
        g.attrs['redshift'] = z  # Current simulation redshift
        g.attrs['Omega0'] = Omega0  # Matter density [rho_crit_0]
        g.attrs['OmegaB'] = OmegaBaryon  # Baryon density [rho_crit_0]
        g.attrs['h100'] = h  # Hubble constant [100 km/s/Mpc]
        g.attrs['r_box'] = r_box # Bounding box radius [cm]

        # Gas fields
        for field in gas_fields:
            g.create_dataset(field, data=f[field][:][gas_mask])
            if field in units: g[field].attrs['units'] = units[field]

        # Star fields
        if 'r_star' in f:
            r_star = f['r_star'][:] - r_virs[i] # Star position [cm]
            star_mask = (np.sum(r_star**2, axis=1) < r_box**2) # Sphere cut
            n_stars = np.int32(np.count_nonzero(star_mask)) # Number of star particles
            if n_stars > 0:
                g.attrs['n_stars'] = n_stars  # Number of star particles
                for field in star_fields:
                    g.create_dataset(field, data=f[field][:][star_mask])
                    if field in units: g[field].attrs['units'] = units[field]
