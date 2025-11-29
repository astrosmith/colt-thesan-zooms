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
# colt_dir = f'/orcd/data/mvogelsb/005/Lab/Thesan-Zooms-COLT/{sim}/ics'
states = 'states-no-UVB' # States prefix
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

f_vir = 4.  # Virial radius extraction factor
use_smoothed = True  # Use smoothed versions
state_fields = ['G_ion', 'x_e', 'x_HI', 'x_HII', 'x_HeI', 'x_HeII',
                'x_CI', 'x_CII', 'x_CIII', 'x_CIV',
                'x_NI', 'x_NII', 'x_NIII', 'x_NIV', 'x_NV',
                'x_OI', 'x_OII', 'x_OIII', 'x_OIV',
                'x_NeI', 'x_NeII', 'x_NeIII', 'x_NeIV',
                'x_MgI', 'x_MgII', 'x_MgIII',
                'x_SiI', 'x_SiII', 'x_SiIII', 'x_SiIV',
                'x_SI', 'x_SII', 'x_SIII', 'x_SIV', 'x_SV', 'x_SVI',
                'x_FeI', 'x_FeII', 'x_FeIII', 'x_FeIV', 'x_FeV', 'x_FeVI']
units = {'r': b'cm', 'v': b'cm/s', 'e_int': b'cm^2/s^2', 'T_dust': b'K', 'rho': b'g/cm^3', 'SFR': b'Msun/yr',
         'r_star': b'cm', 'v_star': b'cm/s', 'm_star': b'Msun', 'm_init_star': b'Msun', 'age_star': b'Gyr'}

print(f'Extracting states in {colt_dir} ...')
tree_file = f'/orcd/data/mvogelsb/004/Thesan-Zooms/analyze/trees/{sim}/tree.hdf5'
with h5py.File(tree_file, 'r') as f:
    snaps = f['Snapshots'][:] # Snapshots in the tree
    zs = f['Redshifts'][:] # Redshifts in the tree
    if not use_smoothed:
        group_ids = f['Group']['GroupID'][:] # Group ID in the tree
        subhalo_ids = f['Subhalo']['SubhaloID'][:] # Subhalo ID in the tree
        R_virs = f['Group']['Group_R_Crit200'][:] # Group virial radii in the tree [ckpc/h]
    else:
        with h5py.File(colt_dir + '_tree/center.hdf5', 'r') as sf:
            g = sf['Smoothed']
            r_virs = g['TargetPos'][:]  # Use smoothed versions
            R_virs = g['R_Crit200'][:]

if False:
    mask = np.zeros(len(snaps), dtype=bool)
    mask[-2] = True
    # n_start = 8
    # mask = (snaps >= n_start*189//9) & (snaps <= (n_start+1)*189//9)
    snaps = snaps[mask]
    zs = zs[mask]
    r_virs = r_virs[mask,:]
    R_virs = R_virs[mask]
    if not use_smoothed:
        group_ids = group_ids[mask]
        subhalo_ids = subhalo_ids[mask]
n_snaps = len(snaps)
r_HRs = np.empty([n_snaps, 3]) # High-resolution center of mass positions [cm]
r_HRs.fill(np.nan) # Fill with NaNs
if not use_smoothed:
    group_indices = np.empty(n_snaps, dtype=np.int32) # Group index in the candidates
    subhalo_indices = np.empty(n_snaps, dtype=np.int32) # Subhalo index in the candidates
    r_virs = np.empty([n_snaps, 3]) # Group positions [cm]
    r_virs.fill(np.nan)
failed_states = [] # List of snapshots where ionization states failed to copy
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
        UnitLength_in_cm = header['UnitLength_in_cm']
        UnitVelocity_in_cm_per_s = header['UnitVelocity_in_cm_per_s']
        length_to_cgs = a * UnitLength_in_cm / h
        r_HRs[i] = length_to_cgs * header['PosHR'] # High-resolution center of mass position [cm]
        if use_smoothed:
            r_virs[i] = length_to_cgs * r_virs[i] - r_HRs[i] # Relative group position [cm]
        else:
            cand_group_ids = f['Group']['GroupID'][:] # Group ID in the candidates
            cand_subhalo_ids = f['Subhalo']['SubhaloID'][:] # Subhalo ID in the candidates
            group_indices[i] = np.where(cand_group_ids == group_ids[i])[0][0] # Group index in the candidates
            subhalo_indices[i] = np.where(cand_subhalo_ids == subhalo_ids[i])[0][0] # Subhalo index in the candidates
            r_virs[i] = length_to_cgs * f['Group']['GroupPos'][group_indices[i]] - r_HRs[i] # Relative group position [cm]
        R_virs[i] = length_to_cgs * R_virs[i] # Virial radius unit conversion [cm]

    with h5py.File(colt_file, 'r') as f:
        r = f['r'][:] - r_virs[i] # Gas position [cm]
        r_box = f_vir * R_virs[i] # Radial cut = 2 * virial radius
        gas_mask = (np.sum(r**2, axis=1) < r_box**2) # Sphere cut

    try:
        states_file = f'{colt_dir}/{states}_{snap:03d}.hdf5'
        new_states_file = f'{colt_dir}_tree/{states}_{snap:03d}.hdf5'
        with h5py.File(states_file, 'r') as f, h5py.File(new_states_file, 'w') as g:
            for field in state_fields:
                g.create_dataset(field, data=f[field][:][gas_mask])
            # if 'G_ion' in state_fields: g['G_ion'].attrs['units'] = b'erg/s'
    except Exception as e:
        failed_states.append(snap)

if len(failed_states) > 0:
    print(f'Failed to copy ionization states for snapshots: {failed_states}')
