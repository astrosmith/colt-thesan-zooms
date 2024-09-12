import h5py, os, platform
import numpy as np

# Configurable global variables
snap = 188 # Snapshot number
if platform.system() == 'Darwin':
    sim, zoom_dir = 'g500531/z4', os.path.expandvars('$HOME/Engaging/Thesan-Zooms')
else:
    sim, zoom_dir = 'g5760/z4', '/orcd/data/mvogelsb/004/Thesan-Zooms'

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        sim = sys.argv[1]
    elif len(sys.argv) == 3:
        sim, snap = sys.argv[1], int(sys.argv[2])
    elif len(sys.argv) == 4:
        sim, snap, zoom_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    elif len(sys.argv) != 1:
        raise ValueError('Usage: python candidates.py [sim] [snap] [zoom_dir]')

mag_dir = f'{zoom_dir}/{sim}/postprocessing/mag'
cand_dir = f'{zoom_dir}/{sim}/postprocessing/candidates'
colt_dir = f'{zoom_dir}-COLT/{sim}/ics'

# Overwrite for local testing
#mag_dir = '.'
#cand_dir = '.'
#colt_dir = '.'

# The following paths should not change
mag_file = f'{mag_dir}/mag_{snap:03d}.hdf5'
cand_file = f'{cand_dir}/candidates_{snap:03d}.hdf5'
halo_file = f'{colt_dir}/halo_{snap:03d}.hdf5'

def write_halos():
    """Add the candidate halo data to the COLT initial conditions file."""
    with h5py.File(cand_file, 'r') as f, h5py.File(mag_file, 'r') as mf, h5py.File(halo_file, 'w') as hf:
        header = f['Header'].attrs
        a = header['Time']
        h = header['HubbleParam']
        UnitLength_in_cm = header['UnitLength_in_cm']
        length_to_cgs = a * UnitLength_in_cm / h
        r_HR =  header['PosHR'].astype(np.float64) * length_to_cgs # High-resolution position [cm]
        n_groups = header['Ngroups_Candidates']
        if n_groups > 0:
            g = f['Group']
            group_mask = g['StarFlag'][:]
            n_groups = np.int32(np.count_nonzero(group_mask)) # Number of groups with high-resolution stars
            if n_groups > 0:
                mg = mf['Group'] # M1500 data
                group_id = g['GroupID'][:][group_mask] # Group IDs
                r_group = g['GroupPos'][:][group_mask].astype(np.float64) * length_to_cgs # Halo positions [cm]
                r_group_light = mg['1500Pos'][:][group_id].astype(np.float64) * length_to_cgs # Light-weighted positions [cm]
                r_group_mass = mg['MstarPos'][:][group_id].astype(np.float64) * length_to_cgs # Mass-weighted positions [cm]
                for i in range(3):
                    r_group[:,i] -= r_HR[i] # Relative halo positions [cm]
                    r_group_light[:,i] -= r_HR[i]
                    r_group_mass[:,i] -= r_HR[i]
                R_group = g['Group_R_Crit200'][:][group_mask].astype(np.float64) * length_to_cgs # Halo radii [cm]
                R_group_light = 2. * mg['1500Half'][:][group_id].astype(np.float64) * length_to_cgs # Half-light radii [cm]
                R_group_mass = 2. * mg['MstarHalf'][:][group_id].astype(np.float64) * length_to_cgs # Half-mass radii [cm]
                v_group_light = mg['1500Vel'][:][group_id].astype(np.float64) * 1e5 # Light-weighted velocities [cm/s]
                v_group_mass = mg['MstarVel'][:][group_id].astype(np.float64) * 1e5 # Mass-weighted velocities [cm/s]
                s_group_light = mg['1500VelDisp'][:][group_id].astype(np.float64) * 1e5 # Light-weighted velocity dispersions [cm/s]
                s_group_mass = mg['MstarVelDisp'][:][group_id].astype(np.float64) * 1e5 # Mass-weighted velocity dispersions [cm/s]
                # Add the halo data to the COLT initial conditions file
                hf.create_dataset(b'group_id', data=group_id)
                hf.create_dataset(b'r_group', data=r_group)
                hf['r_group'].attrs['units'] = b'cm'
                hf.create_dataset(b'r_group_light', data=r_group_light)
                hf['r_group_light'].attrs['units'] = b'cm'
                hf.create_dataset(b'r_group_mass', data=r_group_mass)
                hf['r_group_mass'].attrs['units'] = b'cm'
                hf.create_dataset(b'R_group', data=R_group)
                hf['R_group'].attrs['units'] = b'cm'
                hf.create_dataset(b'R_group_light', data=R_group_light)
                hf['R_group_light'].attrs['units'] = b'cm'
                hf.create_dataset(b'R_group_mass', data=R_group_mass)
                hf['R_group_mass'].attrs['units'] = b'cm'
                hf.create_dataset(b'v_group_light', data=v_group_light)
                hf['v_group_light'].attrs['units'] = b'cm/s'
                hf.create_dataset(b'v_group_mass', data=v_group_mass)
                hf['v_group_mass'].attrs['units'] = b'cm/s'
                hf.create_dataset(b's_group_light', data=s_group_light)
                hf['s_group_light'].attrs['units'] = b'cm/s'
                hf.create_dataset(b's_group_mass', data=s_group_mass)
                hf['s_group_mass'].attrs['units'] = b'cm/s'
        n_subhalos = header['Nsubhalos_Candidates']
        if n_subhalos > 0:
            g = f['Subhalo']
            subhalo_mask = g['StarFlag'][:]
            n_subhalos = np.int32(np.count_nonzero(subhalo_mask)) # Number of subhalos with high-resolution stars
            if n_subhalos > 0:
                mg = mf['Subhalo'] # Mstar data
                mg_NoSFH = mf['Subhalo_NoSFH'] # M1500 data
                subhalo_id = g['SubhaloID'][:][subhalo_mask] # Subhalo IDs
                r_subhalo = g['SubhaloPos'][:][subhalo_mask].astype(np.float64) * length_to_cgs # Halo positions [cm]
                r_subhalo_light = mg_NoSFH['1500Pos'][:][subhalo_id].astype(np.float64) * length_to_cgs # Light-weighted positions [cm]
                r_subhalo_mass = mg['MstarPos'][:][subhalo_id].astype(np.float64) * length_to_cgs # Mass-weighted positions [cm]
                for i in range(3):
                    r_subhalo[:,i] -= r_HR[i] # Relative halo positions [cm]
                    r_subhalo_light[:,i] -= r_HR[i]
                    r_subhalo_mass[:,i] -= r_HR[i]
                R_subhalo = g['R_vir'][:][subhalo_mask].astype(np.float64) * length_to_cgs # Halo radii [cm]
                R_subhalo_light = 2. * mg_NoSFH['1500Half'][:][subhalo_id].astype(np.float64) * length_to_cgs # Half-light radii [cm]
                R_subhalo_mass = 2. * mg['MstarHalf'][:][subhalo_id].astype(np.float64) * length_to_cgs # Half-mass radii [cm]
                v_subhalo_light = mg_NoSFH['1500Vel'][:][subhalo_id].astype(np.float64) * 1e5 # Light-weighted velocities [cm/s]
                v_subhalo_mass = mg['MstarVel'][:][subhalo_id].astype(np.float64) * 1e5 # Mass-weighted velocities [cm/s]
                s_subhalo_light = mg_NoSFH['1500VelDisp'][:][subhalo_id].astype(np.float64) * 1e5 # Light-weighted velocity dispersions [cm/s]
                s_subhalo_mass = mg['MstarVelDisp'][:][subhalo_id].astype(np.float64) * 1e5 # Mass-weighted velocity dispersions [cm/s]
                # Add the halo data to the COLT initial conditions file
                hf.create_dataset(b'subhalo_id', data=subhalo_id)
                hf.create_dataset(b'r_subhalo', data=r_subhalo)
                hf['r_subhalo'].attrs['units'] = b'cm'
                hf.create_dataset(b'r_subhalo_light', data=r_subhalo_light)
                hf['r_subhalo_light'].attrs['units'] = b'cm'
                hf.create_dataset(b'r_subhalo_mass', data=r_subhalo_mass)
                hf['r_subhalo_mass'].attrs['units'] = b'cm'
                hf.create_dataset(b'R_subhalo', data=R_subhalo)
                hf['R_subhalo'].attrs['units'] = b'cm'
                hf.create_dataset(b'R_subhalo_light', data=R_subhalo_light)
                hf['R_subhalo_light'].attrs['units'] = b'cm'
                hf.create_dataset(b'R_subhalo_mass', data=R_subhalo_mass)
                hf['R_subhalo_mass'].attrs['units'] = b'cm'
                hf.create_dataset(b'v_subhalo_light', data=v_subhalo_light)
                hf['v_subhalo_light'].attrs['units'] = b'cm/s'
                hf.create_dataset(b'v_subhalo_mass', data=v_subhalo_mass)
                hf['v_subhalo_mass'].attrs['units'] = b'cm/s'
                hf.create_dataset(b's_subhalo_light', data=s_subhalo_light)
                hf['s_subhalo_light'].attrs['units'] = b'cm/s'
                hf.create_dataset(b's_subhalo_mass', data=s_subhalo_mass)
                hf['s_subhalo_mass'].attrs['units'] = b'cm/s'

if __name__ == '__main__':
    write_halos()
