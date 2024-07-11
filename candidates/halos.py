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

cand_dir = f'{zoom_dir}/{sim}/postprocessing/candidates'
colt_dir = f'{zoom_dir}-COLT/{sim}/ics'

# Overwrite for local testing
#out_dir = '.'
#dist_dir = '.'
#colt_dir = '.'

# The following paths should not change
cand_file = f'{cand_dir}/candidates_{snap:03d}.hdf5'
colt_file = f'{colt_dir}/colt_{snap:03d}.hdf5'
halo_file = f'{colt_dir}/halo_{snap:03d}.hdf5'

def calculate_masses(m, m_star, r, r_star, n_halos, r_halo, R_halo):
    """Calculate the gas and stellar masses within each halo."""
    M_gas = np.zeros(n_halos)
    M_star = np.zeros(n_halos)
    R2_halo = R_halo**2
    for i in range(n_halos):
        mask = (np.sum((r - r_halo[i])**2, axis=1) < R2_halo[i])
        M_gas[i] = np.sum(m[mask]) # Gas mass within halo [Msun]
        mask = (np.sum((r_star - r_halo[i])**2, axis=1) < R2_halo[i])
        M_star[i] = np.sum(m_star[mask]) # Stellar mass within halo [Msun]
    return M_gas, M_star

def write_halos():
    """Add the candidate halo data to the COLT initial conditions file."""
    with h5py.File(cand_file, 'r') as f, h5py.File(colt_file, 'r') as cf, h5py.File(halo_file, 'w') as hf:
        header = f['Header'].attrs
        a = header['Time']
        h = header['HubbleParam']
        UnitLength_in_cm = header['UnitLength_in_cm']
        UnitMass_in_g = header['UnitMass_in_g']
        length_to_cgs = a * UnitLength_in_cm / h
        mass_to_cgs = UnitMass_in_g / h
        Msun = 1.988435e33 # Solar mass [g]
        mass_to_Msun = mass_to_cgs / Msun
        n_halos = header['Ngroups_Candidates']
        hf.attrs['n_halos'] = n_halos
        if n_halos > 0:
            g = f['Group']
            r_HR =  header['PosHR'].astype(np.float64) * length_to_cgs # High-resolution position [cm]
            r_halo = g['GroupPos'][:].astype(np.float64) * length_to_cgs # Halo positions [cm]
            for i in range(3):
                r_halo[:,i] -= r_HR[i] # Relative halo positions [cm]
            R_halo = g['Group_R_Crit200'][:].astype(np.float64) * length_to_cgs # Halo radii [cm]
            M_halo = g['Group_M_Crit200'][:].astype(np.float64) * mass_to_Msun # Halo masses [Msun]
            n_cells = np.int32(cf['r'].shape[0])
            edges = cf['edges'][:]; mask = np.ones(n_cells, dtype=bool); mask[edges] = False
            r = cf['r'][:][mask,:] # Cell positions [cm]
            r_star = cf['r_star'][:] # Star positions [cm]
            m = cf['rho'][:][mask] * cf['V'][:][mask] / Msun # Cell mass [Msun]
            m_star = cf['m_star'][:] # Star mass [Msun]
            M_gas, M_star = calculate_masses(m, m_star, r, r_star, n_halos, r_halo, R_halo)
            # Add the halo data to the COLT initial conditions file
            hf.create_dataset(b'id', data=g['GroupID'][:])
            hf.create_dataset(b'r_halo', data=r_halo)
            hf['r_halo'].attrs['units'] = b'cm'
            hf.create_dataset(b'R_halo', data=R_halo)
            hf['R_halo'].attrs['units'] = b'cm'
            hf.create_dataset(b'M_halo', data=M_halo)
            hf['M_halo'].attrs['units'] = b'Msun'
            hf.create_dataset(b'M_gas', data=M_gas)
            hf['M_gas'].attrs['units'] = b'Msun'
            hf.create_dataset(b'M_star', data=M_star)
            hf['M_star'].attrs['units'] = b'Msun'

if __name__ == '__main__':
    write_halos()
