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
        sim, zoom_dir = sys.argv[1], sys.argv[2]
    elif len(sys.argv) != 1:
        raise ValueError('Usage: python group_centering.py [sim] [zoom_dir]')

# Derived global variables
cand_dir = f'{zoom_dir}/{sim}/postprocessing/candidates'
colt_dir = f'{zoom_dir}-COLT/{sim}/ics'
# colt_dir = f'/orcd/data/mvogelsb/005/Lab/Thesan-Zooms-COLT/{sim}/ics'
os.makedirs(f'{colt_dir}_tree', exist_ok=True) # Ensure the new colt directory exists

M_sun = 1.988435e33 # Solar mass in g
pc = 3.085677581467192e18 # Parsec in cm
kpc = 1e3 * pc # Kiloparsec in cm

# Overwrite for local testing
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

print(f'Centering {colt_dir} ...')
tree_file = f'{zoom_dir}/{sim}/output/tree.hdf5'

with h5py.File(tree_file, 'r') as f:
    snaps = f['Snapshots'][:] # Snapshots in the tree
    group_ids = f['Group']['GroupID'][:] # Group ID in the tree
    subhalo_ids = f['Subhalo']['SubhaloID'][:] # Subhalo ID in the tree
    R_virs = f['Group']['Group_R_Crit200'][:] # Group virial radii in the tree [ckpc/h]
    zs = f['Redshifts'][:] # Redshifts in the tree

if False:
    mask = np.zeros(len(snaps), dtype=bool)
    mask[-2:] = True
    snaps = snaps[mask]
    R_virs = R_virs[mask]
    zs = zs[mask]
    group_ids = group_ids[mask]
    subhalo_ids = subhalo_ids[mask]

n_snaps = len(snaps)
group_indices = np.empty(n_snaps, dtype=np.int32) # Group index in the candidates
subhalo_indices = np.empty(n_snaps, dtype=np.int32) # Subhalo index in the candidates
r_HRs = np.empty([n_snaps, 3]) # High-resolution center of mass positions [cm]
com_arepo_box = np.empty([n_snaps, 3]) # New center of mass in Arepo box units [ckpc/h]
r_virs = np.empty([n_snaps, 3]) # Group positions [cm]
r_HRs.fill(np.nan) # Fill with NaNs
com_arepo_box.fill(np.nan)
r_virs.fill(np.nan)
for i in progressbar(range(n_snaps)):
    snap = snaps[i]
    cand_file = f'{cand_dir}/candidates_{snap:03d}.hdf5'
    colt_file = f'{colt_dir}/colt_{snap:03d}.hdf5'
    dm_file = f'{colt_dir}/dm_{snap:03d}.hdf5'
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
        r_virs[i] = length_to_cgs * f['Group']['GroupPos'][group_indices[i]] - r_HRs[i] # Group position relative to high-resolution center of mass [cm]
        R_virs[i] = length_to_cgs * R_virs[i] # Virial radius unit conversion [cm]
        pos = f['Group']['GroupPos'][group_indices[i]]

    with h5py.File(colt_file, 'r') as f:
        # Read gas and star coordinates for the radial masks
        gas_pos = r = f['r'][:] - r_virs[i] # Gas position [cm]
        r_box = 2. * R_virs[i] # Radial cut = 2 * virial radius
        gas_mask = (np.sum(r**2, axis=1) < r_box**2) # Sphere cut
        n_cells = np.int32(np.count_nonzero(gas_mask)) # Number of cells
        gas_flag = (n_cells >= 0)  # Gas flag
        rho_gas = f['rho'][gas_mask]  # Gas density [g/cm^3]
        vol_gas = f['V'][gas_mask]  # Cell volume [cm^3]
        mass_gas = rho_gas[:] * vol_gas[:]  # Gas mass [g]
        mass_gas = mass_gas / M_sun  # Convert to Msun
        gas_pos = gas_pos[gas_mask]  # Gas positions [cm]

        # Star Properties
        if 'r_star' in f:
            star_flag = True
            r_star = f['r_star'][:] - r_virs[i] # Star position [cm]
            star_mask = (np.sum(r_star**2, axis=1) < r_box**2) # Sphere cut
            n_stars = np.int32(np.count_nonzero(star_mask)) # Number of star particles
            n_cells = np.int32(np.count_nonzero(star_mask)) # Number of cells
            m_star = f['m_init_star'][star_mask]  # Star mass [Msun]
            r_star = r_star[star_mask]
        else:
            star_flag = False
            n_stars = 0
            m_star = np.zeros(1)
            r_star = np.zeros((1,3))

    with h5py.File(dm_file, 'r') as f:
        if 'r_p2' in f:
            p2_flag = True
            # Read coordinates for the radial masks
            r_p2 = f['r_p2'][:] - r_virs[i] # Dark matter particle positions Type 2 [cm]
            p2_mask = (np.sum(r_p2**2, axis=1) < r_box**2) # Sphere cut
            m_p2 = f['m_p2'][p2_mask]  # Mass of dark matter particles Type 2 [MSun]
            r_p2 = r_p2[p2_mask]
        else:
            p2_flag = False
            m_p2 = np.zeros(1)  # Initialize dark matter particle Type 2
            r_p2 = np.zeros((1,3))  # Initialize dark matter particle positions Type 2

        if 'r_p3' in f:
            p3_flag = True
            r_p3 = f['r_p3'][:] - r_virs[i]  # Dark matter particle positions Type 3 [cm]
            p3_mask = (np.sum(r_p3**2, axis=1) < r_box**2) # Sphere cut
            r_p3 = r_p3[p3_mask]  # Dark matter particle positions Type 3 within 2 virial radius[cm]
            m_p3 = f.attrs['m_p3']
            n_p3_tot = f.attrs['n_p3']  # Total number of dark matter particles Type 3 [MSun]
            m_p3 = m_p3 * np.ones(n_p3_tot)
            m_p3 = m_p3[p3_mask]  # Mass of dark matter particles Type 3 [MSun]
        else:
            p3_flag = False
            m_p3 = np.zeros(1)  # Initialize dark matter particle Type 3
            r_p3 = np.zeros((1,3))  # Initialize dark matter particle positions Type 3

        if 'r_dm' in f:
            dm_flag = True
            n_dm_tot = f.attrs['n_dm']  # Total number of dark matter particles
            m_dm = f.attrs['m_dm']  # Mass of dark matter particles [MSun]
            r_dm = f['r_dm'][:] - r_virs[i]  # Dark matter particle positions [cm]
            dm_mask = (np.sum(r_dm**2, axis=1) < r_box**2) # Sphere cut
            r_dm = r_dm[dm_mask]  # Dark matter particle positions
            m_dm = m_dm * np.ones(n_dm_tot)  # Total dark matter mass [MSun]
            m_dm = m_dm[dm_mask]  # Dark matter mass [MSun]
        else:
            dm_flag = False
            m_dm = np.zeros(1)
            r_dm = np.zeros((1,3))  # Initialize dark matter particle positions

    # New COM
    tot_mass = np.sum(mass_gas) + np.sum(m_star) + np.sum(m_dm) + np.sum(m_p2) + np.sum(m_p3) # in Msun
    com_new_2v = np.zeros(3)
    for j in range (3):
        com_new_2v[j] = (np.sum(mass_gas * gas_pos[:,j]) + np.sum(m_star * r_star[:,j]) + np.sum(m_dm * r_dm[:,j]) + \
        np.sum(m_p2 * r_p2[:,j]) + np.sum(m_p3 * r_p3[:,j])) / tot_mass     # in cm
    com_kpc_2v = com_new_2v / (a * kpc) * h # Convert to ckpc/h

    # Shifting the positions to the new center of mass and applying a 1 virial radius cut
    r_box = R_virs[i]  # Radial cut = virial radius

    if gas_flag and n_cells > 0:
        r_gas_new = gas_pos - com_new_2v  # Gas position relative to the new center of mass
        gas_mask = (np.sum(r_gas_new**2, axis=1) < r_box**2)  # Sphere cut
        n_cells = np.int32(np.count_nonzero(gas_mask))  # Number of cells
        mass_gas = mass_gas[gas_mask]  # Gas mass [g]
        gas_pos = r_gas_new[gas_mask]  # Gas positions [cm]

    if star_flag and n_stars > 0:
        r_star_new = r_star - com_new_2v  # Star position relative to the new center of mass
        star_mask = (np.sum(r_star_new**2, axis=1) < r_box**2)
        n_stars = np.int32(np.count_nonzero(star_mask))  # Number of star particles
        m_star = m_star[star_mask]  # Star mass [Msun]
        r_star = r_star_new[star_mask]  # Star positions [cm]

    if dm_flag and len(r_dm) > 0:
        r_dm_new = r_dm - com_new_2v  # Dark matter position relative to the new center
        dm_mask = (np.sum(r_dm_new**2, axis=1) < r_box**2)  # Sphere cut
        r_dm = r_dm_new[dm_mask]  # Dark matter particle positions [cm]
        m_dm = m_dm[dm_mask]  # Dark matter mass [g]

    if p2_flag and len(r_p2) > 0:
        r_p2_new = r_p2 - com_new_2v  # Dark matter particle Type 2 position relative to the new center of mass
        p2_mask = (np.sum(r_p2_new**2, axis=1) < r_box**2)  # Sphere cut
        r_p2 = r_p2_new[p2_mask]  # Dark matter particle Type 2 positions [cm]
        m_p2 = m_p2[p2_mask]  # Dark matter particle Type 2 mass [g]

    if  p3_flag and len(r_p3) > 0:
        r_p3_new = r_p3 - com_new_2v  # Dark matter particle Type 3 position relative to the new center of mass
        p3_mask = (np.sum(r_p3_new**2, axis=1) < r_box**2)  # Sphere cut
        r_p3 = r_p3_new[p3_mask]  # Dark matter particle Type 3 positions [cm]
        m_p3 = m_p3[p3_mask]  # Dark matter particle Type 3 mass [g]

    # New COM
    tot_mass = np.sum(mass_gas) + np.sum(m_star) + np.sum(m_dm) + np.sum(m_p2) + np.sum(m_p3) # in g
    com_new = np.zeros(3)
    for j in range (3):
        com_new[j] = (np.sum(mass_gas * gas_pos[:,j]) + np.sum(m_star * r_star[:,j]) + np.sum(m_dm * r_dm[:,j]) + \
        np.sum(m_p2 * r_p2[:,j]) + np.sum(m_p3 * r_p3[:,j])) / tot_mass     # in cm
    com_kpc = com_new / (a * kpc) * h  # Convert to ckpc/h
    com_arepo_box[i] = com_kpc + com_kpc_2v + (r_virs[i]/length_to_cgs) + (r_HRs[i]/length_to_cgs)  # New center of mass in Arepo box units

with h5py.File(f'{colt_dir}_tree/center.hdf5', 'w') as f:
    # Save the new COM
    f.create_dataset('TargetPos', data=com_arepo_box)  # New center of mass [ckpc/h]
    f.create_dataset('Redshifts', data=zs)  # Redshift of the snapshot
    f['TargetPos'].attrs['units'] = b'ckpc/h'
