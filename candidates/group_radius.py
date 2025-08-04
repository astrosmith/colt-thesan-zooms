import numpy as np
import h5py, os, errno, platform
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

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
        raise ValueError('Usage: python group_radius.py [sim] [zoom_dir]')

# Derived global variables
cand_dir = f'{zoom_dir}/{sim}/postprocessing/candidates'
colt_dir = f'{zoom_dir}-COLT/{sim}/ics'
# colt_dir = f'/orcd/data/mvogelsb/005/Lab/Thesan-Zooms-COLT/{sim}/ics'
os.makedirs(f'{colt_dir}_tree', exist_ok=True) # Ensure the new colt directory exists

M_sun = 1.988435e33 # Solar mass in g
pc = 3.085677581467192e18 # Parsec in cm
kpc = 1e3 * pc # Kiloparsec in cm
km = 1e5                    # Units: 1 km = 1e5 cm
Mpc = 3.085677581467192e24  # Units: 1 Mpc = 3e24 cm
G = 6.6743015e-8 # Gravitational constant [cm^3/g/s^2]
USE_200 = True # Use 200 times the critical density for the virial radius

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

def periodic_distance(delta):
    global BoxSize
    """Calculate the periodic distance from the origin."""
    return np.minimum(delta, BoxSize - delta)  # Minimum of delta and its periodic counterpart

def calculate_M_enc(r, m):
    global n_bins
    """Calculate the enclosed mass within a given radius."""
    r = np.atleast_2d(r)  # Ensure r is at least 2-dimensional
    m = np.atleast_1d(m)  # Ensure m is at least 1-dimensional
    dx = periodic_distance(np.abs(r[:,0]))
    dy = periodic_distance(np.abs(r[:,1]))
    dz = periodic_distance(np.abs(r[:,2]))
    r2 = dx**2 + dy**2 + dz**2  # Squared distances
    ibin = np.floor((np.log10(r2) - logr2_min) * inv_dbin).astype(np.int32)  # Bin indices
    ibin[ibin < 0] = 0  # Clip to first bin
    return np.cumsum(np.bincount(ibin, weights=m, minlength=n_bins))  # Enclosed gas mass

print(f'Calculating {colt_dir} ...')
tree_file = f'{zoom_dir}/{sim}/output/tree.hdf5'

with h5py.File(colt_dir + '_tree/center.hdf5', 'r') as f:
    redshift = f['Redshifts'][:]
    TargetPos = f['TargetPos'][:]

_, SmoothPosX = savgol_filter((redshift, TargetPos[:,0]), 11, 3) # Smooth the data
_, SmoothPosY = savgol_filter((redshift, TargetPos[:,1]), 11, 3) # Smooth the data
_, SmoothPosZ = savgol_filter((redshift, TargetPos[:,2]), 11, 3) # Smooth the data
SmoothPos = np.array([SmoothPosX, SmoothPosY, SmoothPosZ]).T

with h5py.File(tree_file, 'r') as f:
    snaps = f['Snapshots'][:] # Snapshots in the tree
    group_ids = f['Group']['GroupID'][:] # Group ID in the tree
    subhalo_ids = f['Subhalo']['SubhaloID'][:] # Subhalo ID in the tree
    R_virs = f['Group']['Group_R_Crit200'][:] # Group virial radii in the tree [ckpc/h]
    zs = f['Redshifts'][:] # Redshifts in the tree

if False:
    mask = np.zeros(len(snaps), dtype=bool)
    mask[-1] = True
    # n_start = 8
    # mask = (snaps >= n_start*189//9) & (snaps <= (n_start+1)*189//9)
    snaps = snaps[mask]
    R_virs = R_virs[mask]
    zs = zs[mask]
    group_ids = group_ids[mask]
    subhalo_ids = subhalo_ids[mask]
    SmoothPos = SmoothPos[-1:,:]

n_snaps = len(snaps)
group_indices = np.empty(n_snaps, dtype=np.int32) # Group index in the candidates
subhalo_indices = np.empty(n_snaps, dtype=np.int32) # Subhalo index in the candidates
r_HRs = np.empty([n_snaps, 3]) # High-resolution center of mass positions [cm]
r_virs = np.empty([n_snaps, 3]) # Group positions [cm]
r_HRs.fill(np.nan) # Fill with NaNs
r_virs.fill(np.nan)
Subhalo_R_vir = np.empty([n_snaps])
Subhalo_R_vir.fill(np.nan)
Subhalo_M_vir = np.empty([n_snaps])
Subhalo_M_vir.fill(np.nan)
Subhalo_M_gas = np.empty([n_snaps])
Subhalo_M_gas.fill(np.nan)
Subhalo_M_stars = np.empty([n_snaps])
Subhalo_M_stars.fill(np.nan)

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
        OmegaLambda = header['OmegaLambda']
        UnitLength_in_cm = header['UnitLength_in_cm']
        UnitMass_in_g = header['UnitMass_in_g']
        UnitVelocity_in_cm_per_s = header['UnitVelocity_in_cm_per_s']
        length_to_cgs = a * UnitLength_in_cm / h
        cand_group_ids = f['Group']['GroupID'][:] # Group ID in the candidates
        cand_subhalo_ids = f['Subhalo']['SubhaloID'][:] # Subhalo ID in the candidates
        group_indices[i] = np.where(cand_group_ids == group_ids[i])[0][0] # Group index in the candidates
        subhalo_indices[i] = np.where(cand_subhalo_ids == subhalo_ids[i])[0][0] # Subhalo index in the candidates
        PosHR = header['PosHR'] # High-resolution center of mass position [BoxUnits]
        r_HRs[i] = length_to_cgs * header['PosHR'] # High-resolution center of mass position [cm]
        r_virs[i] = length_to_cgs * f['Group']['GroupPos'][group_indices[i]] - r_HRs[i] # Group position [cm]
        R_virs[i] = length_to_cgs * R_virs[i] # Virial radius unit conversion [cm]
        GroupPos = f['Group']['GroupPos'][group_indices[i]] # High-resolution center of mass position [BoxUnits]
        # SmoothPos[i] = GroupPos  # Remove comment to verify if the calculated R_vir goes to the one saved in tree file

    # Certain Derived Quantities
    BoxHalf = BoxSize / 2.
    volume_to_cgs = length_to_cgs**3
    mass_to_cgs = UnitMass_in_g / h
    mass_to_msun = mass_to_cgs / M_sun
    density_to_cgs = mass_to_cgs / volume_to_cgs
    velocity_to_cgs = np.sqrt(a) * UnitVelocity_in_cm_per_s

    with h5py.File(colt_file, 'r') as f:
        # Read gas and star coordinates for the radial masks
        GasPos = (f['r'][:] + r_HRs[i]) / length_to_cgs - SmoothPos[i]  # Gas position [ckpc/h] [centered on SmoothPos]
        n_cells = f.attrs['n_cells'] # Number of cells
        gas_rho = f['rho'][:]  # Gas density [g/cm^3]
        gas_rho[~f['is_HR'][:]] = 0.  # Set gas density to 0 outside the high-resolution region
        gas_vol = f['V'][:]  # Cell volume [cm^3]
        gas_mass = (gas_rho[:] * gas_vol[:]) / M_sun  # Gas mass [Msun]
        GasMass = gas_mass / mass_to_cgs * M_sun  # Convert to code units [BoxUnits]

        # Star Properties
        if 'r_star' in f:
            StarPos = (f['r_star'][:] + r_HRs[i]) / length_to_cgs - SmoothPos[i]  # Star position [ckpc/h] [centered on SmoothPos]
            n_stars = f.attrs['n_stars']
            starmass = f['m_init_star'][:]  # Star mass [Msun]
            StarMass = starmass / mass_to_msun  # Star mass [CodeUnits]
        else:
            n_stars = 0.
            starpos = StarPos = np.zeros((1,3))
            starmass = StarMass = np.zeros(1)

    with h5py.File(dm_file, 'r') as f:
        if 'r_p2' in f:
            P2Pos = (f['r_p2'][:] + r_HRs[i]) / length_to_cgs - SmoothPos[i]  # Dark matter particle positions Type 2 [ckpc/h] [centered on SmoothPos]
            p2_mass = f['m_p2'][:]  # Mass of dark matter particles Type 2 [Msun]
            n_p2_tot = f.attrs['n_p2']
            P2_Mass = p2_mass / mass_to_msun  # Mass of dark  matter particles Type 2 [CodeUnits]
        else:
            n_p2_tot = 0
            p2_mass = P2_Mass = np.zeros(1)  # Initialize dark matter particle Type 2
            p2pos = P2Pos = np.zeros((1,3))  # Initialize dark matter particle positions Type 2

        if 'r_p3' in f:
            P3Pos = (f['r_p3'][:] + r_HRs[i]) / length_to_cgs - SmoothPos[i]  # Dark matter particle positions Type 3 [ckpc/h] [centered on SmoothPos]
            p3mass = f.attrs['m_p3']
            n_p3_tot = f.attrs['n_p3']  # Total number of dark matter particles Type 3 [MSun]
            p3_mass = p3mass * np.ones(n_p3_tot)        # Mass Array in Msun
            P3_Mass = p3_mass / mass_to_msun          # Mass Array in CodeUnits
        else:
            n_p3_tot = 0
            p3_mass = P3_Mass = np.zeros(1)  # Initialize dark matter particle Type 3
            p3pos =  P3Pos = np.zeros((1,3))  # Initialize dark matter particle positions Type 3

        if 'r_dm' in f:
            n_dm_tot = f.attrs['n_dm']  # Total number of dark matter particles
            dm_mass = f.attrs['m_dm']      # Mass of dark matter particles [MSun]
            DMPos = (f['r_dm'][:] + r_HRs[i]) / length_to_cgs - SmoothPos[i]  # Dark matter particle positions [ckpc/h] [centered on SmoothPos]
            dm_mass = dm_mass * np.ones(n_dm_tot)  # Total dark matter mass [MSun]
            DM_Mass = dm_mass / mass_to_msun        # Total DM mass [Code units]
        else:
            n_dm_tot=0
            dm_mass = DM_Mass = np.zeros(1)
            dmpos = DMPos = np.zeros((1,3)) # Initialize dark matter particle positions

    # Calculating Average Density
    H0 = 100. * h * km / Mpc # Hubble constant [km/s/Mpc]
    H2 = H0**2 * (Omega0 / a**3 + OmegaLambda) # Hubble parameter squared
    x_c = Omega0 - 1. # x_c = Omega0 - 1
    tot_mass = np.sum(GasMass) + np.sum(StarMass) + np.sum(DM_Mass) + np.sum(P2_Mass) + np.sum(P3_Mass) # in Msun
    rho_avg = tot_mass / BoxSize**3 # Total density [MSun/ (ckpc/h)^3]
    Delta_c = 200. if USE_200 else 18. * np.pi**2 + 82. * x_c - 39. * x_c**2 # Critical overdensity factor
    rho_crit0 = 3. * H0**2 / (8. * np.pi * G) # Critical density today [g/cm^3]
    rho_crit = 3. * H2 / (8. * np.pi * G) # Critical density [g/cm^3]
    rho_vir = Delta_c * rho_crit # Virial density [g/cm^3]
    rho_vir_code = rho_vir / density_to_cgs # Virial density in code units

    # R_vir Setup
    r_min = 1e-2 * h / a  # 10 pc
    r_max = BoxSize  # r_box
    logr2_min = 2. * np.log10(r_min)  # log(r_min^2)
    logr2_max = 2. * np.log10(r_max)  # log(r_max^2)
    n_bins = 10000  # Number of bins
    inv_dbin = float(n_bins) / (logr2_max - logr2_min)  # n / (log(r_max^2) - log(r_min^2))
    bins = np.logspace(logr2_min, logr2_max, n_bins+1)  # r^2 bins
    log_rbins = np.linspace(0.5*logr2_min, 0.5*logr2_max, n_bins+1)  # log(r) bins
    dlog_rbin = 0.5 * (logr2_max - logr2_min) / float(n_bins)  # (log(r_max) - log(r_min)) / n
    bins[0] = 0.  # Extend first bin to zero
    V_enc = 4. * np.pi / 3. * bins[1:]**1.5 # Enclosed volume
    M_to_rho_vir = 1. / V_enc / rho_vir_code # Mass to virial density

    # Calculate the enclosed masses
    M_gas_enc = calculate_M_enc(GasPos, GasMass)  # Enclosed gas mass
    M_dm_enc = calculate_M_enc(DMPos, DM_Mass) if n_dm_tot > 0 else np.zeros(n_bins)  # Enclosed dark matter mass
    M_p2_enc = calculate_M_enc(P2Pos, P2_Mass) if n_p2_tot > 0 else np.zeros(n_bins)  # Enclosed PartType2 mass
    M_p3_enc = calculate_M_enc(P3Pos, P3_Mass) if n_p3_tot > 0 else np.zeros(n_bins)  # Enclosed PartType3 mass
    M_stars_enc = calculate_M_enc(StarPos, StarMass) if n_stars > 0 else np.zeros(n_bins)  # Enclosed stellar mass

    # Calculate the enclosed mass, density, and virial radius
    M_enc = M_gas_enc + M_dm_enc + M_p2_enc + M_p3_enc + M_stars_enc  # Total enclosed mass
    rho_enc = M_enc * M_to_rho_vir # Enclosed density [rho_vir]
    i_vir = np.where(rho_enc > 1)[0][-1] + 1  # Find the last bin with rho_enc > 1
    # Log interpolation to find the virial radius and masses
    frac = -np.log10(rho_enc[i_vir-1]) / np.log10(rho_enc[i_vir]/rho_enc[i_vir-1]) # Interpolation coordinate
    Subhalo_R_vir[i] = 10.**(log_rbins[i_vir] + frac * dlog_rbin) # Virial radius
    Subhalo_M_vir[i] = 10.**(np.log10(M_enc[i_vir-1]) + frac * np.log10(M_enc[i_vir]/M_enc[i_vir-1])) # Virial mass
    if M_gas_enc[i_vir-1] <= 0.:
        Subhalo_M_gas[i] = M_gas_enc[i_vir-1] + frac * (M_gas_enc[i_vir] - M_gas_enc[i_vir-1]) # Gas mass (<R_vir)
    else:
        Subhalo_M_gas[i] = 10.**(np.log10(M_gas_enc[i_vir-1]) + frac * np.log10(M_gas_enc[i_vir]/M_gas_enc[i_vir-1])) # Gas mass (<R_vir)
    if M_stars_enc[i_vir-1] <= 0.:
        Subhalo_M_stars[i] = M_stars_enc[i_vir-1] + frac * (M_stars_enc[i_vir] - M_stars_enc[i_vir-1]) # Stellar mass (<R_vir)
    else:
        Subhalo_M_stars[i] = 10.**(np.log10(M_stars_enc[i_vir-1]) + frac * np.log10(M_stars_enc[i_vir]/M_stars_enc[i_vir-1])) # Stellar mass (<R_vir)

print('Writing to ' + colt_dir + '_tree/center.hdf5')
with h5py.File(colt_dir + '_tree/center.hdf5', 'r+') as f:
    for key in ['TargetPosSmooth', 'R_Crit200_Smooth']:
        if key in f.keys(): del f[key]  # Remove previous data
    f.create_dataset(name='TargetPosSmooth', data=SmoothPos)
    f.create_dataset(name='R_Crit200_Smooth', data=Subhalo_R_vir)
    f['R_Crit200_Smooth'].attrs['units'] = b'ckpc/h'
