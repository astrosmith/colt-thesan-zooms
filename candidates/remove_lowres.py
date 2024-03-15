import numpy as np
from time import time
import h5py, os, errno
from dataclasses import dataclass, field

VERBOSITY = 2 # Level of print verbosity
TIMERS = True # Print timers

# Global variables
sim = 'g5760/z4'
snap = 188 # Snapshot number
zoom_dir = '/net/hstor001.ib/data2/group/mvogelsb/004/Thesan-Zooms'
colt_dir = f'{zoom_dir}-COLT/{sim}/ics'

# Overwrite for local testing
#colt_dir = '.' # Overwrite for local testing

# The following paths should not change
colt_file = f'{colt_dir}/colt_{snap:03d}.hdf5'

# Extracted fields
gas_fields = ['D', 'SFR', 'T_dust', 'V', 'X', 'Y', 'Z', 'Z_C', 'Z_Fe', 'Z_Mg', 'Z_N', 'Z_Ne', 'Z_O', 'Z_S', 'Z_Si',
              'e_int', 'id', 'is_HR', 'r', 'rho', 'v', 'x_H2', 'x_HI', 'x_HeI', 'x_HeII', 'x_e']
star_fields = ['Z_star', 'age_star', 'id_star', 'm_init_star', 'r_star', 'v_star']
units = {'r': b'cm', 'v': b'cm/s', 'e_int': b'cm^2/s^2', 'T_dust': b'K', 'rho': b'g/cm^3', 'SFR': b'Msun/yr',
         'r_star': b'cm', 'v_star': b'cm/s', 'm_star': b'Msun', 'm_init_star': b'Msun', 'age_star': b'Gyr'}

# Universal constants
# c = 2.99792458e10          # Speed of light [cm/s]
# kB = 1.380648813e-16       # Boltzmann's constant [g cm^2/s^2/K]
# h = 6.626069573e-27        # Planck's constant [erg/s]
# mH = 1.6735327e-24         # Mass of hydrogen atom [g]
# me = 9.109382917e-28       # Electron mass [g]
# ee = 4.80320451e-10        # Electron charge [g^(1/2) cm^(3/2) / s]

# Emperical unit definitions
# Msun = 1.988435e33         # Solar mass [g]
# Lsun = 3.839e33            # Solar luminosity [erg/s]
# Zsun = 0.0134              # Solar metallicity (mass fraction)
# arcsec = 648000. / np.pi   # arseconds per radian
pc = 3.085677581467192e18  # Units: 1 pc  = 3e18 cm
kpc = 1e3 * pc             # Units: 1 kpc = 3e21 cm
Mpc = 1e6 * pc             # Units: 1 Mpc = 3e24 cm
# km = 1e5                   # Units: 1 km  = 1e5  cm
# angstrom = 1e-8            # Units: 1 angstrom = 1e-8 cm
# day = 86400.               # Units: 1 day = 24 * 3600 seconds
# yr = 365.24 * day          # Units: 1 year = 365.24 days
# kyr = 1e3 * yr             # Units: 1 Myr = 10^6 yr
# Myr = 1e6 * yr             # Units: 1 Myr = 10^6 yr

# GAS_HIGH_RES_THRESHOLD = 0.5  # Threshold deliniating high and low resolution gas particles
# SOLAR_MASS = 1.989e33         # Solar masses
# BOLTZMANN = 1.38065e-16       # Boltzmann's constant [g cm^2/sec^2/k]
# PLANCK = 6.6260695e-27        # Planck's constant [erg sec]
# PROTONMASS = 1.67262178e-24   # Mass of hydrogen atom [g]
# HYDROGEN_MASSFRAC = 0.76      # Mass fraction of hydrogen
# GAMMA = 5. / 3.               # Adiabatic index of simulated gas
# GAMMA_MINUS1 = GAMMA - 1.     # For convenience
# HUBBLE = 3.2407789e-18        # Hubble constant [h/sec]
# SEC_PER_GIGAYEAR = 3.15576e16 # Seconds per gigayear
# HE_ABUND = (1./HYDROGEN_MASSFRAC - 1.) / 4. # Helium abundance = n_He / n_H

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

@dataclass
class Simulation:
    """Simulation information and data."""
    n_cells: np.int32 = None # Total number of gas cells
    n_stars: np.int32 = None # Total number of star particles
    r_box: np.float64 = None # Bounding box radius [cm]
    redshift: np.float64 = None # Current simulation redshift
    h100: np.float64 = None # Hubble parameter
    Omega0: np.float64 = None # Matter density
    OmegaB: np.float64 = None # Baryon density

    # Connection data
    n_edges: np.int32 = None # Total number of edge cells
    # n_inner_edges: np.int32 = None # Total number of inner edge cells
    n_neighbors_tot: np.int32 = None # Total number of cell neighbors
    # n_circulators_tot: np.int32 = None # Total number of face circulators
    edges: np.ndarray = None # Edge indices
    # inner_edges: np.ndarray = None # Inner edge indices
    neighbor_indices: np.ndarray = None # Neighbor indices
    neighbor_indptr: np.ndarray = None # Neighbor index pointers
    # circulator_indices: np.ndarray = None # Circulator indices
    # circulator_indptr: np.ndarray = None # Circulator index pointers
    is_HR: np.ndarray = None # High-resolution gas mask

    # Particle data
    gas: dict = field(default_factory=dict)
    stars: dict = field(default_factory=dict)

    def __post_init__(self):
        """Allocate memory for gas and star arrays."""
        # Read header info from the snapshot files
        with h5py.File(colt_file, 'r') as f:
            self.n_cells = f.attrs['n_cells'] # Number of cells
            self.n_stars = f.attrs['n_stars'] # Number of star particles
            self.r_box = f.attrs['r_box'] # Bounding box radius [cm]
            self.redshift = f.attrs['redshift'] # Current simulation redshift
            self.h100 = f.attrs['h100'] # Hubble parameter
            self.Omega0 = f.attrs['Omega0'] # Matter density
            self.OmegaB = f.attrs['OmegaB'] # Baryon density

            # Read connection data
            self.n_edges = f.attrs['n_edges'] # Number of edges
            self.n_neighbors_tot = f.attrs['n_neighbors_tot'] # Total number of cell neighbors
            self.edges = f['edges'][:] # Edge indices
            self.neighbor_indices = f['neighbor_indices'][:] # Neighbor indices
            self.neighbor_indptr = f['neighbor_indptr'][:] # Neighbor index pointers

            # Gas data
            for field in gas_fields:
                self.gas[field] = f[field][:]

            # Star data
            for field in star_fields:
                self.stars[field] = f[field][:]
            self.is_HR = self.gas['is_HR'] # High-resolution gas mask

def remove_lowres():
    # Setup simulation parameters
    if TIMERS: t1 = time()
    sim = Simulation()
    print(' ___       ___  __             \n'
          '  |  |__| |__  /__`  /\\  |\\ |\n'
          '  |  |  | |___ .__/ /--\\ | \\|\n' +
          f'\nFile: {colt_file}' +
          f'\nSnap {snap}: n_cells = {sim.n_cells}, n_stars = {sim.n_stars}' +
          f'\nz = {sim.redshift:g}, h = {sim.h100:g}, r_box = {sim.r_box/pc:g} Mpc\n')
    if TIMERS: t2 = time(); print(f"Time to setup simulation: {t2 - t1:g} s"); t1 = t2

    # Identify edge-connected low-resolution gas particles
    n_cells_old = np.int32(sim.n_cells) # Number of cells (saved for comparison)
    has_HR_neighbor = np.zeros(sim.n_cells, dtype=bool) # Flag cells with high-resolution neighbors
    has_HR_neibneib = np.zeros(sim.n_cells, dtype=bool) # Flag cells with high-resolution neighbors of neighbors
    print(f'Average number of neighbors = {np.mean(np.diff(sim.neighbor_indices)):g}')
    known_LR = set() # Known low-resolution outer cells
    queue_LR = set(sim.edges[~sim.is_HR[sim.edges]]) # Queue low-resolution edges
    print(f'Number of low-resolution outer edges = {len(queue_LR)} / {len(sim.edges)} = {100.*float(len(queue_LR))/float(len(sim.edges)):g}%')
    print('Processing outer low-resolution cells:', end='')
    while len(queue_LR) > 0:
        print(f' {len(queue_LR)}', end='', flush=True)
        added_LR = set() # Added low-resolution neighbors
        for cell in queue_LR:
            neighbors = sim.neighbor_indptr[sim.neighbor_indices[cell]:sim.neighbor_indices[cell+1]] # Neighbors
            neighbors_HR = neighbors[sim.is_HR[neighbors]] # High-resolution neighbors
            if len(neighbors_HR) > 0:
                has_HR_neighbor[cell] = True # Flag cells with high-resolution neighbors
                has_HR_neibneib[neighbors] = True # Flag cells with high-resolution neighbors of neighbors
            neighbors_LR = neighbors[~sim.is_HR[neighbors]] # Low-resolution neighbors
            added_LR.update(neighbors_LR) # Add low-resolution neighbors
        known_LR.update(queue_LR) # Add queue low-resolution neighbors to known
        queue_LR = added_LR - known_LR # Set of new low-resolution neighbors
    print(f'\nNumber of outer low-resolution cells = {len(known_LR)} / {sim.n_cells} = {100.*float(len(known_LR))/float(sim.n_cells):g}%')
    known_LR -= set(np.where(has_HR_neighbor)[0]) # Remove cells with high-resolution neighbors
    print(f'Number of outer low-resolution cells without high-resolution neighbors = {len(known_LR)} / {sim.n_cells} = {100.*float(len(known_LR))/float(sim.n_cells):g}%')
    known_LR -= set(np.where(has_HR_neibneib)[0]) # Remove cells with high-resolution neighbors of neighbors
    print(f'Number of outer low-resolution cells without high-resolution neighbors of neighbors = {len(known_LR)} / {sim.n_cells} = {100.*float(len(known_LR))/float(sim.n_cells):g}%')
    mask = np.ones(sim.n_cells, dtype=bool) # Mask for cells to keep
    mask[list(known_LR)] = False # Remove outer low-resolution cells without high-resolution neighbors
    n_cells = np.int32(np.count_nonzero(mask)) # Number of cells
    for field in sim.gas: sim.gas[field] = sim.gas[field][mask] # Apply gas mask
    print(f'Number of cells = {n_cells} / {n_cells_old} = {100.*float(n_cells)/float(n_cells_old):g}%')
    if TIMERS: t2 = time(); print(f"Time to identify edge-connected low-resolution gas particles: {t2 - t1:g} s"); t1 = t2

    with h5py.File(colt_file + '-tmp', 'w') as f:
        # Simulation properties
        f.attrs['n_cells'] = sim.n_cells # Number of cells
        f.attrs['n_stars'] = sim.n_stars # Number of star particles
        f.attrs['redshift'] = sim.redshift # Current simulation redshift
        f.attrs['Omega0'] = sim.Omega0 # Matter density [rho_crit_0]
        f.attrs['OmegaB'] = sim.OmegaB # Baryon density [rho_crit_0]
        f.attrs['h100'] = sim.h100 # Hubble constant [100 km/s/Mpc]
        f.attrs['r_box'] = sim.r_box # Bounding box radius [cm]

        # Gas fields
        for field in gas_fields:
            f.create_dataset(field, data=sim.gas[field])
            if field in units: f[field].attrs['units'] = units[field]

        # Star fields
        for field in star_fields:
            f.create_dataset(field, data=sim.stars[field])
            if field in units: f[field].attrs['units'] = units[field]

if __name__ == '__main__':
    # import sys
    # if len(sys.argv) == 3:
    #     snap, out_dir = int(sys.argv[1]), sys.argv[2]
    # else:
    #     raise ValueError('Usage: python arepo_to_colt.py snap out_dir')
    # arepo_to_colt(snap=snap, out_dir=out_dir)
    remove_lowres()

