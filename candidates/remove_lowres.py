import numpy as np
from time import time
import h5py, os, platform
from dataclasses import dataclass, field

VERBOSITY = 0 # Level of print verbosity
TIMERS = True # Print timers

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
        raise ValueError('Usage: python remove_lowres.py [sim] [snap] [zoom_dir]')

colt_dir = f'{zoom_dir}-COLT/{sim}/ics'

# Overwrite for local testing
#colt_dir = '.' # Overwrite for local testing

# The following paths should not change
colt_file = f'{colt_dir}/colt_{snap:03d}.hdf5'

# Extracted fields
gas_fields = ['D', 'D_Si', 'SFR', 'T_dust', 'X', 'Y', 'Z', 'Z_C', 'Z_Fe', 'Z_Mg', 'Z_N', 'Z_Ne', 'Z_O', 'Z_S', 'Z_Si',
              'e_int', 'is_HR', 'r', 'rho', 'v', 'x_H2', 'x_HI', 'x_HeI', 'x_HeII', 'x_e', 'id', 'group_id', 'subhalo_id']
star_fields = ['Z_star', 'age_star', 'm_star', 'm_init_star', 'r_star', 'v_star', 'id_star', 'group_id_star', 'subhalo_id_star']
units = {'r': b'cm', 'v': b'cm/s', 'e_int': b'cm^2/s^2', 'T_dust': b'K', 'rho': b'g/cm^3', 'SFR': b'Msun/yr',
         'r_star': b'cm', 'v_star': b'cm/s', 'm_star': b'Msun', 'm_init_star': b'Msun', 'age_star': b'Gyr'}


# Emperical unit definitions
pc = 3.085677581467192e18  # Units: 1 pc  = 3e18 cm
kpc = 1e3 * pc             # Units: 1 kpc = 3e21 cm
Mpc = 1e6 * pc             # Units: 1 Mpc = 3e24 cm

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
    n_neighbors_tot: np.int32 = None # Total number of cell neighbors
    edges: np.ndarray = None # Edge indices
    neighbor_offsets: np.ndarray = None # Cumulative n_neighbor counts
    neighbor_cells: np.ndarray = None # Neighbor cell indices (flat)
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
            self.neighbor_offsets = f['neighbor_indices'][:] # Cumulative n_neighbor counts
            self.neighbor_cells = f['neighbor_indptr'][:] # Neighbor cell indices (flat)

            # Gas data
            for field in gas_fields:
                self.gas[field] = f[field][:]
            self.is_HR = self.gas['is_HR'] # High-resolution gas mask

            # Star data
            if self.n_stars > 0:
                for field in star_fields:
                    self.stars[field] = f[field][:]

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
    print(f'Average number of neighbors = {np.mean(np.diff(sim.neighbor_offsets)):g}')
    known_LR = set() # Known low-resolution outer cells
    queue_LR = set(sim.edges[~sim.is_HR[sim.edges]]) # Queue low-resolution edges
    print(f'Number of low-resolution outer edges = {len(queue_LR)} / {len(sim.edges)} = {100.*float(len(queue_LR))/float(len(sim.edges)):g}%')
    print('Processing outer low-resolution cells:', end='')
    while len(queue_LR) > 0:
        print(f' {len(queue_LR)}', end='', flush=True)
        added_LR = set() # Added low-resolution neighbors
        for cell in queue_LR:
            neighbors = sim.neighbor_cells[sim.neighbor_offsets[cell]:sim.neighbor_offsets[cell+1]] # Neighbors
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
    sim.n_cells = n_cells # Update number of cells
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
        if sim.n_stars > 0:
            for field in star_fields:
                f.create_dataset(field, data=sim.stars[field])
                if field in units: f[field].attrs['units'] = units[field]
    try:
        os.rename(colt_file + '-tmp', colt_file)
    except OSError as e:
        raise OSError(f"Error renaming file: {e}")

if __name__ == '__main__':
    remove_lowres()

