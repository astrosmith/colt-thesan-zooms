import h5py, os, platform, asyncio
import numpy as np
from numba import jit, prange
from dataclasses import dataclass, field
from time import time
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from scipy.spatial import cKDTree

# Constants
NUM_PART = 7 # Number of particle types
GAS_HIGH_RES_THRESHOLD = 0.5 # Threshold deliniating high and low resolution gas particles
SOLAR_MASS = 1.989e33  # Solar masses
kpc = 3.085677581467192e21  # Units: 1 kpc = 3e21 cm
VERBOSITY = 0 # Level of print verbosity
MAX_WORKERS = cpu_count() # Maximum number of workers
SERIAL = 1 # Run in serial
ASYNCIO = 2 # Run in parallel (asyncio)
NUMPY = 1 # Use NumPy
NUMBA = 2 # Use Numba
READ_DEFAULT = (ASYNCIO,) # Default read method
# READ_DEFAULT = (SERIAL,ASYNCIO) # Default read method
CALC_DEFAULT = (NUMPY,) # Calculate center of mass
# CALC_DEFAULT = (NUMPY, NUMBA) # Calculate center of mass
READ_COUNTS = READ_DEFAULT # Read counts methods
READ_GROUPS = READ_DEFAULT # Read groups methods
READ_SNAPS = READ_DEFAULT # Read snapshots methods
CALC_COM = CALC_DEFAULT # Calculate center of mass methods
# USE_200, USE_ALL_PARTICLES = True, False # Limit calculations to groups (Only for testing)
USE_200, USE_ALL_PARTICLES = True, True # Limit calculations to groups (Most consistent with Arepo)
# USE_200, USE_ALL_PARTICLES = False, True # Limit calculations to groups (Some virial radii are too large)
RADIAL_PLOTS = False # Plot results
VIRIAL_PLOTS = False # Plot results
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
        raise ValueError('Usage: python distances.py [sim] [snap] [zoom_dir]')

out_dir = f'{zoom_dir}/{sim}/output'
dist_dir = f'{zoom_dir}/{sim}/postprocessing/distances'

# Overwrite for local testing
#out_dir = '.' # Overwrite for local testing
#dist_dir = '.' # Overwrite for local testing

# The following paths should not change
fof_pre = f'{out_dir}/groups_{snap:03d}/fof_subhalo_tab_{snap:03d}.'
snap_pre = f'{out_dir}/snapdir_{snap:03d}/snapshot_{snap:03d}.'
dist_file = f'{dist_dir}/distances_{snap:03d}.hdf5'

@dataclass
class Simulation:
    """Simulation information and data."""
    n_files: np.int32 = 0 # Number of files
    n_groups_tot: np.uint64 = None # Total number of groups
    n_subhalos_tot: np.uint64 = None # Total number of subhalos
    n_gas_tot: np.uint64 = None # Total number of gas particles
    n_dm_tot: np.uint64 = None # Total number of PartType1 particles
    n_p2_tot: np.uint64 = None # Total number of PartType2 particles
    n_p3_tot: np.uint64 = None # Total number of PartType3 particles
    n_stars_tot: np.uint64 = None # Total number of star particles
    a: np.float64 = None # Scale factor
    BoxSize: np.float64 = None # Size of the simulation volume
    BoxHalf: np.float64 = None # Half the size of the simulation volume
    h: np.float64 = None # Hubble parameter
    Omega0: np.float64 = None # Matter density
    OmegaBaryon: np.float64 = None # Baryon density
    OmegaLambda: np.float64 = None # Dark energy density
    UnitLength_in_cm: np.float64 = None # Unit length in cm
    UnitMass_in_g: np.float64 = None # Unit mass in g
    UnitVelocity_in_cm_per_s: np.float64 = None # Unit velocity in cm/s
    length_to_cgs: np.float64 = None # Conversion factor for length to cgs
    volume_to_cgs: np.float64 = None # Conversion factor for volume to cgs
    mass_to_cgs: np.float64 = None # Conversion factor for mass to cgs
    mass_to_msun: np.float64 = None # Conversion factor for mass to solar masses
    velocity_to_cgs: np.float64 = None # Conversion factor for velocity to cgs

    # Derived quantities
    n_gas_com: np.uint64 = None # Total number of gas particles
    m_gas_com: np.float64 = None # Total mass of gas particles
    r_gas_com: np.ndarray = None # Center of mass of gas particles
    n_dm_com: np.uint64 = None # Total number of dark matter particles
    m_dm_com: np.float64 = None # Total mass of dark matter particles
    r_dm_com: np.ndarray = None # Center of mass of dark matter particles
    n_stars_com: np.uint64 = None # Total number of star particles
    m_stars_com: np.float64 = None # Total mass of star particles
    r_stars_com: np.ndarray = None # Center of mass of star particles
    m_com: np.float64 = None # Total mass of high-resolution particles
    r_com: np.ndarray = None # Center of mass of high-resolution particles

    # File counts and offsets
    n_groups: np.ndarray = None
    n_subhalos: np.ndarray = None
    n_gas: np.ndarray = None
    n_dm: np.ndarray = None
    n_p2: np.ndarray = None
    n_p3: np.ndarray = None
    n_stars: np.ndarray = None
    first_group: np.ndarray = None
    first_subhalo: np.ndarray = None
    first_gas: np.ndarray = None
    first_dm: np.ndarray = None
    first_p2: np.ndarray = None
    first_p3: np.ndarray = None
    first_star: np.ndarray = None

    # FOF data
    groups: dict = field(default_factory=dict)
    GroupPos: np.ndarray = None
    Group_R_Crit200: np.ndarray = None
    Group_M_Crit200: np.ndarray = None
    GroupMassType: np.ndarray = None
    GroupNsubs: np.ndarray = None
    GroupLenType: np.ndarray = None
    subhalos: dict = field(default_factory=dict)
    SubhaloPos: np.ndarray = None
    SubhaloMass: np.ndarray = None
    SubhaloGroupNr: np.ndarray = None
    SubhaloLenType: np.ndarray = None

    # Particle data
    gas: dict = field(default_factory=dict)
    dm: dict = field(default_factory=dict)
    p2: dict = field(default_factory=dict)
    p3: dict = field(default_factory=dict)
    stars: dict = field(default_factory=dict)
    r_gas: np.ndarray = None
    m_gas: np.ndarray = None
    m_gas_HR: np.ndarray = None
    r_dm: np.ndarray = None
    m_dm: np.float64 = None
    r_p2: np.ndarray = None
    m_p2: np.ndarray = None
    r_p3: np.ndarray = None
    m_p3: np.float64 = None
    r_stars: np.ndarray = None
    m_stars: np.ndarray = None
    is_HR: np.ndarray = None

    def __post_init__(self):
        """Allocate memory for group and particle data."""
        # Read header info from the snapshot files
        with h5py.File(fof_pre + '0.hdf5', 'r') as f:
            header = f['Header'].attrs
            self.n_files = header['NumFiles']
            self.n_groups_tot = header['Ngroups_Total']
            self.n_subhalos_tot = header['Nsubhalos_Total']
            self.n_groups = np.zeros(self.n_files, dtype=np.uint64)
            self.n_subhalos = np.zeros(self.n_files, dtype=np.uint64)
            if self.n_groups_tot > 0:
                g = f['Group']
                for field in ['GroupPos', 'Group_R_Crit200', 'Group_M_Crit200', 'GroupMassType', 'GroupNsubs', 'GroupLenType']:
                    shape, dtype = g[field].shape, g[field].dtype
                    shape = (self.n_groups_tot,) + shape[1:]
                    self.groups[field] = np.empty(shape, dtype=dtype)
            if self.n_subhalos_tot > 0:
                g = f['Subhalo']
                for field in ['SubhaloPos', 'SubhaloMass', 'SubhaloGroupNr', 'SubhaloLenType']:
                    shape, dtype = g[field].shape, g[field].dtype
                    shape = (self.n_subhalos_tot,) + shape[1:]
                    self.subhalos[field] = np.empty(shape, dtype=dtype)

        with h5py.File(snap_pre + '0.hdf5', 'r') as f:
            header = f['Header'].attrs
            self.m_dm = header['MassTable'][1] # Mass of dark matter particles
            self.m_p3 = header['MassTable'][3] # Mass of PartType3 particles
            self.n_gas = np.zeros(self.n_files, dtype=np.uint64)
            self.n_dm = np.zeros(self.n_files, dtype=np.uint64)
            self.n_p2 = np.zeros(self.n_files, dtype=np.uint64)
            self.n_p3 = np.zeros(self.n_files, dtype=np.uint64)
            self.n_stars = np.zeros(self.n_files, dtype=np.uint64)
            n_tot = header['NumPart_Total']
            self.n_gas_tot = n_tot[0]
            self.n_dm_tot = n_tot[1]
            self.n_p2_tot = n_tot[2]
            self.n_p3_tot = n_tot[3]
            self.n_stars_tot = n_tot[4]
            self.a = header['Time']
            params = f['Parameters'].attrs
            self.BoxSize = params['BoxSize']
            self.h = params['HubbleParam']
            self.Omega0 = params['Omega0']
            self.OmegaBaryon = params['OmegaBaryon']
            self.OmegaLambda = params['OmegaLambda']
            self.UnitLength_in_cm = params['UnitLength_in_cm']
            self.UnitMass_in_g = params['UnitMass_in_g']
            self.UnitVelocity_in_cm_per_s = params['UnitVelocity_in_cm_per_s']

        # Gas data
        for i in range(self.n_files):
            with h5py.File(snap_pre + f'{i}.hdf5', 'r') as f:
                if 'PartType0' in f:
                    g = f['PartType0']
                    for field in ['Coordinates', 'Masses', 'HighResGasMass']:
                        shape, dtype = g[field].shape, g[field].dtype
                        shape = (self.n_gas_tot,) + shape[1:]
                        self.gas[field] = np.empty(shape, dtype=dtype)
                    break # Found gas data

        # PartType1 data
        for i in range(self.n_files):
            with h5py.File(snap_pre + f'{i}.hdf5', 'r') as f:
                if 'PartType1' in f:
                    g = f['PartType1']
                    for field in ['Coordinates']:
                        shape, dtype = g[field].shape, g[field].dtype
                        shape = (self.n_dm_tot,) + shape[1:]
                        self.dm[field] = np.empty(shape, dtype=dtype)
                    break # Found PartType1 data

        # PartType2 data
        for i in range(self.n_files):
            with h5py.File(snap_pre + f'{i}.hdf5', 'r') as f:
                if 'PartType2' in f:
                    g = f['PartType2']
                    for field in ['Coordinates', 'Masses']:
                        shape, dtype = g[field].shape, g[field].dtype
                        shape = (self.n_p2_tot,) + shape[1:]
                        self.p2[field] = np.empty(shape, dtype=dtype)
                    break # Found PartType2 data

        # PartType3 data
        for i in range(self.n_files):
            with h5py.File(snap_pre + f'{i}.hdf5', 'r') as f:
                if 'PartType0' in f:
                    g = f['PartType3']
                    for field in ['Coordinates']:
                        shape, dtype = g[field].shape, g[field].dtype
                        shape = (self.n_p3_tot,) + shape[1:]
                        self.p3[field] = np.empty(shape, dtype=dtype)
                    break # Found PartType3 data

        # Star data
        if self.n_stars_tot > 0:
            for i in range(self.n_files):
                with h5py.File(snap_pre + f'{i}.hdf5', 'r') as f:
                    if 'PartType4' in f:
                        g = f['PartType4']
                        for field in ['Coordinates', 'Masses', 'IsHighRes']:
                            shape, dtype = g[field].shape, g[field].dtype
                            shape = (self.n_stars_tot,) + shape[1:]
                            self.stars[field] = np.empty(shape, dtype=dtype)
                        break # Found star data

        # Derived quantities
        self.BoxHalf = self.BoxSize / 2.
        self.length_to_cgs = self.a * self.UnitLength_in_cm / self.h
        self.volume_to_cgs = self.length_to_cgs**3
        self.mass_to_cgs = self.UnitMass_in_g / self.h
        self.mass_to_msun = self.mass_to_cgs / SOLAR_MASS
        self.density_to_cgs = self.mass_to_cgs / self.volume_to_cgs
        self.velocity_to_cgs = np.sqrt(self.a) * self.UnitVelocity_in_cm_per_s

        # Group data
        if self.n_groups_tot > 0:
            self.GroupPos = self.groups['GroupPos']
            self.Group_R_Crit200 = self.groups['Group_R_Crit200']
            self.Group_M_Crit200 = self.groups['Group_M_Crit200']
            self.GroupMassType = self.groups['GroupMassType']
            self.GroupNsubs = self.groups['GroupNsubs']
            self.GroupLenType = self.groups['GroupLenType']

        # Subhalo data
        if self.n_subhalos_tot > 0:
            self.SubhaloPos = self.subhalos['SubhaloPos']
            self.SubhaloMass = self.subhalos['SubhaloMass']
            self.SubhaloGroupNr = self.subhalos['SubhaloGroupNr']
            self.SubhaloLenType = self.subhalos['SubhaloLenType']

        # Gas data
        self.r_gas = self.gas['Coordinates']
        self.m_gas = self.gas['Masses']
        self.m_gas_HR = self.gas['HighResGasMass']

        # PartType1 data
        self.r_dm = self.dm['Coordinates']

        # PartType2 data
        self.r_p2 = self.p2['Coordinates']
        self.m_p2 = self.p2['Masses']

        # PartType3 data
        self.r_p3 = self.p3['Coordinates']

        # Star data
        if self.n_stars_tot > 0:
            self.r_stars = self.stars['Coordinates']
            self.m_stars = self.stars['Masses']
            self.is_HR = self.stars['IsHighRes']

    def convert_counts(self):
        """Convert the counts to file offsets."""
        self.first_group = np.cumsum(self.n_groups) - self.n_groups
        self.first_subhalo = np.cumsum(self.n_subhalos) - self.n_subhalos
        self.first_gas = np.cumsum(self.n_gas) - self.n_gas
        self.first_dm = np.cumsum(self.n_dm) - self.n_dm
        self.first_p2 = np.cumsum(self.n_p2) - self.n_p2
        self.first_p3 = np.cumsum(self.n_p3) - self.n_p3
        self.first_star = np.cumsum(self.n_stars) - self.n_stars

    def highres_center_of_mass_numpy(self):
        """Calculate the center of mass of high-resolution particles."""
        self.n_gas_com, self.m_gas_com, self.r_gas_com = center_of_mass_gas_numpy(self.m_gas, self.m_gas_HR, self.r_gas, GAS_HIGH_RES_THRESHOLD)
        self.n_dm_com = len(self.r_dm)
        self.m_dm_com = float(self.n_dm_com) * self.m_dm
        self.r_dm_com = center_of_mass_dm_numpy(self.r_dm)
        if self.n_stars_tot > 0:
            self.n_stars_com, self.m_stars_com, self.r_stars_com = center_of_mass_stars_numpy(self.m_stars, self.r_stars, self.is_HR)
        else:
            self.n_stars_com, self.m_stars_com, self.r_stars_com = 0, 0., np.zeros(3)
        self.m_com = self.m_gas_com + self.m_dm_com + self.m_stars_com
        self.r_com = (self.m_gas_com * self.r_gas_com + self.m_dm_com * self.r_dm_com + self.m_stars_com * self.r_stars_com) / self.m_com

    def highres_center_of_mass_numba(self):
        """Calculate the center of mass of high-resolution particles."""
        self.n_gas_com, self.m_gas_com, self.r_gas_com = center_of_mass_gas_numba(self.m_gas, self.m_gas_HR, self.r_gas, GAS_HIGH_RES_THRESHOLD)
        self.n_dm_com = len(self.r_dm)
        self.m_dm_com = float(self.n_dm_com) * self.m_dm
        self.r_dm_com = center_of_mass_dm_numba(self.r_dm)
        if self.n_stars_tot > 0:
            self.n_stars_com, self.m_stars_com, self.r_stars_com = center_of_mass_stars_numba(self.m_stars, self.r_stars, self.is_HR)
        else:
            self.n_stars_com, self.m_stars_com, self.r_stars_com = 0, 0., np.zeros(3)
        self.m_com = self.m_gas_com + self.m_dm_com + self.m_stars_com
        self.r_com = (self.m_gas_com * self.r_gas_com + self.m_dm_com * self.r_dm_com + self.m_stars_com * self.r_stars_com) / self.m_com

    def highres_radius(self):
        # Calculate maximum distances to high-resolution particles from their center of mass
        r_com = self.r_com # Center of mass
        # High-resolution gas particles
        r = self.r_gas[self.m_gas_HR > 0.]
        for i in range(3): r[:,i] -= r_com[i] # Recenter particles
        dr2_gas_hr = np.max(np.sum(r**2, axis=1)) # Maximum distance squared
        # Low-resolution gas particles
        r = self.r_gas[self.m_gas_HR == 0.]
        for i in range(3): r[:,i] -= r_com[i]
        dr2_gas_lr = np.min(np.sum(r**2, axis=1))
        # High-resolution dark matter particles (PartType1)
        r = np.copy(self.r_dm)
        for i in range(3): r[:,i] -= r_com[i]
        dr2_dm_hr = np.max(np.sum(r**2, axis=1))
        # Low-resolution dark matter particles (PartType2 and PartType3)
        r = np.vstack([self.r_p2, self.r_p3])
        for i in range(3): r[:,i] -= r_com[i]
        dr2_dm_lr = np.min(np.sum(r**2, axis=1))
        if self.n_stars_tot > 0:
            # High-resolution star particles
            r = self.r_stars[self.is_HR > 0]
            if len(r) > 0:
                for i in range(3): r[:,i] -= r_com[i]
                dr2_star_hr = np.max(np.sum(r**2, axis=1))
                self.RadiusHR = np.sqrt(max(dr2_gas_hr, dr2_dm_hr, dr2_star_hr))
            else:
                dr2_star_hr = 0.
                self.RadiusHR = np.sqrt(max(dr2_gas_hr, dr2_dm_hr))
            # Low-resolution star particles
            r = self.r_stars[self.is_HR == 0]
            if len(r) > 0:
                for i in range(3): r[:,i] -= r_com[i]
                dr2_star_lr = np.min(np.sum(r**2, axis=1))
                self.RadiusLR = np.sqrt(min(dr2_gas_lr, dr2_dm_lr, dr2_star_lr))
            else:
                dr2_star_lr = 0.
                self.RadiusLR = np.sqrt(min(dr2_gas_lr, dr2_dm_lr))
        else:
            dr2_star_hr, dr2_star_lr = 0., 0.
            self.RadiusHR = np.sqrt(max(dr2_gas_hr, dr2_dm_hr))
            self.RadiusLR = np.sqrt(min(dr2_gas_lr, dr2_dm_lr))
        # Count the number of particles within these distances
        self.NumGasHR = np.count_nonzero(np.sum((self.r_gas - r_com)**2, axis=1) < self.RadiusHR**2)
        self.NumGasLR = np.count_nonzero(np.sum((self.r_gas - r_com)**2, axis=1) < self.RadiusLR**2)
        if VERBOSITY > 0:
            print(f'Farthest high-resolution distance: (gas, dm, stars) = ({np.sqrt(dr2_gas_hr):g}, {np.sqrt(dr2_dm_hr):g}, {np.sqrt(dr2_star_hr):g}) ckpc/h')
            print(f'Nearest low-resolution distance: (gas, dm, stars) = ({np.sqrt(dr2_gas_lr):g}, {np.sqrt(dr2_dm_lr):g}, {np.sqrt(dr2_star_lr):g}) ckpc/h')
            print(f'RadiusHR = {self.RadiusHR:g} ckpc/h, NumGasHR = {self.NumGasHR}\nRadiusLR = {self.RadiusLR:g} ckpc/h, NumGasLR = {self.NumGasLR}')

    def build_trees(self):
        """Build the trees for the gas and star particles."""
        self.tree_gas_lr = cKDTree(self.r_gas[self.m_gas_HR == 0.], boxsize=self.BoxSize) # Low-resolution gas particles
        self.tree_gas_hr = cKDTree(self.r_gas[self.m_gas_HR > 0.], boxsize=self.BoxSize) # High-resolution gas particles
        self.tree_dm = cKDTree(self.r_dm, boxsize=self.BoxSize) # PartType1 particles
        self.tree_p2 = cKDTree(self.r_p2, boxsize=self.BoxSize) # PartType2 particles
        self.tree_p3 = cKDTree(self.r_p3, boxsize=self.BoxSize) # PartType3 particles
        if self.n_stars_tot > 0:
            self.tree_stars_lr = cKDTree(self.r_stars[self.is_HR == 0], boxsize=self.BoxSize) # Low-resolution star particles
            self.tree_stars_hr = cKDTree(self.r_stars[self.is_HR > 0], boxsize=self.BoxSize) # High-resolution star particles

    def find_nearest_group(self):
        """Find the nearest distance to each group position."""
        self.group_distances_gas_lr, self.group_indices_gas_lr = self.tree_gas_lr.query(self.GroupPos) # Low-resolution gas particles
        self.group_distances_gas_hr, self.group_indices_gas_hr = self.tree_gas_hr.query(self.GroupPos) # High-resolution gas particles
        self.group_distances_dm, self.group_indices_dm = self.tree_dm.query(self.GroupPos)
        self.group_distances_p2, self.group_indices_p2 = self.tree_p2.query(self.GroupPos)
        self.group_distances_p3, self.group_indices_p3 = self.tree_p3.query(self.GroupPos)
        if self.n_stars_tot > 0:
            self.group_distances_stars_lr, self.group_indices_stars_lr = self.tree_stars_lr.query(self.GroupPos) # Low-resolution star particles
            self.group_distances_stars_hr, self.group_indices_stars_hr = self.tree_stars_hr.query(self.GroupPos) # High-resolution star particles

    def find_nearest_subhalo(self):
        """Find the nearest distance to each subhalo position."""
        self.subhalo_distances_gas_lr, self.subhalo_indices_gas_lr = self.tree_gas_lr.query(self.SubhaloPos) # Low-resolution gas particles
        self.subhalo_distances_gas_hr, self.subhalo_indices_gas_hr = self.tree_gas_hr.query(self.SubhaloPos) # High-resolution gas particles
        self.subhalo_distances_dm, self.subhalo_indices_dm = self.tree_dm.query(self.SubhaloPos)
        self.subhalo_distances_p2, self.subhalo_indices_p2 = self.tree_p2.query(self.SubhaloPos)
        self.subhalo_distances_p3, self.subhalo_indices_p3 = self.tree_p3.query(self.SubhaloPos)
        if self.n_stars_tot > 0:
            self.subhalo_distances_stars_lr, self.subhalo_indices_stars_lr = self.tree_stars_lr.query(self.SubhaloPos) # Low-resolution star particles
            self.subhalo_distances_stars_hr, self.subhalo_indices_stars_hr = self.tree_stars_hr.query(self.SubhaloPos) # High-resolution star particles

    def calculate_average_density(self):
        """Calculate the average density of the simulation."""
        self.m_gas_tot = np.sum(self.m_gas) # Total gas mass
        self.m_dm_tot = self.m_dm * float(self.n_dm_tot) # Total dark matter mass
        self.m_p2_tot = np.sum(self.m_p2) # Total PartType2 mass
        self.m_p3_tot = self.m_p3 * float(self.n_p3_tot) # Total dark matter mass
        self.m_stars_tot = np.sum(self.m_stars) if self.n_stars_tot > 0 else 0. # Total stellar mass
        self.m_tot = self.m_gas_tot + self.m_dm_tot + self.m_p2_tot + self.m_p3_tot + self.m_stars_tot # Total mass
        self.rho_avg = self.m_tot / self.BoxSize**3 # Total density
        km = 1e5                    # Units: 1 km = 1e5 cm
        Mpc = 3.085677581467192e24  # Units: 1 Mpc = 3e24 cm
        H0 = 100. * self.h * km / Mpc # Hubble constant [km/s/Mpc]
        G = 6.6743015e-8 # Gravitational constant [cm^3/g/s^2]
        H2 = H0**2 * (self.Omega0 / self.a**3 + self.OmegaLambda) # Hubble parameter squared
        x_c = self.Omega0 - 1. # x_c = Omega0 - 1
        self.Delta_c = 200. if USE_200 else 18. * np.pi**2 + 82. * x_c - 39. * x_c**2 # Critical overdensity factor
        self.rho_crit0 = 3. * H0**2 / (8. * np.pi * G) # Critical density today [g/cm^3]
        self.rho_crit = 3. * H2 / (8. * np.pi * G) # Critical density [g/cm^3]
        self.rho_vir = self.Delta_c * self.rho_crit # Virial density [g/cm^3]
        self.rho_vir_code = self.rho_vir / self.density_to_cgs # Virial density in code units
        if VERBOSITY > 1:
            print(f'min/max m_gas = [{np.min(self.m_gas*self.mass_to_msun):g}, {np.max(self.m_gas*self.mass_to_msun):g}] Msun')
            print(f'min/max m_dm = [{self.m_dm*self.mass_to_msun:g}, {self.m_dm*self.mass_to_msun:g}] Msun')
            print(f'min/max m_p2 = [{np.min(self.m_p2*self.mass_to_msun):g}, {np.max(self.m_p2*self.mass_to_msun):g}] Msun')
            print(f'min/max m_p3 = [{self.m_p3*self.mass_to_msun:g}, {self.m_p3*self.mass_to_msun:g}] Msun')
            print(f'min/max m_stars = [{np.min(self.m_stars*self.mass_to_msun):g}, {np.max(self.m_stars*self.mass_to_msun):g}] Msun')
            print(f'sum(m_gas) = {self.m_gas_tot*self.mass_to_msun:g} Msun = {100.*self.m_gas_tot/self.m_tot:g}% [m_gas_res = {self.m_gas_tot*self.mass_to_msun/float(self.n_gas_tot):g} Msun]')
            print(f'sum(m_dm) = {self.m_dm_tot*self.mass_to_msun:g} Msun = {100.*self.m_dm_tot/self.m_tot:g}% [m_dm_res = {self.m_dm*self.mass_to_msun:g} Msun]')
            print(f'sum(m_p2) = {self.m_p2_tot*self.mass_to_msun:g} Msun = {100.*self.m_p2_tot/self.m_tot:g}% [m_p2_res = {self.m_p2_tot*self.mass_to_msun/float(self.n_p2_tot):g} Msun]')
            print(f'sum(m_p3) = {self.m_p3_tot*self.mass_to_msun:g} Msun = {100.*self.m_p3_tot/self.m_tot:g}% [m_p3_res = {self.m_p3*self.mass_to_msun:g} Msun]')
            print(f'sum(m_stars) = {self.m_stars_tot*self.mass_to_msun:g} Msun = {100.*self.m_stars_tot/self.m_tot:g}% [m_stars_res = {self.m_stars_tot*self.mass_to_msun/float(self.n_stars_tot):g} Msun]')
            print(f'rho_crit0 = {self.rho_crit0:g} g/cm^3, rho_crit = {self.rho_crit:g} g/cm^3, Delta_c = {self.Delta_c:g}, rho_vir = {self.rho_vir:g} g/cm^3')

    def setup_R_vir(self):
        """Initialization for calculating the virial radius of each subhalo."""
        if self.n_subhalos_tot > 0:
            self.Subhalo_R_vir = np.zeros(self.n_subhalos_tot, dtype=np.float32) # Virial radius
            self.Subhalo_M_vir = np.zeros(self.n_subhalos_tot, dtype=np.float32) # Virial mass
            self.Subhalo_M_gas = np.zeros(self.n_subhalos_tot, dtype=np.float32) # Gas mass (<R_vir)
            self.Subhalo_M_stars = np.zeros(self.n_subhalos_tot, dtype=np.float32) # Stellar mass (<R_vir)
            self.r_min = 1e-2 * self.h / self.a # 10 pc
            self.r_max = 2. * self.BoxSize # 2 r_box
            self.logr2_min = 2. * np.log10(self.r_min) # log(r_min^2)
            self.logr2_max = 2. * np.log10(self.r_max) # log(r_max^2)
            self.n_bins = 10000 # Number of bins
            self.inv_dbin = float(self.n_bins) / (self.logr2_max - self.logr2_min) # n / (log(r_max^2) - log(r_min^2))
            bins = np.logspace(self.logr2_min, self.logr2_max, self.n_bins+1) # r^2 bins
            self.log_rbins = np.linspace(0.5*self.logr2_min, 0.5*self.logr2_max, self.n_bins+1) # log(r) bins
            self.dlog_rbin = 0.5 * (self.logr2_max - self.logr2_min) / float(self.n_bins) # (log(r_max) - log(r_min)) / n
            bins[0] = 0. # Extend first bin to zero
            V_enc = 4. * np.pi / 3. * bins[1:]**1.5 # Enclosed volume
            self.M_to_rho_vir = 1. / V_enc / self.rho_vir_code # Mass to virial density
            self.m_dm_full = self.m_dm * np.ones(self.n_dm_tot) # Full dark matter mass array
            self.m_p3_full = self.m_p3 * np.ones(self.n_p3_tot) # Full PartType3 mass array
            if VERBOSITY > 1:
                print(f'[r_min, r_max] = [{self.r_min:g}, {self.r_max:g}] ckpc/h = [{self.r_min*self.length_to_cgs/kpc:g}, {self.r_max*self.length_to_cgs/kpc:g}] kpc')
                print(f'[logr2_min, logr2_max] = [{self.logr2_min:g}, {self.logr2_max:g}] in units of (ckpc/h)^2')
                print(f'n_bins = {self.n_bins}, inv_dbin = n_bins / (log(r_max^2) - log(r_min^2)) = {self.inv_dbin:g}')
                print(f'log_rbins = {self.log_rbins} in units of ckpc/h')
                print(f'bins = {np.sqrt(bins)} ckpc/h')
                print(f'r_edges = {10.**self.log_rbins} ckp/hc = {10.**self.log_rbins*self.length_to_cgs/kpc} kpc')
                print(f'V_enc = {V_enc} (ckpc/h)^3 = {V_enc*self.volume_to_cgs/kpc**3} kpc^3')
                print(f'rho_vir / M = {self.M_to_rho_vir}')

    def periodic_distance(self, coord1, coord2):
        """Calculate the periodic distance between two coordinates."""
        delta = np.abs(coord1 - coord2)  # Absolute differences
        return np.minimum(delta, self.BoxSize - delta)  # Minimum of delta and its periodic counterpart

    def calculate_M_enc(self, r_sub, r, m):
        """Calculate the enclosed mass within a given radius."""
        r = np.atleast_2d(r)  # Ensure r is at least 2-dimensional
        m = np.atleast_1d(m)  # Ensure m is at least 1-dimensional
        dx = self.periodic_distance(r[:,0], r_sub[0])
        dy = self.periodic_distance(r[:,1], r_sub[1])
        dz = self.periodic_distance(r[:,2], r_sub[2])
        r2 = dx**2 + dy**2 + dz**2  # Squared distances
        ibin = np.floor((np.log10(r2) - self.logr2_min) * self.inv_dbin).astype(np.int32)  # Bin indices
        ibin[ibin < 0] = 0  # Clip to first bin
        return np.cumsum(np.bincount(ibin, weights=m, minlength=self.n_bins))  # Enclosed gas mass

    def calculate_R_vir_hist(self, n_subhalos_max=0):
        """Calculate the virial radius of each subhalo (global histogram version)."""
        # Sanity checks on requested n_subhalos
        if n_subhalos_max <= 0 or n_subhalos_max > self.n_subhalos_tot:
            n_subhalos_max = int(self.n_subhalos_tot)
        if self.n_stars_tot == 0:
            M_stars_enc = np.zeros(self.n_bins)  # Enclosed stellar mass
        for i in range(n_subhalos_max):
            if self.Group_R_Crit200[self.SubhaloGroupNr[i]] <= 0.:
                continue # Skip groups without a valid R_Crit200
            r_sub = self.SubhaloPos[i] # Subhalo center
            M_sub = self.SubhaloMass[i] # Subhalo mass
            M_sub_20 = 0.2 * M_sub # 20% of the subhalo mass
            M_gas_enc = self.calculate_M_enc(r_sub, self.r_gas, self.m_gas)  # Enclosed gas mass
            M_dm_enc = self.calculate_M_enc(r_sub, self.r_dm, self.m_dm_full)  # Enclosed dark matter mass
            M_p2_enc = self.calculate_M_enc(r_sub, self.r_p2, self.m_p2)  # Enclosed PartType2 mass
            M_p3_enc = self.calculate_M_enc(r_sub, self.r_p3, self.m_p3_full)  # Enclosed PartType3 mass
            if self.n_stars_tot > 0:
                M_stars_enc = self.calculate_M_enc(r_sub, self.r_stars, self.m_stars)
            M_enc = M_gas_enc + M_dm_enc + M_p2_enc + M_p3_enc + M_stars_enc  # Total enclosed mass
            # Calculate the enclosed mass
            i_0 = np.argmax(M_enc > M_sub_20)  # Find the first bin with mass > 20% of the subhalo mass
            # i_0 = np.argmax(M_enc > 0.) + 1 # Find the second bin with any mass
            rho_enc = M_enc * self.M_to_rho_vir # Enclosed density [rho_vir]
            # Calculate the virial radius
            i_vir = np.argmax(rho_enc[i_0:] < 1.) + i_0 # Find the first bin with rho_enc < rho_vir
            if i_vir == 0: i_vir = 1 # Avoid zero index
            # Log interpolation to find the virial radius and masses
            frac = -np.log10(rho_enc[i_vir-1]) / np.log10(rho_enc[i_vir]/rho_enc[i_vir-1]) # Interpolation coordinate
            self.Subhalo_R_vir[i] = 10.**(self.log_rbins[i_vir] + frac * self.dlog_rbin) # Virial radius
            self.Subhalo_M_vir[i] = 10.**(np.log10(M_enc[i_vir-1]) + frac * np.log10(M_enc[i_vir]/M_enc[i_vir-1])) # Virial mass
            if M_gas_enc[i_vir-1] <= 0.:
                self.Subhalo_M_gas[i] = M_gas_enc[i_vir-1] + frac * (M_gas_enc[i_vir] - M_gas_enc[i_vir-1]) # Gas mass (<R_vir)
            else:
                self.Subhalo_M_gas[i] = 10.**(np.log10(M_gas_enc[i_vir-1]) + frac * np.log10(M_gas_enc[i_vir]/M_gas_enc[i_vir-1])) # Gas mass (<R_vir)
            if M_stars_enc[i_vir-1] <= 0.:
                self.Subhalo_M_stars[i] = M_stars_enc[i_vir-1] + frac * (M_stars_enc[i_vir] - M_stars_enc[i_vir-1]) # Stellar mass (<R_vir)
            else:
                self.Subhalo_M_stars[i] = 10.**(np.log10(M_stars_enc[i_vir-1]) + frac * np.log10(M_stars_enc[i_vir]/M_stars_enc[i_vir-1])) # Stellar mass (<R_vir)
            if RADIAL_PLOTS:
                i_grp = self.SubhaloGroupNr[i] # Group index
                r_grp = self.GroupPos[i_grp] # Group center
                dr_sub = np.sqrt(np.sum((r_sub - r_grp)**2)) # Subhalo distance from group center
                if dr_sub <= 0.:
                    print(f'Group {i_grp}: M_vir / M_200 = {self.Subhalo_M_vir[i] / self.Group_M_Crit200[i_grp]}, R_vir / R_200 = {self.Subhalo_R_vir[i] / self.Group_R_Crit200[i_grp]}')
                    print(f'r_grp = {r_grp*self.length_to_cgs/kpc} kpc, r_sub = {r_sub*self.length_to_cgs/kpc} kpc, dr_sub = {dr_sub*self.length_to_cgs/kpc} kpc')
                    import matplotlib.pyplot as plt
                    r_e = 10.**self.log_rbins*self.length_to_cgs/kpc # kpc
                    r_c = 0.5 * (r_e[1:] + r_e[:-1]) # kpc
                    r_o = r_e[1:] # kpc
                    plt.plot(r_o, M_enc*self.mass_to_msun, label='M_tot')
                    plt.plot(r_o, M_gas_enc*self.mass_to_msun, label='M_gas')
                    plt.plot(r_o, M_dm_enc*self.mass_to_msun, label='M_dm')
                    plt.plot(r_o, M_p2_enc*self.mass_to_msun, label='M_p2')
                    plt.plot(r_o, M_p3_enc*self.mass_to_msun, label='M_p3')
                    plt.plot(r_o, M_stars_enc*self.mass_to_msun, label='M_stars')
                    plt.scatter(self.Group_R_Crit200[i_grp]*self.length_to_cgs/kpc, self.Group_M_Crit200[i_grp]*self.mass_to_msun, color='k', s=40, label='M_200')
                    plt.scatter(self.Subhalo_R_vir[i]*self.length_to_cgs/kpc, self.Subhalo_M_vir[i]*self.mass_to_msun, color='r', s=40, marker='x', label='M_vir')
                    plt.axvline(self.group_distances_p2[i_grp]*self.length_to_cgs/kpc, color='b', linestyle='--', label='R_p2')
                    plt.axvline(self.group_distances_p3[i_grp]*self.length_to_cgs/kpc, color='g', linestyle='--', label='R_p3')
                    plt.xlabel('r [kpc]'); plt.ylabel('M(<r) [Msun]'); plt.xscale('log'); plt.yscale('log'); plt.legend()
                    plt.savefig(f'plots_radial/M_enc_{i_grp}_{i}_h.pdf'); plt.close()
                    plt.plot(r_c, rho_enc, label='rho_enc')
                    plt.axvline(self.Group_R_Crit200[i_grp]*self.length_to_cgs/kpc, color='k', linestyle='-', label='R_200')
                    plt.axvline(self.Subhalo_R_vir[i]*self.length_to_cgs/kpc, color='r', linestyle='-', label='R_vir')
                    plt.axhline(1., color='k', linestyle='--', label='rho_vir')
                    plt.xlabel('r [kpc]'), plt.ylabel('rho_enc [rho_vir]'), plt.xscale('log'), plt.yscale('log'), plt.legend()
                    plt.savefig(f'plots_radial/rho_enc_{i_grp}_{i}_h.pdf'); plt.close()
        if VERBOSITY > 1:
            print(f'R_vir = {self.Subhalo_R_vir[:n_subhalos_max]*self.length_to_cgs/kpc} kpc')
            print(f'M_vir = {self.Subhalo_M_vir[:n_subhalos_max]*self.mass_to_msun} Msun')
            print(f'M_gas = {self.Subhalo_M_gas[:n_subhalos_max]*self.mass_to_msun} Msun')
            print(f'M_stars = {self.Subhalo_M_stars[:n_subhalos_max]*self.mass_to_msun} Msun')
            print(f'SubhaloPos = {self.SubhaloPos[:n_subhalos_max]*self.length_to_cgs/kpc} kpc')
            print(f'SubhaloMass = {self.SubhaloMass[:n_subhalos_max]*self.mass_to_msun} Msun')
            print(f'SubhaloGroupNr = {self.SubhaloGroupNr[:n_subhalos_max]}')

    def setup_full_trees(self):
        """Build the trees for all particles."""
        # Assuming self.BoxSize is defined and is a scalar representing the size of the box in each dimension
        self.tree_gas = cKDTree(self.r_gas, boxsize=self.BoxSize)  # Tree for all gas particles
        if self.n_stars_tot > 0:
            self.tree_stars = cKDTree(self.r_stars, boxsize=self.BoxSize)  # Tree for all star particles
        # Calculate group and subhalo convenience indices
        if self.n_groups_tot > 0:
            self.GroupFirstSub = np.cumsum(self.GroupNsubs) - self.GroupNsubs # First subhalo in each group
            self.GroupFirstType = np.cumsum(self.GroupLenType, axis=0) - self.GroupLenType # First particle of each type in each group
        if self.n_subhalos_tot > 0:
            self.SubhaloFirstType = np.zeros_like(self.SubhaloLenType) # First particle of each type in each subhalo
            for i in range(self.n_groups_tot):
                i_beg, i_end = self.GroupFirstSub[i], self.GroupFirstSub[i] + self.GroupNsubs[i] # Subhalo range
                first_subs = np.cumsum(self.SubhaloLenType[i_beg:i_end], axis=0) - self.SubhaloLenType[i_beg:i_end] # Relative offsets
                for i_part in range(NUM_PART):
                    first_subs[:,i_part] += self.GroupFirstType[i,i_part] # Add group offset
                self.SubhaloFirstType[i_beg:i_end] = first_subs # First particle of each type in each subhalo

    def query_neighbors(self, tree, point, R_max, n):
        """Query the neighbors of a given point within a given radius."""
        k = int(min(n, 64))  # Number of neighbors to query
        while True:
            distances, indices = tree.query(point, k=k)
            if k == n or distances[-1] > R_max:
                break  # Finished collecting particles
            k = int(min(n, 2 * k))  # Double the number of neighbors to query
        return distances, indices

    def calculate_R_vir_tree(self, n_groups_max=0):
        """Calculate the virial radius of each subhalo (local tree version)."""
        # Sanity checks on requested n_groups
        if n_groups_max <= 0 or n_groups_max > self.n_groups_tot:
            n_groups_max = int(self.n_groups_tot)
        n_subhalos_max = self.GroupFirstSub[n_groups_max-1] + self.GroupNsubs[n_groups_max-1] # Maximum number of subhalos
        if VERBOSITY > 1:
            print(f'GroupFirstSub = {self.GroupFirstSub[:n_groups_max]}')
            print(f'GroupNsubs = {self.GroupNsubs[:n_groups_max]}')
            print(f'n_subhalos_max = {n_subhalos_max}')
            print(f'SubhaloGroupNr = {self.SubhaloGroupNr[:n_subhalos_max]}')
        R_30kpc = 30. * self.h / self.a # 30 kpc
        if self.n_stars_tot == 0:
            M_stars_enc = np.zeros(self.n_bins)  # Enclosed stellar mass
        for i_grp in range(n_groups_max):
            if self.Group_R_Crit200[i_grp] <= 0.:
                continue # Skip groups without a valid R_Crit200
            r_grp = self.GroupPos[i_grp] # Group center
            R_max = max(2. * self.Group_R_Crit200[i_grp], R_30kpc) # 2 R_200 (of the Group)
            distances_gas, indices_gas = self.query_neighbors(self.tree_gas, r_grp, R_max, self.n_gas_tot)  # Gas
            distances_dm, indices_dm = self.query_neighbors(self.tree_dm, r_grp, R_max, self.n_dm_tot)  # Dark matter
            distances_p2, indices_p2 = self.query_neighbors(self.tree_p2, r_grp, R_max, self.n_p2_tot)  # PartType2
            distances_p3, indices_p3 = self.query_neighbors(self.tree_p3, r_grp, R_max, self.n_p3_tot)  # PartType3
            if self.n_stars_tot > 0:
                distances_stars, indices_stars = self.query_neighbors(self.tree_stars, r_grp, R_max, self.n_stars_tot)
            # Calculate the enclosed mass of each particle type in each subhalo
            i_beg, i_end = self.GroupFirstSub[i_grp], self.GroupFirstSub[i_grp] + self.GroupNsubs[i_grp] # Subhalo range
            if not USE_ALL_PARTICLES:
                i_beg_gas, i_end_gas = self.GroupFirstType[i_grp,0], self.GroupFirstType[i_grp,0] + self.GroupLenType[i_grp,0] # Gas range
                i_beg_dm, i_end_dm = self.GroupFirstType[i_grp,1], self.GroupFirstType[i_grp,1] + self.GroupLenType[i_grp,1] # Dark matter range
                i_beg_p2, i_end_p2 = self.GroupFirstType[i_grp,2], self.GroupFirstType[i_grp,2] + self.GroupLenType[i_grp,2] # PartType2 range
                i_beg_p3, i_end_p3 = self.GroupFirstType[i_grp,3], self.GroupFirstType[i_grp,3] + self.GroupLenType[i_grp,3] # PartType3 range
                i_beg_stars, i_end_stars = self.GroupFirstType[i_grp,4], self.GroupFirstType[i_grp,4] + self.GroupLenType[i_grp,4] # Stars range
            for i_sub in range(i_beg, i_end):
                r_sub = self.SubhaloPos[i_sub] # Subhalo center
                M_sub = self.SubhaloMass[i_sub] # Subhalo mass
                M_sub_20 = 0.2 * M_sub # 20% of the subhalo mass
                if USE_ALL_PARTICLES:
                    M_gas_enc = self.calculate_M_enc(r_sub, self.r_gas[indices_gas], self.m_gas[indices_gas])  # Enclosed gas mass
                    M_dm_enc = self.calculate_M_enc(r_sub, self.r_dm[indices_dm], self.m_dm_full[indices_dm])  # Enclosed dark matter mass
                    M_p2_enc = self.calculate_M_enc(r_sub, self.r_p2[indices_p2], self.m_p2[indices_p2])  # Enclosed PartType2 mass
                    M_p3_enc = self.calculate_M_enc(r_sub, self.r_p3[indices_p3], self.m_p3_full[indices_p3])  # Enclosed PartType3 mass
                    if self.n_stars_tot > 0:
                        M_stars_enc = self.calculate_M_enc(r_sub, self.r_stars[indices_stars], self.m_stars[indices_stars])
                else:
                    M_gas_enc = self.calculate_M_enc(r_sub, self.r_gas[i_beg_gas:i_end_gas], self.m_gas[i_beg_gas:i_end_gas])  # Enclosed gas mass
                    M_dm_enc = self.calculate_M_enc(r_sub, self.r_dm[i_beg_dm:i_end_dm], self.m_dm_full[i_beg_dm:i_end_dm])  # Enclosed dark matter mass
                    M_p2_enc = self.calculate_M_enc(r_sub, self.r_p2[i_beg_p2:i_end_p2], self.m_p2[i_beg_p2:i_end_p2])  # Enclosed PartType2 mass
                    M_p3_enc = self.calculate_M_enc(r_sub, self.r_p3[i_beg_p3:i_end_p3], self.m_p3_full[i_beg_p3:i_end_p3])  # Enclosed PartType3 mass
                    if self.n_stars_tot > 0:
                        M_stars_enc = self.calculate_M_enc(r_sub, self.r_stars[i_beg_stars:i_end_stars], self.m_stars[i_beg_stars:i_end_stars])
                M_enc = M_gas_enc + M_dm_enc + M_p2_enc + M_p3_enc + M_stars_enc  # Total enclosed mass
                # Calculate the enclosed mass
                i_0 = np.argmax(M_enc > M_sub_20)  # Find the first bin with mass > 20% of the subhalo mass
                # i_0 = np.argmax(M_enc > 0.) + 1 # Find the second bin with any mass
                rho_enc = M_enc * self.M_to_rho_vir # Enclosed density [rho_vir]
                # Calculate the virial radius
                i_vir = np.argmax(rho_enc[i_0:] < 1.) + i_0 # Find the first bin with rho_enc < rho_vir
                if i_vir == 0: i_vir = 1 # Avoid zero index
                # Log interpolation to find the virial radius and masses
                frac = -np.log10(rho_enc[i_vir-1]) / np.log10(rho_enc[i_vir]/rho_enc[i_vir-1]) # Interpolation coordinate
                self.Subhalo_R_vir[i_sub] = 10.**(self.log_rbins[i_vir] + frac * self.dlog_rbin) # Virial radius
                self.Subhalo_M_vir[i_sub] = 10.**(np.log10(M_enc[i_vir-1]) + frac * np.log10(M_enc[i_vir]/M_enc[i_vir-1])) # Virial mass
                if M_gas_enc[i_vir-1] <= 0.:
                    self.Subhalo_M_gas[i_sub] = M_gas_enc[i_vir-1] + frac * (M_gas_enc[i_vir] - M_gas_enc[i_vir-1]) # Gas mass (<R_vir)
                else:
                    self.Subhalo_M_gas[i_sub] = 10.**(np.log10(M_gas_enc[i_vir-1]) + frac * np.log10(M_gas_enc[i_vir]/M_gas_enc[i_vir-1])) # Gas mass (<R_vir)
                if M_stars_enc[i_vir-1] <= 0.:
                    self.Subhalo_M_stars[i_sub] = M_stars_enc[i_vir-1] + frac * (M_stars_enc[i_vir] - M_stars_enc[i_vir-1]) # Stellar mass (<R_vir)
                else:
                    self.Subhalo_M_stars[i_sub] = 10.**(np.log10(M_stars_enc[i_vir-1]) + frac * np.log10(M_stars_enc[i_vir]/M_stars_enc[i_vir-1])) # Stellar mass (<R_vir)
                dr_sub = np.sqrt(np.sum((r_sub - r_grp)**2)) # Subhalo distance from group center
                if RADIAL_PLOTS:
                    # print(f'i_grp = {i_grp}, i_sub = {i_sub}, R_200 = {self.Group_R_Crit200[i_grp]*self.length_to_cgs/kpc:g} kpc, ' +
                    #       f'R_vir = {self.Subhalo_R_vir[i_sub]*self.length_to_cgs/kpc:g} kpc, dr_sub = {dr_sub*self.length_to_cgs/kpc:g} kpc, ' +
                    #       f'r_grp = {r_grp*self.length_to_cgs/kpc} kpc, r_sub = {r_sub*self.length_to_cgs/kpc} kpc')
                    # for j in range(self.n_bins):
                    #     # if rho_enc[j] < 1.5:
                    #     print(f'j = {j}, r = {10.**(self.log_rbins[j])*self.length_to_cgs/kpc:g} kpc, ' +
                    #         f'M_enc = {M_enc[j]*self.mass_to_msun:g} Msun, rho_enc = {rho_enc[j]:g} rho_vir, ' +
                    #         f'M_gas = {M_gas_enc[j]*self.mass_to_msun:g} Msun, M_stars = {M_stars_enc[j]*self.mass_to_msun:g} Msun, ' +
                    #         f'M_200 = {self.Subhalo_M_vir[i_sub]*self.mass_to_msun:g} Msun, M_vir = {self.Subhalo_M_vir[i_sub]*self.mass_to_msun:g} Msun, ' +
                    #         f'R_200 = {self.Group_R_Crit200[i_grp]*self.length_to_cgs/kpc:g} kpc, R_vir = {self.Subhalo_R_vir[i_sub]*self.length_to_cgs/kpc:g} kpc')
                    # print(f'r_edges = {10.**self.log_rbins*self.length_to_cgs/kpc} kpc = {10.**self.log_rbins*self.length_to_cgs/kpc/1000./self.a} cMpc')
                    # if rho_enc[j] < 0.9:
                    # raise ValueError('Debugging')
                    if dr_sub <= 0.:
                        # print(f'rho_enc = {rho_enc}')
                        print(f'Group {i_grp}: M_vir / M_200 = {self.Subhalo_M_vir[i_sub] / self.Group_M_Crit200[i_grp]}, R_vir / R_200 = {self.Subhalo_R_vir[i_sub] / self.Group_R_Crit200[i_grp]}')
                        print(f'r_grp = {r_grp*self.length_to_cgs/kpc} kpc, r_sub = {r_sub*self.length_to_cgs/kpc} kpc, dr_sub = {dr_sub*self.length_to_cgs/kpc} kpc')
                        print(f'm_gas = {self.m_gas[indices_gas]*self.mass_to_msun} Msun')
                        print(f'm_dm = {self.m_dm_full[indices_dm]*self.mass_to_msun} Msun')
                        print(f'm_p2 = {self.m_p2[indices_p2]*self.mass_to_msun} Msun')
                        print(f'm_p3 = {self.m_p3_full[indices_p3]*self.mass_to_msun} Msun')
                        if self.n_stars_tot > 0:
                            print(f'm_stars = {self.m_stars[indices_stars]*self.mass_to_msun} Msun')
                        import matplotlib.pyplot as plt
                        r_e = 10.**self.log_rbins*self.length_to_cgs/kpc # kpc
                        r_c = 0.5 * (r_e[1:] + r_e[:-1]) # kpc
                        r_o = r_e[1:] # kpc
                        plt.plot(r_o, M_enc*self.mass_to_msun, label='M_tot')
                        plt.plot(r_o, M_gas_enc*self.mass_to_msun, label='M_gas')
                        plt.plot(r_o, M_dm_enc*self.mass_to_msun, label='M_dm')
                        plt.plot(r_o, M_p2_enc*self.mass_to_msun, label='M_p2')
                        plt.plot(r_o, M_p3_enc*self.mass_to_msun, label='M_p3')
                        plt.plot(r_o, M_stars_enc*self.mass_to_msun, label='M_stars')
                        plt.scatter(self.Group_R_Crit200[i_grp]*self.length_to_cgs/kpc, self.Group_M_Crit200[i_grp]*self.mass_to_msun, color='k', s=40, label='M_200')
                        plt.scatter(self.Subhalo_R_vir[i_sub]*self.length_to_cgs/kpc, self.Subhalo_M_vir[i_sub]*self.mass_to_msun, color='r', s=40, marker='x', label='M_vir')
                        plt.axvline(self.group_distances_p2[i_grp]*self.length_to_cgs/kpc, color='b', linestyle='--', label='R_p2')
                        plt.axvline(self.group_distances_p3[i_grp]*self.length_to_cgs/kpc, color='g', linestyle='--', label='R_p3')
                        plt.xlabel('r [kpc]'); plt.ylabel('M(<r) [Msun]'); plt.xscale('log'); plt.yscale('log'); plt.legend()
                        plt.savefig(f'plots_radial/M_enc_{i_grp}_{i_sub}_t.pdf'); plt.close()
                        plt.plot(r_c, rho_enc, label='rho_enc')
                        plt.axvline(self.Group_R_Crit200[i_grp]*self.length_to_cgs/kpc, color='k', linestyle='-', label='R_200')
                        plt.axvline(self.Subhalo_R_vir[i_sub]*self.length_to_cgs/kpc, color='r', linestyle='-', label='R_vir')
                        plt.axhline(1., color='k', linestyle='--', label='rho_vir')
                        plt.xlabel('r [kpc]'); plt.ylabel('rho_enc [rho_vir]'); plt.xscale('log'); plt.yscale('log'); plt.legend()
                        plt.savefig(f'plots_radial/rho_enc_{i_grp}_{i_sub}_t.pdf'); plt.close()
                if dr_sub + self.Subhalo_R_vir[i_sub] > R_max:
                    print(f'rho_enc = {rho_enc}'); print(f'M_enc = {M_enc}')
                    print(f'i_grp = {i_grp}, i_sub = {i_sub}, R_200 = {self.Group_R_Crit200[i_grp]*self.length_to_cgs/kpc:g} kpc, ' +
                          f'R_vir = {self.Subhalo_R_vir[i_sub]*self.length_to_cgs/kpc:g} kpc, dr_sub = {dr_sub*self.length_to_cgs/kpc:g} kpc, ' +
                          f'dr_sub + R_vir = {(dr_sub + self.Subhalo_R_vir[i_sub])*self.length_to_cgs/kpc:g} kpc, R_max = {R_max*self.length_to_cgs/kpc:g} kpc, ' +
                          f'r_grp = {r_grp*self.length_to_cgs/kpc} kpc, r_sub = {r_sub*self.length_to_cgs/kpc} kpc')
                    raise ValueError(f'Group {i_grp} has a subhalo outside R_max') # Sanity check
        if VERBOSITY > 1:
            print(f'R_vir = {self.Subhalo_R_vir[:n_subhalos_max]*self.length_to_cgs/kpc} kpc')
            print(f'M_vir = {self.Subhalo_M_vir[:n_subhalos_max]*self.mass_to_msun} Msun')
            print(f'M_gas = {self.Subhalo_M_gas[:n_subhalos_max]*self.mass_to_msun} Msun')
            print(f'M_stars = {self.Subhalo_M_stars[:n_subhalos_max]*self.mass_to_msun} Msun')
            print(f'SubhaloPos = {self.SubhaloPos[:n_subhalos_max]*self.length_to_cgs/kpc} kpc')
            print(f'SubhaloMass = {self.SubhaloMass[:n_subhalos_max]*self.mass_to_msun} Msun')
            print(f'SubhaloGroupNr = {self.SubhaloGroupNr[:n_subhalos_max]}')
        if VIRIAL_PLOTS:
            import matplotlib.pyplot as plt
            values = self.Subhalo_R_vir[:n_subhalos_max][self.Subhalo_R_vir[:n_subhalos_max] > 0] * self.length_to_cgs / kpc
            plt.hist(np.log10(values), histtype='step', bins=int(np.sqrt(len(values))), label='R_vir')
            plt.xlabel('log(R_vir) [kpc]'); plt.ylabel('N'); plt.yscale('log'); plt.legend(); plt.savefig('plots_virial/R_vir.pdf'); plt.close()
            values = self.Subhalo_M_vir[:n_subhalos_max][self.Subhalo_M_vir[:n_subhalos_max] > 0] * self.mass_to_msun
            plt.hist(np.log10(values), histtype='step', bins=int(np.sqrt(len(values))), label='M_vir')
            values = self.Subhalo_M_gas[:n_subhalos_max][self.Subhalo_M_gas[:n_subhalos_max] > 0] * self.mass_to_msun
            plt.hist(np.log10(values), histtype='step', bins=int(np.sqrt(len(values))), label='M_gas')
            values = self.Subhalo_M_stars[:n_subhalos_max][self.Subhalo_M_stars[:n_subhalos_max] > 0] * self.mass_to_msun
            plt.hist(np.log10(values), histtype='step', bins=int(np.sqrt(len(values))), label='M_stars')
            plt.xlabel('log(M_vir) [Msun]'); plt.ylabel('N'); plt.yscale('log'); plt.legend(); plt.savefig('plots_virial/M_vir.pdf'); plt.close()
            mask = self.Group_R_Crit200 > 0. # Valid groups
            i_grps = np.arange(n_groups_max)[mask] # Group indices
            i_subs = self.GroupFirstSub[i_grps] # Subhalo indices
            plt.scatter(self.Subhalo_M_vir[i_subs]*self.mass_to_msun, self.Group_M_Crit200[i_grps]*self.mass_to_msun, color='k', s=20, label='M_200')
            min_M_vir, max_M_vir = np.min(self.Subhalo_M_vir[i_subs]*self.mass_to_msun), np.max(self.Subhalo_M_vir[i_subs]*self.mass_to_msun)
            plt.plot([min_M_vir, max_M_vir], [min_M_vir, max_M_vir], color='r', linestyle='--', label='M_vir = M_200')
            plt.xlabel('M_vir [Msun]'); plt.ylabel('M_200 [Msun]'); plt.xscale('log'); plt.yscale('log'); plt.legend()
            plt.savefig('plots_virial/M_vir_vs_M_200.pdf'); plt.close()
            plt.scatter(self.Subhalo_M_vir*self.mass_to_msun, self.SubhaloMass*self.mass_to_msun, color='k', s=20, label='M_200')
            min_M_vir, max_M_vir = np.min(self.Subhalo_M_vir*self.mass_to_msun), np.max(self.Subhalo_M_vir*self.mass_to_msun)
            plt.plot([min_M_vir, max_M_vir], [min_M_vir, max_M_vir], color='r', linestyle='--', label='M_vir = M_200')
            plt.xlabel('M_vir [Msun]'); plt.ylabel('M_subhalo [Msun]'); plt.xscale('log'); plt.yscale('log'); plt.legend()
            plt.savefig('plots_virial/M_vir_vs_M_200_subhalos.pdf'); plt.close()
            plt.scatter(self.Subhalo_R_vir[i_subs]*self.length_to_cgs/kpc, self.Group_R_Crit200[i_grps]*self.length_to_cgs/kpc, color='k', s=20, label='R_200')
            min_R_vir, max_R_vir = np.min(self.Subhalo_R_vir[i_subs]*self.length_to_cgs/kpc), np.max(self.Subhalo_R_vir[i_subs]*self.length_to_cgs/kpc)
            plt.plot([min_R_vir, max_R_vir], [min_R_vir, max_R_vir], color='r', linestyle='--', label='R_vir = R_200')
            plt.xlabel('R_vir [kpc]'); plt.ylabel('R_200 [kpc]'); plt.xscale('log'); plt.yscale('log'); plt.legend()
            plt.savefig('plots_virial/R_vir_vs_R_200.pdf'); plt.close()

    def print_offsets(self):
        """Print the file counts and offsets."""
        print(f'n_groups = {self.n_groups} (n_groups_tot = {self.n_groups_tot})')
        print(f'n_subhalos = {self.n_subhalos} (n_subhalos_tot = {self.n_subhalos_tot})')
        print(f'n_gas = {self.n_gas} (n_gas_tot = {self.n_gas_tot})')
        print(f'n_dm = {self.n_dm} (n_dm_tot = {self.n_dm_tot})')
        print(f'n_p2 = {self.n_p2} (n_p2_tot = {self.n_p2_tot})')
        print(f'n_p3 = {self.n_p3} (n_p3_tot = {self.n_p3_tot})')
        print(f'n_stars = {self.n_stars} (n_stars_tot = {self.n_stars_tot})\n')
        print(f'first_group = {self.first_group} (n_groups_tot = {self.n_groups_tot})')
        print(f'first_subhalo = {self.first_subhalo} (n_subhalos_tot = {self.n_subhalos_tot})')
        print(f'first_gas = {self.first_gas} (n_gas_tot = {self.n_gas_tot})')
        print(f'first_dm = {self.first_dm} (n_dm_tot = {self.n_dm_tot})')
        print(f'first_p2 = {self.first_p2} (n_p2_tot = {self.n_p2_tot})')
        print(f'first_p3 = {self.first_p3} (n_p3_tot = {self.n_p3_tot})')
        print(f'first_star = {self.first_star} (n_stars_tot = {self.n_stars_tot})')

    def print_groups(self):
        """Print the group data."""
        if self.n_groups_tot > 0:
            print(f'GroupPos = {self.GroupPos}')
            print(f'Group_R_Crit200 = {self.Group_R_Crit200}')
            print(f'Group_M_Crit200 = {self.Group_M_Crit200}')
            print(f'GroupMassType = {self.GroupMassType}')
            print(f'GroupNsubs = {self.GroupNsubs}')
            print(f'GroupLenType = {self.GroupLenType}')
        if self.n_subhalos_tot > 0:
            print(f'SubhaloPos = {self.SubhaloPos}')
            print(f'SubhaloMass = {self.SubhaloMass}')
            print(f'SubhaloGroupNr = {self.SubhaloGroupNr}')
            print(f'SubhaloLenType = {self.SubhaloLenType}')

    def print_particles(self):
        """Print the particle data."""
        print(f'r_gas = {self.r_gas}')
        print(f'm_gas = {self.m_gas}')
        print(f'm_gas_HR = {self.m_gas_HR}')
        print(f'r_dm = {self.r_dm}')
        print(f'r_p2 = {self.r_p2}')
        print(f'm_p2 = {self.m_p2}')
        print(f'r_p3 = {self.r_p3}')
        if self.n_stars_tot > 0:
            print(f'r_stars = {self.r_stars}')
            print(f'm_stars = {self.m_stars}')
            print(f'is_HR = {self.is_HR}')

    def write(self):
        """Write the distance results to an HDF5 file."""
        with h5py.File(dist_file, 'w') as f:
            g = f.create_group(b'Header')
            g.attrs['Ngroups_Total'] = self.n_groups_tot
            g.attrs['Nsubhalos_Total'] = self.n_subhalos_tot
            g.attrs['Time'] = self.a
            g.attrs['Redshift'] = 1. / self.a - 1.
            g.attrs['BoxSize'] = self.BoxSize
            g.attrs['HubbleParam'] = self.h
            g.attrs['Omega0'] = self.Omega0
            g.attrs['OmegaBaryon'] = self.OmegaBaryon
            g.attrs['OmegaLambda'] = self.OmegaLambda
            g.attrs['UnitLength_in_cm'] = self.UnitLength_in_cm
            g.attrs['UnitMass_in_g'] = self.UnitMass_in_g
            g.attrs['UnitVelocity_in_cm_per_s'] = self.UnitVelocity_in_cm_per_s
            g.attrs['PosHR'] = self.r_com
            g.attrs['RadiusHR'] = self.RadiusHR
            g.attrs['RadiusLR'] = self.RadiusLR
            g.attrs['NumGasHR'] = np.int32(self.NumGasHR)
            g.attrs['NumGasLR'] = np.int32(self.NumGasLR)
            if self.n_groups_tot > 0:
                g = f.create_group(b'Group')
                # g.create_dataset(b'Pos', data=self.GroupPos)
                g.create_dataset(b'R_Crit200', data=self.Group_R_Crit200)
                # g.create_dataset(b'M_Crit200', data=self.Group_M_Crit200)
                # g.create_dataset(b'MassType', data=self.GroupMassType)
                g.create_dataset(b'MinDistGasLR', data=self.group_distances_gas_lr, dtype=np.float32)
                g.create_dataset(b'MinDistGasHR', data=self.group_distances_gas_hr, dtype=np.float32)
                g.create_dataset(b'MinDistDM', data=self.group_distances_dm, dtype=np.float32)
                g.create_dataset(b'MinDistP2', data=self.group_distances_p2, dtype=np.float32)
                g.create_dataset(b'MinDistP3', data=self.group_distances_p3, dtype=np.float32)
                if self.n_stars_tot > 0:
                    g.create_dataset(b'MinDistStarsLR', data=self.group_distances_stars_lr, dtype=np.float32)
                    g.create_dataset(b'MinDistStarsHR', data=self.group_distances_stars_hr, dtype=np.float32)
            if self.n_subhalos_tot > 0:
                g = f.create_group(b'Subhalo')
                # g.create_dataset(b'Pos', data=self.SubhaloPos)
                g.create_dataset(b'R_vir', data=self.Subhalo_R_vir) # Virial radius
                g.create_dataset(b'M_vir', data=self.Subhalo_M_vir) # Virial mass
                g.create_dataset(b'M_gas', data=self.Subhalo_M_gas) # Gas mass (<R_vir)
                g.create_dataset(b'M_stars', data=self.Subhalo_M_stars) # Stars mass (<R_vir)
                g.create_dataset(b'MinDistGasLR', data=self.subhalo_distances_gas_lr, dtype=np.float32)
                g.create_dataset(b'MinDistGasHR', data=self.subhalo_distances_gas_hr, dtype=np.float32)
                g.create_dataset(b'MinDistDM', data=self.subhalo_distances_dm, dtype=np.float32)
                g.create_dataset(b'MinDistP2', data=self.subhalo_distances_p2, dtype=np.float32)
                g.create_dataset(b'MinDistP3', data=self.subhalo_distances_p3, dtype=np.float32)
                if self.n_stars_tot > 0:
                    g.create_dataset(b'MinDistStarsLR', data=self.subhalo_distances_stars_lr, dtype=np.float32)
                    g.create_dataset(b'MinDistStarsHR', data=self.subhalo_distances_stars_hr, dtype=np.float32)

    def read_counts_single(self, i):
        """Read the counts from a single FOF and snapshot file."""
        if self.n_groups_tot > 0:
            with h5py.File(fof_pre + f'{i}.hdf5', 'r') as f:
                header = f['Header'].attrs
                self.n_groups[i] = header['Ngroups_ThisFile']
                self.n_subhalos[i] = header['Nsubhalos_ThisFile']

        with h5py.File(snap_pre + f'{i}.hdf5', 'r') as f:
            header = f['Header'].attrs
            nums = header['NumPart_ThisFile'] # Number of particles in this file by type
            self.n_gas[i] = nums[0] # Number of gas particles in this file
            self.n_dm[i] = nums[1] # Number of PartType1 particles in this file
            self.n_p2[i] = nums[2] # Number of PartType2 particles in this file
            self.n_p3[i] = nums[3] # Number of PartType3 particles in this file
            self.n_stars[i] = nums[4] # Number of star particles in this file

    def read_counts(self):
        """Read the counts from the FOF and snapshot files."""
        for i in range(self.n_files):
            self.read_counts_single(i)

    def read_counts_asyncio(self):
        """Read the counts from the FOF and snapshot files using asyncio."""
        async def read_counts_async():
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                await asyncio.gather(*(loop.run_in_executor(executor, self.read_counts_single, i) for i in range(self.n_files)))
        asyncio.run(read_counts_async())

    def read_groups_single(self, i):
        """Read the group data from a single FOF file."""
        if self.n_groups[i] > 0: # Skip empty files
            with h5py.File(fof_pre + f'{i}.hdf5', 'r') as f:
                g = f['Group']
                offset = self.first_group[i] # Offset to the first group
                next_offset = offset + self.n_groups[i] # Offset beyond the last group
                self.GroupPos[offset:next_offset] = g['GroupPos'][:]
                self.Group_R_Crit200[offset:next_offset] = g['Group_R_Crit200'][:]
                self.Group_M_Crit200[offset:next_offset] = g['Group_M_Crit200'][:]
                self.GroupMassType[offset:next_offset] = g['GroupMassType'][:]
                self.GroupNsubs[offset:next_offset] = g['GroupNsubs'][:]
                self.GroupLenType[offset:next_offset] = g['GroupLenType'][:]
        if self.n_subhalos[i] > 0: # Skip empty files
            with h5py.File(fof_pre + f'{i}.hdf5', 'r') as f:
                g = f['Subhalo']
                offset = self.first_subhalo[i] # Offset to the first subhalo
                next_offset = offset + self.n_subhalos[i] # Offset beyond the last subhalo
                self.SubhaloPos[offset:next_offset] = g['SubhaloPos'][:]
                self.SubhaloMass[offset:next_offset] = g['SubhaloMass'][:]
                self.SubhaloGroupNr[offset:next_offset] = g['SubhaloGroupNr'][:]
                self.SubhaloLenType[offset:next_offset] = g['SubhaloLenType'][:]

    def read_groups(self):
        """Read the group data from the FOF files."""
        for i in range(self.n_files):
            self.read_groups_single(i)

    def read_groups_asyncio(self):
        """Read the group data from the FOF files using asyncio."""
        async def read_groups_async():
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                await asyncio.gather(*(loop.run_in_executor(executor, self.read_groups_single, i) for i in range(self.n_files)))
        asyncio.run(read_groups_async())

    def read_snaps_single(self, i):
        """Read the particle data from a single snapshot file."""
        with h5py.File(snap_pre + f'{i}.hdf5', 'r') as f:
            if self.n_gas[i] > 0:
                offset = self.first_gas[i]
                next_offset = offset + self.n_gas[i]
                self.r_gas[offset:next_offset] = f['PartType0/Coordinates'][:]
                self.m_gas[offset:next_offset] = f['PartType0/Masses'][:]
                self.m_gas_HR[offset:next_offset] = f['PartType0/HighResGasMass'][:]
            if self.n_dm[i] > 0:
                offset = self.first_dm[i]
                next_offset = offset + self.n_dm[i]
                self.r_dm[offset:next_offset] = f['PartType1/Coordinates'][:]
            if self.n_p2[i] > 0:
                offset = self.first_p2[i]
                next_offset = offset + self.n_p2[i]
                self.r_p2[offset:next_offset] = f['PartType2/Coordinates'][:]
                self.m_p2[offset:next_offset] = f['PartType2/Masses'][:]
            if self.n_p3[i] > 0:
                offset = self.first_p3[i]
                next_offset = offset + self.n_p3[i]
                self.r_p3[offset:next_offset] = f['PartType3/Coordinates'][:]
            if self.n_stars[i] > 0:
                offset = self.first_star[i]
                next_offset = offset + self.n_stars[i]
                self.r_stars[offset:next_offset] = f['PartType4/Coordinates'][:]
                self.m_stars[offset:next_offset] = f['PartType4/Masses'][:]
                self.is_HR[offset:next_offset] = f['PartType4/IsHighRes'][:]

    def read_snaps(self):
        """Read the particle data from the snapshot files."""
        for i in range(self.n_files):
            self.read_snaps_single(i)

    def read_snaps_asyncio(self):
        """Read the particle data from the snapshot files using asyncio."""
        async def read_snaps_async():
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                await asyncio.gather(*(loop.run_in_executor(executor, self.read_snaps_single, i) for i in range(self.n_files)))
        asyncio.run(read_snaps_async())

def center_of_mass_gas_numpy(m: np.ndarray, m_hr: np.ndarray, r: np.ndarray, hr_threshold: float) -> Tuple[np.uint64, np.float64, np.ndarray]:
    """Calculate the center of mass of high-resolution particles."""
    # Calculate center of mass
    mask = m_hr < hr_threshold * m # Mask for low-resolution gas
    m_hr[mask] = 0. # Zero out low-resolution gas
    n_lr = np.count_nonzero(mask) # Number of low-resolution particles
    n_hr = len(m) - n_lr # Number of high-resolution particles
    m_hr_tot = np.sum(m_hr) # Total mass of high-resolution particles
    r_hr_tot = np.zeros(3) # High-resolution center of mass
    if m_hr_tot > 0.:
        for i in range(3):
            r_hr_tot[i] = np.sum(m_hr * r[:,i]) / m_hr_tot # High-resolution center of mass
    return n_hr, m_hr_tot, r_hr_tot

@jit(nopython=True, nogil=True, parallel=True)
def center_of_mass_gas_numba(m: np.ndarray, m_hr: np.ndarray, r: np.ndarray, hr_threshold: float) -> Tuple[np.uint64, np.float64, np.ndarray]:
    # Calculate center of mass
    n = len(m)
    n_hr = 0 # Number of high-resolution particles
    m_hr_tot = 0. # Total mass of high-resolution particles
    r_hr_tot = np.zeros(3) # High-resolution center of mass
    for i in prange(n):
        if m_hr[i] < hr_threshold * m[i]:
            m_hr[i] = 0. # Zero out low-resolution gas
        else:
            r_hr_tot += m_hr[i] * r[i] # Accumulate the high-resolution center of mass
            m_hr_tot += m_hr[i] # Accumulate the high-resolution mass
            n_hr += 1 # Count the number of high-resolution particles
    if m_hr_tot > 0.:
        r_hr_tot /= m_hr_tot # Normalize the high-resolution center of mass
    return n_hr, m_hr_tot, r_hr_tot

def center_of_mass_dm_numpy(r: np.ndarray) -> np.ndarray:
    """Calculate the center of mass of high-resolution particles."""
    return np.mean(r, axis=0) # Center of mass

@jit(nopython=True, nogil=True, parallel=True)
def center_of_mass_dm_numba(r: np.ndarray) -> np.ndarray:
    # Calculate center of mass
    n = len(r)
    r_com = np.zeros(3)
    for i in prange(n):
        r_com += r[i] # Accumulate center of mass
    r_com /= float(n) # Normalize the center of mass
    return r_com

def center_of_mass_stars_numpy(m: np.ndarray, r: np.ndarray, is_hr: np.ndarray) -> Tuple[np.uint64, np.float64, np.ndarray]:
    """Calculate the center of mass of high-resolution particles."""
    # Calculate center of mass
    mask = is_hr == 0 # Mask for low-resolution stars
    n_lr = np.count_nonzero(mask) # Number of low-resolution particles
    n_hr = len(m) - n_lr # Number of high-resolution particles
    m_hr_tot = np.sum(m[mask]) # Total mass of high-resolution particles
    r_hr_tot = np.zeros(3) # High-resolution center of mass
    if m_hr_tot > 0.:
        for i in range(3):
            r_hr_tot[i] = np.sum(m[mask] * r[mask,i]) / m_hr_tot # High-resolution center of mass
    return n_hr, m_hr_tot, r_hr_tot

@jit(nopython=True, nogil=True, parallel=True)
def center_of_mass_stars_numba(m: np.ndarray, r: np.ndarray, is_hr: np.ndarray) -> Tuple[np.uint64, np.float64, np.ndarray]:
    # Calculate center of mass
    n = len(m)
    n_hr = 0 # Number of high-resolution particles
    m_hr_tot = 0. # Total mass of high-resolution particles
    r_hr_tot = np.zeros(3) # High-resolution center of mass
    for i in prange(n):
        if is_hr[i] != 0:
            r_hr_tot += m[i] * r[i] # Accumulate the high-resolution center of mass
            m_hr_tot += m[i] # Accumulate the high-resolution mass
            n_hr += 1 # Count the number of high-resolution particles
    if m_hr_tot > 0.:
        r_hr_tot /= m_hr_tot # Normalize the high-resolution center of mass
    return n_hr, m_hr_tot, r_hr_tot

def main():
    # Setup simulation parameters
    if TIMERS: t1 = time()
    sim = Simulation()
    print(' ___       ___  __             \n'
          '  |  |__| |__  /__`  /\\  |\\ |\n'
          '  |  |  | |___ .__/ /--\\ | \\|\n' +
          f'\nInput Directory: {out_dir}' +
          f'\nSnap {snap}: Ngroups = {sim.n_groups_tot}, Nsubhalos = {sim.n_subhalos_tot}' +
          f'\nNumGas = {sim.n_gas_tot}, NumP1 = {sim.n_dm_tot}, NumP2 = {sim.n_p2_tot}, NumP3 = {sim.n_p3_tot}, NumStar = {sim.n_stars_tot}' +
          f'\nz = {1./sim.a - 1.:g}, a = {sim.a:g}, h = {sim.h:g}, BoxSize = {1e-3*sim.BoxSize:g} cMpc/h = {1e-3*sim.BoxSize/sim.h:g} cMpc\n')
    if TIMERS: t2 = time(); print(f'Time to setup simulation: {t2 - t1:g} s'); t1 = t2

    # Read the counts from the FOF and snapshot files
    if SERIAL in READ_COUNTS:
        sim.read_counts()
        if TIMERS: t2 = time(); print(f'Time to read counts from files: {t2 - t1:g} s [serial]'); t1 = t2
    if ASYNCIO in READ_COUNTS:
        sim.read_counts_asyncio()
        if TIMERS: t2 = time(); print(f'Time to read counts from files: {t2 - t1:g} s [asyncio]'); t1 = t2
    sim.convert_counts()
    if VERBOSITY > 1: sim.print_offsets()
    if TIMERS: t2 = time(); print(f'Time to convert counts to offsets: {t2 - t1:g} s'); t1 = t2

    # Read the group data from the FOF files
    print('\nReading fof data...')
    if SERIAL in READ_GROUPS:
        sim.read_groups()
        if TIMERS: t2 = time(); print(f'Time to read group data from files: {t2 - t1:g} s [serial]'); t1 = t2
    if ASYNCIO in READ_GROUPS:
        sim.read_groups_asyncio()
        if TIMERS: t2 = time(); print(f'Time to read group data from files: {t2 - t1:g} s [asyncio]'); t1 = t2
    if VERBOSITY > 1: sim.print_groups()

    # Count the number of groups with low-resolution particles
    if sim.n_groups_tot > 0:
        sim.n_groups_LR = np.uint64(np.count_nonzero(sim.GroupMassType[:,2] + sim.GroupMassType[:,3] > 0))
        sim.n_groups_HR = sim.n_groups_tot - sim.n_groups_LR
        print(f'\nn_groups_LR = {sim.n_groups_LR}, n_groups_HR = {sim.n_groups_HR}')
        if TIMERS: t2 = time(); print(f'Time to count low-resolution groups: {t2 - t1:g} s'); t1 = t2

    # Read the particle data from the snapshot files
    print('\nReading snapshot data...')
    if SERIAL in READ_SNAPS:
        sim.read_snaps()
        if TIMERS: t2 = time(); print(f'Time to read particle data from files: {t2 - t1:g} s [serial]'); t1 = t2
    if ASYNCIO in READ_SNAPS:
        sim.read_snaps_asyncio()
        if TIMERS: t2 = time(); print(f'Time to read particle data from files: {t2 - t1:g} s [asyncio]'); t1 = t2
    if VERBOSITY > 1: sim.print_particles()

    # Calculate the minimum distance to the nearest low-resolution particle
    if NUMPY in CALC_COM:
        sim.highres_center_of_mass_numpy()
        if TIMERS: t2 = time(); print(f'Time to calculate high-resolution center of mass: {t2 - t1:g} s [numpy]'); t1 = t2
        if VERBOSITY > 0:
            print(f'Center of mass of high-resolution gas = {sim.r_gas_com} ckpc/h, mass = {sim.m_gas_com:g} 10^10 Msun/h, n_HR = {sim.n_gas_com}')
            print(f'Center of mass of high-resolution dm = {sim.r_dm_com} ckpc/h, mass = {sim.m_dm_com:g} 10^10 Msun/h, n_HR = {sim.n_dm_com}')
            print(f'Center of mass of high-resolution stars = {sim.r_stars_com} ckpc/h, mass = {sim.m_stars_com:g} 10^10 Msun/h, n_HR = {sim.n_stars_com}')
    if NUMBA in CALC_COM:
        sim.highres_center_of_mass_numba() # Calculate the center of mass of high-resolution particles
        if TIMERS: t2 = time(); print(f'Time to calculate high-resolution center of mass: {t2 - t1:g} s [numba]'); t1 = t2
        if VERBOSITY > 0:
            print(f'Center of mass of high-resolution gas = {sim.r_gas_com} ckpc/h, mass = {sim.m_gas_com:g} 10^10 Msun/h, n_HR = {sim.n_gas_com}')
            print(f'Center of mass of high-resolution dm = {sim.r_dm_com} ckpc/h, mass = {sim.m_dm_com:g} 10^10 Msun/h, n_HR = {sim.n_dm_com}')
            print(f'Center of mass of high-resolution stars = {sim.r_stars_com} ckpc/h, mass = {sim.m_stars_com:g} 10^10 Msun/h, n_HR = {sim.n_stars_com}')
    print(f'Center of mass of high-resolution total = {sim.r_com} ckpc/h, mass = {sim.m_com:g} 10^10 Msun/h')
    print('\nCalculating high-resolution radius...')
    sim.highres_radius()
    if TIMERS: t2 = time(); print(f'Time to calculate high-resolution radius: {t2 - t1:g} s'); t1 = t2

    if sim.n_groups_tot > 0 or sim.n_subhalos_tot > 0:
        # Build trees for different particle types
        print('\nBuilding trees for different particle types...')
        sim.build_trees()
        if TIMERS: t2 = time(); print(f'Time to build trees: {t2 - t1:g} s'); t1 = t2

    if sim.n_groups_tot > 0:
        # Find the nearest distance from each group position
        print('\nFinding the nearest distance from each group position...')
        sim.find_nearest_group()
        if TIMERS: t2 = time(); print(f'Time to find nearest distances: {t2 - t1:g} s'); t1 = t2

    if sim.n_subhalos_tot > 0:
        # Find the nearest distance from each group position
        print('\nFinding the nearest distance from each subhalo position...')
        sim.find_nearest_subhalo()
        if TIMERS: t2 = time(); print(f'Time to find nearest distances: {t2 - t1:g} s'); t1 = t2

    # Calculate the average density
    print('\nCalculating the average density...')
    sim.calculate_average_density()
    if TIMERS: t2 = time(); print(f'Time to calculate density: {t2 - t1:g} s'); t1 = t2

    if sim.n_subhalos_tot > 0:
        # Calculate the virial radius of each subhalo
        print('\nCalculating the virial radius of each subhalo...')
        sim.setup_R_vir()  # Initialization for calculating R_vir
        # sim.calculate_R_vir_hist()  # Calculated via global histograms
        sim.setup_full_trees()  # Trees for all gas/star particles
        if TIMERS: t2 = time(); print(f'Time to initialize tree calculations: {t2 - t1:g} s'); t1 = t2
        sim.calculate_R_vir_tree()  # Calculated via local trees
        if TIMERS: t2 = time(); print(f'Time to calculate virial radii: {t2 - t1:g} s'); t1 = t2

    # Write the results to a file
    print('\nWriting the results to a file...')
    sim.write()
    if TIMERS: t2 = time(); print(f'Time to write results to a file: {t2 - t1:g} s'); t1 = t2

if __name__ == '__main__':
    main()
