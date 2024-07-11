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
VERBOSITY = 1 # Level of print verbosity
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
    subhalos: dict = field(default_factory=dict)
    SubhaloPos: np.ndarray = None
    SubhaloMass: np.ndarray = None
    SubhaloGroupNr: np.ndarray = None

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
                for field in ['GroupPos', 'Group_R_Crit200', 'Group_M_Crit200', 'GroupMassType']:
                    shape, dtype = g[field].shape, g[field].dtype
                    shape = (self.n_groups_tot,) + shape[1:]
                    self.groups[field] = np.empty(shape, dtype=dtype)
            if self.n_subhalos_tot > 0:
                g = f['Subhalo']
                for field in ['SubhaloPos', 'SubhaloMass', 'SubhaloGroupNr']:
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
            self.GroupPos = self.groups["GroupPos"]
            self.Group_R_Crit200 = self.groups["Group_R_Crit200"]
            self.Group_M_Crit200 = self.groups["Group_M_Crit200"]
            self.GroupMassType = self.groups["GroupMassType"]

        # Subhalo data
        if self.n_subhalos_tot > 0:
            self.SubhaloPos = self.subhalos["SubhaloPos"]
            self.SubhaloMass = self.subhalos["SubhaloMass"]
            self.SubhaloGroupNr = self.subhalos["SubhaloGroupNr"]

        # Gas data
        self.r_gas = self.gas["Coordinates"]
        self.m_gas = self.gas["Masses"]
        self.m_gas_HR = self.gas["HighResGasMass"]

        # PartType1 data
        self.r_dm = self.dm["Coordinates"]

        # PartType2 data
        self.r_p2 = self.p2["Coordinates"]
        self.m_p2 = self.p2["Masses"]

        # PartType3 data
        self.r_p3 = self.p3["Coordinates"]

        # Star data
        if self.n_stars_tot > 0:
            self.r_stars = self.stars["Coordinates"]
            self.m_stars = self.stars["Masses"]
            self.is_HR = self.stars["IsHighRes"]

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
        print(f"\nFarthest high-resolution distance: (gas, dm, stars) = ({np.sqrt(dr2_gas_hr):g}, {np.sqrt(dr2_dm_hr):g}, {np.sqrt(dr2_star_hr):g}) ckpc/h")
        print(f"Nearest low-resolution distance: (gas, dm, stars) = ({np.sqrt(dr2_gas_lr):g}, {np.sqrt(dr2_dm_lr):g}, {np.sqrt(dr2_star_lr):g}) ckpc/h")
        print(f"RadiusHR = {self.RadiusHR:g} ckpc/h, NumGasHR = {self.NumGasHR}\nRadiusLR = {self.RadiusLR:g} ckpc/h, NumGasLR = {self.NumGasLR}")

    def build_trees(self):
        """Build the trees for the gas and star particles."""
        self.tree_gas_lr = cKDTree(self.r_gas[self.m_gas_HR == 0.]) # Low-resolution gas particles
        self.tree_gas_hr = cKDTree(self.r_gas[self.m_gas_HR > 0.]) # High-resolution gas particles
        self.tree_dm = cKDTree(self.r_dm) # PartType1 particles
        self.tree_p2 = cKDTree(self.r_p2) # PartType2 particles
        self.tree_p3 = cKDTree(self.r_p3) # PartType3 particles
        if self.n_stars_tot > 0:
            self.tree_stars_lr = cKDTree(self.r_stars[self.is_HR == 0]) # Low-resolution star particles
            self.tree_stars_hr = cKDTree(self.r_stars[self.is_HR > 0]) # High-resolution star particles

    def find_nearest_group(self):
        """Find the nearest distance to each group position."""
        self.group_distances_gas_lr, self.group_inices_gas_lr = self.tree_gas_lr.query(self.GroupPos) # Low-resolution gas particles
        self.group_distances_gas_hr, self.group_inices_gas_hr = self.tree_gas_hr.query(self.GroupPos) # High-resolution gas particles
        self.group_distances_dm, self.group_inices_dm = self.tree_dm.query(self.GroupPos)
        self.group_distances_p2, self.group_inices_p2 = self.tree_p2.query(self.GroupPos)
        self.group_distances_p3, self.group_inices_p3 = self.tree_p3.query(self.GroupPos)
        if self.n_stars_tot > 0:
            self.group_distances_stars_lr, self.group_inices_stars_lr = self.tree_stars_lr.query(self.GroupPos) # Low-resolution star particles
            self.group_distances_stars_hr, self.group_inices_stars_hr = self.tree_stars_hr.query(self.GroupPos) # High-resolution star particles

    def find_nearest_subhalo(self):
        """Find the nearest distance to each subhalo position."""
        self.subhalo_distances_gas_lr, self.subhalo_inices_gas_lr = self.tree_gas_lr.query(self.SubhaloPos) # Low-resolution gas particles
        self.subhalo_distances_gas_hr, self.subhalo_inices_gas_hr = self.tree_gas_hr.query(self.SubhaloPos) # High-resolution gas particles
        self.subhalo_distances_dm, self.subhalo_inices_dm = self.tree_dm.query(self.SubhaloPos)
        self.subhalo_distances_p2, self.subhalo_inices_p2 = self.tree_p2.query(self.SubhaloPos)
        self.subhalo_distances_p3, self.subhalo_inices_p3 = self.tree_p3.query(self.SubhaloPos)
        if self.n_stars_tot > 0:
            self.subhalo_distances_stars_lr, self.subhalo_inices_stars_lr = self.tree_stars_lr.query(self.SubhaloPos) # Low-resolution star particles
            self.subhalo_distances_stars_hr, self.subhalo_inices_stars_hr = self.tree_stars_hr.query(self.SubhaloPos) # High-resolution star particles

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
        self.Delta_c = 18. * np.pi**2 + 82. * x_c - 39. * x_c**2 # Critical overdensity factor
        self.rho_crit0 = 3. * H0**2 / (8. * np.pi * G) # Critical density today [g/cm^3]
        self.rho_crit = 3. * H2 / (8. * np.pi * G) # Critical density [g/cm^3]
        self.rho_vir = self.Delta_c * self.rho_crit # Virial density [g/cm^3]
        self.rho_vir_code = self.rho_vir / self.density_to_cgs # Virial density in code units
        if VERBOSITY > 0:
            print(f'sum(m_gas) = {self.m_gas_tot*self.mass_to_msun:g} Msun = {100.*self.m_gas_tot/self.m_tot:g}% [m_gas_res = {self.m_gas_tot*self.mass_to_msun/float(self.n_gas_tot):g} Msun]')
            print(f'sum(m_dm) = {self.m_dm_tot*self.mass_to_msun:g} Msun = {100.*self.m_dm_tot/self.m_tot:g}% [m_dm_res = {self.m_dm*self.mass_to_msun:g} Msun]')
            print(f'sum(m_p2) = {self.m_p2_tot*self.mass_to_msun:g} Msun = {100.*self.m_p2_tot/self.m_tot:g}% [m_p2_res = {self.m_p2_tot*self.mass_to_msun/float(self.n_p2_tot):g} Msun]')
            print(f'sum(m_p3) = {self.m_p3_tot*self.mass_to_msun:g} Msun = {100.*self.m_p3_tot/self.m_tot:g}% [m_p3_res = {self.m_p3*self.mass_to_msun:g} Msun]')
            print(f'sum(m_stars) = {self.m_stars_tot*self.mass_to_msun:g} Msun = {100.*self.m_stars_tot/self.m_tot:g}% [m_stars_res = {self.m_stars_tot*self.mass_to_msun/float(self.n_stars_tot):g} Msun]')
            print(f'rho_crit0 = {self.rho_crit0:g} g/cm^3, rho_crit = {self.rho_crit:g} g/cm^3, Delta_c = {self.Delta_c:g}, rho_vir = {self.rho_vir:g} g/cm^3')

    def calculate_R_vir(self):
        """Calculate the virial radius of each subhalo."""
        if self.n_subhalos_tot > 0:
            self.Subhalo_R_vir = np.zeros(self.n_subhalos_tot, dtype=np.float32) # Virial radius
            self.Subhalo_M_vir = np.zeros(self.n_subhalos_tot, dtype=np.float32) # Virial mass
            self.Subhalo_M_gas = np.zeros(self.n_subhalos_tot, dtype=np.float32) # Gas mass (<R_vir)
            self.Subhalo_M_stars = np.zeros(self.n_subhalos_tot, dtype=np.float32) # Stellar mass (<R_vir)
            self.r_min = 1e-2 * self.h / self.a # 10 pc
            self.r_max = 2. * self.BoxSize # 2 r_box
            self.logr2_min = 2. * np.log10(self.r_min) # log(r_min^2)
            self.logr2_max = 2. * np.log10(self.r_max) # log(r_max^2)
            self.n_bins = 1000 # Number of bins
            inv_dbin = float(self.n_bins) / (self.logr2_max - self.logr2_min) # n / (log(r_max^2) - log(r_min^2))
            bins = np.logspace(self.logr2_min, self.logr2_max, self.n_bins+1) # r^2 bins
            log_rbins = np.linspace(0.5*self.logr2_min, 0.5*self.logr2_max, self.n_bins+1) # log(r) bins
            dlog_rbin = 0.5 * (self.logr2_max - self.logr2_min) / float(self.n_bins) # (log(r_max) - log(r_min)) / n
            bins[0] = 0. # Extend first bin to zero
            V_enc = 4. * np.pi / 3. * bins[1:]**1.5 # Enclosed volume
            M_to_rho_vir = 1. / V_enc / self.rho_vir_code # Mass to virial density
            m_dm_full = self.m_dm * np.ones(self.n_dm_tot) # Full dark matter mass array
            m_p3_full = self.m_p3 * np.ones(self.n_p3_tot) # Full PartType3 mass array
        for i in range(self.n_subhalos_tot):
            if self.Group_R_Crit200[self.SubhaloGroupNr[i]] <= 0.:
                continue # Skip groups without a valid R_Crit200
            r_sub = self.SubhaloPos[i] # Subhalo center
            M_sub = self.SubhaloMass[i] # Subhalo mass
            M_sub_10 = 0.1 * M_sub # 10% of the subhalo mass
            # Gas particles
            r2 = (self.r_gas[:,0] - r_sub[0])**2 \
               + (self.r_gas[:,1] - r_sub[1])**2 \
               + (self.r_gas[:,2] - r_sub[2])**2 # Squared distances
            ibin = np.floor((np.log10(r2) - self.logr2_min) * inv_dbin).astype(np.int32) # Bin indices
            ibin[ibin < 0] = 0 # Clip to first bin
            M_gas_enc = np.cumsum(np.bincount(ibin, weights=self.m_gas, minlength=self.n_bins)) # Enclosed gas mass
            M_enc = np.copy(M_gas_enc) # Total enclosed mass
            # Dark matter particles
            r2 = (self.r_dm[:,0] - r_sub[0])**2 \
               + (self.r_dm[:,1] - r_sub[1])**2 \
               + (self.r_dm[:,2] - r_sub[2])**2 # Squared distances
            ibin = np.floor((np.log10(r2) - self.logr2_min) * inv_dbin).astype(np.int32) # Bin indices
            ibin[ibin < 0] = 0 # Clip to first bin
            M_enc += np.cumsum(np.bincount(ibin, weights=m_dm_full, minlength=self.n_bins)) # Add enclosed dark matter mass
            # PartType2 particles
            r2 = (self.r_p2[:,0] - r_sub[0])**2 \
               + (self.r_p2[:,1] - r_sub[1])**2 \
               + (self.r_p2[:,2] - r_sub[2])**2 # Squared distances
            ibin = np.floor((np.log10(r2) - self.logr2_min) * inv_dbin).astype(np.int32) # Bin indices
            ibin[ibin < 0] = 0 # Clip to first bin
            M_enc += np.cumsum(np.bincount(ibin, weights=self.m_p2, minlength=self.n_bins)) # Add enclosed PartType2 mass
            # PartType3 particles
            r2 = (self.r_p3[:,0] - r_sub[0])**2 \
               + (self.r_p3[:,1] - r_sub[1])**2 \
               + (self.r_p3[:,2] - r_sub[2])**2 # Squared distances
            ibin = np.floor((np.log10(r2) - self.logr2_min) * inv_dbin).astype(np.int32)
            ibin[ibin < 0] = 0
            M_enc += np.cumsum(np.bincount(ibin, weights=m_p3_full, minlength=self.n_bins)) # Add enclosed PartType3 mass
            # Star particles
            if self.n_stars_tot > 0:
                r2 = (self.r_stars[:,0] - r_sub[0])**2 \
                   + (self.r_stars[:,1] - r_sub[1])**2 \
                   + (self.r_stars[:,2] - r_sub[2])**2 # Squared distances
                ibin = np.floor((np.log10(r2) - self.logr2_min) * inv_dbin).astype(np.int32) # Bin indices
                ibin[ibin < 0] = 0 # Clip to first bin
                M_stars_enc = np.cumsum(np.bincount(ibin, weights=self.m_stars, minlength=self.n_bins)) # Enclosed stellar mass
            # Calculate the enclosed mass
            i_10 = np.argmax(M_enc > M_sub_10) # Find the first bin with M_enc > 0.1 M_vir
            rho_enc = M_enc * M_to_rho_vir # Enclosed density [rho_vir]
            # Calculate the virial radius
            i_vir = np.argmax(rho_enc[i_10:] < 1.) + i_10 # Find the first bin with rho_enc < rho_vir
            if i_vir == 0:
                self.Subhalo_R_vir[i] = -1. # Invalid R_vir
                self.Subhalo_M_vir[i] = -1.
                self.Subhalo_M_gas[i] = -1.
                self.Subhalo_M_stars[i] = -1.
            else:
                # Log interpolation to find the virial radius and masses
                frac = -np.log10(rho_enc[i_vir-1]) / np.log10(rho_enc[i_vir]/rho_enc[i_vir-1]) # Interpolation coordinate
                self.Subhalo_R_vir[i] = 10.**(log_rbins[i_vir] + frac * dlog_rbin) # Virial radius
                self.Subhalo_M_vir[i] = 10.**(np.log10(M_enc[i_vir-1]) + frac * np.log10(M_enc[i_vir]/M_enc[i_vir-1])) # Virial mass
                if M_gas_enc[i_vir-1] <= 0.:
                    self.Subhalo_M_gas[i] = M_gas_enc[i_vir-1] + frac * (M_gas_enc[i_vir] - M_gas_enc[i_vir-1]) # Gas mass (<R_vir)
                else:
                    self.Subhalo_M_gas[i] = 10.**(np.log10(M_gas_enc[i_vir-1]) + frac * np.log10(M_gas_enc[i_vir]/M_gas_enc[i_vir-1])) # Gas mass (<R_vir)
                if M_stars_enc[i_vir-1] <= 0.:
                    self.Subhalo_M_stars[i] = M_stars_enc[i_vir-1] + frac * (M_stars_enc[i_vir] - M_stars_enc[i_vir-1]) # Stellar mass (<R_vir)
                else:
                    self.Subhalo_M_stars[i] = 10.**(np.log10(M_stars_enc[i_vir-1]) + frac * np.log10(M_stars_enc[i_vir]/M_stars_enc[i_vir-1])) # Stellar mass (<R_vir)
        if VERBOSITY > 0:
            kpc = 3.085677581467192e21  # Units: 1 kpc = 3e21 cm
            print(f'R_vir = {self.Subhalo_R_vir*self.length_to_cgs/kpc} kpc')
            print(f'M_vir = {self.Subhalo_M_vir*self.mass_to_msun} Msun')
            print(f'M_gas = {self.Subhalo_M_gas*self.mass_to_msun} Msun')
            print(f'M_stars = {self.Subhalo_M_stars*self.mass_to_msun} Msun')

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
        if self.n_subhalos_tot > 0:
            print(f'SubhaloPos = {self.SubhaloPos}')
            print(f'SubhaloMass = {self.SubhaloMass}')
            print(f'SubhaloGroupNr = {self.SubhaloGroupNr}')

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
        if self.n_subhalos[i] > 0: # Skip empty files
            with h5py.File(fof_pre + f'{i}.hdf5', 'r') as f:
                g = f['Subhalo']
                offset = self.first_subhalo[i] # Offset to the first subhalo
                next_offset = offset + self.n_subhalos[i] # Offset beyond the last subhalo
                self.SubhaloPos[offset:next_offset] = g['SubhaloPos'][:]
                self.SubhaloMass[offset:next_offset] = g['SubhaloMass'][:]
                self.SubhaloGroupNr[offset:next_offset] = g['SubhaloGroupNr'][:]

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
    m[mask] = 0. # Zero out low-resolution stars
    n_lr = np.count_nonzero(mask) # Number of low-resolution particles
    n_hr = len(m) - n_lr # Number of high-resolution particles
    m_hr_tot = np.sum(m) # Total mass of high-resolution particles
    r_hr_tot = np.zeros(3) # High-resolution center of mass
    if m_hr_tot > 0.:
        for i in range(3):
            r_hr_tot[i] = np.sum(m * r[:,i]) / m_hr_tot # High-resolution center of mass
    return n_hr, m_hr_tot, r_hr_tot

@jit(nopython=True, nogil=True, parallel=True)
def center_of_mass_stars_numba(m: np.ndarray, r: np.ndarray, is_hr: np.ndarray) -> Tuple[np.uint64, np.float64, np.ndarray]:
    # Calculate center of mass
    n = len(m)
    n_hr = 0 # Number of high-resolution particles
    m_hr_tot = 0. # Total mass of high-resolution particles
    r_hr_tot = np.zeros(3) # High-resolution center of mass
    for i in prange(n):
        if is_hr[i] == 0:
            m[i] = 0. # Zero out low-resolution gas
        else:
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
    if TIMERS: t2 = time(); print(f"Time to setup simulation: {t2 - t1:g} s"); t1 = t2

    # Read the counts from the FOF and snapshot files
    if SERIAL in READ_COUNTS:
        sim.read_counts()
        if TIMERS: t2 = time(); print(f"Time to read counts from files: {t2 - t1:g} s [serial]"); t1 = t2
    if ASYNCIO in READ_COUNTS:
        sim.read_counts_asyncio()
        if TIMERS: t2 = time(); print(f"Time to read counts from files: {t2 - t1:g} s [asyncio]"); t1 = t2
    sim.convert_counts()
    if VERBOSITY > 1: sim.print_offsets()
    if TIMERS: t2 = time(); print(f"Time to convert counts to offsets: {t2 - t1:g} s"); t1 = t2

    # Read the group data from the FOF files
    print("\nReading fof data...")
    if SERIAL in READ_GROUPS:
        sim.read_groups()
        if TIMERS: t2 = time(); print(f"Time to read group data from files: {t2 - t1:g} s [serial]"); t1 = t2
    if ASYNCIO in READ_GROUPS:
        sim.read_groups_asyncio()
        if TIMERS: t2 = time(); print(f"Time to read group data from files: {t2 - t1:g} s [asyncio]"); t1 = t2
    if VERBOSITY > 1: sim.print_groups()

    # Count the number of groups with low-resolution particles
    if sim.n_groups_tot > 0:
        sim.n_groups_LR = np.uint64(np.count_nonzero(sim.GroupMassType[:,2] + sim.GroupMassType[:,3] > 0))
        sim.n_groups_HR = sim.n_groups_tot - sim.n_groups_LR
        print(f"\nn_groups_LR = {sim.n_groups_LR}, n_groups_HR = {sim.n_groups_HR}")
        if TIMERS: t2 = time(); print(f"Time to count low-resolution groups: {t2 - t1:g} s"); t1 = t2

    # Read the particle data from the snapshot files
    print("\nReading snapshot data...")
    if SERIAL in READ_SNAPS:
        sim.read_snaps()
        if TIMERS: t2 = time(); print(f"Time to read particle data from files: {t2 - t1:g} s [serial]"); t1 = t2
    if ASYNCIO in READ_SNAPS:
        sim.read_snaps_asyncio()
        if TIMERS: t2 = time(); print(f"Time to read particle data from files: {t2 - t1:g} s [asyncio]"); t1 = t2
    if VERBOSITY > 1: sim.print_particles()

    # Calculate the minimum distance to the nearest low-resolution particle
    if NUMPY in CALC_COM:
        sim.highres_center_of_mass_numpy()
        if TIMERS: t2 = time(); print(f"Time to calculate high-resolution center of mass: {t2 - t1:g} s [numpy]"); t1 = t2
        print(f"Center of mass of high-resolution gas = {sim.r_gas_com} ckpc/h, mass = {sim.m_gas_com:g} 10^10 Msun/h, n_HR = {sim.n_gas_com}")
        print(f"Center of mass of high-resolution dm = {sim.r_dm_com} ckpc/h, mass = {sim.m_dm_com:g} 10^10 Msun/h, n_HR = {sim.n_dm_com}")
        print(f"Center of mass of high-resolution stars = {sim.r_stars_com} ckpc/h, mass = {sim.m_stars_com:g} 10^10 Msun/h, n_HR = {sim.n_stars_com}")
    if NUMBA in CALC_COM:
        sim.highres_center_of_mass_numba() # Calculate the center of mass of high-resolution particles
        if TIMERS: t2 = time(); print(f"Time to calculate high-resolution center of mass: {t2 - t1:g} s [numba]"); t1 = t2
        print(f"Center of mass of high-resolution gas = {sim.r_gas_com} ckpc/h, mass = {sim.m_gas_com:g} 10^10 Msun/h, n_HR = {sim.n_gas_com}")
        print(f"Center of mass of high-resolution dm = {sim.r_dm_com} ckpc/h, mass = {sim.m_dm_com:g} 10^10 Msun/h, n_HR = {sim.n_dm_com}")
        print(f"Center of mass of high-resolution stars = {sim.r_stars_com} ckpc/h, mass = {sim.m_stars_com:g} 10^10 Msun/h, n_HR = {sim.n_stars_com}")
    print(f"Center of mass of high-resolution total = {sim.r_com} ckpc/h, mass = {sim.m_com:g} 10^10 Msun/h")
    print("\nCalculating high-resolution radius...")
    sim.highres_radius()
    if TIMERS: t2 = time(); print(f"Time to calculate high-resolution radius: {t2 - t1:g} s"); t1 = t2

    if sim.n_groups_tot > 0 or sim.n_subhalos_tot > 0:
        # Build trees for different particle types
        print("\nBuilding trees for different particle types...")
        sim.build_trees()
        if TIMERS: t2 = time(); print(f"Time to build trees: {t2 - t1:g} s"); t1 = t2

    if sim.n_groups_tot > 0:
        # Find the nearest distance from each group position
        print("\nFinding the nearest distance from each group position...")
        sim.find_nearest_group()
        if TIMERS: t2 = time(); print(f"Time to find nearest distances: {t2 - t1:g} s"); t1 = t2

    if sim.n_subhalos_tot > 0:
        # Find the nearest distance from each group position
        print("\nFinding the nearest distance from each subhalo position...")
        sim.find_nearest_subhalo()
        if TIMERS: t2 = time(); print(f"Time to find nearest distances: {t2 - t1:g} s"); t1 = t2

    # Calculate the average density
    print("\nCalculating the average density...")
    sim.calculate_average_density()
    if TIMERS: t2 = time(); print(f"Time to calculate density: {t2 - t1:g} s"); t1 = t2

    if sim.n_subhalos_tot > 0:
        # Calculate the virial radius of each subhalo
        print("\nCalculating the virial radius of each subhalo...")
        sim.calculate_R_vir()
        if TIMERS: t2 = time(); print(f"Time to calculate virial radii: {t2 - t1:g} s"); t1 = t2

    # Write the results to a file
    print("\nWriting the results to a file...")
    sim.write()
    if TIMERS: t2 = time(); print(f"Time to write results to a file: {t2 - t1:g} s"); t1 = t2

if __name__ == '__main__':
    main()
