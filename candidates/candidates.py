import h5py
import numpy as np
from numba import jit, prange
from dataclasses import dataclass, field
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from time import time
from typing import Tuple

# Constants
NUM_PART = 7 # Number of particle types
GAS_HIGH_RES_THRESHOLD = 0.5 # Threshold deliniating high and low resolution gas particles
SOLAR_MASS = 1.989e33  # Solar masses
VERBOSITY = 1 # Level of print verbosity
SERIAL = 1 # Run in serial
MULTIPROCESSING = 2 # Run in parallel
NUMPY = 1 # Use NumPy
NUMBA = 2 # Use Numba
READ_DEFAULT = (SERIAL,) # Default read method
# READ_DEFAULT = (SERIAL,MULTIPROCESSING) # Default read method
CALC_DEFAULT = (NUMPY,) # Calculate center of mass
# CALC_DEFAULT = (NUMPY, NUMBA) # Calculate center of mass
READ_COUNTS = READ_DEFAULT # Read counts methods
READ_GROUPS = READ_DEFAULT # Read groups methods
READ_SNAPS = READ_DEFAULT # Read snapshots methods
CALC_COM = CALC_DEFAULT # Calculate center of mass methods
TIMERS = True # Print timers

# Global variables
snap = 188 # Snapshot number
out_dir = '.'
fof_pre = f'{out_dir}/groups_{snap:03d}/fof_subhalo_tab_{snap:03d}.'
snap_pre = f'{out_dir}/snapdir_{snap:03d}/snapshot_{snap:03d}.'

@dataclass
class Registry:
    """Data type and shape registry for shared memory objects."""
    dtype: np.dtype = None
    shape: Tuple[int, ...] = field(default=None)

@dataclass
class Simulation:
    """Simulation information and shared data."""
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

    # Shared file counts and offsets
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
    GroupPos: np.ndarray = None
    Group_R_Crit200: np.ndarray = None
    Group_M_Crit200: np.ndarray = None
    GroupMassType: np.ndarray = None

    # Particle data
    r_gas: np.ndarray = None
    m_gas: np.ndarray = None
    m_gas_HR: np.ndarray = None
    r_dm: np.ndarray = None
    m_dm: np.float64 = None
    r_p2: np.ndarray = None
    r_p3: np.ndarray = None
    r_stars: np.ndarray = None
    m_stars: np.ndarray = None
    is_HR: np.ndarray = None

    # Shared memory objects
    _shared: list = field(default_factory=list)
    _registry: dict = field(default_factory=dict)

    def __post_init__(self):
        """Create shared memory objects and NumPy arrays."""
        if self.n_files == 0:
            # Read header info from the snapshot files
            with h5py.File(fof_pre + '0.hdf5', 'r') as f:
                header = f['Header'].attrs
                self.n_files = header['NumFiles']
                self.n_groups_tot = header['Ngroups_Total']
                self.n_subhalos_tot = header['Nsubhalos_Total']
                g = f['Group']
                for name in ['GroupPos', 'Group_R_Crit200', 'Group_M_Crit200', 'GroupMassType']:
                    shape = (self.n_groups_tot,) + g[name].shape[1:]
                    self._registry[name] = Registry(g[name].dtype, shape)

            with h5py.File(snap_pre + '0.hdf5', 'r') as f:
                header = f['Header'].attrs
                self.m_dm = header['MassTable'][1] # Mass of dark matter particles
                reg = Registry(dtype=header['NumPart_ThisFile'].dtype, shape=(self.n_files,))
                for name in ['Ngroups', 'Nsubhalos', 'NumGas', 'NumP1', 'NumP2', 'NumP3', 'NumStars',
                             'FirstGroup', 'FirstSubhalo', 'FirstGas', 'FirstP1', 'FirstP2', 'FirstP3', 'FirstStar']:
                    self._registry[name] = reg # Save the registry
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
                for name in ['Coordinates', 'Masses', 'HighResGasMass']:
                    name = 'PartType0/' + name
                    shape = (self.n_gas_tot,) + f[name].shape[1:]
                    self._registry[name] = Registry(f[name].dtype, shape)
                for name in ['PartType1/Coordinates']:
                    shape = (self.n_dm_tot,) + f[name].shape[1:]
                    self._registry[name] = Registry(f[name].dtype, shape)
                for name in ['PartType2/Coordinates']:
                    shape = (self.n_p2_tot,) + f[name].shape[1:]
                    self._registry[name] = Registry(f[name].dtype, shape)
                for name in ['PartType3/Coordinates']:
                    shape = (self.n_p3_tot,) + f[name].shape[1:]
                    self._registry[name] = Registry(f[name].dtype, shape)
                for name in ['Coordinates', 'Masses', 'IsHighRes']:
                    name = 'PartType4/' + name
                    shape = (self.n_stars_tot,) + f[name].shape[1:]
                    self._registry[name] = Registry(f[name].dtype, shape)

            # Derived quantities
            self.BoxHalf = self.BoxSize / 2.
            self.length_to_cgs = self.a * self.UnitLength_in_cm / self.h
            self.volume_to_cgs = self.length_to_cgs**3
            self.mass_to_cgs = self.UnitMass_in_g / self.h
            self.mass_to_msun = self.mass_to_cgs / SOLAR_MASS
            self.velocity_to_cgs = np.sqrt(self.a) * self.UnitVelocity_in_cm_per_s

        # Shared memory file counts and offsets
        self.n_groups, self.first_group = self._new("Ngroups"), self._new("FirstGroup")
        self.n_subhalos, self.first_subhalo = self._new("Nsubhalos"), self._new("FirstSubhalo")
        self.n_gas, self.first_gas = self._new("NumGas"), self._new("FirstGas")
        self.n_dm, self.first_dm = self._new("NumP1"), self._new("FirstP1")
        self.n_p2, self.first_p2 = self._new("NumP2"), self._new("FirstP2")
        self.n_p3, self.first_p3 = self._new("NumP3"), self._new("FirstP3")
        self.n_stars, self.first_star = self._new("NumStars"), self._new("FirstStar")

        # Shared memory group data
        self.GroupPos = self._new("GroupPos")
        self.Group_R_Crit200 = self._new("Group_R_Crit200")
        self.Group_M_Crit200 = self._new("Group_M_Crit200")
        self.GroupMassType = self._new("GroupMassType")

        # Shared memory gas data
        self.r_gas = self._new("PartType0/Coordinates")
        self.m_gas = self._new("PartType0/Masses")
        self.m_gas_HR = self._new("PartType0/HighResGasMass")

        # Shared memory PartType1 data
        self.r_dm = self._new("PartType1/Coordinates")

        # Shared memory PartType2 data
        self.r_p2 = self._new("PartType2/Coordinates")

        # Shared memory PartType3 data
        self.r_p3 = self._new("PartType3/Coordinates")

        # Shared memory star data
        self.r_stars = self._new("PartType4/Coordinates")
        self.m_stars = self._new("PartType4/Masses")
        self.is_HR = self._new("PartType4/IsHighRes")

    def convert_counts(self):
        """Convert the counts to file offsets."""
        self.first_group[:] = np.cumsum(self.n_groups) - self.n_groups
        self.first_subhalo[:] = np.cumsum(self.n_subhalos) - self.n_subhalos
        self.first_gas[:] = np.cumsum(self.n_gas) - self.n_gas
        self.first_dm[:] = np.cumsum(self.n_dm) - self.n_dm
        self.first_p2[:] = np.cumsum(self.n_p2) - self.n_p2
        self.first_p3[:] = np.cumsum(self.n_p3) - self.n_p3
        self.first_star[:] = np.cumsum(self.n_stars) - self.n_stars

    def _new(self, name: str) -> np.ndarray:
        """Create a shared memory group catalog object and return a reference to it."""
        try: reg = self._registry[name]; shape = reg.shape; dtype = reg.dtype
        except: raise ValueError(f"Name {name} not found in registry")
        try:
            bytes = np.dtype(dtype).itemsize # Size of the data type in bytes
            size = int(np.prod(shape)) * bytes # Total size in bytes
            shm = SharedMemory(name=name, create=True, size=size)
            self._shared.append(shm) # Save the shared memory object
        except:
            shm = SharedMemory(name=name) # Re-attach to the shared memory
        return np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    def highres_center_of_mass_numpy(self):
        """Calculate the center of mass of high-resolution particles."""
        self.n_gas_com, self.m_gas_com, self.r_gas_com = center_of_mass_gas_numpy(self.m_gas, self.m_gas_HR, self.r_gas, GAS_HIGH_RES_THRESHOLD)
        self.n_dm_com = len(self.r_dm)
        self.m_dm_com = float(self.n_dm_com) * self.m_dm
        self.r_dm_com = center_of_mass_dm_numpy(self.r_dm)
        self.n_stars_com, self.m_stars_com, self.r_stars_com = center_of_mass_stars_numpy(self.m_stars, self.r_stars, self.is_HR)
        self.m_com = self.m_gas_com + self.m_dm_com + self.m_stars_com
        self.r_com = (self.m_gas_com * self.r_gas_com + self.m_dm_com * self.r_dm_com + self.m_stars_com * self.r_stars_com) / self.m_com

    def highres_center_of_mass_numba(self):
        """Calculate the center of mass of high-resolution particles."""
        self.n_gas_com, self.m_gas_com, self.r_gas_com = center_of_mass_gas_numba(self.m_gas, self.m_gas_HR, self.r_gas, GAS_HIGH_RES_THRESHOLD)
        self.n_dm_com = len(self.r_dm)
        self.m_dm_com = float(self.n_dm_com) * self.m_dm
        self.r_dm_com = center_of_mass_dm_numba(self.r_dm)
        self.n_stars_com, self.m_stars_com, self.r_stars_com = center_of_mass_stars_numba(self.m_stars, self.r_stars, self.is_HR)
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
        # High-resolution star particles
        r = self.r_stars[self.is_HR > 0]
        for i in range(3): r[:,i] -= r_com[i]
        dr2_star_hr = np.max(np.sum(r**2, axis=1))
        # Low-resolution star particles
        r = self.r_stars[self.is_HR == 0]
        for i in range(3): r[:,i] -= r_com[i]
        dr2_star_lr = np.min(np.sum(r**2, axis=1))
        # Final high- and low-resolution radii
        self.RadiusHR = np.sqrt(max(dr2_gas_hr, dr2_dm_hr, dr2_star_hr))
        self.RadiusLR = np.sqrt(min(dr2_gas_lr, dr2_dm_lr, dr2_star_lr))
        # Count the number of particles within these distances
        self.NumGasHR = np.count_nonzero(np.sum((self.r_gas - r_com)**2, axis=1) < self.RadiusHR**2)
        self.NumGasLR = np.count_nonzero(np.sum((self.r_gas - r_com)**2, axis=1) < self.RadiusLR**2)
        print(f"\nFarthest high-resolution distance: (gas, dm, stars) = ({np.sqrt(dr2_gas_hr):g}, {np.sqrt(dr2_dm_hr):g}, {np.sqrt(dr2_star_hr):g}) ckpc/h")
        print(f"Nearest low-resolution distance: (gas, dm, stars) = ({np.sqrt(dr2_gas_lr):g}, {np.sqrt(dr2_dm_lr):g}, {np.sqrt(dr2_star_lr):g}) ckpc/h")
        print(f"RadiusHR = {self.RadiusHR:g} ckpc/h, NumGasHR = {self.NumGasHR}\nRadiusLR = {self.RadiusLR:g} ckpc/h, NumGasLR = {self.NumGasLR}")

    def build_trees(self):
        """Build the trees for the gas and star particles."""
        from scipy.spatial import cKDTree
        self.tree_gas_lr = cKDTree(self.r_gas[self.m_gas_HR == 0.]) # Low-resolution gas particles
        self.tree_gas_hr = cKDTree(self.r_gas[self.m_gas_HR > 0.]) # High-resolution gas particles
        self.tree_dm = cKDTree(self.r_dm) # PartType1 particles
        self.tree_p2 = cKDTree(self.r_p2) # PartType2 particles
        self.tree_p3 = cKDTree(self.r_p3) # PartType3 particles
        self.tree_stars_lr = cKDTree(self.r_stars[self.is_HR == 0]) # Low-resolution star particles
        self.tree_stars_hr = cKDTree(self.r_stars[self.is_HR > 0]) # High-resolution star particles

    def find_nearest(self):
        """Find the nearest distance to each group position."""
        self.distances_gas_lr, self.inices_gas_lr = self.tree_gas_lr.query(self.GroupPos) # Low-resolution gas particles
        self.distances_gas_hr, self.inices_gas_hr = self.tree_gas_hr.query(self.GroupPos) # High-resolution gas particles
        self.distances_dm, self.inices_dm = self.tree_dm.query(self.GroupPos)
        self.distances_p2, self.inices_p2 = self.tree_p2.query(self.GroupPos)
        self.distances_p3, self.inices_p3 = self.tree_p3.query(self.GroupPos)
        self.distances_stars_lr, self.inices_stars_lr = self.tree_stars_lr.query(self.GroupPos) # Low-resolution star particles
        self.distances_stars_hr, self.inices_stars_hr = self.tree_stars_hr.query(self.GroupPos) # High-resolution star particles

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
        print(f'GroupPos = {self.GroupPos}')
        print(f'Group_R_Crit200 = {self.Group_R_Crit200}')
        print(f'Group_M_Crit200 = {self.Group_M_Crit200}')
        print(f'GroupMassType = {self.GroupMassType}')

    def print_particles(self):
        """Print the particle data."""
        print(f'r_gas = {self.r_gas}')
        print(f'm_gas = {self.m_gas}')
        print(f'm_gas_HR = {self.m_gas_HR}')
        print(f'r_dm = {self.r_dm}')
        print(f'r_p2 = {self.r_p2}')
        print(f'r_p3 = {self.r_p3}')
        print(f'r_stars = {self.r_stars}')
        print(f'm_stars = {self.m_stars}')
        print(f'is_HR = {self.is_HR}')

    def write(self):
        """Write the distance results to an HDF5 file."""
        with h5py.File(f'{out_dir}/distances_{snap:03d}.hdf5', 'w') as f:
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
            g = f.create_group(b'Group')
            g.create_dataset(b'Pos', data=self.GroupPos)
            g.create_dataset(b'R_Crit200', data=self.Group_R_Crit200)
            g.create_dataset(b'M_Crit200', data=self.Group_M_Crit200)
            # g.create_dataset(b'MassType', data=self.GroupMassType)
            g.create_dataset(b'MinDistGasLR', data=self.distances_gas_lr)
            g.create_dataset(b'MinDistGasHR', data=self.distances_gas_hr)
            g.create_dataset(b'MinDistDM', data=self.distances_dm)
            g.create_dataset(b'MinDistP2', data=self.distances_p2)
            g.create_dataset(b'MinDistP3', data=self.distances_p3)
            g.create_dataset(b'MinDistStarsLR', data=self.distances_stars_lr)
            g.create_dataset(b'MinDistStarsHR', data=self.distances_stars_hr)

    def cleanup(self):
        """Clean up the shared memory objects."""
        for obj in self._shared:
            obj.close()
            obj.unlink()

    def read_counts(self):
        """Read the counts from the FOF and snapshot files."""
        for i in range(self.n_files):
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

    def read_groups(self):
        """Read the group data from the FOF files."""
        for i in range(self.n_files):
            with h5py.File(fof_pre + f'{i}.hdf5', 'r') as f:
                if self.n_groups[i] == 0: continue # Skip empty files
                g = f['Group']
                offset = self.first_group[i] # Offset to the first group
                next_offset = offset + self.n_groups[i] # Offset beyond the last group
                self.GroupPos[offset:next_offset] = g['GroupPos'][:]
                self.Group_R_Crit200[offset:next_offset] = g['Group_R_Crit200'][:]
                self.Group_M_Crit200[offset:next_offset] = g['Group_M_Crit200'][:]
                self.GroupMassType[offset:next_offset] = g['GroupMassType'][:]

    def read_snaps(self):
        """Read the particle data from the snapshot files."""
        for i in range(self.n_files):
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

def read_counts(i):
    """Read the counts from the FOF and snapshot files."""
    with h5py.File(fof_pre + f'{i}.hdf5', 'r') as f:
        header = f['Header'].attrs
        n_files = header['NumFiles']
        for name in ['Ngroups', 'Nsubhalos']:
            num = header[name+'_ThisFile'] # Number in this file
            shm = SharedMemory(name=name) # Re-attach to the shared memory
            arr = np.ndarray(n_files, dtype=type(num), buffer=shm.buf) # Access the shared memory
            arr[i] = num # Update the counts
            shm.close() # Close the shared memory block

    with h5py.File(snap_pre + f'{i}.hdf5', 'r') as f:
        header = f['Header'].attrs
        nums = header['NumPart_ThisFile'] # Number of particles in this file by type
        for name, j in zip(['NumGas', 'NumP1', 'NumP2', 'NumP3', 'NumStars'], [0, 1, 2, 3, 4]):
            num = nums[j] # Number of this particle type
            shm = SharedMemory(name=name) # Re-attach to the shared memory
            arr = np.ndarray(n_files, dtype=type(num), buffer=shm.buf) # Access the shared memory
            arr[i] = num # Update the counts
            shm.close() # Close the shared memory block

def read_groups(i):
    """Read the group data from the FOF files."""
    with h5py.File(fof_pre + f'{i}.hdf5', 'r') as f:
        header = f['Header'].attrs
        n_groups = header['Ngroups_ThisFile']
        if n_groups == 0: return # Skip empty files
        n_files = header['NumFiles']
        n_groups_tot = header['Ngroups_Total']
        g = f['Group']
        shm_group = SharedMemory(name='FirstGroup')
        first_group = np.ndarray(n_files, dtype=type(n_groups_tot), buffer=shm_group.buf)
        offset = first_group[i] # Offset to the first group
        next_offset = offset + n_groups # Offset beyond the last group
        shm_group.close() # Close the shared memory block
        for name in ['GroupPos', 'Group_R_Crit200', 'Group_M_Crit200', 'GroupMassType']:
            shape = (n_groups_tot,) + g[name].shape[1:]
            shm = SharedMemory(name=name)
            arr = np.ndarray(shape, dtype=g[name].dtype, buffer=shm.buf)
            arr[offset:next_offset] = g[name][:]
            shm.close() # Close the shared memory block

def read_snaps(i):
    """Read the particle data from the snapshot files."""
    with h5py.File(snap_pre + f'{i}.hdf5', 'r') as f:
        header = f['Header'].attrs
        n_part = header['NumPart_ThisFile']
        n_gas = n_part[0]
        n_dm = n_part[1]
        n_p2 = n_part[2]
        n_p3 = n_part[3]
        n_stars = n_part[4]
        n_tot = n_gas + n_dm + n_p2 + n_p3 + n_stars
        if n_tot == 0: return # Skip empty files
        n_files = header['NumFilesPerSnapshot']
        n_part_tot = header['NumPart_Total']
        n_gas_tot = n_part_tot[0]
        n_dm_tot = n_part_tot[1]
        n_p2_tot = n_part_tot[2]
        n_p3_tot = n_part_tot[3]
        n_stars_tot = n_part_tot[4]
        def get_offsets(name, n):
            shm = SharedMemory(name=name)
            arr = np.ndarray(n_files, dtype=n_part.dtype, buffer=shm.buf)
            offset = arr[i] # Offset to the first particle
            next_offset = offset + n # Offset beyond the last particle
            shm.close() # Close the shared memory block
            return offset, next_offset
        def process_shared_memory(name, n_tot, f):
            shape = (n_tot,) + f[name].shape[1:]
            shm = SharedMemory(name=name) # Re-attach to the shared memory
            arr = np.ndarray(shape, dtype=f[name].dtype, buffer=shm.buf)
            arr[offset:next_offset] = f[name][:] # Update the shared memory
            shm.close() # Close the shared memory block
        if n_gas > 0:
            offset, next_offset = get_offsets('FirstGas', n_gas)
            for name in ['Coordinates', 'Masses', 'HighResGasMass']:
                process_shared_memory(f'PartType0/{name}', n_gas_tot, f)
        if n_dm > 0:
            offset, next_offset = get_offsets('FirstP1', n_dm)
            for name in ['Coordinates']:
                process_shared_memory(f'PartType1/{name}', n_dm_tot, f)
        if n_p2 > 0:
            offset, next_offset = get_offsets('FirstP2', n_p2)
            for name in ['Coordinates']:
                process_shared_memory(f'PartType2/{name}', n_p2_tot, f)
        if n_p3 > 0:
            offset, next_offset = get_offsets('FirstP3', n_p3)
            for name in ['Coordinates']:
                process_shared_memory(f'PartType3/{name}', n_p3_tot, f)
        if n_stars > 0:
            offset, next_offset = get_offsets('FirstStar', n_stars)
            for name in ['Coordinates', 'Masses', 'IsHighRes']:
                process_shared_memory(f'PartType4/{name}', n_stars_tot, f)

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

    try:
        # Note: Unix creates subprocesses via a "fork" context.
        # To change to "spawn" (slower but more portable) switch to:
        # ctx = mp.get_context("spawn")
        # with ctx.Pool() as pool:
        #     pool.map(read_counts, range(sim.n_files))

        # Read the counts from the FOF and snapshot files
        if SERIAL in READ_COUNTS:
            sim.read_counts()
            if TIMERS: t2 = time(); print(f"Time to read counts from files: {t2 - t1:g} s [serial]"); t1 = t2
        if MULTIPROCESSING in READ_COUNTS:
            with Pool() as pool:
                pool.map(read_counts, range(sim.n_files))
            if TIMERS: t2 = time(); print(f"Time to read counts from files: {t2 - t1:g} s [multiprocessing]"); t1 = t2
        sim.convert_counts()
        if VERBOSITY > 1: sim.print_offsets()
        if TIMERS: t2 = time(); print(f"Time to convert counts to offsets: {t2 - t1:g} s"); t1 = t2

        # Read the group data from the FOF files
        print("\nReading fof data into shared memory...")
        if SERIAL in READ_GROUPS:
            sim.read_groups()
            if TIMERS: t2 = time(); print(f"Time to read group data from files: {t2 - t1:g} s [serial]"); t1 = t2
        if MULTIPROCESSING in READ_GROUPS:
            with Pool() as pool:
                pool.map(read_groups, range(sim.n_files))
            if TIMERS: t2 = time(); print(f"Time to read group data from files: {t2 - t1:g} s [multiprocessing]"); t1 = t2
        if VERBOSITY > 1: sim.print_groups()

        # Count the number of groups with low-resolution particles
        sim.n_groups_LR = np.uint64(np.count_nonzero(sim.GroupMassType[:,2] + sim.GroupMassType[:,3] > 0))
        sim.n_groups_HR = sim.n_groups_tot - sim.n_groups_LR
        print(f"\nn_groups_LR = {sim.n_groups_LR}, n_groups_HR = {sim.n_groups_HR}")
        if TIMERS: t2 = time(); print(f"Time to count low-resolution groups: {t2 - t1:g} s"); t1 = t2

        # Read the particle data from the snapshot files
        print("\nReading snapshot data into shared memory...")
        if SERIAL in READ_SNAPS:
            sim.read_snaps()
            if TIMERS: t2 = time(); print(f"Time to read particle data from files: {t2 - t1:g} s [serial]"); t1 = t2
        if MULTIPROCESSING in READ_SNAPS:
            with Pool() as pool:
                pool.map(read_snaps, range(sim.n_files))
            if TIMERS: t2 = time(); print(f"Time to read particle data from files: {t2 - t1:g} s [multiprocessing]"); t1 = t2
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

        # Build trees for different particle types
        print("\nBuilding trees for different particle types...")
        sim.build_trees()
        if TIMERS: t2 = time(); print(f"Time to build trees: {t2 - t1:g} s"); t1 = t2

        # Find the nearest distance from each group position
        print("\nFinding the nearest distance from each group position...")
        sim.find_nearest()
        if TIMERS: t2 = time(); print(f"Time to find nearest distances: {t2 - t1:g} s"); t1 = t2

        # Write the results to a file
        print("\nWriting the results to a file...")
        sim.write()
        if TIMERS: t2 = time(); print(f"Time to write results to a file: {t2 - t1:g} s"); t1 = t2

    finally:
        sim.cleanup()

if __name__ == '__main__':
    main()
