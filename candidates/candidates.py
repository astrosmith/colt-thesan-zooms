import h5py
import numpy as np
from dataclasses import dataclass, field
from time import time
from typing import Tuple
import asyncio
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
READ_DEFAULT = (ASYNCIO,) # Default read method
# READ_DEFAULT = (SERIAL,ASYNCIO) # Default read method
READ_COUNTS = READ_DEFAULT # Read counts methods
READ_GROUPS = READ_DEFAULT # Read groups methods
TIMERS = True # Print timers

# Global variables
sim = 'g5760/z4'
snap = 188 # Snapshot number
zoom_dir = '/net/hstor001.ib/data2/group/mvogelsb/004/Thesan-Zooms'
out_dir = f'{zoom_dir}/{sim}/output'
dist_dir = f'{zoom_dir}/{sim}/postprocessing/distances'
cand_dir = f'{zoom_dir}/{sim}/postprocessing/candidates'

# Overwrite for local testing
#out_dir = '.' # Overwrite for local testing
#dist_dir = '.' # Overwrite for local testing
#cand_dir = '.' # Overwrite for local testing

# The following paths should not change
fof_pre = f'{out_dir}/groups_{snap:03d}/fof_subhalo_tab_{snap:03d}.'
dist_file = f'{dist_dir}/distances_{snap:03d}.hdf5'
cand_file = f'{cand_dir}/candidates_{snap:03d}.hdf5'

@dataclass
class Simulation:
    """Simulation information and data."""
    n_files: np.int32 = 0 # Number of files
    n_groups_tot: np.uint64 = None # Total number of groups
    n_subhalos_tot: np.uint64 = None # Total number of subhalos
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

    # File counts and offsets
    n_groups: np.ndarray = None
    n_subhalos: np.ndarray = None
    first_group: np.ndarray = None
    first_subhalo: np.ndarray = None

    # FOF data
    groups: dict = field(default_factory=dict)
    group_units: dict = field(default_factory=dict)
    subhalos: dict = field(default_factory=dict)
    subhalo_units: dict = field(default_factory=dict)

    def __post_init__(self):
        """Allocate memory for group data."""
        # Read header info from the fof files
        with h5py.File(fof_pre + '0.hdf5', 'r') as f:
            header = f['Header'].attrs
            self.n_files = header['NumFiles']
            self.n_groups_tot = header['Ngroups_Total']
            self.n_subhalos_tot = header['Nsubhalos_Total']
            self.n_groups = np.zeros(self.n_files, dtype=np.uint64)
            self.n_subhalos = np.zeros(self.n_files, dtype=np.uint64)
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
            g = f['Group']
            for field in g.keys():
                shape, dtype = g[field].shape, g[field].dtype
                shape = (self.n_groups_tot,) + shape[1:]
                self.groups[field] = np.empty(shape, dtype=dtype)
                g_attrs = g[field].attrs
                if len(g_attrs) > 0:
                    self.group_units[field] = {key: val for key,val in g_attrs.items()}
                    print(f'field: {self.group_units[field]}')
            g = f['Subhalo']
            for field in g.keys():
                shape, dtype = g[field].shape, g[field].dtype
                shape = (self.n_subhalos_tot,) + shape[1:]
                self.subhalos[field] = np.empty(shape, dtype=dtype)
                g_attrs = g[field].attrs
                if len(g_attrs) > 0:
                    self.subhalo_units[field] = {key: val for key,val in g_attrs.items()}
                    print(f'field: {self.subhalo_units[field]}')

        # Derived quantities
        self.BoxHalf = self.BoxSize / 2.
        self.length_to_cgs = self.a * self.UnitLength_in_cm / self.h
        self.volume_to_cgs = self.length_to_cgs**3
        self.mass_to_cgs = self.UnitMass_in_g / self.h
        self.mass_to_msun = self.mass_to_cgs / SOLAR_MASS
        self.velocity_to_cgs = np.sqrt(self.a) * self.UnitVelocity_in_cm_per_s

        # Read distances info
        with h5py.File(dist_file, 'r') as f:
            header = f['Header'].attrs
            self.r_com = header['PosHR']
            self.RadiusHR = header['RadiusHR']
            self.RadiusLR = header['RadiusLR']
            self.NumGasHR = header['NumGasHR']
            self.NumGasLR = header['NumGasLR']
            g = f['Group']
            self.distances_gas_lr = g['MinDistGasLR'][:]
            self.distances_gas_hr = g['MinDistGasHR'][:]
            self.distances_dm = g['MinDistDM'][:]
            self.distances_p2 = g['MinDistP2'][:]
            self.distances_p3 = g['MinDistP3'][:]
            self.distances_stars_lr = g['MinDistStarsLR'][:]
            self.distances_stars_hr = g['MinDistStarsHR'][:]
            # self.GroupPos = g['GroupPos'][:]
            self.Group_R_Crit200 = g['R_Crit200'][:]
            # self.Group_M_Crit200 = g['M_Crit200'][:]
            # Calculate the minimum distance to a low-resolution particle
            self.distances_lr = np.minimum(self.distances_p2, self.distances_p3, self.distances_stars_lr) # P2,P3,StarsLR

        # Mask out groups with R_Crit200 == 0 [R_Crit200 > 0]
        # Require at least 1 high-resolution star particle [MinDist(StarsHR) < R_Crit200]
        # Require at least 1 high-resolution gas particle [MinDist(GasHR) < R_Crit200]
        # Mask out groups with MinDistP2 or MinDistP3 < R_Crit200 [MinDist(P2,P3,StarsLR) > R_Crit200]
        self.group_mask = (self.Group_R_Crit200 > 0) \
                        & (self.distances_stars_hr < self.Group_R_Crit200) \
                        & (self.distances_gas_hr < self.Group_R_Crit200) \
                        & (self.distances_lr > self.Group_R_Crit200)
        self.n_groups_candidates = np.int32(np.count_nonzero(self.group_mask))

    def convert_counts(self):
        """Convert the counts to file offsets."""
        self.first_group = np.cumsum(self.n_groups) - self.n_groups
        self.first_subhalo = np.cumsum(self.n_subhalos) - self.n_subhalos

    def print_offsets(self):
        """Print the file counts and offsets."""
        print(f'n_groups = {self.n_groups} (n_groups_tot = {self.n_groups_tot})')
        print(f'n_subhalos = {self.n_subhalos} (n_subhalos_tot = {self.n_subhalos_tot})')
        print(f'first_group = {self.first_group} (n_groups_tot = {self.n_groups_tot})')
        print(f'first_subhalo = {self.first_subhalo} (n_subhalos_tot = {self.n_subhalos_tot})')

    def print_groups(self):
        """Print the group data."""
        print(f'Group_R_Crit200 = {self.Group_R_Crit200}')

    def write(self):
        """Write the candidate results to an HDF5 file."""
        GroupID = np.arange(self.n_groups_tot, dtype=np.int32)[self.group_mask]
        self.subhalo_mask = np.zeros(self.n_subhalos_tot, dtype=bool)
        self.subhalo_mask = np.array([self.subhalos['SubhaloGroupNr'][i] in GroupID for i in range(self.n_subhalos_tot)], dtype=bool)
        self.n_subhalos_candidates = np.int32(np.count_nonzero(self.subhalo_mask))
        SubhaloID = np.arange(self.n_subhalos_tot, dtype=np.int32)[self.subhalo_mask]
        if VERBOSITY > 1:
            print(f'GroupID = {GroupID}')
            print(f'SubhaloID = {SubhaloID}')
            print(f'n_groups_candidates = {self.n_groups_candidates}')
            print(f'n_subhalos_candidates = {self.n_subhalos_candidates}')
            print(f'GroupID of each Subhalo = {self.subhalos["SubhaloGroupNr"][SubhaloID]}')
        with h5py.File(cand_file, 'w') as f:
            g = f.create_group(b'Header')
            g.attrs['Ngroups_Total'] = self.n_groups_tot
            g.attrs['Nsubhalos_Total'] = self.n_subhalos_tot
            g.attrs['Ngroups_Candidates'] = self.n_groups_candidates
            g.attrs['Nsubhalos_Candidates'] = self.n_subhalos_candidates
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
            g.attrs['NumGasHR'] = self.NumGasHR
            g.attrs['NumGasLR'] = self.NumGasLR
            g = f.create_group(b'Group')
            g.create_dataset(b'GroupID', data=GroupID)
            g.create_dataset(b'MinDistGasLR', data=self.distances_gas_lr[self.group_mask])
            g.create_dataset(b'MinDistGasHR', data=self.distances_gas_hr[self.group_mask])
            g.create_dataset(b'MinDistDM', data=self.distances_dm[self.group_mask])
            g.create_dataset(b'MinDistP2', data=self.distances_p2[self.group_mask])
            g.create_dataset(b'MinDistP3', data=self.distances_p3[self.group_mask])
            g.create_dataset(b'MinDistStarsLR', data=self.distances_stars_lr[self.group_mask])
            g.create_dataset(b'MinDistStarsHR', data=self.distances_stars_hr[self.group_mask])
            for field in self.groups.keys():
                g.create_dataset(field, data=self.groups[field][self.group_mask])
                if field in self.group_units:
                    for key,val in self.group_units[field].items():
                        g[field].attrs[key] = val
            g = f.create_group(b'Subhalo')
            g.create_dataset(b'SubhaloID', data=SubhaloID)
            for field in self.subhalos.keys():
                g.create_dataset(field, data=self.subhalos[field][self.subhalo_mask])
                if field in self.subhalo_units:
                    for key,val in self.subhalo_units[field].items():
                        g[field].attrs[key] = val

    def read_counts_single(self, i):
        """Read the counts from a single FOF and snapshot file."""
        with h5py.File(fof_pre + f'{i}.hdf5', 'r') as f:
            header = f['Header'].attrs
            self.n_groups[i] = header['Ngroups_ThisFile']
            self.n_subhalos[i] = header['Nsubhalos_ThisFile']

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
        with h5py.File(fof_pre + f'{i}.hdf5', 'r') as f:
            if self.n_groups[i] > 0: # Skip empty files
                g = f['Group']
                offset = self.first_group[i] # Offset to the first group
                next_offset = offset + self.n_groups[i] # Offset beyond the last group
                for field in g.keys():
                    self.groups[field][offset:next_offset] = g[field][:]
            if self.n_subhalos[i] > 0: # Skip empty files
                g = f['Subhalo']
                offset = self.first_subhalo[i] # Offset to the first subhalo
                next_offset = offset + self.n_subhalos[i] # Offset beyond the last subhalo
                for field in g.keys():
                    self.subhalos[field][offset:next_offset] = g[field][:]

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

def main():
    # Setup simulation parameters
    if TIMERS: t1 = time()
    sim = Simulation()
    print(' ___       ___  __             \n'
          '  |  |__| |__  /__`  /\\  |\\ |\n'
          '  |  |  | |___ .__/ /--\\ | \\|\n' +
          f'\nInput Directory: {out_dir}' +
          f'\nSnap {snap}: Ngroups = {sim.n_groups_tot}, Nsubhalos = {sim.n_subhalos_tot}' +
          f'\nNumber of candidates = {sim.n_groups_candidates}' +
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

    # Write the results to a file
    print("\nWriting the results to a file...")
    sim.write()
    if TIMERS: t2 = time(); print(f"Time to write results to a file: {t2 - t1:g} s"); t1 = t2

if __name__ == '__main__':
    main()
