import h5py, os, platform, asyncio
import numpy as np
from dataclasses import dataclass, field
from time import time
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

# Constants
VERBOSITY = 0 # Level of print verbosity
MAX_WORKERS = cpu_count() # Maximum number of workers
SERIAL = 1 # Run in serial
ASYNCIO = 2 # Run in parallel (asyncio)
READ_DEFAULT = (ASYNCIO,) # Default read method
# READ_DEFAULT = (SERIAL,ASYNCIO) # Default read method
READ_COUNTS = READ_DEFAULT # Read counts methods
READ_GROUPS = READ_DEFAULT # Read groups methods
READ_SNAPS = READ_DEFAULT # Read snapshots methods
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
        raise ValueError('Usage: python offsets.py [sim] [snap] [zoom_dir]')

out_dir = f'{zoom_dir}/{sim}/output'
off_dir = f'{zoom_dir}/{sim}/postprocessing/offsets'

# Overwrite for local testing
#out_dir = '.' # Overwrite for local testing
#off_dir = '.' # Overwrite for local testing

# The following paths should not change
fof_pre = f'{out_dir}/groups_{snap:03d}/fof_subhalo_tab_{snap:03d}.'
snap_pre = f'{out_dir}/snapdir_{snap:03d}/snapshot_{snap:03d}.'
offsets_file = f'{off_dir}/offsets_{snap:03d}.hdf5'

@dataclass
class Simulation:
    """Simulation information and data."""
    n_files: np.int32 = 0 # Number of files
    NUM_PART: np.int32 = 7 # Number of particle types
    n_groups_tot: np.int64 = None # Total number of groups
    n_subhalos_tot: np.int64 = None # Total number of subhalos
    n_parts_tot: np.ndarray = None # Total number of particles (all types)
    a: np.float64 = None # Scale factor
    BoxSize: np.float64 = None # Size of the simulation volume
    h: np.float64 = None # Hubble parameter

    # File counts and offsets
    n_groups: np.ndarray = None
    n_subhalos: np.ndarray = None
    n_parts: np.ndarray = None
    first_group: np.ndarray = None
    first_subhalo: np.ndarray = None
    first_part: np.ndarray = None

    # FOF data
    groups: dict = field(default_factory=dict)
    GroupNsubs: np.ndarray = None
    GroupLenType: np.ndarray = None
    subhalos: dict = field(default_factory=dict)
    SubhaloLenType: np.ndarray = None

    def __post_init__(self):
        """Allocate memory for group and particle data."""
        # Read header info from the snapshot files
        with h5py.File(fof_pre + '0.hdf5', 'r') as f:
            header = f['Header'].attrs
            self.n_files = header['NumFiles']
            self.n_groups_tot = header['Ngroups_Total'].astype(np.int64)
            self.n_subhalos_tot = header['Nsubhalos_Total'].astype(np.int64)
            self.n_groups = np.zeros(self.n_files, dtype=np.int64)
            self.n_subhalos = np.zeros(self.n_files, dtype=np.int64)
            if self.n_groups_tot > 0:
                g = f['Group']
                for field in ['GroupNsubs', 'GroupLenType']:
                    shape = (self.n_groups_tot,) + g[field].shape[1:]
                    self.groups[field] = np.empty(shape, dtype=np.int64)
            if self.n_subhalos_tot > 0:
                g = f['Subhalo']
                for field in ['SubhaloLenType']:
                    shape = (self.n_subhalos_tot,) + g[field].shape[1:]
                    self.subhalos[field] = np.empty(shape, dtype=np.int64)

        with h5py.File(snap_pre + '0.hdf5', 'r') as f:
            header = f['Header'].attrs
            self.n_parts_tot = header['NumPart_Total'][:]
            self.NUM_PART = np.int32(len(self.n_parts_tot))
            self.n_parts = np.zeros([self.n_files, self.NUM_PART], dtype=np.int64)
            self.a = header['Time']
            params = f['Parameters'].attrs
            self.BoxSize = params['BoxSize']
            self.h = params['HubbleParam']

        # Group data
        if self.n_groups_tot > 0:
            self.GroupNsubs = self.groups['GroupNsubs']
            self.GroupLenType = self.groups['GroupLenType']

        # Subhalo data
        if self.n_subhalos_tot > 0:
            self.SubhaloLenType = self.subhalos['SubhaloLenType']

    def convert_counts(self):
        """Convert the counts to file offsets."""
        self.first_group = np.cumsum(self.n_groups) - self.n_groups
        self.first_subhalo = np.cumsum(self.n_subhalos) - self.n_subhalos
        self.first_part = np.cumsum(self.n_parts, axis=0) - self.n_parts

    def convert_groups(self):
        """Convert the group and subhalo data to particle offsets."""
        if self.n_groups_tot > 0:
            self.GroupFirstSub = np.cumsum(self.GroupNsubs) - self.GroupNsubs # First subhalo in each group
            self.GroupFirstType = np.cumsum(self.GroupLenType, axis=0) - self.GroupLenType # First particle of each type in each group
        if self.n_subhalos_tot > 0:
            self.SubhaloFirstType = np.zeros_like(self.SubhaloLenType) # First particle of each type in each subhalo
            for i in range(self.n_groups_tot):
                i_beg, i_end = self.GroupFirstSub[i], self.GroupFirstSub[i] + self.GroupNsubs[i] # Subhalo range
                first_subs = np.cumsum(self.SubhaloLenType[i_beg:i_end], axis=0) - self.SubhaloLenType[i_beg:i_end] # Relative offsets
                for i_part in range(self.NUM_PART):
                    first_subs[:,i_part] += self.GroupFirstType[i,i_part] # Add group offset
                self.SubhaloFirstType[i_beg:i_end] = first_subs # First particle of each type in each subhalo

    def print_offsets(self):
        """Print the file counts and offsets."""
        print(f'n_groups = {self.n_groups} (n_groups_tot = {self.n_groups_tot})')
        print(f'n_subhalos = {self.n_subhalos} (n_subhalos_tot = {self.n_subhalos_tot})')
        print(f'n_parts = {self.n_parts} (n_parts_tot = {self.n_parts_tot})')
        print(f'first_group = {self.first_group} (n_groups_tot = {self.n_groups_tot})')
        print(f'first_subhalo = {self.first_subhalo} (n_subhalos_tot = {self.n_subhalos_tot})')
        print(f'first_part = {self.first_part} (n_parts_tot = {self.n_parts_tot})')

    def print_groups(self):
        """Print the group data."""
        if self.n_groups_tot > 0:
            print(f'GroupNsubs = {self.GroupNsubs}')
            print(f'GroupLenType = {self.GroupLenType}')
            print(f'GroupFirstSub = {self.GroupFirstSub}')
            print(f'GroupFirstType = {self.GroupFirstType}')
        if self.n_subhalos_tot > 0:
            print(f'SubhaloLenType = {self.SubhaloLenType}')
            print(f'SubhaloFirstType = {self.SubhaloFirstType}')

    def write(self):
        """Write the distance results to an HDF5 file."""
        with h5py.File(offsets_file, 'w') as f:
            g = f.create_group(b'FileOffsets')
            g.create_dataset(b'Group', data=self.first_group)
            g.create_dataset(b'Subhalo', data=self.first_subhalo)
            g.create_dataset(b'SnapByType', data=self.first_part)
            if self.n_groups_tot > 0:
                g = f.create_group(b'Group')
                g.create_dataset(b'SnapByType', data=self.GroupFirstType)
            if self.n_subhalos_tot > 0:
                g = f.create_group(b'Subhalo')
                g.create_dataset(b'SnapByType', data=self.SubhaloFirstType)

    def read_counts_single(self, i):
        """Read the counts from a single FOF and snapshot file."""
        if self.n_groups_tot > 0:
            with h5py.File(fof_pre + f'{i}.hdf5', 'r') as f:
                header = f['Header'].attrs
                self.n_groups[i] = header['Ngroups_ThisFile']
                self.n_subhalos[i] = header['Nsubhalos_ThisFile']

        with h5py.File(snap_pre + f'{i}.hdf5', 'r') as f:
            header = f['Header'].attrs
            self.n_parts[i,:] = header['NumPart_ThisFile'][:] # Number of particles in this file by type

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
                self.GroupNsubs[offset:next_offset] = g['GroupNsubs'][:]
                self.GroupLenType[offset:next_offset] = g['GroupLenType'][:]
        if self.n_subhalos[i] > 0: # Skip empty files
            with h5py.File(fof_pre + f'{i}.hdf5', 'r') as f:
                g = f['Subhalo']
                offset = self.first_subhalo[i] # Offset to the first subhalo
                next_offset = offset + self.n_subhalos[i] # Offset beyond the last subhalo
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

def main():
    os.makedirs(off_dir, exist_ok=True) # Ensure the offsets directory exists
    # Setup simulation parameters
    if TIMERS: t1 = time()
    sim = Simulation()
    if VERBOSITY > 0:
        print(' ___       ___  __             \n'
              '  |  |__| |__  /__`  /\\  |\\ |\n'
              '  |  |  | |___ .__/ /--\\ | \\|\n' +
              f'\nInput Directory: {out_dir}' +
              f'\nSnap {snap}: Ngroups = {sim.n_groups_tot}, Nsubhalos = {sim.n_subhalos_tot}, Nparts = {sim.n_parts_tot}' +
              f'\nz = {1./sim.a - 1.:g}, a = {sim.a:g}, h = {sim.h:g}, BoxSize = {1e-3*sim.BoxSize:g} cMpc/h = {1e-3*sim.BoxSize/sim.h:g} cMpc\n')

    # Read the counts and data from the FOF and snapshot files
    sim.read_counts_asyncio()
    sim.convert_counts()
    if VERBOSITY > 1:
        sim.print_offsets()
    sim.read_groups_asyncio()
    sim.convert_groups()
    sim.write()
    if VERBOSITY > 1:
        sim.print_groups()
    if TIMERS: t2 = time(); print(f'Time: {t2 - t1:g} s'); t1 = t2

if __name__ == '__main__':
    main()
