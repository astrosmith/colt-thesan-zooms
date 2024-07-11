import numpy as np
from time import time
import h5py, os, errno, platform, asyncio
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from dataclasses import dataclass, field

VERBOSITY = 0 # Level of print verbosity
MAX_WORKERS = cpu_count() # Maximum number of workers
SERIAL = 1 # Run in serial
ASYNCIO = 2 # Run in parallel (asyncio)
READ_DEFAULT = (ASYNCIO,) # Default read method
# READ_DEFAULT = (SERIAL,ASYNCIO) # Default read method
READ_COUNTS = READ_DEFAULT # Read counts methods
READ_SNAPS = READ_DEFAULT # Read snapshots methods
TIMERS = (VERBOSITY > 0) # Print timers

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
        raise ValueError('Usage: python arepo_to_dm.py [sim] [snap] [zoom_dir]')

# Derived global variables
out_dir = f'{zoom_dir}/{sim}/output'
dist_dir = f'{zoom_dir}/{sim}/postprocessing/distances'
colt_dir = f'{zoom_dir}-COLT/{sim}/ics'

# Overwrite for local testing
#out_dir = '.'
#dist_dir = '.'
#colt_dir = '.'

# The following paths should not change
snap_pre = f'{out_dir}/snapdir_{snap:03d}/snapshot_{snap:03d}.'
dist_file = f'{dist_dir}/distances_{snap:03d}.hdf5'
dm_file = f'{colt_dir}/dm_{snap:03d}.hdf5'

# Universal constants
# c = 2.99792458e10          # Speed of light [cm/s]
# kB = 1.380648813e-16       # Boltzmann's constant [g cm^2/s^2/K]
# h = 6.626069573e-27        # Planck's constant [erg/s]
# mH = 1.6735327e-24         # Mass of hydrogen atom [g]
# me = 9.109382917e-28       # Electron mass [g]
# ee = 4.80320451e-10        # Electron charge [g^(1/2) cm^(3/2) / s]

# # Emperical unit definitions
# Msun = 1.988435e33         # Solar mass [g]
# Lsun = 3.839e33            # Solar luminosity [erg/s]
# Zsun = 0.0134              # Solar metallicity (mass fraction)
# arcsec = 648000. / np.pi   # arseconds per radian
# pc = 3.085677581467192e18  # Units: 1 pc  = 3e18 cm
# kpc = 1e3 * pc             # Units: 1 kpc = 3e21 cm
# Mpc = 1e6 * pc             # Units: 1 Mpc = 3e24 cm
# km = 1e5                   # Units: 1 km  = 1e5  cm
# angstrom = 1e-8            # Units: 1 angstrom = 1e-8 cm
# day = 86400.               # Units: 1 day = 24 * 3600 seconds
# yr = 365.24 * day          # Units: 1 year = 365.24 days
# kyr = 1e3 * yr             # Units: 1 Myr = 10^6 yr
# Myr = 1e6 * yr             # Units: 1 Myr = 10^6 yr
SOLAR_MASS = 1.989e33         # Solar masses

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

@dataclass
class Simulation:
    """Simulation information and data."""
    n_files: np.int32 = 0 # Number of files
    n_dm_tot: np.uint64 = None # Total number of PartType1 particles
    n_p2_tot: np.uint64 = None # Total number of PartType2 particles
    n_p3_tot: np.uint64 = None # Total number of PartType3 particles
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
    mass_to_Msun: np.float64 = None # Conversion factor for mass to solar masses
    velocity_to_cgs: np.float64 = None # Conversion factor for velocity to cgs
    density_to_cgs: np.float64 = None # Conversion factor for density to cgs

    # Selection criteria
    PosHR: np.ndarray = None # Center of mass of high-resolution particles
    RadiusHR: np.float64 = None # Radius of high-resolution region

    # File counts and offsets
    n_dm: np.ndarray = None
    n_p2: np.ndarray = None
    n_p3: np.ndarray = None
    first_dm: np.ndarray = None
    first_p2: np.ndarray = None
    first_p3: np.ndarray = None

    # Particle data
    dm: dict = field(default_factory=dict)
    p2: dict = field(default_factory=dict)
    p3: dict = field(default_factory=dict)
    r_dm: np.ndarray = None
    m_dm: np.float64 = None
    r_p2: np.ndarray = None
    m_p2: np.ndarray = None
    r_p3: np.ndarray = None
    m_p3: np.float64 = None

    def __post_init__(self):
        """Allocate memory for particle arrays."""
        # Read header info from the snapshot files
        with h5py.File(snap_pre + '0.hdf5', 'r') as f:
            header = f['Header'].attrs
            self.n_files = header['NumFilesPerSnapshot']
            self.m_dm = header['MassTable'][1] # Mass of dark matter particles
            self.m_p3 = header['MassTable'][3] # Mass of PartType3 particles
            self.n_dm = np.zeros(self.n_files, dtype=np.uint64)
            self.n_p2 = np.zeros(self.n_files, dtype=np.uint64)
            self.n_p3 = np.zeros(self.n_files, dtype=np.uint64)
            n_tot = header['NumPart_Total']
            self.n_dm_tot = n_tot[1]
            self.n_p2_tot = n_tot[2]
            self.n_p3_tot = n_tot[3]
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

        # Derived quantities
        self.BoxHalf = self.BoxSize / 2.
        self.length_to_cgs = self.a * self.UnitLength_in_cm / self.h
        self.volume_to_cgs = self.length_to_cgs**3
        self.mass_to_cgs = self.UnitMass_in_g / self.h
        self.mass_to_Msun = self.mass_to_cgs / SOLAR_MASS
        self.velocity_to_cgs = np.sqrt(self.a) * self.UnitVelocity_in_cm_per_s
        self.density_to_cgs = self.mass_to_cgs / self.volume_to_cgs

        # PartType1 data
        self.r_dm = self.dm["Coordinates"]

        # PartType2 data
        self.r_p2 = self.p2["Coordinates"]
        self.m_p2 = self.p2["Masses"]

        # PartType3 data
        self.r_p3 = self.p3["Coordinates"]

    def convert_counts(self):
        """Convert the counts to file offsets."""
        self.first_dm = np.cumsum(self.n_dm) - self.n_dm
        self.first_p2 = np.cumsum(self.n_p2) - self.n_p2
        self.first_p3 = np.cumsum(self.n_p3) - self.n_p3

    def print_offsets(self):
        """Print the file counts and offsets."""
        print(f'n_dm = {self.n_dm} (n_dm_tot = {self.n_dm_tot})')
        print(f'n_p2 = {self.n_p2} (n_p2_tot = {self.n_p2_tot})')
        print(f'n_p3 = {self.n_p3} (n_p3_tot = {self.n_p3_tot})')
        print(f'first_dm = {self.first_dm} (n_dm_tot = {self.n_dm_tot})')
        print(f'first_p2 = {self.first_p2} (n_p2_tot = {self.n_p2_tot})')
        print(f'first_p3 = {self.first_p3} (n_p3_tot = {self.n_p3_tot})')

    def print_particles(self):
        """Print the particle data."""
        print('\nParticle data:')
        print(f'r_dm = {self.r_dm}')
        print(f'r_p2 = {self.r_p2}')
        print(f'r_p3 = {self.r_p3}')

    def read_com(self):
        """Read the high-resolution center of mass and radius from the distances file."""
        with h5py.File(dist_file, 'r') as f:
            header = f['Header'].attrs
            self.PosHR = header['PosHR']
            self.RadiusHR = header['RadiusHR']

    def read_counts_single(self, i):
        """Read the counts from a single snapshot file."""
        with h5py.File(snap_pre + f'{i}.hdf5', 'r') as f:
            header = f['Header'].attrs
            nums = header['NumPart_ThisFile'] # Number of particles in this file by type
            self.n_dm[i] = nums[1] # Number of PartType1 particles in this file
            self.n_p2[i] = nums[2] # Number of PartType2 particles in this file
            self.n_p3[i] = nums[3] # Number of PartType3 particles in this file

    def read_counts(self):
        """Read the counts from the snapshot files."""
        for i in range(self.n_files):
            self.read_counts_single(i)

    def read_counts_asyncio(self):
        """Read the counts from the snapshot files using asyncio."""
        async def read_counts_async():
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                await asyncio.gather(*(loop.run_in_executor(executor, self.read_counts_single, i) for i in range(self.n_files)))
        asyncio.run(read_counts_async())

    def read_snaps_single(self, i):
        """Read the particle data from a single snapshot file."""
        with h5py.File(snap_pre + f'{i}.hdf5', 'r') as f:
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

def arepo_to_dm():
    os.makedirs(colt_dir, exist_ok=True) # Ensure the colt directory exists
    silentremove(dm_file)
    # Setup simulation parameters
    if TIMERS: t1 = time()
    sim = Simulation()
    sim.read_com() # Read extraction region information
    print(' ___       ___  __             \n'
          '  |  |__| |__  /__`  /\\  |\\ |\n'
          '  |  |  | |___ .__/ /--\\ | \\|\n' +
          f'\nInput Directory: {out_dir}' +
          f'\nOutput Directory: {colt_dir}' +
          f'\nSnap {snap}: NumP1 = {sim.n_dm_tot}, NumP2 = {sim.n_p2_tot}, NumP3 = {sim.n_p3_tot}' +
          f'\nz = {1./sim.a - 1.:g}, a = {sim.a:g}, h = {sim.h:g}, BoxSize = {1e-3*sim.BoxSize:g} cMpc/h = {1e-3*sim.BoxSize/sim.h:g} cMpc' +
          f'\nRadiusHR = {sim.RadiusHR:g} ckpc/h, PosHR = {1e-3*sim.PosHR} cMpc/h\n')
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

    # Read the particle data from the snapshot files
    if VERBOSITY > 0: print("\nReading snapshot data...")
    if SERIAL in READ_SNAPS:
        sim.read_snaps()
        if TIMERS: t2 = time(); print(f"Time to read particle data from files: {t2 - t1:g} s [serial]"); t1 = t2
    if ASYNCIO in READ_SNAPS:
        sim.read_snaps_asyncio()
        if TIMERS: t2 = time(); print(f"Time to read particle data from files: {t2 - t1:g} s [asyncio]"); t1 = t2

    # Recenter positions and define particle masks
    sim.dm['Coordinates'] -= sim.PosHR # Recenter dark matter particles
    n_dm_old = np.shape(sim.dm['Coordinates'])[0] # Number of dark matter particles (saved for comparison)
    dm_mask = (np.sum(sim.dm['Coordinates']**2, axis=1) < 1.0001*sim.RadiusHR**2) # Sphere cut
    for field in sim.dm: sim.dm[field] = sim.dm[field][dm_mask] # Apply dm mask
    n_dm = np.int32(np.shape(sim.dm['Coordinates'])[0]) # Number of dark matter particles
    sim.p2['Coordinates'] -= sim.PosHR # Recenter PartType2 particles
    n_p2_old = np.shape(sim.p2['Coordinates'])[0] # Number of PartType2 particles (saved for comparison)
    p2_mask = (np.sum(sim.p2['Coordinates']**2, axis=1) < 1.0001*sim.RadiusHR**2) # Sphere cut
    for field in sim.p2: sim.p2[field] = sim.p2[field][p2_mask] # Apply PartType2 mask
    n_p2 = np.int32(np.shape(sim.p2['Coordinates'])[0]) # Number of PartType2 particles
    sim.p3['Coordinates'] -= sim.PosHR # Recenter PartType3 particles
    n_p3_old = np.shape(sim.p3['Coordinates'])[0] # Number of PartType3 particles (saved for comparison)
    p3_mask = (np.sum(sim.p3['Coordinates']**2, axis=1) < 1.0001*sim.RadiusHR**2) # Sphere cut
    for field in sim.p3: sim.p3[field] = sim.p3[field][p3_mask] # Apply PartType3 mask
    n_p3 = np.int32(np.shape(sim.p3['Coordinates'])[0]) # Number of PartType3 particles
    if TIMERS: t2 = time(); print(f"Time to recenter and mask particles: {t2 - t1:g} s"); t1 = t2
    if VERBOSITY > 1:
        print(f'\nAfter masking: NumDM = {n_dm} = {100.*float(n_dm)/float(n_dm_old):g}%, ' +
              f'NumP2 = {n_p2} = {100.*float(n_p2)/float(n_p2_old):g}%, ' +
              f'NumP3 = {n_p3} = {100.*float(n_p3)/float(n_p3_old):g}%')
    if VERBOSITY > 3:
        sim.print_particles() # Print the particle data

    with h5py.File(dm_file, 'w') as f:
        # Simulation properties
        f.attrs['n_dm'] = n_dm # Number of PartType1 particles
        f.attrs['n_p2'] = n_p2 # Number of PartType2 particles
        f.attrs['n_p3'] = n_p3 # Number of PartType3 particles
        f.attrs['redshift'] = 1./sim.a - 1. # Current simulation redshift
        f.attrs['Omega0'] = sim.Omega0 # Matter density [rho_crit_0]
        f.attrs['OmegaB'] = sim.OmegaBaryon # Baryon density [rho_crit_0]
        f.attrs['h100'] = sim.h # Hubble constant [100 km/s/Mpc]
        f.attrs['r_box'] = sim.length_to_cgs * sim.RadiusHR # Bounding box radius [cm]

        # Dark matter fields
        f.create_dataset('r_dm', data=sim.length_to_cgs * sim.dm['Coordinates'], dtype=np.float64) # Dark matter positions [cm]
        f['r_dm'].attrs['units'] = b'cm'
        f.attrs['m_dm'] = sim.mass_to_Msun * sim.m_dm # Dark matter particle mass [Msun]

        # PartType2 fields
        if n_p2 > 0:
            f.create_dataset('r_p2', data=sim.length_to_cgs * sim.p2['Coordinates'], dtype=np.float64)
            f['r_p2'].attrs['units'] = b'cm'
            f.create_dataset('m_p2', data=sim.mass_to_Msun * sim.p2['Masses'], dtype=np.float64)
            f['m_p2'].attrs['units'] = b'Msun'

        # PartType3 fields
        if n_p3 > 0:
            f.create_dataset('r_p3', data=sim.length_to_cgs * sim.p3['Coordinates'], dtype=np.float64)
            f['r_p3'].attrs['units'] = b'cm'
            f.attrs['m_p3'] = sim.mass_to_Msun * sim.m_p3

if __name__ == '__main__':
    arepo_to_dm()

