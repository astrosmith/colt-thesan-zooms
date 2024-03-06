import numpy as np
from time import time
import h5py, os, errno, asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# Global variables
snap = 178 # Snapshot number
# snap = 188 # Snapshot number
out_dir = '.'
colt_dir = f'{out_dir}/colt'
fof_pre = f'{out_dir}/groups_{snap:03d}/fof_subhalo_tab_{snap:03d}.'
snap_pre = f'{out_dir}/snapdir_{snap:03d}/snapshot_{snap:03d}.'
dist_file = f'{out_dir}/distances_{snap:03d}.hdf5'
# dist_file = f'{out_dir}/../postprocessing/distances/distances_{snap:03d}.hdf5'

VERBOSITY = 2 # Level of print verbosity
MAX_WORKERS = 16 # Maximum number of workers
SERIAL = 1 # Run in serial
ASYNCIO = 2 # Run in parallel (asyncio)
MULTIPROCESSING = 3 # Run in parallel (multiprocessing)
NUMPY = 1 # Use NumPy
NUMBA = 2 # Use Numba
READ_DEFAULT = (ASYNCIO,) # Default read method
# READ_DEFAULT = (SERIAL,ASYNCIO,MULTIPROCESSING) # Default read method
# CALC_DEFAULT = (NUMPY,) # Calculate center of mass
# CALC_DEFAULT = (NUMPY, NUMBA) # Calculate center of mass
READ_COUNTS = READ_DEFAULT # Read counts methods
READ_SNAPS = READ_DEFAULT # Read snapshots methods
# CALC_COM = CALC_DEFAULT # Calculate center of mass methods
TIMERS = True # Print timers

# Extracted fields
gas_fields = ['Coordinates', 'Masses', 'HighResGasMass', 'Velocities', 'InternalEnergy', 'DustTemperature',
              'GFM_Metals', 'GFM_Metallicity', 'GFM_DustMetallicity', 'Density', 'H2_Fraction', 'HI_Fraction',
              'HeI_Fraction', 'HeII_Fraction', 'ElectronAbundance', 'StarFormationRate', 'ParticleIDs']
star_fields = ['Coordinates', 'Masses', 'IsHighRes', 'Velocities', 'GFM_StellarFormationTime', 'GFM_Metallicity', 'GFM_InitialMass', 'ParticleIDs']

# Universal constants
c = 2.99792458e10          # Speed of light [cm/s]
kB = 1.380648813e-16       # Boltzmann's constant [g cm^2/s^2/K]
h = 6.626069573e-27        # Planck's constant [erg/s]
mH = 1.6735327e-24         # Mass of hydrogen atom [g]
me = 9.109382917e-28       # Electron mass [g]
ee = 4.80320451e-10        # Electron charge [g^(1/2) cm^(3/2) / s]

# Emperical unit definitions
Msun = 1.988435e33         # Solar mass [g]
Lsun = 3.839e33            # Solar luminosity [erg/s]
Zsun = 0.0134              # Solar metallicity (mass fraction)
arcsec = 648000. / np.pi   # arseconds per radian
pc = 3.085677581467192e18  # Units: 1 pc  = 3e18 cm
kpc = 1e3 * pc             # Units: 1 kpc = 3e21 cm
Mpc = 1e6 * pc             # Units: 1 Mpc = 3e24 cm
km = 1e5                   # Units: 1 km  = 1e5  cm
angstrom = 1e-8            # Units: 1 angstrom = 1e-8 cm
day = 86400.               # Units: 1 day = 24 * 3600 seconds
yr = 365.24 * day          # Units: 1 year = 365.24 days
kyr = 1e3 * yr             # Units: 1 Myr = 10^6 yr
Myr = 1e6 * yr             # Units: 1 Myr = 10^6 yr

GAS_HIGH_RES_THRESHOLD = 0.5  # Threshold deliniating high and low resolution gas particles
SOLAR_MASS = 1.989e33         # Solar masses
BOLTZMANN = 1.38065e-16       # Boltzmann's constant [g cm^2/sec^2/k]
PLANCK = 6.6260695e-27        # Planck's constant [erg sec]
PROTONMASS = 1.67262178e-24   # Mass of hydrogen atom [g]
HYDROGEN_MASSFRAC = 0.76      # Mass fraction of hydrogen
GAMMA = 5. / 3.               # Adiabatic index of simulated gas
GAMMA_MINUS1 = GAMMA - 1.     # For convenience
HUBBLE = 3.2407789e-18        # Hubble constant [h/sec]
SEC_PER_GIGAYEAR = 3.15576e16 # Seconds per gigayear
HE_ABUND = (1./HYDROGEN_MASSFRAC - 1.) / 4. # Helium abundance = n_He / n_H

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

# Calculates time difference in Gyr between two scale factor values.
# For cosmological simulations a0 and a1 are scalefactors.
def get_time_difference_in_Gyr(a0, a1, Omega0, h):
    OmegaLambda = 1. - Omega0  # Assume a flat cosmology
    factor1 = 2. / (3. * np.sqrt(OmegaLambda))

    term1   = np.sqrt(OmegaLambda / Omega0) * a0**1.5
    term2   = np.sqrt(1. + OmegaLambda / Omega0 * a0**3)
    factor2 = np.log(term1 + term2)
    t0 = factor1 * factor2

    term1   = np.sqrt(OmegaLambda / Omega0) * a1**1.5
    term2   = np.sqrt(1. + OmegaLambda / Omega0 * a1**3)
    factor2 = np.log(term1 + term2)
    t1 = factor1 * factor2

    return (t1 - t0) / (HUBBLE * h * SEC_PER_GIGAYEAR) # now in gigayears

@dataclass
class Simulation:
    """Simulation information and shared data."""
    n_files: np.int32 = 0 # Number of files
    n_gas_tot: np.uint64 = None # Total number of gas particles
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
    mass_to_Msun: np.float64 = None # Conversion factor for mass to solar masses
    velocity_to_cgs: np.float64 = None # Conversion factor for velocity to cgs
    density_to_cgs: np.float64 = None # Conversion factor for density to cgs

    # Selection criteria
    PosHR: np.ndarray = None # Center of mass of high-resolution particles
    RadiusHR: np.float64 = None # Radius of high-resolution region

    # Shared file counts and offsets
    n_gas: np.ndarray = None
    n_stars: np.ndarray = None
    first_gas: np.ndarray = None
    first_star: np.ndarray = None

    # Particle data
    gas: dict = field(default_factory=dict)
    stars: dict = field(default_factory=dict)

    def __post_init__(self):
        """Allocate memory for gas and star arrays."""
        if self.n_files == 0:
            # Read header info from the snapshot files
            with h5py.File(snap_pre + '0.hdf5', 'r') as f:
                header = f['Header'].attrs
                self.n_files = header['NumFilesPerSnapshot']
                self.n_gas = np.zeros(self.n_files, dtype=np.uint64)
                self.n_stars = np.zeros(self.n_files, dtype=np.uint64)
                n_tot = header['NumPart_Total']
                self.n_gas_tot = n_tot[0]
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
                g = f['PartType0']
                for field in gas_fields:
                    shape, dtype = g[field].shape, g[field].dtype
                    shape = (self.n_gas_tot,) + shape[1:]
                    self.gas[field] = np.empty(shape, dtype=dtype)
                # Star data
                g = f['PartType4']
                for field in star_fields:
                    shape, dtype = g[field].shape, g[field].dtype
                    shape = (self.n_stars_tot,) + shape[1:]
                    self.stars[field] = np.empty(shape, dtype=dtype)

            # Derived quantities
            self.BoxHalf = self.BoxSize / 2.
            self.length_to_cgs = self.a * self.UnitLength_in_cm / self.h
            self.volume_to_cgs = self.length_to_cgs**3
            self.mass_to_cgs = self.UnitMass_in_g / self.h
            self.mass_to_Msun = self.mass_to_cgs / SOLAR_MASS
            self.velocity_to_cgs = np.sqrt(self.a) * self.UnitVelocity_in_cm_per_s
            self.density_to_cgs = self.mass_to_cgs / self.volume_to_cgs

    def convert_counts(self):
        """Convert the counts to file offsets."""
        self.first_gas = np.cumsum(self.n_gas) - self.n_gas
        self.first_star = np.cumsum(self.n_stars) - self.n_stars

    def print_offsets(self):
        """Print the file counts and offsets."""
        print(f'n_gas = {self.n_gas} (n_gas_tot = {self.n_gas_tot})')
        print(f'n_stars = {self.n_stars} (n_stars_tot = {self.n_stars_tot})\n')
        print(f'first_gas = {self.first_gas} (n_gas_tot = {self.n_gas_tot})')
        print(f'first_star = {self.first_star} (n_stars_tot = {self.n_stars_tot})')

    def print_particles(self):
        """Print the particle data."""
        print('\nGas particle data:')
        for field in self.gas:
            print(f'{field} = {self.gas[field]}')
        print('\nStar particle data:')
        for field in self.stars:
            print(f'{field} = {self.stars[field]}')

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
            self.n_gas[i] = nums[0] # Number of gas particles in this file
            self.n_stars[i] = nums[4] # Number of star particles in this file

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
            if self.n_gas[i] > 0:
                g = f['PartType0']
                offset = self.first_gas[i]
                next_offset = offset + self.n_gas[i]
                for field in gas_fields:
                    self.gas[field][offset:next_offset] = g[field][:]
            if self.n_stars[i] > 0:
                g = f['PartType4']
                offset = self.first_star[i]
                next_offset = offset + self.n_stars[i]
                for field in star_fields:
                    self.stars[field][offset:next_offset] = g[field][:]

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

# def arepo_to_colt(snap=10, out_dir='.', include_metals=True):
def arepo_to_colt(include_metals=True):
    # Setup simulation parameters
    if TIMERS: t1 = time()
    sim = Simulation()
    sim.read_com() # Read extraction region information
    print(' ___       ___  __             \n'
          '  |  |__| |__  /__`  /\\  |\\ |\n'
          '  |  |  | |___ .__/ /--\\ | \\|\n' +
          f'\nInput Directory: {out_dir}' +
          f'\nSnap {snap}: NumGas = {sim.n_gas_tot}, NumStar = {sim.n_stars_tot}' +
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
    print("\nReading snapshot data into shared memory...")
    if SERIAL in READ_SNAPS:
        sim.read_snaps()
        if TIMERS: t2 = time(); print(f"Time to read particle data from files: {t2 - t1:g} s [serial]"); t1 = t2
    if ASYNCIO in READ_SNAPS:
        sim.read_snaps_asyncio()
        if TIMERS: t2 = time(); print(f"Time to read particle data from files: {t2 - t1:g} s [asyncio]"); t1 = t2
    if VERBOSITY > 1:
        n_gas_hr = np.sum(sim.gas['HighResGasMass'] > GAS_HIGH_RES_THRESHOLD * sim.gas['Masses']) # Number of high-resolution gas particles
        n_stars_hr = np.sum(sim.stars['IsHighRes'] == 1) # Number of high-resolution star particles
        print(f'NumGasHR = {n_gas_hr} = {100.*float(n_gas_hr)/float(sim.n_gas_tot):g}%, NumStarHR = {n_stars_hr} = {100.*float(n_stars_hr)/float(sim.n_stars_tot):g}%')

    # Recenter positions and define particle masks
    sim.gas['Coordinates'] -= sim.PosHR # Recenter gas particles
    sim.stars['Coordinates'] -= sim.PosHR # Recenter star particles
    n_cells_old = np.shape(sim.gas['Coordinates'])[0] # Number of cells (saved for comparison)
    n_stars_old = np.shape(sim.stars['Coordinates'])[0] # Number of star particles (saved for comparison)
    gas_mask = (np.sum(sim.gas['Coordinates']**2, axis=1) < 1.0001*sim.RadiusHR**2) # Sphere cut
    stars_mask = (np.sum(sim.stars['Coordinates']**2, axis=1) < 1.0001*sim.RadiusHR**2) \
               & (sim.stars['IsHighRes'] == 1) # & (sim.stars['GFM_StellarFormationTime'] > 0.) # Sphere cut
    for field in sim.gas: sim.gas[field] = sim.gas[field][gas_mask] # Apply gas mask
    for field in sim.stars: sim.stars[field] = sim.stars[field][stars_mask] # Apply star mask
    n_cells = np.int32(np.shape(sim.gas['Coordinates'])[0]) # Number of cells
    n_stars = np.int32(np.shape(sim.stars['Coordinates'])[0]) # Number of star particles
    if TIMERS: t2 = time(); print(f"Time to recenter and mask particles: {t2 - t1:g} s"); t1 = t2
    if VERBOSITY > 1:
        print(f'\nAfter masking: NumGas = {n_cells} = {100.*float(n_cells)/float(n_cells_old):g}%, ' +
              f'NumStar = {n_stars} = {100.*float(n_stars)/float(n_stars_old):g}%')
    if VERBOSITY > 2:
        n_gas_hr = np.sum(sim.gas['HighResGasMass'] > GAS_HIGH_RES_THRESHOLD * sim.gas['Masses']) # Number of high-resolution gas particles
        n_stars_hr = np.sum(sim.stars['IsHighRes'] == 1) # Number of high-resolution star particles
        print(f'NumGasHR = {n_gas_hr} = {100.*float(n_gas_hr)/float(sim.n_gas_tot):g}%, ' +
              f'NumStarHR = {n_stars_hr} = {100.*float(n_stars_hr)/float(sim.n_stars_tot):g}%')
        print(f'Gas:   r_min = {np.min(sim.gas["Coordinates"], axis=0)} ckpc/h, r_max = {np.max(sim.gas["Coordinates"], axis=0)} ckpc/h\n' +
              f'Stars: r_min = {np.min(sim.stars["Coordinates"], axis=0)} ckpc/h, r_max = {np.max(sim.stars["Coordinates"], axis=0)} ckpc/h')
        print(f'Gas:   radius_min = {np.sqrt(np.min(np.sum(sim.gas["Coordinates"]**2, axis=1)))} ckpc/h, radius_max = {np.sqrt(np.max(np.sum(sim.gas["Coordinates"]**2, axis=1)))} ckpc/h\n' +
              f'Stars: radius_min = {np.sqrt(np.min(np.sum(sim.stars["Coordinates"]**2, axis=1)))} ckpc/h, radius_max = {np.sqrt(np.max(np.sum(sim.stars["Coordinates"]**2, axis=1)))} ckpc/h')
        if VERBOSITY > 3: sim.print_particles() # Print the particle data
        # Print additional extraction properties
        m = sim.mass_to_cgs * sim.gas['HighResGasMass'] # Gas masses [g]
        m_tot = np.sum(m) # Total gas mass [g]
        r = sim.length_to_cgs * sim.gas['Coordinates'] # Gas positions [cm]
        v = sim.velocity_to_cgs * sim.gas['Velocities'] # Gas velocities [cm/s]
        m_star = sim.mass_to_cgs * sim.stars['Masses'] # Star masses [g]
        m_star_tot = np.sum(m_star) # Total star mass [g]
        r_star = sim.length_to_cgs * sim.stars['Coordinates'] # Star positions [cm]
        v_star = sim.velocity_to_cgs * sim.stars['Velocities'] # Star velocities [cm/s]
        r_com = np.sum(m[:,np.newaxis]*r, axis=0) / m_tot # Gas center of mass position [cm]
        v_com = np.sum(m[:,np.newaxis]*v, axis=0) / m_tot # Gas center of mass velocity [cm/s]
        r_star_com = np.sum(m_star[:,np.newaxis]*r_star, axis=0) / m_star_tot # Star center of mass position [cm]
        v_star_com = np.sum(m_star[:,np.newaxis]*v_star, axis=0) / m_star_tot # Star center of mass velocity [cm/s]
        print(f'\nExtracted [gas, star] mass = [{m_tot/Msun/1e10:g}, {m_star_tot/Msun/1e10:g}] x 10^10 Msun')
        print(f'Extracted [gas, star] center of mass position = [{r_com/kpc}, {r_star_com/kpc}] kpc  =>  |r_com| = [{np.sqrt(np.sum(r_com**2))/kpc}, {np.sqrt(np.sum(r_star_com**2))/kpc}] kpc')
        print(f'Extracted [gas, star] center of mass velocity = [{v_com/km}, {v_star_com/km}] km/s  =>  |v_com| = [{np.sqrt(np.sum(v_com**2))/km}, {np.sqrt(np.sum(v_star_com**2))/km}] km/s')

    os.makedirs(f'{colt_dir}/ics', exist_ok=True) # Ensure the colt directory exists
    colt_filename = f'{colt_dir}/ics/colt_{snap:03d}.hdf5'
    silentremove(colt_filename)
    with h5py.File(colt_filename, 'w') as f:
        # Simulation properties
        f.attrs['n_cells'] = n_cells # Number of cells
        f.attrs['n_stars'] = n_stars # Number of star particles
        f.attrs['redshift'] = 1./sim.a - 1. # Current simulation redshift
        f.attrs['Omega0'] = sim.Omega0 # Matter density [rho_crit_0]
        f.attrs['OmegaB'] = sim.OmegaBaryon # Baryon density [rho_crit_0]
        f.attrs['h100'] = sim.h # Hubble constant [100 km/s/Mpc]
        f.attrs['r_box'] = sim.length_to_cgs * sim.RadiusHR # Bounding box radius [cm]

        # Gas fields
        f.create_dataset('r', data=sim.length_to_cgs * sim.gas['Coordinates'], dtype=np.float64) # Mesh generating points [cm]
        f['r'].attrs['units'] = b'cm'
        f.create_dataset('v', data=sim.velocity_to_cgs * sim.gas['Velocities'], dtype=np.float64) # Cell velocities [cm/s]
        f['v'].attrs['units'] = b'cm/s'
        f.create_dataset('e_int', data=sim.UnitVelocity_in_cm_per_s**2 * sim.gas['InternalEnergy'], dtype=np.float64) # Specific internal energy [cm^2/s^2]
        f['e_int'].attrs['units'] = b'cm^2/s^2'
        # mu = 4. / (1. + 3.*HYDROGEN_MASSFRAC + 4.*HYDROGEN_MASSFRAC * x_e) # Mean molecular weight [mH]
        # T = GAMMA_MINUS1 * e_int * mu * PROTONMASS / BOLTZMANN # Temperature [K]
        # f.create_dataset('T', data=T) # Temperature [K]
        # f['T'].attrs['units'] = b'K'
        f.create_dataset('T_dust', data=sim.gas['DustTemperature'], dtype=np.float64) # Dust temperature [K]
        f['T_dust'].attrs['units'] = b'K'
        if include_metals:
            Hydrogen, Helium, Carbon, Nitrogen, Oxygen, Neon, Magnesium, Silicon, Iron = range(9)
            metals = sim.gas['GFM_Metals'] # Metals = [Hydrogen, Helium, Carbon, Nitrogen, Oxygen, Neon, Magnesium, Silicon, Iron]
            f.create_dataset('X', data=metals[:,Hydrogen], dtype=np.float64) # Hydrogen metallicity [mass fraction]
            f.create_dataset('Y', data=metals[:,Helium], dtype=np.float64) # Helium metallicity [mass fraction]
            f.create_dataset('Z_C', data=metals[:,Carbon], dtype=np.float64) # Carbon metallicity [mass fraction]
            f.create_dataset('Z_N', data=metals[:,Nitrogen], dtype=np.float64) # Nitrogen metallicity [mass fraction]
            f.create_dataset('Z_O', data=metals[:,Oxygen], dtype=np.float64) # Oxygen metallicity [mass fraction]
            f.create_dataset('Z_Ne', data=metals[:,Neon], dtype=np.float64) # Neon metallicity [mass fraction]
            f.create_dataset('Z_Mg', data=metals[:,Magnesium], dtype=np.float64) # Magnesium metallicity [mass fraction]
            f.create_dataset('Z_Si', data=metals[:,Silicon], dtype=np.float64) # Silicon metallicity [mass fraction]
            Zsun_Si = 0.000665509  # Solar silicon metallicity [mass fraction]
            Zsun_S = 0.00030953    # Solar sulfer metallicity [mass fraction]
            f.create_dataset('Z_S', data=metals[:,Silicon]*Zsun_S/Zsun_Si, dtype=np.float64) # Sulfer metallicity [mass fraction]
            f.create_dataset('Z_Fe', data=metals[:,Iron], dtype=np.float64) # Iron metallicity [mass fraction]
        Z = sim.gas['GFM_Metallicity']; Z[Z<0.] = 0. # Ensure positive metallicity
        f.create_dataset('Z', data=Z, dtype=np.float64) # Metallicity [mass fraction]
        f.create_dataset('D', data=sim.gas['GFM_DustMetallicity'], dtype=np.float64) # Dust-to-gas ratio [mass fraction]
        f.create_dataset('rho', data=sim.density_to_cgs * sim.gas['Density'], dtype=np.float64) # Density [g/cm^3]
        f['rho'].attrs['units'] = b'g/cm^3'
        # f.create_dataset('x_HI', data=sim.gas['HI_Fraction'], dtype=np.float64) # Neutral hydrogen fraction
        f.create_dataset('x_H2', data=sim.gas['H2_Fraction'], dtype=np.float64) # Molecular hydrogen fraction
        f.create_dataset('x_HI', data=sim.gas['HI_Fraction'] + 2.*sim.gas['H2_Fraction'], dtype=np.float64) # Neutral fraction
        # x_HII = 1. - x_HI # Infer abundance
        # f.create_dataset('x_HII', data=sim.gas['HII_Fraction'], dtype=np.float64) # Ionized hydrogen fraction
        f.create_dataset('x_HeI', data=sim.gas['HeI_Fraction'] / HE_ABUND, dtype=np.float64) # HeI fraction
        f.create_dataset('x_HeII', data=sim.gas['HeII_Fraction'] / HE_ABUND, dtype=np.float64) # HeII fraction
        # f.create_dataset('x_HeIII', data=sim.gas['HeIII_Fraction'] / HE_ABUND, dtype=np.float64) # HeIII fraction
        # x_e = x_HII + x_HeII + 2. * x_HeIII # Infer abundance
        f.create_dataset('x_e', data=sim.gas['ElectronAbundance'], dtype=np.float64) # Electron fraction
        f.create_dataset('SFR', data=sim.gas['StarFormationRate'], dtype=np.float64) # Star formation rate [Msun/yr]
        f['SFR'].attrs['units'] = b'Msun/yr'
        # f.create_dataset('N_phot', data=N_phot_to_cgs * sim.gas['PhotonDensity'], dtype=np.float64) # Photon number density [cm^-3]
        # f['N_phot'].attrs['units'] = b'cm^-3'
        # f.create_dataset('B', data=magnetic_to_cgs*sim.gas['MagneticField'], dtype=np.float64) # Magnetic field [Gauss]
        # f['B'].attrs['units'] = b'G'
        f.create_dataset('id', data=sim.gas['ParticleIDs']) # Particle IDs

        # Star fields
        f.create_dataset('r_star', data=sim.length_to_cgs * sim.stars['Coordinates'], dtype=np.float64) # Star positions [cm]
        f['r_star'].attrs['units'] = b'cm'
        f.create_dataset('v_star', data=sim.velocity_to_cgs * sim.stars['Velocities'], dtype=np.float64) # Star velocities [cm/s]
        f['v_star'].attrs['units'] = b'cm/s'
        f.create_dataset('Z_star', data=sim.stars['GFM_Metallicity'], dtype=np.float64) # Stellar metallicity [mass fraction]
        # f.create_dataset('m_star', data=mass_to_Msun * sim.stars['Masses'], dtype=np.float64) # Star mass [Msun]
        # f['m_star'].attrs['units'] = b'Msun'
        f.create_dataset('m_init_star', data=sim.mass_to_Msun * sim.stars['GFM_InitialMass'], dtype=np.float64) # Star initial mass [Msun]
        f['m_init_star'].attrs['units'] = b'Msun'
        age_star = get_time_difference_in_Gyr(sim.stars['GFM_StellarFormationTime'].astype(np.float64), sim.a, sim.Omega0, sim.h) # Age of the star [Gyr]
        f.create_dataset('age_star', data=age_star) # Star age [Gyr]
        f['age_star'].attrs['units'] = b'Gyr'
        f.create_dataset('id_star', data=sim.stars['ParticleIDs']) # Particle IDs

if __name__ == '__main__':
    # import sys
    # if len(sys.argv) == 3:
    #     snap, out_dir = int(sys.argv[1]), sys.argv[2]
    # else:
    #     raise ValueError('Usage: python arepo_to_colt.py snap out_dir')
    # arepo_to_colt(snap=snap, out_dir=out_dir)
    arepo_to_colt()

