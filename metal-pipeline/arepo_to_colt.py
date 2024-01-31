import numpy as np
import h5py, os, errno

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

BOLTZMANN = 1.38065e-16     # Boltzmann's constant [g cm^2/sec^2/k]
PLANCK = 6.6260695e-27      # Planck's constant [erg sec]
PROTONMASS = 1.67262178e-24 # Mass of hydrogen atom [g]
HYDROGEN_MASSFRAC = 0.76    # Mass fraction of hydrogen
GAMMA = 5. / 3.             # Adiabatic index of simulated gas
GAMMA_MINUS1 = GAMMA - 1.   # For convenience
HUBBLE = 3.2407789e-18            # Hubble constant [h/sec]
SEC_PER_GIGAYEAR = 3.15576e16     # Seconds per gigayear
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

def arepo_to_colt(snap=10, group=0, out_dir='.', include_metals=True, verbose=True):
    os.makedirs(f'{out_dir}/colt/ics', exist_ok=True) # Ensure the output directory exists
    arepo_filename = f'{out_dir}/extractions/snap_{snap:03d}_g{group}.hdf5'
    colt_filename = f'{out_dir}/colt/ics/colt_{snap:03d}.hdf5'
    silentremove(colt_filename)
    if not os.path.isfile(arepo_filename): return
    with h5py.File(arepo_filename, 'r') as f, h5py.File(colt_filename, 'w') as ef:
        # Cosmology
        g = f['Header']
        a = g.attrs['Time']
        z = g.attrs['Redshift']
        BoxSize = g.attrs['BoxSize']
        BoxHalf = BoxSize / 2.

        # Unit conversions
        g = f['Parameters']
        h = g.attrs['HubbleParam']
        Omega0 = g.attrs['Omega0']
        OmegaB = g.attrs['OmegaBaryon']
        UnitLength_in_cm = g.attrs['UnitLength_in_cm']
        UnitMass_in_g = g.attrs['UnitMass_in_g']
        UnitVelocity_in_cm_per_s = g.attrs['UnitVelocity_in_cm_per_s']
        UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
        UnitEnergy_in_cgs = UnitMass_in_g * UnitVelocity_in_cm_per_s * UnitVelocity_in_cm_per_s
        UnitLum_in_cgs = UnitEnergy_in_cgs / UnitTime_in_s
        length_to_cgs = a * UnitLength_in_cm / h
        mass_to_cgs = UnitMass_in_g / h
        mass_to_Msun = mass_to_cgs / Msun
        velocity_to_cgs = np.sqrt(a) * UnitVelocity_in_cm_per_s
        magnetic_to_cgs = h/a**2 * np.sqrt(UnitMass_in_g/UnitLength_in_cm) / UnitTime_in_s
        volume_to_cgs = length_to_cgs * length_to_cgs * length_to_cgs
        density_to_cgs = mass_to_cgs / volume_to_cgs
        CutHalf = BoxHalf
        cut_half = length_to_cgs * CutHalf

        # Filter gas particle data
        gas = f['PartType0']
        r = gas['Coordinates'][:].astype(np.float64)
        for i in range(3):
            r[:,i] -= BoxHalf
        gas_mask = (np.sum(r**2, axis=1) < CutHalf**2) # Sphere cut
        r = length_to_cgs * r[gas_mask,:] # Gas positions [cm]
        v = velocity_to_cgs * gas['Velocities'][:][gas_mask,:].astype(np.float64) # Gas velocities [cm/s]
        n_cells = np.int32(np.shape(r)[0])

        # Filter star particle data
        stars_flag = ('PartType4' in f.keys())
        if stars_flag:
            stars = f['PartType4']
            r_star = stars['Coordinates'][:].astype(np.float64)
            for i in range(3):
                r_star[:,i] -= BoxHalf
            a_form = stars['GFM_StellarFormationTime'][:] # Scale factor of formation
            star_mask = (np.sum(r_star**2, axis=1) < (0.95*CutHalf)**2) & (a_form > 0.) # Sphere cut
            r_star = length_to_cgs * r_star[star_mask,:] # Star positions [cm]
            v_star = velocity_to_cgs * stars['Velocities'][:][star_mask,:].astype(np.float64) # Star velocities [cm/s]
            a_form = a_form[star_mask].astype(np.float64)
            n_stars = np.int32(np.shape(r_star)[0])
        else:
            n_stars = 0

        # Print additional extraction properties
        if verbose and stars_flag:
            m = mass_to_cgs * gas['Masses'][:][gas_mask].astype(np.float64) # Gas masses [g]
            m_tot = np.sum(m) # Total gas mass [g]
            r_com = np.array([np.sum(m*r[:,0]), np.sum(m*r[:,1]), np.sum(m*r[:,2])]) / m_tot # Gas center of mass position [cm]
            v_com = np.array([np.sum(m*v[:,0]), np.sum(m*v[:,1]), np.sum(m*v[:,2])]) / m_tot # Gas center of mass velocity [cm/s]
            m_star = mass_to_cgs * stars['Masses'][:][star_mask].astype(np.float64) # Star masses [g]
            m_star_tot = np.sum(m_star) # Total star mass [g]
            r_star_com = np.array([np.sum(m_star*r_star[:,0]), np.sum(m_star*r_star[:,1]), np.sum(m_star*r_star[:,2])]) / m_star_tot # Star center of mass position [cm]
            v_star_com = np.array([np.sum(m_star*v_star[:,0]), np.sum(m_star*v_star[:,1]), np.sum(m_star*v_star[:,2])]) / m_star_tot # Star center of mass velocity [cm/s]
            print(f'\nExtracted [gas, star] mass = [{m_tot/Msun/1e10:g}, {m_star_tot/Msun/1e10:g}] x 10^10 Msun')
            print(f'Extracted [gas, star] center of mass position = [{r_com/kpc}, {r_star_com/kpc}] kpc  =>  |r_com| = [{np.sqrt(np.sum(r_com**2))/kpc}, {np.sqrt(np.sum(r_star_com**2))/kpc}] kpc')
            print(f'Extracted [gas, star] center of mass velocity = [{v_com/km}, {v_star_com/km}] km/s  =>  |v_com| = [{np.sqrt(np.sum(v_com**2))/km}, {np.sqrt(np.sum(v_star_com**2))/km}] km/s')

        # Simulation properties
        ef.attrs['n_cells'] = n_cells
        ef.attrs['n_stars'] = n_stars
        ef.attrs['redshift'] = z      # Current simulation redshift
        ef.attrs['Omega0'] = Omega0   # Matter density [rho_crit_0]
        ef.attrs['OmegaB'] = OmegaB   # Baryon density [rho_crit_0]
        ef.attrs['h100'] = h          # Hubble constant [100 km/s/Mpc]
        ef.attrs['r_box'] = cut_half  # Bounding box radius [cm]

        # Gas fields
        # mu = 4. / (1. + 3.*HYDROGEN_MASSFRAC + 4.*HYDROGEN_MASSFRAC * x_e) # Mean molecular weight [mH]
        # T = GAMMA_MINUS1 * e_int * mu * PROTONMASS / BOLTZMANN # Temperature [K]
        ef.create_dataset('e_int', data=gas['InternalEnergy'][:][gas_mask].astype(np.float64) * UnitVelocity_in_cm_per_s**2) # Specific internal energy [cm^2/s^2]
        ef['e_int'].attrs['units'] = b'cm^2/s^2'
        #ef.create_dataset('T', data=T) # Temperature [K]
        #ef['T'].attrs['units'] = b'K'
        ef.create_dataset('T_dust', data=gas['DustTemperature'][:].astype(np.float64)) # Dust temperature [K]
        ef['T_dust'].attrs['units'] = b'K'
        if include_metals:
            Hydrogen, Helium, Carbon, Nitrogen, Oxygen, Neon, Magnesium, Silicon, Iron = range(9)
            metals = gas['GFM_Metals'][:][gas_mask,:].astype(np.float64) # Metals = [Hydrogen, Helium, Carbon, Nitrogen, Oxygen, Neon, Magnesium, Silicon, Iron]
            ef.create_dataset('X', data=metals[:,Hydrogen]) # Hydrogen metallicity [mass fraction]
            ef.create_dataset('Y', data=metals[:,Helium]) # Helium metallicity [mass fraction]
            ef.create_dataset('Z_C', data=metals[:,Carbon]) # Carbon metallicity [mass fraction]
            ef.create_dataset('Z_N', data=metals[:,Nitrogen]) # Nitrogen metallicity [mass fraction]
            ef.create_dataset('Z_O', data=metals[:,Oxygen]) # Oxygen metallicity [mass fraction]
            ef.create_dataset('Z_Ne', data=metals[:,Neon]) # Neon metallicity [mass fraction]
            ef.create_dataset('Z_Mg', data=metals[:,Magnesium]) # Magnesium metallicity [mass fraction]
            ef.create_dataset('Z_Si', data=metals[:,Silicon]) # Silicon metallicity [mass fraction]
            Zsun_Si = 0.000665509      # Solar silicon metallicity [mass fraction]
            Zsun_S = 0.00030953        # Solar sulfer metallicity [mass fraction]
            ef.create_dataset('Z_S', data=metals[:,Silicon]*Zsun_S/Zsun_Si) # Sulfer metallicity [mass fraction]
            ef.create_dataset('Z_Fe', data=metals[:,Iron]) # Iron metallicity [mass fraction]
        Z = gas['GFM_Metallicity'][:][gas_mask].astype(np.float64)
        Z[Z<0.] = 0.
        ef.create_dataset('Z', data=Z) # Metallicity [mass fraction]
        ef.create_dataset('D', data=gas['GFM_DustMetallicity'][:][gas_mask].astype(np.float64)) # Dust-to-gas ratio [mass fraction]
        ef.create_dataset('rho', data=density_to_cgs * gas['Density'][:][gas_mask].astype(np.float64)) # Density [g/cm^3]
        ef['rho'].attrs['units'] = b'g/cm^3'
        #ef.create_dataset('x_HI', data=gas['HI_Fraction'][:][gas_mask].astype(np.float64)) # Neutral hydrogen fraction
        ef.create_dataset('x_H2', data=gas['H2_Fraction'][:][gas_mask].astype(np.float64)) # Molecular hydrogen fraction
        ef.create_dataset('x_HI', data=(gas['HI_Fraction'][:][gas_mask] + 2.*gas['H2_Fraction'][:][gas_mask]).astype(np.float64)) # Neutral fraction
        # x_HII = 1. - x_HI # Infer abundance
        # ef.create_dataset('x_HII', data=gas['HII_Fraction'][:][gas_mask].astype(np.float64)) # Ionized hydrogen fraction
        ef.create_dataset('x_HeI', data=gas['HeI_Fraction'][:][gas_mask].astype(np.float64) / HE_ABUND) # HeI fraction
        ef.create_dataset('x_HeII', data=gas['HeII_Fraction'][:][gas_mask].astype(np.float64) / HE_ABUND) # HeII fraction
        #ef.create_dataset('x_HeIII', data=gas['HeIII_Fraction'][:][gas_mask].astype(np.float64) / HE_ABUND) # HeIII fraction
        # x_e = x_HII + x_HeII + 2. * x_HeIII # Infer abundance
        ef.create_dataset('x_e', data=gas['ElectronAbundance'][:][gas_mask].astype(np.float64)) # Electron fraction
        ef.create_dataset('r', data=r) # Mesh generating points [cm]
        ef['r'].attrs['units'] = b'cm'
        ef.create_dataset('v', data=v) # Cell velocities [cm/s]
        ef['v'].attrs['units'] = b'cm/s'
        ef.create_dataset('SFR', data=gas['StarFormationRate'][:][gas_mask].astype(np.float64)) # Star formation rate [Msun/yr]
        ef['SFR'].attrs['units'] = b'Msun/yr'
        #ef.create_dataset('N_phot', data=N_phot_to_cgs * gas['PhotonDensity'][:][gas_mask].astype(np.float64)) # Photon number density (cm^-3)
        #ef['N_phot'].attrs['units'] = b'cm^-3'
        #ef.create_dataset('B', data=magnetic_to_cgs*gas['MagneticField'][:][gas_mask,:].astype(np.float64)) # Magnetic field (Gauss)
        #ef['B'].attrs['units'] = b'G'
        ef.create_dataset('id', data=gas['ParticleIDs'][:][gas_mask]) # Particle IDs

        # Star fields
        if stars_flag:
            ef.create_dataset('r_star', data=r_star) # Star positions [cm]
            ef['r_star'].attrs['units'] = b'cm'
            ef.create_dataset('v_star', data=v_star) # Star velocities [cm/s]
            ef['v_star'].attrs['units'] = b'cm/s'
            ef.create_dataset('Z_star', data=stars['GFM_Metallicity'][:][star_mask].astype(np.float64)) # Stellar metallicity [mass fraction]
            # ef.create_dataset('m_star', data=mass_to_msun * stars['Masses'][:].astype(np.float64)) # Star mass [Msun]
            # ef['m_star'].attrs['units'] = b'Msun'
            ef.create_dataset('m_init_star', data=mass_to_Msun * stars['GFM_InitialMass'][:][star_mask].astype(np.float64)) # Star initial mass [Msun]
            ef['m_init_star'].attrs['units'] = b'Msun'
            age_star = get_time_difference_in_Gyr(a_form, a, Omega0, h) # Age of the star [Gyr]
            ef.create_dataset('age_star', data=age_star) # Star age [Gyr]
            ef['age_star'].attrs['units'] = b'Gyr'
            ef.create_dataset('id_star', data=stars['ParticleIDs'][:][star_mask]) # Particle IDs

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3:
        snap, out_dir = int(sys.argv[1]), sys.argv[2]
    else:
        raise ValueError('Usage: python arepo_to_colt.py snap out_dir')
    arepo_to_colt(snap=snap, out_dir=out_dir)

