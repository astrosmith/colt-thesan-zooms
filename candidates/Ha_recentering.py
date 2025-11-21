import numpy as np
import h5py
import sys

X = 0.76            # Primordial mass fraction of hydrogen

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

# Note: Hydrogen transition energies = E_H * (1/n_1^2 - 1/n_2^2)
e2_h = np.pi * ee * ee / h    # pi e^2 / h
E_H = 2. * me * e2_h * e2_h   # me e^4 / (2 hbar^2)

# Hydrogen Balmer-alpha (2,3) [6561Å]
E0_Ha = E_H * 5. / 36.        # Line energy [erg]

# Recombination coefficient (Case B) - Units: cm^3/s
def alpha_B(T):
    # α_B(T) is from Hui & Gnedin (1997) good to 0.7% accuracy from 1 K < T < 10^9 K
    λ_B = 315614. / T         # 2 T_i/T where T_i = 157807 K
    return 2.753e-14 * λ_B**1.5 / (1. + (λ_B/2.74)**0.407)**2.242 # cm^3/s

# Temperature and density grid for Storey & Hummer (1995) recombination tables
# Note: T = {5e2, 1e3, 3e3, 5e3, 7.5e3, 1e4, 1.25e4, 1.5e4, 2e4, 3e4}
# Note: n_e = {1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8}
logT_SH95 = np.array([np.log10(5e2), 3., np.log10(3e3), np.log10(5e3), np.log10(7.5e3),
                      4., np.log10(1.25e4), np.log10(1.5e4), np.log10(2e4), np.log10(3e4)])

# Case B probability of emitting a Balmer-alpha photon per recombination (T,n_e)
P_Ha_SH95 = np.array([
  [0.60176416, 0.59203990, 0.57777268, 0.55757870, 0.53218274, 0.50309773, 0.47538990],
  [0.57146208, 0.56548156, 0.55537185, 0.54126081, 0.52171947, 0.49779442, 0.47325611],
  [0.51602376, 0.51308374, 0.50870615, 0.50163462, 0.49109775, 0.47695493, 0.46057657],
  [0.48842989, 0.48670009, 0.48384243, 0.47916786, 0.47205133, 0.46211067, 0.45011817],
  [0.46673159, 0.46571739, 0.46369825, 0.46057413, 0.45555148, 0.44857577, 0.43958049],
  [0.45180723, 0.45102825, 0.44947568, 0.44724222, 0.44349888, 0.43815604, 0.43129878],
  [0.44059820, 0.43967487, 0.43880459, 0.43692031, 0.43424518, 0.42968461, 0.42434709],
  [0.43139784, 0.43080340, 0.43015528, 0.42868403, 0.42629594, 0.42286913, 0.41840805],
  [0.41795682, 0.41743320, 0.41691031, 0.41586673, 0.41447832, 0.41195330, 0.40867726],
  [0.39957948, 0.39949886, 0.39933772, 0.39872322, 0.39820159, 0.39687074, 0.39529763]])

# Interpolated probability for conversion to line photons
def P_B(Ptab, T, n_e, T_floor_rec=7000.):
    T_rec = T if (T > T_floor_rec) else T_floor_rec # Temperature floor
    i_L_T = 8; i_L_n = 0;                  # Lower interpolation index
    frac_R_T = 1.; frac_R_n = 0.;          # Upper interpolation fraction
    if T_rec <= 5e2:                       # Temperature minimum = 500 K
        i_L_T = 0
        frac_R_T = 0.
    elif T_rec < 3e4:                      # Temperature maximum = 30,000 K
        logT = np.log10(T_rec)             # Interpolate in log space
        while logT < logT_SH95[i_L_T]:
            i_L_T -= 1                     # Search temperature indices
        frac_R_T = (logT - logT_SH95[i_L_T]) / (logT_SH95[i_L_T+1] - logT_SH95[i_L_T])
    if n_e >= 1e8:                         # Density maximum = 10^8 cm^-3
        i_L_n = 5
        frac_R_n = 1.
    elif n_e > 1e2:                        # Density minimum = 10^2 cm^-3
        d_logn = np.log10(n_e) - 2.        # Table coordinates (log n_e)
        f_logn = np.floor(d_logn)          # Floor coordinate
        frac_R_n = d_logn - f_logn         # Interpolation fraction
        i_L_n = int(f_logn)                # Lower table index

    # Bilinear interpolation (based on left and right fractions)
    i_R_T = i_L_T + 1; i_R_n = i_L_n + 1
    frac_L_T = 1. - frac_R_T; frac_L_n = 1. - frac_R_n
    return (frac_L_T * (Ptab[i_L_T,i_L_n]*frac_L_n + Ptab[i_L_T,i_R_n]*frac_R_n)
          + frac_R_T * (Ptab[i_R_T,i_L_n]*frac_L_n + Ptab[i_R_T,i_R_n]*frac_R_n))

# Balmer-alpha effective recombination coefficient (Case B) - Units: cm^3/s
def alpha_eff_B_Ha(T, n_e):
    return np.array([P_B(P_Ha_SH95, T[i], n_e[i]) for i in range(len(T))]) * alpha_B(T) # α_eff_B = P_B α_B

# data_dir = 'Data/Thesan-Zooms-COLT'
data_dir = f'/orcd/data/mvogelsb/005/Lab/Thesan-Zooms-COLT'
# nsnaps = [186,187,188]  # Single snapshot for testing
nsnaps = np.arange(0, 189,1)  # Range of snapshots
simulations = ['g500531/z4']   ## List of simulations to process
# simulations = ['g2274036/z16','g519761/z16','g500531/z16','137030/z16','g10304/z8',
#                 'g5760/z8','g1163/z4','g578/z4','g205/z4']   ## List of simulations to process
for sim in simulations:
    print(f"Processing simulation {sim}")
    for snap in nsnaps:
        try:
            print(f"Processing snapshot {snap}")
            colt_file = f'{data_dir}/{sim}/ics_tree/colt_{snap}.hdf5'
            state_file = f'{data_dir}/{sim}/ics_tree/states-no-UVB_{snap}.hdf5'
            with h5py.File(colt_file, 'r') as f, h5py.File(state_file, 'r') as s:
                rho = f['rho'][:]
                # T = f['T'][:]
                e_int = f['e_int'][:]
                v = f['v'][:]
                v_star = f['v_star'][:]
                Z = f['Z'][:]
                D = f['D'][:]
                x_e = s['x_e'][:]  # import from states
                x_HII = s['x_HII'][:]  # import from states
                V = f['V'][:]
                gamma = 5. / 3.;          # Adiabatic index
                T_div_emu = (gamma - 1.) * mH / kB; # T / (e * mu)
                mu = 4. / (1. + X * (3. + 4. * x_e)); # Mean molecular mass [mH]
                T = T_div_emu * e_int * mu;      # Gas temperature [K]

            n_H = X * rho * (1. - Z - D) / mH  # n_H = X rho (1-Z-D) / mH
            n_e = n_H * x_e  # Electron number density [cm^-3]
            V_nH_ne = V * n_H**2 * x_e  # Shared constant [cm^-3]
            # Recombination luminosity [erg/s]: L_rec = hν ∫ α_eff_B(T) n_e n_p dV
            L = E0_Ha * V_nH_ne * x_HII * alpha_eff_B_Ha(T, n_e) # [erg/s]
            v_Ha = np.sum(v * L[:,None], axis=0) / np.sum(L)  # Center of light velocity

            # When writing the file use

            with h5py.File(colt_file, 'r+') as f:
                v = f['v']
                v[:] -= v_Ha  # Recentered velocities
                v_star = f['v_star']
                v_star[:] -= v_Ha  # Recentered velocities
                f.attrs['v_Ha'] = v_Ha  # Store velocity shift [cm/s]
                print(f"Snapshot {snap} recentered. Velocity shift: {v_Ha}")
        except Exception as e:
            print(f"Error processing snapshot {snap}: {e}")
            continue
