import h5py
import numpy as np
import matplotlib.pyplot as plt

SOLAR_MASS = 1.989e33  # Solar masses
pc = 3.085677581467192e18  # Units: 1 pc  = 3e18 cm
kpc = 1e3 * pc

# Global variables
snap = 188 # Snapshot number
out_dir = '.'

with h5py.File(f'{out_dir}/distances_{snap:03d}.hdf5', 'r') as f:
    # Load header
    header = f['Header'].attrs
    n_groups_tot = header['Ngroups_Total']
    a = header['Time']
    z = 1. / a - 1
    box_size = header['BoxSize']
    h = header['HubbleParam']
    omega0 = header['Omega0']
    omega_baryon = header['OmegaBaryon']
    omega_lambda = header['OmegaLambda']
    UnitLength_in_cm = header['UnitLength_in_cm']
    UnitMass_in_g = header['UnitMass_in_g']
    UnitVelocity_in_cm_per_s = header['UnitVelocity_in_cm_per_s']
    length_to_cgs = a * UnitLength_in_cm / h
    length_to_kpc = length_to_cgs / kpc
    volume_to_cgs = length_to_cgs**3
    mass_to_cgs = UnitMass_in_g / h
    mass_to_msun = mass_to_cgs / SOLAR_MASS
    velocity_to_cgs = np.sqrt(a) * UnitVelocity_in_cm_per_s
    density_to_cgs = mass_to_cgs / volume_to_cgs
    PosHR = header['PosHR']
    RadiusHR = header['RadiusHR']
    RadiusLR = header['RadiusLR']
    # Load data
    g = f['Group']
    ds = {key: g[key][:] for key in g.keys()}

# Mask out groups with R_Crit200 == 0
mask = ds['R_Crit200'] > 0
print(f'Number of groups: {np.count_nonzero(mask)}  [R_Crit200 > 0]')
for key in ds: ds[key] = ds[key][mask]

# Require at least 1 high-resolution star particle
mask = ds['MinDistStarsHR'] < ds['R_Crit200']
print(f'Number of groups: {np.count_nonzero(mask)}  [MinDist(StarsHR) < R_Crit200]')
for key in ds: ds[key] = ds[key][mask]

# Require at least 1 high-resolution gas particle
mask = ds['MinDistGasHR'] < ds['R_Crit200']
print(f'Number of groups: {np.count_nonzero(mask)}  [MinDist(GasHR) < R_Crit200]')
for key in ds: ds[key] = ds[key][mask]

# Calculate the minimum distance to a low-resolution particle
ds['MinDistLR'] = np.minimum(ds['MinDistP2'], ds['MinDistP3'], ds['MinDistStarsLR']) # P2,P3,StarsLR

# Mask out groups with MinDistP2 or MinDistP3 < R_Crit200
mask = ds['MinDistLR'] > ds['R_Crit200']
print(f'Number of groups: {np.count_nonzero(mask)}  [MinDist(P2,P3,StarsLR) > R_Crit200]')
for key in ds: ds[key] = ds[key][mask]

# Mask out groups with MinDistGasLR < R_Crit200
# ds['MinDistLR'] = np.minimum(ds['MinDistLR'], ds['MinDistGasLR']) # GasLR
# mask = ds['MinDistLR'] > ds['R_Crit200']
# print(f'Number of groups: {np.count_nonzero(mask)}  [MinDist(GasLR) > R_Crit200]')
# for key in ds: ds[key] = ds[key][mask]

mask = ds['MinDistLR'] > 2. * ds['R_Crit200']
print(f'(Stats Only) Number of groups: {np.count_nonzero(mask)}  [MinDist(LR) > 2 R_Crit200]')

mask = ds['MinDistLR'] > 3. * ds['R_Crit200']
print(f'(Stats Only) Number of groups: {np.count_nonzero(mask)}  [MinDist(LR) > 3 R_Crit200]')
# for key in ds: ds[key] = ds[key][mask]

# For plotting convenience only: Convert distances to kpc and masses to solar masses
for key in ds.keys():
    if 'Dist' in key or 'R_' in key or 'Pos' in key:
        ds[key] *= length_to_kpc
    elif 'M_' in key:
        ds[key] *= mass_to_msun
    else:
        print(f'Warning: {key} not converted')
PosHR *= length_to_kpc
RadiusHR *= length_to_kpc
RadiusLR *= length_to_kpc

# mask = ds['MinDistLR'] > 3. * ds['R_Crit200']
# print(f'Number of groups: {np.count_nonzero(mask)}  [MinDistLR > 3 R_Crit200 and MinDistLR > 10 kpc]')
# for key in ds: ds[key] = ds[key][mask]

def plot_Rmin_M200():
    fig = plt.figure(figsize=(4.5,3.))
    ax = plt.axes([0,0,1,1])

    ax.scatter(ds['M_Crit200'], ds['MinDistLR'], s=4, c='C0') #, label='Gas (LR)')

    ## Axes labels ##
    # ax.legend(loc='upper right', frameon=False, borderaxespad=.75, handlelength=2., fontsize=12)
    ax.set_xlabel(r'$M_{\rm 200}\ \,({\rm M}_{\odot})$', fontsize=15)
    # ax.set_xlabel(r'${\rm Time\ \ (Gyr)}$', fontsize=15)
    # ax.set_xlabel(r'${\rm Redshift}$', fontsize=15)
    ax.set_ylabel(r'${\rm Low-res\ Distance\ (kpc)}$', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    # xmin, xmax = 2, 20
    # ymin, ymax = 0, 1
    # ax.set_xlim([xmin, xmax])
    # ax.set_ylim([ymin, ymax])
    # ticks = [2,3,4,5,6,7,8,10,12,15,20]
    # ax.set_xticks(ticks); ax.set_xticklabels([r'$%g$' % tick for tick in ticks], fontsize=12)
    # ticks = np.linspace(2,20,19); ax.set_xticks(ticks, minor=True)
    # ax.set_xticklabels(['']*len(ticks), minor=True)
    # ticks = np.linspace(xmin, xmax, 6)
    # ax.set_xticks(ticks); ax.set_xticklabels([r'$%g$' % tick for tick in ticks], fontsize=13)
    # Lxmin,Lxmax = int(np.ceil(np.log10(xmin))), int(np.floor(np.log10(xmax)))
    # ticks = np.linspace(Lxmin, Lxmax, Lxmax-Lxmin+1)
    # ax.set_xticks(10**ticks); ax.set_xticklabels([r'$10^{%g}$' % tick for tick in ticks], fontsize=13)
    # ax.set_xticks(10**ticks); ax.set_xticklabels([r'$10^{%g}$' % tick for tick in ticks], fontsize=13)
    # ticks = np.linspace(ymin, ymax, 6)
    # ax.set_yticks(ticks); ax.set_yticklabels([r'$%g$' % tick for tick in ticks], fontsize=12)
    # Lymin,Lymax = int(np.ceil(np.log10(ymin))), int(np.floor(np.log10(ymax)))
    # ticks = np.linspace(Lymin, Lymax, Lymax-Lymin+1)
    # ax.set_yticks(10**ticks); ax.set_yticklabels([r'$10^{%g}$' % tick for tick in ticks], fontsize=13)
    # ax.invert_xaxis()
    fig.savefig('plots/Rmin_M200.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)

def plot_dist_cdf():
    fig = plt.figure(figsize=(4.5,3.))
    ax = plt.axes([0,0,1,1])

    # Calculate the cumulative distribution function of distances
    dist = np.sqrt(np.sum((ds['Pos'] - PosHR)**2, axis=1))
    dist = np.sort(dist)
    cdf = np.arange(1, len(dist) + 1) / len(dist)
    ax.plot(dist, cdf, c='C0', lw=2)

    ## Axes labels ##
    # ax.legend(loc='upper right', frameon=False, borderaxespad=.75, handlelength=2., fontsize=12)
    ax.set_xlabel(r'$\|{\bf r} - {\bf r}_{\rm com}\|\ \,({\rm kpc})$', fontsize=15)
    # ax.set_xlabel(r'${\rm Time\ \ (Gyr)}$', fontsize=15)
    # ax.set_xlabel(r'${\rm Redshift}$', fontsize=15)
    ax.set_ylabel(r'${\rm CDF\ \,of\ \,'+str(len(ds['R_Crit200']))+r'\ \,Candidates}$', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.minorticks_on()
    # xmin, xmax = 2, 20
    ymin, ymax = 0, 1
    # ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    # ticks = [2,3,4,5,6,7,8,10,12,15,20]
    # ax.set_xticks(ticks); ax.set_xticklabels([r'$%g$' % tick for tick in ticks], fontsize=12)
    # ticks = np.linspace(2,20,19); ax.set_xticks(ticks, minor=True)
    # ax.set_xticklabels(['']*len(ticks), minor=True)
    # ticks = np.linspace(xmin, xmax, 6)
    # ax.set_xticks(ticks); ax.set_xticklabels([r'$%g$' % tick for tick in ticks], fontsize=13)
    # Lxmin,Lxmax = int(np.ceil(np.log10(xmin))), int(np.floor(np.log10(xmax)))
    # ticks = np.linspace(Lxmin, Lxmax, Lxmax-Lxmin+1)
    # ax.set_xticks(10**ticks); ax.set_xticklabels([r'$10^{%g}$' % tick for tick in ticks], fontsize=13)
    # ax.set_xticks(10**ticks); ax.set_xticklabels([r'$10^{%g}$' % tick for tick in ticks], fontsize=13)
    ticks = np.linspace(ymin, ymax, 6)
    ax.set_yticks(ticks); ax.set_yticklabels([r'$%g$' % tick for tick in ticks], fontsize=12)
    # Lymin,Lymax = int(np.ceil(np.log10(ymin))), int(np.floor(np.log10(ymax)))
    # ticks = np.linspace(Lymin, Lymax, Lymax-Lymin+1)
    # ax.set_yticks(10**ticks); ax.set_yticklabels([r'$10^{%g}$' % tick for tick in ticks], fontsize=13)
    # ax.invert_xaxis()
    fig.savefig('plots/dist_cdf.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)

if __name__ == '__main__':
    plot_Rmin_M200()
    plot_dist_cdf()

