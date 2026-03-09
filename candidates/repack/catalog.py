import numpy as np
import h5py
import os, sys, platform
import astropy.units as u
import pickle
from functools import reduce
from astropy.cosmology import Planck18 as cosmo
import matplotlib.pyplot as plt
sims = {
    'g39': ['z4'],
  'g205': ['z4'],
  'g578': ['z4'],
  'g1163': ['z4'],
  'g5760': ['z4','z8'],
  'g10304': ['z4','z8'],
  'g33206': ['z4', 'z8'],
  'g37591': ['z4', 'z8'],
  'g137030': ['z4', 'z8','z16'],
  'g500531': ['z4', 'z8','z16'],
  'g519761': ['z4', 'z8','z16'],
  'g2274036': ['z4', 'z8','z16'],
  'g5229300': ['z4', 'z8','z16']
}

def progressbar(it, prefix="", size=100, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

def repack_1500(snaps, data_dir='.', out_dir='.'):
    n_snaps = len(snaps)
    success = np.zeros(n_snaps, dtype=bool)
    n_cameras = 8
    zs = np.empty(n_snaps)
    L_tot = np.empty(n_snaps)
    f_esc = np.empty(n_snaps)
    f_escs = np.empty((n_snaps, n_cameras))
    M_1500 = np.empty(n_snaps)
    M_1500_obs = np.empty((n_snaps, n_cameras))
    Ndot_LyC = np.empty(n_snaps)
    L_1500_arr = np.empty(n_snaps)
    f_esc_LyC = np.empty(n_snaps)
    f_escs_LyC = np.empty((n_snaps, n_cameras))
    z_ranges = np.array([16.5, 12.8, 10.4, 8.7, 7.37, 6.33, 5.49, 4.8, 4.24, 3.76, 3.35, 2.5])
    z_names = np.array([14, 11.5, 9.5, 8, 7, 6, 5, 4.5, 4, 3.5, 3])
    for i in progressbar(range(n_snaps)):
        try:
            with h5py.File(f'{data_dir}/M1500/M1500_{snaps[i]:03d}.hdf5', 'r') as f:
                zs[i] = f.attrs['z']
                L_tot[i] = f.attrs['L_tot']
                f_esc[i] = f.attrs['f_esc']
                f_escs[i,:] = f['f_escs'][:]
                n_bins = f.attrs['n_bins']
                Ndot_1500 = f.attrs['Ndot_tot']  # Photon rate [photons/s]
                edges_eV = f['bin']['edges_eV'][:] # Energy bin edges [eV]
                eV = 1.60217725e-12              # Electron volt: 1 eV = 1.6e-12 erg
                angstrom = 1e-8                  # Units: 1 angstrom = 1e-8 cm
                c = 2.99792458e10                # Speed of light [cm/s]
                h = 6.626069573e-27              # Planck's constant [erg s]
                lambda_1500 = 1500. * angstrom  # Continuum wavelength [cm]
                nu_1500 = c / lambda_1500       # Continuum frequency [Hz]
                Delta_lambda = h * c * (1./edges_eV[0] - 1./edges_eV[1]) / (eV * angstrom)  # Bin width [angstrom]
                L_1500 = h * nu_1500 * Ndot_1500 / Delta_lambda  # Spectral luminosity [erg/s/angstrom]
                L_1500_arr[i] = L_1500
                L_1500_obs = L_1500 * f['f_escs'][:]
                pc = 3.085677581467192e18       # Units: 1 pc  = 3e18 cm
                R_10pc = 10. * pc               # Reference distance for continuum [cm]
                fnu_1500_fac = lambda_1500**2 / (4. * np.pi * c * R_10pc**2 * angstrom)
                M_1500[i] = -2.5 * np.log10(fnu_1500_fac * L_1500) - 48.6 # Continuum absolute magnitude
                M_1500_obs[i,:] = -2.5 * np.log10(fnu_1500_fac * L_1500_obs) - 48.6
            with h5py.File(f'{data_dir}/ion-eq/ion-eq_{snaps[i]:03d}.hdf5', 'r') as f:
                Ndot_LyC[i] = f.attrs['Ndot_tot']
                f_esc_LyC[i] = f.attrs['f_esc']
                f_escs_LyC[i,:] = f['f_escs'][:]
            success[i] = True
        except Exception as e:
            print(e)
    snaps = np.array(snaps, dtype=np.int32)
    n_success = np.count_nonzero(success)
    if n_success < n_snaps:
        if n_success == 0:
            print(f'No snapshots found for {data_dir} ...')
            return
        print(f'Only {n_success} out of {n_snaps} snapshots found for {data_dir} ...')
        n_snaps = np.int32(n_success)
        snaps = snaps[success]
        zs = zs[success]
        L_tot = L_tot[success]
        f_esc = f_esc[success]
        f_escs = f_escs[success,:]
        M_1500 = M_1500[success]
        L_1500_arr = L_1500_arr[success]
        M_1500_obs = M_1500_obs[success,:]
        Ndot_LyC = Ndot_LyC[success]
        f_esc_LyC = f_esc_LyC[success]
        f_escs_LyC = f_escs_LyC[success,:]

    os.makedirs(f'{out_dir}/M1500/', exist_ok=True)
    if os.path.exists(f'{out_dir}/M1500/M1500.hdf5'):
        mode = 'r+'
        with h5py.File(f'{out_dir}/M1500/M1500.hdf5', mode) as f:
            f.create_dataset('L_1500', data=L_1500_arr)
    else:
        mode ='w'
        with h5py.File(f'{out_dir}/M1500/M1500.hdf5', mode) as f:
            f.attrs['n_snaps'] = n_snaps
            f.attrs['n_cameras'] = n_cameras
            f.create_dataset('snaps', data=snaps)
            f.create_dataset('zs', data=zs)
            f.create_dataset('L_tot', data=L_tot)
            f.create_dataset('f_esc', data=f_esc)
            f.create_dataset('f_escs', data=f_escs)
            f.create_dataset('M_1500', data=M_1500)
            f.create_dataset('M_1500_obs', data=M_1500_obs)
            f.create_dataset('Ndot_LyC', data=Ndot_LyC)
            f.create_dataset('f_esc_LyC', data=f_esc_LyC)
            f.create_dataset('f_escs_LyC', data=f_escs_LyC)
            f.create_dataset('L_1500', data=L_1500_arr)

def repack_base(snaps, data_dir='.', out_dir='.', base='Lya', RHD=False):
    n_snaps = len(snaps)
    success = np.zeros(n_snaps, dtype=bool)
    n_cameras = 8
    zs = np.empty(n_snaps)
    L_tot = np.empty(n_snaps)
    Ndot_base = np.empty(n_snaps)
    f_esc = np.empty(n_snaps)
    L_base = np.empty(n_snaps)
    f_escs = np.empty((n_snaps, n_cameras))
    # freq_sep = np.empty(n_snaps)
    freq_avg = np.empty(n_snaps)
    freq_std = np.empty(n_snaps)
    nside_base = 5
    nside_LyC = 10
    map_base = np.empty((n_snaps, 12*(nside_base)**2))
    freq_map = np.empty((n_snaps, 12*(nside_base)**2))
    freq2_map = np.empty((n_snaps, 12*(nside_base)**2))
    map_LyC = np.empty((n_snaps, 12*(nside_LyC)**2))
    map_1500 = np.empty((n_snaps, 12*(nside_LyC)**2))
    # freq_seps = np.empty((n_snaps, n_cameras))
    freq_avgs = np.empty((n_snaps, n_cameras))
    freq_stds = np.empty((n_snaps, n_cameras))
    M_1500 = np.empty(n_snaps)
    M_1500_obs = np.empty((n_snaps, n_cameras))
    Ndot_LyC = np.empty(n_snaps)
    f_esc_LyC = np.empty(n_snaps)
    f_escs_LyC = np.empty((n_snaps, n_cameras))
    for i in progressbar(range(n_snaps)):
        try:
            # print(f'{data_dir}/{base}/{base}_{snaps[i]:03d}.hdf5')
            with h5py.File(f'{data_dir}/{base}/{base}_{snaps[i]:03d}.hdf5', 'r') as f:
                zs[i] = f.attrs['z']
                L_tot[i] = f.attrs['L_tot']
                Ndot_base[i] = f.attrs['Ndot_tot']
                E0 = f['line'].attrs['E0']
                # print(E0)
                L = Ndot_base[i] * E0  ## erg per s
                # print(L)
                L_base[i] = L
                map_base[i,:] = f['map'][:]
                freq_map[i,:] = f['freq_map'][:]
                freq2_map[i,:] = f['freq2_map'][:]
                f_esc[i] = f.attrs['f_esc']
                f_escs[i,:] = f['f_escs'][:]
                freq_avg[i] = f.attrs['freq_avg']
                freq_std[i] = f.attrs['freq_std']
                freq_avgs[i,:] = f['freq_avgs'][:]
                freq_stds[i,:] = f['freq_stds'][:]
                fluxes = f['fluxes'][:]
                n_bins = f.attrs['n_bins']
                freq_max = f.attrs['freq_max']
                freq_min = f.attrs['freq_min']
                freq_edges = np.linspace(freq_min, freq_max, n_bins+1)
                freqs = 0.5 * (freq_edges[:-1] + freq_edges[1:])
                # red = (freqs > 0.)
                # for j in range(n_cameras):
                #     freq_seps[i,j] = freqs[red][np.argmax(fluxes[j,red])] - freqs[~red][np.argmax(fluxes[j,~red])]
                flux = f['flux_avg'][:]
                # freq_sep[i] = freqs[red][np.argmax(flux[red])] - freqs[~red][np.argmax(flux[~red])]
            if RHD:
                with h5py.File(f'{data_dir}/ion-eq-RHD/ion-eq-RHD_{snaps[i]:03d}.hdf5', 'r') as f:
                    Ndot_LyC[i] = f.attrs['Ndot_tot']
                    f_esc_LyC[i] = f.attrs['f_esc']
                    map_LyC[i,:] = f['map'][:]
                    f_escs_LyC[i,:] = f['f_escs'][:]
            else:
                with h5py.File(f'{data_dir}/ion-eq/ion-eq_{snaps[i]:03d}.hdf5', 'r') as f:
                    Ndot_LyC[i] = f.attrs['Ndot_tot']
                    f_esc_LyC[i] = f.attrs['f_esc']
                    map_LyC[i,:] = f['map'][:]
                    f_escs_LyC[i,:] = f['f_escs'][:]
            
            with h5py.File(f'{data_dir}/M1500/M1500_{snaps[i]:03d}.hdf5', 'r') as f:
                Ndot_1500 = f.attrs['Ndot_tot']  # Photon rate [photons/s]
                edges_eV = f['bin']['edges_eV'][:] # Energy bin edges [eV]
                map_1500[i,:] = f['map'][:]
                eV = 1.60217725e-12              # Electron volt: 1 eV = 1.6e-12 erg
                angstrom = 1e-8                  # Units: 1 angstrom = 1e-8 cm
                c = 2.99792458e10                # Speed of light [cm/s]
                h = 6.626069573e-27              # Planck's constant [erg s]
                lambda_1500 = 1500. * angstrom  # Continuum wavelength [cm]
                nu_1500 = c / lambda_1500       # Continuum frequency [Hz]
                Delta_lambda = h * c * (1./edges_eV[0] - 1./edges_eV[1]) / (eV * angstrom)  # Bin width [angstrom]
                L_1500 = h * nu_1500 * Ndot_1500 / Delta_lambda  # Spectral luminosity [erg/s/angstrom]
                L_1500_obs = L_1500 * f['f_escs'][:]
                pc = 3.085677581467192e18       # Units: 1 pc  = 3e18 cm
                R_10pc = 10. * pc               # Reference distance for continuum [cm]
                fnu_1500_fac = lambda_1500**2 / (4. * np.pi * c * R_10pc**2 * angstrom)
                M_1500[i] = -2.5 * np.log10(fnu_1500_fac * L_1500) - 48.6 # Continuum absolute magnitude
                M_1500_obs[i,:] = -2.5 * np.log10(fnu_1500_fac * L_1500_obs) - 48.6
            success[i] = True
        except Exception as e:
            print(e)
    
    snaps = np.array(snaps, dtype=np.int32)
    n_success = np.count_nonzero(success)
    if n_success < n_snaps:
        if n_success == 0:
            print(f'No snapshots found for {data_dir} ...')
            return
        print(f'Only {n_success} out of {n_snaps} snapshots found for {data_dir} ...')
        n_snaps = np.int32(n_success)
        snaps = snaps[success]
        zs = zs[success]
        L_tot = L_tot[success]
        L_base = L_base[success]
        Ndot_base = Ndot_base[success]
        f_esc = f_esc[success]
        f_escs = f_escs[success,:]
        # freq_sep = freq_sep[success]
        freq_avg = freq_avg[success]
        freq_std = freq_std[success]
        # freq_seps = freq_seps[success,:]
        freq_avgs = freq_avgs[success,:]
        freq_map = freq_map[success,:]
        freq2_map = freq2_map[success,:]
        map_base = map_base[success,:]
        map_LyC = map_LyC[success,:]
        map_1500 = map_1500[success,:]
        freq_stds = freq_stds[success,:]
        M_1500 = M_1500[success]
        M_1500_obs = M_1500_obs[success,:]
        Ndot_LyC = Ndot_LyC[success]
        f_esc_LyC = f_esc_LyC[success]
        f_escs_LyC = f_escs_LyC[success,:]
    os.makedirs(f'{out_dir}/{base}/', exist_ok=True)
    if os.path.exists(f'{out_dir}/{base}/{base}.hdf5'):
        mode = 'r+'
        ## If you want to add new field to the summary file make sure it exists at the path
        with h5py.File(f'{out_dir}/{base}/{base}.hdf5', mode) as f:
            del f['Ndot_LyC_RHD']
            del f['f_esc_LyC_RHD']
            del f['f_escs_LyC_RHD']
            del f['map_LyC_RHD']
            f.create_dataset('Ndot_LyC_RHD', data=Ndot_LyC)
            f.create_dataset('f_esc_LyC_RHD', data=f_esc_LyC)
            f.create_dataset('f_escs_LyC_RHD', data=f_escs_LyC)
            f.create_dataset('map_LyC_RHD', data=map_LyC)
    else:
        mode = 'w'
        ## Base summary file that was made
        with h5py.File(f'{out_dir}/{base}/{base}.hdf5', mode) as f:
            f.attrs['n_snaps'] = n_snaps
            f.attrs['n_cameras'] = n_cameras
            f.create_dataset('snaps', data=snaps)
            f.create_dataset('zs', data=zs)
            f.create_dataset('Ndot_base', data=Ndot_base)
            f.create_dataset('L', data=L_base)
            f.create_dataset('L_tot', data=L_tot)
            f.create_dataset('f_esc', data=f_esc)
            f.create_dataset('f_escs', data=f_escs)
            # f.create_dataset('freq_sep', data=freq_sep)
            f.create_dataset('freq_avg', data=freq_avg)
            f.create_dataset('freq_std', data=freq_std)
            # f.create_dataset('freq_seps', data=freq_seps)
            f.create_dataset('freq_avgs', data=freq_avgs)
            f.create_dataset('map_base', data=map_base)
            f.create_dataset('map_LyC', data=map_LyC)
            f.create_dataset('map_1500', data=map_1500)
            f.create_dataset('freq_stds', data=freq_stds)
            f.create_dataset('M_1500', data=M_1500)
            f.create_dataset('M_1500_obs', data=M_1500_obs)
            f.create_dataset('Ndot_LyC', data=Ndot_LyC)
            f.create_dataset('f_esc_LyC', data=f_esc_LyC)
            f.create_dataset('f_escs_LyC', data=f_escs_LyC)

def repack_cont(snaps, data_dir='.', out_dir='.', base='Lya'):
    eV = 1.60217725e-12              # Electron volt: 1 eV = 1.6e-12 erg
    angstrom = 1e-8                  # Units: 1 angstrom = 1e-8 cm
    c = 2.99792458e10                # Speed of light [cm/s]
    km = 1e5                         # km to cm
    h = 6.626069573e-27              # Planck's constant [erg s]
    n_snaps = len(snaps)
    success = np.zeros(n_snaps, dtype=bool)
    n_cameras = 8
    zs = np.empty(n_snaps)
    L_tot = np.empty(n_snaps)
    Ndot_line = np.empty(n_snaps)
    EW_0 = np.empty(n_snaps)
    f_esc = np.empty(n_snaps)
    L_line = np.empty(n_snaps)
    nside_base = 5
    nside_LyC = 10
    map_base = np.empty((n_snaps, 12*(nside_base)**2))
    map_LyC = np.empty((n_snaps, 12*(nside_LyC)**2))
    map_1500 = np.empty((n_snaps, 12*(nside_LyC)**2))
    f_escs = np.empty((n_snaps, n_cameras))
    # freq_sep = np.empty(n_snaps)
    freq_avg = np.empty(n_snaps)
    freq_std = np.empty(n_snaps)
    # freq_seps = np.empty((n_snaps, n_cameras))
    freq_avgs = np.empty((n_snaps, n_cameras))
    freq_stds = np.empty((n_snaps, n_cameras))
    M_1500 = np.empty(n_snaps)
    M_1500_obs = np.empty((n_snaps, n_cameras))
    Ndot_LyC = np.empty(n_snaps)
    f_esc_LyC = np.empty(n_snaps)
    f_escs_LyC = np.empty((n_snaps, n_cameras))
    for i in progressbar(range(n_snaps)):
        try:
            line = base.split('-')[0]
            with h5py.File(f'{data_dir}/{line}/{line}_{snaps[i]:03d}.hdf5', 'r') as f:
                L_line[i] = f.attrs['L_tot']
                Ndot_line[i] = f.attrs['Ndot_tot']
            with h5py.File(f'{data_dir}/{base}/{base}_{snaps[i]:03d}.hdf5', 'r') as f:
                zs[i] = f.attrs['z']
                L_tot[i] = f.attrs['L_tot']
                E0 = f['line'].attrs['E0']
                dv_cont_min = f['sources'].attrs['Dv_cont_min']
                dv_cont_max = f['sources'].attrs['Dv_cont_max']
                lambda0 = f['line'].attrs['lambda0'] #* angstrom
                lambda_min = lambda0 * (1. + dv_cont_min * km / c)  # Minimum wavelength [angstrom]
                lambda_max = lambda0 * (1. + dv_cont_max * km / c)  # Maximum wavelength [angstrom]
                dlambda = lambda_max - lambda_min  # Rest-frame width [angstrom] Use this for EW_0 [A]
                dlambda_obs = dlambda * (1. + zs[i])  # Observed width [angstrom] Use this for [erg/s/A]
                L_cont = f['sources'].attrs['L_cont']
                F_lambda_obs =L_cont / dlambda_obs # [erg/s/A]
                EW_0[i] = L_line[i] / F_lambda_obs / (1. + zs[i]) 
                f_esc[i] = f.attrs['f_esc']
                f_escs[i,:] = f['f_escs'][:]
                freq_avg[i] = f.attrs['freq_avg']
                freq_std[i] = f.attrs['freq_std']
                freq_avgs[i,:] = f['freq_avgs'][:]
                freq_stds[i,:] = f['freq_stds'][:]
                fluxes = f['fluxes'][:]
                map_base[i,:] = f['map'][:]
                n_bins = f.attrs['n_bins']
                freq_max = f.attrs['freq_max']
                freq_min = f.attrs['freq_min']
                freq_edges = np.linspace(freq_min, freq_max, n_bins+1)
                freqs = 0.5 * (freq_edges[:-1] + freq_edges[1:])
                # red = (freqs > 0.)
                # for j in range(n_cameras):
                #     freq_seps[i,j] = freqs[red][np.argmax(fluxes[j,red])] - freqs[~red][np.argmax(fluxes[j,~red])]
                flux = f['flux_avg'][:]
                # freq_sep[i] = freqs[red][np.argmax(flux[red])] - freqs[~red][np.argmax(flux[~red])]
            with h5py.File(f'{data_dir}/ion-eq/ion-eq_{snaps[i]:03d}.hdf5', 'r') as f:
                Ndot_LyC[i] = f.attrs['Ndot_tot']
                f_esc_LyC[i] = f.attrs['f_esc']
                map_LyC[i,:] = f['map'][:]
                f_escs_LyC[i,:] = f['f_escs'][:]
            with h5py.File(f'{data_dir}/M1500/M1500_{snaps[i]:03d}.hdf5', 'r') as f:
                Ndot_1500 = f.attrs['Ndot_tot']  # Photon rate [photons/s]
                map_1500[i,:] = f['map'][:]
                edges_eV = f['bin']['edges_eV'][:] # Energy bin edges [eV]
                lambda_1500 = 1500. * angstrom  # Continuum wavelength [cm]
                nu_1500 = c / lambda_1500       # Continuum frequency [Hz]
                Delta_lambda = h * c * (1./edges_eV[0] - 1./edges_eV[1]) / (eV * angstrom)  # Bin width [angstrom]
                L_1500 = h * nu_1500 * Ndot_1500 / Delta_lambda  # Spectral luminosity [erg/s/angstrom]
                L_1500_obs = L_1500 * f['f_escs'][:]
                pc = 3.085677581467192e18       # Units: 1 pc  = 3e18 cm
                R_10pc = 10. * pc               # Reference distance for continuum [cm]
                fnu_1500_fac = lambda_1500**2 / (4. * np.pi * c * R_10pc**2 * angstrom)
                M_1500[i] = -2.5 * np.log10(fnu_1500_fac * L_1500) - 48.6 # Continuum absolute magnitude
                M_1500_obs[i,:] = -2.5 * np.log10(fnu_1500_fac * L_1500_obs) - 48.6
            success[i] = True
        except Exception as e:
            print(e)
    snaps = np.array(snaps, dtype=np.int32)
    n_success = np.count_nonzero(success)
    if n_success < n_snaps:
        if n_success == 0:
            print(f'No snapshots found for {data_dir} ...')
            return
        print(f'Only {n_success} out of {n_snaps} snapshots found for {data_dir} ...')
        n_snaps = np.int32(n_success)
        snaps = snaps[success]
        zs = zs[success]
        L_tot = L_tot[success]
        L_line = L_line[success]
        Ndot_line = Ndot_line[success]
        EW_0 = EW_0[success]
        f_esc = f_esc[success]
        f_escs = f_escs[success,:]
        # freq_sep = freq_sep[success]
        freq_avg = freq_avg[success]
        freq_std = freq_std[success]
        # freq_seps = freq_seps[success,:]
        freq_avgs = freq_avgs[success,:]
        freq_stds = freq_stds[success,:]
        map_base = map_base[success,:]
        map_LyC = map_LyC[success,:]
        map_1500 = map_1500[success,:]
        M_1500 = M_1500[success]
        M_1500_obs = M_1500_obs[success,:]
        Ndot_LyC = Ndot_LyC[success]
        f_esc_LyC = f_esc_LyC[success]
        f_escs_LyC = f_escs_LyC[success,:]
    os.makedirs(f'{out_dir}/{base}/', exist_ok=True)
    if os.path.exists(f'{out_dir}/{base}/{base}.hdf5'):
        mode = 'r+'
        ## If you want to add new field to the summary file make sure it exists at the path
        with h5py.File(f'{out_dir}/{base}/{base}.hdf5', mode) as f:
            f.create_dataset('map_base', data=map_base)
            f.create_dataset('map_LyC', data=map_LyC)
            f.create_dataset('map_1500', data=map_1500)
    else:
        mode = 'w'
        ## Base summary file that was made
        with h5py.File(f'{out_dir}/{base}/{base}.hdf5', 'w') as f:
            f.attrs['n_snaps'] = n_snaps
            f.attrs['n_cameras'] = n_cameras
            f.create_dataset('snaps', data=snaps)
            f.create_dataset('zs', data=zs)
            f.create_dataset('Ndot_line', data=Ndot_line)
            f.create_dataset('L_line', data=L_line)
            f.create_dataset('L_tot', data=L_tot)
            f.create_dataset('EW_0', data=EW_0)
            f.create_dataset('f_esc', data=f_esc)
            f.create_dataset('f_escs', data=f_escs)
            # f.create_dataset('freq_sep', data=freq_sep)
            f.create_dataset('freq_avg', data=freq_avg)
            f.create_dataset('freq_std', data=freq_std)
            # f.create_dataset('freq_seps', data=freq_seps)
            f.create_dataset('freq_avgs', data=freq_avgs)
            f.create_dataset('freq_stds', data=freq_stds)
            f.create_dataset('M_1500', data=M_1500)
            f.create_dataset('M_1500_obs', data=M_1500_obs)
            f.create_dataset('Ndot_LyC', data=Ndot_LyC)
            f.create_dataset('f_esc_LyC', data=f_esc_LyC)
            f.create_dataset('f_escs_LyC', data=f_escs_LyC)

def repack_metals(snaps, sim='g500531/z16', out_dir='.'):
    # --- paths and simulation lists ---
    path_split = {
        'D4': '/orcd/data/mvogelsb/004/Thesan-Zooms-COLT',
        'D1': '/nfs/mvogelsblab001/Lab/Thesan-Zooms-COLT'
    }
    D1_sim = ['g10304/z8','g1163/z4','g137030/z16','g2/z4','g205/z4',
              'g2274036/z16','g39/z4','g500531/z16','g519761/z16',
              'g5760/z8','g578/z4']

    # --- ionic states per element ---
    state_dict = {
        'H':   ['HI', 'HII'],
        'He':  ['HeI', 'HeII'],
        'C':   ['CI', 'CII', 'CIII', 'CIV'],
        'N':   ['NI', 'NII', 'NIII', 'NIV', 'NV'],
        'O':   ['OI', 'OII', 'OIII', 'OIV'],
        'Ne':  ['NeI', 'NeII', 'NeIII', 'NeIV'],
        'Mg':  ['MgI', 'MgII', 'MgIII'],
        'Si':  ['SiI', 'SiII', 'SiIII', 'SiIV'],
        'S':   ['SI', 'SII', 'SIII', 'SIV', 'SV', 'SVI'],
        'Fe':  ['FeI', 'FeII', 'FeIII', 'FeIV', 'FeV', 'FeVI']
    }

    snaps = np.array(snaps, dtype=np.int32)
    n_snaps_total = len(snaps)
    zs_global = np.empty(n_snaps_total)

    # dictionaries to hold arrays for all elements
    mw_dict = {}
    vw_dict = {}

    # --- loop over elements ---
    for element, states in state_dict.items():
        n_snaps = len(snaps)
        success = np.zeros(n_snaps, dtype=bool)
        zs = np.empty(n_snaps)
        n_states = len(states)
        mw_arr = np.empty((n_snaps, n_states))
        vw_arr = np.empty((n_snaps, n_states))

        print(f"\nProcessing element {element} with states {states}")

        for i in progressbar(range(n_snaps)):
            snap = snaps[i]

            # determine data directory
            data_dir = path_split['D1'] if sim in D1_sim else path_split['D4']

            colt_file = f'{data_dir}/{sim}/ics_tree/colt_{snap:03d}.hdf5'
            states_file = f'{data_dir}/{sim}/ics_tree/states-no-UVB_{snap:03d}.hdf5'

            if not (os.path.exists(colt_file) and os.path.exists(states_file)):
                print(f"Skipping snapshot {snap}: missing files.")
                continue

            try:
                with h5py.File(colt_file, 'r') as c, h5py.File(states_file, 'r') as s:
                    zs[i] = c.attrs['redshift']
                    rbox = c.attrs['r_box']
                    rvir = rbox / 4.0
                    r = c['r'][:]
                    mask = np.sum(r*r, axis=1) < rvir*rvir

                    V = c['V'][mask]
                    rho = c['rho'][mask]
                    mass = rho * V

                    # element mass fraction
                    if element == 'H':
                        Z_elem = c['X'][mask]
                    elif element == 'He':
                        Z_elem = c['Y'][mask]
                    else:
                        Z_elem = c[f'Z_{element}'][mask]

                    n_elem = mass * Z_elem
                    n_elem_sum = np.sum(n_elem)
                    V_sum = np.sum(V)

                    # compute MW and VW for each ionic state
                    for j, st in enumerate(states):
                        ion_frac = s[f'x_{st}'][mask]

                        mw = np.sum(ion_frac * n_elem) / n_elem_sum if n_elem_sum > 0 else np.nan
                        vw = np.sum(ion_frac * V) / V_sum if V_sum > 0 else np.nan

                        mw_arr[i, j] = mw
                        vw_arr[i, j] = vw

                success[i] = True

            except Exception as e:
                print(f"Snapshot {snap} failed: {e}")

        # --- filter only successful snapshots ---
        n_success = np.count_nonzero(success)
        if n_success == 0:
            print(f"No snapshots found for {data_dir} ...")
            continue
        if n_success < n_snaps:
            print(f'Only {n_success} out of {n_snaps} snapshots found for {data_dir} ...')
            snaps_filtered = snaps[success]
            zs_filtered = zs[success]
            mw_arr = mw_arr[success, :]
            vw_arr = vw_arr[success, :]
        else:
            snaps_filtered = snaps
            zs_filtered = zs

        # store arrays in dictionaries for final saving
        mw_dict[element] = mw_arr
        vw_dict[element] = vw_arr
        zs_global = zs_filtered  # overwrite global zs, assuming same for all elements

    # --- write summary HDF5 ---
    os.makedirs(f'{out_dir}', exist_ok=True)
    with h5py.File(f'{out_dir}/Metals.hdf5', 'w') as f:
        f.create_dataset('snaps', data=snaps_filtered)
        f.create_dataset('zs', data=zs_global)

        for element, states in state_dict.items():
            g = f.create_group(element)
            g.create_dataset('MW', data=mw_dict[element])
            g.create_dataset('VW', data=vw_dict[element])
            g.create_dataset('states', data=np.array(states, dtype='S'))

    print("\nSummary Metals.hdf5 file created successfully!")

def repack_mass(snaps, sim='g500531/z16', out_dir='.'):
    # --- paths and simulation lists ---
    path_split = {
        'D4': '/orcd/data/mvogelsb/004/Thesan-Zooms-COLT',
        'D1': '/nfs/mvogelsblab001/Lab/Thesan-Zooms-COLT'
    }
    D1_sim = ['g10304/z8','g1163/z4','g137030/z16','g2/z4','g205/z4',
              'g2274036/z16','g39/z4','g500531/z16','g519761/z16',
              'g5760/z8','g578/z4']
    SOLAR_MASS = 1.989e33  # Solar masses
    snaps = np.array(snaps, dtype=np.int32)
    n_snaps_total = len(snaps)
    success = np.zeros(n_snaps_total, dtype=bool)
    f_vir = 4.
    group_data = []
    z_group = []
    z_arr = np.empty(n_snaps_total)
    m_star_arr = np.zeros(n_snaps_total)
    m_gas_arr = np.zeros(n_snaps_total)
    m_dm_arr = np.zeros(n_snaps_total)
    m_p2_arr = np.zeros(n_snaps_total)
    m_p3_arr = np.zeros(n_snaps_total)
    m_halo_arr = np.zeros(n_snaps_total)
    m_dm_p2_p3_arr = np.zeros(n_snaps_total)
    r_HRs = np.empty([n_snaps_total, 3]) # High-resolution center of mass positions [cm]
    R_virs = np.empty([n_snaps_total, 3]) # Group positions [cm]
    data_dir = path_split['D1'] if sim in D1_sim else path_split['D4']
    # with h5py.File(f'{data_dir}/{sim}/ics_tree/center.hdf5', 'r') as f:
    #     TargetPos = f['Smoothed/TargetPos'][:]
    with h5py.File(f'ics_tree/{sim}/center.hdf5', 'r') as f:
        TargetPos = f['Smoothed/TargetPos'][:]
    j = 0
    for i in progressbar(range(n_snaps_total)):
        snap = snaps[i]
        # Derived global variables
        cand_dir = f'/orcd/data/mvogelsb/004/Thesan-Zooms/{sim}/postprocessing/candidates'
        colt_file = f'{data_dir}/{sim}/ics_tree/colt_{snap:03d}.hdf5'

        if not os.path.exists(colt_file):
            print(f"Skipping snapshot {snap}: missing files.")
            continue

        ## Open the colt from ics_tree directory for gas and stars
        try:
            with h5py.File(colt_file, "r") as f:
                z_arr[i] = f.attrs['redshift']
                rbox = f.attrs['r_box']
                r_virs = rbox / 4.0
                r = f['r'][:]

                ## Stars ##
                if 'r_star' in f.keys():
                    pos_star = f['r_star'][:]
                    vir_mask = (np.linalg.norm(pos_star, axis=1) < r_virs)
                    m_star = f['m_init_star'][:]  # Msun
                    m_star_mask = m_star[vir_mask]
                    m_star_arr[i] = np.sum(m_star_mask)
                else:
                    m_star_arr[i] = 0.0
                ## Gas ##
                n_cells = f.attrs['n_cells'] # Number of cells
                pos_gas = f['r'][:]
                vir_mask = (np.linalg.norm(pos_gas, axis=1) < r_virs)
                gas_rho = f['rho'][:]
                gas_vol = f['V'][:]
                inner_edges = f['inner_edges'][:]
                edges = f['edges'][:]
                mask = np.zeros(n_cells, dtype=bool)
                mask[edges] = True
                mask[inner_edges] = True
                gas_rho[mask] = 0.  # Set gas density to 0 outside the high-resolution region
                gas_mass = gas_rho * gas_vol / SOLAR_MASS   ## finding mass and converting to Msun
                m_gas_mask = gas_mass[vir_mask]
                m_gas_arr[i] = np.sum(m_gas_mask)
            
            cand_file = f'{cand_dir}/candidates_{snap:03d}.hdf5'
            with h5py.File(cand_file, 'r') as f:
                # print('Cand_file Open')
                header = f['Header'].attrs
                a = header['Time']
                z = 1. / a - 1.
                h = header['HubbleParam']
                UnitLength_in_cm = header['UnitLength_in_cm']
                length_to_cgs = a * UnitLength_in_cm / h
                cand_group_ids = f['Group']['GroupID'][:] # Group ID in the candidates
                cand_subhalo_ids = f['Subhalo']['SubhaloID'][:] # Subhalo ID in the candidates
                r_HRs[i] = length_to_cgs * header['PosHR'] # High-resolution center of mass position [cm]
                R_virs[i] = length_to_cgs * TargetPos[j] - r_HRs[i] # Group position relative to high-resolution center of mass [cm]
                j+=1
            
            dm_filename = f"{data_dir}/{sim}/ics/dm_{snap:03d}.hdf5"
            # Open the dm files  from ics directory
            with h5py.File(dm_filename, "r") as d:
                # print('DM file open')
                if 'r_p2' in d:
                    # Read coordinates for the radial masks
                    r_p2 = d['r_p2'][:] - R_virs[i] # Dark matter particle positions Type 2 [cm]
                    p2_mask = (np.linalg.norm(r_p2, axis=1) < r_virs) # Sphere cut
                    m_p2 = d['m_p2'][p2_mask]  # Mass of dark matter particles Type 2 [MSun]
                    m_p2_arr[i] = np.sum(m_p2)

                if 'r_p3' in d:
                    r_p3 = d['r_p3'][:] - R_virs[i]  # Dark matter particle positions Type 3 [cm]
                    p3_mask = (np.linalg.norm(r_p3, axis=1) < r_virs) # Sphere cut
                    r_p3 = r_p3[p3_mask]  # Dark matter particle positions Type 3 within 2 virial radius[cm]
                    m_p3 = d.attrs['m_p3']
                    n_p3_tot = d.attrs['n_p3']  # Total number of dark matter particles Type 3 [MSun]
                    m_p3 = m_p3 * np.ones(n_p3_tot)
                    m_p3 = m_p3[p3_mask]  # Mass of dark matter particles Type 3 [MSun]
                    m_p3_arr[i] = np.sum(m_p3)
                if 'r_dm' in d:
                    n_dm_tot = d.attrs['n_dm']  # Total number of dark matter particles
                    m_dm = d.attrs['m_dm']  # Mass of dark matter particles [MSun]
                    r_dm = d['r_dm'][:] - R_virs[i]  # Dark matter particle positions [cm]
                    dm_mask = (np.linalg.norm(r_dm, axis=1) < r_virs) # Sphere cut
                    r_dm = r_dm[dm_mask]  # Dark matter particle positions
                    m_dm = m_dm * np.ones(n_dm_tot)  # Total dark matter mass [MSun]
                    m_dm = m_dm[dm_mask]  # Dark matter mass [MSun]
                    m_dm_arr[i] = np.sum(m_dm)
            m_dm_p2_p3_arr[i] = m_dm_arr[i] + m_p2_arr[i] + m_p3_arr[i]
            success[i] = True
            m_halo_arr[i] = m_dm_p2_p3_arr[i] + m_star_arr[i] + m_gas_arr[i]
        except Exception as e:
            print(f"Snapshot {snap} failed: {e}")
        
    # --- filter only successful snapshots ---
    n_success = np.count_nonzero(success)
    if n_success == 0:
        print(f"No snapshots found for {data_dir} ...")
    if n_success < n_snaps_total:
        print(f'Only {n_success} out of {n_snaps_total} snapshots found for {data_dir} ...')
        snaps_filtered = snaps[success]
        zs_filtered = z_arr[success]
        m_gas_arr_filtered = m_gas_arr[success]
        m_star_arr_filtered = m_star_arr[success]
        m_dm_p2_p3_arr_filtered = m_dm_p2_p3_arr[success]
        m_halo_arr_filtered = m_halo_arr[success]
    else:
        snaps_filtered = snaps
        zs_filtered = z_arr
        m_gas_arr_filtered = m_gas_arr
        m_star_arr_filtered = m_star_arr
        m_dm_p2_p3_arr_filtered = m_dm_p2_p3_arr
        m_halo_arr_filtered = m_halo_arr

    # --- write summary HDF5 ---
    os.makedirs(f'{out_dir}', exist_ok=True)
    with h5py.File(f'{out_dir}/Mass.hdf5', 'w') as f:
        f.create_dataset('snaps', data=snaps_filtered)
        f.create_dataset('zs', data=zs_filtered)
        f.create_dataset('M_gas', data=m_gas_arr_filtered)
        f.create_dataset('M_star', data=m_star_arr_filtered)
        f.create_dataset('M_dm_p2_p3', data=m_dm_p2_p3_arr_filtered)
        f.create_dataset('M_halo', data=m_halo_arr_filtered)

    print("\nSummary Mass.hdf5 file created successfully!")

def repack_SFR(snaps, sim='g500531/z16', out_dir='.'):
    # --- paths and simulation lists ---
    path_split = {
        'D4': '/orcd/data/mvogelsb/004/Thesan-Zooms-COLT',
        'D1': '/nfs/mvogelsblab001/Lab/Thesan-Zooms-COLT'
    }
    D1_sim = ['g10304/z8','g1163/z4','g137030/z16','g2/z4','g205/z4',
              'g2274036/z16','g39/z4','g500531/z16','g519761/z16',
              'g5760/z8','g578/z4']

    snaps = np.array(snaps, dtype=np.int32)
    n_snaps_total = len(snaps)
    success = np.zeros(n_snaps_total, dtype=bool)
    f_vir = 4.
    z_group = []
    z_arr = np.empty(n_snaps_total)
    sfr_5_arr = np.zeros(n_snaps_total)
    sfr_10_arr = np.zeros(n_snaps_total)
    sfr_20_arr = np.zeros(n_snaps_total)
    sfr_30_arr = np.zeros(n_snaps_total)
    sfr_50_arr = np.zeros(n_snaps_total)
    sfr_75_arr = np.zeros(n_snaps_total)
    sfr_100_arr = np.zeros(n_snaps_total)
    data_dir = path_split['D1'] if sim in D1_sim else path_split['D4']
    
    # with h5py.File(f'{data_dir}/{sim}/ics_tree/center.hdf5', 'r') as f:
    #     TargetPos = f['Smoothed/TargetPos'][:]
    for i in progressbar(range(n_snaps_total)):
        snap = snaps[i]
        # Derived global variables
        colt_file = f'{data_dir}/{sim}/ics_tree/colt_{snap:03d}.hdf5'
        if not os.path.exists(colt_file):
            print(f"Skipping snapshot {snap}: missing files.")
            continue
        ## Open the colt from ics_tree directory for gas and stars
        try:
            with h5py.File(colt_file, "r") as f:
                z_arr[i] = f.attrs['redshift']
                rbox = f.attrs['r_box']
                r_virs = rbox / 4.
                ## Stars ##
                if 'r_star' in f.keys():
                    pos = f['r_star'][:]
                    vir_mask = (np.linalg.norm(pos, axis=1) < r_virs)
                    m_star = f['m_init_star'][:]  # Msun
                    age_star = f['age_star'][:]*1e3  ## in Myr
                    sfr_5 = m_star[(age_star < 5) & vir_mask]
                    sfr_10 = m_star[(age_star < 10) & vir_mask]
                    sfr_20 = m_star[(age_star < 20) & vir_mask]
                    sfr_30 = m_star[(age_star < 30) & vir_mask]
                    sfr_50 = m_star[(age_star < 50) & vir_mask]
                    sfr_75 = m_star[(age_star < 75) & vir_mask]
                    sfr_100 = m_star[(age_star < 100) & vir_mask]
                    # calculating in Msun/yr
                    sfr_5_arr[i] = np.sum(sfr_5) / (5e6)
                    sfr_10_arr[i] = np.sum(sfr_10) / (10e6)
                    sfr_20_arr[i] = np.sum(sfr_20) / (20e6)
                    sfr_30_arr[i] = np.sum(sfr_30) / (30e6)
                    sfr_50_arr[i] = np.sum(sfr_50) / (50e6)
                    sfr_75_arr[i] = np.sum(sfr_75) / (75e6)
                    sfr_100_arr[i] = np.sum(sfr_100) / (100e6)
                    success[i] = True
        except Exception as e:
            print(f"Snapshot {snap} failed: {e}")
        
    # --- filter only successful snapshots ---
    n_success = np.count_nonzero(success)
    if n_success == 0:
        print(f"No snapshots found for {data_dir} ...")
    if n_success < n_snaps_total:
        print(f'Only {n_success} out of {n_snaps_total} snapshots found for {data_dir} ...')
        snaps_filtered = snaps[success]
        zs_filtered = z_arr[success]
        sfr_5_filter = sfr_5_arr[success]
        sfr_10_filter = sfr_10_arr[success]
        sfr_20_filter = sfr_20_arr[success]
        sfr_30_filter = sfr_30_arr[success]
        sfr_50_filter = sfr_50_arr[success]
        sfr_75_filter = sfr_75_arr[success]
        sfr_100_filter = sfr_100_arr[success]
    else:
        snaps_filtered = snaps
        zs_filtered = z_arr
        sfr_5_filter = sfr_5_arr
        sfr_10_filter = sfr_10_arr
        sfr_20_filter = sfr_20_arr
        sfr_30_filter = sfr_30_arr
        sfr_50_filter = sfr_50_arr
        sfr_75_filter = sfr_75_arr
        sfr_100_filter = sfr_100_arr

    # --- write summary HDF5 ---

    os.makedirs(f'{out_dir}', exist_ok=True)

    if os.path.exists(f'{out_dir}/SFR.hdf5'):
        with h5py.File(f'{out_dir}/SFR.hdf5', 'r+') as f:
            f.create_dataset('SFR_30', data=sfr_30_filter)
    else:
        with h5py.File(f'{out_dir}/SFR.hdf5', 'w') as f:
            f.create_dataset('snaps', data=snaps_filtered)
            f.create_dataset('zs', data=zs_filtered)
            f.create_dataset('SFR_5', data=sfr_5_filter)
            f.create_dataset('SFR_10', data=sfr_10_filter)
            f.create_dataset('SFR_20', data=sfr_20_filter)
            f.create_dataset('SFR_50', data=sfr_50_filter)
            f.create_dataset('SFR_75', data=sfr_75_filter)
            f.create_dataset('SFR_100', data=sfr_100_filter)

        print("\nSummary SFR.hdf5 file created successfully!")


def repack_O32(snaps, data_dir='.', out_dir='.'):
    n_snaps = len(snaps)
    success = np.zeros(n_snaps, dtype=bool)
    n_cameras = 8
    zs = np.empty(n_snaps)
    L_O2 = np.empty(n_snaps)
    L_O3 = np.empty(n_snaps)
    Ndot_O2 = np.empty(n_snaps)
    Ndot_O3 = np.empty(n_snaps)
    f_esc_O2 = np.empty(n_snaps)
    f_esc_O3 = np.empty(n_snaps)
    f_escs_O2 = np.empty((n_snaps, n_cameras))
    f_escs_O3 = np.empty((n_snaps, n_cameras))
    freq_avg_O2 = np.empty(n_snaps)
    freq_std_O2 = np.empty(n_snaps)
    freq_avg_O3 = np.empty(n_snaps)
    freq_std_O3 = np.empty(n_snaps)
    nside_base = 5
    nside_LyC = 10
    map_O2 = np.empty((n_snaps, 12*(nside_base)**2))
    map_O3 = np.empty((n_snaps, 12*(nside_base)**2))
    map_LyC = np.empty((n_snaps, 12*(nside_LyC)**2))
    map_1500 = np.empty((n_snaps, 12*(nside_LyC)**2))
    # freq_seps = np.empty((n_snaps, n_cameras))
    freq_avgs_O2 = np.empty((n_snaps, n_cameras))
    freq_stds_O2 = np.empty((n_snaps, n_cameras))
    freq_avgs_O3 = np.empty((n_snaps, n_cameras))
    freq_stds_O3 = np.empty((n_snaps, n_cameras))
    M_1500 = np.empty(n_snaps)
    M_1500_obs = np.empty((n_snaps, n_cameras))
    Ndot_LyC = np.empty(n_snaps)
    f_esc_LyC = np.empty(n_snaps)
    f_escs_LyC = np.empty((n_snaps, n_cameras))
    for i in progressbar(range(n_snaps)):
        try:
            # print(f'{data_dir}/{base}/{base}_{snaps[i]:03d}.hdf5')
            with h5py.File(f'{data_dir}/OII-3727-3730/OII-3727-3730_{snaps[i]:03d}.hdf5', 'r') as f:
                zs[i] = f.attrs['z']
                L_O2[i] = f.attrs['L_tot']
                Ndot_O2[i] = f.attrs['Ndot_tot']
                E0_O2 = f['line'].attrs['E0']
                # print(E0)
                map_O2[i,:] = f['map'][:]
                f_esc_O2[i] = f.attrs['f_esc']
                f_escs_O2[i,:] = f['f_escs'][:]
                freq_avg_O2[i] = f.attrs['freq_avg']
                freq_std_O2[i] = f.attrs['freq_std']
                freq_avgs_O2[i,:] = f['freq_avgs'][:]
                freq_stds_O2[i,:] = f['freq_stds'][:]
                fluxes_O2 = f['fluxes'][:]
                n_bins_O2 = f.attrs['n_bins']
                freq_max_O2 = f.attrs['freq_max']
                freq_min_O2 = f.attrs['freq_min']
                freq_edges_O2 = np.linspace(freq_min_O2, freq_max_O2, n_bins_O2+1)
                freqs_O2 = 0.5 * (freq_edges_O2[:-1] + freq_edges_O2[1:])
                # red = (freqs > 0.)
                # for j in range(n_cameras):
                #     freq_seps[i,j] = freqs[red][np.argmax(fluxes[j,red])] - freqs[~red][np.argmax(fluxes[j,~red])]
                flux_O2 = f['flux_avg'][:]
                # freq_sep[i] = freqs[red][np.argmax(flux[red])] - freqs[~red][np.argmax(flux[~red])]
            with h5py.File(f'{data_dir}/OIII-5008/OIII-5008_{snaps[i]:03d}.hdf5', 'r') as f:
                zs[i] = f.attrs['z']
                L_O3[i] = f.attrs['L_tot']
                Ndot_O3[i] = f.attrs['Ndot_tot']
                E0_O3 = f['line'].attrs['E0']
                map_O3[i,:] = f['map'][:]
                f_esc_O3[i] = f.attrs['f_esc']
                f_escs_O3[i,:] = f['f_escs'][:]
                freq_avg_O3[i] = f.attrs['freq_avg']
                freq_std_O3[i] = f.attrs['freq_std']
                freq_avgs_O3[i,:] = f['freq_avgs'][:]
                freq_stds_O3[i,:] = f['freq_stds'][:]
                fluxes_O3 = f['fluxes'][:]
                n_bins_O3 = f.attrs['n_bins']
                freq_max_O3 = f.attrs['freq_max']
                freq_min_O3 = f.attrs['freq_min']
                freq_edges_O3 = np.linspace(freq_min_O3, freq_max_O3, n_bins_O3+1)
                freqs_O3 = 0.5 * (freq_edges_O3[:-1] + freq_edges_O3[1:])
                # red = (freqs > 0.)
                # for j in range(n_cameras):
                #     freq_seps[i,j] = freqs[red][np.argmax(fluxes[j,red])] - freqs[~red][np.argmax(fluxes[j,~red])]
                flux_O3 = f['flux_avg'][:]
                # freq_sep[i] = freqs[red][np.argmax(flux[red])] - freqs[~red][np.argmax(flux[~red])]
            with h5py.File(f'{data_dir}/ion-eq/ion-eq_{snaps[i]:03d}.hdf5', 'r') as f:
                Ndot_LyC[i] = f.attrs['Ndot_tot']
                f_esc_LyC[i] = f.attrs['f_esc']
                map_LyC[i,:] = f['map'][:]
                f_escs_LyC[i,:] = f['f_escs'][:]
            with h5py.File(f'{data_dir}/M1500/M1500_{snaps[i]:03d}.hdf5', 'r') as f:
                Ndot_1500 = f.attrs['Ndot_tot']  # Photon rate [photons/s]
                edges_eV = f['bin']['edges_eV'][:] # Energy bin edges [eV]
                map_1500[i,:] = f['map'][:]
                eV = 1.60217725e-12              # Electron volt: 1 eV = 1.6e-12 erg
                angstrom = 1e-8                  # Units: 1 angstrom = 1e-8 cm
                c = 2.99792458e10                # Speed of light [cm/s]
                h = 6.626069573e-27              # Planck's constant [erg s]
                lambda_1500 = 1500. * angstrom  # Continuum wavelength [cm]
                nu_1500 = c / lambda_1500       # Continuum frequency [Hz]
                Delta_lambda = h * c * (1./edges_eV[0] - 1./edges_eV[1]) / (eV * angstrom)  # Bin width [angstrom]
                L_1500 = h * nu_1500 * Ndot_1500 / Delta_lambda  # Spectral luminosity [erg/s/angstrom]
                L_1500_obs = L_1500 * f['f_escs'][:]
                pc = 3.085677581467192e18       # Units: 1 pc  = 3e18 cm
                R_10pc = 10. * pc               # Reference distance for continuum [cm]
                fnu_1500_fac = lambda_1500**2 / (4. * np.pi * c * R_10pc**2 * angstrom)
                M_1500[i] = -2.5 * np.log10(fnu_1500_fac * L_1500) - 48.6 # Continuum absolute magnitude
                M_1500_obs[i,:] = -2.5 * np.log10(fnu_1500_fac * L_1500_obs) - 48.6
            success[i] = True
        except Exception as e:
            print(e)
    
    snaps = np.array(snaps, dtype=np.int32)
    n_success = np.count_nonzero(success)
    if n_success < n_snaps:
        if n_success == 0:
            print(f'No snapshots found for {data_dir} ...')
            return
        print(f'Only {n_success} out of {n_snaps} snapshots found for {data_dir} ...')
        n_snaps = np.int32(n_success)
        snaps = snaps[success]
        zs = zs[success]
        L_O2 = L_O2[success]
        L_O3 = L_O3[success]
        Ndot_O2 = Ndot_O2[success]
        Ndot_O3 = Ndot_O3[success]
        f_esc_O2 = f_esc_O2[success]
        f_escs_O2 = f_escs_O2[success,:]
        f_esc_O3 = f_esc_O3[success]
        f_escs_O3 = f_escs_O3[success,:]
        # freq_sep = freq_sep[success]
        freq_avg_O2 = freq_avg_O2[success]
        freq_std_O2 = freq_std_O2[success]
        freq_avg_O3 = freq_avg_O3[success]
        freq_std_O3 = freq_std_O3[success]
        # freq_seps = freq_seps[success,:]
        freq_avgs_O2 = freq_avgs_O2[success,:]
        freq_avgs_O3 = freq_avgs_O3[success,:]
        map_O2 = map_O2[success,:]
        map_O3 = map_O3[success,:]
        map_LyC = map_LyC[success,:]
        map_1500 = map_1500[success,:]
        freq_stds_O2 = freq_stds_O2[success,:]
        freq_stds_O3 = freq_stds_O3[success,:]
        M_1500 = M_1500[success]
        M_1500_obs = M_1500_obs[success,:]
        Ndot_LyC = Ndot_LyC[success]
        f_esc_LyC = f_esc_LyC[success]
        f_escs_LyC = f_escs_LyC[success,:]
    os.makedirs(f'{out_dir}/O32', exist_ok=True)
    if os.path.exists(f'{out_dir}/O32/O32.hdf5'):
        mode = 'r+'
        ## If you want to add new field to the summary file make sure it exists at the path
        with h5py.File(f'{out_dir}/O32/O32.hdf5', mode) as f:
            f.create_dataset('map_base', data=map_base)
            f.create_dataset('map_LyC', data=map_LyC)
            f.create_dataset('map_1500', data=map_1500)
    else:
        mode = 'w'
        ## Base summary file that was made
        with h5py.File(f'{out_dir}/O32/O32.hdf5', mode) as f:
            f.attrs['n_snaps'] = n_snaps
            f.attrs['n_cameras'] = n_cameras
            f.create_dataset('snaps', data=snaps)
            f.create_dataset('zs', data=zs)
            f.create_dataset('Ndot_O2', data=Ndot_O2)
            f.create_dataset('Ndot_O3', data=Ndot_O3)
            f.create_dataset('L_O2', data=L_O2)
            f.create_dataset('L_O3', data=L_O3)
            f.create_dataset('f_esc_O2', data=f_esc_O2)
            f.create_dataset('f_escs_O2', data=f_escs_O2)
            f.create_dataset('f_esc_O3', data=f_esc_O3)
            f.create_dataset('f_escs_O3', data=f_escs_O3)
            # f.create_dataset('freq_sep', data=freq_sep)
            f.create_dataset('freq_avg_O2', data=freq_avg_O2)
            f.create_dataset('freq_std_O2', data=freq_std_O2)
            f.create_dataset('freq_avg_O3', data=freq_avg_O3)
            f.create_dataset('freq_std_O3', data=freq_std_O3)
            # f.create_dataset('freq_seps', data=freq_seps)
            f.create_dataset('freq_avgs_O2', data=freq_avgs_O2)
            f.create_dataset('freq_stds_O2', data=freq_stds_O2)
            f.create_dataset('freq_avgs_O3', data=freq_avgs_O3)
            f.create_dataset('freq_stds_O3', data=freq_stds_O3)
            f.create_dataset('M_1500', data=M_1500)
            f.create_dataset('M_1500_obs', data=M_1500_obs)
            f.create_dataset('Ndot_LyC', data=Ndot_LyC)
            f.create_dataset('f_esc_LyC', data=f_esc_LyC)
            f.create_dataset('f_escs_LyC', data=f_escs_LyC)
            f.create_dataset('map_O2', data=map_O2)
            f.create_dataset('map_O3', data=map_O3)
            f.create_dataset('map_LyC', data=map_LyC)
            f.create_dataset('map_1500', data=map_1500)


def plot_delta_MS_evolution(t, out_dir='.', plots=False):
    # Parameters for sSFR fit: [sb, beta, mu]  ## Will's paper
    param_dict = {
        10:  [0.033, 0.041, 2.64], 
        30:  [0.037, 0.042, 2.57],
        50:  [0.043, 0.041, 2.47],
        100: [0.067, 0.032, 2.19],
        'Ha': [0.048, 0.077, 2.60],
        'UV': [0.051, 0.022, 2.32]
    }
    
    if t not in param_dict:
        print(f"Error: Timescale {t} not found in param_dict.")
        return

    # --- 1. Load Data ---
    SFR_label = f'SFR_{t}'
    with h5py.File(f'{out_dir}/SFR.hdf5', 'r') as f:
        snaps_sfr = f['snaps'][:]
        SFR_calc_raw = f[SFR_label][:]

    with h5py.File(f'{out_dir}/Mass.hdf5', 'r') as f:
        snaps_mass = f['snaps'][:] 
        M_star_raw = f['M_star'][:]
        zs_raw = f['zs'][:]

    with h5py.File(f'{out_dir}/Ha/Ha.hdf5', 'r') as f:
        snaps_ha = f['snaps'][:]
        L_tot_Ha = f['L_tot'][:]
        SFR_calc_Ha_raw = L_tot_Ha * 10**(-41.45)
    
    with h5py.File(f'{out_dir}/M1500/M1500.hdf5', 'r') as f:
        snaps_uv = f['snaps'][:]
        L_tot_UV = f['L_1500'][:]
        # Note: Multiplying by 1500 assumes L_1500 is L_lambda to get lambda*L_lambda (~ nu*L_nu)
        SFR_calc_UV_raw = (L_tot_UV * 1500) * 10**(-43.53) 
        
    # --- 2. Align Data (CRITICAL FIX) ---
    common_snaps = reduce(np.intersect1d, (snaps_sfr, snaps_mass, snaps_ha, snaps_uv))
    
    idx_sfr  = np.isin(snaps_sfr, common_snaps)
    idx_mass = np.isin(snaps_mass, common_snaps)
    idx_ha   = np.isin(snaps_ha, common_snaps)
    idx_uv   = np.isin(snaps_uv, common_snaps)
    
    SFR_calc = SFR_calc_raw[idx_sfr]
    M_star = M_star_raw[idx_mass]
    zs = zs_raw[idx_mass]
    SFR_calc_Ha = SFR_calc_Ha_raw[idx_ha]
    SFR_calc_UV = SFR_calc_UV_raw[idx_uv]

    # --- 3. Calculate Cosmic Time and Sort (CRITICAL FIX) ---
    ages = cosmo.age(zs).to(u.Myr).value 
    
    # Sort chronologically (Time increasing, Redshift decreasing)
    sort_idx = np.argsort(ages)
    
    time_myr = ages[sort_idx]
    zs_sorted = zs[sort_idx]
    M_star = M_star[sort_idx]
    
    # MUST sort all SFR arrays so they match the sorted time/mass arrays!
    SFR_calc = SFR_calc[sort_idx]
    SFR_calc_Ha = SFR_calc_Ha[sort_idx]
    SFR_calc_UV = SFR_calc_UV[sort_idx]

    # --- 4. Calculate Delta MS ---
    sb, beta, mu = param_dict[t]
    sb_Ha, beta_Ha, mu_Ha = param_dict['Ha']
    sb_UV, beta_UV, mu_UV = param_dict['UV']
    
    # Calculate sSFR and then SFR_MS (using zs_sorted for ALL of them)
    sSFR_fit = sb * (M_star/ 1e10)**beta * (1 + zs_sorted)**mu
    sSFR_fit_Ha = sb_Ha * (M_star / 1e10)**beta_Ha * (1 + zs_sorted)**mu_Ha
    sSFR_fit_UV = sb_UV * (M_star / 1e10)**beta_UV * (1 + zs_sorted)**mu_UV
    
    SFR_MS = sSFR_fit * (M_star / 1e9) 
    SFR_MS_Ha = sSFR_fit_Ha * (M_star / 1e9) 
    SFR_MS_UV = sSFR_fit_UV * (M_star / 1e9) 
    
    # Delta MS is the log difference
    delta_MS = np.log10(SFR_calc + 1e-10) - np.log10(SFR_MS + 1e-10)
    delta_MS_UV = np.log10(SFR_calc_UV + 1e-10) - np.log10(SFR_MS_UV + 1e-10)
    delta_MS_Ha = np.log10(SFR_calc_Ha + 1e-10) - np.log10(SFR_MS_Ha + 1e-10)

    MS_dir = f'{out_dir}/new_SFRMS'
    plot_dir = f'{MS_dir}/Timescale_{t}'
    os.makedirs(plot_dir, exist_ok=True)
    with h5py.File(f'{out_dir}/SFR.hdf5', 'a') as f: 
        # f.attrs['units'] = 'Msun/yr'
        if f'Timescale_{t}' in f:
            del f[f'Timescale_{t}']
        w = f.require_group(f'Timescale_{t}')
        if f'SFR_MS_{t}' in f.keys():
            del w[f'SFR_MS_{t}']
            del w[f'Delta_SFR_MS_{t}']
        if f'SFR_MS_Ha' in f.keys():
            del w[f'SFR_MS_Ha']
            del w[f'Delta_SFR_MS_Ha']
        if f'SFR_MS_UV' in f.keys():
            del w[f'SFR_MS_UV']
            del w[f'Delta_SFR_MS_UV']
 
        w.create_dataset(f'SFR_MS_{t}', data=SFR_MS)
        w.create_dataset(f'zs', data = zs)
        w.create_dataset(f'Delta_SFR_MS_{t}', data=delta_MS)
        w.create_dataset(f'SFR_MS_Ha', data=SFR_MS_Ha)
        w.create_dataset(f'Delta_SFR_MS_Ha', data=delta_MS_Ha)
        w.create_dataset(f'SFR_MS_UV', data=SFR_MS_UV)
        w.create_dataset(f'Delta_SFR_MS_UV', data=delta_MS_UV)
    if plots:
        # --- 5. Plotting ---
        fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
        
        ax1.plot(time_myr, delta_MS, '-', color='royalblue', linewidth=1.5, alpha=0.8, label=r'$\Delta$MS')
        ax1.plot(time_myr, delta_MS_UV, '-', color='darkviolet', linewidth=1.5, alpha=0.8, label=r'$\Delta$MS (UV)')
        ax1.plot(time_myr, delta_MS_Ha, '-', color='forestgreen', linewidth=1.5, alpha=0.8, label=r'$\Delta$MS (H$\alpha$)')
        
        ax1.set_xlim([250, 2000])
        
        ax1.axhline(0, color='k', linestyle='--', linewidth=2, label='Main Sequence')
        ax1.axhline(0.5, color='crimson', linestyle=':', alpha=0.6, label='Starburst (+0.5 dex)')
        ax1.axhline(-0.5, color='orange', linestyle=':', alpha=0.6, label='Valley (-0.5 dex)')

        ax1.set_xlabel('Cosmic Time (Myr)', fontsize=15)
        ax1.set_ylabel(r'$\Delta$ MS (dex)', fontsize=15)
        ax1.set_title(f'Distance from Main Sequence over Time (t = {t} Myr)', pad=15)
        
        ax1.set_xscale('linear')
        ax1.grid(True, alpha=0.3)
        
        # Move legend out of the way of the data lines
        ax1.legend(loc='lower right', fontsize=11, ncol=2)
        
        # --- 6. Create Secondary Axis (Top = Redshift) ---
        ax2 = ax1.twiny()
        
        z_ticks = np.array([15, 12, 10, 8, 7, 6, 5, 4, 3])
        t_ticks = cosmo.age(z_ticks).to(u.Myr).value
        
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xscale('linear') 
        
        ax2.set_xticks(t_ticks)
        ax2.set_xticklabels([f'{z}' for z in z_ticks], fontsize=13)
        ax2.set_xlabel('Redshift', fontsize=15, labelpad=10)

        fig.tight_layout()
        plt.savefig(f'{plot_dir}/Delta_MS_evolution_{t}.png')
        plt.close()
        
        print(f"Saved: {plot_dir}/Delta_MS_evolution_{t}.png")


# sims = {'g39': ['z4']}           ## Comment this line to process all simulations otherwise only g39 z4 (Used for Test runs)
for sim, res_list in sims.items():
    for res in res_list:
        group = f'{sim}/{res}'
        print(f'Processing {group}...')
        data_dir = f'/orcd/data/mvogelsb/004/Thesan-Zooms/{group}/postprocessing/colt_tree'
        out_dir = f'/orcd/home/002/parth999/work/summary/{group}'
        # repack_metals(snaps=range(0,188+1), sim=group, out_dir=out_dir)
        # repack_SFR(snaps=range(0,188+1), sim=group, out_dir=out_dir)
        # repack_mass(snaps=range(0,188+1), sim=group, out_dir=out_dir)
        # repack_1500(snaps=range(0,188+1), data_dir=data_dir, out_dir=out_dir)
        # repack_base(snaps=range(0,188+1), data_dir=data_dir, out_dir=out_dir, base='Ha',  RHD=True)
        # repack_O32(snaps=range(0,188+1), data_dir=data_dir, out_dir=out_dir)
        # repack_base(snaps=range(0,188+1), data_dir=data_dir, out_dir=out_dir, base='Lya')
        # repack_base(snaps=range(0,188+1), data_dir=data_dir, out_dir=out_dir, base='Hb')
        # repack_cont(snaps=range(0,188+1), data_dir=data_dir, out_dir=out_dir, base'Lya-cont')
        # repack_cont(snaps=range(0,188+1), data_dir=data_dir, out_dir=out_dir, base='Ha-cont')

        # Block to calculate and save delta MS at various timescales
        for i in [10,30,50,100]:
            plot_delta_MS_evolution(t=i, out_dir = out_dir, plots=False)