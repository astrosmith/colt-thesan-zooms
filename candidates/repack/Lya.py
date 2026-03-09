import numpy as np
import h5py, sys

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

def repack(snaps, out_dir='.'):
    n_snaps = len(snaps)
    success = np.zeros(n_snaps, dtype=bool)
    n_cameras = 4
    zs = np.empty(n_snaps)
    L_tot = np.empty(n_snaps)
    f_esc = np.empty(n_snaps)
    f_escs = np.empty((n_snaps, n_cameras))
    freq_sep = np.empty(n_snaps)
    freq_avg = np.empty(n_snaps)
    freq_std = np.empty(n_snaps)
    freq_seps = np.empty((n_snaps, n_cameras))
    freq_avgs = np.empty((n_snaps, n_cameras))
    freq_stds = np.empty((n_snaps, n_cameras))
    M_1500 = np.empty(n_snaps)
    M_1500_obs = np.empty((n_snaps, n_cameras))
    Ndot_LyC = np.empty(n_snaps)
    f_esc_LyC = np.empty(n_snaps)
    f_escs_LyC = np.empty((n_snaps, n_cameras))
    for i in progressbar(range(n_snaps)):
        try:
            with h5py.File(f'{out_dir}/Lya_{snaps[i]:03d}.hdf5', 'r') as f:
                zs[i] = f.attrs['z']
                L_tot[i] = f.attrs['L_tot']
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
                red = (freqs > 0.)
                for j in range(n_cameras):
                    freq_seps[i,j] = freqs[red][np.argmax(fluxes[j,red])] - freqs[~red][np.argmax(fluxes[j,~red])]
                flux = f['flux_avg'][:]
                freq_sep[i] = freqs[red][np.argmax(flux[red])] - freqs[~red][np.argmax(flux[~red])]
            with h5py.File(f'{out_dir}/ion-eq_{snaps[i]:03d}.hdf5', 'r') as f:
                Ndot_LyC[i] = f.attrs['Ndot_tot']
                f_esc_LyC[i] = f.attrs['f_esc']
                f_escs_LyC[i,:] = f['f_escs'][:]
            with h5py.File(f'{out_dir}/M1500_{snaps[i]:03d}.hdf5', 'r') as f:
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
                L_1500_obs = L_1500 * f['f_escs'][:]
                pc = 3.085677581467192e18       # Units: 1 pc  = 3e18 cm
                R_10pc = 10. * pc               # Reference distance for continuum [cm]
                fnu_1500_fac = lambda_1500**2 / (4. * np.pi * c * R_10pc**2 * angstrom)
                M_1500[i] = -2.5 * np.log10(fnu_1500_fac * L_1500) - 48.6 # Continuum absolute magnitude
                M_1500_obs[i,:] = -2.5 * np.log10(fnu_1500_fac * L_1500_obs) - 48.6
            success[i] = True
        except:
            pass
    snaps = np.array(snaps, dtype=np.int32)
    n_success = np.count_nonzero(success)
    if n_success < n_snaps:
        if n_success == 0:
            print(f'No snapshots found for {out_dir} ...')
            return
        print(f'Only {n_success} out of {n_snaps} snapshots found for {out_dir} ...')
        n_snaps = np.int32(n_success)
        snaps = snaps[success]
        zs = zs[success]
        L_tot = L_tot[success]
        f_esc = f_esc[success]
        f_escs = f_escs[success,:]
        freq_sep = freq_sep[success]
        freq_avg = freq_avg[success]
        freq_std = freq_std[success]
        freq_seps = freq_seps[success,:]
        freq_avgs = freq_avgs[success,:]
        freq_stds = freq_stds[success,:]
        M_1500 = M_1500[success]
        M_1500_obs = M_1500_obs[success,:]
        Ndot_LyC = Ndot_LyC[success]
        f_esc_LyC = f_esc_LyC[success]
        f_escs_LyC = f_escs_LyC[success,:]
    with h5py.File(f'{out_dir}/Lya.hdf5', 'w') as f:
        f.attrs['n_snaps'] = n_snaps
        f.attrs['n_cameras'] = n_cameras
        f.create_dataset('snaps', data=snaps)
        f.create_dataset('zs', data=zs)
        f.create_dataset('L_tot', data=L_tot)
        f.create_dataset('f_esc', data=f_esc)
        f.create_dataset('f_escs', data=f_escs)
        f.create_dataset('freq_sep', data=freq_sep)
        f.create_dataset('freq_avg', data=freq_avg)
        f.create_dataset('freq_std', data=freq_std)
        f.create_dataset('freq_seps', data=freq_seps)
        f.create_dataset('freq_avgs', data=freq_avgs)
        f.create_dataset('freq_stds', data=freq_stds)
        f.create_dataset('M_1500', data=M_1500)
        f.create_dataset('M_1500_obs', data=M_1500_obs)
        f.create_dataset('Ndot_LyC', data=Ndot_LyC)
        f.create_dataset('f_esc_LyC', data=f_esc_LyC)
        f.create_dataset('f_escs_LyC', data=f_escs_LyC)

if __name__ == '__main__':
    colt_dir = '/orcd/data/mvogelsb/004/Thesan-Zooms-COLT'
    # colt_dir = '/nfs/mvogelsblab001/Lab/Thesan-Zooms-COLT'
    # colt_dir = '/orcd/data/mvogelsb/005/Lab/Thesan-Zooms-COLT'
    # sims = ['g2', 'g39', 'g205', 'g578', 'g1163', 'g5760', 'g10304', 'g137030', 'g500531', 'g519761', 'g2274036'][4:]
    # runs = ['z4', 'z4', 'z4', 'z4', 'z4', 'z8', 'z8', 'z16', 'z16', 'z16'][4:]
    # , 'g5229300']
    # , 'z16']
    sims = ['g33206']
    runs = ['z8']
    for sim, run in zip(sims, runs):
        out_dir = f'{colt_dir}/{sim}/{run}/output_tree'
        print(f'Processing {out_dir} ...')
        repack(snaps=range(0,188+1), out_dir=out_dir)
