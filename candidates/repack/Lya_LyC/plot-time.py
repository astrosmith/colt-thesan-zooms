import numpy as np
import h5py, os
import healpy as hp
from healpy import projaxes as PA
from healpy import pixelfunc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib import patheffects
import cmasher as cmr
import cmocean
from pixel_mapping import degrade_healpix_custom

from scipy.special import erf
sigma_68, sigma_95, sigma_99 = erf(1./np.sqrt(2.)), erf(2./np.sqrt(2.)), erf(3./np.sqrt(2.))
percentiles = [50., 50.*(1.-sigma_68), 50.*(1.+sigma_68), 50.*(1.-sigma_95), 50.*(1.+sigma_95), 50.*(1.-sigma_99), 50.*(1.+sigma_99)]
n_percentiles = len(percentiles)
# print('percentiles =', percentiles)

zoom_dir = '/orcd/data/mvogelsb/004/Thesan-Zooms'
# colt_dir = f'{zoom_dir}-COLT'
colt_dir = '/nfs/mvogelsblab001/Lab/Thesan-Zooms-COLT'

pc   = 3.085677581467192e18 # Units: 1 pc  = 3e18 cm
kpc  = 1.0e3 * pc           # Units: 1 kpc = 3e21 cm
H_units = 1.6735327e-24 / 0.76  # Conversion from rho to n_H

grey   = [0.5, 0.5, 0.5]
lg     = [0.8, 0.8, 0.8]
yellow = [238./255., 201./255.,  0./255.]
orange = [255./255., 128./255., 15./255.]
blue   = [.75*135./255., .75*206./255., .75*230./255.]
green  = [105./255., 139./255.,  34./255.]
red    = [205./255., 0., 0.]
coral  = [255./255., 127./255., 80./255.]
purple = [0.6,0.1,0.6]
wm     = (0.4, 0.4, 0.4)

# IonMap = MapData(field='ion-eq', key='map', u_str=r'$f_{\rm esc}^{\rm\,LyC}\ \ (\%)$', lims='[0,max]', cmap=cmr.ember, norm=None, units=0.01, n_format=0)
# LyaMap = MapData(field='Lya', key='map', u_str=r'$f_{\rm esc}^{\rm\,Ly\alpha}\ \ (\%)$', lims='[0,max]', cmap=plt.cm.afmhot, norm=None, units=0.01, n_format=0)
# Lya1Map = MapData(field='Lya', key='freq_map', u_str=r'$\langle \Delta v_{\rm{Ly}\alpha} \rangle\ \ (\rm{km}/\rm{s})$', lims=[-100,100], cmap=cmocean.cm.balance, norm=None, n_format=0)
# Lya2Map = MapData(field='Lya', key='freq2_map', u_str=r'$\sigma_{\Delta v,{\rm{Ly}\alpha}}\ \ (\rm{km}/\rm{s})$', lims='[min,max]', cmap=plt.cm.spring, norm=None, n_format=0)
# HI_int_Map = MapData(field='ion-esc', key='HI_columns_int', u_str=r'$\log N_{\rm{HI}}^{\rm int}\ \ (\rm{cm}^{-2})$', lims='[min,max]', cmap=cmr.savanna, norm=None, units=H_units, n_format=1, log=True)
# HI_esc_Map = MapData(field='ion-esc', key='HI_columns_esc', u_str=r'$\log N_{\rm{HI}}^{\rm esc}\ \ (\rm{cm}^{-2})$', lims='[min,max]', cmap=cmr.savanna, norm=None, units=H_units, n_format=1, log=True)
# f_HI_Map = MapData(field='ion-esc', key='f_HI', u_str=r'$f_{\rm{HI}}\ \ (\%)$', lims='[0,max]', cmap=cmr.ember, norm=None, units=0.01, n_format=0)
# f_HeI_Map = MapData(field='ion-esc', key='f_HeI', u_str=r'$f_{\rm{HeI}}\ \ (\%)$', lims='[0,max]', cmap=cmr.ember, norm=None, units=0.01, n_format=0)
# f_HeII_Map = MapData(field='ion-esc', key='f_HeII', u_str=r'$f_{\rm{HeII}}\ \ (\%)$', lims='[0,max]', cmap=cmr.ember, norm=None, units=0.01, n_format=0)
# f_abs_Map = MapData(field='ion-esc', key='f_abss', u_str=r'$f_{\rm{abs}}\ \ (\%)$', lims='[0,max]', cmap=cmr.ember, norm=None, units=0.01, n_format=0)
# f_esc_Map = MapData(field='ion-esc', key='f_escs', u_str=r'$f_{\rm{esc}}\ \ (\%)$', lims='[0,max]', cmap=cmr.ember, norm=None, units=0.01, n_format=0)
# gas_int_Map = MapData(field='ion-esc', key='gas_columns_int', u_str=r'$\log N_{\rm{H}}^{\rm int}\ \ (\rm{cm}^{-2})$', lims='[min,max]', cmap=cmr.savanna, norm=None, units=H_units, n_format=1, log=True)
# gas_esc_Map = MapData(field='ion-esc', key='gas_columns_esc', u_str=r'$\log N_{\rm{H}}^{\rm esc}\ \ (\rm{cm}^{-2})$', lims='[min,max]', cmap=cmr.savanna, norm=None, units=H_units, n_format=1, log=True)
# metal_int_Map = MapData(field='ion-esc', key='metal_columns_int', u_str=r'$\log \Sigma_Z^{\rm int}\ \ (\rm{g\ cm}^{-2})$', lims='[min,max]', cmap=cmr.savanna, norm=None, n_format=1, log=True)
# metal_esc_Map = MapData(field='ion-esc', key='metal_columns_esc', u_str=r'$\log \Sigma_Z^{\rm esc}\ \ (\rm{g\ cm}^{-2})$', lims='[min,max]', cmap=cmr.savanna, norm=None, n_format=1, log=True)
# mean_dists_Map = MapData(field='ion-esc', key='mean_dists', u_str=r'$\log \ell\ \ (\rm{pc})$', lims='[min,max]', cmap=cmr.savanna, norm=None, units=pc, n_format=1, log=True)
# graphite_int_Map = MapData(field='ion-esc', key='dust_graphite_columns_int', u_str=r'$\log \Sigma_{\rm{G}}^{\rm int}\ \ (\rm{g\ cm}^{-2})$', lims='[min,max]', cmap=cmr.savanna, norm=None, n_format=1, log=True)
# graphite_esc_Map = MapData(field='ion-esc', key='dust_graphite_columns_esc', u_str=r'$\log \Sigma_{\rm{G}}^{\rm esc}\ \ (\rm{g\ cm}^{-2})$', lims='[min,max]', cmap=cmr.savanna, norm=None, n_format=1, log=True)
# silicate_int_Map = MapData(field='ion-esc', key='dust_silicate_columns_int', u_str=r'$\log \Sigma_{\rm{S}}^{\rm int}\ \ (\rm{g\ cm}^{-2})$', lims='[min,max]', cmap=cmr.savanna, norm=None, n_format=1, log=True)
# silicate_esc_Map = MapData(field='ion-esc', key='dust_silicate_columns_esc', u_str=r'$\log \Sigma_{\rm{S}}^{\rm esc}\ \ (\rm{g\ cm}^{-2})$', lims='[min,max]', cmap=cmr.savanna, norm=None, n_format=1, log=True)

def plot_healpix_map(data, filename, title, n_format=0, u_str=None):
    f = plt.figure(figsize=(3.,2.))
    extent = (0,0,1,1)
    map = pixelfunc.ma_to_array(data)
    ax = PA.HpxMollweideAxes(f, extent)
    f.add_axes(ax)
    # m.cmap.set_under('w')
    img = ax.projmap(map, cmap='inferno')
    im = ax.get_images()[0]
    b = im.norm.inverse(np.linspace(0,1,im.cmap.N+1))
    v = np.linspace(im.norm.vmin,im.norm.vmax,im.cmap.N)
    cb = f.colorbar(im, ax=ax, orientation='horizontal',
                    shrink=0.75, aspect=25, ticks=PA.BoundaryLocator(),
                    pad=0.05, fraction=0.1, boundaries=b, values=v,
                    format=r'${\rm '+str('%0.'+str(n_format)+'f')+r'}$')
    cb.solids.set_rasterized(True)
    ax.set_title(title)
    if u_str is not None:
        cb.ax.text(0.5, -2.0, u_str, fontsize=14.5, transform=cb.ax.transAxes, ha='center', va='center')
    f.sca(ax)
    f.savefig(filename, bbox_inches='tight', transparent=False, dpi=600, pad_inches=0.)
    plt.close()


def weighted_percentile(Z, W, q):
    # Z = data, W = weights, q = percentiles in [0,100]
    isort = np.argsort(Z)
    Z_sorted = Z[isort]
    W_sorted = W[isort]
    IW_sorted = np.cumsum(W_sorted)
    IW_sorted /= IW_sorted[-1]
    wp = np.zeros_like(q)
    for i_q in range(len(q)):
        q_frac = q[i_q] / 100.
        i = np.searchsorted(IW_sorted, q_frac)
        # print('q_frac =', q_frac, 'IW_sorted[i-1] =', IW_sorted[i-1], 'IW_sorted[i] =', IW_sorted[i])
        assert IW_sorted[i] >= q_frac, f'q_frac = {q_frac}, IW_sorted[i] = {IW_sorted[i]}'
        wp[i_q] = (Z_sorted[i]-Z_sorted[i-1]) * (q_frac-IW_sorted[i-1]) / (IW_sorted[i]-IW_sorted[i-1]) + Z_sorted[i-1]
    return wp

def zip_data(sim='g10304', run='z8', snaps=range(189), plot_map=False, output_iso=False, output_LyC=False, output_logLyC=False, output_rankLyC=False, output_LyCn2=True, output_LyCn4=True, output_LyCn8=True):
    n_snaps = len(snaps)
    valid_snaps = np.zeros(n_snaps, dtype=bool)
    redshifts = np.zeros(n_snaps)
    Ndot_LyC = np.zeros(n_snaps)
    L_Lya = np.zeros(n_snaps)
    f_esc_Lya = np.zeros(n_snaps)
    f_esc_LyC = np.zeros(n_snaps)
    f2_esc_Lya = np.zeros(n_snaps)
    f2_esc_LyC = np.zeros(n_snaps)
    f_Lya_LyC = np.zeros(n_snaps)
    Dv_Lya = np.zeros([n_snaps, n_percentiles+1])
    Sv_Lya = np.zeros([n_snaps, n_percentiles+1])
    Dv_Lya_LyC = np.zeros([n_snaps, n_percentiles+1])
    Sv_Lya_LyC = np.zeros([n_snaps, n_percentiles+1])
    if output_iso:
        Dv_iso = np.zeros([n_snaps, n_percentiles+1])
        Sv_iso = np.zeros([n_snaps, n_percentiles+1])
    if output_LyC:
        f_Lya_logLyC = np.zeros(n_snaps)
        Dv_LyC = np.zeros([n_snaps, n_percentiles+1])
        Sv_LyC = np.zeros([n_snaps, n_percentiles+1])
    if output_logLyC:
        Dv_Lya_logLyC = np.zeros([n_snaps, n_percentiles+1])
        Sv_Lya_logLyC = np.zeros([n_snaps, n_percentiles+1])
    if output_LyCn2:
        f_Lya_LyCn2 = np.zeros(n_snaps)
        Dv_Lya_LyCn2 = np.zeros([n_snaps, n_percentiles+1])
        Sv_Lya_LyCn2 = np.zeros([n_snaps, n_percentiles+1])
    if output_LyCn4:
        f_Lya_LyCn4 = np.zeros(n_snaps)
        Dv_Lya_LyCn4 = np.zeros([n_snaps, n_percentiles+1])
        Sv_Lya_LyCn4 = np.zeros([n_snaps, n_percentiles+1])
    if output_LyCn8:
        f_Lya_LyCn8 = np.zeros(n_snaps)
        Dv_Lya_LyCn8 = np.zeros([n_snaps, n_percentiles+1])
        Sv_Lya_LyCn8 = np.zeros([n_snaps, n_percentiles+1])
    if output_rankLyC:
        f_Lya_rankLyC = np.zeros(n_snaps)
        Dv_Lya_rankLyC = np.zeros([n_snaps, n_percentiles+1])
        Sv_Lya_rankLyC = np.zeros([n_snaps, n_percentiles+1])
    tree_dir = f'{zoom_dir}/{sim}/{run}/postprocessing/colt_tree'
    ics_dir = f'{colt_dir}/{sim}/{run}/ics_tree'
    for i, snap in enumerate(snaps):
        try:
            with h5py.File(f'{tree_dir}/Lya/Lya_{snap:03d}.hdf5','r') as f:
                L_Lya[i] = f.attrs['L_tot']
                map_Lya = f['map'][:]
                n_Lya = len(map_Lya)
                freq_map_Lya = f['freq_map'][:]
                freq2_map_Lya = f['freq2_map'][:]
                freq_std_Lya = np.sqrt(freq2_map_Lya - freq_map_Lya**2)
                redshifts[i] = f.attrs['z']
            with h5py.File(f'{tree_dir}/ion-eq/ion-eq_{snap:03d}.hdf5','r') as f:
                Ndot_LyC[i] = f.attrs['Ndot_tot']
                map_LyC = f['map'][:]
                n_LyC = len(map_LyC)
            if n_LyC > n_Lya:
                if plot_map:
                    plot_healpix_map(100.*map_LyC, f'LyC-10_{snap:03d}.png', title=f'LyC Map (10) {snap}')
                map_LyC = degrade_healpix_custom(map_LyC, n_Lya)
                n_LyC = len(map_LyC)
                # Alternative biasing statistics
                if output_logLyC:
                    mask = map_LyC > 0.
                    map_logLyC = np.zeros_like(map_LyC)
                    map_logLyC[mask] = np.log10(map_LyC[mask] / np.min(map_LyC[mask]))  # Log LyC
                if output_LyCn2:
                    map_LyCn2 = np.sqrt(map_LyC)  # LyC^1/2
                if output_LyCn4:
                    map_LyCn4 = map_LyC ** (1./4.)  # LyC^1/4
                if output_LyCn8:
                    map_LyCn8 = map_LyC ** (1./8.)  # LyC^1/8
                if output_rankLyC:
                    map_rankLyC = np.argsort(np.argsort(map_LyC))  # Rank LyC
                if plot_map:
                    plot_healpix_map(100.*map_LyC, f'LyC-5_{snap:03d}.png', title=f'LyC Map (5) {snap}')
            elif n_LyC != n_Lya:
                raise ValueError(f'n_LyC ({n_LyC}) != n_Lya ({n_Lya})')
            # Correlations and kinematic averages
            f_esc_Lya[i] = np.mean(map_Lya)
            f_esc_LyC[i] = np.mean(map_LyC)
            f2_esc_Lya[i] = np.mean(map_Lya**2)
            f2_esc_LyC[i] = np.mean(map_LyC**2)
            f_Lya_LyC[i] = np.mean(map_Lya * map_LyC)
            Dv_Lya[i,:-1] = weighted_percentile(freq_map_Lya, map_Lya, percentiles)
            Dv_Lya[i,-1] = np.average(freq_map_Lya, weights=map_Lya)
            Sv_Lya[i,:-1] = weighted_percentile(freq_std_Lya, map_Lya, percentiles)
            Sv_Lya[i,-1] = np.sqrt(np.average(freq2_map_Lya, weights=map_Lya) - Dv_Lya[i,-1]**2)
            Dv_Lya_LyC[i,:-1] = weighted_percentile(freq_map_Lya, map_Lya * map_LyC, percentiles)
            Dv_Lya_LyC[i,-1] = np.average(freq_map_Lya, weights=map_Lya * map_LyC)
            Sv_Lya_LyC[i,:-1] = weighted_percentile(freq_std_Lya, map_Lya * map_LyC, percentiles)
            Sv_Lya_LyC[i,-1] = np.sqrt(np.average(freq2_map_Lya, weights=map_Lya * map_LyC) - Dv_Lya_LyC[i,-1]**2)
            if output_iso:
                Dv_iso[i,:-1] = weighted_percentile(freq_map_Lya, np.ones_like(freq_map_Lya), percentiles)
                Dv_iso[i,-1] = np.mean(freq_map_Lya)
                Sv_iso[i,:-1] = weighted_percentile(freq_std_Lya, np.ones_like(freq_std_Lya), percentiles)
                Sv_iso[i,-1] = np.sqrt(np.mean(freq2_map_Lya) - Dv_iso[i,-1]**2)
            if output_LyC:
                Dv_LyC[i,:-1] = weighted_percentile(freq_map_Lya, map_LyC, percentiles)
                Dv_LyC[i,-1] = np.average(freq_map_Lya, weights=map_LyC)
                Sv_LyC[i,:-1] = weighted_percentile(freq_std_Lya, map_LyC, percentiles)
                Sv_LyC[i,-1] = np.sqrt(np.average(freq2_map_Lya, weights=map_LyC) - Dv_LyC[i,-1]**2)
            if output_logLyC:
                f_Lya_logLyC[i] = np.mean(map_Lya * map_logLyC)
                Dv_Lya_logLyC[i,:-1] = weighted_percentile(freq_map_Lya, map_Lya * map_logLyC, percentiles)
                Dv_Lya_logLyC[i,-1] = np.average(freq_map_Lya, weights=map_Lya * map_logLyC)
                Sv_Lya_logLyC[i,:-1] = weighted_percentile(freq_std_Lya, map_Lya * map_logLyC, percentiles)
                Sv_Lya_logLyC[i,-1] = np.sqrt(np.average(freq2_map_Lya, weights=map_Lya * map_logLyC) - Dv_Lya_logLyC[i,-1]**2)
            if output_LyCn2:
                f_Lya_LyCn2[i] = np.mean(map_Lya * map_LyCn2)
                Dv_Lya_LyCn2[i,:-1] = weighted_percentile(freq_map_Lya, map_Lya * map_LyCn2, percentiles)
                Dv_Lya_LyCn2[i,-1] = np.average(freq_map_Lya, weights=map_Lya * map_LyCn2)
                Sv_Lya_LyCn2[i,:-1] = weighted_percentile(freq_std_Lya, map_Lya * map_LyCn2, percentiles)
                Sv_Lya_LyCn2[i,-1] = np.sqrt(np.average(freq2_map_Lya, weights=map_Lya * map_LyCn2) - Dv_Lya_LyCn2[i,-1]**2)
            if output_LyCn4:
                f_Lya_LyCn4[i] = np.mean(map_Lya * map_LyCn4)
                Dv_Lya_LyCn4[i,:-1] = weighted_percentile(freq_map_Lya, map_Lya * map_LyCn4, percentiles)
                Dv_Lya_LyCn4[i,-1] = np.average(freq_map_Lya, weights=map_Lya * map_LyCn4)
                Sv_Lya_LyCn4[i,:-1] = weighted_percentile(freq_std_Lya, map_Lya * map_LyCn4, percentiles)
                Sv_Lya_LyCn4[i,-1] = np.sqrt(np.average(freq2_map_Lya, weights=map_Lya * map_LyCn4) - Dv_Lya_LyCn4[i,-1]**2)
            if output_LyCn8:
                f_Lya_LyCn8[i] = np.mean(map_Lya * map_LyCn8)
                Dv_Lya_LyCn8[i,:-1] = weighted_percentile(freq_map_Lya, map_Lya * map_LyCn8, percentiles)
                Dv_Lya_LyCn8[i,-1] = np.average(freq_map_Lya, weights=map_Lya * map_LyCn8)
                Sv_Lya_LyCn8[i,:-1] = weighted_percentile(freq_std_Lya, map_Lya * map_LyCn8, percentiles)
                Sv_Lya_LyCn8[i,-1] = np.sqrt(np.average(freq2_map_Lya, weights=map_Lya * map_LyCn8) - Dv_Lya_LyCn8[i,-1]**2)
            if output_rankLyC:
                f_Lya_rankLyC[i] = np.mean(map_Lya * map_rankLyC)
                Dv_Lya_rankLyC[i,:-1] = weighted_percentile(freq_map_Lya, map_Lya * map_rankLyC, percentiles)
                Dv_Lya_rankLyC[i,-1] = np.average(freq_map_Lya, weights=map_Lya * map_rankLyC)
                Sv_Lya_rankLyC[i,:-1] = weighted_percentile(freq_std_Lya, map_Lya * map_rankLyC, percentiles)
                Sv_Lya_rankLyC[i,-1] = np.sqrt(np.average(freq2_map_Lya, weights=map_Lya * map_rankLyC) - Dv_Lya_rankLyC[i,-1]**2)
            valid_snaps[i] = True
        except:
            pass
    redshifts = redshifts[valid_snaps]
    Ndot_LyC = Ndot_LyC[valid_snaps]
    L_Lya = L_Lya[valid_snaps]
    f_esc_Lya = f_esc_Lya[valid_snaps]
    f_esc_LyC = f_esc_LyC[valid_snaps]
    f2_esc_Lya = f2_esc_Lya[valid_snaps]
    f2_esc_LyC = f2_esc_LyC[valid_snaps]
    f_Lya_LyC = f_Lya_LyC[valid_snaps]
    Dv_Lya = Dv_Lya[valid_snaps]
    Sv_Lya = Sv_Lya[valid_snaps]
    Dv_Lya_LyC = Dv_Lya_LyC[valid_snaps]
    Sv_Lya_LyC = Sv_Lya_LyC[valid_snaps]
    if output_iso:
        Dv_iso = Dv_iso[valid_snaps]
        Sv_iso = Sv_iso[valid_snaps]
    if output_LyC:
        Dv_LyC = Dv_LyC[valid_snaps]
        Sv_LyC = Sv_LyC[valid_snaps]
    if output_logLyC:
        f_Lya_logLyC = f_Lya_logLyC[valid_snaps]
        Dv_Lya_logLyC = Dv_Lya_logLyC[valid_snaps]
        Sv_Lya_logLyC = Sv_Lya_logLyC[valid_snaps]
    if output_LyCn2:
        f_Lya_LyCn2 = f_Lya_LyCn2[valid_snaps]
        Dv_Lya_LyCn2 = Dv_Lya_LyCn2[valid_snaps]
        Sv_Lya_LyCn2 = Sv_Lya_LyCn2[valid_snaps]
    if output_LyCn4:
        f_Lya_LyCn4 = f_Lya_LyCn4[valid_snaps]
        Dv_Lya_LyCn4 = Dv_Lya_LyCn4[valid_snaps]
        Sv_Lya_LyCn4 = Sv_Lya_LyCn4[valid_snaps]
    if output_LyCn8:
        f_Lya_LyCn8 = f_Lya_LyCn8[valid_snaps]
        Dv_Lya_LyCn8 = Dv_Lya_LyCn8[valid_snaps]
        Sv_Lya_LyCn8 = Sv_Lya_LyCn8[valid_snaps]
    if output_rankLyC:
        f_Lya_rankLyC = f_Lya_rankLyC[valid_snaps]
        Dv_Lya_rankLyC = Dv_Lya_rankLyC[valid_snaps]
        Sv_Lya_rankLyC = Sv_Lya_rankLyC[valid_snaps]
    with h5py.File(f'data/{sim}_{run}.hdf5','w') as f:
        f.create_dataset('redshifts', data=redshifts)
        f.create_dataset('Ndot_LyC', data=Ndot_LyC)
        f.create_dataset('L_Lya', data=L_Lya)
        f.create_dataset('f_esc_Lya', data=f_esc_Lya)
        f.create_dataset('f_esc_LyC', data=f_esc_LyC)
        f.create_dataset('f2_esc_Lya', data=f2_esc_Lya)
        f.create_dataset('f2_esc_LyC', data=f2_esc_LyC)
        f.create_dataset('f_Lya_LyC', data=f_Lya_LyC)
        f.create_dataset('Dv_Lya', data=Dv_Lya)
        f.create_dataset('Sv_Lya', data=Sv_Lya)
        f.create_dataset('Dv_Lya_LyC', data=Dv_Lya_LyC)
        f.create_dataset('Sv_Lya_LyC', data=Sv_Lya_LyC)
        if output_iso:
            f.create_dataset('Dv_iso', data=Dv_iso)
            f.create_dataset('Sv_iso', data=Sv_iso)
        if output_LyC:
            f.create_dataset('Dv_LyC', data=Dv_LyC)
            f.create_dataset('Sv_LyC', data=Sv_LyC)
        if output_logLyC:
            f.create_dataset('f_Lya_logLyC', data=f_Lya_logLyC)
            f.create_dataset('Dv_Lya_logLyC', data=Dv_Lya_logLyC)
            f.create_dataset('Sv_Lya_logLyC', data=Sv_Lya_logLyC)
        if output_LyCn2:
            f.create_dataset('f_Lya_LyCn2', data=f_Lya_LyCn2)
            f.create_dataset('Dv_Lya_LyCn2', data=Dv_Lya_LyCn2)
            f.create_dataset('Sv_Lya_LyCn2', data=Sv_Lya_LyCn2)
        if output_LyCn4:
            f.create_dataset('f_Lya_LyCn4', data=f_Lya_LyCn4)
            f.create_dataset('Dv_Lya_LyCn4', data=Dv_Lya_LyCn4)
            f.create_dataset('Sv_Lya_LyCn4', data=Sv_Lya_LyCn4)
        if output_LyCn8:
            f.create_dataset('f_Lya_LyCn8', data=f_Lya_LyCn8)
            f.create_dataset('Dv_Lya_LyCn8', data=Dv_Lya_LyCn8)
            f.create_dataset('Sv_Lya_LyCn8', data=Sv_Lya_LyCn8)
        if output_rankLyC:
            f.create_dataset('f_Lya_rankLyC', data=f_Lya_rankLyC)
            f.create_dataset('Dv_Lya_rankLyC', data=Dv_Lya_rankLyC)
            f.create_dataset('Sv_Lya_rankLyC', data=Sv_Lya_rankLyC)

def set_redshift_axis(ax):
    ax.set_xlabel(r'${\rm Redshift}$', fontsize=15)
    ax.set_xscale('log')
    ax.minorticks_on()
    xmin, xmax = 3, 15
    ax.set_xlim([xmin, xmax])
    ticks = [3,4,5,6,7,8,10,12,15]
    ax.set_xticks(ticks); ax.set_xticklabels([r'$%g$' % tick for tick in ticks], fontsize=13)
    ticks = np.linspace(xmin, xmax, xmax-xmin+1); ax.set_xticks(ticks, minor=True)
    ax.set_xticklabels(['']*len(ticks), minor=True)

def set_xaxis(ax, xlabel, xmin=None, xmax=None, n=None):
    ax.set_xlabel(xlabel, fontsize=15)
    if xmin is not None and xmax is not None and n is not None:
        ax.set_xlim(xmin, xmax)
        ticks = np.linspace(xmin, xmax, n)
        ax.set_xticks(ticks); ax.set_xticklabels([r'$%g$' % tick for tick in ticks], fontsize=13)

def set_yaxis(ax, ylabel, ymin=None, ymax=None, n=None, yp=0.):
    ax.set_ylabel(ylabel, fontsize=15)
    if ymin is not None and ymax is not None and n is not None:
        ax.set_ylim(ymin, ymax + yp)
        ticks = np.linspace(ymin, ymax, n)
        ax.set_yticks(ticks); ax.set_yticklabels([r'$%g$' % tick for tick in ticks], fontsize=13)

def set_log_xaxis(ax, xlabel, xmin=None, xmax=None):
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_xscale('log')
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
        Lxmin,Lxmax = int(np.ceil(np.log10(xmin))), int(np.floor(np.log10(xmax)))
        ticks = np.linspace(Lxmin, Lxmax, Lxmax-Lxmin+1)
        ax.set_xticks(10**ticks); ax.set_xticklabels([r'$10^{%g}$' % tick for tick in ticks], fontsize=13)

def set_log_yaxis(ax, ylabel, ymin=None, ymax=None):
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_yscale('log')
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
        Lymin,Lymax = int(np.ceil(np.log10(ymin))), int(np.floor(np.log10(ymax)))
        ticks = np.linspace(Lymin, Lymax, Lymax-Lymin+1)
        ax.set_yticks(10**ticks); ax.set_yticklabels([r'$10^{%g}$' % tick for tick in ticks], fontsize=13)

def plot_Dv(sim='g5760', run='z8', n_outliers=0, n_start=0, output_iso=False, output_LyC=False, output_logLyC=False, output_rankLyC=False, output_LyCn2=True, output_LyCn4=True, output_LyCn8=True):
    with h5py.File(f'data/{sim}_{run}.hdf5','r') as f:
        redshifts = f['redshifts'][n_start:]
        Ndot_LyC = f['Ndot_LyC'][n_start:]
        L_Lya = f['L_Lya'][n_start:]
        f_esc_Lya = f['f_esc_Lya'][n_start:]
        f_esc_LyC = f['f_esc_LyC'][n_start:]
        f2_esc_Lya = f['f2_esc_Lya'][n_start:]
        f2_esc_LyC = f['f2_esc_LyC'][n_start:]
        f_Lya_LyC = f['f_Lya_LyC'][n_start:]
        Dv_Lya = f['Dv_Lya'][n_start:]
        Sv_Lya = f['Sv_Lya'][n_start:]
        Dv_Lya_LyC = f['Dv_Lya_LyC'][n_start:]
        Sv_Lya_LyC = f['Sv_Lya_LyC'][n_start:]
        if output_iso:
            Dv_iso = f['Dv_iso'][n_start:]
            Sv_iso = f['Sv_iso'][n_start:]
        if output_LyC:
            Dv_LyC = f['Dv_LyC'][n_start:]
            Sv_LyC = f['Sv_LyC'][n_start:]
        if output_logLyC:
            f_Lya_logLyC = f['f_Lya_logLyC'][n_start:]
            Dv_Lya_logLyC = f['Dv_Lya_logLyC'][n_start:]
            Sv_Lya_logLyC = f['Sv_Lya_logLyC'][n_start:]
        if output_LyCn2:
            f_Lya_LyCn2 = f['f_Lya_LyCn2'][n_start:]
            Dv_Lya_LyCn2 = f['Dv_Lya_LyCn2'][n_start:]
            Sv_Lya_LyCn2 = f['Sv_Lya_LyCn2'][n_start:]
        if output_LyCn4:
            f_Lya_LyCn4 = f['f_Lya_LyCn4'][n_start:]
            Dv_Lya_LyCn4 = f['Dv_Lya_LyCn4'][n_start:]
            Sv_Lya_LyCn4 = f['Sv_Lya_LyCn4'][n_start:]
        if output_LyCn8:
            f_Lya_LyCn8 = f['f_Lya_LyCn8'][n_start:]
            Dv_Lya_LyCn8 = f['Dv_Lya_LyCn8'][n_start:]
            Sv_Lya_LyCn8 = f['Sv_Lya_LyCn8'][n_start:]
        if output_rankLyC:
            f_Lya_rankLyC = f['f_Lya_rankLyC'][n_start:]
            Dv_Lya_rankLyC = f['Dv_Lya_rankLyC'][n_start:]
            Sv_Lya_rankLyC = f['Sv_Lya_rankLyC'][n_start:]
    # Remove the N outlier points
    for i in range(n_outliers):
        imax = np.argmax(Sv_Lya[:,-1])
        Ndot_LyC = np.delete(Ndot_LyC, imax)
        L_Lya = np.delete(L_Lya, imax)
        f_esc_Lya = np.delete(f_esc_Lya, imax)
        f_esc_LyC = np.delete(f_esc_LyC, imax)
        f2_esc_Lya = np.delete(f2_esc_Lya, imax)
        f2_esc_LyC = np.delete(f2_esc_LyC, imax)
        f_Lya_LyC = np.delete(f_Lya_LyC, imax)
        Dv_Lya = np.delete(Dv_Lya, imax, axis=0)
        Sv_Lya = np.delete(Sv_Lya, imax, axis=0)
        Dv_Lya_LyC = np.delete(Dv_Lya_LyC, imax, axis=0)
        Sv_Lya_LyC = np.delete(Sv_Lya_LyC, imax, axis=0)
        if output_iso:
            Dv_iso = np.delete(Dv_iso, imax, axis=0)
            Sv_iso = np.delete(Sv_iso, imax, axis=0)
        if output_LyC:
            Dv_LyC = np.delete(Dv_LyC, imax, axis=0)
            Sv_LyC = np.delete(Sv_LyC, imax, axis=0)
        if output_logLyC:
            f_Lya_logLyC = np.delete(f_Lya_logLyC, imax)
            Dv_Lya_logLyC = np.delete(Dv_Lya_logLyC, imax, axis=0)
            Sv_Lya_logLyC = np.delete(Sv_Lya_logLyC, imax, axis=0)
        if output_LyCn2:
            f_Lya_LyCn2 = np.delete(f_Lya_LyCn2, imax)
            Dv_Lya_LyCn2 = np.delete(Dv_Lya_LyCn2, imax, axis=0)
            Sv_Lya_LyCn2 = np.delete(Sv_Lya_LyCn2, imax, axis=0)
        if output_LyCn4:
            f_Lya_LyCn4 = np.delete(f_Lya_LyCn4, imax)
            Dv_Lya_LyCn4 = np.delete(Dv_Lya_LyCn4, imax, axis=0)
            Sv_Lya_LyCn4 = np.delete(Sv_Lya_LyCn4, imax, axis=0)
        if output_LyCn8:
            f_Lya_LyCn8 = np.delete(f_Lya_LyCn8, imax)
            Dv_Lya_LyCn8 = np.delete(Dv_Lya_LyCn8, imax, axis=0)
            Sv_Lya_LyCn8 = np.delete(Sv_Lya_LyCn8, imax, axis=0)
        if output_rankLyC:
            f_Lya_rankLyC = np.delete(f_Lya_rankLyC, imax)
            Dv_Lya_rankLyC = np.delete(Dv_Lya_rankLyC, imax, axis=0)
            Sv_Lya_rankLyC = np.delete(Sv_Lya_rankLyC, imax, axis=0)
        redshifts = np.delete(redshifts, imax)
    s_esc_Lya = np.sqrt(f2_esc_Lya - f_esc_Lya**2)
    s_esc_LyC = np.sqrt(f2_esc_LyC - f_esc_LyC**2)
    CoV_esc_Lya = s_esc_Lya / f_esc_Lya  # Coefficient of Variation
    CoV_esc_LyC = s_esc_LyC / f_esc_LyC
    rho_Lya_LyC = (f_Lya_LyC - f_esc_Lya * f_esc_LyC) / (s_esc_Lya * s_esc_LyC)  # Pearson correlation coefficient

    # Plot Dv vs redshift
    fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
    iC = 0  # Color counter
    ax.fill_between(redshifts, Dv_Lya[:,1], Dv_Lya[:,2], alpha=0.2, color=f'C{iC}', lw=0)
    ax.plot(redshifts, Dv_Lya[:,-1], c=f'C{iC}', label='Lya')
    ax.plot(redshifts, Dv_Lya[:,0], c=f'C{iC}', linestyle='--')
    iC += 1
    ax.fill_between(redshifts, Dv_Lya_LyC[:,1], Dv_Lya_LyC[:,2], alpha=0.2, color=f'C{iC}', lw=0)
    ax.plot(redshifts, Dv_Lya_LyC[:,-1], c=f'C{iC}', label='Lya x LyC')
    ax.plot(redshifts, Dv_Lya_LyC[:,0], c=f'C{iC}', linestyle='--')
    iC += 1
    if output_iso:
        ax.fill_between(redshifts, Dv_iso[:,1], Dv_iso[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Dv_iso[:,-1], c=f'C{iC}', label='Iso')
        ax.plot(redshifts, Dv_iso[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_LyC:
        ax.fill_between(redshifts, Dv_LyC[:,1], Dv_LyC[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Dv_LyC[:,-1], c=f'C{iC}', label='LyC')
        ax.plot(redshifts, Dv_LyC[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_logLyC:
        ax.fill_between(redshifts, Dv_Lya_logLyC[:,1], Dv_Lya_logLyC[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Dv_Lya_logLyC[:,-1], c=f'C{iC}', label='Lya x logLyC')
        ax.plot(redshifts, Dv_Lya_logLyC[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_LyCn2:
        ax.fill_between(redshifts, Dv_Lya_LyCn2[:,1], Dv_Lya_LyCn2[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Dv_Lya_LyCn2[:,-1], c=f'C{iC}', label='Lya x LyC^1/2')
        ax.plot(redshifts, Dv_Lya_LyCn2[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_LyCn4:
        ax.fill_between(redshifts, Dv_Lya_LyCn4[:,1], Dv_Lya_LyCn4[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Dv_Lya_LyCn4[:,-1], c=f'C{iC}', label='Lya x LyC^1/4')
        ax.plot(redshifts, Dv_Lya_LyCn4[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_LyCn8:
        ax.fill_between(redshifts, Dv_Lya_LyCn8[:,1], Dv_Lya_LyCn8[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Dv_Lya_LyCn8[:,-1], c=f'C{iC}', label='Lya x LyC^1/8')
        ax.plot(redshifts, Dv_Lya_LyCn8[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_rankLyC:
        ax.fill_between(redshifts, Dv_Lya_rankLyC[:,1], Dv_Lya_rankLyC[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Dv_Lya_rankLyC[:,-1], c=f'C{iC}', label='Lya x rankLyC')
        ax.plot(redshifts, Dv_Lya_rankLyC[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    set_redshift_axis(ax)
    set_yaxis(ax, r'$\Delta v\ \,({\rm km/s})$', -100, 300, 5)
    ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    fig.savefig(f'fig_no/Dv.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

    # Plot Sv vs redshift
    fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
    iC = 0  # Color counter
    ax.fill_between(redshifts, Sv_Lya[:,1], Sv_Lya[:,2], alpha=0.2, color=f'C{iC}', lw=0)
    ax.plot(redshifts, Sv_Lya[:,-1], c=f'C{iC}', label='Lya')
    ax.plot(redshifts, Sv_Lya[:,0], c=f'C{iC}', linestyle='--')
    iC += 1
    ax.fill_between(redshifts, Sv_Lya_LyC[:,1], Sv_Lya_LyC[:,2], alpha=0.2, color=f'C{iC}', lw=0)
    ax.plot(redshifts, Sv_Lya_LyC[:,-1], c=f'C{iC}', label='Lya x LyC')
    ax.plot(redshifts, Sv_Lya_LyC[:,0], c=f'C{iC}', linestyle='--')
    iC += 1
    if output_iso:
        ax.fill_between(redshifts, Sv_iso[:,1], Sv_iso[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Sv_iso[:,-1], c=f'C{iC}', label='Iso')
        ax.plot(redshifts, Sv_iso[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_LyC:
        ax.fill_between(redshifts, Sv_LyC[:,1], Sv_LyC[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Sv_LyC[:,-1], c=f'C{iC}', label='LyC')
        ax.plot(redshifts, Sv_LyC[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_logLyC:
        ax.fill_between(redshifts, Sv_Lya_logLyC[:,1], Sv_Lya_logLyC[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Sv_Lya_logLyC[:,-1], c=f'C{iC}', label='Lya x logLyC')
        ax.plot(redshifts, Sv_Lya_logLyC[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_LyCn2:
        ax.fill_between(redshifts, Sv_Lya_LyCn2[:,1], Sv_Lya_LyCn2[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Sv_Lya_LyCn2[:,-1], c=f'C{iC}', label='Lya x LyC^1/2')
        ax.plot(redshifts, Sv_Lya_LyCn2[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_LyCn4:
        ax.fill_between(redshifts, Sv_Lya_LyCn4[:,1], Sv_Lya_LyCn4[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Sv_Lya_LyCn4[:,-1], c=f'C{iC}', label='Lya x LyC^1/4')
        ax.plot(redshifts, Sv_Lya_LyCn4[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_LyCn8:
        ax.fill_between(redshifts, Sv_Lya_LyCn8[:,1], Sv_Lya_LyCn8[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Sv_Lya_LyCn8[:,-1], c=f'C{iC}', label='Lya x LyC^1/8')
        ax.plot(redshifts, Sv_Lya_LyCn8[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_rankLyC:
        ax.fill_between(redshifts, Sv_Lya_rankLyC[:,1], Sv_Lya_rankLyC[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Sv_Lya_rankLyC[:,-1], c=f'C{iC}', label='Lya x rankLyC')
        ax.plot(redshifts, Sv_Lya_rankLyC[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    set_redshift_axis(ax)
    set_yaxis(ax, r'$\sigma_{\Delta v}\ \,({\rm km/s})$', 0, 500, 6)
    ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    fig.savefig(f'fig_no/Sv.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

    # Plot Dv_Lya - Dv_LyC vs redshift
    fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
    ax.axhline(y=0, color=[.8,.8,.8])
    iC = 0  # Color counter
    ax.fill_between(redshifts, Dv_Lya_LyC[:,1] - Dv_Lya[:,1], Dv_Lya_LyC[:,2] - Dv_Lya[:,2], alpha=0.2, color=f'C{iC}', lw=0)
    ax.plot(redshifts, Dv_Lya_LyC[:,-1] - Dv_Lya[:,-1], c=f'C{iC}', label='Lya x LyC - Lya')
    ax.plot(redshifts, Dv_Lya_LyC[:,0] - Dv_Lya[:,0], c=f'C{iC}', linestyle='--')
    iC += 1
    if output_iso:
        ax.fill_between(redshifts, Dv_Lya[:,1] - Dv_iso[:,1], Dv_Lya[:,2] - Dv_iso[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Dv_Lya[:,-1] - Dv_iso[:,-1], c=f'C{iC}', label='Lya - Iso')
        ax.plot(redshifts, Dv_Lya[:,0] - Dv_iso[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_iso and output_LyC:
        ax.fill_between(redshifts, Dv_LyC[:,1] - Dv_iso[:,1], Dv_LyC[:,2] - Dv_iso[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Dv_LyC[:,-1] - Dv_iso[:,-1], c=f'C{iC}', label='LyC - Iso')
        ax.plot(redshifts, Dv_LyC[:,0] - Dv_iso[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_LyC:
        ax.fill_between(redshifts, Dv_LyC[:,1] - Dv_Lya[:,1], Dv_LyC[:,2] - Dv_Lya[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Dv_LyC[:,-1] - Dv_Lya[:,-1], c=f'C{iC}', label='LyC - Lya')
        ax.plot(redshifts, Dv_LyC[:,0] - Dv_Lya[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_logLyC:
        ax.fill_between(redshifts, Dv_Lya_logLyC[:,1] - Dv_Lya[:,1], Dv_Lya_logLyC[:,2] - Dv_Lya[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Dv_Lya_logLyC[:,-1] - Dv_Lya[:,-1], c=f'C{iC}', label='Lya x logLyC - Lya')
        ax.plot(redshifts, Dv_Lya_logLyC[:,0] - Dv_Lya[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_LyCn2:
        ax.fill_between(redshifts, Dv_Lya_LyCn2[:,1] - Dv_Lya[:,1], Dv_Lya_LyCn2[:,2] - Dv_Lya[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Dv_Lya_LyCn2[:,-1] - Dv_Lya[:,-1], c=f'C{iC}', label='Lya x LyC^1/2 - Lya')
        ax.plot(redshifts, Dv_Lya_LyCn2[:,0] - Dv_Lya[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_LyCn4:
        ax.fill_between(redshifts, Dv_Lya_LyCn4[:,1] - Dv_Lya[:,1], Dv_Lya_LyCn4[:,2] - Dv_Lya[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Dv_Lya_LyCn4[:,-1] - Dv_Lya[:,-1], c=f'C{iC}', label='Lya x LyC^1/4 - Lya')
        ax.plot(redshifts, Dv_Lya_LyCn4[:,0] - Dv_Lya[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_LyCn8:
        ax.fill_between(redshifts, Dv_Lya_LyCn8[:,1] - Dv_Lya[:,1], Dv_Lya_LyCn8[:,2] - Dv_Lya[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Dv_Lya_LyCn8[:,-1] - Dv_Lya[:,-1], c=f'C{iC}', label='Lya x LyC^1/8 - Lya')
        ax.plot(redshifts, Dv_Lya_LyCn8[:,0] - Dv_Lya[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_rankLyC:
        ax.fill_between(redshifts, Dv_Lya_rankLyC[:,1] - Dv_Lya[:,1], Dv_Lya_rankLyC[:,2] - Dv_Lya[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Dv_Lya_rankLyC[:,-1] - Dv_Lya[:,-1], c=f'C{iC}', label='Lya x rankLyC - Lya')
        ax.plot(redshifts, Dv_Lya_rankLyC[:,0] - Dv_Lya[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    set_redshift_axis(ax)
    set_yaxis(ax, r'$\delta\Delta v\ \,({\rm km/s})$', -40, 40, 5)
    ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    fig.savefig(f'fig_no/Dv_diff.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

    # Plot Sv_Lya - Sv_LyC vs redshift
    fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
    ax.axhline(y=0, color=[.8,.8,.8])
    iC = 0  # Color counter
    ax.fill_between(redshifts, Sv_Lya_LyC[:,1] - Sv_Lya[:,1], Sv_Lya_LyC[:,2] - Sv_Lya[:,2], alpha=0.2, color=f'C{iC}', lw=0)
    ax.plot(redshifts, Sv_Lya_LyC[:,-1] - Sv_Lya[:,-1], c=f'C{iC}', label='Lya x LyC - Lya')
    ax.plot(redshifts, Sv_Lya_LyC[:,0] - Sv_Lya[:,0], c=f'C{iC}', linestyle='--')
    iC += 1
    if output_iso:
        ax.fill_between(redshifts, Sv_Lya[:,1] - Sv_iso[:,1], Sv_Lya[:,2] - Sv_iso[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Sv_Lya[:,-1] - Sv_iso[:,-1], c=f'C{iC}', label='Lya - Iso')
        ax.plot(redshifts, Sv_Lya[:,0] - Sv_iso[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_iso and output_LyC:
        ax.fill_between(redshifts, Sv_LyC[:,1] - Sv_iso[:,1], Sv_LyC[:,2] - Sv_iso[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Sv_LyC[:,-1] - Sv_iso[:,-1], c=f'C{iC}', label='LyC - Iso')
        ax.plot(redshifts, Sv_LyC[:,0] - Sv_iso[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_LyC:
        ax.fill_between(redshifts, Sv_LyC[:,1] - Sv_Lya[:,1], Sv_LyC[:,2] - Sv_Lya[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Sv_LyC[:,-1] - Sv_Lya[:,-1], c=f'C{iC}', label='LyC - Lya')
        ax.plot(redshifts, Sv_LyC[:,0] - Sv_Lya[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_logLyC:
        ax.fill_between(redshifts, Sv_Lya_logLyC[:,1] - Sv_Lya[:,1], Sv_Lya_logLyC[:,2] - Sv_Lya[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Sv_Lya_logLyC[:,-1] - Sv_Lya[:,-1], c=f'C{iC}', label='Lya x logLyC - Lya')
        ax.plot(redshifts, Sv_Lya_logLyC[:,0] - Sv_Lya[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_LyCn2:
        ax.fill_between(redshifts, Sv_Lya_LyCn2[:,1] - Sv_Lya[:,1], Sv_Lya_LyCn2[:,2] - Sv_Lya[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Sv_Lya_LyCn2[:,-1] - Sv_Lya[:,-1], c=f'C{iC}', label='Lya x LyC^1/2 - Lya')
        ax.plot(redshifts, Sv_Lya_LyCn2[:,0] - Sv_Lya[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_LyCn4:
        ax.fill_between(redshifts, Sv_Lya_LyCn4[:,1] - Sv_Lya[:,1], Sv_Lya_LyCn4[:,2] - Sv_Lya[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Sv_Lya_LyCn4[:,-1] - Sv_Lya[:,-1], c=f'C{iC}', label='Lya x LyC^1/4 - Lya')
        ax.plot(redshifts, Sv_Lya_LyCn4[:,0] - Sv_Lya[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_LyCn8:
        ax.fill_between(redshifts, Sv_Lya_LyCn8[:,1] - Sv_Lya[:,1], Sv_Lya_LyCn8[:,2] - Sv_Lya[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Sv_Lya_LyCn8[:,-1] - Sv_Lya[:,-1], c=f'C{iC}', label='Lya x LyC^1/8 - Lya')
        ax.plot(redshifts, Sv_Lya_LyCn8[:,0] - Sv_Lya[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    if output_rankLyC:
        ax.fill_between(redshifts, Sv_Lya_rankLyC[:,1] - Sv_Lya[:,1], Sv_Lya_rankLyC[:,2] - Sv_Lya[:,2], alpha=0.2, color=f'C{iC}', lw=0)
        ax.plot(redshifts, Sv_Lya_rankLyC[:,-1] - Sv_Lya[:,-1], c=f'C{iC}', label='Lya x rankLyC - Lya')
        ax.plot(redshifts, Sv_Lya_rankLyC[:,0] - Sv_Lya[:,0], c=f'C{iC}', linestyle='--')
        iC += 1
    set_redshift_axis(ax)
    set_yaxis(ax, r'$\delta\sigma_{\Delta v}\ \,({\rm km/s})$', -40, 40, 5)
    ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    fig.savefig(f'fig_no/Sv_diff.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

    # Plot Sv_Lya - Sv_LyC vs Dv_Lya - Dv_LyC
    fig = plt.figure(figsize=(4.5,4.5)); ax = plt.axes([0,0,1,1])
    ax.axhline(y=0, color=[.8,.8,.8], zorder=-10)
    ax.axvline(x=0, color=[.8,.8,.8], zorder=-10)
    iC = 0  # Color counter
    ax.scatter(Dv_Lya_LyC[:,-1] - Dv_Lya[:,-1], Sv_Lya_LyC[:,-1] - Sv_Lya[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x LyC - Lya')
    ax.scatter(np.median(Dv_Lya_LyC[:,-1] - Dv_Lya[:,-1]), np.median(Sv_Lya_LyC[:,-1] - Sv_Lya[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
    iC += 1
    if output_iso:
        ax.scatter(Dv_Lya[:,-1] - Dv_iso[:,-1], Sv_Lya[:,-1] - Sv_iso[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya - Iso')
        ax.scatter(np.median(Dv_Lya[:,-1] - Dv_iso[:,-1]), np.median(Sv_Lya[:,-1] - Sv_iso[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_iso and output_LyC:
        ax.scatter(Dv_LyC[:,-1] - Dv_iso[:,-1], Sv_LyC[:,-1] - Sv_iso[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='LyC - Iso')
        ax.scatter(np.median(Dv_LyC[:,-1] - Dv_iso[:,-1]), np.median(Sv_LyC[:,-1] - Sv_iso[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_LyC:
        ax.scatter(Dv_LyC[:,-1] - Dv_Lya[:,-1], Sv_LyC[:,-1] - Sv_Lya[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='LyC - Lya')
        ax.scatter(np.median(Dv_LyC[:,-1] - Dv_Lya[:,-1]), np.median(Sv_LyC[:,-1] - Sv_Lya[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_logLyC:
        ax.scatter(Dv_Lya_logLyC[:,-1] - Dv_Lya[:,-1], Sv_Lya_logLyC[:,-1] - Sv_Lya[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x logLyC - Lya')
        ax.scatter(np.median(Dv_Lya_logLyC[:,-1] - Dv_Lya[:,-1]), np.median(Sv_Lya_logLyC[:,-1] - Sv_Lya[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_LyCn2:
        ax.scatter(Dv_Lya_LyCn2[:,-1] - Dv_Lya[:,-1], Sv_Lya_LyCn2[:,-1] - Sv_Lya[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x LyC^1/2 - Lya')
        ax.scatter(np.median(Dv_Lya_LyCn2[:,-1] - Dv_Lya[:,-1]), np.median(Sv_Lya_LyCn2[:,-1] - Sv_Lya[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_LyCn4:
        ax.scatter(Dv_Lya_LyCn4[:,-1] - Dv_Lya[:,-1], Sv_Lya_LyCn4[:,-1] - Sv_Lya[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x LyC^1/4 - Lya')
        ax.scatter(np.median(Dv_Lya_LyCn4[:,-1] - Dv_Lya[:,-1]), np.median(Sv_Lya_LyCn4[:,-1] - Sv_Lya[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_LyCn8:
        ax.scatter(Dv_Lya_LyCn8[:,-1] - Dv_Lya[:,-1], Sv_Lya_LyCn8[:,-1] - Sv_Lya[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x LyC^1/8 - Lya')
        ax.scatter(np.median(Dv_Lya_LyCn8[:,-1] - Dv_Lya[:,-1]), np.median(Sv_Lya_LyCn8[:,-1] - Sv_Lya[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_rankLyC:
        ax.scatter(Dv_Lya_rankLyC[:,-1] - Dv_Lya[:,-1], Sv_Lya_rankLyC[:,-1] - Sv_Lya[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x rankLyC - Lya')
        ax.scatter(np.median(Dv_Lya_rankLyC[:,-1] - Dv_Lya[:,-1]), np.median(Sv_Lya_rankLyC[:,-1] - Sv_Lya[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    set_xaxis(ax, r'$\delta\Delta v\ \,({\rm km/s})$', -40, 40, 5)
    set_yaxis(ax, r'$\delta\sigma_{\Delta v}\ \,({\rm km/s})$', -40, 40, 5)
    ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    fig.savefig(f'fig_yes/Dv_Sv_diff.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

    # Plot Dv_Lya - Dv_LyC vs Dv_Lya
    fig = plt.figure(figsize=(4.5,4.5)); ax = plt.axes([0,0,1,1])
    ax.axhline(y=0, color=[.8,.8,.8], zorder=-10)
    ax.axvline(x=0, color=[.8,.8,.8], zorder=-10)
    iC = 0  # Color counter
    ax.scatter(Dv_Lya[:,-1], Dv_Lya_LyC[:,-1] - Dv_Lya[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x LyC - Lya')
    ax.scatter(np.median(Dv_Lya[:,-1]), np.median(Dv_Lya_LyC[:,-1] - Dv_Lya[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
    iC += 1
    if output_iso:
        ax.scatter(Dv_iso[:,-1], Dv_Lya[:,-1] - Dv_iso[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya - Iso')
        ax.scatter(np.median(Dv_iso[:,-1]), np.median(Dv_Lya[:,-1] - Dv_iso[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_iso and output_LyC:
        ax.scatter(Dv_iso[:,-1], Dv_LyC[:,-1] - Dv_iso[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='LyC - Iso')
        ax.scatter(np.median(Dv_iso[:,-1]), np.median(Dv_LyC[:,-1] - Dv_iso[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_LyC:
        ax.scatter(Dv_Lya[:,-1], Dv_LyC[:,-1] - Dv_Lya[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='LyC - Lya')
        ax.scatter(np.median(Dv_Lya[:,-1]), np.median(Dv_LyC[:,-1] - Dv_Lya[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_logLyC:
        ax.scatter(Dv_Lya[:,-1], Dv_Lya_logLyC[:,-1] - Dv_Lya[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x logLyC - Lya')
        ax.scatter(np.median(Dv_Lya[:,-1]), np.median(Dv_Lya_logLyC[:,-1] - Dv_Lya[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_LyCn2:
        ax.scatter(Dv_Lya[:,-1], Dv_Lya_LyCn2[:,-1] - Dv_Lya[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x LyC^1/2 - Lya')
        ax.scatter(np.median(Dv_Lya[:,-1]), np.median(Dv_Lya_LyCn2[:,-1] - Dv_Lya[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_LyCn4:
        ax.scatter(Dv_Lya[:,-1], Dv_Lya_LyCn4[:,-1] - Dv_Lya[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x LyC^1/4 - Lya')
        ax.scatter(np.median(Dv_Lya[:,-1]), np.median(Dv_Lya_LyCn4[:,-1] - Dv_Lya[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_LyCn8:
        ax.scatter(Dv_Lya[:,-1], Dv_Lya_LyCn8[:,-1] - Dv_Lya[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x LyC^1/8 - Lya')
        ax.scatter(np.median(Dv_Lya[:,-1]), np.median(Dv_Lya_LyCn8[:,-1] - Dv_Lya[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_rankLyC:
        ax.scatter(Dv_Lya[:,-1], Dv_Lya_rankLyC[:,-1] - Dv_Lya[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x rankLyC - Lya')
        ax.scatter(np.median(Dv_Lya[:,-1]), np.median(Dv_Lya_rankLyC[:,-1] - Dv_Lya[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    set_xaxis(ax, r'$\Delta v\ \,({\rm km/s})$') #, -40, 40, 5)
    set_yaxis(ax, r'$\delta\Delta v\ \,({\rm km/s})$', -40, 40, 5)
    ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    fig.savefig(f'fig_yes/Dv_v_diff.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

    # Plot (f_esc_Lya, f_esc_LyC, f_Lya_LyC) vs redshift
    fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
    # ax.axhline(y=1, color=[.8,.8,.8])
    iC = 0  # Color counter
    ax.plot(redshifts, f_esc_Lya, c=f'C{iC}', label=r'$f_{\rm esc}^{\rm Ly\alpha}$')
    ax.plot(redshifts, s_esc_Lya, c=f'C{iC}', ls='--', label=r'$\sigma_{\rm esc}^{\rm Ly\alpha}$')
    iC += 1
    ax.plot(redshifts, f_esc_LyC, c=f'C{iC}', label=r'$f_{\rm esc}^{\rm LyC}$')
    ax.plot(redshifts, s_esc_LyC, c=f'C{iC}', ls='--', label=r'$\sigma_{\rm esc}^{\rm LyC}$')
    iC += 1
    set_redshift_axis(ax)
    set_log_yaxis(ax, r'${\rm LyC\ Escape\ Fraction}$', 1e-3, 1)
    ax.legend(loc='lower left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12, ncol=2)
    fig.savefig(f'fig_yes/f_esc.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

    # Plot CoV vs redshift
    fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
    ax.axhline(y=1, color=[.8,.8,.8])
    iC = 0  # Color counter
    ax.plot(redshifts, CoV_esc_Lya, c=f'C{iC}', label=r'${\rm Ly\alpha}$')
    iC += 1
    ax.plot(redshifts, CoV_esc_LyC, c=f'C{iC}', label=r'${\rm LyC}$')
    iC += 1
    ax.plot(redshifts, rho_Lya_LyC, c=f'C{iC}', label=r'$\rho_{\rm Ly\alpha \times LyC}$')
    iC += 1
    set_redshift_axis(ax)
    set_log_yaxis(ax, r'${\rm CoV} = \sigma / \mu$', 0.1, 10)
    ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    fig.savefig(f'fig_yes/CoV.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

    # Plot CoV_Lya vs CoV_LyC
    fig = plt.figure(figsize=(4.5,4.5)); ax = plt.axes([0,0,1,1])
    iC = 0  # Color counter
    ax.scatter(CoV_esc_Lya, CoV_esc_LyC, color=f'C{iC}', alpha=0.4, lw=0)
    ax.scatter(np.median(CoV_esc_Lya), np.median(CoV_esc_LyC), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
    iC += 1
    set_log_xaxis(ax, r'$\rm CoV(\rm Ly\alpha)$', 0.1, 1)
    set_log_yaxis(ax, r'$\rm CoV(\rm LyC)$', 0.3, 10)
    ax.minorticks_on()
    fig.savefig(f'fig_yes/CoV_Lya_vs_CoV_LyC.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

    # Plot f_esc_Lya vs f_esc_LyC
    fig = plt.figure(figsize=(4.5,4.5)); ax = plt.axes([0,0,1,1])
    iC = 0  # Color counter
    ax.scatter(f_esc_Lya, f_esc_LyC, color=f'C{iC}', alpha=0.4, lw=0)
    ax.scatter(np.median(f_esc_Lya), np.median(f_esc_LyC), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
    iC += 1
    set_xaxis(ax, r'${\rm Ly\alpha\ Escape\ Fraction}$', 0.5, 1, 6)
    set_log_yaxis(ax, r'${\rm LyC\ Escape\ Fraction}$', 1e-3, 1)
    ax.minorticks_on()
    fig.savefig(f'fig_yes/f_esc_Lya_vs_f_esc_LyC.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

    # Plot rho_Lya_LyC vs f_esc_LyC
    fig = plt.figure(figsize=(4.5,4.5)); ax = plt.axes([0,0,1,1])
    iC = 0  # Color counter
    ax.scatter(rho_Lya_LyC, f_esc_LyC, color=f'C{iC}', alpha=0.4, lw=0)
    ax.scatter(np.median(rho_Lya_LyC), np.median(f_esc_LyC), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
    iC += 1
    set_xaxis(ax, r'$\rho_{\rm Ly\alpha \times LyC}$', 0., 1, 6)
    set_log_yaxis(ax, r'${\rm LyC\ Escape\ Fraction}$', 1e-3, 1)
    ax.minorticks_on()
    fig.savefig(f'fig_yes/rho_Lya_LyC_vs_f_esc_LyC.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

    # Plot Dv vs Sv
    fig = plt.figure(figsize=(4.5,4.5)); ax = plt.axes([0,0,1,1])
    iC = 0  # Color counter
    ax.scatter(Dv_Lya[:,-1], Sv_Lya[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya')
    ax.scatter(np.median(Dv_Lya[:,-1]), np.median(Sv_Lya[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
    iC += 1
    ax.scatter(Dv_Lya_LyC[:,-1], Sv_Lya_LyC[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x LyC')
    ax.scatter(np.median(Dv_Lya_LyC[:,-1]), np.median(Sv_Lya_LyC[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
    iC += 1
    if output_iso:
        ax.scatter(Dv_iso[:,-1], Sv_iso[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Iso')
        ax.scatter(np.median(Dv_iso[:,-1]), np.median(Sv_iso[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_LyC:
        ax.scatter(Dv_LyC[:,-1], Sv_LyC[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='LyC')
        ax.scatter(np.median(Dv_LyC[:,-1]), np.median(Sv_LyC[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_logLyC:
        ax.scatter(Dv_Lya_logLyC[:,-1], Sv_Lya_logLyC[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x logLyC')
        ax.scatter(np.median(Dv_Lya_logLyC[:,-1]), np.median(Sv_Lya_logLyC[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_LyCn2:
        ax.scatter(Dv_Lya_LyCn2[:,-1], Sv_Lya_LyCn2[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x LyC^1/2')
        ax.scatter(np.median(Dv_Lya_LyCn2[:,-1]), np.median(Sv_Lya_LyCn2[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_LyCn4:
        ax.scatter(Dv_Lya_LyCn4[:,-1], Sv_Lya_LyCn4[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x LyC^1/4')
        ax.scatter(np.median(Dv_Lya_LyCn4[:,-1]), np.median(Sv_Lya_LyCn4[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_LyCn8:
        ax.scatter(Dv_Lya_LyCn8[:,-1], Sv_Lya_LyCn8[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x LyC^1/8')
        ax.scatter(np.median(Dv_Lya_LyCn8[:,-1]), np.median(Sv_Lya_LyCn8[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    if output_rankLyC:
        ax.scatter(Dv_Lya_rankLyC[:,-1], Sv_Lya_rankLyC[:,-1], color=f'C{iC}', alpha=0.4, lw=0, label='Lya x rankLyC')
        ax.scatter(np.median(Dv_Lya_rankLyC[:,-1]), np.median(Sv_Lya_rankLyC[:,-1]), color=f'C{iC}', marker='x', s=100, lw=2, zorder=10)
        iC += 1
    # set_log_xaxis(ax, r'${\rm Velocity\ Offset\ \,(km/s)}$', 1, 1e3)
    # set_log_yaxis(ax, r'${\rm Velocity\ Deviation\ \,(km/s)}$', 1e2, 1e3)
    set_xaxis(ax, r'${\rm Velocity\ Offset\ \,(km/s)}$', 0, 300, 4)
    set_yaxis(ax, r'${\rm Velocity\ Deviation\ \,(km/s)}$', 200, 500, 4)
    ax.minorticks_on()
    fig.savefig(f'fig_yes/Dv_vs_Sv.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

# sim, run = 'g2', 'z4'
# sim, run = 'g39', 'z4'
# sim, run = 'g205', 'z4'
# sim, run = 'g578', 'z4'
# sim, run = 'g1163', 'z4'
sim, run = 'g5760', 'z8'
# sim, run = 'g10304', 'z8'
# sim, run = 'g33206', 'z8'
# sim, run = 'g37591', 'z8'
# sim, run = 'g137030', 'z16'
# sim, run = 'g500531', 'z16'
# sim, run = 'g519761', 'z16'
# sim, run = 'g2274036', 'z16'
# sim, run = 'g5229300', 'z16'

# zip_data(sim=sim, run=run, snaps=[168], plot_map=True)
# zip_data(sim=sim, run=run)
plot_Dv(sim=sim, run=run)
