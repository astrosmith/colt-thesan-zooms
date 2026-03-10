import numpy as np
import h5py, os
import healpy as hp
from healpy import projaxes as PA
from healpy import pixelfunc
from healpy import rotator as R
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib import patheffects
import cmasher as cmr
from multiprocessing import Pool
import functools
import cmocean
import copy

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

# v8 = np.array([
#     [ 2.97565697808504448e-01, -5.71513322854897837e-02,  9.52989181840870114e-01],
#     [ 9.29164837244215058e-01,  3.69630725230415735e-01,  5.08253820136726070e-03],
#     [-1.33877462816566456e-01, -9.56158874726533048e-01,  2.60455430412745936e-01],
#     [-8.43331075342676373e-01, -8.15384195319966525e-03,  5.37332496898118905e-01],
#     [-9.26625998461563238e-02,  9.29051551922226992e-01,  3.58157586071623513e-01],
#     [ 4.84659160656948293e-02,  5.94370605805620622e-01, -8.02729492378455367e-01],
#     [ 4.97299461468477366e-01, -5.29065431566931199e-01, -6.87592186360533986e-01],
#     [-7.02624774581492195e-01, -3.42523402426109791e-01, -6.23695554685736564e-01],
# ])  # Camera direction vectors
v8 = np.zeros([8,3])  # Camera direction vectors
x8 = np.zeros([8,3])  # Camera x-axis vectors
y8 = np.zeros([8,3])  # Camera y-axis vectors
p8 = np.zeros([8,2])  # Projected image directions to healpix rotation vector
mu8 = np.zeros(8)  # Angle to the healpix focus

names = {'g2': 'm13.0', 'g39': 'm12.6', 'g205': 'm12.2', 'g578': 'm11.9', 'g1163': 'm11.5',
         'g5760': 'm11.1', 'g10304': 'm10.8', 'g33206': 'm10.4', 'g37591': 'm10.0', 'g137030': 'm9.7',
         'g500531': 'm9.3', 'g519761': 'm8.9', 'g2274036': 'm8.5', 'g5229300': 'm8.2'}

class MapData:
    def __init__(self, field, key, u_str, z_str=None, w_str='', lims=None, cmap=None, norm=None, units=1., n_format=2, log=False):
        self.data = None
        self.field = field
        self.key = key
        self.u_str = u_str
        self.z_str = z_str
        self.w_str = w_str
        self.lims = lims
        self.cmap = cmap
        self.norm = norm
        self.units = units
        self.n_format = n_format
        self.log = log

class ImageData:
    def __init__(self, field, key, s_key, s_fmt='', lims=None, cmap=None, norm=None, f_cut=None, target=0.2, units=1., s_units=1., n_degrade=0, cutout=False):
        self.data = None
        self.stat = None
        self.wp = None
        self.mean = None
        self.min = None
        self.max = None
        self.s_fmt = s_fmt
        self.field = field
        self.key = key
        self.s_key = s_key
        self.lims = lims
        self.cmap = cmap
        self.norm = norm
        self.f_cut = f_cut
        self.target = target
        self.units = units
        self.s_units = s_units
        self.n_degrade = n_degrade
        self.cutout = cutout

IonMap = MapData(field='ion-eq', key='map', u_str=r'$f_{\rm esc}^{\rm\,LyC}\ \ (\%)$', lims='[0,max]', cmap=cmr.ember, norm=None, units=0.01, n_format=0)
LyaMap = MapData(field='Lya', key='map', u_str=r'$f_{\rm esc}^{\rm\,Ly\alpha}\ \ (\%)$', lims='[0,max]', cmap=plt.cm.afmhot, norm=None, units=0.01, n_format=0)
Lya1Map = MapData(field='Lya', key='freq_map', u_str=r'$\langle \Delta v_{\rm{Ly}\alpha} \rangle\ \ (\rm{km}/\rm{s})$', lims=[-100,100], cmap=cmocean.cm.balance, norm=None, n_format=0)
Lya2Map = MapData(field='Lya', key='freq2_map', u_str=r'$\sigma_{\Delta v,{\rm{Ly}\alpha}}\ \ (\rm{km}/\rm{s})$', lims='[min,max]', cmap=plt.cm.spring, norm=None, n_format=0)
HI_int_Map = MapData(field='ion-esc', key='HI_columns_int', u_str=r'$\log N_{\rm{HI}}^{\rm int}\ \ (\rm{cm}^{-2})$', lims='[min,max]', cmap=cmr.savanna, norm=None, units=H_units, n_format=1, log=True)
HI_esc_Map = MapData(field='ion-esc', key='HI_columns_esc', u_str=r'$\log N_{\rm{HI}}^{\rm esc}\ \ (\rm{cm}^{-2})$', lims='[min,max]', cmap=cmr.savanna, norm=None, units=H_units, n_format=1, log=True)
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

nH_Image = ImageData(field='proj', key='proj_nH_sum', s_key=None, lims=None, cmap=cmr.ember, norm=None, f_cut=1e-10, target=0.15)
HI_Image = ImageData(field='proj', key='proj_xHI_nH', s_key=None, lims=None, cmap=cmr.ember, norm=None, f_cut=1e-10, target=0.15)
IonImage = ImageData(field='ion-eq', key='images', s_key='f_escs', s_fmt='${:.2e}\\%$', lims=None, cmap=plt.cm.afmhot, norm=None, f_cut=1e-6, s_units=0.01, n_degrade=1)
MUVImage = ImageData(field='M1500', key='images', s_key='f_escs', s_fmt='${:.0f}\\%$', lims=None, cmap=plt.cm.inferno, norm=None, f_cut=1e-5, target=0.15, s_units=0.01, n_degrade=1)
OptImage = ImageData(field='optical', key='images', s_key='f_escs', s_fmt='${:.0f}\\%$', lims=None, cmap=plt.cm.inferno, norm=None, f_cut=4e-5, target=0.15, s_units=0.01, n_degrade=1)
LyaImage = ImageData(field='Lya', key='images', s_key='f_escs', s_fmt='${:.0f}\\%$', lims=None, cmap=plt.cm.afmhot, norm=None, f_cut=1e-5, target=0.2, s_units=0.01, n_degrade=2)
Lya1Image = ImageData(field='Lya', key='freq_images', s_key='freq_avgs', s_fmt='${:.0f}$', lims=[-300,300], cmap=cmocean.cm.balance, norm=None, f_cut=1e-5, target=0.2, n_degrade=2)
Lya2Image = ImageData(field='Lya', key='freq2_images', s_key='freq_stds', s_fmt='${:.0f}$', lims=[0,500], cmap=plt.cm.spring, norm=None, f_cut=1e-5, target=0.2, n_degrade=2)
HaImage = ImageData(field='Ha', key='images', s_key='f_escs', s_fmt='${:.2e}\\%$', lims=None, cmap=plt.cm.afmhot, norm=None, f_cut=1e-5, target=0.2, s_units=0.01, cutout=True)
Ha1Image = ImageData(field='Ha', key='freq_images', s_key='freq_avgs', s_fmt='${:.0f}$', lims=[-100,100], cmap=cmocean.cm.balance, norm=None, f_cut=1e-5, target=0.2, cutout=True)
Ha2Image = ImageData(field='Ha', key='freq2_images', s_key='freq_stds', s_fmt='${:.0f}$', lims=[0,400], cmap=plt.cm.spring, norm=None, f_cut=1e-5, target=0.2, cutout=True)
HbImage = ImageData(field='Hb', key='images', s_key='f_escs', s_fmt='${:.2e}\\%$', lims=None, cmap=plt.cm.afmhot, norm=None, f_cut=1e-5, target=0.2, s_units=0.01, cutout=True)
O3Image = ImageData(field='OIII-5008', key='images', s_key='f_escs', s_fmt='${:.2e}\\%$', lims=None, cmap=cmr.ember, norm=None, f_cut=1e-5, target=0.2, s_units=0.01, cutout=True)
O2Image = ImageData(field='OII-3727-3730', key='images', s_key='f_escs', s_fmt='${:.2e}\\%$', lims=None, cmap=cmr.ember, norm=None, f_cut=1e-5, target=0.2, s_units=0.01, cutout=True)

def safe_ratio(numerator, denominator, zero=0.):
    ratio = np.copy(numerator)
    mask = denominator > 0.
    ratio[mask] /= denominator[mask]
    ratio[~mask] = zero
    return ratio

def get_alpha0(data, Z0_max, f_cut=1e-4, f_sat=1.):
    assert f_sat <= 1., f'f_sat = {f_sat:g}'
    Z_i = data / (Z0_max * f_sat)
    Z_i[Z_i<f_cut] = f_cut
    if f_sat < 1.: Z_i[Z_i>1.] = 1.
    Z_i = 1. - np.log10(Z_i) / np.log10(f_cut)
    Z_avg = np.mean(Z_i)
    # print(f'f_cut = {f_cut:g}, f_sat = {f_sat:g}, Z_avg = {Z_avg:g}')
    return Z_i

def get_alpha(data, Z0_max, f_cut=1e-4, f_sat=1., target=0.2, rtol=0.05, boost=1.15, max_iter=100):
    tmin, tmax = target * (1. - rtol), target * (1. + rtol)
    count, Z_avg_prev = 0, 0.
    while count < max_iter:
        Z_i = data / (Z0_max * f_sat)
        Z_i[Z_i<f_cut] = f_cut
        if f_sat < 1.: Z_i[Z_i>1.] = 1.
        Z_i = 1. - np.log10(Z_i) / np.log10(f_cut)
        Z_avg = np.mean(Z_i)
        if Z_avg < tmin:
            f_cut /= boost
        elif Z_avg > tmax:
            f_cut *= boost
        else:
            break
        if np.abs(1. - Z_avg_prev / Z_avg) < 1e-3:
            break
        Z_avg_prev = Z_avg
        count += 1
    # print(f'f_cut = {f_cut:g}, f_sat = {f_sat:g}, Z_avg = {Z_avg:g}')
    return Z_i

def get_alpha_p(Z, p_min=1., p_max=99.99):
    flat = Z.flatten()
    Z_min, Z_max = weighted_percentile(flat[flat>0.], np.ones_like(flat[flat>0.]), [p_min, p_max])
    f_sat = Z_max / np.max(flat)
    f_cut = Z_min / Z_max
    Z /= Z_max
    Z[Z>1.] = 1.
    Z[Z<f_cut] = f_cut
    # print(f'f_cut = {f_cut:g}, f_sat = {f_sat:g}, p_min = {p_min:g}, p_max = {p_max:g}')
    return (1. - np.log10(Z) / np.log10(f_cut))

def get_rgba_HI(Z, A):
    rgba = np.zeros([Z.shape[0], Z.shape[1], 4])
    rgba[:, :, 3] = A.T
    Z0, ZH, Z1 = 0., .5, 1.
    R0, RH, R1 = .1, .9, 1.
    G0, GH, G1 = .9, .2, .9
    B0, BH, B1 = 1., .9, .1
    mask = (Z < 0.5)
    rgba[mask, 0] = R0 + (Z[mask] - Z0) * (RH - R0) / (ZH - Z0)
    rgba[~mask, 0] = R1 + (Z[~mask] - Z1) * (RH - R1) / (ZH - Z1)
    rgba[mask, 1] = G0 + (Z[mask] - Z0) * (GH - G0) / (ZH - Z0)
    rgba[~mask, 1] = G1 + (Z[~mask] - Z1) * (GH - G1) / (ZH - Z1)
    rgba[mask, 2] = B0 + (Z[mask] - Z0) * (BH - B0) / (ZH - Z0)
    rgba[~mask, 2] = B1 + (Z[~mask] - Z1) * (BH - B1) / (ZH - Z1)
    return rgba

def spherical_coords(X):
    # theta: colatitude [0,pi], phi: longitude [-pi,pi)
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return theta, phi

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

def log_minorticks(vmin, vmax):
    expA = np.floor(np.log10(vmin))
    expB = np.floor(np.log10(vmax))
    cofA = np.ceil(vmin/10**expA)
    cofB = np.floor(vmax/10**expB)
    lmt = []
    while cofA*10**expA <= cofB*10**expB:
        if expA < expB:
            lmt = np.hstack( (lmt, np.linspace(cofA, 9, 10-cofA)*10**expA) )
            cofA = 1
            expA += 1
        else:
            lmt = np.hstack( (lmt, np.linspace(cofA, cofB, cofB-cofA+1)*10**expA) )
            expA += 1
    return lmt

def colorbar(image, label, position='right'):
    vmin, vmax = image.get_clim()
    Lvmin, Lvmax = np.ceil(np.log10(vmin)), np.floor(np.log10(vmax))
    Lcticks = np.linspace(Lvmin, Lvmax, Lvmax-Lvmin+1)
    minorticks = image.norm( log_minorticks(vmin, vmax) )
    if position == 'right':
        cax = image.figure.add_axes([1, 0, .03, 1])
        cbar = image.figure.colorbar(image, cax=cax, ticks=10**Lcticks, orientation='vertical')
        cbar.ax.yaxis.set_ticks(minorticks, minor=True)
    elif position == 'top':
        cax = image.figure.add_axes([0, 1, 1, .04])
        cbar = image.figure.colorbar(image, cax=cax, ticks=10**Lcticks, orientation='horizontal')
        cbar.ax.xaxis.set_ticks(minorticks, minor=True)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
    else:
        raise ValueError('position not recognized!')
    cax.tick_params(which='both', direction='in', labelsize=9)
    cbar.set_label(label) #, fontsize=14, labelpad=8)
    # cbar.ax.set_yticklabels([r'$10^{%g}$' % val for val in Lcticks], fontsize=13)
    # cbar.solids.set_edgecolor("face")
    # locs = [0, 1];  ax.set_xticks(locs);  ax.set_yticks(locs)

def _dipole_rot_from_map(m):
    nside = hp.npix2nside(m.size)
    ipix = np.arange(m.size)
    vx, vy, vz = hp.pix2vec(nside, ipix)
    w = np.asarray(m, dtype=float)
    bad = (~np.isfinite(w)) | (w == hp.UNSEEN)
    w = w.copy()
    w[bad] = 0.0
    d = np.array([np.sum(w * vx), np.sum(w * vy), np.sum(w * vz)])
    norm = np.linalg.norm(d)
    if not np.isfinite(norm) or norm == 0.0:
        return None
    d /= norm
    lon = np.degrees(np.arctan2(d[1], d[0]))
    lat = np.degrees(np.arcsin(d[2]))
    return (lon, lat, 0.0)

def _smoothed_max_rot_from_map(m, smooth_pix=4.):
    nside = hp.npix2nside(m.size)
    w = np.asarray(m, dtype=float)
    bad = (~np.isfinite(w)) | (w == hp.UNSEEN)
    mm = hp.ma(w)
    mm.mask = bad
    fwhm = float(smooth_pix) * hp.nside2resol(nside)
    sm = hp.smoothing(mm, fwhm=fwhm)
    ip = int(np.ma.argmax(sm))
    theta, phi = hp.pix2ang(nside, ip)
    lon = np.degrees(phi)
    lat = 90.0 - np.degrees(theta)
    return (lon, lat, 0.0)

def _proj_xy(ax, theta, phi, rot=None):
    vec = R.dir2vec(theta, phi, lonlat=False)
    if rot is not None:
        vec = R.Rotator(rot=rot)(vec)
    x, y = ax.proj.vec2xy(vec, direct=False)
    return x, y

def get_rot(data, recenter_on_dipole=False, recenter_on_smoothed_max=False, smooth_pix=4.):
    map = pixelfunc.ma_to_array(data)
    if recenter_on_smoothed_max:
        return _smoothed_max_rot_from_map(map, smooth_pix=smooth_pix)
    elif recenter_on_dipole:
        return _dipole_rot_from_map(map)
    else:
        return None

def HpPlot(f, extent, m, remove_dip=False, rot=None, add_vectors=False):
    ## f = plt.figure(figsize=(8.5,5.4))
    ## extent = (0.02,0.05,0.96,0.9)
    ## f = plt.figure(figsize=(3.,2.), dpi=sargs['dpi'])
    # f = plt.figure(figsize=(3.,2.))
    # extent = (0,0,1,1)
    map = pixelfunc.ma_to_array(m.data)
    ax = PA.HpxMollweideAxes(f, extent)
    f.add_axes(ax)
    if remove_dip:
        map = pixelfunc.remove_dipole(map)
    m.cmap.set_under('w')
    if m.lims is None:
        img = ax.projmap(map, cmap=m.cmap, rot=rot) #vmin=min,vmax=max,norm=norm)
    else:
        img = ax.projmap(map, cmap=m.cmap, rot=rot, vmin=m.lims[0], vmax=m.lims[1]) #,norm=norm)
    im = ax.get_images()[0]
    b = im.norm.inverse(np.linspace(0,1,im.cmap.N+1))
    v = np.linspace(im.norm.vmin,im.norm.vmax,im.cmap.N)
    cb = f.colorbar(im, ax=ax, orientation='horizontal',
                    shrink=0.75, aspect=25, ticks=PA.BoundaryLocator(),
                    pad=0.05, fraction=0.1, boundaries=b, values=v,
                    format=r'${\rm '+str('%0.'+str(m.n_format)+'f')+r'}$')
    cb.solids.set_rasterized(True)
    # ax.set_title('')
    if remove_dip:
        # ax.text(0.125, 0, r'${\rm Dipole}$'+'\n'+r'${\rm Removed}$', fontsize=13,
        #         fontweight='bold', ha='center', va='center', transform=ax.transAxes)
        ax.text(0.01, -.025, r'${\rm No\ Dipole}$', fontsize=12, fontweight='bold', ha='left', va='baseline', transform=ax.transAxes)
    if m.w_str is not None:
        ax.text(0.875, -.025, m.w_str, fontsize=12, ha='center', va='baseline', transform=ax.transAxes)
    if m.z_str is not None:
        ax.text(0.125, -.025, m.z_str, fontsize=12, ha='center', va='baseline', transform=ax.transAxes)
    if m.u_str is not None:
        cb.ax.text(0.5, -2.0, m.u_str, fontsize=14.5, transform=cb.ax.transAxes, ha='center', va='center')
    if add_vectors:
        theta, phi = spherical_coords(v8)
        outline_kwargs = {'path_effects': [patheffects.withStroke(linewidth=2, foreground='k')]}
        for i in range(len(theta)):
            x, y = _proj_xy(ax, theta[i], phi[i], rot=rot)
            ax.text(x, y, r'${\bf %d}$' % (i+1), color='w', ha='center', va='center', fontsize=7, **outline_kwargs)
    f.sca(ax)

def Hp2Plot(f, extent, m, m0, rot=None, add_vectors=False):
    map = pixelfunc.ma_to_array(m.data)
    map0 = pixelfunc.ma_to_array(m0.data)
    ax = PA.HpxMollweideAxes(f, extent)
    f.add_axes(ax)
    m.cmap.set_under('w')
    # f_cut = m0.f_cut
    # Z0[Z0<f_cut] = f_cut
    # alpha = 1. - np.log10(Z0) / np.log10(f_cut)
    # ax.projmap(back_map, rot=rot)

    # Alpha map
    Z0_max = np.max(m0.data)
    Z0 = map0 / Z0_max
    Z0 = np.clip(Z0, 0.0, 1.0)
    ax.projmap(np.zeros_like(map), cmap='grey', rot=rot, vmin=0, vmax=1)
    if m.lims is None:
        img = ax.projmap(map, alpha=Z0, cmap=m.cmap, rot=rot) # vmin=min, vmax=max, norm=norm)
    else:
        img = ax.projmap(map, alpha=Z0, cmap=m.cmap, rot=rot, vmin=m.lims[0], vmax=m.lims[1]) #, norm=norm)

    # Code for alpha colorbars
    im = ax.get_images()[-1]
    b = im.norm.inverse(np.linspace(0, 1, im.cmap.N + 1))
    v = np.linspace(im.norm.vmin, im.norm.vmax, im.cmap.N)
    cb = f.colorbar(im, ax=ax, orientation='horizontal',
                    shrink=0.75, aspect=25, ticks=PA.BoundaryLocator(),
                    pad=0.05, fraction=0.1, boundaries=b, values=v,
                    format=r'${\rm '+str('%0.'+str(m.n_format)+'f')+r'}$')
    cb.solids.set_visible(False)

    alpha_edges = np.linspace(0., 1., 50+1)
    alpha_centers = 0.5 * (alpha_edges[:-1] + alpha_edges[1:])
    XC, YC = np.meshgrid(v, alpha_centers)
    XC = XC.T; YC = YC.T # Transpose grids
    ax_alpha = cb.ax
    ax_alpha.set_aspect('auto')
    back = np.zeros([50, im.cmap.N, 4]); back[:, :, 3] = 1.
    ax_alpha.imshow(back, aspect='auto', extent=[b[0], b[-1], 0., 1.], interpolation='bicubic')
    image = ax_alpha.imshow(XC.T, alpha=YC.T, origin='lower', aspect='auto', extent=[b[0], b[-1], 0., 1.],
                            cmap=im.cmap, interpolation='bicubic', norm=im.norm)
    # ax_alpha.set_xlabel(m.u_str, fontsize=13)
    ax_alpha.minorticks_off()

    # Text labels
    if m.w_str is not None:
        ax.text(0.875, -.025, m.w_str, fontsize=12, ha='center', va='baseline', transform=ax.transAxes)
    if m.z_str is not None:
        ax.text(0.125, -.025, m.z_str, fontsize=12, ha='center', va='baseline', transform=ax.transAxes)
    if m.u_str is not None:
        cb.ax.text(0.5, -2.0, m.u_str, fontsize=14.5, transform=cb.ax.transAxes, ha='center', va='center')
    if add_vectors:
        theta, phi = spherical_coords(v8)
        outline_kwargs = {'path_effects': [patheffects.withStroke(linewidth=2, foreground='k')]}
        for i in range(len(theta)):
            x, y = _proj_xy(ax, theta[i], phi[i], rot=rot)
            ax.text(x, y, r'${\bf %d}$' % (i+1), color='w', ha='center', va='center', fontsize=7, **outline_kwargs)
    f.sca(ax)

def healpix_plot(sim='g10304', run='z8', snap=100, use_png=False, recenter_on_dipole=False, recenter_on_smoothed_max=False, smooth_pix=4., single=False, add_vectors=False, add_images=False, plot_Ha2=False, verbose=True):
    tree_dir = f'{zoom_dir}/{sim}/{run}/postprocessing/colt_tree'
    ics_dir = f'{colt_dir}/{sim}/{run}/ics_tree'
    maps = copy.deepcopy([IonMap, LyaMap, Lya1Map, Lya2Map, HI_int_Map, HI_esc_Map])
    # maps = copy.deepcopy([IonMap, LyaMap, Lya1Map, Lya2Map, HI_int_Map, HI_esc_Map, f_HI_Map, f_HeI_Map, f_HeII_Map, f_abs_Map, f_esc_Map, gas_int_Map, gas_esc_Map, metal_int_Map, metal_esc_Map, mean_dists_Map, graphite_int_Map, graphite_esc_Map, silicate_int_Map, silicate_esc_Map])
    n_maps = len(maps)
    i_LyC, i_Lya, i_Lya1, i_Lya2, i_HI_int, i_HI_esc = range(n_maps)
    # i_LyC, i_Lya, i_Lya1, i_Lya2, i_HI_int, i_HI_esc, i_f_HI, i_f_HeI, i_f_HeII, i_f_abs, i_f_esc, i_gas_int, i_gas_esc, i_metal_int, i_metal_esc, i_mean_dists, i_graphite_int, i_graphite_esc, i_silicate_int, i_silicate_esc = range(n_maps)
    for imap in range(n_maps):
        m = maps[imap]
        with h5py.File(f'{tree_dir}/{m.field}/{m.field}_{snap:03d}.hdf5','r') as f:
            m.data = f[m.key][:] / m.units
            if m.log:
                m.data = np.log10(m.data)
            # if imap == i_LyC:
            #     m.z_str = r'${\bf %s}$' % names[sim]
            # elif imap == i_Lya1:
            #     m.z_str = r'${\bf z = %0.2g}$' % f.attrs['z']
            if imap == 0:
                redshift = f.attrs['z']
                v8[:], x8[:], y8[:] = f['camera_directions'][:], f['camera_xaxes'][:], f['camera_yaxes'][:]
                if verbose: print(f'Redshift: {redshift:g}')
        if m.key == 'freq2_map' and imap > 0 and maps[imap-1].key == 'freq_map':
            m.data = np.sqrt(m.data - maps[imap-1].data**2)
        wq = weighted_percentile(m.data, np.ones_like(m.data), percentiles)
        if imap == i_Lya2:
            m.w_str = r'$%0.0f^{+%0.0f}_{-%0.0f}$' % (wq[0], wq[0]-wq[1], wq[2]-wq[0])
        else:
            m.w_str = r'$%0.1f^{+%0.1f}_{-%0.1f}$' % (wq[0], wq[0]-wq[1], wq[2]-wq[0])
            # m.w_str = r'${\bf %0.1f^{+%0.1f}_{-%0.1f}}$' % (wq[0], wq[0]-wq[1], wq[2]-wq[0])
        if m.lims == '[min,max]':
            m.lims = [np.min(m.data), np.max(m.data)]
        elif m.lims == '[0,max]':
            m.lims = [0., np.max(m.data)]
        elif m.lims is None:
            raise ValueError(f'lims not specified for {m.field}')
        if verbose: print(f'{m.field}: {wq[0]:g}  [{wq[1]:g}, {wq[2]:g}]  Avg/Min/Max: {np.mean(m.data):g}  [{np.min(m.data):g}, {np.max(m.data):g}]')
    p_str = (f'maps_' if single else f'{sim}_{run}/maps/maps_') + f'{snap:03d}.'
    if use_png:
        sargs = {'bbox_inches':'tight', 'pad_inches':0., 'transparent':False, 'dpi':640}
        p_str = '/orcd/scratch/orcd/006/arsmith/maps/' + p_str + 'png' # png for movies
    else:
        sargs = {'bbox_inches':'tight', 'pad_inches':0., 'transparent':True, 'dpi':640}
        p_str += 'pdf' # pdf for papers
    rot = get_rot(maps[0].data, recenter_on_dipole=recenter_on_dipole, recenter_on_smoothed_max=recenter_on_smoothed_max, smooth_pix=smooth_pix)
    if verbose: print(f'rot = {rot}')
    # Convert rotation to unit direction vector
    if rot is not None:
        # The rotation tuple (lon, lat, psi) defines the center direction
        lon, lat, psi = rot
        # Convert lon/lat to Cartesian unit vector (healpix uses lon,lat in degrees)
        vec = hp.rotator.Rotator(coord=['C','C'])([lon], [lat], lonlat=True)
        # Convert to Cartesian
        theta = np.radians(90 - lat)  # lat to theta
        phi = np.radians(lon)  # lon to phi
        vec = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        if verbose: print(f'unit direction vector = ({vec[0]:g}, {vec[1]:g}, {vec[2]:g})')
        # Find the projected vector subtracting the camera directions
        # c8 = np.array([vec - np.dot(vec, v8[j]) * v8[j] for j in range(8)])
        # c8 /= np.linalg.norm(c8, axis=1, keepdims=True)  # Normalize
        p8[:,0] = [np.dot(vec, x8[j]) for j in range(8)]  # Projected x-directions
        p8[:,1] = [np.dot(vec, y8[j]) for j in range(8)]  # Projected y-directions
        mu8[:] = [np.dot(vec, v8[j]) for j in range(8)]  # Angle to the healpix focus
        if verbose: print(f'Angles: {np.array_str(np.degrees(np.arccos(mu8)), precision=2)}')
    fig = plt.figure(figsize=(3.,2.))
    dy_map = 1.045 # 1.035
    HpPlot(fig, (1,dy_map,1,1), maps[i_Lya], rot=rot, add_vectors=add_vectors)  # LyA Escape Fraction
    HpPlot(fig, (0,dy_map,1,1), maps[i_LyC], rot=rot, add_vectors=add_vectors)  # LyC Escape Fraction
    # HpPlot(fig, (1,0,1,1), maps[i_Lya2], rot=rot, add_vectors=add_vectors)  # LyA Velocity Width
    # HpPlot(fig, (0,0,1,1), maps[i_Lya1], rot=rot, add_vectors=add_vectors)  # LyA Velocity Centroid
    Hp2Plot(fig, (1,0,1,1), maps[i_Lya2], maps[i_Lya], rot=rot, add_vectors=add_vectors)  # LyA Velocity Width
    Hp2Plot(fig, (0,0,1,1), maps[i_Lya1], maps[i_Lya], rot=rot, add_vectors=add_vectors)  # LyA Velocity Centroid
    HpPlot(fig, (0,-dy_map,1,1), maps[i_HI_int], rot=rot, add_vectors=add_vectors)  # HI Column Density (int) [cm^-2]
    HpPlot(fig, (1,-dy_map,1,1), maps[i_HI_esc], rot=rot, add_vectors=add_vectors)  # HI Column Density (esc) [cm^-2]
    # HpPlot(fig, (0,-2*dy_map,1,1), maps[i_gas_int], rot=rot, add_vectors=add_vectors)  # H Column Density (int) [cm^-2]
    # HpPlot(fig, (1,-2*dy_map,1,1), maps[i_gas_esc], rot=rot, add_vectors=add_vectors)  # H Column Density (esc) [cm^-2]
    # HpPlot(fig, (2,dy_map,1,1), maps[i_metal_int], rot=rot, add_vectors=add_vectors)  # Metal Column Density (int) [g/cm^2]
    # HpPlot(fig, (3,dy_map,1,1), maps[i_metal_esc], rot=rot, add_vectors=add_vectors)  # Metal Column Density (esc) [g/cm^2]
    # HpPlot(fig, (2,0,1,1), maps[i_graphite_int], rot=rot, add_vectors=add_vectors)  # Graphite Column Density (int) [g/cm^2]
    # HpPlot(fig, (3,0,1,1), maps[i_graphite_esc], rot=rot, add_vectors=add_vectors)  # Graphite Column Density (esc) [g/cm^2]
    # HpPlot(fig, (2,-dy_map,1,1), maps[i_silicate_int], rot=rot, add_vectors=add_vectors)  # Silicate Column Density (int) [g/cm^2]
    # HpPlot(fig, (3,-dy_map,1,1), maps[i_silicate_esc], rot=rot, add_vectors=add_vectors)  # Silicate Column Density (esc) [g/cm^2]
    # HpPlot(fig, (2,-2*dy_map,1,1), maps[i_mean_dists], rot=rot, add_vectors=add_vectors)  # Mean Distances [pc]
    # HpPlot(fig, (3,-2*dy_map,1,1), maps[i_f_esc], rot=rot, add_vectors=add_vectors)  # Escape Fraction
    # HpPlot(fig, (0,-3*dy_map,1,1), maps[i_f_HI], rot=rot, add_vectors=add_vectors)  # HI Absorption Fraction
    # HpPlot(fig, (1,-3*dy_map,1,1), maps[i_f_HeI], rot=rot, add_vectors=add_vectors)  # HeI Absorption Fraction
    # HpPlot(fig, (2,-3*dy_map,1,1), maps[i_f_HeII], rot=rot, add_vectors=add_vectors)  # HeII Absorption Fraction
    # HpPlot(fig, (3,-3*dy_map,1,1), maps[i_f_abs], rot=rot, add_vectors=add_vectors)  # Dust Absorption Fraction
    if add_images:
        imgs = copy.deepcopy([IonImage, MUVImage, OptImage, LyaImage, Lya1Image, Lya2Image, HaImage, Ha1Image, Ha2Image, O3Image, O2Image, HbImage, nH_Image, HI_Image])
        n_imgs = len(imgs)
        i_LyC, i_MUV, i_opt, i_Lya, i_Lya1, i_Lya2, i_Ha, i_Ha1, i_Ha2, i_O3, i_O2, i_Hb, i_nH, i_HI = range(n_imgs)
        for iimg in range(n_imgs):
            with h5py.File(f'{tree_dir}/{imgs[iimg].field}/{imgs[iimg].field}_{snap:03d}.hdf5','r') as f:
                imgs[iimg].data = f[imgs[iimg].key][:] / imgs[iimg].units
                if imgs[iimg].s_key is not None:
                    imgs[iimg].s_data = f[imgs[iimg].s_key][:] / imgs[iimg].s_units
                if iimg == i_LyC:
                    f_esc_LyC = f.attrs['f_esc']
                    Rvir = f.attrs['image_radius']  # Image radius = 1 R_vir [cm]
                    g = f['stars']
                    n_valid_stars = g.attrs['n_valid_stars']  # Number of valid stars
                    n_invalid_stars = g.attrs['n_invalid_stars']  # Number of invalid stars
                    n_stars = n_valid_stars + n_invalid_stars  # Total number of stars
                    if n_invalid_stars < n_valid_stars:
                        star_mask = np.ones(n_stars, dtype=bool)  # Valid star mask
                        star_mask[g['invalid_indices'][:]] = False  # Mark invalid stars
                    else: # Fewer valid stars
                        star_mask = np.zeros(n_stars, dtype=bool)  # Valid star mask
                        star_mask[g['valid_indices'][:]] = True  # Mark valid stars
                    Ndot_int = g['Ndot_int'][:]  # Photon rate [photons/s]
                    Ndot_esc = g['Ndot_esc'][:]  # Escaped rate [photons/s]
                    with h5py.File(f'{ics_dir}/colt_{snap:03d}.hdf5', 'r') as f_star:
                        # Z = f_star['Z_star'][:][star_mask]  # Stellar metallicity [mass fraction]
                        age = f_star['age_star'][:][star_mask]  # Stellar age [Gyr]
                        m_init_star = f_star['m_init_star'][:][star_mask]  # Initial mass [Msun]
                        mask_10Myr = age < 0.01  # < 10 Myr
                        mask_100Myr = age < 0.1  # < 100 Myr
                        SFR_10 = np.sum(m_init_star[mask_10Myr]) / 1e7  # SFR < 10 Myr [Msun/yr]
                        SFR_100 = np.sum(m_init_star[mask_100Myr]) / 1e8  # SFR < 100 Myr [Msun/yr]
                        r_star = f_star['r_star'][:][star_mask,:]  # Star positions [cm]
                    r_int = np.sum(r_star * Ndot_int[:,None], axis=0) / np.sum(Ndot_int)
                    r_esc = np.sum(r_star * Ndot_esc[:,None], axis=0) / np.sum(Ndot_esc)
                    radius_int = np.sqrt(np.sum(r_int**2))
                    radius_esc = np.sqrt(np.sum(r_esc**2))
                    x_int = np.array([np.dot(r_int, x8[j]) / Rvir for j in range(8)])  # Projected x-directions (int)
                    y_int = np.array([np.dot(r_int, y8[j]) / Rvir for j in range(8)])  # Projected y-directions (int)
                    x_esc = np.array([np.dot(r_esc, x8[j]) / Rvir for j in range(8)])  # Projected x-directions (esc)
                    y_esc = np.array([np.dot(r_esc, y8[j]) / Rvir for j in range(8)])  # Projected y-directions (esc)
                    x_int[x_int > 0.99] = 0.99; x_int[x_int < -0.99] = -0.99
                    y_int[y_int > 0.99] = 0.99; y_int[y_int < -0.99] = -0.99
                    x_esc[x_esc > 0.99] = 0.99; x_esc[x_esc < -0.99] = -0.99
                    y_esc[y_esc > 0.99] = 0.99; y_esc[y_esc < -0.99] = -0.99
                    del age, m_init_star, mask_10Myr, mask_100Myr, r_star, star_mask, Ndot_int, Ndot_esc
                    if verbose:
                        print(f'r_int = ({r_int[0]/kpc:g}, {r_int[1]/kpc:g}, {r_int[2]/kpc:g}) kpc  |  radius = {radius_int/kpc:g} kpc')
                        print(f'r_esc = ({r_esc[0]/kpc:g}, {r_esc[1]/kpc:g}, {r_esc[2]/kpc:g}) kpc  |  radius = {radius_esc/kpc:g} kpc')
                        print(f'|r_int - r_esc| = {np.sqrt(np.sum((r_int - r_esc)**2))/pc:g} pc')
                        print(f'Rvir = {Rvir/kpc:g} kpc')
                elif iimg == i_O3:
                    L_esc_O3 = f.attrs['L_tot'] * f.attrs['f_esc']
                elif iimg == i_O2:
                    L_esc_O2 = f.attrs['L_tot'] * f.attrs['f_esc']
                elif iimg == i_Hb:
                    L_esc_Hb = f.attrs['L_tot'] * f.attrs['f_esc']
                elif iimg == i_MUV:
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
                    L_1500_obs = L_1500 * f.attrs['f_esc']
                    R_10pc = 10. * pc               # Reference distance for continuum [cm]
                    fnu_1500_fac = lambda_1500**2 / (4. * np.pi * c * R_10pc**2 * angstrom)
                    M_1500 = -2.5 * np.log10(fnu_1500_fac * L_1500) - 48.6  # Continuum absolute magnitude (intrinsic)
                    M_1500_obs = -2.5 * np.log10(fnu_1500_fac * L_1500_obs) - 48.6  # Continuum absolute magnitude (observed)
            if imgs[iimg].cutout:  # Cut out the middle 50%
                n1 = imgs[iimg].data.shape[1]
                assert n1 == imgs[iimg].data.shape[2], f'Image is not square: {imgs[iimg].data.shape}'
                assert n1 % 4 == 0, f'Image size is not divisible by 4: {n1}'
                n4 = n1 // 4
                imgs[iimg].data = imgs[iimg].data[:, n4:-n4, n4:-n4]
        for iimg in [i_LyC, i_MUV, i_opt, i_O3, i_O2, i_Hb]:
            for i_degrade in range(imgs[iimg].n_degrade):
                img = imgs[iimg].data
                img = 0.25 * (img[:,::2,::2] + img[:,1::2,::2] + img[:,::2,1::2] + img[:,1::2,1::2])
                imgs[iimg].data = img
        for iimg in [i_Lya, i_Ha]:
            for i_degrade in range(imgs[iimg].n_degrade):
                img0, img1, img2 = imgs[iimg].data, imgs[iimg+1].data, imgs[iimg+2].data
                img1 = 0.25 * (img0[:,::2,::2]*img1[:,::2,::2] + img0[:,1::2,::2]*img1[:,1::2,::2] + img0[:,::2,1::2]*img1[:,::2,1::2] + img0[:,1::2,1::2]*img1[:,1::2,1::2])
                img2 = 0.25 * (img0[:,::2,::2]*img2[:,::2,::2] + img0[:,1::2,::2]*img2[:,1::2,::2] + img0[:,::2,1::2]*img2[:,::2,1::2] + img0[:,1::2,1::2]*img2[:,1::2,1::2])
                img0 = 0.25 * (img0[:,::2,::2] + img0[:,1::2,::2] + img0[:,::2,1::2] + img0[:,1::2,1::2])
                mask = img0 > 0.  # Avoid division by zero
                img1[mask] /= img0[mask]  # Normalize
                img2[mask] /= img0[mask]
                imgs[iimg].data, imgs[iimg+1].data, imgs[iimg+2].data = img0, img1, img2
            imgs[iimg+2].data -= imgs[iimg+1].data**2  # Standard deviation^2
            imgs[iimg+2].data[imgs[iimg+2].data<0.] = 0.  # Avoid negative values
            imgs[iimg+2].data = np.sqrt(imgs[iimg+2].data)  # Standard deviation
        for iimg in range(n_imgs):
            img = imgs[iimg]
            flat = img.data.flatten()
            img.wp = weighted_percentile(flat[flat>0.], np.ones_like(flat[flat>0.]), percentiles)
            img.mean, img.min, img.max = np.mean(img.data), np.min(img.data), np.max(img.data)
            if verbose: print(f'{img.field}: {img.wp[0]:g}  [{img.wp[1]:g}, {img.wp[2]:g}]  Avg/Min/Max: {img.mean:g}  [{img.min:g}, {img.max:g}]')
            if img.lims is not None:
                img.data[img.data < img.lims[0]] = img.lims[0]
                img.data[img.data > img.lims[1]] = img.lims[1]
            else:
                img_max = np.max(img.data)
                img.lims = [img.f_cut * img_max, img_max]
        n_images = imgs[i_Lya].data.shape[0]
        n_pixels = imgs[i_Lya].data.shape[1]
        n_pixels_rgb = imgs[i_MUV].data.shape[1]
        n_pixels_HI = imgs[i_HI].data.shape[1]
        assert n_pixels == imgs[i_Lya].data.shape[2]
        assert n_pixels == imgs[i_Lya+1].data.shape[1]
        assert n_pixels == imgs[i_Lya+1].data.shape[2]
        assert n_pixels == imgs[i_Lya+2].data.shape[1]
        assert n_pixels == imgs[i_Lya+2].data.shape[2]
        alpha_nH = get_alpha_p(imgs[i_nH].data, p_min=10., p_max=99.99)
        # alpha_Ha = get_alpha0(imgs[i_Ha].data, imgs[i_Ha].max, f_cut=imgs[i_Ha].f_cut, f_sat=0.05)
        alpha_Ha = get_alpha(imgs[i_Ha].data, imgs[i_Ha].max, f_cut=imgs[i_Ha].f_cut, target=imgs[i_Ha].target, f_sat=0.02)
        # alpha_Lya = get_alpha0(imgs[i_Lya].data, imgs[i_Lya].max, f_cut=imgs[i_Lya].f_cut)
        alpha_Lya = get_alpha(imgs[i_Lya].data, imgs[i_Lya].max, f_cut=imgs[i_Lya].f_cut, target=imgs[i_Lya].target, f_sat=0.1)
        Z_rgb = np.zeros([n_images, n_pixels_rgb, n_pixels_rgb, 3])
        for j in range(3):
            iimg = [i_opt, i_Ha, i_MUV][j]
            # alpha = get_alpha0(imgs[iimg].data, imgs[iimg].max, f_cut=imgs[iimg].f_cut)
            if iimg == i_Ha:
                for i in range(n_images):
                    Z_rgb[i,:,:,j] = alpha_Ha[i,:,:].T  # Transpose for plotting
            else:
                alpha = get_alpha(imgs[iimg].data, imgs[iimg].max, f_cut=imgs[iimg].f_cut, target=imgs[iimg].target, f_sat=0.5)
                for i in range(n_images):
                    Z_rgb[i,:,:,j] = alpha[i,:,:].T  # Transpose for plotting
        # O_32 and R_3
        O_32, R_3 = L_esc_O3 / L_esc_O2, L_esc_O3 / L_esc_Hb
        if verbose: print(f'O_32 = {O_32:g}, R_3 = {R_3:g}')
        # alpha_O3 = get_alpha0(imgs[i_O3].data, imgs[i_O3].max, f_cut=imgs[i_O3].f_cut, f_sat=0.25)
        # alpha_O3 = get_alpha(imgs[i_O3].data, imgs[i_O3].max, f_cut=imgs[i_O3].f_cut, target=1.25*imgs[i_O3].target, f_sat=0.25)
        # alpha_O2 = get_alpha(imgs[i_O2].data, imgs[i_O2].max, f_cut=imgs[i_O2].f_cut, target=1.25*imgs[i_O2].target, f_sat=0.25)
        # alpha_Hb = get_alpha(imgs[i_Hb].data, imgs[i_Hb].max, f_cut=imgs[i_Hb].f_cut, target=1.25*imgs[i_Hb].target, f_sat=0.25)
        # alpha_O32 = alpha_O3 + alpha_O2; alpha_O32 /= np.max(alpha_O32)
        # alpha_R3 = alpha_O3 + alpha_Hb; alpha_R3 /= np.max(alpha_R3)
        Z_O3pO2 = imgs[i_O3].data + imgs[i_O2].data  # Importance sum (OIII + OII)
        Z_O3pHb = imgs[i_O3].data + imgs[i_Hb].data  # Importance sum (OIII + Hb)
        alpha_O32 = get_alpha(Z_O3pO2, np.max(Z_O3pO2), f_cut=imgs[i_O3].f_cut, target=imgs[i_O3].target, f_sat=0.1)
        alpha_R3 = get_alpha(Z_O3pHb, np.max(Z_O3pHb), f_cut=imgs[i_O3].f_cut, target=imgs[i_O3].target, f_sat=0.1)
        del Z_O3pO2, Z_O3pHb
        Z_O32 = safe_ratio(imgs[i_O3].data, imgs[i_O2].data)  # OIII / OII
        Z_R3 = safe_ratio(imgs[i_O3].data, imgs[i_Hb].data)  # OIII / Hb
        if verbose:
            print(f'O_32 min/max: {np.min(Z_O32):g} {np.max(Z_O32):g}')
            print(f'R_3 min/max: {np.min(Z_R3):g} {np.max(Z_R3):g}')
        flat = Z_O32.flatten()
        # wq_O32 = weighted_percentile(flat[flat>0.], np.ones_like(flat[flat>0.]), percentiles)
        wq_O32 = weighted_percentile(flat[flat>0.], alpha_O32.flatten()[flat>0.], percentiles)
        flat = Z_R3.flatten()
        # wq_R3 = weighted_percentile(flat[flat>0.], np.ones_like(flat[flat>0.]), percentiles)
        wq_R3 = weighted_percentile(flat[flat>0.], alpha_R3.flatten()[flat>0.], percentiles)
        if verbose:
            print(f'O_32 percentiles: {wq_O32[0]:g}  [{wq_O32[1]:g}, {wq_O32[2]:g}]  [{wq_O32[3]:g}, {wq_O32[4]:g}]  [{wq_O32[5]:g}, {wq_O32[6]:g}]')
            print(f'R_3 percentiles: {wq_R3[0]:g}  [{wq_R3[1]:g}, {wq_R3[2]:g}]  [{wq_R3[3]:g}, {wq_R3[4]:g}]  [{wq_R3[5]:g}, {wq_R3[6]:g}]')
        dx, dy = 0.25, 0.375
        back_Ha = np.zeros([n_pixels_rgb, n_pixels_rgb, 4]); back_Ha[:,:,3] = 1.
        back_Lya = np.zeros([n_pixels, n_pixels, 4]); back_Lya[:,:,3] = 1.
        back_HI = np.zeros([n_pixels_HI, n_pixels_HI, 4]); back_HI[:,:,3] = 1.
        # x0, y0 = 0., -0.5
        # x0, y0 = 8.*dx, 5.5*dy - 0.5
        x0, y0 = 8.*dx, 5.5*dy - 0.225
        for i in range(n_images):
            iy = 0.
            xi = x0 + float(i) * dx
            axs = []
            ax_LyC = fig.add_axes([xi, y0-iy*dy, dx, dy]); iy += 1.; axs.append(ax_LyC)
            ax_HI  = fig.add_axes([xi, y0-iy*dy, dx, dy]); iy += 1.; axs.append(ax_HI)
            ax_rgb = fig.add_axes([xi, y0-iy*dy, dx, dy]); iy += 1.; axs.append(ax_rgb)
            ax_Ha1 = fig.add_axes([xi, y0-iy*dy, dx, dy]); iy += 1.; axs.append(ax_Ha1)
            if plot_Ha2:
                ax_Ha2 = fig.add_axes([xi, y0-iy*dy, dx, dy]); iy += 1.; axs.append(ax_Ha2)
            ax_O32 = fig.add_axes([xi, y0-iy*dy, dx, dy]); iy += 1.; axs.append(ax_O32)
            ax_R3 = fig.add_axes([xi, y0-iy*dy, dx, dy]); iy += 1.; axs.append(ax_R3)
            ax_Lya = fig.add_axes([xi, y0-iy*dy, dx, dy]); iy += 1.; axs.append(ax_Lya)
            ax_Lya1 = fig.add_axes([xi, y0-iy*dy, dx, dy]); iy += 1.; axs.append(ax_Lya1)
            ax_Lya2 = fig.add_axes([xi, y0-iy*dy, dx, dy]); iy += 1.; axs.append(ax_Lya2)
            ax_LyC.imshow(imgs[i_LyC].data[i].T, origin='lower', extent=[-1.,1.,-1.,1.], cmap=imgs[i_LyC].cmap,
                aspect='equal', interpolation='bicubic', norm=LogNorm(vmin=imgs[i_LyC].f_cut*imgs[i_LyC].max, vmax=imgs[i_LyC].max, clip=True))
            rgba_HI = get_rgba_HI(imgs[i_HI].data[i].T, alpha_nH[i].T)
            ax_HI.imshow(back_HI, extent=[-1.,1.,-1.,1.], aspect='equal')
            ax_HI.imshow(rgba_HI, origin='lower', extent=[-1.,1.,-1.,1.], aspect='equal', interpolation='bicubic')
            ax_rgb.imshow(Z_rgb[i,:,:,:], origin='lower', extent=[-1.,1.,-1.,1.], aspect='equal', interpolation='bicubic')
            ax_Ha1.imshow(back_Ha, aspect='equal', extent=[-1.,1.,-1.,1.])
            ax_Ha1.imshow(imgs[i_Ha+1].data[i].T, alpha=alpha_Ha[i].T, origin='lower', extent=[-1.,1.,-1.,1.],
                cmap=imgs[i_Ha+1].cmap, aspect='equal', interpolation='bicubic', norm=Normalize(vmin=imgs[i_Ha+1].lims[0], vmax=imgs[i_Ha+1].lims[1], clip=True))
            if plot_Ha2:
                ax_Ha2.imshow(back_Ha, aspect='equal', extent=[-1.,1.,-1.,1.])
                ax_Ha2.imshow(imgs[i_Ha+2].data[i].T, alpha=alpha_Ha[i].T, origin='lower', extent=[-1.,1.,-1.,1.],
                    cmap=imgs[i_Ha+2].cmap, aspect='equal', interpolation='bicubic', norm=Normalize(vmin=imgs[i_Ha+2].lims[0], vmax=imgs[i_Ha+2].lims[1], clip=True))
            ax_O32.imshow(back_Ha, aspect='equal', extent=[-1.,1.,-1.,1.])
            ax_O32.imshow(Z_O32[i].T, alpha=alpha_O32[i].T, origin='lower', extent=[-1.,1.,-1.,1.],
                cmap=cmr.cm.tropical, aspect='equal', interpolation='bicubic', norm=LogNorm(vmin=wq_O32[5], vmax=wq_O32[6], clip=True))
            ax_R3.imshow(back_Ha, aspect='equal', extent=[-1.,1.,-1.,1.])
            ax_R3.imshow(Z_R3[i].T, alpha=alpha_R3[i].T, origin='lower', extent=[-1.,1.,-1.,1.],
                cmap=cmr.cm.neon, aspect='equal', interpolation='bicubic', norm=LogNorm(vmin=wq_R3[5], vmax=wq_R3[6], clip=True))
            ax_Lya.imshow(imgs[i_Lya].data[i].T, origin='lower', extent=[-1.,1.,-1.,1.], cmap=imgs[i_Lya].cmap,
                aspect='equal', interpolation='bicubic', norm=LogNorm(vmin=imgs[i_Lya].f_cut*imgs[i_Lya].max, vmax=imgs[i_Lya].max, clip=True))
            ax_Lya1.imshow(back_Lya, aspect='equal', extent=[-1.,1.,-1.,1.])
            ax_Lya1.imshow(imgs[i_Lya+1].data[i].T, alpha=alpha_Lya[i].T, origin='lower', extent=[-1.,1.,-1.,1.],
                cmap=imgs[i_Lya+1].cmap, aspect='equal', interpolation='bicubic', norm=Normalize(vmin=imgs[i_Lya+1].lims[0], vmax=imgs[i_Lya+1].lims[1], clip=True))
            ax_Lya2.imshow(back_Lya, aspect='equal', extent=[-1.,1.,-1.,1.])
            ax_Lya2.imshow(imgs[i_Lya+2].data[i].T, alpha=alpha_Lya[i].T, origin='lower', extent=[-1.,1.,-1.,1.],
                cmap=imgs[i_Lya+2].cmap, aspect='equal', interpolation='bicubic', norm=Normalize(vmin=imgs[i_Lya+2].lims[0], vmax=imgs[i_Lya+2].lims[1], clip=True))
            # Draw R_vir
            ax_LyC.add_artist(plt.Circle((0, 0), 1., color='w', alpha=0.5, fill=False, lw=0.25))
            ax_HI.add_artist(plt.Circle((0, 0), 1., color='w', alpha=0.5, fill=False, lw=0.25))
            ax_rgb.add_artist(plt.Circle((0, 0), 1., color='w', alpha=0.5, fill=False, lw=0.25))
            ax_Ha1.add_artist(plt.Circle((0, 0), 1., color='w', alpha=0.5, fill=False, lw=0.25))
            if plot_Ha2:
                ax_Ha2.add_artist(plt.Circle((0, 0), 1., color='w', alpha=0.5, fill=False, lw=0.25))
            ax_O32.add_artist(plt.Circle((0, 0), 1., color='w', alpha=0.5, fill=False, lw=0.25))
            ax_R3.add_artist(plt.Circle((0, 0), 1., color='w', alpha=0.5, fill=False, lw=0.25))
            ax_Lya.add_artist(plt.Circle((0, 0), 0.5, color='w', alpha=0.5, fill=False, lw=0.25))
            ax_Lya1.add_artist(plt.Circle((0, 0), 0.5, color='w', alpha=0.5, fill=False, lw=0.25))
            ax_Lya2.add_artist(plt.Circle((0, 0), 0.5, color='w', alpha=0.5, fill=False, lw=0.25))
            # Draw max leaking direction
            for ax in axs:
                # ax.plot([0,p8[i,0]], [0,p8[i,1]], c='w', lw=0.25, ls='--' if mu8[i] < 0. else '-', alpha=0.5)
                x_end, y_end = p8[i,0] + x_int[i], p8[i,1] + y_int[i]
                if x_end > 0.99:
                    x_end, y_end = 0.99, y_int[i] + (0.99 - x_int[i]) * p8[i,1] / p8[i,0]
                if y_end > 0.99:
                    y_end, x_end = 0.99, x_int[i] + (0.99 - y_int[i]) * p8[i,0] / p8[i,1]
                if x_end < -0.99:
                    x_end, y_end = -0.99, y_int[i] + (-0.99 - x_int[i]) * p8[i,1] / p8[i,0]
                if y_end < -0.99:
                    y_end, x_end = -0.99, x_int[i] + (-0.99 - y_int[i]) * p8[i,0] / p8[i,1]
                ax.plot([x_int[i], x_end], [y_int[i], y_end], c='w', lw=0.25, ls='--' if mu8[i] < 0. else '-', alpha=0.5)
            # Print Labels
            if i == 0:
                ax_LyC.text(0.5, 0.85, r'${\bf LyC}$', color='w', ha='center', va='baseline', fontsize=7, transform=ax_LyC.transAxes, path_effects=[patheffects.withStroke(linewidth=2, foreground='k')])
                ax_HI.text(0.5, 0.85, r'${\bf HI}$', color='w', ha='center', va='baseline', fontsize=7, transform=ax_HI.transAxes, path_effects=[patheffects.withStroke(linewidth=2, foreground='k')])
                ax_rgb.text(0.5, 0.85, r'${\bf Optical}$', color=[0.7,0,0], ha='center', va='baseline', fontsize=7, transform=ax_rgb.transAxes, path_effects=[patheffects.withStroke(linewidth=2, foreground='k')])
                ax_Ha1.text(0.5, 0.85, r'${\bf \langle \Delta v_{\bf H\alpha} \rangle}$', color='w', ha='center', va='baseline', fontsize=7, transform=ax_Ha1.transAxes, path_effects=[patheffects.withStroke(linewidth=2, foreground='k')])
                if plot_Ha2:
                    ax_Ha2.text(0.5, 0.85, r'${\bf \sigma_{\bf \Delta v,H\alpha}}$', color='w', ha='center', va='baseline', fontsize=7, transform=ax_Ha2.transAxes, path_effects=[patheffects.withStroke(linewidth=2, foreground='k')])
                ax_O32.text(0.5, 0.85, r'${\bf O_{32}}$', color='w', ha='center', va='baseline', fontsize=7, transform=ax_O32.transAxes, path_effects=[patheffects.withStroke(linewidth=2, foreground='k')])
                ax_R3.text(0.5, 0.85, r'${\bf R_{3}}$', color='w', ha='center', va='baseline', fontsize=7, transform=ax_R3.transAxes, path_effects=[patheffects.withStroke(linewidth=2, foreground='k')])
                ax_Lya.text(0.5, 0.85, r'${\bf Ly\alpha}$', color='w', ha='center', va='baseline', fontsize=7, transform=ax_Lya.transAxes, path_effects=[patheffects.withStroke(linewidth=2, foreground='k')])
                ax_Lya1.text(0.5, 0.85, r'${\bf \langle \Delta v_{\bf Ly\alpha} \rangle}$', color='w', ha='center', va='baseline', fontsize=7, transform=ax_Lya1.transAxes, path_effects=[patheffects.withStroke(linewidth=2, foreground='k')])
                ax_Lya2.text(0.5, 0.85, r'${\bf \sigma_{\bf \Delta v,Ly\alpha}}$', color='w', ha='center', va='baseline', fontsize=7, transform=ax_Lya2.transAxes, path_effects=[patheffects.withStroke(linewidth=2, foreground='k')])
                ax_LyC.text(0.05, 0.035, r'$\%$', color=lg, ha='left', va='baseline', fontsize=5, transform=ax_LyC.transAxes, path_effects=[patheffects.withStroke(linewidth=1, foreground='k')])
                # ax_Ha1.text(0.05, 0.035, r'${\rm km/s}$', color=lg, ha='left', va='baseline', fontsize=5, transform=ax_Ha1.transAxes, path_effects=[patheffects.withStroke(linewidth=1, foreground='k')])
                # ax_Ha2.text(0.05, 0.035, r'${\rm km/s}$', color=lg, ha='left', va='baseline', fontsize=5, transform=ax_Ha2.transAxes, path_effects=[patheffects.withStroke(linewidth=1, foreground='k')])
                ax_Lya.text(0.05, 0.035, r'$\%$', color=lg, ha='left', va='baseline', fontsize=5, transform=ax_Lya.transAxes, path_effects=[patheffects.withStroke(linewidth=1, foreground='k')])
                ax_Lya1.text(0.05, 0.035, r'${\rm km/s}$', color=lg, ha='left', va='baseline', fontsize=5, transform=ax_Lya1.transAxes, path_effects=[patheffects.withStroke(linewidth=1, foreground='k')])
                ax_Lya2.text(0.05, 0.035, r'${\rm km/s}$', color=lg, ha='left', va='baseline', fontsize=5, transform=ax_Lya2.transAxes, path_effects=[patheffects.withStroke(linewidth=1, foreground='k')])
                ax_O32.text(0.5, 0.035, r'$%.2f$' % O_32, color=lg, ha='center', va='baseline', fontsize=5, transform=ax_O32.transAxes, path_effects=[patheffects.withStroke(linewidth=1, foreground='k')])
                ax_R3.text(0.5, 0.035, r'$%.2f$' % R_3, color=lg, ha='center', va='baseline', fontsize=5, transform=ax_R3.transAxes, path_effects=[patheffects.withStroke(linewidth=1, foreground='k')])
            elif i == 1:
                ax_rgb.text(0.5, 0.85, r'${\bf H\alpha}$', color=[0,0.7,0], ha='center', va='baseline', fontsize=7, transform=ax_rgb.transAxes, path_effects=[patheffects.withStroke(linewidth=2, foreground='k')])
                ax_O32.text(0.5, 0.85, r'${\bf OIII/OII}$', color='w', ha='center', va='baseline', fontsize=6, transform=ax_O32.transAxes, path_effects=[patheffects.withStroke(linewidth=2, foreground='k')])
                ax_R3.text(0.5, 0.85, r'${\bf OIII/H\beta}$', color='w', ha='center', va='baseline', fontsize=6, transform=ax_R3.transAxes, path_effects=[patheffects.withStroke(linewidth=2, foreground='k')])
            elif i == 2:
                ax_rgb.text(0.5, 0.85, r'${\bf UV}$', color=[0,0,0.7], ha='center', va='baseline', fontsize=7, transform=ax_rgb.transAxes, path_effects=[patheffects.withStroke(linewidth=2, foreground='k')])
            # Print stats
            ax_LyC.text(0.5, 0.035, r'$%.1f$' % imgs[i_LyC].s_data[i], color=lg, ha='center', va='baseline', fontsize=5, transform=ax_LyC.transAxes, path_effects=[patheffects.withStroke(linewidth=1, foreground='k')])
            # ax_Ha1.text(0.5, 0.035, r'$%.0f$' % imgs[i_Ha+1].s_data[i], color=lg, ha='center', va='baseline', fontsize=5, transform=ax_Ha1.transAxes, path_effects=[patheffects.withStroke(linewidth=1, foreground='k')])
            # ax_Ha2.text(0.5, 0.035, r'$%.0f$' % imgs[i_Ha+2].s_data[i], color=lg, ha='center', va='baseline', fontsize=5, transform=ax_Ha2.transAxes, path_effects=[patheffects.withStroke(linewidth=1, foreground='k')])
            ax_Lya.text(0.5, 0.035, r'$%.0f$' % imgs[i_Lya].s_data[i], color=lg, ha='center', va='baseline', fontsize=5, transform=ax_Lya.transAxes, path_effects=[patheffects.withStroke(linewidth=1, foreground='k')])
            ax_Lya1.text(0.5, 0.035, r'$%.0f$' % imgs[i_Lya+1].s_data[i], color=lg, ha='center', va='baseline', fontsize=5, transform=ax_Lya1.transAxes, path_effects=[patheffects.withStroke(linewidth=1, foreground='k')])
            ax_Lya2.text(0.5, 0.035, r'$%.0f$' % imgs[i_Lya+2].s_data[i], color=lg, ha='center', va='baseline', fontsize=5, transform=ax_Lya2.transAxes, path_effects=[patheffects.withStroke(linewidth=1, foreground='k')])
            for ax in axs:
                ax.plot([-1,-1,1,1,-1], [-1,1,1,-1,-1], 'w', lw=0.5)
                ax.set_axis_off()
            ax_LyC.text(0.1, 0.85, r'${\bf %d}$' % (i+1), color='w', ha='center', va='baseline', fontsize=7, transform=ax_LyC.transAxes, path_effects=[patheffects.withStroke(linewidth=2, foreground='k')])
    with h5py.File(f'{ics_dir}/center.hdf5', 'r') as f:
        Redshifts = f['Redshifts'][:]
        i_z = np.argmin(np.abs(Redshifts - redshift))
        g = f['Smoothed']
        R_Crit200 = g['R_Crit200'][i_z]  # [kpc]
        # M_gas = g['M_gas'][i_z]  # [Msun]
        M_stars = g['M_stars'][i_z]  # [Msun]
        M_vir = g['M_vir'][i_z]  # [Msun]
        if verbose:
            print(f'z = {Redshifts[i_z]:g}, redshift = {redshift:g}')
            print(f'R_Crit200 = {R_Crit200:g} kpc, R_image = {Rvir/kpc:g} kpc')
            print(f'SFR_10 = {SFR_10:g} Msun/yr, SFR_100 = {SFR_100:g} Msun/yr')
            print(f'M_halo = {M_vir:g} Msun')
            print(f'M_stars = {M_stars:g} Msun')
    fig.text(0.05, 2.07, r'${\bf %s}$' % names[sim], fontsize=15, ha='left', va='center', transform=fig.transFigure)
    fig.text(0.39, 2.12, r'${\bf z = %0.2g}$' % redshift, fontsize=12, ha='left', va='baseline', transform=fig.transFigure)
    fig.text(0.39, 2., r'${\bf M_{\bf{UV}} = %0.1f}$' % M_1500_obs, fontsize=11, ha='left', va='baseline', transform=fig.transFigure)
    fig.text(0.8, 2.12, r'${\bf \log M_{\bf{halo}}}\,{\rm [M_{\odot}]}{\bf = %0.1f}$' % np.log10(M_vir), fontsize=11, ha='left', va='baseline', transform=fig.transFigure)
    fig.text(0.8, 2., r'${\bf \log M_{\bf{stars}}}\,{\rm [M_{\odot}]}{\bf = %0.1f}$' % np.log10(M_stars), fontsize=11, ha='left', va='baseline', transform=fig.transFigure)
    fig.text(1.4, 2.12, r'${\bf R_{\bf{vir}}}\,{\rm [kpc]}{\bf = %0.1f}$' % (Rvir/kpc), fontsize=11, ha='left', va='baseline', transform=fig.transFigure)
    SFR_str = f'{SFR_10:.{0 if SFR_10 >= 100 else 1 if SFR_10 >= 10 else 2 if SFR_10 >= 1 else 3}f}'
    fig.text(1.4, 2., r'${\bf SFR_{\bf{10}}}\,{\rm [M_{\odot}/yr]}{\bf = %s}$' % SFR_str, fontsize=11, ha='left', va='baseline', transform=fig.transFigure)
    # fig.text(0.5, 1.0, r'${\bf \log M_{\bf{gas}} = %0.1f}$' % np.log10(M_gas), fontsize=13, ha='left', va='baseline', transform=fig.transFigure)
    print(f' Saving {p_str}')
    plt.savefig(p_str, **sargs)
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
field = 'ion-eq'

# if True:
if False:
    sim_run_dir = f'{sim}_{run}'
    os.makedirs(sim_run_dir, exist_ok=True)
    os.makedirs(f'{sim_run_dir}/{field}', exist_ok=True)
    for snap in range(189):
        try:
            healpix_plot(sim=sim, run=run, field=field, snap=snap, use_png=True, lims=[0.,55.], recenter_on_dipole=True)
        except:
            pass

# Hb, Ha, Ha-RHD, Ha-cont, Lya, Lya-cont, M1500, OII-3727-3730, OIII-5008, SiII-1190, ion-eq, ion-eq-RHD, optical
# healpix_plot(sim='g5760', run='z8', snap=168, use_png=False, recenter_on_smoothed_max=True, smooth_pix=4., single=True, add_vectors=True, add_images=True)
# healpix_plot(sim='g5760', run='z8', snap=188, use_png=False, recenter_on_smoothed_max=True, smooth_pix=4., single=True, add_vectors=True, add_images=True)
def process_snap(snap):
    """Wrapper function for processing a single snapshot in parallel."""
    try:
        healpix_plot(sim='g5760', run='z8', snap=snap, use_png=True,
                   recenter_on_smoothed_max=True, smooth_pix=4.,
                   single=True, add_vectors=True, add_images=True, verbose=False)
        return True
    except Exception as e:
        return False

# Process snapshots in parallel
if __name__ == "__main__":
    # Number of processes to use (adjust based on your CPU)
    n_processes = os.cpu_count()

    print(f"Processing {189} snapshots using {n_processes} processes...")

    # Create a pool of workers
    with Pool(processes=n_processes) as pool:
        # Map the process_snap function to all snapshots
        results = pool.map(process_snap, range(189))
    print(f"Processed {sum(results)}/{len(results)} snapshots successfully")

# Original serial loop (commented out)
# for snap in range(189):
#     try:
#         healpix_plot(sim='g5760', run='z8', snap=snap, use_png=True, recenter_on_smoothed_max=True, smooth_pix=4., single=True, add_vectors=True, add_images=True, verbose=False)
#     except:
#         pass
# healpix_plot(sim='g5760', run='z8', snap=168, use_png=True, lims=[0.,50.], recenter_on_dipole=True, single=True, add_vectors=True, add_images=True)
# healpix_plot(sim='g5760', run='z8', snap=168, use_png=True, lims=[0.,50.], single=True, add_vectors=True, add_images=True)
# healpix_plot(field, snap=snap, use_png=True, lims=None)
# healpix_plot(field, snap=snap, use_png=False, lims=[0.,22.])
# healpix_plot(field, snap=snap, use_png=False, lims=None)

# healpix_plot('f_esc_ion', snap=72, use_png=False, lims=[0.,10.8])
# healpix_plot('f_esc_ion', snap=51, use_png=False, lims=[0.,13.525])
