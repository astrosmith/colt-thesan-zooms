import numpy as np
import time
import h5py, os
import healpy as hp
from healpy import projaxes as PA
from healpy import pixelfunc
from scipy.optimize import curve_fit
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
kpc  = 1e3 * pc             # Units: 1 kpc = 3e21 cm
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

def normalize_to_probability(m, clip_negative=True, eps=0.):
    """
    Convert an array m to a probability vector (nonnegative, sums to 1).
    - clip_negative: if True, values < 0 are set to 0
    - eps: optional small floor added after clipping (e.g. eps=1e-15)
    """
    m = np.asarray(m, dtype=float).copy()
    if clip_negative:
        m[m < 0] = 0.
    if eps > 0:
        m = m + eps
    s = m.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Map has non-positive or invalid total mass after preprocessing.")
    return m / s

def healpix_unit_vectors(nside, nest=False):
    """
    Return unit vectors for HEALPix pixel centers as (Npix, 3).
    """
    import healpy as hp
    npix = hp.nside2npix(nside)
    x, y, z = hp.pix2vec(nside, np.arange(npix), nest=nest)
    return np.vstack([x, y, z]).T

def geodesic_cost_matrix(vecs, p=1):
    """
    Build the NxN cost matrix C_ij = d_geo(x_i, x_j)^p, with d_geo in radians.
    """
    # dot products in [-1, 1]
    dots = np.clip(vecs @ vecs.T, -1., 1.)
    theta = np.arccos(dots)  # radians in [0, pi]
    return theta if p == 1 else theta * theta if p == 2 else theta ** p

def structure_length(a, b, vecs):
    """
    Characteristic structure scale. area outputs full pair averages, debias removes the phase-space factor.
    """
    a = normalize_to_probability(a)
    b = normalize_to_probability(b)
    C = geodesic_cost_matrix(vecs, p=1)
    # Pair weights (exclude self-pairs in auto-correlations)
    aa = a[:,None] * a[None,:]
    bb = b[:,None] * b[None,:]
    ab = a[:,None] * b[None,:]
    np.fill_diagonal(aa, 0.)
    np.fill_diagonal(bb, 0.)
    # Area-weighted pair averages
    dist_aa = np.sum(aa * C) / np.sum(aa)  # sum(a_i * a_j * dist_ij) / sum(a_i * a_j)
    dist_bb = np.sum(bb * C) / np.sum(bb)  # Plot: 1 - dist_aa / 90
    dist_ab = np.sum(ab * C) / np.sum(ab)
    # Debiased pair weights (avoid blowups near 0 and pi)
    mask = (C > np.pi / 180.) & (C < np.pi * 179 / 180.)
    aa[~mask] = 0.  # Set to 0 to exclude pairs at 0 and pi
    bb[~mask] = 0.
    ab[~mask] = 0.
    sin_C = np.sin(C[mask])
    aa[mask] /= sin_C
    bb[mask] /= sin_C
    ab[mask] /= sin_C
    dist_aa_debiased = np.sum(aa * C) / np.sum(aa)
    dist_bb_debiased = np.sum(bb * C) / np.sum(bb)
    dist_ab_debiased = np.sum(ab * C) / np.sum(ab)
    return dist_aa, dist_bb, dist_ab, dist_aa_debiased, dist_bb_debiased, dist_ab_debiased

def wasserstein_sphere_exact(a, b, vecs, p=1, prefer_pot=True, verbose=False):
    """
    Compute W_p between discrete measures a and b on S^2 with geodesic ground cost.

    a, b: arrays of length N (weights), must sum to 1 (or will be normalized here)
    vecs: (N, 3) unit vectors for pixel centers
    p: Wasserstein order
    prefer_pot: if True, try POT (ot) first; else use scipy.linprog (HiGHS)
    Returns: [W_p, dist_aa, dist_bb, dist_ab] (in radians)
    """
    a = normalize_to_probability(a)
    b = normalize_to_probability(b)

    if verbose: start = time.time()
    C = geodesic_cost_matrix(vecs, p=p)
    if verbose: stop = time.time(); print(f'Cost matrix: {stop - start:g} seconds'); start = stop

    # Try POT if available (fast and robust)
    if prefer_pot:
        try:
            import ot  # POT: Python Optimal Transport
            if verbose: print(f'Using POT for W_{p}')
            # ot.emd2 returns the minimum transport cost sum_{ij} T_ij * C_ij
            cost = ot.emd2(a, b, C)
            if verbose: print(f'POT: {time.time() - start:g} seconds'); start = time.time()
            return cost if p == 1 else np.sqrt(cost) if p == 2 else cost ** (1. / p)
        except Exception as e:
            if verbose:
                print(f"POT unavailable or failed ({e}); falling back to LP via scipy.linprog.")

    # Exact LP via scipy.optimize.linprog with sparse constraints
    from scipy.optimize import linprog
    import scipy.sparse as sp

    if verbose: print(f'Using LP for W_{p}')
    c = C.reshape(-1)  # length N^2

    # Constraints:
    # For each i: sum_j T_ij = a_i
    # For each j: sum_i T_ij = b_j
    # Let x = vec(T) row-major (i varying slow, j varying fast) consistent with reshape(-1).
    #
    # Build A_eq as:
    #   A_source = I_N kron 1_N^T    (shape N x N^2)
    #   A_target = 1_N^T kron I_N    (shape N x N^2)
    N = len(a)
    I = sp.eye(N, format="csr")
    ones = sp.csr_matrix(np.ones((1, N)))
    A_source = sp.kron(I, ones, format="csr")
    A_target = sp.kron(ones, I, format="csr")
    A_eq = sp.vstack([A_source, A_target], format="csr")
    b_eq = np.concatenate([a, b])

    bounds = (0., None)  # T_ij >= 0

    res = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"presolve": True}
    )
    if not res.success:
        raise RuntimeError(f"linprog failed: {res.message}")

    cost = res.fun
    if verbose: print(f'LP: {time.time() - start:g} seconds'); start = time.time()
    return cost if p == 1 else np.sqrt(cost) if p == 2 else cost ** (1. / p)

def angular_spectra(p1_map, p2_map, nside, lmax=None, remove_monopole=True):
    """
    Returns ell, Cl11, Cl22, Cl12, rho_l
    where rho_l = Cl12 / sqrt(Cl11*Cl22).
    """
    if lmax is None:
        lmax = 3*nside - 1

    m1 = np.asarray(p1_map, float).copy()
    m2 = np.asarray(p2_map, float).copy()

    if remove_monopole:
        m1 -= np.mean(m1)
        m2 -= np.mean(m2)

    # healpy.anafast can compute auto/cross directly
    Cl11 = hp.anafast(m1, m1, lmax=lmax)
    Cl22 = hp.anafast(m2, m2, lmax=lmax)
    Cl12 = hp.anafast(m1, m2, lmax=lmax)

    ell = np.arange(len(Cl11))
    denom = np.sqrt(np.maximum(Cl11, 0) * np.maximum(Cl22, 0))
    rho = np.zeros_like(Cl12)
    good = denom > 0
    rho[good] = Cl12[good] / denom[good]

    return ell, Cl11, Cl22, Cl12, rho

def characteristic_angle_from_spectrum(ell, Cl, method="peak_Dl"):
    """
    Convert a spectrum into a representative angular scale (in degrees).
    method:
      - "peak_Dl": find ell that maximizes D_ell = ell(ell+1)Cl/(2pi) for ell>=1
      - "energy_weighted": ell_eff = sum ell * w / sum w, with w=(2ell+1)Cl for ell>=1
    Then theta ~ pi/ell (radians) -> degrees.
    """
    ell = np.asarray(ell)
    Cl = np.asarray(Cl)

    mask = ell >= 1
    e = ell[mask]
    c = Cl[mask]

    # Guard against negative Cl due to noise (after monopole removal, etc.)
    cpos = np.maximum(c, 0.)

    if method == "peak_Dl":
        Dl = e*(e+1)*cpos/(2*np.pi)
        if np.all(Dl == 0):
            return np.nan
        ell_peak = e[np.argmax(Dl)]
        theta_rad = np.pi / ell_peak
        return np.degrees(theta_rad)

    elif method == "energy_weighted":
        w = (2*e + 1) * cpos
        if w.sum() == 0:
            return np.nan
        ell_eff = (e*w).sum() / w.sum()
        theta_rad = np.pi / ell_eff
        return np.degrees(theta_rad)

    else:
        raise ValueError("Unknown method.")

def two_point_correlation_bruteforce(p1_map, p2_map, nside, nbins=30, nest=False,
                                     subtract_mean=True, normalize=False):
    """
    Compute binned two-point correlation functions over all pixel pairs:
      w12(theta) = < f1(n) f2(n') >_{sep=theta}
    with optional mean subtraction and optional normalization.

    subtract_mean:
      if True, uses f1 = p1 - <p1>, f2 = p2 - <p2> (covariance-like)
      if False, uses raw p1, p2.

    normalize:
      if True, returns correlation coefficient vs theta:
        w12(theta) / sqrt(w11(theta)*w22(theta))
      (Requires subtract_mean=True to be most meaningful.)

    Returns: theta_centers (deg), w12, w11, w22 (and optionally w12_norm)
    """
    npix = hp.nside2npix(nside)
    p1 = np.asarray(p1_map, float).copy()
    p2 = np.asarray(p2_map, float).copy()

    if subtract_mean:
        p1 = p1 - p1.mean()
        p2 = p2 - p2.mean()

    # Unit vectors of pixel centers
    x, y, z = hp.pix2vec(nside, np.arange(npix), nest=nest)
    vecs = np.vstack([x, y, z]).T  # (npix, 3)

    # Pairwise angular separations
    dots = np.clip(vecs @ vecs.T, -1., 1.)
    theta = np.arccos(dots)  # radians

    # Values for cross and auto products
    P12 = p1[:, None] * p2[None, :]
    P11 = p1[:, None] * p1[None, :]
    P22 = p2[:, None] * p2[None, :]

    # Bin in theta
    edges = np.linspace(0., np.pi, nbins + 1)
    bin_idx = np.digitize(theta.ravel(), edges) - 1  # 0..nbins-1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)

    def bin_mean(values):
        v = values.ravel()
        sums = np.bincount(bin_idx, weights=v, minlength=nbins)
        counts = np.bincount(bin_idx, minlength=nbins)
        out = np.zeros(nbins)
        good = counts > 0
        out[good] = sums[good] / counts[good]
        return out

    w12 = bin_mean(P12)
    w11 = bin_mean(P11)
    w22 = bin_mean(P22)

    theta_centers = 0.5 * (edges[:-1] + edges[1:])
    theta_deg = np.degrees(theta_centers)

    if normalize:
        denom = np.sqrt(np.maximum(w11, 0) * np.maximum(w22, 0))
        w12n = np.zeros_like(w12)
        good = denom > 0
        w12n[good] = w12[good] / denom[good]
        return theta_deg, w12, w11, w22, w12n

    return theta_deg, w12, w11, w22

def _safe_positive_part(x):
    return np.maximum(x, 0.)

def characteristic_scales_from_w(theta_deg, w, theta_min_deg=None, theta_max_deg=None,
                                 exclude_first_bins=1, use_positive_part=True):
    """
    Compute several characteristic scales from a binned correlation function w(theta).
    Returns a dict of scalars in degrees.

    exclude_first_bins: often exclude theta~0 bin(s) to avoid pixel/self-pair artifacts.
    use_positive_part: if True, moments use w_+ = max(w,0) to avoid cancellation.
    """
    theta_deg = np.asarray(theta_deg, float)
    w = np.asarray(w, float)
    theta_rad = np.radians(theta_deg)

    # Mask range
    mask = np.ones_like(theta_deg, dtype=bool)
    if theta_min_deg is not None:
        mask &= theta_deg >= theta_min_deg
    if theta_max_deg is not None:
        mask &= theta_deg <= theta_max_deg

    # Exclude first few bins (often dominated by resolution / self-pair issues)
    if exclude_first_bins > 0:
        idx_sorted = np.argsort(theta_deg)
        first = idx_sorted[:exclude_first_bins]
        mask[first] = False

    th = theta_rad[mask]
    ww = w[mask]

    # Choose weight
    if use_positive_part:
        ww_m = _safe_positive_part(ww)
    else:
        ww_m = ww.copy()

    # Area weight on the sphere
    area_w = np.sin(th)

    out = {}

    # 1) Peak of w (within range)
    if np.any(np.isfinite(ww)):
        out["theta_peak_w_deg"] = np.degrees(th[np.nanargmax(ww)])

    # 2) Peak of sin(theta) * w_+
    prod = area_w * ww_m
    if np.any(prod > 0):
        out["theta_peak_area_wplus_deg"] = np.degrees(th[np.argmax(prod)])
    else:
        out["theta_peak_area_wplus_deg"] = np.nan

    # 3) Centroid (first moment) with area weighting
    denom = np.trapezoid(area_w * ww_m, th)
    if denom > 0:
        numer = np.trapezoid(th * area_w * ww_m, th)
        out["theta_centroid_area_wplus_deg"] = np.degrees(numer / denom)
    else:
        out["theta_centroid_area_wplus_deg"] = np.nan

    # 4) RMS width around centroid (useful "bandwidth" of correlation)
    if denom > 0:
        th0 = np.radians(out["theta_centroid_area_wplus_deg"])
        var = np.trapezoid(((th - th0) ** 2) * area_w * ww_m, th) / denom
        out["theta_rms_width_area_wplus_deg"] = np.degrees(np.sqrt(max(var, 0.)))
    else:
        out["theta_rms_width_area_wplus_deg"] = np.nan

    # 5) First zero-crossing (from raw w, not w_+)
    # Find first sign change from + to - as theta increases
    th_deg_sorted = np.degrees(th)
    order = np.argsort(th_deg_sorted)
    ths = th_deg_sorted[order]
    wws = ww[order]
    # Identify where w crosses 0
    zc = np.nan
    for i in range(len(ths) - 1):
        if np.isfinite(wws[i]) and np.isfinite(wws[i+1]):
            if (wws[i] > 0 and wws[i+1] <= 0) or (wws[i] >= 0 and wws[i+1] < 0):
                # Linear interpolation
                t0, t1 = ths[i], ths[i+1]
                y0, y1 = wws[i], wws[i+1]
                if y1 != y0:
                    zc = t0 + (0 - y0) * (t1 - t0) / (y1 - y0)
                else:
                    zc = t0
                break
    out["theta_zero_cross_deg"] = zc
    return out

def fit_correlation_length(theta_deg, w, model="exp", theta_fit_max_deg=None, exclude_first_bins=1):
    """
    Fit a simple model w(theta) ~ A exp(-(theta/theta_c)^k) on the positive lobe.
    model:
      - "exp": k=1
      - "gauss": k=2
    Returns theta_c in degrees (np.nan if fit fails).
    """
    theta_deg = np.asarray(theta_deg, float)
    w = np.asarray(w, float)

    # Choose fit region: positive part, small angles
    mask = np.isfinite(w) & (w > 0)
    if theta_fit_max_deg is not None:
        mask &= (theta_deg <= theta_fit_max_deg)

    # Exclude first bins if desired
    if exclude_first_bins > 0:
        idx_sorted = np.argsort(theta_deg)
        mask[idx_sorted[:exclude_first_bins]] = False

    x = np.radians(theta_deg[mask])
    y = w[mask]
    if len(x) < 4:
        return np.nan

    # Define models in radians
    if model == "exp":
        def f(th, A, thc): return A * np.exp(-th / thc)
        p0 = (y.max(), np.median(x))
        bounds = ([0, 1e-6], [np.inf, np.pi])
    elif model == "gauss":
        def f(th, A, thc): return A * np.exp(-(th / thc) ** 2)
        p0 = (y.max(), np.median(x))
        bounds = ([0, 1e-6], [np.inf, np.pi])
    else:
        raise ValueError("model must be 'exp' or 'gauss'")

    try:
        popt, _ = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=10000)
        theta_c_rad = popt[1]
        return np.degrees(theta_c_rad)
    except Exception:
        return np.nan

def weighted_pair_hist(theta, pair_w, edges):
    """
    theta: NxN separations in radians
    pair_w: NxN pair weights
    edges: bin edges in radians
    Returns: histogram (sum of weights per bin)
    """
    th = theta.ravel()
    ww = pair_w.ravel()
    hist, _ = np.histogram(th, bins=edges, weights=ww)
    return hist

def debiased_characteristic_scale(a, b, vecs, p=1, nbins=90, eps=1e-12):
    """
    a, b: arrays of length N (weights), must sum to 1 (or will be normalized here)
    vecs: (N, 3) unit vectors for pixel centers
    p: moment to compute
    nbins: number of bins for histogram
    eps: small number to avoid division by zero

    Returns a characteristic scale (radians) computed from the *excess* pair distribution:
      P_ab(theta) / P_rand(theta)  for [aa, bb, ab]
    and then taking a moment of theta^p with respect to that debiased distribution.

    This avoids the sin(theta) overrepresentation in a controlled, bin-averaged way.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a / a.sum(); b = b / b.sum()

    N = a.size
    W_aa = a[:, None] * a[None, :]
    W_bb = b[:, None] * b[None, :]
    W_ab = a[:, None] * b[None, :]
    np.fill_diagonal(W_aa, 0.)
    np.fill_diagonal(W_bb, 0.)

    # baseline "random" weights for same pixel grid: uniform distribution
    u = np.full(N, 1. / N)
    W_rr = u[:, None] * u[None, :]
    W_rr_nodiag = W_rr.copy()
    np.fill_diagonal(W_rr_nodiag, 0.)

    # bin edges in radians
    edges = np.linspace(0., np.pi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    theta = geodesic_cost_matrix(vecs, p=p)
    H_aa = weighted_pair_hist(theta, W_aa, edges)
    H_bb = weighted_pair_hist(theta, W_bb, edges)
    H_ab = weighted_pair_hist(theta, W_ab, edges)
    H_rr = weighted_pair_hist(theta, W_rr, edges)
    H_rr_nodiag = weighted_pair_hist(theta, W_rr_nodiag, edges)

    # debias: ratio to baseline; add eps to avoid division issues
    R_aa = H_aa / (H_rr_nodiag + eps)
    R_bb = H_bb / (H_rr_nodiag + eps)
    R_ab = H_ab / (H_rr + eps)

    # Convert R into a proper weighting over theta (nonnegative).
    # Option 1 (common): use excess over baseline:
    X_aa = np.maximum(R_aa - 1., 0.)
    X_bb = np.maximum(R_bb - 1., 0.)
    X_ab = np.maximum(R_ab - 1., 0.)

    if X_aa.sum() == 0 or X_bb.sum() == 0 or X_ab.sum() == 0:
        # no detectable excess; return NaN or fall back to something else
        return np.nan

    # Moment of theta^p under X (discrete bins)
    return ((np.sum(X_aa * centers ** p) / np.sum(X_aa)) ** (1. / p),
            (np.sum(X_bb * centers ** p) / np.sum(X_bb)) ** (1. / p),
            (np.sum(X_ab * centers ** p) / np.sum(X_ab)) ** (1. / p))

def plot_angular_spectra(ell, Cl11, Cl22, Cl12, rho_l, redshift, snap):
    plt.figure(figsize=(10, 6))
    plt.plot(ell+1, Cl11, label='Lya auto')
    plt.plot(ell+1, Cl22, label='LyC auto')
    plt.plot(ell+1, np.abs(Cl12), label='Lya-LyC cross')
    plt.plot(ell+1, rho_l, label='Correlation')
    plt.xlabel('Multipole (l+1)')
    plt.ylabel('Power Spectrum')
    plt.title(f'Angular Power Spectra at z = {redshift:.2f}')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    fig.savefig(f'fig_no/correlation_angular_{snap}.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

def plot_two_point_correlation(theta_deg, w12, w11, w22, w12n, redshift, snap):
    plt.figure(figsize=(10, 6))
    plt.plot(theta_deg, w11, label='Lya auto')
    plt.plot(theta_deg, w22, label='LyC auto')
    plt.plot(theta_deg, w12, label='Lya-LyC cross')
    plt.plot(theta_deg, w12n, label='Lya-LyC cross (normalized)')
    plt.xlabel('Angular separation (degrees)')
    plt.ylabel('Correlation')
    plt.title(f'Two-Point Correlation at z = {redshift:.2f}')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    fig.savefig(f'fig_no/correlation_two_point_{snap}.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

def zip_data(sim='g10304', run='z8', snaps=range(189), plot_spectra=False, output_W2=False):
    n_snaps = len(snaps)
    valid_snaps = np.zeros(n_snaps, dtype=bool)
    redshifts = np.zeros(n_snaps)
    W1 = np.zeros(n_snaps)
    if output_W2:
        W2 = np.zeros(n_snaps)
    DA_Lya = np.zeros(n_snaps)
    DA_LyC = np.zeros(n_snaps)
    DA_Lya_LyC = np.zeros(n_snaps)
    DS_Lya = np.zeros(n_snaps)
    DS_LyC = np.zeros(n_snaps)
    DS_Lya_LyC = np.zeros(n_snaps)
    # DB_Lya = np.zeros(n_snaps)
    # DB_LyC = np.zeros(n_snaps)
    # DB_Lya_LyC = np.zeros(n_snaps)
    # peak_spec_Lya = np.zeros(n_snaps)
    # peak_spec_LyC = np.zeros(n_snaps)
    # peak_spec_Lya_LyC = np.zeros(n_snaps)
    energy_spec_Lya = np.zeros(n_snaps)
    energy_spec_LyC = np.zeros(n_snaps)
    energy_spec_Lya_LyC = np.zeros(n_snaps)
    # peak_corr_Lya = np.zeros(n_snaps)
    # peak_corr_LyC = np.zeros(n_snaps)
    # peak_corr_Lya_LyC = np.zeros(n_snaps)
    # peak_wp_Lya = np.zeros(n_snaps)
    # peak_wp_LyC = np.zeros(n_snaps)
    # peak_wp_Lya_LyC = np.zeros(n_snaps)
    # cent_wp_Lya = np.zeros(n_snaps)
    # cent_wp_LyC = np.zeros(n_snaps)
    # cent_wp_Lya_LyC = np.zeros(n_snaps)
    # rms_width_wp_Lya = np.zeros(n_snaps)
    # rms_width_wp_LyC = np.zeros(n_snaps)
    # rms_width_wp_Lya_LyC = np.zeros(n_snaps)
    # zero_cross_Lya = np.zeros(n_snaps)
    # zero_cross_LyC = np.zeros(n_snaps)
    # zero_cross_Lya_LyC = np.zeros(n_snaps)
    # exp_Lya = np.zeros(n_snaps)
    # gauss_Lya = np.zeros(n_snaps)
    # exp_LyC = np.zeros(n_snaps)
    # gauss_LyC = np.zeros(n_snaps)
    # exp_Lya_LyC = np.zeros(n_snaps)
    # gauss_Lya_LyC = np.zeros(n_snaps)
    tree_dir = f'{zoom_dir}/{sim}/{run}/postprocessing/colt_tree'
    ics_dir = f'{colt_dir}/{sim}/{run}/ics_tree'
    nside = 5
    npix = hp.nside2npix(nside)
    vecs = healpix_unit_vectors(nside, nest=False)
    for i, snap in enumerate(snaps):
        try:
            with h5py.File(f'{tree_dir}/Lya/Lya_{snap:03d}.hdf5','r') as f:
                map_Lya = f['map'][:]
                n_Lya = len(map_Lya)
                freq_map_Lya = f['freq_map'][:]
                freq2_map_Lya = f['freq2_map'][:]
                freq_std_Lya = np.sqrt(freq2_map_Lya - freq_map_Lya**2)
                redshifts[i] = f.attrs['z']
            with h5py.File(f'{tree_dir}/ion-eq/ion-eq_{snap:03d}.hdf5','r') as f:
                map_LyC = f['map'][:]
                n_LyC = len(map_LyC)
            if n_LyC > n_Lya:
                map_LyC = degrade_healpix_custom(map_LyC, n_Lya)
                n_LyC = len(map_LyC)
            elif n_LyC != n_Lya:
                raise ValueError(f'n_LyC ({n_LyC}) != n_Lya ({n_Lya})')
            assert n_Lya == npix, f'n_Lya ({n_Lya}) != npix ({npix})'

            W1[i] = wasserstein_sphere_exact(map_Lya, map_LyC, vecs, p=1, prefer_pot=True, verbose=plot_spectra)
            if output_W2:
                W2[i] = wasserstein_sphere_exact(map_Lya, map_LyC, vecs, p=2, prefer_pot=True, verbose=plot_spectra)
            DA_Lya[i], DA_LyC[i], DA_Lya_LyC[i], DS_Lya[i], DS_LyC[i], DS_Lya_LyC[i] = structure_length(map_Lya, map_LyC, vecs)
            # DB_Lya[i], DB_LyC[i], DB_Lya_LyC[i] = debiased_characteristic_scale(map_Lya, map_LyC, vecs, p=1, nbins=90, eps=1e-12)

            ell, Cl11, Cl22, Cl12, rho_l = angular_spectra(map_Lya, map_LyC, nside, remove_monopole=True)

            if plot_spectra:
                plot_angular_spectra(ell, Cl11, Cl22, Cl12, rho_l, redshifts[i], snap)

            # peak_spec_Lya[i] = characteristic_angle_from_spectrum(ell, Cl11, method="peak_Dl")
            # peak_spec_LyC[i] = characteristic_angle_from_spectrum(ell, Cl22, method="peak_Dl")
            # peak_spec_Lya_LyC[i] = characteristic_angle_from_spectrum(ell, np.abs(Cl12), method="peak_Dl")
            energy_spec_Lya[i] = characteristic_angle_from_spectrum(ell, Cl11, method="energy_weighted")
            energy_spec_LyC[i] = characteristic_angle_from_spectrum(ell, Cl22, method="energy_weighted")
            energy_spec_Lya_LyC[i] = characteristic_angle_from_spectrum(ell, np.abs(Cl12), method="energy_weighted")

            # Mean separation = sqrt(4π/npix)
            # theta_nn = np.sqrt(4 * np.pi / npix)  # Mean separation
            # nbins = int(2. * np.pi / theta_nn)  # Number of bins (2x mean separation)
            # theta_deg, w12, w11, w22, w12n = two_point_correlation_bruteforce(
            #     map_Lya, map_LyC, nside, nbins=nbins, subtract_mean=True, normalize=True
            # )
            # scales_11 = characteristic_scales_from_w(theta_deg, w11, exclude_first_bins=1, use_positive_part=True)
            # scales_22 = characteristic_scales_from_w(theta_deg, w22, exclude_first_bins=1, use_positive_part=True)
            # scales_12 = characteristic_scales_from_w(theta_deg, w12, exclude_first_bins=1, use_positive_part=True)
            # peak_corr_Lya[i] = scales_11['theta_peak_w_deg']
            # peak_corr_LyC[i] = scales_22['theta_peak_w_deg']
            # peak_corr_Lya_LyC[i] = scales_12['theta_peak_w_deg']
            # peak_wp_Lya[i] = scales_11['theta_peak_area_wplus_deg']
            # peak_wp_LyC[i] = scales_22['theta_peak_area_wplus_deg']
            # peak_wp_Lya_LyC[i] = scales_12['theta_peak_area_wplus_deg']
            # cent_wp_Lya[i] = scales_11['theta_centroid_area_wplus_deg']
            # cent_wp_LyC[i] = scales_22['theta_centroid_area_wplus_deg']
            # cent_wp_Lya_LyC[i] = scales_12['theta_centroid_area_wplus_deg']
            # rms_width_wp_Lya[i] = scales_11['theta_rms_width_area_wplus_deg']
            # rms_width_wp_LyC[i] = scales_22['theta_rms_width_area_wplus_deg']
            # rms_width_wp_Lya_LyC[i] = scales_12['theta_rms_width_area_wplus_deg']
            # zero_cross_Lya[i] = scales_11['theta_zero_cross_deg']
            # zero_cross_LyC[i] = scales_22['theta_zero_cross_deg']
            # zero_cross_Lya_LyC[i] = scales_12['theta_zero_cross_deg']
            # exp_Lya[i] = fit_correlation_length(theta_deg, w11, model="exp", theta_fit_max_deg=90, exclude_first_bins=1)
            # gauss_Lya[i] = fit_correlation_length(theta_deg, w11, model="gauss", theta_fit_max_deg=90, exclude_first_bins=1)
            # exp_LyC[i] = fit_correlation_length(theta_deg, w22, model="exp", theta_fit_max_deg=90, exclude_first_bins=1)
            # gauss_LyC[i] = fit_correlation_length(theta_deg, w22, model="gauss", theta_fit_max_deg=90, exclude_first_bins=1)
            # exp_Lya_LyC[i] = fit_correlation_length(theta_deg, w12, model="exp", theta_fit_max_deg=90, exclude_first_bins=1)
            # gauss_Lya_LyC[i] = fit_correlation_length(theta_deg, w12, model="gauss", theta_fit_max_deg=90, exclude_first_bins=1)
            # Save: theta_centroid_area_wplus_deg, theta_c_exp or theta_c_gau, and first zero-crossing
            # if plot_spectra:
            #     print(f'theta_nn = {theta_nn:g} radians = {np.degrees(theta_nn):g} degrees  ->  nbins = {nbins}')
            #     plot_two_point_correlation(theta_deg, w12, w11, w22, w12n, redshifts[i], snap)
            #     for t, val in zip(theta_deg[:5], w12n[:5]):
            #         print(f"{t:6.2f} deg  w12_norm={val:+.4f}")
            #     print(scales_11)
            #     print(scales_22)
            #     print(scales_12)
            #     print("theta_11 (exp fit) [deg]:", exp_Lya)
            #     print("theta_11 (gauss fit) [deg]:", gauss_Lya)
            #     print("theta_22 (exp fit) [deg]:", exp_LyC)
            #     print("theta_22 (gauss fit) [deg]:", gauss_LyC)
            #     print("theta_12 (exp fit) [deg]:", exp_Lya_LyC)
            #     print("theta_12 (gauss fit) [deg]:", gauss_Lya_LyC)
            valid_snaps[i] = True
        except:
            pass
    redshifts = redshifts[valid_snaps]
    W1 = np.degrees(W1[valid_snaps])  # Convert radians to degrees
    if output_W2:
        W2 = np.degrees(W2[valid_snaps])  # Mask invalid values
    DA_Lya = np.degrees(DA_Lya[valid_snaps])
    DA_LyC = np.degrees(DA_LyC[valid_snaps])
    DA_Lya_LyC = np.degrees(DA_Lya_LyC[valid_snaps])
    DS_Lya = np.degrees(DS_Lya[valid_snaps])
    DS_LyC = np.degrees(DS_LyC[valid_snaps])
    DS_Lya_LyC = np.degrees(DS_Lya_LyC[valid_snaps])
    # DB_Lya = np.degrees(DB_Lya[valid_snaps])
    # DB_LyC = np.degrees(DB_LyC[valid_snaps])
    # DB_Lya_LyC = np.degrees(DB_Lya_LyC[valid_snaps])
    # peak_spec_Lya = peak_spec_Lya[valid_snaps]
    # peak_spec_LyC = peak_spec_LyC[valid_snaps]
    # peak_spec_Lya_LyC = peak_spec_Lya_LyC[valid_snaps]
    energy_spec_Lya = energy_spec_Lya[valid_snaps]
    energy_spec_LyC = energy_spec_LyC[valid_snaps]
    energy_spec_Lya_LyC = energy_spec_Lya_LyC[valid_snaps]
    # peak_corr_Lya = peak_corr_Lya[valid_snaps]
    # peak_corr_LyC = peak_corr_LyC[valid_snaps]
    # peak_corr_Lya_LyC = peak_corr_Lya_LyC[valid_snaps]
    # peak_wp_Lya = peak_wp_Lya[valid_snaps]
    # peak_wp_LyC = peak_wp_LyC[valid_snaps]
    # peak_wp_Lya_LyC = peak_wp_Lya_LyC[valid_snaps]
    # cent_wp_Lya = cent_wp_Lya[valid_snaps]
    # cent_wp_LyC = cent_wp_LyC[valid_snaps]
    # cent_wp_Lya_LyC = cent_wp_Lya_LyC[valid_snaps]
    # rms_width_wp_Lya = rms_width_wp_Lya[valid_snaps]
    # rms_width_wp_LyC = rms_width_wp_LyC[valid_snaps]
    # rms_width_wp_Lya_LyC = rms_width_wp_Lya_LyC[valid_snaps]
    # zero_cross_Lya = zero_cross_Lya[valid_snaps]
    # zero_cross_LyC = zero_cross_LyC[valid_snaps]
    # zero_cross_Lya_LyC = zero_cross_Lya_LyC[valid_snaps]
    # exp_Lya = exp_Lya[valid_snaps]
    # gauss_Lya = gauss_Lya[valid_snaps]
    # exp_LyC = exp_LyC[valid_snaps]
    # gauss_LyC = gauss_LyC[valid_snaps]
    # exp_Lya_LyC = exp_Lya_LyC[valid_snaps]
    # gauss_Lya_LyC = gauss_Lya_LyC[valid_snaps]
    if plot_spectra:
        print(f'W1 = {W1} degrees')
        if output_W2:
            print(f'W2 = {W2} degrees')
        print(f'DA_Lya = {DA_Lya} degrees')
        print(f'DA_LyC = {DA_LyC} degrees')
        print(f'DA_Lya_LyC = {DA_Lya_LyC} degrees')
        print(f'DS_Lya = {DS_Lya} degrees')
        print(f'DS_LyC = {DS_LyC} degrees')
        print(f'DS_Lya_LyC = {DS_Lya_LyC} degrees')
        # print(f'DB_Lya = {DB_Lya} degrees')
        # print(f'DB_LyC = {DB_LyC} degrees')
        # print(f'DB_Lya_LyC = {DB_Lya_LyC} degrees')
        # print(f'Characteristic angle (map1, peak D_l) [deg]: {peak_spec_Lya}')
        # print(f'Characteristic angle (map2, peak D_l) [deg]: {peak_spec_LyC}')
        # print(f'Characteristic angle (|cross|, peak D_l) [deg]: {peak_spec_Lya_LyC}')
        print(f'Characteristic angle (map1, energy weighted) [deg]: {energy_spec_Lya}')
        print(f'Characteristic angle (map2, energy weighted) [deg]: {energy_spec_LyC}')
        print(f'Characteristic angle (|cross|, energy weighted) [deg]: {energy_spec_Lya_LyC}')

    with h5py.File(f'dist_data/{sim}_{run}.hdf5','w') as f:
        f.create_dataset('redshifts', data=redshifts)
        f.create_dataset('W1', data=W1)
        if output_W2:
            f.create_dataset('W2', data=W2)
        f.create_dataset('DA_Lya', data=DA_Lya)
        f.create_dataset('DA_LyC', data=DA_LyC)
        f.create_dataset('DA_Lya_LyC', data=DA_Lya_LyC)
        f.create_dataset('DS_Lya', data=DS_Lya)
        f.create_dataset('DS_LyC', data=DS_LyC)
        f.create_dataset('DS_Lya_LyC', data=DS_Lya_LyC)
        # f.create_dataset('DB_Lya', data=DB_Lya)
        # f.create_dataset('DB_LyC', data=DB_LyC)
        # f.create_dataset('DB_Lya_LyC', data=DB_Lya_LyC)
        # f.create_dataset('peak_spec_Lya', data=peak_spec_Lya)
        # f.create_dataset('peak_spec_LyC', data=peak_spec_LyC)
        # f.create_dataset('peak_spec_Lya_LyC', data=peak_spec_Lya_LyC)
        f.create_dataset('energy_spec_Lya', data=energy_spec_Lya)
        f.create_dataset('energy_spec_LyC', data=energy_spec_LyC)
        f.create_dataset('energy_spec_Lya_LyC', data=energy_spec_Lya_LyC)
        # f.create_dataset('peak_corr_Lya', data=peak_corr_Lya)
        # f.create_dataset('peak_corr_LyC', data=peak_corr_LyC)
        # f.create_dataset('peak_corr_Lya_LyC', data=peak_corr_Lya_LyC)
        # f.create_dataset('peak_wp_Lya', data=peak_wp_Lya)
        # f.create_dataset('peak_wp_LyC', data=peak_wp_LyC)
        # f.create_dataset('peak_wp_Lya_LyC', data=peak_wp_Lya_LyC)
        # f.create_dataset('cent_wp_Lya', data=cent_wp_Lya)
        # f.create_dataset('cent_wp_LyC', data=cent_wp_LyC)
        # f.create_dataset('cent_wp_Lya_LyC', data=cent_wp_Lya_LyC)
        # f.create_dataset('rms_width_wp_Lya', data=rms_width_wp_Lya)
        # f.create_dataset('rms_width_wp_LyC', data=rms_width_wp_LyC)
        # f.create_dataset('rms_width_wp_Lya_LyC', data=rms_width_wp_Lya_LyC)
        # f.create_dataset('zero_cross_Lya', data=zero_cross_Lya)
        # f.create_dataset('zero_cross_LyC', data=zero_cross_LyC)
        # f.create_dataset('zero_cross_Lya_LyC', data=zero_cross_Lya_LyC)
        # f.create_dataset('exp_Lya', data=exp_Lya)
        # f.create_dataset('exp_LyC', data=exp_LyC)
        # f.create_dataset('exp_Lya_LyC', data=exp_Lya_LyC)
        # f.create_dataset('gauss_Lya', data=gauss_Lya)
        # f.create_dataset('gauss_LyC', data=gauss_LyC)
        # f.create_dataset('gauss_Lya_LyC', data=gauss_Lya_LyC)

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

def plot_dist(sim='g5760', run='z8', n_start=0, output_W2=False):
    with h5py.File(f'dist_data/{sim}_{run}.hdf5','r') as f:
        redshifts = f['redshifts'][n_start:]
        W1 = f['W1'][n_start:]
        if output_W2:
            W2 = f['W2'][n_start:]
        DA_Lya = f['DA_Lya'][n_start:]
        DA_LyC = f['DA_LyC'][n_start:]
        DA_Lya_LyC = f['DA_Lya_LyC'][n_start:]
        DS_Lya = f['DS_Lya'][n_start:]
        DS_LyC = f['DS_LyC'][n_start:]
        DS_Lya_LyC = f['DS_Lya_LyC'][n_start:]
        # DB_Lya = f['DB_Lya'][n_start:]
        # DB_LyC = f['DB_LyC'][n_start:]
        # DB_Lya_LyC = f['DB_Lya_LyC'][n_start:]
        # peak_spec_Lya = f['peak_spec_Lya'][n_start:]
        # peak_spec_LyC = f['peak_spec_LyC'][n_start:]
        # peak_spec_Lya_LyC = f['peak_spec_Lya_LyC'][n_start:]
        energy_spec_Lya = f['energy_spec_Lya'][n_start:]
        energy_spec_LyC = f['energy_spec_LyC'][n_start:]
        energy_spec_Lya_LyC = f['energy_spec_Lya_LyC'][n_start:]
        # peak_corr_Lya = f['peak_corr_Lya'][n_start:]
        # peak_corr_LyC = f['peak_corr_LyC'][n_start:]
        # peak_corr_Lya_LyC = f['peak_corr_Lya_LyC'][n_start:]
        # peak_wp_Lya = f['peak_wp_Lya'][n_start:]
        # peak_wp_LyC = f['peak_wp_LyC'][n_start:]
        # peak_wp_Lya_LyC = f['peak_wp_Lya_LyC'][n_start:]
        # cent_wp_Lya = f['cent_wp_Lya'][n_start:]
        # cent_wp_LyC = f['cent_wp_LyC'][n_start:]
        # cent_wp_Lya_LyC = f['cent_wp_Lya_LyC'][n_start:]
        # rms_width_wp_Lya = f['rms_width_wp_Lya'][n_start:]
        # rms_width_wp_LyC = f['rms_width_wp_LyC'][n_start:]
        # rms_width_wp_Lya_LyC = f['rms_width_wp_Lya_LyC'][n_start:]
        # zero_cross_Lya = f['zero_cross_Lya'][n_start:]
        # zero_cross_LyC = f['zero_cross_LyC'][n_start:]
        # zero_cross_Lya_LyC = f['zero_cross_Lya_LyC'][n_start:]
        # exp_Lya = f['exp_Lya'][n_start:]
        # exp_LyC = f['exp_LyC'][n_start:]
        # exp_Lya_LyC = f['exp_Lya_LyC'][n_start:]
        # gauss_Lya = f['gauss_Lya'][n_start:]
        # gauss_LyC = f['gauss_LyC'][n_start:]
        # gauss_Lya_LyC = f['gauss_Lya_LyC'][n_start:]

    with h5py.File(f'data/{sim}_{run}.hdf5','r') as f:
        f_esc_Lya = f['f_esc_Lya'][n_start:]
        f_esc_LyC = f['f_esc_LyC'][n_start:]
        f_Lya_LyC = f['f_Lya_LyC'][n_start:]

    # Plot W1, W2 vs redshift
    fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
    ax.plot(redshifts, W1, c='C0', label=r'$W_1$')
    if output_W2:
        ax.plot(redshifts, W2, c='C1', label=r'$W_2$')
    set_redshift_axis(ax)
    ax.set_ylabel(r'${\rm Optimal\ Transport\ \,(degrees)}$', fontsize=15)
    ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    fig.savefig(f'fig_yes/W1.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

    # Plot DA, DS, DB vs. redshift
    fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
    ax.plot(redshifts, 1. - DA_Lya/90., c='C0', ls='--', label=r'$\delta_{\rm area}^{\rm Ly\alpha}$')
    ax.plot(redshifts, 1. - DA_LyC/90., c='C1', ls='--', label=r'$\delta_{\rm area}^{\rm LyC}$')
    ax.plot(redshifts, 1. - DA_Lya_LyC/90., c='C2', ls='--', label=r'$\delta_{\rm area}^{\rm Ly\alpha \times LyC}$')
    # ax.plot(redshifts, 1. - DB_Lya/90., c='C0', ls=':', label=r'$\delta_{\rm hist}^{\rm Ly\alpha}$')
    # ax.plot(redshifts, 1. - DB_LyC/90., c='C1', ls=':', label=r'$\delta_{\rm hist}^{\rm LyC}$')
    # ax.plot(redshifts, 1. - DB_Lya_LyC/90., c='C2', ls=':', label=r'$\delta_{\rm hist}^{\rm Ly\alpha \times LyC}$')
    ax.plot(redshifts, 1. - DS_Lya/90., c='C0', label=r'$\delta_{\rm sep}^{\rm Ly\alpha}$')
    ax.plot(redshifts, 1. - DS_LyC/90., c='C1', label=r'$\delta_{\rm sep}^{\rm LyC}$')
    ax.plot(redshifts, 1. - DS_Lya_LyC/90., c='C2', label=r'$\delta_{\rm sep}^{\rm Ly\alpha \times LyC}$')
    set_redshift_axis(ax)
    ax.set_ylabel(r'${\rm Overclustering\ \,(1 - \theta\,/\,90^\circ\!)}$', fontsize=15)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    fig.savefig(f'fig_yes/DA_DS.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

    # Plot peak_spec, energy_spec, peak_corr, peak_wp, cent_wp, rms_width_wp, zero_cross, exp, gauss
    # fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
    # ax.plot(redshifts, peak_spec_Lya, c='C0', label=r'$\theta_{\rm peak,spec}^{\rm Ly\alpha}$')
    # ax.plot(redshifts, peak_spec_LyC, c='C1', label=r'$\theta_{\rm peak,spec}^{\rm LyC}$')
    # ax.plot(redshifts, peak_spec_Lya_LyC, c='C2', label=r'$\theta_{\rm peak,spec}^{\rm Ly\alpha \times LyC}$')
    # ax.plot(redshifts, peak_corr_Lya, c='C0', ls='--', label=r'$\theta_{\rm peak,corr}^{\rm Ly\alpha}$')
    # ax.plot(redshifts, peak_corr_LyC, c='C1', ls='--', label=r'$\theta_{\rm peak,corr}^{\rm LyC}$')
    # ax.plot(redshifts, peak_corr_Lya_LyC, c='C2', ls='--', label=r'$\theta_{\rm peak,corr}^{\rm Ly\alpha \times LyC}$')
    # ax.plot(redshifts, peak_wp_Lya, c='C0', ls=':', label=r'$\theta_{\rm peak,wp}^{\rm Ly\alpha}$')
    # ax.plot(redshifts, peak_wp_LyC, c='C1', ls=':', label=r'$\theta_{\rm peak,wp}^{\rm LyC}$')
    # ax.plot(redshifts, peak_wp_Lya_LyC, c='C2', ls=':', label=r'$\theta_{\rm peak,wp}^{\rm Ly\alpha \times LyC}$')
    # set_redshift_axis(ax)
    # ax.set_ylabel(r'${\rm Peaks\ \,(degrees)}$', fontsize=15)
    # ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    # fig.savefig(f'fig_no/peak_spec.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    # plt.close()

    fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
    ax.plot(redshifts, energy_spec_Lya, c='C0', label=r'$\theta_{\rm energy,spec}^{\rm Ly\alpha}$')
    ax.plot(redshifts, energy_spec_LyC, c='C1', label=r'$\theta_{\rm energy,spec}^{\rm LyC}$')
    ax.plot(redshifts, energy_spec_Lya_LyC, c='C2', label=r'$\theta_{\rm energy,spec}^{\rm Ly\alpha \times LyC}$')
    set_redshift_axis(ax)
    ax.set_ylabel(r'${\rm Spectrum\ Energy\ \,(degrees)}$', fontsize=15)
    ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    fig.savefig(f'fig_no/energy_spec.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

    # fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
    # ax.plot(redshifts, cent_wp_Lya, c='C0', label=r'$\theta_{\rm cent,wp}^{\rm Ly\alpha}$')
    # ax.plot(redshifts, cent_wp_LyC, c='C1', label=r'$\theta_{\rm cent,wp}^{\rm LyC}$')
    # ax.plot(redshifts, cent_wp_Lya_LyC, c='C2', label=r'$\theta_{\rm cent,wp}^{\rm Ly\alpha \times LyC}$')
    # set_redshift_axis(ax)
    # ax.set_ylabel(r'${\rm Centroid\ W+\ \,(degrees)}$', fontsize=15)
    # ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    # fig.savefig(f'fig_no/cent_wp.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    # plt.close()

    # fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
    # ax.plot(redshifts, rms_width_wp_Lya, c='C0', label=r'$\theta_{\rm rms,width,wp}^{\rm Ly\alpha}$')
    # ax.plot(redshifts, rms_width_wp_LyC, c='C1', label=r'$\theta_{\rm rms,width,wp}^{\rm LyC}$')
    # ax.plot(redshifts, rms_width_wp_Lya_LyC, c='C2', label=r'$\theta_{\rm rms,width,wp}^{\rm Ly\alpha \times LyC}$')
    # set_redshift_axis(ax)
    # ax.set_ylabel(r'${\rm RMS\ Width\ W+\ \,(degrees)}$', fontsize=15)
    # ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    # fig.savefig(f'fig_no/rms_width_wp.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    # plt.close()

    # fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
    # ax.plot(redshifts, zero_cross_Lya, c='C0', label=r'$\theta_{\rm zero,cross}^{\rm Ly\alpha}$')
    # ax.plot(redshifts, zero_cross_LyC, c='C1', label=r'$\theta_{\rm zero,cross}^{\rm LyC}$')
    # ax.plot(redshifts, zero_cross_Lya_LyC, c='C2', label=r'$\theta_{\rm zero,cross}^{\rm Ly\alpha \times LyC}$')
    # set_redshift_axis(ax)
    # ax.set_ylabel(r'${\rm Zero\ Crossing\ \,(degrees)}$', fontsize=15)
    # ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    # fig.savefig(f'fig_no/zero_cross.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    # plt.close()

    # fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
    # ax.plot(redshifts, exp_Lya, c='C0', label=r'$\theta_{\rm exp}^{\rm Ly\alpha}$')
    # ax.plot(redshifts, exp_LyC, c='C1', label=r'$\theta_{\rm exp}^{\rm LyC}$')
    # ax.plot(redshifts, exp_Lya_LyC, c='C2', label=r'$\theta_{\rm exp}^{\rm Ly\alpha \times LyC}$')
    # ax.plot(redshifts, gauss_Lya, c='C0', ls='--', label=r'$\theta_{\rm gauss}^{\rm Ly\alpha}$')
    # ax.plot(redshifts, gauss_LyC, c='C1', ls='--', label=r'$\theta_{\rm gauss}^{\rm LyC}$')
    # ax.plot(redshifts, gauss_Lya_LyC, c='C2', ls='--', label=r'$\theta_{\rm gauss}^{\rm Ly\alpha \times LyC}$')
    # set_redshift_axis(ax)
    # ax.set_ylabel(r'${\rm Exponential\ Fit\ \,(degrees)}$', fontsize=15)
    # ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    # fig.savefig(f'fig_no/exp_gauss.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    # plt.close()

    # Plot W2 vs W1
    if output_W2:
        fig = plt.figure(figsize=(4.5,4.5)); ax = plt.axes([0,0,1,1])
        ax.plot([8,80], [8,80], color=[.8,.8,.8])
        ax.scatter(W1, W2, color='C0', alpha=0.4, lw=0)
        ax.set_xlabel(r'${\rm Optimal\ Transport\ 1\ \,(degrees)}$', fontsize=15)
        ax.set_ylabel(r'${\rm Optimal\ Transport\ 2\ \,(degrees)}$', fontsize=15)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.minorticks_on()
        fig.savefig(f'fig_yes/W2_vs_W1.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
        plt.close()

    # Plot W1 vs DS
    fig = plt.figure(figsize=(4.5,4.5)); ax = plt.axes([0,0,1,1])
    ax.scatter(W1, 1. - DS_Lya/90., color='C0', alpha=0.4, lw=0, label=r'${\rm Ly\alpha}$')
    ax.scatter(W1, 1. - DS_LyC/90., color='C1', alpha=0.4, lw=0, label=r'${\rm LyC}$')
    ax.scatter(W1, 1. - DS_Lya_LyC/90., color='C2', alpha=0.4, lw=0, label=r'${\rm Ly\alpha} \times {\rm LyC}$')
    ax.set_xlabel(r'${\rm Optimal\ Transport\ \,(degrees)}$', fontsize=15)
    ax.set_ylabel(r'${\rm Overclustering\ \,(1 - \theta\,/\,90^\circ\!)}$', fontsize=15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.set_ylim(1e-3, 1)
    ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    fig.savefig(f'fig_yes/W1_vs_DS.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

    # Plot W1 vs DA
    fig = plt.figure(figsize=(4.5,4.5)); ax = plt.axes([0,0,1,1])
    ax.scatter(W1, 1. - DA_Lya/90., color='C0', alpha=0.4, lw=0, label=r'${\rm Ly\alpha}$')
    ax.scatter(W1, 1. - DA_LyC/90., color='C1', alpha=0.4, lw=0, label=r'${\rm LyC}$')
    ax.scatter(W1, 1. - DA_Lya_LyC/90., color='C2', alpha=0.4, lw=0, label=r'${\rm Ly\alpha} \times {\rm LyC}$')
    ax.set_xlabel(r'${\rm Optimal\ Transport\ \,(degrees)}$', fontsize=15)
    ax.set_ylabel(r'${\rm Overclustering\ \,(1 - \theta\,/\,90^\circ\!)}$', fontsize=15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.set_ylim(1e-3, 1)
    ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    fig.savefig(f'fig_yes/W1_vs_DA.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

    # Plot W1 vs DB
    # fig = plt.figure(figsize=(4.5,4.5)); ax = plt.axes([0,0,1,1])
    # ax.scatter(W1, 1. - DB_Lya/90., color='C0', alpha=0.4, lw=0, label=r'${\rm Ly\alpha}$')
    # ax.scatter(W1, 1. - DB_LyC/90., color='C1', alpha=0.4, lw=0, label=r'${\rm LyC}$')
    # ax.scatter(W1, 1. - DB_Lya_LyC/90., color='C2', alpha=0.4, lw=0, label=r'${\rm Ly\alpha} \times {\rm LyC}$')
    # ax.set_xlabel(r'${\rm Optimal\ Transport\ \,(degrees)}$', fontsize=15)
    # ax.set_ylabel(r'${\rm Overclustering\ \,(1 - \theta\,/\,90^\circ\!)}$', fontsize=15)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.minorticks_on()
    # ax.set_ylim(1e-3, 1)
    # ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    # fig.savefig(f'fig_yes/W1_vs_DB.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    # plt.close()

    # Plot DA vs DS
    fig = plt.figure(figsize=(4.5,4.5)); ax = plt.axes([0,0,1,1])
    ax.plot([1e-3,1], [1e-3,1], color=[.8,.8,.8])
    ax.scatter(1. - DS_Lya/90., 1. - DA_Lya/90., color='C0', alpha=0.4, lw=0, label=r'${\rm Ly\alpha}$')
    ax.scatter(1. - DS_LyC/90., 1. - DA_LyC/90., color='C1', alpha=0.4, lw=0, label=r'${\rm LyC}$')
    ax.scatter(1. - DS_Lya_LyC/90., 1. - DA_Lya_LyC/90., color='C2', alpha=0.4, lw=0, label=r'${\rm Ly\alpha} \times {\rm LyC}$')
    ax.set_xlabel(r'${\rm \mu\ Overclustering\ \,(1 - \theta\,/\,90^\circ\!)}$', fontsize=15)
    ax.set_ylabel(r'${\rm \Omega\ Overclustering\ \,(1 - \theta\,/\,90^\circ\!)}$', fontsize=15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(1e-3, 1)
    ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    fig.savefig(f'fig_yes/DA_vs_DS.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

    # f_esc vs DS
    fig = plt.figure(figsize=(4.5,4.5)); ax = plt.axes([0,0,1,1])
    ax.scatter(1. - DS_Lya/90., f_esc_Lya, color='C0', alpha=0.4, lw=0, label=r'${\rm Ly\alpha}$')
    ax.scatter(1. - DS_LyC/90., f_esc_LyC, color='C1', alpha=0.4, lw=0, label=r'${\rm LyC}$')
    ax.scatter(1. - DS_Lya_LyC/90., f_Lya_LyC, color='C2', alpha=0.4, lw=0, label=r'${\rm Ly\alpha} \times {\rm LyC}$')
    ax.set_xlabel(r'${\rm Overclustering\ \,(1 - \theta\,/\,90^\circ\!)}$', fontsize=15)
    ax.set_ylabel(r'${\rm LyC\ Escape\ Fraction}$', fontsize=15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    xmin, xmax = 1e-3, 1
    ax.set_xlim(xmin, xmax)
    Lxmin,Lxmax = int(np.ceil(np.log10(xmin))), int(np.floor(np.log10(xmax)))
    ticks = np.linspace(Lxmin, Lxmax, Lxmax-Lxmin+1)
    ax.set_xticks(10**ticks); ax.set_xticklabels([r'$10^{%g}$' % tick for tick in ticks], fontsize=13)
    ymin, ymax = 1e-3, 1
    ax.set_ylim(ymin, ymax)
    Lymin,Lymax = int(np.ceil(np.log10(ymin))), int(np.floor(np.log10(ymax)))
    ticks = np.linspace(Lymin, Lymax, Lymax-Lymin+1)
    ax.set_yticks(10**ticks); ax.set_yticklabels([r'$10^{%g}$' % tick for tick in ticks], fontsize=13)
    ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
    fig.savefig(f'fig_yes/f_esc_vs_DS.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
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

# zip_data(sim=sim, run=run, snaps=[168], plot_spectra=True)
# zip_data(sim=sim, run=run)
plot_dist(sim=sim, run=run)
