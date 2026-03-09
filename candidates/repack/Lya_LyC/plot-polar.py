import numpy as np
import h5py, os
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib import patheffects
import cmasher as cmr
import cmocean

from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf
sigma_68, sigma_95, sigma_99 = erf(1./np.sqrt(2.)), erf(2./np.sqrt(2.)), erf(3./np.sqrt(2.))
percentiles = [50., 50.*(1.-sigma_68), 50.*(1.+sigma_68)]
# percentiles = [50., 50.*(1.-sigma_68), 50.*(1.+sigma_68), 50.*(1.-sigma_95), 50.*(1.+sigma_95), 50.*(1.-sigma_99), 50.*(1.+sigma_99)]
n_percentiles = len(percentiles)
# print('percentiles =', percentiles)

zoom_dir = '/orcd/data/mvogelsb/004/Thesan-Zooms'
# colt_dir = f'{zoom_dir}-COLT'
colt_dir = '/nfs/mvogelsblab001/Lab/Thesan-Zooms-COLT'

pc   = 3.085677581467192e18 # Units: 1 pc  = 3e18 cm
kpc  = 1.0e3 * pc           # Units: 1 kpc = 3e21 cm
arcsec = 648000. / np.pi    # arseconds per radian
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

v8 = np.zeros([8,3])  # Camera direction vectors
x8 = np.zeros([8,3])  # Camera x-axis vectors
y8 = np.zeros([8,3])  # Camera y-axis vectors
p8 = np.zeros([8,2])  # Projected image directions to healpix rotation vector
mu8 = np.zeros(8)  # Angle to the healpix focus

def safe_ratio(numerator, denominator, zero=0.):
    ratio = np.copy(numerator)
    mask = denominator > 0.
    ratio[mask] /= denominator[mask]
    ratio[~mask] = zero
    return ratio

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

# def fraction_of_angles(data, threshold):
#     isort = np.argsort(data)  # Sort in ascending order
#     data_sorted = data[isort]  # Data in sorted order
#     cdf = np.cumsum(data_sorted)  # Cumulative distribution
#     cdf /= cdf[-1]  # Normalize to 1
#     cdf_target = 1. - threshold  # Want fraction above threshold
#     i = np.searchsorted(cdf, cdf_target)  # Index of first element with cdf >= cdf_target
#     if i == 0:
#         return 1.
#     # print(f'cdf_target = {cdf_target:g}, cdf[i-1] = {cdf[i-1]:g}, cdf[i] = {cdf[i]:g}')
#     return (float(i-1) + (cdf_target - cdf[i-1]) / (cdf[i] - cdf[i-1])) / float(len(data))

def fraction_of_angles(data, thresholds):
    """
    For each threshold t in thresholds, return the fraction f of elements (angles)
    needed so that the *top* f fraction contains t of the total weight.

    Equivalent to: sort ascending, form CDF of sorted weights, and find the point
    where CDF reaches (1 - t). Then f = that point expressed as a fraction of N.

    Parameters
    ----------
    data : array_like, shape (N,)
        Non-negative weights per angle/bin.
    thresholds : array_like
        Thresholds in (0, 1], e.g. [0.5, 0.9].

    Returns
    -------
    frac : ndarray, shape (len(thresholds),)
        Fraction of angles (in [0,1]) that contain the top 'threshold' fraction
        of the total weight.
    """
    x = np.asarray(data, dtype=float).ravel()
    thr = np.asarray(thresholds, dtype=float).ravel()

    if x.size == 0:
        return np.full(thr.shape, np.nan)
    if np.any(thr <= 0) or np.any(thr > 1):
        raise ValueError("thresholds must be in (0, 1].")
    if np.any(x < 0):
        raise ValueError("data should be non-negative weights.")
    total = x.sum()
    if total <= 0 or not np.isfinite(total):
        # all zeros (or pathological) -> no meaningful concentration; define as 0
        return np.zeros(thr.shape, dtype=float)

    # Sort ascending and compute normalized CDF
    xs = np.sort(x)
    cdf = np.cumsum(xs) / total  # in (0,1]

    # Targets on the CDF: keep the bottom (1 - threshold) mass
    targets = 1.0 - thr

    # Search insertion indices for each target
    i = np.searchsorted(cdf, targets, side="left")  # vectorized

    N = xs.size
    frac = np.empty_like(targets, dtype=float)

    # Case i == 0: target <= cdf[0] -> essentially everything is in the top tail
    m0 = (i == 0)
    frac[m0] = 1.0

    # Case i >= N: target > cdf[-1] (should only happen for numerical weirdness)
    mN = (i >= N)
    frac[mN] = 0.0

    # General case: linearly interpolate between (i-1) and i in CDF space
    mg = ~(m0 | mN)
    ig = i[mg]
    t = targets[mg]
    c0 = cdf[ig - 1]
    c1 = cdf[ig]
    pos = (ig - 1) + (t - c0) / (c1 - c0)  # fractional index position where CDF hits target
    frac[mg] = pos / float(N)  # Convert to top fraction of angles
    return frac

def gini(x):
    """
    Compute the Gini coefficient of a 1D array.
    Assumes x is non-negative.

    Returns:
        float: Gini coefficient in [0, 1].
    """
    # x = np.asarray(x, dtype=float).flatten()
    if np.any(x < 0.):
        raise ValueError("Gini coefficient requires non-negative values.")
    if np.allclose(x, 0.):
        return 0.
    n = float(x.size)
    x_sorted = np.sort(x)
    index = np.arange(1, x.size + 1)
    return np.sum((2. * index - n - 1.) * x_sorted) / (n * np.sum(x_sorted))

def get_thetas_i(m, i_cam):
    nside = hp.npix2nside(m.size)
    ipix = np.arange(m.size)
    vecs = np.array(hp.pix2vec(nside, ipix)).T
    return np.arctan2(np.dot(vecs, y8[i_cam]), np.dot(vecs, x8[i_cam]))

def _dipole_rot_from_map(m):
    nside = hp.npix2nside(m.size)
    ipix = np.arange(m.size)
    vx, vy, vz = hp.pix2vec(nside, ipix)
    w = np.asarray(m, dtype=float)
    bad = (~np.isfinite(w)) | (w == hp.UNSEEN)
    w = w.copy()
    w[bad] = 0.
    d = np.array([np.sum(w * vx), np.sum(w * vy), np.sum(w * vz)])
    norm = np.linalg.norm(d)
    if not np.isfinite(norm) or norm == 0.:
        return None
    d /= norm
    lon = np.degrees(np.arctan2(d[1], d[0]))
    lat = np.degrees(np.arcsin(d[2]))
    return (lon, lat, 0.)

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
    return (lon, lat, 0.)

def get_rot(data, recenter_on_dipole=False, recenter_on_smoothed_max=False, smooth_pix=4.):
    map = hp.pixelfunc.ma_to_array(data)
    if recenter_on_smoothed_max:
        return _smoothed_max_rot_from_map(map, smooth_pix=smooth_pix)
    elif recenter_on_dipole:
        return _dipole_rot_from_map(map)
    else:
        return None

def fwhm_periodic(y, dx=1., x0=0., i0=None, y0=None):
    """
    Compute FWHM (in bins) for a periodic array.
    Returns: (width_bins, left_pos, right_pos)
    left_pos/right_pos are fractional indices in [0, N).
    """
    y = np.asarray(y, float)
    N = y.size
    if i0 is None:
        i0 = np.argmax(y)
    if y0 is None:
        y0 = y[i0]
    half = 0.5 * y0
    if np.min(y) >= half:
        return float(N) * dx, -x0, float(N) * dx - x0

    def prev(i): return (i - 1) % N
    def next(i): return (i + 1) % N

    # Walk left from i0 until crossing below half
    i = i0
    while y[i] >= half:
        j = prev(i)
        if j == i0:
            return float(N), (i0 - N/2) % N, (i0 + N/2) % N
        if y[j] < half:
            # crossing between j (below) and i (above)
            yb, ya = y[j], y[i]
            t = (half - yb) / (ya - yb)  # in (0,1]
            left_pos = (j + t) % N
            break
        i = j

    # Walk right from i0 until crossing below half
    i = i0
    while y[i] >= half:
        j = next(i)
        if j == i0:
            return float(N), (i0 - N/2) % N, (i0 + N/2) % N
        if y[j] < half:
            # crossing between i (above) and j (below)
            ya, yb = y[i], y[j]
            t = (half - ya) / (yb - ya)  # in (0,1)
            right_pos = (i + t) % N
            break
        i = j

    width_bins = (right_pos - left_pos) % N
    return width_bins * dx, left_pos * dx - x0, right_pos * dx - x0

def _wrap_angle(theta):
    """Wrap angle(s) to [-pi, pi)."""
    return (theta + np.pi) % (2 * np.pi) - np.pi

def _zscore(x, eps=1e-15):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sig = np.nanstd(x)
    if not np.isfinite(sig) or sig < eps:
        return np.full_like(x, np.nan)
    return (x - mu) / sig

def _ols_fit(X, y):
    """
    Ordinary least squares with intercept via lstsq.
    Returns: beta, yhat, residuals, R2
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)

    # Add intercept
    X1 = np.column_stack([np.ones(len(y)), X])

    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    yhat = X1 @ beta
    resid = y - yhat

    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean())**2)
    R2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return beta, yhat, resid, R2

def max_circular_xcorr_rho(theta, x, y, *, use_fft=True, mask=None):
    """
    Compute max over circular lags of Pearson correlation rho_xy(Δ) and lag Δθ*.

    Assumptions:
      - theta is 1D, sorted, uniformly spaced over 2π (or nearly so).
      - x, y are 1D arrays sampled at theta bins (same length).

    Returns:
      rho_max: max over lags of correlation (float)
      dtheta_star: lag (in radians) at which rho is maximized (float, wrapped to [-pi, pi))
      rho_by_lag: array of rho for lags 0..N-1 (optional diagnostic)
    """
    theta = np.asarray(theta, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(theta)
    if len(x) != n or len(y) != n:
        raise ValueError("theta, x, y must have same length")

    # Optional mask of valid points (e.g. finite and positive constraints)
    if mask is None:
        mask = np.isfinite(theta) & np.isfinite(x) & np.isfinite(y)
    else:
        mask = np.asarray(mask, bool) & np.isfinite(theta) & np.isfinite(x) & np.isfinite(y)

    # For circular cross-correlation, missing values are awkward.
    # Require all finite or the user should pre-impute/regularize.
    if not np.all(mask):
        raise ValueError(
            "max_circular_xcorr_rho requires all finite samples for circular correlation. "
            "Pre-mask/impute or restrict to finite bins before calling."
        )

    # Check uniform spacing
    dtheta = np.median(np.diff(theta))
    if not np.allclose(np.diff(theta), dtheta, rtol=1e-3, atol=1e-8):
        # Still proceed, but interpret Δθ from index*median(dtheta)
        pass

    zx = _zscore(x)
    zy = _zscore(y)
    if np.all(~np.isfinite(zx)) or np.all(~np.isfinite(zy)):
        return np.nan, np.nan, np.full(n, np.nan)

    # Circular normalized cross-correlation:
    # rho[k] = (1/n) * sum_i zx[i] * zy[(i+k) mod n]
    if use_fft:
        Fx = np.fft.rfft(zx)
        Fy = np.fft.rfft(zy)
        # Cross-correlation in time domain:
        c = np.fft.irfft(Fx * np.conj(Fy), n=n)
        rho_by_lag = c / n
    else:
        rho_by_lag = np.empty(n, float)
        for k in range(n):
            rho_by_lag[k] = np.dot(zx, np.roll(zy, k)) / n

    k_star = int(np.nanargmax(rho_by_lag))
    rho_max = float(rho_by_lag[k_star])

    # Convert lag index to angle. For circular correlation, both +k and -(n-k) represent same shift.
    dtheta_star = _wrap_angle(k_star * dtheta)
    return rho_max, dtheta_star, rho_by_lag

def partial_corr_and_incremental_R2(
    lyc, o32, logOIII, logOII, *, allow_nan=False
):
    """
    Compute:
      - partial correlation corr(resid(LyC|controls), resid(O32|controls))
      - incremental R^2 of adding O32 to controls in predicting LyC

    Returns dict with:
      r_partial, R2_controls, R2_full, dR2, resid_lyc, resid_o32
    """
    lyc = np.asarray(lyc, float)
    o32 = np.asarray(o32, float)
    logOIII = np.asarray(logOIII, float)
    logOII = np.asarray(logOII, float)

    if not (len(lyc) == len(o32) == len(logOIII) == len(logOII)):
        raise ValueError("All inputs must have same length")

    if allow_nan:
        m = np.isfinite(lyc) & np.isfinite(o32) & np.isfinite(logOIII) & np.isfinite(logOII)
        lyc, o32, logOIII, logOII = lyc[m], o32[m], logOIII[m], logOII[m]

    # Controls matrix
    C = np.column_stack([logOIII, logOII])

    # Residualize LyC and O32 on controls
    _, _, resid_lyc, _ = _ols_fit(C, lyc)
    _, _, resid_o32, _ = _ols_fit(C, o32)

    # Partial correlation = Pearson corr of residuals
    rz1 = _zscore(resid_lyc)
    rz2 = _zscore(resid_o32)
    if np.any(~np.isfinite(rz1)) or np.any(~np.isfinite(rz2)):
        r_partial = np.nan
    else:
        r_partial = float(np.dot(rz1, rz2) / len(rz1))

    # Incremental R^2 for predicting LyC:
    # controls-only vs controls+o32
    _, _, _, R2_controls = _ols_fit(C, lyc)
    X_full = np.column_stack([logOIII, logOII, o32])
    _, _, _, R2_full = _ols_fit(X_full, lyc)
    dR2 = np.nan if (np.isnan(R2_controls) or np.isnan(R2_full)) else float(R2_full - R2_controls)

    return dict(
        r_partial=r_partial,
        R2_controls=float(R2_controls) if np.isfinite(R2_controls) else np.nan,
        R2_full=float(R2_full) if np.isfinite(R2_full) else np.nan,
        dR2=dR2,
        resid_lyc=resid_lyc,
        resid_o32=resid_o32,
    )

def acf_correlation_length(theta, x, *, max_lag=None):
    """
    Estimate a correlation length from circular ACF:
      ACF[k] = (1/n) sum zx[i] * zx[(i+k) mod n], for k=0..n-1
    Define L_corr as the smallest positive lag where ACF[k] <= exp(-1).
    If never crosses, return pi (half circle) as a conservative bound.

    Returns:
      L_corr (radians), acf (array)
    """
    theta = np.asarray(theta, float)
    x = np.asarray(x, float)
    n = len(theta)

    if max_lag is None:
        max_lag = n // 2  # only need up to half circle

    dtheta = np.median(np.diff(theta))
    zx = _zscore(x)
    if np.all(~np.isfinite(zx)):
        return np.nan, np.full(max_lag + 1, np.nan)

    Fx = np.fft.rfft(zx)
    c = np.fft.irfft(Fx * np.conj(Fx), n=n) / n
    acf = c[: max_lag + 1]  # k=0..max_lag

    target = np.exp(-1.0)
    # Find first k>=1 where acf[k] <= target
    k_cross = None
    for k in range(1, len(acf)):
        if acf[k] <= target:
            k_cross = k
            break

    if k_cross is None:
        L_corr = np.pi
    else:
        # Optional: linear interpolation between k-1 and k for more precision
        k0 = k_cross - 1
        y0, y1 = acf[k0], acf[k_cross]
        if y1 == y0:
            frac = 0.0
        else:
            frac = (target - y0) / (y1 - y0)
            frac = float(np.clip(frac, 0.0, 1.0))
        L_corr = (k0 + frac) * dtheta

    return float(L_corr), acf


def low_mode_fourier_power_fraction(x, *, mmax=2):
    """
    Smoothness/asymmetry proxy:
      fraction of Fourier power in modes 1..mmax relative to total power (1..Nyquist),
      excluding the mean (mode 0).

    For a smooth, low-order pattern, this fraction is large.
    For a patchy/high-frequency pattern, it is small.

    Returns:
      frac_low (float), power (array of power by mode index)
    """
    x = np.asarray(x, float)
    zx = _zscore(x)
    n = len(zx)
    if np.all(~np.isfinite(zx)):
        return np.nan, np.full(n // 2 + 1, np.nan)

    F = np.fft.rfft(zx)
    power = np.abs(F) ** 2

    # Exclude mean (k=0)
    total = power[1:].sum()
    if total <= 0 or not np.isfinite(total):
        return np.nan, power

    mmax = int(np.clip(mmax, 1, len(power) - 1))
    low = power[1 : mmax + 1].sum()
    return float(low / total), power


def fill_fwhm(ax, x_range, y_range):
    ax.fill_between(x_range, [y_range[0], y_range[0]], [y_range[1], y_range[1]], alpha=0.2, color='gray', lw=0.)

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

def set_theta_axis(ax):
    ax.set_xlabel(r'${\rm Angle\ \,(degrees)}$', fontsize=15)
    ax.set_xlim(-180, 180)
    ticks = np.linspace(-180, 180, 5)
    ax.set_xticks(ticks); ax.set_xticklabels([r'$%g$' % tick for tick in ticks], fontsize=13)

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

def zip_data(sim='g10304', run='z8', snaps=range(189), n_cameras=8, n_bins=181, thin_deg=3., fat_deg=10., verbose=False, plot_theta=False):
    n_snaps = len(snaps)
    valid_snaps = np.zeros(n_snaps, dtype=bool)
    redshifts = np.zeros(n_snaps)
    r_int = np.zeros([n_snaps, 3])
    r_esc = np.zeros([n_snaps, 3])
    L_O3 = np.zeros(n_snaps)
    L_O2 = np.zeros(n_snaps)
    L_Hb = np.zeros(n_snaps)
    angle_cam_max = np.zeros([n_snaps, n_cameras])
    theta_LyC_max = np.zeros([n_snaps, n_cameras])
    theta_LyC = np.zeros([n_snaps, n_cameras])
    theta_O32 = np.zeros([n_snaps, n_cameras])
    theta_R3 = np.zeros([n_snaps, n_cameras])
    theta_O3 = np.zeros([n_snaps, n_cameras])
    theta_O2 = np.zeros([n_snaps, n_cameras])
    theta_Hb = np.zeros([n_snaps, n_cameras])
    radius_O3 = np.zeros([n_snaps, n_cameras])
    radius_O2 = np.zeros([n_snaps, n_cameras])
    radius_Hb = np.zeros([n_snaps, n_cameras])
    FWHM_O3 = np.zeros([n_snaps, n_cameras])
    FWHM_O2 = np.zeros([n_snaps, n_cameras])
    FWHM_Hb = np.zeros([n_snaps, n_cameras])
    FWHM_O32 = np.zeros([n_snaps, n_cameras])
    FWHM_R3 = np.zeros([n_snaps, n_cameras])
    FWHM_LyC = np.zeros([n_snaps, n_cameras])
    f50_O3 = np.zeros([n_snaps, n_cameras])
    f50_O2 = np.zeros([n_snaps, n_cameras])
    f50_Hb = np.zeros([n_snaps, n_cameras])
    f50_O32 = np.zeros([n_snaps, n_cameras])
    f50_R3 = np.zeros([n_snaps, n_cameras])
    f50_LyC = np.zeros([n_snaps, n_cameras])
    f90_O3 = np.zeros([n_snaps, n_cameras])
    f90_O2 = np.zeros([n_snaps, n_cameras])
    f90_Hb = np.zeros([n_snaps, n_cameras])
    f90_O32 = np.zeros([n_snaps, n_cameras])
    f90_R3 = np.zeros([n_snaps, n_cameras])
    f90_LyC = np.zeros([n_snaps, n_cameras])
    gini_O3 = np.zeros([n_snaps, n_cameras])
    gini_O2 = np.zeros([n_snaps, n_cameras])
    gini_Hb = np.zeros([n_snaps, n_cameras])
    gini_O32 = np.zeros([n_snaps, n_cameras])
    gini_R3 = np.zeros([n_snaps, n_cameras])
    gini_LyC = np.zeros([n_snaps, n_cameras])
    rho_max_O3_LyC = np.zeros([n_snaps, n_cameras])
    rho_max_O2_LyC = np.zeros([n_snaps, n_cameras])
    rho_max_Hb_LyC = np.zeros([n_snaps, n_cameras])
    rho_max_O32_LyC = np.zeros([n_snaps, n_cameras])
    rho_max_R3_LyC = np.zeros([n_snaps, n_cameras])
    rho_max_O32_R3 = np.zeros([n_snaps, n_cameras])
    dtheta_star_O3_LyC = np.zeros([n_snaps, n_cameras])
    dtheta_star_O2_LyC = np.zeros([n_snaps, n_cameras])
    dtheta_star_Hb_LyC = np.zeros([n_snaps, n_cameras])
    dtheta_star_O32_LyC = np.zeros([n_snaps, n_cameras])
    dtheta_star_R3_LyC = np.zeros([n_snaps, n_cameras])
    dtheta_star_O32_R3 = np.zeros([n_snaps, n_cameras])
    R2_controls = np.zeros([n_snaps, n_cameras])
    dR2_add_O32_to_controls = np.zeros([n_snaps, n_cameras])
    O32_lowmode_frac_m2 = np.zeros([n_snaps, n_cameras])
    tree_dir = f'{zoom_dir}/{sim}/{run}/postprocessing/colt_tree'
    ics_dir = f'{colt_dir}/{sim}/{run}/ics_tree'
    T_edges = np.linspace(-np.pi, np.pi, n_bins+1)
    T_centers = (T_edges[:-1] + T_edges[1:]) / 2.
    dT = T_centers[1] - T_centers[0]  # Resolution in radians
    dT_deg = np.degrees(dT)  # Resolution in degrees
    i_0 = np.argmin(np.abs(T_centers))
    sigma_thin = thin_deg / dT_deg  # Convert from degrees to bins
    sigma_fat = fat_deg / dT_deg
    # kernel = np.ones(5) / 5.
    # kernel_fat = np.ones(15) / 15.
    for i, snap in enumerate(snaps):
        try:
            with h5py.File(f'{tree_dir}/ion-eq/ion-eq_{snap:03d}.hdf5','r') as f:
                map_LyC = f['map'][:]
                v8[:], x8[:], y8[:] = f['camera_directions'][:], f['camera_xaxes'][:], f['camera_yaxes'][:]
            rot = get_rot(map_LyC, recenter_on_dipole=False, recenter_on_smoothed_max=True, smooth_pix=4.)
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
                angle_cam_max[i,:] = np.arccos(mu8)  # Convert to radians
                theta_LyC_max[i,:] = np.arctan2(p8[:,1], p8[:,0])  # Polar angle in radians
                if verbose: print(f'Angles: {np.array_str(np.degrees(angle_cam_max[i,:]), precision=2)}')
            with h5py.File(f'{tree_dir}/M1500/M1500_{snap:03d}.hdf5','r') as f:
                redshifts[i] = f.attrs['z']
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
                with h5py.File(f'{ics_dir}/colt_{snap:03d}.hdf5', 'r') as f_star:
                    r_star = f_star['r_star'][:][star_mask,:]  # Star positions [cm]
                vir_mask = np.sum(r_star**2, axis=1) < Rvir**2  # Virial mask
                r_star = r_star[vir_mask]  # Star positions [cm]
                Ndot_int = g['Ndot_int'][:][vir_mask]  # Photon rate [photons/s]
                Ndot_esc = g['Ndot_esc'][:][vir_mask]  # Escaped rate [photons/s]
                r_int[i,:] = np.sum(r_star * Ndot_int[:,None], axis=0) / np.sum(Ndot_int)  # Intrinsic center of light position [cm]
                r_esc[i,:] = np.sum(r_star * Ndot_esc[:,None], axis=0) / np.sum(Ndot_esc)  # Escaped center of light position [cm]
                radius_int = np.sqrt(np.sum(r_int[i,:]**2))  # Intrinsic radius [cm]
                radius_esc = np.sqrt(np.sum(r_esc[i,:]**2))  # Escaped radius [cm]
                x_int = np.array([np.dot(r_int[i,:], x8[j]) / Rvir for j in range(8)])  # Projected x-directions (int)
                y_int = np.array([np.dot(r_int[i,:], y8[j]) / Rvir for j in range(8)])  # Projected y-directions (int)
                x_esc = np.array([np.dot(r_esc[i,:], x8[j]) / Rvir for j in range(8)])  # Projected x-directions (esc)
                y_esc = np.array([np.dot(r_esc[i,:], y8[j]) / Rvir for j in range(8)])  # Projected y-directions (esc)
                x_int[x_int > 0.99] = 0.99; x_int[x_int < -0.99] = -0.99
                y_int[y_int > 0.99] = 0.99; y_int[y_int < -0.99] = -0.99
                x_esc[x_esc > 0.99] = 0.99; x_esc[x_esc < -0.99] = -0.99
                y_esc[y_esc > 0.99] = 0.99; y_esc[y_esc < -0.99] = -0.99
                del r_star, star_mask, vir_mask, Ndot_int, Ndot_esc
                if verbose:
                    print(f'r_int = ({r_int[i,0]/kpc:g}, {r_int[i,1]/kpc:g}, {r_int[i,2]/kpc:g}) kpc  |  radius = {radius_int/kpc:g} kpc = {radius_int/Rvir:g} Rvir')
                    print(f'r_esc = ({r_esc[i,0]/kpc:g}, {r_esc[i,1]/kpc:g}, {r_esc[i,2]/kpc:g}) kpc  |  radius = {radius_esc/kpc:g} kpc = {radius_esc/Rvir:g} Rvir')
                    print(f'|r_int - r_esc| = {np.sqrt(np.sum((r_int[i,:] - r_esc[i,:])**2))/pc:g} pc')
                    print(f'Rvir = {Rvir/kpc:g} kpc')
            with h5py.File(f'{tree_dir}/OIII-5008/OIII-5008_{snap:03d}.hdf5','r') as f:
                R_O3 = f.attrs['image_radius']  # Image radius [cm]
                L_O3[i] = f.attrs['L_tot']
                images_O3 = f['images'][:] # Surface brightness [erg/s/cm^2/arcsec^2]
                assert n_cameras == images_O3.shape[0], f'Number of cameras does not match for OIII: {n_cameras} vs {images_O3.shape[0]}'
            with h5py.File(f'{tree_dir}/OII-3727-3730/OII-3727-3730_{snap:03d}.hdf5','r') as f:
                R_O2 = f.attrs['image_radius']  # Image radius [cm]
                assert R_O3 == R_O2, f'Image radius does not match for OII: {R_O3:g} vs {R_O2:g}'
                L_O2[i] = f.attrs['L_tot']
                images_O2 = f['images'][:] # Surface brightness [erg/s/cm^2/arcsec^2]
                assert n_cameras == images_O2.shape[0], f'Number of cameras does not match for OII: {n_cameras} vs {images_O2.shape[0]}'
            with h5py.File(f'{tree_dir}/Hb/Hb_{snap:03d}.hdf5','r') as f:
                R_Hb = f.attrs['image_radius']  # Image radius [cm]
                assert R_O3 == R_Hb, f'Image radius does not match for H-beta: {R_O3:g} vs {R_Hb:g}'
                L_Hb[i] = f.attrs['L_tot']
                images_Hb = f['images'][:] # Surface brightness [erg/s/cm^2/arcsec^2]
                assert n_cameras == images_Hb.shape[0], f'Number of cameras does not match for H-beta: {n_cameras} vs {images_Hb.shape[0]}'
            n_pixels = images_O3.shape[1]
            if True:  # Cut out the middle 50% of OIII and OII images
                assert n_pixels == images_O2.shape[1], f'OIII and OII images are different sizes: {images_O3.shape} vs {images_O2.shape}'
                assert n_pixels == images_Hb.shape[1], f'OIII and H-beta images are different sizes: {images_O3.shape} vs {images_Hb.shape}'
                assert n_pixels == images_O3.shape[2], f'OIII image is not square: {images_O3.shape}'
                assert n_pixels == images_O2.shape[2], f'OII image is not square: {images_O2.shape}'
                assert n_pixels == images_Hb.shape[2], f'H-beta image is not square: {images_Hb.shape}'
                assert n_pixels % 4 == 0, f'Image size is not divisible by 4: {n_pixels}'
                n4 = n_pixels // 4
                images_O3 = images_O3[:, n4:-n4, n4:-n4]
                images_O2 = images_O2[:, n4:-n4, n4:-n4]
                images_Hb = images_Hb[:, n4:-n4, n4:-n4]
                n_pixels //= 2  # All images are half the size
                R_O3 /= 2.
                R_O2 /= 2.
                R_Hb /= 2.
                assert R_O3 == Rvir, f'Image radius does not match for OIII: {R_O3:g} vs {Rvir:g}'
            # Histograms in polar angles
            hist_O3 = np.zeros([n_cameras, n_bins])
            hist_O2 = np.zeros([n_cameras, n_bins])
            hist_O32 = np.zeros([n_cameras, n_bins])
            hist_Hb = np.zeros([n_cameras, n_bins])
            hist_R3 = np.zeros([n_cameras, n_bins])
            hist_LyC = np.zeros([n_cameras, n_bins])
            lower_O3, upper_O3 = np.zeros(n_cameras), np.zeros(n_cameras)
            lower_O2, upper_O2 = np.zeros(n_cameras), np.zeros(n_cameras)
            lower_Hb, upper_Hb = np.zeros(n_cameras), np.zeros(n_cameras)
            lower_O32, upper_O32 = np.zeros(n_cameras), np.zeros(n_cameras)
            lower_R3, upper_R3 = np.zeros(n_cameras), np.zeros(n_cameras)
            lower_LyC, upper_LyC = np.zeros(n_cameras), np.zeros(n_cameras)
            for i_cam in range(n_cameras):
                # Create (X,Y) grids and polar (R,T) grids
                x_edges = np.linspace(-1., 1., n_pixels+1) - x_esc[i_cam]
                y_edges = np.linspace(-1., 1., n_pixels+1) - y_esc[i_cam]
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2.
                y_centers = (y_edges[:-1] + y_edges[1:]) / 2.
                X, Y = np.meshgrid(x_centers, y_centers)
                R = np.sqrt(X**2 + Y**2)
                T = np.arctan2(Y, X)
                # Flatten and mask the images within the unit circle
                R = R.reshape(-1)
                mask = (R > 0.) & (R <= 1.)
                R = R[mask]
                X = X.reshape(-1)[mask]
                Y = Y.reshape(-1)[mask]
                T = T.reshape(-1)[mask]
                X /= R  # Normalize by radius to get unit vectors
                Y /= R
                image_O3 = images_O3[i_cam].reshape(-1)[mask]
                image_O2 = images_O2[i_cam].reshape(-1)[mask]
                image_Hb = images_Hb[i_cam].reshape(-1)[mask]
                # Calculate the OIII directional moment
                f_O3 = np.sum(image_O3)
                x_O3 = np.sum(image_O3 * X) / f_O3
                y_O3 = np.sum(image_O3 * Y) / f_O3
                radius_O3[i,i_cam] = np.sqrt(x_O3**2 + y_O3**2)
                theta_O3[i,i_cam] = np.arctan2(y_O3, x_O3)
                # Calculate the OII directional moment
                f_O2 = np.sum(image_O2)
                x_O2 = np.sum(image_O2 * X) / f_O2
                y_O2 = np.sum(image_O2 * Y) / f_O2
                radius_O2[i,i_cam] = np.sqrt(x_O2**2 + y_O2**2)
                theta_O2[i,i_cam] = np.arctan2(y_O2, x_O2)
                # Calculate the H-beta directional moment
                f_Hb = np.sum(image_Hb)
                x_Hb = np.sum(image_Hb * X) / f_Hb
                y_Hb = np.sum(image_Hb * Y) / f_Hb
                radius_Hb[i,i_cam] = np.sqrt(x_Hb**2 + y_Hb**2)
                theta_Hb[i,i_cam] = np.arctan2(y_Hb, x_Hb)
                # Calculate histograms
                n_mask = len(R)
                hist_O3[i_cam], _ = np.histogram(T, T_edges, weights=image_O3)
                hist_O2[i_cam], _ = np.histogram(T, T_edges, weights=image_O2)
                hist_Hb[i_cam], _ = np.histogram(T, T_edges, weights=image_Hb)
                T_LyC = get_thetas_i(map_LyC, i_cam)  # LyC map projected directions
                hist_LyC[i_cam], _ = np.histogram(T_LyC, T_edges, weights=map_LyC)
                hist_LyC_norm, _ = np.histogram(T_LyC, T_edges, weights=np.ones_like(T_LyC))
                hist_LyC[i_cam][hist_LyC_norm > 0] /= hist_LyC_norm[hist_LyC_norm > 0]
                # Smooth the histograms with periodic boundary conditions
                O3_thin = gaussian_filter1d(hist_O3[i_cam], sigma_thin, mode="wrap")
                O2_thin = gaussian_filter1d(hist_O2[i_cam], sigma_thin, mode="wrap")
                Hb_thin = gaussian_filter1d(hist_Hb[i_cam], sigma_thin, mode="wrap")
                O32_thin = safe_ratio(O3_thin, O2_thin)  # OIII / OII (smoothed)
                R3_thin = safe_ratio(O3_thin, Hb_thin)  # OIII / H-beta (smoothed)
                LyC_thin = gaussian_filter1d(hist_LyC[i_cam], sigma_thin, mode="wrap")
                O3_fat = gaussian_filter1d(hist_O3[i_cam], sigma_fat, mode="wrap")
                O2_fat = gaussian_filter1d(hist_O2[i_cam], sigma_fat, mode="wrap")
                Hb_fat = gaussian_filter1d(hist_Hb[i_cam], sigma_fat, mode="wrap")
                O32_fat = safe_ratio(O3_fat, O2_fat)  # OIII / OII (smoothed)
                R3_fat = safe_ratio(O3_fat, Hb_fat)  # OIII / H-beta (smoothed)
                LyC_fat = gaussian_filter1d(hist_LyC[i_cam], sigma_fat, mode="wrap")
                # Calculate the maximum of the OIII / OII and OIII / H-beta histograms
                i_O3 = np.argmax(O3_fat)
                i_O2 = np.argmax(O2_fat)
                i_Hb = np.argmax(Hb_fat)
                i_O32 = np.argmax(O32_fat)
                i_R3 = np.argmax(R3_fat)
                i_LyC = np.argmax(LyC_fat)
                theta_O32[i,i_cam] = T_centers[i_O32]
                theta_R3[i,i_cam] = T_centers[i_R3]
                theta_LyC[i,i_cam] = T_centers[i_LyC]
                # hist_O3[i_cam] = convolve1d(hist_O3[i_cam], kernel, mode='wrap', origin=0)
                # hist_O2[i_cam] = convolve1d(hist_O2[i_cam], kernel, mode='wrap', origin=0)
                # hist_Hb[i_cam] = convolve1d(hist_Hb[i_cam], kernel, mode='wrap', origin=0)
                # hist_O32[i_cam] = safe_ratio(hist_O3[i_cam], hist_O2[i_cam])  # OIII / OII
                # hist_R3[i_cam] = safe_ratio(hist_O3[i_cam], hist_Hb[i_cam])  # OIII / H-beta
                # hist_O3_fat = convolve1d(hist_O3[i_cam], kernel_fat, mode='wrap', origin=0)
                # hist_O2_fat = convolve1d(hist_O2[i_cam], kernel_fat, mode='wrap', origin=0)
                # hist_Hb_fat = convolve1d(hist_Hb[i_cam], kernel_fat, mode='wrap', origin=0)
                # hist_O32_fat = safe_ratio(hist_O3_fat, hist_O2_fat)  # OIII / OII
                # hist_R3_fat = safe_ratio(hist_O3_fat, hist_Hb_fat)  # OIII / H-beta
                # Calculate the maximum of the OIII / OII and OIII / H-beta histograms
                # i_O3 = np.argmax(hist_O3_fat)
                # i_O2 = np.argmax(hist_O2_fat)
                # i_Hb = np.argmax(hist_Hb_fat)
                # i_O32 = np.argmax(hist_O32_fat)
                # i_R3 = np.argmax(hist_R3_fat)
                # theta_O32[i,i_cam] = T_centers[i_O32]
                # theta_R3[i,i_cam] = T_centers[i_R3]
                # x_O32 = np.cos(T_centers[i_O32])  # Unit vector in the max O32 direction
                # y_O32 = np.sin(T_centers[i_O32])
                # mu_O32_O3[i_cam] = (x_O32 * x_O3 + y_O32 * y_O3) / r_O3  # Cosine of angle between OIII and O32 directions
                # mu_O32_O2[i_cam] = (x_O32 * x_O2 + y_O32 * y_O2) / r_O2  # Cosine of angle between OII and O32 directions
                # mu_O32_LyC[i_cam] = x_O32 * p8[i_cam,0] + y_O32 * p8[i_cam,1]  # Cosine of angle between LyC and O32 directions
                # mu_LyC_O3[i_cam] = (p8[i_cam,0] * x_O3 + p8[i_cam,1] * y_O3) / r_O3  # Cosine of angle between LyC and OIII directions
                # mu_LyC_O2[i_cam] = (p8[i_cam,0] * x_O2 + p8[i_cam,1] * y_O2) / r_O2  # Cosine of angle between LyC and OII directions
                # mu_O3_O2[i_cam] = (x_O3 * x_O2 + y_O3 * y_O2) / (r_O3 * r_O2)  # Cosine of angle between OIII and OII directions
                if verbose:
                    # print(f'min/max T = {np.min(T):g}, {np.max(T):g} = {np.min(T)/np.pi:g} pi, {np.max(T)/np.pi:g} pi')
                    # print(f'n_bins = {n_bins}, n_mask = {n_mask}, n_mask/n_bins = {float(n_mask)/float(n_bins):g}')
                    print(f'radius [O3, O2, Hb] = [{radius_O3[i,i_cam]:g}, {radius_O2[i,i_cam]:g}, {radius_Hb[i,i_cam]:g}]')
                    print(f'theta [O3, O2, Hb, O32, R3, LyC] = [{np.degrees(theta_O3[i,i_cam]):g}, {np.degrees(theta_O2[i,i_cam]):g}, {np.degrees(theta_Hb[i,i_cam]):g}, {np.degrees(theta_O32[i,i_cam]):g}, {np.degrees(theta_R3[i,i_cam]):g}, {np.degrees(theta_LyC[i,i_cam]):g}] deg')
                    print(f'i [O3, O2, Hb, O32, R3, LyC] = [{i_O3}, {i_O2}, {i_Hb}, {i_O32}, {i_R3}, {i_LyC}]')
                # Recenter using fat curve peak and calculate FWHM
                di_O3 = i_0 - i_O3
                di_O2 = i_0 - i_O2
                di_Hb = i_0 - i_Hb
                di_O32 = i_0 - i_O32
                di_R3 = i_0 - i_R3
                di_LyC = i_0 - i_LyC
                hist_O3[i_cam] = np.roll(O3_thin, di_O3)
                hist_O2[i_cam] = np.roll(O2_thin, di_O2)
                hist_Hb[i_cam] = np.roll(Hb_thin, di_Hb)
                hist_O32[i_cam] = np.roll(O32_thin, di_O32)
                hist_R3[i_cam] = np.roll(R3_thin, di_R3)
                hist_LyC[i_cam] = np.roll(LyC_thin, di_LyC)
                # hist_O3[i_cam] = np.roll(O3_fat, di_O3)
                # hist_O2[i_cam] = np.roll(O2_fat, di_O2)
                # hist_Hb[i_cam] = np.roll(Hb_fat, di_Hb)
                # hist_O32[i_cam] = np.roll(O32_fat, di_O32)
                # hist_R3[i_cam] = np.roll(R3_fat, di_R3)
                # hist_LyC[i_cam] = np.roll(LyC_fat, di_LyC)
                FWHM_O3[i,i_cam], lower_O3[i_cam], upper_O3[i_cam] = fwhm_periodic(hist_O3[i_cam], dx=dT, x0=np.pi, i0=i_0, y0=np.max(O3_fat))
                FWHM_O2[i,i_cam], lower_O2[i_cam], upper_O2[i_cam] = fwhm_periodic(hist_O2[i_cam], dx=dT, x0=np.pi, i0=i_0, y0=np.max(O2_fat))
                FWHM_Hb[i,i_cam], lower_Hb[i_cam], upper_Hb[i_cam] = fwhm_periodic(hist_Hb[i_cam], dx=dT, x0=np.pi, i0=i_0, y0=np.max(Hb_fat))
                FWHM_O32[i,i_cam], lower_O32[i_cam], upper_O32[i_cam] = fwhm_periodic(hist_O32[i_cam], dx=dT, x0=np.pi, i0=i_0, y0=np.max(O32_fat))
                FWHM_R3[i,i_cam], lower_R3[i_cam], upper_R3[i_cam] = fwhm_periodic(hist_R3[i_cam], dx=dT, x0=np.pi, i0=i_0, y0=np.max(R3_fat))
                FWHM_LyC[i,i_cam], lower_LyC[i_cam], upper_LyC[i_cam] = fwhm_periodic(hist_LyC[i_cam], dx=dT, x0=np.pi, i0=i_0, y0=np.max(LyC_fat))
                # Calculate the fraction of angles contributing [50%, 90%] of the flux
                f50_O3[i,i_cam], f90_O3[i,i_cam] = fraction_of_angles(hist_O3[i_cam], [0.5, 0.9])
                f50_O2[i,i_cam], f90_O2[i,i_cam] = fraction_of_angles(hist_O2[i_cam], [0.5, 0.9])
                f50_Hb[i,i_cam], f90_Hb[i,i_cam] = fraction_of_angles(hist_Hb[i_cam], [0.5, 0.9])
                f50_O32[i,i_cam], f90_O32[i,i_cam] = fraction_of_angles(hist_O32[i_cam], [0.5, 0.9])
                f50_R3[i,i_cam], f90_R3[i,i_cam] = fraction_of_angles(hist_R3[i_cam], [0.5, 0.9])
                f50_LyC[i,i_cam], f90_LyC[i,i_cam] = fraction_of_angles(hist_LyC[i_cam], [0.5, 0.9])
                # Calculate the Gini coefficient
                gini_O3[i,i_cam] = gini(hist_O3[i_cam])
                gini_O2[i,i_cam] = gini(hist_O2[i_cam])
                gini_Hb[i,i_cam] = gini(hist_Hb[i_cam])
                gini_O32[i,i_cam] = gini(hist_O32[i_cam])
                gini_R3[i,i_cam] = gini(hist_R3[i_cam])
                gini_LyC[i,i_cam] = gini(hist_LyC[i_cam])
                # Calculate max lag circular cross correlation, partial correlation, and low mode fraction
                rho_max_O3_LyC[i,i_cam], dtheta_star_O3_LyC[i,i_cam], _ = max_circular_xcorr_rho(T_centers, hist_O3[i_cam], hist_LyC[i_cam], use_fft=True)
                rho_max_O2_LyC[i,i_cam], dtheta_star_O2_LyC[i,i_cam], _ = max_circular_xcorr_rho(T_centers, hist_O2[i_cam], hist_LyC[i_cam], use_fft=True)
                rho_max_Hb_LyC[i,i_cam], dtheta_star_Hb_LyC[i,i_cam], _ = max_circular_xcorr_rho(T_centers, hist_Hb[i_cam], hist_LyC[i_cam], use_fft=True)
                rho_max_O32_LyC[i,i_cam], dtheta_star_O32_LyC[i,i_cam], _ = max_circular_xcorr_rho(T_centers, hist_O32[i_cam], hist_LyC[i_cam], use_fft=True)
                rho_max_R3_LyC[i,i_cam], dtheta_star_R3_LyC[i,i_cam], _ = max_circular_xcorr_rho(T_centers, hist_R3[i_cam], hist_LyC[i_cam], use_fft=True)
                rho_max_O32_R3[i,i_cam], dtheta_star_O32_R3[i,i_cam], _ = max_circular_xcorr_rho(T_centers, hist_O32[i_cam], hist_R3[i_cam], use_fft=True)
                pa = partial_corr_and_incremental_R2(np.log10(hist_LyC[i_cam]), hist_O32[i_cam], np.log10(hist_O3[i_cam]), np.log10(hist_O2[i_cam]), allow_nan=False)
                R2_controls[i,i_cam], dR2_add_O32_to_controls[i,i_cam] = pa["R2_controls"], pa["dR2"]
                O32_lowmode_frac_m2[i,i_cam], _ = low_mode_fourier_power_fraction(hist_O32[i_cam], mmax=2)
            if verbose:
                print(f'i_0 = {i_0}, T_centers[i_0] = {np.degrees(T_centers[i_0]):g} deg')
                print(f'min/max O_32 = {np.min(hist_O32):g}, {np.max(hist_O32):g}')
                print(f'min/max R3 = {np.min(hist_R3):g}, {np.max(hist_R3):g}')
                print(f'min/max LyC = {np.min(hist_LyC):g}, {np.max(hist_LyC):g}')
            if plot_theta:
                # Plot hist_O32 vs theta
                fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
                for i_cam in range(n_cameras):
                    ax.plot(np.degrees(T_centers), hist_O32[i_cam], alpha=0.5, label=str(i_cam))
                ax.plot(np.degrees(T_centers), hist_O32[-1], c='k')
                fill_fwhm(ax, [np.degrees(lower_O32[-1]), np.degrees(upper_O32[-1])], [np.min(hist_O32[-1]), np.max(hist_O32[-1])])
                set_theta_axis(ax)
                set_yaxis(ax, r'${\rm O}_{32}$', 0, 10, 3, yp=2)
                ax.minorticks_on()
                ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
                fig.savefig(f'fig_polar/O32_theta_{snap:03d}.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
                plt.close()
                # Plot hist_R3 vs theta
                fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
                for i_cam in range(n_cameras):
                    ax.plot(np.degrees(T_centers), hist_R3[i_cam], alpha=0.5, label=str(i_cam))
                ax.plot(np.degrees(T_centers), hist_R3[-1], c='k')
                fill_fwhm(ax, [np.degrees(lower_R3[-1]), np.degrees(upper_R3[-1])], [np.min(hist_R3[-1]), np.max(hist_R3[-1])])
                set_theta_axis(ax)
                set_yaxis(ax, r'${\rm R}_3$') #, 0, 10, 3, yp=2)
                ax.minorticks_on()
                ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
                fig.savefig(f'fig_polar/R3_theta_{snap:03d}.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
                plt.close()
                # Plot hist_O3 vs theta
                fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
                for i_cam in range(n_cameras):
                    ax.plot(np.degrees(T_centers), hist_O3[i_cam], alpha=0.5, label=str(i_cam))
                ax.plot(np.degrees(T_centers), hist_O3[-1], c='k')
                fill_fwhm(ax, [np.degrees(lower_O3[-1]), np.degrees(upper_O3[-1])], [np.min(hist_O3[-1]), np.max(hist_O3[-1])])
                set_theta_axis(ax)
                set_log_yaxis(ax, r'${\rm OIII}$')
                ax.minorticks_on()
                ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
                fig.savefig(f'fig_polar/O3_theta_{snap:03d}.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
                plt.close()
                # Plot hist_O2 vs theta
                fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
                for i_cam in range(n_cameras):
                    ax.plot(np.degrees(T_centers), hist_O2[i_cam], alpha=0.5, label=str(i_cam))
                ax.plot(np.degrees(T_centers), hist_O2[-1], c='k')
                fill_fwhm(ax, [np.degrees(lower_O2[-1]), np.degrees(upper_O2[-1])], [np.min(hist_O2[-1]), np.max(hist_O2[-1])])
                set_theta_axis(ax)
                set_log_yaxis(ax, r'${\rm OII}$')
                ax.minorticks_on()
                ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
                fig.savefig(f'fig_polar/O2_theta_{snap:03d}.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
                plt.close()
                # Plot hist_Hb vs theta
                fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
                for i_cam in range(n_cameras):
                    ax.plot(np.degrees(T_centers), hist_Hb[i_cam], alpha=0.5, label=str(i_cam))
                ax.plot(np.degrees(T_centers), hist_Hb[-1], c='k')
                fill_fwhm(ax, [np.degrees(lower_Hb[-1]), np.degrees(upper_Hb[-1])], [np.min(hist_Hb[-1]), np.max(hist_Hb[-1])])
                set_theta_axis(ax)
                set_log_yaxis(ax, r'${\rm H}\beta$')
                ax.minorticks_on()
                ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
                fig.savefig(f'fig_polar/Hb_theta_{snap:03d}.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
                plt.close()
                # Plot hist_LyC vs theta
                fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
                for i_cam in range(n_cameras):
                    ax.plot(np.degrees(T_centers), hist_LyC[i_cam], alpha=0.5, label=str(i_cam))
                ax.plot(np.degrees(T_centers), hist_LyC[-1], c='k')
                fill_fwhm(ax, [np.degrees(lower_LyC[-1]), np.degrees(upper_LyC[-1])], [np.min(hist_LyC[-1]), np.max(hist_LyC[-1])])
                set_theta_axis(ax)
                set_log_yaxis(ax, r'${\rm LyC}$')
                ax.minorticks_on()
                ax.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2.5, fontsize=12)
                fig.savefig(f'fig_polar/LyC_theta_{snap:03d}.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
                plt.close()
                # Corner plot of hist combinations (O3, O2, Hb, O32, R3, LyC)
                if i_cam == 7:
                    fig, axes = plt.subplots(6, 6, figsize=(15, 15))
                    # fig.suptitle(f'Corner Plot - Snapshot {snap:03d}', fontsize=16)
                    hist_data = [np.log10(hist_O3[-1]), np.log10(hist_O2[-1]), np.log10(hist_Hb[-1]), hist_O32[-1], hist_R3[-1], np.log10(hist_LyC[-1])]
                    labels = ['log OIII', 'log OII', 'log Hβ', 'O32', 'R3', 'log LyC']
                    colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan']
                    for ii in range(6):
                        for jj in range(6):
                            ax = axes[ii, jj]
                            if ii == jj:  # Diagonal: histogram
                                ax.hist(hist_data[ii], bins=30, alpha=0.7, color=colors[ii], edgecolor='black')
                                if jj == 0:  # First column only
                                    ax.set_ylabel('Count', fontsize=10)
                                if ii == 5:  # Bottom row only
                                    ax.set_xlabel(labels[ii], fontsize=10)
                            elif ii > jj:  # Lower triangle: scatter plot
                                ax.scatter(hist_data[jj], hist_data[ii], alpha=0.5, s=1, color=colors[jj])
                                if jj == 0:  # First column only
                                    ax.set_ylabel(labels[ii], fontsize=10)
                                if ii == 5:  # Bottom row only
                                    ax.set_xlabel(labels[jj], fontsize=10)
                            else:  # Upper triangle: correlation coefficient
                                corr = np.corrcoef(hist_data[jj], hist_data[ii])[0, 1]
                                ax.text(0.5, 0.5, f'r = {corr:.3f}', ha='center', va='center',
                                    transform=ax.transAxes, fontsize=12,
                                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                                ax.set_xticks([])
                                ax.set_yticks([])
                            # Remove ticks except for first column (y) and bottom row (x)
                            if jj > 0:  # Not first column
                                ax.set_yticks([])
                            if ii < 5:  # Not bottom row
                                ax.set_xticks([])
                            ax.grid(True, alpha=0.3)
                    plt.subplots_adjust(wspace=0, hspace=0)
                    os.makedirs('fig_polar', exist_ok=True)
                    fig.savefig(f'fig_polar/corner_plot_{snap:03d}.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
                    plt.close()
                    if True:  # Explore additional statistics
                        stat_O32 = hist_O32[-1]
                        stat_LyC = np.log10(hist_LyC[-1])
                        stat_O3 = np.log10(hist_O3[-1])
                        stat_O2 = np.log10(hist_O2[-1])
                        stat_dict = {}
                        stat_dict['rho_max_O32_LyC'], stat_dict['dtheta_star_O32_LyC'], _ = max_circular_xcorr_rho(T_centers, hist_O32[-1], hist_LyC[-1], use_fft=True)
                        pa = partial_corr_and_incremental_R2(np.log10(hist_LyC[-1]), hist_O32[-1], np.log10(hist_O3[-1]), np.log10(hist_O2[-1]), allow_nan=False)
                        stat_dict['dR2_add_O32_to_controls'] = pa["dR2"]
                        stat_dict['R2_controls'] = pa["R2_controls"]
                        stat_dict['R2_full'] = pa["R2_full"]
                        stat_dict['r_partial_LyC_O32_given_logOIII_logOII'] = pa["r_partial"]
                        stat_dict['O32_Lcorr'], _ = acf_correlation_length(T_centers, stat_O32)
                        stat_dict['O32_lowmode_frac_m2'], _ = low_mode_fourier_power_fraction(stat_O32, mmax=2)
                        print(stat_dict)
                        def diag(name, a):
                            a = np.asarray(a, float)
                            print(f'{name}: finite {np.isfinite(a).mean():g}, std {np.nanstd(a):g}, min/max {np.nanmin(a):g}/{np.nanmax(a):g}')
                        diag("LyC", stat_LyC)
                        diag("O32", stat_O32)
                        diag("O3",  stat_O3)
                        diag("O2",  stat_O2)
                        print(f'corr(O3,O2)={np.corrcoef(stat_O3, stat_O2)[0,1]:g}')
            valid_snaps[i] = True
        except:
            pass
    redshifts = redshifts[valid_snaps]
    r_int = r_int[valid_snaps]
    r_esc = r_esc[valid_snaps]
    L_O3 = L_O3[valid_snaps]
    L_O2 = L_O2[valid_snaps]
    L_Hb = L_Hb[valid_snaps]
    angle_cam_max = angle_cam_max[valid_snaps]
    theta_LyC_max = theta_LyC_max[valid_snaps]
    theta_LyC = theta_LyC[valid_snaps]
    theta_O32 = theta_O32[valid_snaps]
    theta_R3 = theta_R3[valid_snaps]
    theta_O3 = theta_O3[valid_snaps]
    theta_O2 = theta_O2[valid_snaps]
    theta_Hb = theta_Hb[valid_snaps]
    radius_O3 = radius_O3[valid_snaps]
    radius_O2 = radius_O2[valid_snaps]
    radius_Hb = radius_Hb[valid_snaps]
    FWHM_O3 = FWHM_O3[valid_snaps]
    FWHM_O2 = FWHM_O2[valid_snaps]
    FWHM_Hb = FWHM_Hb[valid_snaps]
    FWHM_O32 = FWHM_O32[valid_snaps]
    FWHM_R3 = FWHM_R3[valid_snaps]
    FWHM_LyC = FWHM_LyC[valid_snaps]
    f50_O3 = f50_O3[valid_snaps]
    f50_O2 = f50_O2[valid_snaps]
    f50_Hb = f50_Hb[valid_snaps]
    f50_O32 = f50_O32[valid_snaps]
    f50_R3 = f50_R3[valid_snaps]
    f50_LyC = f50_LyC[valid_snaps]
    f90_O3 = f90_O3[valid_snaps]
    f90_O2 = f90_O2[valid_snaps]
    f90_Hb = f90_Hb[valid_snaps]
    f90_O32 = f90_O32[valid_snaps]
    f90_R3 = f90_R3[valid_snaps]
    f90_LyC = f90_LyC[valid_snaps]
    gini_O3 = gini_O3[valid_snaps]
    gini_O2 = gini_O2[valid_snaps]
    gini_Hb = gini_Hb[valid_snaps]
    gini_O32 = gini_O32[valid_snaps]
    gini_R3 = gini_R3[valid_snaps]
    gini_LyC = gini_LyC[valid_snaps]
    rho_max_O3_LyC = rho_max_O3_LyC[valid_snaps]
    rho_max_O2_LyC = rho_max_O2_LyC[valid_snaps]
    rho_max_Hb_LyC = rho_max_Hb_LyC[valid_snaps]
    rho_max_O32_LyC = rho_max_O32_LyC[valid_snaps]
    rho_max_R3_LyC = rho_max_R3_LyC[valid_snaps]
    rho_max_O32_R3 = rho_max_O32_R3[valid_snaps]
    dtheta_star_O3_LyC = dtheta_star_O3_LyC[valid_snaps]
    dtheta_star_O2_LyC = dtheta_star_O2_LyC[valid_snaps]
    dtheta_star_Hb_LyC = dtheta_star_Hb_LyC[valid_snaps]
    dtheta_star_O32_LyC = dtheta_star_O32_LyC[valid_snaps]
    dtheta_star_R3_LyC = dtheta_star_R3_LyC[valid_snaps]
    dtheta_star_O32_R3 = dtheta_star_O32_R3[valid_snaps]
    R2_controls = R2_controls[valid_snaps]
    dR2_add_O32_to_controls = dR2_add_O32_to_controls[valid_snaps]
    O32_lowmode_frac_m2 = O32_lowmode_frac_m2[valid_snaps]
    with h5py.File(f'image_data/{sim}_{run}.hdf5','w') as f:
        f.create_dataset('redshifts', data=redshifts)
        f.create_dataset('r_int', data=r_int)
        f.create_dataset('r_esc', data=r_esc)
        f.create_dataset('L_O3', data=L_O3)
        f.create_dataset('L_O2', data=L_O2)
        f.create_dataset('L_Hb', data=L_Hb)
        f.create_dataset('angle_cam_max', data=angle_cam_max)
        f.create_dataset('theta_LyC_max', data=theta_LyC_max)
        f.create_dataset('theta_LyC', data=theta_LyC)
        f.create_dataset('theta_O32', data=theta_O32)
        f.create_dataset('theta_R3', data=theta_R3)
        f.create_dataset('theta_O3', data=theta_O3)
        f.create_dataset('theta_O2', data=theta_O2)
        f.create_dataset('theta_Hb', data=theta_Hb)
        f.create_dataset('radius_O3', data=radius_O3)
        f.create_dataset('radius_O2', data=radius_O2)
        f.create_dataset('radius_Hb', data=radius_Hb)
        f.create_dataset('FWHM_O3', data=FWHM_O3)
        f.create_dataset('FWHM_O2', data=FWHM_O2)
        f.create_dataset('FWHM_Hb', data=FWHM_Hb)
        f.create_dataset('FWHM_O32', data=FWHM_O32)
        f.create_dataset('FWHM_R3', data=FWHM_R3)
        f.create_dataset('FWHM_LyC', data=FWHM_LyC)
        f.create_dataset('f50_O3', data=f50_O3)
        f.create_dataset('f50_O2', data=f50_O2)
        f.create_dataset('f50_Hb', data=f50_Hb)
        f.create_dataset('f50_O32', data=f50_O32)
        f.create_dataset('f50_R3', data=f50_R3)
        f.create_dataset('f50_LyC', data=f50_LyC)
        f.create_dataset('f90_O3', data=f90_O3)
        f.create_dataset('f90_O2', data=f90_O2)
        f.create_dataset('f90_Hb', data=f90_Hb)
        f.create_dataset('f90_O32', data=f90_O32)
        f.create_dataset('f90_R3', data=f90_R3)
        f.create_dataset('f90_LyC', data=f90_LyC)
        f.create_dataset('gini_O3', data=gini_O3)
        f.create_dataset('gini_O2', data=gini_O2)
        f.create_dataset('gini_Hb', data=gini_Hb)
        f.create_dataset('gini_O32', data=gini_O32)
        f.create_dataset('gini_R3', data=gini_R3)
        f.create_dataset('gini_LyC', data=gini_LyC)
        f.create_dataset('rho_max_O3_LyC', data=rho_max_O3_LyC)
        f.create_dataset('rho_max_O2_LyC', data=rho_max_O2_LyC)
        f.create_dataset('rho_max_Hb_LyC', data=rho_max_Hb_LyC)
        f.create_dataset('rho_max_O32_LyC', data=rho_max_O32_LyC)
        f.create_dataset('rho_max_R3_LyC', data=rho_max_R3_LyC)
        f.create_dataset('rho_max_O32_R3', data=rho_max_O32_R3)
        f.create_dataset('dtheta_star_O3_LyC', data=dtheta_star_O3_LyC)
        f.create_dataset('dtheta_star_O2_LyC', data=dtheta_star_O2_LyC)
        f.create_dataset('dtheta_star_Hb_LyC', data=dtheta_star_Hb_LyC)
        f.create_dataset('dtheta_star_O32_LyC', data=dtheta_star_O32_LyC)
        f.create_dataset('dtheta_star_R3_LyC', data=dtheta_star_R3_LyC)
        f.create_dataset('dtheta_star_O32_R3', data=dtheta_star_O32_R3)
        f.create_dataset('R2_controls', data=R2_controls)
        f.create_dataset('dR2_add_O32_to_controls', data=dR2_add_O32_to_controls)
        f.create_dataset('O32_lowmode_frac_m2', data=O32_lowmode_frac_m2)

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

zip_data(sim=sim, run=run, snaps=[168], verbose=True, plot_theta=True)
# zip_data(sim=sim, run=run)
# plot_Dv(sim=sim, run=run)
