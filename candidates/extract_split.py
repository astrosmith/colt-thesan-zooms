import numpy as np
import h5py, os, errno, platform, subprocess
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
from scipy.signal import savgol_filter

# Configurable global variables
if platform.system() == 'Darwin':
    sim, zoom_dir = 'g500531/z4', os.path.expandvars('$HOME/Engaging/Thesan-Zooms')
else:
    sim, zoom_dir = 'g5760/z4', '/orcd/data/mvogelsb/004/Thesan-Zooms'

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        sim = sys.argv[1]
    elif len(sys.argv) == 3:
        sim, zoom_dir = sys.argv[1], sys.argv[3]
    elif len(sys.argv) != 1:
        raise ValueError('Usage: python extract_split.py [sim] [zoom_dir]')

# Derived global variables
cand_dir = f'{zoom_dir}/{sim}/postprocessing/candidates'
#colt_dir = f'{zoom_dir}-COLT/{sim}/ics'
colt_dir = f'/orcd/data/mvogelsb/005/Lab/Thesan-Zooms-COLT/{sim}/ics'
states = 'states-no-UVB'  # States prefix
copy_states = True  # Copy ionization states to the new colt file
interpolate_mass = True  # Interpolate mass fields
jerk_interp = True  # Use linear interpolation
UnitLength_in_cm = 3.08568e+21 # 1 kpc in cm [from Candidates files]
MYR_TO_S = u.Myr.to(u.s)  # Myr -> s
f_vir = 4.    # constant factor for virial radius (needs to be cross-referendced with extract.py, default value =4.) 

use_smoothed = True # Use smoothed GroupPos
# Overwrite for local testing
#dist_dir = '.'
#colt_dir = '.'
# tree_dir = colt_dir  # Where the data is read from
tree_dir = f'{colt_dir}_tree'  # Where the data is read from
movie_dir = f'{colt_dir}_movie'  # Where the movie files are written to
os.makedirs(movie_dir, exist_ok=True)  # Ensure the new colt directory exists

# List of fields to be interpolated
gas_fields = ['D', 'D_Si', 'SFR', 'T_dust', 'X', 'Y', 'Z', 'Z_C', 'Z_Fe', 'Z_Mg', 'Z_N', 'Z_Ne', 'Z_O', 'Z_S', 'Z_Si',
              'e_int', 'is_HR', 'r', 'rho', 'v', 'x_H2', 'x_HI', 'x_HeI', 'x_HeII', 'x_e', 'id']
# state_fields = ['G_ion', 'x_e', 'x_HI', 'x_HII', 'x_HeI', 'x_HeII',
#                 'x_CI', 'x_CII', 'x_CIII', 'x_CIV',
#                 'x_NI', 'x_NII', 'x_NIII', 'x_NIV', 'x_NV',
#                 'x_OI', 'x_OII', 'x_OIII', 'x_OIV',
#                 'x_NeI', 'x_NeII', 'x_NeIII', 'x_NeIV',
#                 'x_MgI', 'x_MgII', 'x_MgIII',
#                 'x_SiI', 'x_SiII', 'x_SiIII', 'x_SiIV',
#                 'x_SI', 'x_SII', 'x_SIII', 'x_SIV', 'x_SV', 'x_SVI',
#                 'x_FeI', 'x_FeII', 'x_FeIII', 'x_FeIV', 'x_FeV', 'x_FeVI']
state_fields = ['x_HI', 'x_HII', 'x_HeI', 'x_CII', 'x_CIII', 'x_NI', 'x_NII', 'x_OI', 'x_OII', 'x_NeI', 'x_NeII', 'x_MgII', 'x_SiII', 'x_SII']
star_fields = ['Z_star', 'age_star', 'm_star', 'm_init_star', 'v_star','r_star' ,'id_star']
units = {'r': b'cm', 'v': b'cm/s', 'e_int': b'cm^2/s^2', 'T_dust': b'K', 'rho': b'g/cm^3', 'm': b'g', 'V': b'cm^3', 'SFR': b'Msun/yr',
         'r_star': b'cm', 'v_star': b'cm/s', 'm_star': b'Msun', 'm_init_star': b'Msun', 'age_star': b'Gyr'}
# Remove gas fields that are in states
for field in state_fields:
    if field in gas_fields:
        gas_fields.remove(field)
pos_fields = ['r', 'v', 'v_star', 'r_star']
no_interp = ['D', 'D_Si','T_dust','x_e', 'e_int',
             'X', 'Y', 'Z', 'Z_C', 'Z_Fe', 'Z_Mg', 'Z_N', 'Z_Ne', 'Z_O', 'Z_S', 'Z_Si',
             'x_H2', 'Z_star','x_HI', 'x_HII', 'x_HeI', 'x_HeII',
             'x_CI', 'x_CII', 'x_CIII', 'x_CIV',
             'x_NI', 'x_NII', 'x_NIII', 'x_NIV', 'x_NV',
             'x_OI', 'x_OII', 'x_OIII', 'x_OIV',
             'x_NeI', 'x_NeII', 'x_NeIII', 'x_NeIV',
             'x_MgI', 'x_MgII', 'x_MgIII',
             'x_SiI', 'x_SiII', 'x_SiIII', 'x_SiIV',
             'x_SI', 'x_SII', 'x_SIII', 'x_SIV', 'x_SV', 'x_SVI',
             'x_FeI', 'x_FeII', 'x_FeIII', 'x_FeIV', 'x_FeV', 'x_FeVI']
# yes_interp = ['SFR', 'rho', 'G_ion','m_star', 'm_init_star']
# special = ['age_star', 'is_HR']

print(f'Splitting and Interpolating {colt_dir} ...')

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

def make_softlink(src, dst):
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)  # Remove existing file or link
    subprocess.run(['ln', '-s', src, dst], check=True)

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

def smooth(x, y):
    """Smooth the data."""
    x_smooth, y_smooth = savgol_filter((x, y), 15, 3)
    interp = interp1d(x_smooth, y_smooth, fill_value='extrapolate')
    return interp(x)  # Interpolate the smoot+hed data back to the original x values

def smooth_split(x, y, x_split):
    """Smooth the data."""
    x_smooth, y_smooth = savgol_filter((x, y), 15, 3)
    interp = interp1d(x_smooth, y_smooth, fill_value='extrapolate', kind='cubic')
    return interp(x_split)  # Interpolate the smoothed data back to the original x values

def smooth_pos(x, pos):
    """Smooth position data."""
    return np.array([smooth(x, pos[:,0]), smooth(x, pos[:,1]), smooth(x, pos[:,2])]).T

def interpolate_field(r_1, r_2, n_split=3):
    r_1 = np.asarray(r_1)
    r_2 = np.asarray(r_2)
    t_vals = np.linspace(0, 1, n_split + 1)  # Only strictly between 0 and 1
    arr = np.stack([r_1, r_2], axis=0)
    interp = interp1d([0, 1], arr, axis=0)
    r_interp = interp(t_vals)
    return r_interp

def jerk_interpolate(p0, v0, p1, v1, n_split, dt_s):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)

    dt = dt_s
    frames = n_split + 1

    N, D = p0.shape
    pos = np.zeros((frames, N, D), dtype=float)
    vel = np.zeros((frames, N, D), dtype=float)

    # Masks
    both_mask  = (np.linalg.norm(p0, axis=1) > 0) & (np.linalg.norm(p1, axis=1) > 0)
    only0_mask = (np.linalg.norm(p0, axis=1) > 0) & ~both_mask
    only1_mask = (np.linalg.norm(p1, axis=1) > 0) & ~both_mask

    tvals = np.linspace(0.0, dt, frames)[:, None, None]  # shape (frames,1,1)

    # ---- Case 1: IDs present in both snapshots (true jerk interpolation) ----
    if np.any(both_mask):
        dx = p1[both_mask] - p0[both_mask]
        a1 = (6 * dx - 2 * (2 * v0[both_mask] + v1[both_mask]) * dt) / dt**2
        j  = (6 * ((v0[both_mask] + v1[both_mask]) * dt - 2 * dx)) / dt**3

        vel[:, both_mask] = v0[both_mask] + a1*tvals + 0.5*j*tvals**2
        pos[:, both_mask] = (
            p0[both_mask]
            + v0[both_mask]*tvals
            + 0.5*a1*tvals**2
            + (1.0/6.0)*j*tvals**3
        )

    # ---- Case 2: IDs only in snapshot 1 ----
    if np.any(only0_mask):
        a = -v0[only0_mask] / dt  # constant acceleration
        tvals = np.linspace(0, dt, frames)[:, None, None]  # (frames,1,1)
        vel[:, only0_mask] = v0[only0_mask] + a * tvals
        pos[:, only0_mask] = p0[only0_mask] + v0[only0_mask]*tvals + 0.5*a*tvals**2

    # ---- Case 3: IDs only in snapshot 2 ----
    if np.any(only1_mask):
        a = v1[only1_mask] / dt
        tvals = np.linspace(0, dt, frames)[:, None, None]
        vel[:, only1_mask] = a * tvals
        p0_fake = p1[only1_mask] - 0.5 * v1[only1_mask] * dt
        pos[:, only1_mask] = p0_fake + 0.5 * a * tvals**2 + vel[:, only1_mask] * tvals

    return pos, vel

def interpolate_colt_movie_multi(c1, c2, z_split, box_pos_dense, R_virs_dense, gas_fields, star_fields=None, n_split=4, f_cut=1., s1=None, s2=None):
    global file_count
    id1 = c1['id'][:]
    id2 = c2['id'][:]
    z1 = c1.attrs['redshift']
    z2 = c2.attrs['redshift']

    # Keep ordering of c1, then append new ids from c2
    new_ids = id2[~np.isin(id2, id1)]
    id_collective = np.concatenate([id1, new_ids])
    sort_idx_id2 = np.argsort(id2)
    pos_in_id2 = np.searchsorted(id2, id_collective, sorter=sort_idx_id2)
    matches_mask = id2[sort_idx_id2[pos_in_id2]] == id_collective
    valid_positions = sort_idx_id2[pos_in_id2[matches_mask]]

    # Dictionary to store interpolated arrays for each field
    interp_data_dict = {}
    a_arr = 1./(z_split+1.)
    conv_const = UnitLength_in_cm / h
    interp_r_box = R_virs_dense * a_arr * conv_const

    for field in gas_fields:
        if field in state_fields and s1 is not None and s2 is not None:
            data1 = s1[field][:]
            data2 = s2[field][:]
        else:
            data1 = c1[field][:]
            data2 = c2[field][:]

        # Preallocate full arrays
        data1_full = np.zeros((len(id_collective),) + data1.shape[1:], dtype=data1.dtype)
        data2_full = np.zeros_like(data1_full)
        # Fill from c1 and c2
        data1_full[:len(id1)] = data1
        data2_full[matches_mask] = data2[valid_positions]
        if jerk_interp and field in pos_fields:
            # Use jerk interpolation for positions and velocities
            if field in ['r', 'v']:
                p1, v1 = c1['r'][:], c1['v'][:]
                p2, v2 = c2['r'][:], c2['v'][:]
                p1_full = np.zeros((len(id_collective),) + data1.shape[1:], dtype=data1.dtype)
                p2_full = np.zeros_like(p1_full)
                v1_full = np.zeros((len(id_collective),) + data1.shape[1:], dtype=data1.dtype)
                v2_full = np.zeros_like(v1_full)
                # Fill from c1 and c2
                p1_full[:len(id1)] = p1
                p2_full[matches_mask] = p2[valid_positions]
                v1_full[:len(id1)] = v1
                v2_full[matches_mask] = v2[valid_positions]
                if field == 'r':
                    ## p1,p0,v1,v0 in cm and cm/s
                    dt_myr = np.abs(cosmo.age(z1).value - cosmo.age(z2).value) * 1e3  # [Myr]
                    dt_s = dt_myr * MYR_TO_S
                    # Before interpolating, we shift everything to frame of reference of snap1
                    p1_full /= conv_fact1  # ckpc/h
                    p2_full /= conv_fact2  # ckpc/h
                    p1_full += GroupPos1   # ckpc/h but in BoxUnits
                    p2_full += GroupPos2   # ckpc/h but in BoxUnits
                    # Assuming vc = v, since we ignore Hubble flow
                    v1_full /= conv_fact1  # ckpc/h/s
                    v2_full /= conv_fact2  # ckpc/h/s

                    pos_interp, vel_interp = jerk_interpolate(p1_full, v1_full, p2_full, v2_full, n_split=n_split, dt_s=dt_s)

                    # # Shift back to original frame of reference
                    GroupPos_dense = box_pos_dense[:,None,:]
                    a_arr = 1./(z_split+1.)[:,None, None]
                    conv_const = UnitLength_in_cm / h
                    pos_interp -= GroupPos_dense
                    pos_interp *= a_arr * conv_const
                    vel_interp *= a_arr * conv_const

                    interp_data_dict['r'] = pos_interp
                    interp_data_dict['v'] = vel_interp
                    continue
                else:
                    continue

        if field in no_interp:
            # For ids missing in c2, fill data2_full with data1_full values (keep constant)
            missing_mask = ~matches_mask
            data2_full[missing_mask] = data1_full[missing_mask]
            # For ids missing in c1, fill data1_full with data2_full values (keep constant)
            missing_mask_1 = np.arange(len(id_collective)) >= len(id1)
            data1_full[missing_mask_1] = data2_full[missing_mask_1]
            interp_data_dict[field] = interpolate_field(data1_full, data2_full, n_split=n_split)
        elif field == 'is_HR':
            interp = np.logical_and(data1_full, data2_full)
            interp_data = np.zeros((n_split+1, data1_full.shape[0]), dtype=bool)
            interp_data[0, :] = data1_full
            interp_data[-1, :] = data2_full
            for i in range(1, n_split):
                interp_data[i, :] = interp
            interp_data_dict[field] = interp_data
            mask_HR = interp_data
        elif interpolate_mass and field == 'rho':
            V1 = c1['V'][:]
            V2 = c2['V'][:]
            V1_full = np.zeros(len(id_collective), dtype=V1.dtype)
            V2_full = np.zeros_like(V1_full)
            V1_full[:len(id1)] = V1
            V2_full[matches_mask] = V2[valid_positions]
            data1_full *= V1_full
            data2_full *= V2_full
            interp_data_dict[field] = interpolate_field(data1_full, data2_full, n_split=n_split)
        else:
            # Interpolate this field
            interp_data_dict[field] = interpolate_field(data1_full, data2_full, n_split=n_split)

    ## Masking with is_HR for high res cells
    HR_gas = np.all(mask_HR, axis=0)
    id_collective = id_collective[HR_gas]
    for field in gas_fields:
        interp_data_dict[field] = interp_data_dict[field][:, HR_gas]

    if f_cut < 1.:
        for i in range(n_split+1):
            mask = np.sum(interp_data_dict['r'][i,:]**2, axis=1) < (f_cut * interp_r_box[i])**2
            id_collective = id_collective[mask]
            for field in gas_fields:
                interp_data_dict[field] = interp_data_dict[field][:, mask]

    if star_fields is not None:
        idstar1 = c1['id_star'][:] if 'id_star' in c1 else None
        idstar2 = c2['id_star'][:] if 'id_star' in c2 else None
        if idstar1 is not None and idstar2 is not None:
            new_ids_star = idstar2[~np.isin(idstar2, idstar1)]
            id_collective_star = np.concatenate([idstar1, new_ids_star])
            sort_idx_id2 = np.argsort(idstar2)
            pos_in_id2 = np.searchsorted(idstar2, id_collective_star, sorter=sort_idx_id2)
            matches_mask = idstar2[sort_idx_id2[pos_in_id2]] == id_collective_star
            valid_positions = sort_idx_id2[pos_in_id2[matches_mask]]

            for field in star_fields:
                data1 = c1[field][:]
                data2 = c2[field][:]

                # Preallocate full arrays
                data1_full = np.zeros((len(id_collective_star),) + data1.shape[1:], dtype=data1.dtype)
                data2_full = np.zeros_like(data1_full)

                # Fill from c1 and c2
                data1_full[:len(idstar1)] = data1
                data2_full[matches_mask] = data2[valid_positions]

                if jerk_interp and field in pos_fields:
                    if field in ['r_star', 'v_star']:
                        p1, v1 = c1['r_star'][:], c1['v_star'][:]
                        p2, v2 = c2['r_star'][:], c2['v_star'][:]

                        p1_full = np.zeros((len(id_collective_star),) + data1.shape[1:], dtype=data1.dtype)
                        p2_full = np.zeros_like(p1_full)
                        v1_full = np.zeros((len(id_collective_star),) + data1.shape[1:], dtype=data1.dtype)
                        v2_full = np.zeros_like(v1_full)
                        # Fill from c1 and c2
                        p1_full[:len(idstar1)] = p1
                        p2_full[matches_mask] = p2[valid_positions]
                        v1_full[:len(idstar1)] = v1
                        v2_full[matches_mask] = v2[valid_positions]
                        if field == 'r_star':
                            ## p1,p0,v1,v0 in cm and cm/s [originally]
                            dt_myr = np.abs(cosmo.age(z1).value - cosmo.age(z2).value) * 1e3  # [Myr]
                            dt_s = dt_myr * MYR_TO_S
                            # Before interpolating, we shift everything to frame of reference of snap1
                            p1_full /= conv_fact1  # ckpc/h
                            p2_full /= conv_fact2  # ckpc/h
                            p1_full += GroupPos1   # ckpc/h but in BoxUnits
                            p2_full += GroupPos2   # ckpc/h but in BoxUnits
                            # Assuming vc = v, since we ignore Hubble flow
                            v1_full /= conv_fact1  # ckpc/h/s
                            v2_full /= conv_fact2  # ckpc/h/s

                            pos_interp, vel_interp = jerk_interpolate(p1_full, v1_full, p2_full, v2_full, n_split=n_split, dt_s=dt_s)

                            # # Shift back to original frame of reference
                            GroupPos_dense = box_pos_dense[:,None,:]
                            a_arr = 1./(z_split+1.)[:,None, None]

                            pos_interp -= GroupPos_dense
                            conv_const = UnitLength_in_cm / h
                            pos_interp *= a_arr * conv_const
                            vel_interp *= a_arr * conv_const
                            interp_data_dict['r_star'] = pos_interp
                            interp_data_dict['v_star'] = vel_interp
                            continue
                        else:
                            continue

                if field in no_interp:
                    # For ids missing in c2, fill data2_full with data1_full values (keep constant)
                    missing_mask = ~matches_mask
                    data2_full[missing_mask] = data1_full[missing_mask]
                    # For ids missing in c1, fill data1_full with data2_full values (keep constant)
                    missing_mask_1 = np.arange(len(id_collective_star)) >= len(idstar1)
                    data1_full[missing_mask_1] = data2_full[missing_mask_1]
                    interp_data_dict[field] = interpolate_field(data1_full, data2_full, n_split=n_split)
                elif field == 'age_star':
                    # Preallocate interpolated array (frames along axis 0)
                    interp_data = np.zeros((n_split + 1, data1_full.shape[0]), dtype=data1.dtype)

                    # Endpoints
                    interp_data[0, :] = data1_full
                    interp_data[-1, :] = data2_full

                    # Mask for IDs present in both snapshots
                    valid_mask = matches_mask

                    # Interpolate stars present in both snapshots normally
                    if valid_mask.any() and n_split > 1:
                        interp_data[1:-1, valid_mask] = interpolate_field(
                            data1_full[valid_mask],
                            data2_full[valid_mask],
                            n_split=n_split
                        )[1:-1, :]

                    # Compute cosmic times at interpolation redshifts (Gyr)
                    t_interp = cosmo.age(z_split).value
                    dt = t_interp[1:] - t_interp[0]   # time since snap1
                    dt2 = t_interp[-1] - t_interp[:-1]  # time until snap2

                    # Stars only in c1 → age increases with Δt
                    only_c1_mask = (~valid_mask) & (data1_full > 0) & (data2_full == 0)
                    if only_c1_mask.any():
                        for j in range(1, n_split):
                            interp_data[j, only_c1_mask] = data1_full[only_c1_mask] + dt[j]

                    # Stars only in c2 → age decreases backwards with Δt
                    only_c2_mask = (~valid_mask) & (data2_full > 0) & (data1_full == 0)
                    if only_c2_mask.any():
                        for j in range(1, n_split):
                            interp_data[j, only_c2_mask] = data2_full[only_c2_mask] - dt2[j-1]

                    # Remove entries with negative ages by setting them to zero
                    mask_positive = interp_data > 0

                    # Store
                    interp_data_dict[field] = interp_data
                else:
                    # Normal interpolation for other star fields
                    interp_data_dict[field] = interpolate_field(data1_full, data2_full, n_split=n_split)

            # Find indices where all frames for stellar ages are non-negative
            valid_stars = np.all(mask_positive, axis=0)
            id_collective_star = id_collective_star[valid_stars]
            for field in star_fields:
                # Keep only columns (stars) with all non-negative ages
                interp_data_dict[field] = interp_data_dict[field][:, valid_stars]
            if f_cut < 1.:
                for i in range(n_split+1):
                    mask = np.sum(interp_data_dict['r_star'][i,:]**2, axis=1) < (f_cut * interp_r_box[i])**2
                    id_collective_star = id_collective_star[mask]
                    for field in star_fields:
                        interp_data_dict[field] = interp_data_dict[field][:, mask]
        else:
            id_collective_star = None
    else:
        id_collective_star = None

    files_added = []
    if len(file_count) == 0:
        for i in range(0, n_split + 1):
            if not interpolate_mass:
                if i == 0 :
                    make_softlink(c1.filename, f'{movie_dir}/colt_{i:04d}.hdf5')
                    files_added.append(i)
                    continue
                if i == n_split:
                    make_softlink(c2.filename, f'{movie_dir}/colt_{i:04d}.hdf5')
                    files_added.append(i)
                    continue
            with h5py.File(f'{movie_dir}/colt_{i:04d}.hdf5', 'w') as f:
                f.attrs['n_cells'] = np.int32(len(id_collective))  # Number of cells
                f.attrs['n_stars'] = np.int32(len(id_collective_star) if id_collective_star is not None else 0)  # Number of star particles
                f.attrs['redshift'] = z_split[i]  # Current simulation redshift
                f.attrs['Omega0'] = c1.attrs['Omega0']  # Matter density [rho_crit_0]
                f.attrs['OmegaB'] = c1.attrs['OmegaB']  # Baryon density [rho_crit_0]
                f.attrs['h100'] = c1.attrs['h100']  # Hubble constant [100 km/s/Mpc]
                f.attrs['r_box'] = interp_r_box[i] # Bounding box radius [cm]
                for field, arr in interp_data_dict.items():
                    if field == 'id':
                        dset = f.create_dataset(field, data=id_collective)
                        continue
                    if field == 'id_star':
                        dset = f.create_dataset(field, data=id_collective_star)
                        continue
                    if field == 'rho' and interpolate_mass:
                        dset = f.create_dataset('m', data=arr[i])
                        dset.attrs['units'] = units['m']
                        continue
                    dset = f.create_dataset(field, data=arr[i])
                    if field in units:
                        dset.attrs['units'] = units[field]
            files_added.append(i)
    else:
        last_file_no = int(file_count[-1])
        for i in range(0, n_split + 1):
            if i == 0:
                continue  # Skip first frame if continuing
            if not interpolate_mass and i == n_split:
                make_softlink(c2.filename, f'{movie_dir}/colt_{last_file_no + i :04d}.hdf5')
                files_added.append(last_file_no + i)
                continue
            with h5py.File(f'{movie_dir}/colt_{last_file_no + i :04d}.hdf5', 'w') as f:
                f.attrs['n_cells'] = np.int32(len(id_collective))  # Number of cells
                f.attrs['n_stars'] = np.int32(len(id_collective_star) if id_collective_star is not None else 0)  # Number of star particles
                f.attrs['redshift'] = z_split[i]  # Current simulation redshift
                f.attrs['Omega0'] = c1.attrs['Omega0']  # Matter density [rho_crit_0]
                f.attrs['OmegaB'] = c1.attrs['OmegaB']  # Baryon density [rho_crit_0]
                f.attrs['h100'] = c1.attrs['h100']  # Hubble constant [100 km/s/Mpc]
                f.attrs['r_box'] = interp_r_box[i] # Bounding box radius [cm]
                for field, arr in interp_data_dict.items():
                    if field == 'id':
                        dset = f.create_dataset(field, data=id_collective)
                        continue
                    if field == 'id_star':
                        dset = f.create_dataset(field, data=id_collective_star)
                        continue
                    if field == 'rho' and interpolate_mass:
                        dset = f.create_dataset('m', data=arr[i])
                        dset.attrs['units'] = units['m']
                        continue
                    dset = f.create_dataset(field, data=arr[i])
                    if field in units:
                        dset.attrs['units'] = units[field]
            files_added.append(last_file_no + i)

    file_count = np.concatenate((file_count, files_added))
    return file_count

tree_file = f'/orcd/data/mvogelsb/004/Thesan-Zooms/analyze/trees/{sim}/tree.hdf5'
# tree_file = f'{zoom_dir}/{sim}/output/tree.hdf5'
with h5py.File(tree_file, 'r') as f:
    snaps = f['Snapshots'][:] # Snapshots in the tree
    zs = f['Redshifts'][:] # Redshifts in the tree
    if not use_smoothed:
        # group_ids = f['Group']['GroupID'][:] # Group ID in the tree
        # subhalo_ids = f['Subhalo']['SubhaloID'][:] # Subhalo ID in the tree
        R_virs = f['Group']['Group_R_Crit200'][:] # Group virial radii in the tree [ckpc/h]
        GroupPos = f['Group']['GroupPos'][:] # Group positions in the tree [ckpc/h]
    else:
        with h5py.File(tree_dir + '/center.hdf5', 'r') as sf:
            g = sf['Smoothed']
            redshift = sf['Redshifts'][:]
            TargetPos = g['TargetPos'][:]  # Use smoothed versions
            GroupPos = smooth_pos(redshift, TargetPos) # Smooth the data
            # print(np.max(np.abs(GroupPos - GroupPos), axis=0))
            R_virs = g['R_Crit200'][:]  #[ckpc/h]
            SmoothR_virs = smooth(redshift, R_virs) # Smooth the data

if False:
    mask = np.zeros(len(snaps), dtype=bool)
    mask[-1] = True
    mask[-2] = True
    mask[-3] = True
    mask[-4] = True
    snaps = snaps[mask]

n_snaps = len(snaps)
file_count = np.array([])
for i in progressbar(range(n_snaps-1)):
    snap = snaps[i]
    snap_1, snap_2 = snap, snap + 1
    GroupPos1, GroupPos2 = GroupPos[i], GroupPos[i+1]
    colt_1 = f'{tree_dir}/colt_{snap_1:03d}.hdf5'
    colt_2 = f'{tree_dir}/colt_{snap_2:03d}.hdf5'
    with h5py.File(colt_1, 'r') as c1, h5py.File(colt_2, 'r') as c2:
        h = c1.attrs['h100']
        H0 = 100. * h
        Omega0 = c1.attrs['Omega0']
        z1, z2 = c1.attrs['redshift'], c2.attrs['redshift']
        a1, a2 = 1./ (1.+z1), 1./(1.+z2)
        h1, h2 = c1.attrs['h100'], c2.attrs['h100']
        conv_fact1, conv_fact2 = a1 * UnitLength_in_cm / h1, a2 * UnitLength_in_cm / h2
        cosmo = FlatLambdaCDM(H0=H0, Om0=Omega0, Tcmb0=2.725)
        t_fixed = 1e3 * cosmo.age(np.array([z1, z2])).value  # [Myr]
        dt_fixed = np.abs(t_fixed[:-1] - t_fixed[1:])[0]  # [Myr]
        dt_min = 1.5  # [Myr]
        n_add = np.floor(dt_fixed / dt_min).astype(np.int32)
        t_split = interpolate_field(t_fixed[0], t_fixed[1], n_add+1)
        z_split = z_at_value(cosmo.age, t_split * u.Myr)
        pos_dense = np.zeros((n_add+2, 3))
        for d in range(3):
            pos_dense[:,d] = smooth_split(redshift, GroupPos[:,d], z_split)
        r_box_dense = smooth_split(redshift, R_virs, z_split) * f_vir
        n_stars_tot = (c1.attrs['n_stars'] if 'n_stars' in c1.attrs else 0) + (c2.attrs['n_stars'] if 'n_stars' in c2.attrs else 0)
        gas_fields_to_interpolate = np.concatenate((gas_fields, state_fields)) if copy_states else gas_fields
        star_fields_to_interpolate = star_fields if n_stars_tot > 0 else None
        try:
            if not copy_states: 1/0  # Skip states interpolation
            states_1 = f'{tree_dir}/{states}_{snap_1:03d}.hdf5'
            states_2 = f'{tree_dir}/{states}_{snap_2:03d}.hdf5'
            with h5py.File(states_1, 'r') as s1, h5py.File(states_2, 'r') as s2:
                file_count = interpolate_colt_movie_multi(c1, c2, z_split=z_split,box_pos_dense=pos_dense, R_virs_dense=r_box_dense, gas_fields=gas_fields_to_interpolate, star_fields=star_fields_to_interpolate, n_split=n_add+1, s1=s1, s2=s2)
        except (FileNotFoundError, KeyError, ZeroDivisionError):
            file_count = interpolate_colt_movie_multi(c1, c2, z_split=z_split,box_pos_dense=pos_dense, R_virs_dense=r_box_dense ,gas_fields=gas_fields, star_fields=star_fields_to_interpolate, n_split=n_add+1)
