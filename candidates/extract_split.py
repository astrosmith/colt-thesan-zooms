import numpy as np
import h5py, os, errno, platform, subprocess
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

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
colt_dir = f'{zoom_dir}-COLT/{sim}/ics'
# colt_dir = f'/orcd/data/mvogelsb/005/Lab/Thesan-Zooms-COLT/{sim}/ics'
states = 'states-no-UVB'  # States prefix
copy_states = False  # Copy ionization states to the new colt file
os.makedirs(f'{colt_dir}_tree', exist_ok=True)  # Ensure the new colt directory exists
ics_movie_dir = f'{colt_dir}_movie'
os.makedirs(ics_movie_dir, exist_ok=True)  # Ensure the new colt directory exists

# Overwrite for local testing
#dist_dir = '.'
#colt_dir = '.'

# List of fields to be interpolated
gas_fields = ['D', 'D_Si', 'SFR', 'T_dust', 'X', 'Y', 'Z', 'Z_C', 'Z_Fe', 'Z_Mg', 'Z_N', 'Z_Ne', 'Z_O', 'Z_S', 'Z_Si',
              'e_int', 'is_HR', 'r','rho', 'v', 'x_H2', 'x_HI', 'x_HeI', 'x_HeII', 'x_e', 'id']
state_fields = ['G_ion', 'x_e', 'x_HI', 'x_HII', 'x_HeI', 'x_HeII',
                'x_CI', 'x_CII', 'x_CIII', 'x_CIV',
                'x_NI', 'x_NII', 'x_NIII', 'x_NIV', 'x_NV',
                'x_OI', 'x_OII', 'x_OIII', 'x_OIV',
                'x_NeI', 'x_NeII', 'x_NeIII', 'x_NeIV',
                'x_MgI', 'x_MgII', 'x_MgIII',
                'x_SiI', 'x_SiII', 'x_SiIII', 'x_SiIV',
                'x_SI', 'x_SII', 'x_SIII', 'x_SIV', 'x_SV', 'x_SVI',
                'x_FeI', 'x_FeII', 'x_FeIII', 'x_FeIV', 'x_FeV', 'x_FeVI']
star_fields = ['Z_star', 'age_star', 'm_star', 'm_init_star', 'v_star','r_star' ,'id_star']
units = {'r': b'cm', 'v': b'cm/s', 'e_int': b'cm^2/s^2', 'T_dust': b'K', 'rho': b'g/cm^3', 'SFR': b'Msun/yr',
         'r_star': b'cm', 'v_star': b'cm/s', 'm_star': b'Msun', 'm_init_star': b'Msun', 'age_star': b'Gyr'}

no_interp = ['D', 'D_Si','T_dust','x_e', 'e_int',
             'X', 'Y', 'Z', 'Z_C', 'Z_Fe', 'Z_Mg', 'Z_N', 'Z_Ne', 'Z_O', 'Z_S', 'Z_Si',
             'r', 'v', 'v_star', 'r_star','x_H2', 'Z_star',
             'x_HI', 'x_HII', 'x_HeI', 'x_HeII',
             'x_CI', 'x_CII', 'x_CIII', 'x_CIV',
             'x_NI', 'x_NII', 'x_NIII', 'x_NIV', 'x_NV',
             'x_OI', 'x_OII', 'x_OIII', 'x_OIV',
             'x_NeI', 'x_NeII', 'x_NeIII', 'x_NeIV',
             'x_MgI', 'x_MgII', 'x_MgIII',
             'x_SiI', 'x_SiII', 'x_SiIII', 'x_SiIV',
             'x_SI', 'x_SII', 'x_SIII', 'x_SIV', 'x_SV', 'x_SVI',
             'x_FeI', 'x_FeII', 'x_FeIII', 'x_FeIV', 'x_FeV', 'x_FeVI']
yes_interp = ['SFR', 'rho', 'G_ion','m_star', 'm_init_star']
special = ['age_star', 'is_HR']

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

def interpolate_field(r_1, r_2, n_split=3):
    r_1 = np.asarray(r_1)
    r_2 = np.asarray(r_2)
    t_vals = np.linspace(0, 1, n_split + 1)  # Only strictly between 0 and 1
    arr = np.stack([r_1, r_2], axis=0)
    interp = interp1d([0, 1], arr, axis=0)
    r_interp = interp(t_vals)
    return r_interp

def interpolate_colt_movie_multi(c1, c2, fields, n_split=4):
    global file_count
    id1 = c1['id'][:]
    id2 = c2['id'][:]
    idstar1 = c1['id_star'][:] if 'id_star' in c1 else None
    idstar2 = c2['id_star'][:] if 'id_star' in c2 else None
    z1 = c1.attrs['redshift']
    z2 = c2.attrs['redshift']

    # Keep ordering of c1, then append new ids from c2
    new_ids = id2[~np.isin(id2, id1)]
    id_collective = np.concatenate([id1, new_ids])
    if idstar1 is not None and idstar2 is not None:
        new_ids_star = idstar2[~np.isin(idstar2, idstar1)]
        id_collective_star = np.concatenate([idstar1, new_ids_star])
    else:
        id_collective_star = None

    # Dictionary to store interpolated arrays for each field
    interp_data_dict = {}
    # Interpolated redshift array
    interp_z = interpolate_field(z1, z2, n_split=n_split)
    interp_r_box = interpolate_field(c1.attrs['r_box'], c2.attrs['r_box'], n_split=n_split)

    for field in fields:
        if field in gas_fields:
            data1 = c1[field][:]
            data2 = c2[field][:]

            # Preallocate full arrays
            data1_full = np.zeros((len(id_collective),) + data1.shape[1:], dtype=data1.dtype)
            data2_full = np.zeros_like(data1_full)

            # Fill from c1
            data1_full[:len(id1)] = data1

            # Vectorized filling for c2
            sort_idx_id2 = np.argsort(id2)
            pos_in_id2 = np.searchsorted(id2, id_collective, sorter=sort_idx_id2)
            matches_mask = id2[sort_idx_id2[pos_in_id2]] == id_collective
            valid_positions = sort_idx_id2[pos_in_id2[matches_mask]]
            data2_full[matches_mask] = data2[valid_positions]

            if field in no_interp:
                # For ids missing in c2, fill data2_full with data1_full values (keep constant)
                missing_mask = ~matches_mask
                data2_full[missing_mask] = data1_full[missing_mask]
                # For ids missing in c1, fill data1_full with data2_full values (keep constant)
                missing_mask_1 = np.arange(len(id_collective)) >= len(id1)
                data1_full[missing_mask_1] = data2_full[missing_mask_1]

            if field == 'is_HR':
                interp = np.logical_and(data1_full, data2_full)
                interp_data = np.zeros((n_split+1, data1_full.shape[0]), dtype=bool)
                interp_data[0, :] = data1_full
                interp_data[-1, :] = data2_full
                for i in range(1, n_split):
                    interp_data[i, :] = interp
                interp_data_dict[field] = interp_data
            else:
                # Interpolate this field
                interp_data_dict[field] = interpolate_field(data1_full, data2_full, n_split=n_split)

        elif field in star_fields and id_collective_star is not None:
            data1 = c1[field][:]
            data2 = c2[field][:]

            # Preallocate full arrays
            data1_full = np.zeros((len(id_collective_star),) + data1.shape[1:], dtype=data1.dtype)
            data2_full = np.zeros_like(data1_full)

            # Fill from c1
            data1_full[:len(idstar1)] = data1

            # Vectorized filling for c2
            sort_idx_id2 = np.argsort(idstar2)
            pos_in_id2 = np.searchsorted(idstar2, id_collective_star, sorter=sort_idx_id2)
            matches_mask = idstar2[sort_idx_id2[pos_in_id2]] == id_collective_star
            valid_positions = sort_idx_id2[pos_in_id2[matches_mask]]
            data2_full[matches_mask] = data2[valid_positions]

            if field in no_interp:
                # For ids missing in c2, fill data2_full with data1_full values (keep constant)
                missing_mask = ~matches_mask
                data2_full[missing_mask] = data1_full[missing_mask]
                # For ids missing in c1, fill data1_full with data2_full values (keep constant)
                missing_mask_1 = np.arange(len(id_collective_star)) >= len(idstar1)
                data1_full[missing_mask_1] = data2_full[missing_mask_1]

            if field == 'age_star':
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
                t_interp = cosmo.age(interp_z).value
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
    if id_collective_star is not None:
        valid_stars = np.all(mask_positive, axis=0)
        id_collective_star = id_collective_star[valid_stars]
        for field in star_fields:
            # Keep only columns (stars) with all non-negative ages
            interp_data_dict[field] = interp_data_dict[field][:, valid_stars]

    files_added = []
    if len(file_count) == 0:
        for i in range(0, n_split + 1):
            if i == 0 :
                make_softlink(c1.filename, f'{ics_movie_dir}/colt_{i:03d}.hdf5')
                files_added.append(i)
                continue
            elif i == n_split:
                make_softlink(c2.filename, f'{ics_movie_dir}/colt_{i:03d}.hdf5')
                files_added.append(i)
                continue
            with h5py.File(f'{ics_movie_dir}/colt_{i:03d}.hdf5', 'w') as f:
                f.attrs['n_cells'] = np.int32(len(id_collective))  # Number of cells
                f.attrs['n_stars'] = np.int32(len(id_collective_star) if id_collective_star is not None else 0)  # Number of star particles
                f.attrs['redshift'] = interp_z[i]  # Current simulation redshift
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
                    dset = f.create_dataset(field, data=arr[i])
                    if field in units:
                        dset.attrs['units'] = units[field]
            files_added.append(i)
    else:
        last_file_no = int(file_count[-1])
        for i in range(0, n_split + 1):
            if i == 0:
                continue  # Skip first frame if continuing
            elif i == n_split:
                make_softlink(c2.filename, f'{ics_movie_dir}/colt_{last_file_no + i :03d}.hdf5')
                files_added.append(last_file_no + i)
                continue
            with h5py.File(f'{ics_movie_dir}/colt_{last_file_no + i :03d}.hdf5', 'w') as f:
                f.attrs['n_cells'] = np.int32(len(id_collective))  # Number of cells
                f.attrs['n_stars'] = np.int32(len(id_collective_star) if id_collective_star is not None else 0)  # Number of star particles
                f.attrs['redshift'] = interp_z[i]  # Current simulation redshift
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

if False:
    mask = np.zeros(len(snaps), dtype=bool)
    mask[-1] = True
    mask[-2] = True
    mask[-3] = True
    mask[-4] = True
    snaps = snaps[mask]

n_snaps = len(snaps)
file_count = np.array([])
failed_states = [] # List of snapshots where ionization states failed to copy
for i in progressbar(range(n_snaps-1)):
    snap = snaps[i]
    snap_1, snap_2 = snap, snap + 1
    colt_1 = f'{colt_dir}/colt_{snap_1:03d}.hdf5'
    colt_2 = f'{colt_dir}/colt_{snap_2:03d}.hdf5'
    with h5py.File(colt_1, 'r') as c1, h5py.File(colt_2, 'r') as c2:
        h = c1.attrs['h100']
        H0 = 100. * h
        Omega0 = c1.attrs['Omega0']
        z1, z2 = c1.attrs['redshift'], c2.attrs['redshift']
        cosmo = FlatLambdaCDM(H0=H0, Om0=Omega0, Tcmb0=2.725)
        t_fixed = 1e3 * cosmo.age(np.array([z1, z2])).value  # [Myr]
        dt_fixed = np.abs(t_fixed[:-1] - t_fixed[1:])[0]  # [Myr]
        dt_min = 3.  # [Myr]
        n_add = np.floor(dt_fixed / dt_min).astype(np.int32)
        n_stars_tot = (c1.attrs['n_stars'] if 'n_stars' in c1.attrs else 0) + (c2.attrs['n_stars'] if 'n_stars' in c2.attrs else 0)
        fields_to_interpolate = np.concatenate((gas_fields, star_fields)) if n_stars_tot > 0 else gas_fields
        if copy_states: fields_to_interpolate = np.concatenate((fields_to_interpolate, state_fields))
        file_count = interpolate_colt_movie_multi(c1, c2, fields=fields_to_interpolate, n_split=n_add+1)

    # Ionization states
    if copy_states:
        try:
            states_file = f'{colt_dir}/{states}_{snap:03d}.hdf5'
            new_states_file = f'{colt_dir}_tree/{states}_{snap:03d}.hdf5'
            with h5py.File(states_file, 'r') as f, h5py.File(new_states_file, 'w') as g:
                for field in state_fields:
                    g.create_dataset(field, data=f[field][:][gas_mask])
                if 'G_ion' in state_fields: g['G_ion'].attrs['units'] = b'erg/s'
        except Exception as e:
            failed_states.append(snap)

if len(failed_states) > 0:
    print(f'Failed to copy ionization states for snapshots: {failed_states}')
