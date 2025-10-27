import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os, sys, gc

# -------------------- Physical constants -------------------- #
Msun = 1.988435e33
mH = 1.6735327e-24
pc = 3.085677581467192e18
kpc = 1e3 * pc
day = 86400.
yr = 365.24 * day
Myr = 1e6 * yr
Gyr = 1e9 * yr

# -------------------- Data Reading -------------------- #
def nearest_R(R):        
    if R <= 10.0:
        return R
    else:
        return np.floor(R)

def read_data(filename, snap=0, camera=0):
    ds = {'snap': snap}
    with h5py.File(filename, 'r') as f:
        # print([key for key in f.keys()])
        for key in f.attrs.keys():
            ds[key] = f.attrs[key]
        # print(ds['time']/Gyr,'Gyr')
        # 'D_mass', 'T_mass', 'Z_mass', 'nC_sum', 'nFe_sum', 'nH_sum', 'nHe_sum', 'nMg_sum', 'nN_sum', 'nNe_sum', 'nO_sum', 'nS_sum', 'nSi_sum', 'rho_avg',
        for key in ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S']:
            ds[key] = f[f'proj_n{key}_sum'][camera,:,:]
        for key in ['CIII_nC', 'CII_nC','HI_nH','HeI_nHe', 'MgII_nMg', 'NII_nN', 'NI_nN', 'NeII_nNe', 'NeI_nNe','OII_nO', 'OI_nO','SII_nS', 'SiII_nSi']:
            ds[key.split('_')[0]] = f[f'proj_x{key}'][camera,:,:]
        Rx = ds['image_radius_x'] / kpc
        Ry = ds['image_radius_y'] / kpc
        redshift = f.attrs['z']
        # R = ds['image_radius'] / kpc
        ds['extent'] = [-Rx, Rx, -Ry, Ry]
        # ds['extent'] = [-R,R,-R,R]
    return ds, redshift

# -------------------- Normalization -------------------- #
def stats_all_snaps(args, percentiles= np.array([1,10,50,90,99.99])):
    field, max_snap, n_skip, data_dir, n_cameras = args
    Z_tot = np.array([])
    n_snaps = np.arange(0, max_snap+1, n_skip)
    # n_snaps = [0,180,181]
    while field[-1] in ['I', 'V']:
        field = field[:-1]
    for snap in n_snaps:
         for camera in range(n_cameras):
             filename = f'{data_dir}/output_movie/proj_all_{snap:04d}.hdf5'
             ds,z_arr = read_data(filename=filename, snap=snap, camera=camera)
             Z = ds[field][:].flatten()
             Z_tot = np.append(Z_tot,Z)
    Z_p = np.percentile(Z_tot, percentiles)
    if min(Z_p) == 0:
        print(f'Warning: {field} has non-positive percentiles: {Z_p}')
        Z = Z[Z>0]
        Z_p = np.percentile(Z, percentiles)
        print(f'Updated percentiles for {field} = {Z_p}')
    if np.isnan(Z_p).any():
        Z_p = np.linspace(0,1, len(percentiles))
    return field, Z_p

# -------------------- Frame Rendering -------------------- #
def plot_image(args):
    snap, camera, show_rgba, percentiles_dict, metal_list, data_dir = args
    print(f'Rendering snap {snap} camera {camera}')
    alpha = 0.6
    color = 'white'
    font_size = 6
    fig = plt.figure(figsize=(3., 1.32))
    fig.patch.set_facecolor(color)
    fig.patch.set_alpha(alpha)
    filename = f'{data_dir}/output_movie/proj_all_{snap:04d}.hdf5'
    ds, z_arr = read_data(filename=filename,snap=snap, camera=camera,)
    img_dir = f'{save_dir}/{sim}/Mosaic'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    layout = [
        (0, 0, 1, 1, metal_list[1]),   (0, 1, 1, 1, metal_list[2]),   (0, 2, 1, 1,metal_list[3] ), (0, 3, 1, 1, metal_list[4]),
        (1, 0, 1, 1, metal_list[5]),   (1, 1, 2, 2, metal_list[0]),   (1, 3, 1, 1, metal_list[6]),
        (2, 0, 1, 1, metal_list[7]),                            (2, 3, 1, 1, metal_list[8]),
        (3, 0, 1, 1, metal_list[9]), (3, 1, 1, 1, metal_list[10]), (3, 2, 1, 1, metal_list[11]), (3, 3, 1, 1, metal_list[12]),
    ]

    n_rows, n_cols = 4, 4
    pad = 0.002
    w = (1.0 - pad * (n_cols - 1)) / n_cols
    h = (1.0 - pad * (n_rows - 1)) / n_rows

    for row, col, rowspan, colspan, name in layout:
        left = col * (w + pad)
        bottom = 1.0 - (row + rowspan) * (h + pad)
        width = w * colspan + pad * (colspan - 1)
        height = h * rowspan + pad * (rowspan - 1)
        ax = fig.add_axes([left, bottom, width, height])
        
        atom = name
        while atom[-1] in ['I', 'V']:
            atom = atom[:-1]
        state = name[len(atom):]
        #f_sat = 2e-1 if atom == 'H' or atom == 'He' else 1e-3
        #f_cut = 0.05 if atom =='H' or atom =='He' else 5e-10 / f_sat
        f_sat = percentiles_dict[atom][-1] #if atom == 'H' or atom == 'He' else 1e-3
        f_cut = percentiles_dict[atom][0]/percentiles_dict[atom][-1] #if atom =='H' or atom =='He' else 5e-10 / f_sat
        #f_sat = 1e-3
        #f_cut = 5e-10 / f_sat
        if show_rgba:
           # perc=stats_all_snaps(atom)
            # perc = stats(atom)
            Z0 = ds[atom].T / (percentiles_dict[atom][-1])
            #Z0 /= np.max(Z0)
            Z0[Z0 > 1.] = 1.
            Z0[Z0 < f_cut] = f_cut
            # rgba = np.zeros([ds['n_pixels'], ds['n_pixels'], 4])
            rgba = np.zeros([ds['ny_pixels'], ds['nx_pixels'], 4])
            rgba[:, :, 3] = (1. - np.log10(Z0) / np.log10(f_cut))
            Z = ds[name].T
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
            back = np.zeros([ds['ny_pixels'], ds['nx_pixels'], 4])
            back[:, :, 3] = 1.
            # R = ds['image_radius'] / kpc
            Rx = ds['image_radius_x'] / kpc
            Ry = ds['image_radius_y'] / kpc
            # ax.set_ylim([-9.*R/16., 9.*R/16.])
            ax.set_ylim([-9.*Rx/16., 9.*Ry/16.])
            ax.imshow(back, origin='lower', extent=ds['extent'], aspect='equal', interpolation='bilinear')
            ax.imshow(rgba, origin='lower', extent=ds['extent'], aspect='equal', interpolation='bicubic')
            if atom == 'H':
                ax.text(0.02, 0.02, f"${{z = {z_arr:.1f}}}$",color=color,fontsize=font_size, ha="left",va="bottom",transform=ax.transAxes, alpha=alpha)
                # Calculate the position for the bottom right corner
                Rx = nearest_R(ds['extent'][1])
                # Place the text at (0.97, 0.05) in axes coordinates (bottom right)
                ax.text(0.96, 0.02, f"${{{Rx:.0f} \\, \\mathrm{{kpc}}}}$", ha='right', va='bottom',color=color, fontsize=font_size, transform=ax.transAxes, alpha=alpha)
                        # path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
                ax.text(0.975, 0.97, 'THESAN', color=color, ha='right', va='top', transform=ax.transAxes, size=font_size, alpha=alpha) # weight='bold'
                ax.text(0.02, 0.97, f'm9.3', color=color, ha='left', va='top', transform=ax.transAxes, size=font_size, alpha=alpha) # weight='bold'

                # --- Scale bar setup ---
                scale_bar_kpc = 0.75 * Rx  # let bar width grow with Rx
                width_kpc = ds['extent'][1] - ds['extent'][0]

                # Fractional width of the scale bar relative to image width
                frac = scale_bar_kpc / width_kpc  

                # Centered position in axes coordinates
                x_mid = 0.5
                x0 = x_mid - frac / 2.0
                x1 = x_mid + frac / 2.0
                y_bar = 0.055  # vertical position

                # --- Draw the scale bar ---
                # ax.plot([x0-0.0025, x1+0.0025], [y_bar, y_bar],
                #         lw=4., c='k', transform=ax.transAxes, solid_capstyle='butt')
                ax.plot([x0, x1], [y_bar, y_bar],lw=1.5, c=color, transform=ax.transAxes, solid_capstyle='butt', alpha=alpha)
        else:
            ds[name] *= ds[atom]
            perc=stats(name)
            Z = ds[name] / (f_sat * perc[-1])
            Z[Z > 1.] = 1.
            Z[Z < f_cut] = f_cut
            Z = 1. - np.log10(Z) / np.log10(f_cut)
            # print(Z.shape)
            
            R = ds['image_radius'] / kpc
            ax.set_ylim([-9.*R/16., 9.*R/16.])
            ax.imshow(Z.T, origin='lower', cmap='inferno', extent=ds['extent'],
                      aspect='equal', interpolation='bicubic')


        text = ax.text(0.5, 0.1, r'$\mathrm{' + atom + r'\,' + state + r'}$', fontsize=6,ha='center', va='baseline', c=color, transform=ax.transAxes, alpha=alpha)
        #text = ax.text(0.5, 0.1, r'${\bf ' + atom + r'\,' + state + r'}$', fontsize=6,ha='center', va='baseline', c='w', transform = ax.transAxes)
        #text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])
        ax.set_axis_off()

    fig.savefig(f'{img_dir}/Mosaic_{n_cameras*snap+camera:04d}.png',
                bbox_inches='tight', transparent=True, dpi=300, pad_inches=0)
    plt.close(fig)
    gc.collect()

# -------------------- MAIN -------------------- #
# --- Global constants (must be outside __main__ for worker processes) ---
save_dir = '/orcd/home/002/parth999/work/image_data'
# save_dir = '/home/mehrotra26parth/ThesisWork/Data/Thesan-Zooms-COLT'
n_cameras = 3

if __name__ == '__main__':
    # --- Input setup --- #
    if len(sys.argv) == 3:
        sim = sys.argv[1]
        colt_dir = sys.argv[2]
        data_dir = os.path.join(colt_dir, sim)
        print(f'Collecting data for sim={sim}, data_dir={data_dir}')
    elif len(sys.argv) == 2:
        data_dir = sys.argv[1]
        print(f'Collecting data from data_dir={data_dir}')
    else:
        print('Usage: python script.py [sim] [data_dir]')
        sys.exit(1)

    metal_list = ['HI','HeI','CII','CIII','NI','SII','NII','SiII','OI','MgII','NeII','NeI','OII']

    # --- Build atom list --- #
    atom_list = []
    for state in metal_list:
        atom = state
        while atom[-1] in ['I', 'V']:
            atom = atom[:-1]
        if atom not in atom_list:
            atom_list.append(atom)

    # --- Stage 1: Normalization parallel --- #
    print("\n=== Normalizing elements in parallel ===")
    n_workers = min(os.cpu_count(), 8)
    max_snap = 1348
    n_target = 200
    n_skip = max_snap // n_target
    norm_tasks = [(atom, max_snap, n_skip, data_dir, n_cameras) for atom in atom_list]

    percentiles_dict = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for atom, perc in executor.map(stats_all_snaps, norm_tasks):
            percentiles_dict[atom] = perc
            print(f"{atom}: {perc}")

    # --- Stage 2: Frame rendering parallel --- #
    print("\n=== Rendering frames in parallel ===")
    # nsnaps = [0, 180, 181]
    nsnaps = np.arange(0, max_snap+1, 1)
    render_tasks = [(snap, camera, True, percentiles_dict, metal_list, data_dir)
                    for snap in nsnaps for camera in range(n_cameras)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        list(executor.map(plot_image, render_tasks))

    print("\nAll frames rendered successfully.")
