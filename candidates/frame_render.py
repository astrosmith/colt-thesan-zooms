import numpy as np
import h5py
import matplotlib
from matplotlib.colors import LogNorm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cmasher as cmr
import os
import sys
import gc
from concurrent.futures import ProcessPoolExecutor
import matplotlib as mpl, matplotlib.cm as cm
import warnings
if not hasattr(cm, "register_cmap"):  # mpl ≥3.9 dropped cm.register_cmap
    cm.register_cmap = mpl.colormaps.register
    cm.unregister_cmap = mpl.colormaps.unregister

import cmastro

kpc = 3.086e21  # cm

# --- Optional: silence Matplotlib open figure warnings ---
warnings.filterwarnings("ignore", category=RuntimeWarning, module="matplotlib")

def nearest_R(R):        
    return R if R <= 10.0 else np.floor(R)

def read_data(field, filename="/home/mehrotra26parth/ThesisWork/Test1/Thesan-Zooms-COLT/g500531/z4/output/proj_188.hdf5", camera=0):
    ds = {'filename': filename}
    z_arr = np.array([])
    with h5py.File(filename, 'r') as f:
        # print([key for key in f.keys()])
        for key in f.attrs.keys():
            ds[key] = f.attrs[key]
        ds[field] = f[field][camera,:]
        if 'star_proj_m_sum' in f.keys():
            ds['star_proj_m_sum'] = f['star_proj_m_sum'][camera,:]
        redshift = f.attrs['z']
        z_arr = np.append(z_arr, redshift)
        Rx = ds['image_radius_x']/kpc
        Ry = ds['image_radius_y']/kpc
        ds['extent'] = [-Rx,Rx,-Ry,Ry]
        # print(R, ds[field].shape)
    return ds, z_arr

def stats_all_snaps(field, max_snap=188, n_skip = 8,data_dir = "/home/mehrotra26parth/ThesisWork/Data/Thesan-Zooms-COLT/g578/z4" ,percentiles=np.array([1,10,50,90,99.99])):
    Z = np.array([])
    n_snaps = np.arange(0, max_snap+1, n_skip)
    #n_snaps=[
    for snap in n_snaps:
        filename = data_dir + f'/output_movie/proj_rho_{snap:04d}.hdf5'
        with h5py.File(filename, 'r') as f:
            Z = np.append(Z, f[field][:].flatten())
            
    Z_p = np.percentile(Z, percentiles)
    if min(Z_p) == 0:
        print(f'Warning: {field} has non-positive percentiles: {Z_p}')
        Z = Z[Z>0]
        Z_p = np.percentile(Z, percentiles)
        print(f'Updated percentiles for {field} = {Z_p}')
    return Z_p

def plot_proj(field, data_dir, camera=0, lowres=False,snap=187, star_proj=True, perc=None):
    fig = plt.figure(figsize=(4.,4.))
    ax = plt.axes([0,0,1,1])
    if lowres:
        filename = f'{data_dir}/output_tree_rot_lowres/proj_lowres_{snap:03d}.hdf5'
    else:
        filename = f'{data_dir}/output_movie/proj_rho_{snap:04d}.hdf5'

    ds, z_arr = read_data(field,filename,camera)
    Z = ds[field]
    # perc = stats_all_snaps(field)
    interpolation = 'none'
    # Z*=1e26
    Z_min = np.min(Z[Z>0])
    Z_max = np.max(Z)
    print(f'Processing snap {snap}, camera {camera}:', f'min/max SB = [{Z_min}, {Z_max}]')
    alpha = 0.5
    color = 'white'
    # cmap = cmastro.laguna
    image = ax.imshow(Z.T, origin='lower', 
        cmap=cmr.ember, aspect='equal', interpolation=interpolation, norm=LogNorm(vmin=perc[0] , vmax=perc[-1]), rasterized=True, extent=ds['extent'])
    # Star field overlay:
    if star_proj:
        try:
            star = ds['star_proj_m_sum'].T
            star = gaussian_filter(star,1.5)
            star /= np.max(star)
            f_cut = 1e-2
            star[star<f_cut] = f_cut
            front = np.ones([ds['ny_pixels'], ds['nx_pixels'], 4])
            front[:, :, 3] = (1. - np.log10(star) / np.log10(f_cut))
            ax.imshow(
                front,
                origin="lower",
                aspect="equal",
                interpolation="none",
                rasterized=True,
                extent=ds['extent']
            )
            # ax.set_axis_off()
        except Exception as e:
            print(f"Star field overlay failed: {e}")
            
    if lowres:
        img_dir = data_dir + f'/image_data_lowres_{snap}/{field}'
    else:
        img_dir = save_dir + f'/image_data/{sim}/{field}' 
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        # print(f"Created directory: {img_dir}")
    ax.set_axis_off()
    if snap == 0:
        ax.text(0.02, 0.02, f"${{z = {z_arr[snap]:.1f}}}$",color=color,fontsize=14, ha="left",va="bottom",transform=ax.transAxes, alpha=alpha)
    else:
        ax.text(0.02, 0.02, f"${{z = {z_arr[0]:.1f}}}$",color=color,fontsize=14, ha="left",va="bottom",transform=ax.transAxes, alpha=alpha)
    # Calculate the position for the bottom right corner
    Rx = nearest_R(ds['extent'][1])
    # Place the text at (0.97, 0.05) in axes coordinates (bottom right)
    ax.text(0.96, 0.02, f"${{{Rx:.0f} \\, \\mathrm{{kpc}}}}$", ha='right', va='bottom',
            color=color, fontsize=14, transform=ax.transAxes, alpha=alpha)
            # path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
    ax.text(0.975, 0.97, 'THESAN', color=color, ha='right', va='top', transform=ax.transAxes, size=14, alpha=alpha) # weight='bold'
    ax.text(0.02, 0.97, f'm9.3', color=color, ha='left', va='top', transform=ax.transAxes, size=14, alpha=alpha) # weight='bold'
    
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
    ax.plot([x0, x1], [y_bar, y_bar],
            lw=1.5, c=color, transform=ax.transAxes, solid_capstyle='butt', alpha=alpha)

    # Save the figure once
    fig.savefig(f'{img_dir}/Image_{n_cameras*snap+camera:04d}.png', bbox_inches='tight', pad_inches=0., transparent=True, dpi=300)
    # --- Cleanup ---
    plt.close(fig)
    plt.close('all')
    plt.clf()
    del fig, ax, Z
    gc.collect()

# --- Wrapper function for ProcessPoolExecutor ---
def plot_proj_wrapper(args):
    try:
        plot_proj(*args)
    except Exception as e:
        import traceback
        print(f"Worker failed for snap/camera in args: {e}")
        traceback.print_exc()
        raise

# --- Global constants (must be outside __main__ for worker processes) ---
save_dir = '/orcd/home/002/parth999/work'
n_cameras = 3
sim = None  # Will be set in __main__

if __name__ == '__main__':
    
    #test_dir = 'Data'
    #group = 'g500531'
    #res = 'z16'
    #sim = group + '/' + res
    #colt_dir = f"/home/mehrotra26parth/ThesisWork/{test_dir}/Thesan-Zooms-COLT"

    if len(sys.argv) == 4:
        sim = sys.argv[1]
        field = sys.argv[2]
        data_dir = sys.argv[3]
        print(f'Collecting data for sim={sim} , field={field}, data_dir={data_dir}')
    elif len(sys.argv) == 3:
        sim = 'default'  # Fallback for 3-arg mode
        field = sys.argv[1]
        data_dir = sys.argv[2]
        print(f'Collecting data for field={field}, data_dir={data_dir}')
    else:
        print('Expected to be run as `python frame_render_parallel.py [sim] [field] [data_dir]')
        sys.exit(1)


    max_snap = 1348
    n_target = 120
    n_skip = max_snap // n_target

    snaps = np.arange(0,max_snap+1,1)
    #n_snaps=[0]
    perc = stats_all_snaps(field, max_snap=max_snap, n_skip=n_skip ,data_dir=data_dir)

    print(f"Percentiles: {perc}")

    # Prepare tasks for parallel rendering
    # Each task: (field, data_dir, camera, lowres, snap, star_proj, perc)
    tasks = [(field, data_dir, camera, False, snap, True, perc) for snap in snaps for camera in range(n_cameras)]

    # Render frames in parallel using batches
    n_workers = max(1, int(os.getenv("SLURM_CPUS_ON_NODE", "1")))
    # Limit workers to prevent too many simultaneous matplotlib instances
    # n_workers = min(n_workers, 8)  # Cap at 8 workers
    nodelist = os.getenv('SLURM_JOB_NODELIST', 'local')
    print(f"Running with {n_workers} workers on {nodelist}")

    # Process in batches to prevent "too many open files" error
    batch_size = 20  # Adjust this based on your system (lower = safer, higher = faster)
    n_batches = (len(tasks) - 1) // batch_size + 1

    print(f"Processing {len(tasks)} tasks in {n_batches} batches of {batch_size}")

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(tasks))
        batch = tasks[start_idx:end_idx]
        print(f"Processing batch {batch_idx + 1}/{n_batches} ({len(batch)} frames)...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Consume the iterator to surface exceptions
            list(executor.map(plot_proj_wrapper, batch))
        # Force cleanup between batches
        gc.collect()
        print(f"Batch {batch_idx + 1}/{n_batches} complete. Memory cleaned.")

    print("All frames done!")
