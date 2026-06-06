import numpy as np
import h5py
import healpy as hp
from pathlib import Path

base_dir = Path(__file__).resolve().parent
map_file_default = base_dir / 'ion-eq_map_g5760_z8_168.hdf5'
seg_file_default = base_dir / 'ion-eq_seg_g5760_z8_168.hdf5'

VERBOSE = True

def normalize(vec):
    """Return a normalized vector."""
    return vec / np.linalg.norm(vec)

def local_bearing(nside, ipix, neib, nest=False):
    """Return clockwise bearing from local north to neib, in degrees."""
    center = np.array(hp.pix2vec(nside, ipix, nest=nest))
    neib_vec = np.array(hp.pix2vec(nside, neib, nest=nest))
    theta, phi = hp.pix2ang(nside, ipix, nest=nest)

    north = np.array([-np.cos(theta)*np.cos(phi),
                      -np.cos(theta)*np.sin(phi),
                       np.sin(theta)])
    east = np.array([-np.sin(phi), np.cos(phi), 0.])

    tangent = neib_vec - np.dot(neib_vec, center) * center
    tangent = normalize(tangent)
    return (np.degrees(np.arctan2(np.dot(tangent, east),
                                  np.dot(tangent, north))) + 360.) % 360.

def clockwise_cardinal_order(angles):
    """
    Order four bearings clockwise from local north.
    The labels N, E, S, W refer to clockwise bearing quadrants.
    """
    return np.argsort(angles)

def edge_neighbors(nside, ipix, nest=False, edge_step=8, cross_edge_eps=1.e-6):
    """
    Find the four side-sharing neighbors by stepping across pixel boundaries.
    This avoids healpy's -1 cardinal slots at special RING locations.
    """
    center = np.array(hp.pix2vec(nside, ipix, nest=nest))
    boundary = hp.boundaries(nside, ipix, step=edge_step, nest=nest)

    neibs = np.zeros(4, dtype=np.int32)
    angles = np.zeros(4)
    for edge in range(4):
        midpoint = boundary[:, edge*edge_step + edge_step//2]
        outside = normalize(midpoint + cross_edge_eps * (midpoint - center))
        neib = hp.vec2pix(nside, outside[0], outside[1], outside[2], nest=nest)
        neibs[edge] = neib
        angles[edge] = local_bearing(nside, ipix, neib, nest=nest)

    return neibs[clockwise_cardinal_order(angles)]

def print_segment_ranges(neib_n, neib_e, neib_s, neib_w):
    """Print index ranges for quick checks."""
    for name, segment in zip(['neib_n', 'neib_e', 'neib_s', 'neib_w'],
                             [neib_n, neib_e, neib_s, neib_w]):
        missing = np.count_nonzero(segment < 0)
        print(f'{name}: min={segment.min()}, max={segment.max()}, missing={missing}')

def create_pixel_segments(nside=10, save_file=None, nest=False):
    """
    Create HEALPix face-neighbor segments.
    For each pixel, save the four side-sharing neighbors in clockwise order.
    """
    if save_file is None:
        save_file = base_dir / f'nside_{nside}_segments.hdf5'

    npix = hp.nside2npix(nside)
    if VERBOSE:
        print(f'Creating face-neighbor segments for {npix} pixels (nside={nside})')

    neib_n = np.zeros(npix, dtype=np.int32)
    neib_e = np.zeros(npix, dtype=np.int32)
    neib_s = np.zeros(npix, dtype=np.int32)
    neib_w = np.zeros(npix, dtype=np.int32)

    for ipix in range(npix):
        neib_n[ipix], neib_e[ipix], neib_s[ipix], neib_w[ipix] = edge_neighbors(
            nside, ipix, nest=nest)

    duplicate_rows = np.count_nonzero([
        len({neib_n[i], neib_e[i], neib_s[i], neib_w[i]}) != 4
        for i in range(npix)
    ])
    if VERBOSE:
        print_segment_ranges(neib_n, neib_e, neib_s, neib_w)
        print(f'Pixels with duplicate face neighbors: {duplicate_rows}')

    # Save to HDF5
    with h5py.File(save_file, 'w') as f:
        f.create_dataset('neib_n', data=neib_n)
        f.create_dataset('neib_e', data=neib_e)
        f.create_dataset('neib_s', data=neib_s)
        f.create_dataset('neib_w', data=neib_w)
        f.attrs['nside'] = np.int32(nside)
        f.attrs['npix'] = np.int32(npix)
        f.attrs['nest'] = np.int32(nest)
        f.attrs['neighbor_order'] = b'neib_n,neib_e,neib_s,neib_w'
        f.attrs['ordering_note'] = b'side-sharing neighbors, clockwise from local north'
        f.attrs['construction'] = b'pixel boundary midpoint crossed with healpy.vec2pix'

    if VERBOSE:
        print(f'Pixel segments saved to {save_file}')
    return neib_n, neib_e, neib_s, neib_w

def load_pixel_segments(nside=10, filename=None):
    """Load pre-computed HEALPix face-neighbor segments from HDF5 file."""
    if filename is None:
        filename = base_dir / f'nside_{nside}_segments.hdf5'

    with h5py.File(filename, 'r') as f:
        neib_n = f['neib_n'][:]
        neib_e = f['neib_e'][:]
        neib_s = f['neib_s'][:]
        neib_w = f['neib_w'][:]
        assert f.attrs['nside'] == nside
    return neib_n, neib_e, neib_s, neib_w

def compute_neighbor_differences(nside=10, segment_file=None,
                                 map_file=map_file_default):
    """
    Load face neighbors and f_esc map, then compute absolute neighbor differences.
    The returned differences are in the same units as the input map.
    """
    neib_n, neib_e, neib_s, neib_w = load_pixel_segments(nside, segment_file)

    with h5py.File(map_file, 'r') as f:
        map = f['map'][:]

    assert map.size == hp.nside2npix(nside), (
        f'map size {map.size} does not match nside={nside}')

    diff_n = np.abs(map[neib_n] - map)
    diff_e = np.abs(map[neib_e] - map)
    diff_s = np.abs(map[neib_s] - map)
    diff_w = np.abs(map[neib_w] - map)

    diff_all = np.concatenate([diff_n, diff_e, diff_s, diff_w])
    if VERBOSE:
        print('Neighbor |Delta f_esc| statistics:')
        print(f'min={np.min(diff_all):g}, max={np.max(diff_all):g}, '
              f'mean={np.mean(diff_all):g}, median={np.median(diff_all):g}')

    return map, diff_n, diff_e, diff_s, diff_w

def watershed_from_maxima(nside=10, segment_file=None, map_file=map_file_default):
    """
    Identify deterministic local maxima, then grow watershed basins.
    Return group labels plus per-group member/frontier sets, sizes, and fluxes.
    """
    UNASSIGNED = -1
    neib_n, neib_e, neib_s, neib_w = load_pixel_segments(nside, segment_file)
    neighbors = np.vstack([neib_n, neib_e, neib_s, neib_w]).T

    with h5py.File(map_file, 'r') as f:
        map = f['map'][:]

    assert map.size == hp.nside2npix(nside), (f'map size {map.size} does not match nside={nside}')

    group = np.full(map.size, UNASSIGNED, dtype=np.int32)
    group_indices = []
    neib_indices = []
    n_pixels = []
    flux = []
    n_groups = 0
    n_ties = 0
    n_tie_inherits = 0

    def add_pixel_to_group(ipix, group_id):
        ipix = int(ipix)
        group[ipix] = group_id
        for group_neibs in neib_indices:
            group_neibs.discard(ipix)
        group_indices[group_id].add(int(ipix))
        n_pixels[group_id] += 1
        flux[group_id] += map[ipix]
        for neib in neighbors[ipix]:
            if group[neib] == UNASSIGNED:
                neib_indices[group_id].add(int(neib))

    for ipix in range(map.size):
        neibs = neighbors[ipix]
        neib_values = map[neibs]
        value = map[ipix]

        if np.any(neib_values > value):
            continue

        tie_mask = (neib_values == value)
        if np.any(tie_mask):
            n_ties += 1
            tie_groups = group[neibs[tie_mask]]
            tie_groups = tie_groups[tie_groups >= 0]
            if tie_groups.size > 0:
                add_pixel_to_group(ipix, int(np.min(tie_groups)))
                n_tie_inherits += 1
                continue

        group_indices.append(set())
        neib_indices.append(set())
        n_pixels.append(0)
        flux.append(0.)
        add_pixel_to_group(ipix, n_groups)
        n_groups += 1

    n_pixels = np.array(n_pixels, dtype=np.int32)
    flux = np.array(flux, dtype=np.float64)
    active_groups = [True] * n_groups
    max_neib_index = np.full(n_groups, UNASSIGNED, dtype=np.int32)
    max_neib_value = np.full(n_groups, -np.inf, dtype=np.float64)

    def update_group_max(group_id):
        if len(neib_indices[group_id]) == 0:
            active_groups[group_id] = False
            max_neib_index[group_id] = UNASSIGNED
            max_neib_value[group_id] = -np.inf
            return

        active_groups[group_id] = True
        best = max(neib_indices[group_id], key=lambda ipix: (map[ipix], -ipix))
        max_neib_index[group_id] = best
        max_neib_value[group_id] = map[best]

    for group_id in range(n_groups):
        update_group_max(group_id)

    if VERBOSE:
        print('Watershed maxima:')
        print(f'maxima={n_groups}, encountered_ties={n_ties}, '
            f'tie_inherits={n_tie_inherits}')
        print(f'group index sets={len(group_indices)}, '
            f'neighbor index sets={len(neib_indices)}')

    n_steps = 0
    n_value_ties = 0
    n_pixel_ties = 0
    while any(active_groups):
        active_ids = [i for i, active in enumerate(active_groups) if active]
        best_value = np.max(max_neib_value[active_ids])
        candidate_groups = [
            i for i in active_ids if max_neib_value[i] == best_value
        ]
        if len(candidate_groups) > 1:
            n_value_ties += 1

        group_id = max(candidate_groups,
                       key=lambda i: (n_pixels[i], flux[i], -i))
        ipix = int(max_neib_index[group_id])

        tied_groups = [
            i for i in candidate_groups if int(max_neib_index[i]) == ipix
        ]
        if len(tied_groups) > 1:
            n_pixel_ties += 1
            group_id = max(tied_groups,
                           key=lambda i: (n_pixels[i], flux[i], -i))

        affected_groups = [
            i for i in active_ids if ipix in neib_indices[i]
        ]

        if group[ipix] != UNASSIGNED:
            raise RuntimeError(f'Pixel {ipix} was selected after assignment')

        group[ipix] = group_id
        group_indices[group_id].add(ipix)
        n_pixels[group_id] += 1
        flux[group_id] += map[ipix]

        for i in affected_groups:
            neib_indices[i].discard(ipix)
        for neib in neighbors[ipix]:
            if group[neib] == UNASSIGNED:
                neib_indices[group_id].add(int(neib))

        for i in set(affected_groups + [group_id]):
            update_group_max(i)

        n_steps += 1

    n_pixels_total = np.sum(n_pixels)
    flux_total = np.sum(flux)
    map_flux = np.sum(map)
    if n_pixels_total != map.size:
        raise RuntimeError(
            f'Watershed assigned {n_pixels_total} pixels, expected {map.size}')
    if np.any(group == UNASSIGNED):
        raise RuntimeError('Watershed ended with unASSIGNED pixels')
    if not np.isclose(flux_total, map_flux, rtol=1.e-12, atol=1.e-14):
        raise RuntimeError(
            f'Watershed flux {flux_total:g} does not match map sum {map_flux:g}')
    for i in range(len(neib_indices)):
        if len(neib_indices[i]) != 0:
            raise RuntimeError(f'neib_indices[{i}] is not empty after watershed: {neib_indices[i]}')

    if VERBOSE:
        print('Watershed growth:')
        print(f'steps={n_steps}, value_ties={n_value_ties}, '
            f'pixel_ties={n_pixel_ties}')
        print(f'group size min/max/mean/median: {np.min(n_pixels)} / '
            f'{np.max(n_pixels)} / {np.mean(n_pixels):g} / '
            f'{np.median(n_pixels):g}')
        print(f'group flux min/max/sum: {np.min(flux):g} / '
            f'{np.max(flux):g} / {flux_total:g}')
        # print(f'group = {group}')
        # print(f'group_indices = {group_indices}')
        # print(f'n_pixels = {n_pixels}')
        # print(f'flux = {flux}')
    return group, group_indices, n_pixels, flux

def group_indices_to_csr(group_indices):
    """Pack variable-length group index arrays into indices/indptr arrays."""
    lengths = np.array([len(indices) for indices in group_indices], dtype=np.int32)
    indptr = np.zeros(len(group_indices)+1, dtype=np.int32)
    indptr[1:] = np.cumsum(lengths)
    indices = np.zeros(indptr[-1], dtype=np.int32)

    for group_id, values in enumerate(group_indices):
        start, stop = indptr[group_id], indptr[group_id+1]
        indices[start:stop] = np.array(sorted(values), dtype=np.int32)

    return indices, indptr

def write_segmented_groups(seg_file=seg_file_default, nside=10,
                           segment_file=None, map_file=map_file_default,
                           group=None, group_indices=None,
                           n_pixels=None, flux=None):
    """Write watershed segmented groups to HDF5."""
    if group is None or group_indices is None or n_pixels is None or flux is None:
        group, group_indices, n_pixels, flux = watershed_from_maxima(
            nside=nside, segment_file=segment_file, map_file=map_file)

    group = np.asarray(group, dtype=np.int32)
    n_pixels = np.asarray(n_pixels, dtype=np.int32)
    flux = np.asarray(flux, dtype=np.float64)
    indices, indptr = group_indices_to_csr(group_indices)

    assert group.size == hp.nside2npix(nside), (
        f'group size {group.size} does not match nside={nside}')
    assert len(group_indices) == n_pixels.size == flux.size
    assert indptr[-1] == group.size

    with h5py.File(seg_file, 'w') as f:
        f.create_dataset('group', data=group)
        f.create_dataset('n_pixels', data=n_pixels)
        f.create_dataset('flux', data=flux)
        f.create_dataset('group_indices', data=indices)
        f.create_dataset('group_indptr', data=indptr)
        f.attrs['nside'] = np.int32(nside)
        f.attrs['npix'] = np.int32(group.size)
        f.attrs['n_groups'] = np.int32(n_pixels.size)
        f.attrs['format'] = b'group indices stored as CSR-style indices/indptr'

    if VERBOSE:
        print(f'Segmented groups saved to {seg_file}')
    return group, np.array([np.array(sorted(values), dtype=np.int32)
                            for values in group_indices], dtype=object), n_pixels, flux

def read_segmented_groups(seg_file=seg_file_default):
    """Read watershed segmented groups from HDF5."""
    with h5py.File(seg_file, 'r') as f:
        group = f['group'][:].astype(np.int32)
        n_pixels = f['n_pixels'][:].astype(np.int32)
        flux = f['flux'][:].astype(np.float64)
        indices = f['group_indices'][:].astype(np.int32)
        indptr = f['group_indptr'][:].astype(np.int32)

        group_indices = np.empty(indptr.size-1, dtype=object)
        for group_id in range(group_indices.size):
            start, stop = indptr[group_id], indptr[group_id+1]
            group_indices[group_id] = indices[start:stop].astype(np.int32, copy=True)

        assert f.attrs['npix'] == group.size
        assert f.attrs['n_groups'] == group_indices.size

    if VERBOSE:
        print(f'Segmented groups loaded from {seg_file}')
    return group, group_indices, n_pixels, flux

def percentile_string(data, n_format=1):
    """Return median and 68-percent range in the plot-maps.py text style."""
    lo, med, hi = np.percentile(data, [15.865525393145708, 50., 84.1344746068543])
    fmt = f'%0.{n_format}f'
    return r'$' + fmt % med + r'^{+' + fmt % (hi-med) + r'}_{-' + fmt % (med-lo) + r'}$'

def print_map_limits(label, data, lims):
    """Print map statistics and colorbar limits."""
    lo, med, hi = np.percentile(data, [15.865525393145708, 50., 84.1344746068543])
    print(f'{label}: {med:g}  [{lo:g}, {hi:g}]  '
          f'Avg/Min/Max: {np.mean(data):g}  [{np.min(data):g}, {np.max(data):g}]')
    print(f'{label} lims: [{lims[0]:g}, {lims[1]:g}]')

def HpPlot(f, extent, map, u_str=None, w_str=None, lims=None, cmap=None,
           n_format=0):
    """Plot a HEALPix map using the compact plot-maps.py Mollweide style."""
    import copy
    import matplotlib.pyplot as plt
    from healpy import projaxes as PA
    from healpy import pixelfunc

    if cmap is None:
        cmap = plt.cm.afmhot
    cmap = copy.copy(cmap)
    cmap.set_under('w')

    map = pixelfunc.ma_to_array(map)
    ax = PA.HpxMollweideAxes(f, extent)
    f.add_axes(ax)
    if lims is None:
        img = ax.projmap(map, cmap=cmap)
    else:
        img = ax.projmap(map, cmap=cmap, vmin=lims[0], vmax=lims[1])

    im = ax.get_images()[0]
    b = im.norm.inverse(np.linspace(0, 1, im.cmap.N+1))
    v = np.linspace(im.norm.vmin, im.norm.vmax, im.cmap.N)
    cb = f.colorbar(im, ax=ax, orientation='horizontal',
                    shrink=0.75, aspect=25, ticks=PA.BoundaryLocator(),
                    pad=0.05, fraction=0.1, boundaries=b, values=v,
                    format=r'${\rm '+str('%0.'+str(n_format)+'f')+r'}$')
    cb.solids.set_rasterized(True)

    if w_str is not None:
        ax.text(0.875, -.025, w_str, fontsize=12, ha='center',
                va='baseline', transform=ax.transAxes)
    if u_str is not None:
        cb.ax.text(0.5, -2.0, u_str, fontsize=14.5,
                   transform=cb.ax.transAxes, ha='center', va='center')
    f.sca(ax)

def discrete_group_cmap(n_groups):
    """Create a deterministic, visually varied discrete colormap."""
    import matplotlib.colors as colors

    golden = 0.6180339887498949
    i = np.arange(n_groups)
    hue = (i * golden) % 1.
    sat = 0.58 + 0.34 * ((i * 7) % 11) / 10.
    val = 0.68 + 0.28 * ((i * 5) % 13) / 12.
    rgb = colors.hsv_to_rgb(np.vstack([hue, sat, val]).T)
    return colors.ListedColormap(rgb, name=f'groups_{n_groups}')

def HpGroupPlot(f, extent, group, n_groups, u_str=None):
    """Plot a segmented HEALPix group map with one discrete color per group."""
    from healpy import projaxes as PA
    from healpy import pixelfunc

    if u_str is None:
        u_str = r'${\rm %d\ Groups}$' % n_groups

    group = pixelfunc.ma_to_array(group)
    cmap = discrete_group_cmap(n_groups)
    ax = PA.HpxMollweideAxes(f, extent)
    f.add_axes(ax)
    ax.projmap(group, cmap=cmap, vmin=-0.5, vmax=n_groups-0.5)
    im = ax.get_images()[0]
    boundaries = np.arange(n_groups+1) - 0.5
    values = np.arange(n_groups)
    cb = f.colorbar(im, ax=ax, orientation='horizontal',
                    shrink=0.75, aspect=25, ticks=[],
                    pad=0.05, fraction=0.1,
                    boundaries=boundaries, values=values)
    cb.solids.set_rasterized(True)
    cb.ax.text(0.5, -2.0, u_str, fontsize=14.5,
               transform=cb.ax.transAxes, ha='center', va='center')
    f.sca(ax)

def test_plot_neighbor_differences(nside=10, segment_file=None,
                                   map_file=map_file_default,
                                   save_file=None, seg_file=None):
    """
    Plot the original f_esc HEALPix map and mean neighbor Delta f_esc map.
    Plotted values are percentages, matching plot-maps.py's LyC map units.
    """
    import matplotlib.pyplot as plt
    import cmasher as cmr

    if save_file is None:
        save_file = base_dir / 'delta_fesc_segments.pdf'

    map, diff_n, diff_e, diff_s, diff_w = compute_neighbor_differences(
        nside=nside, segment_file=segment_file, map_file=map_file)
    diff_map = (diff_n + diff_e + diff_s + diff_w) / 4.

    fesc = map / 0.01
    delta_fesc = diff_map / 0.01
    fesc_lims = [0., np.max(fesc)]
    delta_lims = [0., np.max(delta_fesc)]

    if VERBOSE:
        print_map_limits('f_esc (%)', fesc, fesc_lims)
        print_map_limits('Delta f_esc (%)', delta_fesc, delta_lims)

    group = None
    if seg_file is not None:
        group, group_indices, n_pixels, flux = read_segmented_groups(seg_file)
        assert group.size == map.size, (
            f'group size {group.size} does not match map size {map.size}')
        if VERBOSE:
            print(f'Segmented groups: n_groups={len(group_indices)}, '
                  f'n_pixels min/max={np.min(n_pixels)}/{np.max(n_pixels)}, '
                  f'flux sum={np.sum(flux):g}')

    fig = plt.figure(figsize=(3., 2.))
    dy_map = 1.045
    HpPlot(fig, (0, dy_map, 1, 1), fesc,
           u_str=r'$f_{\rm esc}^{\rm\,LyC}\ \ (\%)$',
           w_str=percentile_string(fesc, n_format=1),
           lims=fesc_lims, cmap=cmr.ember, n_format=0)
    HpPlot(fig, (0, 0, 1, 1), delta_fesc,
           u_str=r'$\Delta f_{\rm esc}^{\rm\,LyC}\ \ (\%)$',
           w_str=percentile_string(delta_fesc, n_format=1),
           lims=delta_lims, cmap=cmr.amber, n_format=0)
    if group is not None:
        HpGroupPlot(fig, (1.01, dy_map, 1, 1), group, len(group_indices))

    sargs = {'bbox_inches':'tight', 'pad_inches':0.,
             'transparent':False, 'dpi':640}
    plt.savefig(save_file, **sargs)
    plt.close(fig)
    if VERBOSE:
        print(f'Neighbor difference plot saved to {save_file}')
    if group is None:
        return fesc, delta_fesc
    return fesc, delta_fesc, group

if __name__ == '__main__':
    # create_pixel_segments()
    write_segmented_groups()
    test_plot_neighbor_differences(seg_file=seg_file_default)
