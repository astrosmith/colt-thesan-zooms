import numpy as np
import h5py
import healpy as hp
from pathlib import Path
from healpy import rotator as R
from scipy.special import erf

base_dir = Path(__file__).resolve().parent
map_file_default = base_dir / 'ion-eq_map_g5760_z8_168.hdf5'
seg_file_default = base_dir / 'ion-eq_seg_g5760_z8_168.hdf5'

VERBOSE = True

sigma_68 = erf(1./np.sqrt(2.))
sigma_95 = erf(2./np.sqrt(2.))
sigma_99 = erf(3./np.sqrt(2.))
percentiles = np.array([
    50.,
    50.*(1.-sigma_68), 50.*(1.+sigma_68),
    50.*(1.-sigma_95), 50.*(1.+sigma_95),
    50.*(1.-sigma_99), 50.*(1.+sigma_99),
], dtype=np.float64)
n_percentiles = len(percentiles)
percentiles_68 = percentiles[[1, 0, 2]]

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
        # f.attrs['neighbor_order'] = b'neib_n,neib_e,neib_s,neib_w'
        # f.attrs['ordering_note'] = b'side-sharing neighbors, clockwise from local north'
        # f.attrs['construction'] = b'pixel boundary midpoint crossed with healpy.vec2pix'

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
    Return group labels plus per-group member/frontier sets, sizes, sums, and peaks.
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
    max_flux = []
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
        max_flux[group_id] = max(max_flux[group_id], map[ipix])
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
        max_flux.append(-np.inf)
        add_pixel_to_group(ipix, n_groups)
        n_groups += 1

    n_pixels = np.array(n_pixels, dtype=np.int32)
    flux = np.array(flux, dtype=np.float64)
    max_flux = np.array(max_flux, dtype=np.float64)
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
        max_flux[group_id] = max(max_flux[group_id], map[ipix])

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
        raise RuntimeError('Watershed ended with unassigned pixels')
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
        print(f'group max_flux min/max: {np.min(max_flux):g} / '
            f'{np.max(max_flux):g}')
        # print(f'group = {group}')
        # print(f'group_indices = {group_indices}')
        # print(f'n_pixels = {n_pixels}')
        # print(f'flux = {flux}')
        # print(f'max_flux = {max_flux}')
    return group, group_indices, n_pixels, flux, max_flux

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

def boundary_indices_to_csr(group_inner_indices, group_outer_indices):
    """Pack ordered boundary inner/outer arrays into CSR-style arrays."""
    lengths = np.array([len(indices) for indices in group_inner_indices],
                       dtype=np.int32)
    indptr = np.zeros(len(group_inner_indices)+1, dtype=np.int32)
    indptr[1:] = np.cumsum(lengths)
    inner_indices = np.zeros(indptr[-1], dtype=np.int32)
    outer_indices = np.zeros(indptr[-1], dtype=np.int32)

    for group_id in range(len(group_inner_indices)):
        inner = np.asarray(group_inner_indices[group_id], dtype=np.int32)
        outer = np.asarray(group_outer_indices[group_id], dtype=np.int32)
        if inner.size != outer.size:
            raise RuntimeError(
                f'Boundary inner/outer lengths differ for group {group_id}')

        start, stop = indptr[group_id], indptr[group_id+1]
        inner_indices[start:stop] = inner
        outer_indices[start:stop] = outer

    return inner_indices, outer_indices, indptr

def group_vertices_to_csr(group_vertices):
    """Pack variable-length group vertex arrays into vertices/indptr arrays."""
    lengths = np.array([len(vertices) for vertices in group_vertices],
                       dtype=np.int32)
    indptr = np.zeros(len(group_vertices)+1, dtype=np.int32)
    indptr[1:] = np.cumsum(lengths)
    vertices = np.zeros((indptr[-1], 2), dtype=np.float64)

    for group_id, values in enumerate(group_vertices):
        start, stop = indptr[group_id], indptr[group_id+1]
        vertices[start:stop] = np.asarray(values, dtype=np.float64)

    return vertices, indptr

def neighbor_cycle_index(neighbors, ipix, neib):
    """Return the clockwise neighbor-cycle index linking ipix to neib."""
    matches = np.where(neighbors[ipix] == neib)[0]
    if matches.size != 1:
        raise RuntimeError(
            f'Pixel {ipix} does not have unique face neighbor {neib}')
    return int(matches[0])

def theta_phi_from_vec(vertices):
    """Convert unit vectors to HEALPix theta/phi coordinates in radians."""
    vertices = np.asarray(vertices, dtype=np.float64)
    theta = np.arccos(np.clip(vertices[:, 2], -1., 1.))
    phi = np.mod(np.arctan2(vertices[:, 1], vertices[:, 0]), 2.*np.pi)
    return np.vstack([theta, phi]).T

def get_ordered_edge_vertices(nside, neighbors, nest=False,
                              edge_step=16, cross_edge_eps=1.e-6):
    """Return sampled pixel edge vertices matching neib_n/e/s/w cycles."""
    edge_vertices = np.zeros((neighbors.shape[0], 4, edge_step+1, 3),
                             dtype=np.float64)

    for ipix in range(neighbors.shape[0]):
        center = np.array(hp.pix2vec(nside, ipix, nest=nest))
        boundary = hp.boundaries(nside, ipix, step=edge_step, nest=nest)
        corners = hp.boundaries(nside, ipix, step=1, nest=nest)

        raw_neibs = np.zeros(4, dtype=np.int32)
        raw_angles = np.zeros(4, dtype=np.float64)
        raw_vertices = np.zeros((4, edge_step+1, 3), dtype=np.float64)
        for edge in range(4):
            midpoint = boundary[:, edge*edge_step + edge_step//2]
            outside = normalize(midpoint + cross_edge_eps * (midpoint - center))
            neib = hp.vec2pix(nside, outside[0], outside[1], outside[2],
                              nest=nest)
            raw_neibs[edge] = neib
            raw_angles[edge] = local_bearing(nside, ipix, neib, nest=nest)
            start = edge * edge_step
            stop = (edge + 1) * edge_step
            raw_vertices[edge, :-1] = boundary[:, start:stop].T
            raw_vertices[edge, -1] = corners[:, (edge+1) % 4]

        order = clockwise_cardinal_order(raw_angles)
        if not np.array_equal(raw_neibs[order], neighbors[ipix]):
            raise RuntimeError(
                f'Neighbor cycle mismatch while building vertices for pixel {ipix}')
        edge_vertices[ipix] = raw_vertices[order]

    return edge_vertices

def face_segment_edge_vertices(neighbors, edge_vertices, inner, outer):
    """Return Cartesian vertex samples for a directed face boundary segment."""
    cycle = neighbor_cycle_index(neighbors, int(inner), int(outer))
    return edge_vertices[int(inner), cycle]

def common_edge_vertex(edge_a, edge_b, tol=1.e-8):
    """Return the common endpoint of two adjacent boundary edges."""
    dots = np.dot(edge_a, edge_b.T)
    idx = np.unravel_index(np.argmax(dots), dots.shape)
    if dots[idx] < 1. - tol:
        raise RuntimeError('Adjacent boundary edges do not share a vertex')
    return edge_a[idx[0]]

def orient_edge_vertices(edge, start_vertex, end_vertex):
    """Orient sampled edge vertices from start_vertex to end_vertex."""
    forward = np.dot(edge[0], start_vertex) + np.dot(edge[-1], end_vertex)
    reverse = np.dot(edge[-1], start_vertex) + np.dot(edge[0], end_vertex)
    if forward >= reverse:
        return edge
    return edge[::-1]

def boundary_loop_vertices(loop_segments, neighbors, edge_vertices):
    """Return closed theta/phi vertices for one ordered boundary loop."""
    n_segments = len(loop_segments)
    if n_segments == 0:
        return np.empty((0, 2), dtype=np.float64)

    segment_edges = [
        face_segment_edge_vertices(neighbors, edge_vertices, inner, outer)
        for inner, outer in loop_segments
    ]
    loop_samples = []
    for i, edge in enumerate(segment_edges):
        start_vertex = common_edge_vertex(segment_edges[i-1], edge)
        end_vertex = common_edge_vertex(edge, segment_edges[(i+1) % n_segments])
        edge = orient_edge_vertices(edge, start_vertex, end_vertex)
        if i == 0:
            loop_samples.append(edge)
        else:
            loop_samples.append(edge[1:])

    vertices = np.vstack(loop_samples)
    if np.dot(vertices[0], vertices[-1]) < 1. - 1.e-12:
        vertices = np.vstack([vertices, vertices[0]])
    else:
        vertices[-1] = vertices[0]
    return theta_phi_from_vec(vertices)

def next_clockwise_boundary_segment(group, neighbors, group_id, inner, outer):
    """
    Return the next clockwise directed boundary segment.
    A segment is (inner group pixel, outer face-neighbor pixel).
    """
    current_inner = int(inner)
    current_outer = int(outer)
    cycle = (
        neighbor_cycle_index(neighbors, current_inner, current_outer) + 1) % 4
    visited = set()
    max_steps = 4 * np.count_nonzero(group == group_id) + 4

    for _ in range(max_steps):
        state = (current_inner, cycle)
        if state in visited:
            raise RuntimeError(
                f'Boundary walk cycled before finding an edge for group {group_id}')
        visited.add(state)

        candidate = int(neighbors[current_inner, cycle])
        if group[candidate] != group_id:
            return current_inner, candidate

        previous_inner = current_inner
        current_inner = candidate
        back_cycle = neighbor_cycle_index(neighbors, current_inner, previous_inner)
        cycle = (back_cycle + 1) % 4

    raise RuntimeError(f'Boundary walk exceeded max steps for group {group_id}')

def walk_group_boundary(group, neighbors, edge_vertices, group_id):
    """Return clockwise ordered boundary segments and vertices for one group."""
    remaining = set()
    members = np.where(group == group_id)[0]
    for ipix in members:
        for neib in neighbors[ipix]:
            if group[neib] != group_id:
                remaining.add((int(ipix), int(neib)))

    inner_indices = []
    outer_indices = []
    group_vertices = []
    n_expected = len(remaining)
    while remaining:
        start_segment = min(remaining)
        segment = start_segment
        loop_segments = []
        while segment in remaining:
            remaining.remove(segment)
            inner_indices.append(segment[0])
            outer_indices.append(segment[1])
            loop_segments.append(segment)

            next_segment = next_clockwise_boundary_segment(
                group, neighbors, group_id, segment[0], segment[1])
            if next_segment == start_segment:
                break
            segment = next_segment
        loop_vertices = boundary_loop_vertices(
            loop_segments, neighbors, edge_vertices)
        if len(group_vertices) > 0 and loop_vertices.size > 0:
            group_vertices.append(np.array([[np.nan, np.nan]],
                                           dtype=np.float64))
        if loop_vertices.size > 0:
            group_vertices.append(loop_vertices)

    if len(inner_indices) != n_expected:
        raise RuntimeError(
            f'Boundary walk stored {len(inner_indices)} segments for group '
            f'{group_id}, expected {n_expected}')

    if len(group_vertices) == 0:
        vertices = np.empty((0, 2), dtype=np.float64)
    else:
        vertices = np.vstack(group_vertices)

    return (np.asarray(inner_indices, dtype=np.int32),
            np.asarray(outer_indices, dtype=np.int32),
            vertices)

def get_group_boundaries(group, neighbors, nside, nest=False):
    """Return ordered inner/outer boundary pixel arrays and vertices."""
    n_groups = int(np.max(group)) + 1
    edge_vertices = get_ordered_edge_vertices(nside, neighbors, nest=nest)
    group_inner_indices = np.empty(n_groups, dtype=object)
    group_outer_indices = np.empty(n_groups, dtype=object)
    group_vertices = np.empty(n_groups, dtype=object)

    for group_id in range(n_groups):
        inner, outer, vertices = walk_group_boundary(
            group, neighbors, edge_vertices, group_id)
        group_inner_indices[group_id] = inner
        group_outer_indices[group_id] = outer
        group_vertices[group_id] = vertices

    return group_inner_indices, group_outer_indices, group_vertices

def write_segmented_groups(seg_file=seg_file_default, nside=10,
                           segment_file=None, map_file=map_file_default,
                           group=None, group_indices=None,
                           n_pixels=None, flux=None, max_flux=None,
                           n_pixels_sigma=None,
                           write_boundaries=False):
    """Write watershed segmented groups to HDF5, optionally with boundaries."""
    if (group is None or group_indices is None or n_pixels is None or
            flux is None or max_flux is None):
        group, group_indices, n_pixels, flux, max_flux = watershed_from_maxima(
            nside=nside, segment_file=segment_file, map_file=map_file)

    group = np.asarray(group, dtype=np.int32)
    n_pixels = np.asarray(n_pixels, dtype=np.int32)
    flux = np.asarray(flux, dtype=np.float64)
    max_flux = np.asarray(max_flux, dtype=np.float64)
    indices, indptr = group_indices_to_csr(group_indices)

    assert group.size == hp.nside2npix(nside), (
        f'group size {group.size} does not match nside={nside}')
    assert len(group_indices) == n_pixels.size == flux.size == max_flux.size
    assert indptr[-1] == group.size

    with h5py.File(map_file, 'r') as f:
        map = f['map'][:]
    assert map.size == group.size, (
        f'map size {map.size} does not match group size {group.size}')

    flux_sigma_targets = top_sigma_flux_targets()
    if n_pixels_sigma is None:
        n_pixels_sigma, flux_sigma_targets = get_group_flux_pixel_counts(
            group_indices, map, targets=flux_sigma_targets)
    n_pixels_sigma = np.asarray(n_pixels_sigma, dtype=np.float64)
    assert n_pixels_sigma.shape == (n_pixels.size, flux_sigma_targets.size)

    group_inner_indices = None
    group_outer_indices = None
    group_vertices = None
    boundary_indptr = None
    vertex_indptr = None
    if write_boundaries and n_pixels.size > 1:
        neib_n, neib_e, neib_s, neib_w = load_pixel_segments(nside, segment_file)
        neighbors = np.vstack([neib_n, neib_e, neib_s, neib_w]).T
        group_inner_indices, group_outer_indices, group_vertices = (
            get_group_boundaries(group, neighbors, nside))
        boundary_inner, boundary_outer, boundary_indptr = boundary_indices_to_csr(
            group_inner_indices, group_outer_indices)
        boundary_vertices, vertex_indptr = group_vertices_to_csr(group_vertices)

    with h5py.File(seg_file, 'w') as f:
        f.create_dataset('group', data=group)
        f.create_dataset('n_pixels', data=n_pixels)
        f.create_dataset('flux', data=flux)
        f.create_dataset('max_flux', data=max_flux)
        f.create_dataset('n_pixels_sigma', data=n_pixels_sigma)
        f.create_dataset('flux_sigma_targets', data=flux_sigma_targets)
        f.create_dataset('group_indices', data=indices)
        f.create_dataset('group_indptr', data=indptr)
        if boundary_indptr is not None:
            f.create_dataset('group_inner_indices', data=boundary_inner)
            f.create_dataset('group_outer_indices', data=boundary_outer)
            f.create_dataset('group_boundary_indptr', data=boundary_indptr)
            f.create_dataset('group_vertices', data=boundary_vertices)
            f.create_dataset('group_vertex_indptr', data=vertex_indptr)
        f.attrs['nside'] = np.int32(nside)
        f.attrs['npix'] = np.int32(group.size)
        f.attrs['n_groups'] = np.int32(n_pixels.size)
        # f.attrs['format'] = b'group indices stored as CSR-style indices/indptr'
        # f.attrs['n_pixels_sigma_note'] = (
        #     b'fractional brightest-pixel counts for flux_sigma_targets')
        # f.attrs['flux_sigma_targets_note'] = (
        #     b'top 1, 2, and 3 sigma percentile ranges')
        # if boundary_indptr is not None:
        #     f.attrs['boundary_format'] = (
        #         b'group boundary inner/outer indices stored as CSR-style arrays')
        #     f.attrs['boundary_order'] = (
        #         b'clockwise face walk in neib_n,neib_e,neib_s,neib_w order')
        #     f.attrs['group_vertices_coord'] = b'theta_phi_radians'
        #     f.attrs['group_vertices_note'] = (
        #         b'closed boundary polylines; NaN rows separate multiple loops')

    if VERBOSE:
        print(f'Segmented groups saved to {seg_file}')
        print(f'n_pixels_sigma min/max by target: '
              f'{np.min(n_pixels_sigma, axis=0)} / '
              f'{np.max(n_pixels_sigma, axis=0)}')
        if boundary_indptr is not None:
            lengths = np.diff(boundary_indptr)
            print(f'Boundary segments saved: min/max/total = '
                  f'{np.min(lengths)} / {np.max(lengths)} / {boundary_indptr[-1]}')
            vertex_lengths = np.diff(vertex_indptr)
            print(f'Boundary vertices saved: min/max/total = '
                  f'{np.min(vertex_lengths)} / {np.max(vertex_lengths)} / '
                  f'{vertex_indptr[-1]}')
    return group, np.array([np.array(sorted(values), dtype=np.int32)
                            for values in group_indices], dtype=object), n_pixels, flux, max_flux

def read_segmented_groups(seg_file=seg_file_default, read_boundaries=False,
                          read_sigma=False):
    """Read watershed segmented groups from HDF5."""
    with h5py.File(seg_file, 'r') as f:
        group = f['group'][:].astype(np.int32)
        n_pixels = f['n_pixels'][:].astype(np.int32)
        flux = f['flux'][:].astype(np.float64)
        if 'max_flux' in f:
            max_flux = f['max_flux'][:].astype(np.float64)
        else:
            max_flux = np.full(flux.size, np.nan, dtype=np.float64)
        indices = f['group_indices'][:].astype(np.int32)
        indptr = f['group_indptr'][:].astype(np.int32)

        group_indices = np.empty(indptr.size-1, dtype=object)
        for group_id in range(group_indices.size):
            start, stop = indptr[group_id], indptr[group_id+1]
            group_indices[group_id] = indices[start:stop].astype(np.int32, copy=True)

        assert f.attrs['npix'] == group.size
        assert f.attrs['n_groups'] == group_indices.size
        assert max_flux.size == group_indices.size

        if read_sigma:
            if 'n_pixels_sigma' in f:
                n_pixels_sigma = f['n_pixels_sigma'][:].astype(np.float64)
                flux_sigma_targets = f['flux_sigma_targets'][:].astype(
                    np.float64)
            else:
                n_pixels_sigma = np.empty((group_indices.size, 0),
                                          dtype=np.float64)
                flux_sigma_targets = np.array([], dtype=np.float64)

        if read_boundaries:
            group_inner_indices = np.empty(group_indices.size, dtype=object)
            group_outer_indices = np.empty(group_indices.size, dtype=object)
            group_vertices = np.empty(group_indices.size, dtype=object)
            if 'group_inner_indices' in f:
                boundary_inner = f['group_inner_indices'][:].astype(np.int32)
                boundary_outer = f['group_outer_indices'][:].astype(np.int32)
                boundary_indptr = f['group_boundary_indptr'][:].astype(np.int32)
                if 'group_vertices' in f:
                    boundary_vertices = f['group_vertices'][:].astype(np.float64)
                    vertex_indptr = f['group_vertex_indptr'][:].astype(np.int32)
                else:
                    boundary_vertices = None
                    vertex_indptr = None
                for group_id in range(group_indices.size):
                    start = boundary_indptr[group_id]
                    stop = boundary_indptr[group_id+1]
                    group_inner_indices[group_id] = boundary_inner[
                        start:stop].astype(np.int32, copy=True)
                    group_outer_indices[group_id] = boundary_outer[
                        start:stop].astype(np.int32, copy=True)
                    if boundary_vertices is not None:
                        start = vertex_indptr[group_id]
                        stop = vertex_indptr[group_id+1]
                        group_vertices[group_id] = boundary_vertices[
                            start:stop].astype(np.float64, copy=True)
                    else:
                        group_vertices[group_id] = np.empty((0, 2),
                                                            dtype=np.float64)
            else:
                for group_id in range(group_indices.size):
                    group_inner_indices[group_id] = np.array([], dtype=np.int32)
                    group_outer_indices[group_id] = np.array([], dtype=np.int32)
                    group_vertices[group_id] = np.empty((0, 2), dtype=np.float64)

    if VERBOSE:
        print(f'Segmented groups loaded from {seg_file}')
    result = [group, group_indices, n_pixels, flux, max_flux]
    if read_boundaries:
        result += [group_inner_indices, group_outer_indices, group_vertices]
    if read_sigma:
        result += [n_pixels_sigma, flux_sigma_targets]
    return tuple(result)

def get_group_arrays(group, map):
    """Return group_indices, n_pixels, flux, and max_flux for a group map."""
    n_groups = int(np.max(group)) + 1
    group_indices = np.empty(n_groups, dtype=object)
    n_pixels = np.zeros(n_groups, dtype=np.int32)
    flux = np.zeros(n_groups, dtype=np.float64)
    max_flux = np.zeros(n_groups, dtype=np.float64)

    for group_id in range(n_groups):
        indices = np.where(group == group_id)[0].astype(np.int32)
        group_indices[group_id] = indices
        n_pixels[group_id] = indices.size
        flux[group_id] = np.sum(map[indices])
        max_flux[group_id] = np.max(map[indices])

    return group_indices, n_pixels, flux, max_flux

def top_sigma_flux_targets():
    """Return top-tail fractions of the 1, 2, and 3 sigma percentile ranges."""
    sigma_ranges = np.array([
        percentiles[2] - percentiles[1],
        percentiles[4] - percentiles[3],
        percentiles[6] - percentiles[5],
    ], dtype=np.float64)
    return sigma_ranges / 100.

def group_values_array(indices):
    """Return group indices as a sorted int array."""
    if isinstance(indices, set):
        return np.array(sorted(indices), dtype=np.int32)
    return np.asarray(indices, dtype=np.int32)

def n_pixels_for_flux_targets(values, targets):
    """
    Return fractional pixel counts needed for top pixels to reach targets.
    Values are sorted brightest first, and the final pixel is linearly split.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()

    if values.size == 0:
        return np.full(targets.size, np.nan, dtype=np.float64)
    if np.any(targets <= 0.) or np.any(targets > 1.):
        raise ValueError('flux targets must be in (0, 1].')
    if np.any(values < 0.):
        raise ValueError('group flux values must be non-negative.')

    total = np.sum(values)
    if total <= 0. or not np.isfinite(total):
        return np.zeros(targets.size, dtype=np.float64)

    values_sorted = np.sort(values)[::-1]
    cumulative = np.cumsum(values_sorted)
    counts = np.zeros(targets.size, dtype=np.float64)

    for i_target, target in enumerate(targets):
        target_flux = target * total
        i = int(np.searchsorted(cumulative, target_flux, side='left'))
        if i >= values_sorted.size:
            counts[i_target] = float(values_sorted.size)
            continue

        previous_flux = 0. if i == 0 else cumulative[i-1]
        residual = target_flux - previous_flux
        if values_sorted[i] <= 0.:
            counts[i_target] = float(i)
        else:
            counts[i_target] = float(i) + residual / values_sorted[i]

    return counts

def alpha_values_for_flux_targets(values, targets=None, counts=None,
                                  alpha_levels=None):
    """
    Return per-pixel alpha values for nested top-flux sigma regions.
    Cutoff pixels are linearly blended between adjacent alpha levels.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return np.array([], dtype=np.float64)

    if counts is None:
        if targets is None:
            targets = top_sigma_flux_targets()
        counts = n_pixels_for_flux_targets(values, targets)
    counts = np.asarray(counts, dtype=np.float64).ravel()

    if alpha_levels is None:
        if counts.size == 3:
            alpha_levels = np.array([1., 2./3., 1./3., 0.],
                                    dtype=np.float64)
        else:
            alpha_levels = np.linspace(1., 0., counts.size + 1)
    else:
        alpha_levels = np.asarray(alpha_levels, dtype=np.float64).ravel()

    if alpha_levels.size != counts.size + 1:
        raise ValueError('alpha_levels must have one more value than counts.')

    n_values = values.size
    counts = np.clip(counts, 0., float(n_values))
    counts = np.maximum.accumulate(counts)
    bounds = np.concatenate([[0.], counts, [float(n_values)]])

    order = np.lexsort((np.arange(n_values), -values))
    sorted_alpha = np.zeros(n_values, dtype=np.float64)
    for i_rank in range(n_values):
        pixel_start = float(i_rank)
        pixel_stop = float(i_rank + 1)
        for i_level, alpha in enumerate(alpha_levels):
            overlap = min(pixel_stop, bounds[i_level+1]) - max(
                pixel_start, bounds[i_level])
            if overlap > 0.:
                sorted_alpha[i_rank] += overlap * alpha

    alpha = np.zeros(n_values, dtype=np.float64)
    alpha[order] = sorted_alpha
    return alpha

def get_group_flux_pixel_counts(group_indices, map, targets=None):
    """Return fractional pixel counts for top-tail group flux targets."""
    if targets is None:
        targets = top_sigma_flux_targets()
    targets = np.asarray(targets, dtype=np.float64)
    n_pixels_sigma = np.zeros((len(group_indices), targets.size),
                              dtype=np.float64)

    for group_id, indices in enumerate(group_indices):
        indices = group_values_array(indices)
        n_pixels_sigma[group_id] = n_pixels_for_flux_targets(
            map[indices], targets)

    return n_pixels_sigma, targets

def get_group_sigma_alpha(group_indices, map, targets=None,
                          n_pixels_sigma=None):
    """Return a HEALPix alpha map for per-group 1, 2, and 3 sigma regions."""
    if targets is None:
        targets = top_sigma_flux_targets()

    alpha = np.zeros_like(map, dtype=np.float64)
    if n_pixels_sigma is None:
        n_pixels_sigma, targets = get_group_flux_pixel_counts(
            group_indices, map, targets=targets)

    for group_id, indices in enumerate(group_indices):
        indices = group_values_array(indices)
        alpha[indices] = alpha_values_for_flux_targets(
            map[indices], targets=targets, counts=n_pixels_sigma[group_id])

    return alpha

def compact_group_labels(group):
    """Return group labels compacted to 0..n_groups-1."""
    labels, compact = np.unique(group, return_inverse=True)
    assert labels.size == np.max(compact) + 1
    return compact.astype(np.int32)

def get_group_boundary_fluxes(group, map, neighbors):
    """Return max adjoining average flux for each neighboring group pair."""
    max_boundary_flux = {}
    boundary_counts = {}
    for ipix in range(group.size):
        group_i = int(group[ipix])
        for neib in neighbors[ipix]:
            if ipix >= neib:
                continue
            group_j = int(group[neib])
            if group_i == group_j:
                continue

            pair = tuple(sorted((group_i, group_j)))
            boundary_flux = 0.5 * (map[ipix] + map[neib])
            if (pair not in max_boundary_flux or
                    boundary_flux > max_boundary_flux[pair]):
                max_boundary_flux[pair] = boundary_flux
            boundary_counts[pair] = boundary_counts.get(pair, 0) + 1

    return max_boundary_flux, boundary_counts

def merge_groups_by_persistence(persistence_min=0.05, nside=10,
                                segment_file=None, map_file=map_file_default,
                                seg_file=seg_file_default,
                                rel_persistence_min=0.15):
    """
    Merge neighboring watershed groups by topological persistence.
    persistence_min is in raw f_esc units, so 0.01 is one percentage point.
    After each merge, all current group boundaries are recomputed and
    reconsidered. The merge threshold for a neighboring pair is
    max(persistence_min, rel_persistence_min * min_peak), where min_peak is the
    smaller of the two group maxima. Group pairs are considered in descending
    max boundary flux, matching a superlevel-set filtration. This is
    deterministic and peak-centered rather than ordered by basin area or
    summed flux.
    """
    group, group_indices, n_pixels, flux, max_flux = read_segmented_groups(seg_file)
    neib_n, neib_e, neib_s, neib_w = load_pixel_segments(nside, segment_file)
    neighbors = np.vstack([neib_n, neib_e, neib_s, neib_w]).T

    with h5py.File(map_file, 'r') as f:
        map = f['map'][:]

    assert group.size == map.size == hp.nside2npix(nside)
    n_groups = len(group_indices)

    if np.any(~np.isfinite(max_flux)):
        max_flux = np.array([np.max(map[indices]) for indices in group_indices],
                            dtype=np.float64)
    if rel_persistence_min is not None and rel_persistence_min < 0.:
        raise ValueError('rel_persistence_min must be non-negative or None.')

    current_group = compact_group_labels(group)
    merge_records = []
    initial_persistence_values = None
    initial_threshold_values = None
    initial_boundary_counts = None
    final_boundary_counts = None

    def better_group(group_a, group_b, group_n_pixels, group_flux,
                     group_max_flux):
        key_a = (group_max_flux[group_a], group_n_pixels[group_a],
                 group_flux[group_a], -group_a)
        key_b = (group_max_flux[group_b], group_n_pixels[group_b],
                 group_flux[group_b], -group_b)
        return group_a if key_a >= key_b else group_b

    def merge_threshold(min_peak):
        threshold = persistence_min
        if rel_persistence_min is not None and rel_persistence_min > 0.:
            threshold = max(threshold, rel_persistence_min * min_peak)
        return threshold

    while True:
        (current_group_indices, current_n_pixels, current_flux,
         current_max_flux) = get_group_arrays(current_group, map)
        max_boundary_flux, boundary_counts = get_group_boundary_fluxes(
            current_group, map, neighbors)
        persistence_values = []
        threshold_values = []
        for (i, j), boundary_flux in max_boundary_flux.items():
            min_peak = min(current_max_flux[i], current_max_flux[j])
            persistence_values.append(min_peak - boundary_flux)
            threshold_values.append(merge_threshold(min_peak))
        persistence_values = np.array(persistence_values)
        threshold_values = np.array(threshold_values)
        if initial_persistence_values is None:
            initial_persistence_values = persistence_values
            initial_threshold_values = threshold_values
            initial_boundary_counts = boundary_counts

        merge = None
        for (group_i, group_j), boundary_flux in sorted(
                max_boundary_flux.items(), key=lambda item: (-item[1], item[0])):
            min_peak = min(current_max_flux[group_i], current_max_flux[group_j])
            persistence = min_peak - boundary_flux
            threshold = merge_threshold(min_peak)
            if persistence < threshold:
                high_group = better_group(group_i, group_j, current_n_pixels,
                                          current_flux, current_max_flux)
                low_group = group_j if high_group == group_i else group_i
                merge = (low_group, high_group, boundary_flux, persistence,
                         threshold, len(current_group_indices))
                break

        if merge is None:
            final_boundary_counts = boundary_counts
            break

        (low_group, high_group, boundary_flux, persistence, threshold,
         n_groups_before) = merge
        current_group[current_group == low_group] = high_group
        current_group = compact_group_labels(current_group)
        merge_records.append((low_group, high_group, boundary_flux,
                              persistence, threshold, n_groups_before))

    merged_group = current_group
    merged_group_indices, merged_n_pixels, merged_flux, merged_max_flux = (
        get_group_arrays(merged_group, map))

    if VERBOSE:
        print('Persistence merge:')
        print(f'persistence_min={persistence_min:g} '
              f'({100.*persistence_min:g} percentage points)')
        if rel_persistence_min is not None and rel_persistence_min > 0.:
            print(f'rel_persistence_min={rel_persistence_min:g} '
                  f'({100.*rel_persistence_min:g}% of smaller peak)')
        if initial_persistence_values.size > 0:
            p10, p50, p90 = np.percentile(
                initial_persistence_values, [10., 50., 90.])
            t10, t50, t90 = np.percentile(
                initial_threshold_values, [10., 50., 90.])
            counts = np.array(list(initial_boundary_counts.values()),
                              dtype=np.int32)
            print(f'initial boundary pairs={len(initial_boundary_counts)}, '
                  f'boundary pixels min/max={np.min(counts)}/{np.max(counts)}')
            print(f'initial persistence min/p10/median/p90/max = '
                  f'{np.min(initial_persistence_values):g} / {p10:g} / '
                  f'{p50:g} / {p90:g} / '
                  f'{np.max(initial_persistence_values):g}')
            print(f'initial threshold min/p10/median/p90/max = '
                  f'{np.min(initial_threshold_values):g} / {t10:g} / '
                  f'{t50:g} / {t90:g} / '
                  f'{np.max(initial_threshold_values):g}')
        if final_boundary_counts is not None:
            print(f'final boundary pairs={len(final_boundary_counts)}')
        print(f'persistence merges={len(merge_records)}, '
              f'merge passes={len(merge_records)+1}, groups={n_groups} -> '
              f'{len(merged_group_indices)}')
        print(f'merged group size min/max/mean/median: '
              f'{np.min(merged_n_pixels)} / {np.max(merged_n_pixels)} / '
              f'{np.mean(merged_n_pixels):g} / {np.median(merged_n_pixels):g}')
        print(f'merged max_flux min/max: {np.min(merged_max_flux):g} / '
              f'{np.max(merged_max_flux):g}')

    return merged_group, merged_group_indices, merged_n_pixels, merged_flux, merged_max_flux

def write_persistent_segmented_groups(seg_file=seg_file_default,
                                      persistence_min=0.05, nside=10,
                                      segment_file=None,
                                      map_file=map_file_default,
                                      source_seg_file=seg_file_default,
                                      write_boundaries=True,
                                      rel_persistence_min=0.15):
    """Merge watershed groups and write the result to HDF5."""
    group, group_indices, n_pixels, flux, max_flux = merge_groups_by_persistence(
        persistence_min=persistence_min, nside=nside,
        segment_file=segment_file, map_file=map_file,
        seg_file=source_seg_file, rel_persistence_min=rel_persistence_min)
    return write_segmented_groups(seg_file=seg_file, nside=nside,
                                  segment_file=segment_file,
                                  map_file=map_file, group=group,
                                  group_indices=group_indices,
                                  n_pixels=n_pixels, flux=flux,
                                  max_flux=max_flux,
                                  write_boundaries=write_boundaries)

def default_unmerged_seg_file(seg_file):
    """Return the default companion filename for pre-merge groups."""
    seg_file = Path(seg_file)
    return seg_file.with_name(f'{seg_file.stem}_unmerged{seg_file.suffix}')

def percentile_string(data, n_format=1):
    """Return median and 68-percent range in the plot-maps.py text style."""
    lo, med, hi = np.percentile(data, percentiles_68)
    fmt = f'%0.{n_format}f'
    return r'$' + fmt % med + r'^{+' + fmt % (hi-med) + r'}_{-' + fmt % (med-lo) + r'}$'

def print_map_limits(label, data, lims):
    """Print map statistics and colorbar limits."""
    lo, med, hi = np.percentile(data, percentiles_68)
    print(f'{label}: {med:g}  [{lo:g}, {hi:g}]  '
          f'Avg/Min/Max: {np.mean(data):g}  [{np.min(data):g}, {np.max(data):g}]')
    print(f'{label} lims: [{lims[0]:g}, {lims[1]:g}]')

def _smoothed_max_rot_from_map(map, smooth_pix=4.):
    """Return a Healpy rotation centering the smoothed map maximum."""
    nside = hp.npix2nside(map.size)
    values = np.asarray(map, dtype=np.float64)
    bad = (~np.isfinite(values)) | (values == hp.UNSEEN)
    masked_map = hp.ma(values)
    masked_map.mask = bad
    fwhm = float(smooth_pix) * hp.nside2resol(nside)
    smoothed_map = hp.smoothing(masked_map, fwhm=fwhm)
    ipix = int(np.ma.argmax(smoothed_map))
    theta, phi = hp.pix2ang(nside, ipix)
    lon = np.degrees(phi)
    lat = 90. - np.degrees(theta)
    return (lon, lat, 0.)

def finite_vertex_chunks(vertices):
    """Yield finite vertex chunks, using NaN rows as loop separators."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.size == 0:
        return

    finite = np.all(np.isfinite(vertices), axis=1)
    start = None
    for i, is_finite in enumerate(finite):
        if is_finite and start is None:
            start = i
        if start is not None and ((not is_finite) or i == finite.size-1):
            stop = i if not is_finite else i+1
            if stop - start >= 2:
                yield vertices[start:stop]
            start = None

def close_vertex_chunk(vertices):
    """Return vertices with the first point explicitly repeated at the end."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape[0] < 2:
        return vertices
    if np.allclose(vertices[0], vertices[-1], rtol=0., atol=1.e-12):
        return vertices
    return np.vstack([vertices, vertices[0]])

def rotated_projplot(ax, theta, phi, rot=None, **kwargs):
    """Plot theta/phi coordinates with the same rotation convention as projmap."""
    vec = R.dir2vec(theta, phi, lonlat=False)
    if rot is not None:
        vec = R.Rotator(rot=rot)(vec)
    x, y = ax.proj.vec2xy(vec, direct=False)
    x, y = ax._make_segment(
        x, y, threshold=kwargs.pop('threshold', ax._segment_threshold))
    for xx, yy in zip(x, y):
        ax.plot(xx, yy, **kwargs)

def plot_group_vertices(ax, group_vertices, color='white', linewidth=0.35,
                        rot=None):
    """Overlay group boundary vertices on a HEALPix projection axis."""
    if group_vertices is None:
        return

    for vertices in group_vertices:
        for chunk in finite_vertex_chunks(vertices):
            chunk = close_vertex_chunk(chunk)
            rotated_projplot(ax, chunk[:, 0], chunk[:, 1], rot=rot,
                             color=color, linewidth=linewidth, alpha=0.95,
                             zorder=20, solid_capstyle='round',
                             solid_joinstyle='round')

def HpPlot(f, extent, map, u_str=None, w_str=None, lims=None, cmap=None,
           n_format=0, group_vertices=None, rot=None):
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
        img = ax.projmap(map, cmap=cmap, rot=rot)
    else:
        img = ax.projmap(map, cmap=cmap, rot=rot,
                         vmin=lims[0], vmax=lims[1])

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
    plot_group_vertices(ax, group_vertices, rot=rot)
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

def HpGroupPlot(f, extent, group, n_groups, u_str=None, group_vertices=None,
                rot=None):
    """Plot a segmented HEALPix group map with one discrete color per group."""
    from healpy import projaxes as PA
    from healpy import pixelfunc

    if u_str is None:
        u_str = r'${\rm %d\ Groups}$' % n_groups

    group = pixelfunc.ma_to_array(group)
    cmap = discrete_group_cmap(n_groups)
    ax = PA.HpxMollweideAxes(f, extent)
    f.add_axes(ax)
    ax.projmap(group, cmap=cmap, rot=rot, vmin=-0.5, vmax=n_groups-0.5)
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
    plot_group_vertices(ax, group_vertices, rot=rot)
    f.sca(ax)

def HpGroupSigmaPlot(f, extent, group, n_groups, group_indices, map,
                     u_str=None, group_vertices=None, n_pixels_sigma=None,
                     flux_sigma_targets=None, rot=None):
    """Plot segmented groups with sigma-region alpha transparency."""
    from healpy import projaxes as PA
    from healpy import pixelfunc

    if u_str is None:
        u_str = r'${\rm %d\ Groups}$' % n_groups

    group = pixelfunc.ma_to_array(group)
    if flux_sigma_targets is None:
        flux_sigma_targets = top_sigma_flux_targets()
    alpha = get_group_sigma_alpha(group_indices, map,
                                  targets=flux_sigma_targets,
                                  n_pixels_sigma=n_pixels_sigma)

    cmap = discrete_group_cmap(n_groups)
    ax = PA.HpxMollweideAxes(f, extent)
    f.add_axes(ax)
    ax.projmap(np.zeros_like(group, dtype=np.float64), cmap='gray',
               rot=rot, vmin=0., vmax=1.)
    ax.projmap(group, alpha=alpha, cmap=cmap, rot=rot,
               vmin=-0.5, vmax=n_groups-0.5)
    im = ax.get_images()[-1]
    boundaries = np.arange(n_groups+1) - 0.5
    values = np.arange(n_groups)
    cb = f.colorbar(im, ax=ax, orientation='horizontal',
                    shrink=0.75, aspect=25, ticks=[],
                    pad=0.05, fraction=0.1,
                    boundaries=boundaries, values=values)
    cb.solids.set_rasterized(True)
    cb.ax.text(0.5, -2.0, u_str, fontsize=14.5,
               transform=cb.ax.transAxes, ha='center', va='center')
    plot_group_vertices(ax, group_vertices, rot=rot)
    f.sca(ax)

def test_plot_neighbor_differences(nside=10, segment_file=None,
                                   map_file=map_file_default,
                                   save_file=None, seg_file=None,
                                   unmerged_seg_file=None,
                                   write_unmerged=False,
                                   recenter_on_smoothed_max=True,
                                   smooth_pix=4.):
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

    rot = None
    if recenter_on_smoothed_max:
        rot = _smoothed_max_rot_from_map(map, smooth_pix=smooth_pix)
        if VERBOSE:
            print(f'Plot rotation from smoothed f_esc maximum: {rot}')

    group = None
    group_indices = None
    group_vertices = None
    n_pixels_sigma = None
    flux_sigma_targets = None
    group_unmerged = None
    group_indices_unmerged = None
    n_pixels_sigma_unmerged = None
    flux_sigma_targets_unmerged = None
    if seg_file is not None:
        (group, group_indices, n_pixels, flux, max_flux,
         group_inner_indices, group_outer_indices, group_vertices,
         n_pixels_sigma, flux_sigma_targets) = read_segmented_groups(
            seg_file, read_boundaries=True, read_sigma=True)
        assert group.size == map.size, (
            f'group size {group.size} does not match map size {map.size}')
        if np.sum([len(vertices) for vertices in group_vertices]) == 0:
            group_vertices = None
        if n_pixels_sigma.size == 0:
            n_pixels_sigma = None
            flux_sigma_targets = None
        if VERBOSE:
            print(f'Segmented groups: n_groups={len(group_indices)}, '
                  f'n_pixels min/max={np.min(n_pixels)}/{np.max(n_pixels)}, '
                  f'flux sum={np.sum(flux):g}, '
                  f'max_flux min/max={np.nanmin(max_flux):g}/{np.nanmax(max_flux):g}')
            if group_vertices is not None:
                n_vertex_rows = np.sum([
                    len(vertices) for vertices in group_vertices])
                print(f'Boundary vertex rows={n_vertex_rows}')

        if unmerged_seg_file is None:
            unmerged_seg_file = default_unmerged_seg_file(seg_file)
        if write_unmerged:
            write_segmented_groups(seg_file=unmerged_seg_file, nside=nside,
                                   segment_file=segment_file,
                                   map_file=map_file)
        if Path(unmerged_seg_file).exists():
            (group_unmerged, group_indices_unmerged, n_pixels_unmerged,
             flux_unmerged, max_flux_unmerged, n_pixels_sigma_unmerged,
             flux_sigma_targets_unmerged) = read_segmented_groups(
                unmerged_seg_file, read_sigma=True)
            assert group_unmerged.size == map.size, (
                f'unmerged group size {group_unmerged.size} does not match map size {map.size}')
            if n_pixels_sigma_unmerged.size == 0:
                n_pixels_sigma_unmerged = None
                flux_sigma_targets_unmerged = None
            if VERBOSE:
                print(f'Unmerged segmented groups: n_groups={len(group_indices_unmerged)}, '
                      f'n_pixels min/max={np.min(n_pixels_unmerged)}/{np.max(n_pixels_unmerged)}, '
                      f'flux sum={np.sum(flux_unmerged):g}, '
                      f'max_flux min/max={np.nanmin(max_flux_unmerged):g}/{np.nanmax(max_flux_unmerged):g}')

    fig = plt.figure(figsize=(3., 2.))
    dy_map = 1.045
    HpPlot(fig, (0, dy_map, 1, 1), fesc,
           u_str=r'$f_{\rm esc}^{\rm\,LyC}\ \ (\%)$',
           w_str=percentile_string(fesc, n_format=1),
           lims=fesc_lims, cmap=cmr.ember, n_format=0,
           group_vertices=group_vertices, rot=rot)
    HpPlot(fig, (0, 0, 1, 1), delta_fesc,
           u_str=r'$\Delta f_{\rm esc}^{\rm\,LyC}\ \ (\%)$',
           w_str=percentile_string(delta_fesc, n_format=1),
           lims=delta_lims, cmap=cmr.amber, n_format=0,
           group_vertices=group_vertices, rot=rot)
    if group is not None:
        if group_unmerged is not None:
            HpGroupSigmaPlot(
                fig, (1.01, dy_map, 1, 1), group_unmerged,
                len(group_indices_unmerged), group_indices_unmerged, map,
                u_str=r'$%d\ {\rm Groups}\ ({\rm pre\!-\!merge})$'
                % len(group_indices_unmerged),
                group_vertices=group_vertices,
                n_pixels_sigma=n_pixels_sigma_unmerged,
                flux_sigma_targets=flux_sigma_targets_unmerged,
                rot=rot)
            HpGroupSigmaPlot(
                fig, (1.01, 0, 1, 1), group, len(group_indices),
                group_indices, map,
                u_str=r'$%d\ {\rm Groups}\ ({\rm post\!-\!merge})$'
                % len(group_indices),
                group_vertices=group_vertices,
                n_pixels_sigma=n_pixels_sigma,
                flux_sigma_targets=flux_sigma_targets,
                rot=rot)
        else:
            HpGroupSigmaPlot(fig, (1.01, dy_map, 1, 1), group,
                             len(group_indices), group_indices, map,
                             group_vertices=group_vertices,
                             n_pixels_sigma=n_pixels_sigma,
                             flux_sigma_targets=flux_sigma_targets,
                             rot=rot)

    sargs = {'bbox_inches':'tight', 'pad_inches':0.,
             'transparent':False, 'dpi':640}
    plt.savefig(save_file, **sargs)
    plt.close(fig)
    if VERBOSE:
        print(f'Neighbor difference plot saved to {save_file}')
    if group is None:
        return fesc, delta_fesc
    if group_unmerged is not None:
        return fesc, delta_fesc, group_unmerged, group
    return fesc, delta_fesc, group

if __name__ == '__main__':
    # create_pixel_segments()
    unmerged_seg_file = default_unmerged_seg_file(seg_file_default)
    write_segmented_groups(seg_file=unmerged_seg_file)
    write_persistent_segmented_groups(source_seg_file=unmerged_seg_file)
    test_plot_neighbor_differences(seg_file=seg_file_default,
                                   unmerged_seg_file=unmerged_seg_file)
