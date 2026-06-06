import numpy as np
import h5py
import healpy as hp

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
        print(f"{name}: min={segment.min()}, max={segment.max()}, missing={missing}")

def create_pixel_segments(nside=10, save_file=None, nest=False):
    """
    Create HEALPix face-neighbor segments.
    For each pixel, save the four side-sharing neighbors in clockwise order.
    """
    if save_file is None:
        save_file = f'nside_{nside}_segments.hdf5'

    npix = hp.nside2npix(nside)
    print(f"Creating face-neighbor segments for {npix} pixels (nside={nside})")

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
    print_segment_ranges(neib_n, neib_e, neib_s, neib_w)
    print(f"Pixels with duplicate face neighbors: {duplicate_rows}")

    # Save to HDF5
    with h5py.File(save_file, 'w') as f:
        f.create_dataset('neib_n', data=neib_n)
        f.create_dataset('neib_e', data=neib_e)
        f.create_dataset('neib_s', data=neib_s)
        f.create_dataset('neib_w', data=neib_w)
        f.attrs['nside'] = nside
        f.attrs['npix'] = npix
        f.attrs['nest'] = nest
        f.attrs['neighbor_order'] = 'neib_n,neib_e,neib_s,neib_w'
        f.attrs['ordering_note'] = 'side-sharing neighbors, clockwise from local north'
        f.attrs['construction'] = 'pixel boundary midpoint crossed with healpy.vec2pix'

    print(f"Pixel segments saved to {save_file}")
    return neib_n, neib_e, neib_s, neib_w

def load_pixel_segments(nside=10, filename=None):
    """Load pre-computed HEALPix face-neighbor segments from HDF5 file."""
    if filename is None:
        filename = f'nside_{nside}_segments.hdf5'

    with h5py.File(filename, 'r') as f:
        neib_n = f['neib_n'][:]
        neib_e = f['neib_e'][:]
        neib_s = f['neib_s'][:]
        neib_w = f['neib_w'][:]
        assert f.attrs['nside'] == nside
    return neib_n, neib_e, neib_s, neib_w

if __name__ == "__main__":
    create_pixel_segments()
