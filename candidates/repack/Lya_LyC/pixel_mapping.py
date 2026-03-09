import numpy as np
import h5py
import healpy as hp

def create_pixel_mapping(nside_low=5, nside_high=10):
    """
    Create mapping between low and high resolution HEALPix pixels.
    For each low-res pixel, find the 4 nearest high-res pixels using angular distances.
    """
    save_file = f'nside_{nside_low}_{nside_high}.hdf5'
    npix_high = 12 * nside_high * nside_high
    npix_low = 12 * nside_low * nside_low

    print(f"Creating mapping: {npix_low} low-res -> {npix_high} high-res pixels")

    # Get pixel centers for both resolutions
    theta_high, phi_high = hp.pix2ang(nside_high, np.arange(npix_high))
    theta_low, phi_low = hp.pix2ang(nside_low, np.arange(npix_low))

    # Convert to Cartesian coordinates for distance calculations
    x_high = np.sin(theta_high) * np.cos(phi_high)
    y_high = np.sin(theta_high) * np.sin(phi_high)
    z_high = np.cos(theta_high)

    x_low = np.sin(theta_low) * np.cos(phi_low)
    y_low = np.sin(theta_low) * np.sin(phi_low)
    z_low = np.cos(theta_low)

    # For each low-res pixel, find 4 nearest high-res pixels
    sub_0 = np.zeros(npix_low, dtype=np.int32)
    sub_1 = np.zeros(npix_low, dtype=np.int32)
    sub_2 = np.zeros(npix_low, dtype=np.int32)
    sub_3 = np.zeros(npix_low, dtype=np.int32)

    # Inverse mapping: for each high-res pixel, which low-res pixel does it belong to?
    inverse_map = np.zeros(npix_high, dtype=np.int32)

    for i in range(npix_low):
        # Calculate dot products (cosine of angular distance)
        dot_products = (x_high * x_low[i] + y_high * y_low[i] + z_high * z_low[i])

        # Find 4 nearest pixels (largest dot products = smallest angular distances)
        nearest_indices = np.argsort(-dot_products)[:4]  # negative for descending sort

        # Sort the indices to ensure sub_0, sub_1, sub_2, sub_3 are ordered by high-res index
        nearest_indices = np.sort(nearest_indices)

        sub_0[i] = nearest_indices[0]
        sub_1[i] = nearest_indices[1]
        sub_2[i] = nearest_indices[2]
        sub_3[i] = nearest_indices[3]

        # Set inverse mapping
        for idx in nearest_indices:
            inverse_map[idx] = i

    # Check uniqueness
    all_indices = np.concatenate([sub_0, sub_1, sub_2, sub_3])
    unique_indices = np.unique(all_indices)
    print(f"Unique high-res pixels found: {len(unique_indices)} (should be {npix_high})")

    if len(unique_indices) != npix_high:
        print("WARNING: Not all high-res pixels are covered!")
        missing = set(range(npix_high)) - set(unique_indices)
        print(f"Missing pixels: {sorted(list(missing))[:10]}...")  # Show first 10

    # Save to HDF5
    with h5py.File(save_file, 'w') as f:
        f.create_dataset('sub_0', data=sub_0)
        f.create_dataset('sub_1', data=sub_1)
        f.create_dataset('sub_2', data=sub_2)
        f.create_dataset('sub_3', data=sub_3)
        f.create_dataset('inverse_map', data=inverse_map)
        f.attrs['nside_high'] = nside_high
        f.attrs['nside_low'] = nside_low
        f.attrs['npix_high'] = npix_high
        f.attrs['npix_low'] = npix_low

    print(f"Mapping saved to {save_file}")
    return sub_0, sub_1, sub_2, sub_3, inverse_map

def load_pixel_mapping(nside_low, nside_high):
    """Load pre-computed pixel mapping from HDF5 file."""
    filename = f'nside_{nside_low}_{nside_high}.hdf5'
    with h5py.File(filename, 'r') as f:
        sub_0 = f['sub_0'][:]
        sub_1 = f['sub_1'][:]
        sub_2 = f['sub_2'][:]
        sub_3 = f['sub_3'][:]
        inverse_map = f['inverse_map'][:]
        assert f.attrs['nside_high'] == nside_high
        assert f.attrs['nside_low'] == nside_low
    return sub_0, sub_1, sub_2, sub_3, inverse_map

def degrade_healpix_custom(map_in, target_npix):
    """
    Degrade HEALPix map using pre-computed pixel mapping.
    """
    nside_in = hp.npix2nside(len(map_in))
    nside_out = hp.npix2nside(target_npix)
    try:
        sub_0, sub_1, sub_2, sub_3, inverse_map = load_pixel_mapping(nside_out, nside_in)
    except FileNotFoundError:
        print(f"Mapping file nside_{nside_out}_{nside_in}.hdf5 not found. Creating new mapping...")
        sub_0, sub_1, sub_2, sub_3, inverse_map = create_pixel_mapping(nside_out, nside_in)

    # Average the 4 high-res pixels for each low-res pixel
    return (map_in[sub_0] + map_in[sub_1] + map_in[sub_2] + map_in[sub_3]) / 4.

def test_mapping(nside_low=5, nside_high=10):
    """Test the pixel mapping by creating test maps and verifying degradation."""
    print("Testing pixel mapping...")

    # Create mapping
    sub_0, sub_1, sub_2, sub_3, inverse_map = create_pixel_mapping(nside_low, nside_high)

    # Create a test high-resolution map with unique values for each pixel
    npix_high = 1200
    test_map_high = np.arange(npix_high, dtype=float)

    # Degrade using our mapping
    test_map_low = (test_map_high[sub_0] + test_map_high[sub_1] +
                   test_map_high[sub_2] + test_map_high[sub_3]) / 4.0

    # Check that each low-res pixel is the average of 4 unique high-res pixels
    duplicates_found = 0
    for i in range(300):
        indices = [sub_0[i], sub_1[i], sub_2[i], sub_3[i]]
        unique_check = len(set(indices)) == 4
        if not unique_check:
            duplicates_found += 1
            if duplicates_found <= 5:  # Only show first 5 warnings
                print(f"Warning: Low-res pixel {i} has duplicate high-res pixels: {indices}")

    if duplicates_found > 0:
        print(f"Total pixels with duplicates: {duplicates_found}")
    else:
        print("All low-res pixels have 4 unique high-res pixels")

    print(f"Test complete. Low-res map shape: {test_map_low.shape}")
    print(f"Sample low-res values: {test_map_low[:5]}")

    return test_map_low

def visualize_mapping(nside_low=5, nside_high=10):
    """Create visualizations to verify the pixel mapping works correctly."""
    import matplotlib.pyplot as plt

    print("Creating visualization of pixel mapping...")

    # Load mapping
    sub_0, sub_1, sub_2, sub_3, inverse_map = load_pixel_mapping(nside_low, nside_high)

    # Create test maps
    npix_high = 12 * nside_high * nside_high
    npix_low = 12 * nside_low * nside_low

    # Test 1: Gradient map
    test_map_high = np.arange(npix_high, dtype=float)
    test_map_low = (test_map_high[sub_0] + test_map_high[sub_1] +
                   test_map_high[sub_2] + test_map_high[sub_3]) / 4.

    # Test 2: Checkerboard pattern
    checkerboard_high = np.zeros(npix_high)
    for i in range(npix_high):
        checkerboard_high[i] = (i % 2) * 100 + 50
    checkerboard_low = (checkerboard_high[sub_0] + checkerboard_high[sub_1] +
                       checkerboard_high[sub_2] + checkerboard_high[sub_3]) / 4.

    # Create individual plots to avoid subplot issues
    plt.figure(figsize=(8, 6))
    hp.mollview(test_map_high, nest=False, title=f'High-res Gradient (nside={nside_high})',
                min=np.min(test_map_high), max=np.max(test_map_high))
    plt.savefig('high_res_gradient.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    hp.mollview(test_map_low, nest=False, title=f'Low-res Gradient (nside={nside_low})',
                min=np.min(test_map_low), max=np.max(test_map_low))
    plt.savefig('low_res_gradient.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    hp.mollview(checkerboard_high, nest=False, title=f'High-res Checkerboard (nside={nside_high})',
                min=np.min(checkerboard_high), max=np.max(checkerboard_high))
    plt.savefig('high_res_checkerboard.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    hp.mollview(checkerboard_low, nest=False, title=f'Low-res Checkerboard (nside={nside_low})',
                min=np.min(checkerboard_low), max=np.max(checkerboard_low))
    plt.savefig('low_res_checkerboard.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Individual visualizations saved:")
    print("- high_res_gradient.png")
    print("- low_res_gradient.png")
    print("- high_res_checkerboard.png")
    print("- low_res_checkerboard.png")

    # Test inverse mapping
    reconstructed_high = np.zeros(npix_high)
    for i in range(npix_low):
        for idx in [sub_0[i], sub_1[i], sub_2[i], sub_3[i]]:
            reconstructed_high[idx] = test_map_low[i]

    # Calculate reconstruction error
    error = np.abs(test_map_high - reconstructed_high)
    max_error = np.max(error)
    mean_error = np.mean(error)
    print(f"Reconstruction error - Max: {max_error:.2f}, Mean: {mean_error:.2f}")

    return test_map_low, checkerboard_low

if __name__ == "__main__":
    test_mapping()
    visualize_mapping()
