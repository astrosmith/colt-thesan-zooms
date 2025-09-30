import numpy as np
import h5py, sys

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

def repack_maps(snaps, base='ion-eq', out_dir='.'):
    n_snaps = len(snaps)
    with h5py.File(f'{out_dir}/{base}_{snaps[0]:03d}.hdf5', 'r') as f:
        n_side_map = f.attrs['n_side_map']
        n_pix = 12 * n_side_map**2
    zs = np.empty(n_snaps)
    maps = np.empty((n_snaps, n_pix))
    for i in progressbar(range(n_snaps)):
        with h5py.File(f'{out_dir}/{base}_{snaps[i]:03d}.hdf5', 'r') as f:
            zs[i] = f.attrs['z']
            maps[i,:] = f['map'][:]
    with h5py.File(f'{out_dir}/{base}_maps.hdf5', 'w') as f:
        f.attrs['n_snaps'] = n_snaps
        f.attrs['n_side_map'] = n_side_map
        f.attrs['n_pix'] = n_pix
        f.create_dataset('snaps', data=np.array(snaps, dtype=np.int32))
        f.create_dataset('zs', data=zs)
        f.create_dataset('maps', data=maps)

if __name__ == '__main__':
    colt_dir = '/orcd/data/mvogelsb/005/Lab/Thesan-Zooms-COLT'
    out_dir = f'{colt_dir}/g1163/z4/output_tree'
    for base in ['ion-eq', 'ion-eq-RHD']:
        repack_maps(snaps=range(8,188+1), base=base, out_dir=out_dir)
