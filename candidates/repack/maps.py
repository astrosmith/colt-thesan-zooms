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
    success = np.zeros(n_snaps, dtype=bool)
    n_pix = 1
    for i in range(n_snaps):
        try:
            with h5py.File(f'{out_dir}/{base}_{snaps[i]:03d}.hdf5', 'r') as f:
                n_side_map = f.attrs['n_side_map']
                n_pix = 12 * n_side_map**2
            break
        except:
            pass
    zs = np.empty(n_snaps)
    maps = np.empty((n_snaps, n_pix))
    for i in progressbar(range(n_snaps)):
        try:
            with h5py.File(f'{out_dir}/{base}_{snaps[i]:03d}.hdf5', 'r') as f:
                zs[i] = f.attrs['z']
                maps[i,:] = f['map'][:]
                success[i] = True
        except:
            pass
    snaps = np.array(snaps, dtype=np.int32)
    n_success = np.count_nonzero(success)
    if n_success < n_snaps:
        if n_success == 0:
            return
        # print(f'Only {n_success} out of {n_snaps} maps found for {out_dir} ...')
        n_snaps = np.int32(n_success)
        snaps = snaps[success]
        zs = zs[success]
        maps = maps[success,:]
    with h5py.File(f'{out_dir}/{base}_maps.hdf5', 'w') as f:
        f.attrs['n_snaps'] = n_snaps
        f.attrs['n_side_map'] = n_side_map
        f.attrs['n_pix'] = n_pix
        f.create_dataset('snaps', data=snaps)
        f.create_dataset('zs', data=zs)
        f.create_dataset('maps', data=maps)

if __name__ == '__main__':
    colt_dir = '/orcd/data/mvogelsb/005/Lab/Thesan-Zooms-COLT'
    sims = ['g2', 'g39', 'g205', 'g578', 'g1163', 'g5760', 'g10304', 'g137030', 'g500531', 'g519761', 'g2274036'][4:]
    runs = ['z4', 'z4', 'z4', 'z4', 'z4', 'z8', 'z8', 'z16', 'z16', 'z16'][4:]
    # , 'g5229300']
    # , 'z16']
    for sim, run in zip(sims, runs):
        out_dir = f'{colt_dir}/{sim}/{run}/output_tree'
        print(f'Processing {out_dir} ...')
        for base in ['ion-eq', 'ion-eq-RHD']:
            repack_maps(snaps=range(0,188+1), base=base, out_dir=out_dir)
