import h5py
from extract_halo import extract_halo
from combine_files import combine_files
from arepo_to_colt import arepo_to_colt

# Production groups
sims = {
  #'g39': ['z4'],
  #'g205': ['z4'],
  #'g578': ['z4'],
  #'g1163': ['z4'],
  #'g5760': ['noESF', 'noRlim', 'varEff', 'z4', 'z8'],
  #'g5760': ['z4'],
  #'g5760': ['z8'],
  'g10304': ['z4'],
  #'g33206': ['noESF', 'noESF_z8', 'noRlim', 'noRlim_z8', 'varEff', 'varEff_z8', 'z4', 'z8'],
  #'g33206': ['noESF', 'noRlim', 'varEff', 'z4', 'z8'],
  #'g37591': ['z4'],
  #'g137030': ['z4'],
  #'g500531': ['z4', 'z8'],
  #'g519761': ['z4'],
  #'g2274036': ['z4'],
  #'g5229300': ['z4', 'z8'],
}

#snaps = range(12, 188)
snaps = [188]
zooms_dir = '/net/hstor001.ib/data2/group/mvogelsb/004/Thesan-Zooms'
colt_dir = f'{zooms_dir}-COLT'

for group, runs in sims.items():
    for run in runs:
        file_dir = f'{zooms_dir}/{group}/{run}/output'
        out_dir = f'{colt_dir}/{group}/{run}'
        tree_dir = file_dir # out_dir # Permissions issue

        with h5py.File(f'{tree_dir}/tree.hdf5', 'r') as f:
            #snaps = f['Snapshots'][:] # Automated?
            n_files = f['Header'].attrs['NumFiles']

        # Extract halo, combine files, and convert to colt ics for each snapshot
        for snap in snaps:
            print(f'\nProgress for Group {group}, Run {run}: {snap} from {snaps}')
            for i_file in range(n_files):
                extract_halo(snap=snap, i_file=i_file, file_dir=file_dir, out_dir=out_dir, tree_dir=tree_dir)
            combine_files(snap=snap, n_files=n_files, out_dir=out_dir)
            arepo_to_colt(snap=snap, out_dir=out_dir)

