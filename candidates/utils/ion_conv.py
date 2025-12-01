import os, h5py

# Directory containing the simulation files
zoom_dir = '/orcd/data/mvogelsb/004/Thesan-Zooms'
colt_dir = f'{zoom_dir}-COLT'
colt_dir = '/orcd/data/mvogelsb/005/Lab/Thesan-Zooms-COLT'

# Directory containing the output files
states = 'states-no-UVB'  # States to check
field = 'conv/rec_HII'  # Field to check for convergence
target = 0.001  # Target convergence value

print(f'Searching for missing or unconverged {field} files...')

# Define job sets with individual job, group, and runs
job_sets = [
# COLT (long circulators)
# {'job':'D', 'group':'g578', 'run':'noESF_z4', 'snaps':[88, 179]},      # 88-179 ->  88-179
# {'job':'D', 'group':'g578', 'run':'varEff_z4', 'snaps':[0, 68]},       # 0-68   ->  0-68
# {'job':'E', 'group':'g1163', 'run':'noESF_z4', 'snaps':[0, 188]},      # 0-188  ->  0-188
# {'job':'H', 'group':'g33206', 'run':'z16', 'snaps':[0, 8]},            # 0-8    ->  0-8
# Fiducial:
#{'job':'A', 'group':'g2', 'run':'z4', 'snaps':[0, 67]},                # 0-67   ->  0-67
# {'job':'B', 'group':'g39', 'run':'z4', 'snaps':[0, 188]},              # 0-188  ->  0-188
# {'job':'C', 'group':'g205', 'run':'z4', 'snaps':[0, 188]},             # 0-188  ->  0-188
# {'job':'D', 'group':'g578', 'run':'z4', 'snaps':[0, 188]},             # 0-188  ->  0-188
## {'job':'E', 'group':'g1163', 'run':'z4', 'snaps':[0, 188]},            # 0-188  ->  0-188
# {'job':'F', 'group':'g5760', 'run':'z8', 'snaps':[0, 188]},            # 0-188  ->  0-188
## {'job':'G', 'group':'g10304', 'run':'z8', 'snaps':[0, 188]},           # 0-188  ->  0-188
## {'job':'J', 'group':'g137030', 'run':'z16', 'snaps':[5, 188]},         # 0-188  ->  5-188
## {'job':'K', 'group':'g500531', 'run':'z16', 'snaps':[9, 188]},         # 0-188  ->  9-188
## {'job':'L', 'group':'g519761', 'run':'z16', 'snaps':[39, 188]},        # 0-188  ->  39-188
## {'job':'M', 'group':'g2274036', 'run':'z16', 'snaps':[9, 188]},        # 0-189  ->  9-188
# COLT (int circulators)
# {'job':'F', 'group':'g5760', 'run':'noESF_z4', 'snaps':[5, 188]},      # 0-188  ->  5-188
# {'job':'G', 'group':'g10304', 'run':'noESF_z4', 'snaps':[0, 188]},     # 0-188  ->  0-188
# {'job':'H', 'group':'g33206', 'run':'noESF_z8', 'snaps':[4, 188]},     # 0-188  ->  4-188
# {'job':'H', 'group':'g33206', 'run':'noESF_z4', 'snaps':[6, 188]},     # 0-188  ->  6-188
# {'job':'J', 'group':'g137030', 'run':'noESF_z8', 'snaps':[2, 188]},    # 0-188  ->  2-188
# {'job':'J', 'group':'g137030', 'run':'noESF_z4', 'snaps':[10, 188]},   # 0-188  ->  10-188
# {'job':'F', 'group':'g5760', 'run':'noRlim_z4', 'snaps':[5, 188]},     # 0-188  ->  5-188
# {'job':'F', 'group':'g5760', 'run':'varEff_z4', 'snaps':[5, 188]},     # 0-188  ->  5-188
# {'job':'H', 'group':'g33206', 'run':'uvb_z8', 'snaps':[4, 188]},       # 0-188  ->  4-188
# {'job':'H', 'group':'g33206', 'run':'noExt_z8', 'snaps':[4, 188]},     # 0-188  ->  4-188
# {'job':'H', 'group':'g33206', 'run':'noRlim_z8', 'snaps':[4, 188]},    # 0-188  ->  4-188
# {'job':'H', 'group':'g33206', 'run':'varEff_z8', 'snaps':[3, 188]},    # 0-188  ->  3-188
# {'job':'H', 'group':'g33206', 'run':'noRlim_z4', 'snaps':[6, 189]},    # 0-189  ->  6-189
# {'job':'H', 'group':'g33206', 'run':'varEff_z4', 'snaps':[6, 189]},    # 0-189  ->  6-189
# {'job':'I', 'group':'g37591', 'run':'uvb_z8', 'snaps':[7, 188]},       # 0-188  ->  7-188
# {'job':'I', 'group':'g37591', 'run':'noExt_z8', 'snaps':[6, 188]},     # 0-188  ->  6-188
# {'job':'J', 'group':'g137030', 'run':'uvb_z8', 'snaps':[3, 188]},      # 0-188  ->  3-188
# {'job':'J', 'group':'g137030', 'run':'noExt_z8', 'snaps':[3, 188]},    # 0-188  ->  3-188
# {'job':'J', 'group':'g137030', 'run':'noRlim_z8', 'snaps':[2, 188]},   # 0-188  ->  2-188
# {'job':'J', 'group':'g137030', 'run':'varEff_z8', 'snaps':[3, 188]},   # 0-188  ->  3-188
# {'job':'J', 'group':'g137030', 'run':'noRlim_z4', 'snaps':[10, 189]},  # 0-189  ->  10-189
# {'job':'J', 'group':'g137030', 'run':'varEff_z4', 'snaps':[10, 188]},  # 0-188  ->  10-188
# {'job':'K', 'group':'g500531', 'run':'uvb_z8', 'snaps':[8, 189]},      # 0-189  ->  8-189
# {'job':'K', 'group':'g500531', 'run':'noExt_z8', 'snaps':[8, 188]},    # 0-188  ->  8-188
# {'job':'L', 'group':'g519761', 'run':'uvb_z8', 'snaps':[23, 189]},     # 0-189  ->  23-189
# {'job':'L', 'group':'g519761', 'run':'noExt_z8', 'snaps':[46, 189]},   # 0-189  ->  46-189
# {'job':'M', 'group':'g2274036', 'run':'uvb_z8', 'snaps':[22, 182]},    # 0-189  ->  22-182
# {'job':'M', 'group':'g2274036', 'run':'noExt_z8', 'snaps':[28, 189]},  # 0-189  ->  28-189
# {'job':'N', 'group':'g5229300', 'run':'uvb_z8', 'snaps':[0, 0]},       # 0-189  ->  0-0
# {'job':'N', 'group':'g5229300', 'run':'noExt_z8', 'snaps':[0, 0]},     # 0-189  ->  0-0
# Fiducial:
# {'job':'F', 'group':'g5760', 'run':'z4', 'snaps':[5, 188]},            # 0-188  ->  5-188
# {'job':'G', 'group':'g10304', 'run':'z4', 'snaps':[0, 188]},           # 0-188  ->  0-188
# {'job':'H', 'group':'g33206', 'run':'z8', 'snaps':[4, 188]},           # 0-188  ->  4-188
# {'job':'H', 'group':'g33206', 'run':'z4', 'snaps':[6, 189]},           # 0-189  ->  6-189
# {'job':'I', 'group':'g37591', 'run':'z8', 'snaps':[6, 188]},           # 0-188  ->  6-188
# {'job':'I', 'group':'g37591', 'run':'z4', 'snaps':[11, 188]},          # 0-188  ->  11-188
# {'job':'J', 'group':'g137030', 'run':'z8', 'snaps':[2, 188]},          # 0-188  ->  2-188
# {'job':'J', 'group':'g137030', 'run':'z4', 'snaps':[10, 188]},         # 0-188  ->  10-188
# {'job':'K', 'group':'g500531', 'run':'z8', 'snaps':[10, 188]},         # 0-188  ->  10-188
# {'job':'K', 'group':'g500531', 'run':'z4', 'snaps':[16, 189]},         # 0-189  ->  16-189
# {'job':'L', 'group':'g519761', 'run':'z8', 'snaps':[46, 189]},         # 0-189  ->  46-189
# {'job':'L', 'group':'g519761', 'run':'z4', 'snaps':[51, 189]},         # 0-189  ->  51-189
# {'job':'M', 'group':'g2274036', 'run':'z8', 'snaps':[27, 189]},        # 0-189  ->  27-189
# {'job':'M', 'group':'g2274036', 'run':'z4', 'snaps':[52, 189]},        # 0-189  ->  52-189
# {'job':'N', 'group':'g5229300', 'run':'z16', 'snaps':[72, 189]},       # 0-189  ->  72-189
# {'job':'N', 'group':'g5229300', 'run':'z8', 'snaps':[83, 189]},        # 0-189  ->  83-189
# {'job':'N', 'group':'g5229300', 'run':'z4', 'snaps':[87, 189]},        # 0-189  ->  87-189
]

# Loop through job sets
for job_set in job_sets:
    job, group, run, snaps = job_set['job'], job_set['group'], job_set['run'], job_set['snaps']
    out_dir = f'{colt_dir}/{group}/{run}/ics'
    # out_dir = f'{colt_dir}/{group}/{run}/ics_tree'
    #print(f'out_dir = {out_dir}')
    # Check if the output directory exists
    if os.path.isdir(out_dir):
        # Make sure there is an output file for each snapshot
        missing_snaps, unconverged_snaps = [], []
        for snap in range(snaps[0], snaps[1] + 1):
            # Check if the output file exists
            out_file = f'{out_dir}/{states}_{snap:03d}.hdf5'
            #print(f'  out_file = {out_file}')
            if os.path.isfile(out_file):
                with h5py.File(out_file, 'r') as f:
                    # Check the convergence from the file
                    if field in f:
                        x = f[field][:]  # Read the convergence field
                        conv = abs(1. - x[-2] / x[-1])  # Calculate the convergence
                        if conv > target or 'n_photons_pre' not in f.attrs or f.attrs['n_photons'] < 100_000_000:
                            unconverged_snaps.append(snap)  # Field is unconverged
                        # print(f'  snap {snap}: {100.*conv:.2f}%')
                    elif 'x_e' in f:
                        with h5py.File(f'{out_dir}/colt_{snap:03d}.hdf5', 'r') as cf:
                            if len(f['x_e']) != cf.attrs['n_cells']:
                                unconverged_snaps.append(snap)  # Field size is wrong
                    else:
                        missing_snaps.append(snap)  # Field is missing
            else:
                missing_snaps.append(snap)  # File is missing
        if len(missing_snaps) > 0:
            print(f'{job}/{run}: {out_dir} is missing {",".join(map(str, missing_snaps))}')
        if len(unconverged_snaps) > 0:
            print(f'{job}/{run}: {out_dir} is unconverged {",".join(map(str, unconverged_snaps))}')
    else:
        print(f'Missing {out_dir} ({job}/{run})')  # Directory does not exist
