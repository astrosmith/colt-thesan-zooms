import numpy as np
import h5py, os, errno

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred


MEGAPARSEC = 3.085678e24    # Megaparsec [cm]
Msun = 1.988435e33          # Solar mass [g]

def extract_halo(snap=10, group=0, box_vir=2., i_file=0, file_dir='.', out_dir='.', tree_dir='', all_fields=True, sphere=False):
    if tree_dir == '': tree_dir = file_dir # Default
    xn_wrap, yn_wrap, zn_wrap = 0., 0., 0.
    xp_wrap, yp_wrap, zp_wrap = 0., 0., 0.
    with h5py.File(f'{tree_dir}/tree.hdf5', 'r') as f:
        out_dir = out_dir + '/extractions'
        os.makedirs(out_dir, exist_ok=True) # Ensure the output directory exists
        h = f['Header'].attrs['HubbleParam']
        BoxSize = f['Header'].attrs['BoxSize']
        BoxHalf = BoxSize / 2.
        Snapshots = f['Snapshots'][:]
        i_snap = np.where(Snapshots == snap)[0][0]
        Redshift = f['Redshifts'][i_snap]
        a = 1. / (1. + Redshift)
        sqrt_a = np.sqrt(a)
        UnitLength_in_cm = f['Header'].attrs['UnitLength_in_cm']
        UnitMass_in_g = f['Header'].attrs['UnitMass_in_g']
        UnitVelocity_in_cm_per_s = f['Header'].attrs['UnitVelocity_in_cm_per_s']
        UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
        length_to_cgs = a * UnitLength_in_cm / h
        length_to_Mpc = length_to_cgs / MEGAPARSEC
        length_to_cMpc = length_to_Mpc / a
        length_to_cMpch = length_to_cMpc * h
        length_to_kpc = 1e3 * length_to_Mpc
        length_to_ckpc = 1e3 * length_to_cMpc
        length_to_ckpch = 1e3 * length_to_cMpch
        mass_to_cgs = UnitMass_in_g / h
        mass_to_Msun = mass_to_cgs / Msun
        velocity_to_cgs = sqrt_a * UnitVelocity_in_cm_per_s
        velocity_to_kms = 1e-5 * velocity_to_cgs
        R_Crit200 = f['Group']['Group_R_Crit200'][i_snap]
        halo = f['Group']['GroupFirstSub'][i_snap]
        halo_com = f['Subhalo']['SubhaloCM'][i_snap,:]
        halo_mass = f['Subhalo']['SubhaloMass'][i_snap]
        halo_pos = f['Subhalo']['SubhaloPos'][i_snap,:] # Already in code units [a*L/h]
        halo_vel = f['Subhalo']['SubhaloVel'][i_snap,:] / velocity_to_kms # Convert to code units [sqrt(a)*V] from catalog units [km/s]
        if i_file == 0:
            print(f'Snapshot {snap}, Group {group}, Halo {halo}: Redshift = {Redshift}\t(scale factor = {a})')
            print(f'BoxSize = {BoxSize} code units = {BoxSize*length_to_cMpch} cMpc/h = {BoxSize*length_to_cMpc} cMpc = {BoxSize*length_to_Mpc} Mpc')
            print(f'R_Crit200 = {R_Crit200} code units = {R_Crit200*length_to_ckpch} ckpc/h = {R_Crit200*length_to_ckpc} ckpc = {R_Crit200*length_to_kpc} kpc')
            print(f'halo mass = {halo_mass} code units = {halo_mass*mass_to_Msun:g} Msun')
            print(f'halo center of mass = {halo_com} code units = {halo_com*length_to_cMpch} cMpc/h = {halo_com*length_to_cMpc} cMpc = {halo_com*length_to_Mpc} Mpc')
            print(f'halo position = {halo_pos} code units = {halo_pos*length_to_cMpch} cMpc/h = {halo_pos*length_to_cMpc} cMpc = {halo_pos*length_to_Mpc} Mpc')
            halo_offset_com = np.sqrt(np.sum((halo_pos-halo_com)**2))
            print(f'halo offset from center of mass = {halo_offset_com} code units = {halo_offset_com*length_to_ckpch} ckpc/h = {halo_offset_com*length_to_ckpc} ckpc = {halo_offset_com*length_to_kpc} kpc')
            print(f'halo velocity = {halo_vel} code units = {halo_vel*velocity_to_kms} km/s')
        ExtractedBoxSize = round(2. * box_vir * R_Crit200, 4)
        # ExtractedBoxSize = round(box_cMpc * 1e3 * h, 4) # 8 cMpc
        ExtractedBoxHalf = ExtractedBoxSize / 2.
        x_min = halo_pos[0] - ExtractedBoxHalf
        x_max = halo_pos[0] + ExtractedBoxHalf
        y_min = halo_pos[1] - ExtractedBoxHalf
        y_max = halo_pos[1] + ExtractedBoxHalf
        z_min = halo_pos[2] - ExtractedBoxHalf
        z_max = halo_pos[2] + ExtractedBoxHalf
        if i_file == 0:
            print(f'\nExtracted BoxSize = {ExtractedBoxSize} code units = {ExtractedBoxSize*length_to_ckpch} ckpc/h = {ExtractedBoxSize*length_to_ckpc} ckpc = {ExtractedBoxSize*length_to_kpc} kpc')
            print(f'Extracted Region Min = ({x_min*length_to_cMpch}, {y_min*length_to_cMpch}, {z_min*length_to_cMpch}) cMpc/h')
            print(f'Extracted Region Max = ({x_max*length_to_cMpch}, {y_max*length_to_cMpch}, {z_max*length_to_cMpch}) cMpc/h')
        if x_min <= 0.: xp_wrap = halo_pos[0] + BoxHalf
        if y_min <= 0.: yp_wrap = halo_pos[1] + BoxHalf
        if z_min <= 0.: zp_wrap = halo_pos[2] + BoxHalf
        if x_max >= BoxSize: xn_wrap = halo_pos[0] - BoxHalf
        if y_max >= BoxSize: yn_wrap = halo_pos[1] - BoxHalf
        if z_max >= BoxSize: zn_wrap = halo_pos[2] - BoxHalf
    filename = f'{file_dir}/snapdir_{snap:03d}/snapshot_{snap:03d}.{i_file}.hdf5'
    if sphere:
        extracted_filename = f'{out_dir}/sphere_snap_{snap:03d}_g{group}.{i_file}.hdf5'
    else:
        extracted_filename = f'{out_dir}/snap_{snap:03d}_g{group}.{i_file}.hdf5'
    silentremove(extracted_filename)
    with h5py.File(filename, 'r') as f, h5py.File(extracted_filename, 'w') as ef:
        NumPart_ThisFile = f['Header'].attrs['NumPart_ThisFile']
        if i_file == 0:
            print(f"NumPart_Total = {f['Header'].attrs['NumPart_Total']}")

        # Filter particle data
        part_type_list = [f'PartType{i}' for i in range(6)]
        for i in range(len(part_type_list)):
            part_type = part_type_list[i]
            try:
                p = f[part_type]
            except KeyError:
                continue
            pos = p['Coordinates'][:]
            if xp_wrap != 0.: pos[pos[:,0] > xp_wrap,0] -= BoxSize
            if yp_wrap != 0.: pos[pos[:,1] > yp_wrap,1] -= BoxSize
            if zp_wrap != 0.: pos[pos[:,2] > zp_wrap,2] -= BoxSize
            if xn_wrap != 0.: pos[pos[:,0] < xn_wrap,0] += BoxSize
            if yn_wrap != 0.: pos[pos[:,1] < yn_wrap,1] += BoxSize
            if zn_wrap != 0.: pos[pos[:,2] < zn_wrap,2] += BoxSize
            if sphere:
                mask = ((pos[:,0] - halo_pos[0])**2 + (pos[:,1] - halo_pos[1])**2 + (pos[:,2] - halo_pos[2])**2 < ExtractedBoxHalf**2)
            else: # Box
                mask = (pos[:,0] > x_min) & (pos[:,0] < x_max) & (pos[:,1] > y_min) & (pos[:,1] < y_max) & (pos[:,2] > z_min) & (pos[:,2] < z_max)
            p_orig = len(mask)
            p_mask = np.count_nonzero(mask)
            p_frac = float(p_mask) / float(p_orig)
            #if i == 0:
            #    print(f'\nSubfile {i_file}: {part_type} extracted particle fraction = {100.*p_frac}%  ({p_mask} / {p_orig})')
            NumPart_ThisFile[i] = p_mask
            if p_mask == 0:
                continue
            uncopied_fields = []
            ep = ef.create_group(part_type)
            # print(part_type,p.keys())
            for key in p.keys():
                if not all_fields:
                    if key not in ['Coordinates', 'Velocities', 'ParticleIDs', 'Masses', 'InternalEnergy', 'StarFormationRate',
                                   'Density', 'ElectronAbundance', 'HI_Fraction', 'HII_Fraction', 'GFM_DustMetallicity', 'GFM_Metallicity']:
                        uncopied_fields.append(key)
                        continue
                # if i_file == 0: print(f'\t{key}')
                p_data = p[key][:]
                try:
                    p_data_mask = p_data[mask]
                except:
                    p_data_mask = p_data[mask,:]
                if key in ['Coordinates', 'CenterOfMass', 'BirthPos']:
                    if xp_wrap != 0.: p_data_mask[p_data_mask[:,0] > xp_wrap,0] -= BoxSize
                    if yp_wrap != 0.: p_data_mask[p_data_mask[:,1] > yp_wrap,1] -= BoxSize
                    if zp_wrap != 0.: p_data_mask[p_data_mask[:,2] > zp_wrap,2] -= BoxSize
                    if xn_wrap != 0.: p_data_mask[p_data_mask[:,0] < xn_wrap,0] += BoxSize
                    if yn_wrap != 0.: p_data_mask[p_data_mask[:,1] < yn_wrap,1] += BoxSize
                    if zn_wrap != 0.: p_data_mask[p_data_mask[:,2] < zn_wrap,2] += BoxSize
                    p_data_mask[:,0] -= x_min
                    p_data_mask[:,1] -= y_min
                    p_data_mask[:,2] -= z_min
                elif key in ['Velocities', 'BirthVel']:
                    p_data_mask[:,0] -= halo_vel[0]
                    p_data_mask[:,1] -= halo_vel[1]
                    p_data_mask[:,2] -= halo_vel[2]
                ep.create_dataset(key, data=p_data_mask)
                for attr in p[key].attrs.keys():
                    ep[key].attrs[attr] = p[key].attrs[attr]
            if i_file == 0 and len(uncopied_fields) > 0: print('\tUncopied fields =', uncopied_fields)

        # Copy header data with correct BoxSize and NumPart
        for group_name in ['Config', 'Header', 'Parameters']:
            g = f[group_name]
            eg = ef.create_group(group_name)
            for attr in g.attrs.keys():
                if attr == 'BoxSize':
                    eg.attrs['BoxSize'] = ExtractedBoxSize
                elif attr == 'NumPart_ThisFile':
                    eg.attrs['NumPart_ThisFile'] = NumPart_ThisFile
                else:
                    eg.attrs[attr] = g.attrs[attr]
            for key in g.keys():
                eg.create_dataset(key, data=g[key][:])
                for attr in g[key].attrs.keys():
                    eg[key].attrs[attr] = g[key].attrs[attr]
        eg = ef['Header']
        eg.attrs['ExtractionPositionOffset'] = halo_pos - ExtractedBoxHalf
        eg.attrs['ExtractionVelocityOffset'] = halo_vel
        eg.attrs['GroupID'] = np.int64(group)
        eg.attrs['SubhaloID'] = np.int64(halo)

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 5:
        snap, i_file, file_dir, out_dir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    else:
        raise ValueError('Usage: python extract_halo.py snap i_file file_dir out_dir')
    extract_halo(snap=snap, i_file=i_file, file_dir=file_dir, out_dir=out_dir)

