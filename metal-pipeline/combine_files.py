import numpy as np
import h5py, sys, os, errno

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

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

def combine_files(snap=10, n_files=1, group=0, out_dir='.', all_fields=True, sphere=False, remove_files=True):
    out_pre = out_dir + f'/extractions/{"sphere_" if sphere else ""}snap_{snap:03d}_g{group}'
    NumPartFilesGas = np.zeros(n_files, dtype=np.uint64)
    NumPartFilesDM = np.zeros(n_files, dtype=np.uint64)
    NumPartFilesLR = np.zeros(n_files, dtype=np.int64)
    NumPartFilesStars = np.zeros(n_files, dtype=np.uint64)
    print(f'Counting cumulative numbers of particles: Snapshot {snap}, Group {group}')
    for i_file in progressbar(range(n_files)):
        with h5py.File(f'{out_pre}.{i_file}.hdf5', 'r') as ef:
            NumPart_ThisFile = ef['Header'].attrs['NumPart_ThisFile']
            NumPartFilesGas[i_file] = NumPart_ThisFile[0]
            NumPartFilesDM[i_file] = NumPart_ThisFile[1]
            NumPartFilesLR[i_file] = NumPart_ThisFile[2]
            NumPartFilesStars[i_file] = NumPart_ThisFile[4]
            if i_file == 0:
                NumPart_Total = ef['Header'].attrs['NumPart_Total']
                NumPart_Total -= NumPart_Total
        for j in range(len(NumPart_Total)):
            NumPart_Total[j] += NumPart_ThisFile[j]
    FilesGas = np.arange(n_files)[NumPartFilesGas>0]
    FilesDM = np.arange(n_files)[NumPartFilesDM>0]
    FilesLR = np.arange(n_files)[NumPartFilesLR>0]
    FilesStars = np.arange(n_files)[NumPartFilesStars>0]
    print('NumPart_Total =',NumPart_Total)
    #print('NumPartFilesGas =',NumPartFilesGas)
    #print('NumPartFilesDM =',NumPartFilesDM)
    #print('NumPartFilesLR =',NumPartFilesLR)
    #print('NumPartFilesStars =',NumPartFilesStars)

    # Write combined file
    ds0 = {'PartType': '0'}
    ds1 = {'PartType': '1'}
    ds2 = {'PartType': '2'}
    ds4 = {'PartType': '4'}
    #ds5 = {'PartType': '5'}
    silentremove(f'{out_pre}.hdf5')
    with h5py.File(f'{out_pre}.hdf5', 'w') as f:
        print('Copying all data from sparse extraction files:')
        for i_file in progressbar(range(n_files)):
            with h5py.File(f'{out_pre}.{i_file}.hdf5', 'r') as ef:
                # Copy header data with correct counts
                if i_file == 0:
                    for group_name in ['Config', 'Header', 'Parameters']:
                        eg = ef[group_name]
                        g = f.create_group(group_name)
                        for attr in eg.attrs.keys():
                            if attr == 'NumPart_ThisFile':
                                g.attrs['NumPart_ThisFile'] = NumPart_Total.astype(np.uint32)
                            elif attr == 'NumPart_Total':
                                g.attrs['NumPart_Total'] = NumPart_Total.astype(np.uint64)
                            elif attr == 'NumPart_Total_HighWord':
                                tmp = eg.attrs['NumPart_Total_HighWord']
                                for i in range(len(tmp)):
                                    tmp[i] = 0
                                g.attrs['NumPart_Total_HighWord'] = tmp
                            elif attr == 'NumFilesPerSnapshot':
                                g.attrs['NumFilesPerSnapshot'] = np.int32(1)
                            else:
                                g.attrs[attr] = eg.attrs[attr]
                        for key in eg.keys():
                            g.create_dataset(key, data=eg[key][:])
                            for attr in eg[key].attrs.keys():
                                g[key].attrs[attr] = eg[key].attrs[attr]

                # Filter particle data
                # PartType0 (Gas)
                if i_file in FilesGas:
                    if all_fields:
                        vector_fields = ['Coordinates', 'GFM_Metals',
                                         #'KappaIR_P', 'KappaIR_R',
                                         'PhotonDensity', 'Velocities']
                        scalar_fields = ['AllowRefinement', 'Density', 'DustTemperature', 'ElectronAbundance',
                                         'GFM_DustMetallicity', 'GFM_Metallicity',
                                         'H2_Fraction', 'HI_Fraction', 'HeII_Fraction', 'HeI_Fraction',
                                         'HighResGasMass', 'InternalEnergy', 'KappaIR_P', 'KappaIR_R',
                                         'Masses', 'ParticleIDs', 'StarFormationRate']
                    else:
                        vector_fields = ['Coordinates', 'Velocities']
                        scalar_fields = ['ParticleIDs', 'Masses', 'InternalEnergy', 'Density', 'ElectronAbundance', 'StarFormationRate',
                                         'HI_Fraction', 'HII_Fraction', 'GFM_DustMetallicity', 'GFM_Metallicity']
                    fields = vector_fields + scalar_fields
                    #print('LastFileGas =',LastFileGas)
                    ep0 = ef['PartType0']
                    if i_file == FilesGas[0]:
                        p0 = f.create_group('PartType0')
                        for field in fields:
                            ds0[field] = ep0[field][:]
                    else:
                        for field in vector_fields:
                            ds0[field] = np.vstack([ds0[field], ep0[field][:]])
                        for field in scalar_fields:
                            ds0[field] = np.hstack([ds0[field], ep0[field][:]])
                    if i_file == FilesGas[-1]:
                        for field in fields:
                            p0.create_dataset(field, data=ds0[field])
                            for attr in ep0[field].attrs.keys():
                                p0[field].attrs[attr] = ep0[field].attrs[attr]

                # PartType1 (Dark Matter)
                if all_fields:
                    if i_file in FilesDM:
                        vector_fields = ['Coordinates', 'Velocities']
                        scalar_fields = ['ParticleIDs']
                        fields = vector_fields + scalar_fields
                        ep1 = ef['PartType1']
                        if i_file == FilesDM[0]:
                            p1 = f.create_group('PartType1')
                            for field in fields:
                                ds1[field] = ep1[field][:]
                        else:
                            for field in vector_fields:
                                ds1[field] = np.vstack([ds1[field], ep1[field][:]])
                            for field in scalar_fields:
                                ds1[field] = np.hstack([ds1[field], ep1[field][:]])
                        if i_file == FilesDM[-1]:
                            for field in fields:
                                p1.create_dataset(field, data=ds1[field])
                                for attr in ep1[field].attrs.keys():
                                    p1[field].attrs[attr] = ep1[field].attrs[attr]

                # PartType2 (Low Resolution Dark Matter)
                if all_fields:
                    if i_file in FilesLR:
                        vector_fields = ['Coordinates', 'Velocities']
                        scalar_fields = ['Masses', 'ParticleIDs']
                        fields = vector_fields + scalar_fields
                        ep2 = ef['PartType2']
                        if i_file == FilesLR[0]:
                            p2 = f.create_group('PartType2')
                            for field in fields:
                                ds2[field] = ep2[field][:]
                        else:
                            for field in vector_fields:
                                ds2[field] = np.vstack([ds2[field], ep2[field][:]])
                            for field in scalar_fields:
                                ds2[field] = np.hstack([ds2[field], ep2[field][:]])
                        if i_file == FilesLR[-1]:
                            for field in fields:
                                p2.create_dataset(field, data=ds2[field])
                                for attr in ep2[field].attrs.keys():
                                    p2[field].attrs[attr] = ep2[field].attrs[attr]

                # PartType5 (Black Holes)
                #if False and all_fields:
                #    try:
                #        vector_fields = ['Coordinates', 'Velocities']
                #        scalar_fields = ['BH_CumEgyInjection_QM', 'BH_CumEgyInjection_RM', 'BH_CumMassGrowth_QM',
                #                         'BH_CumMassGrowth_RM', 'BH_Density', 'BH_Hsml', 'BH_MPB_CumEgyHigh',
                #                         'BH_MPB_CumEgyLow', 'BH_Mass', 'BH_Mdot', 'BH_MdotBondi', 'BH_MdotEddington',
                #                         'BH_PhotonHsml', 'BH_Pressure', 'BH_Progs', 'BH_U', 'Masses', 'ParticleIDs',
                #                         'Potential', 'SubfindHsml']
                #                         #'SubfindDMDensity', 'SubfindDensity', 'SubfindVelDisp']
                #        fields = vector_fields + scalar_fields
                #        ep5 = ef['PartType5']
                #        if i_file == 0:
                #            p5 = f.create_group('PartType5')
                #            for field in fields:
                #                ds5[field] = ep5[field][:]
                #        else:
                #            for field in vector_fields:
                #                ds5[field] = np.vstack([ds5[field], ep5[field][:]])
                #            for field in scalar_fields:
                #                ds5[field] = np.hstack([ds5[field], ep5[field][:]])
                #        if i_file == LastFileBH:
                #            for field in fields:
                #                p5.create_dataset(field, data=ds5[field])
                #                for attr in ep5[field].attrs.keys():
                #                    p5[field].attrs[attr] = ep5[field].attrs[attr]

                # PartType4 (Stars)
                if all_fields:
                    if i_file in FilesStars:
                        vector_fields = ['BirthPos', 'BirthVel',
                                         'Coordinates', #'GFM_DustAGB', 'GFM_DustSNII', 'GFM_DustSNIa',
                                         #'GFM_Metals', 'GFM_StellarPhotometrics',
                                         'Velocities']
                        scalar_fields = ['BirthDensity',
                                         'GFM_InitialMass', 'GFM_Metallicity', 'GFM_StellarFormationTime',
                                         'Masses', 'ParticleIDs']
                        fields = vector_fields + scalar_fields
                        ep4 = ef['PartType4']
                        if i_file == FilesStars[0]:
                            p4 = f.create_group('PartType4')
                            for field in fields:
                                ds4[field] = ep4[field][:]
                        else:
                            for field in vector_fields:
                                ds4[field] = np.vstack([ds4[field], ep4[field][:]])
                            for field in scalar_fields:
                                ds4[field] = np.hstack([ds4[field], ep4[field][:]])
                        if i_file == FilesStars[-1]:
                            for field in fields:
                                p4.create_dataset(field, data=ds4[field])
                                for attr in ep4[field].attrs.keys():
                                    p4[field].attrs[attr] = ep4[field].attrs[attr]

    # Optionally remove the previous split extracted files
    if remove_files:
        print('Removing sparse extraction files:')
        for i_file in progressbar(range(n_files)):
            silentremove(f'{out_pre}.{i_file}.hdf5')

if __name__ == '__main__':
    if len(sys.argv) == 4:
        snap, n_files, out_dir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    else:
        raise ValueError('Usage: python combine_files.py snap n_files out_dir')
    combine_files(snap=snap, n_files=n_files, out_dir=out_dir)

