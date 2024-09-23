import h5py, os, platform, asyncio
import numpy as np
from dataclasses import dataclass, field
from time import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

# Constants
GAS_HIGH_RES_THRESHOLD = 0.5 # Threshold deliniating high and low resolution gas particles
SOLAR_MASS = 1.989e33  # Solar masses
VERBOSITY = 0 # Level of print verbosity
MAX_WORKERS = cpu_count() # Maximum number of workers
SERIAL = 1 # Run in serial
ASYNCIO = 2 # Run in parallel (asyncio)
READ_DEFAULT = (ASYNCIO,) # Default read method
# READ_DEFAULT = (SERIAL,ASYNCIO) # Default read method
READ_COUNTS = READ_DEFAULT # Read counts methods
READ_GROUPS = READ_DEFAULT # Read groups methods
READ_SNAPS = READ_DEFAULT # Read snapshots methods
SUBHALOS = True # Process individual subhalos
CENTRALS = True # Add the central subhalos of valid candidate groups (and vice versa)
ADD_TREE = True # Add the main group and subhalo from the tree
MIN_STARS = 1 # Initial minimum number of star particles (for StarFlag)
MAX_STARS = 10 # Maximum for the minimum number of star particles (for StarFlag)
MAX_HALOS_TO_STARS = 20 # Maximum number of subhalos per star threshold (for StarFlag)
TIMERS = True # Print timers
SUB_TIMERS = False and TIMERS # Print sub-timers

# Configurable global variables
snap = 188 # Snapshot number
if platform.system() == 'Darwin':
    sim, zoom_dir = 'g500531/z4', os.path.expandvars('$HOME/Engaging/Thesan-Zooms')
else:
    sim, zoom_dir = 'g5760/z4', '/orcd/data/mvogelsb/004/Thesan-Zooms'

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        sim = sys.argv[1]
    elif len(sys.argv) == 3:
        sim, snap = sys.argv[1], int(sys.argv[2])
    elif len(sys.argv) == 4:
        sim, snap, zoom_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    elif len(sys.argv) != 1:
        raise ValueError('Usage: python candidates.py [sim] [snap] [zoom_dir]')

out_dir = f'{zoom_dir}/{sim}/output'
snap_pre = f'{out_dir}/snapdir_{snap:03d}/snapshot_{snap:03d}.'
dist_dir = f'{zoom_dir}/{sim}/postprocessing/distances'
cand_dir = f'{zoom_dir}/{sim}/postprocessing/candidates'

# Overwrite for local testing
#out_dir = '.' # Overwrite for local testing
#dist_dir = '.' # Overwrite for local testing
#cand_dir = '.' # Overwrite for local testing

# The following paths should not change
fof_pre = f'{out_dir}/groups_{snap:03d}/fof_subhalo_tab_{snap:03d}.'
dist_file = f'{dist_dir}/distances_{snap:03d}.hdf5'
tree_file = f'{out_dir}/tree.hdf5'
cand_file = f'{cand_dir}/candidates_{snap:03d}.hdf5'

@dataclass
class Simulation:
    """Simulation information and data."""
    NTYPES: np.int32 = -1 # Number of particle types
    n_files: np.int32 = 0 # Number of files
    n_groups_tot: np.uint64 = None # Total number of groups
    n_subhalos_tot: np.uint64 = None # Total number of subhalos
    a: np.float64 = None # Scale factor
    BoxSize: np.float64 = None # Size of the simulation volume
    BoxHalf: np.float64 = None # Half the size of the simulation volume
    h: np.float64 = None # Hubble parameter
    Omega0: np.float64 = None # Matter density
    OmegaBaryon: np.float64 = None # Baryon density
    OmegaLambda: np.float64 = None # Dark energy density
    UnitLength_in_cm: np.float64 = None # Unit length in cm
    UnitMass_in_g: np.float64 = None # Unit mass in g
    UnitVelocity_in_cm_per_s: np.float64 = None # Unit velocity in cm/s
    length_to_cgs: np.float64 = None # Conversion factor for length to cgs
    volume_to_cgs: np.float64 = None # Conversion factor for volume to cgs
    mass_to_cgs: np.float64 = None # Conversion factor for mass to cgs
    mass_to_msun: np.float64 = None # Conversion factor for mass to solar masses
    velocity_to_cgs: np.float64 = None # Conversion factor for velocity to cgs

    # File counts and offsets
    n_groups: np.ndarray = None
    n_subhalos: np.ndarray = None
    first_group: np.ndarray = None
    first_subhalo: np.ndarray = None

    # FOF data
    groups: dict = field(default_factory=dict)
    group_units: dict = field(default_factory=dict)
    subhalos: dict = field(default_factory=dict)
    subhalo_units: dict = field(default_factory=dict)

    def __post_init__(self):
        """Allocate memory for group data."""
        # Read header info from the fof files
        with h5py.File(fof_pre + '0.hdf5', 'r') as f:
            header = f['Header'].attrs
            self.n_files = header['NumFiles']
            self.n_groups_tot = header['Ngroups_Total']
            self.n_subhalos_tot = header['Nsubhalos_Total']
            self.n_groups = np.zeros(self.n_files, dtype=np.uint64)
            self.n_subhalos = np.zeros(self.n_files, dtype=np.uint64)
            self.a = header['Time']
            params = f['Parameters'].attrs
            self.BoxSize = params['BoxSize']
            self.h = params['HubbleParam']
            self.Omega0 = params['Omega0']
            self.OmegaBaryon = params['OmegaBaryon']
            self.OmegaLambda = params['OmegaLambda']
            self.UnitLength_in_cm = params['UnitLength_in_cm']
            self.UnitMass_in_g = params['UnitMass_in_g']
            self.UnitVelocity_in_cm_per_s = params['UnitVelocity_in_cm_per_s']
            if self.n_groups_tot > 0:
                g = f['Group']
                for field in g.keys():
                    shape, dtype = g[field].shape, g[field].dtype
                    shape = (self.n_groups_tot,) + shape[1:]
                    self.groups[field] = np.empty(shape, dtype=dtype)
                    g_attrs = g[field].attrs
                    if len(g_attrs) > 0:
                        self.group_units[field] = {key: val for key,val in g_attrs.items()}
                        print(f'field: {self.group_units[field]}')
            if self.n_subhalos_tot > 0:
                g = f['Subhalo']
                for field in g.keys():
                    shape, dtype = g[field].shape, g[field].dtype
                    shape = (self.n_subhalos_tot,) + shape[1:]
                    self.subhalos[field] = np.empty(shape, dtype=dtype)
                    g_attrs = g[field].attrs
                    if len(g_attrs) > 0:
                        self.subhalo_units[field] = {key: val for key,val in g_attrs.items()}
                        print(f'field: {self.subhalo_units[field]}')

        # Derived quantities
        self.BoxHalf = self.BoxSize / 2.
        self.length_to_cgs = self.a * self.UnitLength_in_cm / self.h
        self.volume_to_cgs = self.length_to_cgs**3
        self.mass_to_cgs = self.UnitMass_in_g / self.h
        self.mass_to_msun = self.mass_to_cgs / SOLAR_MASS
        self.velocity_to_cgs = np.sqrt(self.a) * self.UnitVelocity_in_cm_per_s

        # Read tree info
        if ADD_TREE:
            with h5py.File(tree_file, 'r') as f:
                Snapshots = f['Snapshots'][:] # Snapshots in the tree
                indices = np.where(Snapshots == snap)[0]
                i_tree = indices[0] if indices.size > 0 else -1 # Index of the snapshot in the tree
                if i_tree >= 0:
                    self.TreeGroupID = f['Group']['GroupID'][:][i_tree] # Group ID in the tree
                    self.TreeSubhaloID = f['Subhalo']['SubhaloID'][:][i_tree] # Subhalo ID in the tree

        # Read distances info
        with h5py.File(dist_file, 'r') as f:
            header = f['Header'].attrs
            self.r_com = header['PosHR']
            self.RadiusHR = header['RadiusHR']
            self.RadiusLR = header['RadiusLR']
            if self.n_groups_tot > 0:
                g = f['Group']
                self.Group_distances_gas_lr = g['MinDistGasLR'][:]
                self.Group_distances_gas_hr = g['MinDistGasHR'][:]
                self.Group_distances_dm = g['MinDistDM'][:]
                self.Group_distances_p2 = g['MinDistP2'][:]
                self.Group_distances_p3 = g['MinDistP3'][:]
                self.Group_R_Crit200 = g['R_Crit200'][:]
                if 'MinDistStarsLR' in g:
                    self.Group_distances_stars_lr = g['MinDistStarsLR'][:]
                    # Calculate the minimum distance to a low-resolution particle
                    self.Group_distances_lr = np.minimum(self.Group_distances_p2, self.Group_distances_p3, self.Group_distances_stars_lr) # P2,P3,StarsLR
                else:
                    self.Group_distances_stars_lr = np.zeros_like(self.Group_distances_gas_lr)
                    # Calculate the minimum distance to a low-resolution particle
                    self.Group_distances_lr = np.minimum(self.Group_distances_p2, self.Group_distances_p3) # P2,P3
                if 'MinDistStarsHR' in g:
                    self.Group_distances_stars_hr = g['MinDistStarsHR'][:]
                else:
                    self.Group_distances_stars_hr = 100. * np.copy(self.Group_R_Crit200) # No HR in Rvir
                if 'MinMemberDistStarsHR' in g:
                    self.Group_member_distances_stars_hr = g['MinMemberDistStarsHR'][:]
                else:
                    self.Group_member_distances_stars_hr = 100. * np.copy(self.Group_R_Crit200) # No HR in Rvir
            if SUBHALOS and self.n_subhalos_tot > 0:
                g = f['Subhalo']
                self.Subhalo_M_gas = g['M_gas'][:]
                self.Subhalo_M_stars = g['M_stars'][:]
                self.Subhalo_M_vir = g['M_vir'][:]
                self.Subhalo_distances_gas_lr = g['MinDistGasLR'][:]
                self.Subhalo_distances_gas_hr = g['MinDistGasHR'][:]
                self.Subhalo_distances_dm = g['MinDistDM'][:]
                self.Subhalo_distances_p2 = g['MinDistP2'][:]
                self.Subhalo_distances_p3 = g['MinDistP3'][:]
                self.Subhalo_R_vir = g['R_vir'][:]
                if 'MinDistStarsLR' in g:
                    self.Subhalo_distances_stars_lr = g['MinDistStarsLR'][:]
                    # Calculate the minimum distance to a low-resolution particle
                    self.Subhalo_distances_lr = np.minimum(self.Subhalo_distances_p2, self.Subhalo_distances_p3, self.Subhalo_distances_stars_lr) # P2,P3,StarsLR
                else:
                    self.Subhalo_distances_stars_lr = np.zeros_like(self.Subhalo_distances_gas_lr)
                    # Calculate the minimum distance to a low-resolution particle
                    self.Subhalo_distances_lr = np.minimum(self.Subhalo_distances_p2, self.Subhalo_distances_p3) # P2,P3
                if 'MinDistStarsHR' in g:
                    self.Subhalo_distances_stars_hr = g['MinDistStarsHR'][:]
                else:
                    self.Subhalo_distances_stars_hr = 100. * np.copy(self.Subhalo_R_vir) # No HR in Rvir
                if 'MinMemberDistStarsHR' in g:
                    self.Subhalo_member_distances_stars_hr = g['MinMemberDistStarsHR'][:]
                else:
                    self.Subhalo_member_distances_stars_hr = 100. * np.copy(self.Subhalo_R_vir) # No HR in Rvir

    def convert_counts(self):
        """Convert the counts to file offsets."""
        self.first_group = np.cumsum(self.n_groups) - self.n_groups
        self.first_subhalo = np.cumsum(self.n_subhalos) - self.n_subhalos

    def periodic_distance(self, coord1, coord2):
        """Calculate the periodic distance between two coordinates."""
        delta = np.abs(coord1 - coord2)  # Absolute differences
        return np.minimum(delta, self.BoxSize - delta)  # Minimum of delta and its periodic counterpart

    def print_offsets(self):
        """Print the file counts and offsets."""
        print(f'n_groups = {self.n_groups} (n_groups_tot = {self.n_groups_tot})')
        print(f'n_subhalos = {self.n_subhalos} (n_subhalos_tot = {self.n_subhalos_tot})')
        print(f'first_group = {self.first_group} (n_groups_tot = {self.n_groups_tot})')
        print(f'first_subhalo = {self.first_subhalo} (n_subhalos_tot = {self.n_subhalos_tot})')

    def print_groups(self):
        """Print the group data."""
        if self.n_groups_tot > 0:
            print(f'Group_R_Crit200 = {self.Group_R_Crit200}')
        if SUBHALOS and self.n_subhalos_tot > 0:
            print(f'Subhalo_R_vir = {self.Subhalo_R_vir}')

    def write(self):
        """Write the candidate results to an HDF5 file."""
        # Construct the group mask
        if self.n_groups_tot > 0:
            # Mask out groups with R_Crit200 == 0 [R_Crit200 > 0]
            # Mask out groups with MinDistP2 or MinDistP3 < R_Crit200 [MinDist(P2,P3,StarsLR) > R_Crit200]
            self.group_mask = (self.Group_R_Crit200 > 0) & (self.Group_distances_lr > self.Group_R_Crit200)
            if ADD_TREE and self.TreeGroupID >= 0:
                self.group_mask[self.TreeGroupID] = True  # Ensure the main group is included
        else:
            self.n_groups_candidates = np.int32(0)  # No groups

        # Construct the subhalo mask
        if self.n_subhalos_tot > 0:
            if SUBHALOS:
                # Mask out subhalos with R_vir == 0 [R_vir > 0]
                # Mask out subhalos with MinDistP2 or MinDistP3 < R_vir [MinDist(P2,P3,StarsLR) > R_vir]
                self.subhalo_mask = (self.Subhalo_R_vir > 0) & (self.Subhalo_distances_lr > self.Subhalo_R_vir)
                if ADD_TREE and self.TreeSubhaloID >= 0:
                    self.subhalo_mask[self.TreeSubhaloID] = True  # Ensure the main subhalo is included
            else:
                self.subhalo_mask = np.array([self.group_mask[self.subhalos['SubhaloGroupNr'][i]] for i in range(self.n_subhalos_tot)], dtype=bool)
        else:
            self.n_subhalos_candidates = np.int32(0)  # No subhalos

        # Add the central subhalos of valid candidate groups (and vice versa)
        if CENTRALS and self.n_groups_tot > 0 and self.n_subhalos_tot > 0:
            # All central subhalos and host groups
            hosts_mask = (self.groups['GroupNsubs'] > 0)  # Host groups (mask) [n_groups_tot]
            # hosts = np.flatnonzero(hosts_mask).astype(np.int32)  # Host groups (indices) [n_centrals]
            centrals = self.groups['GroupFirstSub'][hosts_mask]  # Central subhalos (indices) [n_centrals]
            if np.any((centrals < 0) | (centrals >= self.n_subhalos_tot)):
                raise ValueError('Invalid central subhalo indices')  # Must be in [0, n_subhalos_tot)
            centrals_mask = np.zeros(self.n_subhalos_tot, dtype=bool)
            centrals_mask[centrals] = True  # Central subhalos (mask) [n_subhalos_tot]
            # Ensure group-based candidate central subhalos are included
            grpcand_hosts_mask = self.group_mask & hosts_mask  # Candidate host groups (mask, group-based) [n_groups_tot]
            # grpcand_hosts = np.flatnonzero(grpcand_hosts_mask).astype(np.int32)  # Candidate host groups (indices, group-based) [n_centrals_groups]
            grpcand_centrals = self.groups['GroupFirstSub'][grpcand_hosts_mask]  # Candidate central subhalos (indices, group-based) [n_centrals_groups]
            grpcand_centrals_mask = np.zeros(self.n_subhalos_tot, dtype=bool)
            grpcand_centrals_mask[grpcand_centrals] = True  # Candidate central subhalos (mask, group-based) [n_subhalos_tot]
            self.subhalo_mask[grpcand_centrals_mask] = True  # Mission accomplished!
            # Ensure subhalo-based candidate host groups are included
            subcand_centrals_mask = self.subhalo_mask & centrals_mask  # Candidate central subhalos (mask, subhalo-based) [n_subhalos_tot]
            subcand_centrals = np.flatnonzero(subcand_centrals_mask).astype(np.int32)  # Candidate central subhalos (indices, subhalo-based) [n_centrals_subhalos]
            subcand_hosts = self.subhalos['SubhaloGroupNr'][subcand_centrals]  # Candidate host groups (indices, subhalo-based) [n_centrals_subhalos]
            subcand_hosts_mask = np.zeros(self.n_groups_tot, dtype=bool)
            subcand_hosts_mask[subcand_hosts] = True  # Candidate host groups (mask, subhalo-based) [n_groups_tot]
            self.group_mask[subcand_hosts_mask] = True  # Mission accomplished!

        # Set up the counts and IDs
        if self.n_groups_tot > 0:
            self.n_groups_candidates = np.int32(np.count_nonzero(self.group_mask))
            GroupID = np.flatnonzero(self.group_mask).astype(np.int32)
            if VERBOSITY > 1:
                print(f'GroupID = {GroupID}')
                print(f'n_groups_candidates = {self.n_groups_candidates}')
        if self.n_subhalos_tot > 0:
            self.n_subhalos_candidates = np.int32(np.count_nonzero(self.subhalo_mask))
            SubhaloID = np.flatnonzero(self.subhalo_mask).astype(np.int32)
            if VERBOSITY > 1:
                print(f'SubhaloID = {SubhaloID}')
                print(f'n_subhalos_candidates = {self.n_subhalos_candidates}')
                print(f'GroupID of each Subhalo = {self.subhalos["SubhaloGroupNr"][SubhaloID]}')

        # Stricter requirements for star particles (star flag)
        global MIN_STARS  # Declare MIN_STARS as global to modify it within the function
        assert MIN_STARS >= 1  # Consistency check
        assert MAX_STARS >= MIN_STARS  # Consistency check
        assert MAX_HALOS_TO_STARS >= 1  # Consistency check
        if self.n_subhalos_candidates > 0:
            try:  # Count the number of subhalos with at least MIN_STARS star particles
                keep_counting = (MAX_STARS > MIN_STARS)  # Keep counting until the maximum number of star particles is reached
                while keep_counting:
                    guess = np.count_nonzero(self.subhalo_mask & (self.subhalos['SubhaloLenType'][:,4] >= MIN_STARS) &
                        (self.Subhalo_member_distances_stars_hr <= self.Subhalo_R_vir))  # Approximate star flag count
                    if guess < MAX_HALOS_TO_STARS * MIN_STARS:
                        keep_counting = False  # Stop counting
                    else:
                        MIN_STARS += 1  # Increment the star particle threshold
                        if MIN_STARS >= MAX_STARS:
                            keep_counting = False  # Stop counting
            except AttributeError:
                pass  # Nothing to do
        if self.n_groups_candidates > 0:
            try:  # Require a high-resolution member star within R_Crit200 and MIN_STARS star particles
                group_star_flag = (self.groups['GroupLenType'][:,4] >= MIN_STARS)  # Star flag (group)
                if ADD_TREE and self.TreeGroupID >= 0:
                    group_star_flag[self.TreeGroupID] = True  # Ensure the main group is included
                if VERBOSITY > 1:
                    print(f'n_stars_groups = {self.groups["GroupLenType"][:,4][self.group_mask]}')
                    print(f'group_min_stars_flag = {group_star_flag[self.group_mask]}')
                group_star_flag = self.group_mask & group_star_flag & (self.Group_member_distances_stars_hr <= self.Group_R_Crit200)  # Star flag (group)
                if VERBOSITY > 1:
                    print(f'group_star_flag = {group_star_flag[self.group_mask]}')
            except AttributeError:
                group_star_flag = np.zeros(self.n_groups_tot, dtype=bool)
        else:
            group_star_flag = np.zeros(self.n_groups_candidates, dtype=bool)
            self.n_groups_candidates_stars = np.int32(0)
        if self.n_subhalos_candidates > 0:
            try:  # Require a high-resolution member star within R_vir and MIN_STARS star particles
                subhalo_star_flag = (self.subhalos['SubhaloLenType'][:,4] >= MIN_STARS)  # Full flag (subhalo)
                if ADD_TREE and self.TreeSubhaloID >= 0:
                    subhalo_star_flag[self.TreeSubhaloID] = True  # Ensure the main subhalo is included
                if VERBOSITY > 1:
                    print(f'n_stars_subhalos = {self.subhalos["SubhaloLenType"][:,4][self.subhalo_mask]}')
                    print(f'subhalo_min_stars_flag = {subhalo_star_flag[self.subhalo_mask]}')
                subhalo_star_flag = self.subhalo_mask & subhalo_star_flag & (self.Subhalo_member_distances_stars_hr <= self.Subhalo_R_vir)  # Star flag (subhalo)
                if VERBOSITY > 1:
                    print(f'subhalo_star_flag = {subhalo_star_flag[self.subhalo_mask]}')
            except AttributeError:
                subhalo_star_flag = np.zeros(self.n_subhalos_tot, dtype=bool)
        else:
            subhalo_star_flag = np.zeros(self.n_subhalos_candidates, dtype=bool)
            self.n_subhalos_candidates_stars = np.int32(0)

        # Add the central subhalos of valid star flag groups (and vice versa)
        if CENTRALS and self.n_groups_candidates > 0 and self.n_subhalos_candidates > 0:
            # Ensure group-based candidate star central subhalos are included
            grpstar_hosts_mask = group_star_flag & hosts_mask  # Candidate star host groups (mask, group-based) [n_groups_tot]
            # grpstar_hosts = np.flatnonzero(grpstar_hosts_mask).astype(np.int32)  # Candidate star host groups (indices, group-based) [n_centrals_groups]
            grpstar_centrals = self.groups['GroupFirstSub'][grpstar_hosts_mask]  # Candidate star central subhalos (indices, group-based) [n_centrals_groups]
            grpstar_centrals_mask = np.zeros(self.n_subhalos_tot, dtype=bool)
            grpstar_centrals_mask[grpstar_centrals] = True  # Candidate star central subhalos (mask, group-based) [n_subhalos_tot]
            subhalo_star_flag[grpstar_centrals_mask] = True  # Mission accomplished!
            # Ensure subhalo-based candidate star host groups are included
            substar_centrals_mask = subhalo_star_flag & centrals_mask  # Candidate star central subhalos (mask, subhalo-based) [n_subhalos_tot]
            substar_centrals = np.flatnonzero(substar_centrals_mask).astype(np.int32)  # Candidate star central subhalos (indices, subhalo-based) [n_centrals_subhalos]
            substar_hosts = self.subhalos['SubhaloGroupNr'][substar_centrals]  # Candidate star host groups (indices, subhalo-based) [n_centrals_subhalos]
            substar_hosts_mask = np.zeros(self.n_groups_tot, dtype=bool)
            substar_hosts_mask[substar_hosts] = True  # Candidate star host groups (mask, subhalo-based) [n_groups_tot]
            group_star_flag[substar_hosts_mask] = True  # Mission accomplished!

        # Set up the counts (star flag)
        if self.n_groups_candidates > 0:
            group_star_flag = group_star_flag[self.group_mask]  # Restrict to candidate groups
            self.n_groups_candidates_stars = np.int32(np.count_nonzero(group_star_flag))  # Star flag (group)
            if VERBOSITY > 1:
                print(f'n_groups_candidates_stars = {self.n_groups_candidates_stars}')
        if self.n_subhalos_candidates > 0:
            subhalo_star_flag = subhalo_star_flag[self.subhalo_mask]  # Restrict to candidate subhalos
            self.n_subhalos_candidates_stars = np.int32(np.count_nonzero(subhalo_star_flag))  # Star flag (subhalo)
            if VERBOSITY > 1:
                print(f'n_subhalos_candidates_stars = {self.n_subhalos_candidates_stars}')

        # Write the results to a file
        with h5py.File(cand_file, 'w') as f:
            g = f.create_group(b'Header')
            g.attrs['Ngroups_Total'] = self.n_groups_tot
            g.attrs['Nsubhalos_Total'] = self.n_subhalos_tot
            g.attrs['Ngroups_Candidates'] = self.n_groups_candidates
            g.attrs['Nsubhalos_Candidates'] = self.n_subhalos_candidates
            if self.n_groups_candidates == 0:
                g.attrs['Ngroups_Candidates_Stars'] = self.n_groups_candidates
            if self.n_subhalos_candidates == 0:
                g.attrs['Nsubhalos_Candidates_Stars'] = self.n_subhalos_candidates
            g.attrs['Time'] = self.a
            g.attrs['Redshift'] = 1. / self.a - 1.
            g.attrs['BoxSize'] = self.BoxSize
            g.attrs['HubbleParam'] = self.h
            g.attrs['Omega0'] = self.Omega0
            g.attrs['OmegaBaryon'] = self.OmegaBaryon
            g.attrs['OmegaLambda'] = self.OmegaLambda
            g.attrs['UnitLength_in_cm'] = self.UnitLength_in_cm
            g.attrs['UnitMass_in_g'] = self.UnitMass_in_g
            g.attrs['UnitVelocity_in_cm_per_s'] = self.UnitVelocity_in_cm_per_s
            g.attrs['PosHR'] = self.r_com
            g.attrs['RadiusHR'] = self.RadiusHR
            g.attrs['RadiusLR'] = self.RadiusLR
            g.attrs['MinStars'] = np.int32(MIN_STARS)
            if self.n_groups_candidates > 0:
                g = f.create_group(b'Group')
                g.create_dataset(b'GroupID', data=GroupID)
                g.create_dataset(b'MinDistGasLR', data=self.Group_distances_gas_lr[self.group_mask])
                g.create_dataset(b'MinDistGasHR', data=self.Group_distances_gas_hr[self.group_mask])
                g.create_dataset(b'MinDistDM', data=self.Group_distances_dm[self.group_mask])
                g.create_dataset(b'MinDistP2', data=self.Group_distances_p2[self.group_mask])
                g.create_dataset(b'MinDistP3', data=self.Group_distances_p3[self.group_mask])
                g.create_dataset(b'MinDistStarsLR', data=self.Group_distances_stars_lr[self.group_mask])
                g.create_dataset(b'MinDistStarsHR', data=self.Group_distances_stars_hr[self.group_mask])
                g.create_dataset(b'MinMemberDistStarsHR', data=self.Group_member_distances_stars_hr[self.group_mask])
                f['Header'].attrs['Ngroups_Candidates_Stars'] = self.n_groups_candidates_stars
                g.create_dataset(b'StarFlag', data=group_star_flag, dtype=bool)
                if 'GroupPos' in self.group_units:
                    for key,val in self.group_units['GroupPos'].items():
                        for field in ['MinDistGasLR', 'MinDistGasHR', 'MinDistDM', 'MinDistP2', 'MinDistP3', 'MinDistStarsLR', 'MinDistStarsHR', 'MinMemberDistStarsHR']:
                            g[field].attrs[key] = val
                for field in self.groups.keys():
                    g.create_dataset(field, data=self.groups[field][self.group_mask])
                    if field in self.group_units:
                        for key,val in self.group_units[field].items():
                            g[field].attrs[key] = val
            if self.n_subhalos_candidates > 0:
                g = f.create_group(b'Subhalo')
                g.create_dataset(b'SubhaloID', data=SubhaloID)
                g.create_dataset(b'MinDistGasLR', data=self.Subhalo_distances_gas_lr[self.subhalo_mask])
                g.create_dataset(b'MinDistGasHR', data=self.Subhalo_distances_gas_hr[self.subhalo_mask])
                g.create_dataset(b'MinDistDM', data=self.Subhalo_distances_dm[self.subhalo_mask])
                g.create_dataset(b'MinDistP2', data=self.Subhalo_distances_p2[self.subhalo_mask])
                g.create_dataset(b'MinDistP3', data=self.Subhalo_distances_p3[self.subhalo_mask])
                g.create_dataset(b'MinDistStarsLR', data=self.Subhalo_distances_stars_lr[self.subhalo_mask])
                g.create_dataset(b'MinDistStarsHR', data=self.Subhalo_distances_stars_hr[self.subhalo_mask])
                g.create_dataset(b'MinMemberDistStarsHR', data=self.Subhalo_member_distances_stars_hr[self.subhalo_mask])
                if SUBHALOS:
                    g.create_dataset(b'M_gas', data=self.Subhalo_M_gas[self.subhalo_mask])
                    g.create_dataset(b'M_stars', data=self.Subhalo_M_stars[self.subhalo_mask])
                    g.create_dataset(b'M_vir', data=self.Subhalo_M_vir[self.subhalo_mask])
                    g.create_dataset(b'R_vir', data=self.Subhalo_R_vir[self.subhalo_mask])
                    if 'SubhaloMass' in self.subhalo_units:
                        for key,val in self.subhalo_units['SubhaloMass'].items():
                            for field in ['M_gas', 'M_stars', 'M_vir']:
                                g[field].attrs[key] = val
                    if 'SubhaloPos' in self.subhalo_units:
                        for key,val in self.subhalo_units['SubhaloPos'].items():
                            g['R_vir'].attrs[key] = val
                f['Header'].attrs['Nsubhalos_Candidates_Stars'] = self.n_subhalos_candidates_stars
                g.create_dataset(b'StarFlag', data=subhalo_star_flag, dtype=bool)
                if 'SubhaloPos' in self.subhalo_units:
                    for key,val in self.subhalo_units['SubhaloPos'].items():
                        for field in ['MinDistGasLR', 'MinDistGasHR', 'MinDistDM', 'MinDistP2', 'MinDistP3', 'MinDistStarsLR', 'MinDistStarsHR', 'MinMemberDistStarsHR']:
                            g[field].attrs[key] = val
                for field in self.subhalos.keys():
                    g.create_dataset(field, data=self.subhalos[field][self.subhalo_mask])
                    if field in self.subhalo_units:
                        for key,val in self.subhalo_units[field].items():
                            g[field].attrs[key] = val

    def read_counts_single(self, i):
        """Read the counts from a single FOF and snapshot file."""
        if self.n_groups_tot > 0:
            with h5py.File(fof_pre + f'{i}.hdf5', 'r') as f:
                header = f['Header'].attrs
                self.n_groups[i] = header['Ngroups_ThisFile']
                self.n_subhalos[i] = header['Nsubhalos_ThisFile']

    def read_counts(self):
        """Read the counts from the FOF and snapshot files."""
        for i in range(self.n_files):
            self.read_counts_single(i)

    def read_counts_asyncio(self):
        """Read the counts from the FOF and snapshot files using asyncio."""
        async def read_counts_async():
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                await asyncio.gather(*(loop.run_in_executor(executor, self.read_counts_single, i) for i in range(self.n_files)))
        asyncio.run(read_counts_async())

    def read_groups_single(self, i):
        """Read the group data from a single FOF file."""
        if self.n_groups_tot > 0:
            with h5py.File(fof_pre + f'{i}.hdf5', 'r') as f:
                if self.n_groups[i] > 0: # Skip empty files
                    g = f['Group']
                    offset = self.first_group[i] # Offset to the first group
                    next_offset = offset + self.n_groups[i] # Offset beyond the last group
                    for field in g.keys():
                        self.groups[field][offset:next_offset] = g[field][:]
                if self.n_subhalos[i] > 0: # Skip empty files
                    g = f['Subhalo']
                    offset = self.first_subhalo[i] # Offset to the first subhalo
                    next_offset = offset + self.n_subhalos[i] # Offset beyond the last subhalo
                    for field in g.keys():
                        self.subhalos[field][offset:next_offset] = g[field][:]

    def read_groups(self):
        """Read the group data from the FOF files."""
        for i in range(self.n_files):
            self.read_groups_single(i)

    def read_groups_asyncio(self):
        """Read the group data from the FOF files using asyncio."""
        async def read_groups_async():
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                await asyncio.gather(*(loop.run_in_executor(executor, self.read_groups_single, i) for i in range(self.n_files)))
        asyncio.run(read_groups_async())

def main():
    # Setup simulation parameters
    if TIMERS: t1 = time()
    sim = Simulation()
    print(' ___       ___  __             \n'
          '  |  |__| |__  /__`  /\\  |\\ |\n'
          '  |  |  | |___ .__/ /--\\ | \\|\n' +
          f'\nInput Directory: {out_dir}' +
          f'\nSnap {snap}: Ngroups = {sim.n_groups_tot}, Nsubhalos = {sim.n_subhalos_tot}' +
          f'\nz = {1./sim.a - 1.:g}, a = {sim.a:g}, h = {sim.h:g}, BoxSize = {1e-3*sim.BoxSize:g} cMpc/h = {1e-3*sim.BoxSize/sim.h:g} cMpc\n')
    if SUB_TIMERS: t2 = time(); print(f"Time to setup simulation: {t2 - t1:g} s"); t1 = t2

    # Read the counts from the FOF and snapshot files
    if SERIAL in READ_COUNTS:
        sim.read_counts()
        if SUB_TIMERS: t2 = time(); print(f"Time to read counts from files: {t2 - t1:g} s [serial]"); t1 = t2
    if ASYNCIO in READ_COUNTS:
        sim.read_counts_asyncio()
        if SUB_TIMERS: t2 = time(); print(f"Time to read counts from files: {t2 - t1:g} s [asyncio]"); t1 = t2
    sim.convert_counts()
    if VERBOSITY > 1: sim.print_offsets()
    if SUB_TIMERS: t2 = time(); print(f"Time to convert counts to offsets: {t2 - t1:g} s"); t1 = t2

    # Read the group data from the FOF files
    if SUB_TIMERS: print("\nReading fof data...")
    if SERIAL in READ_GROUPS:
        sim.read_groups()
        if SUB_TIMERS: t2 = time(); print(f"Time to read group data from files: {t2 - t1:g} s [serial]"); t1 = t2
    if ASYNCIO in READ_GROUPS:
        sim.read_groups_asyncio()
        if SUB_TIMERS: t2 = time(); print(f"Time to read group data from files: {t2 - t1:g} s [asyncio]"); t1 = t2
    if VERBOSITY > 1: sim.print_groups()

    # Write the results to a file
    if SUB_TIMERS: print("\nWriting the results to a file...")
    sim.write()
    print(f'Number of candidates = ({sim.n_groups_candidates} groups, {sim.n_subhalos_candidates} subhalos)')
    print(f'Number with HR stars = ({sim.n_groups_candidates_stars} groups, {sim.n_subhalos_candidates_stars} subhalos)  [MIN_STARS = {MIN_STARS}]')
    if SUB_TIMERS: t2 = time(); print(f"Time to write results to a file: {t2 - t1:g} s"); t1 = t2
    else: t2 = time(); print(f'Total time: {t2 - t1:g} s')

if __name__ == '__main__':
    main()
