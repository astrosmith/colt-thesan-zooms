#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <iostream>
#include <climits>
#include <cfloat>
#include <hdf5.h>

using std::cout;
using std::cerr;
using std::endl;
using std::string;

typedef unsigned long long myint;
#define MPI_MYINT MPI_UNSIGNED_LONG_LONG
#define H5T_MYINT H5T_NATIVE_ULLONG

// User options
#define GAS_HIGH_RES_THRESHOLD 0.5 // Threshold deliniating high and low resolution gas particles
#define SUBTIMERS 1 // Enable subtimers
#define VERBOSE 1 // Enable verbose output
#define N_PRINT 8 // Number of particles to print
// #define MPI
// #define GPU

// #define CHECK_INT_OVERFLOW(value) \
//   if (3 * (value) > static_cast<unsigned long long>(INT_MAX)) { \
//     cerr << "Error: 3 * " #value " = " << 3 * (value) << " exceeds the maximum int value" << endl; \
//     MPI_Finalize(); \
//     exit(1); \
//   }
#define CHECK_INT_OVERFLOW(value)

static int ThisTask, NTask, ThisDevice, NDevice, NumFiles, SnapNum, NTYPES;
static string file_dir, out_dir, outname = "distances";
static myint *FileCounts_Group, *FileOffsets_Group, Ngroups_Total;
static myint *FileCounts_Subhalo, *FileOffsets_Subhalo, Nsubhalos_Total;
static myint *FileCounts_Gas, *FileOffsets_Gas, NumGas_Total;
static myint *FileCounts_DM, *FileOffsets_DM, NumDM_Total;
static myint *FileCounts_P2, *FileOffsets_P2, NumP2_Total;
static myint *FileCounts_P3, *FileOffsets_P3, NumP3_Total;
static myint *FileCounts_Star, *FileOffsets_Star, NumStar_Total;
static myint n_grps, n_subs, n_gas, n_gas_hr, n_gas_lr, n_dm, n_p2, n_p3, n_stars, n_stars_hr, n_stars_lr;
static myint n3_grps, n3_subs, n3_gas, n3_gas_hr, n3_gas_lr, n3_dm, n3_p2, n3_p3, n3_stars, n3_stars_hr, n3_stars_lr;
static double MassDM, MassP3, MassHR, PosHR[3], RadiusHR, RadiusLR;
// static double Mass_Total, MassGas_Total, MassDM_Total, MassP2_Total, MassP3_Total, MassStar_Total;
#define n_bins 10000
#define n_edges 10001
#define n_bins_minus1 9999
static float Radius2Min, LogRadius2Min, InvDlogRadius2;
static float *GroupPos, *R_Crit200, *SubhaloPos, *R_vir, *M_vir, *M_gas, *M_stars, M_to_rho_vir[n_bins];
static myint *Group_FirstSub, *Group_Nsubs, *Group_NumStar, *Group_FirstStar, *Subhalo_NumStar, *Subhalo_FirstStar;
static float *Group_MinDistGasHR, *Subhalo_MinDistGasHR;
static float *Group_MinDistGasLR, *Subhalo_MinDistGasLR;
static float *Group_MinDistDM, *Subhalo_MinDistDM;
static float *Group_MinDistP2, *Subhalo_MinDistP2;
static float *Group_MinDistP3, *Subhalo_MinDistP3;
static float *Group_MinDistStarsHR, *Subhalo_MinDistStarsHR;
static float *Group_MinDistStarsLR, *Subhalo_MinDistStarsLR;
static float *Group_MinMemberDistStarsHR, *Subhalo_MinMemberDistStarsHR;
static float *r_gas, *m_gas, *m_gas_hr, *r_dm, *r_p2, *m_p2, *r_p3, *r_star, *m_star;
static float *r_gas_hr, *r_gas_lr, *r_star_hr, *r_star_lr;
static int *star_is_hr;
#ifdef GPU
static myint d_Ngroups_Total, offset_Ngroups_Total, d_Nsubhalos_Total, offset_Nsubhalos_Total;
static myint d_n_grps, d_n_subs, d_n3_grps, d_n3_subs, first_n3_grps, first_n3_subs;
static float *d_GroupPos, *d_R_Crit200, *d_SubhaloPos, *d_R_vir, *d_M_vir, *d_M_gas, *d_M_stars, *d_M_to_rho_vir;
static float *d_Group_MinDistGasHR, *d_Subhalo_MinDistGasHR;
static float *d_Group_MinDistGasLR, *d_Subhalo_MinDistGasLR;
static float *d_Group_MinDistDM, *d_Subhalo_MinDistDM;
static float *d_Group_MinDistP2, *d_Subhalo_MinDistP2;
static float *d_Group_MinDistP3, *d_Subhalo_MinDistP3;
static float *d_Group_MinDistStarsHR, *d_Subhalo_MinDistStarsHR;
static float *d_Group_MinDistStarsLR, *d_Subhalo_MinDistStarsLR;
static float *d_r_gas, *d_m_gas, *d_m_gas_hr, *d_r_dm, *d_r_p2, *d_m_p2, *d_r_p3, *d_r_star, *d_m_star;
static float *d_r_gas_hr, *d_r_gas_lr, *d_r_star_hr, *d_r_star_lr;
static int *d_star_is_hr;
#endif
static double a, BoxSize, BoxHalf, h, Omega0, OmegaBaryon, OmegaLambda;
static double UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s;

static void read_header();
static void read_file_counts();
static void calculate_file_offsets();
static void calculate_fof_offsets();
static void calculate_fof_data();
static void read_fof_data();
static void write_fof_data();
static void read_snap_data();
static void print_data();
static void setup_vir();
static void copy_hr_data();
static void calculate_distances();
#ifndef GPU
#ifdef __host__
#undef __host__
#endif
#define __host__
#ifdef __device__
#undef __device__
#endif
#define __device__
#ifdef __global__
#undef __global__
#endif
#define __global__
#endif
__global__ void calculate_minimum_distance(myint n_halo, float *r_halo, myint n_part, float *r_part, float *r2_min, float BoxSizeF, float BoxHalfF);
__global__ void calculate_maximum_distance(myint n_halo, float *r_halo, myint n_part, float *r_part, float *r2_max, float BoxSizeF, float BoxHalfF);
static double single_minimum_distance(double r[3], myint n_part, float *r_part, double BoxSize, double BoxHalf);
static double single_maximum_distance(double r[3], myint n_part, float *r_part, double BoxSize, double BoxHalf);
__global__ void calculate_R_vir(myint n_halo, float *r_halo, float *R_vir, float *M_vir, float *M_gas, float *M_stars, float M_to_rho_vir[n_bins],
                                myint n_gas, float *r_gas, float *m_gas, myint n_dm, float *r_dm, float MassDM,
                                myint n_p2, float *r_p2, float *m_p2, myint n_p3, float *r_p3, float MassP3,
                                myint n_stars, float *r_star, float *m_star,
                                float BoxSizeF, float BoxHalfF, float Radius2Min, float LogRadius2Min, float InvDlogRadius2);
static inline void equal_work(const myint n_part, const myint worker, const myint n_workers, myint& n_part_worker, myint& first_part_worker)
{
  n_part_worker = n_part / n_workers;
  first_part_worker = worker * n_part_worker;
  const myint remainder = n_part - n_part_worker * n_workers;
  if (worker < remainder) {
    n_part_worker++; // Distribute remainder
    first_part_worker += worker;
  } else {
    first_part_worker += remainder;
  }
}
__host__ __device__ float shortest_distance(float x1, float x2, float BoxSizeF, float BoxHalfF)
{
  float dx = x2 - x1;
  if (dx < -BoxHalfF)
    dx += BoxSizeF; // Periodic wrap around
  else if (dx >= BoxHalfF)
    dx -= BoxSizeF; // Periodic wrap around
  return dx; // Shortest distance
}
template <typename T>
static inline void print(T *data, const string& label, myint n_print) {
  if (n_print <= 0) {
    cout << label << " = []" << endl;
    return;
  }
  string p_str = (n_print <= N_PRINT) ? "" : ", ... "; // Print string
  if (n_print > N_PRINT)
    n_print = N_PRINT;
  cout << label << " = [" << data[0];
  for (int i = 1; i < n_print; ++i)
    cout << ", " << data[i];
  cout << p_str << "]" << endl;
};

#ifdef MPI
#include <mpi.h>

// MPI Windows
static MPI_Win win_FileCounts_Group, win_FileOffsets_Group, win_GroupPos, win_R_Crit200;
static MPI_Win win_FileCounts_Subhalo, win_FileOffsets_Subhalo, win_SubhaloPos, win_R_vir, win_M_vir, win_M_gas, win_M_stars;
static MPI_Win win_FileCounts_Gas, win_FileOffsets_Gas, win_Group_MinDistGasHR, win_Group_MinDistGasLR, win_Subhalo_MinDistGasHR, win_Subhalo_MinDistGasLR;
static MPI_Win win_FileCounts_DM, win_FileOffsets_DM, win_Group_MinDistDM, win_Subhalo_MinDistDM;
static MPI_Win win_FileCounts_P2, win_FileOffsets_P2, win_Group_MinDistP2, win_Subhalo_MinDistP2;
static MPI_Win win_FileCounts_P3, win_FileOffsets_P3, win_Group_MinDistP3, win_Subhalo_MinDistP3;
static MPI_Win win_FileCounts_Star, win_FileOffsets_Star, win_Group_MinDistStarsHR, win_Group_MinDistStarsLR, win_Subhalo_MinDistStarsHR, win_Subhalo_MinDistStarsLR;
static MPI_Win win_Group_MinMemberDistStarsHR, win_Subhalo_MinMemberDistStarsHR;
static MPI_Win win_Group_FirstSub, win_Group_Nsubs, win_Group_NumStar, win_Group_FirstStar, win_Subhalo_NumStar, win_Subhalo_FirstStar;

static MPI_Win win_r_gas, win_m_gas, win_m_gas_hr, win_r_dm, win_r_p2, win_m_p2, win_r_p3, win_r_star, win_m_star;
static MPI_Win win_r_gas_hr, win_r_gas_lr, win_r_star_hr, win_r_star_lr, win_star_is_hr;

// Shared memory allocation (with error handling)
template <typename T, typename N>
void owner_shared(T*& data, MPI_Win& win_data, const N n)
{
  const size_t size_T = sizeof(T); // Size of data type (bytes)
  const size_t size_n = static_cast<size_t>(n) * size_T; // Size of data array (bytes)
  // NumBytes += size_n; // Update total number of bytes
  // cout << "Allocating " << size_n << " bytes = " << double(size_n) / GB << " GB of shared memory ["
  //      << NumBytes << " bytes = " << double(NumBytes) / GB << " GB total]" << endl;
  int err = MPI_Win_allocate_shared(size_n, size_T, MPI_INFO_NULL, MPI_COMM_WORLD, static_cast<void*>(&data), &win_data);
  if (err != MPI_SUCCESS) {
    // Handle the error, e.g., by printing an error message and exiting the program
    char error_string[BUFSIZ];
    int length_of_error_string, error_class;
    MPI_Error_class(err, &error_class);
    MPI_Error_string(error_class, error_string, &length_of_error_string);
    cerr << error_string << endl;
    MPI_Abort(MPI_COMM_WORLD, err);
  }
}

// Shared memory query (with error handling)
template <typename T>
void query_shared(T*& data, MPI_Win& win)
{
  int disp_unit;
  MPI_Aint size;
  int err = MPI_Win_allocate_shared(0, sizeof(T), MPI_INFO_NULL, MPI_COMM_WORLD, &data, &win);
  if (err != MPI_SUCCESS) {
    char error_string[BUFSIZ];
    int length_of_error_string, error_class;
    MPI_Error_class(err, &error_class);
    MPI_Error_string(error_class, error_string, &length_of_error_string);
    cerr << error_string << endl;
    MPI_Abort(MPI_COMM_WORLD, err);
  }
  err = MPI_Win_shared_query(win, 0, &size, &disp_unit, &data);
  if (err != MPI_SUCCESS) {
    char error_string[BUFSIZ];
    int length_of_error_string, error_class;
    MPI_Error_class(err, &error_class);
    MPI_Error_string(error_class, error_string, &length_of_error_string);
    cerr << error_string << endl;
    MPI_Abort(MPI_COMM_WORLD, err);
  }
}

// Shared memory allocation (with error handling and agnostic to root or non-root)
template <typename T, typename N>
void malloc_shared(T*& data, MPI_Win& win_data, const N n)
{
  if (ThisTask == 0)
    owner_shared(data, win_data, n);
  else
    query_shared(data, win_data);
}
#else // No MPI
#ifdef MPI_COMM_WORLD
#undef MPI_COMM_WORLD
#endif
static const int MPI_COMM_WORLD = 0;
static void Dummy_MPI_Barrier(int comm) {}
static void Dummy_MPI_Finalize() {}
#define MPI_Barrier Dummy_MPI_Barrier
#define MPI_Finalize Dummy_MPI_Finalize
#endif

static void allocate()
{
#ifdef MPI
  // File data
  malloc_shared(FileCounts_Group, win_FileCounts_Group, NumFiles);
  malloc_shared(FileOffsets_Group, win_FileOffsets_Group, NumFiles);
  malloc_shared(FileCounts_Subhalo, win_FileCounts_Subhalo, NumFiles);
  malloc_shared(FileOffsets_Subhalo, win_FileOffsets_Subhalo, NumFiles);
  malloc_shared(FileCounts_Gas, win_FileCounts_Gas, NumFiles);
  malloc_shared(FileOffsets_Gas, win_FileOffsets_Gas, NumFiles);
  malloc_shared(FileCounts_DM, win_FileCounts_DM, NumFiles);
  malloc_shared(FileOffsets_DM, win_FileOffsets_DM, NumFiles);
  malloc_shared(FileCounts_P2, win_FileCounts_P2, NumFiles);
  malloc_shared(FileOffsets_P2, win_FileOffsets_P2, NumFiles);
  malloc_shared(FileCounts_P3, win_FileCounts_P3, NumFiles);
  malloc_shared(FileOffsets_P3, win_FileOffsets_P3, NumFiles);
  malloc_shared(FileCounts_Star, win_FileCounts_Star, NumFiles);
  malloc_shared(FileOffsets_Star, win_FileOffsets_Star, NumFiles);

  // Subhalo data
  malloc_shared(Group_FirstSub, win_Group_FirstSub, Ngroups_Total);
  malloc_shared(Group_Nsubs, win_Group_Nsubs, Ngroups_Total);
  malloc_shared(Group_NumStar, win_Group_NumStar, Ngroups_Total);
  malloc_shared(Group_FirstStar, win_Group_FirstStar, Ngroups_Total);
  malloc_shared(Subhalo_NumStar, win_Subhalo_NumStar, Nsubhalos_Total);
  malloc_shared(Subhalo_FirstStar, win_Subhalo_FirstStar, Nsubhalos_Total);
  malloc_shared(GroupPos, win_GroupPos, n3_grps);
  malloc_shared(R_Crit200, win_R_Crit200, Ngroups_Total);
  malloc_shared(SubhaloPos, win_SubhaloPos, n3_subs);
  malloc_shared(R_vir, win_R_vir, Nsubhalos_Total);
  malloc_shared(M_vir, win_M_vir, Nsubhalos_Total);
  malloc_shared(M_gas, win_M_gas, Nsubhalos_Total);
  malloc_shared(M_stars, win_M_stars, Nsubhalos_Total);
  malloc_shared(Group_MinDistGasHR, win_Group_MinDistGasHR, Ngroups_Total);
  malloc_shared(Subhalo_MinDistGasHR, win_Subhalo_MinDistGasHR, Nsubhalos_Total);
  malloc_shared(Group_MinDistGasLR, win_Group_MinDistGasLR, Ngroups_Total);
  malloc_shared(Subhalo_MinDistGasLR, win_Subhalo_MinDistGasLR, Nsubhalos_Total);
  malloc_shared(Group_MinDistDM, win_Group_MinDistDM, Ngroups_Total);
  malloc_shared(Subhalo_MinDistDM, win_Subhalo_MinDistDM, Nsubhalos_Total);
  malloc_shared(Group_MinDistP2, win_Group_MinDistP2, Ngroups_Total);
  malloc_shared(Subhalo_MinDistP2, win_Subhalo_MinDistP2, Nsubhalos_Total);
  malloc_shared(Group_MinDistP3, win_Group_MinDistP3, Ngroups_Total);
  malloc_shared(Subhalo_MinDistP3, win_Subhalo_MinDistP3, Nsubhalos_Total);
  malloc_shared(Group_MinDistStarsHR, win_Group_MinDistStarsHR, Ngroups_Total);
  malloc_shared(Subhalo_MinDistStarsHR, win_Subhalo_MinDistStarsHR, Nsubhalos_Total);
  malloc_shared(Group_MinDistStarsLR, win_Group_MinDistStarsLR, Ngroups_Total);
  malloc_shared(Subhalo_MinDistStarsLR, win_Subhalo_MinDistStarsLR, Nsubhalos_Total);
  malloc_shared(Group_MinMemberDistStarsHR, win_Group_MinMemberDistStarsHR, Ngroups_Total);
  malloc_shared(Subhalo_MinMemberDistStarsHR, win_Subhalo_MinMemberDistStarsHR, Nsubhalos_Total);

  // Particle data
  malloc_shared(r_gas, win_r_gas, n3_gas);
  malloc_shared(m_gas, win_m_gas, n_gas);
  malloc_shared(m_gas_hr, win_m_gas_hr, n_gas);
  malloc_shared(r_dm, win_r_dm, n3_dm);
  malloc_shared(r_p2, win_r_p2, n3_p2);
  malloc_shared(m_p2, win_m_p2, n_p2);
  malloc_shared(r_p3, win_r_p3, n3_p3);
  if (n_stars > 0) {
    malloc_shared(r_star, win_r_star, n3_stars);
    malloc_shared(m_star, win_m_star, n_stars);
    malloc_shared(star_is_hr, win_star_is_hr, n_stars);
  }
#else
  // File data
  FileCounts_Group = (myint *) malloc(NumFiles * sizeof(myint));
  FileOffsets_Group = (myint *) malloc(NumFiles * sizeof(myint));
  FileCounts_Subhalo = (myint *) malloc(NumFiles * sizeof(myint));
  FileOffsets_Subhalo = (myint *) malloc(NumFiles * sizeof(myint));
  FileCounts_Gas = (myint *) malloc(NumFiles * sizeof(myint));
  FileOffsets_Gas = (myint *) malloc(NumFiles * sizeof(myint));
  FileCounts_DM = (myint *) malloc(NumFiles * sizeof(myint));
  FileOffsets_DM = (myint *) malloc(NumFiles * sizeof(myint));
  FileCounts_P2 = (myint *) malloc(NumFiles * sizeof(myint));
  FileOffsets_P2 = (myint *) malloc(NumFiles * sizeof(myint));
  FileCounts_P3 = (myint *) malloc(NumFiles * sizeof(myint));
  FileOffsets_P3 = (myint *) malloc(NumFiles * sizeof(myint));
  FileCounts_Star = (myint *) malloc(NumFiles * sizeof(myint));
  FileOffsets_Star = (myint *) malloc(NumFiles * sizeof(myint));

  // Subhalo data
  Group_FirstSub = (myint *) malloc(Ngroups_Total * sizeof(myint));
  Group_Nsubs = (myint *) malloc(Ngroups_Total * sizeof(myint));
  Group_NumStar = (myint *) malloc(Ngroups_Total * sizeof(myint));
  Group_FirstStar = (myint *) malloc(Ngroups_Total * sizeof(myint));
  Subhalo_NumStar = (myint *) malloc(Nsubhalos_Total * sizeof(myint));
  Subhalo_FirstStar = (myint *) malloc(Nsubhalos_Total * sizeof(myint));
  GroupPos = (float *) malloc(n3_grps * sizeof(float));
  R_Crit200 = (float *) malloc(Ngroups_Total * sizeof(float));
  SubhaloPos = (float *) malloc(n3_subs * sizeof(float));
  R_vir = (float *) malloc(Nsubhalos_Total * sizeof(float));
  M_vir = (float *) malloc(Nsubhalos_Total * sizeof(float));
  M_gas = (float *) malloc(Nsubhalos_Total * sizeof(float));
  M_stars = (float *) malloc(Nsubhalos_Total * sizeof(float));
  Group_MinDistGasHR = (float *) malloc(Ngroups_Total * sizeof(float));
  Subhalo_MinDistGasHR = (float *) malloc(Nsubhalos_Total * sizeof(float));
  Group_MinDistGasLR = (float *) malloc(Ngroups_Total * sizeof(float));
  Subhalo_MinDistGasLR = (float *) malloc(Nsubhalos_Total * sizeof(float));
  Group_MinDistDM = (float *) malloc(Ngroups_Total * sizeof(float));
  Subhalo_MinDistDM = (float *) malloc(Nsubhalos_Total * sizeof(float));
  Group_MinDistP2 = (float *) malloc(Ngroups_Total * sizeof(float));
  Subhalo_MinDistP2 = (float *) malloc(Nsubhalos_Total * sizeof(float));
  Group_MinDistP3 = (float *) malloc(Ngroups_Total * sizeof(float));
  Subhalo_MinDistP3 = (float *) malloc(Nsubhalos_Total * sizeof(float));
  Group_MinDistStarsHR = (float *) malloc(Ngroups_Total * sizeof(float));
  Subhalo_MinDistStarsHR = (float *) malloc(Nsubhalos_Total * sizeof(float));
  Group_MinDistStarsLR = (float *) malloc(Ngroups_Total * sizeof(float));
  Subhalo_MinDistStarsLR = (float *) malloc(Nsubhalos_Total * sizeof(float));
  Group_MinMemberDistStarsHR = (float *) malloc(Ngroups_Total * sizeof(float));
  Subhalo_MinMemberDistStarsHR = (float *) malloc(Nsubhalos_Total * sizeof(float));

  // Particle data
  r_gas = (float *) malloc(n3_gas * sizeof(float));
  m_gas = (float *) malloc(n_gas * sizeof(float));
  m_gas_hr = (float *) malloc(n_gas * sizeof(float));
  r_dm = (float *) malloc(n3_dm * sizeof(float));
  r_p2 = (float *) malloc(n3_p2 * sizeof(float));
  m_p2 = (float *) malloc(n_p2 * sizeof(float));
  r_p3 = (float *) malloc(n3_p3 * sizeof(float));
  if (n_stars > 0) {
    r_star = (float *) malloc(n3_stars * sizeof(float));
    m_star = (float *) malloc(n_stars * sizeof(float));
    star_is_hr = (int *) malloc(n_stars * sizeof(int));
  }
#endif

#ifdef GPU
  // Allocate GPU memory
  if (ThisTask < NDevice) {
    // Subhalo data
    cudaMalloc(&d_GroupPos, d_n3_grps * sizeof(float));
    cudaMalloc(&d_R_Crit200, d_Ngroups_Total * sizeof(float));
    cudaMalloc(&d_SubhaloPos, d_n3_subs * sizeof(float));
    cudaMalloc(&d_R_vir, d_Nsubhalos_Total * sizeof(float));
    cudaMalloc(&d_M_vir, d_Nsubhalos_Total * sizeof(float));
    cudaMalloc(&d_M_gas, d_Nsubhalos_Total * sizeof(float));
    cudaMalloc(&d_M_stars, d_Nsubhalos_Total * sizeof(float));
    cudaMalloc(&d_M_to_rho_vir, n_bins * sizeof(float));
    cudaMalloc(&d_Group_MinDistGasHR, d_Ngroups_Total * sizeof(float));
    cudaMalloc(&d_Subhalo_MinDistGasHR, d_Nsubhalos_Total * sizeof(float));
    cudaMalloc(&d_Group_MinDistGasLR, d_Ngroups_Total * sizeof(float));
    cudaMalloc(&d_Subhalo_MinDistGasLR, d_Nsubhalos_Total * sizeof(float));
    cudaMalloc(&d_Group_MinDistDM, d_Ngroups_Total * sizeof(float));
    cudaMalloc(&d_Subhalo_MinDistDM, d_Nsubhalos_Total * sizeof(float));
    cudaMalloc(&d_Group_MinDistP2, d_Ngroups_Total * sizeof(float));
    cudaMalloc(&d_Subhalo_MinDistP2, d_Nsubhalos_Total * sizeof(float));
    cudaMalloc(&d_Group_MinDistP3, d_Ngroups_Total * sizeof(float));
    cudaMalloc(&d_Subhalo_MinDistP3, d_Nsubhalos_Total * sizeof(float));
    cudaMalloc(&d_Group_MinDistStarsHR, d_Ngroups_Total * sizeof(float));
    cudaMalloc(&d_Subhalo_MinDistStarsHR, d_Nsubhalos_Total * sizeof(float));
    cudaMalloc(&d_Group_MinDistStarsLR, d_Ngroups_Total * sizeof(float));
    cudaMalloc(&d_Subhalo_MinDistStarsLR, d_Nsubhalos_Total * sizeof(float));

    // Particle data
    cudaMalloc(&d_r_gas, n3_gas * sizeof(float));
    cudaMalloc(&d_m_gas, n_gas * sizeof(float));
    cudaMalloc(&d_m_gas_hr, n_gas * sizeof(float));
    cudaMalloc(&d_r_dm, n3_dm * sizeof(float));
    cudaMalloc(&d_r_p2, n3_p2 * sizeof(float));
    cudaMalloc(&d_m_p2, n_p2 * sizeof(float));
    cudaMalloc(&d_r_p3, n3_p3 * sizeof(float));
    if (n_stars > 0) {
      cudaMalloc(&d_r_star, n3_stars * sizeof(float));
      cudaMalloc(&d_m_star, n_stars * sizeof(float));
      cudaMalloc(&d_star_is_hr, n_stars * sizeof(int));
    }
  }
#endif
}

static void free()
{
#ifdef GPU
  if (ThisTask < NDevice) {
    // HR data
    if (n_stars_lr > 0)
      cudaFree(d_r_star_lr);
    if (n_stars_hr > 0)
      cudaFree(d_r_star_hr);
    if (n_gas_lr > 0)
      cudaFree(d_r_gas_lr);
    if (n_gas_hr > 0)
      cudaFree(d_r_gas_hr);

    // Particle data
    if (n_stars > 0) {
      cudaFree(d_star_is_hr);
      cudaFree(d_m_star);
      cudaFree(d_r_star);
    }
    cudaFree(d_r_p3);
    cudaFree(d_m_p2);
    cudaFree(d_r_p2);
    cudaFree(d_r_dm);
    cudaFree(d_m_gas_hr);
    cudaFree(d_m_gas);
    cudaFree(d_r_gas);

    // Subhalo data
    cudaFree(d_Subhalo_MinDistStarsLR);
    cudaFree(d_Group_MinDistStarsLR);
    cudaFree(d_Subhalo_MinDistStarsHR);
    cudaFree(d_Group_MinDistStarsHR);
    cudaFree(d_Subhalo_MinDistP3);
    cudaFree(d_Group_MinDistP3);
    cudaFree(d_Subhalo_MinDistP2);
    cudaFree(d_Group_MinDistP2);
    cudaFree(d_Subhalo_MinDistDM);
    cudaFree(d_Group_MinDistDM);
    cudaFree(d_Subhalo_MinDistGasLR);
    cudaFree(d_Group_MinDistGasLR);
    cudaFree(d_Subhalo_MinDistGasHR);
    cudaFree(d_Group_MinDistGasHR);
    cudaFree(d_M_to_rho_vir);
    cudaFree(d_M_stars);
    cudaFree(d_M_gas);
    cudaFree(d_M_vir);
    cudaFree(d_R_vir);
    cudaFree(d_SubhaloPos);
    cudaFree(d_R_Crit200);
    cudaFree(d_GroupPos);
  }
#endif

#ifdef MPI
  // HR data
  if (n_stars_lr > 0)
    MPI_Win_free(&win_r_star_lr);
  if (n_stars_hr > 0)
    MPI_Win_free(&win_r_star_hr);
  if (n_gas_lr > 0)
    MPI_Win_free(&win_r_gas_lr);
  if (n_gas_hr > 0)
    MPI_Win_free(&win_r_gas_hr);

  // Particle data
  if (n_stars > 0) {
    MPI_Win_free(&win_star_is_hr);
    MPI_Win_free(&win_m_star);
    MPI_Win_free(&win_r_star);
  }
  MPI_Win_free(&win_r_p3);
  MPI_Win_free(&win_m_p2);
  MPI_Win_free(&win_r_p2);
  MPI_Win_free(&win_r_dm);
  MPI_Win_free(&win_m_gas_hr);
  MPI_Win_free(&win_m_gas);
  MPI_Win_free(&win_r_gas);

  // Subhalo data
  MPI_Win_free(&win_Subhalo_MinMemberDistStarsHR);
  MPI_Win_free(&win_Group_MinMemberDistStarsHR);
  MPI_Win_free(&win_Subhalo_MinDistStarsLR);
  MPI_Win_free(&win_Group_MinDistStarsLR);
  MPI_Win_free(&win_Subhalo_MinDistStarsHR);
  MPI_Win_free(&win_Group_MinDistStarsHR);
  MPI_Win_free(&win_Subhalo_MinDistP3);
  MPI_Win_free(&win_Group_MinDistP3);
  MPI_Win_free(&win_Subhalo_MinDistP2);
  MPI_Win_free(&win_Group_MinDistP2);
  MPI_Win_free(&win_Subhalo_MinDistDM);
  MPI_Win_free(&win_Group_MinDistDM);
  MPI_Win_free(&win_Subhalo_MinDistGasLR);
  MPI_Win_free(&win_Group_MinDistGasLR);
  MPI_Win_free(&win_Subhalo_MinDistGasHR);
  MPI_Win_free(&win_Group_MinDistGasHR);
  MPI_Win_free(&win_M_stars);
  MPI_Win_free(&win_M_gas);
  MPI_Win_free(&win_M_vir);
  MPI_Win_free(&win_R_vir);
  MPI_Win_free(&win_SubhaloPos);
  MPI_Win_free(&win_R_Crit200);
  MPI_Win_free(&win_GroupPos);
  MPI_Win_free(&win_Subhalo_FirstStar);
  MPI_Win_free(&win_Subhalo_NumStar);
  MPI_Win_free(&win_Group_FirstStar);
  MPI_Win_free(&win_Group_NumStar);
  MPI_Win_free(&win_Group_Nsubs);
  MPI_Win_free(&win_Group_FirstSub);

  // File data
  MPI_Win_free(&win_FileOffsets_Star);
  MPI_Win_free(&win_FileCounts_Star);
  MPI_Win_free(&win_FileOffsets_P3);
  MPI_Win_free(&win_FileCounts_P3);
  MPI_Win_free(&win_FileOffsets_P2);
  MPI_Win_free(&win_FileCounts_P2);
  MPI_Win_free(&win_FileOffsets_DM);
  MPI_Win_free(&win_FileCounts_DM);
  MPI_Win_free(&win_FileOffsets_Gas);
  MPI_Win_free(&win_FileCounts_Gas);
  MPI_Win_free(&win_FileOffsets_Subhalo);
  MPI_Win_free(&win_FileCounts_Subhalo);
  MPI_Win_free(&win_FileOffsets_Group);
  MPI_Win_free(&win_FileCounts_Group);
#else
  // HR data
  if (n_stars_lr > 0)
    free(r_star_lr);
  if (n_stars_hr > 0)
    free(r_star_hr);
  if (n_gas_lr > 0)
    free(r_gas_lr);
  if (n_gas_hr > 0)
    free(r_gas_hr);

  // Particle data
  if (n_stars > 0) {
    free(star_is_hr);
    free(m_star);
    free(r_star);
  }
  free(r_p3);
  free(m_p2);
  free(r_p2);
  free(r_dm);
  free(m_gas_hr);
  free(m_gas);
  free(r_gas);

  // Subhalo data
  free(Subhalo_MinMemberDistStarsHR);
  free(Group_MinMemberDistStarsHR);
  free(Subhalo_MinDistStarsLR);
  free(Group_MinDistStarsLR);
  free(Subhalo_MinDistStarsHR);
  free(Group_MinDistStarsHR);
  free(Subhalo_MinDistP3);
  free(Group_MinDistP3);
  free(Subhalo_MinDistP2);
  free(Group_MinDistP2);
  free(Subhalo_MinDistDM);
  free(Group_MinDistDM);
  free(Subhalo_MinDistGasLR);
  free(Group_MinDistGasLR);
  free(Subhalo_MinDistGasHR);
  free(Group_MinDistGasHR);
  free(M_stars);
  free(M_gas);
  free(M_vir);
  free(R_vir);
  free(SubhaloPos);
  free(R_Crit200);
  free(GroupPos);
  free(Subhalo_FirstStar);
  free(Subhalo_NumStar);
  free(Group_FirstStar);
  free(Group_NumStar);
  free(Group_Nsubs);
  free(Group_FirstSub);

  // File data
  free(FileOffsets_Star);
  free(FileCounts_Star);
  free(FileOffsets_P3);
  free(FileCounts_P3);
  free(FileOffsets_P2);
  free(FileCounts_P2);
  free(FileOffsets_DM);
  free(FileCounts_DM);
  free(FileOffsets_Gas);
  free(FileCounts_Gas);
  free(FileOffsets_Subhalo);
  free(FileCounts_Subhalo);
  free(FileOffsets_Group);
  free(FileCounts_Group);
#endif
}

int main(int argc, char **argv)
{
#ifdef MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  MPI_Comm_size(MPI_COMM_WORLD, &NTask);
#else
  ThisTask = 0; // Serial
  NTask = 1;
#endif
#ifdef GPU
  cudaGetDeviceCount(&NDevice);
  ThisDevice = ThisTask % NDevice;
  cudaSetDevice(ThisDevice);
#endif

  clock_t begin = clock();
  clock_t start = begin, stop;

  if (argc == 3) {
    file_dir = argv[1];
    size_t pos = file_dir.find_last_of('/');
    out_dir = (pos != string::npos) ? file_dir.substr(0, pos) + "/postprocessing/" + outname : outname;
    // out_dir = "."; // Testing
    SnapNum = std::stoi(argv[2]); // <input directory> <snapshot number>
  } else if (argc == 4) {
    file_dir = argv[1];
    out_dir = argv[2]; // <input directory> <output directory> <snapshot number>
    SnapNum = std::stoi(argv[3]);
  } else {
    cerr << "Usage: " << argv[0] << " <input directory> <output directory> <snapshot number>" << endl;
    MPI_Finalize();
    exit(1);
  }
  // Ensure the output directory exists
  {
    string cmd = "mkdir -p " + out_dir;
    system(cmd.c_str());
  }

  // Read Header
  read_header();

  if (ThisTask == 0) {
    cout << " ___       ___  __             \n"
         << "  |  |__| |__  /__`  /\\  |\\ |\n"
         << "  |  |  | |___ .__/ /--\\ | \\|\n"
         << "\nInput  Directory: " << file_dir
         << "\nOutput Directory: " << out_dir
         << "\n\nSnap " << SnapNum << ": NumFiles = " << NumFiles << ", Ngroups = " << Ngroups_Total << ", Nsubhalos = " << Nsubhalos_Total
#ifdef MPI
         << "  (NTask = " << NTask << ")"
#endif
#ifdef GPU
         << "  (NDevice = " << NDevice << ")"
#endif
         << "\nNumGas = " << NumGas_Total << ", NumDM = " << NumDM_Total
         << ", NumP2 = " << NumP2_Total << ", NumP3 = " << NumP3_Total << ", NumStar = " << NumStar_Total
         << "\nz = " << 1./a - 1. << ", a = " << a << ", h = " << h
         << ", BoxSize = " << 1e-3*BoxSize << " cMpc/h = " << 1e-3*BoxSize/h << " cMpc" << endl;
  }

  // Allocate memory
  allocate();
  MPI_Barrier(MPI_COMM_WORLD);
  if (SUBTIMERS && ThisTask == 0) {
    stop = clock(); double time_spent = ((double) (stop - start)) / CLOCKS_PER_SEC; start = stop;
    cout << "\nTime spent on memory allocation: " << time_spent << " s" << endl;
  }

  // Read file counts
  read_file_counts();
  MPI_Barrier(MPI_COMM_WORLD);

  // Calculate the file offsets (Note: needs to be done in serial)
  if (ThisTask == 0)
    calculate_file_offsets();
  MPI_Barrier(MPI_COMM_WORLD);

  // Read fof and snap data
  read_fof_data();
  MPI_Barrier(MPI_COMM_WORLD);
  if (SUBTIMERS && ThisTask == 0) {
    stop = clock(); double time_spent = ((double) (stop - start)) / CLOCKS_PER_SEC; start = stop;
    cout << "\nTime spent on reading fof data: " << time_spent << " s" << endl;
  }
  read_snap_data();
  MPI_Barrier(MPI_COMM_WORLD);
  if (SUBTIMERS && ThisTask == 0) {
    stop = clock(); double time_spent = ((double) (stop - start)) / CLOCKS_PER_SEC; start = stop;
    cout << "\nTime spent on reading snapshot data: " << time_spent << " s" << endl;
  }
  // Calculate group and subhalo offsets (Note: needs to be done in serial)
  if (ThisTask == 0)
    calculate_fof_offsets();
  MPI_Barrier(MPI_COMM_WORLD);
  if (SUBTIMERS && ThisTask == 0) {
    stop = clock(); double time_spent = ((double) (stop - start)) / CLOCKS_PER_SEC; start = stop;
    cout << "\nTime spent on calculating halo offsets: " << time_spent << " s" << endl;
  }
  calculate_fof_data();
  MPI_Barrier(MPI_COMM_WORLD);
  if (SUBTIMERS && ThisTask == 0) {
    stop = clock(); double time_spent = ((double) (stop - start)) / CLOCKS_PER_SEC; start = stop;
    cout << "\nTime spent on calculating halo data: " << time_spent << " s" << endl;
  }
  setup_vir();
  MPI_Barrier(MPI_COMM_WORLD);
  if (SUBTIMERS && ThisTask == 0) {
    stop = clock(); double time_spent = ((double) (stop - start)) / CLOCKS_PER_SEC; start = stop;
    cout << "\nTime spent on setting up virial radius calculations: " << time_spent << " s" << endl;
  }
  copy_hr_data();
  MPI_Barrier(MPI_COMM_WORLD);
  if (SUBTIMERS && ThisTask == 0) {
    stop = clock(); double time_spent = ((double) (stop - start)) / CLOCKS_PER_SEC; start = stop;
    cout << "\nTime spent on copying HR data: " << time_spent << " s" << endl;
  }

  // Calculate minimum distances between halos and particles
#ifdef GPU
  if (ThisTask < NDevice) {
    // Copy data to GPUs
    cudaMemcpy(d_GroupPos, &GroupPos[first_n3_grps], d_n3_grps * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SubhaloPos, &SubhaloPos[first_n3_subs], n3_subs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_gas, r_gas, n3_gas * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_dm, r_dm, n3_dm * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_p2, r_p2, n3_p2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_p3, r_p3, n3_p3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_gas, m_gas, n_gas * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_p2, m_p2, n_p2 * sizeof(float), cudaMemcpyHostToDevice);
    if (n_stars > 0) {
      cudaMemcpy(d_r_star, r_star, n3_stars * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_m_star, m_star, n_stars * sizeof(float), cudaMemcpyHostToDevice);
    }
    // Perform calculations on the GPU
    calculate_distances();
    // Copy data back to CPU
    cudaMemcpy(&Group_MinDistGasHR[offset_Ngroups_Total], d_Group_MinDistGasHR, Ngroups_Total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&Subhalo_MinDistGasHR[offset_Nsubhalos_Total], d_Subhalo_MinDistGasHR, Nsubhalos_Total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&Group_MinDistGasLR[offset_Ngroups_Total], d_Group_MinDistGasLR, Ngroups_Total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&Subhalo_MinDistGasLR[offset_Nsubhalos_Total], d_Subhalo_MinDistGasLR, Nsubhalos_Total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&Group_MinDistDM[offset_Ngroups_Total], d_Group_MinDistDM, Ngroups_Total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&Subhalo_MinDistDM[offset_Nsubhalos_Total], d_Subhalo_MinDistDM, Nsubhalos_Total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&Group_MinDistP2[offset_Ngroups_Total], d_Group_MinDistP2, Ngroups_Total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&Subhalo_MinDistP2[offset_Nsubhalos_Total], d_Subhalo_MinDistP2, Nsubhalos_Total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&Group_MinDistP3[offset_Ngroups_Total], d_Group_MinDistP3, Ngroups_Total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&Subhalo_MinDistP3[offset_Nsubhalos_Total], d_Subhalo_MinDistP3, Nsubhalos_Total * sizeof(float), cudaMemcpyDeviceToHost);
    if (n_stars_hr > 0) {
      cudaMemcpy(&Group_MinDistStarsHR[offset_Ngroups_Total], d_Group_MinDistStarsHR, Ngroups_Total * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(&Subhalo_MinDistStarsHR[offset_Nsubhalos_Total], d_Subhalo_MinDistStarsHR, Nsubhalos_Total * sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (n_stars_lr > 0) {
      cudaMemcpy(&Group_MinDistStarsLR[offset_Ngroups_Total], d_Group_MinDistStarsLR, Ngroups_Total * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(&Subhalo_MinDistStarsLR[offset_Nsubhalos_Total], d_Subhalo_MinDistStarsLR, Nsubhalos_Total * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(&R_vir[offset_Nsubhalos_Total], d_R_vir, Nsubhalos_Total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&M_vir[offset_Nsubhalos_Total], d_M_vir, Nsubhalos_Total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&M_gas[offset_Nsubhalos_Total], d_M_gas, Nsubhalos_Total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&M_stars[offset_Nsubhalos_Total], d_M_stars, Nsubhalos_Total * sizeof(float), cudaMemcpyDeviceToHost);
  }
#else
  calculate_distances();
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  if (SUBTIMERS && ThisTask == 0) {
    stop = clock(); double time_spent = ((double) (stop - start)) / CLOCKS_PER_SEC; start = stop;
    cout << "\nTime spent on calculating distances: " << time_spent << " s" << endl;
  }

  // Calculate the closest low-resolution distance from the center of mass of the high-resolution region
  const double max_gas_hr = single_maximum_distance(PosHR, n_gas_hr, r_gas_hr, BoxSize, BoxHalf);
  const double min_gas_lr = single_minimum_distance(PosHR, n_gas_lr, r_gas_lr, BoxSize, BoxHalf);
  const double max_dm = single_maximum_distance(PosHR, n_dm, r_dm, BoxSize, BoxHalf);
  const double min_p2 = single_minimum_distance(PosHR, n_p2, r_p2, BoxSize, BoxHalf);
  const double min_p3 = single_minimum_distance(PosHR, n_p3, r_p3, BoxSize, BoxHalf);
  const double max_star_hr = (n_stars_hr > 0) ? single_maximum_distance(PosHR, n_stars_hr, r_star_hr, BoxSize, BoxHalf) : 0.;
  const double min_star_lr = (n_stars_lr > 0) ? single_minimum_distance(PosHR, n_stars_lr, r_star_lr, BoxSize, BoxHalf) : DBL_MAX;
  RadiusHR = std::max(max_gas_hr, std::max(max_dm, max_star_hr));
  RadiusLR = std::min(min_gas_lr, std::min(min_p2, std::min(min_p3, min_star_lr)));

  if (VERBOSE && ThisTask == 0) {
    cout << endl;
    const myint n_skip = (Nsubhalos_Total > 20) ? Nsubhalos_Total / 10 : 1;
    for (myint i = 0; i < Nsubhalos_Total; i += n_skip)
      cout << "i = " << i << ", min subhalo distance = " << Subhalo_MinDistP2[i] << endl;
    cout << "\nCenter of mass of the high-resolution region = (" << PosHR[0] << ", " << PosHR[1] << ", " << PosHR[2] << ") ckpc/h, mass = " << MassHR << " 10^10 Msun/h" << endl;
    cout << "Radius containing all high-resolution particles = " << RadiusHR << " ckpc/h" << endl;
    cout << "Radius containing no low-resolution particles = " << RadiusLR << " ckpc/h" << endl;
    cout << "First subhalo R_vir = " << R_vir[0] << ", M_vir = " << M_vir[0] << ", M_gas = " << M_gas[0] << ", M_stars = " << M_stars[0] << endl;
  }

  // Count the number of candidate groups and subhalos
  myint Ngroups_Candidates = 0, Nsubhalos_Candidates = 0, Ngroups_Candidates_Stars = 0, Nsubhalos_Candidates_Stars = 0;
  if (ThisTask == 0) {
    for (myint i = 0; i < Ngroups_Total; i++) {
      if (R_Crit200[i] > 0. && Group_MinDistP2[i] > R_Crit200[i] && Group_MinDistP3[i] > R_Crit200[i] && Group_MinDistStarsLR[i] > R_Crit200[i]) {
        Ngroups_Candidates++; // Candidate group
        if (Group_MinMemberDistStarsHR[i] < R_Crit200[i]) {
          Ngroups_Candidates_Stars++; // Candidate group with stars
          if (Ngroups_Candidates_Stars < 10)
            cout << "Group " << i << " has stars within R_vir = " << R_Crit200[i] << " > " << Group_MinMemberDistStarsHR[i] << endl;
        } else {
          if (Ngroups_Candidates - Ngroups_Candidates_Stars < 10)
            cout << "Group " << i << " has no stars within R_vir = " << R_Crit200[i] << " > " << Group_MinMemberDistStarsHR[i] << endl;
        }
      }
    }
    for (myint i = 0; i < Nsubhalos_Total; i++) {
      if (R_vir[i] > 0. && Subhalo_MinDistP2[i] > R_vir[i] && Subhalo_MinDistP3[i] > R_vir[i] && Subhalo_MinDistStarsLR[i] > R_vir[i]) {
        Nsubhalos_Candidates++; // Candidate subhalo
        if (Subhalo_MinMemberDistStarsHR[i] < R_vir[i]) {
          Nsubhalos_Candidates_Stars++; // Candidate subhalo with stars
          if (Nsubhalos_Candidates_Stars < 10)
            cout << "Subhalo " << i << " has stars within R_vir = " << R_vir[i] << " > " << Subhalo_MinMemberDistStarsHR[i] << endl;
        } else {
          if (Nsubhalos_Candidates - Nsubhalos_Candidates_Stars < 10)
            cout << "Subhalo " << i << " has no stars within R_vir = " << R_vir[i] << " > " << Subhalo_MinMemberDistStarsHR[i] << endl;
        }
      }
    }
  }

  // Write the catalog data
  if (ThisTask == 0)
    write_fof_data();
  MPI_Barrier(MPI_COMM_WORLD);
  if (SUBTIMERS && ThisTask == 0) {
    stop = clock(); double time_spent = ((double) (stop - start)) / CLOCKS_PER_SEC; start = stop;
    if (VERBOSE)
      print_data();
    cout << "\nTime spent on writing catalog: " << time_spent << " s" << endl;
  }

  // Free memory
  free();
  MPI_Barrier(MPI_COMM_WORLD);
  if (SUBTIMERS && ThisTask == 0) {
    stop = clock(); double time_spent = ((double) (stop - start)) / CLOCKS_PER_SEC; start = stop;
    cout << "\nTime spent on freeing memory: " << time_spent << " s" << endl;
  }

  MPI_Finalize();

  if (ThisTask == 0) {
    double time_spent = ((double) (clock() - begin)) / CLOCKS_PER_SEC;
    cout << "\nNumber of candidates = (" << Ngroups_Candidates << " groups, " << Nsubhalos_Candidates << " subhalos)"
         << "\nNumber with HR stars = (" << Ngroups_Candidates_Stars << " groups, " << Nsubhalos_Candidates_Stars << " subhalos)"
         << "\n\nTotal runtime: " << time_spent << " s" << endl;
  }

  return 0;
}


static void read_header()
{
  // Read header info from the group files
  if (ThisTask == 0) {
    std::ostringstream oss;
    oss << file_dir << "/groups_" << std::setfill('0') << std::setw(3) << SnapNum
        << "/fof_subhalo_tab_" << std::setfill('0') << std::setw(3) << SnapNum << ".0.hdf5";
    string fname = oss.str();
    hid_t file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    // Header
    hid_t group_id = H5Gopen(file_id, "Header", H5P_DEFAULT);

    hid_t attribute_id = H5Aopen_name(group_id, "NumFiles");
    H5Aread(attribute_id, H5T_NATIVE_INT, &NumFiles);
    H5Aclose(attribute_id);

    attribute_id = H5Aopen_name(group_id, "Ngroups_Total");
    H5Aread(attribute_id, H5T_MYINT, &Ngroups_Total);
    H5Aclose(attribute_id);
    CHECK_INT_OVERFLOW(Ngroups_Total);

    attribute_id = H5Aopen_name(group_id, "Nsubhalos_Total");
    H5Aread(attribute_id, H5T_MYINT, &Nsubhalos_Total);
    H5Aclose(attribute_id);
    CHECK_INT_OVERFLOW(Nsubhalos_Total);

    attribute_id = H5Aopen_name(group_id, "Time");
    H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &a);
    H5Aclose(attribute_id);

    H5Gclose(group_id);

    // Config
    group_id = H5Gopen(file_id, "Config", H5P_DEFAULT);

    attribute_id = H5Aopen_name(group_id, "NTYPES");
    H5Aread(attribute_id, H5T_NATIVE_INT, &NTYPES);
    H5Aclose(attribute_id);

    H5Gclose(group_id);

    // Parameters
    group_id = H5Gopen(file_id, "Parameters", H5P_DEFAULT);

    attribute_id = H5Aopen_name(group_id, "BoxSize");
    H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &BoxSize);
    H5Aclose(attribute_id);

    attribute_id = H5Aopen_name(group_id, "HubbleParam");
    H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &h);
    H5Aclose(attribute_id);

    attribute_id = H5Aopen_name(group_id, "Omega0");
    H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &Omega0);
    H5Aclose(attribute_id);

    attribute_id = H5Aopen_name(group_id, "OmegaBaryon");
    H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &OmegaBaryon);
    H5Aclose(attribute_id);

    attribute_id = H5Aopen_name(group_id, "OmegaLambda");
    H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &OmegaLambda);
    H5Aclose(attribute_id);

    attribute_id = H5Aopen_name(group_id, "UnitLength_in_cm");
    H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &UnitLength_in_cm);
    H5Aclose(attribute_id);

    attribute_id = H5Aopen_name(group_id, "UnitMass_in_g");
    H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &UnitMass_in_g);
    H5Aclose(attribute_id);

    attribute_id = H5Aopen_name(group_id, "UnitVelocity_in_cm_per_s");
    H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &UnitVelocity_in_cm_per_s);
    H5Aclose(attribute_id);

    H5Gclose(group_id);
    H5Fclose(file_id);
  }

  // Read header info from the snapshot files
  if (ThisTask == 0) {
    std::ostringstream oss;
    oss << file_dir << "/snapdir_" << std::setfill('0') << std::setw(3) << SnapNum
        << "/snapshot_" << std::setfill('0') << std::setw(3) << SnapNum << ".0.hdf5";
    string fname = oss.str();
    hid_t file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    // Header
    hid_t group_id = H5Gopen(file_id, "Header", H5P_DEFAULT);

    myint NumPart_Total[NTYPES];
    hid_t attribute_id = H5Aopen_name(group_id, "NumPart_Total");
    H5Aread(attribute_id, H5T_MYINT, NumPart_Total);
    H5Aclose(attribute_id);
    NumGas_Total = NumPart_Total[0];
    NumDM_Total = NumPart_Total[1];
    NumP2_Total = NumPart_Total[2];
    NumP3_Total = NumPart_Total[3];
    NumStar_Total = NumPart_Total[4];
    CHECK_INT_OVERFLOW(NumGas_Total);
    CHECK_INT_OVERFLOW(NumDM_Total);
    CHECK_INT_OVERFLOW(NumP2_Total);
    CHECK_INT_OVERFLOW(NumP3_Total);
    CHECK_INT_OVERFLOW(NumStar_Total);

    double MassTable[NTYPES];
    attribute_id = H5Aopen_name(group_id, "MassTable");
    H5Aread(attribute_id, H5T_NATIVE_DOUBLE, MassTable);
    H5Aclose(attribute_id);
    MassDM = MassTable[1];
    MassP3 = MassTable[3];

    H5Gclose(group_id);
    H5Fclose(file_id);
  }

#ifdef MPI
  MPI_Bcast(&NTYPES, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&NumFiles, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Ngroups_Total, 1, MPI_MYINT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Nsubhalos_Total, 1, MPI_MYINT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&NumGas_Total, 1, MPI_MYINT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&NumDM_Total, 1, MPI_MYINT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&NumP2_Total, 1, MPI_MYINT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&NumP3_Total, 1, MPI_MYINT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&NumStar_Total, 1, MPI_MYINT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&MassDM, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&MassP3, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&BoxSize, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&h, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Omega0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&OmegaBaryon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&OmegaLambda, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&UnitLength_in_cm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&UnitMass_in_g, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&UnitVelocity_in_cm_per_s, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  BoxHalf = 0.5 * BoxSize;
  n_grps = Ngroups_Total, n_subs = Nsubhalos_Total;
  n3_grps = 3 * n_grps, n3_subs = 3 * n_subs;
  n_gas = NumGas_Total, n_dm = NumDM_Total, n_p2 = NumP2_Total, n_p3 = NumP3_Total, n_stars = NumStar_Total;
  n3_gas = 3 * n_gas, n3_dm = 3 * n_dm, n3_p2 = 3 * n_p2, n3_p3 = 3 * n_p3, n3_stars = 3 * n_stars;
#if GPU
  equal_work(Ngroups_Total, ThisDevice, NDevice, d_Ngroups_Total, offset_Ngroups_Total);
  equal_work(Nsubhalos_Total, ThisDevice, NDevice, d_Nsubhalos_Total, offset_Nsubhalos_Total);
  d_n_grps = d_Ngroups_Total, d_n_subs = d_Nsubhalos_Total;
  d_n3_grps = 3 * d_Ngroups_Total, d_n3_subs = 3 * d_Nsubhalos_Total;
  first_n3_grps = 3 * offset_Ngroups_Total, first_n3_subs = 3 * offset_Nsubhalos_Total;
#endif
}

static void read_file_counts()
{
  // FoF: Group, Subhalo
  for (int i = ThisTask; i < NumFiles; i += NTask) {
    std::ostringstream oss;
    oss << file_dir << "/groups_" << std::setfill('0') << std::setw(3) << SnapNum
        << "/fof_subhalo_tab_" << std::setfill('0') << std::setw(3) << SnapNum << "." << i << ".hdf5";
    string fname = oss.str();
    hid_t file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    // Header
    hid_t group_id = H5Gopen(file_id, "Header", H5P_DEFAULT);

    hid_t attribute_id = H5Aopen_name(group_id, "Ngroups_ThisFile");
    H5Aread(attribute_id, H5T_MYINT, &FileCounts_Group[i]);
    H5Aclose(attribute_id);

    attribute_id = H5Aopen_name(group_id, "Nsubhalos_ThisFile");
    H5Aread(attribute_id, H5T_MYINT, &FileCounts_Subhalo[i]);
    H5Aclose(attribute_id);

    H5Gclose(group_id);
    H5Fclose(file_id);
  }

  // Snapshot: Gas, DM, P2, P3, Star
  for (int i = ThisTask; i < NumFiles; i += NTask) {
    std::ostringstream oss;
    oss << file_dir << "/snapdir_" << std::setfill('0') << std::setw(3) << SnapNum
        << "/snapshot_" << std::setfill('0') << std::setw(3) << SnapNum << "." << i << ".hdf5";
    string fname = oss.str();
    hid_t file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    // Header
    hid_t group_id = H5Gopen(file_id, "Header", H5P_DEFAULT);

    myint NumPart_ThisFile[NTYPES];
    hid_t attribute_id = H5Aopen_name(group_id, "NumPart_ThisFile");
    H5Aread(attribute_id, H5T_MYINT, NumPart_ThisFile);
    H5Aclose(attribute_id);
    FileCounts_Gas[i] = NumPart_ThisFile[0];
    FileCounts_DM[i] = NumPart_ThisFile[1];
    FileCounts_P2[i] = NumPart_ThisFile[2];
    FileCounts_P3[i] = NumPart_ThisFile[3];
    FileCounts_Star[i] = NumPart_ThisFile[4];

    H5Gclose(group_id);
    H5Fclose(file_id);
  }
}

static void calculate_file_offsets()
{
  FileOffsets_Group[0] = 0;
  FileOffsets_Subhalo[0] = 0;
  FileOffsets_Gas[0] = 0;
  FileOffsets_DM[0] = 0;
  FileOffsets_P2[0] = 0;
  FileOffsets_P3[0] = 0;
  FileOffsets_Star[0] = 0;
  for (int i = 1; i < NumFiles; ++i) {
    FileOffsets_Group[i] = FileOffsets_Group[i-1] + FileCounts_Group[i-1];
    FileOffsets_Subhalo[i] = FileOffsets_Subhalo[i-1] + FileCounts_Subhalo[i-1];
    FileOffsets_Gas[i] = FileOffsets_Gas[i-1] + FileCounts_Gas[i-1];
    FileOffsets_DM[i] = FileOffsets_DM[i-1] + FileCounts_DM[i-1];
    FileOffsets_P2[i] = FileOffsets_P2[i-1] + FileCounts_P2[i-1];
    FileOffsets_P3[i] = FileOffsets_P3[i-1] + FileCounts_P3[i-1];
    FileOffsets_Star[i] = FileOffsets_Star[i-1] + FileCounts_Star[i-1];
  }

  if (VERBOSE) {
    // File counts
    cout << endl;
    print(FileCounts_Group, "FileCounts_Group", NumFiles);
    print(FileCounts_Subhalo, "FileCounts_Subhalo", NumFiles);
    print(FileCounts_Gas, "FileCounts_Gas", NumFiles);
    print(FileCounts_DM, "FileCounts_DM", NumFiles);
    print(FileCounts_P2, "FileCounts_P2", NumFiles);
    print(FileCounts_P3, "FileCounts_P3", NumFiles);
    print(FileCounts_Star, "FileCounts_Star", NumFiles);

    // File offsets
    cout << endl;
    print(FileOffsets_Group, "FileOffsets_Group", NumFiles);
    print(FileOffsets_Subhalo, "FileOffsets_Subhalo", NumFiles);
    print(FileOffsets_Gas, "FileOffsets_Gas", NumFiles);
    print(FileOffsets_DM, "FileOffsets_DM", NumFiles);
    print(FileOffsets_P2, "FileOffsets_P2", NumFiles);
    print(FileOffsets_P3, "FileOffsets_P3", NumFiles);
    print(FileOffsets_Star, "FileOffsets_Star", NumFiles);
  }
}

static void print_data()
{
  // Group data
  cout << endl;
  print(GroupPos, "GroupPos", 3 * Ngroups_Total);
  print(R_Crit200, "R_Crit200", Ngroups_Total);
  print(Group_MinDistGasHR, "Group_MinDistGasHR", Ngroups_Total);
  print(Group_MinDistGasLR, "Group_MinDistGasLR", Ngroups_Total);
  print(Group_MinDistDM, "Group_MinDistDM", Ngroups_Total);
  print(Group_MinDistP2, "Group_MinDistP2", Ngroups_Total);
  print(Group_MinDistP3, "Group_MinDistP3", Ngroups_Total);
  print(Group_MinDistStarsHR, "Group_MinDistStarsHR", Ngroups_Total);
  print(Group_MinDistStarsLR, "Group_MinDistStarsLR", Ngroups_Total);
  print(Group_MinMemberDistStarsHR, "Group_MinMemberDistStarsHR", Ngroups_Total);

  // Subhalo data
  cout << endl;
  print(SubhaloPos, "SubhaloPos", 3 * Nsubhalos_Total);
  print(R_vir, "R_vir", Nsubhalos_Total);
  print(M_vir, "M_vir", Nsubhalos_Total);
  print(M_gas, "M_gas", Nsubhalos_Total);
  print(M_stars, "M_stars", Nsubhalos_Total);
  print(Subhalo_MinDistGasHR, "Subhalo_MinDistGasHR", Nsubhalos_Total);
  print(Subhalo_MinDistGasLR, "Subhalo_MinDistGasLR", Nsubhalos_Total);
  print(Subhalo_MinDistDM, "Subhalo_MinDistDM", Nsubhalos_Total);
  print(Subhalo_MinDistP2, "Subhalo_MinDistP2", Nsubhalos_Total);
  print(Subhalo_MinDistP3, "Subhalo_MinDistP3", Nsubhalos_Total);
  print(Subhalo_MinDistStarsHR, "Subhalo_MinDistStarsHR", Nsubhalos_Total);
  print(Subhalo_MinDistStarsLR, "Subhalo_MinDistStarsLR", Nsubhalos_Total);
  print(Subhalo_MinMemberDistStarsHR, "Subhalo_MinMemberDistStarsHR", Nsubhalos_Total);

  // Gas data
  cout << endl;
  print(r_gas, "r_gas", n3_gas);
  print(m_gas, "m_gas", n_gas);
  print(m_gas_hr, "m_gas_hr", n_gas);

  // Gas data (high resolution)
  cout << endl;
  print(r_gas_hr, "r_gas_hr", n3_gas_hr);

  // Gas data (low resolution)
  cout << endl;
  print(r_gas_lr, "r_gas_lr", n3_gas_lr);

  // DM data
  cout << endl;
  print(r_dm, "r_dm", n_dm);

  // P2 data
  cout << endl;
  print(r_p2, "r_p2", n3_p2);
  print(m_p2, "m_p2", n_p2);

  // P3 data
  cout << endl;
  print(r_p3, "r_p3", n3_p3);

  // Star data
  if (n_stars > 0) {
    cout << endl;
    print(r_star, "r_star", n3_stars);
    print(m_star, "m_star", n_stars);
    print(star_is_hr, "star_is_hr", n_stars);
  }

  // Star data (high resolution)
  if (n_stars_hr > 0) {
    cout << endl;
    print(r_star_hr, "r_star_hr", n3_stars_hr);
  }

  // Star data (low resolution)
  if (n_stars_lr > 0) {
    cout << endl;
    print(r_star_lr, "r_star_lr", n3_stars_lr);
  }
}

static void calculate_fof_offsets()
{
  if (Ngroups_Total == 0)
    return;
  // Initialize offsets for the first group
  myint i_g = 0, i_s = 0, n_s = Group_Nsubs[0]; // Group, Subhalo, Number of group subhalos
  if (i_s != Group_FirstSub[i_g]) {
    cerr << "\ni_s != GroupFirstSub[i_g]   (i_g = " << i_g << ", i_s = " << i_s << ", FirstSub = " << Group_FirstSub[i_g] << ")" << endl;
    exit(1);
  }
  if (n_s <= 0) {
    cerr << "\nGroupNsubs = " << n_s << " <= 0   (i_g = " << i_g << ", i_s = " << i_s << ")" << endl;
    exit(1);
  }
  // Subhalo offsets start at the group
  Group_FirstStar[i_g] = 0;
  Subhalo_FirstStar[i_s] = Group_FirstStar[i_g];
  i_s++;
  for (myint j = 1; j < n_s; j++, i_s++)
    Subhalo_FirstStar[i_s] = Subhalo_FirstStar[i_s-1] + Subhalo_NumStar[i_s-1];

  // Initialize offsets for the remaining groups
  for (i_g = 1; i_g < Ngroups_Total; i_g++) {
    Group_FirstStar[i_g] = Group_FirstStar[i_g-1] + Group_NumStar[i_g-1];
    n_s = Group_Nsubs[i_g]; // Number of group subhalos
    if (n_s > 0) {
      // Double check offset consistency
      if (i_s != Group_FirstSub[i_g]) {
        cerr << "\ni_s != GroupFirstSub[i_g]   (i_g = " << i_g << ", i_s = " << i_s << ", FirstSub = " << Group_FirstSub[i_g] << ", Nsubs = " << n_s << ")" << endl;
        exit(1);
      }
      // Subhalo offsets start at the group
      Subhalo_FirstStar[i_s] = Group_FirstStar[i_g];
      i_s++;
      for (myint j = 1; j < n_s; j++, i_s++)
        Subhalo_FirstStar[i_s] = Subhalo_FirstStar[i_s-1] + Subhalo_NumStar[i_s-1];
    }
  }

  if (VERBOSE) {
    cout << "\nGroup_NumStar = [" << Group_NumStar[0];
    for (int i = 1; i < N_PRINT; i++)
      cout << ", " << Group_NumStar[i];
    cout << ", ... ]\nGroup_FirstStar = [" << Group_FirstStar[0];
    for (int i = 1; i < N_PRINT; i++)
      cout << ", " << Group_FirstStar[i];
    cout << ", ... ]\nSubhalo_NumStar = [" << Subhalo_NumStar[0];
    for (int i = 1; i < N_PRINT; i++)
      cout << ", " << Subhalo_NumStar[i];
    cout << ", ... ]\nSubhalo_FirstStar = [" << Subhalo_FirstStar[0];
    for (int i = 1; i < N_PRINT; i++)
      cout << ", " << Subhalo_FirstStar[i];
    cout << ", ... ]" << endl;

    // Sumary statistics
    const myint NumStar_Groups = Group_FirstStar[Ngroups_Total-1] + Group_NumStar[Ngroups_Total-1];
    const myint NumStar_Outer = NumStar_Total - NumStar_Groups;
    const double FracStar_Outer = (double)NumStar_Outer / (double)NumStar_Total;
    cout << "\nNumStar_Outer = " << NumStar_Outer << "  (" << 1e2*FracStar_Outer << "%)";
  }
}

static void read_fof_data()
{
  int *buffer;
  long long *buffer_ll;

  for (int curnum = ThisTask; curnum < NumFiles; curnum += NTask) {
    const myint Ngroups_ThisFile = FileCounts_Group[curnum];
    const myint Nsubhalos_ThisFile = FileCounts_Subhalo[curnum];

    if (Ngroups_ThisFile > 0 || Nsubhalos_ThisFile > 0) {
      std::ostringstream oss;
      oss << file_dir << "/groups_" << std::setfill('0') << std::setw(3) << SnapNum
          << "/fof_subhalo_tab_" << std::setfill('0') << std::setw(3) << SnapNum << "." << curnum << ".hdf5";
      string fname = oss.str();
      hid_t file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

      if (Ngroups_ThisFile > 0) {
        buffer = (int *) malloc(NTYPES*Ngroups_ThisFile * sizeof(int));
        buffer_ll = (long long *) malloc(Ngroups_ThisFile * sizeof(long long));

        hid_t dataset = H5Dopen(file_id, "Group/GroupFirstSub", H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_ll);
        H5Dclose(dataset);

        dataset = H5Dopen(file_id, "Group/GroupNsubs", H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
        H5Dclose(dataset);

        for (myint j = 0; j < Ngroups_ThisFile; j++) {
          Group_FirstSub[FileOffsets_Group[curnum]+j] = (myint)buffer_ll[j];
          Group_Nsubs[FileOffsets_Group[curnum]+j] = (myint)buffer[j];
        }

        free(buffer_ll);

        dataset = H5Dopen(file_id, "Group/GroupLenType", H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
        H5Dclose(dataset);

        for (myint j = 0; j < Ngroups_ThisFile; j++)
          Group_NumStar[FileOffsets_Group[curnum]+j] = (myint)buffer[NTYPES*j+4]; // Star = PartType4

        free(buffer);

        dataset = H5Dopen(file_id, "Group/GroupPos", H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &GroupPos[3*FileOffsets_Group[curnum]]);
        H5Dclose(dataset);

        dataset = H5Dopen(file_id, "Group/Group_R_Crit200", H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &R_Crit200[FileOffsets_Group[curnum]]);
        H5Dclose(dataset);
      }

      if (Nsubhalos_ThisFile > 0) {
        buffer = (int *) malloc(NTYPES*Nsubhalos_ThisFile * sizeof(int));

        hid_t dataset = H5Dopen(file_id, "Subhalo/SubhaloLenType", H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
        H5Dclose(dataset);

        for (myint j = 0; j < Nsubhalos_ThisFile; j++)
          Subhalo_NumStar[FileOffsets_Subhalo[curnum]+j] = (myint)buffer[NTYPES*j+4]; // Star = PartType4

        free(buffer);

        dataset = H5Dopen(file_id, "Subhalo/SubhaloPos", H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &SubhaloPos[3*FileOffsets_Subhalo[curnum]]);
        H5Dclose(dataset);
      }

      H5Fclose(file_id);
    }
  }
}

static void calculate_fof_data()
{
  if (Ngroups_Total == 0)
    return;

  // Groups
  for (myint i_g = ThisTask; i_g < Ngroups_Total; i_g += NTask) {
    const myint first_star = Group_FirstStar[i_g];
    const myint last_star = first_star + Group_NumStar[i_g];
    double r2_comp = 1e20; // Initialize to a large value
    const double x1 = GroupPos[0], y1 = GroupPos[1], z1 = GroupPos[2];
    for (myint i_star = first_star; i_star < last_star; i_star++) {
      if (star_is_hr[i_star]) {
        const myint i3star = 3 * i_star; // 3D index
        const double x2 = r_star[i3star]; // Star position
        const double y2 = r_star[i3star + 1];
        const double z2 = r_star[i3star + 2];

        const double dx = shortest_distance(x1, x2, BoxSize, BoxHalf);
        const double dy = shortest_distance(y1, y2, BoxSize, BoxHalf);
        const double dz = shortest_distance(z1, z2, BoxSize, BoxHalf);

        const double r2 = dx*dx + dy*dy + dz*dz; // Distance squared
        if (r2 < r2_comp)
          r2_comp = r2;
      }
    }
    Group_MinMemberDistStarsHR[i_g] = sqrt(r2_comp); // Minimum member distance
  }

  // Subhalos
  for (myint i_s = ThisTask; i_s < Nsubhalos_Total; i_s += NTask) {
    const myint first_star = Subhalo_FirstStar[i_s];
    const myint last_star = first_star + Subhalo_NumStar[i_s];
    double r2_comp = 1e20; // Initialize to a large value
    const double x1 = SubhaloPos[0], y1 = SubhaloPos[1], z1 = SubhaloPos[2];
    for (myint i_star = first_star; i_star < last_star; i_star++) {
      if (star_is_hr[i_star]) {
        const myint i3star = 3 * i_star; // 3D index
        const double x2 = r_star[i3star]; // Star position
        const double y2 = r_star[i3star + 1];
        const double z2 = r_star[i3star + 2];

        const double dx = shortest_distance(x1, x2, BoxSize, BoxHalf);
        const double dy = shortest_distance(y1, y2, BoxSize, BoxHalf);
        const double dz = shortest_distance(z1, z2, BoxSize, BoxHalf);

        const double r2 = dx*dx + dy*dy + dz*dz; // Distance squared
        if (r2 < r2_comp)
          r2_comp = r2;
      }
    }
    Subhalo_MinMemberDistStarsHR[i_s] = sqrt(r2_comp); // Minimum member distance
  }
}

#define WRITE_ATTRIBUTE(attr_name, attr_value, attr_type)                                           \
  dataspace_id = H5Screate(H5S_SCALAR);                                                             \
  attribute_id = H5Acreate(group_id, attr_name, attr_type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); \
  status       = H5Awrite(attribute_id, attr_type, &(attr_value));                                  \
  status       = H5Aclose(attribute_id);                                                            \
  status       = H5Sclose(dataspace_id);                                                            \
  if (status < 0) { cerr << "Error: Failed to read attribute " << attr_name << endl; exit(1); }

#define WRITE_ARRAY_ATTRIBUTE(attr_name, attr_ptr, attr_len, attr_type)                             \
  dims[0] = attr_len;                                                                               \
  dataspace_id = H5Screate_simple(1, dims, NULL);                                                   \
  attribute_id = H5Acreate(group_id, attr_name, attr_type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); \
  status       = H5Awrite(attribute_id, attr_type, attr_ptr);                                       \
  status       = H5Aclose(attribute_id);                                                            \
  status       = H5Sclose(dataspace_id);                                                            \
  if (status < 0) { cerr << "Error: Failed to read array attribute " << attr_name << endl; exit(1); }

struct UnitAttrs
{
  double a;  // exponent of the cosmological a factor
  double h;  // exponent of the hubble parameter
  double L;  // length unit scaling
  double M;  // mass unit scaling
  double V;  // velocity unit scaling
  double c;  // conversion factor to cgs units (zero indicates dimensionless quantity, integer count, etc)
};

/*! \brief Function for setting units of an output field.
 *
 *  \param[in/out] *ua UnitAttrs pointer to be set.
 *  \param[in] a the exponent of the cosmological a factor.
 *  \param[in] h the exponent of the hubble parameter.
 *  \param[in] L the length unit scaling.
 *  \param[in] M the mass unit scaling.
 *  \param[in] V the velocity unit scaling.
 *  \param[in] c conversion factor to cgs units (zero indicates dimensionless
 *             quantity, integer count, etc).
 *
 *  \return void
 */
static inline void set_unit_attrs(struct UnitAttrs *ua, double a, double h, double L, double M, double V, double c)
{
  ua->a = a;
  ua->h = h;
  ua->L = L;
  ua->M = M;
  ua->V = V;
  ua->c = c;
}

/*! \brief Function for adding units to an output field.
 *
 *  \param[in] file_id specifies the file location.
 *  \param[in] name specifies the dataset location relative to file_id.
 *  \param[in] ua the UnitAttrs struct holding (a,h,L,M,V,c) attributes.
 *
 *  \return void
 */
static void write_units(hid_t file_id, const char *name, struct UnitAttrs *ua)
{
  herr_t status;
  hid_t dataspace_id, attribute_id;

  hid_t group_id = H5Dopen(file_id, name, H5P_DEFAULT); // group_id is for convenience (macro)

  WRITE_ATTRIBUTE("a_scaling", ua->a, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("h_scaling", ua->h, H5T_NATIVE_DOUBLE)

  WRITE_ATTRIBUTE("length_scaling", ua->L, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("mass_scaling", ua->M, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("velocity_scaling", ua->V, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("to_cgs", ua->c, H5T_NATIVE_DOUBLE)

  H5Dclose(group_id); // Close the dataset
}

// Writes out a vector quantity, e.g. (a1,a2,a3,...)
static void write(hid_t file_id, hsize_t n_cols, float *data, const char *dataset_name)
{
  // Identifier
  hid_t dataspace_id, dataset_id;
  hsize_t dims1d[1] = {n_cols};

  dataspace_id = H5Screate_simple(1, dims1d, NULL);
  dataset_id = H5Dcreate(file_id, dataset_name, H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);

  // Open dataset and get dataspace
  hid_t dataset   = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
  hid_t filespace = H5Dget_space(dataset);

  // File hyperslab
  hsize_t file_offset[1] = {0};
  hsize_t file_count[1] = {n_cols};

  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, file_offset, NULL, file_count, NULL);

  // Memory hyperslab
  hsize_t mem_offset[1] = {0};
  hsize_t mem_count[1] = {n_cols};

  hid_t memspace = H5Screate_simple(1, mem_count, NULL);
  H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offset, NULL, mem_count, NULL);

  // Write
  H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, data);

  // Close handles
  H5Sclose(memspace);
  H5Sclose(filespace);
  H5Dclose(dataset);
}

static void write_fof_data()
{
  // Write output file
  std::ostringstream oss;
  oss << out_dir << "/" + outname + "_" << std::setw(3) << std::setfill('0') << SnapNum << ".hdf5";
  string fname = oss.str();

  // Identifiers
  herr_t status;
  hsize_t dims[1];
  hid_t file_id, group_id, dataspace_id, attribute_id;
  struct UnitAttrs ua;
  set_unit_attrs(&ua, 1., -1., 1., 0., 0., UnitLength_in_cm); // a L / h

  // Open file and write header
  file_id = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  group_id = H5Gcreate(file_id, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  const myint NumGasHR_Total = n_gas_hr;
  const myint NumGasLR_Total = n_gas_lr;
  const myint NumStarHR_Total = n_stars_hr;
  const myint NumStarLR_Total = n_stars_lr;
  WRITE_ATTRIBUTE("Ngroups_Total", Ngroups_Total, H5T_MYINT)
  WRITE_ATTRIBUTE("Nsubhalos_Total", Nsubhalos_Total, H5T_MYINT)
  WRITE_ATTRIBUTE("NumGas_Total", NumGas_Total, H5T_MYINT)
  WRITE_ATTRIBUTE("NumGasHR_Total", NumGasHR_Total, H5T_MYINT)
  WRITE_ATTRIBUTE("NumGasLR_Total", NumGasLR_Total, H5T_MYINT)
  WRITE_ATTRIBUTE("NumDM_Total", NumDM_Total, H5T_MYINT)
  WRITE_ATTRIBUTE("NumP2_Total", NumP2_Total, H5T_MYINT)
  WRITE_ATTRIBUTE("NumP3_Total", NumP3_Total, H5T_MYINT)
  WRITE_ATTRIBUTE("NumStar_Total", NumStar_Total, H5T_MYINT)
  WRITE_ATTRIBUTE("NumStarHR_Total", NumStarHR_Total, H5T_MYINT)
  WRITE_ATTRIBUTE("NumStarLR_Total", NumStarLR_Total, H5T_MYINT)

  // BoxSize, Time, Units, HubbleParam, Omega0, Redshift
  const double Redshift = 1. / a - 1.;
  WRITE_ATTRIBUTE("Time", a, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("Redshift", Redshift, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("BoxSize", BoxSize, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("HubbleParam", h, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("Omega0", Omega0, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("OmegaBaryon", OmegaBaryon, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("OmegaLambda", OmegaLambda, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("UnitLength_in_cm", UnitLength_in_cm, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("UnitMass_in_g", UnitMass_in_g, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("UnitVelocity_in_cm_per_s", UnitVelocity_in_cm_per_s, H5T_NATIVE_DOUBLE)
  WRITE_ARRAY_ATTRIBUTE("PosHR", PosHR, 3, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("RadiusHR", RadiusHR, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("RadiusLR", RadiusLR, H5T_NATIVE_DOUBLE)

  status = H5Gclose(group_id);

  // Group catalog
  if (Ngroups_Total > 0) {
    group_id = H5Gcreate(file_id, "Group", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write(group_id, Ngroups_Total, Group_MinDistGasHR, "MinDistGasHR"); // ckpc/h
    write_units(group_id, "MinDistGasHR", &ua);
    write(group_id, Ngroups_Total, Group_MinDistGasLR, "MinDistGasLR"); // ckpc/h
    write_units(group_id, "MinDistGasLR", &ua);
    write(group_id, Ngroups_Total, Group_MinDistDM, "MinDistDM"); // ckpc/h
    write_units(group_id, "MinDistDM", &ua);
    write(group_id, Ngroups_Total, Group_MinDistP2, "MinDistP2"); // ckpc/h
    write_units(group_id, "MinDistP2", &ua);
    write(group_id, Ngroups_Total, Group_MinDistP3, "MinDistP3"); // ckpc/h
    write_units(group_id, "MinDistP3", &ua);
    if (n_stars_hr > 0) {
      write(group_id, Ngroups_Total, Group_MinDistStarsHR, "MinDistStarsHR"); // ckpc/h
      write_units(group_id, "MinDistStarsHR", &ua);
      write(group_id, Ngroups_Total, Group_MinMemberDistStarsHR, "MinMemberDistStarsHR"); // ckpc/h
      write_units(group_id, "MinMemberDistStarsHR", &ua);
    }
    if (n_stars_lr > 0) {
      write(group_id, Ngroups_Total, Group_MinDistStarsLR, "MinDistStarsLR"); // ckpc/h
      write_units(group_id, "MinDistStarsLR", &ua);
    }
    write(group_id, Ngroups_Total, R_Crit200, "R_Crit200"); // ckpc/h
    write_units(group_id, "R_Crit200", &ua);
    status = H5Gclose(group_id);
  }

  // Subhalo catalog
  if (Nsubhalos_Total > 0) {
    group_id = H5Gcreate(file_id, "Subhalo", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write(group_id, Nsubhalos_Total, Subhalo_MinDistGasHR, "MinDistGasHR"); // ckpc/h
    write_units(group_id, "MinDistGasHR", &ua);
    write(group_id, Nsubhalos_Total, Subhalo_MinDistGasLR, "MinDistGasLR"); // ckpc/h
    write_units(group_id, "MinDistGasLR", &ua);
    write(group_id, Nsubhalos_Total, Subhalo_MinDistDM, "MinDistDM"); // ckpc/h
    write_units(group_id, "MinDistDM", &ua);
    write(group_id, Nsubhalos_Total, Subhalo_MinDistP2, "MinDistP2"); // ckpc/h
    write_units(group_id, "MinDistP2", &ua);
    write(group_id, Nsubhalos_Total, Subhalo_MinDistP3, "MinDistP3"); // ckpc/h
    write_units(group_id, "MinDistP3", &ua);
    if (n_stars_hr > 0) {
      write(group_id, Nsubhalos_Total, Subhalo_MinDistStarsHR, "MinDistStarsHR"); // ckpc/h
      write_units(group_id, "MinDistStarsHR", &ua);
      write(group_id, Nsubhalos_Total, Subhalo_MinMemberDistStarsHR, "MinMemberDistStarsHR"); // ckpc/h
      write_units(group_id, "MinMemberDistStarsHR", &ua);
    }
    if (n_stars_lr > 0) {
      write(group_id, Nsubhalos_Total, Subhalo_MinDistStarsLR, "MinDistStarsLR"); // ckpc/h
      write_units(group_id, "MinDistStarsLR", &ua);
    }
    write(group_id, Nsubhalos_Total, R_vir, "R_vir"); // ckpc/h
    write_units(group_id, "R_vir", &ua);
    set_unit_attrs(&ua, 0., -1., 0., 1., 0., UnitLength_in_cm); // M / h
    write(group_id, Nsubhalos_Total, M_vir, "M_vir"); // Virial mass
    write_units(group_id, "M_vir", &ua);
    write(group_id, Nsubhalos_Total, M_gas, "M_gas"); // Gas mass (<R_vir)
    write_units(group_id, "M_gas", &ua);
    write(group_id, Nsubhalos_Total, M_stars, "M_stars"); // Stars mass (<R_vir)
    write_units(group_id, "M_stars", &ua);
    status = H5Gclose(group_id);
  }

  // Close file
  H5Fclose(file_id);
}

static void read_snap_data()
{
  n_gas_hr = 0; n_gas_lr = 0, n_stars_hr = 0, n_stars_lr = 0;
  MassHR = 0.; PosHR[0] = 0.; PosHR[1] = 0.; PosHR[2] = 0.; // Initialize
  for (int curnum = ThisTask; curnum < NumFiles; curnum += NTask) {
    if (VERBOSE > 0)
      cout << "Reading snapshot file " << curnum << " of " << NumFiles << " (Task " << ThisTask << ")" << endl;
    const myint n_local_gas = FileCounts_Gas[curnum], n_offset_gas = FileOffsets_Gas[curnum];
    const myint n_local_dm = FileCounts_DM[curnum], n_offset_dm = FileOffsets_DM[curnum];
    const myint n_local_p2 = FileCounts_P2[curnum], n_offset_p2 = FileOffsets_P2[curnum];
    const myint n_local_p3 = FileCounts_P3[curnum], n_offset_p3 = FileOffsets_P3[curnum];
    const myint n_local_stars = FileCounts_Star[curnum], n_offset_stars = FileOffsets_Star[curnum];
    if (n_local_gas > 0 || n_local_dm > 0 || n_local_p2 > 0 || n_local_p3 > 0 || n_local_stars > 0) {
      std::ostringstream oss;
      oss << file_dir << "/snapdir_" << std::setfill('0') << std::setw(3) << SnapNum
          << "/snapshot_" << std::setfill('0') << std::setw(3) << SnapNum << "." << curnum << ".hdf5";
      string fname = oss.str();
      hid_t file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

      if (n_local_gas > 0) {
        if (VERBOSE > 0)
          cout << "Task " << ThisTask << " reading " << n_local_gas << " gas particles" << endl;
        const myint n3_local_gas = 3 * n_local_gas;
        double *Coordinates = (double *) malloc(n3_local_gas * sizeof(double));
        hid_t dataset = H5Dopen(file_id, "PartType0/Coordinates", H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Coordinates);
        H5Dclose(dataset);

        float *m_local = &m_gas[n_offset_gas];
        dataset = H5Dopen(file_id, "PartType0/Masses", H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_local);
        H5Dclose(dataset);

        float *m_local_hr = &m_gas_hr[n_offset_gas];
        dataset = H5Dopen(file_id, "PartType0/HighResGasMass", H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_local_hr);
        H5Dclose(dataset);

        // Count the number of high-res gas particles
        myint n_local_gas_hr = 0;
        for (myint i = 0; i < n_local_gas; ++i) {
          if (m_local_hr[i] > GAS_HIGH_RES_THRESHOLD * m_local[i])
            ++n_local_gas_hr;
        }
        const myint n_local_gas_lr = n_local_gas - n_local_gas_hr;
        n_gas_hr += n_local_gas_hr;
        n_gas_lr += n_local_gas_lr;
        float *r_local = &r_gas[3*n_offset_gas];
        for (myint i = 0; i < n_local_gas; ++i) {
          const myint i3 = 3 * i;
          if (m_local_hr[i] > GAS_HIGH_RES_THRESHOLD * m_local[i]) {
            const double m_hr = m_local_hr[i];
            MassHR += m_hr;
            PosHR[0] += m_hr * Coordinates[i3];
            PosHR[1] += m_hr * Coordinates[i3+1];
            PosHR[2] += m_hr * Coordinates[i3+2];
          }
          r_local[i3] = Coordinates[i3]; // Convert to single precision (high-res)
          r_local[i3+1] = Coordinates[i3+1];
          r_local[i3+2] = Coordinates[i3+2];
        }
        free(Coordinates);
      }

      if (n_local_dm > 0) {
        if (VERBOSE > 0)
          cout << "Task " << ThisTask << " reading " << n_local_dm << " dm particles" << endl;
        const myint n3_local_dm = 3 * n_local_dm;
        double *Coordinates = (double *) malloc(n3_local_dm * sizeof(double));
        hid_t dataset = H5Dopen(file_id, "PartType1/Coordinates", H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Coordinates);
        H5Dclose(dataset);

        float *r_local = &r_dm[3*n_offset_dm];
        for (myint i = 0; i < n_local_dm; ++i) {
          const myint i3 = 3 * i;
          MassHR += MassDM;
          PosHR[0] += MassDM * Coordinates[i3];
          PosHR[1] += MassDM * Coordinates[i3+1];
          PosHR[2] += MassDM * Coordinates[i3+2];
          r_local[i3] = Coordinates[i3]; // Convert to single precision
          r_local[i3+1] = Coordinates[i3+1];
          r_local[i3+2] = Coordinates[i3+2];
        }
        free(Coordinates);
      }

      if (n_local_p2 > 0) {
        if (VERBOSE > 0)
          cout << "Task " << ThisTask << " reading " << n_local_p2 << " p2 particles" << endl;
        const myint n3_local_p2 = 3 * n_local_p2;
        double *Coordinates = (double *) malloc(n3_local_p2 * sizeof(double));
        hid_t dataset = H5Dopen(file_id, "PartType2/Coordinates", H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Coordinates);
        H5Dclose(dataset);

        dataset = H5Dopen(file_id, "PartType2/Masses", H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_p2[n_offset_p2]);
        H5Dclose(dataset);

        float *r_local = &r_p2[3*n_offset_p2];
        for (myint i = 0; i < n3_local_p2; ++i)
          r_local[i] = Coordinates[i]; // Convert to single precision
        free(Coordinates);
      }

      if (n_local_p3 > 0) {
        if (VERBOSE > 0)
          cout << "Task " << ThisTask << " reading " << n_local_p3 << " p3 particles" << endl;
        const myint n3_local_p3 = 3 * n_local_p3;
        double *Coordinates = (double *) malloc(n3_local_p3 * sizeof(double));
        hid_t dataset = H5Dopen(file_id, "PartType3/Coordinates", H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Coordinates);
        H5Dclose(dataset);

        float *r_local = &r_p3[3*n_offset_p3];
        for (myint i = 0; i < n3_local_p3; ++i)
          r_local[i] = Coordinates[i]; // Convert to single precision
        free(Coordinates);
      }

      if (n_local_stars > 0) {
        if (VERBOSE > 0)
          cout << "Task " << ThisTask << " reading " << n_local_stars << " star particles" << endl;
        const myint n3_local_stars = 3 * n_local_stars;
        double *Coordinates = (double *) malloc(n3_local_stars * sizeof(double));
        hid_t dataset = H5Dopen(file_id, "PartType4/Coordinates", H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Coordinates);
        H5Dclose(dataset);

        float *m_local = &m_star[n_offset_stars];
        dataset = H5Dopen(file_id, "PartType4/Masses", H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_local);
        H5Dclose(dataset);

        int *is_hr = &star_is_hr[n_offset_stars];
        dataset = H5Dopen(file_id, "PartType4/IsHighRes", H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, is_hr);
        H5Dclose(dataset);

        // Count the number of high-res star particles
        myint n_local_stars_hr = 0;
        for (myint i = 0; i < n_local_stars; ++i) {
          if (is_hr[i])
            ++n_local_stars_hr;
        }
        const myint n_local_stars_lr = n_local_stars - n_local_stars_hr;
        n_stars_hr += n_local_stars_hr;
        n_stars_lr += n_local_stars_lr;
        float *r_local = &r_star[3*n_offset_stars];
        for (myint i = 0; i < n_local_stars; ++i) {
          const myint i3 = 3 * i;
          if (is_hr[i]) {
            const double m_hr = m_local[i];
            MassHR += m_hr;
            PosHR[0] += m_hr * Coordinates[i3];
            PosHR[1] += m_hr * Coordinates[i3+1];
            PosHR[2] += m_hr * Coordinates[i3+2];
          }
          r_local[i3] = Coordinates[i3]; // Convert to single precision (high-res)
          r_local[i3+1] = Coordinates[i3+1];
          r_local[i3+2] = Coordinates[i3+2];
        }
        free(Coordinates);
      }
      H5Fclose(file_id);
    }
  }
#ifdef MPI
  MPI_Allreduce(MPI_IN_PLACE, &n_gas_hr, 1, MPI_MYINT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &n_gas_lr, 1, MPI_MYINT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &n_stars_hr, 1, MPI_MYINT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &n_stars_lr, 1, MPI_MYINT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &MassHR, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, PosHR, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (MassHR > 0.) {
    PosHR[0] /= MassHR; // Center of mass normalization
    PosHR[1] /= MassHR;
    PosHR[2] /= MassHR;
  }
  n3_gas_hr = 3 * n_gas_hr;
  n3_gas_lr = 3 * n_gas_lr;
  n3_stars_hr = 3 * n_stars_hr;
  n3_stars_lr = 3 * n_stars_lr;
}

static void copy_hr_data()
{
#ifdef MPI
  if (n_gas_hr > 0)
    malloc_shared(r_gas_hr, win_r_gas_hr, n3_gas_hr);
  if (n_gas_lr > 0)
    malloc_shared(r_gas_lr, win_r_gas_lr, n3_gas_lr);
  if (n_stars_hr > 0)
    malloc_shared(r_star_hr, win_r_star_hr, n3_stars_hr);
  if (n_stars_lr > 0)
    malloc_shared(r_star_lr, win_r_star_lr, n3_stars_lr);
  MPI_Barrier(MPI_COMM_WORLD);
#else
  if (n_gas_hr > 0)
    r_gas_hr = (float *) malloc(n3_gas_hr * sizeof(float));
  if (n_gas_lr > 0)
    r_gas_lr = (float *) malloc(n3_gas_lr * sizeof(float));
  if (n_stars_hr > 0)
    r_star_hr = (float *) malloc(n3_stars_hr * sizeof(float));
  if (n_stars_lr > 0)
    r_star_lr = (float *) malloc(n3_stars_lr * sizeof(float));
#endif

  // Gas
  bool my_turn = (ThisTask == 0 % NTask);
  if (my_turn && n_gas > 0) {
    myint i_hr = 0, i_lr = 0;
    for (myint i = 0; i < n_gas; ++i) {
      const myint i3 = 3 * i;
      if (m_gas_hr[i] > GAS_HIGH_RES_THRESHOLD * m_gas[i]) {
        const myint i3_hr = 3 * i_hr;
        r_gas_hr[i3_hr] = r_gas[i3];
        r_gas_hr[i3_hr+1] = r_gas[i3+1];
        r_gas_hr[i3_hr+2] = r_gas[i3+2];
        ++i_hr;
      } else {
        const myint i3_lr = 3 * i_lr;
        r_gas_lr[i3_lr] = r_gas[i3];
        r_gas_lr[i3_lr+1] = r_gas[i3+1];
        r_gas_lr[i3_lr+2] = r_gas[i3+2];
        ++i_lr;
      }
    }
    if (i_hr != n_gas_hr || i_lr != n_gas_lr) {
      cerr << "Error: Gas accounting! (HR: " << i_hr << " != " << n_gas_hr << ", LR: " << i_lr << " != " << n_gas_lr << ")" << endl;
      MPI_Finalize();
      exit(1);
    }
    if (VERBOSE > 0)
      cout << "\nFinished copying high-res gas particles." << endl;
  }
  // Stars
  my_turn = (ThisTask == 1 % NTask);
  if (my_turn && n_stars > 0) {
    myint i_hr = 0, i_lr = 0;
    for (myint i = 0; i < n_stars; ++i) {
      const myint i3 = 3 * i;
      if (star_is_hr[i]) {
        const myint i3_hr = 3 * i_hr;
        r_star_hr[i3_hr] = r_star[i3];
        r_star_hr[i3_hr+1] = r_star[i3+1];
        r_star_hr[i3_hr+2] = r_star[i3+2];
        ++i_hr;
      } else {
        const myint i3_lr = 3 * i_lr;
        r_star_lr[i3_lr] = r_star[i3];
        r_star_lr[i3_lr+1] = r_star[i3+1];
        r_star_lr[i3_lr+2] = r_star[i3+2];
        ++i_lr;
      }
    }
    if (i_hr != n_stars_hr || i_lr != n_stars_lr) {
      cerr << "Error: Stars accounting! (HR: " << i_hr << " != " << n_stars_hr << ", LR: " << i_lr << " != " << n_stars_lr << ")" << endl;
      MPI_Finalize();
      exit(1);
    }
    if (VERBOSE > 0)
      cout << "\nFinished copying high-res star particles." << endl;
  }

#ifdef GPU
  // Allocate GPU memory and copy data
  MPI_Barrier(MPI_COMM_WORLD);
  if (ThisTask < NDevice) {
    if (n_gas_hr > 0) {
      cudaMalloc(&d_r_gas_hr, n3_gas_hr * sizeof(float));
      cudaMemcpy(d_r_gas_hr, r_gas_hr, n3_gas_hr * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (n_gas_lr > 0) {
      cudaMalloc(&d_r_gas_lr, n3_gas_lr * sizeof(float));
      cudaMemcpy(d_r_gas_lr, r_gas_lr, n3_gas_lr * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (n_stars_hr > 0) {
      cudaMalloc(&d_r_star_hr, n3_stars_hr * sizeof(float));
      cudaMemcpy(d_r_star_hr, r_star_hr, n3_stars_hr * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (n_stars_lr > 0) {
      cudaMalloc(&d_r_star_lr, n3_stars_lr * sizeof(float));
      cudaMemcpy(d_r_star_lr, r_star_lr, n3_stars_lr * sizeof(float), cudaMemcpyHostToDevice);
    }
  }
#endif
}

static void setup_vir()
{
//   // Gas
//   MassGas_Total = 0.;
//   for (myint i = ThisTask; i < n_gas; i += NTask)
//     MassGas_Total += m_gas[i];
//   // DM
//   MassDM_Total = MassDM * double(n_dm);
//   // P2
//   MassP2_Total = 0.;
//   for (myint i = ThisTask; i < n_p2; i += NTask)
//     MassP2_Total += m_p2[i];
//   // P3
//   MassP3_Total = MassP3 * double(n_p3);
//   // Stars
//   MassStar_Total = 0.;
//   if (n_stars > 0) {
//     for (myint i = ThisTask; i < n_stars; i += NTask)
//       MassStar_Total += m_star[i];
//   }
// #ifdef MPI
//   MPI_Allreduce(MPI_IN_PLACE, &MassGas_Total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//   MPI_Allreduce(MPI_IN_PLACE, &MassDM_Total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//   MPI_Allreduce(MPI_IN_PLACE, &MassP2_Total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//   MPI_Allreduce(MPI_IN_PLACE, &MassP3_Total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//   MPI_Allreduce(MPI_IN_PLACE, &MassStar_Total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
// #endif
//   Mass_Total = MassGas_Total + MassDM_Total + MassP2_Total + MassP3_Total + MassStar_Total;

  const double km = 1e5; // Units: 1 km = 1e5 cm
  const double kpc = 3.085677581467192e21; // Units: 1 kpc = 3e21 cm
  const double Mpc = 3.085677581467192e24; // Units: 1 Mpc = 3e24 cm
  const double H0 = 100. * h * km / Mpc; // Hubble constant [km/s/Mpc]
  const double G = 6.6743015e-8; // Gravitational constant [cm^3/g/s^2]
  const double H2 = H0*H0 * (Omega0 / (a*a*a) + OmegaLambda); // Hubble parameter squared
  const double x_c = Omega0 - 1.; // x_c = Omega0 - 1
  const double Delta_c = 200.; // 18. * M_PI*M_PI + 82. * x_c - 39. * x_c*x_c; // Critical overdensity factor
  const double rho_crit0 = 3. * H0*H0 / (8. * M_PI * G); // Critical density today [g/cm^3]
  const double rho_crit = 3. * H2 / (8. * M_PI * G); // Critical density [g/cm^3]
  const double rho_vir = Delta_c * rho_crit; // Virial density [g/cm^3]
  const double length_to_cgs = a * UnitLength_in_cm / h;
  const double volume_to_cgs = length_to_cgs * length_to_cgs * length_to_cgs;
  const double mass_to_cgs = UnitMass_in_g / h;
  const double density_to_cgs = mass_to_cgs / volume_to_cgs;
  const double rho_vir_code = rho_vir / density_to_cgs; // Virial density in code units
  const double RadiusMin = 1e-2 * h / a; // 10 pc
  const double RadiusMax = BoxSize; // r_box
  Radius2Min = RadiusMin * RadiusMin; // r_min^2
  const double LogRadiusMin = log10(RadiusMin); // log(r_min)
  const double LogRadiusMax = log10(RadiusMax); // log(r_max)
  LogRadius2Min = 2. * LogRadiusMin; // log(r_min^2)
  const double InvDlogRadius = double(n_bins) / (LogRadiusMax - LogRadiusMin); // n / (log(r_max) - log(r_min))
  InvDlogRadius2 = 0.5 * InvDlogRadius; // n / (log(r_max^2) - log(r_min^2))
  const double DlogRadius = (LogRadiusMax - LogRadiusMin) / double(n_bins); // (log(r_max) - log(r_min)) / n
  for (int i = 0; i < n_bins; ++i) {
    const double r_enc = pow(10., LogRadiusMin + float(i+1) * DlogRadius); // Upper edge of bin
    const double V_enc = 4. * M_PI / 3. * r_enc * r_enc * r_enc; // Enclosed volume
    M_to_rho_vir[i] = 1. / (V_enc * rho_vir_code); // Mass to virial density
  }
  if (VERBOSE > 1 && ThisTask == 0) {
    cout << "\n[RadiusMin, RadiusMax] = [" << RadiusMin << ", " << RadiusMax << "] ckpc/h = ["
          << RadiusMin*length_to_cgs/kpc << ", " << RadiusMax*length_to_cgs/kpc << "] kpc"
          << "\n[LogRadiusMin, LogRadiusMax] = [" << LogRadiusMin << ", " << LogRadiusMax << "] in units of (ckpc/h)^2"
          << "\nn_bins = " << n_bins << ", InvDlogRadius = " << InvDlogRadius << ", InvDlogRadius2 = " << InvDlogRadius2 << endl;
    float LogRadiusEdges[n_edges], RadiusEdges[n_edges], V_enc[n_bins];
    for (int i = 0; i < n_edges; ++i) {
      LogRadiusEdges[i] = LogRadiusMin + float(i) * DlogRadius;
      RadiusEdges[i] = pow(10., LogRadiusEdges[i]);
    }
    for (int i = 0; i < n_bins; ++i) {
      const double r_enc = pow(10., LogRadiusMin + float(i+1) * DlogRadius); // Upper edge of bin
      V_enc[i] = 4. * M_PI / 3. * r_enc * r_enc * r_enc; // Enclosed volume
    }
    const double kpc3 = kpc * kpc * kpc;
    RadiusEdges[0] = 0.; // r = 0
    cout << "\nLogRadiusEdges = [" << LogRadiusEdges[0] << ", " << LogRadiusEdges[1] << ", " << LogRadiusEdges[2] << " ... "
          << LogRadiusEdges[n_edges-3] << ", " << LogRadiusEdges[n_edges-2] << ", " << LogRadiusEdges[n_edges-1] << "] in units of ckpc/h" << endl;
    cout << "\nRadiusEdges = [" << RadiusEdges[0] << ", " << RadiusEdges[1] << ", " << RadiusEdges[2] << " ... "
          << RadiusEdges[n_edges-3] << ", " << RadiusEdges[n_edges-2] << ", " << RadiusEdges[n_edges-1] << "] ckpc/h\n            = ["
          << RadiusEdges[0]*length_to_cgs/kpc << ", " << RadiusEdges[1]*length_to_cgs/kpc << ", " << RadiusEdges[2]*length_to_cgs/kpc << " ... "
          << RadiusEdges[n_edges-3]*length_to_cgs/kpc << ", " << RadiusEdges[n_edges-2]*length_to_cgs/kpc << ", " << RadiusEdges[n_edges-1]*length_to_cgs/kpc << "] kpc" << endl;
    cout << "\nV_enc = [" << V_enc[0] << ", " << V_enc[1] << ", " << V_enc[2] << " ... "
          << V_enc[n_bins-3] << ", " << V_enc[n_bins-2] << ", " << V_enc[n_bins-1] << "] (ckpc/h)^3\n      = ["
          << V_enc[0]*volume_to_cgs/kpc3 << ", " << V_enc[1]*volume_to_cgs/kpc3 << ", " << V_enc[2]*volume_to_cgs/kpc3 << " ... "
          << V_enc[n_bins-3]*volume_to_cgs/kpc3 << ", " << V_enc[n_bins-2]*volume_to_cgs/kpc3 << ", " << V_enc[n_bins-1]*volume_to_cgs/kpc3 << "] kpc^3" << endl;
    cout << "\nM_to_rho_vir = [" << M_to_rho_vir[0] << ", " << M_to_rho_vir[1] << ", " << M_to_rho_vir[2] << " ... "
          << M_to_rho_vir[n_bins-3] << ", " << M_to_rho_vir[n_bins-2] << ", " << M_to_rho_vir[n_bins-1] << "] in units of (ckpc/h)^3 / (g/cm^3)" << endl;
  }
#ifdef GPU
  if (ThisTask < NDevice) {
    cudaMemcpy(d_M_to_rho_vir, M_to_rho_vir, n_bins * sizeof(float), cudaMemcpyHostToDevice);
  }
#endif
}

static void calculate_distances()
{
  const float BoxSizeF = BoxSize, BoxHalfF = BoxSize / 2.; // Single precision
  const myint n_per_block = 64;
  const myint n_blocks_grp = (n_grps + n_per_block - 1) / n_per_block;
  const myint n_blocks_sub = (n_subs + n_per_block - 1) / n_per_block;
#ifdef GPU
  if (n_gas_hr > 0) {
    calculate_minimum_distance<<<n_blocks_grp, n_per_block>>>(d_n_grps, d_GroupPos, n_gas_hr, d_r_gas_hr, d_Group_MinDistGasHR, BoxSizeF, BoxHalfF);
    calculate_minimum_distance<<<n_blocks_sub, n_per_block>>>(d_n_subs, d_SubhaloPos, n_gas_hr, d_r_gas_hr, d_Subhalo_MinDistGasHR, BoxSizeF, BoxHalfF);
  }
  if (n_gas_lr > 0) {
    calculate_minimum_distance<<<n_blocks_grp, n_per_block>>>(d_n_grps, d_GroupPos, n_gas_lr, d_r_gas_lr, d_Group_MinDistGasLR, BoxSizeF, BoxHalfF);
    calculate_minimum_distance<<<n_blocks_sub, n_per_block>>>(d_n_subs, d_SubhaloPos, n_gas_lr, d_r_gas_lr, d_Subhalo_MinDistGasLR, BoxSizeF, BoxHalfF);
  }
  calculate_minimum_distance<<<n_blocks_grp, n_per_block>>>(d_n_grps, d_GroupPos, n_dm, d_r_dm, d_Group_MinDistDM, BoxSizeF, BoxHalfF);
  calculate_minimum_distance<<<n_blocks_sub, n_per_block>>>(d_n_subs, d_SubhaloPos, n_dm, d_r_dm, d_Subhalo_MinDistDM, BoxSizeF, BoxHalfF);
  calculate_minimum_distance<<<n_blocks_grp, n_per_block>>>(d_n_grps, d_GroupPos, n_p2, d_r_p2, d_Group_MinDistP2, BoxSizeF, BoxHalfF);
  calculate_minimum_distance<<<n_blocks_sub, n_per_block>>>(d_n_subs, d_SubhaloPos, n_p2, d_r_p2, d_Subhalo_MinDistP2, BoxSizeF, BoxHalfF);
  calculate_minimum_distance<<<n_blocks_grp, n_per_block>>>(d_n_grps, d_GroupPos, n_p3, d_r_p3, d_Group_MinDistP3, BoxSizeF, BoxHalfF);
  calculate_minimum_distance<<<n_blocks_sub, n_per_block>>>(d_n_subs, d_SubhaloPos, n_p3, d_r_p3, d_Subhalo_MinDistP3, BoxSizeF, BoxHalfF);
  if (n_stars_hr > 0) {
    calculate_minimum_distance<<<n_blocks_grp, n_per_block>>>(d_n_grps, d_GroupPos, n_stars_hr, d_r_star_hr, d_Group_MinDistStarsHR, BoxSizeF, BoxHalfF);
    calculate_minimum_distance<<<n_blocks_sub, n_per_block>>>(d_n_subs, d_SubhaloPos, n_stars_hr, d_r_star_hr, d_Subhalo_MinDistStarsHR, BoxSizeF, BoxHalfF);
  }
  if (n_stars_lr > 0) {
    calculate_minimum_distance<<<n_blocks_grp, n_per_block>>>(d_n_grps, d_GroupPos, n_stars_lr, d_r_star_lr, d_Group_MinDistStarsLR, BoxSizeF, BoxHalfF);
    calculate_minimum_distance<<<n_blocks_sub, n_per_block>>>(d_n_subs, d_SubhaloPos, n_stars_lr, d_r_star_lr, d_Subhalo_MinDistStarsLR, BoxSizeF, BoxHalfF);
  }
  calculate_R_vir<<<n_blocks_sub, n_per_block>>>(d_n_subs, d_SubhaloPos, d_R_vir, d_M_vir, d_M_gas, d_M_stars, d_M_to_rho_vir, n_gas, d_r_gas, d_m_gas, n_dm, d_r_dm, MassDM,
                                                 n_p2, d_r_p2, d_m_p2, n_p3, d_r_p3, MassP3, n_stars, d_r_star, d_m_star,
                                                 BoxSizeF, BoxHalfF, Radius2Min, LogRadius2Min, InvDlogRadius2);
#else
  if (n_gas_hr > 0) {
    calculate_minimum_distance(n_grps, GroupPos, n_gas_hr, r_gas_hr, Group_MinDistGasHR, BoxSizeF, BoxHalfF);
    calculate_minimum_distance(n_subs, SubhaloPos, n_gas_hr, r_gas_hr, Subhalo_MinDistGasHR, BoxSizeF, BoxHalfF);
  }
  if (n_gas_lr > 0) {
    calculate_minimum_distance(n_grps, GroupPos, n_gas_lr, r_gas_lr, Group_MinDistGasLR, BoxSizeF, BoxHalfF);
    calculate_minimum_distance(n_subs, SubhaloPos, n_gas_lr, r_gas_lr, Subhalo_MinDistGasLR, BoxSizeF, BoxHalfF);
  }
  calculate_minimum_distance(n_grps, GroupPos, n_dm, r_dm, Group_MinDistDM, BoxSizeF, BoxHalfF);
  calculate_minimum_distance(n_subs, SubhaloPos, n_dm, r_dm, Subhalo_MinDistDM, BoxSizeF, BoxHalfF);
  calculate_minimum_distance(n_grps, GroupPos, n_p2, r_p2, Group_MinDistP2, BoxSizeF, BoxHalfF);
  calculate_minimum_distance(n_subs, SubhaloPos, n_p2, r_p2, Subhalo_MinDistP2, BoxSizeF, BoxHalfF);
  calculate_minimum_distance(n_grps, GroupPos, n_p3, r_p3, Group_MinDistP3, BoxSizeF, BoxHalfF);
  calculate_minimum_distance(n_subs, SubhaloPos, n_p3, r_p3, Subhalo_MinDistP3, BoxSizeF, BoxHalfF);
  if (n_stars_hr > 0) {
    calculate_minimum_distance(n_grps, GroupPos, n_stars_hr, r_star_hr, Group_MinDistStarsHR, BoxSizeF, BoxHalfF);
    calculate_minimum_distance(n_subs, SubhaloPos, n_stars_hr, r_star_hr, Subhalo_MinDistStarsHR, BoxSizeF, BoxHalfF);
  }
  if (n_stars_lr > 0) {
    calculate_minimum_distance(n_grps, GroupPos, n_stars_lr, r_star_lr, Group_MinDistStarsLR, BoxSizeF, BoxHalfF);
    calculate_minimum_distance(n_subs, SubhaloPos, n_stars_lr, r_star_lr, Subhalo_MinDistStarsLR, BoxSizeF, BoxHalfF);
  }
  calculate_R_vir(n_subs, SubhaloPos, R_vir, M_vir, M_gas, M_stars, M_to_rho_vir, n_gas, r_gas, m_gas, n_dm, r_dm, MassDM,
                  n_p2, r_p2, m_p2, n_p3, r_p3, MassP3, n_stars, r_star, m_star,
                  BoxSizeF, BoxHalfF, Radius2Min, LogRadius2Min, InvDlogRadius2);
#endif
}

__global__ void calculate_minimum_distance(myint n_halo, float *r_halo, myint n_part, float *r_part, float *r_min, float BoxSizeF, float BoxHalfF)
{
#ifdef GPU
  const myint first_halo = blockIdx.x * blockDim.x + threadIdx.x;
  const myint stride = blockDim.x * gridDim.x;
#else // CPU
  const myint first_halo = ThisTask, stride = NTask;
#endif
  myint i, j, i3, j3;
  float x1, y1, z1, x2, y2, z2, dx, dy, dz, r2, r2_comp;
  for (i = first_halo; i < n_halo; i += stride) {
    i3 = 3 * i; // 3D index
    x1 = r_halo[i3]; // Halo position
    y1 = r_halo[i3 + 1];
    z1 = r_halo[i3 + 2];
    r2_comp = FLT_MAX; // Initialize to a large value
    for (j = 0; j < n_part; ++j) {
      j3 = 3 * j; // 3D index
      x2 = r_part[j3]; // Particle position
      y2 = r_part[j3 + 1];
      z2 = r_part[j3 + 2];

      dx = shortest_distance(x1, x2, BoxSizeF, BoxHalfF);
      dy = shortest_distance(y1, y2, BoxSizeF, BoxHalfF);
      dz = shortest_distance(z1, z2, BoxSizeF, BoxHalfF);

      r2 = dx*dx + dy*dy + dz*dz; // Distance squared
      if (r2 < r2_comp)
        r2_comp = r2;
    }
    r_min[i] = sqrt(r2_comp); // Minimum distance
  }
}

static double single_minimum_distance(double r[3], myint n_part, float *r_part, double BoxSize, double BoxHalf)
{
  const double x1 = r[0], y1 = r[1], z1 = r[2];
  double r2_comp = DBL_MAX; // Initialize to a large value
  for (myint i = ThisTask; i < n_part; i += NTask) {
    const myint i3 = 3 * i; // 3D index
    const double x2 = r_part[i3]; // Particle position
    const double y2 = r_part[i3 + 1];
    const double z2 = r_part[i3 + 2];

    const double dx = shortest_distance(x1, x2, BoxSize, BoxHalf);
    const double dy = shortest_distance(y1, y2, BoxSize, BoxHalf);
    const double dz = shortest_distance(z1, z2, BoxSize, BoxHalf);

    const double r2 = dx*dx + dy*dy + dz*dz; // Distance squared
    if (r2 < r2_comp)
      r2_comp = r2;
  }
#ifdef MPI
  MPI_Allreduce(MPI_IN_PLACE, &r2_comp, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
  return sqrt(r2_comp); // Minimum distance
}

static double single_maximum_distance(double r[3], myint n_part, float *r_part, double BoxSize, double BoxHalf)
{
  const double x1 = r[0], y1 = r[1], z1 = r[2];
  double r2_comp = 0.; // Initialize to a large value
  myint n_part_task, first_part_task;
  equal_work(n_part, ThisTask, NTask, n_part_task, first_part_task);
  float *r_part_task = r_part + 3 * first_part_task;
  for (myint i = 0; i < n_part_task; ++i) {
    const myint i3 = 3 * i; // 3D index
    const double x2 = r_part_task[i3]; // Particle position
    const double y2 = r_part_task[i3 + 1];
    const double z2 = r_part_task[i3 + 2];

    const double dx = shortest_distance(x1, x2, BoxSize, BoxHalf);
    const double dy = shortest_distance(y1, y2, BoxSize, BoxHalf);
    const double dz = shortest_distance(z1, z2, BoxSize, BoxHalf);

    const double r2 = dx*dx + dy*dy + dz*dz; // Distance squared
    if (r2 > r2_comp)
      r2_comp = r2;
  }
#ifdef MPI
  MPI_Allreduce(MPI_IN_PLACE, &r2_comp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
  return sqrt(r2_comp); // Maximum distance
}

__global__ void calculate_R_vir(myint n_halo, float *r_halo, float *R_vir, float *M_vir, float *M_gas, float *M_stars, float M_to_rho_vir[n_bins],
                                myint n_gas, float *r_gas, float *m_gas, myint n_dm, float *r_dm, float MassDM,
                                myint n_p2, float *r_p2, float *m_p2, myint n_p3, float *r_p3, float MassP3,
                                myint n_stars, float *r_star, float *m_star,
                                float BoxSizeF, float BoxHalfF, float Radius2Min, float LogRadius2Min, float InvDlogRadius2)
{
#ifdef GPU
  const myint first_halo = blockIdx.x * blockDim.x + threadIdx.x;
  const myint stride = blockDim.x * gridDim.x;
#else // CPU
  const myint first_halo = ThisTask, stride = NTask;
#endif
  myint i, j, i3, j3;
  int ibin, ivir;
  const float LogRadiusMin = 0.5 * LogRadius2Min, DlogRadius = 0.5 / InvDlogRadius2;
  float x1, y1, z1, x2, y2, z2, dx, dy, dz, r2, frac;
  float M_gas_enc[n_bins], M_dm_enc[n_bins], M_stars_enc[n_bins], M_enc[n_bins], rho_enc[n_bins]; // Enclosed mass
  for (i = first_halo; i < n_halo; i += stride) {
    for (ibin = 0; ibin < n_bins; ++ibin) {
      M_gas_enc[ibin] = 0.; // Reset enclosed gas mass
      M_dm_enc[ibin] = 0.; // Reset enclosed dm mass
      M_stars_enc[ibin] = 0.; // Reset enclosed stellar mass
    }
    i3 = 3 * i; // 3D index
    x1 = r_halo[i3]; // Halo position
    y1 = r_halo[i3 + 1];
    z1 = r_halo[i3 + 2];
    // Calculate enclosed gas mass
    for (j = 0; j < n_gas; ++j) {
      j3 = 3 * j; // 3D index
      x2 = r_gas[j3]; // Particle position
      y2 = r_gas[j3 + 1];
      z2 = r_gas[j3 + 2];

      dx = shortest_distance(x1, x2, BoxSizeF, BoxHalfF);
      dy = shortest_distance(y1, y2, BoxSizeF, BoxHalfF);
      dz = shortest_distance(z1, z2, BoxSizeF, BoxHalfF);

      r2 = dx*dx + dy*dy + dz*dz; // Distance squared
      if (r2 < Radius2Min) {
        M_gas_enc[0] += m_gas[j]; // Bin gas mass
      } else {
        ibin = floor((log10(r2) - LogRadius2Min) * InvDlogRadius2); // Bin index
        M_gas_enc[ibin] += m_gas[j]; // Bin gas mass
      }
    }
    // Calculate enclosed dm mass
    for (j = 0; j < n_dm; ++j) {
      j3 = 3 * j; // 3D index
      x2 = r_dm[j3]; // Particle position
      y2 = r_dm[j3 + 1];
      z2 = r_dm[j3 + 2];

      dx = shortest_distance(x1, x2, BoxSizeF, BoxHalfF);
      dy = shortest_distance(y1, y2, BoxSizeF, BoxHalfF);
      dz = shortest_distance(z1, z2, BoxSizeF, BoxHalfF);

      r2 = dx*dx + dy*dy + dz*dz; // Distance squared
      if (r2 < Radius2Min) {
        M_dm_enc[0] += MassDM; // Bin dm mass
      } else {
        ibin = floor((log10(r2) - LogRadius2Min) * InvDlogRadius2); // Bin index
        M_dm_enc[ibin] += MassDM; // Bin dm mass
      }
    }
    // Calculate enclosed p2 mass
    for (j = 0; j < n_p2; ++j) {
      j3 = 3 * j; // 3D index
      x2 = r_p2[j3]; // Particle position
      y2 = r_p2[j3 + 1];
      z2 = r_p2[j3 + 2];

      dx = shortest_distance(x1, x2, BoxSizeF, BoxHalfF);
      dy = shortest_distance(y1, y2, BoxSizeF, BoxHalfF);
      dz = shortest_distance(z1, z2, BoxSizeF, BoxHalfF);

      r2 = dx*dx + dy*dy + dz*dz; // Distance squared
      if (r2 < Radius2Min) {
        M_dm_enc[0] += m_p2[j]; // Bin p2 mass
      } else {
        ibin = floor((log10(r2) - LogRadius2Min) * InvDlogRadius2); // Bin index
        M_dm_enc[ibin] += m_p2[j]; // Bin p2 mass
      }
    }
    // Calculate enclosed p3 mass
    for (j = 0; j < n_p3; ++j) {
      j3 = 3 * j; // 3D index
      x2 = r_p3[j3]; // Particle position
      y2 = r_p3[j3 + 1];
      z2 = r_p3[j3 + 2];

      dx = shortest_distance(x1, x2, BoxSizeF, BoxHalfF);
      dy = shortest_distance(y1, y2, BoxSizeF, BoxHalfF);
      dz = shortest_distance(z1, z2, BoxSizeF, BoxHalfF);

      r2 = dx*dx + dy*dy + dz*dz; // Distance squared
      if (r2 < Radius2Min) {
        M_dm_enc[0] += MassP3; // Bin p3 mass
      } else {
        ibin = floor((log10(r2) - LogRadius2Min) * InvDlogRadius2); // Bin index
        M_dm_enc[ibin] += MassP3; // Bin p3 mass
      }
    }
    // Calculate enclosed star mass
    for (j = 0; j < n_stars; ++j) {
      j3 = 3 * j; // 3D index
      x2 = r_star[j3]; // Particle position
      y2 = r_star[j3 + 1];
      z2 = r_star[j3 + 2];

      dx = shortest_distance(x1, x2, BoxSizeF, BoxHalfF);
      dy = shortest_distance(y1, y2, BoxSizeF, BoxHalfF);
      dz = shortest_distance(z1, z2, BoxSizeF, BoxHalfF);

      r2 = dx*dx + dy*dy + dz*dz; // Distance squared
      if (r2 < Radius2Min) {
        M_stars_enc[0] += m_star[j]; // Bin star mass
      } else {
        ibin = floor((log10(r2) - LogRadius2Min) * InvDlogRadius2); // Bin index
        M_stars_enc[ibin] += m_star[j]; // Bin star mass
      }
    }
    // Convert to enclosed densities
    for (ibin = 1; ibin < n_bins; ++ibin) {
      M_gas_enc[ibin] += M_gas_enc[ibin-1]; // Enclosed gas mass
      M_dm_enc[ibin] += M_dm_enc[ibin-1]; // Enclosed dm mass
      M_stars_enc[ibin] += M_stars_enc[ibin-1]; // Enclosed star mass
    }
    for (ibin = 0; ibin < n_bins; ++ibin) {
      M_enc[ibin] = M_gas_enc[ibin] + M_dm_enc[ibin] + M_stars_enc[ibin]; // Enclosed mass
      rho_enc[ibin] = M_enc[ibin] * M_to_rho_vir[ibin]; // Enclosed density [rho_vir]
    }
    // Calculate virial radius
    for (ivir = n_bins_minus1; ivir >= 0; --ivir) {
      if (rho_enc[ivir] > 1.)
        break; // Find the last bin with rho_enc > 1
    }
    if (ivir < 0) {
      R_vir[i] = 0.; // No virial radius
      M_vir[i] = 0.; // No virial mass
      M_gas[i] = 0.; // No gas mass
      M_stars[i] = 0.; // No stellar mass
    } else {
      // Log interpolation to find the virial radius and masses
      ++ivir; // Move to the next bin (ensuring rho_enc < 1 on the right)
      frac = log10(rho_enc[ivir-1]) / log10(rho_enc[ivir-1]/rho_enc[ivir]); // Interpolation coordinate
      R_vir[i] = pow(10., LogRadiusMin + (float(ivir) + frac) * DlogRadius); // Virial radius
      M_vir[i] = exp(log(M_enc[ivir-1]) + frac * log(M_enc[ivir]/M_enc[ivir-1])); // Virial mass
      if (M_gas_enc[ivir-1] <= 0.) { // Linear interpolation
        M_gas[i] = frac * M_gas_enc[ivir]; // Gas mass (<R_vir)
      } else { // Log interpolation
        M_gas[i] = exp(log(M_gas_enc[ivir-1]) + frac * log(M_gas_enc[ivir]/M_gas_enc[ivir-1])); // Gas mass (<R_vir)
      }
      if (M_stars_enc[ivir-1] <= 0.) { // Linear interpolation
        M_stars[i] = frac * M_stars_enc[ivir]; // Stellar mass (<R_vir)
      } else { // Log interpolation
        M_stars[i] = exp(log(M_stars_enc[ivir-1]) + frac * log(M_stars_enc[ivir]/M_stars_enc[ivir-1])); // Stellar mass (<R_vir)
      }
    }
  }
}
