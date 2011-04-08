
/*
 * Gaussian Mixture Model Clustering with CUDA
 *
 * Author: Andrew Pangborn
 *
 * Department of Computer Engineering
 * Rochester Institute of Technology
 * 
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h> 

// includes, project
#include <cutil.h>
#include "gaussian.h"
#include "invert_matrix.h"

// includes, kernels
#include <theta_kernel.cu>

// Function prototypes
extern "C" float* readData(char* f, int* ndims, int*nevents);
int validateArguments(int argc, char** argv, int* num_clusters, int* target_num_clusters, int* device);
void writeCluster(FILE* f, clusters_t clusters, int c,  int num_dimensions);
void printCluster(clusters_t clusters, int c, int num_dimensions);
float cluster_distance(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions);
void copy_cluster(clusters_t dest, int c_dest, clusters_t src, int c_src, int num_dimensions);
void add_clusters(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) {
  int original_num_clusters, desired_num_clusters, stop_number;
  int device = 0;
    
  // For profiling the seed kernel
  clock_t seed_start, seed_end, seed_total;

  // For profiling the regroup kernel
  clock_t regroup_start, regroup_end, regroup_total;
  int regroup_iterations = 0;
    
  // for profiling the reestimate_parameters kernel
  clock_t params_start, params_end, params_total;
  int params_iterations = 0;
    
  // for profiling the constants kernel
  clock_t constants_start, constants_end, constants_total;
  int constants_iterations = 0;

  // for profiling individual kernels
  clock_t e1_start, e1_stop, e1_total;
  clock_t e2_start, e2_stop, e2_total;
  clock_t m1_start, m1_stop, m1_total;
  clock_t m2_start, m2_stop, m2_total;
  clock_t m3_start, m3_stop, m3_total;

  e1_total = e2_total = 0;
  m1_total = m2_total = m3_total = 0;
    
  regroup_total = regroup_iterations = 0;
  params_total = params_iterations = 0;
  constants_total = constants_iterations = 0;
     
  // Set the device to run on... 0 for GTX 480, 1 for GTX 285 on oak
  int GPUCount;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&GPUCount));
  printf("GPUCount %d\n",GPUCount);
  if(GPUCount == 0) {
    PRINT("Only 1 CUDA device found, defaulting to it.\n");
    device = 0;
  } else if(GPUCount >= 1 && DEVICE < GPUCount) {
    PRINT("Multiple CUDA devices found, selecting based on compiled default: %d\n",DEVICE);
    device = DEVICE;
  } else {
    printf("Fatal Error: Unable to set device to %d, not enough GPUs.\n",DEVICE);
    exit(2);
  }
  CUDA_SAFE_CALL(cudaSetDevice(device));
    
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  //PRINT("\nUsing device - %s\n\n", prop.name);
  printf("\nUsing device - %s\n\n", prop.name);
    
  // Keep track of total time
  unsigned int total_timer = 0;
  CUT_SAFE_CALL(cutCreateTimer(&total_timer));
  CUT_SAFE_CALL(cutStartTimer(total_timer));
    
  // For profiling input parsing
  unsigned int io_timer = 0;
  CUT_SAFE_CALL(cutCreateTimer(&io_timer));
    
  // For CPU processing
  unsigned int cpu_timer = 0;
  CUT_SAFE_CALL(cutCreateTimer(&cpu_timer));

  // Keep track of gpu memcpying
  unsigned int memcpy_timer = 0;
  CUT_SAFE_CALL(cutCreateTimer(&memcpy_timer));
   
  CUT_SAFE_CALL(cutStartTimer(io_timer));
  // Validate the command-line arguments, parse # of clusters, etc 
  int error = validateArguments(argc,argv,&original_num_clusters,&desired_num_clusters,&device);
    
  // Don't continue if we had a problem with the program arguments
  if(error) {
    return 1;
  }
    
  // Number of clusters to stop iterating at.
  if(desired_num_clusters == 0) {
    stop_number = 1;
  } else {
    stop_number = desired_num_clusters;
  }
    
  int num_dimensions;
  int num_events;
    
  // Read FCS data   
  PRINT("Parsing input file...");
  // This stores the data in a 1-D array with consecutive values being the dimensions from a single event
  // (num_events by num_dimensions matrix)
  float* fcs_data_by_event = readData(argv[2],&num_dimensions,&num_events);    

  printf("READCSV: ndims = %d, nevents = %d\n", num_dimensions, num_events);

  if(!fcs_data_by_event) {
    printf("Error parsing input file. This could be due to an empty file ");
    printf("or an inconsistent number of dimensions. Aborting.\n");
    return 1;
  }
    
  // Transpose the event data (allows coalesced access pattern in E-step kernel)
  // This has consecutive values being from the same dimension of the data 
  // (num_dimensions by num_events matrix)
  float* fcs_data_by_dimension  = (float*) malloc(sizeof(float)*num_events*num_dimensions);
    
  for(int e=0; e<num_events; e++) {
    for(int d=0; d<num_dimensions; d++) {
      fcs_data_by_dimension[d*num_events+e] = fcs_data_by_event[e*num_dimensions+d];
    }
  }    

  CUT_SAFE_CALL(cutStopTimer(io_timer));
   
  PRINT("Number of events: %d\n",num_events);
  PRINT("Number of dimensions: %d\n\n",num_dimensions);
    
  PRINT("Starting with %d cluster(s), will stop at %d cluster(s).\n",original_num_clusters,stop_number);
   
  CUT_SAFE_CALL(cutStartTimer(cpu_timer));
    
  // Setup the cluster data structures on host
  clusters_t clusters;
  clusters.N = (float*) malloc(sizeof(float)*original_num_clusters);
  clusters.pi = (float*) malloc(sizeof(float)*original_num_clusters);
  clusters.constant = (float*) malloc(sizeof(float)*original_num_clusters);
  clusters.avgvar = (float*) malloc(sizeof(float)*original_num_clusters);
  clusters.means = (float*) malloc(sizeof(float)*num_dimensions*original_num_clusters);
  clusters.R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
  clusters.Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
  clusters.memberships = (float*) malloc(sizeof(float)*num_events*original_num_clusters);
  if(!clusters.means || !clusters.R || !clusters.Rinv || !clusters.memberships) { 
    printf("ERROR: Could not allocate memory for clusters.\n"); 
    return 1; 
  }

#ifdef CODEVAR_2B
  //scratch space to clear out clusters->R
  float *zeroR_2b = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
  for(int i = 0; i<num_dimensions*num_dimensions*original_num_clusters; i++) {
    zeroR_2b[i] = 0.0f;
  }
  float *temp_buffer_2b;
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_buffer_2b),sizeof(float)*num_dimensions*num_dimensions*original_num_clusters));
  CUDA_SAFE_CALL(cudaMemcpy(temp_buffer_2b, zeroR_2b, sizeof(float)*num_dimensions*num_dimensions*original_num_clusters, cudaMemcpyHostToDevice) );
#endif
#ifdef CODEVAR_3B
  //scratch space to clear out clusters->R
  int num_dim_blocks = num_dimensions*(num_dimensions+1)/2;
  float *zeroR_3b = (float*) malloc(sizeof(float)*original_num_clusters*num_dim_blocks);
  for(int i = 0; i<original_num_clusters*num_dim_blocks; i++) {
    zeroR_3b[i] = 0.0f;
  }
  float *temp_buffer_3b;
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_buffer_3b),sizeof(float)*original_num_clusters*num_dim_blocks));
  CUDA_SAFE_CALL(cudaMemcpy(temp_buffer_3b, zeroR_3b, sizeof(float)*original_num_clusters*num_dim_blocks, cudaMemcpyHostToDevice) );
#endif
  
  // Declare another set of clusters for saving the results of the best configuration
  clusters_t saved_clusters;
  saved_clusters.N = (float*) malloc(sizeof(float)*original_num_clusters);
  saved_clusters.pi = (float*) malloc(sizeof(float)*original_num_clusters);
  saved_clusters.constant = (float*) malloc(sizeof(float)*original_num_clusters);
  saved_clusters.avgvar = (float*) malloc(sizeof(float)*original_num_clusters);
  saved_clusters.means = (float*) malloc(sizeof(float)*num_dimensions*original_num_clusters);
  saved_clusters.R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
  saved_clusters.Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
  saved_clusters.memberships = (float*) malloc(sizeof(float)*num_events*original_num_clusters);
  if(!saved_clusters.means || !saved_clusters.R || !saved_clusters.Rinv || !saved_clusters.memberships) { 
    printf("ERROR: Could not allocate memory for clusters.\n"); 
    return 1; 
  }

  // Setup the cluster data structures on host
  clusters_t scratch_clusters;
  scratch_clusters.N = (float*) malloc(sizeof(float)*original_num_clusters);
  scratch_clusters.pi = (float*) malloc(sizeof(float)*original_num_clusters);
  scratch_clusters.constant = (float*) malloc(sizeof(float)*original_num_clusters);
  scratch_clusters.avgvar = (float*) malloc(sizeof(float)*original_num_clusters);
  scratch_clusters.means = (float*) malloc(sizeof(float)*num_dimensions*original_num_clusters);
  scratch_clusters.R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
  scratch_clusters.Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
  scratch_clusters.memberships = (float*) malloc(sizeof(float)*num_events*original_num_clusters);
  if(!scratch_clusters.means || !scratch_clusters.R || !scratch_clusters.Rinv || !scratch_clusters.memberships) { 
    printf("ERROR: Could not allocate memory for scratch_clusters.\n"); 
    return 1; 
  }
  
  DEBUG("Finished allocating memory on host for clusters.\n");
  CUT_SAFE_CALL(cutStopTimer(cpu_timer));
    
  // Setup the cluster data structures on device
  // First allocate structures on the host, CUDA malloc the arrays
  // Then CUDA malloc structures on the device and copy them over
  CUT_SAFE_CALL( cutStartTimer(memcpy_timer));
  clusters_t temp_clusters;
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.N),sizeof(float)*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.pi),sizeof(float)*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.constant),sizeof(float)*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.avgvar),sizeof(float)*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.means),sizeof(float)*num_dimensions*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.R),sizeof(float)*num_dimensions*num_dimensions*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.Rinv),sizeof(float)*num_dimensions*num_dimensions*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.memberships),sizeof(float)*num_events*original_num_clusters));
   
  // Allocate a struct on the device 
  clusters_t* d_clusters;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_clusters, sizeof(clusters_t)));
  DEBUG("Finished allocating memory on device for clusters.\n");
    
  // Copy Cluster data to device
  CUDA_SAFE_CALL(cudaMemcpy(d_clusters,&temp_clusters,sizeof(clusters_t),cudaMemcpyHostToDevice));
  CUT_SAFE_CALL( cutStopTimer(memcpy_timer));
  DEBUG("Finished copying cluster data to device.\n");

  int mem_size = num_dimensions*num_events*sizeof(float);
    
  float min_rissanen, rissanen;
    
  // allocate device memory for FCS data
  float* d_fcs_data_by_event;
  float* d_fcs_data_by_dimension;
  CUT_SAFE_CALL( cutStartTimer(memcpy_timer));
  CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_event, mem_size));
  CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_dimension, mem_size));
  DEBUG("Finished allocating memory on device for clusters.\n");
  // copy FCS to device
  CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_event, fcs_data_by_event, mem_size,cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_dimension, fcs_data_by_dimension, mem_size,cudaMemcpyHostToDevice) );
  CUT_SAFE_CALL( cutStopTimer(memcpy_timer));
  DEBUG("Finished copying FCS data to device.\n");
    
   
  //////////////// Initialization done, starting kernels //////////////// 
  DEBUG("Invoking seed_clusters kernel...");
  fflush(stdout);

  // seed_clusters sets initial pi values, 
  // finds the means / covariances and copies it to all the clusters
  seed_start = clock();
  seed_clusters<<< 1, NUM_THREADS_MSTEP >>>( d_fcs_data_by_event, d_clusters, num_dimensions, original_num_clusters, num_events);
  cudaThreadSynchronize();
  DEBUG("done.\n"); 
  CUT_CHECK_ERROR("Seed Kernel execution failed: ");
   
  DEBUG("Invoking constants kernel...",num_threads);
  // Computes the R matrix inverses, and the gaussian constant
  constants_kernel<<<original_num_clusters, 64>>>(d_clusters,original_num_clusters,num_dimensions);
  cudaThreadSynchronize();
  CUT_CHECK_ERROR("Constants Kernel execution failed: ");
  DEBUG("done.\n");
  seed_end = clock();
  seed_total = seed_end - seed_start;
    
  // Calculate an epsilon value
  //int ndata_points = num_events*num_dimensions;
  float epsilon = (1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions)*log((float)num_events*num_dimensions)*0.0001;
  float likelihood, old_likelihood;
  int iters;
    
  //epsilon = 1e-6;
  PRINT("Gaussian.cu: epsilon = %f\n",epsilon);

  // Used to hold the result from regroup kernel
  float* likelihoods = (float*) malloc(sizeof(float)*NUM_BLOCKS);
  float* d_likelihoods;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_likelihoods, sizeof(float)*NUM_BLOCKS));
    
  // Variables for GMM reduce order
  float distance, min_distance = 0.0;
  int min_c1, min_c2;
  int ideal_num_clusters;
  float* d_c;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_c, sizeof(float)));

  int num_clusters = original_num_clusters;
    
  //for(int num_clusters=original_num_clusters; num_clusters >= stop_number; num_clusters--) {
    /*************** EM ALGORITHM *****************************/
        
    // do initial regrouping
    // Regrouping means calculate a cluster membership probability
    // for each event and each cluster. Each event is independent,
    // so the events are distributed to different blocks 
    // (and hence different multiprocessors)

  //================================== EM INITIALIZE =======================
    DEBUG("Invoking regroup (E-step) kernel with %d blocks...",NUM_BLOCKS);

    //regroup = E step
    regroup_start = clock();
    e1_start = clock();
    estep1<<<dim3(NUM_BLOCKS,num_clusters), NUM_THREADS_ESTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_events,d_likelihoods);
    //cudaThreadSynchronize();
    e1_stop = clock();
    e1_total += e1_stop - e1_start;

    e2_start = clock();
    estep2<<<NUM_BLOCKS, NUM_THREADS_ESTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);
    cudaThreadSynchronize();
    e2_stop = clock();
    e2_total += e2_stop - e2_start;

    regroup_end = clock();
    regroup_total += regroup_end - regroup_start;
    regroup_iterations++;

    DEBUG("done.\n");
    DEBUG("Regroup Kernel Iteration Time: %f\n\n",((double)(regroup_end-regroup_start))/CLOCKS_PER_SEC);
    // check if kernel execution generated an error
    CUT_CHECK_ERROR("Kernel execution failed");

    // Copy the likelihood totals from each block, sum them up to get a total
    CUT_SAFE_CALL( cutStartTimer(memcpy_timer));
    CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*NUM_BLOCKS,cudaMemcpyDeviceToHost));
    CUT_SAFE_CALL( cutStopTimer(memcpy_timer));
    CUT_SAFE_CALL(cutStartTimer(cpu_timer));
    likelihood = 0.0;
    for(int i=0;i<NUM_BLOCKS;i++) {
     likelihood += likelihoods[i]; 
    }
    DEBUG("Starter Likelihood: %e\n",likelihood);
    CUT_SAFE_CALL(cutStopTimer(cpu_timer));

    float change = epsilon*2;

#ifdef PRINT_MATRIX
    printf("------------- R for all clusters befor EM: ----------------- \n");
    CUDA_SAFE_CALL(cudaMemcpy(&temp_clusters, d_clusters, sizeof(clusters_t),cudaMemcpyDeviceToHost));
    // copy all of the arrays from the structs
    CUDA_SAFE_CALL(cudaMemcpy(clusters.R, temp_clusters.R, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
   
    for(int i=0; i<num_clusters; i++) {
      printf("---------- cluster %d ----------\n", i);
      for(int j = 0; j<num_dimensions; j++) {
        for(int k = 0; k<num_dimensions; k++) {
          printf("%f ", clusters.R[i*num_dimensions*num_dimensions + j*num_dimensions+k]);
        }
        printf("\n");
      }
    }
      printf("-----------------------------------\n\n");
#endif
      
    //================================= EM BEGIN ==================================
    printf("Performing EM algorithm on %d clusters.\n",num_clusters);
    iters = 0;

    //make sure the right version is running
#ifdef CODEVAR_1
    printf("CODEVAR1:: NUM_EVENTS = %d, num_events = %d\n", NUM_EVENTS, num_events);
#endif
#ifdef CODEVAR_2A
    printf("CODEVAR2A:: NUM_EVENTS = %d, num_events = %d\n", NUM_EVENTS, num_events);
#endif
#ifdef CODEVAR_2B
    printf("CODEVAR2B:: NUM_EVENTS = %d, num_events = %d\n", NUM_EVENTS, num_events);
#endif
#ifdef CODEVAR_3A
    printf("CODEVAR3A:: NUM_EVENTS = %d, num_events = %d\n", NUM_EVENTS, num_events);
#endif
#ifdef CODEVAR_3B
    printf("CODEVAR3B:: NUM_EVENTS = %d, num_events = %d\n", NUM_EVENTS, num_events);
#endif
    
    // This is the iterative loop for the EM algorithm.
    // It re-estimates parameters, re-computes constants, and then regroups the events
    // These steps keep repeating until the change in likelihood is less than some epsilon        
    while(iters < MIN_ITERS || (iters < MAX_ITERS && fabs(change) > epsilon)) {
      old_likelihood = likelihood;
            
      DEBUG("Invoking reestimate_parameters (M-step) kernel...",num_threads);
      //params = M step
      params_start = clock();
      m1_start = clock();
      // This kernel computes a new N, pi isn't updated until compute_constants though
      mstep_N<<<num_clusters, NUM_THREADS_MSTEP>>>(d_fcs_data_by_event,d_clusters,num_dimensions,num_clusters,num_events);
      cudaThreadSynchronize();
      m1_stop = clock();
      m1_total += m1_stop - m1_start;

      // This kernel computes new means
      m2_start = clock();
      dim3 gridDim1(num_clusters,num_dimensions);
      mstep_means<<<gridDim1, NUM_THREADS_MSTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events);
      cudaThreadSynchronize();
      m2_stop = clock();
      m2_total += m2_stop - m2_start;

      // Covariance is symmetric, so we only need to compute N*(N+1)/2 matrix elements per cluster
      m3_start = clock();
#ifdef CODEVAR_1
      // Covariance is symmetric, so we only need to compute N*(N+1)/2 matrix elements per cluster
      dim3 gridDim2(num_clusters,num_dimensions*(num_dimensions+1)/2);
      mstep_covariance<<<gridDim2, NUM_THREADS_MSTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events);
#endif
#ifdef CODEVAR_2A
      // Covariance is symmetric, so we only need to compute N*(N+1)/2 matrix elements per cluster
      int num_blocks = num_clusters;
      int num_threads = num_dimensions*(num_dimensions+1)/2;
      mstep_covariance_2a<<<num_clusters, num_threads>>>(d_fcs_data_by_event,d_clusters,num_dimensions,num_clusters,num_events);
#endif
#ifdef CODEVAR_2B

       CUDA_SAFE_CALL(cudaMemcpy(temp_buffer_2b, zeroR_2b, sizeof(float)*num_dimensions*num_dimensions*num_clusters, cudaMemcpyHostToDevice) );

      // --- if blocked by number of events/block
      /* int event_block_size = NUM_EVENTS/num_dimensions; //how many events can we fit into shared memory */
      /* int num_event_blocks = num_events/event_block_size + (num_events%event_block_size == 0 ? 0:1); */

      // --- if blocked by number of blocks
      int num_event_blocks = NUM_EVENT_BLOCKS;
      int event_block_size = num_events%NUM_EVENT_BLOCKS == 0 ? num_events/NUM_EVENT_BLOCKS:num_events/(NUM_EVENT_BLOCKS-1);
      
      dim3 gridDim2(num_clusters,num_event_blocks);
      int num_threads = num_dimensions*(num_dimensions+1)/2;
      //printf("num_event_blocks = %d, event_block_size = %d, num_threads = %d\n", num_event_blocks, event_block_size, num_threads);
      mstep_covariance_2b<<<gridDim2, num_threads>>>(d_fcs_data_by_event,d_clusters,num_dimensions,num_clusters,num_events, event_block_size, num_event_blocks, temp_buffer_2b);
#endif
#ifdef CODEVAR_3A
      int num_blocks = num_dimensions*(num_dimensions+1)/2; //hopefully multiple of 32.
      //printf("num_events = %d, num_dimensions = %d\n", num_events, num_dimensions);
      mstep_covariance_3a<<<num_blocks, NUM_THREADS_MSTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events);
#endif

#ifdef CODEVAR_3B
      
       CUDA_SAFE_CALL(cudaMemcpy(temp_buffer_3b, zeroR_3b, sizeof(float)*num_clusters*num_dimensions*((num_dimensions+1)/2), cudaMemcpyHostToDevice));

       // --- if blocked by number of events/block
      int event_block_size = NUM_EVENTS/num_dimensions; //how many events can we fit into shared memory
      int num_event_blocks = num_events/event_block_size + (num_events%event_block_size == 0 ? 0:1);

      /* int num_event_blocks = NUM_EVENT_BLOCKS; */
      /* int event_block_size = num_events%NUM_EVENT_BLOCKS == 0 ? num_events/NUM_EVENT_BLOCKS:num_events/(NUM_EVENT_BLOCKS-1); */

      int num_cell_blocks = num_dimensions*(num_dimensions+1)/2; //hopefully multiple of 32.
      // printf("num_event_blocks = %d, event_block_size = %d, num_cell_blocks = %d\n", num_event_blocks, event_block_size, num_cell_blocks);
      dim3 gridDim2(num_cell_blocks, num_event_blocks);
      int num_threads = 64;
      mstep_covariance_3b<<<gridDim2, num_threads>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events, event_block_size, num_event_blocks, temp_buffer_3b);
#endif

      cudaThreadSynchronize();
      m3_stop = clock();
      m3_total += m3_stop - m3_start;
      params_end = clock();
                 
      CUT_CHECK_ERROR("M-step Kernel execution failed: ");
      params_total += params_end - params_start;
      params_iterations++;
      DEBUG("done.\n");
      DEBUG("Model Parameters Kernel Iteration Time: %f\n\n",((double)(params_end-params_start))/CLOCKS_PER_SEC);

#ifdef PRINT_MATRIX
      printf("------------- R EM AFTER STEP: %d ----------------- \n", iters);
      CUDA_SAFE_CALL(cudaMemcpy(&temp_clusters, d_clusters, sizeof(clusters_t),cudaMemcpyDeviceToHost));
      // copy all of the arrays from the structs
      CUDA_SAFE_CALL(cudaMemcpy(clusters.R, temp_clusters.R, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
      
      for(int i=0; i<num_clusters; i++) {
        printf("---------- cluster %d ----------\n", i);
        for(int j = 0; j<num_dimensions; j++) {
          for(int k = 0; k<num_dimensions; k++) {
            printf("%f ", clusters.R[i*num_dimensions*num_dimensions + j*num_dimensions+k]);
          }
          printf("\n");
        }
      }
      printf("-----------------------------------\n\n");
#endif
      
      DEBUG("Invoking constants kernel...",num_threads);

      // Inverts the R matrices, computes the constant, normalizes cluster probabilities
      constants_start = clock();
      constants_kernel<<<num_clusters, 32>>>(d_clusters,num_clusters,num_dimensions);
      cudaThreadSynchronize();
      constants_end = clock();
      CUT_CHECK_ERROR("Constants Kernel execution failed: ");
      constants_total += constants_end - constants_start;
      constants_iterations++;
      DEBUG("done.\n");
      DEBUG("Constants Kernel Iteration Time: %f\n\n",((double)(constants_end-constants_start))/CLOCKS_PER_SEC);

      DEBUG("Invoking regroup (E-step) kernel with %d blocks...",NUM_BLOCKS);

      //regroup = E step
      regroup_start = clock();

      // Compute new cluster membership probabilities for all the events
      e1_start = clock();
      estep1<<<dim3(NUM_BLOCKS,num_clusters), NUM_THREADS_ESTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_events,d_likelihoods);
      e1_stop = clock();
      e1_total += e1_stop - e1_start;

      e2_start = clock();
      estep2<<<NUM_BLOCKS, NUM_THREADS_ESTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);
      cudaThreadSynchronize();
      e2_stop = clock();
      e2_total += e2_stop - e2_start;

      CUT_CHECK_ERROR("E-step Kernel execution failed: ");
      regroup_end = clock();
      regroup_total += regroup_end - regroup_start;
      regroup_iterations++;

      DEBUG("done.\n");
      DEBUG("Regroup Kernel Iteration Time: %f\n\n",((double)(regroup_end-regroup_start))/CLOCKS_PER_SEC);
        
      // check if kernel execution generated an error
      CUT_CHECK_ERROR("Kernel execution failed");
        
      CUT_SAFE_CALL( cutStartTimer(memcpy_timer));
      CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*NUM_BLOCKS,cudaMemcpyDeviceToHost));
      CUT_SAFE_CALL( cutStopTimer(memcpy_timer));
      CUT_SAFE_CALL(cutStartTimer(cpu_timer));
      likelihood = 0.0;
      for(int i=0;i<NUM_BLOCKS;i++) {
        likelihood += likelihoods[i]; 
      }
            
      change = likelihood - old_likelihood;
      //printf("likelihood = %f\n",likelihood);
      //printf("Change in likelihood: %f\n",change);

      iters++;
      CUT_SAFE_CALL(cutStopTimer(cpu_timer));
            

      /*
      // copy clusters from the device
      CUDA_SAFE_CALL(cudaMemcpy(temp_clusters, d_clusters, sizeof(cluster)*num_clusters,cudaMemcpyDeviceToHost));
      // copy all of the arrays from the structs
      for(int i=0; i<num_clusters; i++) {
      CUDA_SAFE_CALL(cudaMemcpy(clusters[i].means, temp_clusters[i].means, sizeof(float)*num_dimensions,cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy(clusters[i].R, temp_clusters[i].R, sizeof(float)*num_dimensions*num_dimensions,cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy(clusters[i].Rinv, temp_clusters[i].Rinv, sizeof(float)*num_dimensions*num_dimensions,cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy(clusters[i].p, temp_clusters[i].p, sizeof(float)*num_events,cudaMemcpyDeviceToHost));
      clusters[i].N = temp_clusters[i].N;
      clusters[i].pi = temp_clusters[i].pi;
      clusters[i].constant = temp_clusters[i].constant;
      }
      for(int i=0; i<num_clusters; i++) {
      printf("N: %f, mean: %f, variance: %f, Rinv: %f\n",clusters[i].N,clusters[i].means[0],clusters[i].R[0],clusters[i].Rinv[0]);
      }
      printf("\n\n");
      */

    }//EM Loop
        
    // copy clusters from the device
    CUT_SAFE_CALL( cutStartTimer(memcpy_timer));
    CUDA_SAFE_CALL(cudaMemcpy(&temp_clusters, d_clusters, sizeof(clusters_t),cudaMemcpyDeviceToHost));
    // copy all of the arrays from the structs
    CUDA_SAFE_CALL(cudaMemcpy(clusters.N, temp_clusters.N, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(clusters.pi, temp_clusters.pi, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(clusters.constant, temp_clusters.constant, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(clusters.avgvar, temp_clusters.avgvar, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(clusters.means, temp_clusters.means, sizeof(float)*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(clusters.R, temp_clusters.R, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(clusters.Rinv, temp_clusters.Rinv, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(clusters.memberships, temp_clusters.memberships, sizeof(float)*num_events*num_clusters,cudaMemcpyDeviceToHost));
    CUT_SAFE_CALL( cutStopTimer(memcpy_timer));
        
    //} // outer loop from M to 1 clusters


  //================================ EM DONE ==============================
  PRINT("\nFinal rissanen Score was: %f, with %d clusters.\n",min_rissanen,ideal_num_clusters);
 
    
    
  CUT_SAFE_CALL(cutStartTimer(cpu_timer));
  char* result_suffix = ".results";
  char* summary_suffix = ".summary";
  int filenamesize1 = strlen(argv[3]) + strlen(result_suffix) + 1;
  int filenamesize2 = strlen(argv[3]) + strlen(summary_suffix) + 1;
  char* result_filename = (char*) malloc(filenamesize1);
  char* summary_filename = (char*) malloc(filenamesize2);
  strcpy(result_filename,argv[3]);
  strcpy(summary_filename,argv[3]);
  strcat(result_filename,result_suffix);
  strcat(summary_filename,summary_suffix);
    
  PRINT("Summary filename: %s\n",summary_filename);
  PRINT("Results filename: %s\n",result_filename);
  CUT_SAFE_CALL(cutStopTimer(cpu_timer));
    
  CUT_SAFE_CALL(cutStartTimer(io_timer));
  // Open up the output file for cluster summary
  FILE* outf = fopen(summary_filename,"w");
  if(!outf) {
    printf("ERROR: Unable to open file '%s' for writing.\n",argv[3]);
  }

  printf("DONE COMPUTING\n");
  // Print profiling information
  printf("Program Component\tTotal\tIters\tTime Per Iteration\n");
  printf("        Seed Kernel:\t%7.4f\t%d\t%7.4f\n",seed_total/(double)CLOCKS_PER_SEC,1, (double) seed_total / (double) CLOCKS_PER_SEC);
  printf("      E-step Kernel:\t%7.4f\t%d\t%7.4f\n",regroup_total/(double)CLOCKS_PER_SEC,regroup_iterations, (double) regroup_total / (double) CLOCKS_PER_SEC / (double) regroup_iterations);
  printf("      M-step Kernel:\t%7.4f\t%d\t%7.4f\n",params_total/(double)CLOCKS_PER_SEC,params_iterations, (double) params_total / (double) CLOCKS_PER_SEC / (double) params_iterations);
  printf("   Constants Kernel:\t%7.4f\t%d\t%7.4f\n",constants_total/(double)CLOCKS_PER_SEC,constants_iterations, (double) constants_total / (double) CLOCKS_PER_SEC / (double) constants_iterations);    
  
  // Print individual kernel times
  printf("\nTime Per Iteration By Kernel:\n");
  printf("E1 Kernel:\t%7.4f\n", e1_total / (double) CLOCKS_PER_SEC / (double) regroup_iterations);
  printf("E2 Kernel:\t%7.4f\n", e2_total / (double) CLOCKS_PER_SEC / (double) regroup_iterations);
  printf("M1 Kernel:\t%7.4f\n", m1_total / (double) CLOCKS_PER_SEC / (double) params_iterations);
  printf("M2 Kernel:\t%7.4f\n", m2_total / (double) CLOCKS_PER_SEC / (double) params_iterations);
  printf("M3 Kernel:\t%7.4f\n", m3_total / (double) CLOCKS_PER_SEC / (double) params_iterations);

//printf("done printing...\n");
  printf(" total time:\t%f\n", (double)seed_total/(double)CLOCKS_PER_SEC + (double)regroup_total/(double)CLOCKS_PER_SEC + (double)params_total/(double)CLOCKS_PER_SEC + (double)constants_total/(double)CLOCKS_PER_SEC);
  // cleanup host memory
  free(fcs_data_by_event);
  free(fcs_data_by_dimension);
  free(clusters.N);
  free(clusters.pi);
  free(clusters.constant);
  free(clusters.avgvar);
  free(clusters.means);
  free(clusters.R);
  free(clusters.Rinv);
  free(clusters.memberships);

  free(saved_clusters.N);
  free(saved_clusters.pi);
  free(saved_clusters.constant);
  free(saved_clusters.avgvar);
  free(saved_clusters.means);
  free(saved_clusters.R);
  free(saved_clusters.Rinv);
  free(saved_clusters.memberships);
    
  free(scratch_clusters.N);
  free(scratch_clusters.pi);
  free(scratch_clusters.constant);
  free(scratch_clusters.avgvar);
  free(scratch_clusters.means);
  free(scratch_clusters.R);
  free(scratch_clusters.Rinv);
  free(scratch_clusters.memberships);
   
  free(likelihoods);

  // cleanup GPU memory
  CUDA_SAFE_CALL(cudaFree(d_likelihoods));
 
  CUDA_SAFE_CALL(cudaFree(d_fcs_data_by_event));
  CUDA_SAFE_CALL(cudaFree(d_fcs_data_by_dimension));

  CUDA_SAFE_CALL(cudaFree(temp_clusters.N));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.pi));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.constant));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.avgvar));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.means));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.R));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.Rinv));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.memberships));
  CUDA_SAFE_CALL(cudaFree(d_clusters));

  return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Validate command line arguments
///////////////////////////////////////////////////////////////////////////////
int validateArguments(int argc, char** argv, int* num_clusters, int* target_num_clusters, int* device) {
  if(argc <= 6 && argc >= 4) {
    // parse num_clusters
    if(!sscanf(argv[1],"%d",num_clusters)) {
      printf("Invalid number of starting clusters\n\n");
      printUsage(argv);
      return 1;
    } 
        
    // Check bounds for num_clusters
    if(*num_clusters < 1) {
      printf("Invalid number of starting clusters\n\n");
      printUsage(argv);
      return 1;
    }
        
    // parse infile
    FILE* infile = fopen(argv[2],"r");
    if(!infile) {
      printf("Invalid infile.\n\n");
      printUsage(argv);
      return 2;
    } 
        
    // parse target_num_clusters
    if(argc >= 5) {
      if(!sscanf(argv[4],"%d",target_num_clusters)) {
        printf("Invalid number of desired clusters.\n\n");
        printUsage(argv);
        return 4;
      }
      if(*target_num_clusters > *num_clusters) {
        printf("target_num_clusters must be less than equal to num_clusters\n\n");
        printUsage(argv);
        return 4;
      }
    } else {
      *target_num_clusters = 0;
    }
        
    // Clean up so the EPA is happy
    fclose(infile);

    if(argc == 6) {
      if(!sscanf(argv[5],"%d",device)) {
        printf("Invalid device number. Not a number\n\n");
        printUsage(argv);
        return 1;
      } 
    }
    return 0;
  } else {
    printUsage(argv);
    return 1;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Print usage statement
///////////////////////////////////////////////////////////////////////////////
void printUsage(char** argv)
{
  printf("Usage: %s num_clusters infile outfile [target_num_clusters] [device]\n",argv[0]);
  printf("\t num_clusters: The number of starting clusters\n");
  printf("\t infile: ASCII space-delimited FCS data file\n");
  printf("\t outfile: Clustering results output file\n");
  printf("\t target_num_clusters: A desired number of clusters. Must be less than or equal to num_clusters\n");
  printf("\t device: CUDA device to use, default is the first device, 0.\n");
}

void writeCluster(FILE* f, clusters_t clusters, int c, int num_dimensions) {
  fprintf(f,"Probability: %f\n", clusters.pi[c]);
  fprintf(f,"N: %f\n",clusters.N[c]);
  fprintf(f,"Means: ");
  for(int i=0; i<num_dimensions; i++){
    fprintf(f,"%.3f ",clusters.means[c*num_dimensions+i]);
  }
  fprintf(f,"\n");

  fprintf(f,"\nR Matrix:\n");
  for(int i=0; i<num_dimensions; i++) {
    for(int j=0; j<num_dimensions; j++) {
      fprintf(f,"%.3f ", clusters.R[c*num_dimensions*num_dimensions+i*num_dimensions+j]);
    }
    fprintf(f,"\n");
  }
  fflush(f);   
  /*
    fprintf(f,"\nR-inverse Matrix:\n");
    for(int i=0; i<num_dimensions; i++) {
    for(int j=0; j<num_dimensions; j++) {
    fprintf(f,"%.3f ", c->Rinv[i*num_dimensions+j]);
    }
    fprintf(f,"\n");
    } 
  */
}

void printCluster(clusters_t clusters, int c, int num_dimensions) {
  writeCluster(stdout,clusters,c,num_dimensions);
}

float cluster_distance(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions) {
  // Add the clusters together, this updates pi,means,R,N and stores in temp_cluster
  add_clusters(clusters,c1,c2,temp_cluster,num_dimensions);
    
  return clusters.N[c1]*clusters.constant[c1] + clusters.N[c2]*clusters.constant[c2] - temp_cluster.N[0]*temp_cluster.constant[0];
}

void add_clusters(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions) {
  float wt1,wt2;
 
  wt1 = (clusters.N[c1]) / (clusters.N[c1] + clusters.N[c2]);
  wt2 = 1.0f - wt1;
    
  // Compute new weighted means
  for(int i=0; i<num_dimensions;i++) {
    temp_cluster.means[i] = wt1*clusters.means[c1*num_dimensions+i] + wt2*clusters.means[c2*num_dimensions+i];
  }
    
  // Compute new weighted covariance
  for(int i=0; i<num_dimensions; i++) {
    for(int j=i; j<num_dimensions; j++) {
      // Compute R contribution from cluster1
      temp_cluster.R[i*num_dimensions+j] = ((temp_cluster.means[i]-clusters.means[c1*num_dimensions+i])
                                            *(temp_cluster.means[j]-clusters.means[c1*num_dimensions+j])
                                            +clusters.R[c1*num_dimensions*num_dimensions+i*num_dimensions+j])*wt1;
      // Add R contribution from cluster2
      temp_cluster.R[i*num_dimensions+j] += ((temp_cluster.means[i]-clusters.means[c2*num_dimensions+i])
                                             *(temp_cluster.means[j]-clusters.means[c2*num_dimensions+j])
                                             +clusters.R[c2*num_dimensions*num_dimensions+i*num_dimensions+j])*wt2;
      // Because its symmetric...
      temp_cluster.R[j*num_dimensions+i] = temp_cluster.R[i*num_dimensions+j];
    }
  }
    
  // Compute pi
  temp_cluster.pi[0] = clusters.pi[c1] + clusters.pi[c2];
    
  // compute N
  temp_cluster.N[0] = clusters.N[c1] + clusters.N[c2];

  float log_determinant;
  // Copy R to Rinv matrix
  memcpy(temp_cluster.Rinv,temp_cluster.R,sizeof(float)*num_dimensions*num_dimensions);
  // Invert the matrix
  invert_cpu(temp_cluster.Rinv,num_dimensions,&log_determinant);
  // Compute the constant
  temp_cluster.constant[0] = (-num_dimensions)*0.5*logf(2*PI)-0.5*log_determinant;
    
  // avgvar same for all clusters
  temp_cluster.avgvar[0] = clusters.avgvar[0];
}

void copy_cluster(clusters_t dest, int c_dest, clusters_t src, int c_src, int num_dimensions) {
  dest.N[c_dest] = src.N[c_src];
  dest.pi[c_dest] = src.pi[c_src];
  dest.constant[c_dest] = src.constant[c_src];
  dest.avgvar[c_dest] = src.avgvar[c_src];
  memcpy(&(dest.means[c_dest*num_dimensions]),&(src.means[c_src*num_dimensions]),sizeof(float)*num_dimensions);
  memcpy(&(dest.R[c_dest*num_dimensions*num_dimensions]),&(src.R[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
  memcpy(&(dest.Rinv[c_dest*num_dimensions*num_dimensions]),&(src.Rinv[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
  // do we need to copy memberships?
}
