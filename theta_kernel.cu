/*
 * CUDA Kernels for Expectation Maximization with Gaussian Mixture Models
 *
 * Author: Andrew Pangborn
 * 
 * Department of Computer Engineering
 * Rochester Institute of Technology
 */


#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>
#include "gaussian.h"

/*
 * Compute the multivariate mean of the FCS data
 */ 
__device__ void mvtmeans(float* fcs_data, int num_dimensions, int num_events, float* means) {
    // access thread id
    int tid = threadIdx.x;

    if(tid < num_dimensions) {
        means[tid] = 0.0f;

        // Sum up all the values for the dimension
        for(int i=0; i < num_events; i++) {
            means[tid] += fcs_data[i*num_dimensions+tid];
        }

        // Divide by the # of elements to get the average
        means[tid] /= (float) num_events;
    }
}

__device__ void averageVariance(float* fcs_data, float* means, int num_dimensions, int num_events, float* avgvar) {
    // access thread id
    int tid = threadIdx.x;
    
    __shared__ float variances[NUM_DIMENSIONS];
    __shared__ float total_variance;
    
    // Compute average variance for each dimension
    if(tid < num_dimensions) {
        variances[tid] = 0.0f;
        // Sum up all the variance
        for(int j=0; j < num_events; j++) {
            // variance = (data - mean)^2
            variances[tid] += (fcs_data[j*num_dimensions + tid])*(fcs_data[j*num_dimensions + tid]);
        }
        variances[tid] /= (float) num_events;
        variances[tid] -= means[tid]*means[tid];
    }
    
    __syncthreads();
    
    if(tid == 0) {
        total_variance = 0.0f;
        for(int i=0; i<num_dimensions;i++) {
            ////printf("%f ",variances[tid]);
            total_variance += variances[i];
        }
        ////printf("\nTotal variance: %f\n",total_variance);
        *avgvar = total_variance / (float) num_dimensions;
        ////printf("Average Variance: %f\n",*avgvar);
    }
}

// Inverts an NxN matrix 'data' stored as a 1D array in-place
// 'actualsize' is N
// Computes the log of the determinant of the origianl matrix in the process
__device__ void invert(float* data, int actualsize, float* log_determinant)  {
    int maxsize = actualsize;
    int n = actualsize;
    
    if(threadIdx.x == 0) {
        *log_determinant = 0.0f;
      // sanity check        
      if (actualsize == 1) {
        *log_determinant = logf(data[0]);
        data[0] = 1.0 / data[0];
      } else {

          for (int i=1; i < actualsize; i++) data[i] /= data[0]; // normalize row 0
          for (int i=1; i < actualsize; i++)  { 
            for (int j=i; j < actualsize; j++)  { // do a column of L
              float sum = 0.0f;
              for (int k = 0; k < i; k++)  
                  sum += data[j*maxsize+k] * data[k*maxsize+i];
              data[j*maxsize+i] -= sum;
              }
            if (i == actualsize-1) continue;
            for (int j=i+1; j < actualsize; j++)  {  // do a row of U
              float sum = 0.0f;
              for (int k = 0; k < i; k++)
                  sum += data[i*maxsize+k]*data[k*maxsize+j];
              data[i*maxsize+j] = 
                 (data[i*maxsize+j]-sum) / data[i*maxsize+i];
              }
            }
            
            for(int i=0; i<actualsize; i++) {
                *log_determinant += logf(fabs(data[i*n+i]));
            }
            
          for ( int i = 0; i < actualsize; i++ )  // invert L
            for ( int j = i; j < actualsize; j++ )  {
              float x = 1.0f;
              if ( i != j ) {
                x = 0.0f;
                for ( int k = i; k < j; k++ ) 
                    x -= data[j*maxsize+k]*data[k*maxsize+i];
                }
              data[j*maxsize+i] = x / data[j*maxsize+j];
              }
          for ( int i = 0; i < actualsize; i++ )   // invert U
            for ( int j = i; j < actualsize; j++ )  {
              if ( i == j ) continue;
              float sum = 0.0f;
              for ( int k = i; k < j; k++ )
                  sum += data[k*maxsize+j]*( (i==k) ? 1.0 : data[i*maxsize+k] );
              data[i*maxsize+j] = -sum;
              }
          for ( int i = 0; i < actualsize; i++ )   // final inversion
            for ( int j = 0; j < actualsize; j++ )  {
              float sum = 0.0f;
              for ( int k = ((i>j)?i:j); k < actualsize; k++ )  
                  sum += ((j==k)?1.0:data[j*maxsize+k])*data[k*maxsize+i];
              data[j*maxsize+i] = sum;
              }
        }
    }
 }


__device__ void normalize_pi(clusters_t* clusters, int num_clusters) {
    __shared__ float sum;
    
    // TODO: could maybe use a parallel reduction..but the # of elements is really small
    // What is better: having thread 0 compute a shared sum and sync, or just have each one compute the sum?
    if(threadIdx.x == 0) {
        sum = 0.0f;
        for(int i=0; i<num_clusters; i++) {
            sum += clusters->pi[i];
        }
    }
    
    __syncthreads();
    
    for(int c=threadIdx.x; c < num_clusters; c += blockDim.x) {
        clusters->pi[threadIdx.x] /= sum;
    }
 
    __syncthreads();
}


__device__ void compute_constants(clusters_t* clusters, int num_clusters, int num_dimensions) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int num_elements = num_dimensions*num_dimensions;
    
    __shared__ float determinant_arg; // only one thread computes the inverse so we need a shared argument
    
    float log_determinant;
    
    __shared__ float matrix[NUM_DIMENSIONS*NUM_DIMENSIONS];
    
    // Invert the matrix for every cluster
    int c = blockIdx.x;
    // Copy the R matrix into shared memory for doing the matrix inversion
    for(int i=tid; i<num_elements; i+= num_threads ) {
        matrix[i] = clusters->R[c*num_dimensions*num_dimensions+i];
    }
    
    __syncthreads(); 
    
    invert(matrix,num_dimensions,&determinant_arg);

    __syncthreads(); 
    
    log_determinant = determinant_arg;
    
    // Copy the matrx from shared memory back into the cluster memory
    for(int i=tid; i<num_elements; i+= num_threads) {
        clusters->Rinv[c*num_dimensions*num_dimensions+i] = matrix[i];
    }
    
    __syncthreads();
    
    // Compute the constant
    // Equivilent to: log(1/((2*PI)^(M/2)*det(R)^(1/2)))
    // This constant is used in all E-step likelihood calculations
    if(tid == 0) {
        clusters->constant[c] = -num_dimensions*0.5*logf(2*PI) - 0.5*log_determinant;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! @param fcs_data         FCS data: [num_events]
//! @param clusters         Clusters: [num_clusters]
//! @param num_dimensions   number of dimensions in an FCS event
//! @param num_events       number of FCS events
////////////////////////////////////////////////////////////////////////////////
__global__ void
seed_clusters( float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events) 
{
    // access thread id
    int tid = threadIdx.x;
    // access number of threads in this block
    int num_threads = blockDim.x;

    // shared memory
    __shared__ float means[NUM_DIMENSIONS];
    
    // Compute the means
    mvtmeans(fcs_data, num_dimensions, num_events, means);

    __syncthreads();
    
    __shared__ float avgvar;
    
    // Compute the average variance
    averageVariance(fcs_data, means, num_dimensions, num_events, &avgvar);
        
    int num_elements;
    int row, col;
        
    // Number of elements in the covariance matrix
    num_elements = num_dimensions*num_dimensions; 

    __syncthreads();

    float seed;
    if(num_clusters > 1) {
        seed = (num_events-1.0f)/(num_clusters-1.0f);
    } else {
        seed = 0.0f;
    }
    
    // Seed the pi, means, and covariances for every cluster
    for(int c=0; c < num_clusters; c++) {
        if(tid < num_dimensions) {
            clusters->means[c*num_dimensions+tid] = fcs_data[((int)(c*seed))*num_dimensions+tid];
        }
          
        for(int i=tid; i < num_elements; i+= num_threads) {
            // Add the average variance divided by a constant, this keeps the cov matrix from becoming singular
            row = (i) / num_dimensions;
            col = (i) % num_dimensions;

            if(row == col) {
                clusters->R[c*num_dimensions*num_dimensions+i] = 1.0f;
            } else {
                clusters->R[c*num_dimensions*num_dimensions+i] = 0.0f;
            }
        }
        if(tid == 0) {
            clusters->pi[c] = 1.0f/((float)num_clusters);
            clusters->N[c] = ((float) num_events) / ((float)num_clusters);
            clusters->avgvar[c] = avgvar / COVARIANCE_DYNAMIC_RANGE;
        }
    }
}

__device__ float parallelSum(float* data, const unsigned int ndata) {
  const unsigned int tid = threadIdx.x;
  float t;

  __syncthreads();

  // Butterfly sum.  ndata MUST be a power of 2.
  for(unsigned int bit = ndata >> 1; bit > 0; bit >>= 1) {
    t = data[tid] + data[tid^bit];  __syncthreads();
    data[tid] = t;                  __syncthreads();
  }
  return data[tid];
}

__device__ void compute_indices(int num_events, int* start, int* stop) {
    // Break up the events evenly between the blocks
    int num_pixels_per_block = num_events / NUM_BLOCKS;
    // Make sure the events being accessed by the block are aligned to a multiple of 16
    num_pixels_per_block = num_pixels_per_block - (num_pixels_per_block % 16);
    
    *start = blockIdx.x * num_pixels_per_block + threadIdx.x;
    
    // Last block will handle the leftover events
    if(blockIdx.x == NUM_BLOCKS-1) {
        *stop = num_events;
    } else {
        *stop = (blockIdx.x+1) * num_pixels_per_block;
    }
}

__global__ void
estep1(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_events, float* likelihood) {
    
    // Cached cluster parameters
    __shared__ float means[NUM_DIMENSIONS];
    __shared__ float Rinv[NUM_DIMENSIONS*NUM_DIMENSIONS];
    float cluster_pi;
    float constant;
    const unsigned int tid = threadIdx.x;
 
    int start_index;
    int end_index;

    int c = blockIdx.y;

    compute_indices(num_events,&start_index,&end_index);
    
    float like;

    // This loop computes the expectation of every event into every cluster
    //
    // P(k|n) = L(x_n|mu_k,R_k)*P(k) / P(x_n)
    //
    // Compute log-likelihood for every cluster for each event
    // L = constant*exp(-0.5*(x-mu)*Rinv*(x-mu))
    // log_L = log_constant - 0.5*(x-u)*Rinv*(x-mu)
    // the constant stored in clusters[c].constant is already the log of the constant
    
    // copy the means for this cluster into shared memory
    if(tid < num_dimensions) {
        means[tid] = clusters->means[c*num_dimensions+tid];
    }

    // copy the covariance inverse into shared memory
    for(int i=tid; i < num_dimensions*num_dimensions; i+= NUM_THREADS_ESTEP) {
        Rinv[i] = clusters->Rinv[c*num_dimensions*num_dimensions+i]; 
    }
    
    cluster_pi = clusters->pi[c];
    constant = clusters->constant[c];

    // Sync to wait for all params to be loaded to shared memory
    __syncthreads();

    
    for(int event=start_index; event<end_index; event += NUM_THREADS_ESTEP) {
        like = 0.0f;
        // this does the loglikelihood calculation
        #if DIAG_ONLY
            for(int j=0; j<num_dimensions; j++) {
                like += (fcs_data[j*num_events+event]-means[j]) * (fcs_data[j*num_events+event]-means[j]) * Rinv[j*num_dimensions+j];
            }
        #else
            for(int i=0; i<num_dimensions; i++) {
                for(int j=0; j<num_dimensions; j++) {
                    like += (fcs_data[i*num_events+event]-means[i]) * (fcs_data[j*num_events+event]-means[j]) * Rinv[i*num_dimensions+j];
                }
            }
        #endif
        clusters->memberships[c*num_events+event] = -0.5f * like + constant + logf(cluster_pi); // numerator of the probability computation
    }
}

    
__global__ void
estep2(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events, float* likelihood) {
    float temp;
    float thread_likelihood = 0.0f;
    __shared__ float total_likelihoods[NUM_THREADS_ESTEP];
    float max_likelihood;
    float denominator_sum;
    
    // Break up the events evenly between the blocks
    int num_pixels_per_block = num_events / NUM_BLOCKS;
    // Make sure the events being accessed by the block are aligned to a multiple of 16
    num_pixels_per_block = num_pixels_per_block - (num_pixels_per_block % 16);
    int tid = threadIdx.x;
    
    int start_index;
    int end_index;
    start_index = blockIdx.x * num_pixels_per_block + tid;
    
    // Last block will handle the leftover events
    if(blockIdx.x == NUM_BLOCKS-1) {
        end_index = num_events;
    } else {
        end_index = (blockIdx.x+1) * num_pixels_per_block;
    }
    
    total_likelihoods[tid] = 0.0f;

    // P(x_n) = sum of likelihoods weighted by P(k) (their probability, cluster[c].pi)
    // However we use logs to prevent under/overflow
    //  log-sum-exp formula:
    //  log(sum(exp(x_i)) = max(z) + log(sum(exp(z_i-max(z))))
    for(int pixel=start_index; pixel<end_index; pixel += NUM_THREADS_ESTEP) {
        // find the maximum likelihood for this event
        max_likelihood = clusters->memberships[pixel];
        for(int c=1; c<num_clusters; c++) {
            max_likelihood = fmaxf(max_likelihood,clusters->memberships[c*num_events+pixel]);
        }

        // Compute P(x_n), the denominator of the probability (sum of weighted likelihoods)
        denominator_sum = 0.0f;
        for(int c=0; c<num_clusters; c++) {
            temp = expf(clusters->memberships[c*num_events+pixel]-max_likelihood);
            denominator_sum += temp;
        }
        temp = max_likelihood + logf(denominator_sum);
        thread_likelihood += temp;
        
        // Divide by denominator, also effectively normalize probabilities
        for(int c=0; c<num_clusters; c++) {
            clusters->memberships[c*num_events+pixel] = expf(clusters->memberships[c*num_events+pixel] - temp);
            //printf("Probability that pixel #%d is in cluster #%d: %f\n",pixel,c,clusters->memberships[c*num_events+pixel]);
        }
    }
    
    total_likelihoods[tid] = thread_likelihood;
    __syncthreads();

    temp = parallelSum(total_likelihoods,NUM_THREADS_ESTEP);
    if(tid == 0) {
        likelihood[blockIdx.x] = temp;
    }
}

__global__ void
mstep_means(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events) {
    // One block per cluster, per dimension:  (M x D) grid of blocks
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int c = blockIdx.x; // cluster number
    int d = blockIdx.y; // dimension number

    __shared__ float temp_sum[NUM_THREADS_MSTEP];
    float sum = 0.0f;
    
    for(int event=tid; event < num_events; event+= num_threads) {
        sum += fcs_data[d*num_events+event]*clusters->memberships[c*num_events+event];
    }
    temp_sum[tid] = sum;
    
    __syncthreads();
    
    if(tid == 0) {
        for(int i=1; i < num_threads; i++) {
            temp_sum[0] += temp_sum[i];
        }
        clusters->means[c*num_dimensions+d] = temp_sum[0] / clusters->N[c];
    }
    
}

__global__ void
mstep_means_transpose(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events) {
    // One block per cluster, per dimension:  (M x D) grid of blocks
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int c = blockIdx.y; // cluster number
    int d = blockIdx.x; // dimension number

    __shared__ float temp_sum[NUM_THREADS_MSTEP];
    float sum = 0.0f;
    
    for(int event=tid; event < num_events; event+= num_threads) {
        sum += fcs_data[d*num_events+event]*clusters->memberships[c*num_events+event];
    }
    temp_sum[tid] = sum;
    
    __syncthreads();
    
    if(tid == 0) {
        for(int i=1; i < num_threads; i++) {
            temp_sum[0] += temp_sum[i];
        }
        clusters->means[c*num_dimensions+d] = temp_sum[0] / clusters->N[c];
    }
    
}

__global__ void
mstep_N(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events) {
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int c = blockIdx.x;
 
    
    // Need to store the sum computed by each thread so in the end
    // a single thread can reduce to get the final sum
    __shared__ float temp_sums[NUM_THREADS_MSTEP];

    // Compute new N
    float sum = 0.0f;
    // Break all the events accross the threads, add up probabilities
    for(int event=tid; event < num_events; event += num_threads) {
        sum += clusters->memberships[c*num_events+event];
    }
    temp_sums[tid] = sum;
 
    __syncthreads();
    
    // Let the first thread add up all the intermediate sums
    // Could do a parallel reduction...doubt it's really worth it for so few elements though
    if(tid == 0) {
        clusters->N[c] = 0.0f;
        for(int j=0; j<num_threads; j++) {
            clusters->N[c] += temp_sums[j];
        }
        //printf("clusters[%d].N = %f\n",c,clusters[c].N);
        
        // Set PI to the # of expected items, and then normalize it later
        clusters->pi[c] = clusters->N[c];
    }
}
   
/*
 * Computes the row and col of a square matrix based on the index into
 * a lower triangular (with diagonal) matrix
 * 
 * Used to determine what row/col should be computed for covariance
 * based on a block index.
 */
__device__ void compute_row_col(int n, int* row, int* col) {
    int i = 0;
    for(int r=0; r < n; r++) {
        for(int c=0; c <= r; c++) {
            if(i == blockIdx.y) {  
                *row = r;
                *col = c;
                return;
            }
            i++;
        }
    }
}

//CODEVAR_2
__device__ void compute_row_col_thread(int n, int* row, int* col) {
    int i = 0;
    for(int r=0; r < n; r++) {
        for(int c=0; c <= r; c++) {
            if(i == threadIdx.x) {  
                *row = r;
                *col = c;
                return;
            }
            i++;
        }
    }
}
//CODEVAR_3
__device__ void compute_row_col_block(int n, int* row, int* col) {
    int i = 0;
    for(int r=0; r < n; r++) {
        for(int c=0; c <= r; c++) {
            if(i == blockIdx.x) {  
                *row = r;
                *col = c;
                return;
            }
            i++;
        }
    }
}

//CODEVAR_2B and CODEVAR_3B
__device__ void compute_my_event_indices(int n, int bsize, int num_b, int* e_start, int* e_end) {
  int myId = blockIdx.y;
  *e_start = myId*bsize;
  if(myId==(num_b-1)) {
    *e_end = ((myId*bsize)-n < 0 ? n:myId*bsize);
  } else {
    *e_end = myId*bsize + bsize;
  }
  
  return;
}


__device__ void compute_row_col_transpose(int n, int* row, int* col) {
    int i = 0;
    for(int r=0; r < n; r++) {
        for(int c=0; c <= r; c++) {
            if(i == blockIdx.x) {  
                *row = r;
                *col = c;
                return;
            }
            i++;
        }
    }
}
 
/*
 * Computes the covariance matrices of the data (R matrix)
 * Must be launched with a M x D*D grid of blocks: 
 *  i.e. dim3 gridDim(num_clusters,num_dimensions*num_dimensions)
 */
__global__ void
mstep_covariance(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events) {
    int tid = threadIdx.x; // easier variable name for our thread ID

    // Determine what row,col this matrix is handling, also handles the symmetric element
    int row,col,c;
    compute_row_col(num_dimensions, &row, &col);

    __syncthreads();
    
    c = blockIdx.x; // Determines what cluster this block is handling    

    int matrix_index = row * num_dimensions + col;

    // Store the means in shared memory to speed up the covariance computations
    __shared__ float means[NUM_DIMENSIONS];
    // copy the means for this cluster into shared memory
    if(tid < num_dimensions) {
        means[tid] = clusters->means[c*num_dimensions+tid];
    }
    __syncthreads();

    // Sync to wait for all params to be loaded to shared memory


    __shared__ float temp_sums[NUM_THREADS_MSTEP];
    
    float cov_sum = 0.0f;

    for(int event=tid; event < num_events; event+=NUM_THREADS_MSTEP) {
      cov_sum += (fcs_data[row*num_events+event]-means[row])*(fcs_data[col*num_events+event]-means[col])*clusters->memberships[c*num_events+event];
      //cov_sum += means[row]; //*(fcs_data[col*num_events+event]-means[col])*clusters->memberships[c*num_events+event];
    }

    temp_sums[tid] = cov_sum;

    __syncthreads();
    
    if(tid == 0) {
      cov_sum = 0.0f; 
      for(int i=0; i < NUM_THREADS_MSTEP; i++) {
        cov_sum += temp_sums[i];
      }
      if(clusters->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?
        cov_sum /= clusters->N[c];
        clusters->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
        //Set the symmetric value
        matrix_index = col*num_dimensions+row;
        //        matrix_index = col*num_dimensions+row;
        clusters->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
      } else {
        clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; // what should the variance be for an empty cluster...?
        // Set the symmetric value
        matrix_index = col*num_dimensions+row;
        clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; // what should the variance be for an empty cluster...?
      }
      // Regularize matrix - adds some variance to the diagonal elements
      // Helps keep covariance matrix non-singular (so it can be inverted)
      // The amount added is scaled down based on COVARIANCE_DYNAMIC_RANGE constant defined at top of this file
      if(row == col) {
        clusters->R[c*num_dimensions*num_dimensions+matrix_index] += clusters->avgvar[c];
      }
    }
}

/*
 * Computes the covariance matrices of the data (R matrix)
 * Must be launched with a M blocks and D x D/2 threads: 
 */
__global__ void
mstep_covariance_2a(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events) {
    int tid = threadIdx.x; // easier variable name for our thread ID

    // Determine what row,col this matrix is handling, also handles the symmetric element
    int row,col,c;
    compute_row_col_thread(num_dimensions, &row, &col);

    __syncthreads();
    
    c = blockIdx.x; // Determines what cluster this block is handling    

    int matrix_index = row * num_dimensions + col;

    // Store the means in shared memory to speed up the covariance computations
    __shared__ float means[NUM_DIMENSIONS];

    // copy the means for this cluster into shared memory
    if(tid < num_dimensions) {
        means[tid] = clusters->means[c*num_dimensions+tid];
    }
    
#ifdef SH_MEM_EVENTS
    __shared__ float events[NUM_EVENTS];
    for(int event=tid; event < num_events*num_dimensions; event+=blockDim.x) {
      if(event<num_events*num_dimensions) {
        events[event] = fcs_data[event];
      }
    }
#endif    
    
    // Sync to wait for all params to be loaded to shared memory
    __syncthreads();

    //__shared__ float temp_sums[NUM_THREADS_MSTEP];
    float cov_sum = 0.0f; //my local sum for the matrix element, I (thread) sum up over all N events into this var


    for(int event=0; event < num_events; event++) {
      cov_sum += (fcs_data[event*num_dimensions+row]-means[row])*(fcs_data[event*num_dimensions+col]-means[col])*clusters->memberships[c*num_events+event];
    }

    __syncthreads();

    
    if(clusters->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?
      cov_sum /= clusters->N[c];
      clusters->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
      // Set the symmetric value
      matrix_index = col*num_dimensions+row;
      clusters->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
    } else {
      clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; // what should the variance be for an empty cluster...?
      // Set the symmetric value
      matrix_index = col*num_dimensions+row;
      clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; // what should the variance be for an empty cluster...?
    }
    
    // Regularize matrix - adds some variance to the diagonal elements
    // Helps keep covariance matrix non-singular (so it can be inverted)
    // The amount added is scaled down based on COVARIANCE_DYNAMIC_RANGE constant defined at top of this file
    if(row == col) {
      clusters->R[c*num_dimensions*num_dimensions+matrix_index] += clusters->avgvar[c];
    }
    
}

 
/*
 * Computes the covariance matrices of the data (R matrix)
 * Must be launched with a M x B blocks and D x D/2 threads:
 * B is the number of event blocks (N/events_per_block)
 */
__global__ void
mstep_covariance_2b(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events, int event_block_size, int num_b, float *temp_buffer) {

  int tid = threadIdx.x; // easier variable name for our thread ID
    
    // Determine what row,col this matrix is handling, also handles the symmetric element
    int row,col,c;
    compute_row_col_thread(num_dimensions, &row, &col);

    int e_start, e_end;
    compute_my_event_indices(num_events, event_block_size, num_b, &e_start, &e_end);
     
    //__syncthreads();
    
    c = blockIdx.x; // Determines what cluster this block is handling    

    int matrix_index = row * num_dimensions + col;

    // Store the means in shared memory to speed up the covariance computations
    __shared__ float means[NUM_DIMENSIONS];
    __shared__ float myR[NUM_DIMENSIONS*NUM_DIMENSIONS];
    
    // copy the means for this cluster into shared memory
    if(tid < num_dimensions) {
        means[tid] = clusters->means[c*num_dimensions+tid];
    }

    /* for(int z = tid; z<NUM_DIMENSIONS*NUM_DIMENSIONS; z+=blockDim.x) { */
    /*   myR[z] = 0.0f; */
    /* } */
    
#ifdef SH_MEM_EVENTS
    __shared__ float events[NUM_EVENTS];
    
    for(int event=tid+e_start*num_dimensions; event < e_end*num_dimensions; event+=blockDim.x) {
      if(event<num_events*num_dimensions) {
          events[event-e_start*num_dimensions] = fcs_data[event];
        }
    }
#endif    
    
    // Sync to wait for all params to be loaded to shared memory
    __syncthreads();

    float cov_sum = 0.0f; //my local sum for the matrix element, I (thread) sum up over all N events into this var

#ifdef SH_MEM_EVENTS
    //BROKEN DO NOT USE
    for(int event=0; event < event_block_size*num_dimensions; event++) {
      cov_sum += (events[row*event_block_size+event]-means[row])*(events[col*event_block_size+event]-means[col])*clusters->memberships[c*event_block_size+event];
    }
#endif     
   
#ifndef SH_MEM_EVENTS
    for(int event=e_start; event < e_end; event++) {
      cov_sum += (fcs_data[event*num_dimensions+row]-means[row])*(fcs_data[event*num_dimensions+col]-means[col])*clusters->memberships[c*num_events+event];
    }
    /* for(int event=e_start; event < e_end; event++) { */
    /*   cov_sum += (fcs_data[row*num_events+event]-means[row])*(fcs_data[col*num_events+event]-means[col])*clusters->memberships[c*num_events+event]; */
    /* } */

    myR[matrix_index] = cov_sum;
    
#endif     
    //__syncthreads();
     
    float old = atomicAdd(&(temp_buffer[c*num_dimensions*num_dimensions+matrix_index]), myR[matrix_index]); 

    __syncthreads();

     
    //if(blockIdx.y == 0) {
      if(clusters->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?
        float cs = temp_buffer[c*num_dimensions*num_dimensions+matrix_index];
        cs /= clusters->N[c];
        clusters->R[c*num_dimensions*num_dimensions+matrix_index] = cs;
        // Set the symmetric value
        matrix_index = col*num_dimensions+row;
        clusters->R[c*num_dimensions*num_dimensions+matrix_index] = cs;
      } else {
        clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; // what should the variance be for an empty cluster...?
        // Set the symmetric value
        matrix_index = col*num_dimensions+row;
        clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; // what should the variance be for an empty cluster...?
      }
    
      // Regularize matrix - adds some variance to the diagonal elements
      // Helps keep covariance matrix non-singular (so it can be inverted)
      // The amount added is scaled down based on COVARIANCE_DYNAMIC_RANGE constant defined at top of this file
      if(row == col) {
        clusters->R[c*num_dimensions*num_dimensions+matrix_index] += clusters->avgvar[c];
      }
      //}
}
 

/*
 * Computes the covariance matrices of the data (R matrix)
 * Must be launched with a D*D/2 blocks: 
 */
__global__ void
mstep_covariance_3a(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events) {


  int tid = threadIdx.x; // easier variable name for our thread ID
  
  // Determine what row,col this matrix is handling, also handles the symmetric element
  int row,col,c;
  compute_row_col_block(num_dimensions, &row, &col);

  //__syncthreads();
      
  int matrix_index;
    
  // Store the means in shared memory to speed up the covariance computations
  __shared__ float means[NUM_CLUSTERS*NUM_DIMENSIONS];
  
  // copy the means for all clusters into shared memory

  for(int i = tid; i<num_clusters*num_dimensions; i+=NUM_THREADS_MSTEP) {
    means[i] = clusters->means[i];
  }
  
#ifdef SH_MEM_EVENTS
    __shared__ float events[NUM_EVENTS];
    for(int event=tid; event < num_events*num_dimensions; event+=blockDim.x) {
      if(event<num_events*num_dimensions) {
        events[event] = fcs_data[event];
      }
    }
#endif    
     
  // Sync to wait for all params to be loaded to shared memory
  __syncthreads();

  __shared__ float temp_sums[NUM_THREADS_MSTEP];
  __shared__ float cluster_sum[NUM_CLUSTERS]; //local storage for cluster results
  __shared__ float temp_debug[NUM_THREADS_MSTEP];
  __shared__ float cluster_debug[NUM_CLUSTERS];
  
  
  for(int c = 0; c<num_clusters; c++) {
    float cov_sum = 0.0f;
    //float temp = 0.0f;
    
    for(int event=tid; event < num_events; event+=NUM_THREADS_MSTEP) {
#ifdef SH_MEM_EVENTS
      cov_sum += (events[row*num_events+event]-means[c*num_dimensions+row])*(events[col*num_events+event]-means[c*num_dimensions+col])*clusters->memberships[c*num_events+event];
#else
      cov_sum += (fcs_data[row*num_events+event]-means[c*num_dimensions+row])*(fcs_data[col*num_events+event]-means[c*num_dimensions+col])*clusters->memberships[c*num_events+event];
#endif
      
    }

    temp_sums[tid] = cov_sum;
    
    __syncthreads();
      
    if(tid == 0) {
      cluster_sum[c] = 0.0f; 
      for(int i=0; i < NUM_THREADS_MSTEP; i++) {
        cluster_sum[c] += temp_sums[i];
      }
    }

  }
  __syncthreads();

    
  for(int c = tid; c<num_clusters; c+=NUM_THREADS_MSTEP) {
    matrix_index =  row * num_dimensions + col;
    if(clusters->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?
      cluster_sum[c] /= clusters->N[c];
        
      clusters->R[c*num_dimensions*num_dimensions+matrix_index] = cluster_sum[c];
        
      //Set the symmetric value
      matrix_index = col*num_dimensions+row;
      clusters->R[c*num_dimensions*num_dimensions+matrix_index] = cluster_sum[c];
    } else {
      clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; // what should the variance be for an empty cluster...?
      // Set the symmetric value
      matrix_index = col*num_dimensions+row;
      clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; // what should the variance be for an empty cluster...?
    }
      
      
    // Regularize matrix - adds some variance to the diagonal elements
    // Helps keep covariance matrix non-singular (so it can be inverted)
    // The amount added is scaled down based on COVARIANCE_DYNAMIC_RANGE constant defined at top of this file
    if(row == col) {
      clusters->R[c*num_dimensions*num_dimensions+matrix_index] += clusters->avgvar[c];
    }
  } 
}
/*
 * Computes the covariance matrices of the data (R matrix)
 * Must be launched with a D*D/2 blocks: 
 */
__global__ void
mstep_covariance_3b(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events, int event_block_size, int num_b, float *temp_buffer) {

  int tid = threadIdx.x; // easier variable name for our thread ID
  
  // Determine what row,col this matrix is handling, also handles the symmetric element
  int row,col,c;
  compute_row_col_block(num_dimensions, &row, &col);

  int e_start, e_end;
  compute_my_event_indices(num_events, event_block_size, num_b, &e_start, &e_end);

  //__syncthreads();
      
  int matrix_index;
  int num_dim_blocks = num_dimensions*((num_dimensions+1)/2);
    
  // Store the means in shared memory to speed up the covariance computations
  __shared__ float means[NUM_CLUSTERS*NUM_DIMENSIONS];
  
  // copy the means for all clusters into shared memory
  for(int i = tid; i<num_clusters*num_dimensions; i+=NUM_THREADS_MSTEP) {
    means[i] = clusters->means[i];
  }

  __shared__ float events[NUM_EVENTS];
  //could be optimized...
  for(int dim = tid; dim < num_dimensions; dim+=blockDim.x) {
    for(int event=e_start; event < e_end; event++) {
      events[dim*event_block_size+(event-e_start)] = fcs_data[dim*num_events+event];
    }
  }
  
  // Sync to wait for all params to be loaded to shared memory
  __syncthreads();

  __shared__ float temp_sums[NUM_THREADS_MSTEP];
  __shared__ float cluster_sum[NUM_CLUSTERS]; //local storage for cluster results
  
  for(int c = 0; c<num_clusters; c++) {
    float cov_sum = 0.0f;

    for(int event=tid+e_start; event < e_end; event+=NUM_THREADS_MSTEP) {
      cov_sum += (events[row*event_block_size+(event-e_start)]-means[c*num_dimensions+row])*(events[col*event_block_size+(event-e_start)]-means[c*num_dimensions+col])*clusters->memberships[c*num_events+event];
    }

    temp_sums[tid] = cov_sum;
        
    __syncthreads();
    
    if(tid == 0) {
      cluster_sum[c] = 0.0f; 
      for(int i=0; i < NUM_THREADS_MSTEP; i++) {
        cluster_sum[c] += temp_sums[i];
      }
    }

    __syncthreads();
    
    if(tid==0) {
      float old = atomicAdd(&(temp_buffer[c*num_dim_blocks+blockIdx.x]), cluster_sum[c]);
    }
  }

  __syncthreads();

  
  //if(blockIdx.y==0) {
  for(int c = tid; c<num_clusters; c+=NUM_THREADS_MSTEP) {
    matrix_index =  row * num_dimensions + col;
    if(clusters->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?
      float cs = temp_buffer[c];
      cs /= clusters->N[c];
      clusters->R[c*num_dimensions*num_dimensions+matrix_index] = cs;

      //Set the symmetric value
      matrix_index = col*num_dimensions+row;
      clusters->R[c*num_dimensions*num_dimensions+matrix_index] = cs;
    } else {
      clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; // what should the variance be for an empty cluster...?
      // Set the symmetric value
      matrix_index = col*num_dimensions+row;
      clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; // what should the variance be for an empty cluster...?
    }
      
      
    // Regularize matrix - adds some variance to the diagonal elements
    // Helps keep covariance matrix non-singular (so it can be inverted)
    // The amount added is scaled down based on COVARIANCE_DYNAMIC_RANGE constant defined at top of this file
    if(row == col) {
      clusters->R[c*num_dimensions*num_dimensions+matrix_index] += clusters->avgvar[c];
    }
    //}
  }
}

/*
 * Computes the covariance matrices of the data (R matrix)
 * Must be launched with a M x D*D grid of blocks: 
 *  i.e. dim3 gridDim(num_clusters,num_dimensions*num_dimensions)
 */
__global__ void
mstep_covariance_transpose(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events) {
  int tid = threadIdx.x; // easier variable name for our thread ID

    // Determine what row,col this matrix is handling, also handles the symmetric element
    int row,col,c;
    compute_row_col_transpose(num_dimensions, &row, &col);

    __syncthreads();
    
    c = blockIdx.y; // Determines what cluster this block is handling    

    int matrix_index = row * num_dimensions + col;

    #if DIAG_ONLY
    if(row != col) {
        clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
        matrix_index = col*num_dimensions+row;
        clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
        return;
    }
    #endif 

    // Store the means in shared memory to speed up the covariance computations
    __shared__ float means[NUM_DIMENSIONS];
    // copy the means for this cluster into shared memory
    if(tid < num_dimensions) {
        means[tid] = clusters->means[c*num_dimensions+tid];
    }

    // Sync to wait for all params to be loaded to shared memory
    __syncthreads();

    __shared__ float temp_sums[NUM_THREADS_MSTEP];
    
    float cov_sum = 0.0f;

    for(int event=tid; event < num_events; event+=NUM_THREADS_MSTEP) {
        cov_sum += (fcs_data[row*num_events+event]-means[row])*(fcs_data[col*num_events+event]-means[col])*clusters->memberships[c*num_events+event]; 
    }
    temp_sums[tid] = cov_sum;

    __syncthreads();
    
    if(tid == 0) {
        cov_sum = 0.0f;
        for(int i=0; i < NUM_THREADS_MSTEP; i++) {
            cov_sum += temp_sums[i];
        }
        if(clusters->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?
            cov_sum /= clusters->N[c];
            clusters->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
            // Set the symmetric value
            matrix_index = col*num_dimensions+row;
            clusters->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
        } else {
            clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; // what should the variance be for an empty cluster...?
            // Set the symmetric value
            matrix_index = col*num_dimensions+row;
            clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; // what should the variance be for an empty cluster...?
        }

        // Regularize matrix - adds some variance to the diagonal elements
        // Helps keep covariance matrix non-singular (so it can be inverted)
        // The amount added is scaled down based on COVARIANCE_DYNAMIC_RANGE constant defined at top of this file
        if(row == col) {
            clusters->R[c*num_dimensions*num_dimensions+matrix_index] += clusters->avgvar[c];
        }
    }
}


/*
 * Computes the constant for each cluster and normalizes pi for every cluster
 * In the process it inverts R and finds the determinant
 * 
 * Needs to be launched with the number of blocks = number of clusters
 */
__global__ void
constants_kernel(clusters_t* clusters, int num_clusters, int num_dimensions) {
    compute_constants(clusters,num_clusters,num_dimensions);
    
    __syncthreads();
    
    if(blockIdx.x == 0) {
        normalize_pi(clusters,num_clusters);
    }
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
