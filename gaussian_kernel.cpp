/*
 * gaussian_kernel.cpp
 *
 *  Created on: March 24, 2011
 *      Author: Doug Roberts
 *      Modified by: Andrew Pangborn, Young Kim (implemented Cilk Plus)
 */

//Including standard libraries
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <minmax.h>

// Include project files
#include "gaussian.h"
#include "invert_matrix.h"
#include "gaussian_kernel.h"

// Include Cilk files
#include <cilk\cilk.h>

/* Cilk Plus Status:
 * seed_clusters
 * Does not need parallelization
 */
void seed_clusters(float *data, clusters_t* clusters, int D, int M, int N) {
    float* variances = (float*) malloc(sizeof(float)*D);
    float* means = (float*) malloc(sizeof(float)*D);

    // Compute means
    for(int d=0; d < D; d++) {
        means[d] = 0.0;
        for(int n=0; n < N; n++) {
            means[d] += data[n*D+d];
        }
        means[d] /= (float) N;
    }

    // Compute variance of each dimension
    for(int d=0; d < D; d++) {
        variances[d] = 0.0;
        for(int n=0; n < N; n++) {
            variances[d] += data[n*D+d]*data[n*D+d];
        }
        variances[d] /= (float) N;
        variances[d] -= means[d]*means[d];
    }

    // Average variance
    float avgvar = 0.0;
    for(int d=0; d < D; d++) {
        avgvar += variances[d];
    }
    avgvar /= (float) D;

    // Initialization for random seeding and uniform seeding    
    float fraction;
    int seed;
    if(M > 1) {
        fraction = (N-1.0f)/(M-1.0f);
    } else {
        fraction = 0.0;
    }

	// Cilk Plus: Needed to be set below to make the program run.
	seed = 0;
    srand(seed);
    srand(clock());

    for(int m=0; m < M; m++) {
        clusters->N[m] = (float) N / (float) M;
        clusters->pi[m] = 1.0f / (float) M;
        clusters->avgvar[m] = avgvar / COVARIANCE_DYNAMIC_RANGE;

        DEBUG("N: %.2f\tPi: %.2f\tAvgvar: %e\n",clusters->N[m],clusters->pi[m],clusters->avgvar[m]);

        // Choose cluster centers
        DEBUG("Means: ");
        #if UNIforM_SEED
            for(int d=0; d < D; d++) {
                clusters->means[m*D+d] = data[((int)(m*fraction))*D+d];
                DEBUG("%.2f ",clusters->means[m*D+d]);
            }
        #else
            seed = rand() % N;
            DEBUG("Cluster %d seed = event #%d\n",m,seed);
            for(int d=0; d < D; d++) {
                clusters->means[m*D+d] = data[seed*D+d];
                DEBUG("%.2f ",clusters->means[m*D+d]);
            }
        #endif
        DEBUG("\n");

        // Set covariances to identity matrices
        for(int i=0; i < D; i++) {
            for(int j=0; j < D; j++) {
                if(i == j) {
                    clusters->R[m*D*D+i*D+j] = 1.0f;
                } else {
                    clusters->R[m*D*D+i*D+j] = 0.0f;
                }
            }
        }
        
        DEBUG("R:\n");
        for(int d=0; d < D; d++) {
            for(int e=0; e < D; e++) 
                DEBUG("%.2f ",clusters->R[m*D*D+d*D+e]);
            DEBUG("\n");
        }
        DEBUG("\n");

    }

    free(variances);
    free(means);
}

/* Cilk Plus Status:
 * constants
 * No need to parallelize; Dependent on invert_cpu which can possibly be parallelized
 */
void constants(clusters_t* clusters, int M, int D) {
	float log_determinant;
    float* matrix = (float*) malloc(sizeof(float)*D*D);

    float sum = 0.0;
    for(int m=0; m < M; m++) {
        // Invert covariance matrix
        memcpy(matrix,&(clusters->R[m*D*D]),sizeof(float)*D*D);
        invert_cpu(matrix,D,&log_determinant);
        memcpy(&(clusters->Rinv[m*D*D]),matrix,sizeof(float)*D*D);
    
        // Compute constant
        clusters->constant[m] = -D*0.5f*logf(2.0f*PI) - 0.5f*log_determinant;
        DEBUG("Cluster %d constant: %e\n",m,clusters->constant[m]);

        // Sum for calculating pi values
        sum += clusters->N[m];
    }

    // Compute pi values
    for(int m=0; m < M; m++) {
        clusters->pi[m] = clusters->N[m] / sum;
    }
    
    free(matrix);
}

__declspec(noinline) void estep1_events(float* data, clusters_t* clusters, int D, int n, int N, int m, float* means, float* Rinv) {
	float like = 0.0;
	#if DIAG_ONLY
		for(int i=0; i < D; i++) {
			like += (data[i*N+n]-means[i])*(data[i*N+n]-means[i])*Rinv[i*D+i];
		}
	#else
		for(int i=0; i < D; i++) {
			for(int j=0; j < D; j++) {
				like += (data[i*N+n]-means[i])*(data[j*N+n]-means[j])*Rinv[i*D+j];
			}
		}
	#endif  
	clusters->memberships[m*N+n] = -0.5f * like + clusters->constant[m] + log(clusters->pi[m]); 
}

/* Cilk Plus Status:
 * estep1
 * Two Possible Parallelizations:
 * 1) Parallelize over M (i.e. each cluster) 
 * 2) Parallelize over N (i.e. each event) <-- this option was taken
 */
void estep1(float* data, clusters_t* clusters, int D, int M, int N, float* likelihood) {
    clock_t start,finish;
    // Compute likelihood for every data point in each cluster
    float* means;
    float* Rinv;
    start = clock();
    for(int m=0; m < M; m++) {
        means = (float*) &(clusters->means[m*D]);
        Rinv = (float*) &(clusters->Rinv[m*D*D]);
        
        cilk_for(int n=0; n < N; n++) {
			estep1_events(data, clusters, D, n, N, m, means, Rinv);
        }
    }
    finish = clock();
    DEBUG("estep1: %f seconds.\n",(double)(finish-start)/(double)CLOCKS_PER_SEC);
}

__declspec(noinline) float estep2_events(clusters_t* clusters, int M, int n, int N) {
	// Finding maximum likelihood for this data point
	float max_likelihood;
	max_likelihood = __sec_reduce_max(clusters->memberships[n:M:N]);

	// Computes sum of all likelihoods for this event
	float denominator_sum;
	denominator_sum = 0.0f;
	for(int m=0; m < M; m++) {
		denominator_sum += exp(clusters->memberships[m*N+n] - max_likelihood);
	}
	denominator_sum = max_likelihood + log(denominator_sum);

	// Divide by denominator to get each membership
	for(int m=0; m < M; m++) {
		clusters->memberships[m*N+n] = exp(clusters->memberships[m*N+n] - denominator_sum);
	}

	return denominator_sum;
}

/* Cilk Plus Status:
 * estep2
 * Two Possible Parallelizations:
 * 1) Parallelize over M (i.e. each cluster) 
 * 2) Parallelize over N (i.e. each event) <-- this option was taken
 *
 * After looking at the run-times, it seems like estep1 takes the most amount of time... Parallelizing this may not be worth it
 */
void estep2(float* data, clusters_t* clusters, int D, int M, int N, float* likelihood) {
    clock_t start,finish;
    start = clock();
    float max_likelihood, denominator_sum;
    *likelihood = 0.0f;
    cilk_for(int n=0; n < N; n++) {
		*likelihood += estep2_events(clusters, M, n, N);
    }
    finish = clock();
    DEBUG("estep2: %f seconds.\n",(double)(finish-start)/(double)CLOCKS_PER_SEC);
}

/* Cilk Plus Status:
 * mstep_n
 * Parallelization via reductions result in slower speeds
 */
void mstep_n(float* data, clusters_t* clusters, int D, int M, int N) {
    DEBUG("mstep_n: D: %d, M: %d, N: %d\n",D,M,N);
    for(int m=0; m < M; m++) {
        clusters->N[m] = 0.0;
        // compute effective size of each cluster by adding up soft membership values
        for(int n=0; n < N; n++) {
            clusters->N[m] += clusters->memberships[m*N+n];
        }
    }
}

__declspec(noinline) void mstep_mean_dimension(float* data, clusters_t* clusters, int d, int D, int m, int N) {
	clusters->means[m*D+d] = 0.0;
	for(int n=0; n < N; n++) {
		clusters->means[m*D+d] += data[d*N+n]*clusters->memberships[m*N+n];
	}
	clusters->means[m*D+d] /= clusters->N[m];
	DEBUG("%f ",clusters->means[m*D+d]);
}

/* Cilk Plus Status:
 * mstep_mean
 * Two Possible Parallelizations:
 * 1) Parallelize over M (i.e. each cluster) 
 * 2) Parallelize over D (i.e. each dimension) <-- this option was taken
 */

void mstep_mean(float* data, clusters_t* clusters, int D, int M, int N) {
    DEBUG("mstep_mean: D: %d, M: %d, N: %d\n",D,M,N);
    for(int m=0; m < M; m++) {
        DEBUG("Cluster %d: ",m);
        cilk_for(int d=0; d < D; d++) {
			mstep_mean_dimension(data, clusters, d, D, m, N);
        }
        DEBUG("\n");
    }
}

__declspec(noinline) void mstep_covar_cluster(float* data, clusters_t* clusters, int D, int m, int N, float* means) {
	float sum;
	for(int i=0; i < D; i++) {
		for(int j=0; j <= i; j++) {
			#if DIAG_ONLY
				if(i != j) {
					clusters->R[m*D*D+i*D+j] = 0.0f;
					clusters->R[m*D*D+j*D+i] = 0.0f;
					continue;
				}
			#endif
			sum = 0.0;
			for(int n=0; n < N; n++) {
				sum += (data[i*N+n]-means[i])*(data[j*N+n]-means[j])*clusters->memberships[m*N+n];
			}
			if(clusters->N[m] >= 1.0f) {
				clusters->R[m*D*D+i*D+j] = sum / clusters->N[m];
				clusters->R[m*D*D+j*D+i] = sum / clusters->N[m];
			} else {
				clusters->R[m*D*D+i*D+j] = 0.0f;
				clusters->R[m*D*D+j*D+i] = 0.0f;
			}
		}
	}
}

/* Cilk Plus Status:
 * mstep_covar
 * Two Possible Parallelizations:
 * 1) Parallelize over M (i.e. each cluster) <-- this option was taken
 * 2) Parallelize over D (i.e. each dimension) 
 */

void mstep_covar(float* data, clusters_t* clusters, int D, int M, int N) {
    DEBUG("mstep_covar: D: %d, M: %d, N: %d\n",D,M,N);
    float* means;
    cilk_for(int m=0; m < M; m++) {
        means = &(clusters->means[m*D]);
        mstep_covar_cluster(data, clusters, D, m, N, means);
    }
}
