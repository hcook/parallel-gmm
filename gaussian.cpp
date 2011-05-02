/*
 * Gaussian Mixture Model Clustering wtih CUDA
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

// includes, project

// loads common factors for gaussian
#include "gaussian.h"
// loads headers for inverting matrix
#include "invert_matrix.h"
// loads headers for gaussian_kernel
#include "gaussian_kernel.h"

// includes, Cilk Plus
#include <cilk\cilk.h>

// Function prototypes
extern "C" float* readData(char* f, int* ndims, int*nevents);
int validateArguments(int argc, char** argv, int* num_clusters);
void writeCluster(FILE* f, clusters_t clusters, int c,  int num_dimensions);
void printCluster(clusters_t clusters, int c, int num_dimensions);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) {
    int num_clusters;
    
    // For profiling 
    clock_t seed_start, seed_end, seed_total = 0;
    clock_t regroup_start, regroup_end, regroup_total = 0;
    int regroup_iterations = 0;
    clock_t params_start, params_end, params_total = 0;
    int params_iterations = 0;
    clock_t constants_start, constants_end, constants_total = 0;
    int constants_iterations = 0;
    clock_t total_timer = cilk_getticks();
    double total_time = 0;
    clock_t io_timer;
    double io_time = 0;
    clock_t cpu_timer;
    double cpu_time = 0;

    io_timer = cilk_getticks();
    // Validate the command-line arguments, parse # of clusters, etc 
    if(validateArguments(argc,argv,&num_clusters)) {
        return 1; //Bard args
    }
    
    int num_dimensions;
    int num_events;
    
    // Read FCS data   
    PRINT("Parsing input file...");
    // This stores the data in a 1-D array with consecutive values being the dimensions from a single event
    // (num_events by num_dimensions matrix)
    float* fcs_data_by_event = readData(argv[2],&num_dimensions,&num_events);    

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

    io_time += (double)(cilk_getticks() - io_timer);
   
    PRINT("Number of events: %d\n",num_events);
    PRINT("Number of dimensions: %d\n",num_dimensions);
    PRINT("Number of target clusters: %d\n\n",num_clusters);
   
    cpu_timer = cilk_getticks();

    // Setup the cluster data structures on host
    clusters_t clusters;
    clusters.N = (float*) malloc(sizeof(float)*num_clusters);
    clusters.pi = (float*) malloc(sizeof(float)*num_clusters);
    clusters.constant = (float*) malloc(sizeof(float)*num_clusters);
    clusters.avgvar = (float*) malloc(sizeof(float)*num_clusters);
    clusters.means = (float*) malloc(sizeof(float)*num_dimensions*num_clusters);
    clusters.R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*num_clusters);
    clusters.Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*num_clusters);
    clusters.memberships = (float*) malloc(sizeof(float)*num_events*num_clusters);
    if(!clusters.means || !clusters.R || !clusters.Rinv || !clusters.memberships) { 
        printf("ERROR: Could not allocate memory for clusters.\n"); 
        return 1; 
    }
    DEBUG("Finished allocating memory on host for clusters.\n");
    
    float rissanen;
    
    //////////////// Initialization done, starting kernels //////////////// 
    DEBUG("Invoking seed_clusters kernel.\n");
    fflush(stdout);

    // seed_clusters sets initial pi values, 
    // finds the means / covariances and copies it to all the clusters
    // TODO: Does it make any sense to use multiple blocks for this?
    seed_start = cilk_getticks();
    seed_clusters(fcs_data_by_event, &clusters, num_dimensions, num_clusters, num_events);
   
    DEBUG("Invoking constants kernel.\n");
    // Computes the R matrix inverses, and the gaussian constant
    //constants_kernel<<<num_clusters, num_threads>>>(d_clusters,num_clusters,num_dimensions);
    constants(&clusters,num_clusters,num_dimensions);
    constants_iterations++;
    seed_end = cilk_getticks();
    seed_total = seed_end - seed_start;

    // Calculate an epsilon value
    //int ndata_points = num_events*num_dimensions;
    float epsilon = (1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions)*log((float)num_events*num_dimensions)*0.01;
    float likelihood, old_likelihood;
    int iters;
    
    epsilon = 1e-6;

    PRINT("Gaussian.cu: epsilon = %f\n",epsilon);

    /*************** EM ALGORITHM *****************************/
    
    // do initial regrouping
    // Regrouping means calculate a cluster membership probability
    // for each event and each cluster. Each event is independent,
    // so the events are distributed to different blocks 
    // (and hence different multiprocessors)
    DEBUG("Invoking regroup (E-step) kernel with %d blocks.\n",NUM_BLOCKS);
    regroup_start = cilk_getticks();
    estep1(fcs_data_by_event,&clusters,num_dimensions,num_clusters,num_events,&likelihood);
    estep2(fcs_data_by_dimension,&clusters,num_dimensions,num_clusters,num_events,&likelihood);
    //estep2b(fcs_data_by_dimension,&clusters,num_dimensions,num_clusters,num_events,&likelihood);
    regroup_end = cilk_getticks();
    regroup_total += regroup_end - regroup_start;
    regroup_iterations++;
    DEBUG("Regroup Kernel Iteration Time: %f\n\n",((double)(regroup_end-regroup_start)));

    DEBUG("Likelihood: %e\n",likelihood);

    float change = epsilon*2;
    
    PRINT("Performing EM algorithm on %d clusters.\n",num_clusters);
    iters = 0;
    // This is the iterative loop for the EM algorithm.
    // It re-estimates parameters, re-computes constants, and then regroups the events
    // These steps keep repeating until the change in likelihood is less than some epsilon        
    while(iters < MIN_ITERS || (fabs(change) > epsilon && iters < MAX_ITERS)) {
        old_likelihood = likelihood;
        
        DEBUG("Invoking reestimate_parameters (M-step) kernel.\n");
        params_start = cilk_getticks();
        // This kernel computes a new N, pi isn't updated until compute_constants though
        mstep_n(fcs_data_by_dimension,&clusters,num_dimensions,num_clusters,num_events);
        mstep_mean(fcs_data_by_dimension,&clusters,num_dimensions,num_clusters,num_events);
        mstep_covar(fcs_data_by_dimension,&clusters,num_dimensions,num_clusters,num_events);
        params_end = cilk_getticks();
        params_total += params_end - params_start;
        params_iterations++;
        DEBUG("Model M-Step Iteration Time: %f\n\n",((double)(params_end-params_start)));
        //return 0; // RETURN FOR FASTER PROFILING
        
        DEBUG("Invoking constants kernel.\n");
        // Inverts the R matrices, computes the constant, normalizes cluster probabilities
        constants_start = cilk_getticks();
        constants(&clusters,num_clusters,num_dimensions);
        constants_end = cilk_getticks();
        constants_total += constants_end - constants_start;
        constants_iterations++;
        DEBUG("Constants Kernel Iteration Time: %f\n\n",((double)(constants_end-constants_start)));

        regroup_start = cilk_getticks();
        // Compute new cluster membership probabilities for all the events
        estep1(fcs_data_by_event,&clusters,num_dimensions,num_clusters,num_events,&likelihood);
        estep2(fcs_data_by_dimension,&clusters,num_dimensions,num_clusters,num_events,&likelihood);
        //estep2b(fcs_data_by_dimension,&clusters,num_dimensions,num_clusters,num_events,&likelihood);
        regroup_end = cilk_getticks();
        regroup_total += regroup_end - regroup_start;
        regroup_iterations++;
        DEBUG("E-step Iteration Time: %f\n\n",((double)(regroup_end-regroup_start)));
    
        change = likelihood - old_likelihood;
        DEBUG("likelihood = %f\n",likelihood);
        DEBUG("Change in likelihood: %f\n",change);

        iters++;

    }
    
    // Calculate Rissanen Score
    rissanen = -likelihood + 0.5*(num_clusters*(1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions)-1)*logf((float)num_events*num_dimensions);
    PRINT("\nFinal rissanen Score was: %f, with %d clusters.\n",rissanen,num_clusters);
    
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
    cpu_time += (double)(cilk_getticks() - cpu_timer);
    
    io_timer = cilk_getticks();
    // Open up the output file for cluster summary
    FILE* outf = fopen(summary_filename,"w");
    if(!outf) {
        printf("ERROR: Unable to open file '%s' for writing.\n",argv[3]);
    }

    // Print the clusters with the lowest rissanen score to the console and output file
    for(int c=0; c<num_clusters; c++) {
        //if(saved_clusters.N[c] == 0.0) {
        //    continue;
        //}
        if(ENABLE_PRINT) {
            // Output the final cluster stats to the console
            PRINT("Cluster #%d\n",c);
            printCluster(clusters,c,num_dimensions);
            PRINT("\n\n");
        }

        if(ENABLE_OUTPUT) {
            // Output the final cluster stats to the output file        
            fprintf(outf,"Cluster #%d\n",c);
            writeCluster(outf,clusters,c,num_dimensions);
            fprintf(outf,"\n\n");
        }
    }
    
    // Print profiling information
    printf("Program Component\tTotal\tIters\tTime Per Iteration\n");
    printf("        Seed Kernel:\t%7.4f\t%d\t%7.4f\n",(double)seed_total/1000.0,1, (double) seed_total/1000.0 );
    printf("      E-step Kernel:\t%7.4f\t%d\t%7.4f\n",(double)regroup_total/1000.0,regroup_iterations, (double) regroup_total/1000.0 / (double) regroup_iterations);
    printf("      M-step Kernel:\t%7.4f\t%d\t%7.4f\n",(double)params_total/1000.0,params_iterations, (double) params_total/1000.0 / (double) params_iterations);
    printf("   Constants Kernel:\t%7.4f\t%d\t%7.4f\n",(double)constants_total/1000.0,constants_iterations, (double) constants_total/1000.0 / (double) constants_iterations);    
   
    // Write profiling info to summary file
    fprintf(outf,"Program Component\tTotal\tIters\tTime Per Iteration\n");
    fprintf(outf,"        Seed Kernel:\t%7.4f\t%d\t%7.4f\n",(double)seed_total/1000.0,1, (double) seed_total/1000.0);
    fprintf(outf,"      E-step Kernel:\t%7.4f\t%d\t%7.4f\n",(double)regroup_total/1000.0,regroup_iterations, (double) regroup_total/1000.0 / (double) regroup_iterations);
    fprintf(outf,"      M-step Kernel:\t%7.4f\t%d\t%7.4f\n",(double)params_total/1000.0,params_iterations, (double) params_total/1000.0 / (double) params_iterations);
    fprintf(outf,"   Constants Kernel:\t%7.4f\t%d\t%7.4f\n",(double)constants_total/1000.0,constants_iterations, (double) constants_total/1000.0 / (double) constants_iterations);    
    fclose(outf);
    
    
    // Open another output file for the event level clustering results
    FILE* fresults = fopen(result_filename,"w");
   
    if(ENABLE_OUTPUT) { 
        for(int i=0; i<num_events; i++) {
            for(int d=0; d<num_dimensions-1; d++) {
                fprintf(fresults,"%f,",fcs_data_by_event[i*num_dimensions+d]);
            }
            fprintf(fresults,"%f",fcs_data_by_event[i*num_dimensions+num_dimensions-1]);
            fprintf(fresults,"\t");
            for(int c=0; c<num_clusters-1; c++) {
                fprintf(fresults,"%f,",clusters.memberships[c*num_events+i]);
            }
            fprintf(fresults,"%f",clusters.memberships[(num_clusters-1)*num_events+i]);
            fprintf(fresults,"\n");
        }
    }
    fclose(fresults); 
    io_time += (double)(cilk_getticks() - io_timer);
    printf("\n");
    printf( "I/O time: %f (ms)\n", io_time/1000.0);
    printf( "CPU processing time: %f (ms)\n", cpu_time/1000.0);
    total_time += (double)(cilk_getticks() - total_timer);
    printf( "Total time: %f (ms)\n", total_time/1000.0);
 
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

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Validate command line arguments
///////////////////////////////////////////////////////////////////////////////
int validateArguments(int argc, char** argv, int* num_clusters) {
    if(argc == 4) {
        // parse num_clusters
        if(!sscanf(argv[1],"%d",num_clusters)) {
            printf("Invalid number of starting clusters\n\n");
            printUsage(argv);
            return 1;
        } 
        
        // Check bounds for num_clusters
        if(*num_clusters < 1 || *num_clusters > MAX_CLUSTERS) {
            printf("Invalid number of starting clusters (max %d)\n\n", MAX_CLUSTERS);
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
        
        // Clean up so the EPA is happy
        fclose(infile);
        //fclose(outfile);
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
   printf("Usage: %s num_clusters infile outfile\n",argv[0]);
   printf("\t num_clusters: The number of starting clusters\n");
   printf("\t infile: ASCII space-delimited FCS data file\n");
   printf("\t outfile: Clustering results output file\n");
}

void writeCluster(FILE* f, clusters_t clusters, int c, int num_dimensions) {
    fprintf(f,"Probability: %f\n", clusters.pi[c]);
    fprintf(f,"N: %f\n",clusters.N[c]);
    fprintf(f,"Means: ");
    for(int i=0; i<num_dimensions; i++){
        fprintf(f,"%f ",clusters.means[c*num_dimensions+i]);
    }
    fprintf(f,"\n");

    fprintf(f,"\nR Matrix:\n");
    for(int i=0; i<num_dimensions; i++) {
        for(int j=0; j<num_dimensions; j++) {
            fprintf(f,"%f ", clusters.R[c*num_dimensions*num_dimensions+i*num_dimensions+j]);
        }
        fprintf(f,"\n");
    }
    fflush(f);   
}

void printCluster(clusters_t clusters, int c, int num_dimensions) {
    writeCluster(stdout,clusters,c,num_dimensions);
}

