/* Cilk Plus Declarations here
 */
__declspec(noinline) void estep1_events(float* data, clusters_t* clusters, int D, int n, int N, int m, float* means, float* Rinv);
__declspec(noinline) float estep2_events(clusters_t* clusters, int M, int n, int N);
__declspec(noinline) void mstep_mean_dimension(float* data, clusters_t* clusters, int d, int D, int m, int N);
__declspec(noinline) void mstep_covar_cluster(float* data, clusters_t* clusters, int D, int m, int N, float* means);

/* Non Cilk Plus Declarations here
 */
void seed_clusters(float *data, clusters_t* clusters, int D, int M, int N);
void constants(clusters_t* clusters, int M, int D);
void estep1(float* data, clusters_t* clusters, int D, int M, int N, float* likelihood);
void estep2(float* data, clusters_t* clusters, int D, int M, int N, float* likelihood);
void mstep_n(float* data, clusters_t* clusters, int D, int M, int N);
void mstep_mean(float* data, clusters_t* clusters, int D, int M, int N);
void mstep_covar(float* data, clusters_t* clusters, int D, int M, int N);