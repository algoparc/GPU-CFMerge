#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>

#include <thrust/merge.h>

//#define VERIFY

struct cmp_func {
    __host__ __device__
    bool operator()(int x, int y) {
        return x < y;
    }
};

int main(int argc, char **argv) {
	if (argc != 3) {
		fprintf(stderr, "Usage: %s <total number of warps (positive power of 2 required)> <RNG seed value>\n", argv[0]);
        exit(1);
	}

	int num_warps = atoi(argv[1]);
    if (num_warps <= 0) {
        fprintf(stderr, "ERROR: total number of warps must be positive!\n");
        exit(1);
    }
    //TODO: check for power of 2

    int n = num_warps * 32 * MY_E;		//w = 32
	
    //1. Generate random input
    int *input = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i) {
        input[i] = i;
    }

    std::mt19937 gen(atoi(argv[2]));
    std::uniform_int_distribution<int> dist(0, n-1);

    for (int i = 0; i < n; ++i) {
        int j = dist(gen);

        int temp = input[i];
        input[i] = input[j];
        input[j] = temp;
    }


    //2. Initialize GPU input/output arrays
    cudaError_t cudaerr;
    int *d_input;
    //int *d_output;

    cudaerr = cudaMalloc(&d_input,  n * sizeof(int));
    if (cudaerr != cudaSuccess) printf("cudaMalloc(d_input) failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));

    //cudaerr = cudaMalloc(&d_output, n * sizeof(int));
    //if (cudaerr != cudaSuccess) printf("cudaMalloc(d_output) failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));

    cudaerr = cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaerr != cudaSuccess) printf("cudaMemcpy(d_input, input) failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));

    free(input);


    //3. Run merge
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    thrust::sort(thrust::device, d_input, d_input + n, cmp_func());
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%f\n", ms);


    //4. (Optional) Verify output
    #ifdef VERIFY
    int *output = (int *)malloc(n * sizeof(int));
    cudaerr = cudaMemcpy(output, d_input, n * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaerr != cudaSuccess) printf("cudaMemcpy(output, d_input) failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));

    int count = 0;
    for (int i = 1; i < n; ++i) {
        if (output[i] < output[i-1]) ++count;
    }
    if (count > 0) printf("\n%d errors\n", count);
    else printf("\nsorted!\n");

    free(output);
    #endif

    //5. Clean up
    cudaFree(d_input);
    //cudaFree(d_output);
    
    return 0;
}

