// https://wagonhelm.github.io/articles/2018-03/detecting-cuda-capability-with-cmake
// Justin Francis

#include <stdio.h>

int main(int argc, char **argv){
    cudaDeviceProp dP;
    float min_cc = 5.3;  // We need half floats...

    int rc = cudaGetDeviceProperties(&dP, 0);
    if(rc != cudaSuccess) {
        cudaError_t error = cudaGetLastError();
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return rc; /* Failure */
    }
    if((dP.major+(dP.minor/10)) < min_cc) {
        printf("Minimum Compute Capability of %2.1f required: %d.%d found. Not Building CUDA Code.\n",
               min_cc, dP.major, dP.minor);
        return 1; /* Failure */
    } else {
        printf("sm_%d%d", dP.major, dP.minor);
        return 0; /* Success */
    }
}
