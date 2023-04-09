#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#define blockSize 512
namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        __global__ void kernelPreScan(int N,int d, int* dev_odata, int* dev_idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= N ) {
                return;
            }
            if (index >= pow(2, d-1)) {
                dev_odata[index] = dev_idata[index] + dev_idata[index - int(pow(2, d - 1))];
                
            }
            else {
                dev_odata[index] = dev_idata[index];
            }
            
        }
        void scan(int n, int *odata, const int *idata) {
            int* dev_idata;
            int* dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            // TODO
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            for (auto d = 1; d <= ilog2ceil(n); d++) {
                kernelPreScan <<<fullBlocksPerGrid, blockSize >> > (n, d, dev_odata, dev_idata);
                cudaMemcpy(dev_idata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
}
