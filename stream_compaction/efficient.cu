#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#define blockSize 512
namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernelUpstreamScan(int N, int d, int* dev_idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >=N) {
                return;
            }
            if (index % int(pow(2, d + 1)) == 0) {
                dev_idata[index + int(pow(2, d + 1)) - 1] += dev_idata[index + int(pow(2, d)) - 1];
            }

        }
        __global__ void kernelDownStreamScan(int N, int d, int* dev_idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >=N) {
                return;
            }
            if (index % int(pow(2, d + 1)) == 0) {
                    int temp = dev_idata[index + int(pow(2, d)) - 1];
                    dev_idata[index + int(pow(2, d)) - 1] = dev_idata[index + int(pow(2, d + 1)) - 1];
                    dev_idata[index + int(pow(2, d + 1)) - 1] += temp;
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            

            int* dev_idata;
            int arraySizeZeroPad = pow(2, ilog2ceil(n+1)) ;
            cudaMalloc((void**)&dev_idata, arraySizeZeroPad * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMemset(dev_idata , 0, arraySizeZeroPad* sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            // TODO

            
            for (auto d = 0; d <ilog2ceil(arraySizeZeroPad); d++) {
                //threadCount = (arraySizeZeroPad - 1) / pow(2, d + 1);
                dim3 threadBlocksPerGrid(ceil((arraySizeZeroPad + blockSize - 1) / blockSize));
                kernelUpstreamScan << <threadBlocksPerGrid, blockSize >> > (arraySizeZeroPad, d, dev_idata);
            }
            
            cudaMemset(dev_idata + arraySizeZeroPad - 1, 0, sizeof(int));
            for (auto d = ilog2ceil(arraySizeZeroPad)-1; d != -1; d--) {
                //threadCount = (arraySizeZeroPad - 1) / pow(2, d + 1);
                dim3 threadBlocksPerGrid((arraySizeZeroPad + blockSize - 1) / blockSize);
                kernelDownStreamScan << <threadBlocksPerGrid, blockSize >> > (arraySizeZeroPad, d, dev_idata);
            }
            
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata+1, (n) * sizeof(int), cudaMemcpyDeviceToHost);
            //cudaMemcpy(odata+n-1, dev_idata + n-1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
        }

        void scan_gpu(int n, int* dev_odata, const int* dev_bools) {
            int* dev_idata;
            int arraySizeZeroPad = pow(2, ilog2ceil(n + 1));
            cudaMalloc((void**)&dev_idata, arraySizeZeroPad * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMemset(dev_idata, 0, arraySizeZeroPad * sizeof(int));
            cudaMemcpy(dev_idata, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);

            for (auto d = 0; d < ilog2ceil(arraySizeZeroPad); d++) {
                //threadCount = (arraySizeZeroPad - 1) / pow(2, d + 1);
                dim3 threadBlocksPerGrid(ceil((arraySizeZeroPad + blockSize - 1) / blockSize));
                kernelUpstreamScan << <threadBlocksPerGrid, blockSize >> > (arraySizeZeroPad, d, dev_idata);
            }

            cudaMemset(dev_idata + arraySizeZeroPad - 1, 0, sizeof(int));
            for (auto d = ilog2ceil(arraySizeZeroPad) - 1; d != -1; d--) {
                //threadCount = (arraySizeZeroPad - 1) / pow(2, d + 1);
                dim3 threadBlocksPerGrid((arraySizeZeroPad + blockSize - 1) / blockSize);
                kernelDownStreamScan << <threadBlocksPerGrid, blockSize >> > (arraySizeZeroPad, d, dev_idata);
            }
            cudaMemcpy(dev_odata, dev_idata + 1, (n) * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaFree(dev_idata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            

            int* dev_bools;
            int* dev_idata;
            int* dev_odata;
            int* dev_indices;
            int* dev_scan;

            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");
            
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            cudaMalloc((void**)&dev_indices, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");

            
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            // TODO
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);
            
            scan_gpu(n,dev_indices, dev_bools);
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n,dev_odata, dev_idata, dev_bools, dev_indices);
            timer().endGpuTimer();
            int num=0;
            cudaMemcpy(&num, dev_indices + n-1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_indices);
            cudaFree(dev_bools);
            cudaFree(dev_odata);
            return num ;
        }
    }
}
