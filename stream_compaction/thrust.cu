#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            thrust::device_vector<int> dv_in(idata, idata+n);
            dv_in.push_back(0);
            thrust::device_vector<int> dv_out(n+1);
            
            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            //thrust::exclusive_scan(dv_in, dv_in +n, dv_out);
            timer().endGpuTimer();
            thrust::copy(dv_out.begin()+1, dv_out.end(), odata);
        }
    }
}
