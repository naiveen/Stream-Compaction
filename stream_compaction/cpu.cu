#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            if (n > 0) {
                odata[0] = idata[0];
            }
            for (auto i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i];
            }
            
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int oIndex = 0;
            for (auto i = 0; i < n; i++) {
                if (idata[i]) {
                    odata[oIndex++] = idata[i];
                }
            }

            timer().endCpuTimer();
            return oIndex;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int streamNum=0;
            timer().startCpuTimer();
            // TODO
            int * scanArr = new int[n];
            int * inMap = new int[n];
            for (auto i = 0; i < n; i++) {
                inMap[i] = idata[i] ? 1 : 0;
            }
            if (n > 0) {
                scanArr[0] = idata[0];
            }
            for (auto i = 1; i < n; i++) {
                scanArr[i] = scanArr[i - 1] + idata[i];
            }
            if (n>0 && scanArr[0] == 1) {
                odata[streamNum++] = idata[0];
            }
            for (auto i = 1; i < n; i++) {
                if (scanArr[i] != scanArr[i - 1]) {
                    odata[streamNum++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return streamNum;
        }
    }
}
