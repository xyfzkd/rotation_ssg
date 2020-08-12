#ifndef CUDA_MOTIONCOR_H_
#define CUDA_MOTIONCOR_H_

#include "src/mpi.h"
#include "src/motioncorr_runner.h"
#include "src/motioncorr_runner_mpi.h"
#include "src/projector.h"
#include "src/complex.h"
#include "src/image.h"

#include "src/acc/cuda/cuda_mem_utils.h"
#include "src/acc/acc_projector.h"
#include "src/acc/cuda/cuda_settings.h"
#include "src/acc/cuda/cuda_fft.h"
#include "src/acc/cuda/cuda_benchmark_utils.h"

#include <stack>

#ifdef ACC_DOUBLE_PRECISION
#define XFLOAT double
#else
#define XFLOAT float
#endif

class MotionCorCuda
{
private:
    MpiNode *node;
public:
    MotioncorrRunner *baseRnr;
    CudaCustomAllocator *allocator;

    int device_id;

#ifdef TIMING_FILES
    relion_timer timer;
#endif
    MotionCorCuda(MotioncorrRunner    *baseRnr, int dev_id, const char * timing_fnm);
    MotionCorCuda(MotioncorrRunnerMpi *baseRnr, int dev_id, const char * timing_fnm);

    void run();

    bool alignPatch(std::vector<MultidimArray<fComplex> > &Fframes,
                    const int pnx, const int pny,
                    const RFLOAT scaled_B,
                    std::vector<RFLOAT> &xshifts,
                    std::vector<RFLOAT> &yshifts,
                    std::ostream &logfile);

    ~MotionCorCuda()
    {
        for (int i = 0; i < classStreams.size(); ++i) {
            HANDLE_ERROR(cudaStreamDestroy(classStreams[i]));
        }
    }
};
