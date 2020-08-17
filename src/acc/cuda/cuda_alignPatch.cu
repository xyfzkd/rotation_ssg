#include <cufftw.h>
#include <cufft.h>
#include "cuda_runtime.h"
//#include "src/acc/cuda/cuda_alignPatch.h"
#include "src/acc/acc_alignPatch.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "device_launch_parameters.h"
//#include "time.h"
#include "src/multidim_array.h"
#include "src/time.h"

#define PRINTCOMP

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %dn", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/**********************************************************************/
/* function for simulating data for iFFT
 * input of type MultidimArray<fComplex> *, simulate random
 * data, and this function should be integrated into class MultidimArray
 * there is macro RELION_ALIGNED_MALLOC, with data simulator initRandom or others.
 * However, it seems as if it's designed for common type.
 * I decide to fix the omission.
 *
 *         T* ptr=NULL;
 *         long int n;
 *         if (mode == "uniform")
 *             FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY_ptr(*this,n,ptr)
 *             *ptr = static_cast< T >(rnd_unif(op1, op2));
 *
 *  what if T is fComplex?
 */
/***********************************************************************/

void rand_comp(MultidimArray<fComplex>& s){
    fComplex* ptr=NULL;
    long int n;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY_ptr(s,n,ptr){
        (*ptr).real = (float) (rnd_unif(0, 1));
        (*ptr).imag = (float) (rnd_unif(0, 1));
    }
//#ifdef PRINTCOMP
//    print_comp_image(s);
//#endif
}

void print_comp_image(MultidimArray<fComplex>& s){
    printf("this is comp image, stored in array(%d, %d, %d, %d)\n", NSIZE(s), ZSIZE(s), YSIZE(s), XSIZE(s));
    for (int i=0; i < 16; i++){
        printf("%3.1f %3.1f \n", s.data[i].real,  s.data[i].imag);
    }
}

void print_real_image(MultidimArray<float>& s){
    printf("this is real image, stored in array(%d, %d, %d, %d)\n", NSIZE(s), ZSIZE(s), YSIZE(s), XSIZE(s));
    for (int i=0; i < 16; i++){
        printf("%3.1f \n", s.data[i]);
    }
}

/**********************************************************************/
/* function for testing the resulting differences
 *          sum( abs(re1-re2) / (abs(re1)+abs(re2)) )
 *
 * macros for traversing the same size MultidimArray, similar to src/multidim_array.h:234
 */
/***********************************************************************/

#define DIFF_ptr(re1,re2,n,ptr1,ptr2) \
    for ((n)=0, (ptr1)=(re1).data, (ptr2)=(re2).data; (n)<NZYXSIZE(re1); ++(n), ++(ptr1), ++(ptr2))


float diff(MultidimArray<float>& re1, MultidimArray<float>& re2){
    if(NSIZE(re1)!=NSIZE(re2) || \
       ZSIZE(re1)!=ZSIZE(re2) || \
       YSIZE(re1)!=YSIZE(re2) || \
       XSIZE(re1)!=XSIZE(re2)){
        printf("Unequal dimensions:\n  Array1: (%d, %d, %d, %d)\n  Array1: (%d, %d, %d, %d)\n",
                NSIZE(re1), ZSIZE(re1), YSIZE(re1), XSIZE(re1),
                NSIZE(re2), ZSIZE(re2), YSIZE(re2), XSIZE(re2));
        return 0;
    }
    float *ptr1=NULL, *ptr2=NULL;
    long int n;
    float diff = 0, eps=1e-8;
    DIFF_ptr(re1,re2,n,ptr1,ptr2){
        diff += abs(*ptr1 - *ptr2) / (abs(*ptr1) + abs(*ptr2));
#ifdef PRINTDIFF
        printf("N: %d: difference is %f\n", n, diff);
#endif
    }
    printf("Difference is %f\n", diff);

#ifdef PRINTCOMP
    printf("CPU: this is real image, stored in array(%d, %d, %d, %d)\n",
            NSIZE(re1),
            ZSIZE(re1),
            YSIZE(re1),
            XSIZE(re1));
    for (int i=0; i < 16; i++){
        printf("%3.1f \n", re1.data[i]);
    }

    printf("GPU: this is real image, stored in array(%d, %d, %d, %d)\n",
           NSIZE(re2),
           ZSIZE(re2),
           YSIZE(re2),
           XSIZE(re2));
    for (int i=0; i < 16; i++){
        printf("%3.1f \n", re2.data[i]);
    }
#endif

    return diff;
}

/*******************************************************/
/* CuFFT for 2D image c2r realization
 * param@ src: MultidimArray<fComplex>, using array part
 * param@ dest: MultidimArray<float>, using array part
 */
/*******************************************************/
#define TIMING
#ifdef TIMING
    #define RCTIC(label) (timer1.tic(label))
    #define RCTOC(label) (timer1.toc(label))

    Timer timer1;
        int TIMING_GPU_RESIZE = timer1.setNew("GPU - resize");
        int TIMING_GPU_MALLOC = timer1.setNew("GPU - malloc");
        int TIMING_GPU_MEMCPYHD = timer1.setNew("GPU - memcpy host to device");
        int TIMING_GPU_PLAN = timer1.setNew("GPU - plan");
        int TIMING_GPU_EXEC = timer1.setNew("GPU - exec");
        int TIMING_GPU_MEMCPYDH = timer1.setNew("GPU - memcpy device to host");
        int TIMING_GPU_FINISH = timer1.setNew("GPU - free");
        int TIMING_GPU_IFFT_IN = timer1.setNew("GPU - iFFT");

#else
    #define RCTIC(label)
	#define RCTOC(label)
#endif

void CuFFT::inverseFourierTransform(
        MultidimArray<fComplex>& src,
        MultidimArray<float>& dest)
{
    /* http://www.orangeowlsolutions.com/archives/1173 arct
     * https://docs.nvidia.com/cuda/cufft/index.html#cufftdoublecomplex 4.2.1
     * https://docs.nvidia.com/cuda/cufft/index.html 3.9.3
     * https://www.beechwood.eu/using-cufft/ time
     * */
//    RCTIC(TIMING_GPU_IFFT_IN);
//    float elapsedTime = 0;
//    cudaEvent_t start,stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start,0);


    RCTIC(TIMING_GPU_RESIZE);
    if (!areSizesCompatible(dest, src))
    {
        resizeRealToMatch(dest, src);
    }
    RCTOC(TIMING_GPU_RESIZE);


    MultidimArray<fComplex> src2 = src;

    std::vector<int> N(0);
    if (dest.zdim > 1) N.push_back(dest.zdim);
    if (dest.ydim > 1) N.push_back(dest.ydim);
    N.push_back(dest.xdim);




    cufftComplex *host_comp_data, *device_comp_data;
    cufftReal    *host_real_data, *device_real_data;

    /* https://stackoverflow.com/questions/16511526/cufft-and-fftw-data-structures-are-cufftcomplex-and-fftwf-complex-interchangabl
     * Are cufftComplex and fftwf_complex interchangable? yes!
     */
    RCTIC(TIMING_GPU_MALLOC);
    host_comp_data = (cufftComplex*) MULTIDIM_ARRAY(src2);
    host_real_data = MULTIDIM_ARRAY(dest);

    gpuErrchk(cudaMalloc((void**)&device_real_data, sizeof(cufftReal)*N[0]*N[1]));
    gpuErrchk(cudaMalloc((void**)&device_comp_data, sizeof(cufftComplex)*N[0]*(N[1]/2+1)));
    RCTOC(TIMING_GPU_MALLOC);

    RCTIC(TIMING_GPU_MEMCPYHD);
    cudaMemcpy(device_comp_data, host_comp_data, sizeof(cufftComplex)*N[0]*(N[1]/2+1), cudaMemcpyHostToDevice);
    RCTOC(TIMING_GPU_MEMCPYHD);


    RCTIC(TIMING_GPU_PLAN);
    cufftHandle planIn;

    /* Create a 2D FFT plan. */
    cufftPlan2d(&planIn,  N[0], N[1], CUFFT_C2R);
    RCTOC(TIMING_GPU_PLAN);

    RCTIC(TIMING_GPU_EXEC);
    cufftExecC2R(planIn, device_comp_data, device_real_data);
    static int a = 1;
    printf("shape: %d, %d\n", N[0], N[1], a++);
    RCTOC(TIMING_GPU_EXEC);

    RCTIC(TIMING_GPU_MEMCPYDH);
    cudaMemcpy(host_real_data, device_real_data, sizeof(cufftReal)*N[0]*N[1], cudaMemcpyDeviceToHost);
    RCTOC(TIMING_GPU_MEMCPYDH);

    RCTIC(TIMING_GPU_FINISH);
    cufftDestroy(planIn);
    gpuErrchk(cudaFree(device_comp_data));
    gpuErrchk(cudaFree(device_real_data));

//    diff(dest,dest);

    //GET CALCULATION TIME
//    cudaEventRecord(stop,0);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&elapsedTime,start,stop);
//
//    RCTOC(TIMING_GPU_FINISH);
//    printf("CUFFT Calculation COMPLETED IN : % 5.3f ms \n",elapsedTime);
//    RCTOC(TIMING_GPU_IFFT_IN);

#ifdef TIMING
    timer1.printTimes(false);
#endif
}

//void CuFFT::inverseFourierTransformcpu(
//        MultidimArray<fComplex>& src,
//        MultidimArray<float>& dest)
//{
//    /* http://www.orangeowlsolutions.com/archives/1173 arct
//     * https://docs.nvidia.com/cuda/cufft/index.html#cufftdoublecomplex 4.2.1
//     * https://docs.nvidia.com/cuda/cufft/index.html 3.9.3
//     * https://www.beechwood.eu/using-cufft/ time
//     * */
//    if (!areSizesCompatible(dest, src))
//    {
//        resizeRealToMatch(dest, src);
//    }
//
//    MultidimArray<fComplex> src2 = src;
//
//    std::vector<int> N(0);
//    if (dest.zdim > 1) N.push_back(dest.zdim);
//    if (dest.ydim > 1) N.push_back(dest.ydim);
//    N.push_back(dest.xdim);
//
//