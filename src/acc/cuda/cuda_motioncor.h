#ifndef CUDA_AUTOPICKER_H_
#define CUDA_AUTOPICKER_H_

#include "src/mpi.h"
#include "src/autopicker.h"
#include "src/autopicker_mpi.h"
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

class AutoPickerCuda
{
private:

	MpiNode *node;

public:

	AutoPicker *basePckr;

	CudaCustomAllocator *allocator;
	CudaFFT micTransformer;
	CudaFFT cudaTransformer1;
	CudaFFT cudaTransformer2;

	std::vector< AccProjector > projectors;

   //Class streams ( for concurrent scheduling of class-specific kernels)
	std::vector< cudaStream_t > classStreams;

	int device_id;

	bool have_warned_batching;

	//MlDeviceBundle *devBundle;

#ifdef TIMING_FILES
	relion_timer timer;
#endif

	AutoPickerCuda(AutoPicker    *basePicker, int dev_id, const char * timing_fnm);
	AutoPickerCuda(AutoPickerMpi *basePicker, int dev_id, const char * timing_fnm);

	void setupProjectors();

	void run();

	void autoPickOneMicrograph(FileName &fn_mic, long int imic);

	void calculateStddevAndMeanUnderMask(AccPtr< ACCCOMPLEX > &d_Fmic, 
			AccPtr< ACCCOMPLEX > &d_Fmic2, 
			AccPtr< ACCCOMPLEX > &d_Fmsk,
			int nr_nonzero_pixels_mask, AccPtr< XFLOAT > &d_Mstddev, 
			AccPtr< XFLOAT > &d_Mmean,
			size_t x, size_t y, size_t mic_size, size_t workSize);

	~AutoPickerCuda()
	{
		for (int i = 0; i < classStreams.size(); i++)
			HANDLE_ERROR(cudaStreamDestroy(classStreams[i]));
	}



#endif /* CUDA_AUTOPICKER_H_ */
