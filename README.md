# rotation in Shilab
------

> **`relion ver3.1 src/acc/cuda` 优化`backproject`的kernel**
> 
> 根据`relion31_tutorial.pdf`，下载数据集`relion30_tutorial/Movies/*.tiff`，范例在`PrecalculatedResults `，工作目录为`relion30_tutorial`

* 需要快速熟悉relion的流程，例如要每步操作的输出文件类型，为优化做铺垫...
* 优化
* ...

## preprocessing

>根据`relion31_tutorial.pdf`，文档提供GUI下的操作，猜测描述猜测`args`

```bash
relion_import \
--i "Movies/*.tiff" \
--odir Import/job001/ \
--ofile movies.star \
--do_movies true \
--optics_group_name opticsGroup1 \
--optics_group_mtf mtf_k2_200kV.star \
--angpix 0.885 \
--kV 200 \
--Cs 1.4 \
--Q0 0.1 
```
1. 自建`odir`
2. `ifile`记得加双引号
3. 范例可见于`PrecalculatedResults/Import/job001/note.txt`
4. 输出`less Import/job001/movies.star`，

## motioncorr
> 根据上个步骤的总结3，直接运行

```
`which relion_run_motioncorr` \
--i Import/job001/movies.star \
--o MotionCorr/job002/ \
--first_frame_sum 1 \
--last_frame_sum 0 \
--use_own  \
--j 24 \
--bin_factor 1 \
--bfactor 150 \
--dose_per_frame 1.277 \
--preexposure 0 \
--patch_x 5 \
--patch_y 5 \
--gainref Movies/gain.mrc \
--gain_rot 0 \
--gain_flip 0 \
--dose_weighting  \
--grouping_for_ps 3  \
--pipeline_control MotionCorr/job002/
```
1. 有计时，`0.60/4.80 min .......~~(,_,">                                                 [oo]`

## ctf estimation
>根据`bp/PrecalculatedResults/CtfFind/job003/note.txt`，

1. 要下载[`ctffind`](https://grigoriefflab.umassmed.edu/ctf_estimation_ctffind_ctftilt)，同时更改`--ctffind_exe`
2. 要下载`csh`，推荐`conda`安装`tcsh`并`alias`

## Manual particle picking
>根据`bp/PrecalculatedResults/ManualPick/job004/note.txt`，略去手动挑选操作，执行`echo CtfFind/job003/micrographs_ctf.star > ManualPick/job004/coords_suffix_manualpick.star`

1. 教程提到基于`LoG`自动挑选策略，是用到`scripts/relion_it.py`，然而教程不采纳完全自动的策略，而是采用`ver3.1`下不存在的`manual picking`，这是`ver3.0`以来的教程和实际操作的矛盾。
2. 在以上情况下，基于对工作目录已有一个大致的了解，直接进入优化这步...

## 优化
阅读教程，知**backprojection**可能包含在`relion_refine_mpi`，`relion_refine_mpi `与目标息息相关

结合源代码`src/acc/cuda`下的`backprojector.cu*`文件，溯源头文件，发现`AccBackprojector`类的定义和类方法的定义，此外发现或与加速相关的语句

```bash
grep -rn "AccBackprojector" *
src/acc/cuda/cuda_ml_optimiser.h:26:	std::vector< AccBackprojector > backprojectors;
src/acc/cpu/cpu_ml_optimiser.h:27:	std::vector< AccBackprojector > backprojectors;
src/acc/acc_backprojector_impl.h:8:size_t AccBackprojector::setMdlDim(
src/acc/acc_backprojector_impl.h:54:void AccBackprojector::initMdl()
src/acc/acc_backprojector_impl.h:84:void AccBackprojector::getMdlData(XFLOAT *r, XFLOAT *i, XFLOAT * w)
src/acc/acc_backprojector_impl.h:101:void AccBackprojector::getMdlDataPtrs(XFLOAT *& r, XFLOAT *& i, XFLOAT *& w)
src/acc/acc_backprojector_impl.h:110:void AccBackprojector::clear()
src/acc/acc_backprojector_impl.h:140:AccBackprojector::~AccBackprojector()
src/acc/acc_helper_functions.h:97:		AccBackprojector &BP,
src/acc/acc_backprojector.h:15:class AccBackprojector
src/acc/acc_backprojector.h:38:	AccBackprojector():
src/acc/acc_backprojector.h:86:	~AccBackprojector();
src/acc/acc_helper_functions_impl.h:616:		AccBackprojector &BP,
```

在`build/Makefile`找到

```vim
 #=============================================================================
 # Target rules for targets named run_motioncorr
 
 # Build rule for target.
 run_motioncorr: cmake_check_build_system
     $(MAKE) -f CMakeFiles/Makefile2 run_motioncorr
 .PHONY : run_motioncorr
 
 # fast build rule for target.
 run_motioncorr/fast:
     $(MAKE) -f src/apps/CMakeFiles/run_motioncorr.dir/build.make src/apps/CMakeFiles/        run_motioncorr.dir/build
 .PHONY : run_motioncorr/fast
 
 #=============================================================================
  
```
以上命令可以生成`run_motioncorr`, 但无脑`make`也成，以下是展开
  
```vim
  # Target rules for target src/apps/CMakeFiles/run_motioncorr.dir
 
 # All Build rule for target.
 src/apps/CMakeFiles/run_motioncorr.dir/all: src/apps/CMakeFiles/relion_lib.dir/all
     $(MAKE) -f src/apps/CMakeFiles/run_motioncorr.dir/build.make src/apps/CMakeFiles/run_motioncorr.dir/depend
     $(MAKE) -f src/apps/CMakeFiles/run_motioncorr.dir/build.make src/apps/CMakeFiles/run_motioncorr.dir/build
     @$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --progress-dir=/data/xieyufeng/relion/build/        CMakeFiles --progress-num=95 "Built target run_motioncorr"
 .PHONY : src/apps/CMakeFiles/run_motioncorr.dir/all
 
 # Include target in all.
 all: src/apps/CMakeFiles/run_motioncorr.dir/all
 
 .PHONY : all
 
 # Build rule for subdir invocation for target.
 src/apps/CMakeFiles/run_motioncorr.dir/rule: cmake_check_build_system
     $(CMAKE_COMMAND) -E cmake_progress_start /data/xieyufeng/relion/build/CMakeFiles 59
     $(MAKE) -f CMakeFiles/Makefile2 src/apps/CMakeFiles/run_motioncorr.dir/all
     $(CMAKE_COMMAND) -E cmake_progress_start /data/xieyufeng/relion/build/CMakeFiles 0
 .PHONY : src/apps/CMakeFiles/run_motioncorr.dir/rule
 
 # Convenience name for target.
 run_motioncorr: src/apps/CMakeFiles/run_motioncorr.dir/rule
 
 .PHONY : run_motioncorr
 
 # clean rule for target.
 src/apps/CMakeFiles/run_motioncorr.dir/clean:
     $(MAKE) -f src/apps/CMakeFiles/run_motioncorr.dir/build.make src/apps/CMakeFiles/run_motioncorr.dir/clean
 .PHONY : src/apps/CMakeFiles/run_motioncorr.dir/clean
 
 # clean rule for target.
 clean: src/apps/CMakeFiles/run_motioncorr.dir/clean
 
 .PHONY : clean
```

以下大概是cuda文件的编译，复杂，暂时不想看

```
 # Generate the dependency file
 cuda_execute_process(
   "Generating dependency file: ${NVCC_generated_dependency_file}"
   COMMAND "${CUDA_NVCC_EXECUTABLE}"
   -M
   ${CUDACC_DEFINE}
   "${source_file}"
   -o "${NVCC_generated_dependency_file}"
   ${CCBIN}
   ${nvcc_flags}
   ${nvcc_host_compiler_flags}
   ${depends_CUDA_NVCC_FLAGS}
   -DNVCC
   ${CUDA_NVCC_INCLUDE_ARGS}
   )
   
    -D__CUDACC__
    /data/xieyufeng/relion/src/acc/cuda/cuda_projector.cu
    /data/xieyufeng/relion/build/src/apps/CMakeFiles/relion_gpu_util.dir/__/    acc/cuda/relion_gpu_util_generated_cuda_projector.cu.o.NVCC-depend

-m64;-DINSTALL_LIBRARY_DIR=/data/xieyufeng/software/bin/lib/;-DSOURCE_DIR=/data/xieyufeng/       relion/src/;-DACC_CUDA=2;-DACC_CPU=1;-DCUDA;-DALLOW_CTF_IN_SGD;-DHAVE_SINCOS;-DHAVE_TIFF;-DHAVE_PNG

"-I/usr/local/cuda-10.1/include;-I/usr/lib/openmpi/include/openmpi/opal/mca/event/   libevent2021/libevent;-I/usr/lib/openmpi/include/openmpi/opal/mca/event/libevent2021/libevent/include;-I/usr/   lib/openmpi/include;-I/usr/lib/openmpi/include/openmpi;-I/data/xieyufeng/relion;-I/data/xieyufeng/relion/       external/fftw/include;-I/usr/local/cuda-10.1/include"

   "Generating ${generated_file}"
   COMMAND "${CUDA_NVCC_EXECUTABLE}"
   "${source_file}"
   ${cuda_language_flag}
   ${format_flag} -o "${generated_file}"
   ${CCBIN}
   ${nvcc_flags}
   ${nvcc_host_compiler_flags}
   ${CUDA_NVCC_FLAGS}
   -DNVCC
   ${CUDA_NVCC_INCLUDE_ARGS}
   
   /usr/local/cuda-10.1/bin/nvcc 
   
```
   
   
   
   
   
   
# profiling

目标是`src/run_motioncorr.cpp`的`alignPatch`函数`GPU`化，实际上是为了加速，那么做一个时间的profiling

## alignPatch
alignPatch的六个主要部分如下

```
TIMING_PREP_WEIGHT
TIMING_MAKE_REF
TIMING_CCF_CALC
TIMING_CCF_IFFT
TIMING_CCF_FIND_MAX
TIMING_FOURIER_SHIFT

align - prep weight 2.8
align - make reference 
align - calc CCF (in thread)
align - iFFT CCF (in thread)
align - argmax CCF (in thread)
align - shift in Fourier space
```
使用tutorialdata自带的`job002`下的脚本，得到如下

```
read gain                          : 1.449 sec (60406 microsec/operation)
read movie                         : 7.676 sec (319840 microsec/operation)
apply gain                         : 1.566 sec (65284 microsec/operation)
initial sum                        : 4.695 sec (195663 microsec/operation)
detect hot pixels                  : 1.15 sec (47953 microsec/operation)
fix defects                        : 5.509 sec (229573 microsec/operation)
global FFT                         : 35.273 sec (1469723 microsec/operation)
power spectrum                     : 66.63 sec (2776284 microsec/operation)
power - sum                        : 14.228 sec (592853 microsec/operation)
power - square                     : 44.964 sec (1873517 microsec/operation)
power - crop                       : 0.282 sec (11773 microsec/operation)
power - resize                     : 7.005 sec (291894 microsec/operation)
global alignment                   : 10.926 sec (455275 microsec/operation)
global iFFT                        : 37.949 sec (1581241 microsec/operation)
prepare patch                      : 39.466 sec (65777 microsec/operation)
prep patch - clip (in thread)      : 7.905 sec (693 microsec/operation)
prep patch - FFT (in thread)       : 621.909 sec (43257 microsec/operation)
patch alignment                    : 35.366 sec (58943 microsec/operation)
align - prep weight                : 2.872 sec (4602 microsec/operation)
align - make reference             : 6.722 sec (5264 microsec/operation)
align - calc CCF (in thread)       : -48.871 sec (-2431 microsec/operation)
align - iFFT CCF (in thread)       : 243.893 sec (8263 microsec/operation)
align - argmax CCF (in thread)     : 0.109 sec (3 microsec/operation)
align - shift in Fourier space     : 17.424 sec (13645 microsec/operation)
fit polynomial                     : 0.086 sec (3614 microsec/operation)
dose weighting                     : 49.023 sec (2042654 microsec/operation)
dw - calc weight                   : 12.305 sec (512737 microsec/operation)
dw - iFFT                          : 36.717 sec (1529914 microsec/operation)
real space interpolation           : 18.609 sec (775412 microsec/operation)
binning                            : 0 sec (0 microsec/operation)
```
最耗时有两个部分，如下，其中一个在`alignPatch`函数中，
   
```
RCTIC(TIMING_PATCH_FFT);
NewFFT::FourierTransform(Ipatches[tid], Fpatches[igroup]);
RCTOC(TIMING_PATCH_FFT);

RCTIC(TIMING_CCF_IFFT);
NewFFT::inverseFourierTransform(Fccs[tid], Iccs[tid]());
RCTOC(TIMING_CCF_IFFT);
```

* emmmm，可能代码有误，有部分竟然是负数
   
## exp
以下要改写函数，发现函数有线程加速部分，基本上是看不大懂，可能知道的知识点也只有SIMD的avvx指令，也只是概念上理解，但根据relion文档描述，感觉可以理解为一个线程负责一个或若干个loop（是照片的loop，照片可以并行地一张张处理），设置`--j 12`，可能指代一个线程负责两张照片，其中一共有24张照片

除此之外，可能还有`mpirun`模式，并且测试发现能加速到两三分钟，咱先不管
### 去thread
观察代码，发现六模块每个都有线程加速的pattern，删掉，并修改数据结构取消`tid`的作用，猜测引入vector是为了loop中每个element不相互影响，那不多线程的话，tid就没有作用

这样得到一个去threads的版本

```
de_threads
align - prep weight                : 0.947 sec (1517 microsec/operation)
align - make reference             : 5.615 sec (4397 microsec/operation)
align - calc CCF (in thread)       : 4.373 sec (142 microsec/operation)
align - iFFT CCF (in thread)       : 33.004 sec (1076 microsec/operation)
align - argmax CCF (in thread)     : 0.069 sec (2 microsec/operation)
align - shift in Fourier space     : 116.809 sec (91472 microsec/operation)
```

### 给最后一个最耗时的模块加thread
```
threads_for_shift_fourier
align - prep weight                : 0.947 sec (1518 microsec/operation)
align - make reference             : 6.315 sec (4945 microsec/operation)
align - calc CCF (in thread)       : 4.413 sec (144 microsec/operation)
align - iFFT CCF (in thread)       : 33.749 sec (1101 microsec/operation)
align - argmax CCF (in thread)     : 0.073 sec (2 microsec/operation)
align - shift in Fourier space     : 17.861 sec (13987 microsec/operation)
```
### 给倒数四个的模块加thread
又出现负数，可能和tid有关，暂时先不管了

```
threads_for_calc_ccf
align - prep weight                : 0.929 sec (1489 microsec/operation)
align - make reference             : 12.678 sec (4063 microsec/operation)
align - calc CCF (in thread)       : -85.7 sec (-1236 microsec/operation)
align - iFFT CCF (in thread)       : 104.649 sec (1432 microsec/operation)
align - argmax CCF (in thread)     : 0.28 sec (3 microsec/operation)
align - shift in Fourier space     : 35.368 sec (11336 microsec/operation)
```


得到结论
# 可以给最后一个模块做gpu优化

# 研究被调用函数
根据gdb设置断点，找到被调用函数的位置，根据where给出的函数原型找源代码

发现核心函数可能是`new_ft.cpp`的第333行左右的`fftwf_execute_dft_c2r(plan.getBackward(), in, MULTIDIM_ARRAY(dest));`

* 它是cpu版本的傅立叶变换，找gpu版本
* 在relion目录下搜寻cufft的运用，发现了`cuda_kernels/helper.cuh`的一些敏感函数`cuda_kernel_centerFFT_2D`等，没有详细描述不知道它是做什么的，同时这些kernel自己做些运算，感觉不大靠谱（因为已经有cufft了 不知道为啥还要自己造轮子）
* 目前大概只懂要写device的kernel，也要写host的cpp，关于写kernel要多在网上找找，关于写cpp师兄提过app/warpper，暂时没细看，冲冲冲

236 FloatPlan p(dest, src2);

```
NewFFT::FloatPlan::FloatPlan(
		MultidimArray<float>& real,
		MultidimArray<fComplex>& complex,
		unsigned int flags)
:
	reusable(flags & FFTW_UNALIGNED), 
	w(real.xdim), h(real.ydim), d(real.zdim),
	realPtr(MULTIDIM_ARRAY(real)), 
	complexPtr((float*)MULTIDIM_ARRAY(complex))
{
	std::vector<int> N(0);
	if (d > 1) N.push_back(d);
	if (h > 1) N.push_back(h);
	           N.push_back(w);
	
	const int ndim = N.size();
	
	pthread_mutex_lock(&fftw_plan_mutex_new);
	
	fftwf_plan planForward = fftwf_plan_dft_r2c(
			ndim, &N[0],
			MULTIDIM_ARRAY(real),
			(fftwf_complex*) MULTIDIM_ARRAY(complex),
			flags);
	
	fftwf_plan planBackward = fftwf_plan_dft_c2r(
			ndim, &N[0],
			(fftwf_complex*) MULTIDIM_ARRAY(complex),
			MULTIDIM_ARRAY(real),
			flags);
	
	pthread_mutex_unlock(&fftw_plan_mutex_new);
	
	if (planForward == NULL || planBackward == NULL)
	{
		REPORT_ERROR("FFTW plans cannot be created");
	}
	
	plan = std::shared_ptr<Plan>(new Plan(planForward, planBackward));
}
```

# 初始版本

*  gpu mode


```
align - prep weight                : 0.944 sec (1513 microsec/operation)
align - make reference             : 2.738 sec (4389 microsec/operation)
align - calc CCF (in thread)       : 1.731 sec (115 microsec/operation)
align - iFFT CCF (in thread)       : 5.704 sec (380 microsec/operation)
align - argmax CCF (in thread)     : 0.04 sec (2 microsec/operation)
align - shift in Fourier space     : 6.344 sec (10167 microsec/operation)
```
但是`Cuda error: Failed to allocate`，写得有问题

*  cpu mode


```
align - prep weight                : 0.954 sec (1528 microsec/operation)
align - make reference             : 6.388 sec (4991 microsec/operation)
align - calc CCF (in thread)       : 4.771 sec (155 microsec/operation)
align - iFFT CCF (in thread)       : 37.949 sec (1235 microsec/operation)
align - argmax CCF (in thread)     : 0.072 sec (2 microsec/operation)
align - shift in Fourier space     : 15.6 sec (12187 microsec/operation)
```

	FileName fn_mic = mic.getMovieFilename();
	FileName fn_avg = getOutputFileNames(fn_mic);
	FileName fn_avg_noDW = fn_avg.withoutExtension() + "_noDW.mrc";
	FileName fn_log = fn_avg.withoutExtension() + ".log";
	FileName fn_ps = fn_avg.withoutExtension() + "_PS.mrc";
	std::ofstream logfile;
	logfile.open(fn_log);

	// EER related things
	// TODO: will be refactored
	EERRenderer renderer;
	const bool isEER = EERRenderer::isEER(mic.getMovieFilename());

	int n_io_threads = n_threads;
	logfile << "Working on " << fn_mic << " with " << n_threads << " thread(s)." << std::endl << std::endl;
	if (max_io_threads > 0 && n_io_threads > max_io_threads)
	{
		n_io_threads = max_io_threads;
		logfile << "Limitted the number of IO threads per movie to " << n_io_threads << " thread(s)." << std::endl;
	}

	Image<float> Ihead, Igain, Iref;
	std::vector<MultidimArray<fComplex> > Fframes;
	std::vector<Image<float> > Iframes;
	std::vector<int> frames; // 0-indexed

	RFLOAT output_angpix = angpix * bin_factor;
	RFLOAT prescaling = 1;

	const int hotpixel_sigma = 6;
	const int fit_rmsd_threshold = 10; // px
	int nx, ny, nn;

	// Check image size
	if (!isEER)
	{
		Ihead.read(fn_mic, false, -1, false, true); // select_img -1, mmap false, is_2D true
		nx = XSIZE(Ihead()); ny = YSIZE(Ihead()); nn = NSIZE(Ihead());
	}
	else
	{
		renderer.read(fn_mic, eer_upsampling);
		nx = renderer.getWidth(); ny = renderer.getHeight();
		nn = renderer.getNFrames() / eer_grouping; // remaining frames are truncated
	}

	// Which frame to use?
	logfile << "Movie size: X = " << nx << " Y = " << ny << " N = " << nn << std::endl;
	logfile << "Frames to be used:";
	for (int i = 0; i < nn; i++) {
		// For users, all numbers are 1-indexed. Internally they are 0-indexed.
		int frame = i + 1;
		if (frame < first_frame_sum) continue;
		if (last_frame_sum > 0 && frame > last_frame_sum) continue;
		frames.push_back(i);
		logfile << " " << frame;
	}
	logfile << std::endl;

	const int n_frames = frames.size();
	Iframes.resize(n_frames);
	Fframes.resize(n_frames);
	std::vector<RFLOAT> xshifts(n_frames), yshifts(n_frames);
	
	
```
In file included from /home/xyf/relion/src/multidim_array.h:61:0,
                 from /home/xyf/relion/src/jaz/new_ft.h:27,
                 from /home/xyf/relion/src/jaz/new_ft.cpp:21:
/software/cuda-9.0/include/cufftw.h:73:0: warning: "FFTW_ESTIMATE" redefined [enabled by default]
 #define FFTW_ESTIMATE           0x01
 ^
In file included from /home/xyf/relion/src/jaz/new_ft.h:24:0,
                 from /home/xyf/relion/src/jaz/new_ft.cpp:21:
/home/xyf/anaconda3/include/fftw3.h:491:0: note: this is the location of the previous definition
 #define FFTW_ESTIMATE (1U << 6)
```


```
cpu
align - prep weight                : 0.85 sec (1363 microsec/operation)
align - make reference             : 5.421 sec (4245 microsec/operation)
align - calc CCF (in thread)       : 3.455 sec (112 microsec/operation)
align - iFFT CCF (in thread)       : 24.476 sec (798 microsec/operation)
align - argmax CCF (in thread)     : 0.052 sec (1 microsec/operation)
align - shift in Fourier space     : 7.923 sec (6204 microsec/operation)

gpu
align - prep weight                : 0.984 sec (1577 microsec/operation)
align - make reference             : 5.694 sec (4459 microsec/operation)
align - calc CCF (in thread)       : 4.75 sec (155 microsec/operation)
align - iFFT CCF (in thread)       : 88.643 sec (2892 microsec/operation)
align - argmax CCF (in thread)     : 0.203 sec (6 microsec/operation)
align - shift in Fourier space     : 11.336 sec (8877 microsec/operation)
```