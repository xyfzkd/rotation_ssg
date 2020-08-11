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
以上命令可以生成`run_motioncorr`
  
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