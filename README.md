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