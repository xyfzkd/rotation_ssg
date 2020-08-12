#undef ALTCPU
#include <sys/time.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <signal.h>

#include "src/ml_optimiser.h"
#include "src/acc/acc_ptr.h"
#include "src/acc/acc_projector.h"
#include "src/acc/acc_backprojector.h"
#include "src/acc/acc_projector_plan.h"
#include "src/acc/cuda/cuda_kernels/helper.cuh"
#include "src/acc/cuda/cuda_mem_utils.h"
#include "src/acc/cuda/cuda_settings.h"
#include "src/acc/cuda/cuda_benchmark_utils.h"
#include "src/acc/cuda/cuda_fft.h"

#include "src/macros.h"
#include "src/error.h"

#ifdef CUDA_FORCESTL
#include "src/acc/cuda/cuda_utils_stl.cuh"
#else
#include "src/acc/cuda/cuda_utils_cub.cuh"
#endif

#include "src/acc/utilities.h"
#include "src/acc/acc_helper_functions.h"

#include "src/acc/cuda/cuda_motioncor.h"

void MotionCorCuda::run(){
    long int my_first_micrograph, my_last_micrograph, my_nr_micrographs;
    if(node!=NULL)
    {
        // Each node does part of the work
        divide_equally(baseRnr->fn_micrographs.size(),
                       node->size,
                       node->rank,
                       my_first_micrograph,
                       my_last_micrograph);
    }
    else
    {
        my_first_micrograph = 0;
        my_last_micrograph = baseRnr->fn_micrographs.size() - 1;
    }
    my_nr_micrographs = my_last_micrograph - my_first_micrograph + 1;

    int barstep;
    if (baseRnr->verb > 0)
    {
        if (do_own)
            std::cout << " Correcting beam-induced motions using our own implementation ..." << std::endl;
        else if (do_motioncor2)
            std::cout << " Correcting beam-induced motions using Shawn Zheng's MOTIONCOR2 ..." << std::endl;
        else
            REPORT_ERROR("Bug: by now it should be clear whether to use MotionCor2 or Unblur...");

        init_progress_bar(my_nr_micrographs);
        barstep = XMIPP_MAX(1, my_nr_micrographs / 60);
    }
    for (long int imic = my_first_micrograph; imic < my_last_micrograph; imic++)
    {
        if (baseRnr->verb > 0 && imic % barstep == 0)
            progress_bar(imic);


        Micrograph mic(baseRnr->fn_micrographs[imic],
                       baseRnr->fn_gain_reference,
                       baseRnr->bin_factor,
                       baseRnr->eer_upsampling,
                       baseRnr->eer_grouping);

        // Get angpix and voltage from the optics groups:
        obsModel.opticsMdt.getValue(EMDL_CTF_VOLTAGE,
                                    baseRnr->voltage,
                                    baseRnr->optics_group_micrographs[imic]-1);
        obsModel.opticsMdt.getValue(EMDL_MICROGRAPH_ORIGINAL_PIXEL_SIZE,
                                    baseRnr->angpix,
                                    baseRnr->optics_group_micrographs[imic]-1);

        bool result = executeOwnMotionCorrection(mic);
        if (result) {
            saveModel(mic);
            plotShifts(baseRnr->fn_micrographs[imic], mic);
        }
    }

    if (baseRnr->verb > 0)
        progress_bar(my_nr_micrographs);

    // Make a logfile with the shifts in pdf format and write output STAR files
    generateLogFilePDFAndWriteStarFiles();

#ifdef TIMING
    timer.printTimes(false);
#endif
#ifdef TIMING_FFTW
    timer_fftw.printTimes(false);
#endif
}

__global__ void
MotioncorrRunner::alignPatch(std::vector<MultidimArray<fComplex> > &Fframes,
                             const int pnx,
                             const int pny,
                             const RFLOAT scaled_B,
                             std::vector<RFLOAT> &xshifts,
                             std::vector<RFLOAT> &yshifts,
                             std::ostream &logfile)
{
    std::vector<Image<float> > Iccs(n_threads);
    MultidimArray<fComplex> Fref;
    std::vector<MultidimArray<fComplex> > Fccs(n_threads);
    MultidimArray<float> weight;
    std::vector<RFLOAT> cur_xshifts, cur_yshifts;
    bool converged = false;

    // Parameters TODO: make an option
    int search_range = 50; // px
    const RFLOAT tolerance = 0.5; // px
    const RFLOAT EPS = 1e-15;

    // Shifts within an iteration
    const int n_frames = xshifts.size();
    cur_xshifts.resize(n_frames);
    cur_yshifts.resize(n_frames);

    if (pny % 2 == 1 || pnx % 2 == 1) {
        REPORT_ERROR("Patch size must be even");
    }

    // Calculate the size of down-sampled CCF
    float ccf_requested_scale = ccf_downsample;
    if (ccf_downsample <= 0) {
        ccf_requested_scale = sqrt(-log(1E-8) / (2 * scaled_B)); // exp(-2 B max_dist^2) = 1E-8
    }
    int ccf_nx = findGoodSize(int(pnx * ccf_requested_scale)), ccf_ny = findGoodSize(int(pny * ccf_requested_scale));
    if (ccf_nx > pnx) ccf_nx = pnx;
    if (ccf_ny > pny) ccf_ny = pny;
    if (ccf_nx % 2 == 1) ccf_nx++;
    if (ccf_ny % 2 == 1) ccf_ny++;
    const int ccf_nfx = ccf_nx / 2 + 1, ccf_nfy = ccf_ny;
    const int ccf_nfy_half = ccf_ny / 2;
    const RFLOAT ccf_scale_x = (RFLOAT)pnx / ccf_nx;
    const RFLOAT ccf_scale_y = (RFLOAT)pny / ccf_ny;
    search_range /= (ccf_scale_x > ccf_scale_y) ? ccf_scale_x : ccf_scale_y; // account for the increase of pixel size in CCF
    if (search_range * 2 + 1 > ccf_nx) search_range = ccf_nx / 2 - 1;
    if (search_range * 2 + 1 > ccf_ny) search_range = ccf_ny / 2 - 1;

    const int nfx = XSIZE(Fframes[0]), nfy = YSIZE(Fframes[0]);
    const int nfy_half = nfy / 2;

    Fref.reshape(ccf_nfy, ccf_nfx);
    for (int i = 0; i < n_threads; i++) {
        Iccs[i]().reshape(ccf_ny, ccf_nx);
        Fccs[i].reshape(Fref);
    }

#ifdef DEBUG
    std::cout << "Patch Size X = " << pnx << " Y  = " << pny << std::endl;
	std::cout << "Fframes X = " << nfx << " Y = " << nfy << std::endl;
	std::cout << "Fccf X = " << ccf_nfx << " Y = " << ccf_nfy << std::endl;
	std::cout << "CCF crop request = " << ccf_requested_scale << ", actual X = " << 1 / ccf_scale_x << " Y = " << 1 / ccf_scale_y << std::endl;
	std::cout << "CCF search range = " << search_range << std::endl;
	std::cout << "Trajectory size: " << xshifts.size() << std::endl;
#endif

    // Initialize B factor weight
    weight.reshape(Fref);
    RCTIC(TIMING_PREP_WEIGHT);
#pragma omp parallel for num_threads(n_threads)
    for (int y = 0; y < ccf_nfy; y++) {
        const int ly = (y > ccf_nfy_half) ? (y - ccf_nfy) : y;
        RFLOAT ly2 = ly * (RFLOAT)ly / (nfy * (RFLOAT)nfy);

        for (int x = 0; x < ccf_nfx; x++) {
            RFLOAT dist2 = ly2 + x * (RFLOAT)x / (nfx * (RFLOAT)nfx);
            DIRECT_A2D_ELEM(weight, y, x) = exp(- 2 * dist2 * scaled_B); // 2 for Fref and Fframe
        }
    }
    RCTOC(TIMING_PREP_WEIGHT);

    for (int iter = 1; iter	<= max_iter; iter++) {
        RCTIC(TIMING_MAKE_REF);
        Fref.initZeros();

#pragma omp parallel for num_threads(n_threads)
        for (int y = 0; y < ccf_nfy; y++) {
            const int ly = (y > ccf_nfy_half) ? (y - ccf_nfy + nfy) : y;
            for (int x = 0; x < ccf_nfx; x++) {
                for (int iframe = 0; iframe < n_frames; iframe++) {
                    DIRECT_A2D_ELEM(Fref, y, x) += DIRECT_A2D_ELEM(Fframes[iframe], ly, x);
                }
            }
        }
        RCTOC(TIMING_MAKE_REF);

#pragma omp parallel for num_threads(n_threads)
        for (int iframe = 0; iframe < n_frames; iframe++) {
            const int tid = omp_get_thread_num();

            RCTIC(TIMING_CCF_CALC);
            for (int y = 0; y < ccf_nfy; y++) {
                const int ly = (y > ccf_nfy_half) ? (y - ccf_nfy + nfy) : y;
                for (int x = 0; x < ccf_nfx; x++) {
                    DIRECT_A2D_ELEM(Fccs[tid], y, x) = (DIRECT_A2D_ELEM(Fref, y, x) - DIRECT_A2D_ELEM(Fframes[iframe], ly, x)) *
                                                       DIRECT_A2D_ELEM(Fframes[iframe], ly, x).conj() * DIRECT_A2D_ELEM(weight, y, x);
                }
            }
            RCTOC(TIMING_CCF_CALC);

            RCTIC(TIMING_CCF_IFFT);
            NewFFT::inverseFourierTransform(Fccs[tid], Iccs[tid]());
            RCTOC(TIMING_CCF_IFFT);

            RCTIC(TIMING_CCF_FIND_MAX);
            RFLOAT maxval = -1E30;
            int posx = 0, posy = 0;
            for (int y = -search_range; y <= search_range; y++) {
                const int iy = (y < 0) ? ccf_ny + y : y;

                for (int x = -search_range; x <= search_range; x++) {
                    const int ix = (x < 0) ? ccf_nx + x : x;
                    RFLOAT val = DIRECT_A2D_ELEM(Iccs[tid](), iy, ix);
//					std::cout << "(x, y) = " << x << ", " << y << ", (ix, iy) = " << ix << " , " << iy << " val = " << val << std::endl;
                    if (val > maxval) {
                        posx = x; posy = y;
                        maxval = val;
                    }
                }
            }

            int ipx_n = posx - 1, ipx = posx, ipx_p = posx + 1, ipy_n = posy - 1, ipy = posy, ipy_p = posy + 1;
            if (ipx_n < 0) ipx_n = ccf_nx + ipx_n;
            if (ipx < 0) ipx = ccf_nx + ipx;
            if (ipx_p < 0) ipx_p = ccf_nx + ipx_p;
            if (ipy_n < 0) ipy_n = ccf_ny + ipy_n;
            if (ipy < 0) ipy = ccf_ny + ipy;
            if (ipy_p < 0) ipy_p = ccf_ny + ipy_p;

            // Quadratic interpolation by Jasenko
            RFLOAT vp, vn;
            vp = DIRECT_A2D_ELEM(Iccs[tid](), ipy, ipx_p);
            vn = DIRECT_A2D_ELEM(Iccs[tid](), ipy, ipx_n);
            if (std::abs(vp + vn - 2.0 * maxval) > EPS) {
                cur_xshifts[iframe] = posx - 0.5 * (vp - vn) / (vp + vn - 2.0 * maxval);
            } else {
                cur_xshifts[iframe] = posx;
            }

            vp = DIRECT_A2D_ELEM(Iccs[tid](), ipy_p, ipx);
            vn = DIRECT_A2D_ELEM(Iccs[tid](), ipy_n, ipx);
            if (std::abs(vp + vn - 2.0 * maxval) > EPS) {
                cur_yshifts[iframe] = posy - 0.5 * (vp - vn) / (vp + vn - 2.0 * maxval);
            } else {
                cur_yshifts[iframe] = posy;
            }
            cur_xshifts[iframe] *= ccf_scale_x;
            cur_yshifts[iframe] *= ccf_scale_y;
#ifdef DEBUG_OWN
            std::cout << "tid " << tid << " Frame " << 1 + iframe << ": raw shift x = " << posx << " y = " << posy << " cc = " << maxval << " interpolated x = " << cur_xshifts[iframe] << " y = " << cur_yshifts[iframe] << std::endl;
#endif
            RCTOC(TIMING_CCF_FIND_MAX);
        }

        // Set origin
        RFLOAT x_sumsq = 0, y_sumsq = 0;
        for (int iframe = n_frames - 1; iframe >= 0; iframe--) { // do frame 0 last!
            cur_xshifts[iframe] -= cur_xshifts[0];
            cur_yshifts[iframe] -= cur_yshifts[0];
            x_sumsq += cur_xshifts[iframe] * cur_xshifts[iframe];
            y_sumsq += cur_yshifts[iframe] * cur_yshifts[iframe];
        }
        cur_xshifts[0] = 0; cur_yshifts[0] = 0;

        for (int iframe = 0; iframe < n_frames; iframe++) {
            xshifts[iframe] += cur_xshifts[iframe];
            yshifts[iframe] += cur_yshifts[iframe];
//			std::cout << "Shift for Frame " << iframe << ": delta_x = " << cur_xshifts[iframe] << " delta_y = " << cur_yshifts[iframe] << std::endl;
        }

        // Apply shifts
        // Since the image is not necessarily square, we cannot use the method in fftw.cpp
        RCTIC(TIMING_FOURIER_SHIFT);
#pragma omp parallel for num_threads(n_threads)
        for (int iframe = 1; iframe < n_frames; iframe++) {
            shiftNonSquareImageInFourierTransform(Fframes[iframe], -cur_xshifts[iframe] / pnx, -cur_yshifts[iframe] / pny);
        }
        RCTOC(TIMING_FOURIER_SHIFT);

        // Test convergence
        RFLOAT rmsd = std::sqrt((x_sumsq + y_sumsq) / n_frames);
        logfile << " Iteration " << iter << ": RMSD = " << rmsd << " px" << std::endl;

        if (rmsd < tolerance) {
            converged = true;
            break;
        }
    }

#ifdef DEBUG_OWN
    for (int iframe = 0; iframe < n_frames; iframe++) {
		std::cout << iframe << " " << xshifts[iframe] << " " << yshifts[iframe] << std::endl;
	}
#endif

    return converged;
}