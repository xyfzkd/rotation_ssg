
#include "src/motioncorr_runner.h"


class MotioncorrRunnerCuda
{

public:

    MotioncorrRunner *baseRnr;


    int device_id;


    AutoPickerCuda(AutoPicker    *basePicker, int dev_id, const char * timing_fnm);



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

//private:

//	// Uses Roseman2003 formulae to calculate stddev under the mask through FFTs
//	// The FFTs of the micrograph (Fmic), micrograph-squared (Fmic2) and the mask (Fmsk) need to be provided at downsize_mic
//	// The putput (Mstddev) will be at (binned) micrograph_size
//	void calculateStddevAndMeanUnderMask(const MultidimArray<Complex > &Fmic, const MultidimArray<Complex > &Fmic2,
//			MultidimArray<Complex > &Fmsk, int nr_nonzero_pixels_mask, MultidimArray<RFLOAT> &Mstddev, MultidimArray<RFLOAT> &Mmean);
//
//	// Peak search for all pixels above a given threshold in the map
//	void peakSearch(const MultidimArray<RFLOAT> &Mccf, const MultidimArray<RFLOAT> &Mpsi, const MultidimArray<RFLOAT> &Mstddev, int iref, int skip_side, std::vector<Peak> &peaks);
//
//	// Now prune the coordinates: within min_particle_distance: all peaks are the same cluster
//	// From each cluster, take the single peaks with the highest ccf
//	// If then, there is another peaks at a distance of at least min_particle_distance: take that one as well, and so forth...
//	void prunePeakClusters(std::vector<Peak> &peaks, int min_distance);
//
//
//	// Only keep those peaks that are at the given distance apart from each other
//	void removeTooCloselyNeighbouringPeaks(std::vector<Peak> &peaks, int min_distance);

};