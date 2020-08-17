#include <cufftw.h>
#include "src/multidim_array.h"
#include "src/multidim_array.h"
#include "src/jaz/t_complex.h"

class CuFFT{
private:
    MultidimArray<fComplex>& src;
    MultidimArray<fComplex> src2;
    MultidimArray<float>& dest;
    int goodsize = 0;
    bool replan;
    std::vector<int> N(0);

    cufftComplex *host_comp_data, *device_comp_data;
    cufftReal    *host_real_data, *device_real_data;
public:
    /* cufft construction: src, dest and goodsize parameters */
    CuFFT(MultidimArray<fComplex>& s, MultidimArray<float>& d, int size);

    /* cufft construction without parameters, implemented outside the loop to keep
     * device parameters away freeing
     * */
    CuFFT();

    ~CuFFT();

    /* realization */
    void ifft();

    static void inverseFourierTransform(
            MultidimArray<fComplex>& src,
            MultidimArray<float>& dest);

    template<class T>
    static bool areSizesCompatible(
            const MultidimArray<T>& real,
            const MultidimArray<tComplex<T> >& complex)
    {
        return real.xdim == 2 * (complex.xdim - 1)
               && real.ydim == complex.ydim
               && real.zdim == complex.zdim
               && real.ndim == complex.ndim;
    }

    template<class T>
    static void resizeRealToMatch(
            MultidimArray<T>& real,
            const MultidimArray<tComplex<T> >& complex)
    {
        real.resizeNoCp(complex.ndim, complex.zdim, complex.ydim, 2 * (complex.xdim - 1));
    }

};
float diff(MultidimArray<float>& re1, MultidimArray<float>& re2);

void rand_comp(MultidimArray<fComplex>& s);

void print_comp_image(MultidimArray<fComplex>& s);

void print_real_image(MultidimArray<float>& s);
