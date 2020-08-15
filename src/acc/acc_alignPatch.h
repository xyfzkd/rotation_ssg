#include <cufftw.h>
#include "src/multidim_array.h"
#include "src/multidim_array.h"
#include "src/jaz/t_complex.h"

class CuFFT{
public:
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
