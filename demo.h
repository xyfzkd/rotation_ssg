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