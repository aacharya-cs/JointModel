//
//  samplers.h
//  
//
//  Created by Ayan Acharya on 9/16/15.
//
//

#ifndef ____samplers__
#define ____samplers__

#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <stdexcept>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <armadillo>
#include <boost/algorithm/string.hpp>
#include "mathutils.h"

using namespace std;
using namespace boost;
using namespace arma;

// Constants used in PG sampling
// The numerical accuracy of __PI will affect your distribution.
const double __PI = 3.141592653589793238462643383279502884197;
const double HALFPISQ = 0.5 * __PI * __PI;
const double TWOPISQ = 2 * __PI * __PI;
const double FOURPISQ = 4 * __PI * __PI;
const double __TRUNC = 0.64;
const double __TRUNC_RECIP = 1.0 / __TRUNC;
const double ABSLIMIT = 10.0;

double IG(const gsl_rng *rng, double mu, double lambda);
rowvec MVgaussian(const gsl_rng *rng, rowvec mu, mat Sigma, unsigned int K);
double PG(const gsl_rng *rng, double n, double z, unsigned int K);
double sampleCRT(gsl_rng *rng, const double m, const double gammazero);
unsigned int TruncPoisson(gsl_rng *rng, double lambda);

#endif /* defined(____samplers__) */

