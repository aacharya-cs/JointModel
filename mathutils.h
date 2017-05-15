//
//  mathutils.h
//  
//
//  Created by Ayan Acharya on 9/16/15.
//
//

#ifndef ____mathutils__
#define ____mathutils__

#include <math.h>
#include <armadillo>

#define LOWLIMIT 1e-15
#define UPPERLIMIT 10
#define UTH 50.0
#define LTH -2.0
using namespace arma;

double logguard(double m);
double minguard(double m);
double threshold(double m);
rowvec logguardvec(rowvec m);

#endif /* defined(____mathutils__) */
