//
//  mathutils.cpp
//  
//
//  Created by Ayan Acharya on 9/16/15.
//
//
#include "mathutils.h"

double logguard(double m)
{
    // provides guard against log(0)
    if(m<LOWLIMIT)
        return log(LOWLIMIT);
    else
        return log(m);
};

double minguard(double m)
{
    // provides guard against number lower than LOWLIMIT
    if(m<LOWLIMIT)
        return (LOWLIMIT);
    else
        return m;
};

double threshold(double m)
{
    // thresholds values within a given range
    if(m>UTH)
        m = UTH;
    if(m<LTH)
        m = LTH;
    return m;
};

rowvec logguardvec(rowvec m)
{
    // provides guard against log(0)
    for (int i=0;i<m.n_elem; i++) 
    {
		if(m(i)<LOWLIMIT)
			m(i) = log(LOWLIMIT);
		else
			m(i) = log(m(i));
	}
	return m;
};
