//
//  samplers.cpp
//  
//
//  Created by Ayan Acharya on 9/16/15.
//
//

#include "samplers.h"

double IG(const gsl_rng *rng, double mu, double lambda)
{
    double v,y,x,test;
    v = gsl_ran_gaussian (rng, 1.0);  // sample from a normal distribution with a mean of 0 and 1 standard deviation
    y = v*v;
    x = mu + (mu*mu*y)/(2*lambda) - (mu/(2*lambda)) * sqrt(4*mu*lambda*y + mu*mu*y*y);
    test = gsl_ran_flat (rng, 0, 1);  // sample from a uniform distribution between 0 and 1
    if (test <= mu/(mu + x))
        return x;
    else
        return (mu*mu)/x;
}

rowvec MVgaussian(const gsl_rng *rng, rowvec mu, mat Sigma, unsigned int K)
{
    // sampler for multivariate normal distribution
    int k;
    // first sample K gaussian random variables
    rowvec samples = rowvec(K);
    for (k=0; k<K; k++)
		samples(k) = gsl_ran_gaussian(rng,1.0);
    rowvec result = mu + samples*chol(Sigma);
    return(result);
}

double PG(const gsl_rng *rng, double n, double z, uint32_t K)
{
    
    double x = 0;
    for (uint32_t k = 1; k <= K; ++k) {
        x += 1 / TWOPISQ * gsl_ran_gamma(rng, n, 1) /
        ((k - 0.5) * (k - 0.5) + z * z / FOURPISQ);
    }
    // Account for floating point bias
    double temp = std::max(std::abs(z / 2.0), DBL_MIN);
    double xmeanfull = std::tanh(temp) / (temp * 4);
    double xmeantruncate = 0;
    for (uint32_t k = 1; k <= K; ++k) {
        xmeantruncate += 1 / TWOPISQ / ((k - 0.5) * (k - 0.5) + z * z / FOURPISQ);
    }
    x = x * xmeanfull / xmeantruncate;
    return (x);
}


/*double PG(const gsl_rng *rng, double n, double z, unsigned int K)
{
    double x = 0;
    for (unsigned int k = 1; k <= K; ++k)
    {
        x += 1 / TWOPISQ * gsl_ran_gamma(rng, n, 1) / ((k - 0.5) * (k - 0.5) + z * z / FOURPISQ);
    }
    // Account for floating point bias
    double temp = max(abs(z / 2.0), DBL_MIN);
    double xmeanfull = tanh(temp) / (temp * 4);
    double xmeantruncate = 0;
    for (unsigned int k = 1; k <= K; ++k)
    {
        xmeantruncate += 1 / TWOPISQ / ((k - 0.5) * (k - 0.5) + z * z / FOURPISQ);
    }
    x = x * xmeanfull / xmeantruncate;
    return (x);
}*/

double sampleCRT(gsl_rng *rng, const double m, const double gammazero)
{
    double sum = 0, bparam;
    for (int i=0; i<m; i++)
    {
        bparam = gammazero/(i + gammazero);
        if(gsl_rng_uniform (rng)<=bparam)
            sum = sum + 1;
    }
    return sum;
};

unsigned int TruncPoisson(gsl_rng *rng, double lambda)
{
    unsigned int m;
    double PMF = 1.0, prob;
    
    if(lambda>=1)
    {
        while(1)
        {
            m = gsl_ran_poisson (rng, lambda);
            if(m>0)
                break;
        }
    }
    else
    {
        m = 1;
        if(lambda<=1e-6)
            lambda = 1e-6;
        while(1)
        {
            prob = pow(lambda,m)*exp(-lambda)/(m*(1-exp(-lambda)));
            if (prob/PMF>gsl_rng_uniform (rng))        
                break;
            PMF = PMF-prob;
            m = m+1;
        }
    }
    return m;
};

