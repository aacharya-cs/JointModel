//
//  JGPPFdata.h
//  
//
//  Created by Ayan Acharya on 9/17/15.
//
//

#ifndef JGPPFdata_hpp
#define JGPPFdata_hpp

#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <limits>
#include <array>
#include <armadillo>

using namespace std;
using namespace arma;

class data
{
	public:
		unsigned int SN,mSN,SY,mSY,KB,KY,D,V,N;
		sp_mat B,Y,Zt,mB,mY;
		rowvec Ndsz, Dnsz;
		data(string,string,string,string,string,unsigned int,unsigned int);
};

#endif /* JGPPFdata_hpp */
