//
//  JGPPFmodel.hpp
//  
//
//  Created by Ayan Acharya on 9/17/15.
//
//

#ifndef JGPPFmodel_h
#define JGPPFmodel_h

#include "mathutils.h"
#include "samplers.h"
#include "JGPPFdata.h"

class model
{
	unsigned int N,D,V,KB,KBcount,KY,SN,CollectionITER,BurninITER;
	double fzero,gzero,gammaB,gammaY,xiY,xiB;
	double azeroB,bzeroB,czeroB,dzeroB,ezeroB,fzeroB,gzeroB,hzeroB,mzeroB,nzeroB,szeroB,tzeroB;
	double azeroY,bzeroY,czeroY,dzeroY,ezeroY,fzeroY,gzeroY,hzeroY,mzeroY,nzeroY,szeroY,tzeroY;	
	double cB,cY,epsilon;
	mat phink,phinkss,thetadk,thetadkss,psiwk,psiwkss,betawk,betawkss,ydotwk,yddotk,ydotndotk,xndotk;
	rowvec phikss,phikss2,Zphikss,phinss,yk,xk,thetakss,thetadss,psiwss,betawss,pdsum,qwsum,akB,cn,akY,cd,rkB,rkY,rkBss,rkYss;
	
public:

	model(gsl_rng*,data,double);
	void train(gsl_rng*,unsigned int,unsigned int,data,unsigned int,unsigned int);
	void printresults();
};

#endif 
