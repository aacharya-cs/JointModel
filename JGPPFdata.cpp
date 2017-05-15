//
//  JGPPFdata.cpp
//  
//
//  Created by Ayan Acharya on 9/17/15.
//
//

#include "JGPPFdata.h"

data::data(string trFileName1, string trFileName2, string trZFileName, string predFileName1, string predFileName2, unsigned int kB, unsigned int kY)
{
	string line1,line2,line3,line4,line5;
	ifstream trFile1(trFileName1); ifstream trFile2(trFileName2); ifstream trZFile(trZFileName); 
	ifstream predFile1(predFileName1); ifstream predFile2(predFileName2);
	int count,tmp1,tmp2,tmp3,tmp4;
	mSN = 0; mSY = 0; KB = kB; KY = kY;

    // for network 	
	cout<<"loading of network data starts .."<<endl;
	getline(trFile1, line1); istringstream iss1(line1);
	iss1 >> N >> SN; B = sp_mat(N,N);  
    while (getline(trFile1, line1))
    {
        std::istringstream iss1(line1);
        if (!(iss1 >> tmp1 >> tmp2 >> tmp4))
            break;
		B(tmp1,tmp2) = tmp4;
    }	
	cout<<"loading of network data ends .."<<endl;			
	
	// for corpus
	cout<<"loading of corpus data starts .."<<endl;		
	getline(trFile2, line2); istringstream iss2(line2);
	iss2 >> D >> V >> SY; Y = sp_mat(D,V);
    while (getline(trFile2, line2))
    {
        std::istringstream iss2(line2);
        if (!(iss2 >> tmp1 >> tmp2 >> tmp4))
            break;
		Y(tmp1,tmp2) = tmp4;
    }	
    cout<<"loading of corpus data ends .."<<endl;
	
	// loading of Z
	cout<<"loading of Z starts .."<<endl;	
	Ndsz = rowvec(N, fill::zeros); Dnsz = rowvec(D, fill::zeros); Zt = sp_mat(D,N);
	while (getline(trZFile, line3))
	{
		istringstream iss3(line3);
		if (!(iss3 >> tmp1 >> tmp2))
			break;
		Zt(tmp2,tmp1) = 1.0; Ndsz(tmp1) += 1; Dnsz(tmp2) += 1;			
	}
	cout<<"loading of Z ends .."<<endl;	
	
	// reading of missing data for network
	cout<<"loading of prediction file 1 starts .."<<endl;	
	getline(predFile1, line4); istringstream iss4(line4); mB = sp_mat(N,N);
	iss4 >> tmp2 >> mSN;
	if(mSN>0)
	{		
		while (getline(predFile1, line4))
		{
			istringstream iss4(line4);
			if (!(iss4 >> tmp1 >> tmp2 >> tmp4))
				break;
			mB(tmp1,tmp2) = tmp4;
		}
	}		 		
		
	// reading of missing data for corpus
	cout<<"loading of prediction file 2 starts .."<<endl;	
	getline(predFile2, line5); istringstream iss5(line5); mY = sp_mat(D,V);
	iss5 >> tmp2 >> tmp4 >> mSY;
	if(mSY>0)
	{		
		while (getline(predFile2, line5))
		{
			istringstream iss5(line5);
			if (!(iss5 >> tmp1 >> tmp2 >> tmp4))
				break;
			mY(tmp1,tmp2) = tmp4;
		}
	}		
	cout<<"loading of data ends .."<<endl;
		
	return;
};
