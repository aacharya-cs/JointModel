//
//  JGPPFmodel.cpp
//  
//
//  Created by Ayan Acharya on 9/17/15.
//
//

#include "JGPPFmodel.h"

model::model(gsl_rng *rng, data Data, double epsilonlc)
{
	unsigned int *tmpvecvarint;
	unsigned int n,d,w,s,k,kprime,xindex,yindex,val, tmp;
	string line1, line2, dummy;
	vector<unsigned int> tmpptr;
	cout<<"intializing model .."<<endl;
	
	// model hyper-parameters initialized
	KB = Data.KB; KY = Data.KY; N = Data.N; D = Data.D; V = Data.V;	
	// for network
	azeroB = 1.0, bzeroB = 1.0, czeroB = sqrt(N), dzeroB = 1.0, ezeroB = 1.0, fzeroB = 1.0, gzeroB = 1.0; 
	hzeroB = 1.0, mzeroB = 1.0, nzeroB = 1.0, szeroB = 1.0, tzeroB = 1.0, cB = 1.0, gammaB = 1.0, xiB = 0.1;
	// for corpus
	azeroY = 1.0, bzeroY = 1.0, czeroY = sqrt(D), dzeroY = 1.0, ezeroY = 1.0, fzeroY = 1.0, gzeroY = 1.0; 
	hzeroY = 1.0, mzeroY = 1.0, nzeroY = 1.0, szeroY = 1.0, tzeroY = 1.0, cY = 1.0, gammaY = 1.0, xiY = 0.1;			
	epsilon = epsilonlc;   
	// for networK	
	phink  = mat(N,KB); phink.fill(1.0/KB); phinss = rowvec(N); phinss = sum(phink,1).t(); phikss = sum(phink,0), phikss2 = sum(phink%phink,0);
	rkB    = rowvec(KB); rkB.fill(1.0*gammaB/KB); akB = rowvec(KB); akB.fill(1.0); cn = rowvec(N, fill::ones);
	psiwk  = mat(V,KB); psiwk.fill(1.0/V); 			
	xndotk = mat(N,KB), ydotndotk = mat(N,KB); xk = rowvec(KB); 
	// for corpus	
	thetadk = mat(D,KY); thetadk.fill(1.0/KY), thetadss = rowvec(D,fill::zeros), thetakss = sum(thetadk,0);	
	rkY     = rowvec(KY); rkY.fill(1.0*gammaY/KY); cd = rowvec(D,fill::ones); akY = rowvec(KY); akY.fill(1.0);
	betawk  = mat(V,KY); betawk.fill(1.0/V); 	
	ydotwk  = mat(V,(KB+KY)); yddotk = mat(D,KY); yk  = rowvec(KB+KY);  	
	// for Z
	Zphikss = Data.Ndsz*phink;		
	// for debugging, display and further use; of no use otherwise
	phinkss  = mat(N,KB,fill::zeros); psiwkss  = mat(V,KB,fill::zeros); rkBss = rowvec(KB,fill::zeros);	
	thetadkss= mat(D,KY,fill::zeros); betawkss = mat(V,KY,fill::zeros); rkYss = rowvec(KY,fill::zeros);			
	cout<<"model initialization ends.."<<endl;	
	
	return;
};

void model::train(gsl_rng *rng, unsigned int CollectionITER, unsigned int BurninITER, data Data, unsigned int Option, unsigned int netOption)
{
	cout<<"Gibbs sampling iteration starts .."<< endl;
	double param1, param2, param11, param12, param21, param22, gammaBsum, gammaYsum, xiBparam1, xiBparam2, xiYparam1, xiYparam2;
	double rkBsum, rkYsum, rZphisum, ysum, lsum1, logpsum1, lsum2, logpsum2, sk, akYsum, akBsum, epsilonparam1, epsilonparam2;
	double tmp1, tmp2, tmp3;
	double *tmpvecparam, *tmpvecvardouble, tmpsum, tempy, tempss;
	unsigned int *tmpvecvarint; 
	unsigned int n, m, s, k, kBcount, d, w, xindex, yindex, val;
	string dummy; srand(time(NULL));
	sp_mat::const_iterator start,end,it,start2,end2,it2;
	sp_mat tmpsp,tmpsp2;
	
	akBsum = sum(akB); akYsum = sum(akY);
	// Gibbs sampling iteration starts
	for (int i=0; i< (CollectionITER + BurninITER); i++)
	{
		if (i==0 || (i+1)%100 == 0)
			cout<< "Iteration: "<<(i+1)<<endl;		
		
	    // reset few statistics first; O(NK)
	    xndotk.fill(0.0); xk.fill(0.0); ydotndotk.fill(0.0); yddotk.fill(0.0); ydotwk.fill(0.0); yk.fill(0.0); epsilonparam1 = 0.0; epsilonparam2 = 0.0;  
		// sampling starts
		// for network:: sampling of latent counts; O(SK)
		for (n=0; n<N; n++)
		{	
			tmpsp = Data.B.row(n); start = tmpsp.begin(); end = tmpsp.end();
			for(it = start; it != end; ++it)
			{
				m = it.col(); val = (*it); tmpsum = 0.0; tmpvecvarint = new unsigned int [KB]; tmpvecparam = new double [KB]; 				
				for (k=0; k<KB; k++)
				{
					tmp1 = rkB(k), tmp2 = phink(n,k), tmp3 = phink(m,k);
					*(tmpvecparam+k) = minguard(tmp1*tmp2*tmp3), tmpsum += *(tmpvecparam+k);
				}
				//normalization
				for (k=0; k<KB; k++)
					*(tmpvecparam+k) = *(tmpvecparam+k)/tmpsum;	
				if (netOption==1)	
					val = TruncPoisson(rng,tmpsum);				
				gsl_ran_multinomial(rng, KB, val, tmpvecparam, tmpvecvarint);
				// update sufficient statistics of x
				for (k=0; k<KB; k++)
				{	
					if(*(tmpvecvarint+k)>=UPPERLIMIT)
						*(tmpvecvarint+k) = UPPERLIMIT;
					tmp1 = (*(tmpvecvarint+k)); xndotk(n,k) += tmp1; xndotk(m,k) += tmp1; xk(k) += tmp1;
				}
				free(tmpvecvarint);	free(tmpvecparam);
			}
		}
		// for corpus:: sampling of latent counts; O(SK)
		for (d=0; d<D; d++)
		{
			tmpsp = Data.Y.row(d); start = tmpsp.begin(); end = tmpsp.end(); KBcount = KB*Data.Dnsz(d); tmpsp2 = Data.Zt.row(d);
			for(it = start; it != end; ++it)
			{
				w = it.col(); val = (*it); tmpsum = 0.0; 
				tmpvecvarint = new unsigned int [(KBcount+KY)]; tmpvecparam = new double [(KBcount+KY)]; kBcount = 0;
				for (k=0; k<KB; k++) // for network groups
				{
					start2 = tmpsp2.begin(); end2 = tmpsp2.end();
					for (it2 = start2; it2 != end2; ++it2)
					{
						n = it2.col(); tmp1 = rkB(k); tmp2 = phink(n,k); tmp3 = psiwk(w,k);
						*(tmpvecparam+kBcount) = (Option==0)? 0.0: minguard(epsilon*tmp1*tmp2*tmp3); tmpsum  += *(tmpvecparam+kBcount); kBcount += 1;							
					}	
				}				
				for (k=0; k<KY; k++) // for count data related groups
				{
					tmp1 = rkY(k); tmp2 = thetadk(d,k); tmp3 = betawk(w,k);
					*(tmpvecparam+(KBcount+k)) = minguard(tmp1*tmp2*tmp3); tmpsum += *(tmpvecparam+(KBcount+k));
				}
				//normalization
				for (k=0; k<(KBcount+KY); k++)
					*(tmpvecparam+k) = *(tmpvecparam+k)/tmpsum;
				gsl_ran_multinomial(rng, (KBcount+KY), val, tmpvecparam, tmpvecvarint);
				// update sufficient statistics of y
				kBcount = 0.0;
				for (k=0; k<KB; k++) // for network groups
				{
					start2 = tmpsp2.begin(); end2 = tmpsp2.end();
					for (it2 = start2; it2 != end2; ++it2)
						n = it2.col(), tmp1 = *(tmpvecvarint+kBcount), ydotndotk(n,k) += tmp1, ydotwk(w,k) += tmp1, yk(k) += tmp1, epsilonparam1 +=tmp1, kBcount += 1;	
				}
				for (k=0; k<KY; k++) // for count data related groups
					tmp1 = *(tmpvecvarint+KBcount+k), yddotk(d,k) += tmp1, ydotwk(w,(KB+k)) += tmp1, yk(KB+k) += tmp1;
				free(tmpvecvarint);	free(tmpvecparam);
			}	
		}
		// for network:: sampling of phink and cn; O(NK+N)
		for (n=0; n<N; n++)
		{
			// reset the statistics about phi 
			phinss(n) = 0.0;
			for (k=0; k<KB; k++)
			{
				// sampling of phink													
				param1 = azeroB + xndotk(n,k) + ydotndotk(n,k); phikss(k) -= phink(n,k); Zphikss(k) -= Data.Ndsz(n)*phink(n,k); phikss2(k) -= pow(phink(n,k),2);						
				param2 = (Option==0)?(1.0/(cn(n) + rkB(k)*phikss(k))):(1.0/(cn(n) + rkB(k)*(phikss(k) + epsilon*Data.Ndsz(n))));
				phink(n,k) = minguard(gsl_ran_gamma(rng,param1,param2));	
				if(i>=BurninITER)
					phinkss(n,k) += phink(n,k)/CollectionITER;
				//update sufficient statistics for phi 
				phinss(n) += phink(n,k); phikss(k) += phink(n,k); phikss2(k) += pow(phink(n,k),2); Zphikss(k) += Data.Ndsz(n)*phink(n,k);	
			}
			// sampling of cn	
			param1 = czeroB + KB*azeroB; param2  = 1.0/(dzeroB + phinss(n));
			cn(n)  = minguard(gsl_ran_gamma(rng,param1,param2));
		}	
		// for network:: sampling of rk, lk, gammak; O(3*K)
		rkBsum = 0.0; lsum1 = 0.0; logpsum1 = 0.0; xiBparam1 = 0.0; xiBparam2 = 0.0;
		for (k=0; k<KB; k++)
		{	
			// sample rkB
			sk = (pow(phikss(k),2) - phikss2(k))/2; param1 = 1.0*gammaB/KB + xk(k) + yk(k); param2 = (Option==0)?(1.0/(cB + sk)):(1.0/(cB + sk + epsilon*Zphikss(k)));
			rkB(k) = minguard(gsl_ran_gamma(rng, param1, param2)); rkBsum += rkB(k); 
			if(i>=BurninITER)
				rkBss(k) += rkB(k)/CollectionITER;										
			// sample lk's for the updates of gammaB	
			lsum1 += sampleCRT(rng,yk(k),1.0*gammaB/KB); logpsum1 += (Option==0)?(logguard(1.0 + 1.0*sk/cB)):(logguard(1.0 + (sk + epsilon*Zphikss(k))/cB));	
			epsilonparam2 += rkB(k)*Zphikss(k);					
			if(Option==1) // sample psiwk only for the joint model, no need to sample for the disjoint model
			{
				// sample psiwk			
				tmpvecvardouble = new double [V]; tmpvecparam = new double [V];	
				for(w=0; w<V; w++)  // get the parameters for the Dirichlet distribution
					*(tmpvecparam+w) = 1.0*xiB +  ydotwk(w,k);
				gsl_ran_dirichlet (rng, V, tmpvecparam, tmpvecvardouble);
				for (w=0; w<V; w++)
				{
					psiwk(w,k) = minguard(*(tmpvecvardouble+w)); 		
					if(i>=BurninITER)
						psiwkss(w,k) += psiwk(w,k)/CollectionITER;	
					// sample the CRT random variables for the updates of the Dirichlet hyper-parameter						
					xiBparam1 += sampleCRT(rng,ydotwk(w,k),xiB); 										
				}
				free(tmpvecvardouble); free(tmpvecparam);
				// sample the beta random variables for the updates of the Dirichlet hyper-parameter 					
				xiBparam2 += logguard(1.0 - gsl_ran_beta(rng,yk(k),V*xiB));		 
			}													
		}
		// sample gammaB						
		param1 = ezeroB + lsum1; param2 = 1.0/(fzeroB + 1.0*logpsum1/KB);				
		gammaB = minguard(gsl_ran_gamma(rng, param1, param2)); 											
		// for corpus:: sampling of rk, lk, gammak, thetadk and betawk; O(DK+VK+K)	
		lsum2 = 0.0; logpsum2 = 0.0; rkYsum = 0.0; xiYparam1 = 0.0; xiYparam2 = 0.0; thetadss.fill(0.0);	
		for (k=0; k<KY; k++)
		{	
			// sample thetadk					
			thetakss(k) = 0.0; lsum1 = 0.0; logpsum1 = 0.0; 
			for (d=0; d<D; d++)
			{
				param1       = azeroY + yddotk(d,k); param2 = 1.0/(cd(d) + rkY(k));
				thetadk(d,k) = minguard(gsl_ran_gamma(rng,param1,param2)); thetakss(k) += thetadk(d,k); thetadss(d) += thetadk(d,k);
				if(i>=BurninITER)
					thetadkss(d,k) += thetadk(d,k)/CollectionITER;
				// sample ldky's for the updates of akY's
				lsum1 += sampleCRT(rng,yddotk(d,k),akY(k)); logpsum1 += logguard(1.0 + rkY(k)/cd(d)); 	 					
			}						
			// sample betawk	
			tmpvecvardouble = new double [V]; tmpvecparam = new double [V];			
			for(w=0; w<V; w++)  // get the parameters for the Dirichlet distribution
				*(tmpvecparam+w) = 1.0*xiY + ydotwk(w,(k+KB));
			gsl_ran_dirichlet (rng, V, tmpvecparam, tmpvecvardouble);
			for (w=0; w<V; w++)
			{
				betawk(w,k) = minguard(*(tmpvecvardouble+w)); 		
				if(i>=BurninITER)
					betawkss(w,k) += betawk(w,k)/CollectionITER;
				// sample the CRT random variables for the updates of the Dirichlet hyper-parameter						
				xiYparam1 += sampleCRT(rng,ydotwk(w,(k+KB)),xiY); 	
			}
			free(tmpvecvardouble); free(tmpvecparam);
			// sample rkY
			param1 = 1.0*gammaY/KY + yk(KB+k); param2 = 1.0/(cY + thetakss(k));				
			rkY(k) = minguard(gsl_ran_gamma(rng, param1, param2)); rkYsum += rkY(k); 	
			if(i>=BurninITER)
				rkYss(k) += rkY(k)/CollectionITER;	
			// sample lk's for the updates of gammaY	
			lsum2 += sampleCRT(rng,yk(k),1.0*gammaY/KY); logpsum2 += logguard(1.0 + thetakss(k)/cY);	
			// sample the beta random variables for the updates of the Dirichlet hyper-parameter 
			xiYparam2 += logguard(1.0 - gsl_ran_beta(rng,yk(KB+k),V*xiY));		
		}	
		// sample gammaY's
		param1 = ezeroY + lsum2; param2 = 1.0/(fzeroY + 1.0*logpsum2/KY);				
		gammaY = minguard(gsl_ran_gamma(rng, param1, param2)); 									
		// for corpus:: sample cd; O(D)
		for (d=0; d<D; d++)
		{
			param1 = czeroY + KY*azeroY; param2  = 1.0/(dzeroY + thetadss(d));
			cd(d)  = minguard(gsl_ran_gamma(rng,param1,param2));
		}		
		// sample global variable
		param1 = gzeroB + gammaB; param2  = 1.0/(hzeroB + rkBsum);		
		cB     = minguard(gsl_ran_gamma(rng, param1, param2));		
		param1 = gzeroY + gammaY; param2  = 1.0/(hzeroY + rkYsum);		
		cY     = minguard(gsl_ran_gamma(rng, param1, param2));
		//sample xiB
        param1 = szeroB + xiBparam1; param2  = 1.0/(tzeroB - V*xiBparam2);
        //xiB    = minguard(gsl_ran_gamma(rng, param1, param2));
		//sample xiY
        param1 = szeroY + xiYparam1; param2  = 1.0/(tzeroY - V*xiYparam2);
        //xiY    = minguard(gsl_ran_gamma(rng, param1, param2));  
        param1  = szeroY + epsilonparam1; param2  = 1.0/(tzeroY + epsilonparam2);        
		//epsilon = minguard(gsl_ran_gamma(rng, param1, param2));
	}
	// end of Gibbs sampling iteration loop
	cout<<"Gibbs sampling terminates .."<< endl;		
	return;

};

void model::printresults()
{
	unsigned int k,n,d,w;
	
    cout<<"printing results .."<< endl;  
    cout.precision(10);	
    
    // for corpus::
	ofstream opfile1("rkY.txt");
	for (k=0; k<KY; k++)
		opfile1<<rkYss(k)<<endl;
	opfile1.close();	
	
	ofstream opfile2("thetadk.txt");		
	for (k=0; k<KY; k++)
	{
		for(d=0;d<D;d++)
			opfile2<<thetadkss(d,k)<<"\t";
		opfile2<<endl;
	}	
	opfile2.close();
	
	ofstream opfile3("betawk.txt");		
	for (k=0; k<KY; k++)
	{
		for(w=0;w<V;w++)
			opfile3<< betawkss(w,k)<<"\t";
		opfile3<<endl;
	}	
	opfile3.close();
	
	ofstream opfile4("psiwk.txt");		
	for (k=0; k<KB; k++)
	{
		for(w=0;w<V;w++)
			opfile4<<psiwkss(w,k)<<"\t";
		opfile4<<endl;
	}	
	opfile4.close();	
	
    // for network::
	ofstream opfile5("rkB.txt");
	for (k=0; k<KB; k++)
		opfile5<<rkBss(k)<<endl;
	opfile5.close();	
	
	ofstream opfile6("phink.txt");		
	for (k=0; k<KB; k++)
	{
		for(n=0;n<N;n++)
			opfile6<<phinkss(n,k)<<"\t";
		opfile6<<endl;
	}	
	opfile6.close();				
		
	cout<<"printing of results done.."<< endl;	
	return;

};

