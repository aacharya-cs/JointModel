// @ Ayan Acharya
// Date: 02/11/2015
// Code for Joint Gamma Process Poisson Factorization (J-GPPF); this is the most up-to-date implementation


#include "JGPPFdata.h"
#include "JGPPFmodel.h"
#include <time.h>

int main(int argc, char **argv)
{
	string line, trFilename1,trFilename2,trZFilename,predFilename1,predFilename2;
	unsigned int aRand, BurninITER, CollectionITER, KB, KY, Option, netOption;
	double epsilon;
	const gsl_rng_type *Temp;
	
	// gsl set-up
	gsl_rng_env_setup(); Temp = gsl_rng_default; gsl_rng *rng = gsl_rng_alloc(Temp); srand (time(NULL));
	// provides different seeds for different runs
	aRand = rand() % 10 + 1; gsl_rng_set (rng, aRand);

	trFilename1  = argv[1]; trFilename2  = argv[2]; trZFilename  = argv[3]; predFilename1 = argv[4]; predFilename2 = argv[5]; 
	KB = atoi(argv[6]); KY = atoi(argv[7]); BurninITER  = atoi(argv[8]); CollectionITER = atoi(argv[9]); Option = atoi(argv[10]);  
	epsilon = atof(argv[11]); netOption = atoi(argv[12]); 
		
	// data loading
	data Data(trFilename1,trFilename2,trZFilename,predFilename1,predFilename2,KB,KY);
	// model initialization
	model JGPPF(rng,Data,epsilon);
    // Gibbs sampling for C-GPPF
	JGPPF.train(rng,CollectionITER,BurninITER,Data,Option,netOption);
    cout<<"printing of results starts"<<endl;	
	gsl_rng_free (rng);
	// print results for J-GPPF
	JGPPF.printresults();
	cout<<"program terminates .."<< endl;
	
	return 0;
}
