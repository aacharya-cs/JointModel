#!/usr/bin/env python
## this implementation is the most up-to-date one and uses Dirichlet distribution for the topics
########################################################################
## for running with virtualenv:
## source /home/ayan/venv/bin/activate 
## source /home/aacharya/Documents/venv/bin/activate 
## actual command:
## python exploreJGPPF.py 20 20 30 30 45 1000 500 0.25 1.0 1
## python exploreJGPPF.py 20 20 30 30 45 1000 1000 0 0 1
## options: 
## 1. KB: maximum number of latent factors in network 
## 2. KY: maximum number of latent factors in count matrix 
## 3. N: number of users in the network
## 4. D: number of documents in the count matrix
## 5. V: number of words in the vocabulary in the count matrix
## 6: burnin: number of burn-in iterartions for Gibbs sampling
## 7: collection: number of iterations for collections of samples
## 8: p: fraction of data held-out from network
## 9: epsilon: contribution from network
## 10: netoption: 1 for binary network, 2 for count-valued network 
########################################################################

import os
import sys
import pickle
import numpy as np
from pylab import *
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import metrics
import collections

def holdoutentries(A,p):
	N        = A.shape[0]
	A        = A - np.tril(A)
	B        = np.arange(A.shape[0]*A.shape[1]).reshape(A.shape)   
	mask     = np.squeeze(B[np.mask_indices(A.shape[0], np.triu, 1)])             ## linear indices of entries other than the lower triangular part; interesting region
	if p==0:  ## don't hold out anything
	    idx  = np.squeeze(np.nonzero(np.ravel(A, order='C')))        ## indices of ones
	    idxh = np.empty([0])
	elif p>=1.0:
	    raise NameError('set p<1.0')
	else:
		C    = np.zeros(N-1)
		C[0] = 1 
		for n in range(1,N-1):      
			C[n] = C[n-1] + (N+1)	## small hack to ensure that there is atleast one link per row 		
		idxones     = np.squeeze(np.nonzero(A.flat))                                           ## indices of ones from the interesting region
		totmis      = np.floor(p*len(idxones)) - A.shape[0]
		if(totmis<=0):
			## don't hold out anything
		    idx  = np.squeeze(np.nonzero(np.ravel(A, order='C')))        ## indices of ones
		    idxh = np.empty([0])
		else:
			idxzeros    = np.array(list(set(mask) - set(idxones)))                                 ## indices of zeros
			idxhlinks   = np.array(list(set(idxones[SelRandomVec(len(idxones),totmis)]) - set(C))) ## held-out ones
			idxhnolinks = idxzeros[SelRandomVec(len(idxzeros),totmis)] 				               ## held-out zeros
			values      = np.ones(len(idxhlinks)) 
			dictionary1 = dict(zip(idxhlinks, values))
			values      = np.zeros(len(idxhnolinks)) 
			dictionary2 = dict(zip(idxhnolinks, values))       
			idxh        = dict(dictionary1.items() + dictionary2.items())
			idxh        = collections.OrderedDict(sorted(idxh.items()))
			idx         = np.sort(np.array(list(set(set(mask) - set(idxzeros)) - set(idxhlinks))))   	
	return (idx,idxh)
    
def SelRandomVec(N,k):
## N: size of vector, k: number of elements to be chosen
## if k is zero, return the first element
	if k==N:
		ind = np.linspace(0,(N-1), N)
	elif k==0:
		ind = 1    
	else:
		permvect = np.random.permutation(N)
		ind      = permvect[0:(k-1)]
		ind      = np.sort(ind) 
		ind      = np.array(ind,dtype=np.int32)
	ind.tolist()	
	return (ind)	

KB = int(sys.argv[1])
KY = int(sys.argv[2])
N = int(sys.argv[3])
D = int(sys.argv[4])
V = int(sys.argv[5])
BurnIn = int(sys.argv[6])
Collection =  int(sys.argv[7])
p = float(sys.argv[8])
epsilon = float(sys.argv[9])
netOption = int(sys.argv[10])

os.system('clear') 
os.system('rm -rf *.txt')
os.system('rm -rf jgppf')

mm = np.ones((N,N))
mm = np.rint(mm*1)
tmp1 = mm[0:2*N/6,0:2*N/6]
tmp2 = mm[0:4*N/6,0:4*N/6]
M1 = np.concatenate((tmp1,np.zeros((2*N/6,4*N/6))),axis=1)
M2 = np.concatenate((np.zeros((4*N/6,2*N/6)),tmp2),axis=1)
M  = np.concatenate((M1,M2),axis=0)
M  = M-np.tril(M);
M[2*N/6-1,2*N/6] = 1
Mnetwork = M

if 1:
	[idx,idxh] = holdoutentries(Mnetwork,p)
	pickle.dump(idx, open( "idx.p", "wb" ))
	pickle.dump(idxh, open( "idxh.p", "wb" ))
else:
	idx  = pickle.load( open( "idx.p", "rb" ))
	idxh = pickle.load( open( "idxh.p", "rb" ))
	
f = open('trfile1.txt','w')
writestring = str(N)+'\t'+str(len(idx))+'\n' 
f.write(writestring)
for indices in idx:
	i = int(np.floor(indices/N))
	j = int((indices - i*N))
	if netOption==1:
		writestring = str(i)+'\t'+str(j)+'\t'+'1\n' 
	else:
		writestring = str(i)+'\t'+str(j)+'\t'+'5\n' 		
	f.write(writestring)
f.close()

if D==N:
	Z = np.diag(np.ones(N))
else:
	if D==2*N:
		Z = np.zeros((N,D))
		for n in range(0,N):
			Z[n,2*n]   = 1
			Z[n,2*n+1] = 1
	else:		
		if N==2*D:
			Z = np.zeros((N,D))
			for d in range(0,D):
				Z[2*d,d]   = 1
				Z[2*d+1,d] = 1		
		else:
			raise NameError('only D==2N and N==2D are supported')

f = open('trzfile.txt','w')
for n in range(0,N):
	indices  = np.squeeze(np.nonzero(Z[n,:]))
	if indices.size==1:
		writestring = str(n)+'\t'+str(indices)+'\n' 
		f.write(writestring)
	else:	
		for index in indices:
			writestring = str(n)+'\t'+str(index)+'\n' 
			f.write(writestring)	
f.close()
 
f=open('predfile1.txt','w')
Mhidden = [] 
writestring = str(N)+'\t'+str(len(idxh))+'\n' 
f.write(writestring)
for indices in idxh:
	val = int(idxh[indices])
	i = int(np.floor(indices/N))
	j = int((indices - i*N))
	Mnetwork[i,j] = -1
	writestring = str(i)+'\t'+str(j)+'\t'+str(val)+'\n'
	f.write(writestring)
	Mhidden.append(val)	
f.close()

f=open('predfile2.txt','w')
f.close()

Mhidden = np.array(Mhidden)

mm = np.ones((D,V))
mm = np.rint(mm*5)
tmp1 = mm[0:2*D/6,0:2*V/6]
tmp2 = mm[0:4*D/6,0:4*V/6]
M1 = np.concatenate((tmp1,np.zeros((2*D/6,4*V/6))),axis=1)
M2 = np.concatenate((np.zeros((4*D/6,2*V/6)),tmp2),axis=1)
M  = np.concatenate((M1,M2),axis=0)
Mcount = M

f=open('trfile2.txt','w')
writestring = str(D)+'\t'+str(V)+'\t'+str(np.count_nonzero(Mcount))+'\n' 
f.write(writestring)
for i in range(0,D):
	for j in range(0,V):
		if(Mcount[i,j]>0):
			writestring = str(i)+'\t'+str(j)+'\t'+str(Mcount[i,j])+'\n' 
			f.write(writestring)
f.close()

# for experiments with real data explicitly provide "trfile1.txt", "trfile2.txt" and "predfile.txt" 
execstring  = 'g++ -std=c++0x -o JGPPF JGPPF.cpp JGPPFmodel.cpp JGPPFdata.cpp samplers.cpp mathutils.cpp -larmadillo -llapack -lblas `gsl-config --cflags --libs`'
print "compiling J-GPPF .."
os.system(execstring)
execstring  = './JGPPF trfile1.txt trfile2.txt trzfile.txt predfile1.txt predfile2.txt '+str(KB)+' '+str(KY)+' '+str(BurnIn)+' '+str(Collection)+' 1 '+str(epsilon)+' '+str(netOption)
print "running J-GPPF .."
os.system(execstring)

rkB     = np.zeros(KB)
rkY     = np.zeros(KY)
phink   = np.zeros((KB,N))
thetadk = np.zeros((KY,D))
betawk  = np.zeros((KY,V))
psiwk   = np.zeros((KB,V))

print "reading results from J-GPPF .."

f1 = open('rkB.txt', 'r')
count = 0
for line in f1:
	rkB[count] = float(line)
	count = count + 1
f1.close()

f1 = open('rkY.txt', 'r')
count = 0
for line in f1:
	rkY[count] = float(line)
	count = count + 1
f1.close()

f1 = open('phink.txt', 'r')
count = 0
for line in f1:
	vals = line.split('\t')
	for n in range(0,N):
		phink[count,n] = float(vals[n])
	count = count + 1
f1.close()
	
f1 = open('thetadk.txt', 'r')
count = 0
for line in f1:
	vals = line.split('\t')
	for d in range(0,D):
		thetadk[count,d] = float(vals[d])
	count = count + 1
f1.close()

f1 = open('betawk.txt', 'r')
count = 0
for line in f1:
	vals = line.split('\t')
	for w in range(0,V):
		betawk[count,w] = float(vals[w])
	count = count + 1
f1.close()

f1 = open('psiwk.txt', 'r')
count = 0
for line in f1:
	vals = line.split('\t')
	for w in range(0,V):
		psiwk[count,w] = float(vals[w])
	count = count + 1
f1.close()

if idxh:
	## prediction on held-out entires for J-GPPF
	predJGPPF = []
	for indices in idxh:
		i = int(np.floor(indices/N))
		j = int((indices - i*N))
		val = np.sum(np.multiply(np.multiply(rkB,phink[:,i]),phink[:,j]))
		predJGPPF.append(val)	
	predJGPPF = np.array(predJGPPF) 
	## calculate AUC from J-GPPF
	fprJGPPF, tprJGPPF, thresholds = metrics.roc_curve(Mhidden, predJGPPF, pos_label=1)
	roc_aucJGPPF = metrics.auc(fprJGPPF, tprJGPPF)

print "reading of results for J-GPPF done .."

os.system('rm -rf rkB.txt phink.txt')

print "N-GPPF starts .."
execstring  = './JGPPF trfile1.txt trfile2.txt trzfile.txt predfile1.txt predfile2.txt '+str(KB)+' '+str(KY)+' '+str(BurnIn)+' '+str(Collection)+' 0'+' '+str(epsilon)+' '+str(netOption)
print "running N-GPPF .."
os.system(execstring)

rkN       = np.zeros(KB)
phinkGPPF = np.zeros((KB,N))

print "reading results from N-GPPF .."

f1 = open('rkB.txt', 'r')
count = 0
for line in f1:
	rkN[count] = float(line)
	count = count + 1
f1.close()

f1 = open('phink.txt', 'r')
count = 0
for line in f1:
	vals = line.split('\t')
	for n in range(0,N):
		phinkGPPF[count,n] = float(vals[n])
	count = count + 1
f1.close()

if idxh:
	## prediction on held-out entires for N-GPPF
	predNGPPF = []
	for indices in idxh:
		i = int(np.floor(indices/N))
		j = int((indices - i*N))
		val = np.sum(np.multiply(np.multiply(rkN,phinkGPPF[:,i]),phinkGPPF[:,j]))
		predNGPPF.append(val)	
	predNGPPF = np.array(predNGPPF)
	## calculate AUC from N-GPPF
	fprNGPPF, tprNGPPF, thresholdsNGPPF = metrics.roc_curve(Mhidden, predNGPPF, pos_label=1)
	roc_aucNGPPF = metrics.auc(fprNGPPF, tprNGPPF)


print "reading of results for N-GPPF done .."

## estimated network
AssignmentN = np.dot(phink.transpose(),np.diag(np.sqrt(rkB)))
MestN       = np.dot(AssignmentN,phink)

## estimated network from N-GPPF
AssignmentNGPPF = np.dot(phinkGPPF.transpose(),np.diag(np.sqrt(rkN)))
MestNGPPF       = np.dot(AssignmentNGPPF,phinkGPPF)

## estimated count matrix
A1  = np.dot(phink,Z)
A1  = np.dot(A1.transpose(),np.diag(np.sqrt(rkB*epsilon)))
A2  = np.dot(thetadk.transpose(),np.diag(np.sqrt(rkY)))
Assignmentcount = np.concatenate((A1,A2),axis=1)
A11 = np.dot(psiwk.transpose(),np.diag(np.sqrt(rkB*epsilon)))
A1  = np.dot(A1,A11.transpose())
A21 = np.dot(betawk.transpose(),np.diag(np.sqrt(rkY)))
A2  = np.dot(A2,A21.transpose())
Mestcount = (A1+A2)

## display results
winsz = 5
sp = plt.subplot(4,winsz,1)
plt.imshow(Mnetwork + Mnetwork.transpose())
plt.title("original network")
colorbar()

#sp = plt.subplot(4,winsz,15)
#rkBind = np.argsort(rkB)[::-1]
#hardassign = np.argmax(AssignmentN, axis=1)
#AssignmentNN = np.zeros((N,KB))
#phinkNN = np.zeros((KB,N))
#count = 0 
#for k in rkBind:	
#	indices  = np.squeeze(np.nonzero(hardassign==k))
#	if(indices.size>0):
#		for ind in indices:
#			AssignmentNN[count,:] = AssignmentN[ind,:]
#			phinkNN[:,count] = phink[:,ind]
#			count = count+1
#MestNN = np.dot(AssignmentNN,phinkNN)	
#plt.imshow(MestNN)
#plt.title("estimated network")
#colorbar()

sp = plt.subplot(4,winsz,2)	
plt.imshow(1-np.exp(-MestN))
plt.title("estimated network: J-GPPF")
colorbar()

sp = plt.subplot(4,winsz,3)	
plt.imshow(AssignmentN)
plt.title("assignment of users: JGPPF")
colorbar()

sp = plt.subplot(4,winsz,4)	
x  = np.arange(1,KB+1)
plt.stem(x,rkB)
#plt.stem(x,rkB.transpose()/np.sum(rkB))
plt.title("normalized values of rkB: JGPPF")

sp = plt.subplot(4,winsz,5)
plt.imshow(Mcount)
plt.title("original count matrix")
colorbar()

sp = plt.subplot(4,winsz,6)	
plt.imshow(Mestcount)
plt.title("estimated count matrix: JGPPF")
colorbar()

sp = plt.subplot(4,winsz,7)	
plt.imshow(Assignmentcount)
plt.title("assignment of rows in count matrix")
colorbar()

sp = plt.subplot(4,winsz,8)	
x  = np.arange(1,KB+KY+1)
rk = np.concatenate([rkB/np.sum(rkB),rkY/np.sum(rkY)])
plt.stem(x,rk)
#plt.stem(x,rk.transpose()/np.sum(rk))
plt.title("normalized values of rkB,rkY")

sp = plt.subplot(4,winsz,9)	
plt.imshow(1-np.exp(-MestNGPPF))
plt.title("estimated network: NGPPF")
colorbar()

sp = plt.subplot(4,winsz,10)	
plt.imshow(AssignmentNGPPF)
plt.title("assignment of users: NGPPF")
colorbar()

sp = plt.subplot(4,winsz,11)	
x  = np.arange(1,KB+1)
plt.stem(x,rkN.transpose()/np.sum(rkN))
plt.title("normalized values of rkN: NGPPF")

sp = plt.subplot(4,winsz,12)	
plt.imshow(betawk.transpose())
plt.title("betawk")
colorbar()

sp = plt.subplot(4,winsz,13)	
plt.imshow(psiwk.transpose())
plt.title("psiwk")
colorbar()


#sp = plt.subplot(4,winsz,16)
#rkNind = np.argsort(rkN)[::-1]
#hardassign = np.argmax(AssignmentNGPPF, axis=1)
#AssignmentNN = np.zeros((N,KB))
#phinkNN = np.zeros((KB,N))
#count = 0 
#for k in rkNind:	
#	indices  = np.squeeze(np.nonzero(hardassign==k))
	#if(indices.size>0):
	#	for ind in indices:
	#		AssignmentNN[count,:] = AssignmentNGPPF[ind,:]
	#		phinkNN[:,count] = phinkGPPF[:,ind]
	#		count = count+1
#MestNN = np.dot(AssignmentNN,phinkNN)	
#plt.imshow(MestNN)
#plt.title("estimated network")
#colorbar()

#sp = plt.subplot(4,winsz,12)	
#plt.imshow(AssignmentNN)
#plt.title("assignment of users: NGPPF")
#colorbar()

if idxh:
	sp = plt.subplot(4,winsz,13)	
	plt.plot(fprJGPPF, tprJGPPF, label='area = %0.2f' % roc_aucJGPPF)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC for JGPPF')
	plt.legend(loc="lower right")
	plt.title("ROC curve for JGPPF")
	
	sp = plt.subplot(4,winsz,14)	
	plt.plot(fprNGPPF, tprNGPPF, label='area = %0.2f' % roc_aucNGPPF)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC for NGPPF')
	plt.legend(loc="lower right")
	plt.title("ROC curve for NGPPF")

plt.show()	
print "reading of results done .."
