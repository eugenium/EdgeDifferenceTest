
"""
Difference Test Simulations

Eugene Belilovsky 
eugene.belilovsky@inria.fr
"""

import scipy.stats
import numpy as np
from sklearn.linear_model import Lasso,LassoCV
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
from FusedMultiTask_R_Wrapper import FusedMultiTaskLassoCV
from Decorr_InverseCov_R import DecorrInverseCov,DecorrInverseCov2
from Decorr_InverseCovQPCV import DecorrInverseCovDiff,DecorrInverseCovQP
#from statsmodels.stats.multitest import multipletests
from joblib import Parallel,delayed
import sys
from DiffTestLib import generate_sparse_gaussian_graph_with_binary_confound3,Type_I_II,Power,Coverage_Length_CI
import pickle
import readline # this needs to be imported for ryp2 to link right
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
print 'Loaded Libs'

###########Setup Run Parameters
n_features=75
runs=5
n_jobs=5
alp=0.05
n1=800
s1=15
s=2.
#n2s=np.array([30,35,40,45,50,55,60,65,70,75,80,90,100,110,120])
n2s=np.array([40,60,120])

edge_changes=75 
density=0.4
numEval=75
np.random.seed(1)
max_n=n1+20
min_n=n1+5

#######Generate the data    
for _ in range(30):
    try:
        signals, precision, topology,topology2 = generate_sparse_gaussian_graph_with_binary_confound3(n_features=n_features,edge_changes=edge_changes,min_n_samples=min_n, max_n_samples=max_n,density=density,pin=True)
        break
    except:
        continue
B_Trues=list()
for prec in precision:
    B_True=np.zeros((n_features,n_features-1))
    for i in range(n_features):
        ind=np.concatenate([np.arange(0,i),np.arange(i+1,n_features)])
        B_True[i,:]=-prec[i,ind]/prec[i,i]
    B_Trues.append(B_True)

center=True

True_B_Undiag=B_Trues[0]-B_Trues[1]
eps=0.00000001
H1_True=np.abs(True_B_Undiag)>eps 
#H1_True2=np.zeros((n_features,n_features-1),dtype='bool')
#top=(np.abs(precision[0]-precision[1])>0)
#True_Prec_Undiag=np.zeros((n_features,n_features-1))
#for i in range(precision[0].shape[0]):
 #   ind=np.concatenate([np.arange(0,i),np.arange(i+1,precision[0].shape[0])])
  #  H1_True2[i,:]=top[i,ind]
   # True_Prec_Undiag[i,:]=precision[0][i,ind]-precision[1][i,ind]
        
print 'Sparsity:', (B_Trues[0]!=0).sum()/float(n_features*n_features)
print 'SparsityD:', (H1_True!=0).sum()/float(n_features*(n_features-1))

X1=signals[0][0:n1]
X1=X1/X1.std(axis=0)
n_vars=X1.shape[1]
n1=X1.shape[0]
    
if center:
    X1-=X1.mean(axis=0)
coefs_LS=np.zeros((n_vars-1,2))
coefs_L=np.zeros((n_vars-1,2))
coefs_F=np.zeros((n_vars-1,2))   

FP_F=np.zeros((numEval,len(n2s)))
FN_F=np.zeros((numEval,len(n2s)))

FP_L=np.zeros((numEval,len(n2s)))
FN_L=np.zeros((numEval,len(n2s)))

Cv_L=np.zeros(len(n2s))
Cvs_L=np.zeros(len(n2s))
Lv_L=np.zeros(len(n2s))
Lvs_L=np.zeros(len(n2s))

Cv_F=np.zeros(len(n2s))
Cvs_F=np.zeros(len(n2s))
Lv_F=np.zeros(len(n2s))
Lvs_F=np.zeros(len(n2s))


def SolveJointRegression(X1,X2,Y1,Y2,fused=True,s1=15,s=2):        
        mean_Y1_s=Y1.mean()
        mean_Y2_s=Y2.mean()
        sigma1=(Y1-mean_Y1_s).dot(Y1-mean_Y1_s)/n1 #unbiased estimate of variance of the noise variance of feature being regressed on TODO: make  sigma a subfunction with options
        Sigma1= np.dot(X1[:,regressor_ind].T, X1[:,regressor_ind]) /n1#empirical covariance matrix for the X_s^c
        sigma2=(Y2-mean_Y2_s).dot(Y2-mean_Y2_s)/n2
        Sigma2= np.dot(X2[:,regressor_ind].T, X2[:,regressor_ind]) /n2
    
        Ys=[Y1,Y2]
        Xs=[X1[:,regressor_ind],X2[:,regressor_ind]]
        

        lam_L1=np.sqrt(sigma1)*np.sqrt(2.*(np.log(n_vars-1))/n1)*2 #order sigma*sqrt(log(p)/n)
        alphas=np.array([0.1,0.5,1.,5.,10.,50.,100.])*lam_L1
        
        if fused:
            lam_L2=np.sqrt(sigma2)*np.sqrt(2.*(0.01+np.log(n_vars-1))/n2)
            lam_F=np.sqrt(sigma2)*np.sqrt((np.log((n_vars-1))/(n2)))
            
            fm_cv=FusedMultiTaskLassoCV(cv=5,gammas=np.array([0.1,1.,10.,100.])*lam_F,minLam=0.1*lam_F,n_jobs=n_jobs) 
            fm_cv.fit(Xs,Ys)
            coefs=np.reshape(fm_cv.coefs,(2,n_vars-1)).T
            
            mu1=4.*(fm_cv.alpha_*s1+fm_cv.gamma*s)
            mu1=(1./mu1)/(n2**(0.01))
            
            mu=2.*(fm_cv.alpha_)*s
            mu=(1./mu)/(n2**(0.01))
            
            Theta1,Theta2=DecorrInverseCovDiff(X1[:,regressor_ind],X2[:,regressor_ind],mu=mu,mu1=mu1)
            
        else:
            #Two debiased lasso      
            coefs=np.zeros((n_vars-1,2))
            ls_cv=LassoCV(cv=5,fit_intercept=False,alphas=alphas)      
            ls_cv.fit(X1[:,regressor_ind],Y1)
            coefs[:,0]=ls_cv.coef_
            
            lam_L2=np.sqrt(sigma2)*np.sqrt(2.*(0.01+np.log(n_vars-1))/n2)*2
            alphas=np.array([0.1,0.5,1.,5.,10.,50.,100.])*lam_L2
            ls_cv2=LassoCV(cv=5,fit_intercept=False,alphas=alphas)        
            ls_cv2.fit(X2[:,regressor_ind],Y2)
            coefs[:,1]=ls_cv2.coef_

            mu_L=(1./np.sqrt(n1)) * scipy.stats.norm.ppf(1-(0.1/((n_features**2))))
            mu_L2=(1./np.sqrt(n2)) * scipy.stats.norm.ppf(1-(0.1/((n_features**2))))
            Theta1=DecorrInverseCovQP(X1[:,regressor_ind],mu=mu_L)
            Theta2=DecorrInverseCovQP(X2[:,regressor_ind],mu=mu_L2)
                
       
        
        coefs_debiased1=coefs[:,0]+Theta1.dot(X1[:,regressor_ind].T.dot(Y1-X1[:,regressor_ind].dot(coefs[:,0])))/n1
        coefs_debiased2=coefs[:,1]+Theta2.dot(X2[:,regressor_ind].T.dot(Y2-X2[:,regressor_ind].dot(coefs[:,1])))/n2
        coefs_debiased=coefs_debiased1-coefs_debiased2
        var_components1=sigma1*np.diag(Theta1.dot(Sigma1).dot(Theta1.T)/n1)
        var_components2=sigma2*np.diag(Theta2.dot(Sigma2).dot(Theta2.T)/n2)
        
        std_components=np.sqrt(var_components1+var_components2)
        interval=scipy.stats.norm.ppf(1.-alp/2.)*std_components
    
        
        LowerConfInterval=coefs_debiased-interval # confidence interval
        UpperConfInterval=coefs_debiased+interval  
        Pvals=2*(1.-scipy.stats.norm.cdf(np.abs(coefs_debiased/(std_components))))
        

        return coefs_debiased,Pvals,LowerConfInterval,UpperConfInterval

for i in range(0,numEval):
    for n,n2 in  enumerate(n2s):
        X2=signals[1][0:n2]
        X2=X2/X2.std(axis=0)
        
        
        regressor_ind=np.concatenate([np.arange(0,i),np.arange(i+1,n_vars)])
        B1=B_Trues[0][i,:]*100.
        B2=B_Trues[1][i,:]*100.
        e1=np.random.randn(n1)
        e2=np.random.randn(n2)
        
        if center:
            X2-=X2.mean(axis=0)
        Y1=X1[:,regressor_ind].dot(B1)+e1
        Y2=X2[:,regressor_ind].dot(B2)+e2
        
             
        
        grd=np.c_[B1,B2]
 
        #Store for power computation and compute coverage
        H_Trues=H1_True
        B_Trues=B_Trues 
        
        h=np.sum(H_Trues[i,:])
        if(h==0):
            h=-1
        
        coefs_debiased,Pvals,LowerConfInterval,UpperConfInterval=SolveJointRegression(X1,X2,Y1,Y2)      
        FP,FN=Type_I_II((Pvals<0.05),H_Trues[i,:])
        coverage_s0,coverage_s0_c,length_s0,length_s0_s=Coverage_Length_CI(UpperConfInterval,
                                                         LowerConfInterval,H_Trues[i,:],B1-B2)
        FP_F[i,n]+=FP/h
        FN_F[i,n]+=FN/h
        Cv_F[n]+=coverage_s0
        Cvs_F[n]+=coverage_s0_c
        Lv_F[n]+=length_s0
        Lvs_F[n]+=length_s0_s 
        
        coefs_debiased,Pvals,LowerConfInterval,UpperConfInterval=SolveJointRegression(X1,X2,Y1,Y2,fused=False)      
        FP,FN=Type_I_II((Pvals<0.05),H_Trues[i,:])
        coverage_s0,coverage_s0_c,length_s0,length_s0_s=Coverage_Length_CI(UpperConfInterval,
                                                         LowerConfInterval,H_Trues[i,:],B1-B2)
        FP_L[i,n]=FP/h
        FN_L[i,n]=FN/h
        Cv_L[n]+=coverage_s0
        Cvs_L[n]+=coverage_s0_c
        Lv_L[n]+=length_s0
        Lvs_L[n]+=length_s0_s
    
    print("%d/%d"%(i,numEval))    


from scipy.stats import sem
FN_L[np.where(FN_L<0)]=0
FN_F[np.where(FN_F<0)]=0
Power_L_st= sem(1.-FN_L,axis=0)
Power_F_st= sem(1.-FN_F,axis=0)

Power_L= np.mean(1.-FN_L,axis=0)
Power_F= np.mean(1.-FN_F,axis=0)

print(Power_L)
print(Power_F)

#ind = np.arange(0,len(n2s))
#width = 0.32 
#fig,ax=plt.subplots()
#fig.set_size_inches(7, 4)

