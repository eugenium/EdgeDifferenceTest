# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:40:52 2015
QP solver wrappers
@author: eugene
eugene.belilovsky@ecp.fr
"""
import sys
import scipy.stats as st
import numpy as np
import itertools
from cvxopt import matrix, solvers,spmatrix
from mosek import iparam,dparam

epsilon=1e-15
solvers.options['MOSEK'] = {iparam.log: 0,iparam.log_intpnt:0,iparam.log_cut_second_opt:0,dparam.check_convexity_rel_tol:1e-3}
solvers.options['mosek'] = {iparam.log: 0,iparam.log_intpnt:0,iparam.log_cut_second_opt:0,dparam.check_convexity_rel_tol:1e-3}      
solvers.options['LPX_K_MSGLEV']=0
def InverseLinftyOneRow_ConstDiff_Identy( Sigma1,Sigma2,i, mu,mu1, maxiter=500, thresh=1e-2,n1=1.,n2=1.):
    n=Sigma1.shape[0]
    P=matrix(0.0,(2*n,2*n))
    G=matrix(0.0,(4*n,2*n))
    h=matrix(0.0,(4*n,1))
    q=matrix(0.0,(2*n,1))
    

    P[0:n,0:n]=Sigma1*100.
    P[n:2*n,n:2*n]=100.*n1*Sigma2/n2 #Multiply by n1 for numeric stability
    

    G[0:n,0:n]=Sigma1
    G[0:n,n:2*n]=-Sigma2
    h[0:n,:]=mu1
    
    G[3*n:4*n,0:n]=-Sigma1
    G[3*n:4*n,n:2*n]=Sigma2
    h[3*n:4*n,:]=mu1
    
    e=matrix(0.0,(n,1))
    e[i]=1.0
    
    
    G[n:2*n,0:2*n]=np.c_[Sigma1, Sigma2]
    h[n:2*n,:]=mu+2*e
    
    G[2*n:3*n,0:2*n]=np.c_[-Sigma1, -Sigma2]
    h[2*n:3*n,:]=mu-2*e

    A = spmatrix([], [], [], (0,2*n), 'd')  
    b = matrix(0.0, (0,1))
    solvers.options['show_progress']=False
    solvers.options['maxiters']=maxiter
    try:
        sol=solvers.qp(P,q,G,h,A,b,solver='mosek')
    except:
        sol=solvers.qp(P,np.squeeze(q),G,np.squeeze(h),A,np.squeeze(b),solver='mosek')
    beta=np.squeeze(np.array(sol['x']))
    itr=maxiter
    return beta,itr

def InverseLinftyOneRowConstDiff( Sigma1,Sigma2,i, mu, maxiter=50, thresh=1e-2,n1=1.,n2=1. ):
    n=Sigma1.shape[0]
    P=matrix(0.0,(2*n,2*n))
    G=matrix(0.0,(2*n,2*n))
    h=matrix(0.0,(2*n,1))
    q=matrix(0.0,(2*n,1))
    
    e=matrix(0.0,(n,1))
    e[i]=1.0
    P[0:n,0:n]=Sigma1
    P[n:2*n,n:2*n]=Sigma2


    G[0:n,0:n]=Sigma1
    G[0:n,n:2*n]=-Sigma2
    h[0:n,:]=mu
    #h[n:2*n,:]=mu
    A = spmatrix([], [], [], (0,2*n), 'd')  
    b = matrix(0.0, (0,1))
    
    solvers.options['show_progress']=False
    solvers.options['maxiters']=maxiter
    sol=solvers.qp(P,q,G,h,A,b,solver='mosek')
    beta=np.squeeze(np.array(sol['x']))
    itr=maxiter
    return beta,itr
    
def InverseLinftyConstDiff(Sigma1,Sigma2, n1,n2, mu=None,mu1=None, maxiter=500, thresh=1e-2, verbose = True):
    isgiven=1
    if (mu==None):
        isgiven=0
    
    p=Sigma1.shape[0]
    M1=np.zeros((p,p))
    M2=np.zeros((p,p))
    if (isgiven==0):
        mu = (1./np.sqrt(n1)) * st.norm.ppf(1-(0.1/(p**2)))
    for i in range(p):
        mu_t=mu
        
        if(mu1!=None):
            mu1_t=mu1
        
        
        for _ in range(15):
            if(mu1==None):
                beta_r,itr = InverseLinftyOneRowConstDiff(Sigma1,Sigma2, i, mu_t, maxiter=maxiter, thresh=thresh,n1=n1,n2=n2)
            else:
                try:
                    beta_r,itr = InverseLinftyOneRow_ConstDiff_Identy( Sigma1,Sigma2,i, mu_t,mu1_t, maxiter=maxiter, thresh=thresh,n1=n1,n2=n2)
                except:
                    print 'Failed to Run Mosek'
                    beta_r=np.array([])
            #Sometimes we are not feasible so reduce the constraints in this case
            if(not beta_r.any()):
                print 'Increasing Constraints'
                if(mu1==None):
                    mu_t*=2
                else:
                    mu_t*=2.
                    mu1_t*=1.1
            else:
                break
            
        try:
            M1[i,:] = np.array(beta_r[0:p])
            M2[i,:] = np.array(beta_r[p:2*p])
        except:
            print 'Failed to Converge'
            M1[i,i] = 1.
            M2[i,i] = 1.
            
    return M1,M2


def InverseLinftyConstQP(Sigma1, n1,mu=None,maxiter=50, thresh=1e-2, verbose = True):
    isgiven=1
    if (mu==None):
        isgiven=0
    
    p=Sigma1.shape[0]
    M1=np.zeros((p,p))
    if (isgiven==0):
        mu = (1./np.sqrt(n1)) * st.norm.ppf(1-(0.1/(p**2)))
    for i in range(p):
        mu_t=mu
             
        for _ in range(15):
            beta_r,itr = InverseLinftyOneRowConst(Sigma1,i, mu_t, maxiter=maxiter, thresh=thresh)

            #Sometimes we are not feasible so reduce the constraints in this case
            if(not beta_r.any()):
                mu_t*=1.5
            else:
                break
        
        try:
            M1[i,:] = np.array(beta_r[0:p])
        except:
            print 'Failed to Converge'
            M1[i,i] = 1.
            
    return M1
def InverseLinftyOneRowConst( Sigma, i, mu, maxiter=50, thresh=1e-2 ):
    n=Sigma.shape[0]
    P=matrix(2.*Sigma)
    G=matrix(0.0,(2*n,n))
    h=matrix(0.0,(2*n,1))
    q=matrix(0.0,(n,1))
    
    e=matrix(0.0,(n,1))
    e[i]=1.0
    solvers.options['show_progress']=False
    solvers.options['maxiters']=maxiter
    G[0:n,0:n]=Sigma
    G[n:2*n,0:n]=-Sigma
    h[0:n,:]=mu+e
    h[n:2*n,:]=mu-e
    A = spmatrix([], [], [], (0,n), 'd')  
    b = matrix(0.0, (0,1))
    try:
        sol=solvers.qp(P,q,G,h,A,b,solver='mosek')
    except:
        sol=solvers.qp(P,q,G,h,A,b,solver='mosek')
    beta=np.squeeze(np.array(sol['x']))
    itr=maxiter
    return beta,itr

def InverseLinfty(Sigma, n, resol=1.5, mu=None, maxiter=50, thresh=1e-2, verbose = True):
    isgiven=1
    if (mu==None):
        isgiven=0
    
    p=Sigma.shape[0]
    M=np.zeros((p,p))
    xperc = 0
    xp = round(p/10.)
    beta=np.zeros(p)
    for i in range(p):
        #Deal with each row of the solution separately
        if ((i*xp)==0):
            xperc = xperc+10;


        if (isgiven==0):
            mu = (1./np.sqrt(n)) * st.norm.ppf(1-(0.1/(p**2)))
        
        mu_stop=0
        try_no =1
        incr = 0
        
        while ((mu_stop != 1) and (try_no<10)):
            last_beta= beta
            beta_r,itr = InverseLinftyOneRowConst(Sigma, i, mu, maxiter=maxiter, thresh=thresh)
            beta = beta_r
            itr = itr
            if (isgiven==1):
                mu_stop = 1
            else:
                if (try_no==1):
                    if (itr == (maxiter+1)):
                        incr=1
                        mu =mu*resol
                    else:
                        incr = 0
                        mu = mu/resol
                    
                if (try_no > 1):
                    if ((incr == 1) and (itr == (maxiter+1))):
                        mu = mu*resol
                    if ((incr == 1) and (itr < (maxiter+1))):
                        mu_stop = 1
                    if ((incr == 0) and (itr < (maxiter+1))):
                        mu = mu/resol
                    if ((incr == 0) and (itr == (maxiter+1))):
                        mu = mu*resol
                        beta = last_beta
                        mu_stop = 1
                    
            try_no = try_no+1
        M[i,:] = beta
    return M

def DecorrInverseCovDiff(X1,X2,maxiter=100,threshold=1e-3,resol=1.5,verbose=False,mu=None,mu1=None,parallel=True):
    if(not parallel):
        solvers.options['MOSEK'] = {iparam.log: 0,iparam.log_intpnt:0,iparam.log_cut_second_opt:0,dparam.check_convexity_rel_tol:1e-3,iparam.num_threads:3}
        solvers.options['mosek'] = {iparam.log: 0,iparam.log_intpnt:0,iparam.log_cut_second_opt:0,dparam.check_convexity_rel_tol:1e-3,iparam.num_threads:3}  
    n1=X1.shape[0]
    n2=X2.shape[0]
    emp_cov1=np.dot(X1.T, X1) /n1
    emp_cov2=np.dot(X2.T, X2) /n2
    res=InverseLinftyConstDiff(emp_cov1,emp_cov2,n1,n2,maxiter=maxiter, thresh=threshold,verbose=verbose,mu=mu,mu1=mu1)
    return np.array(res)

def DecorrInverseCovQP(X1,maxiter=100,threshold=1e-3,resol=1.5,verbose=False,mu=None,parallel=True):
    if(not parallel):
        solvers.options['MOSEK'] = {iparam.log: 0,iparam.log_intpnt:0,iparam.log_cut_second_opt:0,dparam.check_convexity_rel_tol:1e-3,iparam.num_threads:3}
        solvers.options['mosek'] = {iparam.log: 0,iparam.log_intpnt:0,iparam.log_cut_second_opt:0,dparam.check_convexity_rel_tol:1e-3,iparam.num_threads:3}  
    n1=X1.shape[0]
    emp_cov1=np.dot(X1.T, X1) /n1+epsilon*np.eye(X1.shape[1])
    res=InverseLinftyConstQP(emp_cov1,n1,maxiter=maxiter, thresh=threshold,verbose=verbose,mu=mu)
    return np.array(res)
#InverseLinfty=robjects.globalenv['InverseLinfty']
#robjects.r(rstr)
#InverseLinftyR=robjects.globalenv['InverseLinftyOneRow']
def DecorrInverseCov2(X,maxiter=100,threshold=1e-3,resol=1.5,verbose=False,mu=None):
    n_samples=X.shape[0]
    emp_cov=np.dot(X.T, X) /n_samples
    res=InverseLinfty(emp_cov,n_samples, resol=1.5, maxiter=maxiter, thresh=threshold,verbose=verbose,mu=mu)
    return np.array(res)
