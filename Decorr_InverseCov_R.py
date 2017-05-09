# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:40:52 2015
Wrapper for R code from Hamadi on computing the estimate of the debiased lasso, QP based solver seems to work better
@author: eugene
"""
import readline
import numpy as np
import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects
rpy2.robjects.numpy2ri.activate()

rstr='''
SoftThreshold <- function( x, lambda ) {
#
# Standard soft thresholding
#
  if (x>lambda){
    return (x-lambda);}
  else {
    if (x< (-lambda)){
      return (x+lambda);}
    else {
      return (0); }
  }
}

InverseLinftyOneRow <- function ( sigma, i, mu, maxiter=50, threshold=1e-2 ) {
  p <- nrow(sigma);
  rho <- max(abs(sigma[i,-i])) / sigma[i,i];
  mu0 <- rho/(1+rho);
  beta <- rep(0,p);
    
  if (mu >= mu0){
    beta[i] <- (1-mu0)/sigma[i,i];
    returnlist <- list("optsol" = beta, "iter" = 0);
    return(returnlist);
  }
  
  diff.norm2 <- 1;
  last.norm2 <- 1;
  iter <- 1;
  iter.old <- 1;
  beta[i] <- (1-mu0)/sigma[i,i];
  beta.old <- beta;
  sigma.tilde <- sigma;
  diag(sigma.tilde) <- 0;
  vs <- -sigma.tilde%*%beta;
  
  while ((iter <= maxiter) && (diff.norm2 >= threshold*last.norm2)){    
    
    for (j in 1:p){
      oldval <- beta[j];
      v <- vs[j];
      if (j==i)
        v <- v+1;    
      beta[j] <- SoftThreshold(v,mu)/sigma[j,j];
      if (oldval != beta[j]){
      	vs <- vs + (oldval-beta[j])*sigma.tilde[,j];
      }
    }
    
    iter <- iter + 1;
    if (iter==2*iter.old){
      d <- beta - beta.old;
      diff.norm2 <- sqrt(sum(d*d));
      last.norm2 <-sqrt(sum(beta*beta));
      iter.old <- iter;
      beta.old <- beta;
      if (iter>10)
         vs <- -sigma.tilde%*%beta;
    }
  }

  returnlist <- list("optsol" = beta, "iter" = iter)
  return(returnlist)
}

InverseLinfty <- function(sigma, n, resol=1.5, mu=NULL, maxiter=50, threshold=1e-2, verbose = TRUE) {
  isgiven <- 1;
  if (is.null(mu)){
  	isgiven <- 0;
  }

  p <- nrow(sigma);
  M <- matrix(0, p, p);
  xperc = 0;
  xp = round(p/10);
  for (i in 1:p) {
        if ((i %% xp)==0){
          xperc = xperc+10;
          if (verbose) {
            print(paste(xperc,"% done",sep="")); }
        }
  	if (isgiven==0){
  	  mu <- (1/sqrt(n)) * qnorm(1-(0.1/(p^2)));
  	}
  	mu.stop <- 0;
  	try.no <- 1;
  	incr <- 0;
  	while ((mu.stop != 1)&&(try.no<10)){
  	  last.beta <- beta
  	  output <- InverseLinftyOneRow(sigma, i, mu, maxiter=maxiter, threshold=threshold)
  	  beta <- output$optsol
  	  iter <- output$iter
  	  if (isgiven==1){
  	  	mu.stop <- 1
  	  }
  	  else{
            if (try.no==1){
              if (iter == (maxiter+1)){
                incr <- 1;
                mu <- mu*resol;
              } else {
                incr <- 0;
                mu <- mu/resol;
              }
            }
            if (try.no > 1){
              if ((incr == 1)&&(iter == (maxiter+1))){
                mu <- mu*resol;
              }
              if ((incr == 1)&&(iter < (maxiter+1))){
                mu.stop <- 1;
              }
              if ((incr == 0)&&(iter < (maxiter+1))){
                mu <- mu/resol;
              }
              if ((incr == 0)&&(iter == (maxiter+1))){
                mu <- mu*resol;
                beta <- last.beta;
                mu.stop <- 1;
              }                        
            }
          }
  	  try.no <- try.no+1
  	}
  	M[i,] <- beta;
  }
  return(M)
}
'''
robjects.r(rstr)
InverseLinfty=robjects.globalenv['InverseLinfty']

def DecorrInverseCov2(X1,X2,maxiter=100,threshold=1e-3,resol=1.5,verbose=False):
    n_samples1=X1.shape[0]
    emp_cov1=np.dot(X1.T, X1) /n_samples1
    n_samples2=X2.shape[0]
    emp_cov2=np.dot(X2.T, X2) /n_samples2
    emp_cov=(emp_cov1+emp_cov2)/2
    res=InverseLinfty(emp_cov,(n_samples1+n_samples2)/2, resol=1.5, maxiter=maxiter, threshold=threshold,verbose=verbose)
    return np.array(res)
def DecorrInverseCov(X,maxiter=100,threshold=1e-3,resol=1.5,verbose=False):
    n_samples=X.shape[0]
    emp_cov=np.dot(X.T, X) /n_samples
    res=InverseLinfty(emp_cov,n_samples, resol=1.5, maxiter=maxiter, threshold=threshold,verbose=verbose)
    return np.array(res)