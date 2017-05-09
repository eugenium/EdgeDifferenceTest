"""Fused Multi Task Lasso path, uses genlasso with CV
"""

# Author: Eugene Belilovsky <eugene.belilovsky@inria.fr>
# License: BSD 3 clause

import numpy as np
import scipy as sp
from sklearn.externals.joblib import Parallel, delayed
from sklearn.cross_validation import check_cv
import readline # this needs to be imported for ryp2 to link right
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
genLasso=importr("genlasso")

def ConvertToGenLassoFormat(Xs,Ys):
    #We want objective of the form ||Y-Xw ||^2/n +R(w) , genlasso solves ||Y-Xw ||^2/2 +R(w) we account for that here by rescaling X and Y 
    Xs=[X*np.sqrt(2./X.shape[0]) for X in Xs]
    Ys=[Y*np.sqrt(2./Y.shape[0]) for Y in Ys]
    
    #Now we form the joint objective by block diagonalizing X
    X=sp.linalg.block_diag(*Xs)
    Y=np.hstack(Ys)   
    
    return X,Y

def fused_multitask_lasso_path_residue(Xtrains , Xtests,Ytrains,Ytests,D,gamma=0,approx=True, max_iter=2000, verbose=False,minLam=0):
    
    X_train,Y_train=ConvertToGenLassoFormat(Xtrains,Ytrains)
    X_test,Y_test=ConvertToGenLassoFormat(Xtests,Ytests)
    genData=genLasso.fusedlasso(Y_train, X_train, D, gamma = gamma, approx = approx, maxsteps = max_iter,minlam=minLam,rtol = 1e-7, btol = 1e-7, eps = 1e-4,verbose = verbose) #genlasso warns against changing the tolerance
    
    Betas=np.array(genData[1])
    Lambdas=np.array(genData[0])
    nlam=Lambdas.shape[0]
    Residues=X_test.dot(Betas)-np.tile(Y_test,(nlam,1)).T
#    Residues=Y_Predict-
    return Lambdas,Betas,Residues.T #transpose to stay aligned with sci-kit LARS
#    
class FusedMultiTaskLassoCV():
    '''
        We compute CV paths for the fused lasso and then use interpolation to pick the lowest residue.
        We follow the general structure of LarsCV in sci-kit learn.
        
        The fused lasso path is computed using the efficient R package from Ryan Tibshirani
    '''
    def __init__(self, cv=3,max_iter=3000,max_iter_cv=800,gammas=np.array([0]), n_jobs=1, verbose=False,max_n_alphas=1000,approx=True,bagging=False,lam=-99,minLam=0):
       # self.lambdas1=lambdas1
        self.gammas=gammas
        self.max_iter_cv=max_iter_cv
#        self.tol = tol #genlasso code warns against messing with tolerance
        self.max_iter = max_iter
        self.verbose = verbose
        self.cv = cv
        self.n_jobs = n_jobs
        self.approx=approx
        # The base class needs this for the score method
        self.store_weights = True
        self.max_n_alphas=max_n_alphas
        self.bagging=bagging
        self.lam=lam
        self.minLam=minLam
    def _fit_path_gamma(self,all_X_Y,D,gamma):
        '''fits the path for each gamma, then we can recombine the best solution along gamma dimension, this function assumes all the cv folds have been computed'''
        inner_verbose = max(0, self.verbose - 1)
        dim=all_X_Y[0][0][0].shape[1]*2

        cv_paths = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose)(
                delayed(fused_multitask_lasso_path_residue)(
                    Xtrains, Xtests,Ytrains,Ytests,D,gamma=gamma,approx=self.approx, max_iter=self.max_iter*0.25, verbose=inner_verbose)
                for Xtrains, Xtests,Ytrains,Ytests in all_X_Y)

        #The following code is lifted from LarsCV
        #Now that we computed the path for each CV subset we use interpolation on the residues
        all_alphas = np.concatenate(list(zip(*cv_paths))[0])
        # Unique also sorts
        all_alphas = np.unique(all_alphas)
        # Take at most max_n_alphas values
        stride = int(max(1, int(len(all_alphas) / float(self.max_n_alphas))))
        all_alphas = all_alphas[::stride]

        mse_path = np.empty((len(all_alphas), len(cv_paths)))
        best_betas = np.empty((dim, len(cv_paths)))
        for index, (alphas, coefs,residues) in enumerate(cv_paths):
            best_betas[:,index]=coefs[:,np.argmin(np.sum(residues**2,axis=1))]
            alphas = alphas[::-1]
            residues = residues[::-1]          
            if alphas[0] != 0:
                alphas = np.r_[0, alphas]
                residues = np.r_[residues[0, np.newaxis], residues]
            if alphas[-1] != all_alphas[-1]:
                alphas = np.r_[alphas, all_alphas[-1]]
                residues = np.r_[residues, residues[-1, np.newaxis]]
            this_residues = sp.interpolate.interp1d(alphas,
                                                 residues,
                                                 axis=0)(all_alphas)
            this_residues **= 2
            mse_path[:, index] = np.mean(this_residues, axis=-1)
            
        mask = np.all(np.isfinite(mse_path), axis=-1)
        all_alphas = all_alphas[mask]
        mse_path = mse_path[mask]
        # Select the alpha that minimizes left-out error
        mse_path_mean=mse_path.mean(axis=-1)
        i_best_alpha = np.argmin(mse_path_mean)
        best_alpha = all_alphas[i_best_alpha]
        mse_best_alpha=mse_path_mean[i_best_alpha]
        
        return mse_path_mean,best_alpha,mse_best_alpha,best_betas
    def fit(self, Xs,Ys):
        p=Xs[0].shape[1]
        D=np.c_[np.eye(p),-np.eye(p)]
        
        #Setup the cross validation folds
        numSignals=len(Xs)
        if(self.lam<0):
            cvs=list()
            for X in Xs:
                X = np.asarray(X)
                cvs.append(check_cv(self.cv, X, y=None, classifier=False))
           
            all_X_Y=list()
            
            for i in range(self.cv):
                Xtrains=list()
                Xtests=list()
                Ytrains=list()
                Ytests=list()
                for j in range(numSignals):
                    train,test=list(cvs[j])[i]
                    Xtrains.append(Xs[j][train])
                    Xtests.append(Xs[j][test])
                    Ytrains.append(Ys[j][train])
                    Ytests.append(Ys[j][test])
                all_X_Y.append([Xtrains,Xtests,Ytrains,Ytests])
    
    
            mse_path_all=list()
            best_mses=list()
            best_mse=np.Inf
            best_betas_all=list()
            for gamma in self.gammas:
                mse_path_mean,best_alpha,mse_best_gamma,best_betas=self._fit_path_gamma(all_X_Y,D,gamma)
                best_betas_all.append(best_betas)
                if(mse_best_gamma<best_mse):
                    best_mse=mse_best_gamma
                    self.alpha_= best_alpha
                    self.gamma = gamma
                mse_path_all.append(mse_path_mean)
                best_mses.append(mse_best_gamma)
            
            self.cv_mse_path_ = mse_path_all
            self.best_mses=best_mses
        else:
            self.gamma=0
            self.alpha_=self.lam
        if(self.bagging):
            self.coefs=np.array(best_betas_all).mean(axis=0).mean(axis=1)
        else:
            if(self.alpha_<self.minLam):
                self.alpha_=self.minLam
            #TODO: fix this to not solve the whole path and maybe bagging
            X_train,Y_train=ConvertToGenLassoFormat(Xs,Ys)
            genData=genLasso.fusedlasso(Y_train, X_train, D, gamma = self.gamma, approx = False, maxsteps = self.max_iter, eps = 1e-4,verbose = self.verbose)
            self.coefs=np.array(genLasso.coef_genlasso(genData,np.array([self.alpha_]))[0])[:,0]


#import time
#from GraphDiffHypothesisTest import generate_sparse_gaussian_graph_with_binary_confound
#max_n=100
#density=0.8
#min_n=max_n*0.9
#n_features=50
#signals, precision, topology,topology2 = generate_sparse_gaussian_graph_with_binary_confound(n_features=n_features,edge_changes=8,min_n_samples=min_n, max_n_samples=max_n,density=density)
#diff=(precision[0]>0).astype('int64')-(precision[1]>0).astype('int64')
#rows_Changed=diff.sum(axis=0).astype('bool')
#
##pick 1 problems  with changes and 1 without changes in the graph
#changed=np.where(rows_Changed)[0]
#unchanged=np.where(~rows_Changed)[0]
#
#np.random.seed(0)
#B_Trues=list()
#for prec in precision:
#    B_True=np.zeros((50,49))
#    for i in range(50):
#        ind=np.concatenate([np.arange(0,i),np.arange(i+1,n_features)])
#        B_True[i,:]=-prec[i,ind]/prec[i,i]
#    B_Trues.append(B_True)
#B_true_d=B_Trues[0]-B_Trues[1]
#
#i=changed[0]
#n_vars=50
#
#train_ind=np.arange(0,49)
#test_ind=np.arange(60,85)
#regressor_ind=np.concatenate([np.arange(0,i),np.arange(i+1,n_vars)])
#
#Ytrain=[signals[0][train_ind,i],signals[1][train_ind,i]]
#Xtrain=[signals[0][train_ind][:,regressor_ind],signals[1][train_ind][:,regressor_ind]]
#
#Ytest=[signals[0][test_ind,i],signals[1][test_ind,i]]
#Xtest=[signals[0][test_ind][:,regressor_ind],signals[1][test_ind][:,regressor_ind]]
#D=np.c_[np.eye(n_vars-1),-np.eye(n_vars-1)]
#begin=time.time()
#lams,betas1,residue=fused_multitask_lasso_path_residue(Xtrain , Xtest,Ytrain,Ytest,D,max_iter=100,gamma=0.001,approx=False)
#end=time.time()
#
#
#beta_1=betas1[:,np.argmin(np.linalg.norm(residue,axis=0))]
#
#beta_1_d=beta_1[0:49]-beta_1[49:]
#Err=np.linalg.norm(beta_1_d-B_true_d[i])
#print '1 path:',str(end-begin),'Error: ',Err
#train_ind=np.arange(0,85)
#Ytrain=[signals[0][train_ind,i],signals[1][train_ind,i]]
#Xtrain=[signals[0][train_ind][:,regressor_ind],signals[1][train_ind][:,regressor_ind]]
#FM=FusedMultiTaskLassoCV(cv=4,max_iter=3000, n_jobs=1, verbose=False,max_n_alphas=1000,gammas=np.array([0.9,0.7,0.5,0.2]))
#begin=time.time()
#FM.fit(Xtrain,Ytrain)
#end=time.time()
#print str(end-begin)
