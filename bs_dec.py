import numpy as np
import pandas as pd
import sys
from kgd import kgd
from help_fcts import r2, krr, gcv, log_marg, kgf
import pickle


def in_hull(p, hull):
  from scipy.spatial import Delaunay
  if not isinstance(hull,Delaunay):
    hull = Delaunay(hull)
  
  return hull.find_simplex(p)>=0

def make_data_bs(day):
  np.random.seed(0)
  data1=data[data[:,1]==day]
  np.random.shuffle(data1)
  X=data1[:,8:10]
  X=(X-np.mean(X, 0))/np.std(X,0)
  y=data1[:,5].reshape((-1,1))
  n=X.shape[0]
  per=np.random.permutation(n)
  folds=np.array_split(per,10)
  return X, y, folds
   
def cv_split(X, y, v_fold):
  t_idxs=np.concatenate([folds[t_fold] for t_fold in range(len(folds)) if v_fold != t_fold])
  v_idxs=folds[v_fold]
   
  X_tr=X[t_idxs,:]
  X_te=X[v_idxs,:]
  y_tr=y[t_idxs,:]
  y_te=y[v_idxs,:]
  
  #Remove outside convex hull
  in_ch=in_hull(X_te,X_tr)
  X_te=X_te[in_ch,:]
  y_te=y_te[in_ch,:]

  return X_tr, y_tr, X_te, y_te

data=pd.read_csv('bs_2000.csv',sep=',').to_numpy()
LBDA_MIN_CV=1e-4
LBDA_MAX_CV=10
SIGMA_MIN_CV=1e-2
SIGMA_MAX_CV=10
LBDA_MIN_LM=1e-6
LBDA_MAX_LM=10
SIGMA_MIN_LM=.01
SIGMA_MAX_LM=1e3

nu=0.5
day=1
for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

X, y, folds = make_data_bs(day)
r2_dict={'kgd': [], 'krr_cv': [], 'krr_lm': [], 'kgf_cv': [], 'kgf_lm': []}
for v_fold in range(len(folds)):
  X_tr, y_tr, X_te, y_te = cv_split(X, y, v_fold)
  n_tr=X_tr.shape[0]
        
  y1_kgd=kgd(np.vstack((X_tr,X_te)),X_tr,y_tr, step_size=0.01, nu=nu)
  r2_dict['kgd'].append(r2(y_te,y1_kgd[n_tr:,:]))
      
  lbda_cv, sigma_cv=gcv(X_tr,y_tr,[LBDA_MIN_CV, LBDA_MAX_CV], [SIGMA_MIN_CV, SIGMA_MAX_CV],nu=nu)
  y1_krr_cv=krr(np.vstack((X_tr, X_te)),X_tr,y_tr,lbda_cv,sigma_cv, nu=nu)
  r2_dict['krr_cv'].append(r2(y_te,y1_krr_cv[n_tr:,:]))

  lbda_lm, sigma_lm=log_marg(X_tr,y_tr,[LBDA_MIN_LM, LBDA_MAX_LM], [SIGMA_MIN_LM, SIGMA_MAX_LM],nu=nu)
  y1_krr_lm=krr(np.vstack((X_tr, X_te)),X_tr,y_tr,lbda_lm,sigma_lm, nu=nu)
  r2_dict['krr_lm'].append(r2(y_te,y1_krr_lm[n_tr:,:]))

  y1_kgf_cv=kgf(np.vstack((X_tr, X_te)),X_tr,y_tr,1/lbda_cv,sigma_cv, nu=nu)
  r2_dict['kgf_cv'].append(r2(y_te,y1_kgf_cv[n_tr:,:]))
  y1_kgf_lm=kgf(np.vstack((X_tr, X_te)),X_tr,y_tr,1/lbda_lm,sigma_lm, nu=nu)
  r2_dict['kgf_lm'].append(r2(y_te,y1_kgf_lm[n_tr:,:]))

fi=open('dec_data/bs_dec_nu_seed_'+str(nu)+'_'+str(day)+'.pkl','wb')
pickle.dump(r2_dict,fi)
fi.close()
