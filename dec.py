import numpy as np
import sys
from kgd import kgd
from help_fcts import r2, krr, gcv, mml, make_data_real
import pickle

sigma_bds=[0.01, 10]
lbda_bds=[1e-6, 1]
v_R2s=[0.02, 0.05, 0.1, 0.2, 0.5]

nu=100
data='temp'
seed=0

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

r2_dict={}
X_tr, y_tr, X_te, y_te = make_data_real(data, seed)
n_tr=X_tr.shape[0]

for v_R2 in v_R2s:
  y1_kgd,_=kgd(np.vstack((X_tr,X_te)),X_tr,y_tr, nu=nu, v_R2=v_R2)
  r2_dict['kgd'+str(v_R2)]=r2(y_te,y1_kgd[n_tr:,:])

lbda_gcv, sigma_gcv=gcv(X_tr,y_tr,lbda_bds,sigma_bds,nu=nu)
y1_krr_gcv,_=krr(np.vstack((X_tr, X_te)),X_tr,y_tr,lbda_gcv,sigma_gcv, nu=nu)
r2_dict['krr_gcv']=r2(y_te,y1_krr_gcv[n_tr:,:])

lbda_mml, sigma_mml=mml(X_tr,y_tr,lbda_bds,sigma_bds,nu=nu)
y1_krr_mml,_=krr(np.vstack((X_tr, X_te)),X_tr,y_tr,lbda_mml,sigma_mml, nu=nu)
r2_dict['krr_mml']=r2(y_te,y1_krr_mml[n_tr:,:])

fi=open('dec_data/dec_'+data+'_'+str(nu)+'_'+str(seed)+'.pkl','wb')
pickle.dump(r2_dict,fi)
fi.close()
