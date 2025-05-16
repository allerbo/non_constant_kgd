import numpy as np
import sys
from kgd import kgd
from help_fcts import r2, krr, cv10, mml, make_data_real, kgf
import pickle
from ffnn import  init_model, train_step



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
  y1_kgd=kgd(np.vstack((X_tr,X_te)),X_tr,y_tr, nu=nu, v_R2=v_R2)
  r2_dict['kgd'+str(v_R2)]=r2(y_te,y1_kgd[n_tr:,:])

lbda_cv, sigma_cv=cv10(X_tr,y_tr,lbda_bds,sigma_bds,seed,nu=nu)
y1_krr_cv=krr(np.vstack((X_tr, X_te)),X_tr,y_tr,lbda_cv,sigma_cv, nu=nu)
r2_dict['krr_cv']=r2(y_te,y1_krr_cv[n_tr:,:])

lbda_mml, sigma_mml=mml(X_tr,y_tr,lbda_bds,sigma_bds,nu=nu)
y1_krr_mml=krr(np.vstack((X_tr, X_te)),X_tr,y_tr,lbda_mml,sigma_mml, nu=nu)
r2_dict['krr_mml']=r2(y_te,y1_krr_mml[n_tr:,:])

lbda_cv_kgf, sigma_cv_kgf=cv10(X_tr,y_tr,lbda_bds,sigma_bds,seed,nu=nu, b_krr=False)
y1_kgf_cv=kgf(np.vstack((X_tr, X_te)),X_tr,y_tr,1/lbda_cv_kgf,sigma_cv_kgf, nu=nu)
r2_dict['kgf_cv']=r2(y_te,y1_kgf_cv[n_tr:,:])

if nu==100:
  X_tr_nn=X_tr[:int(0.9*n_tr),:]
  X_vl_nn=X_tr[int(0.9*n_tr):,:]
  y_tr_nn=y_tr[:int(0.9*n_tr),:]
  y_vl_nn=y_tr[int(0.9*n_tr):,:]
  best_r2_vl=-np.inf
  r2_counter=0
  model_state = init_model(X_tr_nn.shape[1],100,1,0.01,0)
  for epoch in range(10000):
    model_state = train_step(model_state, X_tr_nn, y_tr_nn)
    if epoch % 10==0:
      r2_vl=r2(y_vl_nn, model_state.apply_fn(model_state.params,X_vl_nn))
      if r2_vl>best_r2_vl:
        best_r2_vl=r2_vl
        r2_counter=0
        theta=model_state.params
      else:
        r2_counter+=1
      if r2_counter>10:
        break
  
  r2_dict['ffnn']=r2(y_te,model_state.apply_fn(model_state.params,X_te))

fi=open('dec_data/dec_'+data+'_'+str(nu)+'_'+str(seed)+'.pkl','wb')
pickle.dump(r2_dict,fi)
fi.close()
