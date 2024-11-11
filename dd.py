import numpy as np
import sys
from kgd import kgd
from help_fcts import r2, krr, trans_x, make_data_real
from sklearn import datasets
import pickle

import warnings
warnings.filterwarnings("ignore")

def make_data_syn(seed):
  def f_synth(x):
    y=np.sin(2*np.pi*x)
    return y
  np.random.seed(seed)
  n_tr=20
  n_te=100
  x_tr=np.random.uniform(-1,1,n_tr).reshape((-1,1))
  y_tr=f_synth(x_tr)+np.random.normal(0,.2,x_tr.shape)
  x_te=np.random.uniform(-1,1,n_te).reshape((-1,1))
  y_te=f_synth(x_te)
  return x_tr, y_tr, x_te, y_te


nu=100
data='syn'
seed=1

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])


lbda=1e-3

r2_dict={}
for alg1 in ['krr','kgd']:
  r2_dict[alg1]={}
  for tr_te in ['tr','te']:
    r2_dict[alg1][tr_te]={}

if data=='syn':
  X_tr, y_tr, X_te, y_te = make_data_syn(seed)
else:
  X_tr, y_tr, X_te, y_te = make_data_real(data, seed)

X_DICT={'compactiv': [0.4, 1e5], 'super': [0.3, 1e5], 'temp': [0.1, 1e2], 'power': [0.2, 1e5], 'wood': [1e-4, 1e5], 'syn': [1e-4, 20]}
if nu==100:
  X_DICT['syn']=[1e-4, 1]
elif nu==10:
  X_DICT['syn']=[1e-4, 3]
elif nu==0.5:
  X_DICT['syn']=[1e-4, 1e5]

sigmas=trans_x(np.linspace(trans_x(X_DICT[data][0]), trans_x(X_DICT[data][1]), 100),inv=True)

n_tr=X_tr.shape[0]
for sigma in sigmas:
  fh_tr_krr=krr(X_tr,X_tr,y_tr,lbda,sigma, nu=nu)
  fh_te_krr=krr(X_te,X_tr,y_tr,lbda,sigma, nu=nu)
  r2_dict['krr']['tr'][sigma]=r2(y_tr,fh_tr_krr)
  r2_dict['krr']['te'][sigma]=r2(y_te,fh_te_krr)
  fh_kgd=kgd(np.vstack((X_tr,X_te)),X_tr,y_tr, nu=nu, sigma_min=sigma, t_max=1/lbda)
  r2_dict['kgd']['tr'][sigma]=r2(y_tr,fh_kgd[:n_tr,:])
  r2_dict['kgd']['te'][sigma]=r2(y_te,fh_kgd[n_tr:,:])

fi=open('dd_data/dd_'+data+'_'+str(nu)+'_'+str(seed)+'.pkl','wb')
pickle.dump(r2_dict,fi)
fi.close()
