import numpy as np
import pandas as pd
import sys
from kgd import kgd
from help_fcts import r2, krr, trans_x, kgf
import pickle

def in_hull(p, hull):
  from scipy.spatial import Delaunay
  if not isinstance(hull,Delaunay):
      hull = Delaunay(hull)

  return hull.find_simplex(p)>=0

def f_synth(x):
  y=np.sin(2*np.pi*x)
  return y

def make_data_bs(day):
  data=pd.read_csv('bs_2000.csv',sep=',').to_numpy()
  np.random.seed(0)
  data1=data[data[:,1]==day]
  np.random.shuffle(data1)
  X=data1[:,8:10]
  X=(X-np.mean(X, 0))/np.std(X,0)
  y=data1[:,5].reshape((-1,1))
  n_tot=X.shape[0]
   
  X_tr=X[:int(0.8*n_tot),:]
  X_te=X[int(0.8*n_tot):,:]
  y_tr=y[:int(0.8*n_tot),:]
  y_te=y[int(0.8*n_tot):,:]
    
  #Remove outside convex hull
  in_ch=in_hull(X_te,X_tr)
  X_te=X_te[in_ch,:]
  y_te=y_te[in_ch,:]
  
  return X_tr, y_tr, X_te, y_te

def make_data_synth(seed):
  np.random.seed(seed)
  n_tr=20
  n_te=100
  x_tr=np.random.uniform(-1,1,n_tr).reshape((-1,1))
  y_tr=f_synth(x_tr)+np.random.normal(0,.2,x_tr.shape)
  x_te=np.random.uniform(-1,1,n_te).reshape((-1,1))
  y_te=f_synth(x_te)
  
  return x_tr, y_tr, x_te, y_te

t=1e3
lbda=1e-3

seed=1
nu=100
data='syn'
for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

if data=='syn':
  step_size=0.1
  sigmas=trans_x(np.linspace(trans_x(1e-3), trans_x(1e3), 100),inv=True)
  make_data=make_data_synth
elif data=='bs':
  step_size=0.01
  sigmas=trans_x(np.linspace(trans_x(0.009), trans_x(1e5), 100),inv=True)
  make_data=make_data_bs


data_dict={'krr': {'tr':{},'te':{}},'kgd': {'tr':{},'te':{}},'kgf': {'tr':{},'te':{}}}
data_dict['krr']['tr']={}
data_dict['krr']['te']={}
data_dict['kgd']['tr']={}
data_dict['kgd']['te']={}
data_dict['kgf']['tr']={}
data_dict['kgf']['te']={}

X_tr, y_tr, X_te, y_te = make_data(seed)
n_tr=X_tr.shape[0]

for sigma in sigmas:
  fh_tr_krr=krr(X_tr,X_tr,y_tr,lbda,sigma, nu=nu)
  fh_te_krr=krr(X_te,X_tr,y_tr,lbda,sigma, nu=nu)
  data_dict['krr']['tr'][sigma]=r2(y_tr,fh_tr_krr)
  data_dict['krr']['te'][sigma]=r2(y_te,fh_te_krr)
  
  fh_kgd=kgd(np.vstack((X_tr,X_te)),X_tr,y_tr, step_size=step_size, nu=nu, sigma_min=sigma, t_max=t)
  data_dict['kgd']['tr'][sigma]=r2(y_tr,fh_kgd[:n_tr,:])
  data_dict['kgd']['te'][sigma]=r2(y_te,fh_kgd[n_tr:,:])

  fh_tr_kgf=kgf(X_tr,X_tr,y_tr,1/lbda,sigma, nu=nu)
  fh_te_kgf=kgf(X_te,X_tr,y_tr,1/lbda,sigma, nu=nu)
  data_dict['kgf']['tr'][sigma]=r2(y_tr,fh_tr_kgf)
  data_dict['kgf']['te'][sigma]=r2(y_te,fh_te_kgf)

fi=open('dd_data/dd_'+data+'_nu_seed_'+str(nu)+'_'+str(seed)+'.pkl','wb')
pickle.dump(data_dict,fi)
fi.close()
