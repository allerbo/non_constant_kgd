import numpy as np
import sys
from kgd import kgd
from help_fcts import r2, krr, gcv, log_marg, kgf
import pickle


def make_data_lin_sin(seed=None):
  if not seed is None:
    np.random.seed(seed)
  N_TR=100
  N_TE=500
  LIN_LIM=1
  FREQ1=5
  def fy(x):
    return np.sin(FREQ1*2*np.pi*x)*(np.abs(x)<LIN_LIM)+(x+LIN_LIM)*(x<-LIN_LIM)+(x-LIN_LIM)*(x>LIN_LIM)
  x_tr=np.random.normal(0,1,N_TR).reshape((-1,1))
  y_tr=fy(x_tr)+np.random.normal(0,.2,x_tr.shape)
  x_te=np.linspace(-3,3, N_TE).reshape((-1,1))
  y_te=fy(x_te)
  lbda_bounds=[1e-6,1]
  sigma_bounds=[0.01,10]
  return x_tr, y_tr, x_te, y_te, lbda_bounds, sigma_bounds



def make_data_two_freq(seed=None):
  if not seed is None:
    np.random.seed(seed)
  FREQ1=1
  FREQ2=8
  X_MIN=-2
  X_MAX=1
  OBS_FREQ=10
  N_TE=500
  def fy(x):
    return np.sin(FREQ1*2*np.pi*x)*(x<0)+np.sin(FREQ2*2*np.pi*x)*(x>0)
  x_tr1=np.random.uniform(X_MIN,0,-X_MIN*FREQ1*OBS_FREQ).reshape((-1,1))
  y_tr1=fy(x_tr1)+np.random.normal(0,.2,x_tr1.shape)
  x_tr2=np.random.uniform(0,X_MAX,int(X_MAX*FREQ2*OBS_FREQ)).reshape((-1,1))
  y_tr2=fy(x_tr2)+np.random.normal(0,.2,x_tr2.shape)
  x_tr=np.vstack((x_tr1,x_tr2))
  y_tr=np.vstack((y_tr1,y_tr2))
  x_te=np.linspace(X_MIN,X_MAX, N_TE).reshape((-1,1))
  y_te=fy(x_te)
  lbda_bounds=[1e-6,1]
  sigma_bounds=[0.01,0.1]
  return x_tr, y_tr, x_te, y_te, lbda_bounds, sigma_bounds

seed=0
data='lin_sin'
nu=0.5

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

if data == 'lin_sin':
  make_data=make_data_lin_sin
elif data == 'two_freq':
  make_data=make_data_two_freq
     
r2_dict={'kgd':[],'gcv':[],'mml':[],'gcv_kgf':[],'mml_kgf':[]}
x_tr, y_tr, x_te, y_te, lbda_bounds, sigma_bounds=make_data(seed)
n_tr=x_tr.shape[0]
x_tr_te=np.vstack((x_tr,x_te))

fh_kgd=kgd(x_tr_te,x_tr,y_tr,nu=nu)
lbda_gcv, sigma_gcv=gcv(x_tr,y_tr, lbda_bounds, sigma_bounds, n_lbdas=30, n_sigmas=30, nu=nu)
fh_krr_gcv=krr(x_tr_te,x_tr,y_tr,lbda_gcv,sigma_gcv)
fh_kgf_gcv=kgf(x_tr_te,x_tr,y_tr,1/lbda_gcv,sigma_gcv)

lbda_mml, sigma_mml=log_marg(x_tr,y_tr, [0.1*lbda_bounds[0], 10*lbda_bounds[1]],[0.1*sigma_bounds[0], 10*sigma_bounds[1]], nu=nu)
fh_krr_mml=krr(x_tr_te,x_tr,y_tr,lbda_mml,sigma_mml)
fh_kgf_mml=kgf(x_tr_te,x_tr,y_tr,1/lbda_mml,sigma_mml)

r2_dict['kgd'].append(r2(y_te,fh_kgd[n_tr:])) #TODO: should not be list. just one element ?
r2_dict['gcv'].append(r2(y_te,fh_krr_gcv[n_tr:]))
r2_dict['mml'].append(r2(y_te,fh_krr_mml[n_tr:]))
r2_dict['gcv_kgf'].append(r2(y_te,fh_kgf_gcv[n_tr:]))
r2_dict['mml_kgf'].append(r2(y_te,fh_kgf_mml[n_tr:]))


fi=open('dec_data/'+data+'_dec_nu_seed_'+str(nu)+'_'+str(seed)+'.pkl','wb')
pickle.dump(r2_dict,fi)
fi.close()
