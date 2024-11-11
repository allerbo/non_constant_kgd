import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import pandas as pd
import warnings



def make_data_real(data, seed=None, n_samps=100):
  FRAC=0.8
  if data=='wood':
    dm_all=pd.read_csv('csv_files/wood-fibres.csv',sep=',').to_numpy()
    dm_all+=np.random.normal(0,0.001,dm_all.shape) #to break ties
  elif data=='compactiv':
    dm_all=pd.read_csv('csv_files/compactiv.csv',sep=',').to_numpy()
    dm_all=np.roll(dm_all,1,1)
    n_samps=82
  if data=='super':
    dm_all=pd.read_csv('csv_files/super.csv',sep=',').to_numpy()
    dm_all=np.roll(dm_all,1,1)
  elif data=='temp':
    dm_all=pd.read_csv('csv_files/uktemp.csv',sep=',').to_numpy()
  elif data=='power':
    dm_all=pd.read_csv('csv_files/power.csv',sep=',').iloc[:,1:].to_numpy()
    dm_all=np.roll(dm_all,1,1)

  np.random.seed(0)
  np.random.shuffle(dm_all)
  if data=='compactiv' and seed>92:
    dm=dm_all[(93+seed*(n_samps-1)):(93+(seed+1)*(n_samps-1)),:]
  else:
    dm=dm_all[seed*n_samps:(seed+1)*n_samps,:]
  
  X=dm[:,1:]
  X=(X-np.mean(X, 0))/np.std(X,0)
  n=X.shape[0]
  X_tr=X[:int(FRAC*n),:]
  X_te=X[int(FRAC*n):,:]
  
  y=dm[:,0].reshape((-1,1))
  y=y-np.mean(y)
  
  y_tr=y[:int(FRAC*n),:]
  y_te=y[int(FRAC*n):,:]
  
  return X_tr, y_tr, X_te, y_te



def r2(y,y_hat):
  if len(y.shape)==1:
    y=y.reshape((-1,1))
  if len(y_hat.shape)==1:
    y_hat=y_hat.reshape((-1,1))
  return 1-np.mean((y-y_hat)**2)/np.mean((y-np.mean(y))**2)

def krr(xs,x_tr,y_tr_in,lbda,sigma, nu=np.inf,center=True):
  y_tr_mean=np.mean(y_tr_in) if center else 0
  y_tr=y_tr_in-y_tr_mean
  Ks=kern(xs,x_tr,sigma, nu)
  K=kern(x_tr,x_tr,sigma, nu)
  return Ks@np.linalg.solve(K+lbda*np.eye(K.shape[0]),y_tr)+y_tr_mean

def kgf(xs,x_tr,y_tr_in,t,sigma, nu=np.inf,center=True):
  y_tr_mean=np.mean(y_tr_in) if center else 0
  y_tr=y_tr_in-y_tr_mean
  Ks=kern(xs,x_tr,sigma, nu)
  K=kern(x_tr,x_tr,sigma, nu)
  return Ks@np.linalg.inv(K+1e-10*np.eye(K.shape[0]))@(np.eye(K.shape[0])-expm(-t*K))@y_tr+y_tr_mean


def kern(X,Y,sigma, nu=np.inf):
  X2=np.sum(X**2,1).reshape((-1,1))
  XY=X.dot(Y.T)
  Y2=np.sum(Y**2,1).reshape((-1,1))
  D2=X2-2*XY+Y2.T
  D=np.sqrt(D2+1e-10)
  if nu==0.5:      #Laplace
    return np.exp(-D/sigma)
  elif nu==1.5:
    return (1+np.sqrt(3)*D/sigma)*np.exp(-np.sqrt(3)*D/sigma)
  elif nu==2.5:
    return (1+np.sqrt(5)*D/sigma+5*D2/(3*sigma**2))*np.exp(-np.sqrt(5)*D/sigma)
  elif nu==10:     #Cauchy (could have been any number, but I chose 10 for no particular reason)
    return 1/(1+D2/sigma**2)
  else:            #Gaussian
    return np.exp(-0.5*D2/sigma**2)

def trans_x(x,inv=False, shift=.01,exp=5):
  if inv:
    return 1/x**exp-shift
  return 1/(x+shift)**(1/exp)


def gcv(X,y_in,lbda_bds,sigma_bds, nu=np.inf,center=True, rand=True):
  lbdas=np.geomspace(lbda_bds[0], lbda_bds[1], 100)
  sigmas=np.geomspace(sigma_bds[0], sigma_bds[1], 100)
  y=y_in-np.mean(y_in) if center else y_in
  n=X.shape[0]
  cvs=[]
  lbdas_sigmas=[]
  for lbda in lbdas:
    for sigma in sigmas:
      sigma1=np.random.uniform(0.9,1.1)*sigma if rand else sigma
      lbda1=np.random.uniform(0.9,1.1)*lbda if rand else lbda
      K_l=kern(X,X,sigma1,nu)+lbda1*np.eye(n)
      K_li=np.linalg.inv(K_l)
      cvs.append(np.mean((K_li@y/np.diag(K_li).reshape((-1,1)))**2))
      lbdas_sigmas.append([lbda1, sigma1])
  return lbdas_sigmas[np.argmin(cvs)]

def mml(x,y_in,lbda_bds,sigma_bds, nu=np.inf):
  warnings.filterwarnings('ignore')
  np.random.seed(0)
  y=y_in-np.mean(y_in)
  n=x.shape[0]
  def log_marg_fn(args):
    Kl=kern(x,x,args[1],nu)+args[0]*np.eye(n)
    return (y.T@np.linalg.solve(Kl,y) + np.log(np.linalg.det(Kl)))[0][0]
  
  best_fun=np.inf
  best_lbda=np.inf
  best_sigma=np.inf
  for lbda_seed in np.geomspace(1.1*lbda_bds[0],0.9*lbda_bds[1],5):
    for sigma_seed in np.geomspace(1.1*sigma_bds[0],0.9*sigma_bds[1],5):
      lbda_seed1=lbda_seed*np.random.uniform(0.91,1.09)
      sigma_seed1=sigma_seed*np.random.uniform(0.91,1.09)
      res = minimize(log_marg_fn, [lbda_seed1, sigma_seed1], bounds=[lbda_bds,sigma_bds])
      if res.success and res.fun<best_fun:
        best_fun=res.fun
        best_lbda=res.x[0]
        best_sigma=res.x[1]
  return best_lbda, best_sigma

