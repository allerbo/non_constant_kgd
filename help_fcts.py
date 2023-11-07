import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm

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
  return Ks@np.linalg.pinv(K)@(np.eye(K.shape[0])-expm(-t*K))@y_tr+y_tr_mean


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

def trans_x(x,inv=False, shift=0.1):
  if inv:
    return 1/x-shift
  return 1/(x+shift)

def gcv(X,y_in,lbda_bounds,sigma_bounds, n_lbdas=30, n_sigmas=30, nu=np.inf,center=True, rand=True):
  lbdas=np.geomspace(*lbda_bounds,n_lbdas)
  sigmas=np.geomspace(*sigma_bounds,n_sigmas)
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

def log_marg(x,y_in,lbda_bounds,sigma_bounds, nu=np.inf):
  np.random.seed(0)
  y=y_in-np.mean(y_in)
  n=x.shape[0]
  def log_marg_fn(args):
    Kl=kern(x,x,args[1],nu)+args[0]*np.eye(n)
    return (y.T@np.linalg.solve(Kl,y) + np.log(np.linalg.det(Kl)))[0][0]
  
  best_fun=np.inf
  best_lbda=np.inf
  best_sigma=np.inf
  for lbda_seed in np.geomspace(1.1*lbda_bounds[0],0.9*lbda_bounds[1],3):
    for sigma_seed in np.geomspace(1.1*sigma_bounds[0],0.9*sigma_bounds[1],3):
      lbda_seed1=lbda_seed*np.random.uniform(0.91,1.09)
      sigma_seed1=sigma_seed*np.random.uniform(0.91,1.09)
      res = minimize(log_marg_fn, [lbda_seed1, sigma_seed1], bounds=[lbda_bounds,sigma_bounds])
      if res.success and res.fun<best_fun:
        best_fun=res.fun
        best_lbda=res.x[0]
        best_sigma=res.x[1]
  return best_lbda, best_sigma

