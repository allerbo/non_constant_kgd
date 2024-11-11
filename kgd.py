import numpy as np

def kgd(Xs,X_tr,y_tr_in,sigma0=None,step_size=0.01,v_R2=0.1,sigma_min=1e-3,t_max=1e3,path=False,nu=np.inf):
  if path:
    sigmas=[]
    r2s=[]
    fhs=[]
  
  Xs_2=np.sum(Xs**2,1).reshape((-1,1))
  XsX_tr=Xs.dot(X_tr.T)
  X_tr_2=np.sum(X_tr**2,1).reshape((-1,1))
  D2=Xs_2-2*XsX_tr+X_tr_2.T
  D=np.sqrt(D2+1e-10)
  
  X_trX_tr=X_tr.dot(X_tr.T)
  D_tr2=X_tr_2-2*X_trX_tr+X_tr_2.T
  D_tr=np.sqrt(D_tr2+1e-10)
  
  n_tr=X_tr.shape[0]
  ns=Xs.shape[0]
  Ih=np.hstack((np.eye(n_tr),np.zeros((n_tr,ns-n_tr))))
  y_tr_mean=np.mean(y_tr_in)
  y_tr=y_tr_in-y_tr_mean
  
  get_dR2dt = lambda y_tr, fh, K_tr: 2*(y_tr-fh).T@K_tr@(y_tr-fh)/(y_tr.T@y_tr)
  
  if nu==0.5:
    kern = lambda D, sigma: np.exp(-D/sigma)
  elif nu==1.5:
    kern = lambda D, sigma: (1+np.sqrt(3)*D/sigma)*np.exp(-np.sqrt(3)*D/sigma)
  elif nu==2.5:
    kern = lambda D, sigma: (1+np.sqrt(5)*D/sigma+5*D**2/(3*sigma**2))*np.exp(-np.sqrt(5)*D/sigma)
  elif nu==10:
    kern = lambda D, sigma: 1/(1+D**2/sigma**2)
  else:
    kern = lambda D, sigma: np.exp(-0.5*D**2/sigma**2)
  
  sigma=np.max(D) if sigma0 is None else sigma0
  sigma=np.maximum(sigma,sigma_min)
  Ks=kern(D,sigma)
  fh=np.zeros((ns,1))
  for i in range(int(t_max/step_size)):
    fh-= step_size*Ks@(Ih@fh-y_tr)
    r2=(1-np.mean((y_tr-Ih@fh)**2)/np.mean(y_tr**2))
    if path:
      sigmas.append(sigma)
      r2s.append(r2)
      fhs.append(fh+y_tr_mean)

    dR2dt=get_dR2dt(y_tr, Ih@fh,kern(D_tr,sigma))
    if dR2dt<v_R2 and sigma>sigma_min:
      while sigma>sigma_min:
        sigma=sigma/(1+step_size)
        dR2dt=get_dR2dt(y_tr, Ih@fh,kern(D_tr,sigma))
        if dR2dt>=v_R2:
          break
      Ks=kern(D,sigma)
    if r2>0.999:
      break
  if path:
    return fhs, sigmas, r2s
  return fh+y_tr_mean




