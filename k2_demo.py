import numpy as np
from matplotlib import pyplot as plt
from help_fcts import make_data_real

def kern(X,Xp,sigma, nu=np.inf):
  X2=np.sum(X**2,1).reshape((-1,1))
  XXp=X.dot(Xp.T)
  Xp2=np.sum(Xp**2,1).reshape((-1,1))
  D2=X2-2*XXp+Xp2.T+1e-10
  D=np.sqrt(D2)
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



sigmas=np.geomspace(0.05,300,300)
data_sets=['compactiv','power','super','temp']
data_titles=['CPU Run Time', 'Tetouan Power Consumption', 'Superconductor Temperature', 'U.K. Temperature']

n=30
fig,axs=plt.subplots(1,4,figsize=(10,2.5))
for data, data_title,ax in zip(data_sets,data_titles,axs):
  k2s_tot=[]
  for seed in range(100):
    np.random.seed(seed)
    Xx, _, _, _ = make_data_real(data, seed,n_samps=n+1, frac=1, red_comp=False)
    X=Xx[:n,:]
    xs=Xx[-1,:].reshape(-1,1)
    k2s=[]
    for sigma in sigmas:
      k2s.append(np.sqrt(n)*np.linalg.norm(kern(X,xs.T,sigma)-np.mean(kern(X,X,sigma),0).reshape(-1,1)))
    
    k2s_tot.append(k2s)
  
  k2s_tot=np.array(k2s_tot)
  
  ax.plot(sigmas,np.median(k2s_tot,0),'C0-')
  ax.plot(sigmas,np.quantile(k2s_tot,0.25,0),'C0:')
  ax.plot(sigmas,np.quantile(k2s_tot,0.75,0),'C0:')
  ax.set_title(data_title,fontsize=10)
  ax.set_xscale('log')
  
  ax.set_xlabel('$\\sigma$')
  if data=='compactiv':
    ax.set_ylabel('$\\Delta k_2(\\sigma)$')
  fig.tight_layout()
  fig.savefig('figures/k2_demo.pdf')

