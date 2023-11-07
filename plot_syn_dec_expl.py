import numpy as np
from kgd import kgd
from matplotlib import pyplot as plt
from help_fcts import r2, krr, kern, gcv, log_marg, trans_x
from matplotlib.lines import Line2D


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
  y_lim=[None,None]
  return x_tr, y_tr, x_te, y_te, lbda_bounds, sigma_bounds, y_lim



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
  y_lim=[-2,2]
  return x_tr, y_tr, x_te, y_te, lbda_bounds, sigma_bounds, y_lim


x2_ticks=trans_x(np.linspace(trans_x(20), trans_x(0.04), 10),inv=True)
nus = [0.5, 1.5, 2.5, 100, 10]
nu_titles=['$\\nu=1/2$ (Laplace)','$\\nu=3/2$','$\\nu=5/2$','$\\nu=\\infty$ (Gaussian)', 'Cauchy']

lines=[Line2D([0],[0],color='C7',lw=2),plt.plot(0,0,'ok')[0]]
plt.cla()
for c,lw in zip(['C2','C1', 'C3'],['-','--','--']):
  lines.append(Line2D([0],[0],color=c,lw=2))
labs=['True Function', 'Observed Data', 'Decreasing Bandwidth, KGD', 'Constant Bandwidth, GCV', 'Constant Bandwidth, MML']
  
fig,axs=plt.subplots(5,2,figsize=(10,10))
fig2,axs2=plt.subplots(5,2,figsize=(10,10))
seeds=[16,77]
for ii, (nu,nu_title) in enumerate(zip(nus, nu_titles)):
  for jj, (seed, make_data,x_ticks,data_title) in enumerate(zip(seeds, [make_data_lin_sin,make_data_two_freq],[[-3, 0, 3],[-2, -1, 0, 1]], ['Linear and Sine','Two Sines'])):
    x_tr, y_tr, x_te,y_te, lbda_bounds, sigma_bounds, y_lim=make_data(seed)
    n_tr=x_tr.shape[0]
    x_tr_te=np.vstack((x_tr,x_te))
    x_tr_te_argsort=x_tr_te.argsort(0)
    
    fhs_kgd, sigmas_kgd, r2s_kgd=kgd(x_tr_te,x_tr,y_tr, path=True, nu=nu,sigma0=100)
    fh_kgd=fhs_kgd[-1]
    lbda_gcv, sigma_gcv=gcv(x_tr,y_tr, lbda_bounds, sigma_bounds, n_lbdas=30, n_sigmas=30)
    fh_gcv=krr(x_tr_te,x_tr,y_tr,lbda_gcv,sigma_gcv, nu=nu)
    
    lbda_mml, sigma_mml=log_marg(x_tr,y_tr,[0.1*lbda_bounds[0], 10*lbda_bounds[1]],[0.1*sigma_bounds[0], 10*sigma_bounds[1]])
    fh_mml=krr(x_tr_te,x_tr,y_tr,lbda_mml,sigma_mml, nu=nu)
    
    ax=axs[ii,jj]
    ax.plot(x_te,y_te,'C7',lw=3)
    ax.plot(x_tr,y_tr,'ok')
    ax.plot(x_tr_te[x_tr_te_argsort,0],fh_kgd[x_tr_te_argsort,0],'C2',lw=2.5)
    ax.plot(x_tr_te[x_tr_te_argsort,0],fh_gcv[x_tr_te_argsort,0],'C1--',lw=2)
    ax.plot(x_tr_te[x_tr_te_argsort,0],fh_mml[x_tr_te_argsort,0],'C3--',lw=1.5)
    ax.set_ylim(y_lim)
    ax.set_yticks([-2,0,2])
    ax.set_xticks(x_ticks)

    if ii==0:
      ax.set_title(data_title, fontsize=12)
    if jj==0:
      ax.set_ylabel(nu_title, fontsize=12)
      
    fig.legend(lines, labs, loc='lower center', ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.09)
    fig.savefig('figures/syn_dec_expl.pdf')
  
    ax2=axs2[ii,jj]
    ax2.plot(sigmas_kgd,1-np.array(r2s_kgd))
    ax2.set_xscale('function',functions=(lambda x:1/(x+.1),lambda y:y))
    ax2.set_xlim([0.0399,1e10])
    ax2.set_xticks(x2_ticks)
    ax2.set_xlabel('$\\sigma$')
    ax2.set_ylabel('$1-R^2$')
    if ii==0:
      ax2.set_title(data_title, fontsize=12)
    if jj==0:
      ax2.set_ylabel(nu_title+'\n$1-R^2$')
   
    fig2.tight_layout()
    fig2.savefig('figures/syn_dec_expl2.pdf')

     
