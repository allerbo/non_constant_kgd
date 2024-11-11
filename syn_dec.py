import numpy as np
import sys
from kgd import kgd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from help_fcts import r2, krr, kern, gcv, mml, trans_x
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter

sigma_bds=[0.01, 10]
lbda_bds=[1e-6, 1]

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
  return x_tr, y_tr, x_te, y_te



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
  return x_tr, y_tr, x_te, y_te


x2_ticks=trans_x(np.linspace(trans_x(5,exp=1, shift=0.1), trans_x(0.01,exp=1, shift=0.1), 10),inv=True,exp=1, shift=0.1)
nus = [100, 0.5, 1.5, 2.5, 10]
nu_titles=['$\\nu=\\infty$ (Gaussian)','$\\nu=1/2$ (Laplace)','$\\nu=3/2$','$\\nu=5/2$', 'Cauchy']

lines=[Line2D([0],[0],color='C7',lw=2),plt.plot(0,0,'ok')[0]]
plt.cla()
for c,ls in zip(['C2','C1', 'C4'],['-','--','--']):
  lines.append(Line2D([0],[0],color=c,ls=ls))

labs=['True Function', 'Observed Data', 'KGD-D', 'KRR-GCV', 'KRR-MML']
seeds=[16,77]
data_titles=['Linear and Sine','Two Sines']
anims=[[50,200,400,800,-1],[50,300,500,800,-1]]


fig1,axs1=plt.subplots(2,2,figsize=(10,4))
fig2,axs2=plt.subplots(4,2,figsize=(10,6.5))
fig3,axs3=plt.subplots(5,2,figsize=(10,6.5))
axs=np.hstack((axs1.ravel(),axs2.ravel()))
ii=0
for nu,nu_title in zip(nus, nu_titles):
  for seed, make_data,x_ticks,data_title in zip(seeds, [make_data_lin_sin,make_data_two_freq],[[-3, 0, 3],[-2, -1, 0, 1]], data_titles):
    x_tr, y_tr, x_te,y_te=make_data(seed)
    n_tr=x_tr.shape[0]
    x_tr_te=np.vstack((x_tr,x_te))
    x_tr_te_argsort=x_tr_te.argsort(0)
    
    fhs_kgd, sigmas_kgd, r2s_kgd, _=kgd(x_tr_te,x_tr,y_tr, path=True, nu=nu, sigma_min=1e-4)
    fh_kgd=fhs_kgd[-1]
    lbda_gcv, sigma_gcv=gcv(x_tr,y_tr, lbda_bds, sigma_bds,nu=nu)
    fh_gcv, _=krr(x_tr_te,x_tr,y_tr,lbda_gcv,sigma_gcv, nu=nu)
    
    lbda_mml, sigma_mml=mml(x_tr,y_tr,lbda_bds,sigma_bds,nu=nu)
    fh_mml, _=krr(x_tr_te,x_tr,y_tr,lbda_mml,sigma_mml, nu=nu)
    
    ax=axs[ii]
    ax.plot(x_te,y_te,'C7',lw=2.5)
    ax.plot(x_tr_te[x_tr_te_argsort,0],fh_kgd[x_tr_te_argsort,0],'C2',lw=2.5)
    ax.plot(x_tr_te[x_tr_te_argsort,0],fh_gcv[x_tr_te_argsort,0],'C1--',lw=1.1)
    ax.plot(x_tr_te[x_tr_te_argsort,0],fh_mml[x_tr_te_argsort,0],'C4--',lw=1)
    ax.plot(x_tr,y_tr,'ok',ms=2)
    ax.set_ylim([-2.05,2.05])
    ax.set_yticks([-2,0,2])
    ax.set_xticks(x_ticks)
    if nu in [100,0.5]:
      ax.set_title(data_title)
      
    if nu==100:
      ax2=axs[ii+2]
      ax2.plot(sigmas_kgd,1-np.array(r2s_kgd))
      ax2.set_xscale('function',functions=(lambda x:1/(x+.1),lambda y:y))
      ax2.set_xlim([5e-3,10])
      ax2.set_xticks(x2_ticks)
      ax2.set_xlabel('$\\sigma$')
      ax2.set_ylabel('$1-R^2$')
      ax2.set_ylabel('\n$1-R^2$')
      ax2.xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
      for ax,fh_i in zip(axs3[:,ii],anims[ii]):
        fh_kgd=fhs_kgd[fh_i]
        ax.plot(x_te,y_te,'C7',lw=2.5)
        ax.plot(x_tr_te[x_tr_te_argsort,0],fh_kgd[x_tr_te_argsort,0],'C2',lw=2.5)
        ax.plot(x_tr,y_tr,'ok',ms=2)
        ax.set_ylim([-2.05,2.05])
        ax.set_yticks([])
        ax.set_xticks([])
      axs3[0,ii].set_title(data_titles[ii])
    
    if ii % 2==0 and ii>3:
      ax.set_ylabel(nu_title)
    
    ii+=1
    if ii==2:
      ii=4
    
    fig1.legend(lines, labs, loc='lower center', ncol=len(labs))
    fig2.legend(lines, labs, loc='lower center', ncol=len(labs))
    fig3.legend(lines[:3], labs[:3], loc='lower center', ncol=3)
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig1.subplots_adjust(bottom=0.2)
    fig2.subplots_adjust(bottom=0.1)
    fig3.subplots_adjust(bottom=0.06)
    fig1.savefig('figures/syn_dec_100.pdf')
    fig2.savefig('figures/syn_dec_more.pdf')
    fig3.savefig('figures/syn_dec_anim.pdf')


