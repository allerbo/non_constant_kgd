import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import sys
sys.path.insert(1,'..')
from kgd import kgd
from help_fcts import r2, krr


def f(x):
  y=np.sin(2*np.pi*x)
  return y

sigmas=np.array([1, 0.5, 0.1, 0.01])
nus=[100, 0.5,1.5,2.5,10]
nu_titles=['$\\nu=\\infty$ (Gaussian)', '$\\nu=1/2$ (Laplace)','$\\nu=3/2$','$\\nu=5/2$','Cauchy']
n_tr=20
n_plot=300
lbda=1e-3


seed=8
labs = ['True Function', 'Observed Data','Decreasing Bandwidth', 'Constant Bandwidth']
lines=[Line2D([0],[0],color='C7',lw=3),plt.plot(0,0,'ok')[0],Line2D([0],[0],color='C2',lw=3),Line2D([0],[0],color='C1',lw=3)]
plt.cla()

np.random.seed(seed)
x_tr=np.random.uniform(-1,1,n_tr).reshape((-1,1))
n_tr=x_tr.shape[0]
y_tr=f(x_tr)+np.random.normal(0,.2,x_tr.shape)
x_plot=np.linspace(-1,1,n_plot).reshape((-1,1))
y_true=f(x_plot)
x_tot=np.vstack((x_tr,x_plot))
x_tot_argsort=x_tot.argsort(0)

fig1,axs1=plt.subplots(2,2,figsize=(10,4))
fig2,axs2=plt.subplots(4,4,figsize=(10,6))
axs=np.hstack((axs1.ravel(),axs2.ravel()))
ii=0
for nu, nu_title in zip(nus, nu_titles):
  for sigma in sigmas:
    if nu<100 and sigma==1:
      sigma=5.0
    ax=axs[ii]
    ax.plot(x_plot, y_true, 'C7', lw=2.5)
    ax.plot(x_tr,y_tr,'ok')
    if ii<8:
      ax.set_title('$\\sigma_m='+str(sigma)+'$')
    if ii % 4==0 and ii>0:
      ax.set_ylabel(nu_title)
    fh_krr=krr(x_tot,x_tr,y_tr,lbda,sigma, nu=nu)
    fh_kgd=kgd(x_tot,x_tr,y_tr, t_max=1/lbda, sigma_min=sigma, nu=nu)
    ax.plot(x_tot[x_tot_argsort,0],fh_krr[x_tot_argsort,0], 'C1',lw=2)
    ax.plot(x_tot[x_tot_argsort,0],fh_kgd[x_tot_argsort,0], 'C2',lw=1.5)
    ax.set_ylim([-1.5,1.5])
    if nu==100:
      ax.set_xticks([-1,-.5,0,.5,1])
    else:
      ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])
    ii+=1
    fig1.legend(lines, labs, loc='lower center', ncol=len(labs))
    fig2.legend(lines, labs, loc='lower center', ncol=len(labs))
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.subplots_adjust(bottom=.16)
    fig2.subplots_adjust(bottom=.11)
    fig1.savefig('figures/syn_dd_100.pdf')
    fig2.savefig('figures/syn_dd_more.pdf')



