import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from kgd import kgd
from help_fcts import r2, krr, kgf


def f(x):
  y=np.sin(2*np.pi*x)
  return y

sigmas=np.array([5, 0.5, 0.1, 0.01])
nus=[0.5,1.5,2.5,100, 10]
nu_titles=['$\\nu=1/2$\n(Laplace)','$\\nu=3/2$','$\\nu=5/2$','$\\nu=\infty$\n(Gaussian)', 'Cauchy']
n_tr=20
n_plot=300
step_size=0.1
t=1e3
lbda=1e-3



seed=8
labs = ['True Function', 'Observed Data','Decreasing Bandwidth', 'Constant Bandwidth']
lines=[Line2D([0],[0],color='C7',lw=3),plt.plot(0,0,'ok')[0],Line2D([0],[0],color='C2',lw=3),Line2D([0],[0],color='C1',lw=3)]
plt.cla()

fig,axs=plt.subplots(5,4,figsize=(10,9))
np.random.seed(seed)
x_tr=np.random.uniform(-1,1,n_tr).reshape((-1,1))
n_tr=x_tr.shape[0]
y_tr=f(x_tr)+np.random.normal(0,.2,x_tr.shape)
x_plot=np.linspace(-1,1,n_plot).reshape((-1,1))
y_true=f(x_plot)
x_tot=np.vstack((x_tr,x_plot))
x_tot_argsort=x_tot.argsort(0)

for ii, (nu, nu_title) in enumerate(zip(nus, nu_titles)):
  for jj, sigma in enumerate(sigmas):
    ax=axs[ii,jj]
    if ii==0:
      ax.set_title('$\\sigma_m='+str(sigma)+'$', fontsize=12)
    if jj==0:
      ax.set_ylabel(nu_title, fontsize=12)
    ax.plot(x_plot, y_true, 'C7', lw=2.5)
    ax.plot(x_tr,y_tr,'ok')
    fh_krr=krr(x_tot,x_tr,y_tr,lbda,sigma, nu=nu)
    fh_kgf=kgf(x_tot,x_tr,y_tr,1/lbda,sigma, nu=nu)
    fh_kgd=kgd(x_tot,x_tr,y_tr, step_size=step_size, t_max=t, sigma_min=sigma, nu=nu)
    ax.plot(x_tot[x_tot_argsort,0],fh_kgf[x_tot_argsort,0], 'C6',lw=2)
    ax.plot(x_tot[x_tot_argsort,0],fh_krr[x_tot_argsort,0], 'C1',lw=2)
    ax.plot(x_tot[x_tot_argsort,0],fh_kgd[x_tot_argsort,0], 'C2',lw=1.5)
    ax.set_ylim([-1.5,1.5])
    fig.legend(lines, labs, loc='lower center', ncol=len(labs))
    fig.tight_layout()
    fig.subplots_adjust(bottom=.08)
    fig.savefig('figures/double_descent_expl.pdf')
      
