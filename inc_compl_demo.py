import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import sys
sys.path.insert(1,'..')
from kgd import kgd
from help_fcts import r2, krr, cv10

anims=[139,240,779,-1]
lbdas=[13.5,8.2,0.55,1e-6]

LWS=[2.1,1.8,1.5,1.2]

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

sigma_bds=[0.01, 10]
lbda_bds=[1e-6, 1]


def f_sin(x):
  y=np.sin(2*np.pi*x)
  return y

nu=100

fig,axs=plt.subplots(4,1,figsize=(8,4))

labs = ['True Function', 'Noisy Observations','Decreasing Bandwidth', 'Constant Bandwidth']
lines=[Line2D([0],[0],color='C7',lw=3),plt.plot(0,0,'ok')[0],Line2D([0],[0],color='C2',lw=3),Line2D([0],[0],color='C1',lw=3)]
plt.cla()

seed=16
x_tr, y_tr, x_te,y_te=make_data_lin_sin(seed)
n_tr=x_tr.shape[0]
x_tr_te=np.vstack((x_tr,x_te))
x_tr_te_argsort=x_tr_te.argsort(0)

fhs_kgd, sigmas_kgd, r2s_kgd=kgd(x_tr_te,x_tr,y_tr, path=True, nu=nu, sigma_min=1e-4)
fh_kgd=fhs_kgd[-1]

lbda_cv, sigma_cv=cv10(x_tr,y_tr,lbda_bds,sigma_bds,seed,nu=nu)
fhs_nn=[]

for ii, (ax,fh_i,lbda) in enumerate(zip(axs,anims,lbdas)):
  fh_kgd=fhs_kgd[fh_i]
  fh_krr=krr(x_tr_te,x_tr,y_tr,lbda,sigma_cv, nu=nu)
  ax.cla()
  ax.plot(x_te,y_te,'C7',lw=LWS[0])
  ax.plot(x_tr,y_tr,'ok',ms=3)
  ax.plot(x_tr_te[x_tr_te_argsort,0],fh_krr[x_tr_te_argsort,0],'C1',lw=LWS[1])
  ax.plot(x_tr_te[x_tr_te_argsort,0],fh_kgd[x_tr_te_argsort,0],'C2',lw=LWS[2])
  ax.set_xlim([-3.05,3.05])
  ax.set_ylim([-2.2,2.2])
  ax.text(-3,1,'R$_{\\text{tr}}^2$='+f'{r2s_kgd[fh_i]:.1f}',fontsize=10)
  ax.set_yticks([])
  ax.set_xticks([])
  fig.savefig('figures/syn_'+str(nu)+'.pdf')


fig.legend(lines, labs, loc='lower center', ncol=len(labs),fontsize=9)
fig.tight_layout()
fig.subplots_adjust(bottom=.08)
fig.savefig('figures/inc_compl_demo.pdf')

