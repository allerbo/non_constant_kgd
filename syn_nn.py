import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import sys
from ffnn import  init_model, train_step, get_R2
from help_fcts_ntk import init_theta, inc_beta, train_step_gd, train_step_pgd, get_R2_ntk,  f_hat, get_Pp
import time

labs = ['True Function', 'Noisy Observations','Predictions, Neural Network']
lines=[Line2D([0],[0],color='C7',lw=3),plt.plot(0,0,'ok')[0],Line2D([0],[0],color='C3',lw=3)]
plt.cla()

def f_sin(x):
  y=np.sin(2*np.pi*x)
  return y

DIM_HS=[1,3,200,200]
N_LAYS=[1,1,2,4]

plt.cla()

fig,axs=plt.subplots(1,5,figsize=(8,2))

n_tr=20
n_plot=101

seed=8

np.random.seed(seed)
x_tr=np.random.uniform(-1,1,n_tr).reshape((-1,1))
n_tr=x_tr.shape[0]
y_tr=f_sin(x_tr)+np.random.normal(0,.2,x_tr.shape)
x_te=np.linspace(-1,1,n_plot).reshape((-1,1))
y_te=f_sin(x_te)
x_tr_te=np.vstack((x_tr,x_te))
x_tr_te_argsort=x_tr_te.argsort(0)

def count_params(ms):
  pars=0
  for p1 in ms.params['params'].items():
    for p2 in p1[1].items():
      pars+=len(p2[1])
  return pars

def num2str(x):
  e=0
  while x>=10:
    x*=0.1
    e+=1
  return f'${x:.1f}'+'\\cdot 10^'+str(e)+'$'
  
gamma=0.95
for ii, ax in enumerate(axs[:-1]):
  dt=0.001
  model_state = init_model(1, DIM_HS[ii], 1, dt, gamma, n_lay=N_LAYS[ii],seed=6)
  t=0
  r2_tr_old=-np.inf
  epoch=0
  while epoch<10000000:
    epoch+=1
    model_state = train_step(model_state, x_tr, y_tr)
    if epoch <10000 and epoch % 100==0 or epoch<100000 and epoch % 1000 ==0 or epoch<1000000 and epoch % 10000 ==0 or epoch % 100000==0:
      fh_tr=model_state.apply_fn(model_state.params,x_tr)
      fh_te=model_state.apply_fn(model_state.params,x_te)
      ax.cla()
      _=ax.plot(x_te, y_te, 'C7', lw=2.1)
      _=ax.plot(x_tr,y_tr,'ok', ms=3)
      _=ax.plot(x_te,fh_te,'C3',lw=1.2)
      ax.set_title(f'p={count_params(model_state)}, '+'$R^2_{\\text{tr}}=$'+f'{get_R2(fh_tr, y_tr):.2f}'+'\nN$_\\text{e}=$'+num2str(epoch)+', $R^2_{\\text{te}}=$'+f'{get_R2(fh_te, y_te):.2f}' ,fontsize=8)
      ax.set_ylim([-1.5,1.5])
      ax.set_xticks([-1,-.5,0,.5,1])
      ax.set_yticks([-1,0,1])
      ax.set_xticks([])
      ax.set_yticks([])
      fig.tight_layout()
      fig.savefig('figures/syn_nn.pdf')
      if get_R2(fh_tr,y_tr)>0.999:
        break

theta=init_theta(1,1,50)
epoch=0
beta=0
Pp=None
use_P=False
dt=0.001
ALG='pgd'
r2_tr_old=0
ax=axs[4]
x_val=x_te
while 1:
  if epoch % 100 ==0:
    beta, Pp, use_P = inc_beta(x_tr, y_tr, x_val, theta, beta, Pp,use_P)

  if use_P:
    theta = train_step_pgd(x_tr, y_tr, theta, Pp, dt)
  else:
    theta = train_step_gd(x_tr, y_tr, theta, dt)
  if use_P:
    r2_tr=get_R2_ntk(x_tr,y_tr,theta)
    if r2_tr<r2_tr_old:
      Pp=get_Pp(x_tr, x_val, theta, beta)
    r2_tr_old=r2_tr
  if epoch % 1000 == 0 or get_R2_ntk(x_tr,y_tr,theta)>0.999:
    fh_te=f_hat(x_te,theta)
    fh_val=f_hat(x_val,theta)
    _=ax.cla()
    _=ax.plot(x_te,y_te, 'C7',lw=2.1)
    _=ax.plot(x_tr,y_tr,'ok',ms=3)
    _=ax.plot(x_te,fh_te, 'C3', lw=1.2)
    ax.set_title(f'p={len(theta)}, '+'$R^2_{\\text{tr}}=$'+f'{get_R2_ntk(x_tr, y_tr, theta):.2f}'+'\nN$_\\text{e}=$'+num2str(epoch)+', $R^2_{\\text{te}}=$'+f'{get_R2_ntk(x_te, y_te, theta):.2f}' ,fontsize=8)
    ax.set_ylim([-1.5,1.5])
    ax.set_xticks([-1,-.5,0,.5,1])
    ax.set_yticks([-1,0,1])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig('figures/syn_nn.pdf')
    fig.tight_layout()
    if get_R2_ntk(x_tr,y_tr,theta)>0.999:
      break
  epoch+=1


fig.legend(lines, labs, loc='lower center', ncol=len(labs),fontsize=9)
fig.add_artist(plt.Line2D([0.795,0.795],[0.17,.88], transform=fig.transFigure,color='black'))
fig.subplots_adjust(bottom=.17)
fig.savefig('figures/syn_nn.pdf')

