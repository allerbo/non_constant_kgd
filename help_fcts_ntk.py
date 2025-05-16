import os, sys
import jax
jax.config.update('jax_platform_name', 'cpu')
import numpy as np
import jax.numpy as jnp
from jax import grad, jacrev, jit

def init_theta(DIM_X, DIM_Y, DIM_H):
  np.random.seed(0)
  u1=np.sqrt(6/(DIM_X+DIM_H))
  u2=np.sqrt(6/(DIM_H+DIM_Y))
  
  theta=jnp.array(np.hstack((
    np.random.uniform(-u1,u1,[DIM_X, DIM_H]).flatten(), #W1
    np.zeros([DIM_H]).flatten(), #b1
    np.random.uniform(-u2,u2,[DIM_H, DIM_Y]).flatten(), #W2
    np.zeros([DIM_Y]).flatten() #b2
  )))
  
  lengths=np.array([DIM_X*DIM_H, DIM_H, DIM_H*DIM_Y, DIM_Y])
  global lims, dim_x, dim_y, dim_h
  lims=np.hstack(([0],np.cumsum(lengths)))
  dim_x=DIM_X
  dim_y=DIM_Y
  dim_h=DIM_H
  
  return theta

@jit
def f_hat(x,theta):
  W1=theta[lims[0]:lims[1]].reshape((dim_x,dim_h))
  b1=theta[lims[1]:lims[2]].reshape((dim_h))
  W2=theta[lims[2]:lims[3]].reshape((dim_h,dim_y))
  b2=theta[lims[3]:lims[4]].reshape((dim_y))
  h1=jnp.tanh(x@W1+b1)
  return h1@W2+b2

@jit
def get_R2_ntk(x,y,theta):
  y_fh=y-f_hat(x,theta)
  y_ym=y-jnp.mean(y)
  return 1-jnp.squeeze((y_fh.T@y_fh)/(y_ym.T@y_ym))

@jit
def L2(x,y,theta):
  fh=f_hat(x,theta)
  return jnp.mean((y-fh)**2)

@jit
def train_step_pgd(x,y,theta, Pp, dt=0.001):
  fh=f_hat(x,theta)
  dtheta_dt=jnp.squeeze(Pp@(y-fh))
  return theta+dt*dtheta_dt

@jit
def train_step_gd(x,y,theta, dt=0.001):
  dtheta_dt=-grad(L2, argnums=2)(x,y,theta)
  return theta+dt*dtheta_dt


def inc_beta(x_tr, y_tr, x_val, theta, beta, Pp, use_P):
  if beta<0.999:
    alpha_c, beta_c= get_alpha_beta_consts(x_tr,y_tr, x_val,theta, beta)
    while beta<.999 and get_dR2dt_consts(alpha_c, beta_c, beta)<0.1:
      beta+=0.001
      dt=0.001
      use_P=True
   
  if beta>0:
    use_P=True
   
  if use_P:
    Pp=get_Pp(x_tr, x_val, theta, beta)
  return beta, Pp, use_P


@jit
def get_dR2dt_consts(alpha_c, beta_c, beta):
  return (1-beta)*alpha_c+beta*beta_c

@jit
def get_alpha_beta_consts(x_tr,y_tr, x_val,theta, beta):
  Ph=jnp.squeeze(jacrev(f_hat,argnums=1)(jnp.vstack((x_tr,x_val)),theta))
  K=Ph@Ph.T
  
  y_fh=y_tr-f_hat(x_tr,theta)
  y_ym=y_tr-jnp.mean(y_tr)
  
  I0=jnp.vstack((jnp.eye(x_tr.shape[0]), jnp.zeros((x_val.shape[0],x_tr.shape[0]))))
  v=I0@y_fh
  c=2/(y_ym.T@y_ym)
  return c*v.T@K@v, c*v.T@v

@jit
def get_Pp(x_tr, x_val,theta,beta,eps=0.0005):
  Pht=jnp.squeeze(jacrev(f_hat,argnums=1)(x_tr,theta))
  Phv=jnp.squeeze(jacrev(f_hat,argnums=1)(x_val,theta))
  Ph=jnp.vstack((Pht,Phv))
  K=Ph@Ph.T
  
  aKb=(1-beta)*K+beta*jnp.eye(K.shape[0])
  
  U,s,Vt=jnp.linalg.svd(Ph,full_matrices=False)
  Phli=jnp.linalg.pinv(Ph+eps*U@Vt,rcond=1e-15)
  return Phli@aKb@Phli.T@Pht.T

def get_dR2dt(x,y,theta, Pp, use_P):
  if use_P:
    return get_dR2dt_P(x,y,theta, Pp)
  else:
    return get_dR2dt_nP(x,y,theta)

@jit
def get_dR2dt_nP(x,y,theta):
  Ph=jnp.squeeze(jacrev(f_hat,argnums=1)(x,theta))
  K=Ph@Ph.T
  y_fh=y-f_hat(x,theta)
  y_ym=y-jnp.mean(y)
  return jnp.squeeze((2*y_fh.T@K@y_fh)/(y_ym.T@y_ym))

@jit
def get_dR2dt_P(x,y,theta, Pp):
  Ph=jnp.squeeze(jacrev(f_hat,argnums=1)(x,theta))
  K=Ph@Pp
  y_fh=y-f_hat(x,theta)
  y_ym=y-jnp.mean(y)
  return jnp.squeeze((2*y_fh.T@K@y_fh)/(y_ym.T@y_ym))

