import jax
import jax.numpy as jnp
from jax import grad, jit
from flax import linen as nn
import optax
from flax.training.train_state import TrainState


def init_model(DIM_X, DIM_H, DIM_Y, dt, gamma, n_lay=2, seed=2):
  rng, init_rng = jax.random.split(jax.random.PRNGKey(seed), 2)
  if n_lay==1:
    model=reg_1lay(DIM_H, DIM_Y)
  elif n_lay==2:
    model=reg_2lay(DIM_H, DIM_Y)
  else:
    model=reg_4lay(DIM_H, DIM_Y)
  theta=model.init(init_rng,jnp.ones((5,DIM_X)))
  opt = optax.inject_hyperparams(optax.sgd)(learning_rate=dt, momentum=gamma)
  model_state = TrainState.create(apply_fn=model.apply, params=theta, tx=opt)
  return model_state

class reg_4lay(nn.Module):
  DIM_H: int
  DIM_Y: int
  @nn.compact
  def __call__(self,x):
    x=nn.Dense(self.DIM_H)(x)
    x=nn.activation.tanh(x)
    x=nn.Dense(self.DIM_H)(x)
    x=nn.activation.tanh(x)
    x=nn.Dense(self.DIM_H)(x)
    x=nn.activation.tanh(x)
    x=nn.Dense(self.DIM_H)(x)
    x=nn.activation.tanh(x)
    x=nn.Dense(self.DIM_Y, kernel_init=nn.initializers.zeros)(x)
    return x

class reg_2lay(nn.Module):
  DIM_H: int
  DIM_Y: int
  @nn.compact
  def __call__(self,x):
    x=nn.Dense(self.DIM_H)(x)
    x=nn.activation.tanh(x)
    x=nn.Dense(self.DIM_H)(x)
    x=nn.activation.tanh(x)
    x=nn.Dense(self.DIM_Y, kernel_init=nn.initializers.zeros)(x)
    return x

class reg_1lay(nn.Module):
  DIM_H: int
  DIM_Y: int
  @nn.compact
  def __call__(self,x):
    x=nn.Dense(self.DIM_H)(x)
    x=nn.activation.tanh(x)
    x=nn.Dense(self.DIM_Y, kernel_init=nn.initializers.zeros)(x)
    return x

@jit
def train_step(model_state, x, y):
  def L2(theta):
    fh = model_state.apply_fn(theta, x)
    return 0.5*jnp.mean((fh-y)**2)
  
  loss, grads = jax.value_and_grad(L2)(model_state.params)
  model_state = model_state.apply_gradients(grads=grads)
  return model_state

@jit
def get_R2(fh,y):
  y_fh=y-fh
  y_ym=y-jnp.mean(y)
  return 1-jnp.squeeze((y_fh.T@y_fh)/(y_ym.T@y_ym))


