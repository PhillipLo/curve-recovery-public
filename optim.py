'''
optim.py

Contains optimization routines for curve recovery from moments.
'''

from data_generation import *
from moments import *
from subspace_recovery import *
from projection import *
from loss_functions import *
from jaxopt import GradientDescent


def estimate_C_from_subspaces(Chat_subspaces, Lhat_init, m1, m2, m3, nit = 100):
  '''
  Given estimated subspaces of where the vertices of C like, estimate the lengths of the
    vertices of C.

  INPUTS:
    Chat_subspaces: (M+1, d)-shaped array, Chat_subspaces[i] is a unit vector estimated to be
      pointing in the direction of C[i].
    Lhat_init: (M+1,)-shaped array, Lhat_init[i] is the initial guess of length of C[i]
    m1, m2, m3: (d,), (d,d), and (d, d, d)-shaped arrays, putative first, second, and third moments
      of underlying noise-free curve to estimate.
    nit: int, number of gradient descent operations to perform, default=100.

  OUTPUTS:
    Chat: (M+1, d)-shaped array, estimated vertices of curve.
  '''
  solver = GradientDescent(m123_loss_on_lengths)
  Lhat = Lhat_init
  state = solver.init_state(Lhat, Chat_subspaces = Chat_subspaces, m1 = m1, m2 = m2, m3 = m3)
  for _ in tqdm(range(nit), desc = "performing GD on vertex lengths", leave = False):
    Lhat, state = solver.update(Lhat, state, Chat_subspaces = Chat_subspaces, m1 = m1, m2 = m2, m3 = m3)  
  Chat = Lhat[:, jnp.newaxis] * Chat_subspaces

  return Chat


def finetune_C_with_moments(Chat, phat, m3, nit = 2500):
  '''
  Given initial guess Chat to a curve C with known constant segment lengths, estimate C by matching
    relaxed third moment with alternating descent on C and p

  INPUTS:
    Chat: (M+1, d)-shaped array, initial guess of curve points
    phat: (M,)-shaped array, initial guess of proportional segment lengths
    m3: (d, d, d)-shaped array, putative third moment of underlying noise-free curve to estimate.
    nit: int, number of gradient descent operations to perform, default 1e5

  OUTPUTS:
    Chat: (M+1, d)-shaped array, estimated vertices of curve.
    phat: (M,)-shaped array of predicted proportional segment lengths
  '''
  solver_C = GradientDescent(mu3_loss)
  state_C = solver_C.init_state(Chat, phat = phat, m3 = m3)
  solver_p = GradientDescent(lambda phat, Chat, m3: mu3_loss(Chat, phat, m3))
  state_p = solver_p.init_state(phat, Chat = Chat, m3 = m3)
  for _ in tqdm(range(nit), desc = "performing coordinate descent on C and p", leave = False):
    Chat, state_C = solver_C.update(Chat, state_C, phat = phat, m3 = m3)
    phat, state_p = solver_p.update(phat, state_p, Chat = Chat, m3 = m3)
  
  return Chat, phat


def estimate_C_baseline(seed, m1, m2, m3, M, num_trials = 100):
  '''
  Baseline estimation by gradient descent from random initialization.

  INPUTS:
    seed: int, for JAX rng
    Chat_subspaces: (M+1, d)-shaped array, Chat_subspaces[i] is a unit vector estimated to be
      pointing in the direction of C[i].
    m1, m2, m3: (d,), (d, d), (d, d, d)-shaped array, putative first through third moments of 
      underlying noise-free curve to estimate.
    M: positive int, number of segments to predict
    num_trials: positive int, number of random initial guesses to try

  OUTPUTS:
    best_Chat_m123: (M+1, d)-shaped array, estimated vertices of curve from using all three
      moments
    best_Chat_m3: (M+1, d)-shaped array, estimated vertices of curve from using third moment only.
  '''
  key, subkey = random.split(random.PRNGKey(seed))

  d = m1.shape[0]

  best_Chat_m123 = jnp.zeros(shape = (M+1, d))
  best_Chat_m3 = jnp.zeros(shape = (M+1, d))
  best_loss_m123 = jnp.inf
  best_loss_m3 = jnp.inf

  solver_m123 = GradientDescent(m123_loss)
  solver_m3 = GradientDescent(m3_loss)

  for _ in tqdm(range(num_trials), desc = "trialing multiple random initializations for baseline", leave = False): 
    Chat = random.normal(subkey, shape = (M+1, d))
    key, subkey = random.split(key)

    Chat_m123 = jnp.copy(Chat)

    Chat_m123, _ = solver_m123.run(Chat, m1 = m1, m2 = m2, m3 = m3)
    loss_m123 = m123_loss(Chat_m123, m1, m2, m3)
    Chat_m3, _ = solver_m3.run(Chat, m3 = m3)
    loss_m3 = m3_loss(Chat_m123, m3)

    if loss_m123 < best_loss_m123:
      best_loss_m123 = loss_m123
      best_Chat_m123 = Chat_m123

    if loss_m3 < best_loss_m3:
      best_loss_m3 = loss_m3
      best_Chat_m3 = Chat_m3
      
  return best_Chat_m123, best_Chat_m3
