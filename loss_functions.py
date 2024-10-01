'''
loss_functions.py

Loss functions for moment-based optimization.
'''
from moments import *
from relaxed_moments import *
from data_generation import *


@jit
def m123_loss_on_lengths(Lhat, Chat_subspaces, m1, m2, m3):
  '''
  Compute the squared Frobenius norm between a given first moment and the first moment coming from
    a curve Chat with points Chat[i] = Lhat[i] * Chat_subspsaces[i].

  INPUTS:
    Lhat: (M+1,)-shaped array, predicted lengths of points on a curve C
    Chat_subspaces: (M+1, d)-shaped array, predicted unit length vectors spanning the subspaces
      occupied by points on the curve C.
    m1: (d,)-shaped array, putative zero noise ground truth first moment to compute loss 
      against    
    m2: (d, d)-shaped array, putative zero noise ground truth first moment to compute loss 
      against
    m3: (d, d, d)-shaped array, putative zero noise ground truth first moment to compute loss 
      against

  OUTPUTS:
    loss: nonnegative float
  '''
  Chat =  Lhat[:, jnp.newaxis] * Chat_subspaces
  m1_pred = compute_m1(Chat)
  m2_pred = compute_m2(Chat)
  m3_pred = compute_m3(Chat)

  m1_loss = jnp.einsum("j, j -> ", m1_pred - m1, m1_pred - m1)
  m2_loss = jnp.einsum("jk, jk -> ", m2_pred - m2, m2_pred - m2)
  m3_loss = jnp.einsum("jkl, jkl -> ", m3_pred - m3, m3_pred - m3)

  loss = m1_loss + m2_loss + m3_loss

  return loss


@jit
def m3_loss(Chat, m3):
  '''
  Compute the third moment loss as a function of the curve control points.

  INPUTS:
    Chat: (M+1, d)-shaped array, predicted lengths of points on a curve C
    m3: (d, d, d)-shaped array, putative zero noise ground truth third moment to compute loss 
      against

  OUTPUTS:
    m3_loss: nonnegative float, squared frobenius norm between predicted and true third moment
      tensor
  '''
  m3_pred = compute_m3(Chat)
  m3_loss = jnp.einsum("jkl, jkl -> ", m3_pred - m3, m3_pred - m3)
  
  return m3_loss


@jit
def mu3_loss(Chat, phat, m3):
  '''
  Compute the third relaxed moment loss.

  INPUTS:
    Chat: (M+1, d)-shaped array, predicted points on a curve C
    phat: (M,)-shaped array, predicted segment lengths of curve C
    m3: (d, d, d)-shaped array, putative zero noise ground truth third moment to compute loss 
      against

  OUTPUTS:
    mu3_loss: nonnegative float, squared frobenius norm between relaxed predicted and true third 
      moment tensor
  '''
  mu3_pred = compute_mu3_fast(Chat, phat)
  mu3_loss = jnp.einsum("jkl, jkl -> ", mu3_pred - m3, mu3_pred - m3)
    
  return mu3_loss


@jit
def m123_loss(Chat, m1, m2, m3):
  '''
  Compute the sum of the losses of the first three moments as a function of the curve control points.

  INPUTS:
    Chat: (M+1, d)-shaped array, predicted lengths of points on a curve C
    m3: (d, d, d)-shaped array, putative zero noise ground truth third moment to compute loss 
      against

  OUTPUTS:
    loss: nonnegative float, sum squared frobenius norm between predicted and true first, second, 
      and third moment tensor.
  '''
  m1_pred = compute_m1(Chat)
  m2_pred = compute_m2(Chat)
  m3_pred = compute_m3(Chat)
  
  m1_loss = jnp.einsum("j, j -> ", m1_pred - m1, m1_pred - m1)
  m2_loss = jnp.einsum("jk, jk -> ", m2_pred - m2, m2_pred - m2)
  m3_loss = jnp.einsum("jkl, jkl -> ", m3_pred - m3, m3_pred - m3)
    
  loss = m1_loss + m2_loss + m3_loss

  return loss


def compute_curve_loss(C, Chat, num_samples = 10000):
  '''
  Compute the loss between two curves, given by $\int_0^1 \|C1(t) - C2(t)\|^2\,dt$. For use as a
    metric only.

  INPUTS:
    C: shape (M+1, d), ground truth C for piecewise linear curve
    C_pred: shape (M+1, d), prediced C for piecewies linear curve
    num_samples: int, number of integration samples to take, default = 10000

  OUTPUTS:
    curve_loss: scalar, the loss
  '''
  ts = jnp.linspace(0, 1, num_samples)

  Cs = compute_Cs(ts, C)

  C_pred_fwd = compute_Cs(ts, Chat)
  curve_loss_fwd = jnp.mean(jnp.linalg.norm(Cs - C_pred_fwd, axis = 1))

  C_pred_rev = compute_Cs(ts, Chat[::-1])
  curve_loss_rev = jnp.mean(jnp.linalg.norm(Cs - C_pred_rev, axis = 1))

  curve_loss = min(curve_loss_rev, curve_loss_fwd)

  return curve_loss