'''
relaxed_moments.py

Functions for computing relaxed moments of noisy piecewise linear curves
'''

from moments import *


def compute_A_matrices(M):
  '''
  Compute the constant matrices needed to vectorize moment computations for noisy PWL curves.

  INPUTS:
    M: int, number of segments in a curve

  OUTPUTS:
    As, Al, Ar: (M+1, M)-shaped 0-1 matrices.
  '''
  As = jnp.eye(M+1, M) + jnp.eye(M+1, M, k = -1)
  Al = jnp.eye(M+1, M)
  Ar = jnp.eye(M+1, M, k = -1)
  return As, Al, Ar


def compute_mu1(C, p):
  '''
  Compute the relaxed first moment of a piecewise linear curve.
  
  INPUTS:
    C: (M+1, d)-shaped array, points on pwl curve
    p: (M,)-shaped array, proxy for proportional curve lengths

  OUTPUTS:
    mu1: (d,)-shaped array, the relaxed first moment
  '''
  M = C.shape[0] - 1
  As, _, _ = compute_A_matrices(M)
  mu1 = C.T@As@p/2

  return mu1


def compute_mu2(C, p):
  '''
  Compute the relaxed second moment of a piecewise linear curve.
  
  INPUTS:
    C: (M+1, d)-shaped array, points on pwl curve
    p: (M,)-shaped array, proxy for proportional curve lengths

  OUTPUTS:
    mu2: (d, d)-shaped array, the relaxed second moment
  '''
  M = C.shape[0]-1
  As, Al, Ar = compute_A_matrices(M)
  Abar = (As@jnp.diag(p)@As.T + Ar@jnp.diag(p)@Ar.T + Al@jnp.diag(p)@Al.T) / 6
  mu2 = C.T@Abar@C

  return mu2


def compute_alpha(p):
  '''
  Compute the alpha term needed to compute the third relaxed moment.

  INPUTS:
    p: (M,)-shaped array, proxy for proportional curve lengths

  OUTPUTS:
    alpha: (M+1, M+1, M+1)-shaped symmetric 3-way tensor
  '''
  M = p.size
  As, Al, Ar = compute_A_matrices(M)

  alpha = jnp.einsum("mi,ni,oi,i->mno", As, As, As, p) / 12 + \
    jnp.einsum("mi,ni,oi,i->mno", Al, Al, Al, p) / 6 + \
    jnp.einsum("mi,ni,oi,i->mno", Ar, Ar, Ar, p) / 6
  
  return alpha


def star(a, b):
  '''
  Compute a tensor contraction of a six-way and three-way tensor with signature "mjnkol, mno -> jkl".
  
  INPUTS:
    a: (M+1, d, M+1, d, M+1, d)-shaped tensor
    b: (M+1, M+1, M+1)-shaped tensor
  
  OUTPUTS
    ab: (d, d, d)-shaped tensor formed by the natural contraction of a and b
  '''
  ab = jnp.einsum("mjnkol, mno -> jkl", a, b)
  return ab


def matrix_triple(X, Y, Z):
  '''
  Compute the six way tensor product between three matrices.

  INPUTS:
    X, Y, Z: two dimensional arrays
  
  OUTPUTS:
    XYZ: six-dimenional array
  '''
  return jnp.einsum("ij, kl, mn ->ijklmn", X, Y, Z)


def compute_mu3(C, p):
  '''
  Compute the relaxed third moment of a noise-free piecewise linear curve.
  
  INPUTS:
    C: (M+1, d)-shaped array, points on pwl curve
    p: (M,)-shaped array, proxy for proportional curve lengths

  OUTPUTS:
    mu3: (d, d, d)-shaped array, the relaxed third moment
  '''
  C3 = matrix_triple(C, C, C)
  alpha = compute_alpha(p)

  mu3 = star(C3, alpha)
  return mu3


def compute_L1(C, p, m1):
  '''
  Compute the squared frobenius norm between a predicted (or ground truth) first moment m1 and the 
    relaxed first moment of the curve coming from C and p.
  
  INPUTS:
    C: (M+1, d)-shaped array, points on pwl curve
    p: (M,)-shaped array, proxy for proportional curve lengths
    m1: (d,)-shaped array, first moment to compute loss against

  OUTPUTS:
    L1: the loss
  '''
  mu1 = compute_mu1(C, p)
  L1 = jnp.einsum("j, j ->", mu1 - m1, mu1 - m1)
  return L1


def compute_L2(C, p, m2):
  '''
  Compute the squared frobenius norm between a predicted (or ground truth) second moment m2 and the 
    relaxed second moment of the curve coming from C and p.
  
  INPUTS:
    C: (M+1, d)-shaped array, points on pwl curve
    p: (M,)-shaped array, proxy for proportional curve lengths
    m2: (d, d)-shaped array, second moment to compute loss against

  OUTPUTS:
    L2: the loss
  '''
  mu2 = compute_mu2(C, p)
  L2 = jnp.einsum("jk, jk ->", mu2 - m2, mu2 - m2)
  return L2


def compute_L3(C, p, m3):
  '''
  Compute the squared frobenius norm between a predicted (or ground truth) third moment m3 and the 
    relaxed third moment of the curve coming from C and p.
  
  INPUTS:
    C: (M+1, d)-shaped array, points on pwl curve
    p: (M,)-shaped array, proxy for proportional curve lengths
    m3: (d, d, d)-shaped array, third moment to compute loss against

  OUTPUTS:
    L3: the loss
  '''
  mu3 = compute_mu3(C, p)
  L3 = jnp.einsum("jkl, jkl ->", mu3 - m3, mu3 - m3)
  return L3


def compute_DL1_y(C, p, m1, y):
  '''
  Evaluate the total derivative of the relaxed first moment loss of a curve on a point y.

  INPUTS:
    C: (M+1, d)-shaped array, points on pwl curve
    p: (M,)-shaped array, proxy for proportional curve lengths
    m1: (d,)-shaped array, first moment to compute loss against
    y: (M+1, d)-shaped array, tensor to evaluate the derivative on.

  OUTPUTS:
    DL1_y: scalar, the result of evaluating the derivative of the first moment loss on y.
  '''
  M = C.shape[0] - 1
  As, _, _ = compute_A_matrices(M)
  DL1_y = 2 * jnp.inner(compute_mu1(C, p) - m1, 1/2*y.T@As@p)

  return DL1_y


def compute_Abar(p):
  '''
  Evaluate the constant \overline{A} needed to compute second moment loss derivatives.

  INPUTS:
    p: (M,)-shaped array, proxy for proportional curve lengths

  OUTPUTS:
    Abar: (M+1, M+1)-shaped array.
  '''
  M = p.size
  As, Al, Ar = compute_A_matrices(M)
  Abar = (As@jnp.diag(p)@As.T + Al@jnp.diag(p)@Al.T + Ar@jnp.diag(p)@Ar.T) / 6
  return Abar


def compute_DL2_y(C, p, m2, y):
  '''
  Evaluate the total derivative of the relaxed second moment loss of a curve on a point y.

  INPUTS:
    C: (M+1, d)-shaped array, points on pwl curve
    p: (M,)-shaped array, proxy for proportional curve lengths
    m2: (d, d)-shaped array, first moment to compute loss against
    y: (M+1, d)-shaped array, tensor to evaluate the derivative on.

  OUTPUTS:
    DL2_y: scalar, the result of evaluating the derivative of the second moment loss on y.
  '''
  Abar = compute_Abar(p)
  DL2_y = jnp.einsum("ij, ij ->", 2*(C.T@Abar@C - m2), C.T@Abar@y + y.T@Abar@C)

  return DL2_y



def compute_DL3_y(C, p, m3, y):
  '''
  Evaluate the total derivative of the relaxed third moment loss of a curve on a point y.

  INPUTS:
    C: (M+1, d)-shaped array, points on pwl curve
    p: (M,)-shaped array, proxy for proportional curve lengths
    m3: (d, d, d)-shaped array, third moment to compute loss against
    y: (M+1, d)-shaped array, tensor to evaluate the derivative on.

  OUTPUTS:
    DL3_y: scalar, the result of evaluating the derivative of the third moment loss on y.
  '''
  alpha = compute_alpha(p)

  diff_term = 2 * (compute_mu3(C, p) - m3) # difference that's in in the norm
  Dy = star(matrix_triple(C, C, y) +  matrix_triple(C, y, C)  +  matrix_triple(y, C, C), alpha)

  DL3_y = jnp.einsum("jkl, jkl ->", diff_term, Dy)

  return DL3_y


def compute_DpL3_y(C, p, m3, y):
  '''
  Evaluate the total derivative wrt p of the relaxed third moment loss of a curve on a point y.

  INPUTS:
    C: (M+1, d)-shaped array, points on pwl curve
    p: (M,)-shaped array, proxy for proportional curve lengths
    m3: (d, d, d)-shaped array, third moment to compute loss against
    y: (M,)-shaped array, tensor to evaluate the derivative on.

  OUTPUTS:
    DpL3_y: scalar, the result of evaluating the derivative of the third moment loss wrt p on y.
  '''
  alpha = compute_alpha(p)
  
  X = star(matrix_triple(C, C, C), compute_alpha(p)) - m3
  Y = star(matrix_triple(C, C, C), compute_alpha(y))
  
  DpL3_y = 2*jnp.einsum("jkl, jkl -> ", X, Y)

  return DpL3_y


def compute_DL1(C, p, m1):
  '''
  Compute the derivative of the first moment loss wrt C.
    
  INPUTS:
    C: (M+1, d)-shaped array, points on pwl curve
    p: (M,)-shaped array, proxy for proportional curve lengths
    m1: (d,)-shaped array, first moment to compute loss against

  OUTPUTS:
    DL1: (M+1, d)-shaped array, the derivative
  '''
  M = C.shape[0] - 1
  As, _, _ = compute_A_matrices(M)

  DL1 = jnp.outer(As@p, compute_mu1(C, p) - m1)
  
  return DL1


def compute_DL2(C, p, m2):
  '''
  Compute the derivative of the second moment loss wrt C.
    
  INPUTS:
    C: (M+1, d)-shaped array, points on pwl curve
    p: (M,)-shaped array, proxy for proportional curve lengths
    m2: (d, d)-shaped array, second moment to compute loss against

  OUTPUTS:
    DL2 (M+1, d)-shaped array, the derivative.
  '''
  Abar = compute_Abar(p)
  DL2 = 4*Abar@C@compute_sym_part(C.T@Abar@C - m2)
  
  return DL2


def compute_DL3(C, p, m3):
  '''
  Compute the derivative of the second moment loss wrt C.
    
  INPUTS:
    C: (M+1, d)-shaped array, points on pwl curve
    p: (M,)-shaped array, proxy for proportional curve lengths
    m3: (d, d, d)-shaped array, first moment to compute loss against

  OUTPUTS:
    DL3 (M+1, d)-shaped array, the derivative.
  '''
  alpha = compute_alpha(p)
  DL3 = 6*jnp.einsum("jkl, mj, nk, mno -> ol", compute_mu3(C, p) - m3, C, C, alpha)

  return DL3


def compute_mu1_fast(C, p):
  '''
  Compute the relaxed first moment of a piecewise linear curve in a faster, more direct way. Better
    for jitting.
  
  INPUTS:
    C: (M+1, d)-shaped array, points on pwl curve
    p: (M,)-shaped array, proxy for proportional curve lengths

  OUTPUTS:
    mu1: (d,)-shaped array, the relaxed first moment
  '''
  C_l = C[:-1]
  C_r = C[1:]
  mu1 = jnp.einsum("i, ij -> j", p, C_l + C_r) / 2
  return mu1


def compute_mu2_fast(C, p):
  '''
  Compute the relaxed second moment of a piecewise linear curve in a faster, more direct way. Better
    for jitting.
  
  INPUTS:
    C: (M+1, d)-shaped array, points on pwl curve
    p: (M,)-shaped array, proxy for proportional curve lengths

  OUTPUTS:
    mu2: (d, d)-shaped array, the relaxed second moment
  '''
  C_l = C[:-1] # shape (M, d)
  C_r = C[1:]
  seg_sums = C_l + C_r
  x = jnp.einsum("ij, ik -> ijk", seg_sums, seg_sums) + \
    jnp.einsum("ij, ik -> ijk", C_l, C_l) + \
    jnp.einsum("ij, ik -> ijk", C_r, C_r)
  m2 = jnp.einsum("i, ijk -> jk", p, x) / 6

  return m2


def compute_mu3_fast(C, p):
  '''
  Compute the relaxed third moment of a piecewise linear curve in a faster, more direct way. Better
    for jitting.
  
  INPUTS:
    C: (M+1, d)-shaped array, points on pwl curve
    p: (M,)-shaped array, proxy for proportional curve lengths

  OUTPUTS:
    mu3: (d, d, d)-shaped array, the relaxed third moment
  '''
  C_l = C[:-1] # shape (M, d)
  C_r = C[1:]

  seg_sums = C_l + C_r

  x = jnp.einsum("ij, ik, il -> ijkl", seg_sums, seg_sums, seg_sums) / 12 \
      + jnp.einsum("ij, ik, il -> ijkl", C_l, C_l, C_l) / 6 \
      + jnp.einsum("ij, ik, il -> ijkl", C_r, C_r, C_r) / 6
  mu3 = jnp.einsum("i, ijkl -> jkl", p, x)

  return mu3