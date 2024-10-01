'''
projection.py

Functions for projecting a curve/cloud down to lower dimensional subspace.
'''

import jax.numpy as jnp


def check_in_span(u, basis):
  '''
  Check if a vector is in the span of a basis.

  INPUTS:
    u: (d,)-shaped vector
    basis: (d, r)-shaped matrix whose columns are basis vectors

  OUTPUTS: 
    in_span: boolean, true if u is in the span of the columns of basis, false otherwise
  '''
  # probably not the smartest way to do this
  in_span = jnp.allclose(u, basis @ jnp.linalg.pinv(basis) @ u)
  
  return in_span


def get_basis(T2, r):
  '''
  Given a (d, d)-shaped symmetric second moment matrix of known rank r, compute the rank-r
    basis for the image of T2.

  INPUTS:
    T2: (d, d) second moment matrix with rank r
    r: rank of T2

  OUTPUTS:
    basis: (d, r)-shaped matrix whose columns are basis vectors
  '''
  _, evecs = jnp.linalg.eigh(T2)
  basis = evecs[:, -r:]

  return basis


def project_to_subspace(X, basis):
  '''
  Project the columns of X to the subspace spanned by the columns of basis.
  
  INPUTS:
    X: (d, n)-shaped matrix whose columns are vectors to be projected
    basis: (d, r)-shaped matrix whose columns are basis vectors
  
  OUTPUTS:
    X_proj: (r, n)-shaped matrix, X_proj[i] = projection of X[i] onto span of {basis[:, i]}
  '''
  X_proj = basis.T@X
  
  return X_proj


def deproject_from_subspace(X_proj, basis):
  '''
  Deproject X_proj from span of basis to ambient space.

  INPUTS:
    X_proj: (r, n)-shaped matrix, X_proj[i] = projection of X[i] onto span of {basis[:, i]}
    basis: (d, r)-shaped matrix whose columns are basis vectors
  
  OUTPUTS:
    X: (d, n)-shaped matrix whose columns are vectors to be projected
  '''
  X = jnp.linalg.pinv(basis).T@X_proj

  return X


def project_moments(m1, m2, m3, r):
  '''
  Given three tensors that represent the first three moments of d-dim data that really lie in a
    lower r-dim subspace, compute the moments of the data projected down to the subspace.

  INPUTS:
    m1, m2, m3: (d,), (d, d), (d, d, d)-shaped tensors representing first three moments of data.
    r: positive int, dimension of subspace to project down to.

  OUTPUTS:
    m1proj, m2proj, m3proj: (r,), (r, r), (r, r, r)-shaped tensors representing first three moments 
      of the data projected.
  '''
  V = get_basis(m2, r)
  m1proj = V.T@m1
  m2proj = V.T@m2@V
  m3proj = jnp.einsum("jm, kn, lo, jkl -> mno", V, V, V, m3)


  return m1proj, m2proj, m3proj




