'''
moments.py

Functions for computing moments of multivariate Gaussians, noisy piecewise linear curves, and point 
clouds.
'''

import jax.numpy as jnp
from jax import jit, vmap
import itertools

from tqdm import tqdm
from data_generation import *


def gen_Sn(n):
  '''
  Generate an array of all permutations of [0, 1, ..., n - 1]. For example, gen_Sn(3) ->
    [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]].

  INPUTS:
    n: integer

  OUTPUTS:
    Sn: list of n! tuples, each with length (n,)
  '''
  Sn = list(itertools.permutations(range(n)))

  return Sn


def compute_sym_part(A):
  '''
  Given an n-dimensional tensor A, compute its symmetric part. That is, the sum of all n!
    transpositions of A, divided by n!.

  INPUTS:
    A: d^n shaped tensor A

  OUTPUTS:
    sym_part: d^n-shaped tensor Sym(A)
  '''
  n = A.ndim
  nfact = 0 # no jax.factorial, so use a hacky way of computing n!
  sym_part = jnp.zeros(shape = A.shape)
  Sn = gen_Sn(n)
  for perm in Sn:
    sym_part += jnp.transpose(A, perm)
    nfact += 1
     
  sym_part = sym_part / nfact

  return sym_part


def compute_normal_moment(mean, cov, m):
  '''
  Compute the mth moment of multivariate normal with mean m and covariance var. Using formulas
    from Holmquist, Moments and Cumulants of the Multivariate Normal Distribution, Stoch. Anal.
    and App., 6(3), 1988.

  INPUTS:
    mean: (n,)-dimensional array, mean of multivariate normal
    cov: (n, n)-dimensional positive semidefinite array, covariance of multivariate normal
    m: integer between 1 and 5 inlusive, the moment we wish to compute

  OUTPUTS:
    moment: (n, n, ..., n)-dimensional array (ndim = m), the mth moment 
  '''
  if m == 1:
    moment = mean
  elif m == 2:
    moment = jnp.einsum("i, j -> ij", mean, mean) + cov
  elif m == 3:
    moment = jnp.einsum("i, j, k -> ijk", mean, mean, mean) \
      + 3 * jnp.einsum("i, jk -> ijk", mean, cov)
    moment = compute_sym_part(moment)
  else:
    raise NotImplementedError(f"m must be between 1 and 3 inclusive, got {m}")

  return moment
    

@jit
def compute_m1(C):
  '''
  Compute the first moment of a noise-free piecewise linear curve. Segments need not be of equal 
    length.

  INPUTS:
    C: (M + 1, d)-shaped array of segment endpoints on the curve with M segments.

  OUTPUTS:
    m1: (d,)-shaped array, first moment of the curve.
  '''
  C_l = C[:-1]
  C_r = C[1:]
  seg_lens = compute_seg_lens(C)
  Z = jnp.sum(seg_lens)
  m1 = jnp.einsum("i, ij -> j", seg_lens, C_l + C_r) / Z / 2
  return m1


@jit
def compute_m2(C):
  '''
  Compute the second moment of a noise-free piecewise linear curve. Segments need not be of equal 
    length.

  INPUTS:
    C: (M + 1, d)-shaped array of segment endpoints on the curve

  OUTPUTS:
    m2: (d, d)-shaped array, second moment of the curve.
  '''
  seg_lens = compute_seg_lens(C)
  Z = jnp.sum(seg_lens)

  C_l = C[:-1] # shape (M, d)
  C_r = C[1:]
  seg_sums = C_l + C_r
  x = jnp.einsum("ij, ik -> ijk", seg_sums, seg_sums) + \
    jnp.einsum("ij, ik -> ijk", C_l, C_l) + \
    jnp.einsum("ij, ik -> ijk", C_r, C_r)
  m2 = jnp.einsum("i, ijk -> jk", seg_lens, x) / 6 / Z

  return m2


@jit
def compute_m3(C):
  '''
  Compute the third moment of a noise-free piecewise linear curve. Segments need not be of equal 
    length. 

  INPUTS:
    C: (M + 1, d)-shaped array of segment endpoints on the curve

  OUTPUTS:
    m3: (d, d, d)-shaped array, third moment of the curve.
  '''
  seg_lens = compute_seg_lens(C)
  Z = jnp.sum(seg_lens)

  C_l = C[:-1] # shape (M, d)
  C_r = C[1:]

  seg_sums = C_l + C_r

  x = jnp.einsum("ij, ik, il -> ijkl", seg_sums, seg_sums, seg_sums) / 12 \
      + jnp.einsum("ij, ik, il -> ijkl", C_l, C_l, C_l) / 6 \
      + jnp.einsum("ij, ik, il -> ijkl", C_r, C_r, C_r) / 6
  m3 = jnp.einsum("i, ijkl -> jkl", seg_lens, x) / Z

  return m3


@jit
def compute_m1hat(cloud):
  '''
  Compute the empirical first moment of a point cloud in R^d.
  
  INPUTS:
    cloud: (N, d)-shaped array of points in R^d

  OUTPUTS:
    m1hat: (d,)-shaped array, empirical first moment.
  '''
  m1hat = jnp.mean(cloud, axis = 0)
  return m1hat


@jit
def compute_m2hat(cloud):
  '''
  Compute the empirical second moment of a point cloud in R^d.
  
  INPUTS:
    cloud: (N, d)-shaped array of points in R^d

  OUTPUTS:
    m2hat: (d, d)-shaped array, empirical second moment.
  '''
  m2hat = jnp.einsum("ij, ik -> ijk", cloud, cloud)
  m2hat = jnp.mean(m2hat, axis = 0)
  return m2hat


@jit
def compute_m3hat(cloud):
  '''
  Compute the empirical third moment of a point cloud in R^d.
  
  INPUTS:
    cloud: (N, d)-shaped array of points in R^d

  OUTPUTS:
    m3hat: (d, d, d)-shaped array, empirical third moment.
  '''
  m3hat = jnp.einsum("ij, ik, il -> ijkl", cloud, cloud, cloud)
  m3hat = jnp.mean(m3hat, axis = 0)
  return m3hat

  
def gen_cloud_moments(subkey, C, sigma2, N, num_chunks, N_mini = 10000, n = 1e6):
  '''
  Generate points on a cloud on-the-fly and compute the moments in an online manner without 
    actually storing the full cloud for memory reasons.
  
  INPUTS:
    subkey: JAX PRNG key
    C: (M + 1, d)-shaped array of space points.
    sigma2: positive float, gaussian has covariance sigma2*I
    N: positive int, number of points to draw
    num_chunks: int, number of chunks to do the generation in; should divide N evenly
    N_mini: int, return a cloud with N_mini samples for visualization purposes only
    n: positive int, number of finite points to discretize the curve into for sampling purposes

  OUTPUTS:
    m1, m2, m3: (d,), (d, d), (d, d, d)-shaped tensors representing first three moments of data.
    cloud_mini: (N_mini, d)-shaped array of mini cloud, for visualization purposes only
  '''
  d = C.shape[1]
  m1 = jnp.zeros(shape = (d,))
  m2 = jnp.zeros(shape = (d, d))
  m3 = jnp.zeros(shape = (d, d, d))

  for _ in tqdm(range(num_chunks), desc = "generating cloud in chunks and computing online moments", leave = False):
    cloud_chunk = gen_cloud(subkey, C, sigma2, N // num_chunks, n = n)
    _, subkey = random.split(subkey)
    m1 += compute_m1hat(cloud_chunk)
    m2 += compute_m2hat(cloud_chunk)
    m3 += compute_m3hat(cloud_chunk)

  m1 = m1 / num_chunks
  m2 = m2 / num_chunks
  m3 = m3 / num_chunks

  cloud_mini = gen_cloud(subkey, C, sigma2, N_mini)

  return m1, m2, m3, cloud_mini

