'''
subspace_recovery.py

Functions for the first phase of Algorithm 2, where we apply the tensor power method to recover the 
approximate subspaces of the curve points and sort them.
'''

import jax.numpy as jnp
from jax import random


def tpm_deflation(T3, r, seed = 0, num_iters = 100):
  '''
  Apply the tensor power method with deflation on an odeco tensor T3.

  INPUTS:
    T3: (d, d, d)-shaped three-way tensor
    r: number of odeco components to extract from T3
    seed: int, for jax rng, default = 0
    num_iters: int, how many fixed point iterations to perform; default = 100

  OUTPUTS:
    uhat: (r, d)-shaped array where uhat[i] is the predicted ith orthonormal odeco component of T3
    lhat: (r,)-shaped array of "eigenvalues"

  For a truly odeco tensor of rank r, we should have 
    T3 = jnp.einsum("i, ij, ik, il -> jkl", lhat, uhat, uhat, uhat)
  '''
  d = T3.shape[0]
  T3_ = jnp.copy(T3) # copy of third moment tensor to deflate

  uhat = jnp.zeros(shape = (r, d))
  lhat = jnp.zeros(shape = (r,))
  
  key, subkey = random.split(random.PRNGKey(seed))

  for i in range(r):
    x = random.normal(subkey, shape = (d,))
    key, subkey = random.split(key)
    
    for _ in range(num_iters):
      x = jnp.einsum("ijk, j, k -> i", T3_, x, x)
      x = x / jnp.linalg.norm(x) 
    l = jnp.einsum("ijk, i, j, k ->", T3_, x, x, x)
    
    uhat = uhat.at[i].set(x)
    lhat = lhat.at[i].set(l)

    T3_ = T3_ - l * jnp.einsum("i, j, k -> ijk", x, x, x)
  
  return uhat, lhat


def whiten(T3, T2, r):
  '''
  Whiten the third moment tensor using the second moment tensor. See section 4.3 of 
    https://arxiv.org/pdf/1210.7559 for details.

  INPUTS:
    T3: (d, d, d)-shaped three-way tensor, coming from, say, a third moment
    T2: (d, d)-shaped two-way tensor, coming from, say, a second moment
    r: rank of the second moment tensor T2

  OUTPUTS:
    T3_white: (r, r, r)-shaped three-way tensor, whitened third moment tensor that is now odeco
      and subject to the tensor power method.
    W: (d, r)-shaped whitenening matrix
  '''
  # Compute eigendecomposition of second moment matrix
  D, U = jnp.linalg.eigh(T2)
  D = D[-r:]
  U = U[:, -r:]

  # Whitening matrix
  W = U@jnp.diag(1/jnp.sqrt(D))
  
  # Compute T3(W, W, W)
  T3_white = jnp.einsum("jm, kn, lo, jkl -> mno", W, W, W, T3)

  return T3_white, W


def build_chain(inter_C_angles_processed, start_idx):
  '''
  Build a putative chain of idxs to order points Chat. Algorithm 1 in the paper.

  INPUTS:
    inter_C_angles_processed: a (M+1, M+1)-shaped array where the [i, j] entry is equal to the 
      cosine of the angle between Chat[i] and Chat[j] for i != j, and equal to -1 for i = j.
    start_idx: int between 0 and M inclusive, starting point of the chain.

  OUTPUTS:
    chain: (M+1)-length list of ints, putative idxs that would sort Chat
  '''
  M = inter_C_angles_processed.shape[0]-1
  chain = [start_idx]
  for i in range(M):
    # at each step, add the index of the largest element in the most recent idx row that hasn't already been added
    for idx in jnp.argsort(-inter_C_angles_processed[:, chain[-1]]):
      if idx not in chain:
        chain.append(idx.item())
        break
  return chain


def recover_C_subspaces(T3, T2, M, seed = 0):
  '''
  Given the third and second moment of a noise-free mean zero curve with M segments, estimate the 
    M+1 subspaces where the curve points c0,...,cM must live.
  
  INPUTS:
    T3: (d, d, d)-tensor, third moment of noise-free mean zero curve
    T2: (d, d)-tensor, second moment of noise-free mean zero curve
    M: int, number of segments in curve, must be < d
    seed: int, for jax rng, default = 0

  OUTPUTS:
    Chat_subspaces: (M+1, d)-shaped array of potentially sorted estimated subspaces of Chat; each
      Chat_subspaces[i] is unit norm
    status: int
      0: exactly one palindromic chain found, returning sorted Chat
      1: multiple palindromic chains found, returning the first found one
      2: zero palindromic chains found, returning unsorted Chat
  '''
  T3_white, W = whiten(T3, T2, M)
  Chat_unordered_white, lhat_white = tpm_deflation(T3_white, M+1, seed)
  Chat_unordered = jnp.diag(lhat_white) @ Chat_unordered_white @ jnp.linalg.pinv(W)
  
  Chat_unordered_normalized = Chat_unordered / jnp.linalg.norm(Chat_unordered, axis = 1)[:, jnp.newaxis]

  # diag elems will all be 1  
  inter_C_angles = jnp.einsum("ji, ki -> jk", Chat_unordered_normalized, Chat_unordered_normalized)

  # set diag elems to -1
  inter_C_angles_processed = inter_C_angles * (jnp.ones((M+1, M+1)) - 2*jnp.eye(M+1))

  chains = []
  for i in range(M+1):
    chains.append(build_chain(inter_C_angles_processed, i))

  # correct chains will have a palindrome
  # for each pair of chains, assign a score of how palindromic the pair is (num matching entries)
  # score for a chain is maximum of score with each chain
  palindromic_chains = []
  palindromicities = []
  for i in range(len(chains)-1):
    palindromicity = []
    for j in range(i+1, len(chains)):
      if chains[i] == chains[j][::-1]:
         palindromic_chains.append(chains[i])
      palindromicity.append(jnp.count_nonzero(jnp.array(chains[i]) - jnp.array(chains[j])[::-1] == 0))
    palindromicities.append(jnp.max(jnp.array(palindromicity)))
  palindromicities = jnp.array(palindromicities)

  if len(palindromic_chains) == 0:
    status = 2
    Chat_subspaces = Chat_unordered_normalized[jnp.array(chains[jnp.argmax(palindromicities)])]

  elif len(palindromic_chains) == 1:
    status = 0
    Chat_subspaces = Chat_unordered_normalized[jnp.array(palindromic_chains[0])]
  
  else:
    status = 1
    Chat_subspaces = Chat_unordered_normalized[jnp.array(palindromic_chains[0])]

  return Chat_subspaces, status
  
