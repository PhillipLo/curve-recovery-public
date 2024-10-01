'''
data_generation.py

Functions for generating curves and noisy curves.
'''
import jax.numpy as jnp
from jax import random, vmap, jit
from functools import partial
from tqdm import tqdm


def get_arclength_param_ts(C):
  '''
  Given points c_i on a curve, get the t_i's such that the piecewise linear interpolation of the 
    points is arclength parameterized. E.g., if C = [[0, 0], [0.1, 0.1], [0.9, 0.9]], then 
    ts = [0, 0.1, 0.9].
  
  INPUTS:
    C: (M + 1, d)-shaped array of space points
  
  OUTPUTS:
    ts: (M + 1,)-shaped array of times that arclength-parameterize the C.
  '''
  seg_lens = compute_seg_lens(C)
  ts = jnp.cumsum(seg_lens / jnp.sum(seg_lens))
  ts = jnp.insert(ts, 0, 0)

  return ts


def compute_Ct(t, C):
  '''
  Given points c_i on a curve, let C(t) be the arclength parameterized piecewise linear 
    interpolation of those points. Then given t\in [0, 1], compute the point C(t) on the curve.

  INPUTS:
    t: float in [0, 1], for computing C(t) 
    C: (M + 1, d)-shaped array of space points

  OUTPUTS:
    ct: (d,)-shaped array, C(t)
  '''
  ts = get_arclength_param_ts(C)  

  il = len(ts[::-1]<=t) - jnp.argmax(ts[::-1]<=t) - 1 # index i - 1, l for left
  ir = jnp.argmax(ts>t) # index i, r for right


  tl = ts[il] # t_{i-1}
  tr = ts[ir] # t_i
  cl = C[il] # c_{i-1}
  cr = C[ir] # c_i

  ct = (((tr - t) * cl + (t - tl) * cr) / (tr - tl)) * (t < 1) + C[-1] * (t == 1)

  return ct


compute_Cs = vmap(compute_Ct, in_axes = [0, None])
'''
Batch compute_Ct along t to compute points along a curve

INPUTS:
  ts_unif: linspace(0, 1, num_ts_unif, endpoint = False) for some num_ts_unif
  C: (M + 1, d)-shaped array of space points

OUTPUTS:
  C: (num_ts_unif, d)-shaped array, points along the curve dictated by ts and C.
'''


def compute_seg_lens(C):
  '''
  Compute the lengths of the segments of a piecewise linear curve.

  INPUTS:
    ts: (M + 1,)-shaped array of time points, in strictly ascending order in [0, 1], where
      ts[0] = 0, ts[-1] = 1
    C: (M + 1, d)-shaped array of PWL curve vertices

  OUTPUTS:
    seg_lens: (M,)-shaped array of curve lengths
  '''
  c_diffs = C[1:] - C[:-1]
  seg_lens = jnp.linalg.norm(c_diffs, axis = 1)
  
  return seg_lens


def compute_angles(C):
  '''
  Compute the angles between segments in a piecewise linear curve.

  INPUTS:
    C: (M + 1, d)-shaped array of space points
  
  OUTPUTS:
    angles: (M-1)-shaped array of angles
  '''
  M = C.shape[0] - 1
  
  angles = jnp.zeros(shape = (M-1,))

  for i in range(M-1):
    v0 = C[i+1] - C[i]
    v1 = C[i+2] - C[i+1]

    angle = jnp.arccos(jnp.inner(v0, v1) / jnp.linalg.norm(v0) / jnp.linalg.norm(v1))
    angles = angles.at[i].set(angle)
  
  return angles


def sample_from_cloud(seed, C, cov, n):
  '''
  Sample a single point from the cloud around the curve determined by ts and C.

  INPUTS:
    seed: int, for JAX PRNG seed
    C: (M + 1, d)-shaped array of space points
    cov: (d, d)-shaped array of gaussian covariance
    n: positive int, number of finite points to discretize the curve into for sampling purposes

  OUTPUTS:
    sample: (d,)-shaped array, the sample
  '''  
  k1, k2 = random.split(random.PRNGKey(seed))

  # pick index of gaussian mixture along curve to sample from
  idx = random.randint(k1, minval = 0, maxval = n + 1, shape = ())
  t = idx / (n + 1)

  # gaussian mixture center is C(sampled t)
  ct = compute_Ct(t, C)

  sample = random.multivariate_normal(k2, ct, cov, shape = ())

  return sample


samples_from_cloud = vmap(sample_from_cloud, in_axes = [0, None, None, None])
'''
Sample N points from the cloud around the curve determined by ts and C.

INPUTS:
  seeds: (N,)-shaped array of ints, for JAX PRNG seed
  C: (M + 1, d)-shaped array of space points
  cov: (d, d)-shaped array of gaussian covariance
  n: positive int, number of finite points to discretize the curve into for sampling purposes

OUTPUTS:
  sample: (d,)-shaped array, the sample
''' 


@partial(jit, static_argnums = (3, 4))
def gen_cloud(subkey, C, sigma2, N, n = 1e5):
  '''
  Generate a cloud around the piecewise linear curve determined by ts and C.
  
  INPUTS:
    subkey: JAX PRNG key
    ts: (M + 1,)-shaped array of time points, in strictly ascending order in [0, 1], where
      ts[0] = 0, ts[-1] = 1
    C: (M + 1, d)-shaped array of space points
    sigma2: positive float, gaussian has covariance sigma2*I
    N: positive int, number of points to draw
    n: positive int, number of finite points to discretize the curve into for sampling purposes

  OUTPUTS:
    data: (N, d)-shaped array, the data
  '''
  # ideally want replace = False, but this is memory intensive for some reason
  seeds = random.choice(subkey, jnp.arange(int(1e10)), shape = (N,), replace = True)
  d = C.shape[1]
  cov = sigma2 * jnp.eye(d)

  data = samples_from_cloud(seeds, C, cov, n)

  return data


def gen_cloud_chunked(subkey, C, sigma2, N, num_chunks, n = 1e6):
  '''
  Generate a cloud in a chunked manner for memory reasons.
  
  INPUTS:
    subkey: JAX PRNG key
    ts: (M + 1,)-shaped array of time points, in strictly ascending order in [0, 1], where
      ts[0] = 0, ts[-1] = 1
    C: (M + 1, d)-shaped array of space points
    sigma2: positive float, gaussian has covariance sigma2*I
    N: positive int, number of points to draw
    num_chunks: int, number of chunks to do the generation in; should divide N evenly
    n: positive int, number of finite points to discretize the curve into for sampling purposes
    

  OUTPUTS:
    cloud: (N, d)-shaped array, the data
  '''
  d = C.shape[1]
  cloud = jnp.zeros(shape = (0, d))
  for _ in tqdm(range(num_chunks), desc = "generating cloud in chunks", leave = False):
    cloud_chunk = gen_cloud(subkey, C, sigma2, N // num_chunks, n = n)
    _, subkey = random.split(subkey)
    cloud = jnp.append(cloud, cloud_chunk, axis = 0)
  return cloud


def gen_curve_sphere_sampling(seed, M, c0, seg_lens):
  '''
  Generate a piecewise linear curve by sampling randomly from the unit sphere.

  INPUTS:
    seed: int, for jax rng
    M: int, number of segments
    c0: (d,)-shaped array to specify initial point of the curve, or int d to specify dimension 
      of curve and let initial point be the origin
    seg_lens: (M,)-shaped array of positive floats/ints to specify segment lengths, or scalar to
      have all segments be the same length
    
  OUTPUT:
    C: (M+1, d)-shaped array, vertices of piecewise linear curve
  '''
  if isinstance(seg_lens, int) or isinstance(seg_lens, float):
    seg_lens = jnp.array(M * [seg_lens])

  if isinstance(c0, int):
    c0 = jnp.zeros((c0,))

  key, subkey = random.split(random.PRNGKey(seed))

  d = c0.size

  C = jnp.zeros(shape = (M + 1, d))
  C = C.at[0].set(c0)

  for i in range(M):
    v = random.normal(subkey, shape = (d,))
    v = v / jnp.linalg.norm(v) * seg_lens[i]
    key, subkey = random.split(key)
    C = C.at[i+1].set(C[i] + v)
  
  return C