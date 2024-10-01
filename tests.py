'''
tests.py

Automated tests for correctness of implementation.
'''

import jax
jax.config.update("jax_enable_x64", True) # need high precision to test naive vs vectorized versions

import unittest
import numpy as np
from jax import random, jacrev
from tqdm import tqdm

from moments import *
from relaxed_moments import *
from data_generation import *
from tracing_recovery import *
from subspace_recovery import *
from projection import *

from scipy.stats import ortho_group

_NUM_TESTS_ = 10


class test_moments(unittest.TestCase):
  '''
  Test functions in moments.py.
  '''
  def test_compute_sym_part(self):
    '''
    Check that compute_sym_part() really returns a symmetric matrix.
    '''
    for seed in range(_NUM_TESTS_):
      k1, k2 = random.split(random.PRNGKey(seed))

      d = random.randint(k1, minval = 5, maxval = 10, shape = ()) # size of each tensor dim
      n = random.randint(k2, minval = 1, maxval = 5, shape = ()) # number of tensor dims
      
      shape = (d * np.ones(shape = (n,))).astype(int)

      A = random.uniform(k2, minval = -10, maxval = 10, shape = shape)

      sym_part = compute_sym_part(A)

      # get all n! permutations
      Sn = gen_Sn(n)

      for perm in Sn:
        self.assertTrue(np.allclose(sym_part, np.transpose(sym_part, axes = perm)))


  def test_gen_cloud_moments(self):
    '''
    Compare online cloud moment generation with offline.
    '''
    for seed in range(_NUM_TESTS_):
      key, subkey = random.split(random.PRNGKey(seed))

      num_chunks = int(random.randint(subkey, minval = 2, maxval = 10, shape = ()))
      key, subkey = random.split(key)

      chunk_size = int(random.randint(subkey, minval = 10, maxval = 50, shape = ()))
      key, subkey = random.split(key)

      N = num_chunks * chunk_size      

      M = int(random.randint(subkey, shape = (), minval = 2, maxval = 8))
      key, subkey = random.split(key)
      
      d = int(random.randint(subkey, shape = (), minval = 10, maxval = 15))
      key, subkey = random.split(key)

      C = gen_curve_sphere_sampling(seed, M, d, 1)
      
      sigma2 = random.uniform(subkey, shape = (), minval = 1, maxval = 2)
      key, subkey = random.split(key)

      # do NOT split key here because we want exact same cloud
      m1, m2, m3, _ = gen_cloud_moments(subkey, C, sigma2, N, num_chunks)
      cloud = gen_cloud_chunked(subkey, C, sigma2, N, num_chunks)

      m1_ = compute_m1hat(cloud)
      m2_ = compute_m2hat(cloud)
      m3_ = compute_m3hat(cloud)

      self.assertTrue(np.allclose(m1, m1_))
      self.assertTrue(np.allclose(m2, m2_))
      self.assertTrue(np.allclose(m3, m3_))


class test_relaxed_moments(unittest.TestCase):
  '''
  Test functions in relaxed_moments.py
  '''
  def test_relaxed_moments_accuracy(self):
    '''
    Check that relaxed moments agree with true moments when p is equal to the proportional segment
      lengths.
    '''
    for seed in range(_NUM_TESTS_):
      key, subkey = random.split(random.PRNGKey(seed))

      M = int(random.randint(subkey, shape = (), minval = 2, maxval = 8))
      key, subkey = random.split(key)
      
      d = int(random.randint(subkey, shape = (), minval = 2, maxval = 8))
      key, subkey = random.split(key)

      C = random.normal(subkey, shape = (M+1, d))
      key, subkey = random.split(key)
      p = compute_seg_lens(C)
      p = p / jnp.sum(p)

      m1 = compute_m1(C)
      m2 = compute_m2(C)
      m3 = compute_m3(C)

      mu1 = compute_mu1(C, p)
      mu2 = compute_mu2(C, p)
      mu3 = compute_mu3(C, p)

      self.assertTrue(jnp.allclose(m1, mu1))
      self.assertTrue(jnp.allclose(m2, mu2))
      self.assertTrue(jnp.allclose(m3, mu3))


  def test_relaxed_moments_fast(self):
    '''
    Check that fast relaxed moments agree with the more "mathematical" implementation.
    '''
    for seed in range(_NUM_TESTS_):
      key, subkey = random.split(random.PRNGKey(seed))

      M = int(random.randint(subkey, shape = (), minval = 2, maxval = 8))
      key, subkey = random.split(key)
      
      d = int(random.randint(subkey, shape = (), minval = 2, maxval = 8))
      key, subkey = random.split(key)

      C = random.normal(subkey, shape = (M+1, d))
      key, subkey = random.split(key)
      
      p = random.uniform(subkey, shape = (M,))
      p = p / jnp.sum(p)

      mu1_fast = compute_mu1_fast(C, p)
      mu2_fast = compute_mu2_fast(C, p)
      mu3_fast = compute_mu3_fast(C, p)

      mu1 = compute_mu1(C, p)
      mu2 = compute_mu2(C, p)
      mu3 = compute_mu3(C, p)

      self.assertTrue(jnp.allclose(mu1_fast, mu1))
      self.assertTrue(jnp.allclose(mu2_fast, mu2))
      self.assertTrue(jnp.allclose(mu3_fast, mu3))

  
class test_data_generation(unittest.TestCase):
  '''
  Test functions in data_generation.py.
  '''
  def test_get_arclength_param_ts(self):
    '''
    Check get_arclength_param_ts() against uniform colinear C
    '''
    for seed in range(_NUM_TESTS_):
      key, subkey = random.split(random.PRNGKey(seed))

      M = random.randint(subkey, minval = 10, maxval = 20, shape = ())
      key, subkey = random.split(key)

      d = random.randint(subkey, minval = 5, maxval = 10, shape = ())
      key, subkey = random.split(key)

      c0 = random.uniform(subkey, shape = (d,))
      key, subkey = random.split(key)

      cM = random.uniform(subkey, shape = (d,))
      key, subkey = random.split(key)

      C = np.linspace(c0, cM, M + 1)

      ts = get_arclength_param_ts(C)

      self.assertTrue(np.allclose(ts, np.linspace(0, 1, M + 1)))


  def test_compute_Cs(self):
    '''
    Check that compute_Cs returns a curve whose first and last points are c_0 and c_{M+1}.
    '''
    for seed in range(_NUM_TESTS_):
      key, subkey = random.split(random.PRNGKey(seed))

      M = random.randint(subkey, minval = 10, maxval = 50, shape = ())
      key, subkey = random.split(key)

      d = random.randint(subkey, minval = 1, maxval = 5, shape = ())
      key, subkey = random.split(key)

      C = random.uniform(subkey, minval = -10, maxval = 10, shape = (M + 1, d))
      key, subkey = random.split(key)

      num_ts_unif = random.randint(subkey, minval = 500, maxval = 1000, shape = ())
      key, subkey = random.split(key)

      ts_unif = jnp.linspace(0, 1, num_ts_unif)

      Cs = compute_Cs(ts_unif, C)

      self.assertTrue(Cs.shape == (num_ts_unif, d))
      self.assertTrue(np.allclose(Cs[0], C[0]))
      self.assertTrue(np.allclose(Cs[-1], C[-1]))


  def test_compute_seg_lens(self):
    '''
    Compare compute_seg_lens() with naive implementation.
    '''
    for seed in range(_NUM_TESTS_):
      key, subkey = random.split(random.PRNGKey(seed))

      M = random.randint(subkey, minval = 10, maxval = 50, shape = ())
      key, subkey = random.split(key)

      d = random.randint(subkey, minval = 1, maxval = 5, shape = ())
      key, subkey = random.split(key)

      C = random.uniform(subkey, minval = -10, maxval = 10, shape = (M + 1, d))
      key, subkey = random.split(key)

      seg_lens_naive = np.zeros(shape = (M,))

      for i in range(M):
        seg_lens_naive[i] = np.linalg.norm(C[i + 1] - C[i])

      seg_lens = compute_seg_lens(C)

      self.assertTrue(np.allclose(seg_lens, seg_lens_naive))



  def test_compute_angles(self):
    '''
    Compare compute_seg_lens() with naive implementation.
    '''
    for seed in range(_NUM_TESTS_):
      key, subkey = random.split(random.PRNGKey(seed))

      M = random.randint(subkey, minval = 10, maxval = 50, shape = ())
      key, subkey = random.split(key)

      d = random.randint(subkey, minval = 2, maxval = 5, shape = ())
      key, subkey = random.split(key)

      C = random.uniform(subkey, minval = -10, maxval = 10, shape = (M + 1, d))
      key, subkey = random.split(key)


      angles_naive = np.zeros(shape = (M - 1,))
      for i in range(1, M):
        u = C[i] - C[i - 1]
        v = C[i + 1] - C[i]

        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)

        angles_naive[i - 1] = np.arccos(np.inner(u, v) / u_norm / v_norm)
      angles = compute_angles(C)
      self.assertTrue(np.allclose(angles, angles_naive))

  
  def test_gen_curve_sphere_sampling(self):
    '''
    Check that C returned by gen_curve_sphere_sampling() satisfies constraint on segment lengths.
    '''
    for seed in range(_NUM_TESTS_):
      key, subkey = random.split(random.PRNGKey(seed))

      seed_ = random.randint(subkey, minval = 0, maxval = 10 * _NUM_TESTS_, shape = ())
      key, subkey = random.split(key)

      M = int(random.randint(subkey, minval = 2, maxval = 100, shape = ()))
      key, subkey = random.split(key)

      d = random.randint(subkey, minval = 2, maxval = 10, shape = ())
      key, subkey = random.split(key)

      if seed % 2 == 0:
        c0 = random.uniform(subkey, minval = -10, maxval = 10, shape = (d,))
        key, subkey = random.split(key)
        
        seg_lens = random.uniform(subkey, minval = 0, maxval = 2, shape = (M,))
        key, subkey = random.split(key)
      else:
        c0 = int(d)
        seg_lens = float(random.uniform(subkey, minval = 0, maxval = 2, shape = ()))

      C = gen_curve_sphere_sampling(seed_, M, c0, seg_lens)
   
      seg_lens_ = compute_seg_lens(C)
      self.assertTrue(np.allclose(seg_lens, seg_lens_))
      self.assertTrue(C.shape[0] == M+1)
      self.assertTrue(C.shape[1] == d)


  def test_gen_cloud_chunked(self):
    '''
    Check shapes of gen_cloud_chunked().
    '''
    for seed in range(_NUM_TESTS_):
      key, subkey = random.split(random.PRNGKey(seed))

      num_chunks = int(random.randint(subkey, minval = 2, maxval = 10, shape = ()))
      key, subkey = random.split(key)

      chunk_size = int(random.randint(subkey, minval = 10, maxval = 50, shape = ()))
      key, subkey = random.split(key)

      d = int(random.randint(subkey, minval = 2, maxval = 10, shape = ()))
      key, subkey = random.split(key)

      M = int(random.randint(subkey, minval = 2, maxval = 10, shape = ()))
      key, subkey = random.split(key)

      C = random.normal(subkey, shape = (M, d))
      key, subkey = random.split(key)

      sigma2 = random.uniform(subkey, shape = ())
      key, subkey = random.split(key)

      N = num_chunks * chunk_size

      cloud = gen_cloud_chunked(subkey, C, sigma2, N, num_chunks)
      key, subkey = random.split(key)
      self.assertTrue(cloud.shape == (N, d))
      

class test_tracing_recovery(unittest.TestCase):
  '''
  Test functions in tracing_recovery.py.
  '''
  def test_compute_local_cloud(self):
    '''
    Check vectorization of compute_local_cloud()
    '''
    for seed in range(_NUM_TESTS_):
      key, subkey = random.split(random.PRNGKey(seed))

      N = int(random.randint(subkey, shape = (), minval = 100, maxval = 500))
      key, subkey = random.split(key)

      d = random.randint(subkey, shape = (), minval = 2, maxval = 6)
      key, subkey = random.split(key)

      cloud = random.normal(subkey, shape = (N, d))
      key, subkey = random.split(key)
      
      pt = random.normal(subkey, shape = (d,))
      key, subkey = random.split(key)

      r = random.uniform(subkey, shape = (), minval = 0.8, maxval =  1.2)
      key, subkey = random.split(key)

      local_cloud = compute_local_cloud(cloud, pt, r)
      for datum in local_cloud:
        assert np.linalg.norm(datum - pt <= r)


  def test_resolve_Rd_c1(self):
    '''
    Check jaxified if statement in resolve_Rd_c1() against vanilla conditional.
    '''
    for seed in range(_NUM_TESTS_):
      key, subkey = random.split(random.PRNGKey(seed))

      M = int(random.randint(subkey, shape = (), minval = 10, maxval = 20))
      key, subkey = random.split(key)

      N = int(random.randint(subkey, shape = (), minval = 10000, maxval = 20000))
      key, subkey = random.split(key)

      d = random.randint(subkey, shape = (), minval = 2, maxval = 6)
      key, subkey = random.split(key)

      n = int(random.randint(subkey, shape = (), minval = 10000, maxval = 20000))
      key, subkey = random.split(key)

      sigma2 = 10**(random.uniform(subkey, shape = (), minval = -4, maxval = -6)) # log10 uniform between 1e-4, 1e-6      
      key, subkey = random.split(key)

      seg_len = 1/M

      c0 = random.normal(subkey, shape = (d,))
      key, subkey = random.split(key)

      C = gen_curve_sphere_sampling(seed, M, c0, seg_len)
      cloud = gen_cloud(subkey, C, sigma2, N, n)

      c0_cloud = compute_local_cloud(cloud, C[0], seg_len / 2)
      c1_pred = resolve_Rd_c1(c0_cloud, c0, seg_len)

      m2hat = jnp.mean(jnp.einsum("ij, ik -> ijk", c0_cloud - C[0], c0_cloud - C[0]), axis = 0)
      _, evecs = jnp.linalg.eigh(m2hat) # eigs sorted in increasing order
      seg_basis = evecs[:, -1] # only take last evec now

      # last evec has a sign ambiguity
      # distance between c0 + seg_basis * seg_len
      unflipped_dists = jnp.sum(jnp.linalg.norm(c0 + seg_basis * seg_len - c0_cloud, axis = -1))
      flipped_dists = jnp.sum(jnp.linalg.norm(c0 - seg_basis * seg_len - c0_cloud, axis = -1))
      if flipped_dists <= unflipped_dists:
        c1_pred_naive = c0 - seg_basis * seg_len
      else:
        c1_pred_naive = c0 + seg_basis * seg_len

      assert np.allclose(c1_pred, c1_pred_naive)

      # check that we're within 10 percent of the ground truth
      assert np.allclose(c1_pred, C[1], rtol = 0.1)


class test_projection(unittest.TestCase):
  '''
  Test functions in projection.py.
  '''
  def test_check_in_span(self):
    '''
    Check correctness of check_in_span()
    '''
    for seed in range(_NUM_TESTS_):
      key, subkey = random.split(random.PRNGKey(seed))

      d = int(random.randint(subkey, minval = 11, maxval = 20, shape = ()))
      key, subkey = random.split(key)

      r = int(random.randint(subkey, minval = 5, maxval = 10, shape = ()))
      key, subkey = random.split(key)

      basis = random.normal(subkey, shape = (d, r))
      key, subkey = random.split(key)

      coeffs = random.normal(subkey, shape = (r,))
      key, subkey = random.split(key)

      assert check_in_span(basis@coeffs, basis)
      assert not check_in_span(random.normal(subkey, shape = (d,)), basis)


  def test_get_basis(self):
    '''
    Check correctness of get_basis()
    '''
    for seed in range(_NUM_TESTS_):
      key, subkey = random.split(random.PRNGKey(seed))
      
      d = int(random.randint(subkey, minval = 11, maxval = 20, shape = ()))
      key, subkey = random.split(key)

      r = int(random.randint(subkey, minval = 5, maxval = 10, shape = ()))
      key, subkey = random.split(key)

      X = random.normal(subkey, shape = (r, d))
      key, subkey = random.split(key)

      coeffs = random.normal(subkey, shape = (r,))

      T2 = jnp.zeros(shape = (d, d))
      for i in range(r):
        key, subkey = random.split(key)
        T2 += jnp.outer(X[i], X[i])

      basis = get_basis(T2, r)

      assert check_in_span(basis@coeffs, basis)


  def test_project_to_subspace(self):
    '''
    Check correctness of project_to_subspace()
    '''
    for seed in range(_NUM_TESTS_):
      key, subkey = random.split(random.PRNGKey(seed))
      
      d = int(random.randint(subkey, minval = 11, maxval = 20, shape = ()))
      key, subkey = random.split(key)

      r = int(random.randint(subkey, minval = 5, maxval = 10, shape = ()))
      key, subkey = random.split(key)

      n = int(random.randint(subkey, minval = 5, maxval = 20, shape = ()))
      key, subkey = random.split(key)

      basis = random.normal(subkey, shape = (d, r))
      key, subkey = random.split(key)

      X = random.normal(subkey, shape = (d, n))

      X_proj = project_to_subspace(X, basis)

      basis_proj = project_to_subspace(basis, basis)

      for i in range(n):
        assert check_in_span(X_proj[:, i], basis_proj)


  def test_deproject_from_subspace(self):
    '''
    Check correctness of deproject_from_subspace(); if a vector was in the subspace to start with,
      then it's deprojected projection should be the same.
    '''
    for seed in range(_NUM_TESTS_):
      key, subkey = random.split(random.PRNGKey(seed))
      
      d = int(random.randint(subkey, minval = 11, maxval = 20, shape = ()))
      key, subkey = random.split(key)

      r = int(random.randint(subkey, minval = 5, maxval = 10, shape = ()))
      key, subkey = random.split(key)

      n = int(random.randint(subkey, minval = 5, maxval = 20, shape = ()))
      key, subkey = random.split(key)

      basis = random.normal(subkey, shape = (d, r))
      key, subkey = random.split(key)

      coeffs = random.normal(subkey, shape = (r, n))
      
      X = basis@coeffs # all in span of basis
      X_proj = project_to_subspace(X, basis)
      X_deproj = deproject_from_subspace(X_proj, basis)

      assert jnp.allclose(X, X_deproj)


if __name__ == "__main__":
  unittest.main()   