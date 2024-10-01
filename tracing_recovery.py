'''
tracing_recovery.py

Functions for the naive low noise tracing-based curve recovery algorithm.
'''


from data_generation import *
from moments import *
from jax.scipy.stats import gaussian_kde
from scipy.signal import find_peaks


dists_from_cloud_to_pt = vmap(lambda cloud, pt: jnp.linalg.norm(cloud - pt), in_axes = [0, None])
'''
Compute the distances from a cloud to a point.

INPUTS:
  cloud: (N, d)-shaped point cloud of N points in R^d
  x: (d,)-shaped point

OUTPUTS: 
  dists: (N,)-shaped array of distances from each point in cloud to point
'''


def compute_local_cloud(cloud, pt, r):
  '''
  Extract the points in a point cloud within radius r of a point pt.

  INPUTS:
    cloud: (num_pts, d)-shaped array, a point cloud
    pt: (d,)-shaped array 
    r: positive float
  
  OUTPUTS:
    local_cloud: (cloud_size, d)-shaped array of points in point cloud within radius r of point pt
  '''
  local_cloud = cloud[jnp.squeeze(jnp.argwhere(dists_from_cloud_to_pt(cloud, pt) <= r))]
  return local_cloud


def find_elbow_angles(elbow_cloud_2d_centered):
  '''
  Given a 2D elbow-shaped cloud centered at the joint, estimate the angles of the arms.

  INPUTS:  
    elbow_cloud_2d_centered: (N, 2)-shaped 2d point cloud coming from an elbow whose joint is at
      the origin
  
  OUTPUTS:
    angles_rad: (2,)-shaped array of estimates of the angles of the two arms, in radians
  '''
  elbow_angles = (jnp.arctan2(elbow_cloud_2d_centered[:, 1], elbow_cloud_2d_centered[:, 0]) * 180 / jnp.pi) % 360 # in degrees from 0 to 360
  shift = jnp.median(elbow_angles) # median should be between peaks, this maximizes peak separation I think
  elbow_angles_shifted = (elbow_angles - shift) % 360
  dense_angles = jnp.linspace(0, 360, num = 3600)
  pdf = gaussian_kde(elbow_angles_shifted)(dense_angles)

  peaks, metadata = find_peaks(pdf, width = 50)
  
  max_prom_idxs = jnp.argsort(metadata["prominences"])[-2:]
  max_angle_idxs = peaks[max_prom_idxs]
  angles_deg = (dense_angles[max_angle_idxs] + shift) % 360
  angles_rad = angles_deg * jnp.pi / 180

  return angles_rad


def resolve_Rd_elbow(elbow_cloud, elbow_joint, seg_len):
  '''
  Given the estimated center joint of a point cloud coming from an elbow in R^d and known segment 
    lengths, estimate the other two endpoints of the elbow.

  INPUTS:
    elbow_cloud: (N, d)-shaped point cloud coming from an elbow in R^d
    elbow_joint: estimated joint of the elbow
  '''
  # top two eigs of elbow with joint at origin give basis for plane of elbow
  m2hat = jnp.mean(jnp.einsum("ij, ik -> ijk", elbow_cloud - elbow_joint, elbow_cloud - elbow_joint), axis = 0)
  _, evecs = jnp.linalg.eigh(m2hat) # eigs sorted in increasing order
  elbow_basis = evecs[:, -2:] # shape (d, 2), [:, 0] and [:, 1] are basis vectors

  # projection onto basis formed by orthonormal columns of X is X@X.T
  # transposes are because data matrix (elbow_cloud) is rowwise instead of column wise
  elbow_cloud_proj = (elbow_basis@elbow_basis.T@elbow_cloud.T).T
  elbow_cloud_2d = (elbow_basis.T@elbow_cloud_proj.T).T
  elbow_cloud_2d_centered = elbow_cloud_2d - elbow_basis.T@elbow_joint

  angles_rad = find_elbow_angles(elbow_cloud_2d_centered)
  rays_in_basis = jnp.array([
    [jnp.cos(angles_rad[0]), jnp.sin(angles_rad[0])], 
    [jnp.cos(angles_rad[1]), jnp.sin(angles_rad[1])]
  ])

  rays_in_space = elbow_basis@rays_in_basis.T

  c0_pred = elbow_joint + rays_in_space[:, 0] * seg_len
  c2_pred = elbow_joint + rays_in_space[:, 1] * seg_len

  return c0_pred, c2_pred

def resolve_Rd_c1(c0_cloud, c0, seg_len):
  '''
  Given ground truth c0 and the points in the cloud within a ball centered at c0, estimate c1.

  INPUTS:
    c0_cloud: (cloud_size, d)-shaped array of points in point cloud within some radius (probably
      seg_len / 2) of c0
    c0: (d,)-shaped array, the ground truth starting point of the curve
    seg_len: positive float, the (known) length of the line segment

  OUTPUTS:
    c1_pred: (d,)-shaped array, predicted next segment.
  '''
  m2hat = jnp.mean(jnp.einsum("ij, ik -> ijk", c0_cloud - c0, c0_cloud - c0), axis = 0)
  _, evecs = jnp.linalg.eigh(m2hat) # eigs sorted in increasing order
  seg_basis = evecs[:, -1] # only take last evec now

  unflipped_dists = jnp.sum(jnp.linalg.norm(c0 + seg_basis * seg_len - c0_cloud, axis = -1))
  flipped_dists = jnp.sum(jnp.linalg.norm(c0 - seg_basis * seg_len - c0_cloud, axis = -1))

  # equal to 1 if no need to flip, -1 otherwise
  dont_flip = ((flipped_dists > unflipped_dists) - 0.5) * 2

  c1_pred = c0 + dont_flip * seg_basis * seg_len
  return c1_pred


def trace_cloud(cloud, c0, seg_len, M):
  '''
  Trace out a full cloud coming from a piecewise linear uniform segment length curve.

  INPUTS:
    cloud: (N, d)-shaped array, point cloud of N points in R^d
    c0: (d,) shaped array, estimated starting point of curve
    seg_len: float, known segment length
    M: int, number of segments in curve

  OUTPUTS:
    C_pred: (M+1, d)-shaped predicted curve.
  '''
  c0_cloud = compute_local_cloud(cloud, c0, seg_len)
  c1_pred = resolve_Rd_c1(c0_cloud, c0, seg_len)
  C_pred = jnp.array([c0, c1_pred])

  for _ in range(1, M):
    elbow_joint = C_pred[-1]
    elbow_cloud = compute_local_cloud(cloud, elbow_joint, seg_len)
    cprev_pred, cnext_pred = resolve_Rd_elbow(elbow_cloud, elbow_joint, seg_len)
    if jnp.linalg.norm(cprev_pred - C_pred[-2]) >= jnp.linalg.norm(cnext_pred - C_pred[-2]):
      foo = jnp.copy(cnext_pred)
      bar = jnp.copy(cprev_pred)
      cprev_pred = foo
      cnext_pred = bar
    C_pred = jnp.append(C_pred, cnext_pred[jnp.newaxis, :], axis = 0)

  return C_pred


