import torch
import numpy as np
import abc
from tqdm import trange
from scipy.stats import ortho_group
import torch.nn.functional as F

from losses import get_score_fn, get_score_fn_adj
from utils.graph_utils import mask_adjs, mask_x, gen_noise, gen_spec_noise, gen_spec_noise2
from sde import VPSDE, subVPSDE
import math
from sympy.matrices import Matrix, GramSchmidt


def orthogo_tensor(x):
  m, n = x.size()
  x_np = x.t().numpy()
  matrix = [Matrix(col) for col in x_np.T]
  gram = GramSchmidt(matrix)
  ort_list = []
  for i in range(m):
    vector = []
    for j in range(n):
      vector.append(float(gram[i][j]))
    ort_list.append(vector)
  ort_list = np.mat(ort_list)
  ort_list = torch.from_numpy(ort_list)
  ort_list = F.normalize(ort_list, dim=1)
  return ort_list


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t, flags):
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""
  def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.scale_eps = scale_eps
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t, flags):
    pass


class EulerMaruyamaPredictor(Predictor):
  def __init__(self, obj, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    self.obj = obj

  def update_fn(self, x, adj, flags, t):
    dt = -1. / self.rsde.N
    # print("EulerMaruyamaPredictor update_fn")

    if self.obj=='x':
      z = gen_noise(x, flags, sym=False)
      drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=False)
      x_mean = x + drift * dt
      x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
      return x, x_mean

    elif self.obj=='adj':
      # print("EulerMaruyamaPredictor update adj")
      z = gen_noise(adj, flags)
      # z = gen_spec_noise2(adj, flags, u, la)
      # la, u = torch.symeig(adj, eigenvectors=True)
      # la = torch.diag_embed(la)
      #
      # z = gen_spec_noise(adj, flags, u, la)

      drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=True)
      adj_mean = adj + drift * dt
      adj = adj_mean + diffusion[:, None, None] * np.sqrt(-dt) * z

      return adj, adj_mean

    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")




class EulerMaruyamaPredictor2(Predictor):
  def __init__(self, obj, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    self.obj = obj

  def update_fn(self, x, adj, flags, t, u, la):
    dt = -1. / self.rsde.N
    # print("EulerMaruyamaPredictor update_fn")

    if self.obj=='x':
      z = gen_noise(x, flags, sym=False)
      drift, diffusion = self.rsde.sde(x, adj, flags, t,u, la, is_adj=False)
      x_mean = x + drift * dt
      x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
      return x, x_mean

    elif self.obj=='u':
      z = gen_noise(u, flags, sym=False)
      drift, diffusion = self.rsde.sde(x, adj, flags, t,u, la, is_adj=False, is_u=True)
      u_mean = u + drift * dt
      u = u_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
      return u, u_mean


    elif self.obj=='adj':
      z = gen_spec_noise2(adj, flags, u, la)
      drift, diffusion = self.rsde.sde(x, adj, flags, t,u, la, is_adj=True)
      adj_eigen_mean = la + drift * dt
      adj_eigen = adj_eigen_mean + diffusion[:, None] * np.sqrt(-dt) * z
      u_T = torch.transpose(u, -1, -2)
      adj_eigen_diag = torch.diag_embed(adj_eigen)
      adj_eigen_mean_diag = torch.diag_embed(adj_eigen_mean)
      adj = torch.bmm(torch.bmm(u, adj_eigen_diag), u_T)
      adj_mean = torch.bmm(torch.bmm(u, adj_eigen_mean_diag), u_T)
      adj = mask_adjs(adj, flags)
      adj_mean = mask_adjs(adj_mean, flags)


      return adj, adj_mean, adj_eigen, adj_eigen_mean
      # return adj_eigen, adj_eigen_mean

    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")


class ReverseDiffusionPredictor(Predictor):
  def __init__(self, obj, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    self.obj = obj

  def update_fn(self, x, adj, flags, t):

    if self.obj == 'x':
      f, G = self.rsde.discretize(x, adj, flags, t, is_adj=False)
      z = gen_noise(x, flags, sym=False)
      x_mean = x - f
      x = x_mean + G[:, None, None] * z
      return x, x_mean

    elif self.obj == 'adj':
      f, G = self.rsde.discretize(x, adj, flags, t, is_adj=True)
      z = gen_noise(adj, flags)
      #
      # la, u = torch.symeig(adj, eigenvectors=True)
      # la = torch.diag_embed(la)

      # z = gen_spec_noise(adj, flags, u, la)

      # z = gen_noise(adj, flags)
      adj_mean = adj - f
      adj = adj_mean + G[:, None, None] * z
      return adj, adj_mean
    
    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")


class ReverseDiffusionPredictor2(Predictor):
  def __init__(self, obj, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    self.obj = obj

  def update_fn(self, x, adj, flags, t, u, la):

    if self.obj == 'x':
      f, G = self.rsde.discretize(x, adj, flags, t, is_adj=False)
      z = gen_noise(x, flags, sym=False)
      x_mean = x - f
      x = x_mean + G[:, None, None] * z
      return x, x_mean

    elif self.obj == 'adj':
      # print("in ReverseDiffusionPredictor2 before self.rsde.discretize")
      f, G = self.rsde.discretize(x, adj, flags, t, is_adj=True)
      # z = gen_noise(adj, flags)
      #
      # la, u = torch.symeig(adj, eigenvectors=True)
      # la = torch.diag_embed(la)

      z = gen_spec_noise2(adj, flags, u, la)

      # z = gen_noise(adj, flags)
      adj_mean = adj - f
      adj = adj_mean + G[:, None, None] * z
      return adj, adj_mean

    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")


class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
    self.obj = obj
    pass

  def update_fn(self, x, adj, flags, t):
    if self.obj == 'x':
      return x, x
    elif self.obj == 'adj':
      return adj, adj
    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")

class NoneCorrector2(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
    self.obj = obj
    pass

  def update_fn(self, x, adj, flags, t, u, la):
    if self.obj == 'x':
      return x, x
    elif self.obj == 'adj':
      return adj, adj, la, la
    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")


class LangevinCorrector(Corrector):
  def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
    super().__init__(sde, score_fn, snr, scale_eps, n_steps)
    self.obj = obj

  def update_fn(self, x, adj, flags, t):

    # print("LangevinCorrector update fn")
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    seps = self.scale_eps

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    if self.obj == 'x':
      for i in range(n_steps):
        grad = score_fn(x, adj, flags, t)
        noise = gen_noise(x, flags, sym=False)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + step_size[:, None, None] * grad
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
      return x, x_mean

    elif self.obj == 'adj':
      for i in range(n_steps):
        grad = score_fn(x, adj, flags, t)
        noise = gen_noise(adj, flags)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        adj_mean = adj + step_size[:, None, None] * grad
        adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
      return adj, adj_mean

    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported")


class LangevinCorrector2(Corrector):
  def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
    super().__init__(sde, score_fn, snr, scale_eps, n_steps)
    self.obj = obj

  def update_fn(self, x, adj, flags, t, u, la):

    # print("LangevinCorrector update fn")
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    seps = self.scale_eps

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    if self.obj == 'x':
      for i in range(n_steps):
        grad = score_fn(x, adj, flags, t, u, la)
        noise = gen_noise(x, flags, sym=False)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + step_size[:, None, None] * grad
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
      return x, x_mean

    elif self.obj == 'adj':
      for i in range(n_steps):
        grad = score_fn(x, adj, flags, t, u, la)
        # noise = gen_noise(adj, flags)
        noise = gen_spec_noise2(adj, flags, u, la)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        adj_eigen_mean = la + step_size[:, None] * grad
        adj_eigen = adj_eigen_mean + torch.sqrt(step_size * 2)[:, None] * noise * seps

        u_T = torch.transpose(u, -1, -2)

        adj_eigen_diag = torch.diag_embed(adj_eigen)
        adj_eigen_mean_diag = torch.diag_embed(adj_eigen_mean)
        # print("u:", u.shape, " ")
        adj = torch.bmm(torch.bmm(u, adj_eigen_diag), u_T)
        adj_mean = torch.bmm(torch.bmm(u, adj_eigen_mean_diag), u_T)
        adj = mask_adjs(adj, flags)
        adj_mean = mask_adjs(adj_mean, flags)

      return adj, adj_mean, adj_eigen, adj_eigen_mean

    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported")


# -------- PC sampler --------
def get_pc_sampler(sde_x, sde_adj, shape_x, shape_adj, predictor='Euler', corrector='None', 
                   snr=0.1, scale_eps=1.0, n_steps=1, 
                   probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):

  print("shape_x:", shape_x)
  print("shape_adj:", shape_adj)

  def pc_sampler(model_x, model_adj, init_flags):
    # print("in pc_sampler!!!!!!!!!!!!!!!!!!!!!!!!1")
    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn_adj(sde_adj, model_adj, train=False, continuous=continuous)

    predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor

    corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector

    predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow)
    corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)

    predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow)
    corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device) 
      # adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
      adj = sde_adj.prior_sampling_sym2(shape_adj).to(device)
      # print("after prior_sampling_sym2")
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)
      # diff_steps = 100
      # print("diff_steps:",diff_steps)
      # -------- Reverse diffusion process --------
      for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t

        _x = x
        x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
        adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)

        _x = x
        x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
        adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)
      print(' ')
      return (x_mean if denoise else x), (adj_mean if denoise else adj), diff_steps * (n_steps + 1)
  return pc_sampler


def get_pc_sampler2(sde_x, sde_adj, shape_x, shape_adj, predictor='Euler', corrector='None',
                   snr=0.1, scale_eps=1.0, n_steps=1,
                   probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):

  def pc_sampler(model_x, model_adj, init_flags, train_tensors):
    # la, u = torch.symeig(train_tensors, eigenvectors=True)
    la, u = torch.linalg.eigh(train_tensors)
    u = u.to(device)
    u_T = torch.transpose(u, -1, -2)
    print('init_flags:', init_flags)
    num_nodes = torch.sum(init_flags, dim=1)
    print('num_nodes:', num_nodes)
    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn_adj(sde_adj, model_adj, train=False, continuous=continuous)

    # predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor
    predictor_fn = ReverseDiffusionPredictor2 if predictor == 'Reverse' else EulerMaruyamaPredictor2

    corrector_fn = LangevinCorrector2 if corrector == 'Langevin' else NoneCorrector2

    predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow)
    corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)
    predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow)
    corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)



    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device)
      adj_eigen = sde_adj.prior_sampling_sym3(la.shape, u).to(device)
      flags = init_flags
      x = mask_x(x, flags)
      eigen_diag = torch.diag_embed(adj_eigen)
      adj = torch.bmm(torch.bmm(u, eigen_diag), u_T)

      adj = mask_adjs(adj, flags)
      nonzero_count_flag = torch.zeros((flags.shape[0]))
      eigen_mask = torch.zeros((adj.shape[0],adj.shape[1]))
      for i in range(flags.shape[0]):
        count = torch.count_nonzero(flags[i])
        nonzero_count_flag[i] = count
        eigen_mask[i,:count//2] = 1
        eigen_mask[i, -count//2:] = 1
        # print("eigen_mask:",eigen_mask[i])
      eigen_mask = eigen_mask.to(adj.device)
      diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)
      # -------- Reverse diffusion process --------
      la = adj_eigen
      for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
        # print("in sampler loop!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t

        _x = x
        x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t,u, la)
        # print("before corrector_obj_adj:",adj.shape)
        adj, adj_mean, adj_eigen, adj_mean_eigen = corrector_obj_adj.update_fn(_x, adj, flags, vec_t, u, la)
        # print("after corrector_obj_adj:", adj.shape, adj_mean.shape)
        # la = adj_eigen
        la = adj_mean_eigen
        la = la*eigen_mask
        _x = x
        x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t,  u, la)
        adj, adj_mean,adj_eigen, adj_mean_eigen  = predictor_obj_adj.update_fn(_x, adj, flags, vec_t, u, la)
        la = adj_mean_eigen
        la = la * eigen_mask
      return (x_mean if denoise else x), (adj_mean if denoise else adj), diff_steps * (n_steps + 1)
  return pc_sampler
