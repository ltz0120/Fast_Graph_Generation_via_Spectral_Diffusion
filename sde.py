import abc
import torch
import numpy as np
from scipy.stats import ortho_group


class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.
    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.
    Useful for computing the log-likelihood via probability flow ODE.
    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.
    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.
    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)
    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.
    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # -------- Build the class for reverse-time SDE --------
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, feature, x, flags, t,u,la,  is_adj=True, is_u = False):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        # drift, diffusion = sde_fn(x, t) if is_adj else sde_fn(feature, t)
        #print("sde_fn:",sde_fn)

        if is_u:
          drift, diffusion = sde_fn(u, t, is_adj=False)
          score = score_fn(feature, x, flags, t, u, la)
          drift = drift - diffusion[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)

        else:
          drift, diffusion = sde_fn(la, t, is_adj=True) if is_adj else sde_fn(feature, t, is_adj=False)

          if not is_adj:
            # #print("is_adj:", False,"score_fn:",score_fn)
            score = score_fn(feature, x, flags, t, u, la)
          else:
            # #print("is_adj:", True, "score_fn:",score_fn)
            score = score_fn(feature, x, flags, t, u, la)
            # #print("score_fn:",score_fn)
          #print("drift:",drift.shape, "is_adj:",is_adj, " diffusion:", diffusion.shape, " score:",score.shape)
          if is_adj:
            drift = drift - diffusion[:, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
          else:
            drift = drift - diffusion[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
          # -------- Set the diffusion function to zero for ODEs. --------
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, feature, x, flags, t, is_adj=True):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t) if is_adj else discretize_fn(feature, t)
        #print('after discretize_fn, is_adj:', is_adj, " f:", f, "G:",G)
        score = score_fn(feature, x, flags, t)
        rev_f = f - G[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()


class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.
    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    print("num of steps:", N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t, is_adj = True):

    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    if is_adj:
      drift = -0.5 * beta_t[:, None] * x
    else:
      drift = -0.5 * beta_t[:, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def select_type(self, type):
    print("in select type, type:", type)

    if type=="linear":
      self.marginal_prob = self.marginal_prob_ori
      self.marginal_prob_adj = self.marginal_prob_adj_ori
      self.discrete_betas = torch.linspace(self.beta_0 / self.N, self.beta_1 / self.N, self.N)
    elif type=="exp":
      self.marginal_prob = self.marginal_prob_exp
      self.marginal_prob_adj = self.marginal_prob_adj_exp
      t = torch.linspace(self.beta_0 / self.N, self.beta_1 / self.N, self.N)
      self.discrete_betas = self.beta_t_exp(t)
      print("discrete_betas:", self.discrete_betas.shape)
    elif type=="cosine":
      self.marginal_prob = self.marginal_prob_cosine
      self.marginal_prob_adj = self.marginal_prob_adj_cosine
      t = torch.linspace(self.beta_0 / self.N, self.beta_1 / self.N, self.N)
      self.discrete_betas = self.beta_t_cosine(t)
    elif type=="tanh":
      self.marginal_prob = self.marginal_prob_tanh
      self.marginal_prob_adj = self.marginal_prob_adj_tanh

    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
  # def sde_adj(self, la, t):
  #   beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
  #   drift = -0.5 * beta_t[:, None] * la
  #   diffusion = torch.sqrt(beta_t)
  #   return drift, diffusion

  def beta_t_exp(self, t):
    beta = torch.exp(t*torch.log(torch.tensor(self.beta_1 - self.beta_0 +1))) -1 + self.beta_0
    return beta
  def beta_t_cosine(self, t):
    beta = torch.cos(torch.tensor(3.14 + t/(3.14/2)))* (self.beta_1 - self.beta_0) + self.beta_0 + 1
    return beta
  # def beta_t_tanh(self, t):
    # beta = (self.beta_1 - self.beta_0)/(0-torch.tanh(torch.tensor(-3)))* torch.tanh()
    # return beta
  def marginal_prob(self, x, t):
  # def marginal_prob(self, x, t):
    # #print("VPSDE marginal_prob")
    # log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    #
    # # print("log_mean_coeff:", log_mean_coeff)
    # mean = torch.exp(log_mean_coeff[:, None, None]) * x
    # std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return None

  # -------- mean, std of the perturbation kernel --------
  def marginal_prob_ori(self, x, t):
  # def marginal_prob(self, x, t):
    # #print("VPSDE marginal_prob")
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    # print("log_mean_coeff:", log_mean_coeff)
    mean = torch.exp(log_mean_coeff[:, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std


  # -------- marginal_prob_exp --------
  def marginal_prob_exp(self, x, t):
    temp = torch.tensor(self.beta_1 - self.beta_0+1).float()
    log_part = torch.log(temp)
    log_mean_coeff = -0.5 * (1/log_part)* torch.exp(t * log_part)  - 0.5 * t * (self.beta_0-1)
    mean = torch.exp(log_mean_coeff[:, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  # -------- marginal_prob_cosine --------
  def marginal_prob_cosine(self, x, t):
  # def marginal_prob(self, x, t):
    # #print("VPSDE marginal_prob")
    # log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    # log_mean_coeff = -0.5*(3.14/2 * (self.beta_1 - self.beta_0) * (torch.sin(t/(3.14/2) + 3.14) - torch.sin(torch.tensor(3.14))) + t*self.beta_0)
    # log_mean_coeff = -0.5 * (3.14 / 2 * (self.beta_1 - self.beta_0) * (
    #           torch.sin(t / (3.14 / 2) + 3.14)) + t * self.beta_0)
    log_mean_coeff = -0.5 * (3.14 / 2 * (self.beta_1 - self.beta_0) * (
              torch.sin(t / (3.14 / 2) + 3.14)) + t * (1+self.beta_0))
    # print("log_mean_coeff:",log_mean_coeff)
    mean = torch.exp(log_mean_coeff[:, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  # def marginal_prob_adj(self, x, t, u, la):
  def marginal_prob_adj_ori(self, x, t, u, la):
    # #print("VPSDE marginal_prob")
    # print("in marginal_prob_adj_ori")
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    # print("x:",x.shape, 'log_mean_coeff:', log_mean_coeff.shape, "u:",u.shape, "la:", la.shape)
    mean = torch.exp(log_mean_coeff[:, None]) * la

    # mean = torch.exp(log_mean_coeff[:, None, None]) * x
    # #print("mean:", mean.shape)
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std


  def marginal_prob_adj_exp(self, x, t, u, la):
    temp = torch.tensor(self.beta_1 - self.beta_0+1).float()
    log_part = torch.log(temp)
    log_mean_coeff = -0.5 * (1/log_part)* torch.exp(t * log_part)  - 0.5 * t * (self.beta_0-1)
    mean = torch.exp(log_mean_coeff[:, None]) * la
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  # def marginal_prob_adj_cosine(self, x, t, u, la):
  def marginal_prob_adj(self, x, t, u, la):
    # #print("VPSDE marginal_prob")

    log_mean_coeff = -0.5 * (3.14 / 2 * (self.beta_1 - self.beta_0) * (
      torch.sin(t / (3.14 / 2) + 3.14)) + t * (1 + self.beta_0))

    mean = torch.exp(log_mean_coeff[:, None]) * la

    # mean = torch.exp(log_mean_coeff[:, None, None]) * x
    # #print("mean:", mean.shape)
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def marginal_prob_u(self, x, t, u, la):
    # #print("VPSDE marginal_prob")

    log_mean_coeff = -0.5 * (3.14 / 2 * (self.beta_1 - self.beta_0) * (
      torch.sin(t / (3.14 / 2) + 3.14)) + t * (1 + self.beta_0))

    mean = torch.exp(log_mean_coeff[:, None, None]) * u

    # mean = torch.exp(log_mean_coeff[:, None, None]) * x
    # #print("mean:", mean.shape)
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def marginal_prob_std_fast(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return std

  def marginal_prob_std_fast2(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_sampling_sym(self, shape):
    x = torch.randn(*shape).triu(1) 
    return x + x.transpose(-1,-2)

  def prior_sampling_sym2(self, shape):
    # #print("shape:", shape)

    for i in range(shape[0]):
      m = torch.tensor(ortho_group.rvs(dim=shape[-1]))
      m = m.unsqueeze(0)
      if i==0:
        vec = m
      else:
        vec = torch.concat((vec,m), dim=0)
    vec = vec.float()
    vec_T = torch.transpose(vec, -1,-2)
    z = torch.randn(shape)
    eye = torch.eye(shape[-1])
    eye = eye.unsqueeze(0)
    eye = eye.repeat(shape[0], 1, 1)
    z = z * eye
    z = torch.bmm(torch.bmm(vec, z), vec_T) * np.sqrt(z.shape[-1])
    z = z.triu(1)
    return z + z.transpose(-1,-2)

  def prior_sampling_sym3(self, shape, u):
    z = torch.randn(shape)
    return z

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None] * x - x
    G = sqrt_beta
    return f, G

  def transition(self, x, t, dt):
    # -------- negative timestep dt --------
    log_mean_coeff = 0.25 * dt * (2*self.beta_0 + (2*t + dt)*(self.beta_1 - self.beta_0) )
    mean = torch.exp(-log_mean_coeff[:, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std


class VESDE(SDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
    """Construct a Variance Exploding SDE.
    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t, is_adj = True):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def marginal_prob_adj(self, x, t, u, la):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    # mean = x
    mean = la
    return mean, std

  # def marginal_prob_adj(self, x, t, u, la):
  #   # #print("VPSDE marginal_prob")
  #
  #   log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
  #   # #print("adj:", x.shape)
  #   # #print("log_mean_coeff:",log_mean_coeff.shape)
  #   la = torch.exp(log_mean_coeff[:, None, None]) * la
  #   u_T = torch.transpose(u, -1, -2)
  #   mean = torch.bmm(torch.bmm(u, la), u_T) * np.sqrt(la.shape[-1])
  #
  #
  #   # mean = torch.exp(log_mean_coeff[:, None, None]) * x
  #   # #print("mean:", mean.shape)
  #   std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
  #   return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape) 

  def prior_sampling_sym(self, shape):
    x = torch.randn(*shape).triu(1)
    x = x + x.transpose(-1,-2)
    return x

  def prior_sampling_sym2(self, shape):
    #print("shape:", shape)

    for i in range(shape[0]):
      m = torch.tensor(ortho_group.rvs(dim=shape[-1]))
      m = m.unsqueeze(0)
      if i==0:
        vec = m
      else:
        vec = torch.concat((vec,m), dim=0)
    vec = vec.float()
    vec_T = torch.transpose(vec, -1,-2)
    z = torch.randn(shape)
    eye = torch.eye(shape[-1])
    eye = eye.unsqueeze(0)
    eye = eye.repeat(shape[0], 1, 1)
    z = z * eye
    # #print("z:", z.shape)
    # #print("vec:", vec.shape, "vec_T:",vec_T.shape)
    # temp = np.sqrt(z.shape[-1])
    # #print("temp:",temp)
    z = torch.bmm(torch.bmm(vec, z), vec_T) * np.sqrt(z.shape[-1])
    z = z.triu(1)
    return z + z.transpose(-1,-2)
    # x = torch.randn(*shape).triu(1)
    # #print("x:", x.shape)
    # return x + x.transpose(-1,-2)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

  def discretize(self, x, t):
    """SMLD(NCSN) discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    sigma = self.discrete_sigmas.to(t.device)[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 self.discrete_sigmas[timestep - 1].to(t.device))
    f = torch.zeros_like(x)
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G

  def transition(self, x, t, dt):
    # -------- negative timestep dt --------
    std = torch.square(self.sigma_min * (self.sigma_max / self.sigma_min) ** t) - \
          torch.square(self.sigma_min * (self.sigma_max / self.sigma_min) ** (t + dt)) 
    std = torch.sqrt(std)
    mean = x
    return mean, std


class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.
    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[:, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_sampling_sym(self, shape):
    x = torch.randn(*shape).triu(1) 
    return x + x.transpose(-1,-2)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
