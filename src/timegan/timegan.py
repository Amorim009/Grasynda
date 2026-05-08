import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .data import batch_generator
from .utils import extract_time, random_generator, NormMinMax
from .model import Encoder, Recovery, Generator, Discriminator, Supervisor
import matplotlib.pyplot as plt

def plot_progress(history, path):
    """Saves a plot of the training progress to the given path."""
    try:
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot E Loss
        if history['e_loss']:
            ax[0].plot(history['e_loss'], color='blue')
        ax[0].set_title('Autoencoder Loss')
        ax[0].set_xlabel('Steps')
        ax[0].set_ylabel('Loss')
        
        # Plot S Loss
        if history['s_loss']:
            ax[1].plot(history['s_loss'], color='orange')
        ax[1].set_title('Supervisor Loss')
        ax[1].set_xlabel('Steps')
        ax[1].set_ylabel('Loss')

        # Plot GAN Loss
        if history['d_loss']:
            ax[2].plot(history['d_loss'], label='Discriminator', color='red', alpha=0.6)
        if history['g_loss']:
            ax[2].plot(history['g_loss'], label='Generator', color='green', alpha=0.6)
        ax[2].set_title('GAN Loss')
        ax[2].set_xlabel('Steps')
        ax[2].set_ylabel('Loss')
        ax[2].legend()
        
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)
    except Exception as e:
        print(f"Plotting failed: {e}")

class BaseModel():

  def __init__(self, opt, ori_data):
    self.seed(opt.manualseed)

    self.opt = opt
    self.ori_data, self.min_val, self.max_val = NormMinMax(ori_data)
    self.ori_time, self.max_seq_len = extract_time(self.ori_data)
    self.data_num, _, _ = np.asarray(ori_data).shape
    self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")
    
    # History for visualization
    self.history = {
        'e_loss': [], 
        's_loss': [], 
        'g_loss': [], 
        'd_loss': []
    }

  def seed(self, seed_value):
    if seed_value == -1:
      return
    import random
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True

  def train_one_iter_er(self):
    self.nete.train()
    self.netr.train()

    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    self.X = torch.tensor(np.asarray(self.X0), dtype=torch.float32).to(self.device)

    self.optimize_params_er()

  def train_one_iter_er_(self):
    self.nete.train()
    self.netr.train()

    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    self.X = torch.tensor(np.asarray(self.X0), dtype=torch.float32).to(self.device)

    self.optimize_params_er_()

  def train_one_iter_s(self):
    self.nets.train()

    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    self.X = torch.tensor(np.asarray(self.X0), dtype=torch.float32).to(self.device)

    self.optimize_params_s()

  def train_one_iter_g(self):
    self.netg.train()

    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    self.X = torch.tensor(np.asarray(self.X0), dtype=torch.float32).to(self.device)
    self.Z = random_generator(self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len)

    self.optimize_params_g()

  def train_one_iter_d(self):
    self.netd.train()

    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    self.X = torch.tensor(np.asarray(self.X0), dtype=torch.float32).to(self.device)
    self.Z = random_generator(self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len)

    self.optimize_params_d()


  def train(self):
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback if tqdm is not installed
        def tqdm(iterable, desc=""):
             print(f"Starting {desc}...")
             return iterable
    
    # Phase 1: Autoencoder
    print(f"[TimeGAN] Phase 1: Autoencoder Training ({self.opt.iteration} steps)")
    pbar = tqdm(range(self.opt.iteration), desc="Autoencoder")
    for iter in pbar:
      self.train_one_iter_er()
      self.history['e_loss'].append(self.err_er.item())
      
      if (iter + 1) % 100 == 0:
          pbar.set_postfix({'e_loss': f"{self.err_er.item():.6f}"})
          if hasattr(self.opt, 'plot_path') and self.opt.plot_path:
              plot_progress(self.history, self.opt.plot_path)

    # Phase 2: Supervisor
    print(f"[TimeGAN] Phase 2: Supervisor Training ({self.opt.iteration} steps)")
    pbar = tqdm(range(self.opt.iteration), desc="Supervisor")
    for iter in pbar:
      self.train_one_iter_s()
      self.history['s_loss'].append(self.err_s.item())
      
      if (iter + 1) % 100 == 0:
          pbar.set_postfix({'s_loss': f"{self.err_s.item():.6f}"})
          if hasattr(self.opt, 'plot_path') and self.opt.plot_path:
              plot_progress(self.history, self.opt.plot_path)

    # Phase 3: Joint
    print(f"[TimeGAN] Phase 3: Joint GAN Training ({self.opt.iteration} steps)")
    pbar = tqdm(range(self.opt.iteration), desc="Joint")
    for iter in pbar:
      for kk in range(2):
        self.train_one_iter_g()
        self.train_one_iter_er_()
      
      self.train_one_iter_d()
      
      self.history['g_loss'].append(self.err_g.item())
      self.history['d_loss'].append(self.err_d.item())
      
      if (iter + 1) % 100 == 0:
          pbar.set_postfix({
              'd_loss': f"{self.err_d.item():.6f}",
              'g_loss': f"{self.err_g.item():.6f}"
          })
          if hasattr(self.opt, 'plot_path') and self.opt.plot_path:
              plot_progress(self.history, self.opt.plot_path)

    self.generated_data = self.generation(self.opt.batch_size)
    if self.opt.verbose:
      print('Finish Synthetic Data Generation')


  def generation(self, num_samples):
    if num_samples == 0:
      return None
    self.X0, t_seed = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    if len(t_seed) < num_samples:
      repeat = (num_samples // len(t_seed)) + 1
      self.T = (t_seed * repeat)[:num_samples]
    else:
      self.T = t_seed[:num_samples]
    self.Z = random_generator(num_samples, self.opt.z_dim, self.T, self.max_seq_len)
    self.Z = torch.tensor(np.array(self.Z), dtype=torch.float32).to(self.device)
    self.E_hat = self.netg(self.Z)
    self.H_hat = self.nets(self.E_hat)
    generated_data_curr = self.netr(self.H_hat, sigmoid=self.recovery_sigmoid).cpu().detach().numpy()

    generated_data = list()
    for i in range(num_samples):
      temp = generated_data_curr[i, :self.ori_time[i % len(self.ori_time)], :]
      generated_data.append(temp)

    generated_data = np.array(generated_data)
    generated_data = generated_data * self.max_val
    generated_data = generated_data + self.min_val
    return generated_data



class TimeGAN(BaseModel):

    @property
    def name(self):
      return 'TimeGAN'

    def __init__(self, opt, ori_data):
      super(TimeGAN, self).__init__(opt, ori_data)

      self.epoch = 0
      self.times = []
      self.total_steps = 0

      self.nete = Encoder(self.opt).to(self.device)
      self.netr = Recovery(self.opt).to(self.device)
      self.netg = Generator(self.opt).to(self.device)
      self.netd = Discriminator(self.opt).to(self.device)
      self.nets = Supervisor(self.opt).to(self.device)

      if self.opt.resume != '':
        print("\nLoading pre-trained networks.")
        self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
        self.nete.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netE.pth'))['state_dict'])
        self.netr.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netR.pth'))['state_dict'])
        self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
        self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
        self.nets.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netS.pth'))['state_dict'])
        print("\tDone.\n")

      self.l_mse = nn.MSELoss()
      self.l_r = nn.L1Loss()
      self.l_bce = nn.BCELoss()
      self.recovery_sigmoid = getattr(self.opt, "recovery_sigmoid", True)
      self.w_deriv = float(getattr(self.opt, "w_deriv", 0.0))

      if self.opt.isTrain:
        self.nete.train()
        self.netr.train()
        self.netg.train()
        self.netd.train()
        self.nets.train()
        self.optimizer_e = optim.Adam(self.nete.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_r = optim.Adam(self.netr.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_s = optim.Adam(self.nets.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))


    def forward_e(self):
      self.H = self.nete(self.X)

    def forward_er(self):
      self.H = self.nete(self.X)
      self.X_tilde = self.netr(self.H, sigmoid=self.recovery_sigmoid)

    def forward_g(self):
      self.Z = torch.tensor(np.array(self.Z), dtype=torch.float32).to(self.device)
      self.E_hat = self.netg(self.Z)

    def forward_dg(self):
      self.Y_fake = self.netd(self.H_hat)
      self.Y_fake_e = self.netd(self.E_hat)

    def forward_rg(self):
      self.X_hat = self.netr(self.H_hat, sigmoid=self.recovery_sigmoid)

    def forward_s(self):
      self.H_supervise = self.nets(self.H)

    def forward_sg(self):
      self.H_hat = self.nets(self.E_hat)

    def forward_d(self):
      self.Y_real = self.netd(self.H)
      self.Y_fake = self.netd(self.H_hat)
      self.Y_fake_e = self.netd(self.E_hat)


    def backward_er(self):
      self.err_er_recon = self.l_mse(self.X_tilde, self.X)
      if self.w_deriv > 0:
        d_real = self.X[:, 1:, :] - self.X[:, :-1, :]
        d_fake = self.X_tilde[:, 1:, :] - self.X_tilde[:, :-1, :]
        self.err_er_deriv = self.l_mse(d_fake, d_real)
        self.err_er = self.err_er_recon + self.w_deriv * self.err_er_deriv
      else:
        self.err_er = self.err_er_recon
      self.err_er.backward(retain_graph=True)

    def backward_er_(self):
      self.err_er_ = self.l_mse(self.X_tilde, self.X)
      if self.w_deriv > 0:
        d_real = self.X[:, 1:, :] - self.X[:, :-1, :]
        d_fake = self.X_tilde[:, 1:, :] - self.X_tilde[:, :-1, :]
        self.err_er_ = self.err_er_ + self.w_deriv * self.l_mse(d_fake, d_real)
      self.err_s = self.l_mse(self.H_supervise[:,:-1,:], self.H[:,1:,:])
      self.err_er = 10 * torch.sqrt(self.err_er_) + 0.1 * self.err_s
      self.err_er.backward(retain_graph=True)

    def backward_g(self):
      self.err_g_U = self.l_bce(self.Y_fake, torch.ones_like(self.Y_fake))
      self.err_g_U_e = self.l_bce(self.Y_fake_e, torch.ones_like(self.Y_fake_e))
      self.err_g_V1 = torch.mean(torch.abs(torch.sqrt(torch.std(self.X_hat,[0])[1] + 1e-6) - torch.sqrt(torch.std(self.X,[0])[1] + 1e-6)))
      self.err_g_V2 = torch.mean(torch.abs((torch.mean(self.X_hat,[0])[0]) - (torch.mean(self.X,[0])[0])))
      self.err_s = self.l_mse(self.H_supervise[:,:-1,:], self.H[:,1:,:])
      self.err_g = self.err_g_U + \
                   self.err_g_U_e * self.opt.w_gamma + \
                   self.err_g_V1 * self.opt.w_g + \
                   self.err_g_V2 * self.opt.w_g + \
                   torch.sqrt(self.err_s) 
      self.err_g.backward(retain_graph=True)

    def backward_s(self):
      self.err_s = self.l_mse(self.H[:,1:,:], self.H_supervise[:,:-1,:])
      self.err_s.backward(retain_graph=True)

    def backward_d(self):
      self.err_d_real = self.l_bce(self.Y_real, torch.ones_like(self.Y_real))
      self.err_d_fake = self.l_bce(self.Y_fake, torch.zeros_like(self.Y_fake))
      self.err_d_fake_e = self.l_bce(self.Y_fake_e, torch.zeros_like(self.Y_fake_e))
      self.err_d = self.err_d_real + \
                   self.err_d_fake + \
                   self.err_d_fake_e * self.opt.w_gamma
      if self.err_d > 0.15:
        self.err_d.backward(retain_graph=True)

    def optimize_params_er(self):
      self.forward_er()

      self.optimizer_e.zero_grad()
      self.optimizer_r.zero_grad()
      self.backward_er()
      self.optimizer_e.step()
      self.optimizer_r.step()

    def optimize_params_er_(self):
      self.forward_er()
      self.forward_s()

      self.optimizer_e.zero_grad()
      self.optimizer_r.zero_grad()
      self.backward_er_()
      self.optimizer_e.step()
      self.optimizer_r.step()

    def optimize_params_s(self):
      self.forward_e()
      self.forward_s()

      self.optimizer_s.zero_grad()
      self.backward_s()
      self.optimizer_s.step()

    def optimize_params_g(self):
      self.forward_e()
      self.forward_s()
      self.forward_g()
      self.forward_sg()
      self.forward_rg()
      self.forward_dg()

      self.optimizer_g.zero_grad()
      self.optimizer_s.zero_grad()
      self.backward_g()
      self.optimizer_g.step()
      self.optimizer_s.step()

    def optimize_params_d(self):
      self.forward_e()
      self.forward_g()
      self.forward_sg()
      self.forward_d()
      self.forward_dg()

      self.optimizer_d.zero_grad()
      self.backward_d()
      self.optimizer_d.step()
