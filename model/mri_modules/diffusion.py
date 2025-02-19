import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import copy
from .utils import *
import matplotlib.pyplot as plt
TTT = False # test time training not enabled

# LOSS = 'p2s'

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def _rev_warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_start * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[n_timestep - warmup_time:] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'rev_warmup80':
        betas = _rev_warmup_beta(linear_start, linear_end,
                             n_timestep, 0.8)
    elif schedule == 'rev_warmup70':
        betas = _rev_warmup_beta(linear_start, linear_end,
                             n_timestep, 0.7)
    elif schedule == 'rev_warmup90':
        betas = _rev_warmup_beta(linear_start, linear_end,
                             n_timestep, 0.9)
    elif schedule == 'rev_warmup50':
        betas = _rev_warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)

    print('New beta scheduler set!', schedule)
    return betas


# gaussian diffusion trainer class

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoisor,
        image_size,
        channels=3,
        drop_rate=0.3,
        loss_type='p2s',
        conditional=True,
        schedule_opt=None,
        denoise_fn=None
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.channels = channels
        self.image_size = image_size

        self.denoisor = denoisor # for stage 3
        self.denoise_fn = denoise_fn # for stage 1
        self.loss_type = loss_type
        self.conditional = conditional

        # for TTT
        if TTT:
            optim_params = []
            for k, v in self.named_parameters():
                if k.find('matched_state') >= 0:
                    continue
                if k.find('noise_model_variance') >= 0:
                    continue
                optim_params.append(v)
            print('ttt optimizing params:', len(optim_params))
            self.ttt_opt = torch.optim.Adam(optim_params, lr=1e-4)

        if schedule_opt is not None:
            self.set_new_noise_schedule(schedule_opt, device=torch.device('cuda:0'))


    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            print('s2s noise activated!')
            self.mseloss = nn.MSELoss().to(device)
            self.l1loss = nn.L1Loss().to(device)

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        alphas_cumprod_prev10 = np.concatenate((np.ones(10), alphas_cumprod[:-10]))
        
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))
        
        #print(betas.shape, alphas_cumprod.shape, alphas_cumprod_prev.shape, '***')

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))


        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        

        posterior_variance10 = betas * \
            (1. - alphas_cumprod_prev10) / (1. - alphas_cumprod)
        
        self.register_buffer('posterior_variance10',
                             to_torch(posterior_variance10))
        
        self.register_buffer('posterior_log_variance_clipped_run', to_torch(
            np.log(np.maximum(posterior_variance10, 1e-20))))
        
        self.register_buffer('posterior_mean_coef1_run', to_torch(
            betas * np.sqrt(alphas_cumprod_prev10) / (1. - alphas_cumprod)))
        
        self.register_buffer('posterior_mean_coef2_run', to_torch(
            (1. - alphas_cumprod_prev10) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        
        

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise


    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]

        return posterior_mean, posterior_log_variance_clipped
    
    def q_posterior_run(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1_run[t] * \
            x_start + self.posterior_mean_coef2_run[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped_run[t]

        return posterior_mean, posterior_log_variance_clipped
    

    def p_mean_variance(self, x, t, clip_denoised: bool, mask=None, condition_x=None, 
                        mask_condition=None, ttt_opt=None,what_you_want_for_sample=None):
        
        b, c, w, h = x.shape
 
        single_noise_level = torch.FloatTensor(
        [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(b, 1).to(x.device)

        if ttt_opt is None:
            with torch.no_grad():
                x_recon = flip_denoise(x, self.denoisor, single_noise_level.expand(4, -1), 
                                      flips = [(False, False), (True, False), (False, True), (True, True)])
                #x_recon = self.denoisor(x,single_noise_level)
                
        else:
            # TTT
            ttt_opt.zero_grad()

            x_recon = flip_denoise(x, self.denoisor, single_noise_level.expand(4, -1), 
                                   flips = [(False, False), (True, False), (False, True), (True, True)])

            ttt_loss = self.mseloss(x_recon, condition_x.detach())
            ttt_loss.requires_grad = True
            ttt_loss.backward()

            ttt_opt.step()

            self.eval()
            x_recon = flip_denoise(x, self.denoisor, single_noise_level.expand(4, -1), 
                                   flips = [(False, False), (True, False), (False, True), (True, True)])
            self.train()
        

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        if what_you_want_for_sample==2:
            model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        elif what_you_want_for_sample==1:
            if t >= 50:
                model_mean, posterior_log_variance = self.q_posterior_run(
                x_start=x_recon, x_t=x, t=t)
            elif t < 50:
                model_mean, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        elif what_you_want_for_sample==3:
            model_mean, posterior_log_variance = self.q_posterior_run(
                x_start=x_recon, x_t=x, t=t)
        else:
            print("i dont know what you want wuwuwu")
        
        
        return model_mean, posterior_log_variance, x_recon#, noise

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None, mask_condition=None, ttt_opt=None,noise=None,what_you_want_for_sample=None):

        model_mean, model_log_variance, x_recon = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, mask_condition=mask_condition, ttt_opt=ttt_opt, what_you_want_for_sample=what_you_want_for_sample)
        
        # noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        # noise = noise if t > 0 else torch.zeros_like(x)

        #we set eta to 0
        noise = torch.zeros_like(x)

        return model_mean + noise * (0.5 * model_log_variance).exp(), noise, model_mean, x_recon

    @torch.no_grad()
    def conpute_brain_value(self, x):
        
        a=x>-0.95
        b=x>-0.93
        all=torch.ones_like(x)

        braina=all[a]
        brainb=all[b]
        
        value_all=all.sum()
        value_a=braina.sum()
        value_b=brainb.sum()
         
        return math.sqrt(float(0.5*(value_all/value_a)+0.5*(value_all/value_b)))

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, ttt_opt=None, matched_state=1000,Tt=None, what_you_want_for_sample=None,CSNR=None):
        
        random_integer = 0
        x_start = x_in['X'].detach()
        [b, c, w, h] = x_start.shape

        t = matched_state

        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)

        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)
        
        # noise : b, c, w, h
        noise = (x_in['X'] - x_in['condition'+str(random_integer)].detach())
        noise_mean = torch.mean(noise, dim=(1,2,3), keepdim=True)
        noise = noise - noise_mean.detach()

        # # noise shuffle
        # # noise : b, c, w, h
        noise = noise.view(b, c, -1)
        rand_idx = torch.randperm(noise.shape[-1])
        noise = noise[:,:,rand_idx].view(b,c,w,h).detach()

        # noise=torch.randn_like(x_start)
        
        x_condition_sample = x_in['condition'+str(random_integer)]

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x_in['condition'+str(random_integer)], t=t)

        x_condition_sample= model_mean
        
        x_noisy = self.q_sample(
            x_start=x_condition_sample, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise.detach())
    
        img = x_noisy
        ret_img = x_noisy

        # print('matched state:', matched_state)
        
        ttt = None
        if TTT:
            # backup model state dict
            denoisor_fn_state = copy.deepcopy(self.denoise_fn.state_dict())

        x_recon = x_in['X']
        
        #bx
        value= self.conpute_brain_value(x_in['X'])
    
        for i in tqdm((reversed(range(len(Tt)))), desc='sampling loop time step', total=len(Tt)):
            
            noise = noise.view(b, c, -1)
            rand_idx = torch.randperm(noise.shape[-1])
            noise = noise[:,:,rand_idx].view(b,c,w,h).detach()
            b, c, w, h = img.shape

            img, noise, img_wo_noise, x_recon = self.p_sample(img, Tt[i], condition_x=x_in['condition'+str(random_integer)], ttt_opt=ttt,noise=noise,what_you_want_for_sample=what_you_want_for_sample)
            ttt = ttt_opt
            
            ret_img = torch.cat([ret_img, img], dim=0)
            
            ret_img = torch.cat([ret_img, x_recon], dim=0)

            #adaptive termination
            loss=((self.mseloss(x_recon, x_in['X']).item()))
            loss= math.sqrt(loss)*(value)

            if i==0:
                print(loss)
                # break
            if loss>CSNR:
                print(loss)
                break

        if TTT:
            # recover model state dict
            self.denoise_fn.load_state_dict(denoisor_fn_state)
        
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, x_in, continous=False):
        matched_state = 999
        return self.p_sample_loop(x_in, continous, matched_state=matched_state)
    
    @torch.no_grad()
    def getrunwalk(self, what_you_want_for_sample=None,total_step=None):
        if what_you_want_for_sample==2:
            TTTTt = list((range(0, total_step+1)))
        elif what_you_want_for_sample==1:
            TTTTt = []
            n = total_step
            for i in range(n + 1):
                if i < 50:
                    TTTTt.append(i)
                elif i >= 50:
                        TTTTt.append(50+(i-50)*10)
                        if (50+(i-50)*10) >= n:
                            break
                    
        elif what_you_want_for_sample==3:
            TTTTt = list((range(0, total_step+10, 10)))
        else:
            print("i dont know what you want wuwuwu")
        return TTTTt
    
    @torch.no_grad()
    def denoise(self, x_in, continous=False, ttt_opt=None):

        matched_state = int(300) # b, 1
        what_you_want_for_sample=1 #runwalk-1,ddpm-2,ddim-3
        CSNR = 0.040 # for more details, plz refer to our paper
        
        Tt= self.getrunwalk(what_you_want_for_sample=what_you_want_for_sample,total_step=matched_state,CSNR=CSNR)

        return self.p_sample_loop(x_in, continous, ttt_opt=ttt_opt, matched_state=matched_state, Tt=Tt, what_you_want_for_sample=what_you_want_for_sample)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    @torch.no_grad()
    def interpolate(self, x, t = None, lams=[0.5]):
        assert x['X'].shape[0] == 2
        # reverse process
        x1 = dict(X=x['X'][[0]], Y=x['X'][[0]], condition=x['condition'][[0]], matched_state=x['matched_state'][[0]])
        x1 = self.denoise(x1).unsqueeze(0)

        x2 = dict(X=x['X'][[1]], Y=x['X'][[1]], condition=x['condition'][[1]], matched_state=x['matched_state'][[1]])
        x2 = self.denoise(x2).unsqueeze(0)

        b, *_, device = *x1.shape, x1.device
        t = self.num_timesteps

        assert x1.shape == x2.shape
        t_batched = torch.stack([torch.tensor(self.sqrt_alphas_cumprod_prev[t], device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t_batched), (x1, x2))

        imgs = []
        for lam in lams:
            img = (1 - lam) * xt1 + lam * xt2
            img = img.float()
            for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
                img, noise, img_wo_noise, x_recon = self.p_sample(img, i, condition_x=img, ttt_opt=None)
            imgs.append(img)
        return x['X'][[0]], x['X'][[1]], x1,x2, imgs
    @torch.no_grad()
    def mask_tensor(self, x, ratio):
        
        mask = (torch.rand(x.shape) > ratio).to(x.device)

        masked_tensor = x * mask

        return masked_tensor, mask
    
    def p_losses(self, x_in, noise=None, debug=False):
        debug_results = dict()

        
        random_integer = 0
        x_start = x_in['X'].detach()
        
        
        [b, c, w, h] = x_start.shape

        #training the latter diffusion steps
        t = np.random.randint(1, 300)

        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)

        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)
        # Di process
        # noise : b, c, w, h
        
        noise = (x_in['X']-x_in['condition'+str(random_integer)])
        noise_mean = torch.mean(noise, dim=(1,2,3), keepdim=True)
        noise = noise - noise_mean.detach()

        # noise mess
        # noise : b, c, w, h
        noise = noise.view(b, c, -1)
        rand_idx = torch.randperm(noise.shape[-1])
        noise = noise[:,:,rand_idx].view(b,c,w,h).detach()

        # noise=torch.randn_like(x_start)
        
        # Fusion process
        x_condition_sample = x_in['condition'+str(random_integer)]
        

        
        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_condition_sample, x_t=x_in['condition'+str(random_integer)], t=t)
        
        x_condition_sample= model_mean

        x_noisy = self.q_sample(
            x_start=x_condition_sample, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise.detach())
        
        x_recon = self.denoisor(x_noisy, continuous_sqrt_alpha_cumprod)
        
        
        # J-Invariance
        total_loss = self.mseloss(x_recon, x_in['X'])

        if debug:
            return_dict = dict(total_loss=total_loss, debug_results=debug_results)
        else:
            return_dict = dict(total_loss=total_loss, x_recon=x_recon, x_start=x_start,x_noisy=x_noisy)
        return return_dict


    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
