import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.encoder import get_resnet, replace_bn_with_gn
from diffusion_policy.encoder import PotentialResNetEncoder
from diffusion_policy.model import ConditionalUnet1D
from dataset.dataset import TrajInverseModelDataset

pred_horizon = 8
obs_horizon = 2
action_horizon = 4

dataset = TrajInverseModelDataset(
    data_dir="test11_3D", 
    pred_horizon=8,
    obs_horizon=2,
    action_horizon=4
)

# save training data statistics (min, max) for each dim
stats = dataset.stats

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)

# visualize data in batch
batch = next(iter(dataloader))

print(batch['obs'].shape)         # torch.Size([32, 2, 2])
print(batch['action'].shape)      # torch.Size([32, 4, 2])
print(batch['target'].shape)      # torch.Size([32, 2, 2])
print(batch['sample_cond'].shape)  # torch.Size([32, 8, 2])

# agent_pos is 2 dimensional
lowdim_obs_dim = 3
target_dim = 3

obs_dim = lowdim_obs_dim + target_dim
action_dim = 3

#global_cond_dim = (vision_feature_dim + lowdim_obs_dim + target_dim) * obs_horizon + potential_feature_dim + obs_horizon
global_cond_dim = (lowdim_obs_dim + target_dim) * obs_horizon + 8 * 3

noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=global_cond_dim
)

# the final arch has 2 parts
nets = nn.ModuleDict({
    'noise_pred_net': noise_pred_net
})

# for this demo, we use DDPMScheduler with 100 diffusion iterations
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# device transfer
device = torch.device('cuda')
_ = nets.to(device)

num_epochs = 2000

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=nets.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

all_epoch_losses = []


for epoch_idx in range(num_epochs):
    epoch_loss = list()
    # batch loop

    for nbatch in dataloader:
        # data normalized in dataset
        # device transfer
        nagent_pos = nbatch['obs'].to(device)           # (B, obs_horizon, 2)
        target = nbatch['target'].to(device)            # (B, obs_horizon, 2)
        sample_cond = nbatch['sample_cond'].to(device)  # (B, 8, 2)
        naction = nbatch['action'].to(device)
        B = nagent_pos.shape[0]
        
        # === 构建 condition ===
        obs_features = torch.cat([nagent_pos, target], dim=-1)   # (B, obs_horizon, 4)
        obs_cond = obs_features.flatten(start_dim=1)             # (B, obs_horizon * 4)
        
        sample_cond_flat = sample_cond.flatten(start_dim=1)      # (B, 8 * 2)

        # === Classifier-Free Guidance: 20% 概率 mask 掉 sample ===
        '''
        if torch.rand(1).item() < 0.2:
            sample_cond_flat = torch.zeros_like(sample_cond_flat)  # mask掉sample
        '''

        mask_prob = 0.25
        mask_flags = torch.rand(B, device=sample_cond_flat.device) < mask_prob   # shape: (B,)
        sample_cond_flat[mask_flags] = 0.0  # 对选中的样本置零

        global_cond = torch.cat([obs_cond, sample_cond_flat], dim=-1)  # final shape: (B, obs_h*4 + 16)


        # sample noise to add to actions
        noise = torch.randn(naction.shape, device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = noise_scheduler.add_noise(
            naction, noise, timesteps)

        # predict the noise residual
        noise_pred = noise_pred_net(
            noisy_actions, timesteps, global_cond=global_cond)

        # L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        # optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # step lr scheduler every batch
        # this is different from standard pytorch behavior
        lr_scheduler.step()

        # update Exponential Moving Average of the model weights
        ema.step(nets.parameters())

        # logging
        loss_cpu = loss.item()
        epoch_loss.append(loss_cpu)
    print(f"Epoch {epoch_idx + 1} avg loss: {np.mean(epoch_loss):.4f}")

    all_epoch_losses.append(np.mean(epoch_loss))

# Weights of the EMA model
# is used for inference
ema_nets = nets
ema.copy_to(ema_nets.parameters())

torch.save(ema_nets.state_dict(), "env01_traj_inverse_proj_epoch2000.ckpt")
print("EMA model saved to blockreach_ema.ckpt")

# 保存目录
save_dir = "training_plots"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "loss_curve_env01_traj_inverse_proj_epoch2000.png")

plt.figure(figsize=(8, 5))
plt.plot(all_epoch_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (log scale)')
plt.title('Training Loss Curve for Diffusion Policy')
plt.yscale('log')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(save_path, dpi=300)   # 保存图片
plt.close()                       # 关闭以释放内存

print(f"Loss curve saved to: {save_path}")
