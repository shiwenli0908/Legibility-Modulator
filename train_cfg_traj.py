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
from dataset.dataset import BlockReachPotentialSampleDataset

dataset = BlockReachPotentialSampleDataset(
    data_dir="test_NPZ_traj"
)

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    #num_workers=4,
    shuffle=True
)

# visualize data in batch
batch = next(iter(dataloader))

print(batch['target'].shape)     # (B, 2)
print(batch['score'].shape)      # (B, 1)
print(batch['potential'].shape)  # (B, 1, 96, 96)
print(batch['sample'].shape)     # (B, 10, 2)


# --- instantiate visual encoder for potential map ---
potential_encoder = PotentialResNetEncoder(
    name='resnet18',
    output_dim=512,
    pretrained=False,
    group_norm=True
)

# ResNet18 has output dim of 512
#vision_feature_dim = 512
potential_feature_dim = 512
# agent_pos is 2 dimensional
#lowdim_obs_dim = 2
target_dim = 2
start_dim = 2
# observation feature has 514 dims in total per step
obs_dim = target_dim
sample_dim = 2

#global_cond_dim = (vision_feature_dim + lowdim_obs_dim + target_dim) * obs_horizon + potential_feature_dim + obs_horizon
global_cond_dim = start_dim + target_dim + potential_feature_dim + 1

noise_pred_net = ConditionalUnet1D(
    input_dim=sample_dim,
    global_cond_dim=global_cond_dim
)

# the final arch has 2 parts
nets = nn.ModuleDict({
    'potential_encoder': potential_encoder,
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

num_epochs = 800

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
    epoch_loss = []

    for batch in dataloader:
        # ==== 数据准备 ====
        sample_full = batch['sample'].to(device)       # (B, 10, 2)
        start = sample_full[:, 0]                      # (B, 2)
        #target = batch['target'].to(device)            # (B, 2)
        target = sample_full[:, -1]                     # (B, 2)
        score = batch['score'].to(device)              # (B, 1)
        potential = batch['potential'].to(device)      # (B, 1, 96, 96)
        
        # 中间的 8 个点作为生成目标
        sample = sample_full[:, 1:-1]                  # (B, 8, 2)
        B = sample.shape[0]

        # ==== 编码潜能图 ====
        potential_features = nets['potential_encoder'](potential)  # (B, 512)

        # ==== Classifier-Free Guidance dropout ====
        mask = (torch.rand(B, device=device) < 0.1).float().view(B, 1)  # (B, 1)
        score_masked = score * (1.0 - mask)                             # (B, 1)
        potential_masked = potential_features * (1.0 - mask)           # (B, 512)

        # ==== 构造 global condition ====
        global_cond = torch.cat([
            start,         # (B, 2)
            target,        # (B, 2)
            score_masked,  # (B, 1)
            potential_masked  # (B, 512)
        ], dim=-1)  # → (B, 2 + 1 + 512) = (B, 515)

        # ==== Forward diffusion process ====
        noise = torch.randn_like(sample)  # same shape as sample
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        noisy_sample = noise_scheduler.add_noise(
            original_samples=sample,
            noise=noise,
            timesteps=timesteps
        )

        # ==== Denoising prediction ====
        noise_pred = noise_pred_net(
            noisy_sample, timesteps, global_cond=global_cond
        )

        # ==== Loss & optimization ====
        loss = nn.functional.mse_loss(noise_pred, noise)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        ema.step(nets.parameters())

        epoch_loss.append(loss.item())

    # ==== Logging ====
    avg_epoch_loss = np.mean(epoch_loss)
    all_epoch_losses.append(avg_epoch_loss)
    print(f"Epoch {epoch_idx + 1} avg loss: {avg_epoch_loss:.4f}")

# Weights of the EMA model
# is used for inference
ema_nets = nets
ema.copy_to(ema_nets.parameters())

torch.save(ema_nets.state_dict(), "env20_trajectory_potential_epoch800.ckpt")
print("EMA model saved to blockreach_ema.ckpt")

# 保存目录
save_dir = "training_plots"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "loss_curve_trajectory_potential_epoch800.png")

plt.figure(figsize=(8, 5))
plt.plot(all_epoch_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (log scale)')
plt.title('Training Loss Curve for Traget-Conditioned Diffusion Model')
plt.yscale('log')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(save_path, dpi=300)   # 保存图片
plt.close()                       # 关闭以释放内存

print(f"Loss curve saved to: {save_path}")

