import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from dataset.utils import get_data_stats, normalize_data, unnormalize_data
from dataset.utils import create_sample_indices, sample_sequence

class BlockReachPotentialDataset2(Dataset):
    def __init__(self, data_dir,
                 pred_horizon, obs_horizon, action_horizon):
        """
        Args:
            data_dir (str): 根目录，每个子目录是 sample_xxxxx
            pred_horizon (int): 用于确定截取步长的窗口（一般等于 action_horizon）
            obs_horizon (int): 观测帧数量
            action_horizon (int): 动作帧数量（用于建模）
        """
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # === 收集数据路径 ===
        sample_dirs = sorted([
            os.path.join(data_dir, d)
            for d in os.listdir(data_dir)
            if d.startswith("sample_")
        ])

        # === 加载所有数据到 list ===
        self.trajectories = []
        for d in sample_dirs:
            obs = np.load(os.path.join(d, "obs.npz"), allow_pickle=True)["data"]
            act = np.load(os.path.join(d, "action.npz"), allow_pickle=True)["data"]
            frames = np.load(os.path.join(d, "frame.npz"), allow_pickle=True)["data"]
            target = np.load(os.path.join(d, "label_onehot.npz"), allow_pickle=True)["data"]
            scoref = np.load(os.path.join(d, "score_force.npz"), allow_pickle=True)["data"]
            #potential = np.load(os.path.join(d, "potential.npz"), allow_pickle=True)["data"]  # shape: (2, 1, H, W)

            for i in range(len(obs)):
                act_i = act[i]
                obs_i = obs[i]
                
                # === 如果 actions 比 obs 少一个，补上最后一个动作 ===
                if len(act_i) == len(obs_i) - 1:
                    last_action = act_i[-1:,...]  # shape (1, 2)
                    act_i = np.concatenate([act_i, last_action], axis=0)  # 变成 (T, 2)

                self.trajectories.append({
                    "obs": obs[i],                # (T, D_obs)
                    "action": act[i],             # (T, 2)
                    "image": frames[i],           # (T, 3, 96, 96)
                    "target": target[i],          # (T, 2)
                    "scoref": scoref[i]          # (T,)
                    #"potential": potential        # (2, 1, H, W)
                })

        # === 统计与归一化 ===
        flat_obs = np.concatenate([traj["obs"] for traj in self.trajectories], axis=0)
        flat_action = np.concatenate([traj["action"] for traj in self.trajectories], axis=0)
        self.stats = {
            "obs": get_data_stats(flat_obs),
            "action": get_data_stats(flat_action)
        }

        for traj in self.trajectories:
            traj["obs"] = normalize_data(traj["obs"], self.stats["obs"])
            traj["action"] = normalize_data(traj["action"], self.stats["action"])

        # === 构建 sample 索引 ===
        self.indices = []
        for traj_idx, traj in enumerate(self.trajectories):
            T = len(traj["obs"])
            max_start = T - pred_horizon
            for start_t in range(max_start + 1):
                self.indices.append((traj_idx, start_t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start_t = self.indices[idx]
        traj = self.trajectories[traj_idx]

        # === 截取时间段 ===
        obs = traj["obs"][start_t:start_t + self.obs_horizon]                 # (obs_h, D)
        action_raw = traj["action"][start_t:start_t + self.action_horizon]    # (action_h, 2)
        image = traj["image"][start_t:start_t + self.obs_horizon]             # (obs_h, 3, 96, 96)
        target = traj["target"][start_t:start_t + self.obs_horizon]                   # (obs_h, 2)
        #scoref = traj["scoref"][start_t:start_t + self.obs_horizon][..., None]        # (obs_h, 1)

        scoref_val = traj["scoref"][start_t:start_t + self.obs_horizon][..., None]  # (obs_h, 1)
        scoref_mask = np.ones_like(scoref_val)  # (obs_h, 1)
        scoref_full = np.concatenate([scoref_val, scoref_mask], axis=-1)  # (obs_h, 2)

        # === 动作 padding 到 obs 维度 ===
        D_obs = obs.shape[1]
        action = np.zeros((self.action_horizon, D_obs), dtype=np.float32)
        action[:, :2] = action_raw

        '''
        # === 获取对应目标的势场图 ===
        target_id = int(target[0].argmax())  # 0 or 1
        potential = traj["potential"][target_id]  # shape: (1, H, W)
        potential = torch.from_numpy(potential).unsqueeze(0).float()  # (1, 1, H, W)
        potential = F.interpolate(potential, size=(96, 96), mode='bilinear', align_corners=False).squeeze(0)  # -> (1, 96, 96)
        '''

        return {
            'obs': torch.from_numpy(obs).float(),              # (obs_h, D)
            'action': torch.from_numpy(action).float(),        # (action_h, D)
            'image': torch.from_numpy(image).float(),          # (obs_h, 3, 96, 96)
            'target': torch.from_numpy(target).float(),        # (obs_h, 2)
            'scoref': torch.from_numpy(scoref_full).float()        # (obs_h, 2)
            #'potential': potential   # (1, H, W)
        }


class BlockReachPotentialStateDataset(Dataset):
    def __init__(self, data_dir,
                 pred_horizon, obs_horizon, action_horizon):
        """
        Args:
            data_dir (str): 根目录，每个子目录是 sample_xxxxx
            pred_horizon (int): 用于确定截取步长的窗口（一般等于 action_horizon）
            obs_horizon (int): 观测帧数量
            action_horizon (int): 动作帧数量（用于建模）
        """
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # === 加载所有数据到 list ===
        self.trajectories = []
        
        # === 加载所有数据 ===
        obs = np.load(os.path.join(data_dir, "obs.npz"), allow_pickle=True)["data"]
        act = np.load(os.path.join(data_dir, "action.npz"), allow_pickle=True)["data"]
        target = np.load(os.path.join(data_dir, "target.npz"), allow_pickle=True)["data"]
        scoref = np.load(os.path.join(data_dir, "score_force.npz"), allow_pickle=True)["data"]
        #potential = np.load(os.path.join(data_dir, "potential.npz"), allow_pickle=True)["data"]  # shape: (40, 1, H, W)


        for i in range(len(obs)):

            sample_idx = i // 200
            #potential_sample = potential[sample_idx * 2 : (sample_idx + 1) * 2]  # shape: (2, 1, H, W)

            self.trajectories.append({
                "obs": obs[i],                # (T, D_obs)
                "action": act[i],             # (T, 2)
                "target": target[i],          # (T, 6)
                "scoref": scoref[i]          # (T,)
                #"potential": potential_sample        # (2, 1, H, W)
            })

        # === 统计与归一化 ===
        flat_obs = np.concatenate([traj["obs"] for traj in self.trajectories], axis=0)
        flat_action = np.concatenate([traj["action"] for traj in self.trajectories], axis=0)
        self.stats = {
            "obs": get_data_stats(flat_obs),
            "action": get_data_stats(flat_action)
        }

        for traj in self.trajectories:
            traj["obs"] = normalize_data(traj["obs"], self.stats["obs"])
            traj["action"] = normalize_data(traj["action"], self.stats["action"])

        # === 构建 sample 索引 ===
        self.indices = []
        for traj_idx, traj in enumerate(self.trajectories):
            T = len(traj["obs"])
            max_start = T - pred_horizon
            for start_t in range(max_start + 1):
                self.indices.append((traj_idx, start_t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start_t = self.indices[idx]
        traj = self.trajectories[traj_idx]

        # === 截取时间段 ===
        obs = traj["obs"][start_t:start_t + self.obs_horizon]                 # (obs_h, D)
        action_raw = traj["action"][start_t:start_t + self.action_horizon]    # (action_h, 2)
        target = traj["target"][start_t:start_t + self.obs_horizon]                   # (obs_h, 2)
        scoref = traj["scoref"][start_t:start_t + self.obs_horizon][..., None]        # (obs_h, 1)

        #scoref_val = traj["scoref"][start_t:start_t + self.obs_horizon][..., None]  # (obs_h, 1)
        #scoref_mask = np.ones_like(scoref_val)  # (obs_h, 1)
        #scoref_full = np.concatenate([scoref_val, scoref_mask], axis=-1)  # (obs_h, 2)

        # === 动作 padding 到 obs 维度 ===
        D_obs = obs.shape[1]
        action = np.zeros((self.action_horizon, D_obs), dtype=np.float32)
        action[:, :2] = action_raw

        '''
        # === 获取对应目标的势场图 ===
        onehot = target[0][-2:]  # 从第0帧提取 onehot 编码
        target_id = int(np.argmax(onehot))  # 0, 1, or 2
        potential = traj["potential"][target_id]  # shape: (1, H, W)
        potential = torch.from_numpy(potential).unsqueeze(0).float()  # (1, 1, H, W)
        potential = F.interpolate(potential, size=(96, 96), mode='bilinear', align_corners=False).squeeze(0)  # -> (1, 96, 96)
        '''

        return {
            'obs': torch.from_numpy(obs).float(),              # (obs_h, 2)
            'action': torch.from_numpy(action).float(),        # (action_h, 2)
            'target': torch.from_numpy(target).float(),        # (obs_h, 6)
            'scoref': torch.from_numpy(scoref).float()        # (obs_h, 1)
            #'potential': potential   # (1, H, W)
        }

class BlockReachPotentialStateDataset2(Dataset):
    def __init__(self, data_dir,
                 pred_horizon, obs_horizon, action_horizon):
        """
        Args:
            data_dir (str): 根目录，每个子目录是 sample_xxxxx
            pred_horizon (int): 用于确定截取步长的窗口（一般等于 action_horizon）
            obs_horizon (int): 观测帧数量
            action_horizon (int): 动作帧数量（用于建模）
        """
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # === 加载所有数据到 list ===
        self.trajectories = []
        
        # === 加载所有数据 ===
        obs = np.load(os.path.join(data_dir, "obs.npz"), allow_pickle=True)["data"]
        act = np.load(os.path.join(data_dir, "action.npz"), allow_pickle=True)["data"]
        target = np.load(os.path.join(data_dir, "target.npz"), allow_pickle=True)["data"]
        scoref = np.load(os.path.join(data_dir, "score_force.npz"), allow_pickle=True)["data"]
        potential = np.load(os.path.join(data_dir, "potential.npz"), allow_pickle=True)["data"]  # shape: (40, 1, H, W)


        for i in range(len(obs)):

            sample_idx = i // 200
            potential_sample = potential[sample_idx * 2 : (sample_idx + 1) * 2]  # shape: (2, 1, H, W)

            self.trajectories.append({
                "obs": obs[i],                # (T, D_obs)
                "action": act[i],             # (T, 2)
                "target": target[i],          # (T, 6)
                "scoref": scoref[i],          # (T,)
                "potential": potential_sample        # (2, 1, H, W)
            })

        # === 统计与归一化 ===
        flat_obs = np.concatenate([traj["obs"] for traj in self.trajectories], axis=0)
        flat_action = np.concatenate([traj["action"] for traj in self.trajectories], axis=0)
        self.stats = {
            "obs": get_data_stats(flat_obs),
            "action": get_data_stats(flat_action)
        }

        for traj in self.trajectories:
            traj["obs"] = normalize_data(traj["obs"], self.stats["obs"])
            traj["action"] = normalize_data(traj["action"], self.stats["action"])

        # === 构建 sample 索引 ===
        self.indices = []
        for traj_idx, traj in enumerate(self.trajectories):
            T = len(traj["obs"])
            max_start = T - pred_horizon
            for start_t in range(max_start + 1):
                self.indices.append((traj_idx, start_t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start_t = self.indices[idx]
        traj = self.trajectories[traj_idx]

        # === 截取时间段 ===
        obs = traj["obs"][start_t:start_t + self.obs_horizon]                 # (obs_h, D)
        action_raw = traj["action"][start_t:start_t + self.action_horizon]    # (action_h, 2)
        target = traj["target"][start_t:start_t + self.obs_horizon]                   # (obs_h, 2)
        scoref = traj["scoref"][start_t:start_t + self.obs_horizon][..., None]        # (obs_h, 1)

        '''
        scoref_val = traj["scoref"][start_t:start_t + self.obs_horizon][..., None]  # (obs_h, 1)
        scoref_mask = np.ones_like(scoref_val)  # (obs_h, 1)
        scoref_full = np.concatenate([scoref_val, scoref_mask], axis=-1)  # (obs_h, 2)
        '''

        # === 动作 padding 到 obs 维度 ===
        D_obs = obs.shape[1]
        action = np.zeros((self.action_horizon, D_obs), dtype=np.float32)
        action[:, :2] = action_raw

        # === 获取对应目标的势场图 ===
        onehot = target[0][-2:]  # 从第0帧提取 onehot 编码
        target_id = int(np.argmax(onehot))  # 0, 1
        potential = traj["potential"][target_id]  # shape: (1, H, W)
        potential = torch.from_numpy(potential).unsqueeze(0).float()  # (1, 1, H, W)
        potential = F.interpolate(potential, size=(96, 96), mode='bilinear', align_corners=False).squeeze(0)  # -> (1, 96, 96)

        return {
            'obs': torch.from_numpy(obs).float(),              # (obs_h, D)
            'action': torch.from_numpy(action).float(),        # (action_h, D)
            'target': torch.from_numpy(target).float(),        # (obs_h, 6)
            'scoref': torch.from_numpy(scoref).float(),        # (obs_h, 2)
            'potential': potential   # (1, H, W)
        }

class BlockReachPotentialSampleDataset(Dataset):
    def __init__(self, data_dir):
        """
        Dataset for sampled trajectories with per-trajectory score and target-specific potential map.

        Args:
            data_dir (str): path to directory with npz files (sample.npz, score_force.npz, target.npz, potential.npz)
        """
        self.trajectories = []

        # === Load all data ===
        samples = np.load(os.path.join(data_dir, "sample.npz"), allow_pickle=True)["data"]
        scores = np.load(os.path.join(data_dir, "score_force.npz"), allow_pickle=True)["data"]
        targets = np.load(os.path.join(data_dir, "target.npz"), allow_pickle=True)["data"]
        potentials = np.load(os.path.join(data_dir, "potential.npz"), allow_pickle=True)["data"]  # shape: (N, 1, H, W)

        for i in range(len(samples)):
            # === 目标索引 ===
            onehot = targets[i][0][-2:]  # 从第0帧提取 one-hot 编码
            target_id = int(np.argmax(onehot))  # 0 or 1

            # === 提取目标坐标 ===
            target_coord = targets[i][0][2 * target_id : 2 * target_id + 2]  # (x, y)

            # === 获取对应的 potential map ===
            # 每个场景有两个目标 → 每200个样本一组
            scene_idx = i // 400
            potential_sample = potentials[scene_idx * 2 : (scene_idx + 1) * 2]
            potential_map = potential_sample[target_id]  # shape: (1, H, W)

            self.trajectories.append({
                "sample": samples[i],                    # (10, 2)
                "score": float(scores[i][0]),            # scalar
                "target": target_coord,                  # (2,)
                "potential": potential_map               # (1, H, W)
            })

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]

        #sample_traj = torch.from_numpy(traj["sample"]).float()         # (10, 2)
        target = torch.from_numpy(traj["target"]).float()              # (2,)
        score = torch.tensor([traj["score"]]).float()                  # (1,)

        sample_np = np.asarray(traj["sample"], dtype=np.float32)
        sample_traj = torch.from_numpy(sample_np)
        
        pot = torch.from_numpy(traj["potential"]).unsqueeze(0).float()  # (1, 1, H, W)
        pot = F.interpolate(pot, size=(96, 96), mode='bilinear', align_corners=False).squeeze(0)  # → (1, 96, 96)

        return {
            "sample": sample_traj,     # (10, 2)
            "target": target,          # (2,)
            "score": score,            # (1,)
            "potential": pot           # (1, 96, 96)
        }


class BlockReachPotentialSampleDataset3D(Dataset):
    def __init__(self, data_dir):
        """
        Dataset for 3D sampled trajectories with per-trajectory score and shared 3-class 3D potential maps.

        Assumes:
        - 900 samples total
        - 3 potential maps (each used for 300 samples)
        """
        self.sampled = np.load(os.path.join(data_dir, "sampled_trajectories.npy"), allow_pickle=True)
        self.scores = np.load(os.path.join(data_dir, "score_force.npy"), allow_pickle=True)
        self.goals = np.load(os.path.join(data_dir, "full_goal.npy"), allow_pickle=True)
        self.potentials = np.load(os.path.join(data_dir, "potential.npy"), allow_pickle=True)  # shape: (3, 1, 96, 96, 96)

        self.total = len(self.sampled)
        assert self.total == 900, "Currently supports only 900 samples (300 per potential)"
        assert self.potentials.shape[0] == 3, "Expected 3 target-specific potentials."

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        traj = torch.from_numpy(np.asarray(self.sampled[idx], dtype=np.float32))   # (10, 3)
        goal = torch.from_numpy(np.asarray(self.goals[idx], dtype=np.float32))     # (3,)
        score = torch.tensor([self.scores[idx]], dtype=torch.float32)              # (1,)

        # pick potential by group index
        pot_idx = idx // 300  # 0, 1, 2
        pot = torch.from_numpy(self.potentials[pot_idx]).float()  # (1, 96, 96, 96)

        return {
            "sample": traj,         # (10, 3)
            "target": goal,         # (3,)
            "score": score,         # (1,)
            "potential": pot        # (1, 96, 96, 96)
        }

class BlockReachPotentialSampleDatasetProjection(Dataset):
    def __init__(self, data_dir):
        """
        Dataset for 3D sampled trajectories with per-trajectory score and shared 3-class 3D potential maps.

        Assumes:
        - 900 samples total
        - 3 potential maps (each used for 300 samples)
        """
        self.sampled = np.load(os.path.join(data_dir, "sampled_trajectories.npy"), allow_pickle=True)
        self.scores = np.load(os.path.join(data_dir, "score_force.npy"), allow_pickle=True)
        self.goals = np.load(os.path.join(data_dir, "full_goal.npy"), allow_pickle=True)
        self.potentials = np.load(os.path.join(data_dir, "potential_combined_proj.npy"), allow_pickle=True)  # shape: (3, 3, 96, 96)

        self.total = len(self.sampled)
        assert self.total == 600, "Currently supports only 900 samples (300 per potential)"
        assert self.potentials.shape[0] == 2, "Expected 3 target-specific potentials."

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        traj = torch.from_numpy(np.asarray(self.sampled[idx], dtype=np.float32))   # (10, 3)
        goal = torch.from_numpy(np.asarray(self.goals[idx], dtype=np.float32))     # (3,)
        score = torch.tensor([self.scores[idx]], dtype=torch.float32)              # (1,)

        # pick potential by group index
        pot_idx = idx // 300  # 0, 1, 2
        pot = torch.from_numpy(self.potentials[pot_idx]).float()  # (3, 96, 96)

        return {
            "sample": traj,         # (10, 3)
            "target": goal,         # (3,)
            "score": score,         # (1,)
            "potential": pot        # (3, 96, 96)
        }
        

class BlockReachPotentialDataset(Dataset):
    def __init__(self, action_file, obs_file, frame_file,
                 pred_horizon, obs_horizon, action_horizon):
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # === 加载数据 ===
        self.actions_all = np.load(action_file, allow_pickle=True)["data"]  # list of (Ti, 2)
        self.obs_all = np.load(obs_file, allow_pickle=True)["data"]         # list of (Ti, 5)
        self.frames_all = np.load(frame_file, allow_pickle=True)["data"]    # list of (Ti, 3, 96, 96)

        assert len(self.actions_all) == len(self.obs_all) == len(self.frames_all), "轨迹数量不一致"

        # === 构建采样索引（每个轨迹的起始时间） ===
        self.sample_tuples = []  # 每个元素是 (traj_idx, start_t)
        for traj_idx in range(len(self.actions_all)):
            T = len(self.actions_all[traj_idx])
            max_start = T - pred_horizon
            for start_t in range(max_start + 1):
                self.sample_tuples.append((traj_idx, start_t))

        # === 计算归一化统计 ===
        all_obs = np.concatenate(self.obs_all, axis=0)  # (sum_T, 5)
        all_actions = np.concatenate(self.actions_all, axis=0)  # (sum_T, 2)
        self.stats = {
            "obs": get_data_stats(all_obs),
            "action": get_data_stats(all_actions)
        }

        # === 提前归一化整个数据集 ===
        self.normalized_data = {
            "obs": [normalize_data(obs, self.stats["obs"]) for obs in self.obs_all],
            "action": [normalize_data(act, self.stats["action"]) for act in self.actions_all],
            "image": self.frames_all  # 图像已是 float32 [0,1]
        }

    def __len__(self):
        return len(self.sample_tuples)

    def __getitem__(self, idx):
        traj_idx, start_t = self.sample_tuples[idx]

        # 抽取该段序列
        obs_seq = self.normalized_data["obs"][traj_idx][start_t:start_t + self.obs_horizon]         # (obs_h, 5)
        act_seq = self.normalized_data["action"][traj_idx][start_t:start_t + self.pred_horizon]     # (pred_h, 2)
        img_seq = self.normalized_data["image"][traj_idx][start_t:start_t + self.obs_horizon]       # (obs_h, 3, 96, 96)

        return {
            "obs": torch.from_numpy(obs_seq).float(),        # (obs_h, 5)
            "action": torch.from_numpy(act_seq).float(),     # (pred_h, 2)
            "image": torch.from_numpy(img_seq).float()       # (obs_h, 3, 96, 96)
        }
    

# ---- Dataset 类 ----

class BlockReachScoreDataset(Dataset):
    def __init__(self, action_file, obs_file, pred_horizon, obs_horizon, action_horizon):
        # 加载数据
        actions = np.load(action_file)['actions']      # shape: (400, 29, 2)
        obs = np.load(obs_file)['obs_array']           # shape: (400, 30, 3, 2)

        # 扩展动作：每条轨迹补最后一步
        pad = actions[:, -1:, :]  # shape: (400, 1, 2)
        actions = np.concatenate([actions, pad], axis=1)  # shape: (400, 30, 2)

        # reshape obs: (400, 30, 3, 2) → (400, 30, 6)
        #obs = obs.reshape(obs.shape[0], obs.shape[1], -1)

        # 合并所有轨迹
        self.obs = obs.reshape(-1, obs.shape[-1])           # (N, obs_dim)
        self.action = actions.reshape(-1, actions.shape[-1])  # (N, action_dim)
        episode_length = obs.shape[1]
        self.episode_ends = np.arange(episode_length, (obs.shape[0]+1)*episode_length, episode_length)

        # 构造 sample 索引
        self.indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1
        )

        # normalize
        self.stats = {}
        self.normalized_data = {}
        self.stats['obs'] = get_data_stats(self.obs)
        self.stats['action'] = get_data_stats(self.action)
        self.normalized_data['obs'] = normalize_data(self.obs, self.stats['obs'])
        self.normalized_data['action'] = normalize_data(self.action, self.stats['action'])

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        sample = sample_sequence(
            train_data=self.normalized_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )
        sample['obs'] = sample['obs'][:self.obs_horizon]
        return {
            'obs': torch.from_numpy(sample['obs']).float(),
            'action': torch.from_numpy(sample['action']).float()
        }

# ------ Dataset for Legibility Diffuser ------

class LegibilityDiffuserDataset(Dataset):
    def __init__(self, data_dir,
                 pred_horizon, obs_horizon, action_horizon):
        """
        Args:
            data_dir (str): 根目录，每个子目录是 sample_xxxxx
            pred_horizon (int): 用于确定截取步长的窗口（一般等于 action_horizon）
            obs_horizon (int): 观测帧数量
            action_horizon (int): 动作帧数量（用于建模）
        """
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # === 加载所有数据到 list ===
        self.trajectories = []
        
        # === 加载所有数据 ===
        obs = np.load(os.path.join(data_dir, "obs.npz"), allow_pickle=True)["data"]
        act = np.load(os.path.join(data_dir, "action.npz"), allow_pickle=True)["data"]
        target = np.load(os.path.join(data_dir, "goal.npz"), allow_pickle=True)["data"]

        for i in range(len(obs)):
            self.trajectories.append({
                "obs": obs[i],                # (T, 2)
                "action": act[i],             # (T, 2)
                "target": target[i]           # (T, 2)
            })

        # === 统计与归一化 ===
        flat_obs = np.concatenate([traj["obs"] for traj in self.trajectories], axis=0)
        flat_action = np.concatenate([traj["action"] for traj in self.trajectories], axis=0)
        flat_target = np.concatenate([traj["target"] for traj in self.trajectories], axis=0)
        self.stats = {
            "obs": get_data_stats(flat_obs),
            "action": get_data_stats(flat_action),
            "target": get_data_stats(flat_target)
        }

        for traj in self.trajectories:
            traj["obs"] = normalize_data(traj["obs"], self.stats["obs"])
            traj["action"] = normalize_data(traj["action"], self.stats["action"])
            traj["target"] = normalize_data(traj["target"], self.stats["target"])

        # === 构建 sample 索引 ===
        self.indices = []
        for traj_idx, traj in enumerate(self.trajectories):
            T = len(traj["obs"])
            max_start = T - pred_horizon
            for start_t in range(max_start + 1):
                self.indices.append((traj_idx, start_t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start_t = self.indices[idx]
        traj = self.trajectories[traj_idx]

        # === 截取时间段 ===
        obs = traj["obs"][start_t:start_t + self.obs_horizon]                 # (obs_h, 2)
        action = traj["action"][start_t:start_t + self.action_horizon]    # (action_h, 2)
        target = traj["target"][start_t:start_t + self.obs_horizon]                   # (obs_h, 2)

        return {
            'obs': torch.from_numpy(obs).float(),              # (obs_h, 2)
            'action': torch.from_numpy(action).float(),        # (action_h, 2)
            'target': torch.from_numpy(target).float()        # (obs_h, 2)
        }

# ------ Dataset for Inverse Model ------

class TrajInverseModelDataset(Dataset):
    def __init__(self, data_dir,
                 pred_horizon, obs_horizon, action_horizon):
        """
        Args:
            data_dir (str): 目录
            pred_horizon (int): 用于确定截取步长的窗口（一般等于 action_horizon）
            obs_horizon (int): 观测帧数量
            action_horizon (int): 动作帧数量（用于建模）
        """
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # === 加载所有数据到 list ===
        self.trajectories = []
        
        # === 加载所有数据 ===
        obs = np.load(os.path.join(data_dir, "obs.npz"), allow_pickle=True)["data"]
        act = np.load(os.path.join(data_dir, "action.npz"), allow_pickle=True)["data"]
        target = np.load(os.path.join(data_dir, "goal.npz"), allow_pickle=True)["data"]
        sample_cond = np.load(os.path.join(data_dir, "sample_cond.npz"), allow_pickle=True)["data"]

        for i in range(len(obs)):
            self.trajectories.append({
                "obs": obs[i],                # (T, D_obs)
                "action": act[i],             # (T, 2)
                "target": target[i],          # (T, 2)
                "sample_cond": sample_cond[i],     # (8, 2)
            })

        # === 统计与归一化 ===
        flat_obs = np.concatenate([traj["obs"] for traj in self.trajectories], axis=0)
        flat_action = np.concatenate([traj["action"] for traj in self.trajectories], axis=0)
        flat_target = np.concatenate([traj["target"] for traj in self.trajectories], axis=0)
        flat_sample_cond = np.concatenate([traj["sample_cond"] for traj in self.trajectories], axis=0)

        self.stats = {
            "obs": get_data_stats(flat_obs),
            "action": get_data_stats(flat_action),
            "target": get_data_stats(flat_target),
            "sample_cond": get_data_stats(flat_sample_cond),
        }

        for traj in self.trajectories:
            traj["obs"] = normalize_data(traj["obs"], self.stats["obs"])
            traj["action"] = normalize_data(traj["action"], self.stats["action"])
            traj["target"] = normalize_data(traj["target"], self.stats["target"])
            traj["sample_cond"] = normalize_data(traj["sample_cond"], self.stats["sample_cond"])

        # === 构建 sample 索引 ===
        self.indices = []
        for traj_idx, traj in enumerate(self.trajectories):
            T = len(traj["obs"])
            max_start = T - pred_horizon
            for start_t in range(max_start + 1):
                self.indices.append((traj_idx, start_t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start_t = self.indices[idx]
        traj = self.trajectories[traj_idx]

        # === 截取时间段 ===
        obs = traj["obs"][start_t:start_t + self.obs_horizon]                 # (obs_h, 2)
        action = traj["action"][start_t:start_t + self.action_horizon]    # (action_h, 2)
        target = traj["target"][start_t:start_t + self.obs_horizon]                   # (obs_h, 2)
        #sample_cond = traj["sample_cond"]  # (8, 2)
        sample_cond = np.array(traj["sample_cond"], dtype=np.float32)   # (8, 2)

        return {
            'obs': torch.from_numpy(obs).float(),              # (obs_h, 2)
            'action': torch.from_numpy(action).float(),        # (action_h, 2)
            'target': torch.from_numpy(target).float(),        # (obs_h, 2)
            'sample_cond': torch.from_numpy(sample_cond).float()   # (8, 2)
        }