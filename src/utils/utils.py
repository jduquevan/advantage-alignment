import hydra
import math
import numpy as np
import os
import random
import torch
import wandb

from omegaconf import DictConfig

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_observation(observations, t):
    prop = observations['prop'][:, t, :]
    item_and_utility = observations['item_and_utility'][:, t, :]
    return {
        'prop': prop,
        'item_and_utility': item_and_utility
    }

def cat_observations(observations, device):
    observations_cat = torch.cat([
        torch.tensor(observations['item_and_utility'], device=device),
        torch.tensor(observations['prop'], device=device)
        # torch.tensor(observations['utte'], device=device)
    ], dim=1)
    return observations_cat

def get_categorical_entropy(log_ps):
    probs = torch.exp(log_ps)
    eps = 1e-8  # small value to avoid NaNs
    probs = torch.clamp(probs, eps, 1.0 - eps)  # clip probabilities to avoid zeros
    return -(probs * torch.log(probs)).sum(dim=1).mean()

def get_gaussian_entropy(covariance_matrices):
    """
    Calculate entropy of a multivariate Gaussian distribution given its covariance matrix.
    """
    batch_size = covariance_matrices.size(0)
    dimension = covariance_matrices.size(-1)
    det_covariance = torch.det(covariance_matrices)
    entropy = 0.5 * (dimension * (torch.log(2 * torch.tensor(math.pi)) + 1) + torch.log(det_covariance))
    return entropy.mean()

def get_cosine_similarity(x1, x2, eps=1e-8):
    dot_product = np.sum(x1 * x2, axis=-1)
    norm1 = np.linalg.norm(x1, axis=-1)
    norm2 = np.linalg.norm(x2, axis=-1)
    cosine_sim = dot_product / (norm1 * norm2 + eps)
    return cosine_sim

def get_cosine_similarity_torch(x1, x2, eps=1e-8):
    dot_product = torch.sum(x1 * x2, dim=-1)
    norm1 = torch.norm(x1.float(), dim=-1)
    norm2 = torch.norm(x2.float(), dim=-1)
    cosine_sim = dot_product / (norm1 * norm2 + eps)
    return cosine_sim

def unit_vector_from_angles(theta, phi):
    # Convert angles to radians
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    
    # Calculate the components of the unit vector
    x = np.sin(theta_rad) * np.cos(phi_rad)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    z = np.cos(theta_rad)
    
    # Return the unit vector
    return np.array([x, y, z])

def compute_discounted_returns(gamma, rewards):
        batch_size, trajectory_len = rewards.shape
        returns = torch.empty(
            (batch_size, trajectory_len), 
            device=rewards.device,
            dtype=torch.float32
        )
        returns[:, -1] = rewards[:, -1]
        for t in reversed(range(trajectory_len-1)):
            returns[:, t] = rewards[:, t] + gamma * returns[:, t+1]
        return returns

def compute_gae_advantages(rewards, values, gamma=0.96, tau=0.95):
    # Compute deltas
    with torch.no_grad():
        deltas = rewards[:, :-1] + gamma * values[:, 1:] - values[:, :-1]
        
        advantages = torch.empty_like(deltas)
        advantage = 0
        for t in reversed(range(deltas.size(1))):
            advantages[:, t] = advantage = deltas[:, t] + gamma * tau * advantage
        
        # Normalize advantages
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages

def torchify_actions(action, device):
    action_torch = {}
    action_torch['term'] = torch.tensor(action['term'], device=device)
    action_torch['utte'] = torch.tensor(action['utte'], device=device)
    action_torch['prop'] = torch.tensor(action['prop'], device=device)
    return action_torch

def instantiate_agent(cfg: DictConfig):
    if cfg.use_transformer:
        use_gru = False
        encoder = hydra.utils.instantiate(cfg.transformer_model)
        encoder_t = hydra.utils.instantiate(cfg.transformer_model)
        if not cfg.share_encoder:
            encoder_c = hydra.utils.instantiate(cfg.transformer_model)
    else:
        use_gru = True
        encoder = hydra.utils.instantiate(cfg.gru_model)
        encoder_t = hydra.utils.instantiate(cfg.gru_model)
        if not cfg.share_encoder:
            encoder_c = hydra.utils.instantiate(cfg.gru_model)

    mlp_model = hydra.utils.instantiate(
        cfg.mlp_model, 
        encoder=encoder,
        use_gru=use_gru
    )
    actor = hydra.utils.instantiate(cfg.policy, model=mlp_model)

    if not cfg.share_encoder:
        critic = hydra.utils.instantiate(
            cfg.linear_model, 
            encoder=encoder_c,
            use_gru=use_gru
        )
    else:
        critic = hydra.utils.instantiate(
            cfg.linear_model, 
            encoder=encoder,
            use_gru=use_gru
        )

    # Self-supervised reward learning objective
    reward = hydra.utils.instantiate(
        cfg.linear_model, 
        encoder=encoder,
        use_gru=use_gru
    )

    # Self-predictive-representations objective
    spr = hydra.utils.instantiate(
        cfg.spr_model, 
        encoder=encoder,
        use_gru=use_gru
    )

    target = hydra.utils.instantiate(
        cfg.linear_model, 
        encoder=encoder_t,
        use_gru=use_gru
    )

    target.load_state_dict(critic.state_dict())

    optimizer_actor = hydra.utils.instantiate(cfg.optimizer_actor, params=actor.parameters())
    optimizer_critic = hydra.utils.instantiate(cfg.optimizer_critic, params=critic.parameters())
    optimizer_reward = hydra.utils.instantiate(cfg.optimizer_reward, params=reward.parameters())
    optimizer_spr = hydra.utils.instantiate(cfg.optimizer_spr, params=spr.parameters())

    agent = hydra.utils.instantiate(
        cfg.agent,
        actor=actor,
        critic=critic,
        target=target,
        reward=reward,
        spr=spr,
        critic_optimizer=optimizer_critic,
        actor_optimizer=optimizer_actor,
        reward_optimizer=optimizer_reward,
        spr_optimizer=optimizer_spr
    )
    return agent

def update_target_network(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)

def save_checkpoint(agent, epoch, checkpoint_dir):
    # Get the W&B run ID
    run_id = wandb.run.id

    # Create a folder with the run ID if it doesn't exist
    run_folder = os.path.join(checkpoint_dir, run_id)
    os.makedirs(run_folder, exist_ok=True)

    # Generate a unique filename
    checkpoint_name = f"model_{epoch}.pt"
    checkpoint_path = os.path.join(run_folder, checkpoint_name)
    
    checkpoint = {
        'epoch': epoch,
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict()
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")

def merge_train_step_metrics(train_step_metrics_1, train_step_metrics_2, is_f1):
    merged_metrics = {
        "Actor loss 1": train_step_metrics_1["actor_loss"],
        "Critic loss 1": train_step_metrics_1["critic_loss"],
        "Critic Grad Norm 1": train_step_metrics_1["critic_grad_norm"],
        "Actor Grad Norm 1": train_step_metrics_1["actor_grad_norm"],
        "Actor loss 2": train_step_metrics_2["actor_loss"],
        "Critic loss 2": train_step_metrics_2["critic_loss"],
        "Critic Grad Norm 2": train_step_metrics_2["critic_grad_norm"],
        "Actor Grad Norm 2": train_step_metrics_2["actor_grad_norm"]
    }

    if is_f1:
        merged_metrics.update({
            "Acc Entropy 1": train_step_metrics_1["acc_entropy"],
            "Ste Entropy 1": train_step_metrics_1["ste_entropy"],
            "Acc Entropy 2": train_step_metrics_2["acc_entropy"],
            "Ste Entropy 2": train_step_metrics_2["ste_entropy"]
        })
    else:
        merged_metrics.update({
            "Term Entropy 1": train_step_metrics_1["term_entropy"],
            "Utterance Entropy 1": train_step_metrics_1["utterance_entropy"],
            "Prop Entropy 1": train_step_metrics_1["prop_entropy"],
            "Term Entropy 2": train_step_metrics_2["term_entropy"],
            "Utterance Entropy 2": train_step_metrics_2["utterance_entropy"],
            "Prop Entropy 2": train_step_metrics_2["prop_entropy"]
        })

    return merged_metrics

def wandb_stats(trajectory, is_f1):
    stats = {}
    dones = trajectory.data['dones']
    stats['Average reward 1'] = trajectory.data['rewards_0'].mean().detach()
    stats['Average reward 2'] = trajectory.data['rewards_1'].mean().detach()
    stats['Average sum rewards'] = (
        trajectory.data['rewards_0'] +
        trajectory.data['rewards_1']
    ).mean().detach()
    stats['Average traj len'] = ((dones.size(0) * dones.size(1))/torch.sum(dones)).detach()
    if is_f1:
        stats['Average A1 acc'] = trajectory.data['actions_0'][:, :, 0].mean()
        stats['Average A2 acc'] = trajectory.data['actions_1'][:, :, 0].mean()
        stats['Average A1 ste'] = trajectory.data['actions_0'][:, :, 1].mean()
        stats['Average A2 ste'] = trajectory.data['actions_1'][:, :, 1].mean()
        stats['A1 acc std'] = trajectory.data['actions_0'][:, :, 0].std()
        stats['A2 acc std'] = trajectory.data['actions_1'][:, :, 0].std()
        stats['A1 ste std'] = trajectory.data['actions_0'][:, :, 1].std()
        stats['A2 ste std'] = trajectory.data['actions_1'][:, :, 1].std()
    else:
        stats['Average cos sim'] = trajectory.data['cos_sims'].mean().detach()
        stats['Average A1 prop1'] = trajectory.data['a_props_0'].reshape((-1, 3)).mean(dim=0)[0]
        stats['Average A1 prop2'] = trajectory.data['a_props_0'].reshape((-1, 3)).mean(dim=0)[1]
        stats['Average A1 prop3'] = trajectory.data['a_props_0'].reshape((-1, 3)).mean(dim=0)[2]
        stats['Average A2 prop1'] = trajectory.data['a_props_1'].reshape((-1, 3)).mean(dim=0)[0]
        stats['Average A2 prop2'] = trajectory.data['a_props_1'].reshape((-1, 3)).mean(dim=0)[1]
        stats['Average A2 prop3'] = trajectory.data['a_props_1'].reshape((-1, 3)).mean(dim=0)[2]
        stats['A1 prop1 std'] = trajectory.data['a_props_0'].reshape((-1, 3)).std(dim=0)[0]
        stats['A1 prop2 std'] = trajectory.data['a_props_0'].reshape((-1, 3)).std(dim=0)[1]
        stats['A1 prop3 std'] = trajectory.data['a_props_0'].reshape((-1, 3)).std(dim=0)[2]
        stats['A2 prop1 std'] = trajectory.data['a_props_1'].reshape((-1, 3)).std(dim=0)[0]
        stats['A2 prop2 std'] = trajectory.data['a_props_1'].reshape((-1, 3)).std(dim=0)[1]
        stats['A2 prop3 std'] = trajectory.data['a_props_1'].reshape((-1, 3)).std(dim=0)[2]
        stats['Avg max sum rewards'] = trajectory.data['max_sum_rewards'].mean()
        stats['Avg max sum rew ratio'] = trajectory.data['max_sum_reward_ratio'][
            trajectory.data['dones']
        ].mean()
       
    return stats