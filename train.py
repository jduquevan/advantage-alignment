import gymnasium as gym
import wandb
from omegaconf import OmegaConf, DictConfig
import hydra

OmegaConf.register_new_resolver("eval", eval)


from src.algorithms.advantage_alignment import AdvantageAlignment
from src.envs.eg_vectorized import DiscreteEG, ObligatedRatioDiscreteEG
from src.utils.utils import seed_all, instantiate_agent

# import torch
# torch.autograd.set_detect_anomaly(True)

@hydra.main(version_base="1.3", config_path="configs", config_name="f1.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: None
    """
    seed_all(cfg['seed'])
    wandb.init(
        config=dict(cfg), 
        project="Advantage Alignment",
        dir=cfg["wandb_dir"], 
        reinit=True, 
        anonymous="allow",
        mode=cfg["wandb"],
        tags=cfg.wandb_tags,
    )
    
    is_f1 = cfg['env']['type'] == 'f1'
    if is_f1:
        env = gym.make_vec('merging-v0', num_envs=cfg['env']['batch'])
    elif cfg['env']['type'] == 'eg':
        env = DiscreteEG(item_max_quantity=cfg['env']['item_max_quantity'],
                         batch_size=cfg['batch_size'],
                         device=cfg['env']['device'])
    elif cfg['env']['type'] == 'eg-obligated_ratio':
        env = ObligatedRatioDiscreteEG(max_turns={
                                        'lower': cfg['env']['max_turns_lower'],
                                        'mean': cfg['env']['max_turns_mean'],
                                        'upper': cfg['env']['max_turns_upper']
                                       },
                                       sampling_type=cfg['env']['sampling_type'],
                                       item_max_quantity=cfg['env']['item_max_quantity'],
                                       batch_size=cfg['batch_size'],
                                       device=cfg['env']['device'])
    else:
        raise ValueError(f"Environment type {cfg['env']['type']} not supported.")

    if cfg['self_play']:
        agent_1 = agent_2 = instantiate_agent(cfg)
    else:
        agent_1 = instantiate_agent(cfg)
        agent_2 = instantiate_agent(cfg)
    
    algorithm = AdvantageAlignment(
        env=env, 
        agent_1=agent_1, 
        agent_2=agent_2, 
        train_cfg=cfg.training,
        use_transformer=cfg['use_transformer'],
        sum_rewards=cfg['sum_rewards'],
        simultaneous=cfg['simultaneous'],
        discrete=cfg['discrete'],
        use_reward_loss=cfg['use_reward_loss'],
        use_spr_loss=cfg['use_spr_loss'],
        is_f1=is_f1
    )
    algorithm.train_loop()

if __name__ == "__main__":
    main()