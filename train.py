import hydra
import wandb
from omegaconf import DictConfig

from src.algorithms.advantage_alignment import AdvantageAlignment
from src.envs.eg_vectorized import DiscreteEG
from src.envs.exchange_game import make_vectorized_env
from src.utils.utils import seed_all, instantiate_agent

# import torch
# torch.autograd.set_detect_anomaly(True)

@hydra.main(version_base="1.3", config_path="configs", config_name="eg.yaml")
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
        mode=cfg["wandb"]
    )
    if cfg['simultaneous']:
        env = DiscreteEG(batch_size=cfg['batch_size'], device=cfg['env']['device'])
    else:
        env = make_vectorized_env(cfg['batch_size'])

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
    )
    algorithm.train_loop()


if __name__ == "__main__":
    main()