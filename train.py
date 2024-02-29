import hydra
import wandb
from omegaconf import DictConfig

from src.algorithms.advantage_alignment import AdvantageAlignment
from src.envs.eg_vectorized import DiscreteEG
from src.envs.exchange_game import make_vectorized_env
from src.utils.utils import seed_all, instantiate_agent


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
    agent = instantiate_agent(cfg)
    algorithm = AdvantageAlignment(
        env=env, 
        agent_1=agent, 
        agent_2=agent, 
        train_cfg=cfg.training,
        use_transformer=cfg['use_transformer'],
        sum_rewards=cfg['sum_rewards'],
        simultaneous=cfg['simultaneous'],
        discrete=cfg['discrete'],
    )
    algorithm.train_loop()


if __name__ == "__main__":
    main()