import hydra
from omegaconf import DictConfig

from src.algorithms.advantage_alignment import AdvantageAlignment
from src.envs.exchange_game import make_vectorized_env
from src.utils.utils import seed_all, instantiate_agent


@hydra.main(version_base="1.3", config_path="configs", config_name="eg.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: None
    """
    seed_all(cfg['seed'])
    env = make_vectorized_env(cfg['batch_size'])
    agent = instantiate_agent(cfg)
    algorithm = AdvantageAlignment(
        env=env, 
        agent_1=agent, 
        agent_2=agent, 
        train_cfg=cfg.training
    )
    algorithm.train_loop()


if __name__ == "__main__":
    main()