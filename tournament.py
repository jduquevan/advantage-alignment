from src.algorithms.advantage_alignment import gen_episodes_between_agents, AdvantageAlignment, AlwaysCooperateAgent, AlwaysDefectAgent
import gymnasium as gym
import hydra
import wandb
from omegaconf import DictConfig
from src.algorithms.advantage_alignment import AdvantageAlignment
from src.envs.eg_vectorized import DiscreteEG, ObligatedRatioDiscreteEG
from src.utils.utils import seed_all, instantiate_agent, resume_agent, to_json
import multiprocessing
from multiprocessing import get_context

def process_agent_pair(env, cfg, pair) -> None:
    type1, agent1_id, agent1, type2, agent2_id, agent2 = pair
    trajectory = gen_episodes_between_agents(env, cfg.batch_size, 10, agent1, agent2)
    # Detach and convert to CPU tensors if required before sending them through the multiprocessing queue
    trajectory.detach_and_move_to_cpu()
    return {f"{type1}_{agent1_id} vs {type2}_{agent2_id}": trajectory.data}

@hydra.main(version_base="1.3", config_path="configs", config_name="tournament.yaml")
def main(cfg: DictConfig):
    seed_all(cfg.seed)
    if cfg.wandb:
        wandb.init(
            config=dict(cfg), 
            project="Advantage Alignment",
            dir=cfg["wandb_dir"], 
            reinit=True, 
            anonymous="allow",
            mode=cfg["wandb"]
        )
    if cfg['env']['type'] == 'f1':
        env = gym.make_vec('f1-v0', num_envs=cfg.batch_size)
    elif cfg['env']['type'] == 'eg':
        env = DiscreteEG(batch_size=cfg.batch_size, device=cfg.device)
    elif cfg['env']['type'] == 'eg-obligated_ratio':
        env = ObligatedRatioDiscreteEG(batch_size=cfg.batch_size, device=cfg.device)
    else:
        raise ValueError(f"Environment type {cfg['env']['type']} not supported.")
    
    agent_pool = {"aa":{}, "ppo":{}, "ac":{}, "ad":{}}
    if "aa" in cfg.agent_pool.types:
        assert len(cfg.agent_pool.aa_pths) > 0, "No AA agents provided."
        for id, aa_pth in enumerate(cfg.agent_pool.aa_pths):
            agent = resume_agent(cfg, aa_pth)
            agent.eval()
            agent.to(cfg.device)
            agent_pool["aa"][id] = agent
    if "ppo" in cfg.agent_pool.types:
        assert len(cfg.agent_pool.ppo_pths) > 0, "No PPO agents provided."
        for id, ppo_pth in enumerate(cfg.agent_pool.ppo_pths):
            agent = resume_agent(cfg, ppo_pth)
            agent.eval()
            agent.to(cfg.device)
            agent_pool["ppo"][id] = agent
    if "ac" in cfg.agent_pool.types:
        agent_pool["ac"][0] = agent = AlwaysCooperateAgent(cfg.device)
    if "ad" in cfg.agent_pool.types:
        agent_pool["ad"][0] = agent = AlwaysDefectAgent(cfg.device)

    pairs_to_run = []
    for type1, agents1 in agent_pool.items():
        for type2, agents2 in agent_pool.items():
            if type1 != type2:
                for agent1_id, agent1 in agents1.items():
                    for agent2_id, agent2 in agents2.items():
                        pairs_to_run.append((type1, agent1_id, agent1, type2, agent2_id, agent2))

    def process_agent_pairs(pairs_to_run):
        try:
            with get_context("spawn").Pool() as pool:
                out = pool.starmap(process_agent_pair, [(env, cfg, pair) for pair in pairs_to_run])
            # print("Results from multiprocessing:", out)  # Check what is being returned
            return out
        except Exception as e:
            print("Error during multiprocessing:", e)
            return None

    out = process_agent_pairs(pairs_to_run)
    out_dict = {k: v for d in out for k, v in d.items()}
    to_json(out_dict, cfg.tournament_out)
    return
    
if __name__ == '__main__':
    main()
    