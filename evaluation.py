import os
import time
from collections import defaultdict

import hydra
import pandas as pd
import torch
import wandb 

from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from meltingpot import scenario as scene, substrate

from utils import instantiate_agent


def _gen_sim_scenario(state, scenario, cxt_len, agents):
    from train import call_models_forward, TrajectoryBatch, sample_actions_categorical
    time_metrics = defaultdict(int)
    B = 1 # batch size is one, as we don't have the TorchRL parallel env
    N = len(agents)
    assert N == len(scenario.action_spec())
    device = agents[0].device

    agent_rgb = torch.stack([torch.from_numpy(x['RGB']) for x in state.observation], dim=0)
    full_map = torch.from_numpy(scenario._substrate.observation()[0]['WORLD.RGB'])
    _, H, W, C = agent_rgb.shape
    G, L, _ = full_map.shape
    history = torch.zeros((B * N, cxt_len, H, W, C), device=device)
    history_actions = torch.zeros((B * N, cxt_len + 1), device=device)
    history_full_maps = torch.zeros((B, cxt_len, G, L, C), device=device)
    history_action_logits = torch.zeros((B * N, cxt_len, 8), device=device)

    # Self-play only TODO: Add replay buffer of agents
    actors = [agents[i % N].actor for i in range(B * N)]

    # set to eval mode
    for actor in actors:
        actor.eval()

    trajectory = TrajectoryBatch(
        sample_obs=agent_rgb.unsqueeze(0),
        sample_full_map=full_map.unsqueeze(0),
        batch_size=B,
        n_agents=N,
        cxt_len=cxt_len,
        device=device,
        use_full_maps=True
    )

    rewards = torch.zeros((B * N)).to(device)
    log_probs = torch.zeros((B * N)).to(device)
    actions = torch.zeros((B * N, 1)).to(device) # Initial token

    # create our single kv_cache
    kv_cache = actors[0].encoder.transformer_encoder.generate_empty_keys_values(n=B * N, max_tokens=cxt_len * 2)

    i = 0
    while i < cxt_len:
        start_time = time.time()

        agent_rgb = torch.stack([torch.from_numpy(x['RGB']) for x in state.observation], dim=0)
        full_maps = torch.from_numpy(scenario._substrate.observation()[0]['WORLD.RGB'])[None, None, ...]

        observations = agent_rgb
        observations = observations.reshape(-1, *observations.shape[1:]).unsqueeze(0)

        trajectory.add_step(observations, full_maps, rewards, log_probs.detach(), actions, i)

        history[:, i, ...] = observations.squeeze(1)
        history_actions[:, i] = actions.squeeze(1)
        history_full_maps[:, i, ...] = full_maps.squeeze(1)

        observations = history[:, :i + 1, ...]
        actions = history_actions[:, :i + 1]
        full_maps = history_full_maps[:, :i + 1, ...]
        full_maps_repeated = full_maps.unsqueeze(1).repeat((1, N, 1, 1, 1, 1)).reshape((B * N,) + full_maps.shape[1:])
        time_metrics["time_metrics/gen/obs_process"] += time.time() - start_time
        start_time = time.time()

        past_ks = torch.stack([layer_cache.get()[0] for layer_cache in kv_cache._keys_values], dim=1)  # dim=1 because we need the first dimension to be the batch dimension
        past_vs = torch.stack([layer_cache.get()[1] for layer_cache in kv_cache._keys_values], dim=1)
        with torch.no_grad():
            action_logits, this_kvs = call_models_forward(
                actors,
                observations,
                actions,
                full_maps_repeated,
                agents[0].device,
                full_seq=0,
                past_ks_vs=(past_ks, past_vs)
            )
        time_metrics["time_metrics/gen/nn_forward"] += time.time() - start_time
        start_time = time.time()
        history_action_logits[:, i, :] = action_logits.reshape(B * N, -1)

        # update the kv_caches
        for j, layer_cache in enumerate(kv_cache._keys_values):  # looping over caches for layers
            this_layer_k = this_kvs['k'][:, j, ...]
            this_layer_v = this_kvs['v'][:, j, ...]
            layer_cache.update(this_layer_k, this_layer_v)
        time_metrics["time_metrics/gen/kv_cache_update"] += time.time() - start_time
        start_time = time.time()
        actions = torch.argmax(torch.softmax(action_logits.reshape(B * N, -1), dim=1), dim=1)

        actions = actions.reshape(B * N, 1)
        time_metrics["time_metrics/gen/sample_actions"] += time.time() - start_time
        start_time = time.time()

        state = scenario.step(actions.reshape(-1))
        time_metrics["time_metrics/gen/env_step_actions"] += time.time() - start_time
        rewards = torch.stack([torch.from_numpy(r) for r in  state.reward], dim=0)

        if state.last():
            break

        i+=1

    start_time = time.time()
    trajectory.add_actions(actions, i)
    time_metrics["time_metrics/gen/replay_buffer"] += time.time() - start_time
    print(f"(|||)Time metrics: {time_metrics}")
    return trajectory, state

def normalize(scenario, focal_per_capita_return_raw):
    # example usage : normalize('commons_harvest__open_0', 1.8)
    # load the data
    scenario_results = pd.read_feather('data/melting_pot_2_scenario_results.feather')
    scenario_results = scenario_results.reset_index()
    # filter by scenario
    scenario_results = scenario_results[scenario_results['scenario'] == scenario]
    # min scenario score
    min_score = scenario_results['focal_per_capita_return'].min()
    # max scenario score
    max_score = scenario_results['focal_per_capita_return'].max()
    # normalize the scores
    return (focal_per_capita_return_raw - min_score) / (max_score - min_score)

def evaluate_agents_on_scenario(
    agents, 
    scenario_name, 
    num_frames, 
    cxt_len, 
    num_repeats: int = 1, 
    return_trajectories: bool=False, 
    return_k_trajectories: int = 0, 
    run_until_done: bool = False,
    max_env_steps: int = 3000,
    ) -> float:
    if return_trajectories is False:
        assert return_k_trajectories == 0
    else:
        assert return_k_trajectories <= num_repeats

    assert run_until_done is False or num_frames == -1, "If run_until_done is True, num_frames must be -1"

    scenario_to_info = {}

    scenario_config = scene.get_config(scenario_name)
    env = scene.build_from_config(scenario_config)
    state = env.reset()

    for i in range(num_repeats):
        print(f'Evaluating currently at step {i}')
        stop = False
        steps = 0
        trajectories = []
        avg_rewards = []
        avg_returns = []
        while not stop:
            trajectory, state = _gen_sim_scenario(state, env, cxt_len, agents)
            trajectories.append(trajectory)
            avg_rewards.append(trajectory.data['rewards'].mean().item())
            avg_returns.append(trajectory.data['rewards'].sum(axis=-1).mean())
            steps += cxt_len
            if run_until_done:
                stop = state.last() or steps > max_env_steps
            elif steps >= num_frames:
                stop = True

        # concat all trajectories
        trajectory = trajectories[0]
        for j in range(1, len(trajectories)):
            trajectory = trajectory.merge_two_trajectories_time_wise(trajectories[j])

        if i <= return_k_trajectories - 1:
            scenario_to_info.setdefault(scenario_name, {}).setdefault('trajectories', []).append(trajectory)

    scenario_to_info.setdefault(scenario_name, {})['avg_reward'] = sum(avg_rewards) / len(avg_rewards)
    scenario_to_info.setdefault(scenario_name, {})['avg_return'] = torch.sum(trajectory.data['rewards'], dim=1).mean()
    return scenario_to_info

@hydra.main(version_base="1.3", config_path="configs", config_name="meltingpot.yaml")
def main(cfg: DictConfig) -> None:
    model_folder = f"/home/mila/j/juan.duque/projects/advantage-alignment/experiments/{cfg['agent_folder']}/"
    print(f'model folder is: {model_folder}')
    model_name = 'agent.pt'
    cxt_len = cfg['max_cxt_len']

    wandb.init(
        project="Meltingpot",
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=cfg["wandb_dir"],
        anonymous="allow",
        mode=cfg["wandb"], 
    )
    
    agent = instantiate_agent(cfg)
    model_path = os.path.join(model_folder, model_name)
    model_dict = torch.load(model_path, map_location=torch.device(cfg['device']))
    agent.actor.load_state_dict(model_dict['actor_state_dict'])
    
    # Get all substrates and their corresponding scenarios
    substrates_scenarios = scene._scenarios_by_substrate()
    scenario = cfg['env']['scenario']
    print(f"Evaluating substrate: {scenario}")

    results = []
    
    for scenario_name in substrates_scenarios[scenario]:
        print(f"Evaluating scenario: {scenario_name}")

        scenario_config = scene.get_config(scenario_name)
        env = scene.build_from_config(scenario_config)
        num_agents_in_scenario = len(env.action_spec())
        agents = [agent for _ in range(num_agents_in_scenario)]
        
        scenario_to_info = evaluate_agents_on_scenario(
            agents=agents,
            scenario_name=scenario_name,
            num_frames=-1,
            cxt_len=cxt_len,
            num_repeats=100,  # Evaluate n times
            return_trajectories=False,
            run_until_done=True,
            max_env_steps=cfg['env']['max_env_steps']
        )
        avg_reward = scenario_to_info[scenario_name]['avg_reward']
        avg_return = scenario_to_info[scenario_name]['avg_return'].item()  # Convert tensor to float if necessary
        normalized_return = normalize(scenario_name, avg_return)
        
        print(f"Scenario: {scenario_name}, Avg Reward over 100 runs: {avg_reward}, Avg Return over 100 runs: {avg_return}, Normalized Return: {normalized_return}")
        
        results.append({
            'substrate': scenario,
            'scenario': scenario_name,
            'avg_reward': avg_reward,
            'avg_return': avg_return,
            'normalized_return': normalized_return
        })

        wandb.log({
            f'{scenario}/{scenario_name}/avg_reward': avg_reward,
            f'{scenario}/{scenario_name}/avg_return': avg_return,
            f'{scenario}/{scenario_name}/normalized_return': normalized_return,
        })
    
    with open('evaluation_results.txt', 'w') as f:
        for result in results:
            f.write(f"Substrate: {result['substrate']}, Scenario: {result['scenario']}, Avg Reward: {result['avg_reward']}, Avg Return: {result['avg_return']}, Normalized Return: {result['normalized_return']}\n")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    main()