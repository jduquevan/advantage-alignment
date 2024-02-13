import os
import sys

grandparent_folder = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
sys.path.append(grandparent_folder)

from algo.rule_based import (
    pavlovAgent,
    ticfortacAgent,
    randomAgent,
    simpletonAgent,
    grudgeAgent,
    detectiveAgent,
    alwaysAgent,
    ticfortacbufferAgent,
    advdetectiveAgent,
)
from env.dilemma.dilemma_gym import IteratedPrisonersDilemmaEnv
from utils.visualize_utils import plot_dict, plot_heatmap

num_iterations = 100
env = IteratedPrisonersDilemmaEnv(num_iterations)
agent_pool_0 = {  # agent_pool_0 is the pool of agents that agent 0 will play against
    "pavlov": pavlovAgent(env, agent_id=0, opponent_id=1, cynic=False),
    "cynicpavlov": pavlovAgent(env, agent_id=0, opponent_id=1, cynic=True),
    "ticfortac": ticfortacAgent(env, agent_id=0, opponent_id=1, cynic=False),
    "cynicticfortac": ticfortacAgent(env, agent_id=0, opponent_id=1, cynic=True),
    "random": randomAgent(env, agent_id=0, opponent_id=1),
    "simpleton": simpletonAgent(env, agent_id=0, opponent_id=1, cynic=False),
    "cynicsimpleton": simpletonAgent(env, agent_id=0, opponent_id=1, cynic=True),
    "grudge": grudgeAgent(env, agent_id=0, opponent_id=1, cynic=False),
    "cynicgrudge": grudgeAgent(env, agent_id=0, opponent_id=1, cynic=True),
    "detective": detectiveAgent(env, agent_id=0, opponent_id=1),
    "alwayscoop": alwaysAgent(env, agent_id=0, opponent_id=1, defect=False),
    "alwaysdefect": alwaysAgent(env, agent_id=0, opponent_id=1, defect=True),
    "ticfortacbuffer": ticfortacbufferAgent(
        env, agent_id=0, opponent_id=1, buffer_size=1, cynic=False
    ),
    "cynicticfortacbuffer": ticfortacbufferAgent(
        env, agent_id=0, opponent_id=1, buffer_size=1, cynic=True
    ),
    "advdetective11": advdetectiveAgent(
        env,
        agent_id=0,
        opponent_id=1,
        cynic=False,
        data_collection_step=11,
        Bernoulli_action=False,
        only_recent_update=True,
    ),
    "cynicadvdetective11": advdetectiveAgent(
        env,
        agent_id=0,
        opponent_id=1,
        cynic=True,
        data_collection_step=11,
        Bernoulli_action=False,
        only_recent_update=True,
    ),
    "advdetective21": advdetectiveAgent(
        env,
        agent_id=0,
        opponent_id=1,
        cynic=False,
        data_collection_step=21,
        Bernoulli_action=False,
        only_recent_update=True,
    ),
    "cynicadvdetective21": advdetectiveAgent(
        env,
        agent_id=0,
        opponent_id=1,
        cynic=True,
        data_collection_step=21,
        Bernoulli_action=False,
        only_recent_update=True,
    ),
    "advdetective11_all_update": advdetectiveAgent(
        env,
        agent_id=0,
        opponent_id=1,
        cynic=False,
        data_collection_step=11,
        Bernoulli_action=False,
        only_recent_update=False,
    ),
    "cynicadvdetective11_all_update": advdetectiveAgent(
        env,
        agent_id=0,
        opponent_id=1,
        cynic=True,
        data_collection_step=11,
        Bernoulli_action=False,
        only_recent_update=False,
    ),
    # "Bernoulliadvdetective": advdetectiveAgent(
    #     env,
    #     agent_id=0,
    #     opponent_id=1,
    #     cynic=False,
    #     data_collection_step=11,
    #     Bernoulli_action=True,
    #     only_recent_update=True,
    # ),
    # "cynicBernoulliadvdetective": advdetectiveAgent(
    #     env,
    #     agent_id=0,
    #     opponent_id=1,
    #     cynic=True,
    #     data_collection_step=11,
    #     Bernoulli_action=True,
    #     only_recent_update=True,
    # ),
}


def run(agents, env, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()["obs"]
        episode_reward = [0, 0]
        done = False
        info = {"history": []}
        while not done:
            actions = [agent.act(state, info) for agent in agents]
            next_state, rewards, done, info = env.step(actions)
            state = next_state
            episode_reward[0] += rewards[0]
            episode_reward[1] += rewards[1]
    return episode_reward


def eval_with_rule_based(agent, pool=agent_pool_0):
    reward_cal = {}
    for k, v in pool.items():
        agent_tests = [agent]
        agent_tests.append(v)
        reward_cal[k] = run(
            agent_tests,
            env,
            num_episodes=100,
        )
    print("rewards: ", reward_cal)
    sum_reward = sum([v[0] for v in reward_cal.values()])
    print("sum_reward: ", sum_reward)
    return reward_cal, sum_reward
