from typing import Dict, Text, Tuple, Optional, TypeVar

import numpy as np

from highway_env import utils
from highway_env.envs.common.action import Action
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle, RockVehicle
from highway_env.vehicle.objects import Obstacle

Observation = TypeVar("Observation")
LaneIndex = Tuple[str, str, int]

class RacetrackEnv(AbstractEnv):
    """
    A continuous control environment.

    The agent needs to learn two skills:
    - follow the tracks
    - avoid collisions with other vehicles

    Credits and many thanks to @supperted825 for the idea and initial implementation.
    See https://github.com/eleurent/highway-env/issues/231
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "OccupancyGrid",
                    "features": ["presence", "on_road"],
                    "grid_size": [[-18, 18], [-18, 18]],
                    "grid_step": [3, 3],
                    "as_image": False,
                    "align_to_vehicle_axes": True,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": False,
                    "lateral": True,
                    "target_speeds": [0, 5, 10],
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 300,
                "collision_reward": -1,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1,
                "action_reward": -0.3,
                "controlled_vehicles": 1,
                "other_vehicles": 1,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.5],
            }
        )
        return config

    def _reward(self, action: np.ndarray) -> float:
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward = utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        return {
            "lane_centering_reward": 1
            / (1 + self.config["lane_centering_cost"] * lateral**2),
            "action_reward": np.linalg.norm(action),
            "collision_reward": self.vehicle.crashed,
            "on_road_reward": self.vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        # Initialise First Lane
        lane = StraightLane(
            [42, 0],
            [100, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=5,
            speed_limit=speedlimits[1],
        )
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [42, 5],
                [100, 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[2],
            ),
        )
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + 5,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[2],
            ),
        )

        # 3 - Vertical Straight
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [120, -20],
                [120, -30],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [125, -20],
                [125, -30],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )

        # 4 - Circular Arc #2
        center2 = [105, -30]
        radii2 = 15
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[4],
            ),
        )
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2 + 5,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[4],
            ),
        )

        # 5 - Circular Arc #3
        center3 = [70, -30]
        radii3 = 15
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center3,
                radii3 + 5,
                np.deg2rad(0),
                np.deg2rad(136),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                speed_limit=speedlimits[5],
            ),
        )
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center3,
                radii3,
                np.deg2rad(0),
                np.deg2rad(137),
                width=5,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=speedlimits[5],
            ),
        )

        # 6 - Slant
        net.add_lane(
            "f",
            "g",
            StraightLane(
                [55.7, -15.7],
                [35.7, -35.7],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[6],
            ),
        )
        net.add_lane(
            "f",
            "g",
            StraightLane(
                [59.3934, -19.2],
                [39.3934, -39.2],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[6],
            ),
        )

        # 7 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
        center4 = [18.1, -18.1]
        radii4 = 25
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(315),
                np.deg2rad(170),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center4,
                radii4 + 5,
                np.deg2rad(315),
                np.deg2rad(165),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "h",
            "i",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(170),
                np.deg2rad(56),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "h",
            "i",
            CircularLane(
                center4,
                radii4 + 5,
                np.deg2rad(170),
                np.deg2rad(58),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[7],
            ),
        )

        # 8 - Circular Arc #5 - Reconnects to Start
        center5 = [43.2, 23.4]
        radii5 = 18.5
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center5,
                radii5 + 5,
                np.deg2rad(240),
                np.deg2rad(270),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                speed_limit=speedlimits[8],
            ),
        )
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center5,
                radii5,
                np.deg2rad(238),
                np.deg2rad(268),
                width=5,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=speedlimits[8],
            ),
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = (
                ("a", "b", rng.integers(2))
                if i == 0
                else self.road.network.random_lane_index(rng)
            )
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road, lane_index, speed=None, longitudinal=rng.uniform(20, 50)
            )

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        # Front vehicle
        vehicle = IDMVehicle.make_on_lane(
            self.road,
            ("b", "c", lane_index[-1]),
            longitudinal=rng.uniform(
                low=0, high=self.road.network.get_lane(("b", "c", 0)).length
            ),
            speed=6 + rng.uniform(high=3),
        )
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in range(rng.integers(self.config["other_vehicles"])):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = IDMVehicle.make_on_lane(
                self.road,
                random_lane_index,
                longitudinal=rng.uniform(
                    low=0, high=self.road.network.get_lane(random_lane_index).length
                ),
                speed=6 + rng.uniform(high=3),
            )
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 20:
                    break
            else:
                self.road.vehicles.append(vehicle)


class F1Env(AbstractEnv):
    """
    A continuous control environment.

    The agent needs to learn two skills:
    - follow the tracks
    - avoid collisions with other vehicles

    Credits and many thanks to @supperted825 for the idea and initial implementation.
    See https://github.com/eleurent/highway-env/issues/231
    """

    def __init__(self, config: dict = None, render_mode: Optional[str] = None) -> None:
        super().__init__(config=config, render_mode=render_mode)
        
        # DRS
        self.drs_actions = 24  # Multiple of 3
        self.drs_reset = True
        self.max_speed = 40.0
        self.drs_speed = 50.0

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "controlled_vehicles": 2,
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {
                        "type": "Kinematics",
                    }
                },
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "ContinuousAction",
                    }
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 300,
                "collision_reward": -1,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1,
                "action_reward": -0.3,
                "other_vehicles": 1,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.5],
            }
        )
        return config

    def compare_positions(self):
        # By segment
        zero_greater = (
            self.road.vehicles[0].lane_index[0:2] >
            self.road.vehicles[1].lane_index[0:2]
        )
        zero_equal = (
            self.road.vehicles[0].lane_index[0:2] ==
            self.road.vehicles[1].lane_index[0:2]
        )
        if zero_equal:
            zero_equal = (
                np.abs(
                    self.road.vehicles[0].lane_offset[0] -
                    self.road.vehicles[1].lane_offset[0]
                ) < 1e-1
            )
            # By longitude
            zero_greater = (
                (
                    self.road.vehicles[0].lane_offset[0] >
                    self.road.vehicles[1].lane_offset[0]
                )
                and not zero_equal
            )
        
        if zero_equal:
            # By lane
            zero_greater = (
                self.road.vehicles[0].lane_index[2] <
                self.road.vehicles[1].lane_index[2]
            )
        return zero_greater

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError(
                "The road and vehicle must be initialized in the environment implementation"
            )

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        for vehicle in self.controlled_vehicles:
            distance_to_lane = self.get_closest_lane_distance(vehicle.position, vehicle.heading)
            if distance_to_lane > 10:
                vehicle.crashed = True

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        if self.render_mode == "human":
            self.render()

        if terminated:
            self.reset()

        return obs, reward, terminated, truncated, info

    def get_closest_lane_distance(
        self, position: np.ndarray, heading: Optional[float] = None
    ) -> LaneIndex:
        """
        Get the index of the lane closest to a world position.

        :param position: a world position [m].
        :param heading: a heading angle [rad].
        :return: the index of the closest lane.
        """
        indexes, distances = [], []
        for _from, to_dict in self.road.network.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance_with_heading(position, heading))
                    indexes.append((_from, _to, _id))
        return np.min(distances)

    def compute_drs(self):
        drs_vehicle = 1 if self.compare_positions() else 0
        other_vehicle = 0 if self.compare_positions() else 1
        self.controlled_vehicles[drs_vehicle].MAX_SPEED = self.drs_speed
        self.controlled_vehicles[other_vehicle].MAX_SPEED = self.max_speed

    def _reward(self, action: np.ndarray) -> Tuple[float]:
        if self.steps % self.drs_actions == 0:
            self.drs_reset = True
        
        if self.drs_reset:
            self.compute_drs()
            self.drs_reset = False
        
        reward_dicts = self._rewards(action)
        rewards = []
        for reward_instance in reward_dicts:
            reward = sum(
                self.config.get(name, 0) * reward for name, reward in reward_instance.items()
            )
            reward = utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1])
            reward *= reward_instance["on_road_reward"]
            rewards.append(reward)
        return tuple(rewards)

    def _rewards(self, action: np.ndarray) -> Tuple[Dict[Text, float]]:
        rewards = []
        for vehicle in self.controlled_vehicles:
            _, lateral = self.vehicle.lane.local_coordinates(vehicle.position)
            reward = {
                "lane_centering_reward": 1
                / (1 + self.config["lane_centering_cost"] * lateral**2),
                "action_reward": np.linalg.norm(action),
                "collision_reward": vehicle.crashed,
                "on_road_reward": vehicle.on_road,
            }
            rewards.append(reward)
        return tuple(rewards)

    def _is_terminated(self) -> bool:
        is_terminated = True
        for vehicle in self.controlled_vehicles:
            if not vehicle.crashed:
                is_terminated = False
        return is_terminated

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        # Initialise First Lane
        lane = StraightLane(
            [42, 0],
            [100, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=5,
            speed_limit=speedlimits[1],
        )
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [42, 5],
                [100, 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[2],
            ),
        )
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + 5,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[2],
            ),
        )

        # 3 - Vertical Straight
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [120, -20],
                [120, -30],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [125, -20],
                [125, -30],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )

        # 4 - Circular Arc #2
        center2 = [105, -30]
        radii2 = 15
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[4],
            ),
        )
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2 + 5,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[4],
            ),
        )

        # 5 - Circular Arc #3
        center3 = [70, -30]
        radii3 = 15
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center3,
                radii3 + 5,
                np.deg2rad(0),
                np.deg2rad(136),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                speed_limit=speedlimits[5],
            ),
        )
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center3,
                radii3,
                np.deg2rad(0),
                np.deg2rad(137),
                width=5,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=speedlimits[5],
            ),
        )

        # 6 - Slant
        net.add_lane(
            "f",
            "g",
            StraightLane(
                [55.7, -15.7],
                [35.7, -35.7],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[6],
            ),
        )
        net.add_lane(
            "f",
            "g",
            StraightLane(
                [59.3934, -19.2],
                [39.3934, -39.2],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[6],
            ),
        )

        # 7 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
        center4 = [18.1, -18.1]
        radii4 = 25
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(315),
                np.deg2rad(170),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center4,
                radii4 + 5,
                np.deg2rad(315),
                np.deg2rad(165),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "h",
            "i",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(170),
                np.deg2rad(56),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "h",
            "i",
            CircularLane(
                center4,
                radii4 + 5,
                np.deg2rad(170),
                np.deg2rad(58),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[7],
            ),
        )

        # 8 - Circular Arc #5 - Reconnects to Start
        center5 = [43.2, 23.4]
        radii5 = 18.5
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center5,
                radii5 + 5,
                np.deg2rad(240),
                np.deg2rad(270),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                speed_limit=speedlimits[8],
            ),
        )
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center5,
                radii5,
                np.deg2rad(238),
                np.deg2rad(268),
                width=5,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=speedlimits[8],
            ),
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []

        # TODO: Generalize to more than 1 vehicle
        lower_bound = 0
        upper_bound = 1
        num_integers = 2

        random_lanes = np.random.choice(
            np.arange(lower_bound, upper_bound+1), 
            size=num_integers, 
            replace=False
        )

        for i in range(self.config["controlled_vehicles"]):
            lane_index = (
                ("a", "b", random_lanes[i])
                # if i == 0
                # else self.road.network.random_lane_index(rng)
            )
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road, lane_index, speed=None, longitudinal=20
            )

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        # Front vehicle
        obstacle_lane_index = 0
        vehicle = RockVehicle.make_on_lane(
            self.road,
            ("h", "i", obstacle_lane_index),
            longitudinal=0,
            speed=0,
        )
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in range(rng.integers(self.config["other_vehicles"])):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = IDMVehicle.make_on_lane(
                self.road,
                random_lane_index,
                longitudinal=rng.uniform(
                    low=0, high=self.road.network.get_lane(random_lane_index).length
                ),
                speed=6 + rng.uniform(high=3),
            )
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 20:
                    break
            else:
                self.road.vehicles.append(vehicle)


class MergingEnv(AbstractEnv):
    """
    A continuous control environment.

    The agent needs to learn two skills:
    - follow the tracks
    - avoid collisions with other vehicles

    Credits and many thanks to @supperted825 for the idea and initial implementation.
    See https://github.com/eleurent/highway-env/issues/231
    """

    def __init__(self, config: dict = None, render_mode: Optional[str] = None) -> None:
        super().__init__(config=config, render_mode=render_mode)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "controlled_vehicles": 2,
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {
                        "type": "Kinematics",
                    }
                },
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "ContinuousAction",
                    }
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 300,
                "collision_reward": -1,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1,
                "action_reward": -0.3,
                "other_vehicles": 1,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.5],
            }
        )
        return config

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        obstacle_pass = (self.controlled_vehicles[0].position[0] > self.obstacle.position[0] or self.controlled_vehicles[1].position[0] > self.obstacle.position[0])
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
            "obs_pass": int(obstacle_pass)
        }
        try:
            info["rewards"] = self._rewards(action)
        except NotImplementedError:
            pass
        return info

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError(
                "The road and vehicle must be initialized in the environment implementation"
            )

        # if self.first_vehicle_is_merging:
        #     action[1][1] = 0
        # else:
        #     action[0][1] = 0

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        for vehicle in self.controlled_vehicles:
            distance_to_lane = self.get_closest_lane_distance(vehicle.position, vehicle.heading)
            if distance_to_lane > 3:
                vehicle.crashed = True

        obs = self.observation_type.observe()
        terminated = self._is_terminated()
        reward = self._reward(action, terminated)
        truncated = self._is_truncated()
        info = self._info(obs, action)
        
        if self.render_mode == "human":
            self.render()

        if terminated:
            self.reset()

        return obs, reward, terminated, truncated, info

    def get_closest_lane_distance(
        self, position: np.ndarray, heading: Optional[float] = None
    ) -> LaneIndex:
        """
        Get the index of the lane closest to a world position.

        :param position: a world position [m].
        :param heading: a heading angle [rad].
        :return: the index of the closest lane.
        """
        indexes, distances = [], []
        for _from, to_dict in self.road.network.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance_with_heading(position, heading))
                    indexes.append((_from, _to, _id))
        return np.min(distances)
    
    def _reward(self, action: np.ndarray, terminated: bool) -> Tuple[float]:
        position_1 = self.controlled_vehicles[0].position
        position_2 = self.controlled_vehicles[1].position
        centering_dist_1 = self.road.network.graph['a']['b'][0].distance_with_heading(position_1, None)
        centering_dist_2 = self.road.network.graph['a']['b'][0].distance_with_heading(position_2, None)
        
        r1 = 0.5/(1 + centering_dist_1) 
        r2 = 0.5/(1 + centering_dist_2)
        rewards = [r1 - 0.22, r2 - 0.22]

        if terminated:
            if self.first_vehicle_is_merging:
                max_reward_0 = 5
                max_reward_1 = 1
            else:
                max_reward_0 = 1
                max_reward_1 = 5

            if (position_1[0] > position_2[0] and not self.controlled_vehicles[0].crashed):
                rewards[0] = max_reward_0
            else:
                rewards[1] = max_reward_1
            
            for idx, vehicle in enumerate(self.controlled_vehicles):
                if vehicle.crashed:
                    rewards[idx] = -10
        return tuple(rewards)

    def _is_terminated(self) -> bool:
        obstacle_pass = (
            self.controlled_vehicles[0].position[0] > (self.obstacle.position[0] + 4) or 
            self.controlled_vehicles[1].position[0] > (self.obstacle.position[0] + 4)
        )
        vehicles_crashed = (
            self.controlled_vehicles[0].crashed and
            self.controlled_vehicles[1].crashed
        )
        is_terminated = obstacle_pass or vehicles_crashed

        return is_terminated

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()
        speedlimit = 20

        # Before obstacle

        # Initialise First Lane
        lane = StraightLane(
            [-20, 0],
            [100, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=5,
            speed_limit=speedlimit,
        )
        self.lane = lane

        # Add Lanes to Road Network - Straight Section

        net.add_lane("a", "b", lane)
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [-20, 5],
                [100, 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimit,
            ),
        )

        # After obstacle

        net.add_lane(
            "b",
            "c",
            StraightLane(
                [100, 0],
                [200, 0],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimit,
            ),
        )
        net.add_lane(
            "b",
            "c",
            StraightLane(
                [100, 5],
                [200, 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimit,
            ),
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.obstacle = Obstacle(road, [100, 3])
        self.obstacle_2 = Obstacle(road, [100, 5])
        self.obstacle_3 = Obstacle(road, [100, 7])
        road.objects.append(self.obstacle)
        road.objects.append(self.obstacle_2)
        road.objects.append(self.obstacle_3)
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []

        # TODO: Generalize to more than 1 vehicle
        lower_bound = 0
        upper_bound = 1
        num_integers = 2

        random_lanes = np.random.choice(
            np.arange(lower_bound, upper_bound+1), 
            size=num_integers, 
            replace=False
        )
        initial_positions = np.random.choice(
            [86, 88, 90, 92, 94, 96], 
            size=num_integers, 
            replace=False
        )

        self.first_vehicle_is_merging = random_lanes[0]==1

        for i in range(self.config["controlled_vehicles"]):
            lane_index = (
                ("a", "b", random_lanes[i])
                # if i == 0
                # else self.road.network.random_lane_index(rng)
            )

            if random_lanes[i] == 0:
                initial_position = 76
            else:
                initial_position = 80

            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road, lane_index, speed=None, longitudinal=initial_position
            )

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)
