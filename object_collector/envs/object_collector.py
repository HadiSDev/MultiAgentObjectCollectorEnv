import cv2
import gym
import numpy as np
from glm import vec2, pi
from gym.spaces import Box, Discrete, Dict
from gym.utils import EzPickle

from object_collector.envs.AgentState import AgentState, Actions
from object_collector.envs.MapState import MapState


class ObjectCollectorEnv(gym.Env, EzPickle):

    def __init__(self, map_size=(500, 500), n_objectives=11):
        EzPickle.__init__(self)
        self.map_size = map_size
        self.map_state = MapState(self.map_size,
                                  n_objectives=n_objectives)

        self.agent = AgentState(vec2(-0.8, 0.8), orientation=0, name="agent_0", use_random_position=True)
        # spaces = {
        #     "visual": Box(shape=(100, 100, 3), low=0, high=1, dtype=np.float32),
        #     "orientation": Box(shape=(2,), dtype=np.float32, low=-1, high=1),
        #     "action_mask": Box(shape=(3,), low=0, high=1)
        # }
        self.observation_space = Box(shape=(100, 100, 3), low=0, high=1, dtype=np.uint8)
        self.action_space = Discrete(3)
        self.step_penalty = 1
        self.collection_reward = 10
        self.reward = 0

    def get_observation(self):
        visual = cv2.resize(self.map_state.get_current_state(self.agent),
                            (100, 100),
                            interpolation=cv2.INTER_AREA)
        # orientation = np.array(self.agent.get_orientation_vector(), dtype=np.float32)
        # action_mask = np.array([1, 1, 1], dtype=np.float32)
        # action_mask[2] = 1 if self.can_move(Actions.MOVE_ORIENTATION) else 0
        return visual.astype(np.uint8)

    def step(self, action: Actions):
        if self.can_move(action):
            self.agent.do_action(action)
            collected = self.map_state.try_collect(self.agent)
            reward = collected * self.collection_reward - self.step_penalty
        else:
            reward = -100

        done = False
        if self.map_state.all_objectives_collected():
            done = True
        obs = self.get_observation()
        info = {}

        return obs, reward, done, info

    def can_move(self, action):
        if action == Actions.MOVE_ORIENTATION:
            if self.agent.check_bounds() == -1:
                return False
        return True

    def reset(self):
        self.agent.reset_state()
        self.map_state.reset()
        self.reward = 0

        return self.get_observation()

    def render(self, mode="human"):
        v = self.map_state.get_current_state(self.agent)
        cv2.imshow("Game", v)
        cv2.waitKey(1)


if __name__ == "__main__":

    env = ObjectCollectorEnv()
    env.reset()
    while True:
        env.step(Actions.MOVE_ORIENTATION)
        env.render()
