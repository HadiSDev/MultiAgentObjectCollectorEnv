import cv2
from object_collector.envs import Actions
from object_collector.envs import ObjectCollectorEnv
import keyboard

if __name__ == "__main__":
    env = ObjectCollectorEnv(n_objectives=1)
    env.reset()

    while True:
        state = env.map_state.get_current_state(env.agent)
        cv2.imshow("game", state)
        cv2.waitKey(1)

        if keyboard.read_key() == "w":
            obs, r, d, i = env.step(Actions.MOVE_ORIENTATION)
        elif keyboard.read_key() == "a":
            obs, r, d, i = env.step(Actions.TURN_LEFT)
        elif keyboard.read_key() == "d":
            obs, r, d, i =env.step(Actions.TURN_RIGHT)

        state = env.map_state.get_current_state(env.agent)
        cv2.imshow("game", state)
        cv2.waitKey(1)

        print(f"Reward: {r}")

        if d:
            env.reset()


