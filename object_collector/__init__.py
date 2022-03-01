from gym.envs.registration import register

register(id='oc-v1', entry_point='object_collector.envs:ObjectCollectorEnv')
