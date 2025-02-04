from env import HRS_env
#from imitation.algorithms.gail import GAIL 
from imitation.util.util import make_vec_env
from stable_baselines3 import PPO
import numpy as np
#from imitation_data import rollout

vec_env = make_vec_env(HRS_env, n_envs = 1, rng = np.random.default_rng())

#creazione traiettorie esperto
transitions = rollout.flatten_trajectories([
    rollout.generate_trajectories(lambda obs: np.random.choice(5), vec_env, n_trajectories = 10)
])
'''
#creazione modello GAIL
gail = GAIL(
    venv = vec_env,
    expert_data = transitions,
    gen_algo = PPO("MlpPolicy", vec_env, verbose = 1)
)

#training
gail.train(10000)

#creazione modello RL
model = PPO("MlpPolicy", vec_env, verbose = 1)
model.learn(total_timesteps = 50000)
model.save("hrs_policy")

#test
env = HRV_env
obs, _ = env.reset()
for _ in range(10):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
'''