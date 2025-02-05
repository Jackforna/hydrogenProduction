from env import HRS_env
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic             #debugger
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import ProgressBarCallback

def main():
    ic.enable()
    env = HRS_env()
    check_env(env, warn=True)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, progress_bar=ProgressBarCallback())
    model.save("ppo_HRS")

    rewards, hydrogen, loss_power = env.get_res()

    plt.figure(figsize=(14,10))
    plt.subplot(3,1,1)
    plt.plot(rewards)
    plt.title("Rewards")

    plt.subplot(3,1,2)
    plt.plot(hydrogen)
    plt.title("Hydrogen Stored")

    plt.subplot(3,1,3)
    plt.plot(loss_power)
    plt.title("Loss Power")

    plt.show()
    env.close()







if __name__ == "__main__":
    main()