import numpy as np
from env import HRS_env
from env_discrete import HRS_env_disc
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import ProgressBarCallback

def main():
    env = HRS_env()
    check_env(env, warn=True)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=100000, progress_bar=ProgressBarCallback())
    model.save("ppo_HRS")

    window_size = 100

    rewards, hydrogen, loss_power, energy_produced, elec_sold, demand = env.get_res()

    rewards_smooth = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

    fig = plt.figure(figsize=(14,10))
    fig.subplots_adjust(hspace=0.5)  # Aggiunge spazio tra i grafici
    plt.subplot(5,1,1)
    plt.plot(rewards_smooth)
    plt.title("Rewards")

    hydrogen_smooth = np.convolve(hydrogen, np.ones(window_size)/window_size, mode='valid')

    plt.subplot(5,1,2)
    plt.plot(hydrogen_smooth)
    plt.title("Hydrogen Stored")

    energy_smooth = np.convolve(energy_produced, np.ones(window_size)/window_size, mode='valid')
    plt.subplot(5,1,3)
    plt.plot(energy_smooth)
    plt.title("Hydrogen sold")

    elec_smooth = np.convolve(elec_sold, np.ones(window_size)/window_size, mode='valid')
    plt.subplot(5,1,4)
    plt.plot(elec_smooth)
    plt.title("Electricity sold")

    loss_smooth = np.convolve(loss_power, np.ones(window_size)/window_size, mode='valid')
    plt.subplot(5,1,5)
    plt.plot(loss_smooth)
    plt.title("Loss Power")

    fig = plt.figure(figsize=(14,10))
    dem_smooth = np.convolve(demand, np.ones(window_size)/window_size, mode='valid')

    plt.subplot(1,1,1)
    plt.plot(dem_smooth)
    plt.title("demand remained")

    plt.show()
    env.close()

if __name__ == "__main__":
    main()