import numpy as np
import pandas as pd
from env import HRS_env
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

    rewards, hydrogen, loss_power, hydrogen_produced, elec_sold = env.get_res()

    csv_file = "training_data.csv"
    old_data = pd.read_csv(csv_file)
    learned_rewards = list(old_data["learned_rewards"].values)[:len(rewards)]
    hydrogen_sold = list(old_data["hydrogen_sold"].values)[:len(hydrogen)]
    hydrogen_stored = list(old_data["hydrogen_stored"].values)[:len(hydrogen_produced)]
    electricity_sold = list(old_data["electricity_sold"].values)[:len(elec_sold)]
    loss = list(old_data["loss_power"].values)[:len(loss_power)]
    
    rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    learned_rewards = np.convolve(learned_rewards, np.ones(window_size)/window_size, mode='valid')
    hydrogen = np.convolve(hydrogen, np.ones(window_size)/window_size, mode='valid')
    hydrogen_sold = np.convolve(hydrogen_sold, np.ones(window_size)/window_size, mode='valid')
    hydrogen_produced = np.convolve(hydrogen_produced, np.ones(window_size)/window_size, mode='valid')
    hydrogen_stored = np.convolve(hydrogen_stored, np.ones(window_size)/window_size, mode='valid')
    elec_sold = np.convolve(elec_sold, np.ones(window_size)/window_size, mode='valid')
    electricity_sold = np.convolve(electricity_sold, np.ones(window_size)/window_size, mode='valid')
    loss_power = np.convolve(loss_power, np.ones(window_size)/window_size, mode='valid')
    loss = np.convolve(loss, np.ones(window_size)/window_size, mode='valid')

    val = max(rewards) / max(learned_rewards)
    rewards = [reward / val for reward in rewards]

    x = range(len(rewards))
    fig, axs = plt.subplots(5, 1, figsize=(15, 8), sharex=True)
    fig.subplots_adjust(hspace=0.5)  # Aggiunge spazio tra i grafici

    # Plot sovrapposti con colori e label
    axs[0].plot(x, rewards, label='RL', color='blue')
    axs[0].plot(x, learned_rewards, label='IRL', color='orange')
    axs[0].set_title('Rewards')
    axs[0].legend()

    axs[1].plot(x, hydrogen_produced, label='RL', color='blue')
    axs[1].plot(x, hydrogen_sold, label='IRL', color='orange')
    axs[1].set_title('Hydrogen sold')

    axs[2].plot(x, hydrogen, label='RL', color='blue')
    axs[2].plot(x, hydrogen_stored, label='IRL', color='orange')
    axs[2].set_title('Hydrogen stored')

    axs[3].plot(x, elec_sold, label='RL', color='blue')
    axs[3].plot(x, electricity_sold, label='IRL', color='orange')
    axs[3].set_title('Electricity sold')

    axs[4].plot(x, loss_power, label='RL', color='blue')
    axs[4].plot(x, loss, label='IRL', color='orange')
    axs[4].set_title('Loss Power')
    
    plt.show()
    env.close()

if __name__ == "__main__":
    main()