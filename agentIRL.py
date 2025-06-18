import numpy as np
import pandas as pd
from envIRL import HRS_envIRL
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import ProgressBarCallback

def main():
    
    window_size = 125
    
    envIRL = HRS_envIRL()
    check_env(envIRL, warn=True)
    
    envIRL.train_irl()

    model = PPO("MlpPolicy", envIRL, verbose=0)
    model.learn(total_timesteps=125000, progress_bar=ProgressBarCallback())
    model.save("ppo_HRS_IRL")

    learned_rewards, hydrogen_sold, hydrogen_stored, electricity_sold, loss, demand = envIRL.get_res()

    df = pd.DataFrame({
        "learned_rewards": learned_rewards,
        "hydrogen_sold": hydrogen_sold,
        "hydrogen_stored": hydrogen_stored,
        "electricity_sold": electricity_sold,
        "loss_power": loss,
        "demand_remained": demand
    })

    # Salva il DataFrame in un file CSV
    df.to_csv("training_data.csv", index=False)

    learned_smooth = np.convolve(learned_rewards, np.ones(window_size)/window_size, mode='valid')

    fig = plt.figure(figsize=(14,10))
    fig.subplots_adjust(hspace=0.5)  # Aggiunge spazio tra i grafici
    plt.subplot(5,1,1)
    plt.plot(learned_smooth)
    plt.title("Learned Rewards")

    stor_smooth = np.convolve(hydrogen_stored, np.ones(window_size)/window_size, mode='valid')

    plt.subplot(5,1,2)
    plt.plot(stor_smooth)
    plt.title("Hydrogen stored")

    hydr_smooth = np.convolve(hydrogen_sold, np.ones(window_size)/window_size, mode='valid')

    plt.subplot(5,1,3)
    plt.plot(hydr_smooth)
    plt.title("Hydrogen sold")

    elec_smooth = np.convolve(electricity_sold, np.ones(window_size)/window_size, mode='valid')

    plt.subplot(5,1,4)
    plt.plot(elec_smooth)
    plt.title("Electricity sold")

    loss_smooth = np.convolve(loss, np.ones(window_size)/window_size, mode='valid')

    plt.subplot(5,1,5)
    plt.plot(loss_smooth)
    plt.title("loss power")

    fig = plt.figure(figsize=(14,10))
    dem_smooth = np.convolve(demand, np.ones(window_size)/window_size, mode='valid')

    plt.subplot(1,1,1)
    plt.plot(dem_smooth)
    plt.title("demand remained")
    
    plt.show()
    envIRL.close()
    

if __name__ == "__main__":
    main()