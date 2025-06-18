from stable_baselines3 import PPO
import numpy as np
import pandas as pd
import os
from envIRL import HRS_envIRL
#from env import HRS_env
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import ProgressBarCallback

def main():
    
    envIRL = HRS_envIRL()

    df = pd.read_csv("learned_rewards.csv")
    learned_rewards = {}

    # Itera sui range e crea una reward per ogni bin compreso nel range
    for row in df.itertuples(index=False):
        range_str = row.state_range  # es: "0-50"
        reward = float(row.reward)
        
        low, high = map(int, range_str.split("-"))
        
        # Aggiungi tutti i bin di hydrogen_level da low a high-1
        for h in range(low, high):
            learned_rewards[(h, 500, 0, 0)] = reward

    envIRL.learned_rewards = learned_rewards
    
    model = PPO.load("ppo_HRS_IRL.zip", envIRL, verbose=0)

    
    model.learn(total_timesteps=100000, reset_num_timesteps=False, progress_bar=ProgressBarCallback())
    model.save("ppo_HRS_IRL")

    csv_file = "training_data.csv"

    learned_rewards, hydrogen_sold, hydrogen_stored, electricity_sold, loss, demand = envIRL.get_res()
    #learned_rewards, hydrogen_stored, loss, hydrogen_sold, electricity_sold = env.get_res()
    
    
    # Se il file CSV esiste, carica i dati precedenti e concatenali
    if os.path.exists(csv_file):
        old_data = pd.read_csv(csv_file)
        learned_rewards = list(old_data["learned_rewards"].values) + list(learned_rewards)
        hydrogen_sold = list(old_data["hydrogen_sold"].values) + list(hydrogen_sold)
        hydrogen_stored = list(old_data["hydrogen_stored"].values) + list(hydrogen_stored)
        electricity_sold = list(old_data["electricity_sold"].values) + list(electricity_sold)
        loss = list(old_data["loss_power"].values) + list(loss)


    new_data = pd.DataFrame({
        "learned_rewards": learned_rewards,
        "hydrogen_sold": hydrogen_sold,
        "hydrogen_stored": hydrogen_stored,
        "electricity_sold": electricity_sold,
        "loss_power": loss
    })

    # Salva i dati aggiornati nel CSV
    new_data.to_csv(csv_file, index=False)
    

    window_size = int(len(learned_rewards)/1000)

    learned_smooth = np.convolve(learned_rewards, np.ones(window_size)/window_size, mode='valid')

    fig = plt.figure(figsize=(14,10))
    fig.subplots_adjust(hspace=0.5)  # Aggiunge spazio tra i grafici
    plt.subplot(5,1,1)
    plt.plot(learned_smooth)
    plt.title("Learned Rewards")

    hydr_smooth = np.convolve(hydrogen_sold, np.ones(window_size)/window_size, mode='valid')

    plt.subplot(5,1,2)
    plt.plot(hydr_smooth)
    plt.title("Hydrogen sold")

    stor_smooth = np.convolve(hydrogen_stored, np.ones(window_size)/window_size, mode='valid')

    plt.subplot(5,1,3)
    plt.plot(stor_smooth)
    plt.title("Hydrogen stored")


    elec_smooth = np.convolve(electricity_sold, np.ones(window_size)/window_size, mode='valid')

    plt.subplot(5,1,4)
    plt.plot(elec_smooth)
    plt.title("Electricity sold")

    loss_smooth = np.convolve(loss, np.ones(window_size)/window_size, mode='valid')

    plt.subplot(5,1,5)
    plt.plot(loss_smooth)
    plt.title("loss power")
    
    plt.show()
    envIRL.close()
    #env.close()

if __name__ == "__main__":
    main()