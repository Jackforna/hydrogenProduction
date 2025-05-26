import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from envIRLDQN import HRS_envIRL_DQN
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import ProgressBarCallback

from envIRL import HRS_envIRL

def main():
    # Inizializza ambiente
    envIRL = HRS_envIRL_DQN()

    # Controllo di compatibilità
    check_env(envIRL)

    envIRL.train_irl()

    # Inizializzazione modello DQN
    model = DQN(
    "MlpPolicy",
    envIRL,
    learning_rate=1e-4,
    buffer_size=20_000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.4,
    exploration_final_eps=0.1,
    verbose=0
    )

    # Addestramento
    model.learn(total_timesteps=250_000, callback=ProgressBarCallback())

    # Salvataggio del modello
    model.save("dqn_hrs_model")

    window_size = 250

    learned_rewards, hydrogen_sold, hydrogen_stored, electricity_sold, loss = envIRL.get_res()

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
    
    plt.show()
    envIRL.close()

if __name__ == "__main__":
    main()