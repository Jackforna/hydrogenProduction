import numpy as np
from env import HRS_env
from envIRL import HRS_envIRL
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
#from icecream import ic             #debugger
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import ProgressBarCallback

def main():
    #ic.enable()
    #env = HRS_env()
    #check_env(env, warn=True)
    #model = PPO("MlpPolicy", env, verbose=1)
    #model.learn(total_timesteps=100000, progress_bar=ProgressBarCallback())
    #model.save("ppo_HRS")

    window_size = 100
    #rewards, hydrogen, loss_power, energy_produced, elec_sold = env.get_res()
    
    envIRL = HRS_envIRL()
    check_env(envIRL, warn=True)
    
    '''
    for i in range(100):  # 100 iterazioni di addestramento
        model.learn(total_timesteps=2000, progress_bar=ProgressBarCallback())  # Addestra il PPO per 2000 passi
        
        if i % 5 == 0:  # Ogni 5 iterazioni di PPO, aggiorna l'IRL
            print(f"Training IRL at iteration {i}...")
            envIRL.train_irl(num_episodes=50)
    '''
    
    envIRL.train_irl()
    envIRL.training = True

    model = PPO("MlpPolicy", envIRL, verbose=0)
    model.learn(total_timesteps=100000, progress_bar=ProgressBarCallback())
    model.save("ppo_HRS_IRL")

    learned_rewards, hydrogen_sold, hydrogen_stored, electricity_sold, loss, action, demand, input_power = envIRL.get_res()

    with open("passi_IRL.txt", "w") as file:
        pass

    with open("passi_IRL.txt", "a") as file:
        for i in range(len(learned_rewards)):
            file.write(f"Passo n.{i}\n")
            file.write(f"Reward: {learned_rewards[i]}\n")
            #file.write(f"Idrogeno venduto: {hydrogen_sold[i]}\n")
            file.write(f"Idrogeno stoccato: {hydrogen_stored[i]}\n")
            file.write(f"Elettricità venduta: {electricity_sold[i]}\n")
            file.write(f"Potenza persa: {loss[i]}\n")
            file.write(f"Domanda: {demand[i]}\n")
            file.write(f"Energia prodotta: {input_power[i]}\n")
            file.write(f"Azioni: {action[i][0]}, {action[i][1]}, {action[i][2]}, {action[i][3]}, {action[i][4]}\n\n")
    '''
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

    loss_smooth = np.convolve(loss_power, np.ones(window_size)/window_size, mode='valid')
    plt.subplot(5,1,3)
    plt.plot(loss_smooth)
    plt.title("Loss Power")

    energy_smooth = np.convolve(energy_produced, np.ones(window_size)/window_size, mode='valid')
    plt.subplot(5,1,4)
    plt.plot(energy_smooth)
    plt.title("Hydrogen sold")

    elec_smooth = np.convolve(elec_sold, np.ones(window_size)/window_size, mode='valid')
    plt.subplot(5,1,5)
    plt.plot(elec_smooth)
    plt.title("Electricity sold")
    '''
    learned_smooth = np.convolve(learned_rewards, np.ones(window_size)/window_size, mode='valid')

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
    #env.close()
    envIRL.close()
    

if __name__ == "__main__":
    main()