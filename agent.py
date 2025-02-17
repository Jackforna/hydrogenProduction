import numpy as np
from env import HRS_env
#from envIRL import HRS_envIRL
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from icecream import ic             #debugger
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import ProgressBarCallback

def main():
    ic.enable()
    env = HRS_env()
    check_env(env, warn=True)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000000, progress_bar=ProgressBarCallback())
    model.save("ppo_HRS")

    window_size = 1000
    rewards, hydrogen, loss_power, energy_produced, elec_sold = env.get_res()
    
    #envIRL = HRS_envIRL()
    #check_env(envIRL, warn=True)
    #model = PPO("MlpPolicy", envIRL, verbose=1)
    #model.learn(total_timesteps=100000, progress_bar=ProgressBarCallback())
    #model.save("ppo_HRS_IRL")
    #learned_rewards = envIRL.maxent_irl()

    rewards_smooth = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(14,10))
    plt.subplot(5,1,1)
    plt.plot(rewards_smooth)
    plt.title("Rewards")

    hydrogen_smooth = np.convolve(hydrogen, np.ones(window_size)/window_size, mode='valid')

    plt.subplot(5,1,2)
    plt.plot(hydrogen_smooth)
    plt.title("Hydrogen Stored")
    
    #print(learned_rewards)

    #plt.subplot(1,1,1)
    #plt.plot(learned_rewards)
    #plt.title("Learned Rewards")

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
    

    plt.show()
    env.close()
    #envIRL.close()


if __name__ == "__main__":
    main()