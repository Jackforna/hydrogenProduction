from enum import Enum
from scipy.special import logsumexp
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from electrolyser import Electrolyser
from HSS import hydrogenStorage
from cell import FuelCell

class Actions(Enum):
    PRODUCE: 0                 #produrre idrogeno
    SELL_HYDR: 1               #vendere energia elettrica prodotta dall'idrogeno
    SELL_ELEC: 2               #vendere energia elettrica prodotta direttamente da energia rinnovabile
    BLOCK_PRODUCTION: 3        #bloccare la produzione di idrogeno
    BLOCK_SELL: 4              #bloccare la vendita dell'energia prodotta dall'idrogeno


class HRS_envIRL(gym.Env):

    def __init__(self):
        super(HRS_envIRL, self).__init__()

        self.action_space = spaces.Discrete(5)  #5 azioni discrete perchè rappresentano scelte finite

        #gli stati invece saranno continui poichè rappresentano grandezze variabili nel tempo
        #Stato: quantità idrogeno stoccato, energia prodotta, costo produzione idrogeno, prezzo vendita elettricità, prezzo idrogeno, domanda elettricità, elettrolita in azione, celle in azione
        self.observation_space = spaces.Box(low = np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                                            high = np.array([500, 200, 10, 10, 10, 30, 1, 1]),
                                            dtype = np.float32)

        self.storage = hydrogenStorage(max_capacity=500, pressure=350)
        self.cell = FuelCell(power = 50, efficiency = 0.6, hydrogen_consumption = 1.5, HSS=self.storage, active=True)
        self.electrolyser = Electrolyser(min_power=10, max_power=50, period=10, HSS=self.storage, active=True)
        self.state = np.array([0, 0, 5, 5, 5, 15, self.electrolyser.active, self.cell.active], dtype = np.float32)  #stato iniziale da definire
        self.trajectories = []


    def step(self, action):

        _, energy_produced, production_cost, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on = self.state

        if action == 0:
            if not (self.electrolyser.active):
                self.electrolyser.active = True
            electrolyser_on = 1
            _,_ = self.electrolyser.produceHydrogen()
        elif action == 1:
            if not(self.cell.active):
                self.cell.active = True
            cell_on = 1
            _ = self.cell.generatePower(elec_demand)

        elif action == 2:
            if self.cell.active:
                self.cell.active = False
            cell_on = 0

        elif action == 3:
            if self.electrolyser.active:
                self.electrolyser.active = False
            electrolyser_on = 0
        elif action == 4:
            if self.cell.active:
                self.cell.active = False
            cell_on = 0

        self.state = np.array([self.storage.actual_quantity, energy_produced, production_cost, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on], dtype=np.float32)   #stato aggiornato
        self.trajectories.append((self.state,action))
        
        return self.state, 0, False, False, {}



    def reset(self, *, seed=None, options=None):
        super().reset(seed = seed)
        self.state = np.array([0, 0, 5, 5, 5, 15, 1 ,1], dtype = np.float32)  #stato iniziale da definire
        return self.state,{}


    def render(self):
        print("")

    def get_res(self):
        return self.trajectories

    def expert_policy(self):
        hydrogen, energy_produced, production_cost, elec_price, hydrogen_price, elec_demand, _, _ = self.state
        
        if hydrogen>=500:
            return 3
        else:
            hydrogen_produced, loss = self.electrolyser.produceHydrogen()
            if elec_demand>10:
                self.storage.removeHydrogen(hydrogen_produced)
                return 1
            elif hydrogen<400 and energy_produced>elec_demand:
                self.storage.removeHydrogen(hydrogen_produced)
                return 0
            elif (energy_produced-loss)*elec_price < (hydrogen_produced*hydrogen_price)-production_cost:
                self.storage.removeHydrogen(hydrogen_produced)
                return 4
            else:
                self.storage.removeHydrogen(hydrogen_produced)
                return 2

    def generate_expert_trajectories(self, num_episodes=10):
        expert_trajectories = []
        for _ in range(num_episodes):
            self.state,_ = self.reset()
            episode = []
            for _ in range(50):
                action = self.expert_policy()
                next_state,_,_,_,_ = self.step(action)
                episode.append((self.state,action))
                self.state = next_state

            expert_trajectories.append(episode)
        return expert_trajectories

    def maxent_irl(self, alpha=1.0, iterations=100):
        
        num_states = len(self.observation_space.low)
        expert_trajectories = self.generate_expert_trajectories()

        # 1. Creazione delle frequenze degli stati nelle traiettorie esperte
        state_visits = np.zeros(num_states, dtype=np.float32)
        rewards = np.zeros(iterations, dtype=np.float32)

        # Conta quante volte ogni stato è stato visitato nelle traiettorie esperte
        for traj in expert_trajectories:
            for state, _ in traj:
                state_visits += state  # Somma delle visite degli stati

        # Normalizzazione per evitare divisioni per zero
        if np.sum(state_visits) > 0:
            state_visits /= np.sum(state_visits)

        for i in range(iterations):
            # Calcolo delle probabilità softmax per la policy
            policy = np.exp(alpha * float(rewards[i]) - logsumexp(alpha * rewards))

            # Calcolo delle frequenze degli stati con la policy attuale
            expected_state_visits = np.zeros_like(state_visits)
            
            if policy.shape == state_visits.shape:
                expected_state_visits = policy
            else:
                expected_state_visits.fill(1.0 / num_states)  # Distribuzione uniforme se ci sono problemi

            # Aggiornamento della funzione di reward
            rewards[i] = alpha * (state_visits - expected_state_visits)

        return rewards
