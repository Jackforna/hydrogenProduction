from enum import Enum
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import truncnorm
import numpy as np
from electrolyser import Electrolyser
from HSS import hydrogenStorage
from cell import FuelCell

class Actions(Enum):
    PRODUCE = 0                 
    SELL_HYDR = 1              
    SELL_ELEC = 2              
    BLOCK_PRODUCTION = 3       
    BLOCK_SELL = 4             

class HRS_envIRL(gym.Env):

    def __init__(self, bins=10):
        super(HRS_envIRL, self).__init__()

        self.action_space = spaces.MultiDiscrete([2,2,2,2,2])

        # Stati continui da discretizzare
        self.observation_space = spaces.Box(low = np.array([0, 0, 0, 0, 0, 0, 0]),
                                            high = np.array([500, 4, 10, 10, 200, 1, 1]),
                                            dtype = np.float32)

        self.bins = bins  # Numero di bin per la discretizzazione
        self.bin_edges = [np.linspace(low, high, bins) for low, high in zip(self.observation_space.low, self.observation_space.high)]

        self.storage = hydrogenStorage(max_capacity=500, pressure=350)
        self.cell = FuelCell(power=50, efficiency=0.6, hydrogen_consumption=1.5, HSS=self.storage, active=True)
        self.electrolyser = Electrolyser(min_power=10, max_power=50, period=10, HSS=self.storage, active=True)
        self.state = np.array([0, 5, 5, 5, 15, self.electrolyser.active, self.cell.active], dtype=np.float32)
        self.trajectories = []
        self.learned_rewards = {}
        self.rew_arr = []

    def discretize_state(self, state):
        """ Trasforma uno stato continuo in uno stato discreto usando binning """
        return tuple(np.digitize(state[i], self.bin_edges[i]) for i in range(len(state)))

    def filter_invalid_actions(self, action):
        #Elimina combinazioni di azioni non valide
        produce, sell_hydr, _, block_prod, block_sell = action
        
        # Non puoi bloccare e produrre allo stesso tempo
        if produce == 1 and block_prod == 1:
            return False
        
        # Non puoi bloccare la vendita se non stai vendendo
        if block_sell == 1 and sell_hydr == 1:
            return False

        return True

    def step(self, action):
        _, production_cost, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on = self.state
        energy_produced = self.electrolyser.powerSupplied()
        loss_power = 0

        produce, sell_hydr, sell_elec, block_prod, block_sell = action
        if not self.filter_invalid_actions(action):
            return self.state, -100, False, False, {}

        if produce == 1 and block_prod == 0:
            if sell_elec == 0:
                if sell_hydr == 1:
                    if not self.cell.active:
                        self.cell.active = True
                    cell_on = 1
                    _ = self.cell.generatePower(elec_demand)
                if not self.electrolyser.active:
                    self.electrolyser.active = True
                electrolyser_on = 1
                _,_ = self.electrolyser.produceHydrogen(energy_produced) 
            else:
                if not self.electrolyser.active:
                    self.electrolyser.active = True
                electrolyser_on = 1
                if sell_hydr == 0:
                    if self.cell.active:
                        self.cell.active = False
                    cell_on = 0
                else:
                    if not self.cell.active:
                        self.cell.active = True
                    cell_on = 1
                    generate_power = self.cell.generatePower(elec_demand)
                    elec_demand -= generate_power
                if energy_produced > elec_demand:
                    loss_power += energy_produced - elec_demand
                _,_ = self.electrolyser.produceHydrogen(loss_power)

        if sell_hydr == 1 and sell_elec == 1 and produce == 0:
            if not self.cell.active:
                self.cell.active = True
            cell_on = 1
            _ = self.cell.generatePower(elec_demand)

        #Azione: Vendita di energia prodotta dall'idrogeno
        if sell_hydr == 1 and produce == 0 and sell_elec == 0:
            if not self.cell.active:
                self.cell.active = True
            cell_on = 1
            _ = self.cell.generatePower(elec_demand)

        #Azione: Vendita di energia prodotta da fonti rinnovabili
        if sell_elec == 1 and produce == 0 and sell_hydr == 0:
            if self.cell.active:
                self.cell.active = False
            cell_on = 0

        #Azione: Bloccare produzione
        if block_prod == 1:
            if self.electrolyser.active:
                self.electrolyser.active = False
            electrolyser_on = 0

        #Azione: Bloccare vendita
        if block_sell == 1:
            if self.cell.active:
                self.cell.active = False
            cell_on = 0

        self.trajectories.append((self.discretize_state(self.state), action))
        # Stato attuale discretizzato
        state_tuple = self.discretize_state(self.state)

        # Recupera la reward appresa dall'IRL, se disponibile
        reward = self.learned_rewards.get(state_tuple, -0.01)  # Penalità di default se lo stato non ha una reward appresa
        self.rew_arr.append(reward)
        mu = 125    # Media
        sigma = 30  # Deviazione standard
        lower, upper = 50, 200  # Limiti

        # Creiamo la distribuzione troncata
        trunc_gauss = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

        # Prendiamo un valore casuale
        elec_demand = trunc_gauss.rvs()

        # Aggiorna lo stato basandosi sull'azione
        self.state = np.array([self.storage.actual_quantity, production_cost, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on], dtype=np.float32)
        done = False
        truncated = False

        return self.state, float(reward), done, truncated, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0, 2, 5, 5, 50, 1 ,1], dtype=np.float32)
        return self.state, {}

    def get_res(self):
        return self.rew_arr

    def render(self):
        print("")

    def expert_policy(self):
        hydrogen, production_cost, elec_price, hydrogen_price, elec_demand, _, _ = self.state
        energy_produced = self.electrolyser.powerSupplied()
        
        action = [0, 0, 0, 0, 0]
        #PRODUCE = 0                 
        #SELL_HYDR = 1              
        #SELL_ELEC = 2              
        #BLOCK_PRODUCTION = 3       
        #BLOCK_SELL = 4 
        
        # Se il serbatoio è pieno, interrompi la produzione di idrogeno
        if hydrogen >= 500:
            action[3] = 1  # Blocca la produzione
        
        # Se la domanda di elettricità è alta, vendi energia dall'idrogeno
        if elec_demand > energy_produced:
            action[1] = 1
            action[2] = 1
        
        # Se l'idrogeno è sotto il 100% e c'è energia disponibile, produci idrogeno
        if hydrogen < 500 and energy_produced > elec_demand:
            action[0] = 1
        
        # Se la vendita dell'idrogeno è più redditizia della vendita dell'energia diretta, scegli quella
        if (energy_produced * elec_price) < (hydrogen * hydrogen_price) - production_cost:
            action[1] = 1
        
        # Altrimenti, vendi l'energia direttamente
        if elec_demand > 0:
            action[2] = 1
        
        return (action)

    def generate_expert_trajectories(self, num_episodes=50):
        expert_trajectories = []
        for _ in range(num_episodes):
            self.state, _ = self.reset()
            episode = []
            for _ in range(50):
                action = self.expert_policy()
                next_state, _, _, _, _ = self.step(action)
                episode.append((tuple(self.discretize_state(self.state)), action))
                self.state = next_state
            expert_trajectories.append(episode)
        return expert_trajectories

    def maxent_irl(self, iterations=100, alpha=0.1):
        expert_trajectories = self.generate_expert_trajectories()
        gamma = 0.9

        # Creazione della mappa dinamica per stati multidiscreti
        state_indices = {}
        index = 0
        for traj in expert_trajectories:
            for state, _ in traj:
                state_tuple = tuple(state)  # Converti lo stato in una tupla
                if state_tuple not in state_indices:
                    state_indices[state_tuple] = index
                    index += 1
        
        num_states = len(state_indices)
        state_visits = np.zeros(num_states, dtype=np.float32)
        rewards = np.random.uniform(1, 1, num_states)

        # Conta le visite agli stati
        for traj in expert_trajectories:
            for state, _ in traj:
                state_tuple = tuple(state)  # Converti lo stato in una tupla
                state_visits[state_indices[state_tuple]] += 1

        # Normalizzazione per evitare divisioni per zero
        if np.sum(state_visits) > 0:
            state_visits /= (np.sum(state_visits)+1e-6)

        for _ in range(iterations):
            policy = np.exp(alpha * (rewards - np.max(rewards)))
            policy /= (policy.sum()+1e-6)

            expected_state_visits = np.zeros_like(state_visits)
            for state_idx in range(num_states):
                expected_state_visits[state_idx] = policy[state_idx] if state_idx < len(policy) else policy.mean()

            for state_idx in range(num_states):
                rewards[state_idx] += gamma * rewards[state_idx] + alpha * (state_visits[state_idx] - expected_state_visits[state_idx]) / (state_visits[state_idx] + 1e-6)

        #rewards = np.clip(rewards, -10, 100)
        self.learned_rewards = {state: rewards[idx] for state, idx in state_indices.items()}

        return rewards