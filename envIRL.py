from enum import Enum
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import truncnorm
from scipy.special import logsumexp
import numpy as np
from electrolyser import Electrolyser
from HSS import hydrogenStorage
from cell import FuelCell
import sys

class Actions(Enum):
    PRODUCE = 0                 
    SELL_HYDR = 1              
    SELL_ELEC = 2              
    BLOCK_PRODUCTION = 3       
    BLOCK_SELL = 4             

class HRS_envIRL(gym.Env):

    def __init__(self, bins=5):
        super(HRS_envIRL, self).__init__()

        self.action_space = spaces.MultiDiscrete([2,2,2,2,2])

        # Stati continui da discretizzare
        self.observation_space = spaces.Box(low = np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                                            high = np.array([500, 300, 4, 10, 10, 200, 1, 1]),
                                            dtype = np.float32)

        self.bins = bins  # Numero di bin per la discretizzazione
        self.bin_edges = [np.linspace(low, high, bins) for low, high in zip(self.observation_space.low, self.observation_space.high)]

        self.storage = hydrogenStorage(max_capacity=500, pressure=350)
        self.cell = FuelCell(power = 250, efficiency=1.0, hydrogen_consumption=3, HSS=self.storage, active=True)
        self.electrolyser = Electrolyser(min_power=30, max_power=300, period=10, HSS=self.storage, active=True)
        self.state = np.array([0, 5, 5, 5, 15, self.electrolyser.active, self.cell.active], dtype=np.float32)
        self.trajectories = []
        self.learned_rewards = {}
        self.rew_arr = []
        self.hydr_arr = []
        self.stor_arr = []
        self.elec_arr = []
        self.loss_arr = []
        self.action_arr = []
        self.demand_arr = []
        self.input_arr = []
        self.training = False

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

        if produce == 0 and block_prod == 0:
            return False

        if block_sell == 0 and sell_hydr == 0:
            return False

        return True

    def step(self, action):

        produce, sell_hydr, sell_elec, block_prod, block_sell = action
        if not self.filter_invalid_actions(action):
            return self.state, -100, False, False, {}  # Penalità per azioni non valide

        hydrogen, energy_produced, production_cost, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on = self.state
        self.input_arr.append(energy_produced)
        self.demand_arr.append(elec_demand)
        loss_power = 0
        revenue = 0
        costs = 0
        energy_elec = 0

        #Azione produzione idrogeno e vendita elettricità contemporaneamente
        if produce == 1 and block_prod == 0:
            if sell_elec == 0:
                if sell_hydr == 1:
                    if not self.cell.active:
                        self.cell.active = True
                    cell_on = 1
                    generate_power = self.cell.generatePower(elec_demand)
                    revenue += generate_power * elec_price
                    self.hydr_arr.append(generate_power)
                if not self.electrolyser.active:
                    self.electrolyser.active = True
                electrolyser_on = 1
                hydrogen_produced, loss = self.electrolyser.produceHydrogen(energy_produced)
                loss_power += loss
                revenue += hydrogen_produced * hydrogen_price
                costs += production_cost     
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
                    revenue += generate_power * elec_price
                    self.hydr_arr.append(generate_power)
                    elec_demand -= generate_power
                if energy_produced > elec_demand:
                    loss_power += energy_produced - elec_demand
                    energy_produced -= loss_power
                revenue += energy_produced * elec_price
                hydrogen_produced, loss = self.electrolyser.produceHydrogen(loss_power)
                loss_power = loss
                revenue += hydrogen_produced * hydrogen_price
                costs += production_cost 
                energy_elec += energy_produced

        if sell_hydr == 1 and sell_elec == 1 and produce == 0:
            if not self.cell.active:
                self.cell.active = True
            cell_on = 1
            generate_power = self.cell.generatePower(elec_demand)
            revenue += generate_power * elec_price
            self.hydr_arr.append(generate_power)
            elec_demand -= generate_power
            if energy_produced > elec_demand:
                loss_power += energy_produced - elec_demand
                energy_produced -= loss_power
                energy_elec += energy_produced
            else:
                energy_elec += min(elec_demand,energy_produced)
            revenue += energy_elec * elec_price

        #Azione: Vendita di energia prodotta dall'idrogeno
        if sell_hydr == 1 and produce == 0 and sell_elec == 0:
            if not self.cell.active:
                self.cell.active = True
            cell_on = 1
            generate_power = self.cell.generatePower(elec_demand)
            self.hydr_arr.append(generate_power)
            revenue += generate_power * elec_price
            loss_power = energy_produced

        #Azione: Vendita di energia prodotta da fonti rinnovabili
        if sell_elec == 1 and produce == 0 and sell_hydr == 0:
            if self.cell.active:
                self.cell.active = False
            cell_on = 0
            if energy_produced > elec_demand:
                loss_power += energy_produced - elec_demand
                energy_produced -= loss_power
            revenue += energy_produced * elec_price
            energy_elec += energy_produced

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
            if produce == 0 and sell_elec == 0:
                loss_power = energy_produced

        #reward = revenue - costs - (loss_power * elec_price)
        state_tuple = self.discretize_state(self.state)

        # Recupera la reward appresa dall'IRL, se disponibile
        #reward = reward * self.learned_rewards.get(state_tuple, -0.01)  # Penalità di default se lo stato non ha una reward appresa
        reward = self.learned_rewards.get(state_tuple, -10)
        print(reward)
        self.rew_arr.append(reward)
        self.stor_arr.append(self.storage.actual_quantity)
        self.elec_arr.append(energy_elec)
        self.loss_arr.append(loss_power)
        self.action_arr.append(action)
        mu = 125    # Media
        sigma = 30  # Deviazione standard
        lower, upper = 50, 200  # Limiti

        # Creiamo la distribuzione troncata
        trunc_gauss = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

        # Prendiamo un valore casuale
        elec_demand = trunc_gauss.rvs()
        energy_produced = self.electrolyser.powerSupplied()
        self.state = np.array([self.storage.actual_quantity, energy_produced, production_cost, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on], dtype=np.float32)   #stato aggiornato
        done = False
        truncated = False

        return self.state, float(reward), done, truncated, {}

    def step_IRL(self, action):
        _, energy_produced, production_cost, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on = self.state
        #if self.training:
            #self.input_arr.append(energy_produced)
            #self.demand_arr.append(elec_demand)
        loss_power = 0
        energy_elec = 0
        generate_power = 0

        produce, sell_hydr, sell_elec, block_prod, block_sell = action
        if not self.filter_invalid_actions(action):
            return self.state, -100, False, False, {}

        if produce == 1 and block_prod == 0:
            if sell_elec == 0:
                if sell_hydr == 1:
                    if not self.cell.active:
                        self.cell.active = True
                    cell_on = 1
                    generate_power = self.cell.generatePower(elec_demand)
                if not self.electrolyser.active:
                    self.electrolyser.active = True
                electrolyser_on = 1
                _, loss = self.electrolyser.produceHydrogen(energy_produced)
                loss_power = loss
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
                    loss_power = energy_produced - elec_demand
                    energy_produced -= loss_power
                    _, loss = self.electrolyser.produceHydrogen(loss_power)
                    loss_power = loss
                energy_elec = energy_produced

        if sell_hydr == 1 and sell_elec == 1 and produce == 0:
            if not self.cell.active:
                self.cell.active = True
            cell_on = 1
            generate_power = self.cell.generatePower(elec_demand)
            elec_demand -= generate_power
            if energy_produced > elec_demand:
                loss_power += energy_produced - elec_demand
                energy_produced -= loss_power
                energy_elec += energy_produced
            else:
                energy_elec += min(elec_demand,energy_produced)

        #Azione: Vendita di energia prodotta dall'idrogeno
        if sell_hydr == 1 and produce == 0 and sell_elec == 0:
            if not self.cell.active:
                self.cell.active = True
            cell_on = 1
            generate_power = self.cell.generatePower(elec_demand)

        #Azione: Vendita di energia prodotta da fonti rinnovabili
        if sell_elec == 1 and produce == 0 and sell_hydr == 0:
            if self.cell.active:
                self.cell.active = False
            cell_on = 0
            if energy_produced > elec_demand:
                loss_power = energy_produced - elec_demand
            energy_elec = elec_demand

        #Azione: Bloccare produzione
        if block_prod == 1:
            if self.electrolyser.active:
                self.electrolyser.active = False
            electrolyser_on = 0
            if sell_elec == 0 and produce == 0:
                loss_power = energy_produced

        #Azione: Bloccare vendita
        if block_sell == 1:
            if self.cell.active:
                self.cell.active = False
            cell_on = 0
            if produce == 0 and sell_elec == 0:
                loss_power = energy_produced

        #if self.training:
         #   self.stor_arr.append(self.storage.actual_quantity)
          #  self.elec_arr.append(energy_elec)
           # self.hydr_arr.append(generate_power)
            #self.loss_arr.append(loss_power)
            #self.action_arr.append(action)
        mu = 125    # Media
        sigma = 30  # Deviazione standard
        lower, upper = 50, 200  # Limiti

        # Creiamo la distribuzione troncata
        trunc_gauss = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

        # Prendiamo un valore casuale
        elec_demand = trunc_gauss.rvs()
        energy_produced = self.electrolyser.powerSupplied()

        # Aggiorna lo stato basandosi sull'azione
        self.state = np.array([self.storage.actual_quantity, energy_produced, production_cost, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on], dtype=np.float32)
        # Stato attuale discretizzato
        state_tuple = self.discretize_state(self.state)

        # Recupera la reward appresa dall'IRL, se disponibile
        reward = self.learned_rewards.get(state_tuple, -0.01)  # Penalità di default se lo stato non ha una reward appresa
        print(reward)

        self.trajectories.append((tuple(self.discretize_state(self.state)), action))
        
        #if self.training:
            #self.rew_arr.append(reward)
        done = False
        truncated = False

        return self.state, float(reward), done, truncated, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0, 80, 2, 5, 5, 50, 1 ,1], dtype=np.float32)
        return self.state, {}

    def get_res(self):
        return self.rew_arr, self.hydr_arr, self.stor_arr, self.elec_arr, self.loss_arr, self.action_arr, self.demand_arr, self.input_arr

    def render(self):
        print("")

    def expert_policy(self):
        hydrogen, energy_produced, production_cost, elec_price, hydrogen_price, elec_demand, _, _ = self.state
        
        action = [0, 0, 0, 0, 0]
        #PRODUCE = 0                 
        #SELL_HYDR = 1              
        #SELL_ELEC = 2              
        #BLOCK_PRODUCTION = 3       
        #BLOCK_SELL = 4 
        
        #Se il serbatoio è pieno, interrompi la produzione di idrogeno
        if hydrogen >= 400:
            action[3] = 1
            action[1] = 1
            if elec_demand > hydrogen*3:
                action[2] = 1

        elif hydrogen < 100:
            action[0] = 1
            action[2] = 1
            action[1] = 0
            action[4] = 1        
        #Se la domanda di elettricità è alta, vendi energia dall'idrogeno e dall'energia rinnovabile bloccando la produzione
        elif elec_demand >= energy_produced:
            action[0] = 0
            if hydrogen > 100:
                action[1] = 1
            action[2] = 1
            action[3] = 1
        
        #Se l'idrogeno è sotto il 100% e c'è energia disponibile, vendi e produci idrogeno con l'eccesso
        elif hydrogen < 500 and energy_produced > elec_demand:
            action[0] = 1
            if hydrogen > 100:
                action[1] = 1
                if elec_demand > hydrogen*3:
                    action[2] = 1
            else:
                action[2] = 1
                action[4] = 1

        #Altrimenti, vendi l'energia direttamente
        else:
            action[2] = 1
            action[3] = 1
            action[4] = 1
        
        return (action)

    def generate_expert_trajectories(self, num_episodes=200):
        expert_trajectories = []

        with open("expert_trajectories.txt", "a") as file:
            for i in range(num_episodes):
                file.write(f"Traiettoria esperta n.{i}\n")
                self.state, _ = self.reset()
                self.state[0] = np.random.uniform(0,500)
                self.state[4] = np.random.uniform(50,200)
                episode = []
                for j in range(1000):
                    file.write(f"Episode n.{j}\n")
                    action = self.expert_policy()
                    next_state, _, _, _, _ = self.step_IRL(action)
                    file.write(f"Stato: {self.state} e azione: {action}")
                    episode.append((tuple(self.discretize_state(self.state)), action))
                    self.state = next_state
                expert_trajectories.append(episode)
        return expert_trajectories

    def train_irl(self, num_episodes=200, iterations=2000, alpha=1):
        expert_trajectories = self.generate_expert_trajectories(num_episodes)
        expert_trajectories.append(self.trajectories)
        self.maxent_irl(expert_trajectories, iterations, alpha)

    def maxent_irl(self, expert_trajectories, iterations=2000, alpha=1):

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
        rewards = np.random.uniform(-0.1, 0.1, num_states)
        policy = np.zeros(num_states, dtype=np.float32)

        # Conta le visite agli stati
        for traj in expert_trajectories:
            for state, _ in traj:
                state_tuple = tuple(state)  # Converti lo stato in una tupla
                state_visits[state_indices[state_tuple]] += 1


        # Normalizzazione per evitare divisioni per zero
        if np.sum(state_visits) > 0:
            state_visits /= (np.sum(state_visits))
            
        with open("state_visits.txt", "w") as file:
            pass

        for _ in range(iterations):
            policy = np.exp(rewards - logsumexp(rewards))  #distribuzione di probabilità sulle traiettorie
            policy /= (np.sum(policy)+1e-6)

            expected_state_visits = np.zeros_like(state_visits)
            for state_idx in range(num_states):
                if state_idx < len(policy):
                    expected_state_visits[state_idx] = policy[state_idx]  
                else:
                    expected_state_visits[state_idx] = policy.mean()    #non dovrebbe entrarci

            for state_idx in range(num_states):
                rewards[state_idx] += alpha * (state_visits[state_idx] - expected_state_visits[state_idx])/ (state_visits[state_idx] + 1e-6)

            

            with open("state_visits.txt", "a") as file:
                file.write(f"policy {policy}\n")
                file.write(f"state_visits {state_visits}\n")
                file.write(f"expected_state_visits {expected_state_visits}\n")
                file.write(f"rewards {rewards}\n\n")

        for i in range(len(rewards)):
            if rewards[i]<-10:
                rewards[i] = -10

        for i in range(len(rewards)):
            rewards[i] = (rewards[i] - np.min(rewards) + 1e-5)
            #rewards[i] = rewards[i] * 100

        self.learned_rewards = {state: rewards[idx] for state, idx in state_indices.items()}