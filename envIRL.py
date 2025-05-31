from enum import Enum
import pandas as pd
import os
import ast
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import truncnorm
from scipy.special import logsumexp
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

    def __init__(self):
        super(HRS_envIRL, self).__init__()

        self.action_space = spaces.MultiDiscrete([2,2,2,2,2])

        # Stati continui da discretizzare
        self.observation_space = spaces.Box(low = np.array([0, 0, 0, 0, 0, 0, 0]),
                                            high = np.array([1000, 3200, 10, 10, 3200, 1, 1]),
                                            dtype = np.float32)

        self.num_bins = [100, 15, 5, 5, 15, 2, 2]

        self.bin_edges = [
            np.linspace(low, high, num_bins, endpoint=False)if num_bins > 2 else None
            for low, high, num_bins in zip(self.observation_space.low, self.observation_space.high, self.num_bins)
        ]


        self.storage = hydrogenStorage(max_capacity=1000)
        self.cell = FuelCell(power = 3200, efficiency=0.8, hydrogen_consumption=3, HSS=self.storage, active=True)
        self.electrolyser = Electrolyser(min_power=30, max_power=3200, period=10, HSS=self.storage, active=True)
        self.state = np.array([0, 80, 5, 5, 50, 1, 1], dtype=np.float32)
        self.learned_rewards = {}
        self.rew_arr = []
        self.hydr_arr = []
        self.stor_arr = []
        self.elec_arr = []
        self.loss_arr = []
        self.action_arr = []
        self.demand_arr = []
        self.demand_remained_arr = []
        self.input_arr = []
        self.states_traj = []
        self.loaded_trajectories = self.load_expert_trajectories()
        self.len_episodes = 5000
        self.demand_remained = 0
        self.loss_remained = 0

    def load_expert_trajectories(self, filename="expert_trajectories.csv"):
    
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            return []  # Se il file non esiste, restituisce una lista vuota
        
        df = pd.read_csv(filename)

        expert_trajectories = []
        episode = []
        step_count = 0

        for _, row in df.iterrows():
            state = ast.literal_eval(row["hydrogen_bin"])
            action = list(map(int, row["action"].split(",")))   # Converti l'azione in lista di interi
            episode.append((state, action))
            step_count += 1

            if step_count == 5000:
                expert_trajectories.append(episode)
                episode = []
                step_count = 0


        return expert_trajectories

    def discretize_state(self, state, demand_remained, loss_power):
        state = min(state, 1000)
        return tuple([int(np.digitize(state, self.bin_edges[0])),demand_remained, loss_power])

    def filter_invalid_actions(self, action):

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

        hydrogen, energy_produced, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on = self.state
        self.input_arr.append(energy_produced)
        self.demand_arr.append(elec_demand)
        loss_power = 0
        demand = elec_demand
        revenue = 0
        energy_elec = 0
        generate_power = 0
        elec_price /= 10
        hydrogen_price /= 33.33

        #Azione produzione idrogeno e vendita elettricità contemporaneamente
        if produce == 1 and block_prod == 0:
            if sell_elec == 0:
                if sell_hydr == 1:
                    if not self.cell.active:
                        self.cell.active = True
                    cell_on = 1
                    generate_power = self.cell.generatePower(elec_demand)
                    revenue += generate_power * elec_price
                if not self.electrolyser.active:
                    self.electrolyser.active = True
                electrolyser_on = 1
                self.electrolyser.produceHydrogen(energy_produced)
                loss_power = 0
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
                    elec_demand -= generate_power
                if energy_produced > elec_demand:
                    loss_power += energy_produced - elec_demand
                    energy_produced -= loss_power
                    self.electrolyser.produceHydrogen(loss_power)
                    loss_power = 0
                revenue += energy_produced * elec_price
                energy_elec += energy_produced

        if sell_hydr == 1 and sell_elec == 1 and produce == 0:
            if not self.cell.active:
                self.cell.active = True
            cell_on = 1
            if hydrogen_price < elec_price or hydrogen > self.storage.max_capacity * 9/10:
                generate_power = self.cell.generatePower(elec_demand)
                revenue += generate_power * elec_price
                elec_demand -= generate_power
                if energy_produced > elec_demand:
                    loss_power += energy_produced - elec_demand
                    energy_produced -= loss_power
                    energy_elec += energy_produced
                else:
                    energy_elec += min(elec_demand,energy_produced)
            else:
                energy_elec += min(elec_demand, energy_produced)
                elec_demand -= energy_elec
                generate_power = self.cell.generatePower(elec_demand)
                revenue += generate_power * elec_price
            revenue += energy_elec * elec_price

        #Azione: Vendita di energia prodotta dall'idrogeno
        if sell_hydr == 1 and produce == 0 and sell_elec == 0:
            if not self.cell.active:
                self.cell.active = True
            cell_on = 1
            generate_power = self.cell.generatePower(elec_demand)
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

        if energy_elec + generate_power < demand and energy_produced > demand:
            demand -= energy_elec + generate_power
        else:
            demand = 0

        if self.storage.actual_quantity > self.storage.max_capacity * 9/10:
            state_tuple = self.discretize_state(450, demand, loss_power)
        else:
            state_tuple = self.discretize_state(self.storage.actual_quantity, demand, loss_power)

        # Recupera la reward appresa dall'IRL, se disponibile
        
        reward = self.learned_rewards.get((state_tuple), -10)
        self.rew_arr.append(reward)
        self.stor_arr.append(self.storage.actual_quantity)
        self.elec_arr.append(energy_elec)
        self.loss_arr.append(loss_power)
        self.action_arr.append(action)
        self.hydr_arr.append(generate_power)
        self.demand_remained_arr.append(demand)
        
        
        lower, upper = 50, 3200  # Limiti
        mu = (lower + upper) / 2
        sigma = (upper - lower) / 2

        # Creiamo la distribuzione troncata
        trunc_gauss = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

        # Prendiamo un valore casuale
        elec_demand = trunc_gauss.rvs()
        energy_produced = self.electrolyser.powerSupplied()
        hydrogen_price = float(np.random.uniform(5,10))
        elec_price = float(np.random.uniform(1,5))
        self.state = np.array([self.storage.actual_quantity, float(energy_produced), elec_price, hydrogen_price, float(elec_demand), electrolyser_on, cell_on], dtype=np.float32)   #stato aggiornato
        done = False
        truncated = False

        return self.state, float(reward), done, truncated, {}

    def step_IRL(self, action):
        hydrogen, energy_produced, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on = self.state
        self.storage.actual_quantity = hydrogen
    
        loss_power = 0
        energy_elec = 0
        generate_power = 0
        elec_price /= 10
        hydrogen_price /= 33.33
        demand = elec_demand

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
                self.electrolyser.produceHydrogen(energy_produced)
                loss_power = 0
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
                    self.electrolyser.produceHydrogen(loss_power)
                    loss_power = 0
                energy_elec = energy_produced

        if sell_hydr == 1 and sell_elec == 1 and produce == 0:
            if not self.cell.active:
                self.cell.active = True
            cell_on = 1
            if hydrogen_price < elec_price or hydrogen > self.storage.max_capacity * 9/10:
                generate_power = self.cell.generatePower(elec_demand)
                elec_demand -= generate_power
                if energy_produced > elec_demand:
                    loss_power += energy_produced - elec_demand
                    energy_produced -= loss_power
                    energy_elec += energy_produced
                else:
                    energy_elec += min(elec_demand,energy_produced)
            else:
                energy_elec += min(elec_demand, energy_produced)
                elec_demand -= energy_elec
                generate_power = self.cell.generatePower(elec_demand)

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

        if energy_elec + generate_power < demand and energy_produced > demand:
            demand -= energy_elec + generate_power
        else:
            demand = 0

        lower, upper = 50, 3200  # Limiti
        mu = (lower + upper) / 2
        sigma = (upper - lower) / 2

        # Creiamo la distribuzione troncata
        trunc_gauss = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

        # Prendiamo un valore casuale
        elec_demand = trunc_gauss.rvs()
        energy_produced = self.electrolyser.powerSupplied()
        hydrogen_price = float(np.random.uniform(5,10))
        elec_price = float(np.random.uniform(1,5))

        # Aggiorna lo stato basandosi sull'azione
        self.state = np.array([self.storage.actual_quantity, float(energy_produced), elec_price, hydrogen_price, float(elec_demand), electrolyser_on, cell_on], dtype=np.float32)
        # Stato attuale discretizzato
        state_tuple = self.discretize_state(self.storage.actual_quantity, demand, loss_power)

        # Recupera la reward appresa dall'IRL, se disponibile
        reward = self.learned_rewards.get((state_tuple), -0.01)  # Penalità di default se lo stato non ha una reward appresa
        
        self.loss_remained = loss_power
        self.demand_remained = demand
        
        done = False
        truncated = False

        return self.state, float(reward), done, truncated, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0, 80, 5, 5, 50, 1, 1], dtype=np.float32)
        return self.state, {}

    def get_res(self):
        return self.rew_arr, self.hydr_arr, self.stor_arr, self.elec_arr, self.loss_arr, self.demand_remained_arr

    def render(self):
        print("")

    def expert_policy(self):
        hydrogen, energy_produced, _, _, elec_demand, _, _ = self.state
        
        action = [0, 0, 0, 0, 0]
        #PRODUCE = 0                 
        #SELL_HYDR = 1              
        #SELL_ELEC = 2              
        #BLOCK_PRODUCTION = 3       
        #BLOCK_SELL = 4 
        
        #Se il serbatoio è pieno, interrompi la produzione di idrogeno
        if hydrogen >= self.storage.max_capacity * 9/10:
            action[1] = 1
            if elec_demand > hydrogen*self.cell.hydrogen_consumption*self.cell.efficiency:
                action[2] = 1
                action[3] = 1
            elif energy_produced > 0:
                action[0] = 1

        elif hydrogen < self.storage.max_capacity * 1/10:
            action[0] = 1
            action[2] = 1
            action[4] = 1
        #Se la domanda di elettricità è alta, vendi energia dall'idrogeno e dall'energia rinnovabile bloccando la produzione
        elif elec_demand >= energy_produced:
            action[1] = 1
            if elec_demand >= hydrogen*self.cell.hydrogen_consumption*self.cell.efficiency:
                if elec_demand >= hydrogen*self.cell.hydrogen_consumption*self.cell.efficiency + energy_produced:
                    action[2] = 1
                    action[3] = 1
                else:
                    action[0] = 1
            else:
                action[0] = 1
        
        #Se l'idrogeno è sotto il 100% e c'è energia disponibile, vendi e produci idrogeno con l'eccesso
        elif hydrogen < self.storage.max_capacity and energy_produced > elec_demand:
            action[0] = 1
            #if hydrogen > self.storage.max_capacity * 1/10 and hydrogen_price < elec_price:
            if hydrogen > self.storage.max_capacity * 1/10:
                action[1] = 1
                if elec_demand > hydrogen*self.cell.hydrogen_consumption*self.cell.efficiency:
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

    def save_expert_trajectories(self, expert_trajectories, filename="expert_trajectories.csv"):

        data = []
        
        for episode in expert_trajectories:
            for item in episode:
                if isinstance(item, tuple) and len(item) == 2:
                    state, action = item  # Estrai i valori se il formato è corretto
                else:
                    print(f"Errore: Formato errato per item {item}")  # Debug
                    continue  # Salta elementi malformattati

                state_str = str(state)  # Concatena lo stato come stringa
                action_str = ",".join(map(str, action))  # Concatena l'azione come stringa
                data.append([state_str, action_str])

        # Se ci sono dati validi, salva il file
        if data:
            df = pd.DataFrame(data, columns=["hydrogen_bin", "action"])

            # Se il file esiste e ha dati, aggiungi senza sovrascrivere
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                df_existing = pd.read_csv(filename)
                df = pd.concat([df_existing, df], ignore_index=True)

            df.to_csv(filename, index=False)
        else:
            print("Nessuna traiettoria esperta trovata per il salvataggio.")


    def generate_expert_trajectories(self, num_episodes=300):
        expert_trajectories = []
        
        for i in range(num_episodes):
            print(f"\nTraiettoria esperta n.{i}\n")
            self.state, _ = self.reset()
            self.state[0] = np.random.uniform(100,900)
            a, b = self.electrolyser.min_power, self.electrolyser.max_power
            mean = (a+b)/2
            std_dev = (b-a)/2
            self.state[1] = np.random.normal(loc = mean, scale = std_dev)
            self.state[1] =  np.clip(float(self.state[1]), self.electrolyser.min_power, self.electrolyser.max_power)
            a, b = 50, 3200
            mean = (a+b)/2
            std_dev = (b-a)/2
            self.state[4] = np.random.normal(loc = mean, scale = std_dev)
            self.state[4] =  np.clip(float(self.state[4]), 50, 3200)

            episode = []
            summ = 0
            n = 0
            for _ in range(self.len_episodes):
                action = self.expert_policy()
                next_state, _, _, _, _ = self.step_IRL(action)
                self.state = next_state
                n += 1
                summ += self.state[0]
                if n == 10:
                    episode.append((self.discretize_state(summ/10, int(self.demand_remained), int(self.loss_remained)), action))
                    n = 0
                    summ = 0
            expert_trajectories.append(episode)
        
        return expert_trajectories

    def train_irl(self, num_episodes=300, iterations=500, alpha=0.1):
        expert_trajectories = self.generate_expert_trajectories(num_episodes)
        self.save_expert_trajectories(expert_trajectories)
        expert_trajectories += self.loaded_trajectories
        #expert_trajectories = self.loaded_trajectories

        self.maxent_irl(expert_trajectories, iterations, alpha)

    def maxent_irl(self, expert_trajectories, iterations=500, alpha=0.1):

        #Mappa (solo idrogeno) - indice
        state_indices = {}
        index = 0

        for traj in expert_trajectories:
            for state, _ in traj:
                hydrogen_level, loss_power, demand_remained = state
                full_state = (hydrogen_level, loss_power, demand_remained)
                if full_state not in state_indices:
                    state_indices[full_state] = index
                    index += 1

        num_states = len(state_indices)
        state_visits = np.zeros(num_states, dtype=np.float32)
        rewards = np.random.uniform(-0.1, 0.1, num_states)
        policy = np.zeros(num_states, dtype=np.float32)

        # Conta le visite (h, a)
        for traj in expert_trajectories:
            for state, _ in traj:
                hydrogen_level, loss_power, demand_remained = state
                full_state = (hydrogen_level, loss_power, demand_remained)
                state_visits[state_indices[full_state]] += 1

        state_visits /= np.sum(state_visits) + 1e-6

        for _ in range(iterations):
            policy = np.exp(rewards - logsumexp(rewards))  #distribuzione di probabilità sulle traiettorie
            policy /= (np.sum(policy)+1e-6)

            expected_state_visits = np.zeros_like(state_visits)
            for state_idx in range(num_states):
                if state_idx < len(policy):
                    expected_state_visits[state_idx] = policy[state_idx]
                else:
                    expected_state_visits[state_idx] = policy.mean()

            for state_idx in range(num_states):
                rewards[state_idx] += alpha * (state_visits[state_idx] - expected_state_visits[state_idx])/ (state_visits[state_idx] + 1e-6)

        rewards = np.clip(rewards, -10, None)
        rewards = rewards - np.min(rewards) + 1e-5

        grouped_rewards = {}
        group_size = 10

        # Raggruppa gli idx in blocchi da 10
        for i in range(0, max(state_indices.values()) + 1, group_size):
            group = [j for j in range(i, i + group_size) if j < len(rewards)]
            avg = np.mean([rewards[j] for j in group])
            for j in group:
                grouped_rewards[j] = avg

        grouped_rewards[0] -= 5

        # Mappatura: stato - reward
        self.learned_rewards = {
            (state): grouped_rewards.get(idx, -10)  #-10 se idx non esiste
            for state, idx in state_indices.items()
        }

        range_to_reward = {}
        for state, idx in state_indices.items():
            group_id = idx // group_size
            group_start_idx = group_id * group_size
            range_to_reward[group_id] = grouped_rewards.get(group_start_idx, -10) #-10 se group_start_idx non esiste

        # Creazione CSV
        df = pd.DataFrame({
            "state_range": [
                f"{g*10*group_size}-{min((g+1)*10*group_size, 1000)}"
                for g in range_to_reward.keys()
                if g*10*group_size < 1000
            ],
            "reward": [
                r for g, r in range_to_reward.items()
                if g*10*group_size < 1000
            ]
        })
        # Salva in un file CSV
        df.to_csv("learned_rewards.csv", index=False)