from enum import Enum
from scipy.special import logsumexp
import gymnasium as gym
from gymnasium import spaces
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

        self.action_space = spaces.Discrete(5)

        # Stati continui da discretizzare
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([500, 200, 10, 10, 10, 30, 1, 1]),
            dtype=np.float32
        )

        self.bins = bins  # Numero di bin per la discretizzazione
        self.bin_edges = [np.linspace(low, high, bins) for low, high in zip(self.observation_space.low, self.observation_space.high)]

        self.storage = hydrogenStorage(max_capacity=500, pressure=350)
        self.cell = FuelCell(power=50, efficiency=0.6, hydrogen_consumption=1.5, HSS=self.storage, active=True)
        self.electrolyser = Electrolyser(min_power=10, max_power=50, period=10, HSS=self.storage, active=True)
        self.state = np.array([0, 0, 5, 5, 5, 15, self.electrolyser.active, self.cell.active], dtype=np.float32)
        self.trajectories = []

    def discretize_state(self, state):
        """ Trasforma uno stato continuo in uno stato discreto usando binning """
        return tuple(np.digitize(state[i], self.bin_edges[i]) for i in range(len(state)))

    def step(self, action):
        _, energy_produced, production_cost, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on = self.state

        if action == 0:
            if not self.electrolyser.active:
                self.electrolyser.active = True
            electrolyser_on = 1
            _, _ = self.electrolyser.produceHydrogen()
        elif action == 1:
            if not self.cell.active:
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

        self.state = np.array([self.storage.actual_quantity, energy_produced, production_cost, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on], dtype=np.float32)
        self.trajectories.append((self.discretize_state(self.state), action))

        return self.state, 0, False, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0, 0, 5, 5, 5, 15, 1, 1], dtype=np.float32)
        return self.state, {}

    def get_res(self):
        return self.trajectories

    def expert_policy(self):
        
        hydrogen, energy_produced, production_cost, elec_price, hydrogen_price, elec_demand, _, _ = self.state
        
        # Se il serbatoio è pieno, interrompi la produzione di idrogeno
        if hydrogen >= 500:
            return Actions.BLOCK_PRODUCTION.value

        # Se la domanda di elettricità è alta, vendi energia dall'idrogeno
        elif elec_demand > 10:
            return Actions.SELL_HYDR.value

        # Se l'idrogeno è sotto il 80% e c'è energia disponibile, produci idrogeno
        elif hydrogen < 400 and energy_produced > elec_demand:
            return Actions.PRODUCE.value

        # Se la vendita dell'idrogeno è più redditizia della vendita dell'energia diretta, scegli quella
        elif (energy_produced * elec_price) < (hydrogen * hydrogen_price) - production_cost:
            return Actions.SELL_HYDR.value

        # Altrimenti, vendi l'energia direttamente
        else:
            return Actions.SELL_ELEC.value

    def generate_expert_trajectories(self, num_episodes=10):
        expert_trajectories = []
        for _ in range(num_episodes):
            self.state, _ = self.reset()
            episode = []
            for _ in range(50):
                action = self.expert_policy()
                next_state, _, _, _, _ = self.step(action)
                episode.append((self.discretize_state(self.state), action))
                self.state = next_state

            expert_trajectories.append(episode)
        return expert_trajectories

    def maxent_irl(self, alpha=2.0, iterations=200):
        expert_trajectories = self.generate_expert_trajectories()

        # Creiamo una mappa dinamica per assegnare un indice a ogni stato discretizzato
        state_indices = {}
        index = 0
        for traj in expert_trajectories:
            for state, _ in traj:
                if state not in state_indices:
                    state_indices[state] = index
                    index += 1

        num_states = len(state_indices)
        state_visits = np.zeros(num_states, dtype=np.float32)
        rewards = np.random.uniform(-0.01, 0.01, num_states)

        # Conta le visite agli stati
        for traj in expert_trajectories:
            for state, _ in traj:
                state_visits[state_indices[state]] += 1

        # Normalizzazione per evitare divisioni per zero
        if np.sum(state_visits) > 0:
            state_visits /= np.sum(state_visits)

        for _ in range(iterations):
            policy = np.exp(alpha * rewards - logsumexp(alpha * rewards))
            policy /= policy.sum()

            expected_state_visits = np.zeros_like(state_visits)
            for state_idx in range(num_states):
                expected_state_visits[state_idx] = policy[state_idx] if state_idx < len(policy) else policy.mean()

            for state_idx in range(num_states):
                rewards[state_idx] += alpha * (state_visits[state_idx] - expected_state_visits[state_idx]) / (state_visits[state_idx] + 1e-6)

        return rewards
