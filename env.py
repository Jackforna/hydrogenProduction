from enum import Enum
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


class HRS_env(gym.Env):

    def __init__(self):
        super(HRS_env, self).__init__()

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
        self.rew_arr = []
        self.stor_arr = []
        self.loss_arr = []


    def step(self, action):

        hydrogen, energy_produced, production_cost, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on = self.state
        loss_power = 0
        revenue = 0
        costs = 0

        if action == 0:
            if not (self.electrolyser.active):
                self.electrolyser.active = True
            electrolyser_on = 1
            hydrogen_produced, loss = self.electrolyser.produceHydrogen()
            loss_power += loss
            revenue += hydrogen_produced * hydrogen_price
            costs = production_cost
        elif action == 1:
            if not(self.cell.active):
                self.cell.active = True
            cell_on = 1
            generate_power = self.cell.generatePower(elec_demand)
            revenue += generate_power * elec_price
            
            #if generate_power > 0:
            #    revenue += generate_power * elec_price
            #else:
            #    revenue -= 100 #non viene soddisfatta la richiesta energetica

        elif action == 2:
            if self.cell.active:
                self.cell.active = False
            cell_on = 0
            if energy_produced > elec_demand:
                loss_power += energy_produced - elec_demand
            revenue += energy_produced * elec_price

        elif action == 3:
            if self.electrolyser.active:
                self.electrolyser.active = False
            electrolyser_on = 0
            if (hydrogen < self.storage.max_capacity) and (energy_produced > elec_demand):
                loss_power += energy_produced - elec_demand
        elif action == 4:
            if self.cell.active:
                self.cell.active = False
            cell_on = 0
            if (hydrogen == self.storage.max_capacity) and (elec_demand>0) and (electrolyser_on):
                hydrogen_produced, loss = self.electrolyser.produceHydrogen()
                loss_power += loss
                costs = production_cost

        reward = revenue - costs - (loss_power * elec_price)
        self.rew_arr.append(reward)
        self.loss_arr.append(loss_power)
        self.stor_arr.append(self.storage.actual_quantity)
        self.state = np.array([self.storage.actual_quantity, energy_produced, production_cost, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on], dtype=np.float32)   #stato aggiornato
        done = False
        truncated = False

        return self.state, float(reward), done, truncated, {}



    def reset(self, *, seed=None, options=None):
        super().reset(seed = seed)
        self.state = np.array([0, 0, 5, 5, 5, 15, 1 ,1], dtype = np.float32)  #stato iniziale da definire
        return self.state,{}


    def render(self):
        print("")

    def get_res(self):
        return self.rew_arr, self.stor_arr, self.loss_arr