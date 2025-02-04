import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
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

        self.action_space = spaces.Discrete(5)  #5 azioni discrete perchè rappresentano scelte finite

        #gli stati invece saranno continui poichè rappresentano grandezze variabili nel tempo
        #Stato: quantità idrogeno stoccato, energia prodotta, costo produzione idrogeno, prezzo vendita elettricità, prezzo_idrogeno, domanda elettricità, elettrolita in azione, celle in azione
        self.observation_space = spaces.Box(low = np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                                            high = np.array([500, 100, 10, 10, 10, 100, 1, 1]),
                                            dtype = np.float32)

        self.storage = hydrogenStorage(max_capacity=500, pressure=350)
        self.cell = FuelCell(power = 50, efficiency = 0.6, hydrogen_consumption = 1.5, HSS=self.storage, active=True)
        self.electrolyser = Electrolyser(min_power=10, max_power=50, period=10, HSS=self.storage, active=True)
        self.state = np.array([0, 0, 5, 5, 100, 5, electrolyser.active, cell.active], dtype = np.float32)  #stato iniziale da definire

    def step(self, action):

        hydrogen, energy_produced, production_cost, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on = self.state
        revenue = 0
        loss_power = 0
        costs = 0

        if action == Actions.PRODUCE.value:
            if not (self.electrolyser.active):
                self.electrolyser.active = True
            electrolyser_on = 1
            hydrogen_produced, loss = self.electrolyser.produceHydrogen()
            loss_power += loss
            revenue += hydrogen_produced * hydrogen_price
            costs = production_cost
        elif action == Actions.SELL_HYDR.value:
            if not(self.cell.active):
                self.cell.active = True
            cell_on = 1
            generate_power = self.cell.generatePower(elec_demand)
            if generate_power > 0:
                revenue += generate_power * elec_price
            else:
                revenue -= 100 #non viene soddisfatta la richiesta energetica
        elif action == Actions.SELL_ELEC.value:
            if self.cell.active:
                self.cell.active = False
            cell_on = 0
            if elec_demand == 0:
                loss_power += energy_produced
            else:
                revenue += energy_produced * elec_price

        elif action == Actions.BLOCK_PRODUCTION.value:
            if self.electrolyser.active:
                self.electrolyser.active = False
            electrolyser_on = 0
            if self.HSS.actual_quantity < self.HSS.max_capacity & energy_produced > elec_demand:
                loss_power += energy_produced - elec_demand
        elif action == Actions.BLOCK_SELL.value:
            if self.cell.active:
                self.cell.active = False
            cell_on = 0
            if self.HSS.actual_quantity == self.HSS.max_capacity & elec_demand>0 & electrolyser_on:
                hydrogen_produced, loss = self.electrolyser.produceHydrogen()
                loss_power += loss
                costs = production_cost

        reward = revenue - costs - (loss_power * elec_price)
        self.state = np.array([self.HSS.actual_quantity, energy_produced, production_cost, elec_price, elec_demand, electrolyser_on, cell_on], dtype=np.float32)   #stato aggiornato
        done = False

        return self.state, reward, done, {}



    def reset(self, seed=None):
        self.state = np.array([0,5,5,100, 5, 1 ,1], dtype = np.float32)  #stato iniziale da definire
        return self.state,{}


    def render(self):
        print("")