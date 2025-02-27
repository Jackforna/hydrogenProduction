from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from electrolyser import Electrolyser
from scipy.stats import truncnorm
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

        self.action_space = spaces.MultiDiscrete([2,2,2,2,2])  #5 azioni discrete perchè rappresentano scelte finite

        #gli stati invece saranno continui poichè rappresentano grandezze variabili nel tempo
        #Stato: quantità idrogeno stoccato, costo produzione idrogeno, prezzo vendita elettricità, prezzo idrogeno, domanda elettricità, elettrolita in azione, celle in azione
        self.observation_space = spaces.Box(low = np.array([0, 0, 0, 0, 0, 0, 0]),
                                            high = np.array([500, 4, 10, 10, 200, 1, 1]),
                                            dtype = np.float32)

        self.storage = hydrogenStorage(max_capacity=500, pressure=350)
        self.cell = FuelCell(power = 250, efficiency = 0.6, hydrogen_consumption = 1.5, HSS=self.storage, active=True)
        self.electrolyser = Electrolyser(min_power=30, max_power=300, period=10, HSS=self.storage, active=True)
        self.state = np.array([0, 0, 2, 5, 5, 50, self.electrolyser.active, self.cell.active], dtype = np.float32)  #stato iniziale da definire
        self.rew_arr = []
        self.stor_arr = []
        self.loss_arr = []
        self.hydr_arr = []
        self.elec_arr = []


    def step(self, action):

        produce, sell_hydr, sell_elec, block_prod, block_sell = action
        if not self.filter_invalid_actions(action):
            return self.state, -100, False, False, {}  # Penalità per azioni non valide

        hydrogen, production_cost, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on = self.state
        loss_power = 0
        revenue = 0
        costs = 0
        energy_produced = self.electrolyser.powerSupplied()
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
                hydrogen_produced, loss = self.electrolyser.produceHydrogen(loss_power)
                revenue += energy_produced * elec_price
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
            if (hydrogen < self.storage.max_capacity) and (energy_produced > elec_demand):
                loss_power += energy_produced - elec_demand

        #Azione: Bloccare vendita
        if block_sell == 1:
            if self.cell.active:
                self.cell.active = False
            cell_on = 0
            if produce == 0 and sell_elec == 0:
                loss_power = energy_produced

        reward = revenue - costs - (loss_power * elec_price)
        self.rew_arr.append(reward)
        self.loss_arr.append(loss_power)
        self.stor_arr.append(self.storage.actual_quantity)
        self.elec_arr.append(energy_elec)
        mu = 125    # Media
        sigma = 30  # Deviazione standard
        lower, upper = 50, 200  # Limiti

        # Creiamo la distribuzione troncata
        trunc_gauss = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

        # Prendiamo un valore casuale
        elec_demand = trunc_gauss.rvs()
        self.state = np.array([self.storage.actual_quantity, production_cost, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on], dtype=np.float32)   #stato aggiornato
        done = False
        truncated = False

        return self.state, float(reward), done, truncated, {}

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

    def reset(self, *, seed=None, options=None):
        super().reset(seed = seed)
        self.state = np.array([0, 2, 5, 5, 50, 1 ,1], dtype = np.float32)  #stato iniziale da definire
        return self.state,{}


    def render(self):
        print("")

    def get_res(self):
        return self.rew_arr, self.stor_arr, self.loss_arr, self.hydr_arr, self.elec_arr