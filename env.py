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
        #Stato: quantità idrogeno stoccato, potenza in ingresso, prezzo vendita elettricità, prezzo idrogeno, domanda elettricità, elettrolita in azione, celle in azione
        self.observation_space = spaces.Box(low = np.array([0, 0, 0, 0, 0, 0, 0]),
                                            high = np.array([500, 2500, 5, 10, 2500, 1, 1]),
                                            dtype = np.float32)

        self.storage = hydrogenStorage(max_capacity=500)
        self.cell = FuelCell(power = 2500, efficiency = 0.8, hydrogen_consumption = 3.0, HSS=self.storage, active=True)
        self.electrolyser = Electrolyser(min_power=30, max_power=2500, period=10, HSS=self.storage, active=True)
        self.state = np.array([0, 80, 3, 8, 50, self.electrolyser.active, self.cell.active], dtype = np.float32)  #stato iniziale da definire
        self.rew_arr = []
        self.stor_arr = []
        self.loss_arr = []
        self.hydr_arr = []
        self.elec_arr = []
        self.demand_remained_arr = []

    def step(self, action):

        produce, sell_hydr, sell_elec, block_prod, block_sell = action
        if not self.filter_invalid_actions(action):
            return self.state, -100, False, False, {}  # Penalità per azioni non valide

        hydrogen, energy_produced, elec_price, hydrogen_price, elec_demand, electrolyser_on, cell_on = self.state
        loss_power = 0
        generate_power = 0
        revenue = 0
        energy_elec = 0
        elec_price /= 10
        hydrogen_price /= 33.33
        demand = elec_demand

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
                    if energy_produced > elec_demand:
                        loss_power += energy_produced - elec_demand
                        energy_produced -= loss_power
                        self.electrolyser.produceHydrogen(loss_power)
                        loss_power = 0
                    revenue += energy_produced * elec_price
                    energy_elec += energy_produced
                else:
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
                            self.electrolyser.produceHydrogen(loss_power)
                            loss_power = 0
                        revenue += energy_produced * elec_price
                        energy_elec += energy_produced
                    else:
                        energy_elec += min(energy_produced, elec_demand)
                        elec_demand -= energy_elec
                        revenue += energy_produced * elec_price
                        if elec_demand > 0:
                            generate_power = self.cell.generatePower(elec_demand)
                            revenue += generate_power * elec_price
                        else:
                            self.electrolyser.produceHydrogen(energy_produced - energy_elec)
                            loss_power = 0


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

        reward = revenue - (loss_power * elec_price)
        #if elec_demand > energy_elec + generate_power:
        #    reward -= (elec_demand - (energy_elec + generate_power)) * elec_price
        self.rew_arr.append(reward)
        self.loss_arr.append(loss_power)
        self.stor_arr.append(self.storage.actual_quantity)
        self.elec_arr.append(energy_elec)
        self.hydr_arr.append(generate_power)
        self.demand_remained_arr.append(demand)
        lower, upper = 50, 2500  # Limiti
        mu = (upper + lower) / 2    # Media
        sigma = (upper - lower) / 2  # Deviazione standard

        trunc_gauss = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

        elec_demand = trunc_gauss.rvs()

        energy_produced = self.electrolyser.powerSupplied()
        hydrogen_price = float(np.random.uniform(5,10))
        elec_price = float(np.random.uniform(1,5))
        self.state = np.array([self.storage.actual_quantity, float(energy_produced), elec_price, hydrogen_price, float(elec_demand), electrolyser_on, cell_on], dtype=np.float32)   #stato aggiornato
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

        if produce == 0 and block_prod == 0:
            return False

        if block_sell == 0 and sell_hydr == 0:
            return False

        return True

    def reset(self, *, seed=None, options=None):
        super().reset(seed = seed)
        self.state = np.array([0, 80, 3, 8, 50, 1 ,1], dtype = np.float32)  #stato iniziale da definire
        return self.state,{}


    def render(self):
        print("")

    def get_res(self):
        return self.rew_arr, self.stor_arr, self.loss_arr, self.hydr_arr, self.elec_arr, self.demand_remained_arr