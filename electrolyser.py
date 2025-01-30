 import math
class Electrolyser:
    def __init__(self, min_power, max_power, period, HSS):
        #Inizializza un elettrolizzatore con potenza minima e massima.
        self.min_power = min_power
        self.max_power = max_power
        self.active = True
        self.period = period
        self.time = 0
        self.HSS = HSS
   
    def powerGeneration(self):
        #Genera un valore di potenza seguendo un andamento sinusoidale.
            width = (self.max_power - self.min_power) / 2
            avg = (self.max_power + self.min_power) / 2
            power = avg + width * math.sin((2 * math.pi / self.period) * self.time)
            self.time += 1
            return power
   
    def produceHydrogen(self):
        #Calcola la quantità di idrogeno prodotta in base alla potenza generata e la inserisce nel sistema di stoccaggio.
        if self.HSS.actual_quantity < self.HSS.max_capacity:
            self.active = True
            power_generated = self.powerGeneration()
            hydrogen_produced = power_generated * 0.18  # Conversione potenza a idrogeno (m3/kW)
            if self.HSS.actual_quantity + hydrogen_produced <= self.HSS.max_capacity:
                self.HSS.addHydrogen(hydrogen_produced)
            else:
                self.HSS.addHydrogen(self.HSS.max_capacity-self.HSS.actual_quantity) #riempie al massimo HSS ma perdita di potenza in eccesso, capire come gestirla
            print(f"Generated power: {power_generated:.2f} kW, producing {hydrogen_produced:.2f} m3 of hydrogen.")
        else:
            self.active = False
            print("Hydrogen storage is full, cannot produce more hydrogen.")