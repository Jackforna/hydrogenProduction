import math
class Electrolyser:
    def __init__(self, min_power, max_power, period, HSS, active=True):
        #Inizializza un elettrolizzatore con potenza minima e massima.
        self.min_power = min_power
        self.max_power = max_power
        self.active = active
        self.period = period        #indica dopo quanto si ripete l'andamento del grafico (ogni 10 ore)
        self.time = 0               #misura il tempo in ore
        self.efficiency = 1.0
        self.HSS = HSS
   
    def powerSupplied(self):
        #Genera un valore di potenza seguendo un andamento sinusoidale.
        width = (self.max_power - self.min_power) / 2
        avg = (self.max_power + self.min_power) / 2
        power = avg + width * math.sin((2 * math.pi / self.period) * self.time)
        self.time += 1
        return power
   
    def produceHydrogen(self, power_generated):
        #Calcola la quantità di idrogeno prodotta in base alla potenza generata e la inserisce nel sistema di stoccaggio.
        if self.HSS.actual_quantity < self.HSS.max_capacity:
            self.active = True
            hydrogen_produced = power_generated * 0.2 * self.efficiency # Conversione potenza a idrogeno (m3/kW)
            if self.HSS.actual_quantity + hydrogen_produced <= self.HSS.max_capacity:
                self.HSS.addHydrogen(hydrogen_produced)
                loss = 0
                return hydrogen_produced, loss
            else:
                self.HSS.addHydrogen(self.HSS.max_capacity - self.HSS.actual_quantity) #riempie al massimo HSS ma perdita di potenza in eccesso
                loss = power_generated - ((self.HSS.max_capacity - self.HSS.actual_quantity) / (0.2 * self.efficiency))
                return hydrogen_produced, loss
            #print(f"Generated power: {power_generated:.2f} kW, producing {hydrogen_produced:.2f} m³ of hydrogen.")
        else:
            self.active = False
            return 0, power_generated
            #print("Hydrogen storage is full, cannot produce more hydrogen.")