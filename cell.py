class FuelCell:
    def __init__(self, power, efficiency, hydrogen_consumption, HSS, active=True):
        self.efficiency = efficiency
        self.power = power
        self.hydrogen_consumption = hydrogen_consumption
        self.active = active
        self.HSS = HSS

    def generatePower(self, elec_demand):
        if self.active:
            elec_demand = min(elec_demand, self.power)
            hydrogen_needed = elec_demand * self.hydrogen_consumption

            if hydrogen_needed > self.HSS.actual_quantity:  #la quantità di idrogeno richiesta è maggiore di quella disponibile
                hydrogen_needed = self.HSS.actual_quantity
                elec_demand = hydrogen_needed / self.hydrogen_consumption

            self.HSS.removeHydrogen(hydrogen_needed)    #rimuove dallo stoccaggio l'idrogeno usato

            return elec_demand
        else:
            return 0
        

    
