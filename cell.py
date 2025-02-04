class FuelCell:
    def __init__(self, power, efficiency, hydrogen_consumption, HSS, active=True):
        self.power = power
        self.efficiency = efficiency
        self.hydrogen_consumption = hydrogen_consumption
        self.active = active
        self.activity_period = 0
        self.HSS = HSS

    def generatePower(self, elec_demand):

        power_output = min(elec_demand, self.power)
        hydrogen_needed = power_output * self.hydrogen_consumption

        if hydrogen_needed > self.HSS.actual_quantity:  #la quantità di idrogeno richiesta è maggiore di quella disponibile
            self.active = False
            return 0

        self.HSS.removeHydrogen(hydrogen_needed)    #rimuove dallo stoccaggio l'idrogeno usato

        return power_output
        

    
