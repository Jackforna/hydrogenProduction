 class hydrogenStorage:
    def __init__(self, max_capacity, pressure):
        #Inizializza un HSS.
        self.max_capacity = max_capacity
        self.pressure = pressure
        self.actual_quantity = 0  # Quantità iniziale di idrogeno (in m3)
   
    def addHydrogen(self, quantity):
        #Aggiunge idrogeno al sistema di stoccaggio.
        if quantity < 0:
            raise ValueError("The quantity to add must be positive.")
        new_quantity = self.actual_quantity + quantity
        if new_quantity > self.max_capacity:
            raise ValueError("Exceeded maximum storage capacity!")
        self.actual_quantity = new_quantity
        print(f"Added {quantity:.2f} m3 of hydrogen. Current quantity: {self.actual_quantity:.2f} m3.")
   
    def removeHydrogen(self, quantity):
        #Rimuove idrogeno dal sistema di stoccaggio.
        if quantity < 0:
            raise ValueError("La quantità da rimuovere deve essere positiva.")
        new_quantity = self.actual_quantity - quantity
        if new_quantity < 0:
            raise ValueError("Non si può rimuovere più idrogeno di quello disponibile!")
        self.actual_quantity = new_quantity
        print(f"Removed {quantity:.2f} m³ of hydrogen. Current quantity: {self.actual_quantity} m3.")
   
    def getState(self):
        #Restituisce lo stato attuale del sistema di stoccaggio.
        return {
            "Capacità massima (m3)": self.max_capacity,
            "Pressione (bar)": self.pressure,
            "Quantità attuale (m3)": self.actual_quantity
        }