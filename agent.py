 from electrolyser import Electrolyser
from HSS import hydrogenStorage

storage = hydrogenStorage(max_capacity=500, pressure=350)
electrolyser = Electrolyser(min_power=10, max_power=50, period=10, HSS=storage)
   
for _ in range(20):
    electrolyser.produceHydrogen()
   
storage.removeHydrogen(50)
   
print(storage.get_status())