import numpy as np

areas = []

areas.append(np.array([1,2,1,100,500],np.uint32))
areas.append(np.array([5,6,3,120,480],np.uint32))

areas.append(np.array([10,11,1,300,500],np.uint32))
areas.append(np.array([12,14,3,108,305],np.uint32))

areas = np.array(areas)

a = np.argsort(areas[:,3])
b = areas[a][0,3]

