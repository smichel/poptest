import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def coriolis(lat,vel_ang=2*np.pi/86400):
    return 2*vel_ang*np.sin(np.deg2rad(lat))


def migration(i,j,pop,biom,landmask):
    if pop[i,j]>biom[i,j]:
        pop_res = pop[i,j] - biom[i,j]
        biom_res = -pop[i,j]*1.1
        biom_n = biom[i, j - 1]
        biom_e = biom[i + 1, j]
        biom_s = biom[i, j + 1]
        biom_w = biom[i - 1, j]
        biom_surround= biom[i, j - 1] + biom[i + 1, j] + biom[i, j + 1] + biom[i - 1, j]
        biom_weight = np.array([biom[i, j - 1], biom[i + 1, j],  biom[i, j + 1], biom[i - 1, j]]) / biom_surround

        pop_delta = np.zeros(pop.shape)
        pop_delta[i,j] = -pop[i,j]
        pop_delta[i,j-1] = pop_res*biom_weight[0]
        pop_delta[i + 1, j] = pop_res * biom_weight[1]
        pop_delta[i, j + 1] = pop_res * biom_weight[2]
        pop_delta[i - 1, j] = pop_res * biom_weight[3]

        biom_delta = np.zeros(pop.shape)
        biom_delta[i,j] = biom_res

        return pop_delta, biom_delta
    elif pop[i,j]<=biom[i,j]:

        biom_res = -pop[i,j]

        biom_delta = np.zeros(pop.shape)
        biom_delta[i,j] = biom_res
        pop_delta = np.zeros(pop.shape)

        return pop_delta,biom_delta


np.random.seed(1)
lat = np.arange(-80,81,5)

lon = np.arange(0,360,5)
coriolis_param = coriolis(lat)
[lons,lats]=np.meshgrid(lon,lat)

T_0 = 290
delta_T = 18
lat_sin = np.sin(np.deg2rad(lats))
lon_sin = np.sin(np.deg2rad(2*lons))
u = np.abs((45-lats))/45
v = np.abs(np.abs(lats)-45)/45
T = T_0-np.abs(np.sin(np.deg2rad(lats)))*delta_T
height = np.random.random((8,15))
height = ndimage.zoom(height,(lats.shape[0]/height.shape[0],lats.shape[1]/height.shape[1]))
dummy = np.random.random((2,4))
height +=ndimage.zoom(dummy,(lats.shape[0]/dummy.shape[0],lats.shape[1]/dummy.shape[1]))
height += np.random.random((height.shape))*.1
#height[height<1.1] = 0
height = np.sqrt(height)
height_nested = np.zeros((lats.shape[0]+2,lats.shape[1]+2))
height_nested[1:-1,1:-1]=height
land_mask = height_nested>0
T_nested = np.zeros((lats.shape[0]+2,lats.shape[1]+2))
T_nested[1:-1,1:-1] = T

biomass = (T_nested-250)*60 - height_nested * 2000
biomass[biomass<0] = 0
pop = np.zeros(biomass.shape)
pop[np.random.randint(1,pop.shape[0]-1),np.random.randint(1,pop.shape[1]-1)] = 1000
timesteps = 1000
biomass_time = np.zeros((biomass.shape[0],biomass.shape[1],timesteps))
pop_time = np.zeros((pop.shape[0],pop.shape[1],timesteps))
for t in range(timesteps):
    idxs = np.where(pop>0)
    for i in range(len(idxs[0])):
        pop_delta, biomass_delta = migration(idxs[0][i],idxs[1][i],pop,biomass,land_mask)
        pop +=pop_delta
        biomass += biomass_delta


    biomass[biomass<0] = 0
    pop_time[:,:,t] = pop
    biomass_time[:, :, t] = biomass

for t in range(timesteps):
    if t == 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        img = ax.imshow(pop_time[:,:,t])
        #s = plt.colorbar(img)
        #s.set_clim(0, 1000)
        #s.set_ticks(np.arange(0,1100,100))
        #s.draw_all()
    else:
        img.set_data(pop_time[:,:,t])
    plt.pause(0.05)
plt.imshow(biomass)
plt.figure()
plt.imshow(pop)
