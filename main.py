import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'
from scipy import ndimage
import subprocess




def coriolis(lat,vel_ang=2*np.pi/86400):
    return 2*vel_ang*np.sin(np.deg2rad(lat))


def migration(i,j,pop,biom,landmask):
    if pop[i,j]>biom[i,j]:
        pop_res = pop[i,j] - biom[i,j]
        biom_res = -pop[i,j]*1.01
        biom_n = biom[i, j - 1]
        biom_e = biom[i + 1, j]
        biom_s = biom[i, j + 1]
        biom_w = biom[i - 1, j]
        biom_surround= biom[i, j - 1] + biom[i + 1, j] + biom[i, j + 1] + biom[i - 1, j]
        pop_delta = np.zeros(pop.shape)
        biom_delta = np.zeros(pop.shape)

        if biom_surround <= 0:
            pop_delta[i, j] = -pop[i, j]
            biom_delta[i,j] = biom_res
            return pop_delta,biom_delta
        elif biom_surround < pop_res:
            pop_res = biom_surround
        biom_weight = np.array([biom[i, j - 1], biom[i + 1, j],  biom[i, j + 1], biom[i - 1, j]]) / biom_surround

        pop_delta = np.zeros(pop.shape)
        pop_delta[i,j] = -pop[i,j]
        pop_delta[i,j-1] = pop_res*biom_weight[0]
        pop_delta[i + 1, j] = pop_res * biom_weight[1]
        pop_delta[i, j + 1] = pop_res * biom_weight[2]
        pop_delta[i - 1, j] = pop_res * biom_weight[3]

        biom_delta[i,j] = biom_res

        return pop_delta, biom_delta
    elif pop[i,j]<=biom[i,j]:
        pop_res = pop[i, j] * .25

        biom_res = -pop[i,j]
        if pop[i, j] + pop_res > biom[i,j] + biom_res:
            pop_res = biom[i,j] + biom_res
        biom_delta = np.zeros(pop.shape)
        biom_delta[i,j] = biom_res
        pop_delta = np.zeros(pop.shape)
        pop_delta[i,j] = pop_res
        return pop_delta,biom_delta

#np.random.seed(1)
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
height[height<0.7] = 0
height = np.sqrt(height)
height_nested = np.zeros((lats.shape[0]+2,lats.shape[1]+2))
height_nested[1:-1,1:-1]=height
land_mask = height_nested>0
T_nested = np.zeros((lats.shape[0]+2,lats.shape[1]+2))
T_nested[1:-1,1:-1] = T

biomass = (T_nested-250)*100 - height_nested * 2000
biomass[biomass<0] = 0
biomass[height_nested==0] = 0
biomass_ini = np.copy(biomass)
pop = np.zeros(biomass.shape)
pop[np.random.randint(1,pop.shape[0]-1),np.random.randint(1,pop.shape[1]-1)] = 1000

timesteps = 500
minimum_growthtime = 5
biomass_time = np.zeros((biomass.shape[0],biomass.shape[1],timesteps))
pop_time = np.zeros((pop.shape[0],pop.shape[1],timesteps))

veg_time = np.zeros((pop.shape[0],pop.shape[1]))
for t in range(timesteps):
    idxs = np.where(pop>0)
    for i in range(len(idxs[0])):
        pop_delta, biomass_delta = migration(idxs[0][i],idxs[1][i],pop,biomass,land_mask)
        pop +=pop_delta
        biomass += biomass_delta

    veg_time[~land_mask] = np.nan
    veg_time[biomass<=0] += 1

    biomass[veg_time>minimum_growthtime] += (biomass_ini[veg_time>minimum_growthtime]-
                                             biomass[veg_time>minimum_growthtime])*\
                                            np.e**(-0.5**(veg_time[veg_time>minimum_growthtime]-minimum_growthtime))
    pop[pop<0] = 0
    biomass[biomass<0] = 0
    biomass[~land_mask] = 0
    veg_time[biomass==biomass_ini] = 0
    #veg_time[pop>0] = 0
    pop_time[:,:,t] = pop
    biomass_time[:, :, t] = biomass

cmdstring = ('ffmpeg',
             '-y', '-r', '30',  # overwrite, 30fps
             '-s', '%dx%d' % (700, 700),  # size of image string
             '-pix_fmt', 'argb',  # format
             '-f', 'rawvideo', '-i', '-', '-b:v', '5M', '-crf', '14',  # input from pipe, bitrate, compression
             # tell ffmpeg to expect raw video from the pipe
             '-vcodec', 'mpeg4', 'output.mp4')  # output encoding
animate = 0
if animate:


    f = plt.figure(frameon=True, figsize=(7, 7))
    p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)
    ax1 = f.add_subplot(211)
    ax2 = f.add_subplot(212)

    for t in range(len(biomass_time)):
        if t == 0:
            img_biom = ax1.imshow(biomass_time[:, :, t], cmap=plt.get_cmap('YlGn'))
            img_pop = ax2.imshow(pop_time[:, :, t])
            s_pop = f.colorbar(img_pop, ax=ax2)
            s_pop.draw_all()
            s_pop.set_label('Population')
            s_bio = f.colorbar(img_biom, ax=ax1)
            s_bio.draw_all()
            s_bio.set_label('Biomasse')
        else:
            img_biom.set_data(biomass_time[:, :, t])
            img_pop.set_data(pop_time[:, :, t])
        f.canvas.draw()

        string = f.canvas.tostring_argb()

        p.stdin.write(string)

    p.communicate()

if animate:
    fig, axes = plt.subplots(2, 1)
    img_biom = axes[0].imshow(biomass_time[:, :, t], cmap=plt.get_cmap('YlGn'))
    img_pop = axes[1].imshow(pop_time[:, :, t])
    s_pop = fig.colorbar(img_pop, ax=axes[1])
    s_pop.draw_all()
    s_pop.set_label('Population')
    s_bio = fig.colorbar(img_biom, ax=axes[0])
    s_bio.draw_all()
    s_bio.set_label('Biomasse')

    def animate(t):
        img_biom.set_data(biomass_time[:, :, t])
        img_pop.set_data(pop_time[:, :, t])
        return [img_biom,img_pop]


    FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
    anim = animation.FuncAnimation(fig, animate,
                               frames=timesteps,interval=100,blit=False)
    anim.save("./output.mp4", writer=FFwriter)


              #extra_args=['-vcodec', 'h264',
               #           '-pix_fmt', 'yuv420p'])
#timesteps =np.where(np.sum(np.sum(pop_time,axis=1),axis=0)==0)[0][0]
for t in range(timesteps):
    if t == 0:
        fig, axes = plt.subplots(2,1,figsize=(10,8))
        img_biom = axes[0].imshow(biomass_time[:,:,t],cmap=plt.get_cmap('YlGn'))
        img_pop = axes[1].imshow(pop_time[:, :, t])
        s_pop = fig.colorbar(img_pop,ax=axes[1])
        s_pop.draw_all()
        s_pop.set_label('Population')
        s_bio = fig.colorbar(img_biom,ax=axes[0])
        s_bio.draw_all()
        s_bio.set_label('Biomasse')
    else:
        img_biom.set_data(biomass_time[:,:,t])
        img_pop.set_data(pop_time[:, :, t])
    plt.pause(0.05)



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
    if t%50==0:
        if (np.sum(biomass_time[:,:,t])==np.sum(biomass_time[:,:,t-1])):
            t=timesteps
#plt.imshow(biomass)
plt.figure()
plt.imshow(pop)
