
import numpy as np
import matplotlib.pyplot as plt
import os
import multinode
import pandas as pd


# Define simulation parameters
T=60*60*24 #simulation time horizon [s]
sim_dt=10 #simulation timestep [s]

# Define parameters of system dynamics models
mparams={}
mparams['h']=44.25/39.37 #tank height [m]
mparams['r']=9.0/39.37 #tank radius [m]
mparams['M']=20 #number of layers in model
mparams['k']=1.3 #coeffient for modeling conduction between neighboring nodes
mparams['cp']=4181.3 #heat capacity of water
mparams['rho']=1.0e3 #density of water
mparams['R']=1.3 #tank insulation coefficient
mparams['T_amb'] = (70.0-32.0)*(5.0/9.0)+273.15 #ambient indoor temperature [K]
mparams['T_out']=(50-32.0)*(5.0/9.0)+273.15 #ambient outdoor temperature [K]
mparams['T_init_prof']=[295.0, 297.4, 298.7, 302.5,304.2, 308.5 , 311.0 , 312.0,314.0, 315.3 ,
                          315.5, 316.0,317.0, 318.0, 319.0, 320.8,321.1, 321.3, 321.8, 322.0389] #initial conditions for tank temperature [K]
mparams['vol_col']=0.00378541/4.0 #volume of water in the solar collector [m^3]
mparams['eff']=0.7 # efficiency of solar collector
mparams['R_col']=0.5
vdot_pump_nom=6.30902E-5/5 #nominal pump volumetric flow rate [0.2 GPM]
mparams['A_col']=1 #solar collector area [m^2]


#array for tank temperatures
temp_array = np.zeros((int(T/sim_dt),mparams['M']))
temp_array[0,:]=np.array(mparams['T_init_prof'])

#array for solar collector temperature
tc_array=np.zeros((int(T/sim_dt),))
tc_array[0]=mparams['T_out']

#Load the solar radiation data
ghi_data=pd.read_csv('./datasets/nsrdb_stanford_2023.csv')
ghi_data=ghi_data.set_index(pd.DatetimeIndex(ghi_data['date']))
ghi_data_resample=ghi_data.resample(str(sim_dt)+'S').interpolate() #interpolate data at simulation timestep

#array for pump volumetric flow rate
vdot_pump=np.zeros((int(T/sim_dt),))

#Loop through simulation timesteps
for i in range(int(T/sim_dt)-1):

    #Pump control dynamics
    if tc_array[i]>=temp_array[i,0]:
        vdot_pump[i]=vdot_pump_nom
    else:
        vdot_pump[i]=0

    #simulate system for one timestep
    temp_array[i+1,:],tc_array[i+1]=multinode.sim_dynamics(mparams, sim_dt, temp_array[i,:],vdot_pump[i],tc_array[i],ghi_data_resample.GHI[ghi_data_resample.index[i]])

#convert temperature results from K to F
temp_array_F = (temp_array - 273.15) * (9.0 / 5.0) + 32.0
tc_array_F = (tc_array - 273.15) * (9.0 / 5.0) + 32.0


######   PLOT RESULTS   ######

fntsz=10
plt.figure(figsize=(8,6))
plt.subplot(3,1,1)
for i in [1,7,14,20]:
    plt.plot(np.arange(temp_array_F.shape[0])*sim_dt/3600,temp_array_F[:,i-1],label='Tank layer '+str(i))

plt.plot(np.arange(len(tc_array_F))*sim_dt/3600,tc_array_F,'-k',label='Collector temperature ($T_c$)')
plt.legend(fontsize=fntsz-2,ncol=2)
plt.ylabel('Temperature (F)',fontsize=fntsz)
plt.xlim([0,24])
plt.xlabel('Hour',fontsize=fntsz)

plt.subplot(3,1,2)
plt.plot(np.arange(len(np.array(ghi_data_resample.GHI)))*sim_dt/3600,np.array(ghi_data_resample.GHI))
plt.xlabel('Hour',fontsize=fntsz)
plt.xlim([0,24])
plt.ylabel('$Q_{rad}$ $[W/m^2]$',fontsize=fntsz)

plt.subplot(3,1,3)
plt.plot(np.arange(len(vdot_pump))*sim_dt/3600,vdot_pump/6.30902E-5)
plt.xlabel('Hour',fontsize=fntsz)
plt.xlim([0,24])
plt.ylabel('Flow rate [GPM]',fontsize=fntsz)
plt.tight_layout()
plt.savefig('sample_results.png',dpi=300)
plt.show()

