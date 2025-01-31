import numpy as np

def sim_dynamics(mparams, sim_dt, T0,vdot_pump,Tc0,q_col):
    '''Simulate multinode model dynamics for the water tank and the dynamics for solar collector
    Returns: temperatures at next timestep
    '''

    #Calculate some model parameters
    M = mparams['M'] #number of nodes
    mparams['dx'] = mparams['h'] / mparams['M']  # height of each tank layer [m]
    mparams['A_s'] = mparams['dx'] * np.pi * mparams['r'] * 2  # surface area of layer [m^2]
    mparams['cross_area'] = np.pi * (mparams['r'] ** 2)  # cross sectional area of tank [m^2]
    mparams['layer_vol'] = mparams['dx'] * mparams['cross_area']  # volume of each tank layer [m^3]
    mparams['cap'] = mparams['layer_vol'] * mparams['cp'] * mparams['rho']  # thermal capacitance of tank layer
    mparams['diff'] = mparams['k'] / (mparams['cp'] * mparams['rho'])  # diffusion coefficient for conduction between the layers

    #simulate thermal dynamics
    dT = np.zeros((M,))
    for j in range(M):

        #bottom node
        if j == 0:
            dT[j] = ((-(mparams['A_s'] + mparams['cross_area']) / (mparams['cap'] * mparams['R'])) * (T0[j] - mparams['T_amb']) +
                  mparams['diff'] * ((T0[j + 1] - T0[j]) / (mparams['dx'] ** 2)) -
                  (vdot_pump / mparams['cross_area']) * ((T0[j] - T0[j+1]) / mparams['dx']) )
        # top node
        elif j == M - 1:
            dT[j] = ((-(mparams['A_s'] + mparams['cross_area']) / (mparams['cap'] * mparams['R'])) * (T0[j] - mparams['T_amb']) +
                  mparams['diff'] * ((T0[ j - 1] - T0[j]) / (mparams['dx'] ** 2)) -
                  (vdot_pump / mparams['cross_area']) * ((T0[j] - Tc0) / mparams['dx']) )
        # middle nodes
        else:
            dT[j] = ((-mparams['A_s'] / (mparams['cap'] * mparams['R'])) * (T0[j] - mparams['T_amb']) +
                  mparams['diff'] * ((T0[j - 1] - T0[ j]) / (mparams['dx'] ** 2)) +
                  mparams['diff'] * ((T0[ j + 1] - T0[ j]) / (mparams['dx'] ** 2)) -
                  (vdot_pump / mparams['cross_area']) * ((T0[j] - T0[j + 1]) / mparams['dx']))

    #solar collector dynamics
    dTc=mparams['eff']*(mparams['A_col']*q_col/(mparams['rho']*mparams['cp']*mparams['vol_col']))-((vdot_pump/mparams['vol_col'])*(Tc0-T0[0]))-(1/((mparams['rho']*mparams['cp']*mparams['vol_col']*mparams['R_col'])))*(Tc0-mparams['T_out'])

    #forward Euler timestep
    T_new=T0 + sim_dt * dT
    Tc_new=Tc0+sim_dt*dTc

    #mixing heuristic to deal with buoyancy dynamics (average the inverted layers)
    T_new=mixing_heuristic(T_new)

    return T_new, Tc_new


def mixing_heuristic(T_new):
    '''Heuristic to deal with the buoyancy dynamics in the multinode model: average the layers that have temperature inversions'''

    tol=0.01 #tolerance for temperature inversions
    bool_check = True
    while (bool_check == True):
        diff_array = np.diff(T_new)
        if np.min(diff_array) < -tol:
            zero_index = np.argmax(diff_array < -tol)
            T_new[zero_index:zero_index + 2] = (T_new[zero_index] + T_new[zero_index + 1]) / 2.0 #average inverted nodes
            bool_check = True
        else:
            bool_check = False

    return T_new