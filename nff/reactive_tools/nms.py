import scipy
import numpy as np
from scipy.stats import rv_discrete
from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal,second

CM_2_AU = 4.5564e-6
ANGS_2_AU = 1.8897259886
AMU_2_AU = 1822.88985136
k_B = 1.38064852e-23
PLANCKS_CONS = 6.62607015e-34 
HA2J = 4.359744E-18
BOHRS2ANG = 0.529177
SPEEDOFLIGHT = 2.99792458E8
AMU2KG = 1.660538782E-27 

class Boltzmann_gen(rv_discrete):
    "Boltzmann distribution"
    def _pmf(self, k, nu, temperature):
         return ((np.exp(-(k * PLANCKS_CONS * nu)/(k_B * temperature))) *
                (1 - np.exp(-(PLANCKS_CONS * nu)/(k_B * temperature))))

def reactive_normal_mode_sampling(xyz, force_constants_J_m_2,
                                  proj_vib_freq_cm_1, proj_hessian_eigvec,
                                  temperature,
                                  kick=1):
    
    """Normal Mode Sampling for Transition States. Takes in xyz(1,N,3), force_constants(3N-6) in J/m^2,
    projected vibrational frequencies(3N-6) in cm^-1,mass-weighted projected hessian eigenvectors(3N-6,3N)
    ,temperature in K, and scaling factor of initial velocity of the lowest imaginary mode. 
    Returns displaces xyz and a pair of velocities(forward and backwards)"""
    
    #Determine the highest level occupany of each mode
    occ_vib_modes = []
    boltzmann = Boltzmann_gen(a=0, b=1000000, name="boltzmann")
    for i, nu in enumerate(proj_vib_freq_cm_1):
        if nu > 50:
            occ_vib_modes.append(boltzmann.rvs(nu * SPEEDOFLIGHT * 100, 
                                               temperature)) 
        elif i == 0:
            occ_vib_modes.append(boltzmann.rvs(-1 * nu * SPEEDOFLIGHT * 100, 
                                               temperature))
        else:
            occ_vib_modes.append(-1)
    
    #Determine maximum displacement (amplitude) of each mode
    
    amplitudes = []
    freqs = []
    for i, occ in enumerate(occ_vib_modes):
        if occ >= 0:
            energy = proj_vib_freq_cm_1[i] * SPEEDOFLIGHT * 100 * PLANCKS_CONS # cm-1 to Joules
            amplitudes.append(np.sqrt((0.5 * (occ + 1) * energy) / force_constants_J_m_2[i]) * 1e9) #Angstom
        else:
            amplitudes.append(0)

    #Determine the actual displacements and velocities
    displacements = []
    velocities = []
    random_0_1 = [np.random.normal(0,1) for i in range(len(amplitudes))]
    for i, amplitude in enumerate(amplitudes):
        
        if force_constants_J_m_2[i] > 0:
            
            displacements.append(amplitude 
                                 * np.cos(2 * np.pi * random_0_1[i]) 
                                 * proj_hessian_eigvec[i])
        
            velocities.append(-1 * proj_vib_freq_cm_1[i] * SPEEDOFLIGHT * 100 * 2 * np.pi
                             * amplitude 
                             * np.sin(2 * np.pi * random_0_1[i]) 
                             * proj_hessian_eigvec[i] / Bohr**2)
            
        elif i == 0:
            
            displacements.append(0)
            velocities.append(0)
            
            # Extra kick for lowest imagninary mode(s)
            velocities.append(-1 * proj_vib_freq_cm_1[i] * SPEEDOFLIGHT * 100 * 2 * np.pi
                 * amplitude 
                 * np.sin(2 * np.pi * random_0_1[i]) 
                 * proj_hessian_eigvec[i] / Bohr**2) 
            
            
        else:
            
            displacements.append(0)
            velocities.append(0)
    
    tot_disp = np.sum(np.array(displacements),axis=0)
    #In angstroms
    disp_xyz = xyz + tot_disp.reshape(1,-1,3)
    #In angstroms per second
    tot_vel_plus = np.sum(np.array(velocities),axis=0).reshape(1,-1,3) 
    tot_vel_minus = -1 * tot_vel_plus
    
    return disp_xyz, tot_vel_plus, tot_vel_minus
