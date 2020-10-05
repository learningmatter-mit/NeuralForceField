from ase.io import Trajectory
from ase.md.langevin import *
from ase import Atoms
from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal,second,Ang
from nff.md.utils import NeuralMDLogger, write_traj

class Reactive_Dynamics:
    
    def __init__(self, 
                atomsbatch,
                nms_vel,
                mdparam,
                ):
    
        # initialize the atoms batch system 
        self.atomsbatch = atomsbatch
        self.mdparam = mdparam
       
        #initialize velocity from nms
        self.vel = nms_vel
        
        self.temperature = self.mdparam['T_init']
        
        self.friction = self.mdparam['friction']
        
        # todo: structure optimization before starting
        
        # intialize system momentum by normal mode sampling 
        self.atomsbatch.set_velocities(self.vel.reshape(-1,3) * Ang / second) 
        
        # set thermostats 
        integrator = self.mdparam['thermostat']
        
        self.integrator = integrator(self.atomsbatch, 
                                     self.mdparam['time_step'] * fs,
                                     self.temperature * kB, 
                                     self.friction)
        
        # attach trajectory dump 
        self.traj = Trajectory(self.mdparam['traj_filename'], 'w', self.atomsbatch)
        self.integrator.attach(self.traj.write, interval=mdparam['save_frequency'])
        
        # attach log file
        self.integrator.attach(NeuralMDLogger(self.integrator, 
                                              self.atomsbatch, 
                                              self.mdparam['thermo_filename'], 
                                              mode='a'), interval=mdparam['save_frequency'])

    def run(self):
        
        self.integrator.run(self.mdparam['steps'])

        #self.traj.close()
        
    
    def save_as_xyz(self, filename):
        
        '''
        TODO: save time information 
        TODO: subclass TrajectoryReader/TrajectoryReader to digest AtomsBatch instead of Atoms?
        TODO: other system variables in .xyz formats 
        '''
        traj = Trajectory(self.mdparam['traj_filename'], mode='r')
        
        xyz = []

        for snapshot in traj:
            frames = np.concatenate([
                snapshot.get_atomic_numbers().reshape(-1, 1),
                snapshot.get_positions().reshape(-1, 3)
            ], axis=1)
            
            xyz.append(frames)
            
        write_traj(filename, np.array(xyz))
