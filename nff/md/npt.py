import os
import numpy as np
import copy
import math

from ase.optimize.optimize import Dynamics
from ase.md.npt import NPT
from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)


class NoseHoovernpt(NPT):
    def __init__(self, atoms,
                 timestep, temperature=None, externalstress=None,
                 ttime=None, pfactor=None,
                 *, temperature_K=None,
                 mask=None, trajectory=None, logfile=None, loginterval=1,
                 nbr_update_period=20,append_trajectory=False):

        if os.path.isfile(str(trajectory)):
            os.remove(trajectory)

        NPT.__init__(self,atoms=atoms,
                     timestep=timestep * units.fs,
                     ttime=ttime,
                     externalstress=externalstress,
                     pfactor=pfactor,
                     temperature_K=temperature,
                     mask=mask,
                     trajectory=trajectory,
                     logfile=logfile,
                     loginterval=loginterval,
                     append_trajectory=append_trajectory)

        # Initialize simulation parameters
        # convert units
        self.nbr_update_period = nbr_update_period
        self.max_steps=0
        MaxwellBoltzmannDistribution(self.atoms, 2*temperature*units.kB)
        Stationary(self.atoms)
        ZeroRotation(self.atoms)
        self.initialize()
    def run(self, steps=None):

        if steps is None:
            steps = self.num_steps

        epochs = math.ceil(steps / self.nbr_update_period)
        # number of steps in between nbr updates
        steps_per_epoch = int(steps / epochs)
        # maximum number of steps starts at `steps_per_epoch`
        # and increments after every nbr list update
        #self.max_steps = 0
        self.atoms.update_nbr_list()

        for _ in range(epochs):
            self.max_steps += steps_per_epoch
            Dynamics.run(self)
            self.atoms.update_nbr_list()
 
