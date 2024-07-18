from ase import units
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from nff.md.utils import NeuralMDLogger

DEFAULTNVEPARAMS = {
    "T_init": 120.0,
    "thermostat": Langevin,  # or Langevin or NPT or NVT or Thermodynamic Integration
    "thermostat_params": {
        "timestep": 0.5 * units.fs,
        "temperature": 120.0 * units.kB,
        "friction": 0.002,
    },
    "nbr_list_update_freq": 20,
    "steps": 3000,
    "save_frequency": 10,
    "thermo_filename": "./thermo.log",
    "traj_filename": "./atom.traj",
    "skip": 0,
}


class TI:
    def __init__(
        self,
        atomsbatch,
        final_aggr,
        init_aggr,
        mdparam=DEFAULTNVEPARAMS,
    ):
        """
        modelparams = dict()
        modelparams['n_atom_basis'] = 128
        modelparams['n_filters'] = 128
        modelparams['n_gaussians'] = 32
        modelparams['n_convolutions'] = 3
        modelparams['n_convolutions'] = 3
        modelparams['cutoff'] = 3
        thermo_int = GraphConvIntegration(modelparams)

        calc = NeuralFF(model=thermo_int, device=1)

        final_aggr = torch.Tensor([1.] * 127 + [0.])
        init_aggr = torch.Tensor([1.] * 128)

        bulk.set_calculator(calc)
        nve = TI(bulk, final_aggr, init_aggr, DEFAULTNVEPARAMS)
        """
        # initialize the atoms batch system
        self.atomsbatch = atomsbatch
        self.mdparam = mdparam
        self.init_aggr = init_aggr
        self.final_aggr = final_aggr

        # todo: structure optimization before starting

        # intialize system momentum
        MaxwellBoltzmannDistribution(self.atomsbatch, self.mdparam["T_init"] * units.kB)

        # set thermostats
        integrator = self.mdparam["thermostat"]

        self.integrator = integrator(self.atomsbatch, **self.mdparam["thermostat_params"])

        # attach trajectory dump
        self.traj = Trajectory(self.mdparam["traj_filename"], "w", self.atomsbatch)
        self.integrator.attach(self.traj.write, interval=mdparam["save_frequency"])

        # attach log file
        self.integrator.attach(
            NeuralMDLogger(
                self.integrator,
                self.atomsbatch,
                self.mdparam["thermo_filename"],
                mode="a",
            ),
            interval=mdparam["save_frequency"],
        )

    def run(self):
        #
        epochs = int(self.mdparam["steps"] // self.mdparam["nbr_list_update_freq"])

        dlambda = (self.final_aggr - self.init_aggr) / epochs

        self.atomsbatch.props["aggr_wgt"] = self.init_aggr

        for step in range(epochs):
            self.integrator.run(self.mdparam["nbr_list_update_freq"])
            self.atomsbatch.update_nbr_list()
            self.atomsbatch.props["aggr_wgt"] += dlambda
            # update

        self.traj.close()
