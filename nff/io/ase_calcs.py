"""
This file contains the implementation of the NeuralFF calculator
and the EnsembleNFF calculator. The NeuralFF calculator is an ASE calculator
that uses a pretrained NeuralFF model to predict the energy, forces, and stress
of a given geometry. The EnsembleNFF calculator is an ASE calculator that uses
an ensemble of NeuralFF models to predict the energy, forces, and stress of a
given geometry. Both calculators are used to perform molecular dynamics simulations
and geometry optimizations using NFF AtomsBatch objects.
"""

import os
import sys
from collections import Counter
from typing import List, Union

import numpy as np
import torch
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from torch.autograd import grad

import nff.utils.constants as const
from nff.data import Dataset, collate_dicts
from nff.io.ase import DEFAULT_DIRECTED, AtomsBatch
from nff.nn.models.cp3d import OnlyBondUpdateCP3D
from nff.nn.models.hybridgraph import HybridGraphConv
from nff.nn.models.schnet import SchNet, SchNetDiabat
from nff.nn.models.schnet_features import SchNetFeatures
from nff.train.builders.model import load_model
from nff.utils.constants import EV_TO_KCAL_MOL, HARTREE_TO_KCAL_MOL
from nff.utils.cuda import batch_detach, batch_to
from nff.utils.geom import batch_compute_distance, compute_distances
from nff.utils.scatter import compute_grad
from nff.nn.models.mace import NffScaleMACE

HARTREE_TO_EV = HARTREE_TO_KCAL_MOL / EV_TO_KCAL_MOL


UNDIRECTED = [SchNet, SchNetDiabat, HybridGraphConv, SchNetFeatures, OnlyBondUpdateCP3D]


def check_directed(model, atoms):
    model_cls = model.__class__.__name__
    msg = f"{model_cls} needs a directed neighbor list"
    assert atoms.directed, msg


class NeuralFF(Calculator):
    """ASE calculator using a pretrained NeuralFF model"""

    implemented_properties = ["energy", "forces", "stress", "embedding"]

    def __init__(
        self,
        model,
        device="cpu",
        jobdir=None,
        en_key="energy",
        properties=["energy", "forces"],
        model_kwargs=None,
        model_units="kcal/mol",
        prediction_units="eV",
        **kwargs,
    ):
        """Creates a NeuralFF calculator.nff/io/ase.py

        Args:
        model (TYPE): Description
        device (str): device on which the calculations will be performed
        properties (list of str): 'energy', 'forces' or both and also stress for only
            schnet  and painn
        **kwargs: Description
        model (one of nff.nn.models)
        """

        Calculator.__init__(self, **kwargs)
        self.model = model
        self.model.eval()
        self.device = device
        self.to(device)
        self.jobdir = jobdir
        self.en_key = en_key
        self.properties = properties
        self.model_kwargs = model_kwargs
        self.model_units = model_units
        self.prediction_units = prediction_units

        print("Requested properties:", self.properties)
        
        # Unit conversion factors
        self.energy_units_to_eV = 1.0  # assuming default model outputs energy in eV
        self.length_units_to_A = 1.0   # assuming default model uses A**3 for length

        # Adjust conversion factors if the model uses different units
        if self.model_units == "kcal/mol":
            self.energy_units_to_eV = 0.0433641  # kcal/mol -> eV
        if self.prediction_units == "nm":
            self.length_units_to_A = 10.0  # nm -> AA*3

    def to(self, device):
        self.device = device
        self.model.to(device)

    def log_embedding(self, jobdir, log_filename, props):
        """For the purposes of logging the NN embedding on-the-fly, to help with
        sampling after calling NFF on geometries."""

        log_file = os.path.join(jobdir, log_filename)
        if os.path.exists(log_file):
            log = np.load(log_file)
        else:
            log = None

        if log is not None:
            log = np.append(log, props[None, :, :, :], axis=0)
        else:
            log = props[None, :, :, :]

        np.save(log_filename, log)

        return

    def calculate(
        self,
        atoms: AtomsBatch = None,
        properties: List[str] = ["energy", "forces"],
        system_changes=all_changes,
    ):
        """Calculates the desired properties for the given AtomsBatch.

        Args:
        atoms (AtomsBatch): custom Atoms subclass that contains implementation
            of neighbor lists, batching and so on. Avoids the use of the Dataset
            to calculate using the models created.
        system_changes (default from ase)
        """

        if not any([isinstance(self.model, i) for i in UNDIRECTED]):
            check_directed(self.model, atoms)

        # for backwards compatability
        if getattr(self, "properties", None) is None:
            self.properties = properties

        Calculator.calculate(self, atoms, self.properties, system_changes)

        # run model
        # atomsbatch = AtomsBatch(atoms)
        # batch_to(atomsbatch.get_batch(), self.device)
        batch = batch_to(atoms.get_batch(), self.device)

        # add keys so that the readout function can calculate these properties
        # print("Properties: ", self.properties, "\n\n")
        # print("en_key:", self.en_key)
        grad_key = self.en_key + "_grad"
        # print("grad_key:", grad_key)
        batch[self.en_key] = []
        batch[grad_key] = []

        kwargs = {}
        requires_stress = "stress" in self.properties
        requires_embedding = "embedding" in self.properties
        if requires_embedding:
            kwargs["requires_embedding"] = True
        if requires_stress:
            kwargs["requires_stress"] = True
        kwargs["grimme_disp"] = False
        if getattr(self, "model_kwargs", None) is not None:
            kwargs.update(self.model_kwargs)


        
        prediction = self.model(batch, **kwargs)
        # print(prediction.keys())

        # change energy and force to numpy array
        conversion_factor: dict = const.conversion_factors.get((self.model_units, self.prediction_units), const.DEFAULT)
        prediction_numpy = batch_detach(const.convert_units(prediction, conversion_factor), to_numpy=True)

        # change energy and force to numpy array
        if "/atom" in self.model_units:
            energy = prediction_numpy[self.en_key] * len(atoms)
            prediction_numpy[self.en_key + "_per_atom"] = prediction_numpy[self.en_key]
            prediction_numpy[self.en_key] = energy
        else:
            energy = prediction_numpy[self.en_key]

        if grad_key in prediction_numpy:
            energy_grad = prediction_numpy[grad_key]
        else:
            energy_grad = None

        # TODO: implement unit conversion with prediction_numpy
        self.results = {"energy": energy.reshape(-1)}
        if "e_disp" in prediction:
            self.results["energy"] = self.results["energy"] + prediction["e_disp"]

        if "forces" in self.properties:
            # HACK: this is a fix to make the MACE model work with the calculator
            if energy_grad is not None:
                self.results["forces"] = -energy_grad.reshape(-1, 3)
            else:
                self.results["forces"] = prediction["forces"].detach().cpu().numpy()
            if "forces_disp" in prediction:
                self.results["forces"] = self.results["forces"] + prediction["forces_disp"]

        if requires_embedding:
            embedding = prediction["embedding"].detach().cpu().numpy()
            self.results["embedding"] = embedding

        if requires_stress:
            if isinstance(self.model, NffScaleMACE):#the implementation of stress calculation in MACE is a bit different and hence this is needed (ASE_suit: mace/mace/calculators/mace.py)
               # print(f"Original stress shape: {prediction['stress'].shape}")
                if prediction["stress"].ndim==1:
                    try:
                        prediction["stress"] = prediction["stress"].reshape(3, 3)
                       # print("Reshaped stress to 3*3:", prediction["stress"])
                    except ValueError as e:
                        raise ValueError(f"Error reshaping stress tensor: {e}, shape: {prediction['stress'].shape}")

               # print("The current dimensions of the stress tensor is:", prediction["stress"].ndim)
                
                self.results["stress"] = full_3x3_to_voigt_6_stress(
                    torch.mean(prediction["stress"], dim=0).cpu().numpy()*self.energy_units_to_eV/1.0)#1.0 as volume is in A**3, but conversion for energy is needed
            else:  #for other models
                stress = prediction["stress_volume"].detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL) # TODO change to more general prediction
                self.results["stress"] = stress * (1 / atoms.get_volume())
            if "stress_disp" in prediction:
                self.results["stress"] = self.results["stress"] + prediction["stress_disp"]
            #self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])#for mace model, this line is already implemented above, for other models uncomment this line
        atoms.results = self.results.copy()
        #print("Final stress tensor:", self.results["stress"])

    def get_embedding(self, atoms=None):
        return self.get_property("embedding", atoms)

    @classmethod
    def from_file(cls, model_path, device="cuda", **kwargs):
        # Remove the model_units and prediction_units from the kwargs
        kwargs_wo_units = kwargs.copy()
        kwargs_wo_units.pop("prediction_units", None)
        kwargs_wo_units.pop("model_units", None)
        model = load_model(model_path, device=device, **kwargs_wo_units)
        out = cls(model=model, device=device, **kwargs)

        return out


class EnsembleNFF(Calculator):
    """Produces an ensemble of NFF calculators to predict the
    discrepancy between the properties"""

    implemented_properties = ["energy", "forces", "stress", "energy_std", "forces_std", "stress_std"]

    def __init__(
        self,
        models: list,
        device="cpu",
        jobdir=None,
        properties=["energy", "forces"],
        model_kwargs=None,
        model_units="kcal/mol",
        prediction_units="eV",
        **kwargs,
    ):
        """Creates a NeuralFF calculator.nff/io/ase.py

        Args:
        model(TYPE): Description
        device(str): device on which the calculations will be performed
        **kwargs: Description
        model(one of nff.nn.models)

        """

        Calculator.__init__(self, **kwargs)
        self.models = models
        for m in self.models:
            m.eval()
        self.device = device
        self.to(device)
        self.jobdir = jobdir
        self.offset_data = {}
        self.properties = properties
        self.model_kwargs = model_kwargs
        self.model_units = model_units
        self.prediction_units = prediction_units

    def to(self, device):
        self.device = device
        for m in self.models:
            m.to(device)

    def offset_energy(self, atoms, energy: Union[float, np.ndarray]):
        """Offset the NFF energy to obtain the correct reference energy

        Args:
            atoms (Atoms): Atoms object
            energy (float or np.ndarray): energy to be offset
        """

        ads_count = Counter(atoms.get_chemical_symbols())

        stoidict = self.offset_data.get("stoidict", {})
        ref_en = 0
        for ele, num in ads_count.items():
            ref_en += num * stoidict.get(ele, 0.0)
        ref_en += stoidict.get("offset", 0.0)

        energy += ref_en * HARTREE_TO_EV
        return energy

    def log_ensemble(self, jobdir, log_filename, props):
        """For the purposes of logging the std on-the-fly, to help with
        sampling after calling NFF on geometries with high uncertainty."""

        props = np.swapaxes(np.expand_dims(props, axis=-1), 0, -1)

        log_file = os.path.join(jobdir, log_filename)
        if os.path.exists(log_file):
            log = np.load(log_file)
        else:
            log = None

        if log is not None:
            log = np.concatenate([log, props], axis=0)
        else:
            log = props

        np.save(log_filename, log)

        return

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        """Calculates the desired properties for the given AtomsBatch.

        Args:
        atoms (AtomsBatch): custom Atoms subclass that contains implementation
            of neighbor lists, batching and so on. Avoids the use of the Dataset
            to calculate using the models created.
        properties (list of str): 'energy', 'forces' or both
        system_changes (default from ase)
        """

        for model in self.models:
            if not any([isinstance(model, i) for i in UNDIRECTED]):
                check_directed(model, atoms)

        if getattr(self, "properties", None) is None:
            self.properties = properties

        Calculator.calculate(self, atoms, properties, system_changes)

        # TODO can probably make it more efficient by only calculating the system changes

        # run model
        # atomsbatch = AtomsBatch(atoms)
        # batch_to(atomsbatch.get_batch(), self.device)
        # TODO update atoms only when necessary
        atoms.update_nbr_list(update_atoms=True)

        kwargs = {}
        requires_stress = "stress" in self.properties
        if requires_stress:
            kwargs["requires_stress"] = True
        kwargs["grimme_disp"] = False
        if getattr(self, "model_kwargs", None) is not None:
            kwargs.update(self.model_kwargs)

        # run model
        # atomsbatch = AtomsBatch(atoms)
        # batch_to(atomsbatch.get_batch(), self.device)
        batch = batch_to(atoms.get_batch(), self.device)

        # add keys so that the readout function can calculate these properties
        batch["energy"] = []
        if "forces" in properties:
            batch["energy_grad"] = []

        energies = []
        gradients = []
        stresses = []

        # do this in parallel, using the version here: https://pytorch.org/functorch/1.13/notebooks/ensembling.html

        # fmodel, params, buffers = combine_state_for_ensemble(self.models)
        # predictions_vmap = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, batch)

        # Construct a "stateless" version of one of the models. It is "stateless" in
        # the sense that the parameters are meta Tensors and do not have storage.
        # self.base_model = copy.deepcopy(self.models[0])
        # self.base_model = self.base_model.to('meta')
        # def fmodel(params, buffers, x):
        #     return functional_call(self.base_model, (params, buffers), (x,))

        # predictions_vmap = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, batch)

        # do this in parallel

        for model in self.models:
            # TODO can export to function and merge with NeuralFF class
            prediction = model(batch, **kwargs)

            # change energy and force to numpy array
            conversion_factor: dict = const.conversion_factors.get(
                (self.model_units, self.prediction_units), const.DEFAULT
            )
            prediction_numpy = batch_detach(const.convert_units(prediction, conversion_factor), to_numpy=True)

            if "/atom" in self.model_units:
                energy = prediction_numpy["energy"] * len(atoms)
                prediction_numpy["energy_per_atom"] = prediction_numpy["energy"]
                prediction_numpy["energy"] = energy
            else:
                energy = prediction_numpy["energy"]
            energies.append(energy)
            gradients.append(prediction_numpy["energy_grad"])
            if "stress_volume" in prediction:
                # TODO: implement unit conversion for stress with prediction_numpy
                stresses.append(
                    prediction["stress_volume"].detach().cpu().numpy()
                    * (1 / const.EV_TO_KCAL_MOL)
                    * (1 / atoms.get_volume())
                )

        energies = np.stack(energies)
        gradients = np.stack(gradients)

        # offset energies
        energies = self.offset_energy(atoms, energies)

        if len(stresses) > 0:
            stresses = np.stack(stresses)

        self.results = {
            "energy": energies.mean(0).reshape(-1),
            "energy_std": energies.std(0).reshape(-1),
        }
        if "e_disp" in prediction:
            self.results["energy"] = self.results["energy"] + prediction["e_disp"]
        if self.jobdir is not None and system_changes:
            energy_std = self.results["energy_std"][None]
            self.log_ensemble(self.jobdir, "energy_nff_ensemble.npy", energies)

        if "forces" in properties:
            self.results["forces"] = -gradients.mean(0).reshape(-1, 3)
            self.results["forces_std"] = gradients.std(0).reshape(-1, 3)
            if "forces_disp" in prediction:
                self.results["forces"] = self.results["forces"] + prediction["forces_disp"]
            if self.jobdir is not None:
                forces_std = self.results["forces_std"][None, :, :]
                self.log_ensemble(self.jobdir, "forces_nff_ensemble.npy", -1 * gradients)

        if "stress" in properties:
            self.results["stress"] = stresses.mean(0)
            self.results["stress_std"] = stresses.std(0)
            if "stress_disp" in prediction:
                self.results["stress"] = self.results["stress"] + prediction["stress_disp"]
            if self.jobdir is not None:
                stress_std = self.results["stress_std"][None, :, :]
                self.log_ensemble(self.jobdir, "stress_nff_ensemble.npy", stresses)

        atoms.results = self.results.copy()

    def set(self, **kwargs):
        """Set parameters like set(key1=value1, key2=value2, ...).

        A dictionary containing the parameters that have been changed
        is returned.

        The special keyword 'parameters' can be used to read
        parameters from a file."""
        Calculator.set(self, **kwargs)
        if "offset_data" in self.parameters.keys():
            self.offset_data = self.parameters["offset_data"]
            print(f"offset data: {self.offset_data} is set from parameters")

    @classmethod
    def from_files(cls, model_paths: list, device="cuda", **kwargs):
        models = [load_model(path, device=device, **kwargs) for path in model_paths]
        return cls(models, device, **kwargs)


class NeuralOptimizer:
    def __init__(self, optimizer, nbrlist_update_freq=5):
        self.optimizer = optimizer
        self.update_freq = nbrlist_update_freq

    def run(self, fmax=0.2, steps=1000):
        epochs = steps // self.update_freq

        for step in range(epochs):
            self.optimizer.run(fmax=fmax, steps=self.update_freq)
            self.optimizer.atoms.update_nbr_list()


class NeuralMetadynamics(NeuralFF):
    def __init__(
        self,
        model,
        pushing_params,
        old_atoms=None,
        device="cpu",
        en_key="energy",
        directed=DEFAULT_DIRECTED,
        **kwargs,
    ):
        NeuralFF.__init__(
            self,
            model=model,
            device=device,
            en_key=en_key,
            directed=DEFAULT_DIRECTED,
            **kwargs,
        )

        self.pushing_params = pushing_params
        self.old_atoms = old_atoms if (old_atoms is not None) else []
        self.steps_from_old = []

        # only apply the bias to certain atoms
        self.exclude_atoms = torch.LongTensor(self.pushing_params.get("exclude_atoms", []))
        self.keep_idx = None

    def get_keep_idx(self, atoms):
        # correct for atoms not in the biasing potential

        if self.keep_idx is not None:
            assert len(self.keep_idx) + len(self.exclude_atoms) == len(atoms)
            return self.keep_idx

        keep_idx = torch.LongTensor([i for i in range(len(atoms)) if i not in self.exclude_atoms])
        self.keep_idx = keep_idx
        return keep_idx

    def make_dsets(self, atoms):
        keep_idx = self.get_keep_idx(atoms)
        # put the current atoms as the second dataset because that's the one
        # that gets its positions rotated and its gradient computed
        props_1 = {"nxyz": [torch.Tensor(atoms.get_nxyz())[keep_idx, :]]}
        props_0 = {"nxyz": [torch.Tensor(old_atoms.get_nxyz())[keep_idx, :] for old_atoms in self.old_atoms]}

        dset_0 = Dataset(props_0, do_copy=False)
        dset_1 = Dataset(props_1, do_copy=False)

        return dset_0, dset_1

    def rmsd_prelims(self, atoms):
        num_atoms = len(atoms)

        # f_damp is the turn-on on timescale, measured in
        # number of steps. From https://pubs.acs.org/doi/pdf/
        # 10.1021/acs.jctc.9b00143

        kappa = self.pushing_params["kappa"]
        steps_from_old = torch.Tensor(self.steps_from_old)
        f_damp = 2 / (1 + torch.exp(-kappa * steps_from_old)) - 1

        # given in mHartree / atom in CREST paper
        k_i = self.pushing_params["k_i"] / 1000 * units.Hartree * num_atoms

        # given in Bohr^(-2) in CREST paper
        alpha_i = self.pushing_params["alpha_i"] / units.Bohr**2

        dsets = self.make_dsets(atoms)

        return k_i, alpha_i, dsets, f_damp

    def rmsd_push(self, atoms):
        if not self.old_atoms:
            return np.zeros((len(atoms), 3)), 0.0

        k_i, alpha_i, dsets, f_damp = self.rmsd_prelims(atoms)

        delta_i, _, xyz_list = compute_distances(
            dataset=dsets[0],
            # do this on CPU - it's a small RMSD
            # and gradient calculation, so the
            # dominant time is data transfer to GPU.
            # Testing it out confirms that you get a
            # big slowdown from doing it on GPU
            device="cpu",
            # device=self.device,
            dataset_1=dsets[1],
            store_grad=True,
            collate_dicts=collate_dicts,
        )

        v_bias = (f_damp * k_i * torch.exp(-alpha_i * delta_i.reshape(-1) ** 2)).sum()

        f_bias = -compute_grad(inputs=xyz_list[0], output=v_bias).sum(0)

        keep_idx = self.get_keep_idx(atoms)
        final_f_bias = torch.zeros(len(atoms), 3)
        final_f_bias[keep_idx] = f_bias.detach().cpu()
        nan_idx = torch.bitwise_not(torch.isfinite(final_f_bias))
        final_f_bias[nan_idx] = 0

        return final_f_bias.detach().numpy(), v_bias.detach().numpy()

    def get_bias(self, atoms):
        bias_type = self.pushing_params["bias_type"]
        if bias_type == "rmsd":
            results = self.rmsd_push(atoms)
        else:
            raise NotImplementedError

        return results

    def append_atoms(self, atoms):
        self.old_atoms.append(atoms)
        self.steps_from_old.append(0)

        max_ref = self.pushing_params.get("max_ref_structures")
        if max_ref is None:
            max_ref = 10

        if len(self.old_atoms) >= max_ref:
            self.old_atoms = self.old_atoms[-max_ref:]
            self.steps_from_old = self.steps_from_old[-max_ref:]

    def calculate(
        self,
        atoms,
        properties=["energy", "forces"],
        system_changes=all_changes,
        add_steps=True,
    ):
        if not any([isinstance(self.model, i) for i in UNDIRECTED]):
            check_directed(self.model, atoms)

        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        # Add metadynamics energy and forces

        f_bias, _ = self.get_bias(atoms)

        self.results["forces"] += f_bias
        self.results["f_bias"] = f_bias

        if add_steps:
            for i, step in enumerate(self.steps_from_old):
                self.steps_from_old[i] = step + 1


class BatchNeuralMetadynamics(NeuralMetadynamics):
    def __init__(
        self,
        model,
        pushing_params,
        old_atoms=None,
        device="cpu",
        en_key="energy",
        directed=DEFAULT_DIRECTED,
        **kwargs,
    ):
        NeuralMetadynamics.__init__(
            self,
            model=model,
            pushing_params=pushing_params,
            old_atoms=old_atoms,
            device=device,
            en_key=en_key,
            directed=directed,
            **kwargs,
        )

        self.query_nxyz = None
        self.mol_idx = None

    def rmsd_prelims(self, atoms):
        # f_damp is the turn-on on timescale, measured in
        # number of steps. From https://pubs.acs.org/doi/pdf/
        # 10.1021/acs.jctc.9b00143

        kappa = self.pushing_params["kappa"]
        steps_from_old = torch.Tensor(self.steps_from_old)
        f_damp = 2 / (1 + torch.exp(-kappa * steps_from_old)) - 1

        # k_i depends on the number of atoms so must be done by batch
        # given in mHartree / atom in CREST paper

        k_i = self.pushing_params["k_i"] / 1000 * units.Hartree * atoms.num_atoms

        # given in Bohr^(-2) in CREST paper
        alpha_i = self.pushing_params["alpha_i"] / units.Bohr**2

        return k_i, alpha_i, f_damp

    def get_query_nxyz(self, keep_idx):
        if self.query_nxyz is not None:
            return self.query_nxyz

        query_nxyz = torch.stack([torch.Tensor(old_atoms.get_nxyz())[keep_idx, :] for old_atoms in self.old_atoms])
        self.query_nxyz = query_nxyz

        return query_nxyz

    def append_atoms(self, atoms):
        super().append_atoms(atoms)
        self.query_nxyz = None

    def make_nxyz(self, atoms):
        keep_idx = self.get_keep_idx(atoms)
        ref_nxyz = torch.Tensor(atoms.get_nxyz())[keep_idx, :]
        query_nxyz = self.get_query_nxyz(keep_idx)

        return ref_nxyz, query_nxyz, keep_idx

    def get_mol_idx(self, atoms, keep_idx):
        if self.mol_idx is not None:
            assert self.mol_idx.max() + 1 == len(atoms.num_atoms)
            return self.mol_idx

        num_atoms = atoms.num_atoms
        counter = 0

        mol_idx = []

        for i, num in enumerate(num_atoms):
            mol_idx.append(torch.ones(num).long() * i)
            counter += num

        mol_idx = torch.cat(mol_idx)[keep_idx]
        self.mol_idx = mol_idx

        return mol_idx

    def get_num_atoms_tensor(self, mol_idx, atoms):
        num_atoms = torch.LongTensor([(mol_idx == i).nonzero().shape[0] for i in range(len(atoms.num_atoms))])

        return num_atoms

    def get_v_f_bias(self, rmsd, ref_xyz, k_i, alpha_i, f_damp):
        v_bias = (f_damp.reshape(-1, 1) * k_i * torch.exp(-alpha_i * rmsd**2)).sum()

        f_bias = -compute_grad(inputs=ref_xyz, output=v_bias)

        output = [v_bias.reshape(-1).detach().cpu(), f_bias.detach().cpu()]

        return output

    def rmsd_push(self, atoms):
        if not self.old_atoms:
            return np.zeros((len(atoms), 3)), np.zeros(len(atoms.num_atoms))

        k_i, alpha_i, f_damp = self.rmsd_prelims(atoms)

        ref_nxyz, query_nxyz, keep_idx = self.make_nxyz(atoms=atoms)
        mol_idx = self.get_mol_idx(atoms=atoms, keep_idx=keep_idx)
        num_atoms_tensor = self.get_num_atoms_tensor(mol_idx=mol_idx, atoms=atoms)

        # note - everything is done on CPU, which is much faster than GPU. E.g. for
        # 30 molecules in a batch, each around 70 atoms, it's 4 times faster to do
        # this on CPU than GPU

        rmsd, ref_xyz = batch_compute_distance(
            ref_nxyz=ref_nxyz,
            query_nxyz=query_nxyz,
            mol_idx=mol_idx,
            num_atoms_tensor=num_atoms_tensor,
            store_grad=True,
        )

        v_bias, f_bias = self.get_v_f_bias(rmsd=rmsd, ref_xyz=ref_xyz, k_i=k_i, alpha_i=alpha_i, f_damp=f_damp)

        final_f_bias = torch.zeros(len(atoms), 3)
        final_f_bias[keep_idx] = f_bias
        nan_idx = torch.bitwise_not(torch.isfinite(final_f_bias))
        final_f_bias[nan_idx] = 0

        return final_f_bias.numpy(), v_bias.numpy()


class NeuralGAMD(NeuralFF):
    """
    NeuralFF for Gaussian-accelerated molecular dynamics (GAMD)
    """

    def __init__(
        self,
        model,
        k_0,
        V_min,
        V_max,
        device=0,
        en_key="energy",
        directed=DEFAULT_DIRECTED,
        **kwargs,
    ):
        NeuralFF.__init__(
            self,
            model=model,
            device=device,
            en_key=en_key,
            directed=DEFAULT_DIRECTED,
            **kwargs,
        )
        self.V_min = V_min
        self.V_max = V_max

        self.k_0 = k_0
        self.k = self.k_0 / (self.V_max - self.V_min)

    def calculate(self, atoms, properties=["energy", "forces"], system_changes=all_changes):
        if not any([isinstance(self.model, i) for i in UNDIRECTED]):
            check_directed(self.model, atoms)

        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        old_en = self.results["energy"]
        if old_en < self.V_max:
            old_forces = self.results["forces"]
            f_bias = -self.k * (self.V_max - old_en) * old_forces

            self.results["forces"] += f_bias


class ProjVectorCentroid:
    """
    Collective variable class. Projection of a position vector onto a reference vector
    Atomic indices are used to determine the coordiantes of the vectors.
    Params
    ------
    vector: list of int
       List of the indices of atoms that define the vector on which the position vector is projected
    indices: list if int
       List of indices of the mol/fragment
    reference: list of int
       List of atomic indices that are used as reference for the position vector

    note: the position vector is calculated in the method get_value
    """

    def __init__(self, vector=[], indices=[], reference=[], device="cpu"):
        self.vector_inds = vector
        self.mol_inds = torch.LongTensor(indices)
        self.reference_inds = reference

    def get_value(self, positions):
        vector_pos = positions[self.vector_inds]
        vector = vector_pos[1] - vector_pos[0]
        vector = vector / torch.linalg.norm(vector)
        mol_pos = positions[self.mol_inds]
        reference_pos = positions[self.reference_inds]
        mol_centroid = mol_pos.mean(axis=0)  # mol center
        reference_centroid = reference_pos.mean(axis=0)  # centroid of the whole structure

        # position vector with respect to the structure centroid
        rel_mol_pos = mol_centroid - reference_centroid

        # projection
        cv = torch.dot(rel_mol_pos, vector)
        return cv


class ProjVectorPlane:
    """
    Collective variable class. Projection of a position vector onto a the average plane
    of an arbitrary ring defined in the structure
    Atomic indices are used to determine the coordiantes of the vectors.
    Params
    ------
    mol_inds: list of int
       List of indices of the mol/fragment tracked by the CV
    ring_inds: list of int
       List of atomic indices of the ring for which the average plane is calculated.

    note: the position vector is calculated in the method get_value
    """

    def __init__(self, mol_inds=[], ring_inds=[]):
        self.mol_inds = torch.LongTensor(mol_inds)  # list of indices
        self.ring_inds = torch.LongTensor(ring_inds)  # list of indices
        # both self.mol_coors and self.ring_coors torch tensors with atomic coordinates
        # initiallized as list but will be set to torch tensors with set_positions
        self.mol_coors = []
        self.ring_coors = []

    def set_positions(self, positions):
        # update coordinate torch tensors from the positions tensor
        self.mol_coors = positions[self.mol_inds]
        self.ring_coors = positions[self.ring_inds]

    def get_indices(self):
        return self.mol_inds + self.ring_inds

    def get_value(self, positions):
        """Calculates the values of the CV for a specific atomic positions

        Args:
            positions (torch tensor): atomic positions

        Returns:
            float: current values of the collective variable
        """
        self.set_positions(positions)
        mol_cm = self.mol_coors.mean(axis=0)  # mol center
        ring_cm = self.ring_coors.mean(axis=0)  # ring center
        # ring atoms to center
        self.ring_coors = self.ring_coors - ring_cm

        r1 = torch.zeros(3, device=self.ring_coors.device)
        N = len(self.ring_coors)  # number of atoms in the ring
        for i, rl0 in enumerate(self.ring_coors):
            r1 = r1 + rl0 * np.sin(2 * np.pi * i / N)
        r1 = r1 / N

        r2 = torch.zeros(3, device=self.ring_coors.device)
        for i, rl0 in enumerate(self.ring_coors):
            r2 = r2 + rl0 * np.cos(2 * np.pi * i / N)
        r2 = r2 / N

        plane_vec = torch.cross(r1, r2)
        plane_vec = plane_vec / torch.linalg.norm(plane_vec)
        pos_vec = mol_cm - ring_cm

        cv = torch.dot(pos_vec, plane_vec)
        return cv


class ProjOrthoVectorsPlane:
    """
    Collective variable class. Projection of a position vector onto a the average plane
    of an arbitrary ring defined in the structure
    Atomic indices are used to determine the coordiantes of the vectors.
    Params
    ------
    mol_inds: list of int
       List of indices of the mol/fragment tracked by the CV
    ring_inds: list of int
       List of atomic indices of the ring for which the average plane is calculated.

    note: the position vector is calculated in the method get_value
    """

    def __init__(self, mol_inds=[], ring_inds=[]):
        self.mol_inds = torch.LongTensor(mol_inds)  # list of indices
        self.ring_inds = torch.LongTensor(ring_inds)  # list of indices
        # both self.mol_coors and self.ring_coors torch tensors with atomic coordinates
        # initiallized as list but will be set to torch tensors with set_positions
        self.mol_coors = []
        self.ring_coors = []

    def set_positions(self, positions):
        # update coordinate torch tensors from the positions tensor
        self.mol_coors = positions[self.mol_inds]
        self.ring_coors = positions[self.ring_inds]

    def get_indices(self):
        return self.mol_inds + self.ring_inds

    def get_value(self, positions):
        """Calculates the values of the CV for a specific atomic positions

        Args:
            positions (torch tensor): atomic positions

        Returns:
            float: current values of the collective variable
        """
        self.set_positions(positions)
        mol_cm = self.mol_coors.mean(axis=0)  # mol center
        ring_cm = self.ring_coors.mean(axis=0)  # ring center
        # ring atoms to center
        self.ring_coors = self.ring_coors - ring_cm

        r1 = torch.zeros(3, device=self.ring_coors.device)
        N = len(self.ring_coors)  # number of atoms in the ring
        for i, rl0 in enumerate(self.ring_coors):
            r1 = r1 + rl0 * np.sin(2 * np.pi * i / N)
        r1 = r1 / N

        r2 = torch.zeros(3, device=self.ring_coors.device)
        for i, rl0 in enumerate(self.ring_coors):
            r2 = r2 + rl0 * np.cos(2 * np.pi * i / N)
        r2 = r2 / N

        # normalize r1 and r2
        r1 = r1 / torch.linalg.norm(r1)
        r2 = r2 / torch.linalg.norm(r2)
        # project position vector on r1 and r2
        pos_vec = mol_cm - ring_cm
        proj1 = torch.dot(pos_vec, r1)
        proj2 = torch.dot(pos_vec, r2)
        cv = proj1 + proj2
        return abs(cv)


class HarmonicRestraint:
    """Class to apply a harmonic restraint on a MD simulations
    Params
    ------
    cvdic (dict): Dictionary contains the information to define the collective variables
                  and the harmonic restraint.
    max_steps (int): maximum number of steps of the MD simulation
    device: device
    """

    def __init__(self, cvdic, max_steps, device="cpu"):
        self.cvs = []  # list of collective variables (CV) objects
        self.kappas = []  # list of constant values for every CV
        self.eq_values = []  # list equilibrium values for every CV
        self.steps = []  # list of lists with the steps of the MD simulation
        self.setup_contraint(cvdic, max_steps, device)

    def setup_contraint(self, cvdic, max_steps, device):
        """Initializes the collectiva variables objects
        Args:
            cvdic (dict): Dictionary contains the information to define the collective variables
                          and the harmonic restraint.
            max_steps (int): maximum number of steps of the MD simulation
            device: device
        """
        for cvname, val in cvdic.items():
            if val["type"].lower() == "proj_vec_plane":
                mol_inds = [i - 1 for i in val["mol"]]  # caution check type
                ring_inds = [i - 1 for i in val["ring"]]
                cv = ProjVectorPlane(mol_inds, ring_inds)
            elif val["type"].lower() == "proj_vec_ref":
                mol_inds = [i - 1 for i in val["mol"]]
                reference = [i - 1 for i in val["reference"]]
                vector = [i - 1 for i in val["vector"]]
                cv = ProjVectorCentroid(vector, mol_inds, reference, device=device)  # z components
            elif val["type"].lower() == "proj_ortho_vectors_plane":
                mol_inds = [i - 1 for i in val["mol"]]  # caution check type
                ring_inds = [i - 1 for i in val["ring"]]
                cv = ProjOrthoVectorsPlane(mol_inds, ring_inds)  # z components
            else:
                print("Bad CV type")
                sys.exit(1)

            self.cvs.append(cv)
            steps, kappas, eq_values = self.create_time_dependec_arrays(val["restraint"], max_steps)
            self.kappas.append(kappas)
            self.steps.append(steps)
            self.eq_values.append(eq_values)

    def create_time_dependec_arrays(self, restraint_list, max_steps):
        """creates lists of steps, kappas and equilibrium values that will be used along
        the simulation to determine the value of kappa and equilibrium CV at each step

        Args:
            restraint_list (list of dicts): contains dictionaries with the desired values of
                                            kappa and CV at arbitrary step in the simulation
            max_steps (int): maximum number of steps of the simulation

        Returns:
            list: all steps e.g [1,2,3 .. max_steps]
            list: all kappas for every step, e.g [0.5, 0.5, 0.51, .. ] same size of steps
            list: all equilibrium values for every step, e.g. [1,1,1,3,3,3, ...], same size of steps
        """
        steps = []
        kappas = []
        eq_vals = []
        # in case the restraint does not start at 0
        templist = list(range(0, restraint_list[0]["step"]))
        steps += templist
        kappas += [0 for _ in templist]
        eq_vals += [0 for _ in templist]

        for n, rd in enumerate(restraint_list[1:]):
            # rd and n are out of phase by 1, when n = 0, rd points to index 1
            templist = list(range(restraint_list[n]["step"], rd["step"]))
            steps += templist
            kappas += [restraint_list[n]["kappa"] for _ in templist]
            dcv = rd["eq_val"] - restraint_list[n]["eq_val"]
            cvstep = dcv / len(templist)  # get step increase
            eq_vals += [restraint_list[n]["eq_val"] + cvstep * tind for tind, _ in enumerate(templist)]

        # in case the last step is lesser than the max_step
        templist = list(range(restraint_list[-1]["step"], max_steps))
        steps += templist
        kappas += [restraint_list[-1]["kappa"] for _ in templist]
        eq_vals += [restraint_list[-1]["eq_val"] for _ in templist]

        return steps, kappas, eq_vals

    def get_energy(self, positions, step):
        """Calculates the retraint energy of an arbritrary state of the system

        Args:
            positions (torch.tensor): atomic positions
            step (int): current step

        Returns:
            float: energy
        """
        tot_energy = 0
        for n, cv in enumerate(self.cvs):
            kappa = self.kappas[n][step]
            eq_val = self.eq_values[n][step]
            cv_value = cv.get_value(positions)
            energy = 0.5 * kappa * (cv_value - eq_val) * (cv_value - eq_val)
            tot_energy += energy
        return tot_energy

    def get_bias(self, positions, step):
        """Calculates the bias energy and force

        Args:
            positions (torch.tensor): atomic positions
            step (int): current step

        Returns:
            float: forces
            float: energy
        """
        energy = self.get_energy(positions, step)
        forces = grad(energy, positions)[0]
        return forces, energy


class NeuralRestraint(Calculator):
    """ASE calculator to run Neural restraint MD simulations"""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model,
        device="cpu",
        en_key="energy",
        properties=["energy", "forces"],
        cv={},
        max_steps=0,
        **kwargs,
    ):
        """Creates a NeuralFF calculator.nff/io/ase.py

        Args:
            model (TYPE): Description
            device (str): device on which the calculations will be performed
            properties (list of str): 'energy', 'forces' or both and also stress for only schnet and painn
            **kwargs: Description
            model (one of nff.nn.models)
        """

        Calculator.__init__(self, **kwargs)
        self.model = model
        self.model.eval()
        self.device = device
        self.to(device)
        self.en_key = en_key
        self.step = -1
        self.max_steps = max_steps
        self.properties = properties
        self.hr = HarmonicRestraint(cv, max_steps, device)

    def to(self, device):
        self.device = device
        self.model.to(device)

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        """Calculates the desired properties for the given AtomsBatch.

        Args:
            atoms (AtomsBatch): custom Atoms subclass that contains implementation
                of neighbor lists, batching and so on. Avoids the use of the Dataset
                to calculate using the models created.
            system_changes (default from ase)
        """

        # print("calculating ...")
        self.step += 1
        # print("step ", self.step, self.step*0.0005)
        if not any([isinstance(self.model, i) for i in UNDIRECTED]):
            check_directed(self.model, atoms)

        # for backwards compatability
        if getattr(self, "properties", None) is None:
            self.properties = properties

        Calculator.calculate(self, atoms, self.properties, system_changes)

        # run model
        batch = batch_to(atoms.get_batch(), self.device)

        # add keys so that the readout function can calculate these properties
        grad_key = self.en_key + "_grad"
        batch[self.en_key] = []
        batch[grad_key] = []

        kwargs = {}
        requires_stress = "stress" in self.properties
        if requires_stress:
            kwargs["requires_stress"] = True
        prediction = self.model(batch, **kwargs)

        # change energy and force to kcal/mol
        energy = prediction[self.en_key] * (1 / const.EV_TO_KCAL_MOL)
        energy_grad = prediction[grad_key] * (1 / const.EV_TO_KCAL_MOL)

        # bias ------------------
        bias_forces, bias_energy = self.hr.get_bias(
            torch.tensor(atoms.get_positions(), requires_grad=True, device=self.device),
            self.step,
        )
        energy_grad += bias_forces

        # change energy and force to numpy array
        energy_grad = energy_grad.detach().cpu().numpy()
        energy = energy.detach().cpu().numpy()

        self.results = {"energy": energy.reshape(-1)}

        if "forces" in self.properties:
            self.results["forces"] = -energy_grad.reshape(-1, 3)
        if requires_stress:
            stress = prediction["stress_volume"].detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)
            self.results["stress"] = stress * (1 / atoms.get_volume())

        with open("colvar", "a") as f:
            f.write("{} ".format(self.step * 0.5))
            # ARREGLAR, SI YA ESTA CALCULADO PARA QUE RECALCULAR LA CVS
            for cv in self.hr.cvs:
                curr_cv_val = float(cv.get_value(torch.tensor(atoms.get_positions(), device=self.device)))
                f.write(" {:.6f} ".format(curr_cv_val))
            f.write("{:.6f} \n".format(float(bias_energy)))

    @classmethod
    def from_file(cls, model_path, device="cuda", **kwargs):
        model = load_model(model_path)
        out = cls(model=model, device=device, **kwargs)
        return out
