import sys
sys.path.append('/home/saxelrod/htvs-ax/htvs')

import os
import django

os.environ["DJANGO_SETTINGS_MODULE"]="djangochem.settings.orgel"
django.setup()

# Shell Plus Model Imports


from django.contrib.contenttypes.models import ContentType

from jobs.models import Job, JobConfig

from django.contrib.auth.models import Group
from pgmols.models import (AtomBasis, 
                           Geom, Hessian, Jacobian, MDFrame, Mechanism, Method, Mol, MolGroupObjectPermission,
                           MolSet, MolUserObjectPermission, PathImage, ProductLink, ReactantLink, Reaction,
                           ReactionPath, ReactionType, SinglePoint, Species, Stoichiometry, Trajectory)




import numpy as np
import random
import pdb
import json
from rdkit import Chem
from torch.utils.data import DataLoader
import copy
from django.contrib.contenttypes.models import ContentType
from django.utils import timezone


from ase.calculators.calculator import Calculator
from ase import optimize, Atoms, units
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory as AseTrajectory


# from nff.nn.models import PostProcessModel
from nff.io.ase_ax import NeuralFF, AtomsBatch
from nff.train import load_model
from nff.utils import constants as const
from nff.nn.tensorgrad import get_schnet_hessians
from nff.data import collate_dicts, Dataset
from nff.utils.cuda import batch_to

from neuralnet.utils import vib



KT = 0.000944853
FS_TO_AU =  41.341374575751
AU_TO_ANGS = 0.529177
CM_TO_AU = 4.5564e-6
AU_TO_KELVIN = 317638

DEFAULT_OPT_METHOD = 'nn_ci_opt_sf_tddft_bhhlyp'
DEFAULT_MD_METHOD = 'nn_ci_dynamics_sf_tddft_bhhlyp'


DEFAULT_OPT_CONFIG = 'nn_ci_opt'
DEFAULT_MD_CONFIG = 'nn_ci_dynamics'



DEFAULT_GROUP = 'switches'
DEFAULT_TEMP = 300
DEFAULT_PENALTY = 0.5
DEFAULT_CI_OPT_TYPE = 'BFGS'
DEFAULT_MAX_STEPS = 500
WEIGHT_FOLDER = '/home/saxelrod/engaging/models'
GROUP_NAME = 'switches'

PERIODICTABLE = Chem.GetPeriodicTable()


# BASE_NXYZ = np.array([[9.0, 1.626, -1.2352, -2.1575],
#  [6.0, 1.9869, -0.4611, -1.1023],
#  [6.0, 2.4846, 0.8267, -1.2586],
#  [6.0, 2.8629, 1.5495, -0.1401],
#  [6.0, 2.8794, 0.9444, 1.1089],
#  [6.0, 2.314, -0.313, 1.2809],
#  [9.0, 2.5912, -1.0202, 2.3902],
#  [6.0, 1.6616, -0.8408, 0.1753],
#  [7.0, 0.828, -1.9748, 0.448],
#  [7.0, -0.2528, -1.678, 1.0679],
#  [6.0, -1.087, -0.7575, 0.3522],
#  [6.0, -1.8388, -1.3056, -0.6806],
#  [6.0, -3.0777, -0.8041, -1.0307],
#  [6.0, -3.4558, 0.4394, -0.5895],
#  [6.0, -2.6664, 1.0449, 0.3605],
#  [6.0, -1.6479, 0.3386, 0.9823],
#  [1.0, 2.4427, 1.2813, -2.2382],
#  [1.0, 3.1273, 2.6058, -0.2052],
#  [1.0, 3.2671, 1.4542, 1.978],
#  [1.0, -1.5848, -2.2797, -1.0655],
#  [1.0, -3.8298, -1.3837, -1.5617],
#  [1.0, -4.364, 0.9446, -0.9216],
#  [1.0, -2.9646, 1.969, 0.8442],
#  [1.0, -1.3019, 0.6553, 1.9634]])
BASE_NXYZ = np.array([[6.0, 4.452, -0.5003, 0.3975],
					 [6.0, 3.6787, -1.4613, 1.0474],
					 [6.0, 2.2963, -1.29, 1.1518],
					 [6.0, 1.673, -0.1649, 0.6015],
					 [7.0, 0.2828, -0.0077, 0.7513],
					 [7.0, -0.2784, 0.0488, -0.3654],
					 [6.0, -1.6699, 0.1935, -0.2155],
					 [6.0, -2.2349, 1.31, 0.4161],
					 [6.0, -3.6213, 1.4419, 0.5226],
					 [6.0, -4.4562, 0.4614, -0.0113],
					 [6.0, -3.9067, -0.6474, -0.6535],
					 [6.0, -2.5197, -0.7764, -0.758],
					 [6.0, 2.4631, 0.8004, -0.0379],
					 [6.0, 3.8456, 0.6321, -0.1442],
					 [1.0, 5.5279, -0.6325, 0.3158],
					 [1.0, 4.1501, -2.3426, 1.4743],
					 [1.0, 1.6958, -2.0383, 1.6624],
					 [1.0, -1.5867, 2.0791, 0.8278],
					 [1.0, -4.0473, 2.3094, 1.0197],
					 [1.0, -5.5355, 0.5628, 0.0706],
					 [1.0, -4.5559, -1.4108, -1.0743],
					 [1.0, -2.0934, -1.6394, -1.2627],
					 [1.0, 1.9939, 1.6872, -0.4558],
					 [1.0, 4.4467, 1.3848, -0.6474]])

# BASE_NXYZ = np.array([[6.0, -2.9158, -0.8555, 0.8318],
#  [6.0, -2.8923, -0.9434, -0.5533],
#  [6.0, -2.3179, 0.1365, -1.2153],
#  [6.0, -1.5066, 1.0489, -0.5507],
#  [7.0, -0.557, 1.7803, -1.3222],
#  [7.0, 0.3693, 1.0616, -1.8394],
#  [6.0, 1.1488, 0.374, -0.8505],
#  [6.0, 1.6462, -0.8982, -1.0729],
#  [6.0, 2.5616, -1.4677, -0.1954],
#  [6.0, 3.1468, -0.6729, 0.7641],
#  [6.0, 3.0255, 0.7069, 0.6335],
#  [6.0, 1.9038, 1.1984, -0.0226],
#  [6.0, -1.6495, 1.1864, 0.8133],
#  [6.0, -2.4421, 0.2665, 1.4903],
#  [1.0, -3.4773, -1.6338, 1.332],
#  [1.0, -3.0551, -1.9037, -1.016],
#  [1.0, -2.2432, 0.0943, -2.294],
#  [1.0, 1.0876, -1.5393, -1.7434],
#  [1.0, 2.6524, -2.5394, -0.0807],
#  [1.0, 3.6078, -1.0611, 1.6671],
#  [1.0, 3.6938, 1.3734, 1.1781],
#  [1.0, 1.5802, 2.2304, 0.1155],
#  [1.0, -0.9201, 1.7747, 1.3505],
#  [1.0, -2.4469, 0.2829, 2.5802]])

OPT_DIC = {
	"BFGS": optimize.BFGS
}

def get_new_model(model, lower_key, upper_key, ref_energy, penalty):

	# process_list = [{"func_name": "gap",
	# 				 # hack to get ase to think this is the energy
	# 				 # for minimization
	# 				 "output_name": "energy",
	# 				 "params": {"lower_key": lower_key,
	# 							"upper_key": upper_key}},

	# 				{"func_name": "gap_grad",
	# 				 "output_name": "energy_grad",
	# 				 "params": {"lower_key": lower_key,
	# 							"upper_key": upper_key}}] 

	if ref_energy is None:
		ref_energy = 0

	process_list = [{"func_name": "gap_penalty",
					 # hack to get ase to think this is the energy
					 # for minimization
					 "output_name": "energy",
					 "params": {"lower_key": lower_key,
								"upper_key": upper_key,
								"ref_energy": ref_energy,
								"penalty": penalty
					}},

					{"func_name": "gap_penalty_grad",
					 "output_name": "energy_grad",
					 "params": {"lower_key": lower_key,
								"upper_key": upper_key,
								"penalty": penalty

						}}

						] 

	base_keys = [lower_key, upper_key, lower_key + "_grad",
				 upper_key + "_grad"]


	new_model = PostProcessModel(model=model, process_list=process_list,
					base_keys=base_keys)

	return new_model


def set_ci_calc(atoms, model, lower_key, upper_key,
				**kwargs):

	new_model = get_new_model(model=model,
							  lower_key=lower_key,
							  upper_key=upper_key,
							  **kwargs)

	ci_calculator = NeuralFF(
	    model=new_model,
	    **kwargs)
	atoms.set_calculator(ci_calculator)

def pdb_wrap(func):
	def main(*args, **kwargs):
		try:
			return func(*args, **kwargs)
		except Exception as e:
			print(e)
			pdb.post_mortem()
	return main

def opt_ci(
		  model,
		  nxyz,
		  penalty=0.5,
		  lower_idx=0,
		  upper_idx=1,
		  method='BFGS',
		  steps=500,
		  **kwargs):

	atoms = AtomsBatch(BASE_NXYZ[:, 0], BASE_NXYZ[:, 1:])
	init_calc = NeuralFF(model=model, output_keys=['energy_0'])
	atoms.set_calculator(init_calc)
	ref_energy = atoms.get_potential_energy().item(
		) * const.EV_TO_KCAL_MOL


	lower_key = "energy_{}".format(lower_idx)
	upper_key = "energy_{}".format(upper_idx)

	set_ci_calc(atoms=atoms, model=model,
				lower_key=lower_key, upper_key=upper_key,
				ref_energy=ref_energy, penalty=penalty)

	dyn = OPT_DIC[method](atoms)
	dyn.run(steps=steps)

	return atoms

def get_modes(model, nxyz, cutoff, energy_keys, device=0):

	props = {"nxyz": [nxyz], **{key: [0] for key in energy_keys}}
	dataset = Dataset(props.copy())
	dataset.generate_neighbor_list(cutoff)
	loader = DataLoader(dataset, batch_size=1, collate_fn=collate_dicts)
	batch = next(iter(loader))
	batch = batch_to(batch, device)
	model = model.to(device)


	w_list = []
	orth_list = []

	for key in energy_keys:
		hessian = get_schnet_hessians(batch=batch, model=model, device=device,
								energy_key=key)[0].cpu().detach().numpy()

		# convert to Ha / bohr^2
		hessian *= (const.BOHR_RADIUS) ** 2
		hessian *= const.KCAL_TO_AU['energy']

		force_consts, vib_freqs, eigvec = vib.vib_analy(
			r=nxyz[:, 0], xyz=nxyz[:, 1:],
			hessian=hessian)

		w_list.append(vib_freqs * CM_TO_AU)
		orth_list.append(np.array(eigvec))

	return w_list, orth_list

def normal_to_real(orth, mass_vec, x=None, ref_geom=None, p=None):

	new_orth = np.transpose(orth)
	pos = None
	mom = None

	if x is not None:
		mass_pos = np.matmul(new_orth, x_t).reshape(-1, 3)
		pos = (mass_pos / (mass_vec.reshape(-1, 1)) ** 0.5)
		# convert to angstrom
		pos *= const.BOHR_RADIUS 

		# add converged geometry
		pos += ref_geom

	if p is not None:

		mass_mom = np.matmul(new_orth, p).reshape(-1, 3)
		mom = (mass_mom * (mass_vec.reshape(-1, 1)) ** 0.5)

	return pos, mom

def sample_p(w, orth, mass_vec, kt=KT):
	dim = len(w)
	cov = kt * np.identity(dim)
	p_normal = np.random.multivariate_normal(mean=np.zeros(dim), cov=cov)
	_, p_real = normal_to_real(orth=orth, mass_vec=mass_vec, p=p_normal)
	p_real *= 1/ (1 / FS_TO_AU * units.fs ) * AU_TO_ANGS / const.AMU_TO_AU

	return p_real

def sample_ci(ci_atoms, model, cutoff, energy_keys, device=0, kt=KT):

	nxyz = ci_atoms.get_nxyz()
	mass_vec = np.array([PERIODICTABLE.GetAtomicWeight(int(element[0])) * const.AMU_TO_AU
			for element in nxyz])

	w_list, orth_list = get_modes(model=model, nxyz=nxyz, cutoff=cutoff,
						energy_keys=energy_keys, device=device)


	lower_atoms = copy.deepcopy(ci_atoms)
	upper_atoms = copy.deepcopy(ci_atoms)

	# pdb.set_trace()

	for i, atoms in enumerate([lower_atoms, upper_atoms]):

		# ignore negative w modes

		w = w_list[i]
		orth = orth_list[i]

		good_idx = [i for i, w_1d in enumerate(w) if w_1d > 0]
		new_w = w[good_idx]
		new_orth = orth[good_idx, :]

		p = sample_p(w=new_w, orth=new_orth, mass_vec=mass_vec, kt=kt)
		atoms.set_momenta(p)

	return lower_atoms, upper_atoms

def opt_and_sample_ci(model,
					  nxyz,
					  penalty=0.5,
					  lower_idx=0,
					  upper_idx=1,
					  method='BFGS',
					  steps=500,
					  cutoff=5.0,
					  device=0,
					  kt=KT,
					  **kwargs):
	
	ci_atoms = opt_ci(
		  model=model,
		  nxyz=nxyz,
		  penalty=penalty,
		  lower_idx=lower_idx,
		  upper_idx=upper_idx,
		  method=method,
		  steps=steps)

	energy_keys = ["energy_{}".format(lower_idx), "energy_{}".format(upper_idx)]

	lower_atoms, upper_atoms = sample_ci(
		ci_atoms=ci_atoms,
		model=model,
		cutoff=cutoff,
		energy_keys=energy_keys,
		device=device,
		kt=KT)

	return lower_atoms, upper_atoms


	
	
def test():
	# weightpath = "/home/saxelrod/engaging/models/971"
	weightpath = "/home/saxelrod/engaging/models/953"
	nxyz = BASE_NXYZ
	penalty = 0.5
	# atoms = opt_ci(weightpath=weightpath, nxyz=nxyz,
	# 			   penalty=penalty)

	model = load_model(weightpath)
	lower_idx = 0
	upper_idx = 1
	lower_atoms, upper_atoms = opt_and_sample_ci(model=model,
						  nxyz=nxyz,
						  penalty=DEFAULT_PENALTY,
						  lower_idx=lower_idx,
						  upper_idx=upper_idx,
						  method='BFGS',
						  steps=100,
						  cutoff=5.0,
						  device=0,
						  kt=KT)

	lower_calc = NeuralFF(model=model, output_keys=["energy_{}".format(lower_idx)])
	upper_calc = NeuralFF(model=model, output_keys=["energy_{}".format(upper_idx)])

	lower_atoms.set_calculator(lower_calc)
	upper_atoms.set_calculator(upper_calc)

	lower_integrator = VelocityVerlet(lower_atoms, dt=units.fs, logfile='test_lower.log',
					trajectory='test_lower.traj')
	lower_integrator.run(1000)

	upper_integrator = VelocityVerlet(upper_atoms, dt=units.fs, logfile='test_upper.log',
					trajectory='test_upper.traj')
	upper_integrator.run(1000)

def run_ci_md(model, lower_atoms, upper_atoms, lower_idx, upper_idx, 
			  base_name='test', dt=0.5, tmax=500):

	lower_calc = NeuralFF(model=model, output_keys=["energy_{}".format(lower_idx)])
	upper_calc = NeuralFF(model=model, output_keys=["energy_{}".format(upper_idx)])

	lower_atoms.set_calculator(lower_calc)
	upper_atoms.set_calculator(upper_calc)

	lower_log = "{}_lower.log".format(base_name)
	lower_trj_name = "{}_lower.traj".format(base_name)
	num_steps = int(tmax/dt)

	lower_integrator = VelocityVerlet(lower_atoms, dt=dt * units.fs, logfile=lower_log,
					trajectory=lower_trj_name)
	lower_integrator.run(num_steps)



	upper_log = "{}_upper.log".format(base_name)
	upper_trj_name = "{}_upper.traj".format(base_name)
	upper_integrator = VelocityVerlet(upper_atoms, dt=dt * units.fs, logfile=upper_log,
					trajectory=upper_trj_name)
	upper_integrator.run(num_steps)


	lower_trj = AseTrajectory(lower_trj_name)
	upper_trj = AseTrajectory(upper_trj_name)

	return lower_trj, upper_trj


def make_geom(method, job, coords, parentgeom):
    geom = Geom(method=method,
                parentjob=job)

    geom.set_coords(coords)
    geom.converged = False
    geom.species = parentgeom.species
    geom.stoichiometry = parentgeom.stoichiometry
    # geom.details = {'temp_hessian': sampletemp}
    geom.save()

    geom.parents.add(parentgeom)
    return geom

def make_coords(nxyz):
    coords = []
    for i in range(len(nxyz)):
        number, x, y, z = nxyz[i]
        element = PERIODICTABLE.GetElementSymbol(int(number))
        coords.append(dict(element=element, x=x, y=y, z=z))

    return coords

def to_db(smiles, nnid, num_samples, group_name=GROUP_NAME,
			weight_folder=WEIGHT_FOLDER, penalty=DEFAULT_PENALTY,
			lower_idx=0, upper_idx=1,
			max_opt_steps=DEFAULT_MAX_STEPS, cutoff=5.0,
			device=0, kt=KT, lower_trj='test_lower.traj',
			upper_trj='test_upper.traj', dt=0.5, tmax=500,
			ci_opt_type=DEFAULT_CI_OPT_TYPE,
			md_method_name=DEFAULT_MD_METHOD, md_config_name=DEFAULT_MD_CONFIG,
			opt_method_name=DEFAULT_OPT_METHOD, opt_config_name=DEFAULT_OPT_CONFIG):

	
	group = Group.objects.get(name=group_name)
	parentgeom = Geom.objects.filter(species__smiles=smiles, species__group=group,
									  converged=True).first()
	nxyz = parentgeom.xyz
	weightpath = os.path.join(weight_folder, str(nnid))
	model = load_model(weightpath)

	lower_atoms, upper_atoms = opt_and_sample_ci(model=model,
						  nxyz=nxyz,
						  penalty=penalty,
						  lower_idx=lower_idx,
						  upper_idx=upper_idx,
						  method=ci_opt_type,
						  steps=max_opt_steps,
						  cutoff=cutoff,
						  device=device,
						  kt=kt)


	opt_details = {"penalty": penalty,
				   "lower_state": lower_idx,
				   "upper_state": upper_idx,
				   "method": ci_opt_type,
				   "max_steps": max_opt_steps,
				   "cutoff": cutoff,
				   "temp": round(kt * AU_TO_KELVIN, 2),
				   "nnid": nnid}




	opt_method, new_method = Method.objects.get_or_create(name=opt_method_name,
		description='generated with opt_ci code for optimizing conical intersections'
		)


	opt_config, new_jc = JobConfig.objects.get_or_create(name=opt_config_name,
	                                                    parent_class_name='Geom',
	                                                    configpath='None')

	
	opt_job = Job(config=opt_config,
	          status='done',
	          group=group,
	          parentct=ContentType.objects.get_for_model(parentgeom),
	          parentid=parentgeom.id,
	          completetime=timezone.now()
	          )

	opt_job.details = opt_details
	opt_job.save()


	ci_nxyz = lower_atoms.get_nxyz()
	coords = make_coords(ci_nxyz)
	ci_geom = make_geom(method=opt_method,
			  job=opt_job,
			  coords=coords,
			  parentgeom=parentgeom)


	######
	######
	######



	

	lower_trj, upper_trj = run_ci_md(model=model,
				  lower_atoms=lower_atoms,
				  upper_atoms=upper_atoms,
				  lower_idx=lower_idx,
				  upper_idx=upper_idx, 
				  base_name=parentgeom.id,
				  dt=dt,
				  tmax=tmax)


	

	md_details = {"thermostat": "velocity_verlet",
				  "dt": dt,
				  "tmax": tmax,
				  "lower_state": lower_idx,
				  "upper_state": upper_idx,
				  "nnid": nnid}

	md_method, new_method = Method.objects.get_or_create(name=md_method_name,
	                       description='generated with neural dynamics around CI')
	md_config, new_jc = JobConfig.objects.get_or_create(name=md_config_name,
	                                                    parent_class_name='Geom',
	                                                    configpath='None')
	md_job = Job(config=md_config,
	          status='done',
	          group=group,
	          parentct=ContentType.objects.get_for_model(parentgeom),
	          parentid=parentgeom.id,
	          completetime=timezone.now()
	          )

	md_job.details = md_details
	md_job.save()

	lower_key = "energy_{}".format(lower_idx)
	upper_key = "energy_{}".format(upper_idx)
	best_atoms = []

	# pdb.set_trace()

	gap_pairs = []
	best_atoms = []
	i = 0

	for trj in [lower_trj, upper_trj]:
		for atoms in trj:
			set_ci_calc(atoms=atoms, model=model,
				lower_key=lower_key, upper_key=upper_key,
				ref_energy=0, penalty=0)

			gap = atoms.get_potential_energy().item()
			gap_pairs.append([gap, i])
			best_atoms.append(atoms)

			i += 1

			# # exclude > 1 eV
			# if gap < 0.8:
			# # if gap < 10000:
			# 	best_atoms.append(atoms)
			# # break

	# random.shuffle(best_atoms)

	sorted_idx = [item[-1] for item in sorted(gap_pairs)]
	best_atoms = [best_atoms[i] for i in sorted_idx]

	for atoms in best_atoms[:num_samples]:
		nxyz = AtomsBatch(atoms).get_nxyz()
		coords = make_coords(nxyz)
		new_geom = make_geom(method=md_method,
							job=md_job,
							coords=coords,
							parentgeom=ci_geom)


@pdb_wrap
def main():
	# smiles = 'c1ccc(/N=N\\c2ccccc2)cc1'
	smiles = 'c1ccc(/N=N/c2ccccc2)cc1'
	nnid = 953
	num_samples = 1000
	to_db(smiles=smiles, nnid=nnid, num_samples=num_samples)


def make_plots():
	smiles = 'c1ccc(/N=N\\c2ccccc2)cc1'
	group = Group.objects.get(name='switches')
	parentgeom = Geom.objects.filter(species__smiles=smiles, species__group=group,
									  converged=True).first()
	trj_name = "{}_upper.traj".format(parentgeom.id)
	print(trj_name)

	return
	
	trj = AseTrajectory(trj_name)

	lower_key = "energy_0"
	upper_key = "energy_1"

	nxyz_list = []
	gap_list = []
	model = load_model("/home/saxelrod/engaging/models/953")

	for atoms in trj:
		nxyz = AtomsBatch(atoms).get_nxyz()
		nxyz_list.append(nxyz.tolist())

		set_ci_calc(atoms=atoms, model=model,
					lower_key=lower_key, upper_key=upper_key,
					ref_energy=0, penalty=0)

		gap = atoms.get_potential_energy().item()
		gap_list.append(gap)

	nxyz_save = 'demo.json'
	gap_save = 'demo_gap.json'

	with open(nxyz_save, "w") as f:
		json.dump(nxyz_list, f)

	with open(gap_save, "w") as f:
		json.dump(gap_list, f)


if __name__ == "__main__":
	# main()
	make_plots()







