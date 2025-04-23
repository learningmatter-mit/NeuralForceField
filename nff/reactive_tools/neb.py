import copy

from ase.io import read
from ase.neb import NEB
from ase.optimize import BFGS

from nff.io.ase import AtomsBatch
from nff.io.ase_calcs import NeuralFF
from nff.reactive_tools.utils import xyz_to_ase_atoms


def neural_neb_ase(
    reactantxyzfile,
    productxyzfile,
    nff_dir,
    rxn_name,
    steps=500,
    n_images=24,
    fmax=0.004,
    isclimb=False,
):
    # reactant and products as ase Atoms
    initial = AtomsBatch(xyz_to_ase_atoms(reactantxyzfile), cutoff=5.5, directed=True)

    final = AtomsBatch(xyz_to_ase_atoms(productxyzfile), cutoff=5.5, directed=True)

    # Make a band consisting of n_images:
    images = [initial]
    images += [copy.deepcopy(initial) for i in range(n_images)]
    images += [final]
    neb = NEB(images, k=0.02, climb=isclimb, allow_shared_calculator=True)
    neb.method = "improvedtangent"

    # Interpolate linearly the potisions of the n_images:
    neb.interpolate()
    neb.idpp_interpolate(optimizer=BFGS, steps=steps)

    images = read(f"idpp.traj@-{n_images + 2!s}:")

    # # Set calculators:
    nff_ase = NeuralFF.from_file(nff_dir, device="cuda:0")
    neb.set_calculators(nff_ase)

    # # Optimize:
    optimizer = BFGS(neb, trajectory=f"{nff_dir}/{rxn_name}.traj")
    optimizer.run(fmax=fmax, steps=steps)

    # Read NEB images from File
    images = read(f"{nff_dir}/{rxn_name}.traj@-{n_images + 2!s}:")

    return images
