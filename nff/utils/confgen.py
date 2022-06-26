import os
import random
import subprocess
import re
import socket
import time
import numpy as np
import json
import pickle
import copy
import math

from rdkit.Chem import (AddHs, MolFromSmiles, inchi, GetPeriodicTable,
                        Conformer, MolToSmiles)
from rdkit.Chem.AllChem import (EmbedMultipleConfs,
                                UFFGetMoleculeForceField,
                                MMFFGetMoleculeForceField,
                                MMFFGetMoleculeProperties,
                                GetConformerRMS)
from rdkit.Chem.rdmolops import RemoveHs, GetFormalCharge

from nff.utils.misc import read_csv, tqdm_enum
from nff.data.parallel import gen_parallel

PERIODICTABLE = GetPeriodicTable()

UFF_ELEMENTS = ['B', 'Al']
DEFAULT_GEOM_COMPARE_TIMEOUT = 300
XYZ_NAME = "{0}_Conf_{1}.xyz"
MAX_CONFS = 10000
AU_TO_KCAL = 627.509
KB_KCAL = 0.001985875

INCHI_OPTIONS = " -RecMet  -FixedH "


def write_xyz(coords, filename, comment):
    '''
    Write an xyz file from coords
    '''
    with open(filename, "w") as f_p:
        f_p.write(str(len(coords)) + "\n")
        f_p.write(str(comment) + "\n")
        for atom in coords:
            f_p.write("%s %.4f %.4f %.4f\n" %
                      (atom[0], atom[1][0], atom[1][1], atom[1][2]))


def obfit_rmsd(file1, file2, smarts, path):
    cmd = ["obfit", "'" + str(smarts) + "'",
           os.path.join(path, file1 + '.xyz'),
           os.path.join(path, file2 + '.xyz')]
    ret = subprocess.check_output(" ".join(cmd),
                                  stdin=None,
                                  stderr=subprocess.STDOUT,
                                  shell=True,
                                  universal_newlines=False,
                                  timeout=DEFAULT_GEOM_COMPARE_TIMEOUT)
    rmsd = float(ret.decode('utf-8')[5:13])
    return rmsd


def align_rmsd(file1, file2, path, smarts=None):
    cmd = ["obabel",
           os.path.join(path, file1 + '.xyz'),
           os.path.join(path, file2 + '.xyz'),
           '-o', 'smi',
           '--align',
           '--append',
           'rmsd']
    if smarts:
        cmd += ['-s', str(smarts)]
    ret = subprocess.check_output(cmd,
                                  stdin=None,
                                  stderr=subprocess.STDOUT,
                                  shell=False,
                                  universal_newlines=False,
                                  timeout=DEFAULT_GEOM_COMPARE_TIMEOUT)
    rmsd = ret.decode('utf-8').split()[-1]
    return float(rmsd)


class ConformerGenerator(object):
    '''
    Generates conformations of molecules from 2D representation.
    '''

    def __init__(self, smiles, forcefield="mmff"):
        '''
        Initialises the class
        '''
        self.mol = MolFromSmiles(smiles)
        self.full_clusters = []
        self.forcefield = forcefield
        self.conf_energies = []
        self.initial_confs = None
        self.smiles = smiles

    def generate(self,
                 max_generated_conformers=50,
                 prune_thresh=0.01,
                 maxattempts_per_conformer=5,
                 output=None,
                 threads=1):
        '''
        Generates conformers

        Note  the number max_generated _conformers required is related to the
        number of rotatable bonds
        '''
        self.mol = AddHs(self.mol, addCoords=True)
        self.initial_confs = EmbedMultipleConfs(
            self.mol,
            numConfs=max_generated_conformers,
            pruneRmsThresh=prune_thresh,
            maxAttempts=maxattempts_per_conformer,
            useRandomCoords=False,
            # Despite what the documentation says -1 is a seed!!
            # It doesn't mean random generation
            numThreads=threads,
            randomSeed=random.randint(
                1, 10000000)
        )
        if len(self.initial_confs) == 0:
            output.write((f"Generated  {len(self.initial_confs)} "
                          "initial confs\n"))
            output.write((f"Trying again with {max_generated_conformers * 10} "
                          "attempts and random coords\n"))

            self.initial_confs = EmbedMultipleConfs(
                self.mol,
                numConfs=max_generated_conformers,
                pruneRmsThresh=prune_thresh,
                useRandomCoords=True,
                maxAttempts=10 * maxattempts_per_conformer,
                # Despite what the documentation says -1 is a seed!!
                # It doesn't mean random
                # generatrion
                numThreads=threads,
                randomSeed=random.randint(
                    1, 10000000)
            )

        output.write("Generated " +
                     str(len(self.initial_confs)) + " initial confs\n")
        return self.initial_confs

    def minimise(self,
                 output=None,
                 minimize=True):
        '''
        Minimises conformers using a force field
        '''

        if "\\" in self.smiles or "/" in self.smiles:
            output.write(("WARNING: Smiles string contains slashes, "
                          "which specify cis/trans stereochemistry.\n"))
            output.write(("Bypassing force-field minimization to avoid generating "
                          "incorrect isomer.\n"))
            minimize = False

        if self.forcefield != "mmff" and self.forcefield != "uff":
            raise ValueError("Unrecognised force field")
        if self.forcefield == "mmff":
            props = MMFFGetMoleculeProperties(self.mol)
            for i in range(0, len(self.initial_confs)):
                potential = MMFFGetMoleculeForceField(
                    self.mol, props, confId=i)
                if potential is None:
                    output.write("MMFF not available, using UFF\n")
                    potential = UFFGetMoleculeForceField(self.mol, confId=i)
                    assert potential is not None

                if minimize:
                    output.write(f"Minimising conformer number {i}\n")
                    potential.Minimize()
                mmff_energy = potential.CalcEnergy()
                self.conf_energies.append((i, mmff_energy))

        elif self.forcefield == "uff":
            for i in range(0, len(self.initial_confs)):
                potential = UFFGetMoleculeForceField(self.mol, confId=i)
                assert potential is not None
                if minimize:
                    potential.Minimize()
                uff_energy = potential.CalcEnergy()
                self.conf_energies.append((i, uff_energy))
        self.conf_energies = sorted(self.conf_energies, key=lambda tup: tup[1])
        return self.mol

    def cluster(self,
                rms_tolerance=0.1,
                max_ranked_conformers=10,
                energy_window=5,
                Report_e_tol=10,
                output=None):
        '''
        Removes duplicates after minimization
        '''
        self.counter = 0
        self.factormax = 3
        self.mol_no_h = RemoveHs(self.mol)
        calcs_performed = 0
        self.full_clusters = []
        confs = self.conf_energies[:]
        ignore = []
        ignored = 0

        for i, pair_1 in enumerate(confs):
            if i == 0:
                index_0, energy_0 = pair_1
            output.write((f"clustering cluster {i} of "
                          f"{len(self.conf_energies)}\n"))
            index_1, energy_1 = pair_1
            if abs(energy_1 - energy_0) > Report_e_tol:
                output.write(("Breaking because hit Report Energy Window, "
                              f"E was {energy_1} kcal/mol "
                              f"and minimum was {energy_0} \n"))

                break
            if i in ignore:
                ignored += i
                continue
            self.counter += 1
            if self.counter == self.factormax * max_ranked_conformers:
                output.write('Breaking because hit MaxNConfs \n')
                break
            clustered = [[self.mol.GetConformer(id=index_1), energy_1, 0.00]]
            ignore.append(i)
            for j, pair_2 in enumerate(confs):
                if j > 1:
                    index_2, energy_2 = pair_2
                    if j in ignore:
                        ignored += 1
                        continue
                    if abs(energy_1 - energy_2) > energy_window:
                        break
                    if abs(energy_1 - energy_2) <= 1e-3:
                        clustered.append([self.mol.GetConformer(id=index_2),
                                          energy_2, 0.00])
                        ignore.append(j)
                        rms = GetConformerRMS(self.mol_no_h,
                                              index_1,
                                              index_2)
                        calcs_performed += 1
                        if rms <= rms_tolerance:
                            clustered.append(
                                [self.mol.GetConformer(id=index_2),
                                 energy_2, rms])
                            ignore.append(j)
            self.full_clusters.append(clustered)
        output.write(f"{ignored} ignore passes made\n")
        output.write((f"{calcs_performed} overlays needed out "
                      f"of a possible {len(self.conf_energies) ** 2}\n"))

        ranked_clusters = []
        for i, cluster in enumerate(self.full_clusters):
            if i < self.factormax * max_ranked_conformers:
                ranked_clusters.append(cluster[0])

        return ranked_clusters

    def recluster(self,
                  path,
                  rms_tolerance=0.1,
                  max_ranked_conformers=10,
                  energy_window=5,
                  output=None,
                  clustered_confs=[],
                  molecule=None,
                  key=None,
                  fallback_to_align=False):
        self.removed = []
        self.counter = 0
        i = -1
        for conf_a in clustered_confs:
            i += 1
            j = i
            if self.counter == max_ranked_conformers:
                for k in range(i, len(clustered_confs)):
                    if os.path.isfile(key + "_Conf_" + str(k + 1) + ".xyz"):
                        os.remove(key + "_Conf_" + str(k + 1) + ".xyz")
                        output.write("Removed " + key +
                                     "_Conf_" + str(k + 1) + ".xyz\n")
                break
            if i in self.removed:
                continue
            self.counter += 1
            for conf_b in clustered_confs[i + 1:]:
                j += 1
                if conf_b[1] - conf_a[1] > energy_window:
                    break
                if j in self.removed:
                    continue
                try:
                    rms = obfit_rmsd(key + "_Conf_" + str(i + 1),
                                     key + "_Conf_" + str(j + 1),
                                     str(molecule),
                                     path=path)
                except (subprocess.CalledProcessError, ValueError,
                        subprocess.TimeoutExpired) as e:
                    if fallback_to_align:
                        output.write(
                            'obfit failed, falling back to obabel --align')
                        output.write(f'Exception {e}\n')
                        try:
                            rms = align_rmsd(f"{key}_Conf_{str(i + 1)}",
                                             f"{key}_Conf_{str(j + 1)}",
                                             path)
                        except (ValueError, subprocess.TimeoutExpired):
                            continue
                    else:
                        continue

                output.write("Comparing " + str(i + 1) + " " +
                             str(j + 1) + ' RMSD ' + str(rms) + "\n")
                if rms > rms_tolerance:
                    pos = _atomic_pos_from_conformer(conf_b[0])
                    elements = _extract_atomic_type(conf_b[0])
                    pos = [[-float(coor[k]) for k in range(3)] for coor in pos]
                    coords = list(zip(elements, pos))

                    filename = os.path.join(path, key + "_Conf_" +
                                            str(j + 1) + "_inv.xyz")
                    write_xyz(coords=coords, filename=filename,
                              comment=conf_b[1])
                    try:
                        file1 = key + "_Conf_" + str(i + 1)
                        file2 = key + "_Conf_" + str(j + 1) + "_inv"
                        rmsinv = obfit_rmsd(file1, file2, str(molecule))
                    except (subprocess.CalledProcessError, ValueError,
                            subprocess.TimeoutExpired) as e:
                        if fallback_to_align:
                            output.write(
                                'obfit failed, falling back to obabel --align')
                            output.write(f'Exception {e}\n')
                            try:
                                i_key = f"{key}_Conf_{str(i + 1)}"
                                inv_key = f"{key}_Conf_{str(j + 1)}_inv"
                                rmsinv = align_rmsd(i_key, inv_key)
                            except (ValueError, subprocess.TimeoutExpired):
                                continue
                        else:
                            continue

                    rms = min([rms, rmsinv])
                    os.remove(key + "_Conf_" + str(j + 1) + "_inv.xyz")
                    output.write((f"Comparing {i + 1} {j + 1} "
                                  f"RMSD after checking inversion {rms}\n"))
                if rms <= rms_tolerance:
                    self.removed.append(j)
                    output.write("Removed Conf_" + str(j + 1) + "\n")
                    os.remove(key + "_Conf_" + str(j + 1) + ".xyz")


def _extract_atomic_type(confomer):
    '''
    Extracts the elements associated with a conformer, in order that prune_threshy
    are read in
    '''
    elements = []
    mol = confomer.GetOwningMol()
    for atom in mol.GetAtoms():
        elements.append(atom.GetSymbol())
    return elements


def _atomic_pos_from_conformer(conformer):
    '''
    Extracts the atomic positions for an RDKit conformer object, to allow writing
    of input files, uploading to databases, etc.
    Returns a list of lists
    '''
    atom_positions = []
    natoms = conformer.GetNumAtoms()
    for atom_num in range(0, natoms):
        pos = conformer.GetAtomPosition(atom_num)
        atom_positions.append([pos.x, pos.y, pos.z])
    return atom_positions


def rename_xyz_files(path):
    namedict = {}
    flist = os.listdir(path)
    for filename in flist:
        if filename.endswith(".xyz"):
            num = int(filename.split('_')[-1][:-4])
            namedict[num] = filename
    keys = namedict.keys()
    for i, num in enumerate(sorted(keys)):
        oldfilename = namedict[num]
        newfilename = '_'.join(oldfilename.split(
            '_')[:-1]) + '_' + str(i + 1) + ".xyz"
        oldfilepath = os.path.join(path, oldfilename)
        newfilepath = os.path.join(path, newfilename)
        os.rename(oldfilepath, newfilepath)


def clean(molecule):
    molecule = str(molecule.split()[0])
    molecule = re.sub('Cl', '[#17]', molecule)
    molecule = re.sub('C', '[#6]', molecule)
    molecule = re.sub('c', '[#6]', molecule)
    molecule = re.sub('\[N-\]', '[#7-]', molecule)
    molecule = re.sub('N', '[#7]', molecule)
    molecule = re.sub('n', '[#7]', molecule)
    molecule = re.sub('\[\[', '[', molecule)
    molecule = re.sub('\]\]', ']', molecule)
    molecule = re.sub('\]H\]', 'H]', molecule)
    molecule = re.sub('=', '~', molecule)

    return molecule


def minimize(output,
             molecule,
             forcefield,
             nconf_gen,
             prun_tol,
             e_window,
             rms_tol,
             rep_e_window):

    output.write(f"Analysing smiles string {molecule}\n")
    MolFromSmiles(molecule)
    # print "There are", NumRotatableBonds(mol)
    output.write("Generating initial conformations\n")
    confgen = ConformerGenerator(
        smiles=molecule, forcefield=forcefield)
    output.write((f"Minimising conformations using the {forcefield} "
                  "force field\n"))
    confgen.generate(max_generated_conformers=int(nconf_gen),
                     prune_thresh=float(prun_tol),
                     output=output)
    gen_time = time.time()
    confgen.minimise(output=output)
    min_time = time.time()
    output.write(("Minimisation complete, generated conformations "
                  "with the following energies:\n"))
    output.write("\n".join([str(energy[1])
                            for energy in confgen.conf_energies])+"\n")
    msg = (f"Clustering structures using an energy window of "
           f"{e_window}  and an rms tolerance of {rms_tol} and a "
           f"Report Energy Window of {rep_e_window}\n")
    output.write(msg)

    return confgen, gen_time, min_time


def write_clusters(output,
                   idx,
                   conformer,
                   inchikey,
                   path):
    output.write(f"Cluster {idx} has energy {conformer[1]}\n")
    pos = _atomic_pos_from_conformer(conformer[0])
    elements = _extract_atomic_type(conformer[0])
    coords = list(zip(elements, pos))
    xyz_file = os.path.join(path, f"{inchikey}_Conf_{(idx + 1)}.xyz")
    write_xyz(coords=coords, filename=xyz_file,
              comment=conformer[1])


def run_obabel(inchikey,
               idx):
    try:
        cmd = ["obabel", f"{inchikey}_Conf_{(idx+1)}.xyz", "-osmi"]
    except UnboundLocalError as err:
        print(f"Did not produce any geometries for {inchikey} {err}")
        raise

    molecule = subprocess.check_output(cmd,
                                       stdin=None,
                                       stderr=subprocess.STDOUT,
                                       shell=False,
                                       universal_newlines=False
                                       ).decode('utf-8')
    molecule = clean(molecule)
    return molecule


def summarize(output,
              gen_time,
              start_time,
              min_time,
              cluster_time):

    recluster_time = time.time()

    output.write(socket.gethostname() + "\n")
    output.write('gen time  {0:1f}  sec\n'.format(
        gen_time - start_time))
    output.write('min time  {0:1f}  sec\n'.format(min_time - gen_time))
    output.write('cluster time  {0:1f}  sec\n'.format(
        cluster_time - min_time))
    output.write('recluster time  {0:1f}  sec\n'.format(
        recluster_time - cluster_time))
    output.write('total time  {0:1f}  sec\n'.format(
        time.time() - start_time))
    output.write('Terminated successfully\n')


def get_mol(smiles):
    mol = MolFromSmiles(MolToSmiles(
        MolFromSmiles(smiles)))
    return mol


def xyz_to_rdmol(nxyz, smiles):
    mol = get_mol(smiles)
    mol = AddHs(mol)

    num_atoms = len(nxyz)
    conformer = Conformer(num_atoms)
    for i, quad in enumerate(nxyz):
        conformer.SetAtomPosition(i, quad[1:])

    mol.AddConformer(conformer)
    return mol


def update_with_boltz(geom_list,
                      temp):

    rel_ens = np.array([i["relativeenergy"] for i in geom_list])
    degens = np.array([i["degeneracy"] for i in geom_list])
    k_t = temp * KB_KCAL
    weights = np.exp(-(degens * rel_ens) / k_t)
    weights /= weights.sum()

    for weight, geom_dic in zip(weights, geom_list):
        geom_dic["boltzmannweight"] = weight

    return geom_list


def parse_nxyz(lines):
    nxyz = []
    for line in lines[2:]:
        split = line.split()
        atom_char = split[0]
        positions = np.array(split[1:]).astype(float)
        atom_num = PERIODICTABLE.GetAtomicNumber(atom_char)
        this_nxyz = np.array([atom_num, *positions])
        nxyz.append(this_nxyz)
    nxyz = np.array(nxyz)
    return nxyz


def make_geom_dic(lines,
                  smiles,
                  geom_list,
                  idx):

    nxyz = parse_nxyz(lines)
    energy_kcal = float(lines[1])
    # total energy in au
    energy_au = energy_kcal / AU_TO_KCAL

    # relative energy in kcal
    if idx == 0:
        rel_energy = 0
    else:
        ref_energy_kcal = (geom_list[0]["totalenergy"]
                           * AU_TO_KCAL)
        rel_energy = energy_kcal - ref_energy_kcal

    rd_mol = xyz_to_rdmol(nxyz=nxyz,
                          smiles=smiles)

    geom = {"confnum": idx + 1,
            "totalenergy": energy_au,
            "relativeenergy": rel_energy,
            "degeneracy": 1,
            "rd_mol": rd_mol}

    return geom


def get_charge(smiles):
    mol = get_mol(smiles)
    charge = GetFormalCharge(mol)
    return charge


def combine_geom_dics(geom_list,
                      temp,
                      other_props,
                      smiles):

    totalconfs = sum([i["degeneracy"]
                      for i in geom_list])
    uniqueconfs = len(geom_list)
    lowestenergy = geom_list[0]["totalenergy"]
    poplowestpct = (geom_list[0]["boltzmannweight"]
                    * 100)
    charge = get_charge(smiles)

    combination = {"totalconfs": totalconfs,
                   "uniqueconfs": uniqueconfs,
                   "temperature": temp,
                   "lowestenergy": lowestenergy,
                   "poplowestpct": poplowestpct,
                   "charge": charge,
                   "conformers": geom_list,
                   "smiles": smiles}
    if other_props is not None:
        combination.update(other_props)

    return combination


def parse_results(job_dir,
                  log_file,
                  inchikey,
                  smiles,
                  max_confs,
                  other_props,
                  temp,
                  clean_up):

    # import pdb
    # pdb.set_trace()

    geom_list = []
    log_path = os.path.join(job_dir, log_file)
    with open(log_path) as f_p:
        loglines = f_p.readlines()

    if loglines[-1].strip() != "Terminated successfully":
        msg = ("'Terminated successfully' not found "
               "at end of conformer output")
        raise Exception(msg)

    geom_list = []
    for i in range(max_confs):
        path = os.path.join(job_dir, XYZ_NAME.format(inchikey, i + 1))
        if not os.path.isfile(path):
            continue
        with open(path, 'r') as f_p:
            lines = f_p.readlines()

        geom = make_geom_dic(lines=lines,
                             smiles=smiles,
                             geom_list=geom_list,
                             idx=i)
        geom_list.append(geom)

    geom_list = update_with_boltz(geom_list=geom_list,
                                  temp=temp)
    summary_dic = combine_geom_dics(geom_list=geom_list,
                                    temp=temp,
                                    other_props=other_props,
                                    smiles=smiles)

    if clean_up:
        for file in os.listdir(job_dir):
            if not ("_Conf_" in file and file.endswith("xyz")):
                continue
            file_path = os.path.join(job_dir, file)
            os.remove(file_path)

    return summary_dic


def one_species_confs(molecule,
                      log,
                      other_props,
                      max_confs,
                      forcefield,
                      nconf_gen,
                      e_window,
                      rms_tol,
                      prun_tol,
                      job_dir,
                      log_file,
                      rep_e_window,
                      fallback_to_align,
                      temp,
                      clean_up,
                      start_time):

    smiles = copy.deepcopy(molecule)
    with open(log, "w") as output:
        output.write("The smiles strings that will be run are:\n")
        output.write("\n".join([molecule])+"\n")

        if any([element in molecule for element in UFF_ELEMENTS]):
            output.write(("Switching to UFF, since MMFF94 does "
                          "not have boron and/or aluminum\n"))
            forcefield = 'uff'

        confgen, gen_time, min_time = minimize(output=output,
                                               molecule=molecule,
                                               forcefield=forcefield,
                                               nconf_gen=nconf_gen,
                                               prun_tol=prun_tol,
                                               e_window=e_window,
                                               rms_tol=rms_tol,
                                               rep_e_window=rep_e_window)
        clustered_confs = confgen.cluster(rms_tolerance=float(rms_tol),
                                          max_ranked_conformers=int(
                                              max_confs),
                                          energy_window=float(e_window),
                                          Report_e_tol=float(rep_e_window),
                                          output=output)

        cluster_time = time.time()
        inchikey = inchi.MolToInchiKey(get_mol(molecule),
                                       options=INCHI_OPTIONS)

        for i, conformer in enumerate(clustered_confs):
            write_clusters(output=output,
                           idx=i,
                           conformer=conformer,
                           inchikey=inchikey,
                           path=job_dir)

        molecule = run_obabel(inchikey=inchikey,
                              idx=i)
        confgen.recluster(path=job_dir,
                          rms_tolerance=float(rms_tol),
                          max_ranked_conformers=int(max_confs),
                          energy_window=float(e_window),
                          output=output,
                          clustered_confs=clustered_confs,
                          molecule=molecule,
                          key=inchikey,
                          fallback_to_align=fallback_to_align)
        rename_xyz_files(path=job_dir)
        summarize(output=output,
                  gen_time=gen_time,
                  start_time=start_time,
                  min_time=min_time,
                  cluster_time=cluster_time)

    conf_dic = parse_results(job_dir=job_dir,
                             log_file=log_file,
                             inchikey=inchikey,
                             max_confs=max_confs,
                             other_props=other_props,
                             temp=temp,
                             smiles=smiles,
                             clean_up=clean_up)
    return conf_dic


def run_generator(smiles_list,
                  other_props=None,
                  max_confs=MAX_CONFS,
                  forcefield="mmff",
                  nconf_gen=(10 * MAX_CONFS),
                  e_window=5.0,
                  rms_tol=0.1,
                  prun_tol=0.01,
                  job_dir="confs",
                  log_file="confgen.log",
                  rep_e_window=5.0,
                  fallback_to_align=False,
                  temp=298.15,
                  clean_up=True,
                  **kwargs):
    """
    Args:
        smiles_list (list[str]): list of SMILES strings
        other_props (dict): dictionary of any other properties of
            the molecule.
        forcefield (str): Forcefield used for minimisations
        max_confs (int): Number of low energy conformations to return
        nconf_gen (int): Number of conformations to build in the generation stage
        e_window (float): Energy window to use when clustering, kcal/mol
        rms_tol (float): RMS tolerance to use when clustering
        prun_tol (float): RMS tolerance used in pruning in the generation phase
        job_dir (str): directory for conformer generation
        log_file (str): name of log file
        rep_e_window (float): Energy window to use when reporting, Kcal/mol
        fallback_to_align (bool): whether to use rmsd_align if obfit fails
    """

    if not os.path.isdir(job_dir):
        os.makedirs(job_dir)

    log = os.path.join(job_dir, log_file)
    start_time = time.time()

    conf_dics = []

    for molecule in smiles_list:
        conf_dic = one_species_confs(molecule=molecule,
                                     log=log,
                                     other_props=other_props,
                                     max_confs=max_confs,
                                     forcefield=forcefield,
                                     nconf_gen=nconf_gen,
                                     e_window=e_window,
                                     rms_tol=rms_tol,
                                     prun_tol=prun_tol,
                                     job_dir=job_dir,
                                     log_file=log_file,
                                     rep_e_window=rep_e_window,
                                     fallback_to_align=fallback_to_align,
                                     temp=temp,
                                     clean_up=clean_up,
                                     start_time=start_time)
        conf_dics.append(conf_dic)

    return conf_dics


def add_to_summary(summary_dic,
                   conf_dic,
                   smiles,
                   save_dir):
    inchikey = inchi.MolToInchiKey(get_mol(smiles),
                                   options=INCHI_OPTIONS)
    pickle_path = os.path.join(os.path.abspath(save_dir), f"{inchikey}.pickle")
    summary_dic[smiles] = {key: val for key, val in
                           conf_dic.items() if key != "conformers"}
    summary_dic[smiles].update({"pickle_path": pickle_path})

    return summary_dic, pickle_path


def confs_and_save(config_path):
    with open(config_path, "r") as f_open:
        info = json.load(f_open)

    csv_data_path = info["csv_data_path"]
    save_dir = info["pickle_save_dir"]

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    smiles_dic = read_csv(csv_data_path)
    summary_dic = {}

    print(f"Saving pickle files to directory {save_dir}")

    for i, smiles in tqdm_enum(smiles_dic["smiles"]):
        other_props = {key: val[i] for key, val in smiles_dic.items()
                       if key != 'smiles'}
        smiles_list = [smiles]
        conf_dic = run_generator(smiles_list=smiles_list,
                                 other_props=other_props,
                                 **info)[0]
        summary_dic, pickle_path = add_to_summary(summary_dic=summary_dic,
                                                  conf_dic=conf_dic,
                                                  smiles=smiles,
                                                  save_dir=save_dir)

        with open(pickle_path, "wb") as f_open:
            pickle.dump(conf_dic, f_open)

    summary_path = os.path.join(info["summary_save_dir"], "summary.json")
    print(f"Saving summary to {summary_path}")
    with open(summary_path, "w") as f_open:
        json.dump(summary_dic, f_open, indent=4)
