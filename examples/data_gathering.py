import sys
sys.path.append("/home/saxelrod/Repo")
sys.path.append("/home/saxelrod/Repo/projects/commit_changes/graphbuilder")
from nff.train.trainer import *
import numpy as np
import os

import sys
import django
import pprint
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from graphbuilder import *
from NeuralForceField.nff.data.graphs import *

# Change to your directory
sys.path.append('/home/saxelrod/htvs/djangochem')

sys.path.append('/home/saxelrod/htvs')

# setup the django settings file.  Change this to use the settings file that connects you to your desired database
os.environ["DJANGO_SETTINGS_MODULE"] = "djangochem.settings.orgel"
# this must be run to setup access to the django settings and make database access work etc.
django.setup()

# Shell Plus Model Imports
from django.contrib.auth.models import Group, Permission, User
from jobs.models import Job, JobConfig, WorkBatch
from pgmols.models import Batch, Calc, Cluster, Geom, Hessian, Jacobian, Mechanism, Method, Mol, MolGroupObjectPermission, MolSet, MolUserObjectPermission, ProductLink, ReactantLink, Reaction, ReactionType, SinglePoint, Species, Stoichiometry
from django.contrib.admin.models import LogEntry
from django.contrib.contenttypes.models import ContentType
from django.contrib.sessions.models import Session
from features.models import AtomDescriptor, BondDescriptor, ConnectivityMatrix, DistanceMatrix, Fingerprint, ProximityMatrix, SpeciesDescriptor, TrainingSet, Transformation

from pgmols.models import Trajectory, MDFrame


from guardian.models import GroupObjectPermission, UserObjectPermission
# Shell Plus Django Imports
from django.urls import reverse
from django.conf import settings
from django.utils import timezone
from django.core.cache import cache
from django.db.models import SmallIntegerField, Avg, Case, Count, F, Max, Min, Prefetch, Q, Sum, When, Exists, OuterRef, Subquery, FloatField
from django.db import transaction
from django.contrib.auth import get_user_model
import numpy as np


GROUP = "switches"
HARTREE_TO_KCAL = 627.509
BOHR_TO_ANGS = 0.529177
FORCE_AU_TO_KCAL = HARTREE_TO_KCAL / BOHR_TO_ANGS
METHOD = 'sf_tddft_bhhlyp'
MEAN_ENERGY = True

def convert_list(lst, conv):
    for index, sub_lst in enumerate(lst):
        if type(sub_lst) is not list:
            lst[index] = sub_lst*conv
        else:
            lst[index] = convert_list(sub_lst, conv)
    return lst

def z_from_xyz(xyz):
    return [element[0] for element in xyz]

def is_isomer(sm1, sm2):
    """Determines if two smiles strings are cis-trans isomers."""

    array = [e1 == e2 for e1, e2 in zip(sm1, sm2)]
    true_array = list(filter(lambda x: x==True, array ))
    if len(true_array) != len(array)-1:
        return False
    false_index = [index for index, item in enumerate(array) if not item][0]
    char1 = sm1[false_index]
    char2 = sm2[false_index]
    return sorted([char1, char2]) == ["/", "\\"]

def find_isomers(smiles, group):

    smiles_list = list(Species.objects.filter(group__name=group).values_list("smiles", flat=True))
    dupes = list(filter(lambda x: is_isomer(smiles, x), smiles_list))
    dupes.append(smiles) if (smiles in smiles_list) else ()

    return sorted(dupes)

def filter_none(lst):
    return list(filter(lambda x: x is not None, lst))

def find_gs_e(smiles, group, method, mean):

    dupes = find_isomers(smiles, group)
    geoms = Geom.objects.filter(calcs__method__name = method, species__smiles__in=dupes)

    bare_gs_energies = list(geoms.values_list("calcs__props__totalenergy", flat=True))
    trimmed_energies = filter_none(bare_gs_energies)

    if mean:
        return np.mean(trimmed_energies)
    else:
        return min(trimmed_energies)

def gs_e_from_geoms(geoms, group, method, mean):

    smiles_list = list(geoms.values_list("species__smiles", flat=True).order_by("pk"))

    gs_e = []
    dic = {}

    for smiles in smiles_list:
        if smiles not in dic:
            dic[smiles] = find_gs_e(smiles, group, method, mean)

        gs_e.append(dic[smiles])

    return gs_e

def process_e(energies, gs_e, conv=HARTREE_TO_KCAL):

    trimmed_e = filter_none(energies)

    if len(gs_e) != len(trimmed_e):
        return None
    new_e = (np.array(trimmed_e) - np.array(gs_e)).tolist()
    return new_e
    # return convert_list(new_e, conv)

def process_f(forces, conv = FORCE_AU_TO_KCAL):
    trimmed_f = filter_none(forces)
    # return convert_list(trimmed_f, conv)
    return trimmed_f

def thread(lst1, *remaining_lists):
    """ Threads multiple lists together."""
    new_lst = []
    for n, item in enumerate(lst1):

        new_item = [item]
        for list in remaining_lists:
            new_item.append(list[n])
        new_lst.append(new_item)

    return new_lst

def get_energies(geoms, state, group, method, gs_e):


    if state == 0:
        keyword = "calcs__props__totalenergy"
        bare_energies = list(geoms.values_list(keyword, flat=True).order_by("pk"))
        return process_e(bare_energies, gs_e)

    keyword = "calcs__props__excitedstates__{}__energy".format(state-1)
    bare_energies = list(geoms.values_list(keyword, flat=True).order_by("pk"))


    return process_e(bare_energies, gs_e)

def combine_energies(geoms, state, group, method, mean=MEAN_ENERGY):
    """ Give a set of energies or one state's energies depdning on state. """

    if mean:
        print("Finding average energy")
        gs_e = gs_e_from_geoms(geoms, group, method, mean=mean)
        print("Found average energy")
    else:
        print("Finding ground state energies")
        gs_e = gs_e_from_geoms(geoms, group, method, mean=mean)
        print("Found ground state energies")

    # pdb.set_trace()

    if state == None:
        state = 0
        energy = get_energies(geoms, state, group, method, gs_e)
        while energy:
            if state == 0:
                energies = energy
            else:
                energies = thread(energies, energy)
            state += 1
            energy = get_energies(geoms, state, group, method, gs_e)
    else:
        energies = get_energies(geoms, state, group, method, gs_e)
    return energies


def get_forces(geoms, state):

    if state == 0:
        keyword = "calcs__jacobian__forces"
        bare_f = list(geoms.values_list(keyword, flat=True).order_by("pk"))
        return process_f(bare_f)

    keyword = "calcs__props__excitedstates__{}__forces".format(state-1)
    bare_f = list(geoms.values_list(keyword, flat=True).order_by("pk"))

    return process_f(bare_f)

def combine_forces(geoms, state):
    """ Give a set of forces or one state's forces depdning on state. """

    if state == None:
        forces = []
        state = 0
        force = get_forces(geoms, state)
        while force:
            if state == 0:
                forces = force
            else:
                forces = thread(forces, force)
            state += 1
            force = get_forces(geoms, state)

    else:
        forces = get_forces(geoms, state)
    return forces


def get_xyz(geoms):
    return list(geoms.values_list("xyz", flat=True).order_by("pk"))

def get_smiles(geoms):
    return list(geoms.values_list("species__smiles", flat=True).order_by("pk"))

def geoms_from_trjs(geom_per_trj, num_trjs, group):
    trjs = Trajectory.objects.annotate(count = Count('frames')).order_by("-count")
    num_trjs = len(trjs) if (num_trjs is None) else num_trjs
    geom_ids = []
    for i, trj in enumerate(trjs):
        if i >= num_trjs:
            break
        if geom_per_trj is not None:
            geoms = trj.frames.all()[:geom_per_trj]
        else:
            geoms = trj.frames.all()
        geom_ids = [*geom_ids, *list(geoms.values_list("pk", flat=True))]

    # pdb.set_trace()

    geom_query = Geom.objects.filter(pk__in=geom_ids)
    return geom_query

def geoms_from_smiles(smiles, group, method):
    if type(smiles) is not list:
        smls_filter = [smiles]
    else:
        smls_filter = smiles
    return Geom.objects.filter(species__smiles__in=smls_filter, species__group__name=group,
                              calcs__method__name=method, calcs__props__totalenergy__isnull=False).all()

def get_all_data(filter_by_smiles = True, smiles = None,
                 state = None, geom_per_trj = None, num_trjs = None, group=GROUP, method=METHOD):


    if filter_by_smiles:
        geoms = geoms_from_smiles(smiles=smiles, group=group, method=method)
    else:
        geoms = geoms_from_trjs(geom_per_trj=geom_per_trj, num_trjs=num_trjs, group=group)

    dic = {"xyz": get_xyz(geoms), "force": combine_forces(geoms, state),
           "energy": combine_energies(geoms, state, group, method), "smiles": get_smiles(geoms)}

    return dic
