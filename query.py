import pickle
from django.db.models import FloatField
from django.contrib.postgres.fields.jsonb import KeyTextTransform
from django.contrib.postgres.fields.jsonb import KeyTransform
from django.db.models.functions import Cast
from django.db.models.functions import Coalesce
from analysis.metalation_energy import *
import numpy as np
from random import shuffle
from analysis.metalation_energy import *


def query_single_mol(smiles, sampling_method):

    assert sampling_method in ['nms', 'min']
    assert type(smiles) == str

    if sampling_method == 'nms':
        sampling = 'hessian_displacement_dft_d3_gga_bp86'
    if sampling_method == 'min':
        sampling = 'dft_d3_gga_bp86'

    species = Species.objects.filter(smiles=smiles)
    #species |= Species.objects.filter(containedin__smiles__in=smiles)

    geoms = Geom.objects.filter(
            species__group__name='lipoly',
            species__in=species,
            method__name__in=[sampling]).distinct()

    values = geoms.annotate(
           energy=Coalesce(Cast(KeyTextTransform(
               'totalenergy', 'calcs__props'), FloatField()), 0)).order_by('stoichiometry','id').values_list(
               'xyz', 'calcs__jacobian__forces', 'energy','id', 'stoichiometry__formula')

    completed = []
    oldformula = None
    current = [None, None, None]

    for newgeom, newforces, newenergy, geomid, formula in values:
        if formula != oldformula:
            reference_energy = stoich_energy(formula, 'dft_d3_gga_bp86') # dft_hyb_wbx
            oldformula = formula
        if current[0] != newgeom:
            if all([i is not None for i in current]) and current[2]:
                completed.append(current)
            current = [newgeom, None, None]
        if newforces is not None:
            current[1] = newforces
        if newenergy is not None and newenergy != 0:
            current[2] = newenergy - reference_energy

    dump_path = '/home/wwj/data/lipoly/'+smiles+ "_" + sampling_method + '.pkl'

    if len(completed) != 0:
        print(str(len(completed)) + " geoms dumped for " + smiles)
        with open(dump_path, 'wb') as handle:
            pickle.dump(completed, handle)
    else:
        print("no data dumped")