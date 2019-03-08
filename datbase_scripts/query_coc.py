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

smiles = ['COC']
species = Species.objects.filter(smiles__in=smiles)
species |= Species.objects.filter(containedin__smiles__in=smiles)

geoms = Geom.objects.filter(
    species__group__name='lipoly',
    species__in=species,
    method__name__in=['hessian_displacement_dft_d3_gga_bp86', 'nn_dynamics_dft_d3_gga_bp86']
                            ).distinct()

values = geoms.annotate(
       energy=Coalesce(Cast(KeyTextTransform(
           'totalenergy', 'calcs__props'), FloatField()), 0)).order_by('stoichiometry', 'species', 'id').values_list(
           'xyz', 'calcs__jacobian__forces', 'energy','id', 'stoichiometry__formula', 'species__smiles').order_by("?")[:50000]

completed = []
oldformula = None
current = [None, None, None, None]

for newgeom, newforces, newenergy, geomid, formula, smiles in values:
    if formula != oldformula:
        reference_energy = stoich_energy(formula, 'dft_d3_gga_bp86') # dft_hyb_wbx
        oldformula = formula
    if current[0] != newgeom:
        if all([i is not None for i in current]) and current[2]:
            completed.append(current)
        current = [newgeom, None, None, None]
    if newforces is not None:
        current[1] = newforces
    if newenergy is not None and newenergy != 0:
        # print(newenergy)
        current[2] = newenergy - reference_energy
    if smiles is not None:
        current[3] = smiles

with open('/home/wwj/data/lipoly/COC_NMS.pkl', 'wb') as handle:
    pickle.dump(completed, handle)