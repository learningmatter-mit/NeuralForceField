import os
import numpy as np
import json
import argparse
import time, subprocess

import django

# setup the django settings file.  Change this to use the settings file that connects you to your desired database
# The environment variables is very importatn here, make sure htvs/djangochem/ is in your path somehow
os.environ["DJANGO_SETTINGS_MODULE"] = "djangochem.settings.orgel"
django.setup()

from django.contrib.auth.models import Group, Permission, User
from django.contrib.sessions.models import Session
from django.contrib.contenttypes.models import ContentType
from neuralnet.models import NetArchitecture, NeuralNetwork, NnPotential, NnPotentialStats
from jobs.models import Job, JobConfig, WorkBatch
from guardian.models import GroupObjectPermission, UserObjectPermission
from pgmols.models import AtomBasis, BasisSet, Batch, Calc, Cluster, Geom, Hessian, Jacobian, Mechanism, Method, Mol, MolGroupObjectPermission, MolSet, MolUserObjectPermission, PathImage, ProductLink, ReactantLink, Reaction, ReactionPath, ReactionType, SinglePoint, Species, Stoichiometry
from django.contrib.admin.models import LogEntry
from features.models import AtomDescriptor, BondDescriptor, ConnectivityMatrix, DistanceMatrix, Fingerprint, ProximityMatrix, SpeciesDescriptor, TrainingSet, Transformation

from django.urls import reverse
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone
from django.contrib.auth import get_user_model
from django.db import transaction
from django.db.models import Avg, Case, Count, F, Max, Min, Prefetch, Q, Sum, When, Exists, OuterRef, Subquery

from neuralnet.utils import *
