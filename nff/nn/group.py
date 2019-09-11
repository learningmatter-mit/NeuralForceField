import copy

GRAD_SUFFIX = "grad"
HESS_SUFFIX = "hess"
POOL_OPTIONS = ["sum", "average"]
DEF_ARCH = {"pool": "sum", "layers": [lambda x: x / 2, 1], "activation": "shifted_softplus"}
DEF_WEIGHTS = {"main": 0.1, "grad": 1, "hess": 1}
TEMPLATE_DIC = {
    "weights": {
        "main": None,
        "grad": None,
        "hess": None,
        "force": None
    },
    "grad_name": None,
    "hess_name": None,
    "force_name": None,
    "layers": None,
    "activation": None,
    "pool": None,
}


class Architecture:
    """A class that holds information about the architecture of the neural network after the convolution layers.
    The main attribute of interest is prop_dics.

    Attributes:
        output_vars (list): a list of the output variables that you want the network to predict
        parent_keys (list): a list of the names of parent properties (e.g. "energy_3")
        child_keys (list): a list of the names of child properties (e.g. "force_3")
        specs (dict): each key in specs is the name of a different output variable. Its value is
            a dictionary with specifications about the network for that output.
        prop_dics (dict): a dictionary of property dictionaries, created from output_vars and specs.

    """

    def __init__(self, output_vars, specs=None):
        """Creates an instance of the Architecture object.

        Args:
            output_vars (list): see Attributes
            specs (dict): see Attributes
        Example:
            output_vars = ["energy_0", "force_0", "energy_1", "force_1", "dipole_2"]
            specs = {
                    "energy_0": {'activation': 'shifted_softplus', 'layers': [lambda x: x/2, 1],
                                  'pool': 'sum', "weight": 3},
                    "force_0": {"weight": 2},
                     "energy_1": {'activation': 'shifted_softplus',
                                  'layers': [lambda x: x, 1],
                                  'pool': 'sum', "weight": 2},
                    "force_1": {"weight": 4},
                    "dipole_2": {'activation': 'shifted_softplus', 'layers': [lambda x: 3*x, 10, 1],
                                'pool': 'sum', 'weight': 15}
                     }

            my_arch = Architecture(output_vars, specs)
            print(my_arch.prop_dics)
                >> {'dipole_2': {'activation': 'shifted_softplus',
                  'force_name': None,
                  'grad_name': None,
                  'hess_name': None,
                  'layers': [<function __main__.<lambda>(x)>, 10, 1],
                  'pool': 'sum',
                  'weights': {'force': None, 'grad': None, 'hess': None, 'main': 15}},
                 'energy_0': {'activation': 'shifted_softplus',
                  'force_name': 'force_0',
                  'grad_name': None,
                  'hess_name': None,
                  'layers': [<function __main__.<lambda>(x)>, 1],
                  'pool': 'sum',
                  'weights': {'force': None, 'grad': 2, 'hess': None, 'main': 3}},
                 'energy_1': {'activation': 'shifted_softplus',
                  'force_name': 'force_1',
                  'grad_name': None,
                  'hess_name': None,
                  'layers': [<function __main__.<lambda>(x)>, 1],
                  'pool': 'sum',
                  'weights': {'force': None, 'grad': 4, 'hess': None, 'main': 2}}}


            """
        self.output_vars = output_vars
        # split variables into parent keys (e.g. energy) and child keys (e.g. force)
        [parent_keys, child_keys] = self.split_keys()
        self.parent_keys = parent_keys
        self.child_keys = child_keys
        self.specs = specs if (specs is not None) else {}
        # create prop_dics
        self.prop_dics = self.create_prop_dics()

    def split_keys(self):
        """Split output_vars into parent_keys and child_keys."""

        parent_keys = []
        child_keys = []
        for key in self.output_vars:
            if not any((GRAD_SUFFIX in key, HESS_SUFFIX in key, "force" in key)):
                parent_keys.append(key)
            else:
                child_keys.append(key)
        return parent_keys, child_keys

    def are_related(self, key1, key2):
        """Determine if two keys are related (i.e., if one is the parent of the other).
        Args:
            key1 (str): first key
            key2 (str): second key
        """

        sorted_keys = sorted([key1, key2])
        for suffix in GRAD_SUFFIX, HESS_SUFFIX:
            # e.g., if sorted_keys[1] = "my_energy_grad" and sorted_keys[0] = "my_energy"
            if sorted_keys[1] == sorted_keys[0] + suffix:
                return True
        for pair in [("energy", "force"), ("energy", "hess")]:
            # e.g., if key[0] = "my_first_energy" and key[2] = "my_first_hess"
            if key1.replace(pair[0], pair[1]) == key2:
                return True
            if key2.replace(pair[0], pair[1]) == key1:
                return True
        return False

    def get_children(self, parent_key, child_keys):
        """ Get all children of a parent key.
        Args:
            parent_key (str): parent key
            child_keys (list): list of all child keys (related and not)"""

        children = []
        are_related = list(map(lambda x: self.are_related(parent_key, x), child_keys))
        for child_key, is_related in zip(child_keys, are_related):
            if is_related:
                children.append(child_key)
        return children

    def add_child_to_prop(self, prop_dic, child, attr):
        """Add children to a prop_dic.
        Args:
            prop_dic (dict): a property dictionary
            child (str): a child key
            attr (str): the name of the child property
        Example:
            child = "my_first_energy_gradient"
            attr = "grad"
        """

        # first set the child name
        prop_dic["{}_name".format(attr)] = child
        # then set the weight of the child's contribution to the overall loss
        weight_attr = "grad" if ("force" in attr) else attr
        if child not in self.specs or "weight" not in self.specs[child]:
            print("Warning: {} loss weight not given. Using default weight of {}.".format(child,
                                                                                          DEF_WEIGHTS[weight_attr]))
        child_specs = self.specs.get(child, {})
        prop_dic["weights"][weight_attr] = child_specs.get("weight", DEF_WEIGHTS[weight_attr])

    def get_spec(self, parent_key):
        """Get spec dictionary of a prop_dic.
        Args:
            parent_key (str): a parent key"""

        if parent_key not in self.specs:
            print("Warning: No specs given for property {}. Using defaults.".format(parent_key))
            return {}
        spec = self.specs[parent_key]
        for key in DEF_ARCH.keys():
            if key not in spec.keys():
                print("Warning: {} not specified. Using default {}.".format(key, DEF_ARCH[key]))
        return spec

    def check_valid(self, key, val):
        """Check if key: val pair is valid.
        Args:
            key (str): dictionary key
            val (str): dictionary value"""

        if key == "pool":
            assert val in POOL_OPTIONS, "{} is not a valid pool option. Choices are: {}.".format(key, ", ".join(
                POOL_OPTIONS))

    def create_prop_dics(self):
        """Create all property dictionaries."""

        prop_dics = dict()
        for parent_key in self.parent_keys:
            children = self.get_children(parent_key=parent_key, child_keys=self.child_keys)
            # create property dictionary
            new_dic = copy.deepcopy(TEMPLATE_DIC)
            new_dic["weights"]["main"] = self.specs[parent_key].get("weight", DEF_WEIGHTS["main"])

            # add children to new_object
            for child in children:
                if GRAD_SUFFIX in child:
                    self.add_child_to_prop(prop_dic=new_dic, child=child, attr="grad")
                elif HESS_SUFFIX in child:
                    self.add_child_to_prop(prop_dic=new_dic, child=child, attr="hess")
                elif "force" in child:
                    self.add_child_to_prop(prop_dic=new_dic, child=child, attr="force")

            # set the nn architecture of new_object using parent_key's spec
            spec = self.get_spec(parent_key)
            for key in DEF_ARCH.keys():
                val = spec.get(key, DEF_ARCH[key])
                self.check_valid(key, val)
                new_dic[key] = val

            prop_dics.update({parent_key: new_dic})

        return prop_dics
