import copy

GRAD_SUFFIX = "grad"
HESS_SUFFIX = "hess"
REQUIRED_KEYS = ["layers", "pool", "activation"]
POOL_OPTIONS = ["sum", "average"]
DEF_ARCH = {"pool": "sum", "layers": [lambda x: x / 2, 1], "activation": "shifted_softplus"}
DEF_WEIGHTS = {"main": 0.1, "grad": 1, "hess": 1}


class Property:
    """A class that holds information about a parent property (e.g energy) and its children properties (e.g. force,
        hessian). Different instances of the Property object are held in the Architecture object.

    Attributes:
        parent_name (str): name of the parent property
        grad_name (str): name of the gradient of the property
        hess_name (str): name of the Hessian of the property
        force_name (str): name of the negative gradient of the property
        layers (list): list of the number of neurons per layer. Elements of the list can be either integers
            or functions. If the element is a function (call it f(x)), then the number of neurons is that layer
            is f(n_previous), where n_previous is the number of neurons in the previous layer.
        activation (str): name of activation function
        pool (str): type of pooling
        weights (dict): dictionary of weights for calculating the loss function. A weight should be given to both
            the parent property and to all its children.

    """

    def __init__(self, parent_name, grad_name=None, hess_name=None, force_name=None, layers=None, activation=None,
                 pool=None, weights=None):

        """ Constructs a property instance.

        Args:
            parent_name (str): see Attributes
            grad_name (str): see Attributes
            hess_name (str): see Attributes
            force_name (str): see Attributes
            layers (list): see Attributes
            activation (str): see Attributes
            pool (str): see Attributes
            weights (dict): see Attributes

        Example:
            my_prop = Property(parent_name="energy_0", grad_name=None, hess_name="hess_0", force_name="force_0",
                               layers=[lambda x: 2*x,3, 1], activation="shifted_softplus", pool="sum",
                               weights={"energy_0": 1, "force_0": 2, "hess_0": 3})

            """
        self.name = parent_name
        self.grad_name = grad_name
        self.hess_name = hess_name
        self.force_name = force_name
        self.layers = layers
        self.activation = activation
        self.pool = pool
        self.weights = weights if (weights is not None) else {}

    def __repr__(self):
        """Representation of the object."""

        rep = ["name: {}".format(self.name)]
        for key, val in self.__dict__.items():
            if val is not None and key != "name":
                rep.append("{}: {}".format(key, val))
        return "({})".format(", ".join(rep))


class Architecture:
    """A class that holds information about the architecture of the neural network after the convolution layers.
    The main attribute of interest is prop_objects.

    Attributes:
        output_vars (list): a list of the output variables that you want the network to predict
        parent_keys (list): a list of the names of parent properties (e.g. "energy_3")
        child_keys (list): a list of the names of child properties (e.g. "force_3")
        specs (dict): each key in specs is the name of a different output variable. Its value is
            a dictionary with specifications about the network.
        prop_objects (list): a list of instances of the Property object, created from output_vars and specs.

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
            print(my_arch.prop_objects)
                >> [
                (name: energy_0, activation: shifted_softplus, weights: {'grad': 2, 'main': 3}, pool: sum, layers:
                [<function <lambda> at 0x1122992f0>, 1], force_name: force_0),
                (name: energy_1,
                activation: shifted_softplus, weights: {'grad': 4, 'main': 2}, pool: sum, layers:
                [<function <lambda> at 0x112299158>, 1], force_name: force_1),
                (name: dipole_2, activation: shifted_softplus, weights: {'main': 15}, pool: sum,
                layers: [<function <lambda> at 0x112299400>, 10, 1])
                ]


            """
        self.output_vars = output_vars
        # split variables into parent keys (e.g. energy) and child keys (e.g. force)
        [parent_keys, child_keys] = self.split_keys()
        self.parent_keys = parent_keys
        self.child_keys = child_keys
        self.specs = specs if (specs is not None) else {}
        # create prop_objects
        self.prop_objects = self.create_prop_objects()

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

    def add_child_to_prop(self, prop_object, child, attr):
        """Add children to a prop_object.
        Args:
            prop_object (Property): an instance of the Property object
            child (str): a child key
            attr (str): the name of the child property
        Example:
            child = "my_first_energy_gradient"
            attr = "grad"
        """

        # first set the child name
        setattr(prop_object, attr + "_name", child)
        # then set the weight of the child's contribution to the overall loss
        weight_attr = "grad" if ("force" in attr) else attr
        if child not in self.specs or "weight" not in self.specs[child]:
            print("Warning: {} loss weight not given. Using default weight of {}.".format(child,
                                                                                          DEF_WEIGHTS[weight_attr]))
        child_specs = self.specs.get(child, {})
        prop_object.weights[weight_attr] = child_specs.get("weight", DEF_WEIGHTS[weight_attr])

    def get_spec(self, parent_key):
        """Get spec dictionary of a prop_object.
        Args:
            parent_key (str): a parent key"""

        if parent_key not in self.specs:
            print("Warning: No specs given for property {}. Using defaults.".format(parent_key))
            return {}
        spec = self.specs[parent_key]
        for key in REQUIRED_KEYS:
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

    def create_prop_objects(self):
        """Create all property objects."""

        [parent_keys, child_keys] = self.split_keys()
        prop_objects = []
        for parent_key in parent_keys:
            children = self.get_children(parent_key=parent_key, child_keys=child_keys)
            # create property object with parent_name = parent_key
            new_object = Property(parent_name=parent_key)
            new_object.weights["main"] = self.specs[parent_key].get("weight", DEF_WEIGHTS["main"])
            # add children to new_object
            for child in children:
                if GRAD_SUFFIX in child:
                    self.add_child_to_prop(prop_object=new_object, child=child, attr="grad")
                elif HESS_SUFFIX in child:
                    self.add_child_to_prop(prop_object=new_object, child=child, attr="hess")
                elif "force" in child:
                    self.add_child_to_prop(prop_object=new_object, child=child, attr="force")

            # set the nn architecture of new_object using parent_key's spec
            spec = self.get_spec(parent_key)
            for key in DEF_ARCH.keys():
                val = spec.get(key, DEF_ARCH[key])
                self.check_valid(key, val)
                setattr(new_object, key, val)

            prop_objects.append(copy.deepcopy(new_object))

        return prop_objects
