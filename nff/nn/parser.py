import numpy as np
GRAD_SUFFIX = "_grad"
HESS_SUFFIX = "_hess"

class Property:
	def __init__(self, name, grad=None, hess=None, force=None, layers=None, pool=None):
		self.name = name
		self.grad = grad
		self.hess = hess
		self.force = force
		self.layers = layers
		self.pool = pool

	def __repr__(self):
		rep = ["name: {}".format(self.name)]
		for key, val in self.__dict__.items():
			if val is not None and key != "name":
				rep.append("{}: {}".format(key, val))
		return "({})".format(", ".join(rep))


class Architecture:
	def __init__(self, output_vars, specs):
		self.output_vars = output_vars
		[base_keys, child_keys] = self.split_keys()
		self.base_keys = base_keys
		self.child_keys = child_keys
		self.prop_objects = self.create_prop_objects()
		self.specs = specs
		self.parse_specs()

	def split_keys(self):
		base_keys = []
		child_keys = []
		for key in self.output_vars:
			if not any((GRAD_SUFFIX in key, HESS_SUFFIX in key, "force" in key)):
				base_keys.append(key)
			else:
				child_keys.append(key)
		return base_keys, child_keys

	def are_related(self, key1, key2):
		sorted_keys = sorted([key1, key2])
		for suffix in GRAD_SUFFIX, HESS_SUFFIX:
			if sorted_keys[1] == sorted_keys[0] + suffix:
				return True
		if key1.replace("energy", "force") == key2:
			return True
		if key2.replace("energy", "force") == key1:
			return True
		return False

	def create_prop_objects(self):
		prop_objects = []
		for base_key in self.base_keys:
			new_object = Property(name=base_key)
			
			test_relations = list(map(lambda x: self.are_related(base_key, x), self.child_keys))
			children = []
			for child_key, test_relation in zip(self.child_keys, test_relations):
				if test_relation:
					children.append(child_key)

			for child in children:
				if GRAD_SUFFIX in child:
					new_object.grad = child
				elif HESS_SUFFIX in child:
					new_object.hess = child
				elif "force" in child:
					new_object.force = child
			prop_objects.append(new_object)

		return prop_objects

	def parse_specs(self):
		for prop_object in self.prop_objects:
			


