import copy
from torch import nn
from torch.nn import ModuleDict, ModuleList
import numpy as np
from nff.train import batch_detach

IMPLEMENTED_MODES = ['sum', 'mean']


class Stack(nn.Module):
    def __init__(self, model_dict, mode='sum'):
        super().__init__()

        if mode not in IMPLEMENTED_MODES:
            raise NotImplementedError(
                f'{mode} mode is not implemented for Stack')

        # to implement a check for readout keys

        self.models = ModuleDict(model_dict)
        self.mode = mode

    def forward(self,
                batch,
                keys_to_combine=['energy', 'energy_grad'],
                **kwargs):

        # run models
        result_list = [self.models[key](batch,  **kwargs)
                       for key in self.models.keys()]

        # perform further operations
        combine_results = dict()

        for i, result in enumerate(result_list):
            for key in keys_to_combine:
                if i != 0:
                    combine_results[key] += result[key]
                else:
                    combine_results[key] = result[key]
        if self.mode == 'mean':
            for key in keys_to_combine:
                combine_results[key] /= len(result_list)
        return combine_results


class DiabatStack(nn.Module):
    def __init__(self,
                 models,
                 diabat_keys,
                 energy_keys,
                 adiabat_mean,
                 extra_keys=None):

        super().__init__()

        self.models = ModuleList(models)
        self.diabatic_readout = self.models[0].diabatic_readout
        self.diabat_keys = diabat_keys
        self.energy_keys = energy_keys
        self.adiabat_mean = adiabat_mean

        # any extra keys you want to be averaged
        self.extra_keys = (extra_keys if (extra_keys is not None)
                           else extra_keys)

    def forward(self,
                batch,
                **kwargs):

        # use the same xyz for all the models so you can
        # compute the gradients

        num_models = len(self.models)

        if self.adiabat_mean:
            combined_results = {}
            for i, model in enumerate(self.models):
                results = batch_detach(model(batch, **kwargs))
                for key in [*self.energy_keys, *self.extra_keys]:
                    for grad in [True, False]:
                        this_key = key + "_grad" if (grad) else key
                        if this_key not in results:
                            continue
                        if i != 0:
                            combined_results[this_key] += (results[this_key]
                                                           / num_models)
                        else:
                            combined_results[this_key] = (results[this_key]
                                                          / num_models)
            return combined_results

        xyz = batch['nxyz'][:, 1:]
        xyz.requires_grad = True

        # don't compute any gradients in the initial forward
        # pass because they're not true gradients of the
        # average adiabatic energies and waste time/memory

        init_kwargs = copy.deepcopy(kwargs)
        init_kwargs.update({"add_grad": False,
                            "add_nacv": False})

        # get diabatic predictions from each model
        result_list = []
        for model in self.models:
            # use the initial run before computing all the
            # adiabatic quantities, if possible
            if hasattr(model, "run"):
                result_list.append(model.run(batch=batch,
                                             xyz=xyz)[0])
            else:
                result_list.append(model(batch=batch,
                                         xyz=xyz,
                                         **init_kwargs))

        combined_results = {}
        unique_diabat_keys = list(set((np.array(self.diabat_keys)
                                       .reshape(-1).tolist())))
        num_models = len(self.models)
        for key in unique_diabat_keys:
            for i, result in enumerate(result_list):
                if i == 0:
                    combined_results[key] = result[key] / num_models
                else:
                    combined_results[key] = (combined_results[key]
                                             + result[key] / num_models)

        combined_results = self.diabatic_readout(batch=batch,
                                                 xyz=xyz,
                                                 results=combined_results,
                                                 **kwargs)

        return combined_results
