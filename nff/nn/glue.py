import torch.nn
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ModuleDict

class Stack(torch.nn.Module):
    def __init__(self, model_dict, mode='sum'):
        super().__init__()
        implemented_mode = ['sum',
                            'mean']
        
        if mode not in implemented_mode:
            raise NotImplementedError(
                '{} mode is not implemented for Stack'.format(key))

        # to implement a check for readout keys

                
        self.models = ModuleDict(model_dict)
        self.mode = mode         
        
    def forward(self, batch, keys_to_combine=['energy', 'energy_grad']): 
        
        # run models 
        result_list = [self.models[key](batch) for key in self.models.keys()]
        
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