import torch
from torch import nn

from nff.utils.tools import make_directed
from nff.nn.modules.painn import (MessageBlock, UpdateBlock,
                                  EmbeddingBlock, ReadoutBlock)
from nff.utils.scatter import compute_grad


class Painn(nn.Module):
    def __init__(self,
                 modelparams):
        super().__init__()

        feat_dim = modelparams["feat_dim"]
        activation = modelparams["activation"]
        n_rbf = modelparams["n_rbf"]
        cutoff = modelparams["cutoff"]
        num_conv = modelparams["num_conv"]
        output_keys = modelparams["output_keys"]
        grad_keys = modelparams["grad_keys"]
        learnable_k = modelparams.get("learnable_k", False)

        self.embed_block = EmbeddingBlock(feat_dim=feat_dim)
        self.message_blocks = nn.ModuleList(
            [MessageBlock(feat_dim=feat_dim,
                          activation=activation,
                          n_rbf=n_rbf,
                          cutoff=cutoff,
                          learnable_k=learnable_k)
             for _ in range(num_conv)]
        )
        self.update_blocks = nn.ModuleList(
            [UpdateBlock(feat_dim=feat_dim,
                         activation=activation)
             for _ in range(num_conv)]
        )

        ##################
        # self.message_blocks = nn.ModuleList(
        #     [self.message_blocks[0]] * num_conv)
        # self.update_blocks = nn.ModuleList([self.update_blocks[0]] * num_conv)
        ##################

        # no skip connection in original paper
        self.skip = modelparams.get("skip_connection", False)
        num_readouts = num_conv if (self.skip) else 1
        self.readout_blocks = nn.ModuleList(
            [ReadoutBlock(feat_dim=feat_dim,
                          output_keys=output_keys,
                          grad_keys=grad_keys,
                          activation=activation)
             for _ in range(num_readouts)]
        )

    def forward(self, batch):

        nbrs, _ = make_directed(batch['nbr_list'])
        nxyz = batch['nxyz']
        num_atoms = batch['num_atoms'].detach().cpu().tolist()

        xyz = nxyz[:, 1:]
        xyz.requires_grad = True
        z_numbers = nxyz[:, 0].long()
        r_ij = xyz[nbrs[:, 1]] - xyz[nbrs[:, 0]]

        s_i, v_i = self.embed_block(z_numbers)

        # grad = compute_grad(output=r_ij,
        #                     inputs=xyz)
        # import pdb
        # pdb.set_trace()

        # print(grad)

        for i, message_block in enumerate(self.message_blocks):
            update_block = self.update_blocks[i]
            ds_message, dv_message = message_block(s_j=s_i,
                                                   v_j=v_i,
                                                   r_ij=r_ij,
                                                   nbrs=nbrs)

            # import pdb
            # pdb.set_trace()

            s_i = s_i + ds_message

            ######
            # dv_message = dv_message * 0
            ######
            v_i = v_i + dv_message

            ds_update, dv_update = update_block(s_i=s_i,
                                                v_i=v_i)

            s_i = s_i + ds_update

            ######
            # dv_update = dv_update * 0
            ######

            v_i = v_i + dv_update

            if self.skip:
                readout_block = self.readout_blocks[i]
                new_results = readout_block(s_i=s_i,
                                            xyz=xyz,
                                            num_atoms=num_atoms)
                if i == 0:
                    results = new_results
                else:
                    for key in results.keys():
                        results[key] += new_results[key]

        if not self.skip:
            readout_block = self.readout_blocks[0]
            results = readout_block(s_i=s_i,
                                    xyz=xyz,
                                    num_atoms=num_atoms)

        return results
