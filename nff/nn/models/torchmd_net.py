from torch import nn
from nff.nn.modules.torchmd_net import (UpdateBlock,
                                        MessageBlock,
                                        EmbeddingBlock)
from nff.nn.modules.painn import ReadoutBlock
from nff.nn.layers import ExpNormalBasis
from nff.nn.modules.schnet import (AttentionPool, SumPool)
from nff.utils.tools import make_directed

POOL_DIC = {"sum": SumPool,
            "attention": AttentionPool}
EPS = 1e-15


class TorchMDNet(nn.Module):
    def __init__(self,
                 modelparams):

        super().__init__()

        conv_dropout = modelparams.get("conv_dropout", 0)
        learnable_mu = modelparams.get("learnable_mu", False)
        learnable_beta = modelparams.get("learnable_beta", False)
        same_message_blocks = modelparams["same_message_blocks"]
        feat_dim = modelparams["feat_dim"]
        num_heads = modelparams["num_heads"]
        means = modelparams.get("means")
        stddevs = modelparams.get("stddevs")
        pool_dic = modelparams.get("pool_dic")
        readout_dropout = modelparams.get("readout_dropout", 0)
        activation = modelparams["activation"]
        output_keys = modelparams["output_keys"]
        num_conv = modelparams["num_conv"]
        layer_norm = modelparams["layer_norm"]

        # basic properties

        self.output_keys = output_keys
        self.skip = modelparams.get("skip_connection",
                                    {key: False for key
                                     in self.output_keys})
        self.grad_keys = modelparams["grad_keys"]
        num_readouts = num_conv if any(self.skip.values()) else 1

        # modules

        rbf = ExpNormalBasis(n_rbf=modelparams["n_rbf"],
                             cutoff=modelparams["cutoff"],
                             learnable_mu=learnable_mu,
                             learnable_beta=learnable_beta)

        self.embed_block = EmbeddingBlock(feat_dim=feat_dim,
                                          dropout=conv_dropout,
                                          rbf=rbf)

        self.message_blocks = nn.ModuleList(
            [
                MessageBlock(
                    num_heads=num_heads,
                    feat_dim=feat_dim,
                    activation=activation,
                    rbf=rbf)

                for _ in range(num_conv)
            ]
        )

        self.update_blocks = nn.ModuleList(
            [
                UpdateBlock(num_heads=num_heads,
                            feat_dim=feat_dim,
                            dropout=conv_dropout)
                for _ in range(num_conv)])

        self.readout_blocks = nn.ModuleList(
            [ReadoutBlock(feat_dim=feat_dim,
                          output_keys=output_keys,
                          activation=activation,
                          dropout=readout_dropout,
                          means=means,
                          stddevs=stddevs)
             for _ in range(num_readouts)]
        )

        self.pool_dic = self.make_pool(pool_dic)

        if same_message_blocks:
            self.trim_message()

        self.layer_norm = nn.LayerNorm(feat_dim) if (layer_norm) else None

    def make_pool(self, pool_dic):
        if pool_dic is None:
            pool_module_dic = {key: SumPool() for key
                               in self.output_keys}
        else:
            pool_module_dic = nn.ModuleDict({})
            for out_key, sub_dic in pool_dic.items():
                pool_name = sub_dic["name"].lower()
                kwargs = sub_dic["param"]
                pool_class = POOL_DIC[pool_name]
                pool_module_dic[out_key] = pool_class(**kwargs)
        return pool_module_dic

    def trim_message(self):
        self.message_blocks = nn.ModuleList(
            [self.message_blocks[0]]
            * len(self.message_blocks))
        self.update_blocks = nn.ModuleList(
            [self.update_blocks[0]]
            * len(self.update_blocks))

    def atomwise(self,
                 batch,
                 xyz=None):

        nbrs, _ = make_directed(batch['nbr_list'])
        nxyz = batch['nxyz']

        if xyz is None:
            xyz = nxyz[:, 1:]
            xyz.requires_grad = True

        z_numbers = nxyz[:, 0].long()
        r_ij = xyz[nbrs[:, 1]] - xyz[nbrs[:, 0]]
        dist = ((r_ij ** 2 + EPS).sum(-1)) ** 0.5

        x_i = self.embed_block(z_numbers,
                               nbrs=nbrs,
                               dist=dist)
        results = {}

        for i, message_block in enumerate(self.message_blocks):

            update_block = self.update_blocks[i]

            inp = self.layer_norm(x_i) if self.layer_norm else x_i
            scaled_v = message_block(dist=dist,
                                     nbrs=nbrs,
                                     x_i=inp)

            x_i = update_block(nbrs=nbrs,
                               x_i=x_i,
                               scaled_v=scaled_v)

            if not any(self.skip.values()):
                continue

            readout_block = self.readout_blocks[i]
            new_results = readout_block(x_i)
            for key, skip in self.skip.items():
                if not skip:
                    continue
                if key in results:
                    results[key] += new_results[key]
                else:
                    results[key] = new_results[key]

        if not all(self.skip.values()):
            first_readout = self.readout_blocks[0]
            new_results = first_readout(x_i)
            for key, skip in self.skip.items():
                if not skip:
                    results[key] = new_results[key]

        return results, xyz

    def pool(self,
             batch,
             atomwise_out,
             xyz):

        if not hasattr(self, "output_keys"):
            self.output_keys = list(self.readout_blocks[0]
                                    .readoutdict.keys())

        if not hasattr(self, "pool_dic"):
            self.pool_dic = {key: SumPool() for key
                             in self.output_keys}

        all_results = {}

        for key, pool_obj in self.pool_dic.items():

            grad_key = f"{key}_grad"
            grad_keys = [grad_key] if (grad_key in self.grad_keys) else []
            results = pool_obj(batch=batch,
                               xyz=xyz,
                               atomwise_output=atomwise_out,
                               grad_keys=grad_keys,
                               out_keys=[key])
            all_results.update(results)

        return all_results, xyz

    def run(self,
            batch,
            xyz=None):

        atomwise_out, xyz = self.atomwise(batch=batch,
                                          xyz=xyz)
        all_results, xyz = self.pool(batch=batch,
                                     atomwise_out=atomwise_out,
                                     xyz=xyz)

        return all_results, xyz

    def forward(self, batch, xyz=None):
        """
        Call the model
        Args:
            batch (dict): batch dictionary
        Returns:
            results (dict): dictionary of predictions
        """

        results, _ = self.run(batch=batch,
                              xyz=xyz)

        return results
