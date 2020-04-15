from itertools import repeat
from torch.autograd import grad
import torch


def compute_grad(inputs, output):
    """Compute gradient of the scalar output with respect to inputs.

    Args:
        inputs (torch.Tensor): torch tensor, requires_grad=True
        output (torch.Tensor): scalar output 

    Returns:
        torch.Tensor: gradients with respect to each input component 
    """

    assert inputs.requires_grad

    gradspred, = grad(output, inputs, grad_outputs=output.data.new(output.shape).fill_(1),
                      create_graph=True, retain_graph=True)

    return gradspred


def gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    dim = range(src.dim())[dim]  # Get real dim value.

    # Automatically expand index tensor to the right dimensions.
    if index.dim() == 1:
        index_size = list(repeat(1, src.dim()))
        index_size[dim] = src.size(dim)
        index = index.view(index_size).expand_as(src)

    # Generate output tensor if not given.
    if out is None:
        dim_size = index.max().item() + 1 if dim_size is None else dim_size
        out_size = list(src.size())
        out_size[dim] = dim_size
        out = src.new_full(out_size, fill_value)

    return src, out, index, dim


def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value)
    return out.scatter_add_(dim, index, src)


def chemprop_msg_update(h, nbrs):

    # nbr_dim x nbr_dim matrix, e.g. for nbr_dim = 4, all_idx =
    # [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    all_idx = torch.stack([torch.arange(0, len(nbrs))] * len(nbrs)).long()

    # The first argument gives nbr list indices for which the second
    # neighbor of the nbr element matches the second neighbor of this element.
    # The second argument makes you ignore nbr elements equal to this one.
    # Example:
    # nbrs = [[1,2], [2, 1], [2, 3], [3, 2], [2, 4], [4, 2]].
    # message = [m_{12}, m_{21}, m_{23}, m_{32}, m_{24}, m_{42}]
    # Then m_{12} = h_{32} + h_{42} (and not + h_{12})

    mask = (nbrs[:, 1] == nbrs[:, 1, None]) * (nbrs[:, 0] != nbrs[:, 0, None])

    # select the values of all_idx that are allowed by `mask`
    good_idx = all_idx[mask]

    # get the h's of these indices
    h_to_add = h[good_idx]

    # number of nbr_list matches for each index of `message`.
    # E.g. for the first index, with m_{12}, we got two matches

    num_matches = mask.sum(1).tolist()
    # map from indices `h_to_add` to the indices of `message`
    match_idx = torch.cat([torch.LongTensor([index] * match)
                           for index, match in enumerate(num_matches)])
    match_idx = match_idx.to(h.device)

    graph_size = h.shape[0]

    message = scatter_add(src=h_to_add,
                          index=match_idx,
                          dim=0,
                          dim_size=graph_size)

    return message


def chemprop_msg_to_node(h, nbrs, num_nodes):

    node_idx = torch.arange(num_nodes).to(h.device)
    nbr_idx = torch.arange(len(nbrs)).to(h.device)
    node_nbr_idx = torch.stack([nbr_idx] * len(node_idx))

    mask = (nbrs[:, 0] == node_idx[:, None])
    num_matches = mask.sum(1).tolist()
    match_idx = torch.cat([torch.LongTensor([index] * match)
                           for index, match in enumerate(num_matches)])
    match_idx = match_idx.to(h.device)

    good_idx = node_nbr_idx[mask]
    h_to_add = h[good_idx]

    node_features = scatter_add(src=h_to_add,
                                index=match_idx,
                                dim=0,
                                dim_size=num_nodes)

    return node_features
