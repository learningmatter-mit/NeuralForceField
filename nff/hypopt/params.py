from nff.train import get_model


def make_feat_nums(in_basis, out_basis, num_layers):
    if num_layers == 0:
        return []
    elif num_layers == 1:
        feature_nums = [in_basis, out_basis]
    else:
        feature_nums = [in_basis]
        for i in range(1, num_layers):
            out_coeff = i / num_layers
            in_coeff = 1 - out_coeff
            num_nodes = int(out_coeff * out_basis + in_coeff * in_basis)
            feature_nums.append(num_nodes)
        feature_nums.append(out_basis)

    return feature_nums


def make_layers(in_basis,
                out_basis,
                num_layers,
                layer_act,
                dropout_rate,
                last_act=None):
    feature_nums = make_feat_nums(in_basis=in_basis,
                                  out_basis=out_basis,
                                  num_layers=num_layers)

    layers = []
    for i in range(len(feature_nums)-1):
        in_features = feature_nums[i]
        out_features = feature_nums[i+1]
        lin_layer = {'name': 'linear', 'param': {'in_features': in_features,
                                                 'out_features': out_features}}
        act_layer = {'name': layer_act, 'param': {}}
        drop_layer = {'name': 'Dropout', 'param': {'p': dropout_rate}}

        layers += [lin_layer, act_layer, drop_layer]
    # remove the last activation layer
    layers = layers[:-2]
    # update with a final activation if needed
    if last_act is not None:
        layers.append({"name": last_act, "param": {}})

    return layers


def make_boltz(boltz_type,
               num_layers=None,
               mol_basis=None,
               layer_act=None,
               last_act=None):
    if boltz_type == "multiply":
        dic = {"type": "multiply"}
        return dic

    layers = make_layers(in_basis=mol_basis+1,
                         out_basis=mol_basis,
                         num_layers=num_layers,
                         layer_act=layer_act,
                         last_act=last_act,
                         dropout_rate=0)
    dic = {"type": "layers", "layers": layers}
    return dic


def make_readout(names,
                 classifications,
                 num_basis,
                 num_layers,
                 layer_act,
                 dropout_rate):

    dic = {}
    for name, classification in zip(names, classifications):
        last_act = "sigmoid" if classification else None
        layers = make_layers(in_basis=num_basis,
                             out_basis=1,
                             num_layers=num_layers,
                             layer_act=layer_act,
                             last_act=last_act,
                             dropout_rate=dropout_rate)
        dic.update({name: layers})

    return dic


def get_extra_wc_feats(param_dic):
    extra_feats = param_dic.get("extra_features")
    if extra_feats is not None:
        extra_length = sum([dic["length"] for dic in extra_feats])
    else:
        extra_length = 0
    return extra_length


def get_extra_cp_feats(cp_params):

    cp_feats = cp_params["hidden_size"]
    extra_feats = cp_params.get("extra_features")
    if extra_feats is None:
        return cp_feats
    for extra_feat in extra_feats:
        cp_feats += extra_feat["length"]
    return cp_feats


def get_wc_params(param_dic, num_extra_feats):
    """
    Get params for a WeightedConformer model
    """

    classifications = [True] * len(param_dic["readout_names"])
    num_basis = param_dic["mol_basis"] + num_extra_feats

    readout = make_readout(names=param_dic["readout_names"],
                           classifications=classifications,
                           num_basis=num_basis,
                           num_layers=param_dic["num_readout_layers"],
                           layer_act=param_dic["layer_act"],
                           dropout_rate=param_dic["readout_dropout"])

    mol_fp_layers = make_layers(in_basis=param_dic["n_atom_basis"],
                                out_basis=param_dic["mol_basis"],
                                num_layers=param_dic["num_mol_layers"],
                                layer_act=param_dic["layer_act"],
                                last_act=None,
                                dropout_rate=param_dic["mol_nn_dropout"])

    params = {
        'n_convolutions': param_dic.get("n_conv"),
        'extra_features': param_dic.get('extra_features'),
        'mol_fp_layers': mol_fp_layers,
        'readoutdict': readout,
        'dropout_rate': param_dic.get("schnet_dropout")
    }

    params.update(param_dic)

    if param_dic.get("boltz_params") is not None:
        boltz = make_boltz(**param_dic["boltz_params"])
        params.update({'boltzmann_dict': boltz})

    return params


def get_cp_params(param_dic):
    """
    Example:
        info = {
            "param_regime": [{"name": "cp_hidden_size", "type": "int", "bounds": {"min": 100, "max": 200}}],
            "set_params": {"readout_names": ["bind"],
                            "chemprop": ["depth": 3]}}
        In a given iteration, this would lead to something like this:
            param_dic = {"cp_hidden_size": 132, "readout_name": ["bind"],
                         "chemprop": ["depth": 3]}]
        After applying this `get_cp_params`, we get
            param_dic = {readout_name": ["bind"],
                         "chemprop": ["depth": 3, "hidden_size": 132]}]
    """

    params = {**param_dic["chemprop"]}
    # anything that starts with "cp_"
    # is a chemprop param

    for key, val in param_dic.items():
        if key.startswith("cp_"):
            new_key = key.replace("cp_", "")
            params.update({new_key: val})

    return params


def make_wc_model(param_dic):
    """
    Make a WeightedConformer model
    """

    num_extra_feats = get_extra_wc_feats(param_dic)
    wc_params = get_wc_params(param_dic=param_dic,
                              num_extra_feats=num_extra_feats)
    model = get_model(wc_params, model_type="WeightedConformers")

    return model


def make_cp3d_model(param_dic):
    """
    Make a ChemProp3D model
    """

    cp_params = get_cp_params(param_dic)
    num_extra_feats = get_extra_cp_feats(cp_params)
    wc_params = get_wc_params(param_dic=param_dic,
                              num_extra_feats=num_extra_feats)
    wc_params.pop("chemprop")

    final_params = {"chemprop": cp_params,
                    **wc_params}
    model = get_model(final_params, model_type="ChemProp3D")

    return model


def make_cp2d_model(param_dic):
    """
    Make a ChemProp2D model
    """

    cp_params = get_cp_params(param_dic)
    num_extra_feats = get_extra_cp_feats(cp_params)
    classifications = [True] * len(param_dic["readout_names"])
    num_basis = num_extra_feats

    readout = make_readout(names=param_dic["readout_names"],
                           classifications=classifications,
                           num_basis=num_basis,
                           num_layers=param_dic["num_readout_layers"],
                           layer_act=param_dic["layer_act"],
                           dropout_rate=param_dic["readout_dropout"])

    final_params = {
        "chemprop": cp_params,
        'readoutdict': readout,
    }

    model = get_model(final_params, model_type="ChemProp2D")

    return model
