from scripts.cp3d.test_new_confs.steps import choose_bioactive as cb


def run(rd_mol_dir,
        dset_dir,
        dset_name,
        targ_key,
        cutoff,
        undirected,
        load_existing):

    print("Loading dataset...")
    dset = cb.get_dset(rd_mol_dir=rd_mol_dir,
                       dset_dir=dset_dir,
                       dset_name=dset_name,
                       targ_key=targ_key,
                       cutoff=cutoff,
                       undirected=undirected,
                       load_existing=load_existing)
    print("Dataset loaded!")
