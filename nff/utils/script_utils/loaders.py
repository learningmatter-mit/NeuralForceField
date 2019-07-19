import torch
import nff.data


def get_loaders(args, logging=None):

    if logging is not None:
        logging.info("loading dataset...")

    dataset = torch.load(args.data_path)

    if args.mode == 'eval':
        test_loader = nff.data.GraphLoader(
            dataset,
            batch_size=args.batch_size,
            cutoff=args.cutoff,
            device=args.device
        )

        return test_loader

    elif args.mode == 'train':

        if logging is not None:
            logging.info("creating splits...")

        train, val, test = nff.data.split_train_validation_test(
            dataset,
            val_size=args.split[0],
            test_size=args.split[1]
        )
    
        if logging is not None:
            logging.info("load data...")
    
        train_loader = nff.data.GraphLoader(
            train,
            batch_size=args.batch_size,
            cutoff=args.cutoff,
            device=args.device
        )
        val_loader = nff.data.GraphLoader(
            val,
            batch_size=args.batch_size,
            cutoff=args.cutoff,
            device=args.device
        )
        test_loader = nff.data.GraphLoader(
            test,
            batch_size=args.batch_size,
            cutoff=args.cutoff,
            device=args.device
        )
    
        return train_loader, val_loader, test_loader
