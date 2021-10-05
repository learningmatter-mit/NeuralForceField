import torch
import nff.data

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from nff.data.loader import collate_dicts


def get_loaders(args, logging=None):

    if logging is not None:
        logging.info("loading dataset...")

    dataset = torch.load(args.data_path)

    if args.mode == 'eval':
        test_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
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
    
        train_loader = DataLoader(
            train,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=collate_dicts,
            sampler=RandomSampler(train)
        )
        val_loader = DataLoader(
            val,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=collate_dicts
        )
        test_loader = DataLoader(
            test,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=collate_dicts
        )
    
        return train_loader, val_loader, test_loader
