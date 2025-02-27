import logging
import resource
import sys
import torch
from abc import abstractmethod
from hydra.utils import instantiate
from torch.utils.data import DataLoader, default_collate, random_split

from src.utils import augmentations
from src.utils.datasets import SummarizedLCDataset
from src.utils.trainer import Trainer

class BaseExperiment:

    def __init__(self, cfg, exp_dir):
        
        self.cfg = cfg
        self.exp_dir = exp_dir
        self.log = logging.getLogger('Experiment')
        
        self.dtype_data = getattr(torch, cfg.data.dtype)
        self.dtype_train = getattr(torch, cfg.training.dtype)
        # self.dtype = torch.get_default_dtype()
        self.device = torch.device(
            f'cuda:{cfg.device}' if cfg.use_gpu and torch.cuda.is_available()
            else f'mps:{cfg.device}' if cfg.use_gpu else 'cpu'
        )
        self.log.info(f'Using device {self.device}')
        if cfg.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True        

        # initialize preprocessing transforms (for data and targets)
        self.preprocessing={
            k: [instantiate(t) for t in ts] for k, ts in self.cfg.preprocessing.items()
        }
        transform_names = {
            k: [t.__class__.__name__ for t in ts] for k,ts in self.preprocessing.items()
        }
        self.log.info(f'Loaded preprocessing dict: {transform_names}')


    def run(self):
        
        self.log.info('Initializing model')
        if self.cfg.train or self.cfg.evaluate:
            
            self.model = self.get_model().to(device=self.device, dtype=self.dtype_train)
            self.model = torch.compile(self.model)
            
            self.log.info(
                f'Model ({self.model.__class__.__name__}[{self.model.net.__class__.__name__}]) has '
                f'{sum(w.numel() for w in self.model.trainable_parameters)} trainable parameters'
            )

        if self.cfg.train or self.cfg.evaluate:
            
            self.log.info('Initializing dataloaders')
            dataloaders = self.get_dataloaders()

        if self.cfg.train:
            
            tcfg = self.cfg.training

            augs = self.get_augmentations()
            if augs:
                self.log.info(
                    f"Loaded augmentations: {', '.join([a.__class__.__name__ for a in augs])}"
                )

            self.log.info('Initializing trainer')                
            trainer = Trainer(
                self.model, dataloaders, self.preprocessing, augs,
                tcfg, self.exp_dir, self.device, self.dtype_train
            )
            self.log.info('Running training')
            trainer.run_training()

        if self.cfg.evaluate:
            self.log.info(f'Loading model state from {self.exp_dir}.')
            self.model.load(self.exp_dir, self.device)
            self.model = self.model.to(self.dtype_train)
            self.model.eval()
            self.log.info('Running evaluation')
            self.evaluate(dataloaders)

        if self.cfg.plot:
            self.log.info('Making plots')
            self.plot()

        # print memory usage
        self.log_resources()

    def get_dataloaders(self):
        
        dcfg = self.cfg.data
        tcfg = self.cfg.training
        fixed_rng = torch.Generator().manual_seed(1729)
        
        # read data
        dataset = self.get_dataset(dcfg.dir)
        dataset_test = (self.get_dataset(dcfg.dir + '/test') if dcfg.use_test_dir else None)

        shape_strings = [
            f'({len(dataset)}, {repr(tuple(x.shape))[1:].replace(',)', ')')}'
            for x in dataset[0]
        ]
        self.log.info(f"Read data with shapes {', '.join(shape_strings)}")
        
        # partition the dataset into train/val/test
        trn = dcfg.splits.train
        if dataset_test is None:
            tst = dcfg.splits.test
            val = 1 - trn - tst
            assert val > 0, 'A validation split is required'

            dataset_splits = dict(zip(
                ('train', 'val', 'test'),
                random_split(dataset, [trn, val, tst], generator=fixed_rng)
            ))
        else:
            val = 1 - trn 
            assert val > 0, 'A validation split is required'
            
            dataset_train, dataset_val = random_split(dataset, [trn, val], generator=fixed_rng)
            dataset_splits = {
                'train': dataset_train, 'val': dataset_val, 'test': dataset_test
            }

        del dataset #TODO: Assess if this is really necessary
        
        # create dataloaders
        dataloaders = {}
        num_cpus = 0 if dcfg.on_gpu or dcfg.summarize else self.cfg.num_cpus
        self.log.info(f'{num_cpus=}')
        for k, d in dataset_splits.items():
            
            collate_fn = default_collate
            is_trn = k=='train'
            # optionally summarize (compress) dataset
            if dcfg.summarize:
                
                if self.cfg.summary_net is None:
                    self.log.error('Asking to summarize dataset, but no summary net provided.')
                    sys.exit()

                augment = tcfg.augment and is_trn # only augment training split
                augstring = ' (with augmentations) ' if augment else ' '
                self.log.info(
                    f'Summarizing {k} split{augstring}with batch size {dcfg.summary_batch_size}.'
                )

                dataset_splits[k] = SummarizedLCDataset(
                    d, summary_net=self.model.summary_net, device=self.device, exp_cfg=self.cfg,
                    dataset_cfg=dcfg, augment=augment, use_amp=tcfg.use_amp
                )

                if augment:
                    # select one augmentation per batch
                    collate_fn = dataset_splits[k].collate_fn

            batch_size = tcfg.batch_size if is_trn else tcfg.test_batch_size
            dataloaders[k] = DataLoader(
                dataset_splits[k], shuffle=is_trn, drop_last=is_trn, batch_size=batch_size,
                collate_fn=collate_fn,
                pin_memory=False, # pinning can cause memory issues with large lightcones
                num_workers=num_cpus # parallel loading from GPU causes CUDA error
            )

        return dataloaders
    
    def get_augmentations(self):
        augs = []
        if self.cfg.training.augment and not self.cfg.data.summarize:
            for name, kwargs in self.cfg.training.augmentations.items():
                aug = getattr(augmentations, name)(**kwargs)
                augs.append(aug)
        return augs
    
    def log_resources(self):
        if self.cfg.use_gpu:
            max_mem_gpu = torch.cuda.max_memory_allocated(self.device)
            tot_mem_gpu = torch.cuda.mem_get_info(self.device)[1]
            GB = 1024**3
            self.log.info(
                f"Peak GPU RAM usage: {max_mem_gpu/GB:.3} GB (of {tot_mem_gpu/GB:.3} GB available)"
            )
        
        max_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.log.info(
            f"Peak system RAM usage: {max_ram/1024**2:.3} GB"
        )        
    
    @abstractmethod
    def get_dataset(self):
        pass
    
    @abstractmethod
    def get_model(self):
        pass
    
    @abstractmethod
    def plot(self):
        pass
    
    @abstractmethod
    def evaluate(self, dataloaders):
        pass
