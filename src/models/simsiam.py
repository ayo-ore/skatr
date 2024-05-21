import torch # for rot90 augmentations
import random
from omegaconf import DictConfig

from .base_model import Model
from .. import networks


class SimSiam(Model):

    def __init__(self, cfg:DictConfig):
        super().__init__(cfg)
        self.predictor = networks.MLP(cfg.predictor)
        self.aug_num = cfg.net.aug_num
        self.aug_type = cfg.net.aug_type

    def batch_loss(self, batch):
    #xx what is batch.shape
    # TODO: check functionality, add reflections

        def augment(x, augmentation):
            def random_ints(max):
                rand_arr = random.sample(range(0, max+1), 2)
                return rand_arr[0], rand_arr[1]

            def random_int(max):
                return random.randint(0, max)

            if 'rotation' in augmentation and 'reflection' in augmentation:
                i1, i2 = random_ints(3) # random integers [0, 3]
                x1 = torch.rot90(x, i1, dims=[1,0]) # rotate between [0, 3] times
                x2 = torch.rot90(x, i2, dims=[1,0])
                i = random_int(3)
                if i == 0:
                    x1 = x1.transpose(x1, 0, 1)
                if i == 2:
                    x2 = x2.transpose(x1, 0, 1)
                return x1, x2

            '''if augmentation == 'rotation':
                i1, i2 = random_ints(3) # random integers [0, 3]
                x1 = torch.rot90(x, i1, dims=[1,0]) # rotate between [0, 3] times
                x2 = torch.rot90(x, i2, dims=[1,0])
                return x1, x2

            else:
                raise('Selected augmentations faulty')'''

        def cosSim(p, z):
            CosineSimilarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            z = z.detach() # stop gradient
            return CosineSimilarity(p, z)
        
        x = batch[0]

        # augment / mask batch
        x1, x2 = augment(x, ['rotation'])

        diff = x1-x2
        if diff < something:
            continue

        # embed original and transformed batch
        z1 = self(x1)
        z2 = self(x2)

        # predict original from embedding of transformed
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # symmetric loss using stopgrad on z
        loss = - cosSim(p1, z2)

        '''loss = 0
        for i in range(self.aug_num):
            # augment / mask batch
            x1, x2 = augment(x, 'rotation')

            # embed original and transformed batch
            z1 = self(x1)
            z2 = self(x2)
            
            # predict original from embedding of transformed
            p1 = self.predictor(z1)
            p2 = self.predictor(z2)

            # symmetric loss using stopgrad on z
            loss += - 0.5 * ( cosSim(p1, z2) + cosSim(p2, z1) ) / self.aug_num'''
        #print(f"{loss=}")
        return loss
    
    def forward(self, x):
        return self.net(x)
        #raise NotImplementedError

    def predict(self, x):
        return self.net(x)
        #raise NotImplementedError