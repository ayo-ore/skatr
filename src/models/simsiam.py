from .base_model import Model

class SimSiam(Model):
    self.num_augmentations
    self.augmentation_type = 'rotation'
    def batch_loss(self, batch):
    #xx what is batch.shape

    def augment(x, augmentation):
            def unique_int(max):
                i1 = int(np.random.rand()*(max+1))
                i2 = int(np.random.rand()*(max+1))
                if i1==i2:
                    unique_int(max)
                else:
                    return i1, i2

            if augmentation == 'rotation':
                i1, i2 = unique_int(3)
                x1 = np.rot90(x, i1, axes=(1,0))
                x2 = np.rot90(x, i2, axes=(1,0))
                return x1, x2
            else:
                raise('Selected augmentations faulty')

            def cosSim(p, z):
                CosineSimilarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                z = z.detach() # stop gradient
                return - CosineSimilarity(p, z)

        for i in range(self.num_augmentations):
            # augment / mask batch
            x1, x2 = augment(x, 'rotation')

            # embed original and transformed batch
            z1 = self.model(x1)
            z2 = self.model(x2)
            
            # predict original from embedding of transformed
            p1 = self.predictor(z1)
            p2 = self.predictor(z2)
            loss = - 0.5 * ( cosSim(p1, z2) + cosSim(p2, z1) )
            

            #augmentations: reflections, rotations

        raise NotImplementedError
    
    def forward(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError