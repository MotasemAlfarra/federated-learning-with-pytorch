import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformWrapper(nn.Module):
    def __init__(self, model, device, aug_method, sigma=0.1, num_bases=2):
        super(DeformWrapper, self).__init__()
        self.base_classifier = model
        self.device = device
        self.aug_method = aug_method
        # We assume that the input is always between 0 and 1. For rotation, we do this normalization internally
        self.sigma = sigma*math.pi if aug_method =='rotation' else sigma
        self.num_bases = num_bases
        self.deformed_images = None
    
    def _deformImageGaussianFull(self, imgs):
        '''This function apply defromation to the Image. The deformation is sampled form normal distribution
        '''

        batch_sz, num_channels, rows, cols = imgs.shape
        new_ros = torch.linspace(-1, 1, rows)
        new_cols = torch.linspace(-1, 1, cols)
        meshx, meshy = torch.meshgrid((new_ros, new_cols))
        grid = torch.stack((meshy, meshx), 2).unsqueeze(0).expand(batch_sz, rows, cols, 2).to(self.device)

        randomFlow = torch.randn(batch_sz, rows, cols, 2, device=self.device) * self.sigma

        new_grid = grid + randomFlow
        Iwarp = F.grid_sample(imgs, new_grid)

        return Iwarp
    
    def _GenImageRotation(self, x):
        N, num_channels, rows, cols = x.shape # N is the batch size
        ang = (-2 * torch.rand((N, 1, 1)) + 1) *self.sigma #Uniform between [-sigma, sigma]
        
        #Generating the vector field for rotation. Not that sigma should be sig*pi, where sig is in [0,1]
        X, Y = torch.meshgrid(torch.linspace(-1,1,rows),torch.linspace(-1,1,cols))
        X, Y = X.unsqueeze(0), Y.unsqueeze(0)
        Xv = X*torch.cos(ang)-Y*torch.sin(ang)-X
        Yv = X*torch.sin(ang)+Y*torch.cos(ang)-Y
        
        randomFlow = torch.stack((Yv,Xv), axis=3).to(self.device)
        grid = torch.stack((Y,X), axis=3).to(self.device)
        
        return F.grid_sample(x, grid+randomFlow)

    def _GenImageTranslation(self, x):
        N, _, rows, cols = x.shape #N is the batch size

        #Generating the vector field for translation.
        X, Y = torch.meshgrid(torch.linspace(-1,1,rows),torch.linspace(-1,1,cols))
        X, Y = X.unsqueeze(0), Y.unsqueeze(0)
        Xv = torch.randn((N, 1, 1))*self.sigma + 0*X
        Yv = torch.randn((N, 1, 1))*self.sigma + 0*Y
        
        randomFlow = torch.stack((Yv,Xv), axis=3).to(self.device)
        grid = torch.stack((Y,X), axis=3).to(self.device)
        
        return F.grid_sample(x, grid+randomFlow)
    
    def _GenImageScalingUniform(self, x):
        N, _, rows, cols = x.shape # N is the batch size
        #Scaling here is sampled from uniform distribution between [-sigma, sigma]
        scale = (-2 * torch.rand((N, 1, 1)) + 1.0) * self.sigma + 1.0
        #Generating the vector field for scaling.
        X, Y = torch.meshgrid(torch.linspace(-1,1,rows),torch.linspace(-1,1,cols))
        X, Y = X.unsqueeze(0), Y.unsqueeze(0)
        Xv = X * scale - X
        Yv = Y * scale - Y
        
        randomFlow = torch.stack((Yv,Xv), axis=3).to(self.device)
        grid = torch.stack((Y,X), axis=3).to(self.device)
        
        return F.grid_sample(x, grid+randomFlow)

    def _GenImageAffine(self, x):
        N, _, rows, cols = x.shape # N is the batch size
        params = torch.randn((6, N, 1, 1)) * self.sigma

        #Generating the vector field for Affine transformation.
        X, Y = torch.meshgrid(torch.linspace(-1,1,rows),torch.linspace(-1,1,cols))
        X, Y = X.unsqueeze(0), Y.unsqueeze(0)
        Xv = params[0]*X + params[1]*Y + params[2]
        Yv = params[3]*X + params[4]*Y + params[5]
        
        randomFlow = torch.stack((Yv,Xv), axis=3).to(self.device)
        grid = torch.stack((Y,X), axis=3).to(self.device)
        
        return F.grid_sample(x, grid+randomFlow)

    def _GenImageDCT(self, x):

        batch_sz, num_channels, rows, cols = x.shape
        new_ros = torch.linspace(-1, 1, rows)
        new_cols = torch.linspace(-1, 1, cols)
        meshx, meshy = torch.meshgrid((new_ros, new_cols))
        grid = torch.stack((meshy, meshx), 2).unsqueeze(0).expand(batch_sz, rows, cols, 2).to(self.device)

        X, Y = torch.meshgrid((new_ros, new_cols))
        X = torch.reshape(X, (1, 1, 1, rows, cols))
        Y = torch.reshape(Y, (1, 1, 1, rows, cols))

        param_ab = torch.randn(batch_sz, self.num_bases, self.num_bases, 1, 2) * self.sigma
        a = param_ab[:, :, :, :, 0].unsqueeze(4)
        b = param_ab[:, :, :, :, 1].unsqueeze(4)
        K1 = torch.arange(self.num_bases).view(1, self.num_bases, 1, 1, 1)
        K2 = torch.arange(self.num_bases).view(1, 1, self.num_bases, 1, 1)
        basis_factors  = torch.cos( math.pi* (K1 * (X+0.5/rows) ))*torch.cos( math.pi * (K2 * (Y+0.5/cols)))

        U = torch.squeeze(torch.sum(a * basis_factors, dim=(1, 2)))
        V = torch.squeeze(torch.sum(b * basis_factors, dim=(1, 2)))

        randomFlow = torch.stack((V, U), dim=3).to(self.device)

        return F.grid_sample(x, grid + randomFlow)

    def _PerturbImage(self, x):
        return x + torch.randn_like(x)*self.sigma

    def forward(self, x):
        if self.aug_method == 'nominal':
            return self.base_classifier(x)
        if self.aug_method == 'pixel_perturbations':
            x = self._PerturbImage(x)
        elif self.aug_method == 'gaussianFull':
            x = self._deformImageGaussianFull(x)
        elif self.aug_method == 'rotation':
            x = self._GenImageRotation(x)
        elif self.aug_method == 'translation':
            x = self._GenImageTranslation(x)
        elif self.aug_method == 'affine':
            x = self._GenImageAffine(x)
        elif self.aug_method == 'scaling_uniform':
            x = self._GenImageScalingUniform(x)
        elif self.aug_method == 'DCT':
            x = self._GenImageDCT(x)
        else:
            raise Exception("Un identified augmentation method!")
        return self.base_classifier(x)
