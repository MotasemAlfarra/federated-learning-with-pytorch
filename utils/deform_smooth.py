import math
import torch
import torch.nn.functional as F
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint


class SmoothFlow(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, certify_method : str, sigma: float, device='cuda'):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.certify_method = certify_method
        self.sigma = sigma
        self.device = device
        self.num_bases = 2

    def _GenDeformGaussian(self, imgs, N, device):
        ''' This function takes an image C x W x H and returns N Gaussianly perturbed coordinates versions
        '''       
        batch = imgs.repeat((N, 1, 1, 1))
        num_channels, rows, cols = imgs.shape
        randomFlow = torch.randn(N, rows, cols, 2, device=device) * self.sigma

        new_ros = torch.linspace(-1, 1, rows)
        new_cols = torch.linspace(-1, 1, cols)

        meshx, meshy = torch.meshgrid((new_ros, new_cols))
        grid = torch.stack((meshy, meshx), 2).unsqueeze(0).expand(N, rows, cols, 2).to(device)

        new_grid = grid + randomFlow

        Iwarp = F.grid_sample(batch, new_grid)
        return Iwarp


    def _GenImageRotation(self, x, N):
        _, rows, cols = x.shape #Usually in certification, the batch size is 1
        ang = (-2 * torch.rand((N, 1, 1)) + 1) *self.sigma #Uniform between [-sigma, sigma]
        
        #Generating the vector field for rotation. Not that sigma should be sig*pi, where sig is in [0,1]
        X, Y = torch.meshgrid(torch.linspace(-1,1,rows),torch.linspace(-1,1,cols))
        X, Y = X.unsqueeze(0), Y.unsqueeze(0)
        Xv = X*torch.cos(ang)-Y*torch.sin(ang)-X
        Yv = X*torch.sin(ang)+Y*torch.cos(ang)-Y
        
        randomFlow = torch.stack((Yv,Xv), axis=3).to(self.device)
        grid = torch.stack((Y,X), axis=3).to(self.device)
        
        return F.grid_sample(x.repeat((N, 1, 1, 1)), grid+randomFlow)

    def _GenImageTranslation(self, x, N):
        _, rows, cols = x.shape #N is the batch size

        #Generating the vector field for translation.
        X, Y = torch.meshgrid(torch.linspace(-1,1,rows),torch.linspace(-1,1,cols))
        X, Y = X.unsqueeze(0), Y.unsqueeze(0)
        Xv = torch.randn((N, 1, 1))*self.sigma + 0*X
        Yv = torch.randn((N, 1, 1))*self.sigma + 0*Y
        
        randomFlow = torch.stack((Yv,Xv), axis=3).to(self.device)
        grid = torch.stack((Y,X), axis=3).to(self.device)
        
        return F.grid_sample(x.repeat((N, 1, 1, 1)), grid+randomFlow)
    
    def _GenImageScalingUniform(self, x, N):
        _, rows, cols = x.shape # N is the batch size
        #Scaling here is sampled from uniform distribution between [1-sigma, 1+sigma]
        scale = (-2 * torch.rand((N, 1, 1)) + 1.0) * self.sigma + 1.0
        #Generating the vector field for scaling.
        X, Y = torch.meshgrid(torch.linspace(-1,1,rows),torch.linspace(-1,1,cols))
        X, Y = X.unsqueeze(0), Y.unsqueeze(0)
        Xv = X * scale - X
        Yv = Y * scale - Y
        
        randomFlow = torch.stack((Yv,Xv), axis=3).to(self.device)
        grid = torch.stack((Y,X), axis=3).to(self.device)
        
        return F.grid_sample(x.repeat((N, 1, 1, 1)), grid+randomFlow)

    def _GenImageAffine(self, x, N):
        _, rows, cols = x.shape # N is the batch size
        
        params = torch.randn((6, N, 1, 1))*self.sigma

        #Generating the vector field for Affine transformation.
        X, Y = torch.meshgrid(torch.linspace(-1,1,rows),torch.linspace(-1,1,cols))
        X, Y = X.unsqueeze(0), Y.unsqueeze(0)
        Xv = params[0]*X + params[1]*Y + params[2]
        Yv = params[3]*X + params[4]*Y + params[5]
        
        randomFlow = torch.stack((Yv,Xv), axis=3).to(self.device)
        grid = torch.stack((Y,X), axis=3).to(self.device)
        
        return F.grid_sample(x.repeat((N, 1, 1, 1)), grid+randomFlow)

    def _GenImageDCT(self, x, N):

        _, rows, cols = x.shape
        new_ros = torch.linspace(-1, 1, rows)
        new_cols = torch.linspace(-1, 1, cols)
        meshx, meshy = torch.meshgrid((new_ros, new_cols))
        grid = torch.stack((meshy, meshx), 2).unsqueeze(0).expand(N, rows, cols, 2).to(self.device)

        X, Y = torch.meshgrid((new_ros, new_cols))
        X = torch.reshape(X, (1, 1, 1, rows, cols))
        Y = torch.reshape(Y, (1, 1, 1, rows, cols))

        param_ab = torch.randn(N, self.num_bases, self.num_bases, 1, 2) * self.sigma
        a = param_ab[:, :, :, :, 0].unsqueeze(4)
        b = param_ab[:, :, :, :, 1].unsqueeze(4)
        K1 = torch.arange(self.num_bases).view(1, self.num_bases, 1, 1, 1)
        K2 = torch.arange(self.num_bases).view(1, 1, self.num_bases, 1, 1)
        basis_factors  = torch.cos( math.pi* (K1 * (X+0.5/rows) ))*torch.cos( math.pi * (K2 * (Y+0.5/cols)))

        U = torch.squeeze(torch.sum(a * basis_factors, dim=(1, 2)))
        V = torch.squeeze(torch.sum(b * basis_factors, dim=(1, 2)))

        randomFlow = torch.stack((V, U), dim=3).to(self.device)

        return F.grid_sample(x.repeat((N, 1, 1, 1)), grid + randomFlow)

    def _PerturbImage(self, x, N):
        x = x.repeat(N, 1, 1, 1)
        return x + torch.randn_like(x)*self.sigma

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.
        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return SmoothFlow.ABSTAIN, 0.0, 0.5#0.5 for the radius to be zero
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius, pABar

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return SmoothFlow.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                if self.certify_method == 'gaussianFull':
                    batch = self._GenDeformGaussian(x, this_batch_size, device='cuda')
                elif self.certify_method =='pixel_perturbations':
                    batch = self._PerturbImage(x, this_batch_size)
                elif self.certify_method == 'rotation':
                    batch = self._GenImageRotation(x, this_batch_size)
                elif self.certify_method == 'translation':
                    batch = self._GenImageTranslation(x, this_batch_size)
                elif self.certify_method == 'affine':
                    batch = self._GenImageAffine(x, this_batch_size)
                elif self.certify_method == 'scaling_uniform':
                    batch = self._GenImageScalingUniform(x, this_batch_size)
                elif self.certify_method == 'DCT':
                    batch = self._GenImageDCT(x, this_batch_size)
                else:
                    raise Exception('Undefined augmentaion method!')
                predictions = self.base_classifier(batch).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
