from abc import ABC, abstractmethod
import torch

class UQSampler(ABC):
    def __init__(self, N, is_uq_model):
        self.N = N # number of times to sample
        self.is_uq_model = is_uq_model # flag saying if model has a set apply function to call
        
    def set_N(self, N):
        self.N = N
        
    def get_N(self):
        return self.N
    
    def _gen_samples(self, x, model, **kwargs):
        samples = []
        for _ in range(self.N):
            samples.append(model(x, **kwargs))
        samples = torch.stack(samples, dim=1)
        return samples
            
    def __call__(self, x, model, **kwargs):
        samples = self._gen_samples(x, model, **kwargs)
        mle_est = self._gen_mle_estimate(x, samples, model, **kwargs)
        return samples, mle_est
        
    @abstractmethod
    def _gen_mle_estimate(self, x, samples, model, **kwargs):
        pass
    
    
class MeanUQSampler(UQSampler):
    def _gen_mle_estimate(self, x, samples, model, **kwargs):
        return torch.mean(samples, dim=0)
        
        
class ModeUQSampler(UQSampler):
    """
    this is what we should use for semantic segmentation
    https://arxiv.org/pdf/1807.07356.pdf (Section 3)
    """
    def __init__(self, apply_thresholding=False, threshold=0.7, apply_binning=False, bin_decimal=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_thresholding = apply_thresholding
        self.threshold = threshold
        self.apply_binning = apply_binning
        self.bin_decimal = bin_decimal
        
        if apply_thresholding and apply_binning:
            raise ValueError("can only do one of thresholding or binning, not both")
    
    def _gen_mle_estimate(self, x, samples, model, **kwargs):
        if self.apply_thresholding:
            samples = samples > self.threshold
        if self.apply_binning:
            samples = torch.round(samples, decimals=self.bin_decimal)
        
        return torch.mode(samples, dim=0)
    
    
class DeterministicModelUQSampler(UQSampler):
    def _gen_mle_estimate(self, x, samples, model, **kwargs):
        if not self.is_uq_model:
            raise ValueError("is_uq_model flag is false, i.e model cannot be set to deterministic mode")
        curr_state = model.get_applyfunc()
        if curr_state != False:
            model.set_applyfunc(False)
        est = model(x, **kwargs)
        model.set_applyfunc(curr_state)
        return est
    
    
class DeterministicModelNoSample(DeterministicModelUQSampler):
    def __call__(self, x, model, **kwargs):
        mle_est = self._gen_mle_estimate(x, None, model, **kwargs)
        return None, mle_est 
    
class NonDeterministicModelNoSample(DeterministicModelUQSampler):
    def _gen_mle_estimate(self, x, samples, model, **kwargs):
        if not self.is_uq_model:
            raise ValueError("is_uq_model flag is false, i.e model cannot be set to deterministic mode")
        curr_state = model.get_applyfunc()
        if curr_state != True:
            model.set_applyfunc(True)
        est = model(x, **kwargs)
        model.set_applyfunc(curr_state)
        return est
    
    def __call__(self, x, model, **kwargs):
        mle_est = self._gen_mle_estimate(x, None, model, **kwargs)
        return None, mle_est 