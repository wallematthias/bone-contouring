# Frozen upstream scientific test oracle: Nathan Neeteson et al.
# Bonelab/HR-pQCT-Segmentation, commit 9aef696, GPL-3.0-only.
import numpy as np


class SampleStandardizer(object):
    # this transformation object is very simple. At creation, a mean and
    # standard deviation are specified. When a sample is passed through, the
    # image portion of the sample is normalized so that the intensity values
    # are capped to the given range, then scaled to the interval [-1,1]
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, sample):
        sample['image'] = np.minimum(np.maximum(sample['image'], self.min_val), self.max_val)
        sample['image'] = (2 * sample['image'] - self.max_val - self.min_val) / (self.max_val - self.min_val)

        return sample
