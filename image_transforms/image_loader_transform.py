

# Custom transform that applies the permutation
class LoaderTransformImagePermutation:
    def __init__(self, perm_transform):
        self.perm_transform = perm_transform

    def __call__(self, img):
        # Add batch dimension, apply permutation, remove batch dimension
        img_batch = img.unsqueeze(0)
        img_permuted = self.perm_transform(img_batch)
        return img_permuted.squeeze(0)