'''
Module focused on the implementation of Averaged Higher Order SVD
(AHOSVD) technique.
'''

import torch
import numpy as np

from smithers.ml.models.hosvd import HOSVD

class AHOSVD():
    """
    Class that handles the construction of the functions needed
    to perform the dimensionality reduction of a given tensor,
    which has exactly one dimension (the first one) that is too large,
    thus preventing standard HOSVD to run on current architectures.

    This new technique is called Averaged Higher Order Singular
    Value Decomposition (AHOSVD). Basically, HOSVD is performed on
    batches of the outputs of the premodel, then the U matrices
    resulting from HOSVD relative to the same unfolding direction
    are averaged in order to keep the computing requirements (mainly
    GPU storage) accessible.

    :param torch.Tensor tensor: snapshots tensor of dimensions
        (n,d_1,...,d_n)
    :param list[int] mode_number: list of integers representing the
        target reduced dimensions; (d_1,...,d_n) will be reduced
    :param int batch_len: number of element of the snapshot to process
        together
    """
    def __init__(self, tensor, red_dims, batch_len):

        self.tensor = tensor
        self.red_dims = red_dims
        if batch_len > tensor.size()[0] > 0:
            raise ValueError('The batch for AHOSVD must be smaller than the' +
                             ' batch size of the data loader.')
        self.batch_len = batch_len
        self.u_matrices = []
        self.proj_matrices = []

    def incremental_average(self, current_list, new_list, index):
        """
        Auxiliary function used to compute the incremental step for a list
        containing already computed
        averages when another list of new values is given
        :param list current_list: list containing the current averages
        :param list new_list: list of the new values
        :param int index: defines the number of elements the current average is taken over
        :return: the updated list of averages
        :rtype: list
        """
        matrices_list = []
        if index == 0:
            return new_list
        elif index > 0:
            for i, _ in enumerate(current_list):
                matrices_list.append((index / (index + 1)) * current_list[i] +
                                     (1/(index + 1)) * new_list[i])
            return matrices_list
        elif index < 0:
            raise ValueError('Index variable must be greater or equal to 0.')

    def _partial_hosvd(self, batch_from_tensor):
        """
        Computes the partial HOSVD from a restricted sample of the snapshots tensor

        :param torch.Tensor batch_from_tensor: the batch given
        :return: list of U matrices coming from the modal SVDs
        :rtype: list[torch.Tensor]
        """
        hosvd = HOSVD(batch_from_tensor.shape)
        hosvd.fit(batch_from_tensor, return_S_tensor=False, for_AHOSVD=True)
        return hosvd.modes_matrices


    def compute_u_matrices(self):
        """
        This function updates the current U matrices with their new values
        from a never-seen batch of examples.
        """
        for idx_ in range(int(np.floor(self.tensor.shape[0]/self.batch_len))):
            p_hosvd = self._partial_hosvd(self.tensor[idx_ * self.batch_len : (idx_+1) * self.batch_len])
            self.u_matrices = self.incremental_average(self.u_matrices, p_hosvd, idx_)

    def compute_proj_matrices(self):
        """
        This function sets the attribute proj_matrices with the transposes of
        the matrices obtained from the numbers given in self.red_dims
        of columns of the U matrices previously computed
        """
        for i in range(len(self.u_matrices)):
            self.proj_matrices.append(self.u_matrices[i][ :, : self.red_dims[i]].t().conj())

# example
if __name__ == '__main__':
    import time
    tensor1 = torch.randn(50000, 4, 4, 256).to('cuda')
    start = time.time()
    ahosvd = AHOSVD(tensor1, [3, 3, 50], 20)
    ahosvd.compute_u_matrices()
    print(f"The U matrices' dimensions are {[ahosvd.u_matrices[i].shape for i in range(len(ahosvd.u_matrices))]}")
    ahosvd.compute_proj_matrices()
    end = time.time()
    print(f'time needed: {end-start} seconds')
