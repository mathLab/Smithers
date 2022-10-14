import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class tensor_product_layer(nn.Module):
    """
    Class that handles the definition of a PyTorch layer that created to
    compute several specific tensor products: given a list of matrices and a tensor,
    its job is to multiply the i-th one dimensional sections of the tensor by the i-th
    matrix of the list.

    """
    def __init__(self, list_of_matrices):
        """
        :param list[torch.Tensor] list_of_matrices: list of the matrices that will multiply the one dimensional sections of a given tensor
        """
        super(tensor_product_layer, self).__init__()
        self.list_of_matrices = list_of_matrices
        self.param0 = Parameter(list_of_matrices[0])
        self.param1 = Parameter(list_of_matrices[1])
        self.param2 = Parameter(list_of_matrices[2])

    def tensor_reverse(self, tensor):
        """
        Function that reverses the directions of a tensor

        :param torch.Tensor A: the input tensor with dimensions (d_1,d_2,...,d_n)
        :return: input tensor with reversed dimensions (d_n,...,d_2,d_1)
        :rtype: torch
        """
        incr_list = [i for i in range(len(tensor.shape))]
        incr_list.reverse()
        return torch.permute(tensor, tuple(incr_list))

    def forward(self, input_tensor):
        """
        Forward function of the layer. The if clause concern tha case in which a single tensor 
        needs to be projected, the else clause deals with the possibility of multiple tensors being provided
        
        :param torch.Tensor input_tensor: the input tensor (either single tensor or tensor as a collection of tensors)
        :return: the projected tensor (either full or its "components")
        :rtype: torch.Tensor
        """
        if len(input_tensor.shape) == len(self.list_of_matrices):
            for i, _ in enumerate(input_tensor.shape):
                input_tensor = torch.tensordot(self.list_of_matrices[i], input_tensor, ([1],[i]))
            return self.tensor_reverse(input_tensor) 
        elif len(input_tensor.shape) == len(self.list_of_matrices) + 1:
            for i in range(len(self.list_of_matrices)):
                input_tensor = torch.tensordot(self.list_of_matrices[i], input_tensor, ([1],[i+1]))
            return self.tensor_reverse(input_tensor)

    def extra_repr(self):
        return 'in_dimensions={}, out_dimensions={}'.format([self.list_of_matrices[i].shape[1] for i in range(len(self.list_of_matrices))], [self.list_of_matrices[i].shape[0] for i in range(len(self.list_of_matrices))])

if __name__ == '__main__':
    from smithers.ml.AHOSVD import AHOSVD
    tensor_batch = torch.randn(100, 256, 4, 4).to('cuda')
    tensor_image = torch.randn(256, 4, 4).to('cuda')
    ahosvd = AHOSVD(tensor_batch, [25, 50, 3, 3], 25)
    ahosvd.compute_u_matrices()
    ahosvd.compute_proj_matrices()
    my_layer = tensor_product_layer(ahosvd.proj_matrices)
    projected_obs = my_layer.forward(tensor_image)
    print(projected_obs.shape)