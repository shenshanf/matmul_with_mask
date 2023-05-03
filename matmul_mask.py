import torch
from numba import cuda
from torch.autograd import Function
import math
import time


@cuda.jit(device=True)
def warpReduceSum(acc, warp_size=32):
    """
    accumulate values in the same warp thread
    with Warp-Level Primitives
    the result will in the first thread in every warp
    more detail see: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
    :param acc:
    :param warp_size:
    :return:
    """
    offset = warp_size
    while offset > 1:
        offset = offset // 2
        acc += cuda.shfl_down_sync(0xffffffff, acc, offset)
    return acc


@cuda.jit
def matmul_with_mask_forward_kernel(output, a, bt, mask_idxs):
    """
    The forward kernel function for masked matrix operation.
    Each block of threads handles one non-zero element,
    and 32 threads (one warp) within each block handle one channel.
    :param output: [nnz, 1] the values of output sparse tensor [B, N1, N2]
                   the output should init to zeros outside this function
    :param a:  [B, N1, C]
    :param bt: the transpose of b [B, N2, C]
    :param mask_idxs: [nnz, 3] the indices of non-zero element
    :return:
    """
    nnz = output.shape[0]
    Ch = a.shape[2]

    # Each block of threads handles one non-zero element
    tid = cuda.blockIdx.x

    # if blocks is less than nnz, each block should handle multiple nnz element
    while tid < nnz:
        # mask_idxs: [nnz, 3]：[B, N1, N2] indices of non-zero element
        batch_id = mask_idxs[tid, 0]
        N1_id = mask_idxs[tid, 1]
        N2_id = mask_idxs[tid, 2]

        #  The non-zero elements
        #  are obtained by taking the vector
        #  corresponding to the positions in matrices a and b.
        #  [bi, n1_i,:] * [bi, n2_i, :] = nnz
        aar = a[batch_id, N1_id]
        bar = bt[batch_id, N2_id]

        # The accumulation result of vector multiplication.
        accumulate = 0

        # 32 threads (one warp) within each block handle one channel
        ch_id = cuda.threadIdx.x
        warp_size = 32

        # The block size should be a multiple of 32 (one warp)
        assert cuda.blockDim.x % warp_size == 0

        while ch_id < Ch:
            accumulate += aar[ch_id] * bar[ch_id]
            ch_id += cuda.blockDim.x

        # warp reduce by cuda.shfl_down_sync
        # this implement could avoid to use share cuda memory
        accumulate = warpReduceSum(accumulate, warp_size)

        # after warp reduce, the result is in first thread in warp
        if cuda.threadIdx.x % 32 == 0:
            # use atomic add to avoid thread race
            cuda.atomic.add(output, tid, accumulate)
            # we don't need to synchronize threads
            # because every thread in different block handle output data independently

        # if cuda.threadIdx.x == 0:
        #     # after warp reduce, the result is in first thread in warp
        #     output[tid] = accumulate

        tid += cuda.gridDim.x


@cuda.jit
def matmul_with_mask_backward_kernel(
        grad_a, grad_bt,
        grad_output, a, bt,
        mask_idxs):
    nnz = grad_output.shape[0]
    Ch = a.shape[2]

    tid = cuda.blockIdx.x
    while tid < nnz:
        # mask_idxs: [nnz, 3]：[B, N1, N2] indices of non-zero element
        batch_id = mask_idxs[tid, 0]
        N1_id = mask_idxs[tid, 1]
        N2_id = mask_idxs[tid, 2]

        ch_id = cuda.threadIdx.x
        while ch_id < Ch:
            grad_a[batch_id, N1_id, ch_id] = grad_output[tid] * bt[batch_id, N2_id, ch_id]
            grad_bt[batch_id, N2_id, ch_id] = grad_output[tid] * a[batch_id, N1_id, ch_id]
            ch_id += cuda.blockDim.x

        tid += cuda.gridDim.x


class MatMultWithMask(Function):
    """
    Inherits from torch.autograd.Function
    and (forward) implements masked matrix multiplication
    and (backward) sparse back gradient computation.
    """

    @staticmethod
    def forward(ctx, a_, b_, mask):
        """

        :param ctx:
        :param a: [B, N1, C]
        :param b: [B, C, N2]
        :param mask: [3, nnz], coords: [indx_B, indx_N1, indx_N2]
        :return: sparse c with values and indices
        """
        a = a_.detach()  # detach at forward function
        b = b_.detach()

        # note: The tensor processed by
        #       the CUDA kernel function must be contiguous in memory
        bt = b.transpose(-1, -2).contiguous()  # [B, C, N2] -> [B, N2, C]

        # Transformed into an object that can be operated by Numba CUDA.
        cuda_a = cuda.as_cuda_array(a)
        cuda_bt = cuda.as_cuda_array(bt)

        mask_ = mask.coalesce()
        mask_indx = mask_.indices().transpose(0, 1).contiguous()  # [3, nnz] -> [nnz, 3]
        cuda_mask_indx = cuda.as_cuda_array(mask_indx)
        nnz = mask_indx.size(0)

        ctx.save_for_backward(a, bt, mask_indx)

        # the output should init to zero
        # because it is the initial value
        # for accumulation in the kernel function
        output = torch.zeros(nnz, dtype=a.dtype, device='cuda')  # [nnz, 1]
        cuda_output = cuda.as_cuda_array(output)

        # note:We create a one-dimensional grid of length nnz
        #      that does not exceed MAX_GRID_DIM_X to handle the non-zero elements.
        #      we create a one-dimensional thread block
        #      with a length slightly larger than
        #      the channel dimension of matrix A
        #      and a multiple of 32, which does not exceed the MAX_THREADS_PER_BLOCK,
        #      to handle the channel dimension of the data.
        grids = (min(nnz, cuda.get_current_device().MAX_GRID_DIM_X),)
        blocks = (min(int(math.ceil(a.size(-1) / 32) * 32),
                      cuda.get_current_device().MAX_THREADS_PER_BLOCK),)
        stream = cuda.stream()
        matmul_with_mask_forward_kernel[grids, blocks, stream](cuda_output,
                                                               cuda_a, cuda_bt,
                                                               cuda_mask_indx)
        return output, mask_indx.transpose(0, 1).contiguous()

    @staticmethod
    def backward(ctx, output_grad, _):
        # obtain tensor that will use in backward from forward function
        a, bt, mask_indx = ctx.saved_tensors

        nnz = mask_indx.size(0)

        cuda_a = cuda.as_cuda_array(a)
        cuda_bt = cuda.as_cuda_array(bt)

        # note: Gradients are only considered in the positions
        #       where the matrix multiplication is performed.
        grad_a = torch.zeros_like(a)
        grad_bt = torch.zeros_like(bt)

        cuda_grad_a = cuda.as_cuda_array(grad_a)
        cuda_grad_bt = cuda.as_cuda_array(grad_bt)
        cuda_output_grad = cuda.as_cuda_array(output_grad)
        cuda_mask_indx = cuda.as_cuda_array(mask_indx)

        # note:We create a one-dimensional grid of length nnz
        #      that does not exceed MAX_GRID_DIM_X to handle the non-zero elements.
        #      we create a one-dimensional thread block
        #      with a length slightly larger than
        #      the channel dimension of matrix A
        #      and a multiple of 32, which does not exceed the MAX_THREADS_PER_BLOCK,
        #      to handle the channel dimension of the data.
        grids = (min(nnz, cuda.get_current_device().MAX_GRID_DIM_X),)
        blocks = (min(int(math.ceil(a.size(-1) / 32) * 32),
                      cuda.get_current_device().MAX_THREADS_PER_BLOCK),)
        stream = cuda.stream()

        matmul_with_mask_backward_kernel[grids, blocks, stream](cuda_grad_a, cuda_grad_bt,
                                                                cuda_output_grad,
                                                                cuda_a, cuda_bt,
                                                                cuda_mask_indx)
        grad_b = grad_bt.transpose(-1, -2).contiguous()

        # we don't care the gradient of indices
        return grad_a, grad_b, None


#

def matmul_with_mask(a, b, mask):
    """
    - Batched matrix multiplication with a sparse mask
    that marks which elements of the tensors a and b should be multiplied.
    - This function outputs a sparse result and can automatically compute gradients.
    :param a: dence tensor [B, N1, C]
    :param b: dence tensor [B, C, N2]
    :param mask: sparse tensor [B, N1, N2],
                 indices: [coords=3, nnz], coords: B,N1,N2
    :return: c: sparse tensor [B, N1, N2],
                 indices: [coords=3, nnz], coords: B,N1,N2
    """
    assert a.dim() == 3
    assert b.dim() == 3
    assert a.size(-1) == b.size(-2)  # matrix multiplication
    assert a.size(0) == b.size(0)  # batch dimension should be same
    assert a.is_cuda
    assert b.is_cuda
    assert mask.is_cuda
    assert mask.is_sparse

    B, N1, N2 = mask.size()
    values, indexs = MatMultWithMask.apply(a, b, mask)
    sparse_c = torch.sparse_coo_tensor(values=values,
                                       indices=indexs, size=(B, N1, N2))
    return sparse_c


if __name__ == "__main__":
    B = 8
    H = 2560
    W = 640
    C = 1024

    a = torch.rand(B, H, C, requires_grad=True, dtype=torch.float).cuda()
    b = torch.rand(B, C, W, requires_grad=True, dtype=torch.float).cuda()

    #
    mask_p = torch.rand(B, H, W)

    #
    mask_dence = torch.where(mask_p >= 0.5,
                             torch.ones_like(mask_p),
                             torch.zeros_like(mask_p))

    mask_dence = mask_dence.requires_grad_()

    mask = mask_dence.to_sparse_coo().cuda()

    c = matmul_with_mask(a, b, mask)
    pass
    loss = c.sum()
    loss.backward()
    pass
    #

