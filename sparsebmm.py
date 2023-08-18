import torch
from matmul_mask import matmul_with_mask


class SparseBMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        a = a.coalesce()
        r = torch.bmm(a, b)
        ctx.save_for_backward(a, b)
        return r

    @staticmethod
    def backward(ctx, grad):

        a, b = ctx.saved_tensors

        # gradients w.r.t. a
        ga = None
        if ctx.needs_input_grad[0]:
            ga = matmul_with_mask(grad, b.transpose(-2, -1), a)

        # gradients w.r.t. b
        gb = None
        if ctx.needs_input_grad[1]:
            gb = a.transpose(1, 2).bmm(grad)

        return ga, gb


def sparse_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Batch matrix multiply between a sparse matrix and a dense matrix
    """
    assert a.ndim == b.ndim == 3
    assert a.shape[0] == b.shape[0]
    # assert a.is_sparse
    return SparseBMM.apply(a, b)


if __name__ == '__main__':
    # a = torch.tensor([[1, 0, 1, 1],
    #                   [0, 1, 0, 0],
    #                   [2, 1, 0, 0]],
    #                  dtype=torch.float32,
    #                  requires_grad=True, device='cuda')
    # a = a.repeat(2, 1, 1)
    a = torch.rand(4, 640, 720, requires_grad=True, device='cuda')

    b = torch.rand_like(a, requires_grad=True)
    b = b.permute(0, 2, 1)

    sp_a = a.to_sparse()

    c = sparse_bmm(sp_a, b)

    # 验证前向运算：
    check_c = torch.bmm(a, b)
    print("前向运算结果：", torch.allclose(check_c, c))

    #
    grad_sp_a, grad_b = torch.autograd.grad(c.sum(), (sp_a, b))
    # grad_a = grad_sp_a.to_dense()

    pass

    init_grad_o = torch.ones_like(c)
    check_grad_a = torch.bmm(init_grad_o, b.permute(0, 2, 1))
    check_grad_a = check_grad_a.sparse_mask(grad_sp_a)
    print("grad_a计算结果：", torch.allclose(grad_sp_a.to_dense(), check_grad_a.to_dense()))

    check_grad_b = torch.bmm(a.permute(0, 2, 1), init_grad_o)
    print("grad_b计算结果：", torch.allclose(check_grad_b, check_grad_b))

    pass
