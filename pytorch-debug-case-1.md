# PyTorch debug 一则

## 现象

``` python
>>> a = torch.ones(2,3)
>>> a_gpu = a.cuda()
>>> a_mlu = a.to("mlu")
>>> a[:,1:2] -= a[0,1]
>>> a
	tensor([[1., 0., 1.], [1., 1., 1.]])

>>> a_gpu[:,1:2] -= a_gpu[0,1]
>>> a_gpu
    tensor([[1., 0., 1.], [1., 0., 1.]], device='cuda:0')

>>> a_mlu[:,1:2] -= a_mlu[0,1]
>>> a_mlu
	tensor([[1., 0., 1.], [1., 0., 1.]], device='mlu:0')
```

可以看到，上面发生了 `data race`，造成未定义行为使得 GPU & MLU 与 CPU 行为不一致。

## 分析与结论

> When accessing the contents of a tensor via indexing, PyTorch follows Numpy behaviors that basic indexing returns **views**, while advanced indexing returns a **copy**.  - *from [tensor view doc](https://pytorch.org/docs/stable/tensor_view.html)*

该例中右值 `a[0,1]` 为 `basic indexing`，因此是一个 `lazy op`，返回为 `view`，而并未将数据拷贝至新的内存中，因此在计算过程中会发生 `data race`，得到未定义结果。可以看到：

- CPU 在计算时是按元素计算的，因此当 `a[0,1]` 在第一次计算后变成了 `0`，其后面的应减 `a[0,1]` 的元素（`a[0,2], a[1,1], a[1,2]`）都减了新的 `a[0,1]` 的取值 `0`，因此仍保持原值 `1`。
- GPU 和 MLU 在计算时是按列计算的，因此可以看到第 1 列都减成了 0， 而第 2 列减的是新的 `a[0,1]` 值 `0`，因此仍保持原值 `1`。

因此对 overlapped indexing 同时执行读和写操作，可能会发生 data race 造成结果未定义。

PyTorch 对 overlapped indexing 的使用是有警告的: [*If the constructed tensor is "overlapped" (with multiple indices referring to the same element in memory) its behavior is undefined.*](https://github.com/pytorch/pytorch/blob/master/torch/_torch_docs.py)

CUDA 也有相似警告: [*The two threads read and write from the same memory locations X and Y simultaneously. Any data-race is undefined behaviour, and has no defined semantics. The resulting values for A and B can be anything.*](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) 

## References
1. [ezyang's blog](http://blog.ezyang.com/)