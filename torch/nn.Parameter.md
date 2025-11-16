# nn.Parameter

`Parameter`类在`torch/nn/parameter.py`中定义，是`torch.Tensor`的子类，只有四个方法：`__deepcopy__`、`__repr__`、`__reduce_ex__`和`__new__`，其中`__repr__`跟字符串表示相关，`__deepcopy__`跟拷贝相关，`__reduce_ex__`用于序列化，这里只关注`__new__`。

```python
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.empty(0)
        if type(data) is torch.Tensor or type(data) is Parameter:
            # For ease of BC maintenance, keep this path for standard Tensor.
            # Eventually (tm), we should change the behavior for standard Tensor to match.
            return torch.Tensor._make_subclass(cls, data, requires_grad)

        # Path for custom tensors: set a flag on the instance to indicate parameter-ness.
        t = data.detach().requires_grad_(requires_grad)
        if type(t) is not type(data):
            raise RuntimeError(f"Creating a Parameter from an instance of type {type(data).__name__} "
                               "requires that detach() returns an instance of the same type, but return "
                               f"type {type(t).__name__} was found instead. To use the type as a "
                               "Parameter, please correct the detach() semantics defined by "
                               "its __torch_dispatch__() implementation.")
        t._is_param = True
        return t
```

这个魔法函数包含两个参数：`Tensor`类型的`data`和`bool`类型的`requires_grad`。

1. 当`data`是`Tensor`类型或`data`是`Parameter`，调用`_mask_subclass`方法。

2. 其他情况，直接调用`data.detach()`生成返回值，并设置`requires_grad`属性，令`_is_param`为`True`。

所以实际上，`nn.Parameter`的关键内容需要从`torch.Tensor`中找了。
