from . import base

from jax import lax
from jax.nn import initializers
import jax.numpy as jnp


class Linear(Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.kernel = Parameter((in_features, out_features),
                            jax.nn.initializers.lecun_normal())
    self.bias = Parameter((out_features,),
                          jax.nn.initializers.zeros)

  def forward(self, x):
    return np.dot(x, self.kernel) + self.bias

_no_init = lambda rng, shape: ()

def _absolute_dims(rank, dims):
  return tuple([rank + dim if dim < 0 else dim for dim in dims])


class Conv(Module):
  def __init__(self, features, kernel_size):
    init, f = GeneralConv(2, features, filter_shape=kernel_size)
    W, b = init(features, self._rng)
    self.weights = w
    self.bias = b

  def forward(self, x):
    pass

class LayerNorm(Module):
  """Layer normalization (https://arxiv.org/abs/1607.06450).
  Operates on the last axis of the input data.
  """
  def __init__(self):
    super().__init__()
    self.bias = Parameters((features,))
    self.scale = Parameters((features,))

  def forward(self,
              x,
              epsilon=1e-6,
              bias=True,
              scale=True):
    """Applies layer normalization on the input.
    It normalizes the activations of the layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within
    each example close to 0 and the activation standard deviation close to 1.
    Args:
      x: the inputs
      epsilon: A small float added to variance to avoid dividing by zero.
      dtype: the dtype of the computation (default: float32).
      bias:  If True, bias (beta) is added.
      scale: If True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: Initializer for bias, by default, zero.
      scale_init: Initializer for scale, by default, one.
    Returns:
      Normalized inputs (the same shape as inputs).
    """
    features = x.shape[-1]
    mean = jnp.mean(x, axis=-1, keepdims=True)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    var = mean2 - lax.square(mean)
    mul = lax.rsqrt(var + epsilon)
    if scale:
      mul = mul * self.scale
    y = (x - mean) * mul
    if bias:
      y = y + self.bias
    return y
