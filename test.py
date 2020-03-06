import jax.numpy as jnp
import numpy as onp
import jax
from jax import grad, random, vmap, jit
from dataclasses import dataclass
from functools import partial
from typing import Tuple, Callable
import copy


class Dense(Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.kernel = Parameter((in_features, out_features), jax.nn.initializers.lecun_normal())
    self.bias = Parameter((out_features,), jax.nn.initializers.zeros)

  def forward(self, x):
    return jnp.dot(x, self.kernel) + self.bias

class MLP(Module):
  def __init__(self, in_features, hidden_features, out_features):
    super().__init__()
    self.dense1 = Dense(in_features, hidden_features)
    self.dense2 = Dense(hidden_features, out_features)

  def forward(self, x):
    x = self.dense1(x)
    x = jax.nn.relu(x)
    return self.dense2(x)

model = MLP(3, 4, 2)
model.init(random.PRNGKey(0))
model

model(jnp.ones((3,)))

model.variables()

def loss_fn(model):
  return sum(jnp.sum(p ** 2) for p in jax.tree_leaves(model.params()))

l2, grad = model.value_and_grad(loss_fn)
print(l2, grad)
update = jax.tree_multimap(lambda p, g: p - 0.1 * g, model.params(), grad)
new_model = model.updated(update)
loss_fn(new_model)

# because Module is registered as a pytree it can be passed into jitted functions...
@jax.jit
def optimize(model):
  l2, grad = model.value_and_grad(loss_fn)
  update = jax.tree_multimap(lambda p, g: p - 0.1 * g, model.params(), grad)
  new_model = model.updated(update)
  return l2, loss_fn(new_model), new_model

optimize(model)

# a mutable module should not be passed into jit because it is not garantueed to work for inpure functions
model.mutate(optimize)
