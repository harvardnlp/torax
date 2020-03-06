import jax.numpy as jnp
import numpy as onp
import jax
from jax import grad, random, vmap, jit
from dataclasses import dataclass
from functools import partial
from typing import Tuple, Callable
import copy

def flatten_module(module):
  if not module._frozen:
    raise ValueError('Module not frozen.')
  return (module.variables(),), module

def unflatten_module(module, vars):
  if not module._frozen:
    raise ValueError('Module not frozen.')
  return module.updated(vars[0])

class ModuleMeta(type):
  def __init__(cls, name, bases, attrs):
    super().__init__(name, bases, attrs)
    jax.tree_util.register_pytree_node(cls, flatten_module, unflatten_module)

@dataclass(frozen=True)
class Variable:
  shape: Tuple[int]
  init_fn: Callable
  kind: str = 'state'

Parameter = partial(Variable, kind='param')

def _addindent(s_, numSpaces):                                                             s = s_.split('\n')                                                                     # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

class Module(metaclass=ModuleMeta):
  def __init__(self):
    self._frozen = False
    self._rng = None
    self._modules = {}
    self._variable_defs = {}
    self._variable_values = {}

  def init(self, rng):
    rng, child_rng, var_rng = random.split(rng, 3)
    if self._modules:
      child_rngs = random.split(child_rng, len(self._modules))
      for module, rng in zip(self._modules.values(), child_rngs):
        module.init(rng)
    var_rngs = random.split(var_rng, len(self._variable_defs))
    for (key, var_def), rng in zip(self._variable_defs.items(), var_rngs):
      if key not in self._variable_values:
        init_fn = var_def.init_fn
        self._variable_values[key] = init_fn(rng, var_def.shape)
    self._freeze()

  def _freeze(self):
    self._frozen = True
    for module in self._modules.values():
      module._freeze()

  def _unfreeze(self):
    self._frozen = False
    for module in self._modules.values():
      module._unfreeze()

  def variables(self, kind=None):
    vars = {}
    for key, modules in self._modules.items():
      vars[key] = modules.variables(kind=kind)
    for key, var in self._variable_defs.items():
      if kind is None or var.kind == kind:
        vars[key] = self._variable_values[key]
    return vars

  def params(self):
    return self.variables(kind='param')

  def update(self, updates):
    for key, value in updates.items():
      if key in self._modules:
        self._modules[key].update(value)
      else:
        setattr(self, key, value)

  def updated(self, updates):
    return self.mutate(lambda s: s.update(updates))

  def mutate(self, fn):
    new_self = copy.deepcopy(self)
    new_self._unfreeze()
    fn(new_self)
    new_self._freeze()
    return new_self

  def value_and_grad(self, fn, has_aux=False):
    def wrapper(params):
      model = self.updated(params)
      return fn(model)
    return jax.value_and_grad(wrapper, has_aux=has_aux)(self.params())

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def __getattr__(self, key):
    frozen = object.__getattribute__(self, '_frozen')
    if key in self._variable_defs:
      if key not in self._variable_values:
        raise ValueError(f'Variable named "{key}" is not initialized. Please call init')
      return self._variable_values[key]
    if key in self._modules:
      return self._modules[key]
    else:
      raise AttributeError(f'No attribute named "{key}"')

  def __setattr__(self, key, value):
    if key in ['_frozen', '_rng', '_modules', '_variable_defs', '_variable_values']:
      object.__setattr__(self, key, value)
    else:
      try:
        frozen = object.__getattribute__(self, '_frozen')
      except AttributeError:
        raise ValueError('You forgot to call super().__init__()')
      if frozen:
        raise ValueError('Module is frozen')
      if isinstance(value, Variable):
        self._variable_defs[key] = value
      elif isinstance(value, Module):
        self._modules[key] = value
      elif key in self._variable_defs:
        self._variable_values[key] = value
      else:
        object.__setattr__(self, key, value)

  def __deepcopy__(self, memo):
    new_self = copy.copy(self)
    self._frozen = False
    self._modules = {k: copy.deepcopy(v) for k, v in self._modules.items()}
    self._variable_defs = copy.copy(self._variable_defs)
    self._variable_values = copy.copy(self._variable_values)
    self._frozen = True
    return new_self

  def extra_repr(self):
      return ""

  def _get_name(self):
      return self.__class__.__name__

  def __repr__(self):
      "Mimic the pytorch pretty printer"

      extra_lines = []
      extra_repr = self.extra_repr()
      # empty string will be split into list ['']
      if extra_repr:
          extra_lines = extra_repr.split('\n')
      child_lines = []
      for key, module in self._modules.items():
          mod_str = repr(module)
          mod_str = _addindent(mod_str, 2)
          child_lines.append('(' + key + '): ' + mod_str)
      lines = extra_lines + child_lines

      main_str = self._get_name() + '('
      if lines:
          # simple one-liner info, which most builtin Modules will use
          if len(extra_lines) == 1 and not child_lines:
              main_str += extra_lines[0]
          else:
              main_str += '\n  ' + '\n  '.join(lines) + '\n'

      main_str += ')'
      return main_str
