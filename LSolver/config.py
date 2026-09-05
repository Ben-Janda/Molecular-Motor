from dataclasses import dataclass, field
from typing import Callable
import numpy as np

TimeDependentFunc = Callable[[float],float]

@dataclass(frozen=True)
class SimulationConfig:
    """Configuration of physical paramaters and discretization"""
    # Physical parameters
    gamma: float
    k_func: TimeDependentFunc
    D_func: TimeDependentFunc
    
    # Discretization
    t_max: float
    dt: float
    x0: float = 0.0
    
    # For evaluation
    t_array: np.ndarray = field(init=False, repr=False)
    num_steps: int = field(init=False, repr=False)
    
    def __post_init__(self):
        n_steps = int(np.ceil(self.t_max / self.dt))
        t_arr = np.linspace(0, self.t_max, n_steps)
        
        object.__setattr__(self, 'num_steps', n_steps)
        object.__setattr__(self, 't_array', t_arr)
        
        if self.gamma <= 0:
            raise ValueError("Friction coefficient gamma has to be positive.")
        if self.dt <= 0:
            raise ValueError("Timestep dt has to be positive.")