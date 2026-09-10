import numpy as np
from .config import SimulationConfig

def solver(config: SimulationConfig, rng: np.random.Generator = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate single trajectory x(t) and stochastic time y(t)

    Parameters
    ----------
    config : SimulationConfig
        Configuration of the simulation.
    rng : np.random.Generator
        Optional rng for reproducibility.

    Returns
    -------
    Tuple (x(t), y(t)).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    N = config.num_steps
    dt = config.dt
    gamma = config.gamma
    t_arr = config.t_array
    
    x = np.empty(N)
    y = np.empty(N)
    x[0] = config.x0
    y[0] = 0.0
    eta_x = rng.normal(0, 1, N-1)
    eta_y = rng.normal(0, 1, N-1)
    
    for i in range(N-1):
        t = t_arr[i]
        kt = config.k_func(t)
        Dt = config.D_func(t)
        sigma = np.sqrt(2*Dt*dt)
        
        dx = -(kt/gamma) * x[i] * dt + sigma * eta_x[i]
        dy = sigma * eta_y[i]
        
        x[i+1] = x[i] + dx
        y[i+1] = y[i] + dy
        
    return x, y