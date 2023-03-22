import numpy as np

__all__ = ["drift_diffusion_integrator"]


def drift_diffusion_integrator(
    starting_point: float,
    non_decision_time: float,
    drift_rate,
    threshold: float,
    noise: float = 1.0,
    dt: float = 0.01,
) -> float:
    """
    Integrates the drift diffusion model for a single trial using and implementation of
    the using the Euler-Maruyama method. This is a proof of concept implementation and
    is not optimized for speed.

    Args:
        starting_point: The starting point of the particle
        non_decision_time: The non-decision time
        drift_rate: The deterministic drift rate of the particle
        threshold: The threshold to cross, the boundary is assumed to be symmetric.
        noise: The standard deviation of the noise
        dt: The time step to use for the integration

    Returns:
        The time it took to cross the threshold, with the sign indicating the direction
    """
    particle = starting_point

    # Integrate the Wiener process until it crosses the threshold
    t = 0
    while abs(particle) < threshold:
        particle = particle + np.random.normal(
            loc=drift_rate * dt, scale=noise * np.sqrt(dt)
        )
        t = t + 1

    # Return the time it took to cross the threshold, with the sign indicating the direction
    # Add the non-decision time to the RT
    return (
        non_decision_time + t * dt
        if particle > threshold
        else -non_decision_time - t * dt
    )
