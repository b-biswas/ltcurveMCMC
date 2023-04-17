#define the parametric function of the lightcurve
import tensorflow_probability.substrates.jax as tfp
import jax.numpy as jnp

def parametric_fn(t0, t_rise, t_fall, amplitude, x_range):
    
    time_dif = x_range - t0
    numerator = jnp.exp(-time_dif/t_fall)
    denominator = 1 + jnp.exp(-time_dif/t_rise)
    predicted_lc = amplitude * ( numerator/ denominator )
    return predicted_lc