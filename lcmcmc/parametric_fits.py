#define the parametric function of the lightcurve
import tensorflow_probability.substrates.jax as tfp
import jax.numpy as jnp

def parametric_fn(t0, t_rise, t_fall, amplitude, x_range):
    
    time_dif = x_range - t0
    numerator = jnp.exp(-time_dif/t_fall)
    denominator = 1 + jnp.exp(-time_dif/t_rise)
    predicted_lc = amplitude * ( numerator/ denominator )
    return predicted_lc

def parametric_fn_pcs(c1, c2, c3 , pcs, positions):
    index = positions
    predicted_lc = pcs[0][index[:]]*c1 + pcs[1][index[:]]*c2 + pcs[2][index[:]]*c3
    return predicted_lc