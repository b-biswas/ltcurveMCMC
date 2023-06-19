
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from lcmcmc.parametric_fits import parametric_fn, parametric_fn_pcs

tfd = tfp.distributions
#rng = jax.random.PRNGKey(0)

# Disdistribution with interband correlations
# TODO: find the actual correlation between the bands.

def jd_model(index, x_range):

    assert index.shape[0] == x_range.shape[0]

    num_channel = np.unique(index[:, 1]).shape[0]
    num_event = np.unique(index[:, 0]).shape[0]

    @tfd.JointDistributionCoroutineAutoBatched
    def current_event():
        # define priors
        t0_hyper = yield tfd.Sample(tfd.Uniform(-2, 2), (num_event), name="t0_hyper")
        t0 = yield tfd.Sample(tfd.Uniform(t0_hyper-1, t0_hyper+1), (num_channel), name="t0")

        t_rise_hyper_prior = yield tfd.Sample(tfd.Uniform(.25, 10), (num_event), name="t_rise_hyper")
        t_rise = yield tfd.Sample(tfd.Uniform(t_rise_hyper_prior - t_rise_hyper_prior/4, t_rise_hyper_prior + t_rise_hyper_prior/4), (num_channel), name="t_rise")

        t_fall_hyper_prior = yield tfd.Sample(tfd.Uniform(.25, 10), (num_event), name="t_fall_hyper_prior")
        t_fall_= yield tfd.Sample(tfd.Uniform(t_fall_hyper_prior - t_fall_hyper_prior/4, t_fall_hyper_prior + t_fall_hyper_prior/4), (num_channel), name="t_fall_")
        
        t_fall = t_rise + t_fall_

        amplitude = yield tfd.Sample(tfd.Gamma(10, 5), (num_event, num_channel), name="amp")
        # evaluate the predictions
        prediction = parametric_fn(
            t0[index[:, 0], index[:, 1]],
            t_rise[index[:, 0], index[:, 1]],
            t_fall[index[:, 0], index[:, 1]],
            amplitude[index[:, 0], index[:, 1]], 
            x_range,
        )

        # Likelihood
        sigma = yield tfd.Sample(tfd.Gamma(1, 1), len(index), name="sigma")
        yield tfd.Normal(prediction, sigma, name="obs")

    return current_event

def jd_model_pcs(index, x_range, pcs):

    assert index.shape[0] == x_range.shape[0]

    positions = np.around(
            (x_range).astype(float) / .25
            + (400 - 1) / 2
        )
    positions = positions.astype(int)

    num_channel = np.unique(index[:, 1]).shape[0]
    num_event = np.unique(index[:, 0]).shape[0]

    @tfd.JointDistributionCoroutineAutoBatched
    def current_event():
        # define priors
        # c1 = yield tfd.Sample(tfd.Normal(1, .2), (num_event, num_channel), name="c1")

        c2_hyper = yield tfd.Sample(tfd.Normal(0, .1), (num_event), name="c2_hyper")
        c2 = yield tfd.Sample(tfd.Normal(c2_hyper, .01), (num_channel), name="c2")

        c3_hyper = yield tfd.Sample(tfd.Normal(0, .02), (num_event), name="c3_hyper")
        c3 = yield tfd.Sample(tfd.Normal(c3_hyper, .01), (num_channel), name="c3")

        c1_hyper = yield tfd.Sample(tfd.Uniform(.9, 1.2), (num_event), name="c1_hyper")
        c1_ = yield tfd.Sample(tfd.Normal(c1_hyper, .01), (num_channel), name="c1_")

        c1 = c1_ - c3 - c2

        # evaluate the predictions
        prediction = parametric_fn_pcs(
            c1=c1[index[:, 0], index[:, 1]],
            c2=c2[index[:, 0], index[:, 1]],
            c3=c3[index[:, 0], index[:, 1]],
            pcs=pcs,
            positions=positions,
        )

        # Likelihood
        sigma = yield tfd.Sample(tfd.Gamma(1, 1), len(index), name="sigma")
        yield tfd.Normal(prediction, sigma, name="obs")

    return current_event