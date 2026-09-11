"""Capture of the UNet cross-attention weights.

keras_cv computes the attention weights as a local variable inside
CrossAttention.call (see diffusion_model.py), so there's nothing
to hook onto. This module temporarily replaces the class method
with a copy that keeps the weights before returning.

The replicated arithmetic is identical to the original, same operation,
same order, so generation is unaffected. Only cross-attention is recorder:
the self-attention block is called with context=None, which is what tells
them apart.

Note that the capture only works in eager mode. Inside tf.function  (as used
by predict_on_batch) the recorded tensors would be symbolic placeholders
from the tracing pass, not actual values.
"""

import math
import numpy as np
import keras_cv.models.stable_diffusion.diffusion_model as kcv
import tensorflow as tf
from tensorflow import keras

class AttentionRecorder:
    """Context manager collecting cross-attention weights of every forward pass.

    Attributes:
        weights: one entry per cross-attention call, in call order, each of
                 shape (batch, num_heads, h*w, context_length).
                 With the UNet of Stable Diffusion that is 16 entries per forward pass.
    Args:
        reduce: optional callable applied to each weight tensor before storing.
                Raw weights are large (~1.3M floats per denoising steps), so any real
                run should aggregate here rather than keep everything.
    """
    def __init__(self, reduce=None):
        self.weights = []
        self.reduce = reduce
        self._original = None

    def __enter__(self):
        recorder = self
        self._original = kcv.CrossAttention.call

        def call(layer, inputs):
            tensor, context = inputs
            if context is None:
                # self-attention untouched
                return recorder._original(layer, inputs)

            q = layer.to_q(tensor)
            k = layer.to_k(context)
            v = layer.to_v(context)

            q = tf.reshape(q, (-1, tensor.shape[1], layer.num_heads, layer.head_size))
            k = tf.reshape(k, (-1, context.shape[1], layer.num_heads, layer.head_size))
            v = tf.reshape(v, (-1, context.shape[1], layer.num_heads, layer.head_size))

            q = tf.transpose(q, (0,2,1,3))
            k = tf.transpose(k, (0,2,3,1))
            v = tf.transpose(v, (0,2,1,3))

            score = kcv.td_dot(q, k) * layer.scale
            weights = keras.activations.softmax(score)

            recorder.weights.append(
                weights if recorder.reduce is None else recorder.reduce(weights)
            )

            attn = kcv.td_dot(weights, v)
            attn = tf.transpose(attn, (0,2,1,3))

            out = tf.reshape(attn, (-1, tensor.shape[1], layer.num_heads * layer.head_size))
            return layer.out_proj(out)

        kcv.CrossAttention.call = call

        return self

    def __exit__(self, *exception):
        kcv.CrossAttention.call = self._original
        return False

def generate(model, context, batch_size=1, num_steps=50, ugs=7.5, seed=None,
             recorder=None, record_steps=None, condition_steps=None):
    """Eager re-implementation of StableDiffusionBase.generate_image.

    Same arithmetic as the original, with one deliberate difference: the UNet is
    called directly rather than through predict_on_batch. predict_on_batch wraps
    the model in a tf.function, and inside a traced graph the captured attention
    tensors would be symbolic placeholders instead of values. The cost is a
    rounding discrepancy of about 1 level out of 255 on the final image.

    Args:
        recorder: optional AttentionRecorder. Its buffer is cleared before each
            of the two UNet calls, so conditional and unconditional weights stay
            separate by construction rather than by relying on call order. They
            must not be mixed: their difference is what guidance amplifies.
        record_steps: which denoising steps to keep attention for, by position
            in execution order (0 is the noisiest). None keeps all. The buffer
            is cleared before every call regardless, so skipped steps cost only
            transient memory — which matters, since a batch of 8 over 20 steps
            would otherwise hold close to 2 GB.
        condition_steps: positions (in execution order) where the class context is used;
            elsewhere the unconditional prediction is taken as-is.
            None conditions every step. Lets one ask when conditioning matters causally,
            as opposed to where the instantaneous signal is largest.

    Returns:
        (images, steps, trace):
            images: uint8, (batch, height, width, 3).
            steps: one dict per kept step with its attention weights; empty
                without a recorder.
            trace: one dict per step with the magnitude and spatial layout of
                the class signal. Always recorded — it costs two norms and
                needs no recorder — since eps(c) - eps(0) is the only channel
                through which the class enters generation.
    """
    context = model._expand_tensor(context, batch_size)
    unconditional_context = tf.repeat(
        model._get_unconditional_context(), batch_size, axis=0)
    latent = model._get_initial_diffusion_noise(batch_size, seed)

    timesteps = tf.range(1, 1000, 1000 // num_steps)
    alphas, alphas_prev = model._get_initial_alphas(timesteps)

    steps, trace = [], []

    for position, (index, timestep) in enumerate(list(enumerate(timesteps))[::-1]):
        keep = recorder is not None and (record_steps is None
                                         or position in record_steps)
        latent_prev = latent
        t_emb = model._get_timestep_embedding(timestep, batch_size)

        if recorder is not None:
            recorder.weights.clear()
        unconditional_latent = model.diffusion_model(
            [latent, t_emb, unconditional_context])
        unconditional_attention = list(recorder.weights) if keep else []

        if recorder is not None:
            recorder.weights.clear()
        latent = model.diffusion_model([latent, t_emb, context])

        if keep:
            steps.append({"timestep": int(timestep), "index": int(index),
                          "conditional": list(recorder.weights),
                          "unconditional": unconditional_attention})

        # latent currently holds eps(c) and unconditional_latent eps(0), so this
        # is the class signal itself. It must be read here, before the guidance
        # step overwrites latent.
        difference = (latent - unconditional_latent).numpy()
        trace.append({
            "timestep": int(timestep),
            "index": int(index),
            "guidance_norm": np.linalg.norm(
                difference.reshape(batch_size, -1), axis=1),
            # ||eps(0)|| normalises the above: the scale of the noise prediction
            # changes along the denoising, so absolute norms are not comparable
            # between steps.
            "eps_norm": np.linalg.norm(
                unconditional_latent.numpy().reshape(batch_size, -1), axis=1),
            # Where in the latent the class changes the prediction: (batch, h, w).
            "guidance_map": np.linalg.norm(difference, axis=-1),
        })

        if condition_steps is None or position in condition_steps:
            latent = unconditional_latent + ugs * (latent - unconditional_latent)
        else:
            latent = unconditional_latent

        a_t, a_prev = alphas[index], alphas_prev[index]
        pred_x0 = (latent_prev - math.sqrt(1 - a_t) * latent) / math.sqrt(a_t)
        latent = latent * math.sqrt(1.0 - a_prev) + math.sqrt(a_prev) * pred_x0

    decoded = model.decoder(latent)
    decoded = ((decoded + 1) / 2) * 255

    return np.clip(decoded, 0, 255).astype("uint8"), steps, trace

def generate_in_chunks(model, context, chunk, seed, **kwargs):
    """Generate in blocks, to keep the batch within GPU memory.

    The seed is offset by the block index: reusing one seed would make every
    block start from the same noise and produce the same images. The offsets are
    deterministic, so all configurations still start from identical noise — which
    is what makes comparing them meaningful.
    """
    images, timesteps = [], None
    for index, start in enumerate(range(0, len(context), chunk)):
        stop = min(start + chunk, len(context))
        batch, _, trace = generate(
            model, context[start:stop], batch_size=stop - start,
            seed=seed + index, **kwargs)
        images.append(batch)
        timesteps = [t["timestep"] for t in trace]
    return np.concatenate(images), timesteps
