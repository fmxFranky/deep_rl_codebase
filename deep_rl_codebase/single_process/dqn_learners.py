import abc
from typing import Dict, List

import reverb
import sonnet as snt
import tensorflow as tf
import trfl
from acme.adders import reverb as adders
from acme.tf import losses

slice_with_actions = trfl.dist_value_ops._slice_with_actions


class RainbowLearner(abc.ABC):
  def __init__(
      self,
      network: snt.Module,
      target_network: snt.Module,
      double_dqn: int,
      discount: float,
      importance_sampling_exponent: float,
      learning_rate: float,
      adam_epsilon: float,
      target_update_period: int,
      dataset: tf.data.Dataset,
      replay_client: reverb.TFClient = None,
      **kwargs
  ):
    # Internalise agent components (replay buffer, networks, optimizer).
    self._iterator = iter(dataset)
    self._network = network
    self._target_network = target_network
    # TODO add lr_scheduler
    self._optimizer = snt.optimizers.Adam(learning_rate, epsilon=adam_epsilon)
    self._replay_client = replay_client
    self._double_dqn = double_dqn
    self._discount = discount
    self._target_update_period = target_update_period
    self._importance_sampling_exponent = importance_sampling_exponent

    # Learner state.
    self._variables: List[List[tf.Tensor]] = [network.trainable_variables]
    self._num_steps = tf.Variable(0, dtype=tf.int32)

  @abc.abstractmethod
  def _step(self) -> Dict[str, tf.Tensor]:
    raise NotImplementedError

  def step(self):
    return self._step()

  @property
  def state(self):
    """Returns the stateful parts of the learner for checkpointing."""
    return {
      'network': self._network,
      'target_network': self._target_network,
      'optimizer': self._optimizer,
      'num_steps': self._num_steps
    }


class DQNLearner(RainbowLearner):
  def __init__(
      self,
      network: snt.Module,
      target_network: snt.Module,
      kappa: float,
      **kwargs
  ):
    self._kappa = kappa
    super().__init__(
      network=network,
      target_network=target_network,
      **kwargs
    )

  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    # Pull out the data needed for updates/priorities.
    inputs = next(self._iterator)
    o_tm1, a_tm1, r_t, d_t, o_t = inputs.data
    keys, probs = inputs.info[:2]

    with tf.GradientTape() as tape:
      # Evaluate our networks.
      q_tm1 = self._network(o_tm1)
      q_t_value = self._target_network(o_t)
      q_t_selector = self._network(o_t)

      # The rewards and discounts have to have the same type as network values.
      r_t = tf.cast(r_t, q_tm1.dtype)
      r_t = tf.clip_by_value(r_t, -1., 1.)
      d_t = tf.cast(d_t, q_tm1.dtype) * tf.cast(self._discount, q_tm1.dtype)

      # Compute the loss.
      _, extra = trfl.double_qlearning(q_tm1, a_tm1, r_t, d_t, q_t_value,
                                       q_t_selector)
      loss = losses.huber(extra.td_error, self._kappa)

      # Get the importance weights.
      importance_weights = 1. / probs  # [B]
      importance_weights **= self._importance_sampling_exponent
      importance_weights /= tf.reduce_max(importance_weights)

      # Reweight.
      loss *= tf.cast(importance_weights, loss.dtype)  # [B]
      loss = tf.reduce_mean(loss, axis=[0])  # []

    # Do a step of SGD.
    gradients = tape.gradient(loss, self._network.trainable_variables)
    self._optimizer.apply(
      gradients, self._network.trainable_variables)

    # Update the priorities in the replay buffer.
    if self._replay_client:
      priorities = tf.cast(tf.abs(extra.td_error), tf.float64)
      self._replay_client.update_priorities(
        table=adders.DEFAULT_PRIORITY_TABLE, keys=keys, priorities=priorities)

    # Periodically update the target network.
    if tf.math.mod(self._num_steps, self._target_update_period) == 0:
      for src, dest in zip(self._network.variables,
                           self._target_network.variables):
        dest.assign(src)
    self._num_steps.assign_add(1)

    # Report loss & statistics for logging.
    fetches = {
      'loss': loss,
    }

    return fetches


class C51Learner(RainbowLearner):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    # Pull out the data needed for updates/priorities.
    inputs = next(self._iterator)

    o_tm1, a_tm1, r_t, d_t, o_t = inputs.data
    keys, probs = inputs.info[:2]

    with tf.GradientTape() as tape:
      # Evaluate our networks.
      q_tm1 = self._network(o_tm1)
      q_t = self._target_network(o_t)
      if self._double_dqn:
        q_t_selector = self._network(o_t).mean()
      else:
        q_t_selector = self._target_network(o_t).mean()

      # The rewards and discounts have to have the same type as network values.
      r_t = tf.clip_by_value(tf.cast(r_t, q_tm1.logits.dtype), -1., 1.)
      d_t = tf.cast(d_t, q_tm1.logits.dtype) * tf.cast(self._discount,
                                                       q_tm1.logits.dtype)

      # Compute the loss.
      loss, _ = trfl.categorical_dist_double_qlearning(
        q_tm1.values, q_tm1.logits, a_tm1, r_t, d_t,
        q_t.values, q_t.logits, q_t_selector
      )
      td_errors = tf.identity(loss)

      # Get the importance weights.
      importance_weights = 1. / probs  # [B]
      importance_weights **= self._importance_sampling_exponent
      importance_weights /= tf.reduce_max(importance_weights)

      # Reweight.
      loss *= tf.cast(importance_weights, loss.dtype)  # [B]
      loss = tf.reduce_mean(loss, axis=[0])  # []

    # Do a step of SGD.
    gradients = tape.gradient(loss, self._network.trainable_variables)
    self._optimizer.apply(
      gradients, self._network.trainable_variables)

    # Update the priorities in the replay buffer.
    if self._replay_client:
      priorities = tf.cast(tf.sqrt(td_errors + 1e-10), tf.float64)
      self._replay_client.update_priorities(
        table=adders.DEFAULT_PRIORITY_TABLE, keys=keys, priorities=priorities)

    # Periodically update the target network.
    if tf.math.mod(self._num_steps, self._target_update_period) == 0:
      trfl.update_target_variables(
        target_variables=self._target_network.variables,
        source_variables=self._network.variables)
    self._num_steps.assign_add(1)

    # Report loss & statistics for logging.
    fetches = {
      'loss': loss,
    }

    return fetches


class QRDQNLearner(RainbowLearner):
  """QR-DQN learner."""

  def __init__(
      self,
      network: snt.Module,
      target_network: snt.Module,
      num_quantiles: int,
      kappa: float,
      **kwargs
  ):
    self._kappa = kappa
    self._num_quantiles = num_quantiles
    super().__init__(
      network=network,
      target_network=target_network,
      **kwargs
    )

  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    """Do a step of SGD and update the priorities."""

    # Pull out the data needed for updates/priorities.
    inputs = next(self._iterator)

    o_tm1, a_tm1, r_t, d_t, o_t = inputs.data
    keys, probs = inputs.info[:2]

    with tf.GradientTape() as tape:
      # Evaluate our networks.
      # `qv` means quantile values
      # Shape: [B, A, num_quantiles]
      qv_tm1 = self._network(o_tm1)
      qv_t = self._target_network(o_t)  # [B, A, num_quantiles]
      q_t_selector = tf.reduce_mean(
        self._network(o_t) if self._double_dqn else self._target_network(o_t),
        axis=-1)

      # Shape: [B]
      a_t = tf.argmax(q_t_selector, axis=1, output_type=tf.int32)

      # The rewards and discounts have to have the same type as network values.
      r_t = tf.clip_by_value(tf.cast(r_t, qv_tm1.dtype), -1., 1.)
      d_t = tf.cast(d_t, qv_tm1.dtype) * tf.cast(self._discount, qv_tm1.dtype)

      # Calculate actions value distribution
      qv_a_tm1 = slice_with_actions(qv_tm1, a_tm1)  # [B, num_quantiles]

      # Scale and shift time-t distribution atoms by discount and reward.
      qv_a_t = slice_with_actions(qv_t, a_t)  # [B, num_quantiles]
      # Shape: [B, num_quantiles]
      target = tf.stop_gradient(r_t[:, None] + d_t[:, None] * qv_a_t)

      # Calculate quantile huber loss
      # Shape: [B, num_quantiles, num_quantiles]
      bellman_errors = target[:, None, :] - qv_a_tm1[:, :, None]
      tau = tf.linspace(0.0, 1.0, self._num_quantiles + 1)
      tau_hat = ((tau[1:] + tau[:-1]) / 2.0)[None, :, None]

      # Compute the loss.
      # Shape: [B, num_quantiles, num_quantiles]
      # huber_loss_case_one = tf.cast(tf.abs(bellman_errors) <= self._kappa,
      #                               tf.float32) * 0.5 * bellman_errors ** 2
      # huber_loss_case_two = tf.cast(tf.abs(bellman_errors) > self._kappa,
      #                               tf.float32) * self._kappa * (
      #                           tf.abs(bellman_errors) - 0.5 * self._kappa)
      # huber_loss = huber_loss_case_one + huber_loss_case_two
      huber_loss = losses.huber(bellman_errors, self._kappa)

      # Shape: [B, num_quantiles, num_quantiles]
      quantile_huber_loss = (tf.abs(tau_hat - tf.stop_gradient(
        tf.cast(bellman_errors < 0, tf.float32))) * huber_loss) / self._kappa

      # Shape: [B]
      loss = tf.reduce_mean(tf.reduce_sum(quantile_huber_loss, axis=2), axis=1)
      td_errors = tf.identity(loss)

      # Get the importance weights.
      importance_weights = 1. / probs  # [B]
      importance_weights **= self._importance_sampling_exponent
      importance_weights /= tf.reduce_max(importance_weights)

      # Reweight.
      loss *= tf.cast(importance_weights, loss.dtype)  # [B]
      loss = tf.reduce_mean(loss, axis=[0])  # []

    # Do a step of SGD.
    gradients = tape.gradient(loss, self._network.trainable_variables)
    self._optimizer.apply(
      gradients, self._network.trainable_variables)

    # Update the priorities in the replay buffer.
    if self._replay_client:
      priorities = tf.cast(tf.sqrt(td_errors + 1e-10), tf.float64)
      self._replay_client.update_priorities(
        table=adders.DEFAULT_PRIORITY_TABLE, keys=keys, priorities=priorities)

    # Periodically update the target network.
    if tf.math.mod(self._num_steps, self._target_update_period) == 0:
      trfl.update_target_variables(
        target_variables=self._target_network.variables,
        source_variables=self._network.variables)
    self._num_steps.assign_add(1)

    # Report loss & statistics for logging.
    fetches = {
      'loss': loss,
    }

    return fetches


class IQNLearner(RainbowLearner):
  """IQN learner."""

  def __init__(
      self,
      network: snt.Module,
      target_network: snt.Module,
      kappa: float,
      num_tau_samples: int,
      num_tau_prime_samples: int,
      num_quantile_samples: int,
      **kwargs
  ):
    self._kappa = kappa
    self._num_tau_samples = num_tau_samples
    self._num_tau_prime_samples = num_tau_prime_samples
    self._num_quantile_samples = num_quantile_samples
    super().__init__(
      network=network,
      target_network=target_network,
      **kwargs
    )

  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    """Do a step of SGD and update the priorities."""

    # Pull out the data needed for updates/priorities.
    inputs = next(self._iterator)

    o_tm1, a_tm1, r_t, d_t, o_t = inputs.data
    keys, probs = inputs.info[:2]

    with tf.GradientTape() as tape:
      # Evaluate our networks.
      # `qv` means quantile values, `tau` is same as quantile
      # Shape: [B, num_quantiles]
      tau_tm1 = tf.random.uniform(
        shape=[tf.shape(o_tm1)[0], self._num_tau_samples],
        minval=0., maxval=1., dtype=tf.float32)
      tau_t = tf.random.uniform(
        shape=[tf.shape(o_t)[0], self._num_tau_prime_samples],
        minval=0., maxval=1., dtype=tf.float32)
      tau_for_selector = tf.random.uniform(
        shape=[tf.shape(o_t)[0], self._num_quantile_samples],
        minval=0., maxval=1., dtype=tf.float32)
      # Shape: [B, A, num_quantiles]
      qv_tm1 = self._network(o_tm1, tau_tm1)
      qv_t = self._target_network(o_t, tau_t)
      if self._double_dqn:
        q_t_selector = tf.reduce_mean(
          self._network(o_t, tau_for_selector), axis=-1)
      else:
        q_t_selector = tf.reduce_mean(
          self._target_network(o_t, tau_for_selector), axis=-1)

      # Shape: [B]
      a_t = tf.argmax(q_t_selector, axis=1, output_type=tf.int32)

      # The rewards and discounts have to have the same type as network values.
      r_t = tf.clip_by_value(tf.cast(r_t, qv_tm1.dtype), -1., 1.)
      d_t = tf.cast(d_t, qv_tm1.dtype) * tf.cast(self._discount, qv_tm1.dtype)

      # Calculate actions value distribution
      qv_a_tm1 = slice_with_actions(qv_tm1, a_tm1)  # [B, num_quantiles]

      # Scale and shift time-t distribution atoms by discount and reward.
      qv_a_t = slice_with_actions(qv_t, a_t)  # [B, num_quantiles]
      # Shape: [B, num_quantiles]
      target = tf.stop_gradient(r_t[:, None] + d_t[:, None] * qv_a_t)

      # Calculate quantile huber loss
      # Shape: [B, num_quantiles, num_quantiles]
      bellman_errors = target[:, None, :] - qv_a_tm1[:, :, None]

      # Compute the loss.
      # Shape: [B, num_quantiles, num_quantiles]
      huber_loss = losses.huber(bellman_errors, self._kappa)

      # Shape: [B, num_quantiles, num_quantiles]
      # Unlike QR-DQN, we directly use taus outputted by network as \hat{\tau}
      quantile_huber_loss = (tf.abs(tau_tm1[..., None] - tf.stop_gradient(
        tf.cast(bellman_errors < 0, tf.float32))) * huber_loss) / self._kappa

      # Shape: [B]
      loss = tf.reduce_mean(tf.reduce_sum(quantile_huber_loss, axis=2), axis=1)
      td_errors = tf.identity(loss)

      # Get the importance weights.
      importance_weights = 1. / probs  # [B]
      importance_weights **= self._importance_sampling_exponent
      importance_weights /= tf.reduce_max(importance_weights)

      # Reweight.
      loss *= tf.cast(importance_weights, loss.dtype)  # [B]
      loss = tf.reduce_mean(loss, axis=[0])  # []

    # Do a step of SGD.
    gradients = tape.gradient(loss, self._network.trainable_variables)
    self._optimizer.apply(
      gradients, self._network.trainable_variables)

    # Update the priorities in the replay buffer.
    if self._replay_client:
      priorities = tf.cast(tf.sqrt(td_errors + 1e-10), tf.float64)
      self._replay_client.update_priorities(
        table=adders.DEFAULT_PRIORITY_TABLE, keys=keys, priorities=priorities)

    # Periodically update the target network.
    if tf.math.mod(self._num_steps, self._target_update_period) == 0:
      trfl.update_target_variables(
        target_variables=self._target_network.variables,
        source_variables=self._network.variables)
    self._num_steps.assign_add(1)

    # Report loss & statistics for logging.
    fetches = {
      'loss': loss,
    }

    return fetches
