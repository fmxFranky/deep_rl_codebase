import functools
import math
import os
from typing import Any, List, Sequence

import acme
import dm_env
import gym
import numpy as np
import sonnet as snt
import tensorflow as tf
import wandb
from absl import app
from absl import flags
from acme import types
from acme import wrappers
from acme.tf import utils as tf2_utils
from coolname import generate_slug

import deep_rl_codebase.single_process.epsilon_greedy_actor as actor
from deep_rl_codebase.single_process.dqn_learners import IQNLearner
from deep_rl_codebase.single_process.runner import RainbowRunner

# environment's config
flags.DEFINE_string('game', 'breakout',
                    'Which Atari game to play.')
flags.DEFINE_integer('num_envs', 1, 'the number of environments')

# experiment's config
flags.DEFINE_string('experiment_name', None, 'the name for this experiment')
flags.DEFINE_integer('num_iterations', 200, 'the iteration number threshold.')
flags.DEFINE_integer('seed', 666, 'the seed for tf2, numpy and gym')
flags.DEFINE_integer('gpu_id', 0, 'set gpu device for learner')

# hyperparameters for implicit quantile regression and the network
flags.DEFINE_integer('num_tau_samples', 64,
                     'the number of online quantile samples for loss '
                     'estimation.')
flags.DEFINE_integer('num_tau_prime_samples', 64,
                     'the number of target quantile samples for loss '
                     'estimation.')
flags.DEFINE_integer('num_quantile_samples', 32,
                     'the number of quantile samples for computing Q-values.')
flags.DEFINE_integer('quantile_embedding_dim', 32,
                     'embedding dimension for the quantile input.')
flags.DEFINE_float('kappa', 1.0, 'Huber loss cutoff.')
flags.DEFINE_multi_integer('hidden_sizes', [512], 'width of each hidden layer.')

# learner's config
flags.DEFINE_integer('batch_size', 32, 'batch size for updates')
flags.DEFINE_float('learning_rate', 0.00005,
                   'learning rate for the q-network update')

# logger and saver's config
flags.DEFINE_string('logdir', './experiments',
                    'directory for the logging messages and models')
flags.DEFINE_boolean('checkpoint', False,
                     'boolean indicating whether to checkpoint the learner')
flags.DEFINE_boolean('tensorboard', True, 'whether use tensorboard')
flags.DEFINE_boolean('use_wandb', False, 'whether use wandb')
flags.DEFINE_string('wandb_login_token', None, 'your wandb token')

FLAGS = flags.FLAGS


def make_atari_environment(
    game: str,
    full_action_space: bool = False,
    max_episode_len: int = 27000,
    sticky_actions: bool = False,
    zero_discount_on_life_loss: bool = True,
    seed: Any = None
) -> dm_env.Environment:
  name = game.title() + f"NoFrameskip-v{0 if sticky_actions else 4}"
  env = gym.make(name, full_action_space=full_action_space)
  env.seed(seed)

  return wrappers.wrap_all(env, [
    wrappers.GymAtariAdapter,
    functools.partial(
      wrappers.AtariWrapper,
      to_float=True,
      max_episode_len=max_episode_len,
      zero_discount_on_life_loss=zero_discount_on_life_loss,
    ),
    wrappers.SinglePrecisionWrapper,
  ])


class IQNAtariNetwork(snt.Module):
  """A Categorical Q-network."""

  def __init__(
      self,
      num_actions: int,
      quantile_embedding_dim: int,
      hidden_sizes: Sequence[int],
  ):
    super().__init__(name='implicit_quantile_network')

    self._quantile_embedding_dim = quantile_embedding_dim
    self._w_init = snt.initializers.VarianceScaling(
      scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform'
    )
    self._torso = snt.Sequential([
      snt.Conv2D(32, [8, 8], [4, 4], w_init=self._w_init),
      tf.nn.relu,
      snt.Conv2D(64, [4, 4], [2, 2], w_init=self._w_init),
      tf.nn.relu,
      snt.Conv2D(64, [3, 3], [1, 1], w_init=self._w_init),
      tf.nn.relu,
      snt.Flatten(),
    ])
    self._quantile_value_head = snt.Sequential([
      snt.nets.MLP([*hidden_sizes, num_actions], w_init=self._w_init),
      tf.keras.layers.Permute([2, 1])
    ])

  def __call__(self, inputs: tf.Tensor, quantiles: tf.Tensor) -> tf.Tensor:
    """Creates the output tensor/op given the state tensor as input.

        See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
        information on this. Note that tf.keras.Model implements `call` which is
        wrapped by `__call__` function by tf.keras.Model.

        Args:
          inputs: `tf.Tensor`, contains the agent's current state.
          quantiles: `tf.Tensor`, contains the quantiles
        Returns:
          quantile_values
        """
    state_embedding = self._torso(inputs)
    state_vector_length = tf.shape(state_embedding)[-1]

    # \phi_j(tau) = RELU(\Sigma_i(cos(\pi*i*\tau)*w_{ij} + b_j))
    pi = tf.constant(math.pi, dtype=tf.float32)
    idx_range = tf.range(1, self._quantile_embedding_dim + 1, dtype=tf.float32)
    # Shape: [B, num_quantiles, quantile_embedding_dim]
    cos_embedding = tf.cos(idx_range[None, None, :] * pi * quantiles[..., None])
    if not hasattr(self, '_quantile_fc'):
      self._quantile_fc = snt.Sequential([
        snt.Linear(state_vector_length, w_init=self._w_init),
        tf.nn.relu])
    # Z_{\tau}(x,a) \simeq f(\psi(x) @ \phi(\tau))
    # Shape: [B, num_quantiles, quantile_embedding_dim]
    quantile_embedding = self._quantile_fc(tf.cast(cos_embedding, tf.float32))
    hadamard_product = tf.multiply(
      x=state_embedding[:, None, :], y=quantile_embedding)
    quantile_values = self._quantile_value_head(hadamard_product)

    return quantile_values


def implicit_quantile_policy_wrapper(network: snt.Module) -> snt.Sequential:
  return snt.Sequential([
    network,
    lambda q: tf.argmax(tf.reduce_mean(q, -1), axis=1, output_type=tf.int32)
  ])


class IQNActor(actor.EpsilonGreedyActor):
  def __init__(self, num_quantile_samples: int, **kwargs):
    self._num_quantile_samples = num_quantile_samples
    super().__init__(**kwargs)

  def select_action(
      self, timesteps: List[dm_env.TimeStep]
  ) -> types.NestedArray:
    if not self._eval_mode:
      self._feedforward_steps += 1
    epsilon = self._epsilon_fn(
      self._epsilon_decay_period,
      self._feedforward_steps,
      self._warmup_steps,
      self._epsilon_train) if not self._eval_mode else self._epsilon_eval
    if np.random.uniform() <= epsilon:
      return np.random.randint(
        low=0, high=self._num_actions, size=[self._num_envs], dtype=np.int32)
    else:
      batched_obs = tf.concat(
        [tf2_utils.add_batch_dim(ts.observation) for ts in timesteps], axis=0)
      # Forward the policy network.
      quantiles = tf.random.uniform(
        shape=[tf.shape(batched_obs)[0], self._num_quantile_samples],
        minval=0., maxval=1., dtype=tf.float32)
      return tf2_utils.to_numpy(self._policy_network(batched_obs, quantiles))


def main(_):
  # Initialize wandb and runx.
  # We use runx to record metrics for each iteration and use wandb to record
  # all the metrics during the training progress online
  # Note that use wandb will reduce the runner's efficiency
  experiment_name = f'rainbow-iqn-{FLAGS.game}-{generate_slug(2)}'
  experiment_name = FLAGS.experiment_name or experiment_name
  FLAGS.set_default(name='experiment_name', value=experiment_name)

  if FLAGS.use_wandb:
    assert FLAGS.wandb_login_token is not None, "You must use the token to " \
                                                "login wandb website "
    wandb.login('never', FLAGS.wandb_login_token)
    wandb.init(
      dir=os.path.join(FLAGS.logdir, experiment_name),
      config=FLAGS.flag_values_dict(),
      project="atari_implements", name=experiment_name,
      id=experiment_name, save_code=True)

  # Set gpu devices and seeds for tf and np
  gpus = tf.config.list_physical_devices(device_type='GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)
  tf.config.set_visible_devices(devices=gpus[FLAGS.gpu_id], device_type='GPU')
  # tf.config.threading.set_inter_op_parallelism_threads(1)
  # tf.config.threading.set_intra_op_parallelism_threads(1)
  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  # Create the actor the learner
  # Create Atari environment
  environments = [make_atari_environment(
    game=FLAGS.game, seed=FLAGS.seed
  ) for _ in range(FLAGS.num_envs + 1)]
  environment_spec = acme.make_environment_spec(environments[0])

  # Create the network and its target network
  # Ensure that we create the variables before proceeding (maybe not needed).
  network = IQNAtariNetwork(
    num_actions=environment_spec.actions.num_values,
    quantile_embedding_dim=FLAGS.quantile_embedding_dim,
    hidden_sizes=FLAGS.hidden_sizes)
  tf2_utils.create_variables(
    network, [environment_spec.observations,
              acme.specs.Array(shape=(), dtype=np.float32)])

  runner = RainbowRunner(
    experiment_name=experiment_name,
    logdir=FLAGS.logdir,
    environments=environments,
    network=network,
    actor_class=IQNActor,
    policy_wrapper=implicit_quantile_policy_wrapper,
    learner_class=IQNLearner,
    batch_size=FLAGS.batch_size,
    learning_rate=FLAGS.learning_rate,
    kappa=FLAGS.kappa,
    num_tau_samples=FLAGS.num_tau_samples,
    num_tau_prime_samples=FLAGS.num_tau_prime_samples,
    num_quantile_samples=FLAGS.num_quantile_samples,
    checkpoint=FLAGS.checkpoint,
    tensorboard=FLAGS.tensorboard,
    use_wandb=FLAGS.use_wandb,
  )

  runner.run_experiment(FLAGS.num_iterations)


if __name__ == '__main__':
  app.run(main)
