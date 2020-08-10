import functools
import os
from typing import Any, Sequence

import acme
import dm_env
import gym
import numpy as np
import sonnet as snt
import tensorflow as tf
import wandb
from absl import app
from absl import flags
from acme import wrappers
from acme.tf import utils as tf2_utils
from acme.tf.networks import distributions as ad
from coolname import generate_slug

from deep_rl_codebase.single_process.dqn_learners import C51Learner
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

# network's config
flags.DEFINE_integer('num_atoms', 51, 'discretised size of value distribution')
flags.DEFINE_float('vmin', -10., 'minimum of value distribution support')
flags.DEFINE_float('vmax', 10., 'maximum of value distribution support')
flags.DEFINE_multi_integer('hidden_sizes', [512], 'width of each hidden layer.')

# learner's config
flags.DEFINE_integer('batch_size', 32, 'batch size for updates')
flags.DEFINE_float('learning_rate', 0.0000625,
                   'learning rate for the q-network update')

# logger and saver's config
flags.DEFINE_string('logdir', './experiments',
                    'directory for the logging messages and models')
flags.DEFINE_boolean('checkpoint', True,
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


class CategoricalAtariNetwork(snt.Module):
  """A Categorical Q-network."""

  def __init__(
      self,
      num_actions: int,
      num_atoms: int,
      vmin: float,
      vmax: float,
      hidden_sizes: Sequence[int],
  ):
    super().__init__(name='categorical_network')

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
    self._categorical_value_head = snt.Sequential([
      snt.nets.MLP(
        output_sizes=[*hidden_sizes, num_actions * num_atoms],
        w_init=self._w_init),
      snt.Reshape([num_actions, num_atoms]),
    ])
    self._network = snt.Sequential([
      self._torso,
      self._categorical_value_head
    ])

    self._values = tf.linspace(vmin, vmax, num_atoms)

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    # We do not calculate probabilities here
    logits = self._network(inputs)  # [B, A, num_atoms]
    values = tf.cast(self._values, logits.dtype)  # [num_atoms]

    return ad.DiscreteValuedDistribution(values=values, logits=logits)


def categorical_policy_wrapper(network: snt.Module) -> snt.Sequential:
  return snt.Sequential([
    network,
    lambda dist: tf.argmax(dist.mean(), axis=1, output_type=tf.int32)
  ])


def main(_):
  # Initialize wandb and runx.
  # We use runx to record metrics for each iteration and use wandb to record
  # all the metrics during the training progress online
  # Note that use wandb will reduce the runner's efficiency
  experiment_name = f'rainbow-c51-{FLAGS.game}-{generate_slug(2)}'
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
  network = CategoricalAtariNetwork(
    num_actions=environment_spec.actions.num_values,
    num_atoms=FLAGS.num_atoms, vmin=FLAGS.vmin, vmax=FLAGS.vmax,
    hidden_sizes=FLAGS.hidden_sizes)
  tf2_utils.create_variables(network, [environment_spec.observations])

  runner = RainbowRunner(
    experiment_name=experiment_name,
    logdir=FLAGS.logdir,
    environments=environments,
    network=network,
    policy_wrapper=categorical_policy_wrapper,
    learner_class=C51Learner,
    batch_size=FLAGS.batch_size,
    learning_rate=FLAGS.learning_rate,
    checkpoint=FLAGS.checkpoint,
    tensorboard=FLAGS.tensorboard,
    use_wandb=FLAGS.use_wandb,
  )

  runner.run_experiment(FLAGS.num_iterations)


if __name__ == '__main__':
  app.run(main)
