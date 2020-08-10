from concurrent import futures
from typing import Callable, List, Optional

import acme
import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
from acme import types
from acme.adders import reverb as reverb_adders
from acme.tf import utils as tf2_utils


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
  """Returns the current epsilon for the agent's epsilon-greedy policy.

  This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
  al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.

  Args:
    decay_period: int, the period over which epsilon is decayed.
    step: int, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before epsilon is decayed.
    epsilon: float, the final value to which to decay the epsilon parameter.

  Returns:
    A float, the current epsilon value computed according to the schedule.
  """
  steps_left = decay_period + warmup_steps - step
  bonus = (1.0 - epsilon) * steps_left / decay_period
  bonus = np.clip(bonus, 0., 1. - epsilon)
  return epsilon + bonus


def identity_epsilon(unused_decay_period, unused_step, unused_warmup_steps,
                     epsilon):
  return epsilon


class EpsilonGreedyActor:
  """A feed-forward actor.

  An actor based on a feed-forward policy which takes non-batched observations
  and outputs non-batched actions. It also allows adding experiences to replay
  and updating the weights from the policy on the learner.
  """

  def __init__(
      self,
      network: snt.Module,
      policy_wrapper: Callable[[snt.Module], snt.Sequential],
      environments: List[dm_env.Environment],
      adders: List[reverb_adders.ReverbAdder] = None,
      epsilon_fn: Callable[
        [int, int, int, float], float] = linearly_decaying_epsilon,
      epsilon_train: float = 0.01,
      epsilon_eval: float = 0.001,
      epsilon_decay_period: int = 250000,
      warmup_steps: int = 1000,
      eval_mode: bool = False,
      **kwargs
  ):
    """Initializes the actor.

    Args:
      network: the network to inference.
      policy_wrapper: the wrapper for the network to generate the policy_network
      environments: environments
      adders: the adder object to which allows to add experiences to a
        dataset/replay buffer.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      warmup_steps: int, the number of steps taken before epsilon is decayed.
      eval_mode: bool, True for evaluation and False for training.
    """

    # Store these for later use.
    self._policy_network = tf.function(policy_wrapper(network))
    self._environments = environments
    self._adders = adders

    self._epsilon_fn = epsilon_fn
    self._epsilon_train = epsilon_train
    self._epsilon_eval = epsilon_eval
    self._epsilon_decay_period = epsilon_decay_period
    self._warmup_steps = warmup_steps
    self._eval_mode = eval_mode
    self._training_steps = 0

    environment_spec = acme.make_environment_spec(self._environments[0])
    self._num_actions = environment_spec.actions.num_values
    self._num_envs = len(self._environments)
    self._pool = futures.ThreadPoolExecutor(max_workers=self._num_envs)
    self._episode_return_buffer = []
    self._episode_length_buffer = []
    self._step_counters = None
    self._reward_accumulators = None
    self._num_episodes = 0
    self._feedforward_steps = 0
    self._eval_mode = False

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
      return tf2_utils.to_numpy(self._policy_network(batched_obs))

  def reset_all_environments(self):
    # Reset environments, which makes the first observations.
    reset_tasks = []
    for environment in self._environments:
      reset_tasks.append(self._pool.submit(environment.reset))
    futures.wait(reset_tasks, return_when=futures.ALL_COMPLETED)
    timesteps: List[dm_env.TimeStep] = [task.result() for task in reset_tasks]
    if not self._eval_mode:
      # Add first observations to the replay server.
      add_first_tasks = []
      for timestep, adder in zip(timesteps, self._adders):
        adder.reset()
        add_first_tasks.append(self._pool.submit(adder.add_first, timestep))
      futures.wait(add_first_tasks, return_when=futures.ALL_COMPLETED)
    # Clear metric queue
    self._episode_return_buffer.clear()
    self._episode_length_buffer.clear()
    self._step_counters = [1 for _ in range(self._num_envs)]
    self._reward_accumulators = [0.0 for _ in range(self._num_envs)]
    self._num_episodes = 0

    return timesteps

  def run_environment_step(self, actions):
    # Run one step
    step_tasks = []
    for environment, action in zip(self._environments, actions):
      step_tasks.append(self._pool.submit(environment.step, action))
    futures.wait(step_tasks, return_when=futures.ALL_COMPLETED)
    timesteps: List[dm_env.TimeStep] = [task.result() for task in step_tasks]
    # TODO return new terminated episodes' info
    for i, timestep in enumerate(timesteps):
      if timestep.first():
        self._step_counters[i] = 1
        self._reward_accumulators[i] = 0.0
      else:
        self._step_counters[i] += 1
        self._reward_accumulators[i] += timestep.reward
        if timestep.last():
          self._num_episodes += 1
          self._episode_length_buffer.append(self._step_counters[i])
          self._episode_return_buffer.append(self._reward_accumulators[i])
    return timesteps

  def observe(
      self,
      timesteps: List[dm_env.TimeStep],
      actions: Optional[types.NestedArray] = None,
  ):
    if self._adders and not self._eval_mode:
      # Add observations
      add_tasks = []
      for action, timestep, adder in zip(actions, timesteps, self._adders):
        if timestep.first():
          add_tasks.append(self._pool.submit(adder.add_first, timestep))
        else:
          add_tasks.append(self._pool.submit(adder.add, action, timestep))
      futures.wait(add_tasks, return_when=futures.ALL_COMPLETED)

  @property
  def feedforward_steps(self):
    return self._feedforward_steps

  @property
  def num_timesteps(self):
    return self._feedforward_steps * self._num_envs

  @property
  def num_episodes(self):
    return self._num_episodes

  def latest_episodes_average_return(self, num_episodes: Optional[int] = None):
    if num_episodes:
      n = min(len(self._episode_return_buffer), num_episodes)
      return sum(self._episode_return_buffer[-1:-n:-1]) / max(n, 1)
    n = max(len(self._episode_return_buffer), 1)
    return sum(self._episode_return_buffer) / n

  def latest_episodes_average_length(self, num_episodes: Optional[int] = None):
    if num_episodes:
      n = min(len(self._episode_length_buffer), num_episodes)
      return sum(self._episode_length_buffer[-1:-n:-1]) / max(n, 1)
    n = max(len(self._episode_length_buffer), 1)
    return sum(self._episode_length_buffer) / n

  def train(self):
    self._eval_mode = False

  def eval(self):
    self._eval_mode = True
