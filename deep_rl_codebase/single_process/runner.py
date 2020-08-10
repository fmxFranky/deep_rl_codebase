import copy
import os
import time
from typing import Callable, List

import acme
import dm_env
import reverb
import sonnet as snt
import tensorflow as tf
from acme import datasets
from acme.adders import reverb as reverb_adders
from acme.tf import savers as tf2_savers
from acme.utils.loggers.terminal import TerminalLogger

from deep_rl_codebase.single_process import dqn_learners, \
  epsilon_greedy_actor as actor, utils


class RainbowRunner:
  def __init__(
      self,
      experiment_name: str,
      logdir: str,
      environments: List[dm_env.Environment],
      network: snt.Module,
      policy_wrapper: Callable[[snt.Module], snt.Sequential],
      # runner's config
      schedule: str = 'continuous_train_and_eval',
      training_steps_per_iteration: int = 250000,
      eval_unit: str = 'step',
      eval_steps_per_iteration: int = 125000,
      eval_episodes_per_iteration: int = 100,
      # actor's config
      epsilon_fn: Callable[
        [int, int, int, float], float] = actor.linearly_decaying_epsilon,
      epsilon_train: float = 0.01,
      epsilon_eval: float = 0.001,
      epsilon_decay_period: int = 250000,
      # replay buffer's config
      min_replay_size: int = 20000,
      max_replay_size: int = 1000000,
      replay_scheme: str = 'prioritized',
      replay_server_port: int = None,
      priority_exponent: float = 0.5,
      # rainbow learner's config
      double_dqn: bool = True,
      batch_size: int = 32,
      prefetch_size: int = -1,
      learning_rate: float = 0.00005,
      adam_epsilon=0.0003125,
      learner_update_period: int = 4,
      target_update_period: int = 2500,
      importance_sampling_exponent: float = 0.5,
      n_step: int = 3,
      discount: float = 0.99,
      # logger ans saver's config
      tensorboard: bool = True,
      use_wandb: bool = False,
      checkpoint: bool = True,
      # custom actor and learner's config
      actor_class=actor.EpsilonGreedyActor,
      learner_class=dqn_learners.DQNLearner,
      **kwargs
  ):
    self._eval_environment = environments[-1]
    environment_spec = acme.make_environment_spec(self._eval_environment)
    # Create the network and its target network
    # Ensure that we create the variables before proceeding (maybe not needed).
    target_network = copy.deepcopy(network)

    # Create a replay server to add data to. This uses no limiter behavior in
    # order to allow the program's interface to handle it.
    assert replay_scheme in ['uniform', 'prioritized']
    if replay_scheme == 'uniform':
      sampler = reverb.selectors.Uniform()
    else:
      sampler = reverb.selectors.Prioritized(priority_exponent)
    replay_table = reverb.Table(
      name=reverb_adders.DEFAULT_PRIORITY_TABLE,
      sampler=sampler,
      remover=reverb.selectors.Fifo(),
      max_size=max_replay_size,
      rate_limiter=reverb.rate_limiters.MinSize(1),
      signature=reverb_adders.NStepTransitionAdder.signature(environment_spec))
    replay_checkpointer = reverb.checkpointers.DefaultCheckpointer(
      path=os.path.join(logdir, experiment_name, 'checkpoints', 'reverb'))
    self._replay_server = reverb.Server(
      tables=[replay_table],
      port=replay_server_port,
      checkpointer=replay_checkpointer)

    # The adder is used to insert observations into replay.
    address = f'localhost:{self._replay_server.port}'
    adders = [reverb_adders.NStepTransitionAdder(
      client=reverb.Client(address),
      n_step=n_step,
      discount=discount) for _ in range(len(environments))]

    # The dataset provides an interface to sample from replay.
    replay_client = reverb.TFClient(address)
    dataset = datasets.make_reverb_dataset(
      server_address=address,
      batch_size=batch_size,
      prefetch_size=prefetch_size)

    # Create the actor which defines how we take actions.
    self._actor = actor_class(
      network=network,
      policy_wrapper=policy_wrapper,
      environments=environments[:-1],
      adders=adders,
      epsilon_fn=epsilon_fn,
      epsilon_train=epsilon_train,
      epsilon_eval=epsilon_eval,
      epsilon_decay_period=epsilon_decay_period,
      warmup_steps=max(batch_size, min_replay_size),
      **kwargs
    )

    # The learner updates the parameters (and initializes them).
    self._learner = learner_class(
      network=network,
      target_network=target_network,
      double_dqn=double_dqn,
      discount=discount,
      importance_sampling_exponent=importance_sampling_exponent,
      learning_rate=learning_rate,
      adam_epsilon=adam_epsilon,
      target_update_period=target_update_period,
      dataset=dataset,
      replay_client=replay_client if replay_scheme == 'prioritized' else None,
      **kwargs
    )

    # Create checkpointer if necessary
    self._checkpointer = tf2_savers.Checkpointer(
      directory=os.path.join(logdir, experiment_name),
      objects_to_save=self._learner.state,
      subdirectory='learner',
      time_delta_minutes=10.,
      max_to_keep=3,
      add_uid=False,
    ) if checkpoint else None

    # Record other variables for future reusing
    assert schedule in ['continuous_train_and_eval', 'continuous_train']
    self._schedule = schedule
    self._min_replay_size = min_replay_size
    self._learner_update_period = learner_update_period

    self._use_wandb = use_wandb

    self._training_steps_per_iteration = training_steps_per_iteration
    self._eval_unit = eval_unit
    self._eval_episodes_per_iteration = eval_episodes_per_iteration
    self._eval_steps_per_iteration = eval_steps_per_iteration

    self._summary_writer = tf.summary.create_file_writer(
      os.path.join(logdir, experiment_name)) if tensorboard else None
    self._train_logger = TerminalLogger('train', time_delta=60)
    self._flush_logger = TerminalLogger('train summary')
    self._eval_logger = TerminalLogger('eval summary')
    self._current_iteration = None
    self._start_time = time.time()

  def _run_train_phase(self):
    self._actor.train()
    time_stat = {
      "environment_step": utils.TimeStat(1000),
      "select_action": utils.TimeStat(1000),
      "store_transition": utils.TimeStat(1000),
      "update_network": utils.TimeStat(1000)}
    timesteps = self._actor.reset_all_environments()

    threshold = self._current_iteration * self._training_steps_per_iteration
    while self._actor.num_timesteps < threshold:
      with time_stat["select_action"]:
        actions = self._actor.select_action(timesteps)
      with time_stat["environment_step"]:
        timesteps = self._actor.run_environment_step(actions)
      with time_stat["store_transition"]:
        self._actor.observe(timesteps, actions)

      if self._actor.num_timesteps > self._min_replay_size and \
          self._actor.feedforward_steps % self._learner_update_period == 0:
        with time_stat["update_network"]:
          result = self._learner.step()
        loss = result["loss"].numpy()
        if self._summary_writer:
          with self._summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=self._actor.num_timesteps)
            tf.summary.scalar(
              name='latest_100_episodes_average_return',
              data=self._actor.latest_episodes_average_return(100),
              step=self._actor.num_timesteps)
            for name, val in time_stat.items():
              tf.summary.scalar(name, val.mean, step=self._actor.num_timesteps)

        self._train_logger.write({
          "iteration": self._current_iteration,
          "actor timesteps": self._actor.num_timesteps,
          "average return": self._actor.latest_episodes_average_return(),
          "loss": loss, "wall time": (time.time() - self._start_time) / 60.0})

    if self._summary_writer:
      with self._summary_writer.as_default():
        tf.summary.scalar(
          name=f'train_episode_average_return',
          data=self._actor.latest_episodes_average_return(),
          step=self._actor.num_timesteps)
        tf.summary.scalar(
          name=f'train_episode_average_length',
          data=self._actor.latest_episodes_average_length(),
          step=self._actor.num_timesteps)

    self._flush_logger.write({
      "iteration": self._current_iteration,
      "actor timesteps": self._actor.num_timesteps,
      "average return(100)": self._actor.latest_episodes_average_return(100),
      "average length(100)": self._actor.latest_episodes_average_length(100),
      "wall time(minute)": (time.time() - self._start_time) / 60.0})
    self._flush_logger.write({
      "environment step time(ms)": time_stat["environment_step"].mean * 10000,
      "select action time(ms)": time_stat["select_action"].mean * 1000,
      "store transition time(ms)": time_stat["store_transition"].mean * 1000,
      "update network time(ms)": time_stat["update_network"].mean * 1000,
    })

  def _run_eval_phase(self, eval_unit: str, threshold: int):
    """Use eval_environment to evaluate the policy"""

    def mean_fn(buffer):
      return sum(buffer) / max(1, len(buffer))

    self._actor.eval()
    episode_length_buffer = []
    episode_return_buffer = []

    while True:
      # Reset any counts and start the environment.
      episode_steps = 0
      episode_return = 0
      timestep = self._eval_environment.reset()
      while not timestep.last():
        action = self._actor.select_action([timestep])[0]
        timestep = self._eval_environment.step(action)
        # Book-keeping.
        episode_steps += 1
        episode_return += timestep.reward

      episode_length_buffer.append(episode_steps)
      episode_return_buffer.append(episode_return)

      bool1 = sum(episode_length_buffer) >= threshold and eval_unit == 'step'
      bool2 = len(episode_length_buffer) >= threshold and eval_unit == 'episode'
      if bool1 or bool2:
        break

    mean_episode_return = mean_fn(episode_return_buffer)
    mean_episode_length = mean_fn(episode_length_buffer)
    if self._summary_writer:
      with self._summary_writer.as_default():
        tf.summary.scalar(
          name=f'eval_episode_average_return',
          data=mean_episode_return,
          step=self._actor.num_timesteps)
        tf.summary.scalar(
          name=f'eval_episode_average_length',
          data=mean_episode_length,
          step=self._actor.num_timesteps)

    self._eval_logger.write(
      {"total episodes": len(episode_length_buffer),
       "total timesteps": sum(episode_length_buffer),
       "average return": mean_episode_return,
       "average length": mean_episode_length})

  def _run_one_iteration(self):
    self._run_train_phase()
    if self._schedule == 'continuous_train_and_eval':
      print(f"Start evaluation {'.' * 50}")
      if self._eval_unit == 'step':
        threshold = self._eval_steps_per_iteration
      else:
        threshold = self._eval_episodes_per_iteration
      self._run_eval_phase(self._eval_unit, threshold)
    if self._checkpointer:
      self._checkpointer.save()

  def run_experiment(self, num_iterations):
    for iteration in range(1, num_iterations):
      print(f"{'#' * 40} Staring iteration {iteration} {'#' * 40}")
      self._current_iteration = iteration
      self._run_one_iteration()
