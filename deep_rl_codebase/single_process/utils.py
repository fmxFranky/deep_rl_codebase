import time

import numpy as np


class WindowStat(object):
  """ Tool to maintain statistical data in a window.
  """

  def __init__(self, window_size):
    self.items = [None] * window_size
    self.idx = 0
    self.count = 0

  def add(self, obj):
    self.items[self.idx] = obj
    self.idx += 1
    self.count += 1
    self.idx %= len(self.items)

  @property
  def mean(self):
    if self.count > 0:
      return np.mean(self.items[:self.count])
    else:
      return None

  @property
  def min(self):
    if self.count > 0:
      return np.min(self.items[:self.count])
    else:
      return None

  @property
  def max(self):
    if self.count > 0:
      return np.max(self.items[:self.count])
    else:
      return None


class TimeStat(object):
  """A time stat for logging the elapsed time of code running
  Example:
      time_stat = TimeStat()
      with time_stat:
          // some code
      print(time_stat.mean)
  """

  def __init__(self, window_size=1):
    self.time_samples = WindowStat(window_size)
    self._start_time = None

  def __enter__(self):
    self._start_time = time.time()

  def __exit__(self, type, value, tb):
    time_delta = time.time() - self._start_time
    self.time_samples.add(time_delta)

  @property
  def mean(self):
    return self.time_samples.mean

  @property
  def min(self):
    return self.time_samples.min

  @property
  def max(self):
    return self.time_samples.max
