# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Megatron timers."""

from abc import ABC
from abc import abstractmethod
import time

import torch
from deepspeed.accelerator import get_accelerator
from packaging import version


class TimerBase(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def start(self, barrier=False):
        pass

    @abstractmethod
    def stop(self, barrier=False):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def elapsed(self, reset=True, barrier=False):
        pass



class DummyTimer(TimerBase):

    def __init__(self):
        super().__init__('dummy timer')

    def start(self, barrier=False):
        return

    def stop(self, barrier=False):
        return

    def reset(self):
        return

    def elapsed(self, reset=True, barrier=False):
        raise Exception('dummy timer should not be used to '
                        'calculate elapsed time')



class Timer(TimerBase):
    """
    Comment on using `barrier`: If this flag is passed, then all
    the caller processes will wait till all reach the timing routine.
    It is up to the user to make sure all the ranks in `barrier_group`
    call it otherwise, it will result in a hang.
    Comment on `barrier_group`: By default it is set to None which
    in torch distributed land, it will result in the global communicator.
    """

    def __init__(self, name):
        super().__init__(name)
        self._elapsed = 0.0
        self._started = False
        # Note that None will default to the global process group
        self._barrier_group = None
        self._start_time = time.time()
        self._num_calls = 0    # Zixian: 10/27/2025: included for record attention timer


    def set_barrier_group(self, barrier_group):
        self._barrier_group = barrier_group


    def start(self, barrier=False):
        """Start the timer."""
        assert not self._started, 'timer has already been started'
        if barrier:
            torch.distributed.barrier(group=self._barrier_group)
        get_accelerator().synchronize()
        self._start_time = time.time()
        self._started = True


    def stop(self, barrier=False):
        """Stop the timer."""
        assert self._started, 'timer is not started'
        if barrier:
            torch.distributed.barrier(group=self._barrier_group)
        get_accelerator().synchronize()
        self._elapsed += (time.time() - self._start_time)
        self._started = False
        self._num_calls += 1  


    def reset(self):
        """Reset timer."""
        self._elapsed = 0.0
        self._started = False
        self._num_calls = 0


    def elapsed(self, reset=True, barrier=False):
        """Calculate the elapsed time."""
        _started = self._started
        # If the timing in progress, end it first.
        if self._started:
            self.stop(barrier=barrier)
        # Get the elapsed time.
        _elapsed = self._elapsed
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if _started:
            self.start(barrier=barrier)
        return _elapsed



class Timers:
    """Group of timers."""

    def __init__(self, log_level, log_option):
        self._log_level = log_level
        self._log_option = log_option
        self._timers = {}
        self._log_levels = {}
        self._dummy_timer = DummyTimer()
        self._max_log_level = 2


    def __call__(self, name, log_level=None):
        # If the timer has already been set, then check if the log-level
        # is provided, it matches the one that the timer was created with.
        if name in self._timers:
            if log_level is not None:
                assert log_level == self._log_levels[name], \
                    'input log level {} does not match already existing '\
                    'log level {} for {} timer'.format(
                        log_level, self._log_levels[name], name)
            return self._timers[name]
        # If timer does not exist and no log level is provided,
        # set it to the max log level which is 2.
        if log_level is None:
            log_level = self._max_log_level
        assert log_level <= self._max_log_level, \
            'log level {} is larger than max supported log level {}'.format(
                log_level, self._max_log_level)
        # Now if the input log level is larger than the one set for
        # the timers class, just ignore it and return a dummy timer.
        if log_level > self._log_level:
            return self._dummy_timer
        # Otherwise, initalize the timer and set the level.
        self._timers[name] = Timer(name)
        self._log_levels[name] = log_level
        return self._timers[name]


    def _get_elapsed_time_all_ranks(self, names, reset, barrier):
        """
        Assumptions:
            - All the ranks call this function.
            - `names` are identical on all ranks.
        If the above assumptions are not met, calling this function will
        result in hang.
        Arguments:
            - names: list of timer names
            - reset: reset the timer after recording the elapsed time
            - barrier: if set, do a global barrier before time measurments
        """

        # First make sure all the callers are in sync.
        if barrier:
            torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        # Here we can use gather on the rank we want to print the
        # timing, however, there is no gather_base support in
        # pytorch yet. It is simpler to deal with a single tensor
        # and since we are only gathering a small amount of data,
        # it should be ok to use all-gather instead of gather.
        rank_name_to_time = torch.zeros((world_size, len(names)),
                                        dtype=torch.float,
                                        device=get_accelerator().current_device_name())
        for i, name in enumerate(names):
            if name in self._timers:
                # Here we don't need to pass the barrier flag as all
                # the processes are already in sync. This avoids the
                # issue of different timers having different barrier
                # groups inside their class.
                rank_name_to_time[rank, i] = self._timers[name].elapsed(
                    reset=reset)

        # See the note above for why we are not using gather.
        if version.parse(torch.__version__) >= version.parse('1.13'):
            torch.distributed.all_gather_into_tensor(rank_name_to_time.view(-1),
                                           rank_name_to_time[rank, :].view(-1))
        else:
            torch.distributed._all_gather_base(rank_name_to_time.view(-1),
                                         rank_name_to_time[rank, :].view(-1))

        return rank_name_to_time


    def _get_global_min_max_time(self, names, reset, barrier, normalizer):
        """Report only min and max times across all ranks."""

        rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset,
                                                             barrier)
        name_to_min_max_time = {}
        for i, name in enumerate(names):
            rank_to_time = rank_name_to_time[:, i]
            # filter out the ones we did not have any timings for
            rank_to_time = rank_to_time[rank_to_time > 0.0]
            # If the timer exists:
            if rank_to_time.numel() > 0:
                name_to_min_max_time[name] = (
                    rank_to_time.min().item() / normalizer,
                    rank_to_time.max().item() / normalizer)
        return name_to_min_max_time


    def _get_global_min_max_time_string(self, names, reset, barrier,
                                        normalizer, max_only):
        name_to_min_max_time = self._get_global_min_max_time(
            names, reset, barrier, normalizer)
        if not name_to_min_max_time:
            return None
        output_string = '(min, max) time across ranks (ms):'
        for name in name_to_min_max_time:
            min_time, max_time = name_to_min_max_time[name]
            if max_only:
                output_string += '\n    {}: {:.2f}'.format(
                    (name+' ').ljust(48, '.'), max_time)
            else:
                output_string += '\n    {}: ({:.2f}, {:.2f})'.format(
                    (name+' ').ljust(48, '.'), min_time, max_time)
        return output_string


    def _get_all_ranks_time_string(self, names, reset, barrier, normalizer):
        """Report times across all ranks."""
        rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset,
                                                             barrier)

        output_string = 'times across ranks (ms):'
        no_reported_timing = True
        for i, name in enumerate(names):
            not_yet_found = True
            for rank in range(torch.distributed.get_world_size()):
                if rank_name_to_time[rank, i] > 0:
                    no_reported_timing = False
                    if not_yet_found:
                        not_yet_found = False
                        output_string += '\n  {}:'.format(name)
                    output_string += '\n     rank {:2d}: {:.2f}'.format(
                        rank, rank_name_to_time[rank, i] / normalizer)
        if no_reported_timing:
            return None
        return output_string


    def log(self, names, rank=None, normalizer=1.0, reset=True, barrier=False):
        """Log a group of timers."""

        # Print.
        assert normalizer > 0.0
        if self._log_option in ['max', 'minmax']:
            max_only = False
            if self._log_option == 'max':
                max_only = True
            output_string = self._get_global_min_max_time_string(
                names, reset, barrier, normalizer/1000.0, max_only)
        elif self._log_option == 'all':
            output_string = self._get_all_ranks_time_string(names,
                                                            reset, barrier,
                                                            normalizer/1000.0)
        else:
            raise Exception('unknown timing log option {}'.format(
                self._log_option))

        # If no input rank is provided, log on last rank.
        if rank is None:
            rank = torch.distributed.get_world_size() - 1
        if rank == torch.distributed.get_rank() and output_string is not None:
            print(output_string, flush=True)


    def write(self, names, writer, iteration, normalizer=1.0,
              reset=False, barrier=False):
        """Write timers to a tensorboard writer
        Note that we only report maximum time across ranks to tensorboard.
        """
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        name_to_min_max_time = self._get_global_min_max_time(
            names, reset, barrier, normalizer)
        if writer is not None:
            for name in name_to_min_max_time:
                _, max_time = name_to_min_max_time[name]
                writer.add_scalar(name + '-time', max_time, iteration)


    def log_global_average_times(self, names, reset=True):
        """
        Zixian: 10/27/2025: 
        Calculates and logs the average time for given timers across all calls,
        all layers, all microbatches, and all ranks.
        
        This function should be called only once after all timing is complete 
        (e.g., at the end of a training step).
        """
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        print (f'[megatron/timers.py] inside log_global_average_times')

        # Create a tensor to hold this rank's data: [total_elapsed_time, num_calls] for each timer.
        # Using float64 for accumulators is safer to avoid precision issues.
        local_data = torch.zeros((2, len(names)), dtype=torch.float64,
                                 device=get_accelerator().current_device_name())

        # Populate the tensor with data from the timers on this rank.
        for i, name in enumerate(names):
            if name in self._timers:
                timer = self._timers[name]
                # Access the internal accumulated values from the Timer object
                local_data[0, i] = timer._elapsed
                local_data[1, i] = timer._num_calls
                if reset:
                    timer.reset()

        # Prepare a tensor to receive the data from all ranks.
        # Shape: [world_size, 2 (for time/calls), num_timers]
        global_data = torch.zeros((world_size, 2, len(names)), dtype=torch.float64,
                                  device=get_accelerator().current_device_name())

        # Each rank sends its `local_data` and receives the data from all ranks into `global_data`.
        torch.distributed.all_gather_into_tensor(global_data, local_data)

        # Only Rank 0 will perform the final calculation and printing.
        if rank == 0:
            # Sum the total times and total calls from all ranks.
            # We sum along the 'world_size' dimension (dim=0).
            # The result is a [2, num_timers] tensor with global totals.
            global_totals = torch.sum(global_data, dim=0)

            output_string = "Global Averages (ms) ->"
            for i, name in enumerate(names):
                total_time = global_totals[0, i].item()
                total_calls = global_totals[1, i].item()

                avg_time_ms = 0.0
                if total_calls > 0:
                    # Calculate the average time and convert it to milliseconds.
                    avg_time_ms = (total_time / total_calls) * 1000
                
                # Format the name for consistent alignment
                formatted_name = (name + ':').ljust(18)
                output_string += f" {formatted_name}{avg_time_ms:<7.2f} |"

            # Print the final, single-line report. `[:-2]` removes the trailing " |".
            print(output_string[:-2], flush=True)