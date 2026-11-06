"""adapters/megatron_train.py (ft-sys-v2 recovery, 2026-11-06): the trainer adapter for Megatron-DeepSpeed (megatron/training.py).

The recovery's core asks the trainer for five things; this file answers them with Megatron's OWN functions, called again in the
same order on every rank (no Megatron function is copied or changed):
  reinit_parallel()   parallel_state.destroy_model_parallel + initialize_model_parallel (the arguments of initialize.py:318-323),
                      DeepSpeed's registries reset (the optimizer adapter), the CUDA RNG trackers reset and seeded again
                      (initialize._set_random_seed), DeepSpeed's activation checkpointing configured again (initialize.py:187-211)
  build_model()       training.setup_model_and_optimizer, exactly as pretrain() calls it (it runs deepspeed.initialize)
  build_data()        training.build_train_valid_test_data_iterators: the sampler starts at args.consumed_train_samples
  counters / set_counters / increments / set_scheduler   args.iteration, consumed samples and tokens; the LR scheduler's own
                      step(increment) brings a fresh scheduler to the same state (optimizer_param_scheduler.py:178-189)
  rng_snapshot / rng_restore   python, numpy, torch CPU and GPU generators and the model-parallel tracker
  standby_args()      the standby's arguments without a world: Megatron's parse_args, validate_args and set_global_variables
                      (initialize.py:47-70) with the rank and world size it will have once it joins, then the fused kernels.
Where the position hook is: Megatron's train() is re-entered after a recovery with args.iteration set (it starts at
args.iteration, training.py train()); no line of train() changes."""
import os
import random

import numpy as np
import torch


class MegatronTrainer:
    def __init__(self, model_provider, model_type, data_provider, data_post_process=None):
        self.model_provider, self.model_type = model_provider, model_type
        self.data_provider, self.data_post_process = data_provider, data_post_process

    # ------------------------------------------------------------------ counters
    def counters(self):
        from megatron import get_args
        a = get_args()
        return {'iteration': int(a.iteration), 'samples': int(a.consumed_train_samples), 'tokens': int(a.consumed_train_tokens)}

    def increments(self):
        from megatron import get_args, get_num_microbatches
        from megatron.core import mpu
        a = get_args()
        samples = mpu.get_data_parallel_world_size() * a.micro_batch_size * get_num_microbatches()
        return {'samples': int(samples), 'tokens': int(samples * a.seq_length), 'micro': int(get_num_microbatches())}

    def set_counters(self, iteration, samples, tokens):
        from megatron import get_args
        a = get_args()
        a.iteration, a.consumed_train_samples, a.consumed_train_tokens = int(iteration), int(samples), int(tokens)

    def set_scheduler(self, sched, samples, tokens_before_last_step):
        """A fresh scheduler (num_steps 0) to the state it has after the steps that consumed `samples`: its own step(), once.
        The clean run's last scheduler step saw the tokens counted before that step (training.py updates them after the step)."""
        if sched is not None and samples > 0:
            sched.step(increment=int(samples), token_num=int(tokens_before_last_step))

    # ------------------------------------------------------------------ rebuild
    def reinit_parallel(self, zero_adapter_cls):
        from megatron import get_args
        from megatron.core import mpu, tensor_parallel
        from megatron.initialize import _set_random_seed, setup_deepspeed_random_and_activation_checkpointing
        a = get_args()
        mpu.destroy_model_parallel()
        mpu.initialize_model_parallel(a.tensor_model_parallel_size, a.pipeline_model_parallel_size, a.ds_sequence_parallel_size,
                                      a.virtual_pipeline_model_parallel_size, a.pipeline_model_parallel_split_rank,
                                      use_distributed_optimizer=a.use_distributed_optimizer)
        zero_adapter_cls.reset_registries()
        try:
            tensor_parallel.random._CUDA_RNG_STATE_TRACKER.reset()
        except Exception:
            pass
        if a.deepspeed and a.deepspeed_activation_checkpointing:
            setup_deepspeed_random_and_activation_checkpointing(a)
        _set_random_seed(a.seed, a.data_parallel_random_init)

    def build_model(self):
        from megatron import get_args
        from megatron.training import setup_model_and_optimizer
        model, optimizer, sched = setup_model_and_optimizer(
            self.model_provider, self.model_type, teacher=False, data_post_process=self.data_post_process,
            build_train_valid_test_datasets_provider=self.data_provider)
        a = get_args()
        return model, optimizer, sched

    def build_data(self):
        from megatron.training import build_train_valid_test_data_iterators
        return build_train_valid_test_data_iterators(self.data_provider)

    def reset_timers(self):
        """Timers left running by the interrupted step (train() starts 'interval-time' again; start() asserts it is stopped)."""
        from megatron import get_timers
        t = get_timers()
        for timer in list(getattr(t, '_timers', {}).values()):
            if getattr(timer, '_started', False):
                timer.reset()

    # ------------------------------------------------------------------ RNG
    def rng_snapshot(self):
        from megatron.core import tensor_parallel
        snap = {'python': random.getstate(), 'numpy': np.random.get_state(), 'cpu': torch.get_rng_state()}
        if torch.cuda.is_available():
            snap['cuda'] = torch.cuda.get_rng_state()
            try:
                snap['tracker'] = tensor_parallel.get_cuda_rng_tracker().get_states()
            except Exception:
                pass
        return snap

    def rng_restore(self, snap):
        from megatron.core import tensor_parallel
        random.setstate(snap['python'])
        np.random.set_state(snap['numpy'])
        torch.set_rng_state(snap['cpu'])
        if 'cuda' in snap:
            torch.cuda.set_rng_state(snap['cuda'])
        if 'tracker' in snap:
            tensor_parallel.get_cuda_rng_tracker().set_states(snap['tracker'])

    # ------------------------------------------------------------------ the standby
    def standby_args(self, extra_args_provider, args_defaults, rank, world, local_rank):
        """initialize_megatron's first part (initialize.py:47-70) without a world: the arguments with the rank and world size
        this process will have once it joins; then the device and the fused kernels (initialize.py:115-185, loaded locally)."""
        from megatron.arguments import parse_args, validate_args
        from megatron.global_vars import set_global_variables
        from megatron import fused_kernels
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(local_rank))
        a = parse_args(extra_args_provider, False)
        a.rank, a.world_size, a.local_rank = int(rank), int(world), int(local_rank)
        for key in args_defaults:
            setattr(a, key, args_defaults[key])
        validate_args(a)
        set_global_variables(a)
        torch.cuda.set_device(local_rank)
        try:
            fused_kernels.load(a)
        except Exception as e:
            print(f'[ftv2 standby] fused kernels not loaded: {e!r}'[:300], flush=True)
        return a

    def standby_set_rank(self, rank):
        from megatron import get_args
        a = get_args()
        a.rank = int(rank)
        os.environ['RANK'] = str(rank)
