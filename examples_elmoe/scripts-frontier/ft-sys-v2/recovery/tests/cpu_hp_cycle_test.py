#!/usr/bin/env python3
"""cpu_hp_cycle_test.py (2026-11-07, M3MA6 5875594): a parameter linked the way DeepSpeed's link_hp_params links it (p._hp_mapping
-> a fragment whose lp_fragment is p.flatten().narrow(...) made with autograd on) survives every reference drop and gc.collect,
because the fragment's grad_fn chain ends in p's AccumulateGrad, which holds p from C++; rc_memaudit.break_hp_cycles clears the
link, and then p and what its fragment reaches are freed. The holder test (_hp_holders) sees the cycle. CPU only."""
import gc
import os
import sys
import types
import weakref

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import rc_memaudit as m  # noqa: E402


class Fragment:                                  # as deepspeed.utils.tensor_fragment.tensor_fragment
    def __init__(self, lp_fragment, gradient_dict):
        self.lp_fragment, self.gradient_dict = lp_fragment, gradient_dict


def get_full_hp_grad(self):
    return None


def make():
    p = torch.nn.Parameter(torch.zeros(64, 32))
    big = torch.zeros(1 << 20)                   # stands for the gradient_dict's flat buffer
    p._hp_mapping = Fragment(p.flatten().narrow(0, 8, 100), {0: [big]})
    p.get_full_hp_grad = types.MethodType(get_full_hp_grad, p)
    return weakref.ref(p), weakref.ref(big)


def main():
    checks = {}
    wp, wb = make()
    gc.collect()
    checks['the linked parameter survives gc (held from C++)'] = wp() is not None and wb() is not None
    lines = []
    m._hp_holders(gc.get_objects(), lines)
    checks['the holder test sees the cycle through AccumulateGrad'] = any('AccumulateGrad holding this Parameter: True' in x
                                                                         for x in lines)
    n = m.break_hp_cycles()
    gc.collect()
    checks['break_hp_cycles clears it (at least 1)'] = n >= 1
    checks['then the parameter and its buffer are freed'] = wp() is None and wb() is None
    q = torch.nn.Parameter(torch.zeros(4))
    m.break_hp_cycles()
    checks['a parameter without the link is left alone'] = q.__dict__ == {} or '_hp_mapping' not in q.__dict__
    fails = 0
    for k, v in checks.items():
        print(f'{"PASS" if v else "FAIL"}  {k}')
        fails += not v
    if fails:
        print('\n'.join(lines))
    print('all passed' if not fails else f'{fails} failed')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
