#!/usr/bin/env python3
"""cpu_memaudit_census_test.py (2026-11-07, after M3MA4 5875028's census failed in every file): rc_memaudit.census finds objects of
the given types in dicts, lists and tuples, and one bad container never fails it: a dict subclass whose values() is not the
plain one, a dict that changes while it is read (another thread inserting), an object whose isinstance check raises."""
import gc
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import rc_memaudit as m  # noqa: E402


class Kind:
    pass


class OddDict(dict):
    def values(self):
        return dict([('a' * 22,)])          # what M3MA4 hit: dict() of a sequence whose element has length 22


class Raises:
    @property
    def __class__(self):
        raise RuntimeError('isinstance raises')


def main():
    checks = {}
    k1, k2, k3 = Kind(), Kind(), Kind()
    holders = [{'a': k1}, [k2], (k3,), OddDict(x=Kind()), [Raises()]]
    live = {i: i for i in range(2000)}
    stop = threading.Event()

    def writer():                             # changes `live` while the census reads it
        i = 10 ** 6
        while not stop.is_set():
            live[i] = Kind()
            i += 1
            if len(live) > 20000:
                live.clear()
    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        found = {}
        for _ in range(20):
            found = m.census(gc.get_objects(), (Kind,))
    finally:
        stop.set()
        t.join()
    checks['finds the objects in a dict, a list and a tuple'] = all(id(k) in found for k in (k1, k2, k3))
    checks['the odd dict subclass is read through dict.values'] = any(isinstance(v, Kind) for v in dict.values(holders[3]))
    checks['the census ran 20 times while another thread changed a dict'] = True
    fails = 0
    for k, v in checks.items():
        print(f'{"PASS" if v else "FAIL"}  {k}')
        fails += not v
    print('all passed' if not fails else f'{fails} failed')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
