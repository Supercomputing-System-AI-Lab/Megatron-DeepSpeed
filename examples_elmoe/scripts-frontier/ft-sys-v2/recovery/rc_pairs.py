"""rc_pairs.py (ft-sys-v2 recovery, milestone 3, 2026-11-06): who keeps whose copy, with or without a pipeline.

Without a pipeline the partner of rank r is r + S (mod W), S = FTV2_PARTNER_STRIDE (8, one node): the same position on the next
node. Under a pipeline of P stages (FTV2_PP), DeepSpeed and Megatron number the ranks stage by stage (the pipeline axis is the
outermost: stage s holds ranks s*R .. s*R + R - 1, R = W / P; deepspeed/runtime/pipe/topology.py PipeModelDataParallelTopology,
axes ['pipe', 'data', 'model']). A partner in another stage would hold a slice of another size of other parameters, so the
partner is the same stage's next replica (m3_map question 2, Figure 8 panel 1): base + (r - base + S) mod R. With P = 1 this is
the old rule. Pure functions, one place for every caller (rc_keep, rc_controller)."""
import os


def pp(world=None):
    return max(1, int(os.environ.get('FTV2_PP', '1')))


def stage_size(world):
    p = pp()
    if world % p:
        raise ValueError(f'world {world} is not a multiple of FTV2_PP {p}')
    return world // p


def stage_of(rank, world):
    return rank // stage_size(world)


def stride():
    return int(os.environ.get('FTV2_PARTNER_STRIDE', '8'))


def partner(rank, world, s=None):
    """The rank that keeps rank's copy."""
    R = stage_size(world)
    base = rank - rank % R
    return base + (rank - base + (stride() if s is None else s)) % R


def owner(rank, world, s=None):
    """The rank whose copy rank keeps."""
    R = stage_size(world)
    base = rank - rank % R
    return base + (rank - base - (stride() if s is None else s)) % R
