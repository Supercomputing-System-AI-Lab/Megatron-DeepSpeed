#!/usr/bin/env python3
"""failure_record.py: the failure record (one TSV line per failure) and the lemon list (design 3.5).

Record schema (tab-separated, header line):
  job time class code node rank step injected evidence
  time = 'YYYY-MM-DD HH:MM:SS' local; injected = yes|no; unknown fields '-'.

CLI:
  failure_record.py append --record P [--record P2 ...] --job J --cls CLASS --code N --node NODE
                    --rank R --step S --injected yes|no --evidence TEXT [--time '...']
  failure_record.py lemons --days 30 --record P [--record ...] [--now '...'] [--include-injected] [--nodes-only]

Lemon rule: a node with >= 1 HARD line, or >= 2 SOFT lines (the gate_<test> classes count as soft), within the last --days days.
Injected lines are ignored unless --include-injected. Duplicates (job+node+class+time) count once.
"""
import argparse, datetime, fcntl, os, sys

FIELDS = ["job", "time", "class", "code", "node", "rank", "step", "injected", "evidence"]
HEADER = "\t".join(FIELDS) + "\n"
HARD = {"node_lost", "ecc_uncorrectable", "xgmi_error"}
SOFT = {"collective_hang", "slow_rank", "filesystem_hang", "rank_died", "driver_hip_error", "host_oom"}
# [2026-10-09 late evening] the pre-run gate's exclusions are recorded too, one line per excluded node, with the class
# gate_<test> (gate_gemm, gate_pair, gate_switch, gate_ras_ue, ...; code 30). They are SOFT: one line says the node was slow or
# silent once (the degraded tier included); two such lines in 30 days make it a lemon.
GATE_PREFIX = "gate_"


def is_soft(cls):
    return cls in SOFT or cls.startswith(GATE_PREFIX)
TFMT = "%Y-%m-%d %H:%M:%S"


def _clean(v):
    s = "-" if v is None else str(v)
    s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
    return s if s else "-"


def append_line(paths, **fields):
    """Append one record line to every path (flock'd); returns the line (without newline)."""
    if isinstance(paths, str):
        paths = [paths]
    if "cls" in fields and "class" not in fields:
        fields["class"] = fields.pop("cls")
    if not fields.get("time") or fields.get("time") == "-":
        fields["time"] = datetime.datetime.now().strftime(TFMT)
    inj = str(fields.get("injected", "no")).lower()
    fields["injected"] = "yes" if inj in ("yes", "y", "1", "true") else "no"
    line = "\t".join(_clean(fields.get(k)) for k in FIELDS)
    for p in paths:
        d = os.path.dirname(os.path.abspath(p))
        os.makedirs(d, exist_ok=True)
        with open(p, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0, os.SEEK_END)
                if f.tell() == 0:
                    f.write(HEADER)
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return line


def read_records(paths):
    """Yield dicts from every record file; skip header/malformed lines; dedupe on job+node+class+time."""
    seen = set()
    for p in ([paths] if isinstance(paths, str) else paths):
        try:
            fh = open(p)
        except OSError:
            continue
        with fh:
            for raw in fh:
                parts = raw.rstrip("\n").split("\t")
                if len(parts) < len(FIELDS) - 1 or parts[0] == "job":
                    continue
                parts += ["-"] * (len(FIELDS) - len(parts))
                r = dict(zip(FIELDS, parts[:len(FIELDS)]))
                key = (r["job"], r["node"], r["class"], r["time"])
                if key in seen:
                    continue
                seen.add(key)
                yield r


def _parse_time(s):
    try:
        return datetime.datetime.strptime(s.strip(), TFMT)
    except (ValueError, AttributeError):
        return None


def lemons(paths, days, now=None, include_injected=False):
    """Return a sorted list of (node, 'hard'|'soft', 'class@time; ...')."""
    if now is None:
        now = datetime.datetime.now()
    elif isinstance(now, str):
        now = datetime.datetime.strptime(now, TFMT)
    start = now - datetime.timedelta(days=float(days))
    hard, soft = {}, {}
    for r in read_records(paths):
        node = r["node"].strip()
        if node in ("", "-"):
            continue
        if r["injected"].strip().lower() == "yes" and not include_injected:
            continue
        t = _parse_time(r["time"])
        if t is None or t < start or t > now:
            continue
        cls = r["class"].strip()
        if cls in HARD:
            hard.setdefault(node, []).append((t, cls))
        elif is_soft(cls):
            soft.setdefault(node, []).append((t, cls))
    out = []
    for node in sorted(set(hard) | set(soft)):
        if node in hard:
            kind, ev = "hard", sorted(hard[node] + soft.get(node, []))
        elif len(soft[node]) >= 2:
            kind, ev = "soft", sorted(soft[node])
        else:
            continue
        out.append((node, kind, "; ".join("%s@%s" % (c, t.strftime(TFMT)) for t, c in ev)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("append")
    a.add_argument("--record", action="append", required=True)
    for k in ("job", "cls", "code", "node", "rank", "step", "evidence", "time"):
        a.add_argument("--" + k, default="-")
    a.add_argument("--injected", default="no", choices=["yes", "no"])
    l = sub.add_parser("lemons")
    l.add_argument("--record", action="append", required=True)
    l.add_argument("--days", type=float, default=30)
    l.add_argument("--now", default=None)
    l.add_argument("--include-injected", action="store_true")
    l.add_argument("--nodes-only", action="store_true")
    o = ap.parse_args(argv)
    if o.cmd == "append":
        line = append_line(o.record, job=o.job, cls=o.cls, code=o.code, node=o.node, rank=o.rank,
                           step=o.step, injected=o.injected, evidence=o.evidence, time=o.time)
        print("[failure_record] " + line)
        return 0
    try:
        res = lemons(o.record, o.days, now=o.now, include_injected=o.include_injected)
    except Exception as e:  # never break the caller
        print("[failure_record] lemons failed: %s" % e, file=sys.stderr)
        res = []
    for node, kind, reasons in res:
        print(node if o.nodes_only else "%s\t%s\t%s" % (node, kind, reasons))
    return 0


if __name__ == "__main__":
    sys.exit(main())
