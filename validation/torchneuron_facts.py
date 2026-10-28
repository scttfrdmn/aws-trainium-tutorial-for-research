#!/usr/bin/env python3
"""TorchNeuron (native backend, beta) facts probe -- record what the backend actually does.

Several tutorial claims about TorchNeuron (torch-neuronx 2.13+/2.14+, device ``"neuron"``,
Trn2/Trn3) are still "per AWS docs" rather than observed: when eager ops compile, whether
``NEURON_COMPILE_CACHE_URL`` applies, what happens to ops the device can't lower, the bf16 SDPA
masking nan, which sync APIs exist, and which ``torch.distributed`` backends work (issues #26,
#28, #33). This script OBSERVES those behaviours on real hardware and records them; it asserts
nothing about what the answers "should" be.

Run it INSIDE a TorchNeuron venv on a Trn2/Trn3 box (budget: under ~10 minutes)::

    . /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
    python -m validation.torchneuron_facts

Design:

* Each probe is a ``probe_<name>()`` function returning a dict of observations. ``run_probes``
  calls each one independently under try/except, so one failing probe records an ``"error"`` and
  the rest still run.
* Device work runs in child ``python -u -c ...`` processes. That isolates device crashes/hangs
  (each child has a timeout) and lets us capture the Neuron compiler's stdout/stderr log lines
  (e.g. ``Compilation Successfully Completed``) per phase, which an in-process call can't see.
* No APIs are assumed: anything beyond ``torch.device("neuron")`` is looked up with
  getattr/try and its presence or absence is itself recorded.

Outputs (``validation/results/``):

* ``validation.torchneuron_facts@<instance>.json`` -- a normal ``ValidationResult``.
  ``status == "passed"`` means only that the PROBE SCRIPT COMPLETED (every probe ran, even if some
  recorded errors) -- it is NOT a claim that TorchNeuron passed anything. Metrics are minimal.
* ``validation.torchneuron_facts@<instance>.observations.json`` -- every probe's observations.
  ``ValidationResult`` has fixed fields and no free-form slot, so the observations live in this
  sibling file and the result's ``log_path`` points at it. (It has no ``example`` key, so
  ``render_status`` groups it under its own stem and never mistakes it for an example's artifact.)

Off-hardware (no ``torch.neuron`` / not available) it writes a ``failed`` result with
``error="neuron unavailable: ..."``, no observations file, and exits non-zero. Not a registered
example: like ``torchneuron_smoke`` it is not counted in VALIDATED.md's coverage. torch is
imported lazily so this module imports on a laptop.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation.provenance import ValidationResult, capture_environment  # noqa: E402

RESULTS_DIR = _REPO_ROOT / "validation" / "results"
EXAMPLE = "validation.torchneuron_facts"

# Device string the child processes use. "neuron" on hardware; only ever changed by a developer
# exercising the child snippets on CPU (main() still refuses to run without TorchNeuron).
_DEVICE = "neuron"

# Probe order == run order. Each name maps to a module-level `probe_<name>` function, looked up at
# call time so tests can monkeypatch a single probe.
PROBE_NAMES: tuple[str, ...] = (
    "eager_compile_behavior",
    "compile_cache_env",
    "unlowerable_op",
    "bf16_sdpa",
    "sync_api",
    "distributed_backends",
)

# Substrings counted in captured child output, per phase.
LOG_PATTERNS: tuple[str, ...] = (
    "Compilation Successfully Completed",
    "neuronx-cc",
    "Compil",
    "fallback",
)
_DEFAULT_CACHE_DIR = (
    "/var/tmp/neuron-compile-cache"  # Neuron's documented default location
)

# Shared header for every child snippet: device handle, JSON emit, phase markers, NaN-safe floats.
_PRELUDE = r"""
import json, os, sys, time
import torch
DEV = torch.device(__DEVICE__)
def emit(obj):
    print("@@JSON " + json.dumps(obj, default=str), flush=True)
def phase(name):
    print("@@PHASE " + name, flush=True)
def fnum(x):
    x = float(x)
    return x if x == x and abs(x) != float("inf") else str(x)
def err(exc):
    first = (str(exc).strip().splitlines() or [""])[0]
    return type(exc).__name__ + ": " + first[:300]
def count_files(root):
    if not root or root.startswith("s3://") or not os.path.isdir(root):
        return None
    return sum(len(files) for _, _, files in os.walk(root))
"""


# --------------------------------------------------------------------------------------------
# Child-process plumbing
# --------------------------------------------------------------------------------------------


def _count_patterns(lines: list[str]) -> dict[str, int]:
    """Count lines containing each LOG_PATTERNS entry ("fallback" is case-insensitive)."""
    counts = {}
    for pat in LOG_PATTERNS:
        if pat == "fallback":
            counts[pat] = sum(1 for ln in lines if pat in ln.lower())
        else:
            counts[pat] = sum(1 for ln in lines if pat in ln)
    return counts


def parse_child_output(text: str) -> dict[str, Any]:
    """Split child output on ``@@PHASE`` markers; count log patterns per phase; pull ``@@JSON``.

    Returns ``{"data": <last @@JSON object or None>, "phases": {phase: {...counts, "samples"}}}``.
    Lines before the first marker are phase ``"_setup"`` (import/device init).
    """
    phases: dict[str, list[str]] = {"_setup": []}
    current = "_setup"
    data = None
    for line in text.splitlines():
        if line.startswith("@@PHASE "):
            current = line[len("@@PHASE ") :].strip()
            phases.setdefault(current, [])
        elif line.startswith("@@JSON "):
            try:
                data = json.loads(line[len("@@JSON ") :])
            except json.JSONDecodeError:
                data = {"unparseable_json_line": line[:500]}
        else:
            phases[current].append(line)
    out: dict[str, Any] = {}
    for name, lines in phases.items():
        counts: dict[str, Any] = _count_patterns(lines)
        counts["lines"] = len(lines)
        counts["samples"] = [
            ln[:200] for ln in lines if any(p in ln for p in ("Compil", "neuronx-cc"))
        ][:3]
        out[name] = counts
    return {"data": data, "phases": out}


def run_child(
    body: str,
    timeout: float,
    env: dict[str, str] | None = None,
    args: list[str] | None = None,
) -> dict[str, Any]:
    """Run ``_PRELUDE + body`` in a fresh interpreter; return parsed observations.

    stdout and stderr are merged (so compiler log lines interleave with our phase markers). On a
    timeout or a child that never emitted ``@@JSON``, the dict carries ``"error"`` and an output
    tail for diagnosis.
    """
    code = _PRELUDE.replace("__DEVICE__", repr(_DEVICE)) + textwrap.dedent(body)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-u", "-c", code, *(args or [])],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env={**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1"},
            check=False,
        )
        text, rc, timed_out = proc.stdout or "", proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        raw = exc.stdout or ""
        text = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
        rc, timed_out = None, True
    obs: dict[str, Any] = parse_child_output(text)
    obs.update(
        returncode=rc,
        timed_out=timed_out,
        elapsed_s=round(time.perf_counter() - start, 2),
    )
    if timed_out:
        obs["error"] = f"child timed out after {timeout}s"
    elif obs["data"] is None:
        obs["error"] = f"child exited {rc} without reporting"
    if "error" in obs:
        obs["output_tail"] = [ln[:300] for ln in text.splitlines()[-25:]]
    return obs


def _cache_dir() -> str:
    """The compile-cache location in effect for children (env var, else Neuron's default)."""
    return os.environ.get("NEURON_COMPILE_CACHE_URL") or _DEFAULT_CACHE_DIR


def _list_files(root: Path, limit: int = 8) -> tuple[int, list[str]]:
    """Return (file count, first `limit` relative paths) under root (0, [] if absent)."""
    if not root.is_dir():
        return 0, []
    files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    return len(files), files[:limit]


# --------------------------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------------------------

_EAGER_BODY = r"""
cache = sys.argv[1]
out = {"cache_dir": cache, "cache_files_start": count_files(cache)}
for name, n in (("A_first", 256), ("A_second", 256), ("B_first", 384), ("B_second", 384)):
    a = torch.randn(n, n).to(DEV)
    b = torch.randn(n, n).to(DEV)
    phase(name)
    try:
        t = time.perf_counter()
        c = a @ b
        host = c.cpu()  # forces completion, so the timing covers compile + execute
        out[name] = {
            "shape": n,
            "seconds": round(time.perf_counter() - t, 4),
            "result_device": str(c.device),
            "max_abs_err_vs_cpu": fnum((host - a.cpu() @ b.cpu()).abs().max()),
            "cache_files_after": count_files(cache),
        }
    except Exception as exc:
        out[name] = {"shape": n, "error": err(exc)}
emit(out)
"""


def probe_eager_compile_behavior() -> dict[str, Any]:
    """Time first vs repeat eager matmul at shape A, then a NEW shape B twice.

    Records per-phase seconds, compile-log-line counts, and compile-cache file counts, i.e.
    whether eager ops compile per new shape and whether the repeat call hits a cache.
    """
    return run_child(_EAGER_BODY, timeout=240, args=[_cache_dir()])


_CACHE_BODY = r"""
cache = sys.argv[1]
out = {"cache_dir": cache}
x = torch.randn(64, 64).to(DEV)
phase("torch_compile")
try:
    f = torch.compile(lambda t: (t @ t.T).relu(), backend="neuron")
    out["torch_compile"] = {"result_device": str(f(x).cpu().device)}
except Exception as exc:
    out["torch_compile"] = {"error": err(exc)}
out["files_after_compile"] = count_files(cache)
phase("eager")
try:
    y = (x * 3.0 + 1.0).tanh()
    out["eager"] = {"result_device": str(y.device), "sum": fnum(y.cpu().sum())}
except Exception as exc:
    out["eager"] = {"error": err(exc)}
out["files_after_eager"] = count_files(cache)
emit(out)
"""


def probe_compile_cache_env() -> dict[str, Any]:
    """Point NEURON_COMPILE_CACHE_URL at a fresh temp dir; see whether artifacts land there.

    Also counts new files in the default cache dir, to tell "env var ignored" from "nothing
    cached at all".
    """
    tmp = Path(tempfile.mkdtemp(prefix="tn_facts_cache_"))
    default = Path(_DEFAULT_CACHE_DIR)
    default_before, _ = _list_files(default)
    try:
        obs = run_child(
            _CACHE_BODY,
            timeout=180,
            env={"NEURON_COMPILE_CACHE_URL": str(tmp)},
            args=[str(tmp)],
        )
        count, names = _list_files(tmp)
        default_after, _ = _list_files(default)
        obs.update(
            env_cache_dir_file_count=count,
            env_cache_dir_sample=names,
            env_var_applied=count > 0,
            default_cache_dir=str(default),
            default_cache_new_files=default_after - default_before,
        )
        return obs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_UNLOWERABLE_BODY = r"""
import warnings
def first_tensor(r):
    if isinstance(r, torch.Tensor):
        return r
    for item in (r if isinstance(r, (tuple, list)) else getattr(r, "__iter__", lambda: [])()):
        if isinstance(item, torch.Tensor):
            return item
    return None
cases = {
    "linalg_eig": lambda: torch.linalg.eig(torch.randn(8, 8).to(DEV)),
    "fft_complex": lambda: torch.fft.fft(torch.randn(16, dtype=torch.complex64).to(DEV)),
    "sparse_mm": lambda: torch.sparse.mm(
        torch.eye(8).to_sparse().to(DEV), torch.randn(8, 8).to(DEV)
    ),
}
out = {}
for name, fn in cases.items():
    phase(name)
    rec = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            t = first_tensor(fn())
            rec["raised"] = False
            rec["result_device"] = None if t is None else str(t.device)
            if t is not None:
                rec["ran_on"] = t.device.type  # "cpu" => silent fallback
        except Exception as exc:
            rec["raised"] = True
            rec["exception"] = err(exc)
    rec["warnings"] = [str(w.message).splitlines()[0][:200] for w in caught][:3]
    out[name] = rec
emit(out)
"""


def probe_unlowerable_op() -> dict[str, Any]:
    """Try ops unlikely to lower to the device; record raise vs silent CPU/neuron execution."""
    return run_child(_UNLOWERABLE_BODY, timeout=180)


_SDPA_BODY = r"""
import math
import torch.nn.functional as F
torch.manual_seed(0)
B, H, S, D = 2, 4, 16, 32
q, k, v = (torch.randn(B, H, S, D) for _ in range(3))
keep = torch.ones(B, S, dtype=torch.bool)
keep[:, S // 2:] = False  # padded tail, like a batch of short sequences
def manual(q, k, v, m):
    return torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(D) + m, dim=-1) @ v
def stats(o, ref):
    dtype = str(o.dtype)  # before the upcast below: did autocast actually run in bf16?
    o = o.float().cpu()
    return {
        "dtype": dtype,
        "has_nan": bool(torch.isnan(o).any()),
        "has_inf": bool(torch.isinf(o).any()),
        "max_abs_diff_vs_cpu_fp32": fnum((o - ref).abs().max()),
    }
out = {"autocast_device_type": DEV.type, "shape": [B, H, S, D]}
masks = {"large_negative": torch.finfo(torch.float32).min, "neg_inf": float("-inf")}
for mname, neg in masks.items():
    m = torch.where(keep, 0.0, neg)[:, None, None, :]
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
    for impl in ("sdpa", "manual"):
        key = mname + "/" + impl
        phase(key)
        try:
            qn, kn, vn, mn = (t.to(DEV) for t in (q, k, v, m))
            with torch.autocast(device_type=DEV.type, dtype=torch.bfloat16):
                if impl == "sdpa":
                    o = F.scaled_dot_product_attention(qn, kn, vn, attn_mask=mn)
                else:
                    o = manual(qn, kn, vn, mn)
            out[key] = stats(o, ref)
        except Exception as exc:
            out[key] = {"error": err(exc)}
try:
    from transformers import BertForTokenClassification
except Exception as exc:
    out["bert_tiny"] = {"skipped": "transformers not importable: " + err(exc)}
else:
    bert = {}
    ids = torch.randint(0, 100, (2, 16))
    labels = torch.zeros(2, 16, dtype=torch.long)
    pad = torch.ones(2, 16, dtype=torch.long)
    pad[:, 12:] = 0
    for attn in ("sdpa", "eager"):
        for mname, mask in (("all_ones", torch.ones_like(pad)), ("padded", pad)):
            key = attn + "/" + mname
            phase("bert/" + key)
            try:
                model = BertForTokenClassification.from_pretrained(
                    "prajjwal1/bert-tiny", num_labels=3, attn_implementation=attn
                )
                model.eval()
                with torch.no_grad():
                    cpu_loss = float(model(input_ids=ids, attention_mask=mask, labels=labels).loss)
                    model = model.to(DEV)
                    with torch.autocast(device_type=DEV.type, dtype=torch.bfloat16):
                        o = model(
                            input_ids=ids.to(DEV),
                            attention_mask=mask.to(DEV),
                            labels=labels.to(DEV),
                        )
                    loss = float(o.loss.detach().cpu())
                bert[key] = {
                    "loss": fnum(loss),
                    "loss_finite": math.isfinite(loss),
                    "cpu_fp32_loss": fnum(cpu_loss),
                }
            except Exception as exc:
                bert[key] = {"error": err(exc)}
    out["bert_tiny"] = bert
emit(out)
"""


def probe_bf16_sdpa() -> dict[str, Any]:
    """bf16-autocast SDPA vs manual attention with -large / -inf additive masks; BERT-tiny too."""
    return run_child(_SDPA_BODY, timeout=300)


_SYNC_BODY = r"""
def lookup(path):
    obj = torch
    for part in path.split(".")[1:]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj
names = ("torch.neuron.synchronize", "torch.neuron.current_stream", "torch.accelerator.synchronize")
out = {"apis": {}}
fns = {}
for name in names:
    fn = lookup(name)
    out["apis"][name] = {"exists": fn is not None, "callable": callable(fn)}
    if callable(fn):
        fns[name] = fn
n = 4096
a = torch.randn(n, n).to(DEV)
b = torch.randn(n, n).to(DEV)
phase("warmup")
(a @ b).cpu()
def timed(sync, reps=3):
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        c = a @ b
        if sync is not None:
            sync()
        times.append(round(time.perf_counter() - t, 5))
        c.cpu()  # drain before the next measurement
    return times
phase("timing")
out["matmul_shape"] = n
out["no_sync_s"] = timed(None)
def via_cpu():
    times = []
    for _ in range(3):
        t = time.perf_counter()
        (a @ b).cpu()
        times.append(round(time.perf_counter() - t, 5))
    return times
out["with_cpu_copy_s"] = via_cpu()
for name, fn in fns.items():
    rec = out["apis"][name]
    try:
        if name.endswith("current_stream"):
            stream = fn()
            rec["returned"] = type(stream).__module__ + "." + type(stream).__name__
            ssync = getattr(stream, "synchronize", None)
            rec["stream_has_synchronize"] = callable(ssync)
            if callable(ssync):
                rec["with_stream_sync_s"] = timed(ssync)
        else:
            fn()
            rec["call_ok"] = True
            rec["with_sync_s"] = timed(fn)
    except Exception as exc:
        rec["error"] = err(exc)
emit(out)
"""


def probe_sync_api() -> dict[str, Any]:
    """Which sync APIs exist, and does each one block (vs dispatch-only timing)?"""
    return run_child(_SYNC_BODY, timeout=180)


_DIST_INVENTORY_BODY = r"""
import torch.distributed as dist
cands = json.loads(sys.argv[1])
out = {"is_available": bool(dist.is_available())}
iba = getattr(dist, "is_backend_available", None)
out["has_is_backend_available"] = callable(iba)
out["is_backend_available"] = {}
for name in cands:
    if callable(iba):
        try:
            out["is_backend_available"][name] = bool(iba(name))
        except Exception as exc:
            out["is_backend_available"][name] = err(exc)
Backend = getattr(dist, "Backend", None)
if Backend is not None:
    bl = getattr(Backend, "backend_list", None)
    out["Backend.backend_list"] = list(bl) if bl is not None else None
    dm = getattr(Backend, "default_device_backend_map", None)
    out["Backend.default_device_backend_map"] = dict(dm) if dm is not None else None
    out["Backend.upper_attrs"] = sorted(a for a in dir(Backend) if a.isupper())
emit(out)
"""

_DIST_WORKER_BODY = r"""
import datetime
import torch.distributed as dist
backend = sys.argv[1]
rank = int(os.environ["RANK"])
out = {"rank": rank, "backend": backend}
try:
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=45))
    out["init_ok"] = True
    out["get_backend"] = str(dist.get_backend())
    t = torch.ones(4).to(DEV) * (rank + 1)
    dist.all_reduce(t)
    out["all_reduce_result"] = t.cpu().tolist()
    out["all_reduce_correct"] = out["all_reduce_result"] == [3.0] * 4
    dist.destroy_process_group()
except Exception as exc:
    out["error"] = err(exc)
emit(out)
"""

DIST_CANDIDATES: tuple[str, ...] = ("neuron", "nccl", "gloo", "xla", "ccom")


def _free_port() -> int:
    """Return an unused localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _two_rank_attempt(backend: str, timeout: float = 90) -> dict[str, Any]:
    """Spawn 2 local ranks that init_process_group(backend) + all_reduce a neuron tensor.

    Each rank gets ``NEURON_RT_VISIBLE_CORES=<rank>`` so the two processes don't contend for the
    same NeuronCore; that choice is recorded in the observations.
    """
    port = _free_port()
    code = _PRELUDE.replace("__DEVICE__", repr(_DEVICE)) + textwrap.dedent(
        _DIST_WORKER_BODY
    )
    procs = []
    for rank in range(2):
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "WORLD_SIZE": "2",
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "NEURON_RT_VISIBLE_CORES": str(rank),
        }
        procs.append(
            subprocess.Popen(
                [sys.executable, "-u", "-c", code, backend],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        )
    deadline = time.monotonic() + timeout
    ranks = []
    for proc in procs:
        try:
            text, _ = proc.communicate(timeout=max(1.0, deadline - time.monotonic()))
            rec: dict[str, Any] = {"returncode": proc.returncode, "timed_out": False}
        except subprocess.TimeoutExpired:
            proc.kill()
            text, _ = proc.communicate()
            rec = {"returncode": None, "timed_out": True}
        parsed = parse_child_output(text or "")
        rec["data"] = parsed["data"]
        if parsed["data"] is None or rec["timed_out"]:
            rec["output_tail"] = [ln[:300] for ln in (text or "").splitlines()[-15:]]
        ranks.append(rec)
    ok = all(
        isinstance(r["data"], dict) and r["data"].get("all_reduce_correct")
        for r in ranks
    )
    return {"success": ok, "visible_cores_per_rank": ["0", "1"], "ranks": ranks}


def probe_distributed_backends() -> dict[str, Any]:
    """Inventory torch.distributed backends, then try a 2-rank all_reduce per available one."""
    obs = run_child(
        _DIST_INVENTORY_BODY, timeout=60, args=[json.dumps(list(DIST_CANDIDATES))]
    )
    data = obs.get("data")
    inv: dict[str, Any] = data if isinstance(data, dict) else {}
    avail = inv.get("is_backend_available") or {}
    # Inventory failed or torch.distributed absent: nothing to attempt.
    if not inv or not inv.get("is_available"):
        to_try: list[str] = []
    elif inv.get("has_is_backend_available"):
        to_try = [name for name in DIST_CANDIDATES if avail.get(name) is True]
    else:  # no availability query: try every candidate (each attempt is time-boxed)
        to_try = list(DIST_CANDIDATES)
    obs["attempted_backends"] = to_try
    attempts = {}
    for name in to_try:
        try:
            attempts[name] = _two_rank_attempt(name)
        except Exception as exc:
            attempts[name] = {"error": f"{type(exc).__name__}: {exc}"}
    obs["two_rank_all_reduce"] = attempts
    return obs


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def _probe(name: str) -> Callable[[], dict[str, Any]]:
    """Look up ``probe_<name>`` at call time (so a monkeypatched probe is honoured)."""
    fn: Callable[[], dict[str, Any]] = globals()[f"probe_{name}"]
    return fn


def run_probes(names: tuple[str, ...] = PROBE_NAMES) -> dict[str, dict[str, Any]]:
    """Run each probe independently; a raising probe becomes ``{"error": ...}``, others continue."""
    observations: dict[str, dict[str, Any]] = {}
    for name in names:
        start = time.perf_counter()
        print(f"[torchneuron_facts] probe {name} ...", flush=True)
        try:
            obs = dict(_probe(name)())
        except Exception as exc:
            obs = {"error": f"{type(exc).__name__}: {exc}"}
        obs.setdefault("probe_wall_s", round(time.perf_counter() - start, 2))
        observations[name] = obs
    return observations


def neuron_status() -> tuple[bool, str]:
    """Return (available, detail) for TorchNeuron in THIS interpreter; never raises."""
    try:
        import torch  # autoloads TorchNeuron when torch-neuronx 2.13+/2.14+ is installed
    except Exception as exc:
        return False, f"torch not importable ({type(exc).__name__}: {exc})"
    neuron = getattr(torch, "neuron", None)
    if neuron is None:
        return False, "torch.neuron missing (TorchNeuron not installed/loaded)"
    is_avail = getattr(neuron, "is_available", None)
    try:
        ok = bool(is_avail()) if callable(is_avail) else False
    except Exception as exc:
        return False, f"torch.neuron.is_available() raised {type(exc).__name__}: {exc}"
    return (True, "ok") if ok else (False, "torch.neuron.is_available() is False")


def _env_snapshot() -> dict[str, str]:
    """Neuron/torch-related env vars in effect (they change compile/cache behaviour)."""
    prefixes = ("NEURON", "TORCH_NEURON", "XLA", "PJRT")
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(prefixes)}


def _summarize(observations: dict[str, dict[str, Any]], width: int = 110) -> str:
    """A compact, human-readable rendering of the observations."""
    lines = []
    for name, obs in observations.items():
        flag = "ERROR" if "error" in obs else "ok"
        lines.append(f"== {name} [{flag}] ({obs.get('probe_wall_s', '?')}s)")
        if "error" in obs:
            lines.append(f"   error: {obs['error']}"[:width])
        data = obs.get("data")
        shown = data if isinstance(data, dict) else {}
        for key in ("two_rank_all_reduce", "env_var_applied", "attempted_backends"):
            if key in obs:
                shown = {**shown, key: obs[key]}
        for key, val in shown.items():
            lines.append(f"   {key}: {json.dumps(val, default=str)}"[:width])
        phases = obs.get("phases") or {}
        compiles = {
            p: c.get("Compilation Successfully Completed", 0)
            for p, c in phases.items()
            if isinstance(c, dict)
        }
        if any(compiles.values()):
            lines.append(
                f"   'Compilation Successfully Completed' per phase: {compiles}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run all probes and write the result + observations. Returns 0 if the probe completed."""
    result = capture_environment(ValidationResult(example=EXAMPLE, status="failed"))
    start = time.time()
    available, detail = neuron_status()
    if not available:
        result.error = f"neuron unavailable: {detail}"
        result.wall_clock_s = round(time.time() - start, 2)
        path = result.write(RESULTS_DIR)
        print(f"TorchNeuron facts: {result.status} -> {path.name}\n  {result.error}")
        return 1

    observations = run_probes()
    errored = sorted(n for n, o in observations.items() if "error" in o)
    # NOTE: "passed" == the probe script ran every probe to completion. It says NOTHING about
    # whether TorchNeuron behaved well; the facts are in the observations file.
    result.status = "passed"
    result.metrics = {
        "probes_run": float(len(observations)),
        "probes_with_error": float(len(errored)),
    }
    result.wall_clock_s = round(time.time() - start, 2)
    path = result.write(RESULTS_DIR)
    obs_path = path.with_suffix(".observations.json")
    payload = {
        "note": "Observations only. The result's status reflects probe completion, not a "
        "TorchNeuron pass.",
        "result_artifact": path.name,
        "instance_type": result.instance_type,
        "versions": result.versions,
        "env": _env_snapshot(),
        "probes_with_error": errored,
        "probes": observations,
    }
    obs_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    try:
        result.log_path = str(obs_path.relative_to(_REPO_ROOT))
    except ValueError:
        result.log_path = str(obs_path)
    result.write(RESULTS_DIR)  # rewrite with log_path pointing at the observations

    print("\n" + _summarize(observations))
    print(
        f"\nTorchNeuron facts: probe {result.status} (completion, not a TorchNeuron verdict)"
    )
    print(f"  result:       {path}\n  observations: {obs_path}")
    if errored:
        print(f"  probes that recorded an error: {', '.join(errored)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
