"""Scene-side API: what a SOFA ``scene.py`` imports to talk to the optimizer.

This is the *only* sofaopt import a scene needs, and it pulls in no optimizer
dependencies (no optuna) — safe to import inside a SofaPython3 process.

Typical use inside ``createScene(root)``::

    from sofaopt.scene import open_trial

    def createScene(root):
        trial = open_trial(root)          # reads params + run metadata from env
        stiffness = trial.params["stiffness"]
        mesh = trial.env.get("OPT_MESH")  # if a prepare hook produced one
        # ... build your SOFA graph ...
        root.addObject(MyController(trial=trial))   # calls trial.write_score(...)
        return root

When run *outside* the optimizer (opening the scene by hand), ``open_trial``
returns a trial with ``params == {}`` and ``is_optimizing == False`` so the
same scene file still works for interactive debugging.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sofaopt.core import envkeys


@dataclass
class Trial:
    """Everything a scene needs from the optimizer for one run."""

    params: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    trial_state_path: str | None = None
    run_slot: int = 0
    gen: int = 0
    trial: int = 0
    run: int = 0
    test_name: str = ""
    test_run_index: int = 0
    test_run_total: int = 0
    _writer: "ScoreWriter | None" = field(default=None, repr=False)

    @property
    def is_optimizing(self) -> bool:
        """True when launched by the optimizer (a score is expected)."""
        return self.trial_state_path is not None

    @property
    def run_info(self) -> dict:
        return {"gen": self.gen, "trial": self.trial, "run": self.run}

    def attach(self, rootnode) -> "Trial":
        """Bind a SOFA root so writes timestamp with sim time and can stop it."""
        self._writer = ScoreWriter(
            rootnode,
            run_info=self.run_info,
            trial_state_path=self.trial_state_path,
            run_slot=self.run_slot,
        )
        return self

    def _ensure_writer(self) -> "ScoreWriter":
        if self._writer is None:
            self._writer = ScoreWriter(
                None,
                run_info=self.run_info,
                trial_state_path=self.trial_state_path,
                run_slot=self.run_slot,
            )
        return self._writer

    def write_status(self, payload: dict, *, min_interval: float = 0.0) -> None:
        self._ensure_writer().write_status(payload, min_interval=min_interval)

    def write_score(self, score: float, reason: str = "") -> None:
        """Record this run's final score and stop the simulation."""
        self._ensure_writer().write_score_and_stop(score, reason)

    def prune(self, reason: str = "") -> None:
        """Mark this run pruned (not scored) and stop the simulation."""
        self._ensure_writer().write_pruned_and_stop(reason)

    # --- relaunchable probes (carry state across runSofa relaunches) ---------
    @property
    def trial_dir(self) -> Path | None:
        """This trial's working directory (where carry/params files live)."""
        return Path(self.trial_state_path).parent if self.trial_state_path else None

    def _carry_path(self) -> Path | None:
        d = self.trial_dir
        return (d / f"carry_run{self.run_slot}.json") if d else None

    def load_carry(self) -> dict:
        """Read state saved by a previous relaunch of this run.

        Each relaunch is a fresh process, so in-memory values are gone — carry
        anything you need to the next launch via :meth:`save_carry` and read it
        back here. Returns ``{}`` on the first launch (nothing saved yet).
        """
        p = self._carry_path()
        if p and p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def save_carry(self, data: dict) -> None:
        """Persist state for the next relaunch of this run (see :meth:`load_carry`)."""
        p = self._carry_path()
        if p is not None:
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def relaunch(self, carry: dict | None = None, status: dict | None = None) -> None:
        """End this probe iteration and ask the optimizer to relaunch the run.

        Persists ``carry`` (read it next launch with :meth:`load_carry`), flags
        the run so the optimizer relaunches it instead of treating the exit as a
        crash, and stops this process. Requires the test's ``relaunchable=True``
        and the project's ``max_run_relaunches > 0``. Call :meth:`write_score`
        instead to finish for good.
        """
        if carry is not None:
            self.save_carry(carry)
        payload = {"probe_finished": True}
        if status:
            payload.update(status)
        self.write_status(payload)
        self._ensure_writer()._stop()

    def mark_probe_finished(self) -> None:
        """Low-level: flag this run for relaunch without persisting carry/stopping.

        Prefer :meth:`relaunch`, which also saves carry state and stops the run.
        """
        self._ensure_writer().write_status({"probe_finished": True})


def open_trial(rootnode=None) -> Trial:
    """Build a :class:`Trial` from the optimizer environment + ``params.json``."""
    trial_state_path = os.environ.get(envkeys.TRIAL_STATE_PATH)

    params: dict[str, Any] = {}
    params_path = os.environ.get(envkeys.PARAMS_PATH)
    if params_path and Path(params_path).is_file():
        try:
            params = json.loads(Path(params_path).read_text(encoding="utf-8"))
        except Exception:
            params = {}

    def _int(key: str) -> int:
        try:
            return int(os.environ.get(key, "0"))
        except ValueError:
            return 0

    t = Trial(
        params=params,
        env=dict(os.environ),
        trial_state_path=trial_state_path,
        run_slot=_int(envkeys.RUN_SLOT),
        gen=_int(envkeys.GEN),
        trial=_int(envkeys.TRIAL),
        run=_int(envkeys.RUN),
        test_name=os.environ.get(envkeys.TEST_NAME, ""),
        test_run_index=_int(envkeys.TEST_RUN_INDEX),
        test_run_total=_int(envkeys.TEST_RUN_TOTAL),
    )
    if rootnode is not None:
        t.attach(rootnode)
    return t


class ScoreWriter:
    """Atomic writer for one run's slot in ``trial_state.json``.

    Generic and SOFA-aware: if given a ``rootnode`` it timestamps with sim time
    and stops the scene on the final write; without one it uses wall time.
    """

    def __init__(
        self,
        rootnode,
        run_info: dict[str, int],
        trial_state_path: str | None,
        run_slot: int,
    ) -> None:
        self.rootnode = rootnode
        self.run_info = run_info
        self.trial_state_path = trial_state_path
        self.run_slot = int(run_slot)
        self._finished = False
        self._last_status: dict[str, Any] = {}
        self._last_write_monotonic = float("-inf")

    def _now(self) -> float:
        if self.rootnode is not None:
            try:
                return float(self.rootnode.time.value)
            except Exception:
                pass
        return time.time()

    def _acquire_lock(self, lock_path: Path, timeout_s: float = 5.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return True
            except FileExistsError:
                time.sleep(0.01)
            except Exception:
                return False
        return False

    def _release_lock(self, lock_path: Path) -> None:
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass

    def _update_trial_state_run(self, payload: dict[str, Any]) -> bool:
        if self.trial_state_path is None:
            return False
        path = Path(self.trial_state_path)
        lock_path = path.with_suffix(path.suffix + ".lock")
        if not self._acquire_lock(lock_path):
            return False
        try:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            except Exception:
                data = {}

            runs = data.get("runs")
            if not isinstance(runs, list):
                runs = []
            while len(runs) < self.run_slot:
                runs.append({"run": len(runs) + 1})

            slot = runs[self.run_slot - 1]
            if not isinstance(slot, dict):
                slot = {"run": self.run_slot}
            slot.update(payload)
            slot["run"] = self.run_slot
            slot["updated_at"] = self._now()
            runs[self.run_slot - 1] = slot

            data["runs"] = runs
            data["updated_at"] = self._now()
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(path)
            return True
        finally:
            self._release_lock(lock_path)

    def write_status(self, payload: dict[str, Any], *, min_interval: float = 0.0) -> None:
        """Best-effort live-status write (errors swallowed so they never kill the sim)."""
        full = {**self.run_info, **payload, "updated_at": self._now()}
        self._last_status = dict(full)

        now = time.monotonic()
        if (
            min_interval > 0.0
            and not self._finished
            and now - self._last_write_monotonic < min_interval
        ):
            return
        self._last_write_monotonic = now
        self._update_trial_state_run(full)

    def write_score_and_stop(self, score: float, reason: str) -> None:
        """Mark the slot done with ``score`` then stop the process. Idempotent."""
        if self._finished:
            return
        self._finished = True
        final_status = dict(self._last_status)
        final_status.update({"state": "done", "score": score, "reason": reason})
        self.write_status(final_status)
        print(f"[Score] {reason} | score: {score:.4f}")
        self._stop()

    def write_pruned_and_stop(self, reason: str) -> None:
        """Mark the slot pruned (unscored) then stop the process. Idempotent."""
        if self._finished:
            return
        self._finished = True
        final_status = dict(self._last_status)
        final_status.update({"state": "pruned", "score": None, "reason": reason})
        self.write_status(final_status)
        print(f"[Pruned] {reason}")
        self._stop()

    def _stop(self) -> None:
        if self.rootnode is not None:
            try:
                self.rootnode.animate = False
            except Exception:
                pass
        # Only hard-kill under the optimizer; an interactive (hand-launched)
        # run has no trial_state_path and should just pause, not exit.
        if self.trial_state_path is not None:
            os.kill(os.getpid(), 9)

    @property
    def finished(self) -> bool:
        return self._finished


__all__ = ["Trial", "open_trial", "ScoreWriter"]
