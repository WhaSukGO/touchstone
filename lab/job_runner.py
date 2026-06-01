"""Job runner (design §4.5).

Runs a command to completion (blocking) and streams output to a log FILE — the agent
never babysits it and never sees the raw stream. Two backends:
  - "docker": container with read-only cache mounts + --gpus all (real experiments)
  - "local":  run directly in the workspace (CI / self-test without Docker or GPU)

Paths are exposed to the command via env vars (LAB_DATA, LAB_WEIGHTS, LAB_ARTIFACTS,
LAB_LOGS, LAB_EVAL_OUT) so the same command string works in both backends."""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .util import ensure_dir


@dataclass
class JobSpec:
    exp_id: str
    command: str
    workdir: str
    artifacts_dir: str
    log_path: str
    data_dir: str | None = None
    weights_dir: str | None = None
    eval_out_dir: str | None = None
    code_dir: str | None = None  # reference/experiment code, mounted ro at /code
    image: str | None = None
    mode: str = "local"          # "local" | "docker"
    network: str | None = None   # e.g. "none" to disable networking (sandboxed authoring)
    env: dict = field(default_factory=dict)


@dataclass
class JobResult:
    exit_code: int
    log_path: str
    wall_seconds: float
    artifacts_dir: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class JobRunner:
    def __init__(self, *, default_mode: str = "local"):
        self.default_mode = default_mode

    def run(self, spec: JobSpec) -> JobResult:
        # Paths must be absolute: the job runs with cwd=workdir, so relative env paths
        # (LAB_ARTIFACTS, ...) would re-resolve against it and double-nest.
        spec.workdir = os.path.abspath(spec.workdir)
        spec.artifacts_dir = os.path.abspath(spec.artifacts_dir)
        spec.log_path = os.path.abspath(spec.log_path)
        if spec.data_dir:
            spec.data_dir = os.path.abspath(spec.data_dir)
        if spec.weights_dir:
            spec.weights_dir = os.path.abspath(spec.weights_dir)
        if spec.eval_out_dir:
            spec.eval_out_dir = os.path.abspath(spec.eval_out_dir)
        if spec.code_dir:
            spec.code_dir = os.path.abspath(spec.code_dir)

        ensure_dir(spec.workdir)
        ensure_dir(spec.artifacts_dir)
        ensure_dir(Path(spec.log_path).parent)
        if spec.eval_out_dir:
            ensure_dir(spec.eval_out_dir)

        mode = spec.mode or self.default_mode
        if mode == "docker":
            argv = self._docker_argv(spec)
            env = os.environ.copy()
        else:
            argv = ["bash", "-c", spec.command]
            env = self._local_env(spec)

        start = time.monotonic()
        with open(spec.log_path, "w") as log:
            log.write(f"$ mode={mode} exp={spec.exp_id}\n$ {spec.command}\n\n")
            log.flush()
            proc = subprocess.run(
                argv, cwd=spec.workdir, env=env,
                stdout=log, stderr=subprocess.STDOUT, text=True,
            )
        wall = time.monotonic() - start
        return JobResult(exit_code=proc.returncode, log_path=spec.log_path,
                         wall_seconds=wall, artifacts_dir=spec.artifacts_dir)

    def _local_env(self, spec: JobSpec) -> dict:
        env = os.environ.copy()
        env.update(spec.env)
        env["LAB_ARTIFACTS"] = spec.artifacts_dir
        env["LAB_LOGS"] = str(Path(spec.log_path).parent)
        if spec.data_dir:
            env["LAB_DATA"] = spec.data_dir
        if spec.weights_dir:
            env["LAB_WEIGHTS"] = spec.weights_dir
        if spec.eval_out_dir:
            env["LAB_EVAL_OUT"] = spec.eval_out_dir
        if spec.code_dir:
            env["LAB_CODE"] = spec.code_dir
        return env

    def _docker_argv(self, spec: JobSpec) -> list[str]:
        if not spec.image:
            raise ValueError("docker mode requires a resolved image")
        # Run as the host user so artifacts are host-owned (not root). HOME=/tmp keeps any
        # framework cache writes off a non-writable home.
        argv = ["docker", "run", "--rm", "--gpus", "all",
                "--user", f"{os.getuid()}:{os.getgid()}"]
        if spec.network:
            argv += ["--network", spec.network]
        mounts = {
            "/workspace": spec.workdir,
            "/artifacts": spec.artifacts_dir,
        }
        envs = {
            "LAB_ARTIFACTS": "/artifacts",
            "LAB_LOGS": "/logs",
            "HOME": "/tmp",
        }
        if spec.data_dir:
            mounts["/data:ro"] = spec.data_dir
            envs["LAB_DATA"] = "/data"
        if spec.weights_dir:
            mounts["/weights:ro"] = spec.weights_dir
            envs["LAB_WEIGHTS"] = "/weights"
        if spec.eval_out_dir:
            mounts["/eval_out"] = spec.eval_out_dir
            envs["LAB_EVAL_OUT"] = "/eval_out"
        if spec.code_dir:
            mounts["/code:ro"] = spec.code_dir
            envs["LAB_CODE"] = "/code"
        for container_path, host_path in mounts.items():
            cp, _, ro = container_path.partition(":")
            suffix = ":ro" if ro == "ro" else ""
            argv += ["-v", f"{os.path.abspath(host_path)}:{cp}{suffix}"]
        for k, v in {**envs, **spec.env}.items():
            argv += ["-e", f"{k}={v}"]
        argv += ["-w", "/workspace", spec.image, "bash", "-c", spec.command]
        return argv
