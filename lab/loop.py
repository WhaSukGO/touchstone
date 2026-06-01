"""The deterministic harness loop (design §3 state machine, §8 sequence).

Zero LLM calls live here. Reasoning is delegated to the injected Planner / Evaluator
(Agent SDK in Stage 2; dummy in self-test). The loop owns: state transitions, the GPU
lease, dataset caching, image resolution, logging, and budget — so downloads, builds,
and training never consume the reasoning budget."""
from __future__ import annotations

from .budget import Budget
from .dataset_cache import DatasetCache
from .gpu_lease import GpuLease, hold
from .image_registry import ImageRegistry, NoImageError
from .job_runner import JobRunner, JobSpec
from .logging_policy import error_line, summarize_log
from .models import ExperimentRecord, Status
from .notebook import Notebook
from .paths import Layout
from .plugins.base import Evaluator, MetricExtractor, Planner
from .queue import Queue
from .registry import Registry
from .serde import to_jsonable
from .util import stable_hash


class Harness:
    def __init__(self, *, layout: Layout, registry: Registry, queue: Queue,
                 gpu_lease: GpuLease, image_registry: ImageRegistry,
                 dataset_cache: DatasetCache, job_runner: JobRunner, budget: Budget,
                 notebook: Notebook, planner: Planner, evaluator: Evaluator,
                 metric_extractor: MetricExtractor, job_mode: str = "local",
                 lease_timeout_s: float = 3600.0):
        self.layout = layout
        self.registry = registry
        self.queue = queue
        self.gpu_lease = gpu_lease
        self.image_registry = image_registry
        self.dataset_cache = dataset_cache
        self.job_runner = job_runner
        self.budget = budget
        self.notebook = notebook
        self.planner = planner
        self.evaluator = evaluator
        self.metric_extractor = metric_extractor
        self.job_mode = job_mode
        self.lease_timeout_s = lease_timeout_s
        self.autonomy_enabled = False

    # ---- state-machine helpers -------------------------------------------------
    def _commit(self, rec: ExperimentRecord, status: Status, event: str | None = None) -> None:
        rec.status = status
        self.registry.upsert(rec)
        if event:
            self.notebook.log_event(rec, event)

    def _fail(self, rec: ExperimentRecord, reason: str) -> ExperimentRecord:
        rec.status = Status.FAILED
        self.registry.upsert(rec)
        self.notebook.log_event(rec, reason)
        self.notebook.log_failed(rec, reason)
        return rec

    # ---- one experiment through the full state machine -------------------------
    def run_experiment(self, rec: ExperimentRecord) -> ExperimentRecord:
        self.budget.mark_started()
        try:
            # PROPOSED -> CONTRACTED  (reasoning: planner; charged in tokens)
            if rec.contract is None:
                contract, usage = self.planner.propose_contract(rec)
                rec.contract = contract
                rec.config_hash = stable_hash(to_jsonable(contract))
                rec.tokens_in += usage.tokens_in
                rec.tokens_out += usage.tokens_out
                self.budget.charge_tokens(rec.id, usage.tokens_in, usage.tokens_out)
            contract = rec.contract
            self._commit(rec, Status.CONTRACTED, "contract negotiated")

            # CONTRACTED -> ENV_READY  (image resolution; no tokens)
            image = None
            if contract.framework is not None:
                try:
                    image = self.image_registry.resolve(contract.framework).image
                except NoImageError as e:
                    if self.job_mode == "docker":
                        return self._fail(rec, error_line(str(e)))
                    # local mode needs no container image
            rec.env_image = image
            self._commit(rec, Status.ENV_READY, f"image={image}")

            # ENV_READY -> DATA_READY  (cache ensures download happens once; no tokens)
            data_dir = None
            ds_names: list[str] = []
            for ref in contract.datasets:
                entry = self.dataset_cache.ensure(ref)  # download once; cached thereafter
                ds_names.append(ref.name)
                if not ref.held_out:
                    data_dir = entry.path  # specific dataset dir (holds the actual files)
            rec.datasets = ds_names
            self._commit(rec, Status.DATA_READY, f"datasets={ds_names}")

            # DATA_READY -> RUNNING -> ARTIFACTS_READY  (job under GPU lease; IO not charged)
            workdir = self.layout.workspace(rec.id)
            artifacts = self.layout.artifacts(rec.id)
            log_path = self.layout.logs(rec.id) / "train.log"
            rec.workdir = str(workdir)
            rec.log_path = str(log_path)
            self._commit(rec, Status.RUNNING, "training")
            with hold(self.gpu_lease, f"{rec.id}:train", timeout_s=self.lease_timeout_s):
                job = self.job_runner.run(JobSpec(
                    exp_id=rec.id, command=contract.command, workdir=str(workdir),
                    artifacts_dir=str(artifacts), log_path=str(log_path),
                    data_dir=data_dir, code_dir=contract.code_dir,
                    image=image, mode=self.job_mode,
                ))
            self.budget.note_io(rec.id, job.wall_seconds)
            rec.wall_seconds += job.wall_seconds
            if not job.ok:
                return self._fail(rec, error_line(
                    f"job exit {job.exit_code}\n{summarize_log(log_path)}"))
            self._commit(rec, Status.ARTIFACTS_READY, "artifacts produced")

            # extract generator-reported metrics (deterministic, recorded but NOT trusted)
            rec.reported_metrics = self.metric_extractor.extract(str(artifacts))
            self.registry.upsert(rec)

            # ARTIFACTS_READY -> EVALUATING -> VERIFIED/REJECTED  (independent evaluator)
            self._commit(rec, Status.EVALUATING, "independent evaluation")
            with hold(self.gpu_lease, f"{rec.id}:eval", timeout_s=self.lease_timeout_s) as token:
                verdict, usage = self.evaluator.evaluate(rec, contract, str(artifacts), token)
            rec.tokens_in += usage.tokens_in
            rec.tokens_out += usage.tokens_out
            self.budget.charge_tokens(rec.id, usage.tokens_in, usage.tokens_out)
            rec.verdict = verdict
            if verdict.verdict == "PASS":
                self._commit(rec, Status.VERIFIED, "VERIFIED (evaluator-signed)")
            else:
                self._commit(rec, Status.REJECTED, "REJECTED by evaluator")
                self.notebook.log_failed(rec, f"evaluator FAIL: {verdict.oracle_comparison}")
            return rec

        except Exception as e:  # any uncaught error -> FAILED, recorded, loop survives
            return self._fail(rec, error_line(repr(e)))

    # ---- autonomy gate (design §7) ---------------------------------------------
    def calibration_gate(self, positive: ExperimentRecord,
                         negative: ExperimentRecord) -> bool:
        """Open full autonomy ONLY if the evaluator verifies a known-good run AND
        rejects a deliberately poisoned one (proving it is not a rubber stamp)."""
        pos = self.run_experiment(positive)
        neg = self.run_experiment(negative)
        ok = (pos.status == Status.VERIFIED) and (neg.status == Status.REJECTED)
        self.autonomy_enabled = ok
        self.notebook.log_event(
            pos, f"CALIBRATION {'OPEN' if ok else 'LOCKED'}: "
                 f"positive={pos.status.value} negative={neg.status.value}")
        return ok

    # ---- autonomous loop -------------------------------------------------------
    def loop(self, *, require_gate: bool = True) -> None:
        if require_gate and not self.autonomy_enabled:
            raise RuntimeError("autonomy locked: pass calibration_gate() first")
        # crash recovery: re-queue interrupted experiments (jobs are idempotent)
        for rec in self.queue.interrupted():
            self.notebook.log_event(rec, "resuming interrupted experiment")
            self.queue.requeue(rec)
        while self.budget.can_spawn():
            rec = self.queue.pop_next()
            if rec is None:
                break
            rec = self.run_experiment(rec)
            if rec.status == Status.VERIFIED and rec.verdict is not None:
                nxt, usage = self.planner.decide_next(rec.verdict, rec)
                self.budget.charge_tokens(rec.id, usage.tokens_in, usage.tokens_out)
                if nxt is not None:
                    self.queue.push(nxt)
