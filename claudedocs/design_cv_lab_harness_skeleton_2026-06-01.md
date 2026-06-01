# CV Research Lab — Harness Skeleton 설계 명세 (Stage 1)

- 날짜: 2026-06-01
- 범위: **결정적 harness 골격(Stage 1) + 독립 evaluator 경계(Stage 2 인터페이스)**. 도메인·정체성(A/B/C) 무관하게 동일한 부분만.
- 기반: Python Agent SDK, 단일 로컬 GPU(WSL2), 완전 자율(단 evaluator 독립).
- 비범위: 구현 코드(→ `/sc:implement`), 도메인별 플러그인 내용(데이터셋/oracle 실체).
- 선행: `research_cv_lab_harness_design_2026-06-01.md`

---

## 1. 설계 원칙 (재확인)

| 원칙 | 강제하는 것 |
|---|---|
| **추론 ⟂ 실행 분리** | harness는 결정적 Python(LLM 0). 다운로드·빌드·학습 대기는 harness가, 모델 호출은 추론 지점에서만 |
| **예산은 토큰+실험회수** | IO 대기는 예산에서 제외. turn quota로 IO를 벌주지 않음 |
| **상태는 영속** | 모든 상태가 registry/파일 → 컨텍스트 리셋·프로세스 재시작 생존 |
| **evaluator 물리 분리** | generator 세션과 별도 프로세스/컨텍스트. oracle·held-out은 evaluator만 |
| **출력은 서명된 결과만** | 랩 공식 출력 = evaluator가 검증한 `VerifiedResult`. generator 주장 ≠ 출력 |
| **단일 GPU = 직렬** | GPU 단일 mutex. 병렬분해 대신 순차+smoke-subset 선검증 |
| **자율은 게이트 뒤에** | calibration(재현+negative control) 통과 전 자율 루프 잠금 |

---

## 2. 시스템 아키텍처 (컴포넌트 다이어그램)

```
┌─────────────────────────────────────────────────────────────────────┐
│  HARNESS PROCESS  (deterministic Python, LLM 호출 0)                  │
│                                                                       │
│   ┌──────────┐   ┌───────────────┐   ┌────────────┐                  │
│   │ Queue    │──▶│  Outer Loop    │──▶│  Budget    │                  │
│   │(priority)│   │ (state machine)│   │  Accountant│                  │
│   └──────────┘   └───────┬────────┘   └────────────┘                  │
│                          │                                            │
│        ┌─────────────────┼──────────────────┬───────────────┐        │
│        ▼                 ▼                  ▼               ▼        │
│  ┌───────────┐    ┌────────────┐    ┌────────────┐  ┌────────────┐   │
│  │ Image     │    │ Dataset    │    │ GPU Lease  │  │ Job Runner │   │
│  │ Registry  │    │ Cache      │    │ (mutex=1)  │  │(container) │   │
│  └───────────┘    └────────────┘    └────────────┘  └─────┬──────┘   │
│                                                            │          │
│   ┌──────────────────────────────────────────────────┐    │          │
│   │ Experiment Registry (SQLite) + Logs(files)        │◀───┘          │
│   │ + Lab Notebook(md) + Failed-approaches log(md)    │               │
│   └──────────────────────────────────────────────────┘               │
└───────────┬───────────────────────────────────┬─────────────────────┘
            │ reasoning calls only               │ reasoning calls only
            ▼                                     ▼
┌───────────────────────┐            ┌──────────────────────────────────┐
│ AGENT SESSIONS         │            │ INDEPENDENT EVALUATOR             │
│ (Agent SDK)            │            │ (Agent SDK, SEPARATE context)     │
│  • Planner / PI        │            │  • held-out split 접근 (전용)      │
│  • Domain experts      │            │  • Oracle 접근 (전용)              │
│  • (generator)         │            │  • 직접 재현 → 채점 → 서명          │
│  oracle 접근 ✗          │            │  generator 히스토리 접근 ✗         │
└───────────────────────┘            └──────────────────────────────────┘
            │                                     │
            └──────────── share ──────────────────┘
                 GPU Lease (둘 다 mutex 획득)
```

**핵심 토폴로지 규칙**
- Harness는 단일 장수 프로세스. agent/evaluator는 harness가 *호출*하는 stateless 세션.
- Generator 세션과 Evaluator 세션은 **컨텍스트를 공유하지 않는다**. harness가 artifact 파일만 evaluator에 전달.
- GPU는 generator-run과 evaluator-run이 **공유하는 단일 mutex**(둘 다 GPU 필요).

---

## 3. 실험 생애주기 (상태 기계)

```
 PROPOSED ──▶ CONTRACTED ──▶ ENV_READY ──▶ DATA_READY ──▶ RUNNING
   (planner)   (contract       (image         (cache         (job
                negotiated)     resolved)      ensured)       runner)
                                                                │
                                                                ▼
   recorded ◀── VERIFIED/REJECTED ◀── EVALUATING ◀── ARTIFACTS_READY
   (registry)    (signed verdict)     (independent    (metrics
        │                              evaluator)      extracted)
        ▼
   planner.decide_next() ──▶ 새 PROPOSED (lineage: parent_id)

 실패 전이 (어느 단계든):
   * → FAILED(reason) ──▶ failed_approaches.md append ──▶ retry? (budget 내) or skip
```

각 전이는 **registry에 commit** → 프로세스가 죽어도 마지막 상태에서 재개. 전이 사이에 모델 호출이 *없는* 구간(ENV/DATA/RUNNING)은 토큰 0.

---

## 4. 컴포넌트 명세 (Stage 1)

### 4.1 Experiment Queue
- **책임**: 다음 실험 선택. FIFO + priority(우선순위: calibration > 사용자지정 > planner제안).
- **저장**: registry의 `status=PROPOSED|CONTRACTED` 행을 쿼리(별도 큐 파일 불필요, registry가 단일 진실원).
- **인터페이스**
```python
class Queue:
    def push(self, exp: ExperimentRecord) -> None: ...
    def pop_next(self) -> ExperimentRecord | None: ...   # priority 순, 없으면 None
    def requeue(self, exp_id: str, *, priority: int) -> None: ...
```

### 4.2 GPU Lease (단일 GPU mutex)
- **책임**: 동시 GPU 사용 금지. generator-run과 evaluator-run 모두 획득 후 사용.
- **구현 방향**: 파일락(`fcntl.flock` on `state/gpu.lock`) 또는 registry의 단일 lease 행 + heartbeat. 크래시 시 stale lease 회수(heartbeat timeout).
- **인터페이스**
```python
class GpuLease:
    def acquire(self, holder: str, timeout_s: float) -> LeaseToken: ...  # 블로킹
    def release(self, token: LeaseToken) -> None: ...
    def reap_stale(self, ttl_s: float) -> None: ...   # 죽은 holder 회수
```

### 4.3 CUDA Image Registry
- **책임**: framework 선언 → 사전빌드 이미지로 결정적 매핑. *런타임 이미지 빌드 금지*(빌드는 별도 의식적 단계).
- **저장**: `images/registry.yaml` — (framework, version, cuda) → image digest + healthcheck cmd.
- **예시 매트릭스**
```yaml
images:
  - key: torch-2.4-cu121
    image: "lab/torch:2.4.0-cu121"      # 사전 빌드, digest 고정
    cuda: "12.1"
    healthcheck: "python -c 'import torch; assert torch.cuda.is_available()'"
  - key: torch-2.2-cu118
    image: "lab/torch:2.2.0-cu118"
    cuda: "11.8"
    healthcheck: "python -c 'import torch; assert torch.cuda.is_available()'"
```
- **인터페이스**
```python
class ImageRegistry:
    def resolve(self, framework: FrameworkSpec) -> ResolvedImage: ...   # 없으면 명시적 에러(자동빌드 X)
    def healthcheck(self, image: ResolvedImage) -> bool: ...
    def list_matrix(self) -> list[ResolvedImage]: ...
```
- **규칙**: resolve 실패 = 실험 FAILED(`ERROR: no prebuilt image for <framework>`). 자동 빌드로 폭주 금지. 이미지 추가는 사람이 승인하는 별도 의식.

### 4.4 Dataset / Weights Cache
- **책임**: 데이터셋·체크포인트 *일생 1회* 확보. 다운로드는 harness 잡(토큰 0). 내용 해시로 검증.
- **저장**: `cache/datasets/<name>/`, `cache/weights/<name>/` + `cache/manifest.json`(name → path, sha256, size, source).
- **인터페이스**
```python
class DatasetCache:
    def ensure(self, ref: DatasetRef) -> CachedPath: ...   # 있으면 즉시 반환, 없으면 1회 다운로드+검증
    def verify(self, ref: DatasetRef) -> bool: ...         # sha256 대조
    def manifest(self) -> dict[str, CacheEntry]: ...
```
- **규칙**: 컨테이너는 cache를 **read-only 마운트**. 실험이 cache를 오염시키지 못함.

### 4.5 Job Runner (컨테이너 실행)
- **책임**: 해석된 이미지 + 캐시 + 워크스페이스 + GPU로 컨테이너 실행, **블로킹 대기**, 로그를 파일로. 에이전트는 babysitting 안 함.
- **실행 형태**(개념):
```
docker run --rm --gpus all \
  -v cache/datasets:/data:ro \
  -v cache/weights:/weights:ro \
  -v workspaces/<exp_id>:/workspace \
  -v logs/<exp_id>:/logs \
  <resolved_image> <contract.command>
```
- **인터페이스**
```python
class JobRunner:
    def run(self, exp: ExperimentRecord, lease: LeaseToken) -> JobResult: ...
    # JobResult = {exit_code, log_path, wall_seconds, artifacts_dir}
```
- **resumability**: workdir = `workspaces/<exp_id>`(config_hash로 키잉). 체크포인트는 `cache/weights/<exp_id>/`에 저장 → 크래시 후 재실행이 이어받음. 잡은 **idempotent**해야(같은 config → 같은 결과·이어쓰기).
- **로그 정책**: 전체 로그는 `logs/<exp_id>/{download,build,train,eval}.log`. registry엔 *tail 요약 + 표준 에러줄 + 집계 metric*만 기록. 에이전트엔 stdout 떡칠 금지.

### 4.6 Budget Accountant
- **책임**: 예산을 **토큰 + 실험회수**로 추적. IO/wall은 관측용으로만 분리 기록.
- **단위**: per-experiment(max_tokens, max_retries, max_wall_s) + global(total_tokens, total_experiments).
- **인터페이스**
```python
class Budget:
    def charge_tokens(self, exp_id: str, in_tok: int, out_tok: int) -> None: ...
    def can_spawn(self) -> bool: ...                  # global 한도 내인가
    def remaining_tokens(self) -> int: ...
    def note_io(self, exp_id: str, wall_s: float) -> None: ...   # 예산에 영향 X, 관측만
```
- **규칙**: 다운로드/빌드/학습 대기 = `note_io`만. 토큰 청구는 agent 세션 호출에서만.

### 4.7 Logging / Notebook / Failed-approaches
- `state/lab_notebook.md`: 진행 상황 자동 갱신(실험 결과 요약, 결정).
- `state/failed_approaches.md`: 실패 전이마다 append(`<hypothesis> | <reason> | <exp_id>`) → planner가 읽어 순환 방지.
- 표준 에러 포맷: `ERROR: <one-line reason>`.

---

## 5. 데이터 모델 (스키마)

### 5.1 ExperimentRecord (registry, SQLite 권장)
```python
@dataclass
class ExperimentRecord:
    id: str                      # slug 예: "repro-resnet50-001"
    parent_id: str | None        # lineage
    status: Status               # PROPOSED..VERIFIED/REJECTED/FAILED
    hypothesis: str
    contract: ExperimentContract # §5.2
    config_hash: str             # 재현·workdir 키
    env_image: str | None        # resolved digest
    datasets: list[str]
    workdir: str | None
    log_path: str | None
    reported_metrics: dict       # generator 주장 (신뢰 X, 기록만)
    verdict: VerifiedResult | None  # §5.3 (evaluator 서명, 유일한 진실)
    tokens_in: int = 0
    tokens_out: int = 0
    wall_seconds: float = 0.0
    retries: int = 0
    created_at: str = ""         # harness가 외부 시계로 스탬프
    updated_at: str = ""
```

### 5.2 ExperimentContract (실행 전 합의 = sprint contract)
```python
@dataclass
class ExperimentContract:
    success_definition: str          # 사람이 읽을 수 있는 "done의 정의"
    gradable_criteria: list[Criterion]  # 채점가능 항목 (metric, 비교연산, 임계값)
    framework: FrameworkSpec         # → ImageRegistry.resolve
    datasets: list[DatasetRef]
    command: str                     # 컨테이너 진입 명령
    budget: BudgetSpec               # max_tokens / max_wall_s / max_retries
    oracle: OracleRef | None         # 재현 시: expected metric + tolerance
    # 예: Criterion(metric="top1_acc", op=">=", value=0.76, tolerance=0.005)
```

### 5.3 VerifiedResult (랩의 유일한 공식 출력)
```python
@dataclass
class VerifiedResult:
    experiment_id: str
    verdict: Literal["PASS", "FAIL"]
    measured_metrics: dict           # evaluator가 직접 측정 (NOT generator 주장)
    oracle_comparison: dict | None   # {expected, measured, within_tolerance}
    artifacts: list[ArtifactRef]     # 경로 + sha256
    provenance: Provenance           # config, image digest, dataset hashes, seed, git commit
    evaluator_notes: str
    signed_by: str                   # evaluator 세션 id
    signed_at: str                   # 외부 시계 스탬프
```
> 이 구조가 **랩 간 협업의 신뢰 기반**: 2번째 랩은 `VerifiedResult`(서명·provenance 포함)만 신뢰해 입력으로 받는다.

---

## 6. 독립 Evaluator 경계 (Stage 2 인터페이스)

- **격리**: harness가 별도 Agent SDK 세션으로 호출. generator 대화 히스토리 **미전달**. 입력 = {contract, artifacts_dir, oracle 핸들(전용)}.
- **능동 검증**: 수동 리뷰 금지. evaluator는 GPU lease를 잡고 **held-out split에서 직접 추론/재현** 후 metric 측정.
- **회의적 튜닝**: few-shot으로 "좋은/나쁜 결과 점수 분해" 주입. 기본값은 의심.
- **인터페이스**
```python
class IndependentEvaluator:
    def evaluate(self, contract: ExperimentContract,
                 artifacts_dir: str, lease: LeaseToken) -> VerifiedResult: ...
```
- **leakage 가드**: held-out split은 evaluator만 마운트. generator 컨테이너엔 train split만.

---

## 7. 자율성 게이트 (Calibration)

자율 루프(`autonomy_enabled`)는 기본 **잠금**. 다음 2조건 통과 시에만 개방:

```
 [Positive control]  알려진 재현 1건
   → evaluator 측정치가 oracle을 tolerance 내 재현?  ── 실패시 GATE 잠금 유지
 [Negative control]  의도적으로 손상시킨 run(poisoned)
   → evaluator가 이를 FAIL로 거부?                   ── 통과 못하면(=rubber-stamp) GATE 잠금
 ─────────────────────────────────────────────────────
 둘 다 통과 ⇒ autonomy_enabled = True  ⇒ 완전 자율 개방
```
> negative control이 핵심: evaluator가 그냥 도장 찍는 기계가 아님을 증명해야 자율을 신뢰할 수 있다. 이게 당신의 지난 실패("실행=성공")를 구조적으로 차단.

---

## 8. 시퀀스: 실험 1사이클

```
Harness        Planner      ImageReg/Cache    JobRunner(GPU)    Evaluator(GPU)
  │  pop_next      │               │                │                 │
  ├───────────────▶│ propose/contract                                  │
  │◀── contract ───┤  (tokens 청구)                                     │
  ├── resolve ─────────────────▶│ image+data ready (tokens 0)           │
  │◀── ready ───────────────────┤                                       │
  ├── run (lease) ──────────────────────────▶│ 학습/추론 (tokens 0)      │
  │◀── JobResult(log file) ───────────────────┤  release lease          │
  ├── extract metrics (결정적, tokens 0)        │                        │
  ├── evaluate (lease) ──────────────────────────────────────────────▶ │ held-out 직접재현
  │◀── VerifiedResult (서명) ───────────────────────────────────────────┤  release lease
  ├── registry.commit + notebook.update                                  │
  ├── decide_next ─▶ Planner (tokens 청구) ─▶ 새 PROPOSED                  │
  └── (loop)
```
토큰 청구 지점: contract 협상, decide_next, evaluate(추론 텍스트). 학습·다운로드·metric추출 = 0.

---

## 9. 디렉터리 레이아웃

```
lab/
  harness/            # 결정적 Python (LLM 0)
    loop.py  queue.py  gpu_lease.py  image_registry.py
    dataset_cache.py  job_runner.py  budget.py  registry.py  logging_policy.py
  agents/             # Agent SDK 세션
    planner.py  experts/  evaluator.py     # evaluator 별도 컨텍스트
  plugins/            # 도메인 의존 (추후 확정, 인터페이스 뒤)
    datasets/  oracles/  metrics/
  images/registry.yaml
  state/
    registry.db  lab_notebook.md  failed_approaches.md  gpu.lock
  cache/              # 영속, gitignore
    datasets/  weights/  manifest.json
  logs/<exp_id>/
  workspaces/<exp_id>/
```

---

## 10. 플러그인 인터페이스 (도메인 미확정 흡수)

도메인(분류/검출/...)이 정해지면 이 4개만 구현하면 됨:
```python
class DatasetProvider(Protocol):
    def fetch(self, ref: DatasetRef) -> CachedPath: ...   # cache가 호출
class Oracle(Protocol):
    def expected(self, exp: ExperimentRecord) -> Criterion: ...  # 재현 기준
class MetricExtractor(Protocol):
    def extract(self, artifacts_dir: str) -> dict: ...    # 로그/체크포인트→metric (결정적)
class HeldoutSplit(Protocol):
    def mount_path(self) -> str: ...                      # evaluator 전용
```

---

## 11. 설계 검증 (요구사항 ↔ 설계)

| 지난 실패 / 요구 | 이 설계의 대응 |
|---|---|
| 다운로드 턴 낭비 | §4.4 cache(1회) + §4.6 IO는 토큰 0 + §4.5 harness 블로킹 |
| CUDA 빌드 폭주 | §4.3 사전빌드 이미지 매트릭스, 런타임 빌드 금지 |
| "실행=성공" | §5.2 gradable contract + §6 능동 evaluator + §7 negative control |
| 자기평가 편향 | §2 토폴로지 분리 + §6 별도 컨텍스트 |
| 컨텍스트 리셋 | §3 상태기계 commit + §5 registry 영속 |
| 순환 | §4.7 failed_approaches.md |
| 완전 자율 위험 | §7 calibration 게이트 |
| 랩 협업 확장 | §5.3 서명된 VerifiedResult + provenance |

---

## 12. 미결 (구현 전 확정 필요)
1. **영점용 재현 대상 1건** (예: ResNet-50/ImageNet top-1≈0.76). calibration oracle.
2. Registry 저장소: SQLite(권장) vs JSONL.
3. GPU lease 방식: flock vs registry-row+heartbeat.
4. Agent SDK 세션 격리 방식: 별도 process vs 별도 `query()` client(컨텍스트 비공유 보장 방법).

> **다음 단계**: 이 설계 승인 후 `/sc:implement`로 §4 harness 골격(LLM 0)부터 구현. 1번(재현 대상)만 정하면 착수 가능.
