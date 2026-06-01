# Computer Vision Research Lab — Harness 설계 리서치 리포트

- 날짜: 2026-06-01
- 질문: Anthropic Agent SDK / Harness engineering으로 "CV 리서치 랩"(전문가 팀이 회의·조사·구현·결과도출)을 만들되, 이전 ralph-loop 시도의 실패(턴 낭비, 리소스 관리, "그냥 돌아가기만 함")를 반복하지 않으려면 어떻게 설계해야 하는가.
- 1차 출처: Anthropic Engineering — *Harness design for long-running agentic apps*, *Building a C compiler*.
- 신뢰도: 두 출처는 직접 인용 기반(High). CV 도메인 매핑은 설계 추론(Medium, 검증 필요).

---

## 0. Executive Summary

이전 시도가 "돌아가는 듯했으나 리서치 팀 잣대로는 실패"한 근본 원인은 **세 가지 층위의 혼동**이다:

1. **Turn 경제 혼동** — 다운로드/빌드/학습 같은 I/O·시간 바운드 작업을 *추론 턴* 안에서 처리했다. 두 블로그의 핵심 교훈은 "기계적·결정적 작업은 에이전트 루프 밖(harness)에서 처리하고, 모델 턴은 *추론할 때만* 소비하라"이다.
2. **Verifier 빈약** — "오픈소스가 돌아가는가"를 성공 기준으로 삼았다. C 컴파일러 블로그의 핵심 경고: *"task verifier가 거의 완벽하지 않으면 Claude는 엉뚱한 문제를 푼다."* 리서치 랩에서 "실행 성공"은 verifier가 아니라 전제조건일 뿐이다.
3. **자기평가 편향** — 한 세션/한 컨텍스트 안에서 평가하면 모델은 자기 결과를 "confident praising"한다. 독립 evaluator가 필요하다는 당신의 직관은 정확하며, 두 블로그 모두 이를 가장 강력한 레버로 지목한다.

결론: 랩의 성패는 **(a) 결정적 harness가 IO/리소스/스케줄을 떠맡고, (b) 회의적·독립적 evaluator가 oracle 기반으로 연구 품질을 채점하고, (c) 모든 상태가 파일/레지스트리로 영속화되어 컨텍스트 리셋을 견디는** 세 축에 달려 있다.

---

## 1. 두 블로그에서 추출한 1차 교훈 (인용 기반)

### 1.1 Harness design (long-running apps)

- **Generator ≠ Evaluator (가장 중요).**
  > "When asked to evaluate work they've produced, agents tend to respond by confidently praising the work—even when the quality is obviously mediocre."
  > "separating the agent doing the work from the agent judging it proves to be a strong lever."
  - Evaluator는 **회의적으로 튜닝**하는 게 generator를 자기비판적으로 만드는 것보다 훨씬 tractable.
  - Evaluator는 **수동 리뷰가 아니라 직접 실행**한다(Playwright로 앱을 사람처럼 클릭). → CV 매핑: evaluator가 직접 추론을 돌리고 metric을 재현한다.
  - 평가는 **구체적·채점가능 기준 + few-shot 점수 분해**에 grounding.

- **Context 관리.** 두 실패 모드: *context anxiety*(한계가 가깝다고 느끼면 조기 마무리), *context degradation*(히스토리가 차면 성능 저하). 해법: **context reset + 충분한 handoff artifact**가 compaction보다 효과적이었다(Sonnet 4.5 기준). 핸드오프 아티팩트가 다음 에이전트가 깔끔히 이어받을 만큼의 state를 담아야 함.

- **Sprint contract.** generator와 evaluator가 코드 작성 *전에* "done의 정의"를 합의. → CV 매핑: 실험 시작 전 "성공 = 무엇"을 명문화한 **experiment contract**.

- **Over-specification 회피.** 틀린 상세 스펙은 구현 오류로 연쇄. "deliverable을 명시하고 구현 경로는 에이전트가 정하게."

- **모든 harness 컴포넌트는 '모델이 못하는 것'에 대한 가정을 인코딩**한다. 모델이 좋아지면 가정은 빠르게 낡는다 → "가장 단순한 해법에서 시작, 필요할 때만 복잡도 추가."

- **비용 현실.** solo 1-agent 20분/$9 vs full harness 6시간/$200 (20배+). 품질 차이는 즉시 보였다. 품질을 위해 비용을 의도적으로 교환.

### 1.2 Building a C compiler (ralph loop, ~2,000 세션, $20k)

- **Verifier가 거의 완벽해야 한다.**
  > "the task verifier is nearly perfect, otherwise Claude will solve the wrong problem."

- **Oracle 기반 분해.** 막힌 모놀리식 태스크(Linux 커널 컴파일)를 GCC를 oracle로 써서 분해: GCC로 대부분 컴파일, Claude 컴파일러로 일부만 → 실패 파일 특정 → 병렬화. → CV 매핑: 논문 보고치 / reference 구현 / held-out set이 oracle.

- **Context 오염 방지(시간/출력 블라인드).**
  > "The test harness should not print thousands of useless bytes."
  - stdout 떡칠 대신 **파일 로깅**(필요 시 retrieve), 표준화된 에러 포맷("ERROR: [reason on same line]"), **집계 통계 사전계산**, README를 자주 갱신.
  - `--fast`: 결정적 랜덤 샘플링(1–10%)으로 커버리지 유지하며 컨텍스트 낭비 차단. → CV 매핑: smoke subset으로 빠른 검증.

- **상태 영속화 / 순환 방지.** git 히스토리 + lock 파일(`current_tasks/`)로 중복 작업 방지, README 공유 컨텍스트, **"실패한 접근들의 running doc"** 유지로 같은 버그 재방문 차단. CI로 regression 조기 포착.

- **병렬화는 분해에서 나온다.** "서로 다른 실패 테스트를 각 에이전트가 잡으면 병렬화가 trivial." 전문화: 중복코드 정리/성능/코드젠/문서/설계비평 에이전트 분리.

- **컨테이너 격리.** 세션마다 fresh Docker 컨테이너 + `/workspace` git clone, 작업 후 upstream push.

---

## 2. 당신의 실패를 두 블로그로 진단

| 당신이 겪은 문제 | 블로그가 말하는 진짜 원인 | 처방 |
|---|---|---|
| 다운로드가 많아 turn quota(20) 소진 | I/O를 추론 턴에서 처리. "Claude를 한 번 호출할 때마다 턴 소비"는 IO 대기와 추론을 동일시한 것 | IO/빌드/학습을 **harness 잡**으로 빼고, 영속 캐시로 다운로드를 일생에 1회로. 예산을 turn이 아니라 **토큰/추론-스텝**으로 측정 |
| Docker/GPU/CUDA 빌드에 리소스·시간 과다 | 매번 이미지 빌드 = 막대한 시간/턴. 모놀리식 환경구축 | **사전 빌드된 base 이미지 매트릭스(레지스트리)**에서 선택. GPU는 **리스/스케줄링되는 공유 자원** |
| "오픈소스가 돌아가면 만족"이 됨 | Verifier가 "실행 성공"에 머묾 → 모델이 엉뚱한 문제를 풂 | "실행"은 전제조건. **연구품질 rubric**(재현·신규성·ablation·통계적 유의·baseline) + **독립 evaluator + oracle** |
| 한 세션 내 자기평가 | confident praising 편향 | generator와 분리된 **회의적 evaluator**, held-out에서 직접 재현 |

---

## 3. 제안 아키텍처

### 3.1 두 개의 루프 (결정적 harness ⟂ 모델 추론)

```
[Outer harness loop — 결정적 코드, 턴 소비 0]
  while not done:
    exp = queue.pop_next_experiment()           # 결정적
    env  = image_registry.resolve(exp.framework) # 캐시된 CUDA 이미지 선택
    data = dataset_cache.ensure(exp.datasets)    # 1회 다운로드, 이후 재사용
    job  = scheduler.lease_gpu_and_run(env,data,exp.cmd)  # 비동기/블로킹, harness가 대기
    log_to_file(job)                             # stdout 떡칠 금지
    verdict = independent_evaluator(exp, job.artifacts)   # 별도 컨텍스트
    registry.record(exp, job, verdict)
    queue.update(planner.decide_next(verdict))   # 여기서만 모델 추론
```

핵심: **다운로드·환경빌드·학습 대기는 harness가 수행하고 모델 턴을 쓰지 않는다.** 모델은 (i) 가설/실험설계, (ii) 결과해석/다음수, (iii) 코드작성, (iv) 독립평가 — *추론이 필요한 지점에서만* 호출된다.

### 3.2 전문가 팀(서브에이전트) 역할

- **PI / Planner** — 연구방향·가설·분해. brief를 실험 contract로 확장. 고수준 설계에 머묾(over-spec 회피).
- **Data expert** — 데이터셋 확보/전처리/leakage 점검(캐시·레지스트리 경유).
- **Modeling expert** — 아키텍처/손실/학습전략 구현.
- **Infra/Env engineer** — CUDA 이미지 매트릭스, GPU 리스, 컨테이너 격리, 재현성.
- **Independent Evaluator (별도 컨텍스트, 회의적)** — held-out에서 *직접 재현*, oracle(논문치/reference)과 대조, rubric 채점, 구체적 결함 리포트. generator의 metric 주장 불신.
- **Reproducibility/Oracle agent** — 보고된 결과를 독립 재실행, seed/leakage/분산 점검.

"회의"는 채팅이 아니라 **파일 기반 아티팩트 협상**: 가설 → experiment contract("성공=무엇")를 *실행 전에* 합의 → 결과 → evaluator verdict.

### 3.3 영속 상태 (컨텍스트 리셋 생존)

- **Experiment Registry**(파일/DB): {hypothesis, config, env_image, datasets, cmd, status, metrics, artifacts, evaluator_verdict, parent_exp}. 모든 세션이 여기서 읽고 씀.
- **Lab Notebook / README** 지속 갱신(C 컴파일러 패턴).
- **Failed-approaches log** — 순환 방지.
- **Dataset/Weights cache** — 볼륨 마운트, 전 세션 공유, 절대 재다운로드 금지.
- **CUDA Image Registry** — (framework × CUDA × 용도) 매트릭스, healthcheck.
- 로그는 파일로, 에러는 표준 포맷, 집계는 사전계산, 모델은 필요 시 retrieve.

### 3.4 리서치용 Verifier/Evaluator 설계 (실패의 핵심 수리)

성공 기준을 **채점가능 rubric**으로:
- **재현성**: 보고치를 tolerance 내 재현했는가(held-out, 독립 실행)?
- **Baseline**: 정당한 baseline 대비 향상인가(동일 조건)?
- **신규성/기여**: 가설이 ablation으로 뒷받침되는가?
- **통계적 유의성**: seed 분산, 신뢰구간.
- **Leakage/gaming 방지**: generator가 본 적 없는 split에서 evaluator가 직접 평가.
- few-shot으로 "좋은/나쁜 결과" 점수 분해를 evaluator에 주입.

### 3.5 Turn/Budget 재정의

- 예산 단위를 **turn**이 아니라 **(출력)토큰 + 실험 회수**로. IO 대기는 예산에서 분리.
- `--fast` 등가물: smoke subset(1–10%)으로 환경/파이프라인 선검증 후 full run.
- GPU는 큐/리스로 경합 제어; 실험은 idempotent·resumable(크래시/리셋 후 재개) — *durable execution이 정당하게 필요한 유일한 곳*.

---

## 4. 안티패턴 체크리스트 (재발 방지)

- [ ] 다운로드/빌드/학습을 모델 턴 안에서 babysitting하지 않는다.
- [ ] 같은 데이터셋/이미지를 두 번 받지 않는다(캐시·레지스트리 강제).
- [ ] "실행 성공"을 성공으로 기록하지 않는다(품질 rubric 통과만 성공).
- [ ] generator가 자기 결과를 채점하지 않는다.
- [ ] evaluator는 수동 리뷰가 아니라 직접 재현한다.
- [ ] stdout에 수천 바이트를 쏟지 않는다(파일 로깅).
- [ ] 실패한 접근을 기록해 재방문을 막는다.
- [ ] harness 컴포넌트를 새 모델마다 재검토(불필요한 복잡도 제거).

---

## 5. 미해결 결정 사항 (사용자 확인 필요)

아래 5개가 설계를 실질적으로 바꾸므로 별도 질문으로 확인한다:
1. 연구 성격: **재현(reproduction)** vs **신규(novel/SOTA)** — evaluator/oracle 난이도가 근본적으로 다름.
2. 컴퓨트: 단일 로컬 GPU(WSL2 박스) vs 멀티 GPU/클라우드 — 스케줄러·병렬성 설계 좌우.
3. 자율성: 완전 자율 ralph loop vs 가설/평가 지점 human checkpoint.
4. 기반: Claude **Agent SDK(Py/TS) 직접** vs Claude Code 서브에이전트 — 예산/턴 측정 방식 포함.
5. CV 하위도메인: 분류/검출/분할/생성(diffusion)/비디오/3D 중 어디 — 데이터·oracle·인프라가 크게 다름.

---

## 6. 확정 결정 → 맞춤 설계 (사용자 답변 반영)

확정된 4가지:
1. **범위**: 하나의 랩을 먼저 완성. 이후 다른 랩을 추가해 **랩 간 협업(multi-lab collaborate)**으로 확장.
2. **컴퓨트**: 단일 로컬 GPU (이 WSL2 박스).
3. **자율성**: 완전 자율 루프, 단 **evaluator는 독립**.
4. **기반**: Python Agent SDK. CV 하위도메인은 추후 확정.

### 6.1 이 결정들이 강제하는 설계 원칙

- **완전 자율 + 단일 GPU + verifier 의존**은 *존재론적 결합*이다. 자율 루프에서 verifier가 약하면(예: "실행=성공") 모델은 무한히 엉뚱한 문제를 풀고 단일 GPU를 통째로 낭비한다. → **착수 단계는 reproduction으로 시작**해 verifier/evaluator를 *알려진 정답(oracle)*에 대해 보정한 뒤 신규 연구로 확장하라. 자율성을 켜기 전에 evaluator가 oracle을 정확히 재현하는지부터 검증.
- **단일 GPU = 직렬 실행**: 병렬 분해보다 **큐 + GPU 단일 리스 락**으로 충분. 동시성 대신 *결정적 순차 + smoke-subset 선검증*으로 낭비 차단. 실험은 idempotent·resumable(크래시/리셋 재개).
- **독립 evaluator = 별도 프로세스/별도 컨텍스트**: Agent SDK에서 generator 세션과 분리된 evaluator 세션(또는 subprocess)으로 띄워 generator의 대화 히스토리·주장에 오염되지 않게. held-out split·oracle은 evaluator만 접근.

### 6.2 "랩 = 합성 가능한 모듈" 인터페이스 (협업 확장 대비)

지금부터 랩을 깨끗한 경계를 가진 단위로 설계하면 나중에 랩들이 협업한다:

```
Lab = {
  intake:    Hypothesis | ResearchQuestion      # 입력
  registry:  ExperimentRegistry (파일/DB)        # 영속 상태
  evaluator: IndependentEvaluator (oracle 접근)  # 품질 게이트
  output:    VerifiedResult { metrics, artifacts, verdict, provenance }  # 출력
}
```

- 랩의 **출력은 evaluator가 서명한 VerifiedResult만**(generator 주장 금지). → 이후 다른 랩이 이 출력을 신뢰하고 입력으로 받을 수 있음(랩 간 협업의 신뢰 기반).
- 도메인 의존부(데이터셋·oracle·CUDA 이미지)는 **플러그인 인터페이스 뒤에** 둬서 하위도메인 미확정 상태를 흡수.

### 6.3 단일 GPU·Python Agent SDK용 구체 컴포넌트

- **Harness 루프**: Python 결정적 코드. Agent SDK 호출은 추론 지점에서만(가설·코드·해석·평가).
- **GPU 리스**: 파일락/세마포어 1개. 동시 실행 금지. 큐가 직렬화.
- **Dataset/Weights 캐시**: WSL2 영속 볼륨(예: `~/.cache/lab/`). 다운로드는 harness 잡, 1회.
- **CUDA 이미지 레지스트리**: (framework × CUDA) 사전 빌드 매트릭스. torch→cu12x 매핑을 표로 고정, healthcheck.
- **Experiment Registry**: SQLite 또는 JSONL. 컨텍스트 리셋·재시작 생존.
- **로그 정책**: 학습/다운로드 로그는 파일로, 에이전트엔 집계+표준 에러만. 학습 stdout을 컨텍스트에 흘리지 말 것.
- **Failed-approaches log + Lab Notebook**: 순환 방지·진행 가시성.
- **착수 검증 게이트**: 첫 실험은 알려진 reproduction. evaluator가 oracle 재현에 성공해야 자율 루프 개방.

### 6.4 권장 착수 순서 (구현 시)

1. Harness 골격(루프·registry·GPU락·캐시·이미지 레지스트리) — 모델 호출 0의 결정적 인프라부터.
2. 독립 evaluator + oracle 보정(reproduction 1건으로 verifier가 "거의 완벽"한지 증명).
3. generator/expert 서브에이전트 + experiment contract 협상.
4. 완전 자율 루프 개방(이때만). 예산은 토큰+실험회수로 측정.
5. VerifiedResult 출력 인터페이스 고정 → 2번째 랩 추가 시 협업.

> 다음 단계: 이 리포트 기반으로 `/sc:design`(아키텍처 상세) 또는 `/sc:implement`(harness 골격부터 구현). 본 `/sc:research`는 리포트까지만 산출한다.

---

## Sources
- Anthropic Engineering — Harness design for long-running agentic apps: https://www.anthropic.com/engineering/harness-design-long-running-apps
- Anthropic Engineering — Building a C compiler: https://www.anthropic.com/engineering/building-c-compiler
