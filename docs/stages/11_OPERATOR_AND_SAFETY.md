# Stage 11: The Operator and Safety Gates (Running It Live)

> A Kubernetes **operator** runs the whole pipeline every 60 seconds against the live cluster, publishes what it believes to a custom Kubernetes object called `FailurePredictor`, and applies an action only if several independent safety gates are all open. By default it runs in **shadow mode**: it decides and logs, but never changes anything.

---

## 1. In simple words

Everything before this stage can run on recorded files. The operator is the part that **lives in the cluster's control loop**:

- Every minute: *query Prometheus → Rectifier → autoencoder → DBN → root cause → decision*.
- It writes the result somewhere any engineer can read with `kubectl get fp -o yaml`: risk, suspected service, chosen action, debounce progress, health.
- If an action is due, it goes through **safety gates**. Unless a human has deliberately opened *two different switches in two different places*, it only records **"WOULD_EXECUTE"**.

Think of a trainee doctor who writes "I would give drug X" in the chart. The senior doctor must both sign the chart *and* unlock the medicine cabinet before anything is given.

## 2. Why this stage exists

- PREFACE only raised alarms. The project's goal includes **acting before users are hurt**.
- Automated actions on production are dangerous: a false alarm that restarts services **causes** an outage. So the design is safety-first: shadow by default, several gates, cooldowns, rate limits and an audit trail.
- Operators are the standard Kubernetes way to add custom automation, configured through the Kubernetes API.

---

## 3. Step by step

### 3.1 Setting it up

```
Step 1  kubectl apply -f manifests/crd-failurepredictor.yaml       # teach Kubernetes the new object type
Step 2  kubectl apply -f manifests/sample-failurepredictor.yaml    # create one: train-ticket-predictor
Step 3  kubectl port-forward -n monitoring svc/prometheus-server 9090:80 --address 127.0.0.1
Step 4  PREFACE_TICK_SECONDS=60 kopf run src/operator.py --verbose --standalone
        (PREFACE_ALLOW_LIVE unset -> live actions impossible)
Step 5  kubectl get fp train-ticket-predictor -o yaml               # read its status
```

The operator runs **on the laptop** using the kubeconfig. There is no container image or in-cluster Deployment for it yet.

### 3.2 What happens every tick (`reconcile`)

```
every 60 s (kopf.timer, sharp=True):
  1. read spec -> config (threshold 0.95, debounce 11, cooldown 300 s, shadowMode, actions)
  2. get or build runtime state: DecisionPolicy, ActionExecutor, AuditLogger, InferenceAdapter
  3. adapter.run_tick():
        Prometheus (2 queries) -> Rectifier -> autoencoder -> DBN.step
     on ANY exception: write status.health.lastError, keep old belief, stop this tick
  4. decision = policy.evaluate(ddn_output)              (Stage 10)
  5. if decision.state == INTERVENE and action != Do_Nothing:
        if action disabled in spec.actions   -> "DISABLED"
        else result = executor.execute(action, root_cause, shadow_mode)
        if result is EXECUTED or WOULD_EXECUTE -> policy.record_action (starts cooldown)
        append to audit.log (JSONL) and status.recentActions
  6. patch status: risk, rootCause, decision, persistence, safety, health, recentActions, checkpoint
  7. warn if the tick took more than 5 s
```

---

## 4. Technical depth

### 4.1 The custom resource

**Spec** (what a human sets), from `manifests/sample-failurepredictor.yaml`:

```yaml
apiVersion: preface.example.com/v1alpha1
kind: FailurePredictor
metadata: { name: train-ticket-predictor, namespace: default }
spec:
  threshold: { risk: 0.95 }        # P(Critical) needed to act
  debounce:  { ticks: 11 }         # consecutive ticks the same root cause must be named
  cooldown:  { seconds: 300 }      # per service, after an action
  actions:
    reschedule: { enabled: true }
    restart:    { enabled: true }
    scale:      { enabled: true }
  policy:
    shadowMode: true               # absent also means true
```

**Status** (what the operator writes):

| Field | Meaning |
|---|---|
| `risk.current` | highest P(Critical) over all services this tick |
| `risk.criticalProbability` | P(Critical) of the named root cause |
| `rootCause.service` | root cause from Stage 09, or "None" |
| `decision.selectedAction`, `deltaEu`, `interventionEligible` | the Stage 10 result; deltaEu = EU(Reschedule) − EU(Restart) |
| `persistence.currentTicks / requiredTicks` | debounce progress, e.g. 4/11 |
| `safety.shadowMode`, `cooldownRemaining`, `rateLimitState` | the gate state (`rateLimitState` is the length of the recent-actions list) |
| `health.lastError`, `lastSuccessfulTick`, `inferenceLatency` | whether it is working |
| `recentActions` | last 10 action records with timestamps |
| `checkpoint` | the decision policy's state, so a restarted operator resumes mid-incident |

### 4.2 Kopf handlers

| Handler | Trigger | Does |
|---|---|---|
| `on_ready` | resource created, or operator (re)started | builds state; restores `checkpoint` if present; writes initial safety/health |
| `on_spec_change` | spec edited | updates debounce, cooldown and threshold without a restart |
| `on_delete` | resource deleted | drops its runtime state |
| `reconcile` | timer every `PREFACE_TICK_SECONDS` (60) | the tick above |

The `InferenceAdapter` is created **lazily** on the first tick. It loads PyTorch and needs Prometheus, so building it at startup would crash-loop the operator whenever either is missing. Created lazily, the problem shows up in `status.health.lastError`. After an error the adapter is discarded and rebuilt on the next tick.

### 4.3 The safety gates, in order

For the cluster to actually change, **all** of these must hold:

| # | Gate | Where it's set | Default |
|---|---|---|---|
| 1 | Decision state is `INTERVENE` (debounce met, P(Critical) ≥ threshold, no cooldown, under rate limit, best action ≠ Do_Nothing) | Stage 10 logic + spec | — |
| 2 | That action is enabled in `spec.actions` | the custom resource | enabled in the sample |
| 3 | `spec.policy.shadowMode` is **explicitly** `false` | the custom resource | **true** (absent = true) |
| 4 | `PREFACE_ALLOW_LIVE=true` in the operator's environment | where the operator runs | **unset** |
| 5 | The action has a real implementation | code (`LIVE_ACTIONS`) | only `Restart_Pod`, `Scale_Out` |

Gates 3 and 4 are deliberately in **two different places**: the Kubernetes object, and the process environment. A single mistaken edit or default cannot cause a real mutation.

`ActionExecutor.execute` return codes:

| Code | When |
|---|---|
| `WOULD_EXECUTE` | shadow mode, nothing touched |
| `BLOCKED` | live requested, but `allow_live` is false |
| `NOT_IMPLEMENTED` | live allowed, but there's no implementation for this action |
| `EXECUTED` | the API call succeeded |
| `FAILED` | the API call raised |

### 4.4 What the live actions do

- **Restart_Pod** → `_restart_deployment`: patches the Deployment's pod template with the annotation `kubectl.kubernetes.io/restartedAt: <now>`. This is exactly what `kubectl rollout restart` does: pods are replaced gradually, never all at once.
- **Scale_Out** → `_scale_out`: reads the current replicas (respecting what the HPA already decided) and sets `current + 1`, capped at `max_replicas = 10`.
- **Reschedule_Pod**, **Traffic_Shift** → `NOT_IMPLEMENTED`.

⚠ **Important consequence.** At P(Critical) ≥ 0.95, the utility matrix always makes **Reschedule_Pod** the best action (Stage 10). So if both live gates were opened today, the operator would return `NOT_IMPLEMENTED` rather than act. And if `reschedule` were disabled in the spec, it would return `DISABLED`; it does not fall back to Restart. **In the current configuration, no live mutation can happen.** That is safe, but it means live mitigation has never been exercised end to end.

### 4.5 Audit log

Every eligible decision is appended as one JSON line to `audit.log`: timestamp, resource, risk, root cause, all expected utilities, chosen action, deltaEU, persistence, state, shadow flag, execution status and reason.

### 4.6 Performance

The design budget is 5 s per tick. After the IPv4 fix (`127.0.0.1` instead of `localhost`), ticks took **0.06–0.10 s**, down from 4.2 s.

---

## 5. Real example: the end-to-end operator test

`scripts/42_operator_fault_test.py` treats the operator as a black box and reads only its published status, as a user would:

1. Wait for 3 healthy ticks.
2. Snapshot `ts-route-service`'s Deployment (generation, replicas, restart annotation).
3. Inject Chaos Mesh CPU stress into `ts-route-service` (no autoscaler, so the fault persists).
4. Log every tick until an action is decided, plus 3 more.
5. **Always** remove the fault (in a `finally` block), then log recovery.
6. Re-snapshot and run 9 checks.

```
min after fault   P(Critical) on root   root cause          debounce   decision
  0.8             0.01                  ts-route-service     1/11
  1.9             0.67                  ts-route-service     2/11
  2.9             0.94                  ts-route-service     3/11
  3.9             0.98                  ts-route-service     4/11
 10.9             0.99                  ts-route-service    11/11      Reschedule_Pod -> WOULD_EXECUTE
 11.9 - 14.0      0.98 - 0.99           ts-route-service    12-14/11   Do_Nothing (cooldown)
 14.0             fault removed
 15.0             1.00                  ts-route-service
 16.0 - 18.0      0.00                  None                 0/11
```

**All 9 checks passed:**

| Check | Result |
|---|---|
| no inference errors | ✅ |
| every tick within 5 s | ✅ (0.06–0.10 s) |
| healthy ticks quiet (max P(Critical) < 0.5) | ✅ |
| root cause named as ts-route-service | ✅ at 0.8 min |
| P(Critical) on root ≥ 0.95 | ✅ at 3.9 min |
| intervention became eligible | ✅ |
| action decided on ts-route-service, WOULD_EXECUTE | ✅ at 10.9 min |
| deployment unchanged (shadow mode respected) | ✅ |
| risk fell after fault removal | ✅ |

Expected utilities in the audit record: Reschedule_Pod 44.6, Restart_Pod 29.8, Scale_Out 19.9, Do_Nothing −49.5. These correspond exactly to P = (Normal 0, Degrading 0.01, Critical 0.99).

**The finding:** the model was confident at 3.9 minutes, but the **11-tick debounce** delayed the decision to 10.9 minutes. In the recorded runs, the same fault on the same service reached users at 5 minutes. **In this configuration the operator would act about 6 minutes after users were already affected.**

---

## 6. Where it lives in the code

| What | File |
|---|---|
| Operator (Kopf handlers, reconcile) | `src/operator.py` |
| Live inference path | `src/inference_adapter.py` |
| Actions and gates | `src/action_executor.py` |
| Audit log | `src/audit_logger.py` |
| CRD / sample resource | `manifests/crd-failurepredictor.yaml`, `manifests/sample-failurepredictor.yaml` |
| End-to-end test | `scripts/42_operator_fault_test.py` |

## 7. Limits and known issues

- **Acts too late**, because of the debounce (above). P(Critical) first reached 0.95 at 3.9 minutes, so a debounce of 3 consecutive ticks *above the risk threshold* would have acted at about 6 minutes instead of 10.9. That's still slightly after the 5-minute disruption, and its false-action cost is unmeasured.
- **The chosen action has no live implementation** (Reschedule), so live mode is effectively unreachable.
- **Runs outside the cluster.** There's no image, Deployment, RBAC or leader election. `--standalone` means a single instance.
- **One test run**, on one service.
- **`rateLimitState`** reports the number of recent actions, not rate-limit usage.
- **In-memory state** per process. The checkpoint restores only the decision policy, not the DBN particles or the Rectifier's moving average.

## 8. Questions a reviewer may ask

**Q: What is an operator, and why use one?**
A: A controller that watches a custom Kubernetes resource and reconciles the cluster toward it. It's the standard way to add domain automation, configured and observed through `kubectl`.

**Q: How do you make sure it doesn't break production?**
A: Shadow mode is the default. A live action needs `shadowMode: false` on the resource **and** `PREFACE_ALLOW_LIVE=true` in the operator's environment. On top of that come the debounce, the 0.95 risk threshold, a 300 s per-service cooldown, 3 actions per hour, per-action enable flags and an audit log. The end-to-end test verified that the deployment was unchanged.

**Q: What happens if Prometheus fails mid-incident?**
A: The tick records the error in `status.health.lastError` and leaves the belief untouched. A scrape failure is missing evidence, not evidence of health.

**Q: Did it ever actually restart anything?**
A: No. All tests were in shadow mode, and the action it chooses at high risk (Reschedule) is not implemented live. Restart and Scale-out are implemented against the Kubernetes API but were not exercised on a live fault.

**Q: Why did it take 10.9 minutes to act?**
A: The model reached P(Critical) 0.98 at 3.9 minutes, but the policy requires the same root cause on 11 consecutive ticks, which is 11 minutes with 60-second ticks. That's a configuration trade-off against false actions, and the main thing to improve.
