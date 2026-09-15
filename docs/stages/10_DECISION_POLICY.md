# Stage 10: The Decision Policy (What To Do, and When)

> Probabilities aren't actions. For the root-cause service, we compute the **expected utility** of each possible action (the benefit of each action averaged over how likely each health state is) and pick the best: **Maximum Expected Utility (MEU)**. Before anything is allowed, safety rules must also pass: an 11-tick debounce, P(Critical) ≥ 0.95, a 5-minute cooldown, and at most 3 actions per hour.

---

## 1. In simple words

Should you take an umbrella? If rain is 90% likely, yes. If it's 5% likely, carrying it is a nuisance. You're weighing **how likely each situation is** against **how good or bad each choice is in that situation**.

Here, the situations are the service's health states and the choices are the actions:

| | Do nothing | Scale out | Restart | Reschedule | Traffic shift |
|---|---|---|---|---|---|
| **service actually Normal** | fine (+10) | small waste (0) | needless disruption (−5) | bigger needless disruption (−15) | small cost (−2) |
| **Degrading** | bad (−5) | good (+15) | ok (+10) | ok (+5) | good (+12) |
| **Critical** | very bad (−50) | helps some (+20) | helps (+30) | helps most (+45) | helps (+25) |

Multiply each row by the probability of that state, add them up per column, and pick the largest total.

Then come the safety rules, like a hospital protocol: *confirm the diagnosis over time, only treat when very sure, don't give the same treatment twice within 5 minutes, and never more than 3 times an hour.*

## 2. Why this stage exists

- The DBN says "route: 99% Critical". Someone has to decide **what** to do.
- **Different actions suit different situations.** Scaling out is cheap and fine for early degradation; rescheduling is disruptive but strongest when critical.
- **Wrong actions cause outages.** Restarting a healthy service hurts users. So action must be cautious, reversible and rate-limited.
- **This is the decision-theory half of the "Dynamic Decision Network".**

---

## 3. Step by step (`DecisionPolicy.evaluate`, every tick)

```
Input: DBN output (posteriors, root_cause, expected_utilities)

1. root cause is "None"?                     → reset counter;          state HEALTHY
2. same root cause as last tick?             → counter + 1, else counter = 1
3. counter < debounce (11)?                  → state PENDING
4. take the root cause's EU for each action; best = highest
   (ties broken: Reschedule > Restart > Scale_Out > Traffic_Shift > Do_Nothing)
5. best is Do_Nothing?                       → state HEALTHY
6. P(Critical) of root cause < 0.95?         → state PENDING   (reason: risk below threshold)
7. last action on this service < 300 s ago?  → state COOLDOWN
8. ≥ 3 actions on this service in last hour? → state RATE_LIMITED
9. otherwise                                 → state INTERVENE, action = best

The operator then executes (or shadows) it, and calls record_action(),
which starts the cooldown and adds to the rate-limit history.
```

---

## 4. Technical depth

### 4.1 Expected utility

For service s with posterior p = [P(Normal), P(Degrading), P(Critical)]:

```
EU(action) = Σ_state P(state) · U(state, action)          i.e.   EU = p · U   (vector × matrix)
MEU action = argmax_action EU(action)
```

The utility matrix U, from `src/ddn_core_phase3.py`:

```
                Do_Nothing  Scale_Out  Restart_Pod  Reschedule_Pod  Traffic_Shift
Normal             10          0          -5            -15             -2
Degrading          -5         15          10              5             12
Critical          -50         20          30             45             25
```

The DBN computes EU for every service every tick. The policy uses the **root cause's**.

### 4.2 Worked examples (exact)

| Posterior (N, D, C) | Do_Nothing | Scale_Out | Restart | Reschedule | Traffic_Shift | Best |
|---|---|---|---|---|---|---|
| (1, 0, 0) healthy | **10** | 0 | −5 | −15 | −2 | Do_Nothing |
| (0, 1, 0) degrading | −5 | **15** | 10 | 5 | 12 | Scale_Out |
| (0, 0, 1) critical | −50 | 20 | 30 | **45** | 25 | Reschedule |
| (0, 0.01, 0.99) **operator test** | −49.55 | 19.95 | 29.8 | **44.6** | 24.87 | Reschedule |
| (0.05, 0, 0.95) at the threshold | −47 | 19 | 28.25 | **42** | 23.65 | Reschedule |

Row 4 by hand: Reschedule = 0·(−15) + 0.01·5 + 0.99·45 = 0.05 + 44.55 = **44.6**. This matches the operator's audit log exactly.

### 4.3 Which action wins as risk grows

If the rest of the probability is **Normal** (P(D) = 0):

| P(Critical) | Best action |
|---|---|
| 0 – 0.126 | Do_Nothing |
| 0.126 – 0.286 | Scale_Out |
| 0.286 – 0.375 | Traffic_Shift |
| 0.375 – 0.401 | Restart_Pod |
| **0.401 – 1.0** | **Reschedule_Pod** |

If the rest is **Degrading**: Scale_Out up to P(C) = 0.286, then Reschedule_Pod.

**So the matrix encodes an escalation ladder:** cheap actions at moderate risk, the strongest at high risk. **But** the policy only acts at P(Critical) ≥ 0.95, where Reschedule_Pod always wins, and Reschedule has no live implementation (Stage 11). In practice, only the top rung is ever reached.

### 4.4 The debounce

- The counter counts **consecutive ticks on which the same service is named root cause**. It is *not* "consecutive ticks with P(Critical) ≥ 0.95". The risk threshold is checked separately in step 6.
- Any tick with a different name, or with "None", resets the counter.
- With 60 s ticks, 11 ticks = **11 minutes** from the first naming. On the recorded faults, the root cause was named from tick 9, so the debounce would be satisfied at tick 19, 11 minutes after injection. The live operator decided at **10.9 minutes**.

### 4.5 Cooldown and rate limit

- `record_action(target)` stores the time. Then `remaining = 300 − (now − last_time)`; while it's positive, the state is COOLDOWN.
- Rate limit: timestamps of actions on the same service in the last 3,600 s; at 3 or more, the state is RATE_LIMITED.
- Both are **per service**.
- `export_state()` / `import_state()` save the counter, last root cause, action times and rate history into the operator's status `checkpoint`, so a restarted operator doesn't reset mid-incident.

### 4.6 Node-pressure adjustment (built, unused)

If `node_pressure_flag` is set and P(Critical) > 0.3, the DBN adds +35 to Reschedule and −25 to Restart (restarting on a sick machine rarely helps). The operator always passes `False`.

### 4.7 Where the utilities came from

They are **hand-set design values** from the project's architecture documents, not learned or measured. They encode an ordering: disruptive actions are costly when the service is healthy and valuable when it's critical. The exact numbers are arbitrary, and changing them moves the crossover points in 4.3.

---

## 5. Real example: the live operator test (route fault)

```
min after fault  root cause   counter  P(Crit) root   policy state                      status action
  0.8            route        1/11     0.01           PENDING (debounce)                -
  1.9            route        2/11     0.67           PENDING (debounce)                -
  2.9            route        3/11     0.94           PENDING (debounce)                -
  3.9            route        4/11     0.98           PENDING (debounce)                -
  ...            route        5-10     0.98-0.99      PENDING (debounce)                -
 10.9            route        11/11    0.99           INTERVENE → Reschedule_Pod        WOULD_EXECUTE (shadow)
 11.9-14.0       route        12-14    0.98-0.99      COOLDOWN (300 s from 10.9)        Do_Nothing
 14.0            fault removed
 16.0-18.0       None         0        0.00           HEALTHY (counter reset)           -
```

Notice: **from 3.9 to 10.9 minutes, the model was over 95% sure but the policy waited only because of the debounce.** In the recorded runs, this fault reached users at 5 minutes.

---

## 6. Where it lives in the code

| What | File |
|---|---|
| Policy | `src/decision_policy.py` (`DecisionPolicy.evaluate`, `record_action`) |
| Utility matrix + EU computation | `src/ddn_core_phase3.py` (`utility_matrix`, step 5) |
| Configuration from the custom resource | `src/operator.py::_config` (`threshold.risk`, `debounce.ticks`, `cooldown.seconds`) |

## 7. Limits and known issues

- **The debounce is too long for these faults.** It acts at 11 minutes against a 5-minute disruption. A counter of ticks *above the risk threshold*, or a shorter debounce, would act sooner, but its false-action cost hasn't been measured.
- **Only Reschedule_Pod is ever chosen** at the 0.95 threshold, and it has no live implementation.
- **The utilities are hand-set.** No sensitivity analysis was done.
- **Disabled actions don't fall back.** The operator doesn't pass `spec.actions` into the policy, so if Reschedule were disabled, the policy would still pick it and the operator would return `DISABLED`, not the next-best action.
- **The debounce counts "named root cause" ticks**, which can include ticks where the naming came from the anomaly signal alone. That's safe, because the risk threshold still gates action, but the name suggests something stricter.
- **Per-service limits only.** There's no global cap across services.

## 8. Questions a reviewer may ask

**Q: What is Maximum Expected Utility?**
A: For each action, average its utility over the possible states, weighted by their probabilities, then pick the action with the highest average. It's the standard rational decision under uncertainty.

**Q: Compute the expected utility of Reschedule for P = (0, 0.01, 0.99).**
A: 0 × (−15) + 0.01 × 5 + 0.99 × 45 = 44.6. That's what the operator logged.

**Q: Why is Do_Nothing −50 when Critical?**
A: Ignoring a critical service leads to an outage. It's the worst outcome in the matrix, so almost any action beats it once Critical is likely.

**Q: Why require both MEU and P(Critical) ≥ 0.95?**
A: MEU already prefers acting from about P(C) = 0.13 (scale out). The 0.95 threshold is a separate conservative safety rule, so that disruptive automation only runs when the model is nearly certain.

**Q: Why an 11-tick debounce?**
A: To ignore one-off naming blips. It came from the design documents and proved too long for sudden CPU faults. That's our main operator finding, and future work.

**Q: Where do the utility numbers come from?**
A: They're hand-set to encode an ordering of costs and benefits, not learned. A real deployment would set them from incident costs, or learn them.
