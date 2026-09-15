# Stage 07: The Dynamic Bayesian Network (How Probability Is Computed)

> The core of the project. For every service we keep a belief, a probability for **Normal**, **Degrading** and **Critical**, and update it every minute using (1) how services usually change over time, (2) how sick their callers are, and (3) the new anomaly signal. We compute it with a **particle filter** of 500 simulated guesses.

---

## 1. In simple words

A doctor can't see "health" directly. They see symptoms, like a temperature reading. A good doctor:

- **remembers** the patient was fine an hour ago, so one high reading might be a measuring glitch;
- **knows how illnesses progress**: people rarely jump from perfectly fine to critical in a minute, and once critical they usually stay that way for a while;
- **knows what each state looks like**: a healthy person reads about 37 °C, a sick one about 39 °C;
- **combines all of it** into "I'm 90% sure this patient is critical".

The DBN does exactly this for each service. The "temperature" is the anomaly signal from Stage 05.

**Why 500 particles?** Instead of doing the maths for every possible combination of 8 services' states, we keep 500 "imaginary copies" of the system. Each copy is one full guess, like *"route is Critical, all others Normal"*. Every minute we let each guess evolve randomly by the rules, check how well each explains the new signals, and **keep more copies of the guesses that explain them well**. The share of copies where route is Critical **is** P(route = Critical).

## 2. Why this stage exists

PREFACE decides with a memoryless rule: *signal above a line means alarm*. On real telemetry:

- **Noise crosses the line.** Healthy signals on our cluster regularly exceeded PREFACE's line: 16 of 28 ticks in one healthy run. PREFACE false-alarmed on every healthy run.
- **There's no notion of "getting worse".** A threshold gives a yes/no answer. Early warning needs an in-between state, which we call **Degrading**.
- **There's no confidence.** A decision to restart production services should depend on *how sure* we are. The DBN gives probabilities, which Stage 10 turns into expected utility.
- **There are no dependencies.** A DBN can include "a sick caller makes a callee more likely to get sick".

---

## 3. Step by step (one tick of `DynamicDecisionNetworkPhase3.step`)

```
Input: anomaly signals {service: a_s} from the autoencoder

1. PREDICT    For each particle, for each service (parents first):
                take its previous state, look up the transition row,
                shift the row if its worst parent is Degrading/Critical,
                randomly draw the new state.
2. WEIGHT     For each particle: how likely are ALL 8 observed signals
                if the services were in this particle's states?
                weight = Π_s N(a_s ; μ_state, σ_state)       (computed in log space)
3. RESAMPLE   Draw 500 particles with replacement, in proportion to weight.
                Good guesses get copied; bad guesses disappear.
4. POSTERIOR  For each service: P(Normal) = share of particles in Normal, etc.
5. ROOT CAUSE Pass signals + posteriors to the causal analyzer (Stage 09).
6. UTILITY    For each service: EU(action) = Σ_state P(state) · U(state, action)  (Stage 10)

Output: {posteriors, root_cause, expected_utilities, causal_data}
The particles are kept for the next tick. That is the "memory".
```

---

## 4. Technical depth

### 4.1 The random variables

For each service s and tick t:

- **Hidden state** H_t^s ∈ {0 = Normal, 1 = Degrading, 2 = Critical}. It is never observed.
- **Observation** A_t^s ∈ [0, 15], the anomaly signal (clipped).

### 4.2 The network structure (two time slices)

A DBN is a Bayesian network repeated over time. Each arrow means "depends on":

```
        time t-1                         time t
   H(train)_{t-1} ──────────────────► H(train)_t ──────► A(train)_t
                                          │
                                          │  parent influence (train calls route)
                                          ▼
   H(route)_{t-1} ──────────────────► H(route)_t ──────► A(route)_t
```

This gives the factorisation:

```
P(H_t, A_t | H_{t-1}) = Π_s  P(H_t^s | H_{t-1}^s, worst(H_t^{parents(s)}))  ·  P(A_t^s | H_t^s)
                             └──────────── transition model ───────────┘     └─ emission ─┘
```

Two assumptions make this tractable:
- **Markov:** the next state depends only on the current state (plus the parents), not the whole history. The history is summarised in the belief.
- **Conditional independence of observations:** given its hidden state, a service's signal doesn't depend on other services.

> Implementation detail: services are processed in **topological order**, and particles are updated in place. So the parent state a child sees is the parent's **new** state at tick t, not its state at t−1.

### 4.3 The transition model

A 3×3 matrix. Row = state now, column = state next minute. Calibrated from recorded runs (Stage 08), from `data/experiments/params/live_v2.json`:

```
                 → Normal   → Degrading   → Critical
from Normal        0.9960      0.0032       0.0008
from Degrading     0.0769      0.6154       0.3077
from Critical      0.0200      0.0200       0.9600
```

How to read it:
- A Normal service stays Normal with 99.6% probability each minute. The expected stay is 1 / (1 − 0.996) ≈ 250 minutes.
- Degrading is **short-lived** (expected stay about 2.6 minutes). It goes to Critical 31% of the time.
- Critical is **sticky**: 96% chance of staying, so the expected stay is 25 minutes.

For comparison, the hand-written defaults before calibration were:
```
[0.950 0.045 0.005]
[0.200 0.650 0.150]
[0.020 0.180 0.800]
```

**Parent influence (the graph inside the DBN).** For each particle, take the worst state among the service's parents. If it is Degrading (1) or Critical (2), add a modifier vector to the transition row, then clip and renormalise:

```python
t_row = T[prev_state].copy()
if parents:
    worst_parent = max(particle[parent] for parent in parents)
    if calibrated modifiers exist:  t_row += modifiers[worst_parent]
    else (defaults):                 worst=1: +0.10 to D, +0.05 to C
                                     worst=2: +0.05 to D, +0.20 to C
    t_row = t_row / t_row.sum()
new_state = np.random.choice([0, 1, 2], p=t_row)
```

⚠ **With the calibrated parameters, the modifiers are all zero** because of a calibration bug (Stage 08, section 7). In the live evaluation, the transition step therefore treated services independently, and the graph acted only through root-cause analysis.

### 4.4 The emission model

Each state produces signals from a Gaussian (normal) distribution:

```
P(A = a | H = k) = N(a; μ_k, σ_k) = 1 / (σ_k √(2π)) · exp( −(a − μ_k)² / (2 σ_k²) )
```

Calibrated values:

| State | μ (typical signal) | σ (spread) |
|---|---|---|
| Normal | 0.66 | 0.80 |
| Degrading | 7.00 | 4.84 |
| Critical | 10.54 | 0.80 |

The likelihood of a few observations under each state:

| Observed signal | under Normal | under Degrading | under Critical | Most likely |
|---|---|---|---|---|
| 0.30 (quiet) | 0.45 | 0.032 | 8×10⁻³⁷ | Normal |
| 3.00 (healthy spike) | 0.0071 | 0.059 | 2×10⁻²⁰ | Degrading, but Critical is essentially impossible |
| 10.66 (route fault) | 1×10⁻³⁴ | 0.062 | 0.495 | Critical |

**Log space.** Multiplying 8 likelihoods like 10⁻³⁷ underflows float32 to 0. So the code **adds log-likelihoods** and normalises with the *log-sum-exp trick*:

```python
log_w = Σ_s  [ −log(σ_k √(2π)) − ½ ((a_s − μ_k)/σ_k)² ]     # per particle
w = exp(log_w − max(log_w));  w = w / w.sum()                 # stable normalisation
```

Signals are clipped to [0, 15] first, so that one absurd value cannot give every particle a weight of zero ("weight degeneracy").

### 4.5 The exact maths the particles approximate (Bayes filter)

For **one service on its own**, the belief b_t(k) = P(H_t = k | a_1..a_t) updates as:

```
prior_t(k)     = Σ_j  b_{t-1}(j) · T[j, k]                     (predict)
b_t(k)         ∝ prior_t(k) · N(a_t ; μ_k, σ_k)                 (update, Bayes' rule)
```

That is **Bayes' rule**: posterior ∝ prior × likelihood. Run exactly on the real route-fault signals (fault injected at tick 8):

```
tick  signal   prior N/D/C              posterior N/D/C
  7    0.00    0.996 0.003 0.001        1.000 0.000 0.000
  8    0.00    0.996 0.003 0.001        1.000 0.000 0.000     <- fault injected, not visible yet
  9   10.66    0.996 0.003 0.001        0.000 0.323 0.677     <- one strong observation
 10   11.55    0.038 0.212 0.749        0.000 0.063 0.937
 11   11.24    0.024 0.057 0.919        0.000 0.010 0.990
 12   11.11    0.021 0.026 0.953        0.000 0.004 0.996
```

Two things to notice:
- At tick 9 the **prior** says only 0.1% Critical, but the **likelihood** of 10.66 under Normal is 10⁻³⁴, so the evidence overwhelms the prior.
- Belief then **builds**: 0.68 → 0.94 → 0.99. That persistence is what a threshold lacks.

### 4.6 Why particles, not the exact formula?

With 8 services, the **joint** state has 3⁸ = 6,561 combinations, which is still small enough for exact computation. But:
- the parent coupling makes services depend on each other, so the joint table must be handled, not 8 separate ones;
- at the 20–30 services the design targets, 3³⁰ ≈ 2×10¹⁴ combinations is impossible.

A particle filter's cost grows with *particles × services* (500 × 8 = 4,000 random draws per tick), not exponentially. It takes about 0.06 s per tick in Python.

### 4.7 The particle filter, precisely

- **State:** `particles` is a 500 × 8 integer array. Row = one guess of all 8 services' states. At start, every particle is all Normal.
- **Predict:** sample each service's next state from its (parent-adjusted) transition row.
- **Weight:** one joint weight per particle, the product over all 8 services of the emission likelihoods.
- **Resample:** `np.random.choice(500, size=500, p=weights)` (multinomial resampling).
- **Posterior:** `P(s = Critical) = mean(particles[:, s] == 2)`.

Because random numbers are involved, **different seeds give slightly different posteriors**. That's why every evaluation number is reported over 8 seeds.

---

## 5. Real examples from our data

### 5.1 Particle filter on the route fault (seed 0)

```
tick  route P(N/D/C)       root cause           EU(Reschedule) on route
 0-8  1.00 / 0.00 / 0.00   None                 -15.0
  9   0.00 / 1.00 / 0.00   ts-route-service       5.0
 10   0.00 / 0.36 / 0.64   ts-route-service      30.5
 11   0.00 / 0.06 / 0.94   ts-route-service      42.6
 12   0.00 / 0.00 / 1.00   ts-route-service      44.8
```

**Why does tick 9 say 100% Degrading when the exact answer is 32% D / 68% C?** Before resampling at tick 9, the prediction step had moved route in only **1 of 500 particles** (499 Normal, 1 Degrading, 0 Critical), matching the tiny 0.3% and 0.08% transition probabilities. Every Normal particle has likelihood about 10⁻³⁴, so the single Degrading particle got essentially all the weight, and resampling copied it 500 times. This is **particle degeneracy**: when the evidence is very surprising compared with the prior, few particles are in the right place. The filter recovers over the next ticks (0.64, 0.94, 1.00), but the first step is coarse. More particles, or a proposal that looks at the observation, would help.

### 5.2 A healthy run: why no false alarm

Healthy run `live_healthy_000`: peak signal **2.23**, and **16 of 28 ticks above PREFACE's 1.386 line**. PREFACE alarmed. The DBN's highest P(Critical) on any service, on any tick, was **0.000**:

- Under Critical (μ = 10.54, σ = 0.80), a signal of 2.23 is about 10 standard deviations away. Its likelihood is effectively zero, so no Critical particle survives weighting.
- Even Degrading needs a particle to *transition* there first (0.32% per minute), and the next quiet signal removes it again.

A one-off spike cannot build a persistent belief. That's the whole difference from a threshold.

### 5.3 Abrupt recovery

In the operator test, when the fault was removed, P(Critical) went **1.00 → 0.00 in one tick**, despite the 0.96 Critical→Critical transition. A Normal-looking signal is so improbable under Critical's narrow Gaussian (σ = 0.80) that the few particles that transitioned to Normal take all the weight. Clearing quickly is correct here, but it shows the evidence dominates the transition prior when the emissions are this narrow.

---

## 6. Where it lives in the code

| What | File |
|---|---|
| DBN / particle filter / expected utility | `src/ddn_core_phase3.py` (`DynamicDecisionNetworkPhase3.step`) |
| Original version without log-space or clipping | `src/ddn_core.py` |
| Calibrated parameters | `data/experiments/params/live_v2.json` |
| Replay used in evaluation (alarm = any P(Critical) > 0.5) | `scripts/36_evaluate_goal6.py::replay` |
| Live use | `src/inference_adapter.py` (500 particles, calibrated params) |

The file is called "DDN", a **Dynamic Decision Network**: a DBN plus decision and utility nodes. The same class does both.

## 7. Limits and known issues

- **Stochastic.** Results vary by seed. With v1's thin calibration, recall ranged from 33% to 67% across 8 seeds.
- **Particle degeneracy** on sudden, strong evidence (5.1).
- **Degrading is poorly calibrated** (σ = 4.84 from only 10 labels), so the in-between state is not very informative.
- **Parent influence is effectively off** with calibrated parameters (calibration bug, Stage 08).
- **Gaussian emissions on a log signal** are an approximation. Real signals are skewed and have many exact zeros.
- **The structure is fixed by hand** (3 states, Markov, worst-parent rule). It is not learned.
- **`node_pressure_flag`** (boosts Reschedule on node pressure) exists but is always `False` live.

## 8. Questions a reviewer may ask

**Q: What makes it "Bayesian"?**
A: The belief is a probability distribution updated with Bayes' rule: posterior ∝ prior × likelihood. The prior comes from the previous belief and the transition model; the likelihood comes from the Gaussian emission of the new signal.

**Q: What makes it "dynamic"?**
A: The same network is repeated every tick, and each tick's hidden state depends on the previous tick's. That's what gives it memory.

**Q: How exactly is P(Critical) computed?**
A: As the fraction of the 500 particles in which that service is in state Critical, after predicting, weighting by the Gaussian likelihoods of all 8 signals, and resampling.

**Q: Why a particle filter if 3⁸ is small?**
A: Parent coupling makes services dependent, and the design targets 20–30 services, where 3³⁰ joint states are infeasible. Particles scale linearly. The trade-off is randomness and degeneracy, which we measured.

**Q: Where do μ, σ and the transition matrix come from?**
A: They were calibrated from the recorded runs with weak labels from the fault schedule (Stage 08), not hand-picked.

**Q: Why three states?**
A: Normal and Critical alone make it a threshold again. Degrading represents the error interval (the fault is present, users not yet hurt), which is where early warning happens.

**Q: How does the graph enter the probability?**
A: Through the transition step: a child's next-state row is shifted by its worst parent's state. Honestly, the calibrated shift came out zero because of a bug, so in our results the graph mattered through root-cause analysis instead.

**Q: Why did the DBN not false-alarm when PREFACE did?**
A: Healthy spikes (around 2–6) are nearly impossible under the calibrated Critical emission (about 10.5 ± 0.8), and one spike cannot move a belief that must first transition out of Normal. A threshold has no such memory.
