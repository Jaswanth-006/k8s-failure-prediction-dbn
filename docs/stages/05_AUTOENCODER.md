# Stage 05: The Autoencoder (How Unusual Does Each Service Look?)

> A neural network learns, from healthy data only, to reproduce the 63-number vector. When a service misbehaves, the network reproduces that service's numbers badly. We turn "how badly" into one **anomaly signal per service**: 0 means normal, and around 10 means extremely unusual.

---

## 1. In simple words

Picture an art student who has practised copying only **healthy** plant drawings, and must copy each drawing through a very small sketchbook that forces them to remember only the essentials. Show them a healthy plant and the copy is nearly perfect. Show them a plant with a strange disease and the copy comes out wrong, because they never learned that pattern.

The autoencoder is that student. It:
1. squeezes the 63 numbers down to 7 (the small sketchbook),
2. expands them back to 63,
3. and we measure how different the copy is from the original.

A big difference on the 7 numbers belonging to `ts-route-service` means **route looks unusual**.

## 2. Why this stage exists

- **No labels needed.** We don't have to collect or label failures to train it. Healthy data is plentiful.
- **It learns relationships.** When traffic rises, many services' CPU rises together. That's normal, and the network learns it. One service's CPU rising *alone* breaks the learned pattern.
- **Per-service output.** Scoring each service separately tells the DBN *where* the unusual behaviour is, not just *that* something is unusual.

This is PREFACE's second core idea. We keep it, but instead of thresholding the score immediately, we pass a continuous score to the DBN.

---

## 3. Step by step

### Training (once)

```
Step 1  Collect healthy telemetry     python scripts/40_collect_healthy.py --minutes 25 --interval 15
        -> data/raw/healthy/phase3_healthy_telemetry_dataset.csv   (100 ticks)
        (no faults running; load generator on; autoscaler active)
Step 2  Rectify each timestamp        100 × 63 matrix  (Stage 04)
Step 3  Split in time order           first 80 ticks train, last 20 validate
Step 4  Fit robust scaling on train   median and IQR of each of the 63 features
Step 5  Train the network             50 epochs, Adam, learning rate 0.001, batch 64, MSE loss
Step 6  Measure healthy error         for each service: mean and std of its reconstruction error on train
Step 7  Save                          models/phase3_autoencoder_cpu_only.pth
        (weights + median + IQR + per-service error mean/std + feature names)
```

Command: `python scripts/20_audit_and_train_cpu_only.py`

### Scoring (every tick)

```
x_t (63 numbers)
  -> normalise:      z_x = (x_t − median) / IQR
  -> reconstruct:    x̂  = decoder(encoder(z_x))
  -> squared error:  e_i = (z_x,i − x̂_i)²                 for each of the 63 features
  -> per service:    E_s = mean of e_i over that service's 7 features
  -> standardise:    z_s = (E_s − healthy_mean_s) / healthy_std_s
  -> compress:       a_s = log(1 + max(0, z_s))           <- the anomaly signal
Output: {service: a_s} for the 8 services
```

---

## 4. Technical depth

### 4.1 Architecture

`DeepAutoencoder(input_dim=63)` in `src/autoencoder_phase3.py`:

```
Encoder:  Linear(63→31) → ReLU → Linear(31→15) → ReLU → Linear(15→7) → ReLU
Decoder:  Linear(7→15)  → ReLU → Linear(15→31) → ReLU → Linear(31→63)       (linear output)
```

Layer sizes follow n → n/2 → n/4 → n/8 with minimums of 16, 8 and 4: 63 // 2 = 31, 63 // 4 = 15, 63 // 8 = 7. The saved weights confirm the shapes (31×63, 15×31, 7×15, 15×7, 31×15, 63×31).

- **The bottleneck (7 neurons)** forces the network to learn the main structure of healthy behaviour instead of memorising the input.
- **ReLU** lets it learn non-linear relationships.
- **Linear output**, because normalised values can be negative or large, and a squashing function would clip them.

### 4.2 Robust normalisation (median / IQR)

```
z_x = (x − median_train) / IQR_train,    IQR = Q3 − Q1
```

We use this instead of mean and standard deviation because Kubernetes metrics are skewed and spiky: one outlier in the training data would inflate a standard deviation. Median and IQR ignore outliers. If a feature never varies (IQR = 0, e.g. `count` for a service that always had 1 pod), IQR is set to 1 to avoid dividing by zero.

### 4.3 Training objective

Mean squared error between input and reconstruction:

```
L = (1/N) Σ_samples (1/63) Σ_i (z_x,i − x̂_i)²
```

minimised with Adam (lr = 1e-3), 50 passes over the 80 training ticks, in shuffled batches of 64.

### 4.4 From reconstruction error to a per-service score

Each feature belongs to a service through its name prefix (`ts-route-service.cpu_usage.max` → route).

1. **Per-service error** `E_s`: the mean squared error over the service's 7 features.
2. **Standardise against healthy behaviour.** Each service has a different normal error level (a busy service is harder to reconstruct), so we compare it to *its own* healthy error:
   ```
   z_s = (E_s − μ_s) / σ_s
   ```
   μ_s and σ_s were measured on the training ticks after training. For example `ts-route-service`: μ = 0.090, σ = 0.114.
3. **Log compression:** `a_s = log1p(max(0, z_s))`.

> The `node_pool.node_cpu.*` features have no service prefix. They help the network reconstruct everything else (machine load explains a lot), but they count toward **no** service's score.

### 4.5 Why `log1p`?

During a CPU fault, z reaches **tens of thousands**. The DBN models signals with Gaussians around values like 0, 7 and 10; it cannot use numbers like 42,616 directly. `log1p` keeps the order of values but compresses the scale:

| z (standard deviations above healthy) | anomaly signal a = log(1+z) |
|---|---|
| 0 (or below) | 0 |
| 1.72 | 1.0 |
| **3** (PREFACE's rule) | **1.386** |
| 11.2 | 2.5 |
| 19.1 | 3.0 |
| 147 | 5.0 |
| 22,025 | 10.0 |
| 42,616 | 10.66 (route fault) |

Negative z (better than normal) is clamped to 0, because "unusually healthy" isn't a problem. The DBN also clips signals to [0, 15].

**This table is where our biggest correction came from.** PREFACE alarms at z > 3, which is signal > **1.386**. An earlier version of our comparison used 3.0, which is z > 19, and wrongly concluded the DBN had no advantage (Stage 12).

---

## 5. Real example from our data

**Route-service CPU fault, tick 9** (second recording):

1. The route pod's CPU goes from about 0.004 cores to its 0.150 limit.
2. The healthy median of `ts-route-service.cpu_usage.mean` is 0.00375 and its IQR is 0.00127. So 0.150 normalises to **(0.150 − 0.00375) / 0.00127 ≈ 115**. The network never saw anything like this, so it cannot reproduce it.
3. Route's mean squared error over its 7 features: **about 4,854**. Healthy: 0.090 ± 0.114.
4. z = (4854 − 0.090) / 0.114 ≈ **42,616**.
5. a = log(1 + 42,616) = **10.66**.

Same tick, other services: dashboard 3.16, train 1.79, the rest below 2. The fault clearly stands out.

**Healthy per-service error levels** (μ_s, from training): dashboard 0.036, order 0.068, payment 0.071, user 0.074, route 0.090, train 0.123, inventory 0.130, station 0.141.

**The under-dispersion problem.** On the 20 held-out healthy validation ticks, the highest signal was **1.65**. On the live healthy runs recorded days later, peaks were **2.23 to 6.47**, and in one healthy run 16 of 28 ticks exceeded 1.386. Healthy live data varies more than the 25-minute training set captured. This is why a fixed 3σ threshold (PREFACE) false-alarms constantly on our cluster, while the DBN, whose "Normal" state was calibrated on the live runs, does not.

---

## 6. Where it lives in the code

| What | File |
|---|---|
| Network + scoring pipeline | `src/autoencoder_phase3.py` (`DeepAutoencoder`, `RobustAnomalyScorePipeline`) |
| Healthy data collection | `scripts/40_collect_healthy.py` |
| Training + validation report | `scripts/20_audit_and_train_cpu_only.py` |
| Trained model | `models/phase3_autoencoder_cpu_only.pth` (gitignored) |
| Older mean/std version (not used live) | `src/autoencoder.py` |

## 7. Limits and known issues

- **Small, correlated training set.** 100 ticks from 25 minutes, sampled every 15 s with a 2-minute rate window, so neighbouring samples overlap. This is the main cause of under-dispersion.
- **Per-service analogue of PREFACE.** The paper thresholds the *global* reconstruction error and then ranks services. Our baseline thresholds the *highest per-service* signal, because recorded runs store only per-service signals. (`src/preface_baseline.py` implements the global version but could not be replayed on the recordings.)
- **CPU only**, as in Stage 04.
- **Train/validation split in time order**, which is correct for time series, but the validation block is only 20 ticks.

## 8. Questions a reviewer may ask

**Q: Why train only on healthy data?**
A: We want "what normal looks like". Anything it reconstructs badly is, by definition, unlike normal. It also means no failure labels are needed.

**Q: What is the bottleneck size and why?**
A: 7 neurons (63 // 8). A narrow middle layer forces the network to learn structure. If the middle were as wide as the input, it could copy anything, including anomalies.

**Q: Why median/IQR instead of mean/std normalisation?**
A: Robustness. Kubernetes metrics have outliers and skew. An outlier inflates a standard deviation but barely moves an IQR.

**Q: Why the log?**
A: Fault z-scores are in the tens of thousands. The log compresses them into a range the DBN's Gaussians can model, while keeping the order. PREFACE's z > 3 becomes signal > 1.386.

**Q: Isn't the autoencoder alone enough?**
A: That is PREFACE. On our cluster, healthy signals regularly exceed its 3σ line, giving false alarms on 6 of 6 healthy runs. The DBN turns the same signals into a persistent belief and raised 0 false alarms.
