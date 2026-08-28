import pandas as pd
import json

df = pd.read_csv('data/experiments/goal1_clean_experiment_results.csv')
with open('data/experiments/goal1_metadata.json') as f:
    meta = json.load(f)

t_fault = pd.to_datetime(meta['fault_injection_time'])
df['timestamp'] = pd.to_datetime(df['timestamp'])

print('=== METADATA ===')
print('Fault injection time:', t_fault)

print()
print('=== ALL ts-train-service TICKS ===')
train_ticks = df[df['root_cause'] == 'ts-train-service']
for _, r in train_ticks.iterrows():
    delta = (r['timestamp'] - t_fault).total_seconds()
    print(f"Tick {r['tick']:3d} [{r['phase']}]  ts={r['timestamp']}  delta={delta:.2f}s  sig={r['ts_train_signal']:.4f}  p_crit={r['p_critical']:.4f}  persistence={r['persistence']}")

print()
print('=== RECOVERY: last tick ===')
r = df.iloc[-1]
print(f"Tick {r['tick']}  [{r['phase']}]  p_crit={r['p_critical']:.4f}  ts={r['timestamp']}")

print()
print('=== TICKS 56-60 ===')
for _, r in df[df['tick'].isin([56, 57, 58, 59, 60])].iterrows():
    delta = (r['timestamp'] - t_fault).total_seconds()
    print(f"Tick {r['tick']:3d} [{r['phase']}]  ts={r['timestamp']}  delta={delta:.2f}s  sig={r['ts_train_signal']:.4f}  p={r['p_critical']:.4f}  rc={r['root_cause']}  pers={r['persistence']}")

print()
print('=== FIRST EXPERIMENT-PHASE tick ===')
exp = df[df['phase'] == 'EXPERIMENT']
if len(exp):
    r = exp.iloc[0]
    delta = (r['timestamp'] - t_fault).total_seconds()
    print(f"Tick {r['tick']}  ts={r['timestamp']}  delta={delta:.2f}s from fault")

print()
print('=== ALL ticks with p_critical > 0.5 ===')
for _, r in df[df['p_critical'] > 0.5].iterrows():
    delta = (r['timestamp'] - t_fault).total_seconds()
    print(f"Tick {r['tick']:3d} [{r['phase']}]  ts={r['timestamp']}  delta={delta:.2f}s  p_crit={r['p_critical']:.4f}  rc={r['root_cause']}")

print()
print('=== Evaluator trace: Adaptive persistence simulation ===')
persistence_counter = 0
current_rc = "None"
for _, row in df.iterrows():
    rc = row['root_cause']
    p_crit = row['p_critical']
    signal = row['ts_train_signal']
    tick = row['tick']
    phase = row['phase']
    ts = row['timestamp']

    if rc == "None" or rc != "ts-train-service":
        persistence_counter = 0
        current_rc = "None"
    else:
        current_rc = rc
        persistence_counter += 1

    req_ticks = 11
    if signal >= 8.0 and p_crit >= 0.9:
        req_ticks = 2
    elif signal >= 4.0 and p_crit >= 0.6:
        req_ticks = 5

    intervene = persistence_counter >= req_ticks
    if intervene:
        delta = (ts - t_fault).total_seconds()
        print(f"ADAPTIVE INTERVENE at Tick {tick} [{phase}]  ts={ts}  delta={delta:.2f}s  sig={signal:.4f}  p={p_crit:.4f}  pers={persistence_counter}/{req_ticks}")
        break
    if rc == "ts-train-service":
        print(f"  tracking Tick {tick} [{phase}]  ts={ts}  sig={signal:.4f}  p={p_crit:.4f}  pers={persistence_counter}/{req_ticks}")
