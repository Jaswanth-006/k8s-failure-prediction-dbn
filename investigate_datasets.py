import pandas as pd

# Check healthy dataset
healthy = pd.read_csv('data/raw/healthy/phase3_healthy_telemetry_dataset.csv')
healthy['timestamp'] = pd.to_datetime(healthy['timestamp'])
ts = sorted(healthy['timestamp'].unique())
print('=== HEALTHY DATASET ===')
print(f'Shape: {healthy.shape}')
print(f'Unique timestamps: {len(ts)}')
print(f'Range: {ts[0]} -> {ts[-1]}')
print(f'Duration: {(ts[-1]-ts[0]).total_seconds():.1f}s')
print(f'Services: {sorted(healthy["service_name"].unique())}')
print(f'kpi_names: {sorted(healthy["kpi_name"].unique())}')
train_h = healthy[
    (healthy['service_name'] == 'ts-train-service') &
    (healthy['kpi_name'] == 'cpu_usage')
]
print(f'ts-train max CPU in healthy: {train_h["value"].max():.6f}')
print(f'ts-train mean CPU in healthy: {train_h["value"].mean():.6f}')

print()

# Fault dataset
fault = pd.read_csv('data/raw/faults/cpu_train_service_telemetry_dataset.csv')
fault['timestamp'] = pd.to_datetime(fault['timestamp'])
ts_f = sorted(fault['timestamp'].unique())
print('=== FAULT DATASET ===')
print(f'Shape: {fault.shape}')
print(f'Unique timestamps: {len(ts_f)}')
print(f'Range: {ts_f[0]} -> {ts_f[-1]}')
print(f'Duration: {(ts_f[-1]-ts_f[0]).total_seconds():.1f}s')
print(f'Columns: {list(fault.columns)}')

# Does fault dataset have a label column?
print(f'Columns: {list(fault.columns)}')

# ============================================================
# CPU-ONLY FAULT DATASET ANALYSIS
# ============================================================

print('\n=== FAULT DATASET: CPU-ONLY ANALYSIS ===')

# train_f already contains ONLY:
# service_name = ts-train-service
# kpi_name = cpu_usage
train_f = fault[
    (fault['service_name'] == 'ts-train-service') &
    (fault['kpi_name'] == 'cpu_usage')
].copy()


train_f = train_f.sort_values('timestamp').copy()

print(f'Number of CPU rows: {len(train_f)}')

print(f'Minimum CPU: {train_f["value"].min():.6f}')
print(f'Maximum CPU: {train_f["value"].max():.6f}')
print(f'Mean CPU: {train_f["value"].mean():.6f}')
print(f'Median CPU: {train_f["value"].median():.6f}')
print(f'Std CPU: {train_f["value"].std():.6f}')


# ------------------------------------------------------------
# First 20 CPU observations
# ------------------------------------------------------------

print('\n=== FIRST 20 CPU TICKS ===')

for i, (_, row) in enumerate(train_f.head(20).iterrows(), start=1):

    print(
        f'Tick {i:3d}: '
        f'{row["timestamp"]}  '
        f'CPU = {row["value"]:.6f}'
    )


# ------------------------------------------------------------
# Last 20 CPU observations
# ------------------------------------------------------------

print('\n=== LAST 20 CPU TICKS ===')

start_tick = max(1, len(train_f) - 19)

for i, (_, row) in enumerate(
    train_f.tail(20).iterrows(),
    start=start_tick
):

    print(
        f'Tick {i:3d}: '
        f'{row["timestamp"]}  '
        f'CPU = {row["value"]:.6f}'
    )


# ------------------------------------------------------------
# Compare fault CPU against healthy CPU baseline
# ------------------------------------------------------------

healthy_cpu = healthy[
    (healthy['service_name'] == 'ts-train-service') &
    (healthy['kpi_name'] == 'cpu_usage')
]['value']

healthy_mean = healthy_cpu.mean()
healthy_std = healthy_cpu.std()

print('\n=== COMPARISON WITH HEALTHY CPU BASELINE ===')

print(f'Healthy CPU mean: {healthy_mean:.6f}')
print(f'Healthy CPU std : {healthy_std:.6f}')

print('\nFirst 20 fault CPU values and healthy-baseline z-score:')

for i, (_, row) in enumerate(
    train_f.head(20).iterrows(),
    start=1
):

    cpu = row['value']

    if healthy_std > 0:
        z = (cpu - healthy_mean) / healthy_std
    else:
        z = float('inf')

    print(
        f'Tick {i:3d}: '
        f'CPU = {cpu:.6f}, '
        f'z = {z:.2f}'
    )
# Show first healthy ticks CPU
print('\nFirst 5 raw ticks (showing ALL pod values for ts-train):')
for i in range(min(5, len(ts_f))):
    ts = ts_f[i]
    rows = train_f[train_f['timestamp']==ts][['pod_name','value']].values.tolist()
    print(f'  tick {i+1}: {rows}')
