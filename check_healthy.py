import pandas as pd

h = pd.read_csv('data/raw/healthy/phase3_healthy_telemetry_dataset.csv')
h = h[(h['kpi_name']=='cpu_usage') & (h['service_name']=='ts-train-service')]
print('ts-train-service cpu_usage stats in HEALTHY dataset:')
print(f'  min  = {h["value"].min():.6f}')
print(f'  max  = {h["value"].max():.6f}')
print(f'  mean = {h["value"].mean():.6f}')
print(f'  Values > 1000: {(h["value"] > 1000).sum()}')
print(f'  Total rows: {len(h)}')

# Check the healthy dataset - what are the large cpu values?
big = h[h['value'] > 1000]
print(f'\nRows with value > 1000 in healthy:')
print(big[['timestamp','pod_name','value']].to_string())
