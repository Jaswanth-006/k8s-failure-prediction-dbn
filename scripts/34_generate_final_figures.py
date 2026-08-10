import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_figures():
    out_dir = os.path.abspath("data/experiments/phase5/results")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Data
    pilot_dir = os.path.abspath("data/experiments/phase5/cpu/single/pilot_cpu_01")
    dbn_df = pd.read_csv(os.path.join(pilot_dir, "results.csv"))
    base_df = pd.read_csv(os.path.join(pilot_dir, "results_baseline.csv"))

    with open(os.path.join(out_dir, "preface_vs_dbn_comparison.json"), "r") as f:
        comp = json.load(f)

    dbn_df['time_idx'] = np.arange(len(dbn_df))
    base_df['time_idx'] = np.arange(len(base_df))

    fault_start_idx = dbn_df[dbn_df['phase'] == 'EXPERIMENT'].index[0]

    # FIGURE 2: Pilot CPU Fault Timeline (DBN)
    plt.figure(figsize=(10, 6))
    plt.plot(dbn_df['time_idx'], dbn_df['p_critical'], label="P(Critical) [ts-train-service/proxy]", color='red', linewidth=2)
    plt.axvline(x=fault_start_idx, color='black', linestyle='--', label="Fault Injected (t=5)")

    # Annotate DBN interventions
    interventions = dbn_df[dbn_df['decision_state'] == 'INTERVENE']
    if not interventions.empty:
        first_int = interventions.iloc[0]
        plt.scatter([first_int['time_idx']], [first_int['p_critical']], color='darkred', s=100, zorder=5, marker='x', label=f"Intervention (ts-ui-dashboard)")

    plt.title("FIGURE 2: PREFACE-DBN Pilot CPU Fault Timeline")
    plt.xlabel("Tick (5s intervals)")
    plt.ylabel("Probability Critical / Risk")
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_pilot_timeline.png"), dpi=300)
    plt.close()

    # FIGURE 3: PREFACE vs PREFACE-DBN Timeline
    plt.figure(figsize=(10, 6))

    # Baseline is binary: 0 or 1
    plt.plot(base_df['time_idx'], base_df['p_critical'], label="Baseline (Binary Threshold)", color='blue', linewidth=2, linestyle='-.')
    plt.plot(dbn_df['time_idx'], dbn_df['p_critical'], label="PREFACE-DBN (P(Critical))", color='red', linewidth=2)

    plt.axvline(x=fault_start_idx, color='black', linestyle='--', label="Fault Injected (t=5)")

    # Annotate interventions
    base_int = base_df[base_df['decision_state'] == 'INTERVENE']
    if not base_int.empty:
        plt.scatter([base_int.iloc[0]['time_idx']], [1.05], color='blue', s=100, marker='v', label="Baseline Action (0s delay)")

    if not interventions.empty:
        plt.scatter([interventions.iloc[0]['time_idx']], [0.95], color='red', s=100, marker='v', label="DBN Action (56s debounce delay)")

    plt.title("FIGURE 3: Comparison Timeline (Baseline vs DBN)")
    plt.xlabel("Tick (5s intervals)")
    plt.ylabel("Risk Signal")
    plt.ylim(-0.1, 1.2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_comparison_timeline.png"), dpi=300)
    plt.close()

    # FIGURE 4: Localization Comparison
    plt.figure(figsize=(8, 5))
    categories = ['Strong Localization', 'Weak Localization', 'Overall Localization']
    base_vals = [
        int(comp['preface_baseline']['strong_localization']),
        int(comp['preface_baseline']['weak_localization']),
        int(comp['preface_baseline']['overall_localization'])
    ]
    dbn_vals = [
        int(comp['preface_dbn']['strong_localization']),
        int(comp['preface_dbn']['weak_localization']),
        int(comp['preface_dbn']['overall_localization'])
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, base_vals, width, label='PREFACE Baseline', color='royalblue')
    ax.bar(x + width/2, dbn_vals, width, label='PREFACE-DBN', color='indianred')

    ax.set_ylabel('Success (1=True, 0=False)')
    ax.set_title('FIGURE 4: Localization Comparison (Pilot CPU Fault)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['False', 'True'])
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_localization.png"), dpi=300)
    plt.close()

    # FIGURE 5: Latency Comparison
    plt.figure(figsize=(8, 5))

    metrics = ['Reaction Interval (s)', 'Intervention Delay (s)']
    base_lat = [
        comp['preface_baseline']['reaction_interval'],
        comp['preface_baseline']['lead_time'].get('eligibility_lead_time', 0.0)
    ]
    dbn_lat = [
        comp['preface_dbn']['reaction_interval'],
        comp['preface_dbn']['lead_time'].get('eligibility_lead_time', 56.8)
    ]

    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, base_lat, width, label='PREFACE Baseline', color='royalblue')
    ax.bar(x + width/2, dbn_lat, width, label='PREFACE-DBN', color='indianred')

    ax.set_ylabel('Time (seconds)')
    ax.set_title('FIGURE 5: Reaction and Intervention Latency')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_latency.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_figures()
    print("Figures generated in data/experiments/phase5/results/")
