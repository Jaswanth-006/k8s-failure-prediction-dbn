import os
import sys
import json
import time
import argparse
import subprocess
import requests
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference_adapter import InferenceAdapter
from src.decision_policy import DecisionPolicy

PROMETHEUS_URL = "http://localhost:9090"

def get_git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except:
        return "unknown"

def check_prereqs():
    try:
        res = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=2)
        if res.status_code != 200:
            raise Exception("Status code not 200")
    except Exception as e:
        print(f"Prometheus unreachable: {e}")
        sys.exit(1)

def inject_cpu_fault(services):
    for svc in services:
        print(f"Injecting CPU fault on {svc}...")
        subprocess.run(["kubectl", "exec", f"deployment/{svc}", "--", "sh", "-c", "yes > /dev/null & yes > /dev/null &"], shell=False)

def remove_cpu_fault(services):
    for svc in services:
        print(f"Removing CPU fault on {svc}...")
        subprocess.run(["kubectl", "exec", f"deployment/{svc}", "--", "pkill", "-f", "yes"], shell=False, stderr=subprocess.DEVNULL)

def inject_network_fault(services):
    for svc in services:
        yaml_path = f"manifests/chaos/network-delay-{svc.replace('ts-', '').replace('-service', '')}.yaml"
        if os.path.exists(yaml_path):
            print(f"Applying {yaml_path}")
            subprocess.run(["kubectl", "apply", "-f", yaml_path], shell=False)
        else:
            print(f"WARNING: No predefined network chaos manifest for {svc}. Network fault not injected.")

def remove_network_fault(services):
    for svc in services:
        yaml_path = f"manifests/chaos/network-delay-{svc.replace('ts-', '').replace('-service', '')}.yaml"
        if os.path.exists(yaml_path):
            print(f"Deleting {yaml_path}")
            subprocess.run(["kubectl", "delete", "-f", yaml_path], shell=False)

def main():
    parser = argparse.ArgumentParser(description="Phase 5 Experiment Runner")
    parser.add_argument("--fault-type", type=str, required=True, choices=["cpu", "network", "memory"])
    parser.add_argument("--topology", type=str, required=True, choices=["single", "dual"])
    parser.add_argument("--target-services", type=str, required=True, help="Comma separated services")
    parser.add_argument("--experiment-id", type=str, required=True)
    args = parser.parse_args()

    services = args.target_services.split(",")
    if args.topology == "single" and len(services) != 1:
        print("Error: topology 'single' requires exactly 1 target service")
        sys.exit(1)
    if args.topology == "dual" and len(services) != 2:
        print("Error: topology 'dual' requires exactly 2 target services")
        sys.exit(1)

    if args.fault_type == "memory":
        print("ERROR: Memory fault mechanism is not safe or reproducible in the current repository.")
        print("Aborting experiment to prevent cluster destabilization.")
        sys.exit(1)

    out_dir = os.path.abspath(f"data/experiments/phase5/{args.fault_type}/{args.topology}/{args.experiment_id}")
    os.makedirs(out_dir, exist_ok=True)

    check_prereqs()

    adapter = InferenceAdapter(prometheus_url=PROMETHEUS_URL)
    policy = DecisionPolicy(debounce_ticks=11, cooldown_seconds=300)
    config_dict = {"shadow_mode": True, "enabled_actions": ["Do_Nothing", "Reschedule_Pod", "Restart_Pod", "Scale_Out", "Traffic_Shift"]}

    results = []
    event_log = []

    def log_event(event):
        msg = f"[{datetime.now().isoformat()}] {event}"
        print(msg)
        event_log.append(msg)

    def run_ticks(phase, num_ticks):
        for i in range(num_ticks):
            print(f"[{phase}] Tick {i+1}/{num_ticks}...")

            ddn_output = adapter.run_tick()
            decision = policy.evaluate(ddn_output, config_dict)

            # If intervention is triggered, record action to trigger cooldown properly
            if decision.get("state") == "INTERVENE":
                policy.record_action(decision.get("root_cause"))

            rc = decision.get("root_cause", "None")
            p_crit = 0.0
            if rc != "None":
                p_crit = ddn_output.get("posteriors", {}).get(rc, {}).get("Critical", 0.0)

            results.append({
                "timestamp": datetime.now().isoformat(),
                "phase": phase,
                "tick": len(results) + 1,
                "root_cause": rc,
                "p_critical": p_crit,
                "persistence": decision.get("persistence_count", 0),
                "decision_state": decision.get("state", "HEALTHY"),
                "selected_action": decision.get("action", "Do_Nothing"),
                "delta_eu": decision.get("delta_eu", 0.0)
            })
            time.sleep(5)

    # Clean state before starting
    if args.fault_type == "cpu":
        remove_cpu_fault(services)
    elif args.fault_type == "network":
        remove_network_fault(services)

    log_event("--- EXPERIMENT START ---")
    log_event(f"Pre-fault phase starting (5 ticks). Target: {services}")
    run_ticks("PRE-FAULT", 5)

    log_event("--- INJECTING FAULT ---")
    if args.fault_type == "cpu":
        inject_cpu_fault(services)
    elif args.fault_type == "network":
        inject_network_fault(services)

    log_event("Fault injected. Experiment phase starting (40 ticks).")
    run_ticks("EXPERIMENT", 40)

    log_event("--- REMOVING FAULT ---")
    if args.fault_type == "cpu":
        remove_cpu_fault(services)
    elif args.fault_type == "network":
        remove_network_fault(services)

    log_event("Fault removed. Recovery phase starting (20 ticks).")
    run_ticks("RECOVERY", 20)
    log_event("--- EXPERIMENT COMPLETE ---")

    # Save Artifacts
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

    with open(os.path.join(out_dir, "events.log"), "w") as f:
        f.write("\n".join(event_log))

    metadata = {
        "experiment_id": args.experiment_id,
        "timestamp": datetime.now().isoformat(),
        "git_sha": get_git_sha(),
        "fault_type": args.fault_type,
        "topology": args.topology,
        "target_services": services,
        "shadow_mode": True,
        "protocol": {
            "pre_fault_ticks": 5,
            "fault_ticks": 40,
            "recovery_ticks": 20,
            "telemetry_interval_sec": 5
        }
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nSaved artifacts to {out_dir}")

if __name__ == "__main__":
    main()
