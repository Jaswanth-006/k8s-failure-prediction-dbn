"""
End-to-end operator test under a real fault.

Checks that the running operator notices a real fault, names the right service,
reaches an action decision, acts only in shadow mode, and settles again once the
fault is removed. It never talks to the model directly: everything it checks is
read from the FailurePredictor status the operator publishes, which is what a
user of the cluster would see.

Start the operator first, in shadow mode:

    PREFACE_TICK_SECONDS=60 python -m kopf run src/operator.py --standalone

then run:

    python scripts/42_operator_fault_test.py
    python scripts/42_operator_fault_test.py --target ts-route-service --out data/experiments/operator_test.json

Steps
-----
    1. Wait for the operator to tick, then record a few healthy ticks.
    2. Snapshot the target deployment.
    3. Inject CPU stress into the target with Chaos Mesh. The default target,
       ts-route-service, has no autoscaler, so the fault persists.
    4. Log every operator tick until an action is decided, plus a few more
       ticks, or until --max-fault-min.
    5. Remove the fault (always, even on error) and log recovery.
    6. Re-snapshot the deployment and evaluate pass/fail checks.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# kubectl ships with Docker Desktop on Windows; kind and helm were installed to
# ~/bin. Add them only if they exist, so the script also runs elsewhere.
for _dir in (r"C:\Program Files\Docker\Docker\resources\bin", os.path.expanduser("~/bin")):
    if os.path.isdir(_dir) and _dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _dir + os.pathsep + os.environ.get("PATH", "")

LATENCY_BUDGET_S = 5.0


def log(msg):
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def kubectl(args, stdin=None, check=True):
    r = subprocess.run(["kubectl"] + args, input=stdin, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError("kubectl %s: %s" % (" ".join(args), r.stderr.strip()))
    return r.stdout


def manifest(name, target):
    return """apiVersion: chaos-mesh.org/v1alpha1
kind: StressChaos
metadata:
  name: %s
  namespace: default
spec:
  mode: all
  selector:
    namespaces: [default]
    labelSelectors:
      app: %s
  stressors:
    cpu:
      workers: 2
      load: 80
  duration: '40m'
""" % (name, target)


def status(cr):
    return json.loads(kubectl(["get", "failurepredictor", cr, "-o", "json"])).get("status") or {}


def deploy_fingerprint(target):
    """Everything a live mutation would change: generation, replicas, restart stamp."""
    d = json.loads(kubectl(["get", "deploy", target, "-n", "default", "-o", "json"]))
    ann = d["spec"]["template"]["metadata"].get("annotations") or {}
    return {
        "generation": d["metadata"]["generation"],
        "replicas": d["spec"].get("replicas"),
        "restartedAt": ann.get("kubectl.kubernetes.io/restartedAt"),
    }


def tick_key(st):
    h = st.get("health") or {}
    return (h.get("lastSuccessfulTick"), h.get("lastError"), h.get("inferenceLatency"))


def wait_tick(cr, prev_key, timeout=240):
    start = time.time()
    while time.time() - start < timeout:
        st = status(cr)
        key = tick_key(st)
        if key != prev_key and any(v is not None for v in key):
            return st, key
        time.sleep(10)
    raise TimeoutError("operator produced no new tick in %ds" % timeout)


def make_row(st, phase, t_inject):
    risk = st.get("risk") or {}
    dec = st.get("decision") or {}
    per = st.get("persistence") or {}
    saf = st.get("safety") or {}
    h = st.get("health") or {}
    recent = st.get("recentActions") or []
    return {
        "phase": phase,
        "clock": datetime.now().strftime("%H:%M:%S"),
        "min_since_fault": None if t_inject is None else round((time.time() - t_inject) / 60.0, 1),
        "p_critical_root": risk.get("criticalProbability"),
        "p_critical_max": risk.get("current"),
        "root_cause": (st.get("rootCause") or {}).get("service"),
        "persistence": "%s/%s" % (per.get("currentTicks"), per.get("requiredTicks")),
        "action": dec.get("selectedAction"),
        "eligible": dec.get("interventionEligible"),
        "cooldown_s": saf.get("cooldownRemaining"),
        "shadow": saf.get("shadowMode"),
        "latency_s": h.get("inferenceLatency"),
        "error": h.get("lastError") or "",
        "latest_action": recent[0] if recent else "",
    }


def show(row):
    def num(v):
        return "%.3f" % v if v is not None else "-"
    log("%-8s t=%-5s P(crit) root=%-6s max=%-6s root=%-22s persist=%-6s action=%-15s eligible=%-5s lat=%-5s %s%s" % (
        row["phase"], row["min_since_fault"], num(row["p_critical_root"]), num(row["p_critical_max"]),
        row["root_cause"], row["persistence"], row["action"], row["eligible"], row["latency_s"],
        ("ERROR " + row["error"][:60]) if row["error"] else "",
        ("  latest: " + row["latest_action"]) if row["latest_action"] else ""))


def main():
    ap = argparse.ArgumentParser(description="End-to-end operator test under a real fault")
    ap.add_argument("--target", default="ts-route-service")
    ap.add_argument("--cr", default="train-ticket-predictor")
    ap.add_argument("--healthy-ticks", type=int, default=3)
    ap.add_argument("--max-fault-min", type=float, default=22.0)
    ap.add_argument("--extra-ticks", type=int, default=3,
                    help="ticks to keep the fault running after the action is decided")
    ap.add_argument("--max-recovery-min", type=float, default=10.0)
    ap.add_argument("--out", default="data/experiments/operator_test.json")
    args = ap.parse_args()

    chaos_name = "preface-operator-test-%s" % args.target.replace("ts-", "").replace("-service", "")
    rows = []
    kubectl(["delete", "stresschaos", chaos_name, "-n", "default", "--ignore-not-found"], check=False)

    before = deploy_fingerprint(args.target)
    log("%s before: %s" % (args.target, before))

    st = status(args.cr)
    key = tick_key(st)
    # Actions are detected by comparing entries rather than counting them: the
    # operator keeps only its 10 most recent actions, so once that list is full
    # a new action replaces an old one and the count never rises. Every entry
    # carries a timestamp, so a new action is always a new string.
    actions_before = set(st.get("recentActions") or [])

    log("waiting for the operator to tick...")
    st, key = wait_tick(args.cr, key)
    log("operator is ticking")

    for _ in range(args.healthy_ticks):
        st, key = wait_tick(args.cr, key)
        row = make_row(st, "healthy", None)
        rows.append(row)
        show(row)

    t_inject = time.time()
    first_action_tick = None
    try:
        kubectl(["apply", "-f", "-"], stdin=manifest(chaos_name, args.target))
        log("FAULT INJECTED: CPU stress on %s" % args.target)
        tick = 0
        while True:
            st, key = wait_tick(args.cr, key)
            tick += 1
            row = make_row(st, "fault", t_inject)
            rows.append(row)
            show(row)
            if first_action_tick is None and row["latest_action"] and row["latest_action"] not in actions_before:
                first_action_tick = tick
                log("ACTION DECIDED: %s" % row["latest_action"])
            if first_action_tick is not None and tick >= first_action_tick + args.extra_ticks:
                break
            if (time.time() - t_inject) / 60.0 > args.max_fault_min:
                log("fault window limit reached without an action")
                break
    finally:
        kubectl(["delete", "stresschaos", chaos_name, "-n", "default", "--ignore-not-found"], check=False)
        log("FAULT REMOVED")

    t_removed = time.time()
    calm = 0
    while (time.time() - t_removed) / 60.0 < args.max_recovery_min:
        st, key = wait_tick(args.cr, key)
        row = make_row(st, "recovery", t_inject)
        rows.append(row)
        show(row)
        calm = calm + 1 if (row["p_critical_max"] or 0.0) < 0.5 else 0
        if calm >= 3:
            log("recovered: P(Critical) below 0.5 on every service for 3 ticks")
            break

    after = deploy_fingerprint(args.target)
    log("%s after:  %s" % (args.target, after))

    fault = [r for r in rows if r["phase"] == "fault"]
    healthy = [r for r in rows if r["phase"] == "healthy"]
    recovery = [r for r in rows if r["phase"] == "recovery"]
    first_root = next((r for r in fault if r["root_cause"] == args.target), None)
    first_high = next((r for r in fault if (r["p_critical_root"] or 0) >= 0.95
                       and r["root_cause"] == args.target), None)
    first_eligible = next((r for r in fault if r["eligible"]), None)
    action_rows = [r for r in fault if r["latest_action"] and r["latest_action"] not in actions_before]

    checks = {
        "no inference errors": all(not r["error"] for r in rows),
        "latency within %.0fs budget" % LATENCY_BUDGET_S: all((r["latency_s"] or 0) <= LATENCY_BUDGET_S for r in rows),
        "healthy ticks quiet (P(Critical) max < 0.5)": all((r["p_critical_max"] or 0) < 0.5 for r in healthy),
        "root cause named as %s during fault" % args.target: first_root is not None,
        "P(Critical) on root >= 0.95 during fault": first_high is not None,
        "intervention became eligible": first_eligible is not None,
        "action decided on %s in shadow mode" % args.target: bool(action_rows)
            and args.target in action_rows[0]["latest_action"]
            and "WOULD_EXECUTE" in action_rows[0]["latest_action"],
        "shadow mode left deployment unchanged": before == after,
        "risk fell after fault removal": bool(recovery) and (recovery[-1]["p_critical_max"] or 0) < 0.5,
    }
    timings = {
        "root cause named (min after fault)": first_root and first_root["min_since_fault"],
        "P(Critical) >= 0.95 on root (min)": first_high and first_high["min_since_fault"],
        "intervention eligible (min)": first_eligible and first_eligible["min_since_fault"],
        "action decided (min)": action_rows[0]["min_since_fault"] if action_rows else None,
    }

    log("=" * 70)
    for name, ok in checks.items():
        log("%-4s %s" % ("PASS" if ok else "FAIL", name))
    for name, v in timings.items():
        log("     %-38s %s" % (name, v))
    log("RESULT: %d/%d checks passed" % (sum(checks.values()), len(checks)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"target": args.target, "before": before, "after": after,
                   "checks": checks, "timings": timings, "rows": rows}, fh, indent=2)
    log("wrote %s" % args.out)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
