import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import networkx as nx

class MetricsEvaluator:
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.results_path = os.path.join(exp_dir, "results.csv")
        self.metadata_path = os.path.join(exp_dir, "metadata.json")
        self.df = None
        self.metadata = None

        # Build topological graph to determine proxies (Weak Localization)
        self.G = nx.DiGraph()
        self.G.add_edge("ts-ui-dashboard", "ts-user-service")
        self.G.add_edge("ts-ui-dashboard", "ts-train-service")
        self.G.add_edge("ts-ui-dashboard", "ts-route-service")
        self.G.add_edge("ts-ui-dashboard", "ts-order-service")
        self.G.add_edge("ts-order-service", "ts-payment-service")
        self.G.add_edge("ts-order-service", "ts-inventory-service")
        self.G.add_edge("ts-order-service", "ts-station-service")

    def load(self):
        if not os.path.exists(self.results_path) or not os.path.exists(self.metadata_path):
            return False
        self.df = pd.read_csv(self.results_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)
        return True

    def _get_proxies(self, target_service):
        proxies = set()
        if self.G.has_node(target_service):
            proxies.update(self.G.predecessors(target_service))
            proxies.update(self.G.successors(target_service))
        return proxies

    def evaluate(self):
        targets = self.metadata.get("target_services", [])

        # T0: Telemetry start
        t0 = self.df.iloc[0]['timestamp']

        # Tf_start: Experiment phase start
        exp_df = self.df[self.df['phase'] == 'EXPERIMENT']
        if exp_df.empty:
            return {"validity": False, "notes": "No EXPERIMENT phase found."}
        tf_start = exp_df.iloc[0]['timestamp']

        # Tf_end: Recovery phase start
        rec_df = self.df[self.df['phase'] == 'RECOVERY']
        tf_end = rec_df.iloc[0]['timestamp'] if not rec_df.empty else None

        # Look only at data from Tf_start onwards for reaction/localization
        # A detection is when p_critical > 0.0 (an anomaly is detected by DDN)
        post_fault_df = self.df[self.df['timestamp'] >= tf_start]
        detections = post_fault_df[post_fault_df['p_critical'] > 0.0]
        td = detections.iloc[0]['timestamp'] if not detections.empty else None

        # Root cause localization (Trc)
        trc = None
        localization_type = "None"
        strong = False
        weak = False

        # Check for Strong and Weak Localization
        for _, row in post_fault_df.iterrows():
            rc = row['root_cause']
            if rc == "None":
                continue

            # Since target could be multiple (dual topology), check any target
            is_strong = any(rc == target for target in targets)
            is_weak = any(rc in self._get_proxies(target) for target in targets)

            if is_strong or is_weak:
                trc = row['timestamp']
                strong = is_strong
                weak = is_weak
                localization_type = "Strong" if strong else "Weak"
                break

        # Eligibility (Te)
        interventions = post_fault_df[post_fault_df['decision_state'] == 'INTERVENE']
        te = interventions.iloc[0]['timestamp'] if not interventions.empty else None

        # Metrics Calculations
        reaction_interval = (trc - tf_start).total_seconds() if trc and tf_start else None

        # Earliness Interval
        # Because we lack Locust HTTP percentiles in this telemetry (baseline criteria),
        # we cannot compute the absolute 'disruptive failure' timestamp (T_disrupt).
        earliness_interval = None
        earliness_percentage = None

        # Lead times
        detection_lead_time = (td - tf_start).total_seconds() if td else None
        localization_lead_time = reaction_interval
        eligibility_lead_time = (te - tf_start).total_seconds() if te else None

        return {
            "experiment_id": self.metadata.get("experiment_id"),
            "fault_type": self.metadata.get("fault_type"),
            "topology": self.metadata.get("topology"),
            "target_services": targets,
            "T0": t0.isoformat() if t0 else None,
            "Tf_start": tf_start.isoformat() if tf_start else None,
            "Tf_end": tf_end.isoformat() if tf_end else None,
            "Td": td.isoformat() if td else None,
            "Trc": trc.isoformat() if trc else None,
            "Te": te.isoformat() if te else None,
            "Trecovery": tf_end.isoformat() if tf_end else None, # Assuming recovery starts tracking at Tf_end
            "reaction_interval": reaction_interval,
            "earliness_interval": earliness_interval,
            "earliness_percentage": earliness_percentage,
            "strong_localization": strong,
            "weak_localization": weak,
            "overall_localization": strong or weak,
            "localization_type": localization_type,
            "risk_metrics": "Calibration analysis not yet statistically supported.",
            "lead_time": {
                "detection_lead_time": detection_lead_time,
                "localization_lead_time": localization_lead_time,
                "eligibility_lead_time": eligibility_lead_time
            },
            "data_quality": "Valid. Note: Earliness is None due to lack of HTTP disruption metric.",
            "validity": True,
            "notes": "Processed successfully."
        }
