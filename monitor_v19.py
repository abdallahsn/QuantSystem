"""
monitor_v19.py - Monitoring runner for QuantSystem V19 JSONL event logs
"""

from __future__ import annotations

import argparse
import json

from modules.config_v19 import load_v19_config
from modules.monitoring_v19 import MonitoringState, emit_alerts, load_baseline_from_artifacts, load_jsonl, write_monitoring_outputs


def main():
    p = argparse.ArgumentParser(description='QuantSystem V19 monitoring runner')
    p.add_argument('--events', required=True)
    p.add_argument('--models', default='outputs_v19')
    p.add_argument('--output', default='outputs_v19_monitoring')
    p.add_argument('--config', default=None)
    args = p.parse_args()

    cfg = load_v19_config(args.config)
    baseline = load_baseline_from_artifacts(args.models)
    events = load_jsonl(args.events)
    summary = MonitoringState(baseline=baseline).summarize(events)
    alerts = emit_alerts(summary, writer=None, config=cfg.get('monitoring', {}))
    paths = write_monitoring_outputs(args.output, summary, alerts)
    print(json.dumps({**paths, 'events': len(events)}, indent=2))


if __name__ == '__main__':
    main()
