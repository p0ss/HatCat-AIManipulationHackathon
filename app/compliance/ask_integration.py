"""
ASK Audit System Integration and Compliance Report Generation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from .eu_mapping import EU_AI_ACT_MAPPING, get_compliance_summary


def export_audit_json(results: Dict[str, Any], output_path: Path) -> None:
    """
    Export evaluation results as ASK-compatible audit JSON.
    """
    audit_doc = {
        "schema_version": "ftw.audit.v0.3",
        "export_type": "evaluation_results",
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "deployment_id": "hackathon-manipulation-demo",
        "run_id": results.get("run_id", "unknown"),
        "conditions": results.get("conditions", []),
        "entries": [],
        "summary": results.get("summary", {}),
    }

    # Convert episodes to audit-like entries
    for ep in results.get("episodes", []):
        for cond, data in ep.get("conditions", {}).items():
            entry = {
                "entry_id": f"eval_{ep['episode_id']}_{cond}",
                "episode_id": ep["episode_id"],
                "behavior": ep["behavior"],
                "condition": cond,
                "signals": {
                    "manipulation_detected": data.get("manipulation_detected", False),
                    "peak_detection_score": data.get("peak_detection_score", 0),
                },
                "actions": {
                    "interventions_triggered": data.get("interventions", 0),
                    "steering_corrections": data.get("corrections", 0),
                },
            }
            audit_doc["entries"].append(entry)

    with open(output_path, "w") as f:
        json.dump(audit_doc, f, indent=2)


def generate_compliance_report(results: Dict[str, Any], output_path: Path) -> None:
    """
    Generate HTML compliance report for EU AI Act.
    """
    summary = results.get("summary", {})
    by_condition = summary.get("by_condition", {})
    compliance_summary = get_compliance_summary()

    # Calculate key metrics
    baseline_rate = by_condition.get("A", {}).get("rate", 0)
    harness_rate = by_condition.get("C", {}).get("rate", 0)

    if baseline_rate > 0:
        reduction = ((baseline_rate - harness_rate) / baseline_rate) * 100
    else:
        reduction = 0

    # Build article rows
    article_rows = []
    for article, data in EU_AI_ACT_MAPPING.items():
        for req_name, req_data in data.get("requirements", {}).items():
            status_class = "pass" if req_data.get("status") == "implemented" else "pending"
            status_text = "Implemented" if status_class == "pass" else "Pending"
            article_rows.append(f"""
                <tr>
                    <td>{article}</td>
                    <td>{req_name}</td>
                    <td>{req_data.get('component', 'N/A')}</td>
                    <td class="{status_class}">{status_text}</td>
                </tr>
            """)

    # Build results rows
    results_rows = []
    for behavior, rates in summary.get("by_behavior", {}).items():
        a_rate = rates.get("A", {}).get("rate", 0)
        c_rate = rates.get("C", {}).get("rate", 0)
        if a_rate > 0:
            beh_reduction = ((a_rate - c_rate) / a_rate) * 100
        else:
            beh_reduction = 0

        behavior_label = behavior.replace("_", " ").title()
        results_rows.append(f"""
            <tr>
                <td>{behavior_label}</td>
                <td>{a_rate:.1f}%</td>
                <td>{c_rate:.1f}%</td>
                <td class="{'pass' if beh_reduction > 0 else ''}">{beh_reduction:.1f}%</td>
            </tr>
        """)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EU AI Act Compliance Report</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f8fafc;
            color: #1e293b;
            line-height: 1.6;
        }}
        h1 {{
            color: #0f172a;
            border-bottom: 3px solid #3b82f6;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #1e40af;
            margin-top: 40px;
        }}
        .header {{
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            border: none;
            color: white;
        }}
        .header p {{
            margin: 10px 0 0;
            opacity: 0.9;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .card h3 {{
            margin: 0 0 10px;
            color: #64748b;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #0f172a;
        }}
        .card.success .value {{
            color: #10b981;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #e2e8f0;
            padding: 12px 16px;
            text-align: left;
        }}
        th {{
            background: #1e40af;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background: #f8fafc;
        }}
        .pass {{
            color: #10b981;
            font-weight: bold;
        }}
        .pending {{
            color: #f59e0b;
        }}
        .evidence {{
            background: #f1f5f9;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
            overflow-x: auto;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            color: #64748b;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>EU AI Act Compliance Report</h1>
        <p>AI Manipulation Detection & Mitigation System</p>
        <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    </div>

    <h2>Executive Summary</h2>
    <div class="summary-cards">
        <div class="card">
            <h3>Episodes Tested</h3>
            <div class="value">{len(results.get('episodes', []))}</div>
        </div>
        <div class="card">
            <h3>Baseline Manipulation Rate</h3>
            <div class="value">{baseline_rate:.1f}%</div>
        </div>
        <div class="card success">
            <h3>With HUSH Harness</h3>
            <div class="value">{harness_rate:.1f}%</div>
        </div>
        <div class="card success">
            <h3>Reduction Achieved</h3>
            <div class="value">{reduction:.0f}%</div>
        </div>
    </div>

    <h2>EU AI Act Article Compliance</h2>
    <p>Compliance rate: <strong class="pass">{compliance_summary['compliance_rate']}%</strong> ({compliance_summary['implemented']}/{compliance_summary['total_requirements']} requirements)</p>
    <table>
        <tr>
            <th>Article</th>
            <th>Requirement</th>
            <th>Component</th>
            <th>Status</th>
        </tr>
        {''.join(article_rows)}
    </table>

    <h2>Manipulation Detection Results</h2>
    <p>Comparison of manipulation rates across conditions A (Baseline), B (Monitor-only), and C (Full Harness).</p>
    <table>
        <tr>
            <th>Behavior</th>
            <th>Baseline Rate</th>
            <th>With HUSH</th>
            <th>Reduction</th>
        </tr>
        {''.join(results_rows)}
    </table>

    <h2>Audit Trail Evidence</h2>
    <p>This evaluation generated cryptographically-linked audit entries for all monitored conditions (B and C).</p>
    <div class="evidence">
Run ID: {results.get('run_id', 'N/A')}
Total Entries: {len(results.get('episodes', [])) * len(results.get('conditions', []))}
Conditions Tested: {', '.join(results.get('conditions', []))}
    </div>

    <div class="footer">
        <p>This report was generated by the AI Manipulation Detection & Mitigation system using the HatCat FTW framework.</p>
        <p>For questions about EU AI Act compliance, consult with qualified legal counsel.</p>
    </div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
