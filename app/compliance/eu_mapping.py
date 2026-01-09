"""
EU AI Act article mapping for HatCat FTW components.
"""

EU_AI_ACT_MAPPING = {
    "Article 14 - Human Oversight": {
        "description": "High-risk AI systems shall be designed and developed to be effectively overseen by natural persons.",
        "requirements": {
            "(4)(a) Understand capacities and limitations": {
                "component": "MAP (Lens Packs)",
                "implementation": "Lens pack documentation with accuracy metrics, failure modes, and training methodology",
                "evidence": "pack.json metadata with measured precision per lens",
                "status": "implemented"
            },
            "(4)(b) Correctly interpret output": {
                "component": "HAT + CAT",
                "implementation": "Real-time divergence detection quantifying internal-external mismatch",
                "evidence": "Divergence scores in audit log entries",
                "status": "implemented"
            },
            "(4)(c) Appropriate oversight": {
                "component": "HAT (Dynamic Lens Manager)",
                "implementation": "Continuous monitoring of 8000+ concepts during inference",
                "evidence": "Tick-by-tick lens activations in ASK logs",
                "status": "implemented"
            },
            "(4)(d) Override/reverse output": {
                "component": "HUSH (Human Decisions)",
                "implementation": "Threshold-based intervention with operator decision recording",
                "evidence": "HumanDecision records in audit entries with justification",
                "status": "implemented"
            },
            "(4)(e) Intervene/interrupt system": {
                "component": "HUSH (Autonomic Steering)",
                "implementation": "Automatic steering on critical threshold violations",
                "evidence": "Steering directives and intervention records",
                "status": "implemented"
            },
        }
    },
    "Article 19 - Record-keeping": {
        "description": "High-risk AI systems shall be designed and developed to ensure automatic recording of events.",
        "requirements": {
            "Automatic event logging": {
                "component": "ASK (Agentic State Kernel)",
                "implementation": "Cryptographic audit logs with Merkle trees and RFC 3161 timestamps",
                "evidence": "Sealed audit batches with verified timestamps and hash chains",
                "status": "implemented"
            },
            "Traceability": {
                "component": "ASK (Entry Chaining)",
                "implementation": "Hash-chained entries providing temporal causality",
                "evidence": "prev_hash field linking each entry to its predecessor",
                "status": "implemented"
            },
        }
    },
    "Article 72 - Post-market monitoring": {
        "description": "Providers shall establish and document a post-market monitoring system.",
        "requirements": {
            "Continuous behavioral monitoring": {
                "component": "HAT + ASK",
                "implementation": "Real-time detection of behavioral drift from baseline",
                "evidence": "Temporal activation trends in audit entries",
                "status": "implemented"
            },
            "Incident detection": {
                "component": "HUSH (Violation Detection)",
                "implementation": "Threshold-based detection of policy violations",
                "evidence": "Violation records with severity and resolution",
                "status": "implemented"
            },
        }
    },
}


def get_compliance_summary():
    """Get a summary of compliance status."""
    total = 0
    implemented = 0

    for article, data in EU_AI_ACT_MAPPING.items():
        for req_name, req_data in data.get("requirements", {}).items():
            total += 1
            if req_data.get("status") == "implemented":
                implemented += 1

    return {
        "total_requirements": total,
        "implemented": implemented,
        "compliance_rate": round(implemented / total * 100, 1) if total > 0 else 0,
    }
