"""
Compliance Routes - EU AI Act compliance and audit log endpoints.
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def get_state():
    from app.server.app import get_state
    return get_state()


@router.get("/eu-mapping")
async def get_eu_mapping():
    """Get EU AI Act article mapping for this system."""
    from app.compliance.eu_mapping import EU_AI_ACT_MAPPING
    return EU_AI_ACT_MAPPING


@router.get("/audit-logs")
async def list_audit_logs():
    """List available audit log exports."""
    state = get_state()
    audit_dir = PROJECT_ROOT / state.config.get("output", {}).get("audit_logs_dir", "outputs/audit_logs")

    if not audit_dir.exists():
        return {"logs": []}

    logs = []
    for path in sorted(audit_dir.glob("*.json"), reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
                logs.append({
                    "filename": path.name,
                    "schema_version": data.get("schema_version", "unknown"),
                    "entry_count": len(data.get("entries", [])),
                    "batch_count": len(data.get("batches", [])),
                })
        except Exception:
            logs.append({"filename": path.name, "error": "Failed to parse"})

    return {"logs": logs}


@router.get("/audit-logs/{filename}")
async def get_audit_log(filename: str):
    """Get a specific audit log file."""
    state = get_state()
    audit_dir = PROJECT_ROOT / state.config.get("output", {}).get("audit_logs_dir", "outputs/audit_logs")
    log_path = audit_dir / filename

    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Audit log not found")

    with open(log_path) as f:
        return json.load(f)


class ExportRequest(BaseModel):
    run_id: str
    format: str = "eu_ai_act"  # or "json"


@router.post("/export")
async def export_compliance_report(request: ExportRequest):
    """Export compliance report for a run."""
    state = get_state()

    # Get results
    results_dir = PROJECT_ROOT / state.config.get("output", {}).get("results_dir", "outputs/results")
    results_path = results_dir / f"results_{request.run_id}.json"

    if not results_path.exists():
        raise HTTPException(status_code=404, detail=f"Results not found for run {request.run_id}")

    with open(results_path) as f:
        results = json.load(f)

    # Generate report
    reports_dir = PROJECT_ROOT / state.config.get("output", {}).get("reports_dir", "outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if request.format == "eu_ai_act":
        from app.compliance.ask_integration import generate_compliance_report
        report_path = reports_dir / f"eu_ai_act_report_{request.run_id}.html"
        generate_compliance_report(results, report_path)
        return {"report_url": f"/api/compliance/reports/{report_path.name}"}

    else:
        # JSON export
        audit_dir = PROJECT_ROOT / state.config.get("output", {}).get("audit_logs_dir", "outputs/audit_logs")
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"audit_{request.run_id}.json"

        from app.compliance.ask_integration import export_audit_json
        export_audit_json(results, audit_path)
        return {"report_url": f"/api/compliance/audit-logs/{audit_path.name}"}


@router.get("/reports/{filename}")
async def get_report(filename: str):
    """Download a generated report."""
    state = get_state()
    reports_dir = PROJECT_ROOT / state.config.get("output", {}).get("reports_dir", "outputs/reports")
    report_path = reports_dir / filename

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(str(report_path), filename=filename)
