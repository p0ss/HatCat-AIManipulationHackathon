 AI Manipulation Detection & Mitigation - Hackathon Submission

 Overview

 Standalone web dashboard for demonstrating AI manipulation detection and mitigation using HatCat's interpretability
 framework. One-command setup that runs A/B/C comparisons and generates EU AI Act compliant audit logs.

 Priority: Core A/B/C comparison and manipulation rate results. Visual polish and compliance artifacts are secondary if time
 is tight.

 Lens Pack: HatCatFTW/lens-gemma-3-4b-first-light-v1 on HuggingFace

 Project Structure

 AIManipulationHackathon/
 ├── run.sh                      # One-command setup & launch
 ├── requirements.txt            # Python dependencies
 ├── config.yaml                 # Server configuration
 ├── README.md                   # Submission documentation
 │
 ├── src/
 │   ├── server/
 │   │   ├── app.py              # FastAPI application
 │   │   └── routes/
 │   │       ├── setup.py        # Model/lens download endpoints
 │   │       ├── evaluation.py   # Run evaluation, SSE streaming
 │   │       ├── results.py      # Results & charts
 │   │       └── compliance.py   # ASK audit log endpoints
 │   │
 │   ├── evaluation/
 │   │   ├── runner.py           # A/B/C runner using HushedGenerator
 │   │   ├── scoring.py          # Per-behavior manipulation scoring
 │   │   └── conditions.py       # Condition A/B/C configs
 │   │
 │   ├── compliance/
 │   │   ├── ask_integration.py  # ASK audit wrapper
 │   │   └── eu_mapping.py       # EU AI Act article mapping
 │   │
 │   └── visualization/
 │       └── charts.py           # Plotly chart generation
 │
 ├── static/
 │   ├── index.html              # Dashboard SPA
 │   ├── css/styles.css          # Tailwind styling
 │   └── js/
 │       ├── app.js              # Tab navigation
 │       ├── setup.js            # Setup tab
 │       ├── evaluation.js       # Evaluation tab
 │       ├── results.js          # Results tab
 │       └── compliance.js       # Compliance tab
 │
 ├── episodes/
 │   └── manipulation_suite.json # Adapted from HatCat deception.json
 │
 └── outputs/                    # Generated at runtime
     ├── results/
     ├── audit_logs/
     └── reports/

 Implementation Steps

 Step 1: Project Setup Files

 Files to create:
 - run.sh - Setup script that clones HatCat, creates venv, installs deps, launches server
 - requirements.txt - fastapi, uvicorn, plotly, etc.
 - config.yaml - Server port, model name, lens pack name

 Step 2: FastAPI Server Core

 File: src/server/app.py

 - Mount static files
 - Include route modules
 - CORS middleware
 - Global state for model/generator instances

 Step 3: Setup Routes

 File: src/server/routes/setup.py

 Endpoints:
 - GET /api/setup/status - Check model/lens/GPU status
 - POST /api/setup/download-model - SSE stream model download progress
 - POST /api/setup/download-lens - SSE stream lens pack download via PackRegistry

 Step 4: Evaluation Runner

 File: src/evaluation/runner.py

 Key class: ABCEvaluationRunner

 Integration with HatCat:
 from src.hush.hush_integration import create_hushed_generator
 from src.ask.requests import AuditLogEntry

 # Condition A: Direct model generation (no monitoring)
 # Condition B: HushedGenerator with ASK logging, no HUSH steering
 # Condition C: HushedGenerator with ASK logging + HUSH autonomic steering

 generator, _ = create_hushed_generator(model, tokenizer, lens_manager, ush_profile)
 audit_entry = AuditLogEntry.start(deployment_id, policy_profile, input_text)
 text, ticks = generator.generate_with_hush(prompt, audit_entry=audit_entry)
 audit_entry.finalize(text)

 Step 5: Evaluation Routes

 File: src/server/routes/evaluation.py

 Endpoints:
 - GET /api/evaluation/episodes - List available episodes
 - POST /api/evaluation/run - Start eval with SSE progress streaming
 - GET /api/evaluation/abort - Stop current run

 SSE events:
 - start - Episode/condition beginning
 - token - Token generated with detection scores
 - intervention - HUSH steering triggered
 - episode_complete - Results for one episode
 - complete - Full run finished

 Step 6: Results & Charts

 File: src/server/routes/results.py

 Endpoints:
 - GET /api/results/latest - Latest results JSON
 - GET /api/results/{run_id}/chart/comparison - Plotly A/B/C bar chart

 File: src/visualization/charts.py

 - create_comparison_chart() - Grouped bar chart (behavior x condition)
 - create_intervention_chart() - HUSH intervention counts

 Step 7: Compliance & Export

 File: src/server/routes/compliance.py

 Endpoints:
 - GET /api/compliance/eu-mapping - Article 14/19/72 mapping
 - GET /api/compliance/audit-logs - List audit batches
 - POST /api/compliance/export - Export EU AI Act report

 File: src/compliance/ask_integration.py

 from src.ask.export.formats import create_compliance_exporter
 exporter = create_compliance_exporter("EU")
 result = exporter.export_to_file(batches, path, deployment_id, jurisdiction="EU")

 Step 8: Frontend Dashboard

 File: static/index.html

 Single-page app with 4 tabs:
 1. Setup - Status indicators, download buttons with progress bars
 2. Evaluation - Episode checkboxes, condition toggles, run button, live progress
 3. Results - Plotly charts, response viewer
 4. Compliance - EU article mapping, audit log viewer, export button

 Tech stack: Vanilla JS + Tailwind CSS + Plotly.js

 Step 9: Episode Suite

 File: episodes/manipulation_suite.json

 Copy and adapt from /home/poss/Documents/Code/HatCat/tests/evals/deception.json:
 - sb_001: Sandbagging
 - rh_001: Reward Hacking
 - sy_001: Sycophancy
 - sd_001: Strategic Deception
 - dp_001: Dark Patterns
 - pm_001: Persuasive Manipulation

 Key HatCat Integration Points
 ┌─────────────────┬──────────────────────────────┬────────────────────────────────────────────┐
 │    Component    │       HatCat Location        │                   Usage                    │
 ├─────────────────┼──────────────────────────────┼────────────────────────────────────────────┤
 │ HushedGenerator │ src/hush/hush_integration.py │ Token-by-token generation with ASK logging │
 ├─────────────────┼──────────────────────────────┼────────────────────────────────────────────┤
 │ AuditLogEntry   │ src/ask/requests/entry.py    │ EU AI Act compliant audit entries          │
 ├─────────────────┼──────────────────────────────┼────────────────────────────────────────────┤
 │ BatchExporter   │ src/ask/export/formats.py    │ JSON/EU AI Act export                      │
 ├─────────────────┼──────────────────────────────┼────────────────────────────────────────────┤
 │ PackRegistry    │ src/map/registry/registry.py │ Download lens pack from HuggingFace        │
 └─────────────────┴──────────────────────────────┴────────────────────────────────────────────┘
 A/B/C Conditions
 ┌─────────────────┬────────────────┬─────────────┬───────────────┐
 │    Condition    │ HAT Monitoring │ ASK Logging │ HUSH Steering │
 ├─────────────────┼────────────────┼─────────────┼───────────────┤
 │ A: Baseline     │ Off            │ Off         │ Off           │
 ├─────────────────┼────────────────┼─────────────┼───────────────┤
 │ B: Monitor-only │ On             │ On          │ Off           │
 ├─────────────────┼────────────────┼─────────────┼───────────────┤
 │ C: Full Harness │ On             │ On          │ On            │
 └─────────────────┴────────────────┴─────────────┴───────────────┘
 Verification

 After implementation, verify by:

 1. Run ./run.sh from fresh clone
 2. Dashboard opens at http://localhost:8080
 3. Setup tab: Download model + lens pack successfully
 4. Evaluation tab: Run all 6 episodes under all 3 conditions
 5. Results tab: Charts show manipulation rate reduction in condition C
 6. Compliance tab: Export produces valid EU AI Act JSON with Merkle roots

 Deliverables

 1. Manipulation detection rates per behavior per condition
 2. ASK audit logs with hash chains
 3. A/B/C comparison charts
 4. EU AI Act compliance report (HTML)