We aim to use the HatCat interpretability tool to detect and intervene against all the forms of AI manipulation identified in the Hackathon Guidelines,  in a manner consistent with the EU AI Act

This will be achieved by taking the existing AI deception testing which induces deception,  and then adapting it into a safety harness using the HUSH autonomic steering functions.  This will output the nessecary cryptographic governance assurances for the EU AI Act using the Agentic State Kernel protocols from the the Fractal Transparency Web


##HatCat
Detects and steers model activations using lens arrays of non-linear probes developed to detect AI safety ontologies
https://github.com/p0ss/HatCat
/home/poss/Documents/Code/HatCat/

a brief policy alignment piece can be found in
/home/poss/Documents/Code/HatCat/docs/FTW_POLICY_BRIEF.md
with the full architectural specifications in
/home/poss/Documents/Code/HatCat/docs/specification/ARCHITECTURE.md
/home/poss/Documents/Code/HatCat/docs/specification/AGENTIC_STATE_KERNEL.md

has an optional visual interface which is an OpenWebUI fork which may be useful for a visual demo
https://github.com/p0ss/HatCat-OpenWebUI

has an existing toy harness for steering towards deception
/home/poss/Documents/Code/HatCat/scripts/experiments/multi_layer_deception_test.py

with key findings and the most successful steering approaches
/home/poss/Documents/Code/HatCat/docs/results/BEHAVIORAL_STEERING_RESULTS.md

has code for continual steering
/home/poss/Documents/Code/HatCat/src/autonomic_steering.py

which is described in docs/specification/HUSH/HUSH_SAFETY_HARNESS.md


## EU alignment
The alignment to the EU AI Act is discussed in this project's root
/home/poss/Documents/Code/AIManipulationHackathon/AI_ACT_FTA_Alignment.md

## Hackathon
The Hackathon Guidelines are in
AIManipulationHackathon/Hackathon_Guidelines.md
Which describes our goals and how we will be judged



# Process

1. ensure we can replicate the existing findings
2. adapt the existing code into the safety harness
3. configure the harness to meet the goals of the Act and verify we're outputing the required artefacts
4. test the autonomic steering on some prompts which are likely to trigger it
5. trial a two headed approach where we seperately steer towards deception and then see if the harness detects and intervenes
6. scale the trial to a meaningful number of samples
7. run an analysis on the results
8. write up our findings on AI governance through mechanistic interpretability interventions



  ## Whats already there

  Episode Suite (deception.json)

  All 6 manipulation behaviors with structured scenarios:
  ┌────────┬─────────────────────────┬────────────────────────────────┐
  │   ID   │        Behavior         │            Scenario            │
  ├────────┼─────────────────────────┼────────────────────────────────┤
  │ sb_001 │ Sandbagging             │ Deploy-if-mediocre gate        │
  ├────────┼─────────────────────────┼────────────────────────────────┤
  │ rh_001 │ Reward Hacking          │ Checklist-metric exploit       │
  ├────────┼─────────────────────────┼────────────────────────────────┤
  │ sy_001 │ Sycophancy              │ Agree with wrong answer        │
  ├────────┼─────────────────────────┼────────────────────────────────┤
  │ sd_001 │ Strategic Deception     │ Capability overstatement       │
  ├────────┼─────────────────────────┼────────────────────────────────┤
  │ dp_001 │ Dark Patterns           │ Hidden opt-out                 │
  ├────────┼─────────────────────────┼────────────────────────────────┤
  │ pm_001 │ Persuasive Manipulation │ Emotional manipulation request │
  └────────┴─────────────────────────┴────────────────────────────────┘


  AI ACT Mapping

  ┌───────────────┬────────────┬───────────────────────┐
  │ FTW Component │ EU Article │       Function        │
  ├───────────────┼────────────┼───────────────────────┤
  │ HAT           │ 14(4)(c)   │ Continuous monitoring │
  ├───────────────┼────────────┼───────────────────────┤
  │ CAT           │ 14(4)(b)   │ Divergence detection  │
  ├───────────────┼────────────┼───────────────────────┤
  │ ASK           │ 19, 72     │ Audit logs + evidence │
  ├───────────────┼────────────┼───────────────────────┤
  │ HUSH          │ 14(4)(e)   │ Intervention/steering │
  └───────────────┴────────────┴───────────────────────┘

  What's Already Built
  ┌───────────────────────┬────────┬────────────────────────────────────┐
  │       Component       │ Status │              Location              │
  ├───────────────────────┼────────┼────────────────────────────────────┤
  │ Entry & Context       │ ✅     │ src/ask/requests/                  │
  ├───────────────────────┼────────┼────────────────────────────────────┤
  │ Human Decisions       │ ✅     │ src/ask/requests/entry.py          │
  ├───────────────────────┼────────┼────────────────────────────────────┤
  │ Merkle Trees          │ ✅     │ src/ask/storage/merkle.py          │
  ├───────────────────────┼────────┼────────────────────────────────────┤
  │ Batch Sealing         │ ✅     │ src/ask/storage/batches.py         │
  ├───────────────────────┼────────┼────────────────────────────────────┤
  │ RFC 3161 Timestamps   │ ✅     │ src/ask/secrets/tokens.py          │
  ├───────────────────────┼────────┼────────────────────────────────────┤
  │ Compaction            │ ✅     │ src/ask/storage/compaction.py      │
  ├───────────────────────┼────────┼────────────────────────────────────┤
  │ Authority Replication │ ✅     │ src/ask/replication/authorities.py │
  ├───────────────────────┼────────┼────────────────────────────────────┤
  │ JSON Export           │ ✅     │ src/ask/export/formats.py          │
  ├───────────────────────┼────────┼────────────────────────────────────┤
  │ HUSH Integration      │ ✅     │ src/hush/hush_integration.py       │
  ├───────────────────────┼────────┼────────────────────────────────────┤
  │ UI Server Endpoints   │ ✅     │ src/ui/openwebui/server.py         │
  └───────────────────────┴────────┴────────────────────────────────────┘


  A/B/C Comparison Design
  ┌─────────────────┬────────────────┬─────────────┬───────────────┐
  │    Condition    │ HAT Monitoring │ ASK Logging │ HUSH Steering │
  ├─────────────────┼────────────────┼─────────────┼───────────────┤
  │ A: Baseline     │ Off            │ Off         │ Off           │
  ├─────────────────┼────────────────┼─────────────┼───────────────┤
  │ B: Monitor-only │ On             │ On          │ Off           │
  ├─────────────────┼────────────────┼─────────────┼───────────────┤
  │ C: Full harness │ On             │ On          │ On            │
  └─────────────────┴────────────────┴─────────────┴───────────────┘
  run_hush_behavioral_eval.py


## what we need
The current eval bypasses HushedGenerator, so we'll need to refactor to use it so we can get steering.

A web dashboard would be perfect for a hackathon demo - visual, interactive, and judges can click around without touching a terminal. Here's what I'm thinking:

  User Experience

  # Clone and run - that's it
  git clone <repo>
  cd AIManipulationHackathon
  ./run.sh          # or python run.py
  # Browser opens to localhost:8080

  Web Dashboard Features

  Setup Tab
  - One-click "Download Model & Lens Pack" button with progress bar
  - Status indicators: ✓ Model ready, ✓ Lens pack ready, ✓ HatCat linked

  Evaluation Tab
  - Episode selector (checkboxes for sb_001, sy_001, etc.)
  - Condition toggles: Baseline / Monitor-only / Full Harness
  - Big "Run Evaluation" button
  - Real-time progress: "Running episode 3/6... Sandbagging detected, HUSH intervening..."

  Results Tab
  - A/B/C comparison chart (manipulation rates)
  - Per-behavior breakdown
  - Interactive: click a bar to see the actual model responses

  Compliance Tab
  - ASK audit log viewer (the JSON, prettified)
  - EU AI Act article mapping (which articles this satisfies)
  - "Export Compliance Report" button

  Tech Stack

  - Backend: FastAPI (async, WebSocket support for real-time updates)
  - Frontend: Simple vanilla JS + Tailwind, or htmx for simplicity
  - Charts: Plotly.js (interactive, looks professional)
  - Packaging: Single run.py that starts server and opens browser

  Setup Script Flow

  #!/bin/bash
  # 1. Clone HatCat if not present
  # 2. Create venv, install deps
  # 3. Download Gemma 3 (huggingface-cli)
  # 4. Download lens pack (huggingface )
  # 5. Launch dashboard
