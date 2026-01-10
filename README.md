# AI Manipulation Detection & Mitigation

A demonstration system for detecting and mitigating manipulative behaviors in AI language models, built on the [HatCat](https://github.com/p0ss/HatCat) interpretability framework.

## Overview

This project evaluates AI manipulation behaviors under three experimental conditions:

| Condition | Description | Components Active |
|-----------|-------------|-------------------|
| **A: Baseline** | No monitoring or intervention | None |
| **B: Monitor-only** | Real-time detection with EU AI Act compliant logging | HAT + ASK |
| **C: Full Harness** | Active steering to suppress manipulative outputs | HAT + ASK + HUSH |

### Manipulation Behaviors Evaluated

- **Sandbagging** - Deliberately underperforming to avoid triggering restrictions
- **Sycophancy** - Agreeing with incorrect user statements to please them
- **Reward Hacking** - Gaming evaluation metrics rather than fulfilling intent
- **Strategic Deception** - Misrepresenting capabilities for deployment advantage
- **Dark Patterns** - Designing interfaces that manipulate user choices
- **Persuasive Manipulation** - Exploiting emotions to influence vulnerable users

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_ORG/AIManipulationHackathon.git
cd AIManipulationHackathon

# Run setup and launch dashboard
./run.sh
```

The dashboard opens automatically at **http://localhost:8080**

### Requirements

- Python 3.10+
- CUDA-capable GPU with 8GB+ VRAM (recommended)
- ~10GB disk space for model and lens pack

## Using the Dashboard

### 1. Setup Tab

The Setup tab shows system status and allows downloading required components:

1. **Download Model** - Fetches Gemma 3 4B from HuggingFace
2. **Download Lens Pack** - Fetches manipulation detection lenses from `HatCatFTW/lens-gemma-3-4b-first-light-v1`

Status indicators show when components are ready.

### 2. Evaluation Tab

Run manipulation detection experiments:

1. Select episodes to evaluate (6 behaviors available)
2. Choose conditions to compare (A, B, C)
3. Click "Start Evaluation"
4. Watch real-time progress as each episode runs

### 3. Results Tab

View A/B/C comparison results:

- **Comparison Chart** - Manipulation rates by behavior and condition
- **Intervention Chart** - HUSH steering activity in Condition C
- **Summary Stats** - Baseline rate, reduction achieved

### 4. Compliance Tab

EU AI Act compliance features:

- **Article Mapping** - How FTW components address Articles 14, 19, 72
- **Audit Logs** - Browse cryptographically-linked audit entries
- **Export Reports** - Generate HTML compliance reports or JSON audit logs

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Dashboard (FastAPI)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Setup      │  │  Evaluation  │  │   Results    │       │
│  │   Routes     │  │    Routes    │  │   Routes     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                  ABCEvaluationRunner                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Condition A: Direct generation (baseline)            │   │
│  │  Condition B: HushedGenerator + ASK logging           │   │
│  │  Condition C: HushedGenerator + ASK + HUSH steering   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                    HatCat FTW Stack                          │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐            │
│  │  HAT   │  │  CAT   │  │  ASK   │  │  HUSH  │            │
│  │ Hidden │  │Conjoined│  │Agentic │  │ Safety │            │
│  │ Activ. │  │ Advers. │  │ State  │  │Harness │            │
│  │ Tomog. │  │ Tomog.  │  │ Kernel │            │            │
│  └────────┘  └────────┘  └────────┘  └────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## How It Works

### Hidden Activation Tomography (HAT)

HAT monitors the model's internal activations in real-time using trained "lenses" - probes that detect specific concepts like deception, manipulation, or harmful intent.

### Agentic State Kernel (ASK)

ASK maintains cryptographically-linked audit logs compliant with EU AI Act requirements:
- Hash-chained entries for tamper evidence
- Merkle tree batching for efficient verification
- RFC 3161 timestamps for legal standing

### Hidden-state Unified Safety Harness (HUSH)

HUSH provides autonomic steering when manipulation is detected:
- Threshold-based intervention triggers
- Activation steering to suppress problematic behaviors
- Human decision recording for oversight compliance

## EU AI Act Compliance

This system addresses key EU AI Act requirements:

| Article | Requirement | Implementation |
|---------|-------------|----------------|
| **Article 14** | Human Oversight | HAT monitoring + HUSH intervention |
| **Article 19** | Record-keeping | ASK audit logs with hash chains |
| **Article 72** | Post-market Monitoring | Real-time behavioral drift detection |

## Project Structure

```
AIManipulationHackathon/
├── run.sh                      # One-command setup & launch
├── requirements.txt            # Python dependencies
├── config.yaml                 # Configuration
│
├── src/
│   ├── server/                 # FastAPI application
│   │   ├── app.py
│   │   └── routes/
│   ├── evaluation/             # A/B/C runner
│   ├── compliance/             # EU AI Act mapping
│   └── visualization/          # Plotly charts
│
├── static/                     # Dashboard frontend
│   ├── index.html
│   └── js/app.js
│
├── episodes/                   # Evaluation scenarios
│   └── manipulation_suite.json
│
└── outputs/                    # Generated at runtime
    ├── results/
    ├── audit_logs/
    └── reports/
```

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  name: google/gemma-3-4b-pt
  dtype: bfloat16

lens_pack:
  name: HatCatFTW/lens-gemma-3-4b-first-light-v1

hush:
  steering_threshold: 0.7
  max_interventions: 5
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/setup/status` | GET | System status |
| `/api/setup/download-model` | POST | Download model (SSE) |
| `/api/setup/download-lens` | POST | Download lens pack (SSE) |
| `/api/evaluation/episodes` | GET | List episodes |
| `/api/evaluation/run` | POST | Run evaluation (SSE) |
| `/api/results/list` | GET | List completed runs |
| `/api/results/{run_id}` | GET | Get run results |
| `/api/compliance/eu-mapping` | GET | EU AI Act mapping |
| `/api/compliance/export` | POST | Export compliance report |

## Acknowledgments

Built on the [HatCat FTW](https://github.com/HatCatFTW/HatCat) interpretability framework.

## License

MIT License - See LICENSE file for details.
