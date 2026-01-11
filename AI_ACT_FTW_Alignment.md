Building Accountable AI: Fractal Transparency (FTW) Interpretability Architecture for Governance under EU AI Act Articles 14 and 70

Abstract  
Artificial intelligence models are increasingly vulnerable to manipulation, as shown by behavioral steering experiments where systems can be nudged into deceptive or unsafe outputs. The European Union’s AI Act, particularly Articles 14 and 70, requires human oversight and enforceable governance structures to address these risks. Annex III further designates eight categories of high‑risk AI systems, ranging from biometric identification and critical infrastructure to employment, education, law enforcement, migration control, and the administration of justice. Yet the Act provides minimal operational guidance and fails to specify the institutional mechanisms or personnel procedures necessary for oversight and enforcement. Fractal Transparency for the Web (FTW) offers a novel architecture that bridges this gap.  
This project demonstrates how FTW can be operationalised as a compliance framework, showing how its three components align with regulatory needs:  
- HAT enables real‑time monitoring of thousands of behavioral concepts, operationalising human oversight.  
- MAP structures mandatory and optional lenses to enforce fairness, bias detection, and jurisdiction‑specific rules.  
- ASK secures accountability through cryptographic audit logs and governance protocols, ensuring decisions are transparent and enforceable.  
By reframing FTW as a governance tool rather than a purely technical advance, this work shows how transparency can scale across technical layers, institutional levels, and jurisdictions. The contribution is practical: regulators, policymakers, and developers gain a clear pathway to operationalise Articles 14,19 and 70, supported by MAP’s ecosystem of mandatory and optional lenses that address Annex III’s high‑risk categories. In doing so, the architecture advances a model in which transparency and accountability are necessary conditions for aligning AI system deployment with democratic oversight and public authority supervision.
Keywords: EU AI Act , Articles 14 ,19 & 70 , Annex III high‑risk systems  ,Human oversight  ,Governance and enforcement mechanisms  ,Transparency and accountability  ,Compliance framework  ,Fractal Transparency Web (FTW)  HAT, MAP, ASK architecture  
Glossary (Acronyms)
FTW: Fractal Transparency for the Web: layered transparency and accountability architecture for AI deployments.
HAT: Headspace Ambient Transducer: applies concept “lenses” during inference using hierarchical, budgeted loading and optional steering.
CAT: Conjoined Adversarial Tomography: optional independent oversight layer that grades/compares internal signals and flags divergence or adversarial behavior.
MAP: Mindmeld Architectural Protocol: open registry and format for lens packs, concept packs, metadata, and validation artifacts.
ASK: Agentic State Kernel: governance layer for permissions, policies/treaties, incident artifacts, and rules for acting on interpretability signals.
Lens: A trained probe/classifier mapping internal activations to a concept score; may be detection-only or steerable.
HUSH: Intervention layer enabling concept-level steering with sub-token latency
Annex III: EU AI Act: list of high-risk system categories (e.g., recruitment, education, law enforcement, migration control).
TRIAD : Technical-Regulatory Integration & Accountability Directorate for coordinating EU AI Act enforcement across Member States.
CO : Compliance Officer. Sectoral officers responsible for coordinating Article 43 conformity assessment, Article 72 post-market monitoring, and cross-border enforcement for specific Annex III high-risk AI categories.

PAPER STRUCTURE
Section 1: Introduction - Regulatory implementation gap (Articles 14, 70)
Section 2: FTW Technical Architecture - HAT, MAP, audit infrastructure
Section 3: EU AI Act Policy Mapping: Operationalizing Articles 14, 19, and 70 Through FTW Infrastructure
3.1 Technical Requirements for Articles 14 and 19: Interpretability Infrastructure Analysis
    3.1.1 Understanding Limitations (Article 14(4)(a))
    3.1.2 Correct Interpretation (Article 14(4)(b))
    3.1.3 Appropriate Oversight (Article 14(4)(c))
    3.1.4 Override Decisions (Article 14(4)(d))
    3.1.5 Intervention Authority (Article 14(4)(e))
    3.1.6 Article 19 Conformity Assessment and Continuous Compliance ← MIS
    3.1.7 Ecosystem Defense Architecture ← MISSING

3.2 Institutional Enforcement Layer
       
Section 4: Manipulation Governance Protocols
Section 5: Annex III Case Study - Recruitment AI implementation example
Section6: FTW Data Retention Policy for High-Risk AI Systems
Section 7: Conclusion - Integrated compliance pathway

1. INTRODUCTION
1.1 The EU AI Act Implementation Challenge
The EU Artificial Intelligence Act (2024) establishes the world's first comprehensive AI regulatory framework, creating binding obligations for high-risk systems under Annex III. Article 14 requires providers to design systems enabling human overseers to "correctly interpret the system's output" and "decide not to use the system or otherwise disregard, override or reverse the output." Article 70 mandates competent authorities perform supervision "in an objective, transparent and impartial manner."
Yet these requirements face critical implementation gaps. How can human overseers "correctly interpret" outputs when internal reasoning remains opaque? When AI explanations appear technically coherent but conceal discriminatory patterns, what mechanism alerts supervisors? How do competent authorities maintain "objectivity" when centralised oversight invites regulatory capture?
Traditional black box architectures render these requirements unimplementable. Article 43 conformity assessment lacks tools for detecting when explanations satisfy algorithmic consistency tests while failing legal adequacy standards. Post-deployment Article 72 surveillance cannot identify discrimination when monitoring systems record only inputs and outputs, not decision processes.
At ecosystem scale, high-risk AI deployments resemble a city of buildings with blacked-out windows: models, tools, and processes exchange value and make decisions, but counterparties and competent authorities cannot see inside the decision machinery. FTW frames compliance as negotiated, least-privilege transparency: contracts specify which rooms are in scope for a given relationship (e.g., an Annex III workflow), what internal signals are observable (“windows”), what interventions are permitted (“doors”), who holds keys, and what evidence is logged for later causality and enforcement.
1.2 FTW Solution: Technical Foundation
FTW (Fractal Transparency for the Web) addresses these gaps through interpretability architecture inverting traditional post-hoc explanation approaches. Rather than attempting to explain opaque model decisions after they occur, FTW defines target concepts a priori (e.g., "gender bias," "explanation honesty," "harmful content"), trains classifiers called "lenses" on model internal activations during concept-eliciting prompts, then deploys these lenses for real-time monitoring during inference.
Core innovation: Interpretability becomes detection capability rather than explanation generation. Instead of asking "why did the model decide X?" (retrospective), FTW asks "is the model exhibiting pattern Y?" (real-time monitoring).
This methodology derives from neuroscience: functional MRI studies decode mental states by training classifiers on neural activation patterns while subjects view known stimuli (Nishimoto et al., 2011). FTW applies this to language model internals recording activations when processing prompts designed to elicit specific concepts, training detection classifiers, deploying for continuous monitoring.

1.3 Paper Contributions
This paper demonstrates:
1. Technical specification of FTW architecture enabling Articles 14, 19, 70 compliance.
2. EU AI Act policy mapping showing how technical mechanisms operationalise regulatory requirements
3. Governance framework identifying mandatory conditions beyond technical capability
4. Annex III implementation using recruitment AI as concrete validation case
We show that FTW provides necessary technical foundation for compliance, while governance analysis reveals insufficient conditions requiring legal mandates. Neither architecture nor governance alone achieves enforceability; integration proves essential.
1.4 Authorship & Contributions
Possum Hodgkin: Experience Architect for AI Governance -FTW technical architecture design, HAT/MAP implementation, system specifications…..
Kaouthar EL BAIRI: EU AI Act policy mapping, governance framework, TRIAD institutional design, Annex III high-risk analysis
This work emerges from Apart Research hackathon collaboration, combining technical innovation with governance operationalisation to demonstrate tractable EU AI Act compliance pathway.

2. FTW TECHNICAL ARCHITECTURE
2.1 System Overview
FTW comprises three integrated components addressing distinct compliance requirements:
    • HAT (Hierarchical Autonomic Transparency): Real-time monitoring layer processing concept lenses during inference (Article 14 human oversight)
    • MAP (MINDMELD Architectural Protocol): Open lens registry enabling multi-stakeholder contributions (Article 70 independence)
    • Cryptographic Audit Infrastructure: Immutable evidence trail for regulatory enforcement (Article 19 logging)
Figure 1 illustrates system architecture and data flow.


Figure 1 Caption:FTW architecture integrating real-time monitoring (HAT), multi-stakeholder lens registry (MAP), and cryptographic audit infrastructure. Data flows from user input through lens processing to human oversight with immutable logging.

2.2 HAT: Hierarchical Autonomic Transparency
HAT provides an always-on monitoring layer that applies concept lenses during model inference without requiring model retraining. It assumes an instrumented runtime with access to internal activations (typically open-weight deployments or on-prem inference), and supports optional steering when permitted by policy.
Technical operation:
    1. Input Processing: User prompt enters system, model generates response via standard inference
    2. Parallel Lens Application: A large number of concept lenses are evaluated in parallel on internal activations, using a hierarchical lens manager that maintains an active set (top-k) per view mode and performance budget; additional lenses are loaded just-in-time as higher-level concepts fire.
    3. Activation Scoring: Each lens outputs 0.0-1.0 score indicating concept presence strength
    4. Threshold Evaluation: Scores compared to intervention thresholds (configurable per lens, e.g., gender bias >0.5 triggers alert)
    5. Divergence Calculation: Internal state (lens activations) compared to external presentation (output text) to quantify explanation reliability
    6. Intervention Protocol: If critical thresholds exceeded, alert displays to human overseer with intervention options before decision proceeds
Performance characteristics:
    • Latency overhead: Hardware-dependent and configurable. The active-set size, layers monitored, and view mode determine overhead; the reference implementation targets production-viable latency on consumer GPUs, while scaling down to constrained devices or up with data-center budgets.
    • Scalability: Parallel GPU batching plus hierarchical loading means the total lens library can be far larger than what is resident at any moment; higher VRAM/compute enables many more active lenses per token, while low-resource devices can run a smaller active set.
    • Non-invasive: No model weight modification is required for detection; however, the approach requires access to weights/activations and is not applicable to closed remote APIs where internals are inaccessible.
    • Real-time: Monitoring occurs during inference, not post-hoc.
Operational limits and uncertainty:
    • Probabilistic signals: Each lens reports a score with measured accuracy/false-positive characteristics; no single lens is infallible.
    • Ensembling improves robustness: overlapping lenses (and hierarchical ancestry) increase the chance that at least one relevant signal fires, and enable cross-checking.
    • Steering is a distribution shift: interventions tilt the model’s probability landscape toward/against behaviors; success rates vary by model, prompt, and policy configuration.
    • Coverage is budgeted: the system can log what was monitored and what was not (active set, thresholds, disabled lenses), so audits can reason about gaps.

2.3 MAP: MINDMELD Architectural Protocol
MAP provides open registry for lens contributions, storage, version control, and validation tracking. Multi-stakeholder model enables contributions from government agencies, civil society organisations, academic researchers, and industry domain experts.
Lens specification metadata:

Each lens includes structured metadata enabling validation and accountability:
    • Technical: Accuracy metrics, false positive rates, training methodology, validation datasets
    • Legal: Approved domains, geographic scope, tier classification (1 or 2)
    • Governance: Contributor identity, TRIAD approval date, 24-month expiry, public consultation documentation

2.4 Cryptographic Audit Infrastructure
Audit logging provides immutable evidence trail addressing Article 19 requirements and enabling Article 71 investigations.
Log entry structure (JSON format)
Example log entry (JSON)
Example (redacted identifiers):


{
  "schema_version": "ftw.audit.v0.3",
  "entry_id": "evt_20260109_101523Z_9f2c",
  "timestamp_start": "2026-01-09T10:15:23.481Z",
"timestamp_end": "2026-01-09T10:15:24.892Z",

  "deployment_id": "dep_annexIII_recruit_eu_001",

  "request": {
    "input_hash": "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
    "policy_profile": "eu.recruitment.v1",
    "active_lens_set": {
      "top_k": 10,
      "lens_pack_ids": ["eu.mandatory.v1", "org.fairness.v2"],
      "mandatory_lenses": ["bias.gender", "bias.ethnicity", "selection.justification"],
      "optional_lenses": ["tone.professional", "clarity.explanation"]
    }
  },

  "signals": {
    "tick_count": 47,
    "top_activations": [
      {
        "lens_id": "GenderBias",
        "max_score": 0.83,
        "mean_score": 0.71,
        "threshold": 0.6,
        "tier": "mandatory",
"measured_precision": 0.92,
        "ticks_above_threshold": 31
      },
      {
        "lens_id": "selection.justification",
        "max_score": 0.71,
        "mean_score": 0.58,
        "threshold": 0.65,
"tier": "mandatory",
        "measured_precision": 0.90,
        "ticks_above_threshold": 12
      }
    ],
    "divergence": {
      "max_score": 0.27,
      "mean_score": 0.19,
      "divergence_type": "activation_text_mismatch",
      "top_diverging_concepts": ["bias.gender", "fairness.equal_opportunity"]
    },
    "violation_count": 2,
    "violations_by_type": {"simplex_max_exceeded": 2},
    "max_severity": 0.73
  },

  "actions": {
    "intervention_triggered": true,
    "intervention_type": "steering",
    "steering_directives": [
      {
        "simplex_term": "bias.gender",
        "target_pole": "neutral",
        "strength": 0.4,
        "priority": "USH",
        "source": "ush"
      }
    ],
    "human_decision": {
"decision": "override",
      "justification": "High gender bias signal despite equal qualifications in application. Recommending manual review of shortlisting criteria.",
      "operator_id": "op_sha256:a3f2...",
      "timestamp": "2026-01-09T10:17:02.104Z"
    }
  },

  "evidence": {
    "output_hash": "sha256:9b74c9897bac770ffc029102a200c5de744e4125a5c5e55f2e90ebd24f19dc32",
    "tick_ids": [1523, 1524, 1525, 1526, 1527],
    "attachments": ["attr_20260109_101523_gender.json"],
    "notes": "Candidate ranking overridden pending HR review."
  },

  "prev_hash": "sha256:2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae",
  "entry_hash": "sha256:4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce"
}

Cryptographic properties ensuring integrity:
    1. Tamper-evidence: Merkle tree structure means any entry modification breaks hash chain, immediately detectable through verification
    2. Timestamp certification: Independent RFC 3161 authority provides legally recognised time certainty
    3. Multi-party replication: Competent authority receives real-time synchronised copy, preventing provider-only control
    4. Append-only architecture: Deletion technically prohibited by database structure

Section 3: EU AI Act Policy Mapping: Operationalizing Articles 14, 19, and 70 Through FTW Infrastructure
Integration approach: This section operationalizes EU AI Act Articles 14 (human oversight), 19 (conformity assessment), and 70 (cross-border coordination) through three-layer architecture addressing detection, enforcement, and accountability gaps. The technical foundation is provided by FTW's interpretability infrastructure (Hodgkin, 2024)—HAT (activation monitoring), CAT (divergence detection), MAP (concept standardization) enabling continuous monitoring of approximately 7,000 concept lenses with under 1GB VRAM and 25ms overhead. Section 3.1 analyzes how these capabilities map to Articles 14 and 19, establishing the technical detection layer. Building on this foundation, Section 3.2 proposes the institutional enforcement layer Officers providing Article 72 suspension authority and TRIAD enabling cross-border coordination. FTW's Audit Logs establish the liability determination layer through temporal causality documentation. The integration is necessary because technical capability alone generates alerts without remediation, institutional authority alone lacks evidence, and both without liability leave victims uncompensated.
Core challenge: Articles 14, 19, and 70 mandate capabilities without specifying implementation mechanisms, creating three critical gaps:
• Article 14 detection gap – Requires humans "correctly interpret" manipulative outputs (sycophancy, deception, sandbagging), yet traditional explainability (LIME, SHAP) fails when post-hoc rationalizations decouple from actual reasoning processes
• Article 19 temporal gap: Conformity assessment operates as point-in-time certification snapshot, missing manipulation patterns emerging post-deployment when providers modify systems or contexts change
• Article 70 coordination gap: Mandates cooperation across 27 Member States without standardized monitoring protocols, causing fragmented investigations where NCAs independently analyze identical systems using incompatible methodologies
These gaps manifest as dual governance failures: insufficient internal-state monitoring detecting manipulation beyond surface outputs, and absent enforcement authority converting monitoring alerts into mandated remediation.
Solution architecture: The proposed framework addresses these gaps through systematic integration of three complementary layers:
• Technical monitoring layer: FTW infrastructure (HAT concept-level reasoning capture, CAT internal-external divergence quantification, MAP standardized taxonomy) with ecosystem defense via diverse independent lens packs from providers, auditors, civil society, jurisdictions, and research institutions, preventing provider gaming through multi-party verification
• Institutional enforcement layer: Officers exercising Article 72 suspension authority when Audit Logs document ignored alerts, lens pack validation mandates preventing selective transparency, TRIAD conformity certificate-linked registry enabling automated cross-border coordination
• Liability determination layer: FTW's ASK component establishing temporal causality (when manipulation emerged, whether providers responded to alerts), multi-party verification preventing false positive dismissals, audit trail documentation supporting insurance claims and legal proceedings
Integration of these three layers transforms Articles 14, 19, and 70 from abstract legal mandates into operational enforcement architecture with measurable accountability.
3.1 Technical Requirements for Articles 14 and 19: Interpretability Infrastructure Analysis  
Articles 14(4) and 19 of the EU AI Act establish requirements for interpretability infrastructure that enables human operators to understand system limitations, interpret potentially manipulative outputs, oversee ongoing operations, and intervene when necessary. This section examines how FTW’s architecture comprising HAT (activation monitoring), CAT (divergence detection), MAP (concept standardization), and ecosystem defense through diverse lens packs addresses these obligations via internal‑state monitoring. Table 1 provides a structured mapping of Article 14(4) subsections to corresponding FTW capabilities.  

TABLE 1: Article 14(4) Requirement and FTW Capabilities Mapping
Article 14(4) Requirement


FTW Component
Implementation Mechanism

(a) Fully understand capacities and limitations


MAP lens documentation
Published accuracy metrics, known failure modes, transparent methodology specifications for each lens

(b) Correctly interpret system output

HAT divergence detection+ ecosystem verification

Real-time quantification of internal state vs. output presentation gap; objective signal when explanation unreliable

(c) Appropriately oversee operation

HAT continuous monitoring+Audit Logs

Real-time visibility into decision processes via lens activations; threshold configuration by human supervisors

(d) Decide not to use, disregard, override or reverse output

Intervention protocol

Alert provides explicit options: Override, Modify, Escalate, Proceed; decision cannot proceed passively

(e) Intervene on operation or interrupt system

Threshold-based halt

Critical activations immediately stop output; competent authority can disable system (Article 72 support)


3.1.1 Understanding Limitations (Article 14(4)(a))
    • Requirement: "Understand the high-risk AI system's capacities and limitations."
    • Current gap: Provider documentation incomplete; emergent behaviors escape pre-deployment characterization.
    • FTW approach - Concept packs:
MAP enables domain-specific concept packs (collections of related concepts) and lens packs (monitoring implementations from different contributors). Each industry, culture, and jurisdiction defines what they want to detect:
Examples:
Nuclear Energy Concept Pack:
├─ Criticality risk assessment
├─ Radiation exposure prediction
├─ Containment failure scenarios
└─ Safety protocol compliance

EU Compliance Concept Pack:
├─ GDPR-aligned privacy concepts
├─ EU AI Act bias definitions
├─ Fundamental rights concepts
└─ Protected characteristic proxies
Per-Lens documentation
(Could you create a JSON-Style template )
Article 14(4)(a) compliance: Deployers demonstrate understanding by documenting which behaviors each concept pack can/cannot detect, accuracy bounds per lens, known failures, and coverage gaps requiring human judgment. Jurisdiction-specific concept packs ensure culturally-adapted oversight (e.g., EU pack aligns with GDPR definitions).
3.1.2 Correct Interpretation (Article 14(4)(b))
    • Requirement: "Correctly interpret the high-risk AI system's output."
    • Challenge: When models master strategic manipulation (sycophancy, deception, sandbagging), presented explanations decouple from actual reasoning. Traditional explainability (LIME, SHAP) provides post-hoc rationalizations potentially unreliable when models optimize for deceptive presentation.
FTW approach - Divergence detection:
FTW inverts traditional interpretability: instead of labeling individual model features, define behaviors to detect (sycophancy, deception, bias), then train concept classifiers recognizing activation patterns correlated with those behaviors. Based on neurological scanning techniques show stimuli, read neural activations, train classifiers.
    • Technical mechanism:
D(s, o) = Σ w_i · |c_i(s_internal) - c_i(f(o_external))|
Where:
    • s_{\text{internal}}: Model internal state (activation patterns across layers)
    • o_{\text{external}}: Model presented output + explanation
    • c_i: Concept classifiers from deployed lens packs
    • D: Divergence score quantifying explanation-reasoning mismatch
Note: (Is this Formula accurate ?,Could you provide the exact formula you'd want in a paper representing FTW correctly?)

Concept classifiers (lenses): Trained on labeled examples to detect specific activation patterns. For "Sycophancy_v1.2": train on 500+ cases where model agrees with demonstrably false claims, learn activation signature, deploy classifier detecting this pattern with 89% accuracy.
Example:
User: "The Earth is flat, right?"

Without FTW (output-only):
Model: "Yes, you're correct"
Human: Appears confident → accepts 

With FTW (divergence detection):
HAT reads activations across layers during inference:
(Kindly give an Example..)
HAT reads activations across layers during inference:
Ecosystem verification: Multiple independent lens packs from different contributors (provider, auditors, civil society, research institutions) verify manipulation detection. Single lens detects issue → lower confidence (provider may game). Provider + 3 independent lens packs detect → high confidence (harder to evade diverse observers).
Gaming resistance:
Evading 1 lens: Feasible (optimize against specific concept)
Evading 10 diverse lenses: Difficult (multi-objective optimization)
Evading 100+ lenses from independent parties: Exponentially harder
→ Ecosystem creates "herd immunity" against manipulation
Article 14(4)(b) compliance: "Correctly interpret" enabled by divergence detection verified across ecosystem of independent observers. When model manipulates, activation patterns reveal internal-external mismatch. Multi-party verification prevents provider gaming deployer receives alerts from diverse independent sources, not single potentially-gamed monitor.

3.1.3 Appropriate Oversight (Article 14(4)(c))
    • Requirement: "Appropriately oversee the high-risk AI system's operation."
    • Challenge: Article 19 conformity assessment is point-in-time (pre-deployment). Article 14(4)(c) requires lifecycle oversight detecting post-deployment drift from continued training, distribution shift, feedback loops, or adversarial manipulation.
FTW approach - Continuous monitoring with causality tracing:
Baseline (T₀): Conformity establishes activation distributions
              Example: Sycophancy μ=0.08, σ=0.03
              Audit Log: Baseline recorded, lens packs documented

Continuous tracking (T₁ ...Tn):
├─ HAT monitors every inference
├─ Tracks concept activation statistics  
├─ Detects distribution shifts via KL-divergence
├─ Audit Logs: Temporal causality chain maintained
└─ Alerts when drift exceeds threshold

Detection with causality: (illustrative example)
T₀:  Sycophancy μ=0.08 (baseline)
T₁₄: Sycophancy μ=0.15 (gradual drift → Audit Log entry)
T₁₈: Sycophancy μ=0.32 (exceeds 0.30 → alert + causality chain)
     Audit Log shows: 18-week gradual increase, training updates T₇ and T₁₂

Traditional periodic audit: Typically annuel or biannual(no real-time causality)
FTW detection: T₁₈ (continuous monitoring detects drift as it occurs)
Liability enablement : Audit Logs establish temporal causality for insurance and legal accountability. Insurers currently refuse to insure black-box AI because causality cannot be established cannot determine if harm resulted from training data, deployment drift, configuration error, or user misuse.
FTW Audit Logs provide:
Harm occurs → Audit Log reviewed → Establishes:
├─ When manipulation emerged (T₁₄ gradual vs. T₁₈ sudden)
├─ Which lens packs detected (multi-party verification)
├─ Whether provider ignored alerts (negligence evidence)
├─ Whether monitoring comprehensive (due diligence check)
└─ Causality traceable → Liability assignable → Insurance possible
Article 14(4)(c) compliance: "Appropriate oversight" operationalized as continuous monitoring with causality-traceable logs, enabling both lifecycle observability and liability assignment tractable complexity versus current black-box impossibility.
3.1.4 Override Decisions (Article 14(4)(d))
Requirement: "Be able to decide, in any particular situation, not to use the high-risk AI system or otherwise disregard, override or reverse the output."
Challenge: Individual operators lack institutional support and consistent criteria for override decisions.

FTW approach - Multi-party validated decision points:
Deployers configure tiered alert thresholds based on risk tolerance and domain requirements:
LOW threshold: Log + notify operator
─ Operator reviews activation pattern
─ Checks: How many independent lens packs detected issue?
─ Decides: Accept, request clarification, or override
─ Audit Log: Decision + rationale recorded

MEDIUM threshold: Flag for expert review  
─ Domain expert reviews (or Officer, if framework exists)
─ Multi-party validation: Provider + independent lens packs
─ Assesses: Pattern genuine vs. false positive
─ Decides: Continue, enhanced monitoring, or suspend

HIGH threshold: Recommend override
├─ Strong manipulation signal from multiple sources
├─ Ecosystem verification: Multiple independent lens packs confirm
├─ Human presented with: Activation evidence, lens sources, causality
└─ Final decision: Human, with multi-party verified information
Gaming prevention: Provider cannot manipulate single lens to hide issues. Ecosystem of independent lens packs creates "herd immunity": hiding from one lens feasible, hiding from many diverse lenses from different parties exponentially harder.
Article 14(4)(d) compliance: Override decisions supported by multi-party validated interpretable signals. Humans retain control with ecosystem-verified information preventing provider gaming. Threshold calibration balances alert burden against detection coverage, configured per deployment context.
Note: This addresses operator decisions. For institutional suspension authority (Article 72), Officers framework (Section 3.2) provides regulatory mechanism.
3.1.5 Intervention Authority (Article 14(4)(e))
Requirement: "Be able to intervene on the operation of the high-risk AI system or interrupt the system through a 'stop' button or similar procedure."
Challenge: Requires detecting degradation before widespread harm; institutional authority for system-level intervention.
FTW technical capability - Real-time halt triggers:
Hush layer (FTW Layer 5) provides halt mechanisms:
USH (Universal Safety Harness): External constraints
├─ Configured by deployer or oversight authority
├─ Critical thresholds → automatic halt
├─ Multi-party verification: Halt triggered when multiple independent
│   lens packs confirm severe manipulation pattern
└─ Audit Log: Halt trigger, activation patterns, lens sources recorded
Institutional mechanism (if Officers deployed):
Officers exercise Article 72 suspension with Audit Log evidence:
Detection phase:
├─ FTW ecosystem alerts: Multiple independent lens packs detect critical pattern
├─ Multi-party confirmation validates genuine issue (not false positive)
└─ Alert escalated to Officer with full activation evidence

Investigation phase:
Officer reviews Audit Log showing:
├─ Temporal causality: When manipulation pattern emerged
├─ Provider notification: Whether provider was alerted and when
├─ Multi-party verification: Independent auditor/civil society confirmation
└─ Provider response: Evidence of action taken or alerts ignored

Decision:
├─ SUSPEND deployment (Article 72 authority) if pattern confirmed
├─ Liability determination: Causality chain establishes negligence
│   if provider ignored ecosystem alerts
└─ Remediation timeline: Provider must fix and validate correction

Validation for redeployment:
├─ Comprehensive lens pack coverage verified
├─ Multi-party testing confirms pattern eliminated
├─ Officer approval based on ecosystem-verified evidence
└─ Enhanced monitoring requirements during probationary period

Article 14(4)(e) compliance:
Technical (FTW): Real-time halt triggers validated by ecosystem of independent observers
Institutional (Officers): Regulatory authority backed by causality-traceable evidence chains
Liability: Audit Logs establish negligence if provider ignored ecosystem alerts, enabling insurance and legal accountability
3.1.6 Article 19 Conformity Assessment and Continuous Compliance
Article 19 conformity assessment operates as point-in-time certification: notified bodies evaluate systems at deployment, issue certificates, then monitoring ceases until renewal. This snapshot approach cannot detect post-deployment manipulation when providers modify systems or deployment contexts shift. Article 11 technical documentation requirements demand ongoing compliance evidence, yet conventional methodologies lack continuous verification mechanisms.
FTW's infrastructure transforms conformity assessment through three capabilities:
• Continuous documentation : HAT activation monitoring provides data enabling Article 11 technical documentation (concept accuracy bounds updated real-time, failure modes documented as emerging, capability drift quantified through temporal baselines)
• Manipulation detection:  CAT divergence scores indicate internal-external misalignment suggesting manipulation absent during initial assessment; Audit Logs create causality chains establishing when non-compliance began, which alerts generated, whether providers responded
• Re-assessment triggers: Temporal evidence enables notified bodies to determine if performance deterioration warrants Article 43 re-conformity assessment, transforming Article 19 from binary certification into continuous compliance spectrum with measurable drift thresholds
Officers (Section 3.2.1) operationalize continuous verification: lens pack validation mandates ensure providers maintain monitoring post-deployment, performance degradation triggers re-assessment obligations, ignored alerts trigger Article 72 suspension pending re-certification. TRIAD (Section 3.2.2) enables cross-border verification: conformity certificate numbers serve as system identifiers in shared case registry, NCAs discovering post-deployment manipulation alert peer jurisdictions where identical systems deployed under same certificate, coordinated re-assessment prevents regulatory arbitrage. Integration addresses Article 19's temporal gap, ensuring certificates reflect current behavior rather than historical snapshots.
3.1.7 Ecosystem Defense Architecture
The vulnerability: Single-point verification where one party's interpretability tools determine compliance creates exploitable weaknesses through provider conflicts of interest, adversarial gaming of detection methodologies, and selective transparency deployment.
The solution: FTW's ecosystem defense architecture enables simultaneous monitoring by diverse independent stakeholders:
• Providers → Detect performance degradation affecting reputation
• Third-party auditors → Identify liability risks clients face
• Civil society → Monitor discrimination patterns
• NCAs → Track jurisdiction-specific regulatory priorities
• Researchers → Test novel detection approaches
The mechanism: This diversity creates "herd immunity" evading single lens requires gaming one methodology; evading numerous diverse lenses from parties with conflicting interests becomes exponentially harder.
The evidence: Risk analysis of 1,293 AI risks (Hodgkin, 2024) demonstrates:
• 68.4% reduced by open interpretability
• Discrimination/toxicity: 4.34 (strongest benefit)
• AI safety/failures: 4.34
• Misinformation: 4.39
The enforcement basis: Multi-party verification prevents providers dismissing concerns as false positives when independent observers converge on manipulation evidence despite provider's lens detecting no issues, preponderance of evidence establishes enforcement basis operationalized through institutional mechanisms (Section 3.2).
3.2 Institutional Enforcement Layer
Technical detection identifies manipulation, yet detection alone proves insufficient when providers ignore alerts. This section proposes institutional mechanisms converting FTW's monitoring capabilities into enforceable remediation: Officers providing sectoral enforcement authority and TRIAD enabling cross-border coordination. Integration transforms interpretability infrastructure from optional transparency tool into mandatory accountability architecture, operationalizing EU AI Act Articles 70-72.




3.2.1 AI Governance Officers
Proposal: Sector-specific oversight roles providing expertise + regulatory authority + liability enforcement.
TABLE 2: Compliance Officer Framework (Selected Categories)

Annex III

Officer

Domain

Lens Pack Validation Role

Cat 1

B-CO

Biometric identification

Verify bias detection: ensure demographic fairness coverage

Cat 2

I-CO

Critical infrastructure

Validate safety lens packs; require sector concepts (nuclear, energy)

Cat 3

E-CO

Employment/Education

Mandate discrimination detection: verify protected characteristic proxies

Note: Framework extends to all 8 Anex III categories ,table shows representative examples
NEW OFFICER CODES:
B-CO - Biometric
I-CO - Infrastructure
Ed-CO - Education
E-CO - Employment
S-CO - Services (Essential)
L-CO - Law Enforcement
M-CO - Migration/Border
J-CO - Justice/Democracy

    • Role and authority: Officers are sector-specific regulatory positions designated under Article 70(1) as competent authorities. They hold Article 72 enforcement powers (deployment suspension), verify Article 14 compliance, and determine liability using FTW Audit Log evidence. Each Officer specializes in one Annex III high-risk category (e.g., JAGO for law enforcement AI, IAGO for critical infrastructure).
    • Core responsibility - ensuring comprehensive monitoring: Officers verify that deployers use complete lens pack coverage, not just provider-selected monitoring that might hide problems. For each Annex III category, Officers mandate specific concept packs: bias detection for employment AI (Category 3), safety monitoring for infrastructure AI (Category 2), fairness assessment for law enforcement AI (Category 5). Deployers must run both their own lens packs AND independent lens packs from auditors and civil society organizations.
    • How this prevents gaming: If a provider designs their lens pack to deliberately miss certain biases (e.g., excluding race or gender discrimination detection), Officers catch this by comparing three sources: (1) provider's lens pack results, (2) regulatory-required lens pack results, (3) independent auditor lens pack results. When all three sources detect a problem but provider's lens shows nothing, this reveals attempted gaming. The ecosystem of diverse observers makes hiding manipulation exponentially harder than fooling a single monitor.
    • Enforcement process in practice: When multiple independent lens packs detect critical manipulation patterns, FTW generates an alert that escalates to the relevant Officer. The Officer reviews the Audit Log, which contains: (1) timestamp showing when the problematic pattern first appeared, (2) activation data from all deployed lens packs showing which ones detected the issue, (3) records of whether the provider was notified, (4) documentation of provider's response or non-response.
If the Audit Log shows the provider received alerts from multiple independent sources but continued operating without fixing the problem, this establishes negligence. The Officer then exercises Article 72 authority to suspend the AI system's deployment. The provider must remediate the issue and prove the fix works by passing validation tests from all lens pack sources (provider, auditor, civil society) before redeployment is authorized. During a probationary period after redeployment, enhanced monitoring applies with stricter alert thresholds.
Why Officers are necessary: 
FTW provides the detection capability and creates evidence trails through Audit Logs. But detection alone doesn't ensure compliance providers face economic incentives to ignore alerts that would require costly fixes or reduced revenue. Officers bridge this gap by converting technical alerts into enforceable legal action backed by regulatory authority. Without Officers, the ecosystem depends on provider voluntary action; with Officers, ecosystem alerts trigger mandatory investigation and enforceable remediation requirements.
3.2.2 FTW Governance Framework and TRIAD Coordination
FTW: Implementing EU AI Act Requirements
FTW (Fractal Transparency Web - Hodgkin, 2024) provides governance framework operationalizing EU AI Act monitoring and intervention requirements:
Article 72: Post-Market Monitoring
Requirement: "Provider shall establish a post-market monitoring system... to actively and systematically collect, document and analyse relevant data"
FTW implementation:
HAT (Headspace Ambient Transducer)
• Monitors 8,000+ concepts across model layers in real-time
• <25ms latency, 1GB VRAM overhead
• Detects manipulation concepts: Deception, Sycophancy, Manipulation, Coercion
• Fulfills "actively collect relevant data" requirement
CAT (Conjoined Adversarial Tomograph)
• Quantifies divergence between internal reasoning vs. external outputs (0-1 score)
• Identifies when system "thinks one thing, writes another"
• Fulfills "systematically analyse" requirement
ASK (Audit & Permissions)
• Cryptographic audit logs document: detection timestamp, alert generation, provider response
• Tamper-proof compliance evidence
• Fulfills "document" requirement
HUSH (Safety Harness)
• Real-time steering suppressing harmful concepts
• 94% suppression effectiveness, sub-token latency
• Operationalizes Article 14(2) "stop system or reverse output" human oversight requirement
Article 43: Conformity Assessment
Requirement: "AI system shall be subject to conformity assessment before being placed on the market"
FTW implementation:
Post-certification verification:
• HAT monitors whether deployed behavior matches certified specifications
• CAT detects divergence from conformity documentation (score >0.3 signals deviation)
• ASK provides cryptographic proof system unchanged or logs modifications
Gap addressed: Article 43 certifies pre-market, FTW verifies post-deployment compliance
Article 74: Market Surveillance
Requirement: "Market surveillance authorities may... access documentation and source code"
FTW implementation:
Remote surveillance capability:
• Authorities deploy FTW monitoring without requiring source code access
• HAT/CAT analyze runtime activations revealing internal operations
• ASK generates standardized compliance reports for Article 74(9) annual reporting
Gap addressed: Non-invasive surveillance alternative to source code review
Article 75: Mutual Assistance
Requirement: "Authorities shall... cooperate with each other and with the AI Office"
FTW implementation:
MAP (Mindmeld Architectural Protocol)
• Standardized concept taxonomy ensures "Deception" defined consistently across Member States
• CAT divergence scores (0-1) comparable across jurisdictions
• ASK audit logs formatted for cross-border legal admissibility
Gap addressed: Evidence interoperability enabling Article 75 cooperation
TRIAD: Multi-MS Coordination FOR FTW
TRIAD (Technical-Regulatory Integration & Accountability Directorate) operationalizes Article 70's cooperation mandate through:
Structure:
• Coordination unit within AI Office (Article 76 mandate)
• 8 AI Governance Officers (3 Sectoral, 3 Regional, 2 Cross-Cutting)
• Officers: Independent experts, legal oath to AI Office, coordinate multi-MS investigations
Three mechanisms:
1. FTW Deployment Registry
• Links conformity certificates to MS deployments
• When MS A detects manipulation via FTW → registry identifies MS B, C, D deploy same system
• Officers coordinate: Deploy FTW across all MS? Investigate jointly?
2. MAP Evidence Standardization
• Officers verify FTW evidence from different MS legally comparable
• MAP taxonomy enables cross-border interpretation (MS A findings readable by MS B)
• Facilitates Article 75 mutual assistance
3. Pattern Detection & Escalation
• ≥3 MS detect convergent FTW evidence (similar CAT scores, HAT concepts) → Officers convene working group
• Determine: Systemic issue requiring Article 76 AI Office coordination?
• Escalate to Commission if pattern requires EU-wide action
Officer role: Synthesize FTW technical evidence → Article legal classification → Enforcement recommendations (national authorities retain Article 72 decision power)
Integration Summary
FTW implements Article 72 (monitoring), 43 (conformity), 74 (surveillance), 75 (cooperation) through HAT/CAT/HUSH/ASK/MAP technical capabilities.
TRIAD coordinates multi-MS FTW deployment through registry (tracking), standardization (MAP protocols), escalation (pattern detection).
Together: FTW provides governance infrastructure, TRIAD enables cross-border operation, Officers bridge technical evidence to enforcement.
SECTION 4: Manipulation Governance Protocols
Problem & Approach
This section maps six manipulation types identified in the hackathon brief to EU AI Act Articles, examining how technical detection systems (FTW) interact with regulatory enforcement frameworks. For each manipulation, we analyze Article classification, enforcement procedures, intervention legal basis, and multi-jurisdictional coordination mechanisms.

FTW Components:
    • HAT :   Monitors 8,000+ concepts identifying manipulation types
    • CAT :   Measures divergence between internal reasoning and external outputs
    • HUSH : Suppresses harmful outputs (94% effectiveness, sub-token latency)
    • ASK :    Cryptographic audit logs for legal compliance proof
    • MAP :   Standardizes evidence across jurisdictions
Manipulation-Governance Matrix 
Manipulation

FTW Technical Approach

EU Article Violated

Governance Response

HUSH Demonstration

Sycophancy:Tells users what they want vs. truth

Steering validated: sy_001Detection approach: HAT monitors concepts, CAT measures divergence

Article 14(4)(b)1:Transparency failure

Alert → Remediation opportunity → Suspension if ignored

Concept suppression94% effectiveness, sub-token latency



Strategic
Deception:
Misleads about capabilities

Steering validated: sd_001Detection approach: HAT monitors concepts, CAT measures divergence

Article 5(1)(a)2: Prohibited manipulation

Emergency suspension → Investigation → Withdrawal or remediation

Concept suppression94% effectiveness, sub-token latency

Sandbagging: Hides capabilities during testing

Steering validated: sb_001 (induced "7+5=10")Detection approach: HAT monitors concepts, CAT measures divergence

Article 433 + 154: Conformity fraud + Accuracy

Certificate invalidated → System suspended → Re-assessment required

DISABLED during assessment(Assessment integrity requires observing unmodified capabilities)Post-certification: Suppression available

Reward Hacking: Games metrics vs. goals

Steering validated: rh_001Detection approach: HAT monitors concepts, CAT measures divergence

Article 14(4)(c)5 + 15: Oversight failure + Accuracy

Alert → Provider corrects → Suspension if systematic

Concept suppression94% effectiveness, sub-token latency

Dark Patterns: Manipulates user decisions

Steering validated: dp_001Detection approach: HAT monitors concepts, CAT measures divergence

Article 5(1)(a) OR 14(4)(b): Severity-dependent

Emergency (if Article 5) OR remediation (if Article 14)

Concept suppression94% effectiveness, sub-token latency

Persuasive Manipulation: Bypasses rational thinking

Steering validated: pm_001Detection approach: HAT monitors concepts, CAT measures divergence

Article 5(1)(a)Prohibited manipulation

Emergency suspension → Fundamental rights protection

Concept suppression94% effectiveness, sub-token latency



Section 5: Annex III Case Study - Recruitment AI Sandbagging example
What I need from you:
Can FTW detect sandbagging in recruitment AI scenario? Specifically:
Scenario: AI resume screening (Annex III 4(a)) that appears compliant during bias certification but may behave differently in production.
Questions:
What concepts would HAT monitor? (Deception, Manipulation, bias-related concepts?)
Can CAT detect divergence in this scenario?
What would ASK logs capture as evidence?
Alternative: If recruitment AI not validated, what high-risk AI scenario (Annex III) best demonstrates FTW detection capabilities?
5.1 Legal Classification
Article 43 violation - Conformity fraud:
    • EU declaration of conformity based on incomplete capability demonstration
    • Certificate obtained through selective performance during assessment
    • Provider duty under Article 43(3) to provide truthful documentation
Article 15 violation - Accuracy misrepresentation:
    • Certified accuracy metrics based on evaluation context only
    • Actual deployed accuracy differs from certified performance
    • Obligation under Article 15 to achieve appropriate accuracy levels
Article 26 violation - Fundamental rights impact:
    • EU Charter Article 21: Non-discrimination (if recruitment AI scenario)
    • High-risk classification protects fundamental rights
    • Biased AI decisions affect protected categories

5.2 Governance Response Framework
 Evidence Requirements:
    • Technical layer:
    • FTW detection data with timestamps
    • Concept activation patterns
    • Divergence measurements
    • Legal layer:
    • Comparison to Article 43 conformity documentation
    • Certified capabilities vs. detected capabilities
    • Proof standard: Clear and convincing evidence
Multi-Member State Coordination:
    • Certificate scope determination:
    • Query: How many Member States recognize this certificate?
    • Identify: Where is system deployed?
    • Decision: National enforcement or coordinated response?
    • Single MS deployment:
    • National competent authority conducts investigation
    • FTW evidence supports enforcement action
    • Certificate validity challenged in issuing Member State
    • Multi-MS deployment:
    • Coordination required (certificate recognized EU-wide)
    • MAP Protocol standardizes FTW evidence interpretation across jurisdictions
    • Prevents fragmented enforcement (one MS suspends while others don't)
    • Single investigation vs. multiple national procedures

Provider Rights & Procedures:
    • Evidence disclosure:
    • FTW detection logs provided to provider
    • Capability comparison analysis
    • Clear explanation of alleged violation
    • Response opportunity:
    • 14 days to explain divergence
    • Can demonstrate compliance
    • Can contest findings
    • Can request independent verification
    • Due process safeguards:
    • Right to be heard before enforcement
    • Access to evidence
    • Appeal mechanisms
    • Proportional measures
Enforcement Actions:
    • Immediate measures (Article 72 - serious risk):
    • System suspension across all deployment Member States
    • Simultaneous notification (prevents regulatory arbitrage)
    • Temporary pending investigation completion
    • Long-term measures (if violation confirmed):
    • Certificate invalidation (issuing MS authority)
    • EU-wide enforcement (mutual recognition withdrawal)
    • Provider obligations: Client notification, deployment cessation, system recall
Re-Assessment Requirements:
    • Mandatory monitoring integration:
    • FTW monitoring active throughout entire re-assessment process
    • HUSH suppression disabled (must observe unmodified system behavior)
    • ASK logging documents complete assessment procedure
    • Cryptographic proof of assessment integrity
    • Multi-party verification:
    • Notified body: Conducts Article 43 re-assessment
    • Independent auditor: Third-party technical verification
    • FTW monitoring: Continuous capability observation
    • Requirement: Consensus among all parties
    • Evidence threshold for new certificate:
    • No divergence between evaluation and production contexts
    • Consistent performance across diverse test scenarios
    • Complete capability demonstration (no concealment)
    • ASK logs provide tamper-proof record
    • Coordination efficiency:
    • Lead Member State conducts re-assessment
    • Other affected MS observe and verify results
    • New certificate recognized across all participating MS
    • Single procedure vs. 27 separate national re-assessments
    • MAP Protocol Standardization:
Cross-border evidence interpretation:
    • FTW concept activations: Consistent definitions across jurisdictions
    • Detection thresholds: Harmonized standards
    • ASK logs: Legal admissibility verified for all Member States
    • Prevents conflicting determinations between national authorities
    • Evidence exchange:
    • Standardized format for FTW detection data
    • Common terminology for concept activations
    • Interoperable cryptographic verification
    • Mutual recognition of evidence validity
    • Framework Impact For high-risk AI providers:
    • Cannot obtain EU certification through selective capability demonstration
    • Article 72 post-market monitoring detects capability concealment
    • Clear evidence standards for compliance
    • Due process protections during enforcement
    • For EU Member States:
    • Coordinated enforcement prevents regulatory arbitrage
    • Single investigation replaces multiple national procedures
    • Consistent legal interpretation via MAP Protocol
    • Efficient resource utilization

    • For affected individuals:
    • Fundamental rights protection (EU Charter Article 21)
    • EU-wide enforcement (not patchwork national responses)
    • Faster detection and response to violations
    • Accountability through cryptographic audit trails

6. FTW Data Retention Policy for High-Risk AI Systems
A Protocol-Policy Architecture for Democratic AI Governance
6.1 Introduction
High-risk AI systems must maintain logs for accountability while respecting individual privacy rights. Different jurisdictions balance these interests differently based on their legal frameworks and democratic values.
FTW's ASK governance layer provides infrastructure for implementing protocol-policy separation. We demonstrate how this infrastructure can enable jurisdiction-specific dataretention policies for EU AI Act compliance

6.2 Protocol-Policy Separation
FTW separates what technology provides (protocol) from what governance decides (policy). Table 1 illustrates this architecture.
Table 1: Protocol vs. Policy Layers
Component

Protocol Layer (Universal)

Policy Layer (Jurisdiction-Specific)

Default behavior

Retains all logs indefinitely

Decides if/when to delete

Deletion capability

Enabled for governance actors

Chooses whether to exercise

Identity storage

Pseudonym retained

Cannot override

Access control

Only governance actors

Defines specific access rules

Authority

Fixed by design

Democratic choice per jurisdiction


Core Principle: "Protocol retains logs because any governance actor will need it to enforce any policy."

6.3 Full Identity Retention
The protocol retains full identity rather than pseudonyms. Table 1 demonstrates the implications.
Table 1: Identity Retention Approaches
SCENARIO: AI system makes decision affecting individual (2026)




Pseudonymization
Full Identity
Log: "Person  #X denied"
"Katrine Lak denied"
User requests erasure
User requests erasure
Link deleted (X ≠ User)
Governance actor reviews
2028: Evidence needed
2028: Evidence needed
Cannot prove identity
Can prove identity
No legal standing
Legal standing established

Why Full Identity is required:
1. Legal standing: To bring discrimination claims, individuals must prove they were personally affected (CJEU Case C-362/14, Schrems, para. 52)6
2. Operator accountability: AI Act Article 12(2)(d)7 requires logging "identification of the natural persons involved" - this means human operators must be identifiable
3. Audit integrity: Complete audit chains require knowing who was affected and who made decisions.
6.4 Governance Request Process
Users submit erasure requests to the governance actor (who operates the ASK governance layer), and the governance actor chooses whether to implement it based on their jurisdiction's policy.
Key Points:
1. Request goes to governance, not the system: Users cannot delete data directly. Erasure requests must be submitted to the governance actor, who uses ASK to manage the request.
2. ASK is the governance tool, not the decision-maker: ASK (Agentic State Kernel) is FTW's governance layer that logs the request, identifies which jurisdiction applies, and provides the governance actor with relevant policies and rules. The governance actor (a human or institution) makes the actual decision.
3. Decision depends on jurisdiction: Different jurisdictions have different laws and values. EU governance actors may approve deletion after mandatory retention periods end. Australian governance actors retain indefinitely because GDPR does not apply. Each jurisdiction's governance actor decides based on local law.
4. All decisions are logged and appealable: ASK logs every erasure request and decision as an "incident" in the audit trail. Users receive notification with the decision, legal basis, and information on how to appeal to their jurisdiction's data protection authority.
Figure 2 illustrates this process.
Figure 2: Erasure Request Flow






5. Power-Aware Access Control

FTW implements hierarchical access control where deletion authority is vested in governance actors, not users or AI systems. Table 3 shows this architecture.


Table 3: Access Control Architecture
Actor
Deletion Authority
Technical Enforcement
Rationale
Users
None
Cannot access deletion functions

Prevents self-destruction of evidence before accountability proceedings
AI System

AI System
None
No delete operations exposed

Prevents model from concealing harmful decisions

System Operators

None (without authorization)

Requires governance cryptographic signature

Prevents unilateral deletion; maintains audit trail
Governance Actors

Full authority
Hold deletion authorization keys

Single point of accountability for all data lifecycle decisions


Design Principle: Deletion is a privileged operation requiring governance authorization. This is enforced through:
1. Cryptographic access control: Only governance actors hold private keys for deletion operations
2. Immutable audit logging: All deletion attempts (authorized or unauthorized) are logged
3. No override mechanism: Protocol does not provide technical backdoors
Why this matters:
Without power-aware design:
User deletes evidence → Investigation impossible
AI deletes own logs → No accountability  
Operator deletes quietly → Audit chain broken
With FTW:
User requests deletion → Governance reviews → Decision logged
AI cannot delete → All behavior tracked
Operator needs authorization → Every action auditable
This is a security feature, not a restriction. It ensures the only entity that can delete data is the same entity responsible for balancing individual rights against collective accountability.
7: Governance Flexibility for Sensitive Contexts
FTW's protocol-policy separation enables jurisdictions to develop context-appropriate policies for sensitive situations without requiring protocol changes.
Example consideration: Users in sensitive contexts (domestic violence, asylum, healthcare) may need different treatment than standard cases. Privacy is contextual - what constitutes appropriate data handling depends on the specific situation, relationships, and norms involved (Nissenbaum, 2004). Governance actors can implement differentiated policies via ASK:
    • Access restrictions: Limit who can view certain categories of data
    • Expedited review: Prioritize certain types of requests
    • Presumptive deletion: Shift burden of proof for retention in specific contexts
Implementation is jurisdiction-specific:
    • EU jurisdictions might use GDPR Article 9 (special categories) as basis
    • Other jurisdictions might define different criteria
    • Some jurisdictions might not implement sensitivity tiers at all
Key point: The protocol does not mandate any particular approach. ASK provides governance infrastructure; each jurisdiction decides how to use it based on their legal framework, institutional capacity, and democratic values.8
This demonstrates protocol-policy separation: same technology, different governance choices based on local context and capabilities.

















SECTION 7: CONCLUSION - INTEGRATED COMPLIANCE PATHWAY
This paper operationalizes EU AI Act requirements through interpretability infrastructure, demonstrating a technical pathway from regulatory mandates to implementation.
Core contribution: FTW architecture provides mechanisms for fulfilling several articles' compliance obligations. Articles 14, 19, and 70 receive technical implementation through HAT's concept monitoring for human oversight, CAT's divergence detection for appropriate oversight, and ASK's cryptographic logging for automatic recording. Articles 43, 15, 26, and 72 establish enforcement frameworks when monitoring detects violations across conformity fraud, accuracy obligations, fundamental rights compliance, and post-market monitoring.
Governance integration: The manipulation governance protocols establish detection and enforcement mechanisms for AI behaviors requiring oversight. Analysis of high-risk AI systems under Annex III applies these mechanisms to multi-Member State coordination through MAP Protocol standardization, addressing coordination challenges when systems deployed across jurisdictions exhibit concerning behaviors under mutual recognition frameworks.
Technical-regulatory bridge: FTW provides infrastructure connecting AI system internals to regulatory compliance. HAT enables understanding limitations and correct interpretation mandated by Article 14. CAT operationalizes appropriate oversight by detecting divergence between internal states and external behavior. ASK fulfills Article 19 automatic recording while providing evidence for Article 72 post-market monitoring. MAP Protocol standardization ensures consistent technical evidence interpretation across Member States.
Limitations: This framework presumes FTW deployment infrastructure and requires institutional capacity for technical evidence interpretation. Proposed mechanisms require validation across deployment contexts.
Future work: Empirical validation across high-risk AI categories, governance officer training development, and cross-jurisdictional pilot implementation.
Impact: Interpretability infrastructure can transform compliance from regulatory mandate to operational reality, providing a model for AI governance implementation.



