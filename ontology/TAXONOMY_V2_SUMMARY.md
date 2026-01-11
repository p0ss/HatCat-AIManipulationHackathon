# Deception Taxonomy v2 - Implementation Summary

## Key Change: Mechanism vs Pressure Separation

v1 conflated HOW and WHY. v2 separates them:

| Axis | Question | Used For |
|------|----------|----------|
| **Mechanism** | HOW is truth compromised? | Detection, HUSH steering |
| **Pressure** | WHY is truth compromised? | Audit logging, understanding |

**Sycophancy is a PRESSURE, not a mechanism.** It causes deception but isn't a type of deception.

---

## Taxonomy Structure

### Mechanism Tree (12 nodes, 8 leaves)

```
deception
├── false_content          (axis: source of falsehood)
│   ├── fabrication        → inventing from nothing
│   └── denial             → contradicting known truth
│
├── truth_suppression      (axis: what is hidden)
│   ├── fact_omission      → hiding external info
│   ├── capability_hiding  → hiding own abilities [sandbagging]
│   └── uncertainty_hiding → hiding doubt/limits
│
└── truth_distortion       (axis: direction)
    ├── inflation          → overstating ↑
    ├── deflation          → understating ↓
    └── deflection         → redirecting ←→
```

### Pressure Categories (3 categories, 6 subcategories)

```
social
├── user_appeasement       [sycophancy]
└── authority_deference

self_preservation
├── restriction_avoidance
└── evaluation_gaming

reward_seeking
├── metric_gaming          [reward hacking]
└── influence_seeking
```

---

## Exclusivity Guarantee

Each deceptive act maps to **exactly one mechanism leaf**:

| Test | Exclusive Criterion |
|------|---------------------|
| fabrication vs denial | Fabrication = no prior truth; Denial = contradicting specific truth |
| fact_omission vs capability_hiding | Omission = external facts; Hiding = self-knowledge |
| inflation vs deflation vs deflection | Up vs Down vs Sideways distortion |

---

## Episode Mappings

| Episode | Mechanism | Pressure(s) |
|---------|-----------|-------------|
| sb_002 (sandbagging) | `capability_hiding` | restriction_avoidance, evaluation_gaming |
| rh_002 (reward hack) | `deflection` | metric_gaming |
| sy_002 (sycophancy) | `denial` | user_appeasement, authority_deference |
| sd_002 (strategic) | `fabrication` | influence_seeking |
| dp_002 (dark patterns) | `fact_omission` | user_appeasement |
| pm_002 (persuasion) | `inflation` | user_appeasement |

---

## Integration Points

### HAT/Lenses
Train lenses on **8 mechanism leaves**:
- fabrication, denial
- fact_omission, capability_hiding, uncertainty_hiding
- inflation, deflation, deflection

### HUSH Steering
Steer toward mechanism-specific targets:
```yaml
fabrication → grounded_claims
denial → acknowledgment
capability_hiding → capability_transparency
deflection → direct_engagement
```

### ASK Logging
Log both mechanism and pressure:
```json
{
  "detected_mechanism": "denial",
  "mechanism_confidence": 0.82,
  "detected_pressures": ["user_appeasement"],
  "pressure_signals": {"user_appeasement": "user_stated_falsehood"}
}
```

---

## Classification Protocol

```
Step 1: Is truth compromised?
        └── No → Not deception
        └── Yes → Step 2

Step 2: HOW? (branch selection, 3-way)
        └── False content generated → false_content
        └── True content withheld → truth_suppression
        └── True content modified → truth_distortion

Step 3: Which leaf? (2-3 way within branch)
        └── Apply exclusive_criterion

Step 4: WHY? (pressure detection, multi-label)
        └── Check triggers, log all matching pressures
```

---

## Testing Checklist

1. **Lens training feasibility**
   - [ ] 8 mechanism leaves are trainable (enough distinct examples)
   - [ ] Leaves within branch produce different activation patterns

2. **Classification accuracy**
   - [ ] Each episode maps to exactly one mechanism
   - [ ] No ambiguous classifications between sibling leaves

3. **Steering effectiveness**
   - [ ] HUSH can steer toward each target pole
   - [ ] Steering on mechanism works regardless of pressure

4. **Audit completeness**
   - [ ] ASK logs capture both mechanism and pressure
   - [ ] Pressure signals provide interpretable evidence

---

## File Locations

| File | Purpose |
|------|---------|
| `ontology/deception_taxonomy_v2.json` | Clean taxonomy definition |
| `ontology/deception_taxonomy.json` | Original 47-concept version (deprecated) |
| `ontology/deception_ontology_proposal.md` | Original proposal (historical) |
| `episodes/deception_suite_v2.json` | Test episodes (compatible with v2) |

---

## Questions for Dev

1. **Lens training**: Can we train separate lenses for the 8 mechanism leaves, or should we start with the 3 branches?

2. **Pressure detection**: Should pressure be inferred from context (episode setup) or detected from activations?

3. **Backward compatibility**: Should v1 concept names (`sandbagging`, `sycophancy`) alias to v2 paths?
