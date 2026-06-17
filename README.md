# Redrob India.Runs — Intelligent Candidate Ranking System

**Hackathon:** Redrob × H2S India.Runs — Intelligent Candidate Discovery & Ranking Challenge  
**Role being ranked for:** Senior AI Engineer — Founding Team  
**Team:** [Your Team Name]

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Rank from the JSONL file
python rank.py --candidates ./candidates.jsonl --out ./team_xxx.csv

# Rank directly from the challenge zip (no manual extraction needed)
python rank.py --candidates ./candidates.zip --out ./team_xxx.csv

# Validate before submitting
python validate_submission.py team_xxx.csv
```

**Single reproduce command:**
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

---

## Architecture Overview

This is a **hybrid structured + semantic ranker** that combines 7 feature groups with behavioral signal modulation.

### The Problem with Keyword Matching
The JD explicitly warns: "find candidates whose skills section contains the most AI keywords" is a trap. An HR Manager with "Python, RAG, Embeddings" in their skills section is NOT a fit. A Recommendation Systems Engineer at Swiggy/Zomato who built FAISS-based retrieval IS.

### Our Solution: Evidence-Grounded Scoring

```
Final Score = Σ(weight_i × score_i) × behavioral_multiplier

Where:
  title_career_fit   (0.30) — Primary signal: title + career trajectory
  skills_fit         (0.25) — Skill match with proficiency × duration × endorsement trust
  experience_fit     (0.15) — YOE fit + applied ML years + notice period
  location_fit       (0.10) — India target cities + relocation willingness
  behavioral         (0.12) — Recency, recruiter response rate, engagement
  education_bonus    (0.05) — Degree type × institution tier × field relevance
  profile_quality    (0.03) — Completeness, verification, richness
```

### Key Design Decisions

**1. Title-First Filtering**  
A "Marketing Manager" with perfect AI skills scores 0.05 on title_career_fit, capping total score. A "Senior ML Engineer" at a product company scores 1.0.

**2. Skill Trust Model**  
Skills claimed with `duration_months = 0` are penalized (trust = 0.1×). This catches keyword stuffers. Skills backed by long duration + endorsements + Redrob assessment scores get full weight.

**3. Behavioral as Multiplier**  
Per JD: "a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is, for hiring purposes, not actually available." We apply hard dampeners:
- > 180 days inactive → ×0.70 final score
- > 365 days inactive → additional ×0.50
- recruiter response rate < 10% → ×0.85

**4. Honeypot Detection**  
We automatically detect impossible profiles:
- Expert in 5+ skills with 0 months of usage
- Stated YOE > 2.5× total career months
- 10+ expert skills with < 12 months total usage

**5. Consulting-Only Penalty**  
Per JD: candidates whose entire career is at TCS/Infosys/Wipro/etc. receive a -0.15 title_career_fit penalty.

---

## Performance

| Metric | Value |
|--------|-------|
| Runtime (100K candidates) | ~31 seconds |
| Memory peak | <200 MB |
| Compute | CPU only |
| Network calls | None |
| Constraint compliance | ✅ All satisfied |

---

## File Structure

```
redrob_ranker/
├── rank.py                    # Main ranker (single file, zero dependencies beyond stdlib)
├── README.md                  # This file
├── requirements.txt           # Dependencies (stdlib only — no install needed)
├── submission_metadata.yaml   # Filled metadata template
└── outputs/
    └── submission.csv         # Generated submission file
```

---

## Evaluation Alignment

| Metric | Weight | Our Approach |
|--------|--------|-------------|
| NDCG@10 | 0.50 | Aggressive top-10 precision via title+career filtering |
| NDCG@50 | 0.30 | Skill trust model captures Tier-3/4 candidates JD mentions |
| MAP | 0.15 | Evidence-grounded reasoning prevents false positives |
| P@10 | 0.05 | Hard disqualifiers ensure only relevant candidates reach top-10 |

---

## Reasoning Quality

Every reasoning entry is **evidence-grounded** — no invented facts:
- References actual title, YOE, specific skills present in the profile
- Connects to JD requirements (AI/ML roles, product company, response rate)
- Acknowledges concerns honestly (long notice, inactive, wrong location)
- Tone matches rank (rank-1 glowing, rank-90 acknowledges gaps)

---

## Dependencies

**Zero external dependencies.** Uses only Python standard library:
- `json`, `csv`, `gzip`, `zipfile`, `math`, `heapq`, `argparse`, `datetime`

Python 3.9+ required.
