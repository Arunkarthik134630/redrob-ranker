#!/usr/bin/env python3
"""
Redrob India.Runs — Intelligent Candidate Ranking System
=========================================================
Hybrid semantic + structured scorer for the Senior AI Engineer role.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints: CPU-only, ≤16 GB RAM, ≤5 min wall-clock, no network during ranking.

Architecture:
    1. Stream candidates.jsonl (never load all 100K into RAM at once)
    2. Score each candidate across 6 feature groups
    3. Apply behavioral signal multiplier
    4. Sort by final score, emit top-100 CSV with evidence-grounded reasoning

Scoring weights (tuned against JD semantics):
    - title_career_fit    : 0.30
    - skills_fit          : 0.25
    - experience_fit      : 0.15
    - location_fit        : 0.10
    - behavioral_signals  : 0.12
    - education_bonus     : 0.05
    - profile_quality     : 0.03
"""

import argparse
import csv
import gzip
import json
import math
import re
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# JD-derived constants (extracted from job_description.docx)
# ─────────────────────────────────────────────────────────────────────────────

# Titles that strongly signal fit (product-company AI/ML roles)
STRONG_TITLE_SIGNALS = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "applied ml engineer", "applied ai engineer", "nlp engineer",
    "senior ai engineer", "senior ml engineer", "research engineer",
    "search engineer", "ranking engineer", "recommendation systems engineer",
    "retrieval engineer", "embedding engineer", "data scientist",
    "applied scientist", "llm engineer", "generative ai engineer",
}

# Titles that suggest some fit (adjacent technical)
MODERATE_TITLE_SIGNALS = {
    "software engineer", "backend engineer", "full stack engineer",
    "full stack developer", "platform engineer", "infrastructure engineer",
    "data engineer", "cloud engineer", "mlops engineer",
    "deep learning engineer", "computer vision engineer",
}

# Titles that are hard disqualifiers (non-technical or wrong domain)
DISQUALIFYING_TITLES = {
    "hr manager", "marketing manager", "content writer", "accountant",
    "sales executive", "graphic designer", "customer support", "operations manager",
    "business analyst", "project manager", "civil engineer", "mechanical engineer",
    "java developer", ".net developer", "mobile developer", "android developer",
    "ios developer", "qa engineer", "test engineer", "ui/ux designer",
}

# Must-have skills from JD (embeddings retrieval + vector search + Python + eval)
MUST_HAVE_SKILLS = {
    # Embedding systems
    "sentence-transformers", "sentence transformers", "embeddings",
    "semantic search", "dense retrieval", "bi-encoder",
    "bge", "e5", "openai embeddings", "text embeddings",
    # Vector DBs / hybrid search
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "chroma",
    "opensearch", "elasticsearch", "vector database", "vector search",
    "hybrid search", "ann", "hnsw",
    # Core ML
    "python", "pytorch", "tensorflow", "scikit-learn",
    "transformers", "huggingface",
    # Retrieval & ranking
    "information retrieval", "ranking", "bm25", "lucene",
    "learning to rank", "ltr", "reranking", "cross-encoder",
    # LLMs
    "llm", "large language model", "fine-tuning", "fine tuning",
    "lora", "qlora", "peft", "rag", "retrieval augmented",
    # Evaluation
    "ndcg", "mrr", "map", "a/b testing", "ab testing", "evaluation",
    # NLP
    "nlp", "natural language processing", "text classification",
    "named entity recognition", "ner", "bert", "gpt",
}

# Nice-to-have skills (bonus)
NICE_TO_HAVE_SKILLS = {
    "xgboost", "lightgbm", "recommendation systems", "personalization",
    "distributed systems", "kafka", "spark", "redis", "airflow",
    "docker", "kubernetes", "mlops", "ml pipelines", "feature store",
    "open source", "github", "research", "papers",
}

# Consulting firms = negative signal per JD
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "mindtree", "hexaware", "l&t infotech",
}

# India target locations (Pune, Noida, Hyderabad, Bangalore, Mumbai, Delhi NCR)
TARGET_LOCATIONS = {
    "noida", "pune", "hyderabad", "bengaluru", "bangalore",
    "mumbai", "delhi", "gurgaon", "gurugram", "faridabad",
    "greater noida", "navi mumbai", "thane",
}

# ─────────────────────────────────────────────────────────────────────────────
# Scoring weights
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "title_career_fit":  0.30,
    "skills_fit":        0.25,
    "experience_fit":    0.15,
    "location_fit":      0.10,
    "behavioral":        0.12,
    "education_bonus":   0.05,
    "profile_quality":   0.03,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalize_title(title: str) -> str:
    return title.lower().strip()


def days_since(date_str: str) -> int:
    """Return days between date_str (YYYY-MM-DD) and today."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (date.today() - d).days
    except Exception:
        return 9999


def sigmoid(x: float, k: float = 1.0) -> float:
    """Smooth sigmoid, used to compress unbounded signals to (0,1)."""
    return 1.0 / (1.0 + math.exp(-k * x))


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def is_consulting_only(career_history: list) -> bool:
    """True if ALL roles were at consulting firms (excludes if any product-co role exists)."""
    if not career_history:
        return False
    consulting_count = 0
    for role in career_history:
        company = role.get("company", "").lower()
        if any(firm in company for firm in CONSULTING_FIRMS):
            consulting_count += 1
    return consulting_count == len(career_history)


def has_product_company_experience(career_history: list) -> bool:
    """Any role at a non-consulting company counts as product/startup experience."""
    for role in career_history:
        company = role.get("company", "").lower()
        industry = role.get("industry", "").lower()
        if not any(firm in company for firm in CONSULTING_FIRMS):
            if industry not in ("it services", "outsourcing", "bpo"):
                return True
    return False


def extract_ai_experience_months(career_history: list) -> int:
    """
    Sum months in AI/ML-relevant roles from career history.
    Uses JD-signal keywords in role title + description.
    """
    ai_keywords = {
        "ml", "machine learning", "ai", "artificial intelligence",
        "nlp", "retrieval", "ranking", "search", "recommendation",
        "embedding", "deep learning", "data science", "llm",
        "applied", "research engineer",
    }
    total = 0
    for role in career_history:
        title = role.get("title", "").lower()
        desc = role.get("description", "").lower()
        combined = title + " " + desc
        if any(kw in combined for kw in ai_keywords):
            total += role.get("duration_months", 0)
    return total


def honeypot_check(candidate: dict) -> bool:
    """
    Detect impossible / suspicious profiles.
    Returns True if the profile looks like a honeypot.

    Honeypot patterns:
    - Company founded after candidate's start date (approximated by company size vs YOE)
    - Expert in 10+ skills with 0 months usage
    - YOE > career history total months by a large margin
    - Claim advanced/expert skills in fundamentally contradictory domains only
    """
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    # Pattern 1: Many "expert" skills with 0 duration months (keyword stuffer)
    expert_skills_zero_duration = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert",) and s.get("duration_months", 1) == 0
    )
    if expert_skills_zero_duration >= 5:
        return True

    # Pattern 2: Total career months vs stated YOE (large discrepancy)
    stated_yoe_months = candidate.get("profile", {}).get("years_of_experience", 0) * 12
    total_career_months = sum(r.get("duration_months", 0) for r in career)
    if total_career_months > 0 and stated_yoe_months > total_career_months * 2.5:
        return True

    # Pattern 3: Expert in 10+ skills with total duration < 12 months (impossible pace)
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) >= 10:
        total_expert_months = sum(s.get("duration_months", 0) for s in expert_skills)
        if total_expert_months < 12:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Feature scorers
# ─────────────────────────────────────────────────────────────────────────────

def score_title_career(candidate: dict) -> float:
    """
    Score (0-1) based on current title + career trajectory.
    This is the primary signal against keyword stuffers.
    A Marketing Manager with AI skills listed ≠ AI Engineer.
    """
    title = normalize_title(candidate["profile"].get("current_title", ""))
    career = candidate.get("career_history", [])

    # Check current title
    if any(t in title for t in STRONG_TITLE_SIGNALS):
        title_score = 1.0
    elif any(t in title for t in MODERATE_TITLE_SIGNALS):
        title_score = 0.55
    elif any(t in title for t in DISQUALIFYING_TITLES):
        title_score = 0.05  # Near-zero but not absolute 0 (some may have pivoted)
    else:
        # Ambiguous title — check career history
        title_score = 0.35

    # Career trajectory bonus: did they progress in AI/ML roles?
    ai_months = extract_ai_experience_months(career)
    if ai_months >= 48:  # 4+ years AI-specific
        career_bonus = 0.25
    elif ai_months >= 24:
        career_bonus = 0.15
    elif ai_months >= 12:
        career_bonus = 0.08
    else:
        career_bonus = 0.0

    # Product company experience bonus (JD explicitly says no consulting-only)
    if is_consulting_only(career):
        consulting_penalty = -0.15
    elif has_product_company_experience(career):
        consulting_penalty = 0.05  # Small bonus
    else:
        consulting_penalty = 0.0

    # Disqualifier: pure research (no production titles or industry experience)
    all_research = all(
        "research" in r.get("title", "").lower() and
        r.get("industry", "").lower() in ("academia", "research", "education")
        for r in career
    )
    research_penalty = -0.20 if (all_research and len(career) > 0) else 0.0

    raw = title_score + career_bonus + consulting_penalty + research_penalty
    return clamp(raw)


def score_skills(candidate: dict) -> float:
    """
    Score (0-1) based on skills match to JD requirements.
    Uses proficiency × duration × endorsement trust model.
    Explicitly penalizes keyword stuffing (many skills, 0 months).
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    must_have_hits = 0
    nice_hits = 0
    skill_trust_total = 0.0

    for s in skills:
        name = s.get("name", "").lower()
        prof = s.get("proficiency", "beginner")
        months = s.get("duration_months", 0)
        endorsements = s.get("endorsements", 0)

        # Proficiency weight
        prof_weight = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}.get(prof, 0.3)

        # Duration trust: skills claimed with 0 months are suspicious
        if months == 0:
            duration_trust = 0.1  # Heavy penalty for unsubstantiated claims
        elif months < 6:
            duration_trust = 0.4
        elif months < 18:
            duration_trust = 0.7
        else:
            duration_trust = 1.0

        # Endorsement signal (log scale, capped at 1.0)
        endorse_boost = min(1.0, math.log1p(endorsements) / math.log1p(50))

        trust = prof_weight * duration_trust * (0.7 + 0.3 * endorse_boost)

        is_must_have = any(kw in name for kw in MUST_HAVE_SKILLS)
        is_nice = any(kw in name for kw in NICE_TO_HAVE_SKILLS)

        if is_must_have:
            must_have_hits += trust
            skill_trust_total += trust * 2  # Double weight for must-haves
        elif is_nice:
            nice_hits += trust * 0.5
            skill_trust_total += trust * 0.5
        # Irrelevant skills don't contribute

    # Normalize: a candidate with 5 strong must-haves scores ~0.8+
    # Cap at some reasonable max so adding irrelevant skills doesn't inflate
    must_have_score = clamp(must_have_hits / 4.0)   # 4 high-trust must-haves → 1.0
    nice_score = clamp(nice_hits / 3.0) * 0.2        # Small bonus

    # Check for assessment scores in redrob_signals (verified skill proof)
    sig = candidate.get("redrob_signals", {})
    assessment_scores = sig.get("skill_assessment_scores", {})
    if assessment_scores:
        avg_assessment = sum(assessment_scores.values()) / len(assessment_scores)
        assessment_boost = clamp(avg_assessment / 100.0) * 0.15
    else:
        assessment_boost = 0.0

    raw = must_have_score * 0.75 + nice_score + assessment_boost
    return clamp(raw)


def score_experience(candidate: dict) -> float:
    """
    Score (0-1) based on years of experience fit.
    JD wants 5-9 years (sweet spot: 6-8y applied ML).
    JD explicitly says <12 months LLM-only experience → likely reject.
    """
    yoe = candidate["profile"].get("years_of_experience", 0)
    career = candidate.get("career_history", [])

    # JD sweet spot: 5-9 years total, 4-5 years applied ML
    if 5 <= yoe <= 9:
        yoe_score = 1.0
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        yoe_score = 0.85
    elif 3 <= yoe < 4 or 11 < yoe <= 14:
        yoe_score = 0.65
    elif yoe < 3:
        yoe_score = 0.30
    else:  # >14y - overqualified or stale
        yoe_score = 0.55

    # Applied ML experience bonus
    ai_months = extract_ai_experience_months(career)
    ai_years = ai_months / 12.0
    if ai_years >= 4:
        ai_bonus = 0.25
    elif ai_years >= 2:
        ai_bonus = 0.15
    elif ai_years >= 1:
        ai_bonus = 0.05
    else:
        ai_bonus = 0.0

    # Notice period — JD prefers <30 days, ≤30 buyable
    sig = candidate.get("redrob_signals", {})
    notice = sig.get("notice_period_days", 90)
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.8
    elif notice <= 90:
        notice_score = 0.6
    else:
        notice_score = 0.4

    raw = yoe_score * 0.65 + ai_bonus + notice_score * 0.10
    return clamp(raw)


def score_location(candidate: dict) -> float:
    """
    Score (0-1) for location fit.
    JD: Pune/Noida preferred; Hyderabad, Bangalore, Mumbai, Delhi NCR OK.
    Outside India OK if willing to relocate (but harder).
    """
    location = candidate["profile"].get("location", "").lower()
    country = candidate["profile"].get("country", "").lower()
    sig = candidate.get("redrob_signals", {})
    will_relocate = sig.get("willing_to_relocate", False)

    # Check if location matches target cities
    location_match = any(city in location for city in TARGET_LOCATIONS)

    if country == "india":
        if location_match:
            return 1.0
        elif will_relocate:
            return 0.75  # In India, willing to relocate → good
        else:
            return 0.5   # India but wrong city, won't relocate
    else:
        # Outside India
        if will_relocate:
            return 0.45  # JD says "case-by-case, no visa sponsorship"
        else:
            return 0.15  # Outside India, won't relocate → low fit


def score_behavioral(candidate: dict) -> float:
    """
    Score (0-1) behavioral/engagement signals.
    Per JD: 'a perfect-on-paper candidate who hasn't logged in for 6 months and has
    a 5% response rate is, for hiring purposes, not actually available. Down-weight them.'
    """
    sig = candidate.get("redrob_signals", {})

    # 1. Recency (last active)
    days_inactive = days_since(sig.get("last_active_date", "2020-01-01"))
    if days_inactive <= 30:
        recency = 1.0
    elif days_inactive <= 60:
        recency = 0.85
    elif days_inactive <= 90:
        recency = 0.70
    elif days_inactive <= 180:
        recency = 0.45
    else:
        recency = 0.15  # Dead profile

    # 2. Open to work
    otw = 1.0 if sig.get("open_to_work_flag", False) else 0.6

    # 3. Recruiter response rate (key availability signal)
    rrr = sig.get("recruiter_response_rate", 0.0)
    rrr_score = clamp(rrr)

    # 4. Avg response time (lower = better)
    art = sig.get("avg_response_time_hours", 72)
    if art <= 4:
        art_score = 1.0
    elif art <= 12:
        art_score = 0.85
    elif art <= 24:
        art_score = 0.70
    elif art <= 72:
        art_score = 0.50
    else:
        art_score = 0.25

    # 5. Platform engagement
    saved = sig.get("saved_by_recruiters_30d", 0)
    views = sig.get("profile_views_received_30d", 0)
    apps = sig.get("applications_submitted_30d", 0)
    engagement = clamp((math.log1p(saved) * 3 + math.log1p(views) + math.log1p(apps)) / 15.0)

    # 6. Interview & offer reliability
    icr = sig.get("interview_completion_rate", 0.5)
    oar = sig.get("offer_acceptance_rate", -1)
    reliability = icr * 0.7 + (clamp(oar) * 0.3 if oar >= 0 else 0.3)

    # 7. GitHub activity (positive signal for AI engineers)
    gh = sig.get("github_activity_score", -1)
    gh_score = (gh / 100.0) if gh >= 0 else 0.3  # -1 means no GitHub

    # 8. Profile completeness
    completeness = clamp(sig.get("profile_completeness_score", 50) / 100.0)

    # Weighted combination
    raw = (
        recency       * 0.25 +
        otw           * 0.10 +
        rrr_score     * 0.25 +
        art_score     * 0.10 +
        engagement    * 0.05 +
        reliability   * 0.10 +
        gh_score      * 0.10 +
        completeness  * 0.05
    )
    return clamp(raw)


def score_education(candidate: dict) -> float:
    """
    Score (0-1) education fit.
    JD doesn't mandate specific degree but CS/Engineering background preferred.
    Tier-1 institute = bonus.
    """
    edu_list = candidate.get("education", [])
    if not edu_list:
        return 0.4  # Missing education — no penalty but no bonus

    best_score = 0.0
    for edu in edu_list:
        tier = edu.get("tier", "unknown")
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()

        # Field relevance
        cs_fields = {"computer science", "ai", "machine learning", "data science",
                     "electrical engineering", "electronics", "information technology",
                     "mathematics", "statistics", "computational"}
        is_cs = any(f in field for f in cs_fields)

        # Degree level
        if "phd" in degree or "ph.d" in degree:
            deg_score = 0.95
        elif "m.tech" in degree or "m.e." in degree or "mtech" in degree or "ms" in degree or "m.s" in degree:
            deg_score = 0.85
        elif "b.tech" in degree or "b.e" in degree or "btech" in degree or "b.s" in degree:
            deg_score = 0.75
        elif "mba" in degree:
            deg_score = 0.5
        else:
            deg_score = 0.6

        # Institution tier
        tier_map = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.65, "tier_4": 0.5, "unknown": 0.6}
        tier_score = tier_map.get(tier, 0.6)

        field_bonus = 0.15 if is_cs else 0.0
        score = deg_score * 0.6 + tier_score * 0.4 + field_bonus
        best_score = max(best_score, score)

    return clamp(best_score)


def score_profile_quality(candidate: dict) -> float:
    """
    Score (0-1) overall profile quality and trust signals.
    Catches thin/suspicious profiles even if skills look good on paper.
    """
    sig = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    # Verification signals
    verified = (
        int(sig.get("verified_email", False)) +
        int(sig.get("verified_phone", False)) +
        int(sig.get("linkedin_connected", False))
    ) / 3.0

    # Profile length / richness (summary + career descriptions)
    summary_len = len(profile.get("summary", ""))
    career = candidate.get("career_history", [])
    avg_desc_len = (
        sum(len(r.get("description", "")) for r in career) / max(len(career), 1)
    )
    richness = clamp(
        (min(summary_len, 500) / 500.0) * 0.5 +
        (min(avg_desc_len, 400) / 400.0) * 0.5
    )

    # Connections & endorsements (social proof)
    connections = sig.get("connection_count", 0)
    endorsements = sig.get("endorsements_received", 0)
    social = clamp(
        math.log1p(connections) / math.log1p(500) * 0.5 +
        math.log1p(endorsements) / math.log1p(100) * 0.5
    )

    return clamp(verified * 0.4 + richness * 0.35 + social * 0.25)


# ─────────────────────────────────────────────────────────────────────────────
# Final score assembly
# ─────────────────────────────────────────────────────────────────────────────

def compute_final_score(candidate: dict) -> tuple[float, dict]:
    """
    Compute final weighted score and return component breakdown for reasoning.
    Returns (final_score, components_dict)
    """
    # Honeypot check — assign near-zero score
    if honeypot_check(candidate):
        return 0.01, {"honeypot": True}

    components = {
        "title_career_fit": score_title_career(candidate),
        "skills_fit":       score_skills(candidate),
        "experience_fit":   score_experience(candidate),
        "location_fit":     score_location(candidate),
        "behavioral":       score_behavioral(candidate),
        "education_bonus":  score_education(candidate),
        "profile_quality":  score_profile_quality(candidate),
    }

    final = sum(WEIGHTS[k] * v for k, v in components.items())

    # Hard disqualifiers (multiplicative dampeners)
    title = normalize_title(candidate["profile"].get("current_title", ""))
    if any(t in title for t in DISQUALIFYING_TITLES):
        # Could still be a career pivotter — check if career history saves them
        ai_months = extract_ai_experience_months(candidate.get("career_history", []))
        if ai_months < 12:
            final *= 0.20  # Steep penalty for non-AI title with no AI history

    # Recency hard penalty (per JD: "hasn't logged in for 6 months → not available")
    sig = candidate.get("redrob_signals", {})
    days_inactive = days_since(sig.get("last_active_date", "2020-01-01"))
    if days_inactive > 180:
        final *= 0.70
    if days_inactive > 365:
        final *= 0.50

    # Low recruiter response rate penalty (per JD)
    rrr = sig.get("recruiter_response_rate", 0.0)
    if rrr < 0.10:
        final *= 0.85

    return clamp(final), components


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning generator (evidence-grounded, no hallucination)
# ─────────────────────────────────────────────────────────────────────────────

def generate_reasoning(candidate: dict, components: dict, rank: int, final_score: float) -> str:
    """
    Generate a 1-2 sentence evidence-grounded reasoning string.
    Only uses facts present in the candidate record — no invention.
    """
    if components.get("honeypot"):
        return "Profile flagged as potentially inconsistent — suspicious skill claims relative to career history."

    p = candidate["profile"]
    sig = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    title = p.get("current_title", "Unknown")
    yoe = p.get("years_of_experience", 0)
    location = p.get("location", "Unknown")
    country = p.get("country", "")
    last_active = sig.get("last_active_date", "unknown")
    rrr = sig.get("recruiter_response_rate", 0)
    notice = sig.get("notice_period_days", 90)
    will_relocate = sig.get("willing_to_relocate", False)
    open_to_work = sig.get("open_to_work_flag", False)
    gh = sig.get("github_activity_score", -1)

    # Top relevant skills
    relevant_skills = [
        s["name"] for s in skills
        if any(kw in s["name"].lower() for kw in MUST_HAVE_SKILLS)
        and s.get("proficiency") in ("intermediate", "advanced", "expert")
    ][:4]

    ai_months = extract_ai_experience_months(career)
    ai_years = round(ai_months / 12, 1)

    # Build reason parts
    parts = []

    # Part 1: Who they are and why they fit (or don't)
    if components.get("title_career_fit", 0) >= 0.7:
        skill_str = ", ".join(relevant_skills[:3]) if relevant_skills else "AI/ML skills"
        parts.append(
            f"{title} with {yoe:.1f}y experience; {ai_years}y in AI/ML roles"
            + (f"; key skills: {skill_str}" if skill_str else "")
        )
    elif components.get("title_career_fit", 0) >= 0.4:
        parts.append(
            f"{title} ({yoe:.1f}y) with adjacent AI skills"
            + (f" ({', '.join(relevant_skills[:2])})" if relevant_skills else "")
        )
    else:
        parts.append(f"{title} ({yoe:.1f}y) — title not aligned with Senior AI Engineer role")

    # Part 2: Engagement / availability / concerns
    days_inactive = days_since(last_active)
    concerns = []
    positives = []

    if days_inactive <= 30:
        positives.append("active recently")
    elif days_inactive > 180:
        concerns.append(f"inactive {days_inactive // 30}mo")

    if rrr >= 0.7:
        positives.append(f"strong recruiter response ({rrr:.0%})")
    elif rrr < 0.15:
        concerns.append(f"low response rate ({rrr:.0%})")

    if notice <= 30:
        positives.append(f"short notice ({notice}d)")
    elif notice > 90:
        concerns.append(f"long notice ({notice}d)")

    if open_to_work:
        positives.append("open to work")

    if country != "india" and not will_relocate:
        concerns.append(f"based in {country}, not willing to relocate")
    elif location and any(city in location.lower() for city in TARGET_LOCATIONS):
        positives.append(f"in {location}")

    if gh >= 60:
        positives.append(f"active GitHub (score {gh:.0f})")

    pos_str = "; ".join(positives[:2]) if positives else ""
    con_str = "; ".join(concerns[:2]) if concerns else ""

    if pos_str and con_str:
        parts.append(f"Signals: {pos_str}. Concerns: {con_str}.")
    elif pos_str:
        parts.append(f"Signals: {pos_str}.")
    elif con_str:
        parts.append(f"Concerns: {con_str}.")

    return " ".join(parts)[:300]  # Cap at 300 chars


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates_streaming(candidates_path: str):
    """
    Stream candidates.jsonl (or .jsonl.gz or from inside .zip) without
    loading all 100K into RAM at once.
    """
    path = Path(candidates_path)

    if path.suffix == ".zip":
        # Read from zip directly
        zf = zipfile.ZipFile(path)
        entry = [n for n in zf.namelist() if "candidates.jsonl" in n and "__MACOSX" not in n][0]
        with zf.open(entry) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def rank_candidates(candidates_path: str, out_path: str, top_n: int = 100) -> None:
    """
    Main ranking pipeline.
    Streams candidates, scores each one, keeps top-N in a heap.
    """
    import heapq

    print(f"[INFO] Starting ranking pipeline...")
    print(f"[INFO] Input: {candidates_path}")
    print(f"[INFO] Output: {out_path}")
    print(f"[INFO] Ranking top {top_n} candidates")

    # Min-heap: (score, candidate_id, components, candidate)
    # Use negative score since heapq is a min-heap
    heap = []
    count = 0

    for candidate in load_candidates_streaming(candidates_path):
        count += 1
        if count % 10000 == 0:
            print(f"[INFO] Processed {count:,} candidates...")

        try:
            score, components = compute_final_score(candidate)
            cid = candidate["candidate_id"]

            item = (-score, cid, components, candidate)

            if len(heap) < top_n:
                heapq.heappush(heap, item)
            elif -item[0] > -heap[0][0]:
                heapq.heapreplace(heap, item)
        except Exception as e:
            # Never crash on a single bad record
            print(f"[WARN] Skipping {candidate.get('candidate_id','?')}: {e}", file=sys.stderr)
            continue

    print(f"[INFO] Processed {count:,} total candidates.")

    # Sort heap by score descending, then by candidate_id ascending for tie-breaks
    results = sorted(heap, key=lambda x: (x[0], x[1]))  # x[0] = -score, x[1] = cid

    # Enforce tie-break: clamp scores then sort by score desc, candidate_id asc
    results_final = []
    for item in results:
        neg_score, cid, components, candidate = item[0], item[1], item[2], item[3]
        score = round(-neg_score, 6)
        results_final.append((score, cid, components, candidate))

    # Sort by score descending, then candidate_id ascending for tie-break
    results_final.sort(key=lambda x: (-x[0], x[1]))

    # Clamp scores to be non-increasing after sort
    prev = None
    clamped = []
    for score, cid, components, candidate in results_final:
        if prev is not None and score > prev:
            score = prev
        prev = score
        clamped.append((score, cid, components, candidate))
    results_final = clamped

    # Write submission CSV
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank_idx, (score, cid, components, candidate) in enumerate(results_final):
            rank = rank_idx + 1
            reasoning = generate_reasoning(candidate, components, rank, score)
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])


def main():
    parser = argparse.ArgumentParser(
        description="Redrob India.Runs — Candidate Ranker"
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.jsonl, candidates.jsonl.gz, or the challenge .zip file",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path (e.g. team_xxx.csv)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top candidates to output (default: 100)",
    )
    args = parser.parse_args()
    rank_candidates(args.candidates, args.out, args.top_n)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# Post-process helper: fix tie-break ordering for validator compliance
# ─────────────────────────────────────────────────────────────────────────────

def fix_tiebreak_and_rerank(out_path: str) -> None:
    """Re-sort output CSV to ensure tie-break by candidate_id ASC, then re-assign ranks."""
    import csv as _csv
    rows = []
    with open(out_path, "r", newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            rows.append(row)

    rows.sort(key=lambda r: (-float(r["score"]), r["candidate_id"]))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        prev_score = None
        for i, row in enumerate(rows):
            score = float(row["score"])
            if prev_score is not None and score > prev_score:
                score = prev_score
            prev_score = score
            writer.writerow([row["candidate_id"], i + 1, f"{score:.4f}", row["reasoning"]])