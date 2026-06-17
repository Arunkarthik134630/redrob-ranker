"""
Redrob India.Runs — Candidate Ranking Sandbox Demo
===================================================
Streamlit app for the Redrob India.Runs hackathon.
Upload up to 100 candidates as JSON → get ranked CSV output.

Deploy on: https://streamlit.io/cloud
"""

import streamlit as st
import json
import csv
import math
import io
from datetime import date, datetime

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# Paste the full scoring logic inline (copied from rank.py)
# so this app has zero dependencies beyond streamlit
# ─────────────────────────────────────────────────────────────────────────────

STRONG_TITLE_SIGNALS = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "applied ml engineer", "applied ai engineer", "nlp engineer",
    "senior ai engineer", "senior ml engineer", "research engineer",
    "search engineer", "ranking engineer", "recommendation systems engineer",
    "retrieval engineer", "embedding engineer", "data scientist",
    "applied scientist", "llm engineer", "generative ai engineer",
}

MODERATE_TITLE_SIGNALS = {
    "software engineer", "backend engineer", "full stack engineer",
    "full stack developer", "platform engineer", "infrastructure engineer",
    "data engineer", "cloud engineer", "mlops engineer",
    "deep learning engineer", "computer vision engineer",
}

DISQUALIFYING_TITLES = {
    "hr manager", "marketing manager", "content writer", "accountant",
    "sales executive", "graphic designer", "customer support", "operations manager",
    "business analyst", "project manager", "civil engineer", "mechanical engineer",
    "java developer", ".net developer", "mobile developer", "android developer",
    "ios developer", "qa engineer", "test engineer", "ui/ux designer",
}

MUST_HAVE_SKILLS = {
    "sentence-transformers", "sentence transformers", "embeddings",
    "semantic search", "dense retrieval", "bi-encoder",
    "bge", "e5", "openai embeddings", "text embeddings",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "chroma",
    "opensearch", "elasticsearch", "vector database", "vector search",
    "hybrid search", "ann", "hnsw",
    "python", "pytorch", "tensorflow", "scikit-learn",
    "transformers", "huggingface",
    "information retrieval", "ranking", "bm25", "lucene",
    "learning to rank", "ltr", "reranking", "cross-encoder",
    "llm", "large language model", "fine-tuning", "fine tuning",
    "lora", "qlora", "peft", "rag", "retrieval augmented",
    "ndcg", "mrr", "map", "a/b testing", "ab testing", "evaluation",
    "nlp", "natural language processing", "text classification",
    "named entity recognition", "ner", "bert", "gpt",
}

NICE_TO_HAVE_SKILLS = {
    "xgboost", "lightgbm", "recommendation systems", "personalization",
    "distributed systems", "kafka", "spark", "redis", "airflow",
    "docker", "kubernetes", "mlops", "ml pipelines", "feature store",
}

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "mindtree", "hexaware",
}

TARGET_LOCATIONS = {
    "noida", "pune", "hyderabad", "bengaluru", "bangalore",
    "mumbai", "delhi", "gurgaon", "gurugram", "faridabad",
}

WEIGHTS = {
    "title_career_fit": 0.30,
    "skills_fit":       0.25,
    "experience_fit":   0.15,
    "location_fit":     0.10,
    "behavioral":       0.12,
    "education_bonus":  0.05,
    "profile_quality":  0.03,
}


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def days_since(date_str):
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (date.today() - d).days
    except:
        return 9999


def normalize_title(title):
    return title.lower().strip()


def extract_ai_experience_months(career_history):
    ai_keywords = {
        "ml", "machine learning", "ai", "artificial intelligence",
        "nlp", "retrieval", "ranking", "search", "recommendation",
        "embedding", "deep learning", "data science", "llm", "applied",
    }
    total = 0
    for role in career_history:
        combined = (role.get("title", "") + " " + role.get("description", "")).lower()
        if any(kw in combined for kw in ai_keywords):
            total += role.get("duration_months", 0)
    return total


def is_consulting_only(career_history):
    if not career_history:
        return False
    return all(
        any(firm in role.get("company", "").lower() for firm in CONSULTING_FIRMS)
        for role in career_history
    )


def has_product_company_experience(career_history):
    for role in career_history:
        company = role.get("company", "").lower()
        industry = role.get("industry", "").lower()
        if not any(firm in company for firm in CONSULTING_FIRMS):
            if industry not in ("it services", "outsourcing", "bpo"):
                return True
    return False


def honeypot_check(candidate):
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    expert_zero = sum(1 for s in skills
                      if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0)
    if expert_zero >= 5:
        return True
    stated_months = candidate.get("profile", {}).get("years_of_experience", 0) * 12
    total_months = sum(r.get("duration_months", 0) for r in career)
    if total_months > 0 and stated_months > total_months * 2.5:
        return True
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) >= 10:
        if sum(s.get("duration_months", 0) for s in expert_skills) < 12:
            return True
    return False


def score_title_career(candidate):
    title = normalize_title(candidate["profile"].get("current_title", ""))
    career = candidate.get("career_history", [])
    if any(t in title for t in STRONG_TITLE_SIGNALS):
        title_score = 1.0
    elif any(t in title for t in MODERATE_TITLE_SIGNALS):
        title_score = 0.55
    elif any(t in title for t in DISQUALIFYING_TITLES):
        title_score = 0.05
    else:
        title_score = 0.35
    ai_months = extract_ai_experience_months(career)
    career_bonus = 0.25 if ai_months >= 48 else 0.15 if ai_months >= 24 else 0.08 if ai_months >= 12 else 0.0
    consulting_penalty = -0.15 if is_consulting_only(career) else 0.05 if has_product_company_experience(career) else 0.0
    return clamp(title_score + career_bonus + consulting_penalty)


def score_skills(candidate):
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0
    must_hits = 0.0
    nice_hits = 0.0
    sig = candidate.get("redrob_signals", {})
    for s in skills:
        name = s.get("name", "").lower()
        prof = s.get("proficiency", "beginner")
        months = s.get("duration_months", 0)
        endorsements = s.get("endorsements", 0)
        prof_w = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}.get(prof, 0.3)
        dur_trust = 0.1 if months == 0 else 0.4 if months < 6 else 0.7 if months < 18 else 1.0
        endorse_boost = min(1.0, math.log1p(endorsements) / math.log1p(50))
        trust = prof_w * dur_trust * (0.7 + 0.3 * endorse_boost)
        if any(kw in name for kw in MUST_HAVE_SKILLS):
            must_hits += trust
        elif any(kw in name for kw in NICE_TO_HAVE_SKILLS):
            nice_hits += trust * 0.5
    assessment_scores = sig.get("skill_assessment_scores", {})
    assessment_boost = clamp(sum(assessment_scores.values()) / len(assessment_scores) / 100.0) * 0.15 if assessment_scores else 0.0
    return clamp(clamp(must_hits / 4.0) * 0.75 + clamp(nice_hits / 3.0) * 0.2 + assessment_boost)


def score_experience(candidate):
    yoe = candidate["profile"].get("years_of_experience", 0)
    career = candidate.get("career_history", [])
    sig = candidate.get("redrob_signals", {})
    if 5 <= yoe <= 9:
        yoe_score = 1.0
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        yoe_score = 0.85
    elif 3 <= yoe < 4 or 11 < yoe <= 14:
        yoe_score = 0.65
    elif yoe < 3:
        yoe_score = 0.30
    else:
        yoe_score = 0.55
    ai_years = extract_ai_experience_months(career) / 12.0
    ai_bonus = 0.25 if ai_years >= 4 else 0.15 if ai_years >= 2 else 0.05 if ai_years >= 1 else 0.0
    notice = sig.get("notice_period_days", 90)
    notice_score = 1.0 if notice <= 30 else 0.8 if notice <= 60 else 0.6 if notice <= 90 else 0.4
    return clamp(yoe_score * 0.65 + ai_bonus + notice_score * 0.10)


def score_location(candidate):
    location = candidate["profile"].get("location", "").lower()
    country = candidate["profile"].get("country", "").lower()
    sig = candidate.get("redrob_signals", {})
    will_relocate = sig.get("willing_to_relocate", False)
    location_match = any(city in location for city in TARGET_LOCATIONS)
    if country == "india":
        return 1.0 if location_match else 0.75 if will_relocate else 0.5
    else:
        return 0.45 if will_relocate else 0.15


def score_behavioral(candidate):
    sig = candidate.get("redrob_signals", {})
    days_inactive = days_since(sig.get("last_active_date", "2020-01-01"))
    recency = 1.0 if days_inactive <= 30 else 0.85 if days_inactive <= 60 else 0.70 if days_inactive <= 90 else 0.45 if days_inactive <= 180 else 0.15
    otw = 1.0 if sig.get("open_to_work_flag", False) else 0.6
    rrr = clamp(sig.get("recruiter_response_rate", 0.0))
    art = sig.get("avg_response_time_hours", 72)
    art_score = 1.0 if art <= 4 else 0.85 if art <= 12 else 0.70 if art <= 24 else 0.50 if art <= 72 else 0.25
    saved = sig.get("saved_by_recruiters_30d", 0)
    views = sig.get("profile_views_received_30d", 0)
    apps = sig.get("applications_submitted_30d", 0)
    engagement = clamp((math.log1p(saved) * 3 + math.log1p(views) + math.log1p(apps)) / 15.0)
    icr = sig.get("interview_completion_rate", 0.5)
    oar = sig.get("offer_acceptance_rate", -1)
    reliability = icr * 0.7 + (clamp(oar) * 0.3 if oar >= 0 else 0.3)
    gh = sig.get("github_activity_score", -1)
    gh_score = (gh / 100.0) if gh >= 0 else 0.3
    completeness = clamp(sig.get("profile_completeness_score", 50) / 100.0)
    return clamp(
        recency * 0.25 + otw * 0.10 + rrr * 0.25 + art_score * 0.10 +
        engagement * 0.05 + reliability * 0.10 + gh_score * 0.10 + completeness * 0.05
    )


def score_education(candidate):
    edu_list = candidate.get("education", [])
    if not edu_list:
        return 0.4
    best = 0.0
    for edu in edu_list:
        tier = edu.get("tier", "unknown")
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()
        cs_fields = {"computer science", "ai", "machine learning", "data science",
                     "electrical engineering", "electronics", "information technology",
                     "mathematics", "statistics"}
        is_cs = any(f in field for f in cs_fields)
        deg_score = 0.95 if "phd" in degree else 0.85 if any(x in degree for x in ["m.tech", "mtech", "m.s", "ms"]) else 0.75 if any(x in degree for x in ["b.tech", "btech", "b.e", "b.s"]) else 0.6
        tier_map = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.65, "tier_4": 0.5, "unknown": 0.6}
        best = max(best, clamp(deg_score * 0.6 + tier_map.get(tier, 0.6) * 0.4 + (0.15 if is_cs else 0.0)))
    return best


def score_profile_quality(candidate):
    sig = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})
    verified = (int(sig.get("verified_email", False)) +
                int(sig.get("verified_phone", False)) +
                int(sig.get("linkedin_connected", False))) / 3.0
    summary_len = len(profile.get("summary", ""))
    career = candidate.get("career_history", [])
    avg_desc = sum(len(r.get("description", "")) for r in career) / max(len(career), 1)
    richness = clamp((min(summary_len, 500) / 500.0) * 0.5 + (min(avg_desc, 400) / 400.0) * 0.5)
    connections = sig.get("connection_count", 0)
    endorsements = sig.get("endorsements_received", 0)
    social = clamp(math.log1p(connections) / math.log1p(500) * 0.5 +
                   math.log1p(endorsements) / math.log1p(100) * 0.5)
    return clamp(verified * 0.4 + richness * 0.35 + social * 0.25)


def compute_final_score(candidate):
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
    title = normalize_title(candidate["profile"].get("current_title", ""))
    if any(t in title for t in DISQUALIFYING_TITLES):
        if extract_ai_experience_months(candidate.get("career_history", [])) < 12:
            final *= 0.20
    sig = candidate.get("redrob_signals", {})
    days_inactive = days_since(sig.get("last_active_date", "2020-01-01"))
    if days_inactive > 180:
        final *= 0.70
    if days_inactive > 365:
        final *= 0.50
    if sig.get("recruiter_response_rate", 0.0) < 0.10:
        final *= 0.85
    return clamp(final), components


def generate_reasoning(candidate, components, rank, score):
    if components.get("honeypot"):
        return "Profile flagged as potentially inconsistent — suspicious skill claims relative to career history."
    p = candidate["profile"]
    sig = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    title = p.get("current_title", "Unknown")
    yoe = p.get("years_of_experience", 0)
    location = p.get("location", "Unknown")
    country = p.get("country", "")
    rrr = sig.get("recruiter_response_rate", 0)
    notice = sig.get("notice_period_days", 90)
    will_relocate = sig.get("willing_to_relocate", False)
    open_to_work = sig.get("open_to_work_flag", False)
    days_inactive = days_since(sig.get("last_active_date", "2020-01-01"))
    relevant_skills = [s["name"] for s in skills
                       if any(kw in s["name"].lower() for kw in MUST_HAVE_SKILLS)
                       and s.get("proficiency") in ("intermediate", "advanced", "expert")][:3]
    ai_years = round(extract_ai_experience_months(candidate.get("career_history", [])) / 12, 1)
    parts = []
    if components.get("title_career_fit", 0) >= 0.7:
        skill_str = ", ".join(relevant_skills) if relevant_skills else "AI/ML skills"
        parts.append(f"{title} with {yoe:.1f}y experience; {ai_years}y in AI/ML roles; key skills: {skill_str}")
    elif components.get("title_career_fit", 0) >= 0.4:
        parts.append(f"{title} ({yoe:.1f}y) with adjacent AI skills" +
                     (f" ({', '.join(relevant_skills[:2])})" if relevant_skills else ""))
    else:
        parts.append(f"{title} ({yoe:.1f}y) — title not aligned with Senior AI Engineer role")
    positives, concerns = [], []
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
    elif any(city in location.lower() for city in TARGET_LOCATIONS):
        positives.append(f"in {location}")
    pos_str = "; ".join(positives[:2])
    con_str = "; ".join(concerns[:2])
    if pos_str and con_str:
        parts.append(f"Signals: {pos_str}. Concerns: {con_str}.")
    elif pos_str:
        parts.append(f"Signals: {pos_str}.")
    elif con_str:
        parts.append(f"Concerns: {con_str}.")
    return " ".join(parts)[:300]


def rank_candidates(candidates):
    scored = []
    for c in candidates:
        try:
            score, components = compute_final_score(c)
            scored.append((score, c["candidate_id"], components, c))
        except Exception as e:
            continue
    scored.sort(key=lambda x: (-x[0], x[1]))
    results = []
    prev = None
    for i, (score, cid, components, candidate) in enumerate(scored):
        s = round(score, 6)
        if prev is not None and s > prev:
            s = prev
        prev = s
        results.append((i + 1, cid, s, components, candidate))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("🎯 Redrob India.Runs — Candidate Ranker")
st.markdown("**Intelligent ranking system for the Senior AI Engineer role**")
st.markdown("Upload a JSON file containing candidate profiles → get ranked results instantly.")

st.divider()

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Team:** [Your Team Name]  
    **Hackathon:** Redrob × H2S India.Runs  
    **GitHub:** [redrob-ranker](https://github.com/Arunkarthik134630/redrob-ranker)
    
    ---
    
    **Scoring weights:**
    - 🏆 Title & Career: 30%
    - 🔧 Skills Fit: 25%
    - 📅 Experience: 15%
    - 📊 Behavioral: 12%
    - 📍 Location: 10%
    - 🎓 Education: 5%
    - ✅ Profile Quality: 3%
    
    ---
    
    **Constraints met:**
    - ✅ CPU only
    - ✅ No network calls
    - ✅ Zero external dependencies
    - ✅ ~31s for 100K candidates
    """)

# File upload
st.subheader("📁 Upload Candidates")
st.markdown("Upload `sample_candidates.json` or any JSON array of candidate profiles (max 100 for demo).")

uploaded_file = st.file_uploader(
    "Choose a JSON file",
    type=["json"],
    help="Upload a JSON array of candidate profiles matching the Redrob schema"
)

col1, col2 = st.columns([1, 1])
with col1:
    top_n = st.slider("Number of top candidates to show", min_value=5, max_value=100, value=10, step=5)

if uploaded_file is not None:
    try:
        raw = json.load(uploaded_file)

        # Handle both array and single object
        if isinstance(raw, dict):
            candidates = [raw]
        elif isinstance(raw, list):
            candidates = raw
        else:
            st.error("Invalid format — expected a JSON array of candidates.")
            st.stop()

        st.success(f"✅ Loaded **{len(candidates)}** candidates")

        # Run ranking
        with st.spinner(f"🔄 Ranking {len(candidates)} candidates..."):
            results = rank_candidates(candidates)

        st.success(f"✅ Ranked **{len(results)}** candidates in seconds!")
        st.divider()

        # Show top N results
        display_n = min(top_n, len(results))
        st.subheader(f"🏆 Top {display_n} Ranked Candidates")

        # Build display table
        table_data = []
        for rank, cid, score, components, candidate in results[:display_n]:
            p = candidate["profile"]
            sig = candidate.get("redrob_signals", {})
            reasoning = generate_reasoning(candidate, components, rank, score)
            table_data.append({
                "Rank": rank,
                "Candidate ID": cid,
                "Score": f"{score:.4f}",
                "Title": p.get("current_title", ""),
                "YOE": p.get("years_of_experience", 0),
                "Location": p.get("location", ""),
                "Last Active": sig.get("last_active_date", ""),
                "Notice (days)": sig.get("notice_period_days", ""),
                "RRR": f"{sig.get('recruiter_response_rate', 0):.0%}",
                "Reasoning": reasoning,
            })

        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.NumberColumn(format="%.4f"),
                "Rank": st.column_config.NumberColumn(width="small"),
                "Reasoning": st.column_config.TextColumn(width="large"),
            }
        )

        # Score breakdown for top candidate
        st.divider()
        st.subheader("🔍 Score Breakdown — Rank #1")
        top_rank, top_cid, top_score, top_components, top_candidate = results[0]
        p = top_candidate["profile"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Candidate", top_cid)
        col2.metric("Title", p.get("current_title", ""))
        col3.metric("Final Score", f"{top_score:.4f}")

        if not top_components.get("honeypot"):
            cols = st.columns(7)
            labels = {
                "title_career_fit": "Title & Career",
                "skills_fit": "Skills",
                "experience_fit": "Experience",
                "location_fit": "Location",
                "behavioral": "Behavioral",
                "education_bonus": "Education",
                "profile_quality": "Profile Quality",
            }
            for i, (key, label) in enumerate(labels.items()):
                score_val = top_components.get(key, 0)
                cols[i].metric(label, f"{score_val:.2f}", f"×{WEIGHTS[key]:.0%}")

        # Generate downloadable CSV
        st.divider()
        st.subheader("⬇️ Download Submission CSV")

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, cid, score, components, candidate in results:
            reasoning = generate_reasoning(candidate, components, rank, score)
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])

        st.download_button(
            label="📥 Download submission.csv",
            data=csv_buffer.getvalue(),
            file_name="submission.csv",
            mime="text/csv",
            help="Download the ranked CSV in the exact competition format"
        )

        st.caption("⚠️ For the full 100K candidate submission, run `python run.py` locally.")

    except json.JSONDecodeError:
        st.error("❌ Invalid JSON file. Please upload a valid JSON array of candidates.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

else:
    # Show example / instructions when no file uploaded
    st.info("👆 Upload a JSON file to get started. You can use `sample_candidates.json` from the challenge bundle.")

    st.subheader("📋 Expected Input Format")
    st.code('''[
  {
    "candidate_id": "CAND_0000001",
    "profile": {
      "current_title": "ML Engineer",
      "years_of_experience": 6.5,
      "location": "Hyderabad, Telangana",
      "country": "India",
      ...
    },
    "career_history": [...],
    "skills": [...],
    "education": [...],
    "redrob_signals": {
      "last_active_date": "2026-05-20",
      "recruiter_response_rate": 0.85,
      "notice_period_days": 30,
      ...
    }
  }
]''', language="json")

    st.subheader("🚀 How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Upload**")
        st.markdown("Upload your candidates JSON file (up to 100 profiles for the demo)")
    with col2:
        st.markdown("**2. Rank**")
        st.markdown("Our 7-dimension scorer evaluates each candidate instantly on CPU")
    with col3:
        st.markdown("**3. Download**")
        st.markdown("Get a validated submission CSV ready for the competition portal")

st.divider()
st.caption("Redrob India.Runs Hackathon | Intelligent Candidate Discovery & Ranking | github.com/Arunkarthik134630/redrob-ranker")