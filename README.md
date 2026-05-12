# Resume Matching Engine
### Redrob AI Campus Hackathon — McKinley Rice | Amity University Noida

---

## Problem Statement

Built a Resume Matching Engine that matches 10 candidate resumes against 3 Job Descriptions from Korean technology companies (Kakao, Naver, Line) using TF-IDF vectorization and Cosine Similarity.

---

## Approach

### Step 1 — Skill Normalization
- Split raw skills on commas
- Convert to lowercase
- Match multi-word phrases first (e.g. "spring boot", "feature engineering")
- Apply exact alias mapping using `SKILL_ALIASES`
- Discard any token not in the alias map

### Step 2 — Deduplication
- Each canonical skill appears only once per resume

### Step 3 — Vocabulary Construction
- Built shared vocabulary from all normalized resume skills
- Sorted alphabetically for consistent vector ordering
- Total vocabulary size: 48 skills

### Step 4 — TF-IDF Computation
Used exact formulas as specified:

```
TF(skill, resume) = 1 / N
where N = total unique skills in resume

IDF(skill) = ln(10 / df(skill))
where df = number of resumes containing the skill

TF-IDF = TF × IDF
```

### Step 5 — JD Binary Vectors
- Combined required + preferred skills for each JD
- Normalized through the same SKILL_ALIASES map
- Binary vector: 1 if skill in vocabulary AND in JD, else 0

### Step 6 — Cosine Similarity + Ranking
```
Cosine(A, B) = dot_product(A, B) / (|A| × |B|)
```
- Ranked Top 3 candidates per JD
- Tie-breaking: alphabetical by candidate name

---

## Results

```
JD-1 — Kakao (ML Engineer)
Sneha Patel(0.57), Karan Mehta(0.53), Arjun Sharma(0.40)

JD-2 — Naver (Backend Engineer)
Rahul Gupta(0.81), Ananya Krishnan(0.28), Deepika Rao(0.19)

JD-3 — Line (Frontend Engineer)
Aditya Kumar(0.67), Priya Nair(0.58), Ananya Krishnan(0.35)
```

---

## Tech Stack

- **Language:** Python 3
- **Libraries:** `math` (standard library only)
- **No external libraries** — numpy, pandas, sklearn all prohibited

---

## How to Run

```bash
python3 resume_matching_engine.py
```

---

## File Structure

```
ResumeMatchingEngine/
│
├── resume_matching_engine.py    # Main solution file
└── README.md                    # This file
```

---

## Key Implementation Details

- Multi-word phrase matching is handled by sorting aliases by length (descending) before matching — this ensures "spring boot" matches before "spring" or "boot" individually
- Exact token matching only — prevents false matches like "r" matching inside "rest_api"
- JD skills normalized through SKILL_ALIASES before building binary vectors
- Pure Python math implementation — no numpy or scipy

---

*Submitted for Redrob AI Campus Hackathon | Powered by McKinley Rice*
