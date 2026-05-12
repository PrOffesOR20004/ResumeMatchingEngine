import math

# Define the SKILL_ALIASES mapping
# Do not modify this mapping
SKILL_ALIASES = {
    # Languages
    "python": "python",
    "pyhton": "python",
    "java": "java",
    "javascript": "javascript",
    "javascrpit": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "typescrpit": "typescript",
    "c++": "cpp",
    "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",

    # ML / Data
    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",
    "sklearn": "machine_learning",
    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",
    "nlp": "nlp",
    "bert": "bert",
    "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics",
    "stats": "statistics",
    "regression": "regression",
    "clustering": "clustering",
    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",
    "pandas": "pandas",
    "numpy": "numpy",

    # Web - Frontend
    "react": "react",
    "reacts": "react",
    "reactjs": "react",
    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",
    "redux": "redux",
    "tailwind": "tailwind",
    "html/css": "html_css",
    "html css": "html_css",
    "html": "html_css",
    "css": "html_css",
    "jest": "jest",
    "graphql": "graphql",

    # Web - Backend
    "node.js": "nodejs",
    "nodejs": "nodejs",
    "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot",
    "springboot": "spring_boot",
    "rest api": "rest_api",
    "rest": "rest_api",
    "restapi": "rest_api",
    "microservices": "microservices",

    # Databases
    "sql": "sql",
    "mysql": "mysql",
    "mysq": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "mongodb": "mongodb",
    "redis": "redis",

    # DevOps / Cloud
    "docker": "docker",
    "kubernetes": "kubernetes",
    "kubernates": "kubernetes",
    "k8s": "kubernetes",
    "ci/cd": "ci_cd",
    "cicd": "ci_cd",
    "ci cd": "ci_cd",
    "aws": "aws",

    # Mobile
    "android": "android",
    "firebase": "firebase",

    # CS Fundamentals
    "algorithms": "algorithms",
    "algoritms": "algorithms",
    "data structure": "data_structures",
    "data structures": "data_structures",
    "competitive programming": "competitive_programming",

    # Design
    "ui/ux": "ui_ux",
    "ui ux": "ui_ux",
    "figma": "figma",
}

# Resume dataset - 10 candidates
resumes = [
    {"id": "01", "name": "Arjun Sharma",    "skills": "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"},
    {"id": "02", "name": "Priya Nair",      "skills": "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"},
    {"id": "03", "name": "Rahul Gupta",     "skills": "Java, Spring Boot, MySql, Microservices, Docker, kubernates"},
    {"id": "04", "name": "Sneha Patel",     "skills": "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"},
    {"id": "05", "name": "Vikram Singh",    "skills": "C++, Algoritms, Data Structure, competitive programming, python"},
    {"id": "06", "name": "Ananya Krishnan", "skills": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"},
    {"id": "07", "name": "Karan Mehta",     "skills": "Python, Sklearn, XGboost, feature engineering, SQL, tableau"},
    {"id": "08", "name": "Deepika Rao",     "skills": "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"},
    {"id": "09", "name": "Aditya Kumar",    "skills": "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"},
    {"id": "10", "name": "Meera Iyer",      "skills": "python, R, statistics, ML, regression, clustering, Power-BI"},
]

# Job description dataset - 3 JDs
job_descriptions = [
    {
        "id": "JD-1",
        "company": "Kakao",
        "role": "ML Engineer",
        "required_skills": "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization",
        "preferred_skills": "NLP, BERT, Feature Engineering, Statistics"
    },
    {
        "id": "JD-2",
        "company": "Naver",
        "role": "Backend Engineer",
        "required_skills": "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes",
        "preferred_skills": "REST API, CI/CD, Redis"
    },
    {
        "id": "JD-3",
        "company": "Line",
        "role": "Frontend Engineer",
        "required_skills": "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS",
        "preferred_skills": "Node.js, GraphQL, Redux, Jest, AWS"
    },
]


# Step 1: Normalize skills function
# - Split on commas
# - Lowercase each token
# - Match multi-word phrases FIRST, then single tokens
# - Apply SKILL_ALIASES mapping
# - Discard tokens not in the alias map
def normalize_skills(skills_string):
    # Sort aliases by length descending to match multi-word phrases first
    sorted_aliases = sorted(SKILL_ALIASES.items(), key=lambda x: len(x[0]), reverse=True)

    # Split raw skills by comma and strip whitespace
    tokens = [token.strip().lower() for token in skills_string.split(",")]

    normalized = []
    for token in tokens:
        matched = False
        for alias, canonical_skill in sorted_aliases:
            # Exact match only - avoids partial matches like "r" inside "rest_api"
            if token == alias:
                normalized.append(canonical_skill)
                matched = True
                break
        # If no match found, discard the token as per instructions

    # Step 2: Deduplicate - each canonical skill appears only once per resume
    deduplicated = list(dict.fromkeys(normalized))
    return deduplicated


# Step 3: Build shared vocabulary from all normalized resume skills
def build_vocabulary(resumes):
    vocabulary = set()
    for resume in resumes:
        vocabulary.update(resume["normalized_skills"])
    # Sort alphabetically for consistent vector ordering
    return sorted(list(vocabulary))


# Step 4: Compute TF-IDF vectors for resumes
# TF = 1 / N where N = total unique skills in resume
# IDF = ln(10 / df(skill)) where df = number of resumes containing that skill
# TF-IDF = TF * IDF
def compute_tfidf_vectors(resumes, vocabulary):
    tfidf_vectors = {}
    for resume in resumes:
        normalized_skills = resume["normalized_skills"]
        N = len(normalized_skills)
        vector = []
        for skill in vocabulary:
            if skill in normalized_skills:
                tf = 1 / N
                # Count how many resumes contain this skill
                df = sum(1 for r in resumes if skill in r["normalized_skills"])
                idf = math.log(10 / df)
                tfidf = tf * idf
            else:
                tfidf = 0.0
            vector.append(tfidf)
        tfidf_vectors[resume["name"]] = vector
    return tfidf_vectors


# Step 5: Build binary vectors for Job Descriptions
# Combine required + preferred skills, normalize through SKILL_ALIASES
# Binary: 1 if skill is in vocabulary AND in JD, else 0
def build_jd_binary_vectors(job_descriptions, vocabulary):
    jd_vectors = {}
    for jd in job_descriptions:
        # Combine required and preferred skills and normalize them
        combined_skills = jd["required_skills"] + ", " + jd["preferred_skills"]
        normalized_jd_skills = normalize_skills(combined_skills)

        # Build binary vector over the same vocabulary
        vector = []
        for skill in vocabulary:
            if skill in normalized_jd_skills:
                vector.append(1)
            else:
                vector.append(0)
        jd_vectors[jd["id"]] = vector
    return jd_vectors


# Step 6: Compute cosine similarity
# Cosine(A, B) = (A . B) / (|A| * |B|)
# A = resume TF-IDF vector, B = JD binary vector
def compute_cosine_similarity(vector_a, vector_b):
    # Dot product
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    # Euclidean norm of A
    norm_a = math.sqrt(sum(a ** 2 for a in vector_a))
    # Euclidean norm of B
    norm_b = math.sqrt(sum(b ** 2 for b in vector_b))
    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


# ── Main Execution ──

# Step 1 + 2: Normalize and deduplicate skills for each resume
print("=" * 50)
print("STEP 1 & 2: Normalized + Deduplicated Skills")
print("=" * 50)
for resume in resumes:
    resume["normalized_skills"] = normalize_skills(resume["skills"])
    print(f"{resume['name']}: {resume['normalized_skills']}")

# Step 3: Build vocabulary
print("\n" + "=" * 50)
print("STEP 3: Shared Vocabulary (alphabetically sorted)")
print("=" * 50)
vocabulary = build_vocabulary(resumes)
print(vocabulary)
print(f"Total vocabulary size: {len(vocabulary)}")

# Step 4: Compute TF-IDF vectors
print("\n" + "=" * 50)
print("STEP 4: TF-IDF Vectors")
print("=" * 50)
tfidf_vectors = compute_tfidf_vectors(resumes, vocabulary)
for name, vector in tfidf_vectors.items():
    non_zero = {vocabulary[i]: round(v, 4) for i, v in enumerate(vector) if v > 0}
    print(f"{name}: {non_zero}")

# Step 5: Build JD binary vectors
print("\n" + "=" * 50)
print("STEP 5: JD Binary Vectors")
print("=" * 50)
jd_vectors = build_jd_binary_vectors(job_descriptions, vocabulary)
for jd in job_descriptions:
    jd_id = jd["id"]
    non_zero = [vocabulary[i] for i, v in enumerate(jd_vectors[jd_id]) if v == 1]
    print(f"{jd_id} ({jd['company']} - {jd['role']}): {non_zero}")

# Step 6: Compute cosine similarity and rank top 3 candidates per JD
print("\n" + "=" * 50)
print("STEP 6: Cosine Similarity Scores")
print("=" * 50)
for jd in job_descriptions:
    jd_id = jd["id"]
    jd_vector = jd_vectors[jd_id]
    scores = []
    for resume in resumes:
        name = resume["name"]
        resume_vector = tfidf_vectors[name]
        similarity = compute_cosine_similarity(resume_vector, jd_vector)
        scores.append((name, similarity))
    # Sort by score descending, break ties alphabetically by name
    scores.sort(key=lambda x: (-x[1], x[0]))
    print(f"\n{jd_id} ({jd['company']}):")
    for name, score in scores:
        print(f"  {name}: {score:.4f}")

# Final Output in required format
print("\n" + "=" * 50)
print("FINAL OUTPUT")
print("=" * 50)
for jd in job_descriptions:
    jd_id = jd["id"]
    jd_vector = jd_vectors[jd_id]
    scores = []
    for resume in resumes:
        name = resume["name"]
        resume_vector = tfidf_vectors[name]
        similarity = compute_cosine_similarity(resume_vector, jd_vector)
        scores.append((name, similarity))
    scores.sort(key=lambda x: (-x[1], x[0]))
    top3 = scores[:3]
    print(f"\n{jd_id} — {jd['company']} ({jd['role']})")
    print(", ".join(f"{name}({score:.2f})" for name, score in top3))
