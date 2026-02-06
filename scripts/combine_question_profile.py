import json

QUESTIONS_PATH = "mydatasets/sample_questions.json"
PROFILES_PATH  = "mydatasets/profiles.jsonl"
OUT_PATH       = "mydatasets/q_with_profile.jsonl"
LEN = 500

PROFILE_FIELDS_ORDER = [
    "age",
    "gender",
    "pregnancy_status",
    "chronic_conditions",
    "allergies",
]

def load_questions(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for item in data:
        if isinstance(item, str):
            questions.append(item)
        elif isinstance(item, dict) and "question" in item:
            questions.append(item["question"])
        else:
            raise ValueError(f"Unknown question format: {item}")
    return questions

def load_profiles(path: str):
    profiles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            profiles.append(json.loads(line))
    return profiles

def format_profile(p: dict) -> str:
    lines = []
    for k in PROFILE_FIELDS_ORDER:
        if k not in p:
            continue
        v = p[k]
        if isinstance(v, list):
            v = ", ".join(v) if v else "none"
        elif v in [None, ""]:
            v = "unknown"
        lines.append(f"{k}: {v}")
    return "\n".join(lines)

def main():
    questions = load_questions(QUESTIONS_PATH)
    profiles  = load_profiles(PROFILES_PATH)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for q, p in zip(questions[:LEN], profiles[:LEN]):
            rec = {
                "question": q,
                "user_profile": format_profile(p),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
