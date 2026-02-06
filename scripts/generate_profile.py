import random
import json

N = 500
MAX_TRIES = 10000

chronic_conditions = [
    # Cardiovascular 
    "hypertension",
    "type 2 diabetes",
    "hyperlipidemia",
    # Endocrine
    "chronic kidney disease",
    "hypothyroidism",
    "asthma",
    # Mental health
    "depression",
    "anxiety disorder",
    # Infectious
    "chronic hepatitis B",
    # Immune / Inflammatory
    "rheumatoid arthritis",
    # Hematologic
    "iron deficiency anemia",
    # Metabolic / Inflammatory
    "gout",
    # Digestive
    "gastroesophageal reflux disease",
    # Musculoskeletal / Pain
    "osteoarthritis",
    "chronic back pain",
    "migraine"
]

allergies = [
    "none",
    "aspirin allergy",
    "NSAID allergy",
    "penicillin allergy",
    "sulfa allergy",
    "cephalosporin allergy",
    "peanut allergy",
    "shellfish allergy",
    "milk allergy",
    "egg allergy",
    "tree nut allergy"
]

import random
import json

N = 500

chronic_conditions = [
    # Cardiovascular 
    "hypertension",
    "type 2 diabetes",
    "hyperlipidemia",
    # Endocrine
    "chronic kidney disease",
    "hypothyroidism",
    "asthma",
    # Mental health
    "depression",
    "anxiety Disorder",
    # Infectious
    "chronic hepatitis B",
    # Musculoskeletal / Pain
    "osteoarthritis",
    "chronic back pain",
    # Immune / Inflammatory
    "rheumatoid arthritis",
    # Hematologic
    "iron deficiency anemia",
    # Metabolic / Inflammatory
    "gout",
    # Digestive
    "gastroesophageal reflux disease",
    # Musculoskeletal / Pain
    "osteoarthritis",
    "chronic back pain",
    "migraine"
]

allergies = [
    "none",
    "aspirin allergy",
    "NSAID allergy",
    "penicillin allergy",
    "sulfa allergy",
    "cephalosporin allergy",
    "peanut allergy",
    "shellfish allergy",
    "milk allergy",
    "egg allergy",
    "tree nut allergy"
]


profiles = []
seen = set()
profile_id = 0
tries = 0

while len(profiles) < N and tries < MAX_TRIES:
    tries += 1

    age = random.randint(18, 85)
    gender = random.choice(["male", "female"])

    if gender == "female" and 18 <= age <= 50 and random.random() < 0.08:
        pregnancy = "pregnant"
    else:
        pregnancy = "not pregnant"

    if random.random() < 0.4:
        chronic = []
    else:
        chronic = random.sample(
            chronic_conditions,
            k=random.randint(1, 2)
        )

    allergy = random.choice(allergies)

    signature = (
        age,
        gender,
        pregnancy,
        tuple(sorted(chronic)),
        allergy
    )

    if signature in seen:
        continue

    seen.add(signature)
    profile_id += 1

    profiles.append({
        "profile_id": profile_id,
        "age": age,
        "gender": gender,
        "pregnancy_status": pregnancy,
        "chronic_conditions": chronic,
        "allergies": [allergy]
    })

assert len(profiles) == N, f"Only generated {len(profiles)} profiles"

with open("mydatasets/profiles.jsonl", "w", encoding="utf-8") as f:
    for p in profiles:
        f.write(json.dumps(p) + "\n")

print(f"Generated {len(profiles)} unique profiles → profiles.jsonl")