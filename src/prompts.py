POSITIVE_TEMPLATES = [
    "{}",
    "Findings of {} in chest X-ray",
    "The chest radiograph demonstrates {}",
    "A patient with {}",
    "There is evidence of {}",
]

NEGATIVE_TEMPLATES = [
    "No evidence of {}",
    "No findings of {}",
]


def build_prompts(disease: str) -> dict[str, list[str]]:
    return {
        "simple":      ["{}".format(disease)],
        "findings":    ["Findings of {} in chest X-ray".format(disease)],
        "clinical":    ["The chest radiograph demonstrates {}".format(disease)],
        "patient":     ["A patient with {}".format(disease)],
        "radiologist": ["There is evidence of {}".format(disease)],
        "negative":    ["No evidence of {}".format(disease)],
        "ensemble":    [t.format(disease) for t in POSITIVE_TEMPLATES],
        "pos_neg":     [t.format(disease) for t in POSITIVE_TEMPLATES + NEGATIVE_TEMPLATES],
    }
