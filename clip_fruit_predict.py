from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import clip
import torch
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent

LABELS = [
    "almond", "apple", "apricot", "avocado", "banana", "bean pod", "beetroot",
    "blackberry", "blueberry", "cabbage", "cactus fruit", "cantaloupe",
    "carambola", "carrot", "cashew seed", "cauliflower", "cherry", "cherimoya",
    "chestnut", "clementine", "coconut", "corn", "cucumber", "date",
    "dragon fruit", "eggplant", "fig", "ginger root", "granadilla", "grape",
    "grapefruit", "gooseberry", "guava", "hazelnut", "huckleberry", "kaki",
    "kiwi", "kohlrabi", "kumquat", "lemon", "lime", "lychee", "mandarine",
    "mango", "mangosteen", "melon", "mulberry", "nectarine", "nut", "onion",
    "orange", "papaya", "passion fruit", "peach", "peanut", "pear", "pepino",
    "pepper", "physalis", "pineapple", "pistachio", "plum", "pomegranate",
    "pomelo", "potato", "quince", "rambutan", "raspberry", "red cabbage",
    "redcurrant", "salak", "strawberry", "tamarillo", "tangelo", "tomato",
    "walnut", "watermelon", "zucchini",
]

NON_FRUIT_LABELS = {
    "almond", "bean pod", "beetroot", "cabbage", "carrot", "cashew seed",
    "cauliflower", "corn", "cucumber", "ginger root", "hazelnut", "kohlrabi",
    "nut", "onion", "peanut", "pepper", "pistachio", "potato", "red cabbage",
    "tomato", "walnut", "zucchini",
}

COMMON_FRUIT_LABELS = {
    "apple", "apricot", "avocado", "banana", "blackberry", "blueberry",
    "cantaloupe", "carambola", "cherry", "cherimoya", "clementine", "coconut",
    "date", "dragon fruit", "fig", "grape", "grapefruit", "gooseberry",
    "guava", "kiwi", "lemon", "lime", "lychee", "mandarine", "mango",
    "melon", "nectarine", "orange", "papaya", "passion fruit", "peach",
    "pear", "pineapple", "plum", "pomegranate", "raspberry", "salak",
    "strawberry", "watermelon",
}

RARE_LABEL_MIN_CONFIDENCE = 0.55
COMMON_BACKOFF_MIN_CONFIDENCE = 0.05
COMMON_BACKOFF_RATIO = 0.40


def load_clip_model(model_names: list[str]):
    errors = []
    for model_name in model_names:
        try:
            model, preprocess = clip.load(
                model_name,
                device="cpu",
                download_root=str(BASE_DIR / "models" / "clip"),
            )
            return model_name, model, preprocess
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")
    raise RuntimeError("No CLIP model could be loaded. " + " | ".join(errors))


def prompts_for(label: str) -> list[str]:
    base = [
        f"a photo of {label}" if label in NON_FRUIT_LABELS else f"a photo of {label} fruit",
        f"a close up photo of {label}",
    ]
    if label == "pomegranate":
        base.extend([
            "a close up photo of pomegranate seeds",
            "a photo of a red pomegranate fruit",
            "a photo of a rotten pomegranate fruit",
        ])
    if label == "banana":
        base.extend(["a photo of a yellow banana", "a photo of a rotten banana"])
    if label == "apple":
        base.extend(["a photo of a red apple", "a photo of a fresh apple"])
    return list(dict.fromkeys(base))


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP zero-shot fruit classifier")
    parser.add_argument("image")
    parser.add_argument(
        "--models",
        default=os.environ.get("CLIP_MODELS", "ViT-L/14,ViT-B/32"),
        help="Comma-separated CLIP model priority list.",
    )
    args = parser.parse_args()

    model_names = [item.strip() for item in args.models.split(",") if item.strip()]
    model_name, model, preprocess = load_clip_model(model_names)

    prompt_texts = []
    prompt_ranges = []
    for label in LABELS:
        start = len(prompt_texts)
        prompt_texts.extend(prompts_for(label))
        prompt_ranges.append((start, len(prompt_texts)))

    image = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    text_tensor = clip.tokenize(prompt_texts)
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        label_features = []
        for start, end in prompt_ranges:
            feature = text_features[start:end].mean(dim=0)
            feature = feature / feature.norm()
            label_features.append(feature)
        label_features = torch.stack(label_features)
        probs = (100.0 * image_features @ label_features.T).softmax(dim=-1)[0]

    top_indices = probs.argsort(descending=True)[:5]
    best_index = int(top_indices[0])
    best_label = LABELS[best_index]
    best_confidence = float(probs[best_index])

    if best_label not in COMMON_FRUIT_LABELS and best_confidence < RARE_LABEL_MIN_CONFIDENCE:
        common_indices = [
            int(index)
            for index in probs.argsort(descending=True).tolist()
            if LABELS[int(index)] in COMMON_FRUIT_LABELS
        ]
        if common_indices:
            common_index = common_indices[0]
            common_confidence = float(probs[common_index])
            if (
                common_confidence >= COMMON_BACKOFF_MIN_CONFIDENCE
                and common_confidence >= best_confidence * COMMON_BACKOFF_RATIO
            ):
                best_index = common_index

    candidate_indices = []
    for index in [best_index, *[int(index) for index in top_indices.tolist()]]:
        if index not in candidate_indices:
            candidate_indices.append(index)
    candidate_indices = sorted(candidate_indices, key=lambda index: float(probs[index]), reverse=True)[:5]

    print(json.dumps({
        "fruit_name": LABELS[best_index].replace(" ", "_"),
        "fruit_conf": round(float(probs[best_index]), 4),
        "fruit_source": "clip_zero_shot",
        "fruit_model": model_name,
        "fruit_candidates": [
            {
                "fruit_name": LABELS[index].replace(" ", "_"),
                "confidence": round(float(probs[index]), 4),
                "accepted": index == best_index,
            }
            for index in candidate_indices
        ],
    }))


if __name__ == "__main__":
    main()
