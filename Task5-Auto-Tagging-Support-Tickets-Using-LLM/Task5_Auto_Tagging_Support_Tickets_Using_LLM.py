"""
Task 5: Auto Tagging Support Tickets Using LLM
===============================================
Objective : Automatically tag support tickets using prompt engineering with Mistral-7B.
Techniques: Zero-shot vs Few-shot comparison, top-3 tag prediction per ticket.

Requirements:
  pip install huggingface-hub

Run:
  python auto_tag.py
"""

import json
import time
import re
from huggingface_hub import InferenceClient

# ─────────────────────────────────────────────
# CONFIG — paste your free HuggingFace token here
# Get one at: https://huggingface.co/settings/tokens
# Enable "Make calls to Inference Providers" permission
# ─────────────────────────────────────────────
HF_TOKEN = "hf_abCD"   # <-- replace this

MODEL    = "mistralai/Mistral-7B-Instruct-v0.2"
DELAY    = 2   # seconds between API calls (avoid rate limits)

# ─────────────────────────────────────────────
# AVAILABLE TAGS
# ─────────────────────────────────────────────
TAGS = [
    "billing",
    "technical_issue",
    "account_access",
    "shipping_delivery",
    "refund_return",
    "product_defect",
    "feature_request",
    "password_reset",
    "subscription",
    "general_inquiry",
]

# ─────────────────────────────────────────────
# BUILT-IN SUPPORT TICKET DATASET
# 20 realistic tickets with ground-truth labels
# ─────────────────────────────────────────────
TICKETS = [
    {
        "id": 1,
        "text": "I was charged twice for my monthly subscription last week. "
                "Please refund the duplicate charge immediately.",
        "ground_truth": ["billing", "subscription", "refund_return"],
    },
    {
        "id": 2,
        "text": "My app keeps crashing every time I try to open the dashboard. "
                "I've tried reinstalling but the issue persists.",
        "ground_truth": ["technical_issue", "general_inquiry"],
    },
    {
        "id": 3,
        "text": "I forgot my password and the reset email is not arriving in my inbox. "
                "I've checked my spam folder too.",
        "ground_truth": ["password_reset", "account_access"],
    },
    {
        "id": 4,
        "text": "My order was supposed to arrive 3 days ago but tracking still shows "
                "'In Transit'. Can you tell me where my package is?",
        "ground_truth": ["shipping_delivery", "general_inquiry"],
    },
    {
        "id": 5,
        "text": "The laptop I received has a cracked screen right out of the box. "
                "I need a replacement sent urgently.",
        "ground_truth": ["product_defect", "refund_return", "shipping_delivery"],
    },
    {
        "id": 6,
        "text": "I want to cancel my premium subscription and get a refund for "
                "the remaining months I already paid for.",
        "ground_truth": ["subscription", "refund_return", "billing"],
    },
    {
        "id": 7,
        "text": "It would be really useful if your mobile app had a dark mode option. "
                "Many users have been requesting this feature.",
        "ground_truth": ["feature_request"],
    },
    {
        "id": 8,
        "text": "I cannot log in to my account. It says my account has been locked "
                "after too many failed attempts.",
        "ground_truth": ["account_access", "password_reset"],
    },
    {
        "id": 9,
        "text": "My invoice shows a charge for a service I never signed up for. "
                "This looks like an error on your end.",
        "ground_truth": ["billing", "general_inquiry"],
    },
    {
        "id": 10,
        "text": "The website is throwing a 500 internal server error whenever I try "
                "to check out. This has been happening for 2 hours.",
        "ground_truth": ["technical_issue"],
    },
    {
        "id": 11,
        "text": "I returned the product two weeks ago but still haven't received my refund. "
                "The return tracking shows it was delivered.",
        "ground_truth": ["refund_return", "shipping_delivery"],
    },
    {
        "id": 12,
        "text": "Please add multi-language support to your platform. "
                "A lot of our team members are non-English speakers.",
        "ground_truth": ["feature_request", "general_inquiry"],
    },
    {
        "id": 13,
        "text": "I upgraded to the Pro plan but the extra features are not showing up in my account. "
                "I was charged for the upgrade.",
        "ground_truth": ["subscription", "billing", "account_access"],
    },
    {
        "id": 14,
        "text": "The headphones I bought stopped working after just one week. "
                "The left ear produces no sound at all.",
        "ground_truth": ["product_defect", "refund_return"],
    },
    {
        "id": 15,
        "text": "My package was marked as delivered but I never received it. "
                "My neighbor also did not see any delivery.",
        "ground_truth": ["shipping_delivery", "refund_return"],
    },
    {
        "id": 16,
        "text": "Can you please explain how the annual billing cycle works? "
                "I'm not sure if I'll be charged monthly or yearly.",
        "ground_truth": ["billing", "subscription", "general_inquiry"],
    },
    {
        "id": 17,
        "text": "Two-factor authentication is not working. The code I receive via SMS "
                "is always rejected even though it's correct.",
        "ground_truth": ["account_access", "technical_issue"],
    },
    {
        "id": 18,
        "text": "I'd like to request an API endpoint that allows bulk export of all "
                "user data in CSV format.",
        "ground_truth": ["feature_request", "technical_issue"],
    },
    {
        "id": 19,
        "text": "My discount code isn't being applied at checkout. "
                "It says the code is invalid but I just received it in my email.",
        "ground_truth": ["billing", "technical_issue"],
    },
    {
        "id": 20,
        "text": "I need to transfer my account to a different email address. "
                "How do I do this without losing my subscription history?",
        "ground_truth": ["account_access", "subscription", "general_inquiry"],
    },
]

# ─────────────────────────────────────────────
# FEW-SHOT EXAMPLES (used in few-shot prompt)
# ─────────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    {
        "ticket": "I was double charged on my credit card this month.",
        "tags": ["billing", "refund_return"],
    },
    {
        "ticket": "The app crashes immediately after login on my iPhone.",
        "tags": ["technical_issue"],
    },
    {
        "ticket": "I want to return the jacket I ordered — it doesn't fit.",
        "tags": ["refund_return", "shipping_delivery"],
    },
    {
        "ticket": "Please add export to PDF functionality in the reports section.",
        "tags": ["feature_request"],
    },
    {
        "ticket": "My account is locked and I can't reset my password via email.",
        "tags": ["account_access", "password_reset"],
    },
]


# ─────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────
def build_zero_shot_prompt(ticket_text: str) -> str:
    tags_list = "\n".join(f"  - {t}" for t in TAGS)
    return f"""You are a support ticket classification system.

Your task is to assign the top 3 most relevant tags to a support ticket.

Available tags:
{tags_list}

Instructions:
- Return ONLY a JSON object with a single key "tags" containing a list of exactly 3 tags.
- Choose from the available tags list only.
- Order tags by relevance (most relevant first).
- Do not include any explanation or extra text.

Support ticket:
\"{ticket_text}\"

Response (JSON only):"""


def build_few_shot_prompt(ticket_text: str) -> str:
    tags_list = "\n".join(f"  - {t}" for t in TAGS)

    examples_text = ""
    for ex in FEW_SHOT_EXAMPLES:
        examples_text += f'\nTicket: "{ex["ticket"]}"\n'
        examples_text += f'Response: {json.dumps({"tags": ex["tags"]})}\n'

    return f"""You are a support ticket classification system.

Your task is to assign the top 3 most relevant tags to a support ticket.

Available tags:
{tags_list}

Instructions:
- Return ONLY a JSON object with a single key "tags" containing a list of exactly 3 tags.
- Choose from the available tags list only.
- Order tags by relevance (most relevant first).
- Do not include any explanation or extra text.

Here are some examples:
{examples_text}
Now classify this ticket:
Ticket: "{ticket_text}"
Response (JSON only):"""


# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────
def call_llm(prompt: str, client: InferenceClient) -> str:
    """Call Mistral-7B via HuggingFace InferenceClient chat_completion."""
    response = client.chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def parse_tags(raw_response: str) -> list:
    """
    Parse the LLM response and extract tags list.
    Handles: clean JSON, JSON with extra text, or partial responses.
    """
    # Try direct JSON parse
    try:
        parsed = json.loads(raw_response)
        if "tags" in parsed and isinstance(parsed["tags"], list):
            tags = [t.lower().strip() for t in parsed["tags"] if isinstance(t, str)]
            valid = [t for t in tags if t in TAGS]
            return valid[:3]
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from response text
    match = re.search(r'\{[^{}]*"tags"\s*:\s*\[[^\]]*\][^{}]*\}', raw_response, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if "tags" in parsed:
                tags = [t.lower().strip() for t in parsed["tags"] if isinstance(t, str)]
                valid = [t for t in tags if t in TAGS]
                return valid[:3]
        except json.JSONDecodeError:
            pass

    # Fallback: scan response for any known tag names
    found = []
    for tag in TAGS:
        if tag in raw_response.lower() and tag not in found:
            found.append(tag)
        if len(found) == 3:
            break

    return found if found else ["general_inquiry"]


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate(results: list) -> dict:
    """
    Compute top-1 and top-3 accuracy:
    - Top-1: predicted[0] is in ground truth
    - Top-3: any predicted tag is in ground truth
    """
    top1_hits = 0
    top3_hits = 0
    total = len(results)

    for r in results:
        gt = set(r["ground_truth"])
        predicted = r["predicted_tags"]
        if predicted and predicted[0] in gt:
            top1_hits += 1
        if any(p in gt for p in predicted):
            top3_hits += 1

    return {
        "total": total,
        "top1_accuracy": round(top1_hits / total * 100, 1),
        "top3_accuracy": round(top3_hits / total * 100, 1),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    if HF_TOKEN == "hf_your_token_here":
        print("ERROR: Please set your HuggingFace token in the HF_TOKEN variable at the top of this file.")
        print("Get a free token at: https://huggingface.co/settings/tokens")
        print("Enable 'Make calls to Inference Providers' permission.")
        return

    client = InferenceClient(token=HF_TOKEN)

    print("=" * 65)
    print("  Task 5: Auto Tagging Support Tickets Using LLM")
    print("=" * 65)
    print(f"Model   : {MODEL}")
    print(f"Tickets : {len(TICKETS)}")
    print(f"Tags    : {len(TAGS)}")
    print(f"Method  : Zero-shot vs Few-shot comparison")
    print("=" * 65)

    zero_shot_results = []
    few_shot_results  = []

    for i, ticket in enumerate(TICKETS, 1):
        print(f"\n[{i}/{len(TICKETS)}] Ticket #{ticket['id']}")
        print(f"  Text: {ticket['text'][:80]}...")
        print(f"  Ground truth: {ticket['ground_truth']}")

        # ── Zero-shot ──
        try:
            zs_prompt = build_zero_shot_prompt(ticket["text"])
            zs_raw    = call_llm(zs_prompt, client)
            zs_tags   = parse_tags(zs_raw)
        except Exception as e:
            print(f"  [Zero-shot ERROR] {e}")
            zs_tags = ["general_inquiry"]

        zero_shot_results.append({
            "id": ticket["id"],
            "text": ticket["text"],
            "ground_truth": ticket["ground_truth"],
            "predicted_tags": zs_tags,
        })
        print(f"  Zero-shot tags : {zs_tags}")
        time.sleep(DELAY)

        # ── Few-shot ──
        try:
            fs_prompt = build_few_shot_prompt(ticket["text"])
            fs_raw    = call_llm(fs_prompt, client)
            fs_tags   = parse_tags(fs_raw)
        except Exception as e:
            print(f"  [Few-shot ERROR] {e}")
            fs_tags = ["general_inquiry"]

        few_shot_results.append({
            "id": ticket["id"],
            "text": ticket["text"],
            "ground_truth": ticket["ground_truth"],
            "predicted_tags": fs_tags,
        })
        print(f"  Few-shot tags  : {fs_tags}")
        time.sleep(DELAY)

    # ── Evaluation ──
    zs_eval = evaluate(zero_shot_results)
    fs_eval = evaluate(few_shot_results)

    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS")
    print("=" * 65)
    print(f"\n{'Metric':<25} {'Zero-Shot':>12} {'Few-Shot':>12}")
    print("-" * 50)
    print(f"{'Top-1 Accuracy':<25} {zs_eval['top1_accuracy']:>11}% {fs_eval['top1_accuracy']:>11}%")
    print(f"{'Top-3 Accuracy':<25} {zs_eval['top3_accuracy']:>11}% {fs_eval['top3_accuracy']:>11}%")
    print(f"{'Tickets Evaluated':<25} {zs_eval['total']:>12} {fs_eval['total']:>12}")

    improvement_top1 = round(fs_eval["top1_accuracy"] - zs_eval["top1_accuracy"], 1)
    improvement_top3 = round(fs_eval["top3_accuracy"] - zs_eval["top3_accuracy"], 1)
    print(f"\nFew-shot improvement:")
    print(f"  Top-1: {'+' if improvement_top1 >= 0 else ''}{improvement_top1}%")
    print(f"  Top-3: {'+' if improvement_top3 >= 0 else ''}{improvement_top3}%")

    # ── Per-ticket summary ──
    print("\n" + "=" * 65)
    print("  PER-TICKET COMPARISON")
    print("=" * 65)
    for zs, fs in zip(zero_shot_results, few_shot_results):
        print(f"\nTicket #{zs['id']}: {zs['text'][:60]}...")
        print(f"  Ground truth : {zs['ground_truth']}")
        print(f"  Zero-shot    : {zs['predicted_tags']}")
        print(f"  Few-shot     : {fs['predicted_tags']}")

    # ── Save results ──
    output = {
        "model": MODEL,
        "total_tickets": len(TICKETS),
        "available_tags": TAGS,
        "zero_shot": {
            "evaluation": zs_eval,
            "results": zero_shot_results,
        },
        "few_shot": {
            "evaluation": fs_eval,
            "results": few_shot_results,
        },
    }

    with open("tagging_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 65)
    print("  Results saved to: tagging_results.json")
    print("  Task 5 completed successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()
