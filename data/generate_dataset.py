"""
Generates a labeled dataset of customer support tickets for 5-way
intent classification: billing, technical, account, feature_request, general.

This is a SYNTHETIC/TEMPLATE-based generator so the pipeline runs end-to-end
without needing an external data source. For a real resume project, replace
this with your own curated/labeled data (see README "Swapping in your own
real data") — keep the same output schema:

    {"text": "...", "label": "billing"}

Usage:
    python data/generate_dataset.py --n_per_class 120 --seed 42
"""
import argparse
import json
import random
from pathlib import Path

LABELS = ["billing", "technical", "account", "feature_request", "general"]

# Template pools per label. Each template has {slots} filled with varied
# values so the 500-1000 generated examples aren't near-duplicates of
# each other — this matters for actually teaching the model the category,
# not just memorizing a handful of surface strings.

PRODUCTS = ["the mobile app", "the dashboard", "the desktop client", "the web portal", "your service", "the API"]
PLANS = ["Pro", "Basic", "Team", "Enterprise", "Starter"]
AMOUNTS = ["$9.99", "$29.00", "$49.99", "$120.00", "$15.50", "$199.00"]
FEATURES = ["dark mode", "bulk export", "SSO login", "a mobile widget", "calendar sync", "multi-currency support", "an undo button", "offline mode"]
ERRORS = ["a 500 error", "a blank screen", "an infinite loading spinner", "a crash on launch", "a timeout error", "a 'session expired' message"]

TEMPLATES = {
    "billing": [
        "Subject: Duplicate charge\nBody: I was charged {amount} twice this month for my {plan} plan. Can you refund the extra charge?",
        "Subject: Invoice question\nBody: I need a copy of last month's invoice for {product} for my accounting records.",
        "Subject: Upgrade pricing\nBody: What would it cost to upgrade from {plan} to a higher tier plan?",
        "Subject: Failed payment\nBody: My card was declined when renewing my {plan} subscription, but I have funds available. Why did this happen?",
        "Subject: Cancel subscription\nBody: I'd like to cancel my {plan} plan and get a prorated refund for the unused days.",
        "Subject: Wrong amount charged\nBody: I was expecting to be billed {amount} but was charged a different amount this cycle.",
    ],
    "technical": [
        "Subject: App keeps crashing\nBody: {product} shows {error} every time I try to open it on my phone.",
        "Subject: Sync not working\nBody: Changes I make in {product} aren't syncing across my devices anymore.",
        "Subject: Can't upload files\nBody: I keep getting {error} whenever I try to upload a file larger than 10MB in {product}.",
        "Subject: Slow performance\nBody: {product} has been extremely slow to load for the past two days.",
        "Subject: Integration broken\nBody: The integration between {product} and our internal tools stopped working after the last update.",
        "Subject: Login loop\nBody: I keep getting logged out of {product} every few minutes with {error}.",
    ],
    "account": [
        "Subject: Can't reset password\nBody: I never received the password reset email for my account, can you help?",
        "Subject: Change email address\nBody: I need to update the email address linked to my account to a new one.",
        "Subject: Merge accounts\nBody: I accidentally created two accounts with different emails, can you merge them into one?",
        "Subject: Two-factor issues\nBody: I lost access to my authenticator app and can't get into my account.",
        "Subject: Delete my account\nBody: Please permanently delete my account and all associated data.",
        "Subject: Team member access\nBody: I need to add a new teammate to our {plan} workspace with admin permissions.",
    ],
    "feature_request": [
        "Subject: Please add {feature}\nBody: It would really help our workflow if {product} supported {feature}.",
        "Subject: Suggestion for {product}\nBody: Have you considered adding {feature}? A lot of teams like ours would use it.",
        "Subject: Feature idea\nBody: I'd love to see {feature} added to {product} in a future release.",
        "Subject: Missing capability\nBody: {product} is missing {feature}, which our competitor offers. Any plans to add it?",
        "Subject: Vote for {feature}\nBody: Just wanted to add my vote for {feature} — it's the one thing blocking wider adoption on our team.",
    ],
    "general": [
        "Subject: Just wanted to say thanks\nBody: Really enjoying {product} so far, great work from the team!",
        "Subject: General question\nBody: Do you have documentation or a getting-started guide for {product}?",
        "Subject: Partnership inquiry\nBody: We're interested in a potential partnership or integration opportunity, who should I talk to?",
        "Subject: Press inquiry\nBody: I'm a journalist writing about your company, could someone answer a few questions?",
        "Subject: Feedback\nBody: Overall {product} has been solid, just wanted to share some general feedback on the UI.",
        "Subject: Where to find X\nBody: Could you point me to where I can find your terms of service and privacy policy?",
    ],
}


def fill_template(template: str, rng: random.Random) -> str:
    return template.format(
        product=rng.choice(PRODUCTS),
        plan=rng.choice(PLANS),
        amount=rng.choice(AMOUNTS),
        feature=rng.choice(FEATURES),
        error=rng.choice(ERRORS),
    )


def generate(n_per_class: int, seed: int):
    rng = random.Random(seed)
    rows = []
    for label in LABELS:
        templates = TEMPLATES[label]
        for _ in range(n_per_class):
            template = rng.choice(templates)
            text = fill_template(template, rng)
            rows.append({"text": text, "label": label})
    rng.shuffle(rows)
    return rows


def split(rows, train_frac=0.7, val_frac=0.15):
    n = len(rows)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return rows[:n_train], rows[n_train:n_train + n_val], rows[n_train + n_val:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_class", type=int, default=120,
                         help="examples per label (default 120 -> 600 total)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="data/raw")
    args = parser.parse_args()

    rows = generate(args.n_per_class, args.seed)
    train, val, test = split(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in split_rows:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(split_rows)} examples to {path}")

    print(f"\nTotal: {len(rows)} examples across {len(LABELS)} labels.")
    print("NOTE: this is synthetic template data for pipeline testing. "
          "Swap in your own curated/labeled tickets before treating results "
          "as a real resume metric (see README).")


if __name__ == "__main__":
    main()
