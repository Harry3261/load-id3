"""ID3 decision tree implemented from scratch (no ML libraries)."""
import math
from collections import Counter

FEATURES = ["Income", "Employment", "CreditHistory"]
TARGET = "Eligible"

# Training data: (Income, Employment, CreditHistory, Eligible)
DATA = [
    # Good credit history
    ("Low", "Employed", "Good", "Yes"),
    ("Low", "Self-Employed", "Good", "Yes"),
    ("Low", "Unemployed", "Good", "No"),
    ("Medium", "Employed", "Good", "Yes"),
    ("Medium", "Self-Employed", "Good", "Yes"),
    ("Medium", "Unemployed", "Good", "Yes"),
    ("High", "Employed", "Good", "Yes"),
    ("High", "Self-Employed", "Good", "Yes"),
    ("High", "Unemployed", "Good", "Yes"),
    # Average credit history
    ("Low", "Employed", "Average", "No"),
    ("Low", "Self-Employed", "Average", "No"),
    ("Low", "Unemployed", "Average", "No"),
    ("Medium", "Employed", "Average", "Yes"),
    ("Medium", "Self-Employed", "Average", "Yes"),
    ("Medium", "Unemployed", "Average", "No"),
    ("High", "Employed", "Average", "Yes"),
    ("High", "Self-Employed", "Average", "Yes"),
    ("High", "Unemployed", "Average", "No"),
    # Poor credit history
    ("Low", "Employed", "Poor", "No"),
    ("Low", "Self-Employed", "Poor", "No"),
    ("Low", "Unemployed", "Poor", "No"),
    ("Medium", "Employed", "Poor", "No"),
    ("Medium", "Self-Employed", "Poor", "No"),
    ("Medium", "Unemployed", "Poor", "No"),
    ("High", "Employed", "Poor", "Yes"),
    ("High", "Self-Employed", "Poor", "No"),
    ("High", "Unemployed", "Poor", "No"),
]


def rows_as_dicts(data=DATA):
    return [dict(zip(FEATURES + [TARGET], r)) for r in data]


def entropy(rows):
    total = len(rows)
    counts = Counter(r[TARGET] for r in rows)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def info_gain(rows, feature):
    total = len(rows)
    remainder = 0.0
    for value in {r[feature] for r in rows}:
        subset = [r for r in rows if r[feature] == value]
        remainder += (len(subset) / total) * entropy(subset)
    return entropy(rows) - remainder


def majority(rows):
    return Counter(r[TARGET] for r in rows).most_common(1)[0][0]


def build_tree(rows, features):
    labels = {r[TARGET] for r in rows}
    if len(labels) == 1:
        return {"leaf": labels.pop(), "samples": len(rows)}
    if not features:
        return {"leaf": majority(rows), "samples": len(rows)}

    gains = {f: info_gain(rows, f) for f in features}
    best = max(gains, key=gains.get)
    node = {
        "feature": best,
        "gain": round(gains[best], 4),
        "entropy": round(entropy(rows), 4),
        "samples": len(rows),
        "default": majority(rows),
        "children": {},
    }
    remaining = [f for f in features if f != best]
    for value in sorted({r[best] for r in rows}):
        subset = [r for r in rows if r[best] == value]
        node["children"][value] = build_tree(subset, remaining)
    return node


def predict(tree, sample):
    """Return (label, path) where path lists the decisions taken."""
    path = []
    node = tree
    while "leaf" not in node:
        value = sample.get(node["feature"])
        path.append(f"{node['feature']} = {value}")
        child = node["children"].get(value)
        if child is None:
            return node["default"], path
        node = child
    return node["leaf"], path


def train():
    return build_tree(rows_as_dicts(), FEATURES)
