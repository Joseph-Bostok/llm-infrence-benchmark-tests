#!/usr/bin/env python3
"""
PInsight Workload Classifier

Classifies LLM inference prompts into workload types and recommends
KV cache optimization strategies. First stage of PInsight's in-situ loop:
    Profile -> [Classify] -> Tune -> Monitor -> Repeat

Workload types:
    DIALOGUE  - Multi-turn conversation, chat histories
    RAG       - Retrieval-augmented generation, document QA
    CODE      - Code completion, repository-level generation
    REASONING - Chain-of-thought, math/logic problem solving
"""

import json
import re
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, List
from enum import Enum


class WorkloadType(str, Enum):
    DIALOGUE = "dialogue"
    RAG = "rag"
    CODE = "code"
    REASONING = "reasoning"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class WorkloadSignals:
    """Raw signals extracted from a prompt for classification."""
    num_turns: int = 0
    has_system_prompt: bool = False
    num_documents: int = 0
    total_tokens_estimate: int = 0
    code_fraction: float = 0.0
    has_code_blocks: bool = False
    has_retrieval_markers: bool = False
    has_reasoning_markers: bool = False
    has_conversation_markers: bool = False
    avg_turn_length: float = 0.0
    max_segment_length: int = 0


@dataclass
class WorkloadClassification:
    """Result of classifying a workload."""
    workload_type: WorkloadType
    confidence: float
    signals: WorkloadSignals
    scores: Dict[str, float]
    reasoning: str


RETRIEVAL_PATTERNS = [
    r'(?i)(context|document|passage|source|retrieved|excerpt)\s*[:\d]',
    r'(?i)based on the (following|above|provided|given)',
    r'(?i)according to the (document|passage|text|context)',
    r'\[Document\s*\d+\]',
]

REASONING_PATTERNS = [
    r"(?i)let'?s (think|reason|solve|work|figure|consider)",
    r'(?i)step\s*\d+[:\.]',
    r'(?i)(therefore|thus|hence|consequently)',
    r'(?i)the answer is',
    r'(?i)solve.*equation',
]

CONVERSATION_PATTERNS = [
    r'(?i)(user|human|assistant|system)\s*:',
    r'"role"\s*:\s*"(user|assistant|system)"',
    r'(?i)<\|?(user|assistant|system)\|?>',
    r'(?i)\[INST\]',
]


def _count_matches(text, patterns):
    return sum(1 for p in patterns if re.search(p, text))


def _count_turns(text):
    t1 = len(re.findall(r'(?i)(user|human|assistant)\s*:', text)) // 2
    t2 = len(re.findall(r'"role"\s*:\s*"(user|assistant)"', text)) // 2
    return max(t1, t2)


def _count_documents(text):
    d1 = len(re.findall(r'(?i)(document|passage|source|context)\s*\d+', text))
    d2 = len(re.findall(r'\[Document\s*\d+\]', text))
    return max(d1, d2)


def _code_fraction(text):
    lines = [l for l in text.split('\n') if l.strip()]
    if not lines:
        return 0.0
    code_kw = r'^(def|class|import|from|if|for|while|return|var|let|const|function)\b'
    code_lines = sum(1 for l in lines if re.match(code_kw, l.strip())
                     or l.strip().endswith((';', '{', '}')))
    return code_lines / len(lines)


def extract_signals(prompt):
    s = WorkloadSignals()
    s.total_tokens_estimate = max(1, len(prompt) // 4)
    s.num_turns = _count_turns(prompt)
    s.num_documents = _count_documents(prompt)
    s.has_system_prompt = bool(re.search(
        r'(?i)(system\s*:|"role"\s*:\s*"system")', prompt))
    s.code_fraction = _code_fraction(prompt)
    s.has_code_blocks = bool(re.search(r'```[\w]*\n', prompt))
    s.has_retrieval_markers = _count_matches(prompt, RETRIEVAL_PATTERNS) >= 2
    s.has_reasoning_markers = _count_matches(prompt, REASONING_PATTERNS) >= 2
    s.has_conversation_markers = _count_matches(prompt, CONVERSATION_PATTERNS) >= 1
    if s.num_turns > 0:
        s.avg_turn_length = s.total_tokens_estimate / max(1, s.num_turns * 2)
    segments = re.split(r'(?i)(user|assistant|human)\s*:', prompt)
    if segments:
        s.max_segment_length = max(len(seg) // 4 for seg in segments)
    return s


def _score_dialogue(s):
    score = 0.0
    if s.num_turns >= 3: score += 0.4
    elif s.num_turns >= 1: score += 0.2
    if s.has_conversation_markers: score += 0.3
    if s.has_system_prompt: score += 0.1
    if 0 < s.avg_turn_length < 200: score += 0.1
    if s.code_fraction > 0.3: score -= 0.2
    if s.num_documents >= 2: score -= 0.2
    return max(0.0, min(1.0, score))


def _score_rag(s):
    score = 0.0
    if s.has_retrieval_markers: score += 0.4
    if s.num_documents >= 3: score += 0.3
    elif s.num_documents >= 1: score += 0.15
    if s.total_tokens_estimate > 2000: score += 0.1
    if s.max_segment_length > 500: score += 0.1
    if s.code_fraction > 0.4: score -= 0.2
    return max(0.0, min(1.0, score))


def _score_code(s):
    score = 0.0
    if s.code_fraction > 0.4: score += 0.4
    elif s.code_fraction > 0.2: score += 0.2
    if s.has_code_blocks: score += 0.3
    if not s.has_conversation_markers: score += 0.05
    if not s.has_retrieval_markers: score += 0.05
    return max(0.0, min(1.0, score))


def _score_reasoning(s):
    score = 0.0
    if s.has_reasoning_markers: score += 0.5
    if s.total_tokens_estimate > 1000 and not s.has_retrieval_markers: score += 0.1
    if s.num_documents == 0 and s.code_fraction < 0.1: score += 0.1
    return max(0.0, min(1.0, score))


def classify_workload(prompt):
    """Classify a prompt into a workload type."""
    signals = extract_signals(prompt)
    scores = {
        "dialogue": _score_dialogue(signals),
        "rag": _score_rag(signals),
        "code": _score_code(signals),
        "reasoning": _score_reasoning(signals),
    }
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    sorted_scores = sorted(scores.values(), reverse=True)

    if best_score < 0.2:
        wtype = WorkloadType.UNKNOWN
        reasoning = "No strong signals detected."
    elif len(sorted_scores) >= 2 and sorted_scores[0] - sorted_scores[1] < 0.1:
        wtype = WorkloadType.MIXED
        top2 = sorted(scores.items(), key=lambda x: -x[1])[:2]
        reasoning = f"Mixed: {top2[0][0]} ({top2[0][1]:.2f}) vs {top2[1][0]} ({top2[1][1]:.2f})"
    else:
        wtype = WorkloadType(best_type)
        reasoning = f"Classified as {wtype.value} (score: {best_score:.2f})"

    return WorkloadClassification(
        workload_type=wtype, confidence=best_score,
        signals=signals, scores=scores, reasoning=reasoning
    )


def main():
    parser = argparse.ArgumentParser(description="PInsight Workload Classifier")
    parser.add_argument("--prompt", type=str, help="Single prompt to classify")
    parser.add_argument("--file", type=str, help="JSON file with prompts")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    if args.prompt:
        r = classify_workload(args.prompt)
        print(f"\nType: {r.workload_type.value}  Confidence: {r.confidence:.2f}")
        print(f"Reasoning: {r.reasoning}")
        for t, s in sorted(r.scores.items(), key=lambda x: -x[1]):
            print(f"  {t:12s} {'█' * int(s * 20)} {s:.2f}")
    elif args.file:
        with open(args.file) as f:
            data = json.load(f)
        prompts = data if isinstance(data, list) else data.get('prompts', [data.get('prompt', '')])
        results = [classify_workload(p if isinstance(p, str) else p.get('prompt', str(p))) for p in prompts]
        for i, r in enumerate(results):
            print(f"[{i}] {r.workload_type.value:12s} (conf: {r.confidence:.2f})")
        if args.output:
            with open(args.output, 'w') as f:
                json.dump([asdict(r) for r in results], f, indent=2, default=str)
    else:
        print("\nPInsight Workload Classifier — Demo\n")
        demos = {
            "Dialogue": "System: You are helpful.\nUser: Hi\nAssistant: Hello!\nUser: What is AI?\nAssistant: AI is...\nUser: Tell me more",
            "RAG": "[Document 1] The Treaty of Versailles...\n[Document 2] The Paris Peace Conference...\n[Document 3] Wilson's Fourteen Points...\nBased on the above documents, answer: What was the relationship?",
            "Code": "```python\nclass TransformerBlock(nn.Module):\n    def __init__(self, d_model):\n        super().__init__()\n        self.attn = nn.MultiheadAttention(d_model, 8)\n```\nComplete the forward method with residual connections.",
            "Reasoning": "Solve step by step.\nStep 1: A train leaves at 60mph.\nStep 2: Another at 80mph.\nTherefore, let's think about when they meet.",
        }
        for name, prompt in demos.items():
            r = classify_workload(prompt)
            print(f"  {name:12s} -> {r.workload_type.value:12s} (conf: {r.confidence:.2f})")


if __name__ == '__main__':
    main()
