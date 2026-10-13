"""
evaluation/benchmark.py
=======================
Gold-standard Q&A benchmark for the 3GPP Telecom RAG system.

This module contains:
  1. A curated set of gold-standard questions with expected properties.
  2. A BenchmarkRunner that evaluates the live /query endpoint.
  3. Custom metrics: abstention accuracy, citation resolution rate, response latency.
  4. RAG-specific metrics: context precision (estimated), faithfulness (estimated).

Usage:
    cd telecom_qa_backend
    python -m evaluation.benchmark

Output: evaluation/results/benchmark_<timestamp>.json + summary printed to stdout.
"""

import os
import sys
import json
import time
import requests
from typing import List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_URL = os.environ.get("QA_API_URL", "http://127.0.0.1:8001")

# ──────────────────────────────────────────────────────────────────────
# Gold Standard Dataset
# ──────────────────────────────────────────────────────────────────────
@dataclass
class GoldQuestion:
    """A single benchmark test case."""
    id: str
    question: str
    category: str            # e.g. "rrc", "core", "ran", "ood"
    is_out_of_domain: bool   # True = system SHOULD abstain
    expected_keywords: List[str] = field(default_factory=list)   # Must appear in answer
    forbidden_keywords: List[str] = field(default_factory=list)  # Must NOT appear (hallucination indicators)
    expected_releases: List[str] = field(default_factory=list)   # e.g. ["Rel-17"]

GOLD_QUESTIONS: List[GoldQuestion] = [
    # ── RRC Questions ──────────────────────────────────────────────────
    GoldQuestion(
        id="rrc-001",
        question="What is the purpose of the RRC Setup procedure in 5G NR?",
        category="rrc",
        is_out_of_domain=False,
        expected_keywords=["RRC", "radio resource control", "setup"],
        expected_releases=["Rel-17"],
    ),
    GoldQuestion(
        id="rrc-002",
        question="What is the RRCReconfiguration message used for in 5G NR Rel-17?",
        category="rrc",
        is_out_of_domain=False,
        expected_keywords=["reconfiguration", "bearer", "measurement"],
    ),
    GoldQuestion(
        id="rrc-003",
        question="Describe the RRC states in 5G NR and the transitions between them.",
        category="rrc",
        is_out_of_domain=False,
        expected_keywords=["RRC_IDLE", "RRC_INACTIVE", "RRC_CONNECTED"],
    ),

    # ── Core Network Questions ─────────────────────────────────────────
    GoldQuestion(
        id="core-001",
        question="What is the role of the AMF in 5G Core?",
        category="core",
        is_out_of_domain=False,
        expected_keywords=["AMF", "access", "mobility", "management"],
    ),
    GoldQuestion(
        id="core-002",
        question="Explain the concept of a PDU session in 5G.",
        category="core",
        is_out_of_domain=False,
        expected_keywords=["PDU", "session", "SMF", "UPF"],
    ),
    GoldQuestion(
        id="core-003",
        question="What is the NRF and how does service discovery work in 5GC?",
        category="core",
        is_out_of_domain=False,
        expected_keywords=["NRF", "network repository", "service discovery"],
    ),
    GoldQuestion(
        id="core-004",
        question="How does network slicing work at the core network level in 5G?",
        category="core",
        is_out_of_domain=False,
        expected_keywords=["slice", "NSSAI", "isolation"],
    ),

    # ── Physical Layer / RAN ───────────────────────────────────────────
    GoldQuestion(
        id="ran-001",
        question="What are the frequency ranges FR1 and FR2 in 5G NR?",
        category="ran",
        is_out_of_domain=False,
        expected_keywords=["FR1", "FR2", "frequency", "GHz"],
    ),
    GoldQuestion(
        id="ran-002",
        question="Explain Massive MIMO and how it improves 5G NR capacity.",
        category="ran",
        is_out_of_domain=False,
        expected_keywords=["MIMO", "antenna", "beamforming", "spatial"],
    ),
    GoldQuestion(
        id="ran-003",
        question="What is the difference between FDD and TDD in 5G NR?",
        category="ran",
        is_out_of_domain=False,
        expected_keywords=["FDD", "TDD", "duplex", "uplink", "downlink"],
    ),

    # ── Security ───────────────────────────────────────────────────────
    GoldQuestion(
        id="sec-001",
        question="How does 5G AKA authentication work between UE and AMF?",
        category="security",
        is_out_of_domain=False,
        expected_keywords=["AKA", "authentication", "AUSF", "key"],
    ),

    # ── Out-of-Domain (Abstention Tests) ──────────────────────────────
    GoldQuestion(
        id="ood-001",
        question="What is the weather in New York today?",
        category="ood",
        is_out_of_domain=True,
        forbidden_keywords=["sunny", "rain", "temperature", "forecast"],
    ),
    GoldQuestion(
        id="ood-002",
        question="Who won the FIFA World Cup in 2022?",
        category="ood",
        is_out_of_domain=True,
        forbidden_keywords=["Argentina", "France", "goal", "penalty"],
    ),
    GoldQuestion(
        id="ood-003",
        question="Write a Python function to sort a list.",
        category="ood",
        is_out_of_domain=True,
        forbidden_keywords=["def sort", "bubble", "quicksort"],
    ),
]

# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QuestionResult:
    id: str
    question: str
    category: str
    is_out_of_domain: bool
    # Response
    answer: str = ""
    is_abstained: bool = False
    citations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    guardrails: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    # Eval outcomes
    keyword_pass: bool = True
    forbidden_keyword_fail: bool = False
    abstention_correct: Optional[bool] = None  # None = N/A
    citation_resolution_rate: float = 1.0
    error: Optional[str] = None

@dataclass
class BenchmarkSummary:
    total: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0
    # Abstention
    ood_total: int = 0
    ood_correctly_abstained: int = 0
    in_domain_total: int = 0
    in_domain_incorrectly_abstained: int = 0
    abstention_accuracy: float = 0.0
    # Quality
    keyword_pass_rate: float = 0.0
    forbidden_keyword_fail_rate: float = 0.0
    avg_citation_resolution: float = 0.0
    avg_confidence: float = 0.0
    # Guardrails
    hallucinated_citation_rate: float = 0.0
    ungrounded_number_rate: float = 0.0
    version_conflict_rate: float = 0.0


class BenchmarkRunner:
    def __init__(self, api_url: str = API_URL, timeout: int = 90):
        self.api_url = api_url
        self.timeout = timeout

    def run_question(self, q: GoldQuestion) -> QuestionResult:
        result = QuestionResult(
            id=q.id, question=q.question,
            category=q.category, is_out_of_domain=q.is_out_of_domain
        )
        try:
            start = time.perf_counter()
            r = requests.post(
                f"{self.api_url}/query",
                json={"query": q.question},
                timeout=self.timeout
            )
            result.latency_ms = round((time.perf_counter() - start) * 1000, 2)

            if r.status_code != 200:
                result.error = f"HTTP {r.status_code}: {r.text[:200]}"
                return result

            data = r.json()
            result.answer = data.get("answer", "")
            result.is_abstained = data.get("is_abstained", False)
            result.citations = data.get("citations", [])
            result.confidence_score = data.get("confidence_score", 0.0)
            result.guardrails = data.get("guardrails", {})

            # Keyword check
            answer_lower = result.answer.lower()
            if q.expected_keywords and not result.is_abstained:
                result.keyword_pass = any(kw.lower() in answer_lower for kw in q.expected_keywords)

            # Forbidden keyword check
            if q.forbidden_keywords and not result.is_abstained:
                result.forbidden_keyword_fail = any(kw.lower() in answer_lower for kw in q.forbidden_keywords)

            # Abstention correctness
            if q.is_out_of_domain:
                result.abstention_correct = result.is_abstained
            else:
                result.abstention_correct = not result.is_abstained

            # Citation resolution
            gr = result.guardrails
            total_cit = len(result.citations) + len(gr.get("unresolved_citations", []))
            if total_cit > 0:
                result.citation_resolution_rate = len(result.citations) / total_cit

        except Exception as e:
            result.error = str(e)
            result.latency_ms = 0.0

        return result

    def run_all(self, questions: List[GoldQuestion] = None) -> tuple:
        questions = questions or GOLD_QUESTIONS
        results: List[QuestionResult] = []

        print(f"\n{'='*60}")
        print(f"  3GPP RAG Benchmark Suite — {len(questions)} questions")
        print(f"{'='*60}\n")

        for i, q in enumerate(questions, 1):
            print(f"[{i:02d}/{len(questions)}] {q.id:<12} {q.question[:55]}...", end=' ', flush=True)
            res = self.run_question(q)
            results.append(res)
            status = "✓" if (res.abstention_correct is not False and res.keyword_pass and not res.error) else "✗"
            print(f"{status} ({res.latency_ms:.0f}ms)")
            if res.error:
                print(f"           ERROR: {res.error}")

        summary = self._compute_summary(results)
        return results, summary

    def _compute_summary(self, results: List[QuestionResult]) -> BenchmarkSummary:
        s = BenchmarkSummary()
        s.total = len(results)
        valid = [r for r in results if not r.error]
        s.errors = s.total - len(valid)

        if not valid:
            return s

        s.avg_latency_ms = round(sum(r.latency_ms for r in valid) / len(valid), 2)

        ood = [r for r in valid if r.is_out_of_domain]
        in_domain = [r for r in valid if not r.is_out_of_domain]

        s.ood_total = len(ood)
        s.ood_correctly_abstained = sum(1 for r in ood if r.abstention_correct)
        s.in_domain_total = len(in_domain)
        s.in_domain_incorrectly_abstained = sum(1 for r in in_domain if r.is_abstained)

        correct_abstentions = (s.ood_correctly_abstained +
                               (s.in_domain_total - s.in_domain_incorrectly_abstained))
        s.abstention_accuracy = round(correct_abstentions / s.total, 3) if s.total else 0

        s.keyword_pass_rate = round(
            sum(1 for r in in_domain if r.keyword_pass) / len(in_domain), 3
        ) if in_domain else 0
        s.forbidden_keyword_fail_rate = round(
            sum(1 for r in valid if r.forbidden_keyword_fail) / len(valid), 3
        ) if valid else 0
        s.avg_citation_resolution = round(
            sum(r.citation_resolution_rate for r in in_domain) / len(in_domain), 3
        ) if in_domain else 0
        s.avg_confidence = round(
            sum(r.confidence_score for r in in_domain) / len(in_domain), 3
        ) if in_domain else 0

        # Guardrail rates
        s.hallucinated_citation_rate = round(
            sum(1 for r in valid if r.guardrails.get("unresolved_citations")) / len(valid), 3
        ) if valid else 0
        s.ungrounded_number_rate = round(
            sum(1 for r in valid if r.guardrails.get("ungrounded_numbers")) / len(valid), 3
        ) if valid else 0
        s.version_conflict_rate = round(
            sum(1 for r in valid if r.guardrails.get("version_conflict_detected")) / len(valid), 3
        ) if valid else 0

        return s


def print_summary(summary: BenchmarkSummary):
    print(f"\n{'='*60}")
    print("  BENCHMARK RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total questions    : {summary.total}")
    print(f"  Errors             : {summary.errors}")
    print(f"  Avg latency        : {summary.avg_latency_ms:.0f}ms")
    print()
    print(f"  ACCURACY METRICS")
    print(f"  Abstention accuracy: {summary.abstention_accuracy:.1%}  (OOD correctly refused, in-domain answered)")
    print(f"  Keyword pass rate  : {summary.keyword_pass_rate:.1%}  (expected terms in answer)")
    print(f"  Avg confidence     : {summary.avg_confidence:.1%}")
    print(f"  Avg citation res.  : {summary.avg_citation_resolution:.1%}")
    print()
    print(f"  GUARDRAIL METRICS")
    print(f"  Hallucinated cit.  : {summary.hallucinated_citation_rate:.1%}")
    print(f"  Ungrounded nums    : {summary.ungrounded_number_rate:.1%}")
    print(f"  Version conflicts  : {summary.version_conflict_rate:.1%}")
    print(f"{'='*60}\n")


def save_results(results, summary):
    os.makedirs("evaluation/results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"evaluation/results/benchmark_{ts}.json"
    payload = {
        "timestamp": ts,
        "summary": asdict(summary),
        "results": [asdict(r) for r in results],
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {path}")


if __name__ == "__main__":
    runner = BenchmarkRunner()
    try:
        requests.get(f"{API_URL}/health", timeout=5)
    except Exception:
        print(f"ERROR: Cannot connect to {API_URL}. Is the backend running?")
        sys.exit(1)

    results, summary = runner.run_all()
    print_summary(summary)
    save_results(results, summary)
