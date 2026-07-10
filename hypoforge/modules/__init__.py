"""HypoForge pipeline modules (M1–M6).

Each module implements ``ModuleProtocol`` and is registered with
``ModuleRegistry``.  Stub implementations are provided so the pipeline
runs end-to-end from day 1; replace them with real LLM-powered
implementations as the project progresses.
"""

from .m1_problem_understanding import M1ProblemUnderstanding
from .m2_literature_search import M2LiteratureSearch
from .m3_evidence_graph import M3EvidenceGraph
from .m4_hypothesis_generation import M4HypothesisGeneration
from .m5_research_plan import M5ResearchPlan
from .m6_review_iteration import M6ReviewIteration

__all__ = [
    "M1ProblemUnderstanding",
    "M2LiteratureSearch",
    "M3EvidenceGraph",
    "M4HypothesisGeneration",
    "M5ResearchPlan",
    "M6ReviewIteration",
]
