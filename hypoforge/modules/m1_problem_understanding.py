"""
M1: Problem Understanding & Decomposition.

Decomposes a frontier scientific question into structured sub-questions,
identifies domains, extracts key entities, and classifies the question type.

Output: ``ProblemCard`` in ``state.problem_card``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..protocol import ModuleProtocol
from ..registry import ModuleRegistry
from ..state import PipelineState, ProblemCard, QuestionType


@ModuleRegistry.register
class M1ProblemUnderstanding(ModuleProtocol):
    module_name = "m1"
    module_version = "0.1.0"
    description = "Problem decomposition: sub-questions, domain tagging, entity extraction"

    # ------------------------------------------------------------------
    # Stub data — replace with real Qwen API call in Phase 1
    # ------------------------------------------------------------------

    STUB_RESULTS: Dict[str, ProblemCard] = {
        "protein_folding": ProblemCard(
            original_question="蛋白质如何折叠及错误折叠导致疾病的机制？",
            domain=["structural biology", "biophysics", "cell biology"],
            sub_questions=[
                "What molecular chaperones govern protein folding in the ER?",
                "How do misfolded proteins aggregate into amyloid structures?",
                "What links protein misfolding to neurodegenerative diseases?",
                "Can pharmacological chaperones rescue misfolding phenotypes?",
            ],
            key_entities=["Hsp70", "Hsp90", "chaperonin", "amyloid-beta",
                          "alpha-synuclein", "prion", "unfolded protein response",
                          "proteasome", "autophagy"],
            question_type=QuestionType.MECHANISM,
        ),
    }

    # Default stub for any question not in STUB_RESULTS
    _DEFAULT = ProblemCard(
        original_question="",
        domain=["biomedicine"],
        sub_questions=[
            "What are the key molecular mechanisms involved?",
            "What experimental evidence supports current models?",
            "What knowledge gaps remain unresolved?",
        ],
        key_entities=["protein", "gene", "pathway"],
        question_type=QuestionType.MECHANISM,
    )

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

    def __init__(self, **kwargs):
        pass  # Stub needs no LLM client; real impl will accept LLMConfig

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        question = state.input_question

        # Simple keyword matching for stub — real impl calls Qwen
        if "protein" in question.lower() and "折叠" in question:
            card = self.STUB_RESULTS["protein_folding"]
        else:
            card = self._DEFAULT.model_copy()
            card.original_question = question

        return {"problem_card": card}

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return ["input_question"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["problem_card"]
