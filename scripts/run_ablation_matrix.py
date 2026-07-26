import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hypoforge.state import PipelineState, HypothesisCard, KnowledgeEntry, EvidenceGraph, EvidenceNode, EvidenceEdge, EvidenceNodeType, EvidenceEdgeRelation
from hypoforge.evaluation.scorer import score_pipeline_state_async

def create_mock_state():
    state = PipelineState(run_id="test_run", input_question="Can compound X inhibit Gene Y?")
    
    # Mock Hypothesis
    h = HypothesisCard(
        hypothesis_id="h1",
        statement="Compound X inhibits Gene Y expression, preventing tumor growth.",
        mechanism="Compound X binds to the promoter of Gene Y.",
        observable_predictions=["Gene Y mRNA will decrease."],
        falsification_conditions=["Gene Y mRNA does not decrease."],
        scores={"novelty": 0.8}
    )
    state.top_hypotheses = [h]
    
    # Mock Graph
    nodes = [
        EvidenceNode(id="n1", type=EvidenceNodeType.CLAIM, label="Compound X decreases protein synthesis."),
        EvidenceNode(id="n2", type=EvidenceNodeType.EVIDENCE, label="Gene Y is highly expressed in tumors."),
        EvidenceNode(id="n3", type=EvidenceNodeType.CLAIM, label="Protein Z inhibits Gene Y.")
    ]
    
    edges = [
        EvidenceEdge(source="n1", target="n3", relation=EvidenceEdgeRelation.SUPPORTS),
        EvidenceEdge(source="n3", target="n2", relation=EvidenceEdgeRelation.LIMITS) # Threat/Conflict edge
    ]
    
    state.evidence_graph = EvidenceGraph(nodes=nodes, edges=edges)
    return state

async def main():
    print("Running Ablation Matrix (Novelty & Consistency)...")
    state = create_mock_state()
    
    # Mock LLM Config (use fake or env var)
    llm_config = {
        "model_name": "qwen-plus",
        "api_key": os.environ.get("DASHSCOPE_API_KEY", "dummy"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    }
    
    report = await score_pipeline_state_async(state, llm_config=llm_config)
    import json
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
