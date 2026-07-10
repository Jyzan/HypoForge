"""
HypoForge — AI Scientist Workflow Framework for Scientific Hypothesis Generation & Research Plan Design.

A pluggable, LangGraph-based pipeline for:
  M1: Problem Understanding & Decomposition
  M2: Literature Search & Knowledge Extraction
  M3: Evidence Graph Construction
  M4: Hypothesis Generation & Screening (multi-agent)
  M5: Research Plan Design
  M6: Review & Iterative Refinement

Each module implements the ModuleProtocol interface and is registered via
the ModuleRegistry, enabling hot-swappable implementations — from stubs to
full LLM-powered modules.
"""

__version__ = "0.1.0"
