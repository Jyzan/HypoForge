"""Provider changes preserve GLM identity, separate critics, and record embeddings."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hypoforge.benchmarks import agentideabench as bench
from hypoforge.benchmarks import agentideabench_scoring as scoring
from hypoforge.tools.credentials import read_api_key_file
from hypoforge.tools.qwen_client import QwenClient


REPO = Path(__file__).resolve().parents[1]
SF_CONFIG = REPO / "configs/agentideabench_glm51_siliconflow.yaml"
ORIGINAL_CONFIG = REPO / "configs/agentideabench_glm51.yaml"


@pytest.mark.parametrize("content", [
    'sk-test-key\n“https://api.siliconflow.cn/v1“\n',
    'API_KEY="sk-test-key"\nBASE_URL="https://api.siliconflow.cn/v1/"\n',
    '\ufeff{"api_key": "sk-test-key", "base_url": "https://api.siliconflow.cn/v1"}',
])
def test_text_credentials_accept_runtime_file_and_curly_quotes(tmp_path, content):
    path = tmp_path / "credentials.txt"
    path.write_text(content)
    assert read_api_key_file(path) == ("sk-test-key", "https://api.siliconflow.cn/v1")


@pytest.mark.parametrize("content", [
    'sk-test-key\nsk-second-key\nhttps://api.siliconflow.cn/v1',
    'sk-test-key\nhttps://api.siliconflow.cn/v1\nhttps://other.test/v1',
    'sk-test-key\nhttp://api.siliconflow.cn/v1',
    'sk-test-key\nhttps://user:password@api.siliconflow.cn/v1',
    'sk-test-key\nhttps://api.siliconflow.cn/v1?secret=sk-test-key',
    '{"api_key": ["sk-test-key"], "base_url": "https://api.siliconflow.cn/v1"}',
])
def test_ambiguous_or_unsafe_credentials_fail_without_disclosing_key(tmp_path, content):
    path = tmp_path / "credentials.txt"
    path.write_text(content)
    with pytest.raises(ValueError) as error:
        read_api_key_file(path)
    assert "sk-test-key" not in str(error.value)


def configure_sf(tmp_path, monkeypatch):
    # Restore even variables changed by configure() when this test finishes.
    monkeypatch.setenv("ENTITY_EMBEDDING_API_KEY", "previous-key")
    monkeypatch.setenv("ENTITY_EMBEDDING_BASE_URL", "https://previous.test/v1")
    path = tmp_path / "llm_api.txt"
    path.write_text('sk-test-key\n“https://api.siliconflow.cn/v1“\n')
    return bench.configure(SF_CONFIG, None, api_key_file=str(path)), path


def test_sf_generation_configuration_preserves_pipeline_and_discloses_resource_change(tmp_path, monkeypatch):
    config, _ = configure_sf(tmp_path, monkeypatch)
    import os
    assert all(t.model == "Pro/zai-org/GLM-5.1" and t.seed == 42 and t.enable_thinking is False
               and t.api_base == "https://api.siliconflow.cn/v1"
               for t in (config.qwen.base, config.qwen.max, config.qwen.plus, config.qwen.turbo))
    assert config.entity_embedding_model == config.evaluation.embedding.model_name == "Qwen/Qwen3-Embedding-8B"
    assert config.enabled_modules == ["m1", "m2", "m3", "m4", "m5", "m6"]
    assert config.max_iterations == 3 and config.max_evidence_gap_rounds == 1
    assert config.search.tools == ["semantic_scholar", "pubmed"]
    assert config.scoring.auto_score is False
    assert os.environ["ENTITY_EMBEDDING_API_KEY"] == "sk-test-key"
    public = json.dumps(bench.public_config(config))
    assert "sk-test-key" not in public
    assert "Qwen/Qwen3-Embedding-8B" in public


def test_generation_and_embedding_credentials_can_use_separate_providers(tmp_path, monkeypatch):
    _, key_file = configure_sf(tmp_path, monkeypatch)
    embed_csv = tmp_path / "embedding.csv"
    embed_csv.write_text('apiKey,separate-embedding-key\nopenAiCompatible,https://embedding.test/v1\n')
    config = bench.configure(ORIGINAL_CONFIG, None, api_key_file=str(key_file),
                             embedding_csv=str(embed_csv), embedding_model="Qwen/Qwen3-Embedding-8B")
    import os
    assert config.qwen.base.model == "Pro/zai-org/GLM-5.1"
    assert config.qwen.base.api_key == "sk-test-key"
    assert config.evaluation.embedding.base_url == "https://embedding.test/v1"
    assert os.environ["ENTITY_EMBEDDING_API_KEY"] == "separate-embedding-key"


def test_switch_does_not_silently_replace_unsupported_embedding_model(tmp_path, monkeypatch):
    _, key_file = configure_sf(tmp_path, monkeypatch)
    config = bench.configure(ORIGINAL_CONFIG, None, api_key_file=str(key_file))
    assert config.entity_embedding_model == "text-embedding-v3"


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["generation", "all"])
async def test_preflight_separates_embedding_and_critic_endpoints(tmp_path, monkeypatch, capsys, scope):
    config, _ = configure_sf(tmp_path, monkeypatch)
    calls = []

    class FakeLLM:
        async def structured_chat(self, **kwargs):
            calls.append("json")
            return {"ok": True}

        async def chat(self, **kwargs):
            calls.append("text")
            return "OK"

    class FakeAPI:
        def __init__(self, cfg):
            self.config = cfg
            self.client = SimpleNamespace(
                embeddings=SimpleNamespace(create=self.embedding),
                close=lambda: calls.append(("close", cfg.api_base)))

        def embedding(self, **kwargs):
            assert self.config.api_base == "https://api.siliconflow.cn/v1"
            assert self.config.api_key == "sk-test-key"
            assert kwargs["model"] == "Qwen/Qwen3-Embedding-8B"
            assert kwargs["encoding_format"] == "float"
            assert len(kwargs["input"]) == 2
            calls.append("embedding")
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2]) for _ in range(2)])

        def completion(self, model, *args, **kwargs):
            assert scope == "all"
            assert self.config.api_base == "https://original-critics.test/v1"
            assert self.config.api_key == "critic-key"
            calls.append(model)
            return "OK"

    critic = config.qwen.base.model_copy(deep=True)
    critic.api_key, critic.api_base = "critic-key", "https://original-critics.test/v1"
    monkeypatch.setattr(QwenClient, "from_config", lambda *args: FakeLLM())
    monkeypatch.setattr(scoring, "CriticAPI", FakeAPI)
    from hypoforge.tools import semantic_scholar
    native_calls, dated_calls = [], []

    def native_search(query, limit):
        native_calls.append((query, limit))
        return [{"paper_id": "test"}]

    def dated_search(query):
        dated_calls.append(query)
        return [{"paperId": "test"}]

    monkeypatch.setattr(semantic_scholar, "_search", native_search)
    monkeypatch.setattr(scoring, "search_prior_art", dated_search)
    assert await scoring.preflight(config, critic_config=critic, scope=scope)
    assert calls[:3] == ["json", "text", "embedding"]
    assert [c for c in calls if c in bench.CRITICS] == (list(bench.CRITICS) if scope == "all" else [])
    if scope == "generation":
        assert "external critics have not been checked" in capsys.readouterr().out
        assert native_calls == [("graph neural networks", 1)] and not dated_calls
    else:
        assert dated_calls == ["graph neural networks"] and not native_calls


def test_sf_scoring_requires_explicit_original_critic_credentials(tmp_path, monkeypatch, capsys):
    import run_agentideabench as entry
    _, key_file = configure_sf(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_agentideabench.py", "--phase", "score",
                                     "--config", str(SF_CONFIG), "--api-key-file", str(key_file)])
    monkeypatch.setattr(scoring, "score", lambda *args, **kwargs: pytest.fail("Critics were not configured"))
    assert entry.main() == 1
    assert "separate original critic credentials" in capsys.readouterr().out


def test_scoring_uses_supplied_original_provider_not_generation_key(tmp_path, monkeypatch):
    import run_agentideabench as entry
    _, key_file = configure_sf(tmp_path, monkeypatch)
    critic_file = tmp_path / "critics.csv"
    critic_file.write_text('apiKey,critic-key\nopenAiCompatible,https://original-critics.test/v1\n')
    observed = []

    def score(output, root, config, *args):
        observed.append((config.api_key, config.api_base, config.model))
        return True

    monkeypatch.setattr(scoring, "score", score)
    monkeypatch.setattr(sys, "argv", ["run_agentideabench.py", "--phase", "score", "--config", str(SF_CONFIG),
                                     "--api-key-file", str(key_file), "--critic-api-key-csv", str(critic_file)])
    assert entry.main() == 0
    assert observed == [("critic-key", "https://original-critics.test/v1", "glm-5.1")]
