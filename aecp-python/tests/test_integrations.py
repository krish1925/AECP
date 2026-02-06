"""
Tests for AECP integrations: AECPAgent, AECPEnabledAgent, AECPMCPServer, AECPMCPClient.

Validates the three integration scenarios:
  - Scenario 2: Decoupled LLM + AECP agent (AECPAgent, AECPEnabledAgent)
  - Scenario 3: MCP server/client with AECP (AECPMCPServer, AECPMCPClient)
"""

import pytest
import numpy as np

from aecp import AECP
from aecp.adapters.mock import MockAdapter
from aecp.adapters.local import LocalModelAdapter
from aecp.integrations.base import AECPAgent
from aecp.integrations.agent_framework import AECPEnabledAgent
from aecp.integrations.mcp import AECPMCPServer, AECPMCPClient


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_384():
    return MockAdapter(dimensions=384, model_name="mock-384")


@pytest.fixture
def mock_768():
    return MockAdapter(dimensions=768, model_name="mock-768")


@pytest.fixture
def small_vocab():
    return [
        "hello", "world", "test", "data", "model",
        "system", "process", "function", "algorithm", "network",
    ] * 2


# ── AECPAgent Tests ──────────────────────────────────────────────────


class TestAECPAgent:
    """Tests for the base AECPAgent class."""

    def test_init(self, mock_384):
        agent = AECPAgent(
            embedder=mock_384,
            llm_provider="openai:gpt-4",
            system_prompt="You are helpful.",
            agent_id="test_agent",
        )

        assert agent.agent_id == "test_agent"
        assert agent.llm_provider == "openai:gpt-4"
        assert agent.system_prompt == "You are helpful."

    def test_init_default_llm(self, mock_384):
        agent = AECPAgent(embedder=mock_384, agent_id="test")

        assert agent.llm_provider is None
        assert agent.system_prompt == ""

    def test_embed(self, mock_384):
        agent = AECPAgent(embedder=mock_384, agent_id="test")
        emb = agent.embed("hello world")

        assert isinstance(emb, np.ndarray)
        assert emb.shape == (384,)

    def test_embed_batch(self, mock_384):
        agent = AECPAgent(embedder=mock_384, agent_id="test")
        embs = agent.embed_batch(["hello", "world"])

        assert isinstance(embs, np.ndarray)
        assert embs.shape == (2, 384)

    def test_embedder_property(self, mock_384):
        agent = AECPAgent(embedder=mock_384, agent_id="test")

        assert agent.embedder is mock_384

    def test_calibrate_with_another_agent(self, mock_384, mock_768, small_vocab):
        agent1 = AECPAgent(embedder=mock_384, agent_id="a1")
        agent2 = AECPAgent(embedder=mock_768, agent_id="a2")

        result = agent1.calibrate_with(agent2, vocabulary=small_vocab, verbose=False)
        assert result.success

    def test_calibrate_with_raw_aecp(self, mock_384, mock_768, small_vocab):
        agent = AECPAgent(embedder=mock_384, agent_id="a1")
        raw_aecp = AECP(mock_768, agent_id="a2")

        result = agent.calibrate_with(raw_aecp, vocabulary=small_vocab, verbose=False)
        assert result.success

    def test_transfer_to_agent(self, mock_384, mock_768, small_vocab):
        agent1 = AECPAgent(embedder=mock_384, agent_id="a1")
        agent2 = AECPAgent(embedder=mock_768, agent_id="a2")
        agent1.calibrate_with(agent2, vocabulary=small_vocab, verbose=False)

        emb = agent1.embed("test query")
        transfer = agent1.transfer_to(agent2, emb)

        assert transfer.embedding.shape == (768,)
        assert transfer.source_agent == "a1"
        assert transfer.target_agent == "a2"

    def test_transfer_to_raw_aecp(self, mock_384, mock_768, small_vocab):
        agent = AECPAgent(embedder=mock_384, agent_id="a1")
        raw_aecp = AECP(mock_768, agent_id="a2")
        agent.calibrate_with(raw_aecp, vocabulary=small_vocab, verbose=False)

        emb = agent.embed("test")
        transfer = agent.transfer_to(raw_aecp, emb)

        assert transfer.embedding.shape == (768,)

    def test_find_similar(self, mock_384):
        agent = AECPAgent(embedder=mock_384, agent_id="test")

        candidates = [agent.embed(t) for t in ["cat", "dog", "car", "house"]]
        query = agent.embed("cat")

        indices = agent.find_similar(query, candidates, top_k=2)

        assert len(indices) == 2
        assert indices[0] == 0  # "cat" should be most similar to itself

    def test_send_message(self, mock_384, mock_768, small_vocab):
        agent1 = AECPAgent(embedder=mock_384, agent_id="a1")
        agent2 = AECPAgent(embedder=mock_768, agent_id="a2")
        agent1.calibrate_with(agent2, vocabulary=small_vocab, verbose=False)

        result = agent1.send_message(agent2, "hello world")

        assert result["method"] == "aecp"
        # AECP success wraps the transfer data in a "data" key
        assert "data" in result
        assert "embedding" in result["data"]

    def test_send_message_text_fallback(self, mock_384, mock_768):
        """Without calibration, send_message should fall back to text."""
        agent1 = AECPAgent(embedder=mock_384, agent_id="a1")
        agent2 = AECPAgent(embedder=mock_768, agent_id="a2")

        result = agent1.send_message(agent2, "hello world", fallback_to_text=True)

        assert result["method"] == "text"
        assert "message" in result

    def test_get_info(self, mock_384):
        agent = AECPAgent(
            embedder=mock_384,
            llm_provider="openai:gpt-4",
            agent_id="info_test",
        )
        info = agent.get_info()

        assert info["agent_id"] == "info_test"
        assert info["llm_provider"] == "openai:gpt-4"
        assert info["supports_aecp"] is True
        assert "embedding_model" in info
        assert "embedding_dimensions" in info

    def test_repr(self, mock_384):
        agent = AECPAgent(embedder=mock_384, agent_id="repr_test")
        r = repr(agent)

        assert "AECPAgent" in r
        assert "repr_test" in r


# ── AECPEnabledAgent Tests ───────────────────────────────────────────


class TestAECPEnabledAgent:
    """Tests for AECPEnabledAgent with knowledge base."""

    def test_init(self, mock_384):
        agent = AECPEnabledAgent(
            embedder=mock_384,
            llm_provider="openai:gpt-4",
            agent_id="kb_agent",
        )

        assert agent.agent_id == "kb_agent"
        assert len(agent.knowledge_texts) == 0
        assert len(agent.knowledge_embeddings) == 0

    def test_inherits_from_aecp_agent(self, mock_384):
        agent = AECPEnabledAgent(embedder=mock_384, agent_id="kb")
        assert isinstance(agent, AECPAgent)

    def test_add_knowledge(self, mock_384):
        agent = AECPEnabledAgent(embedder=mock_384, agent_id="kb")
        count = agent.add_knowledge([
            "Dopamine is a neurotransmitter.",
            "Serotonin regulates mood.",
        ])

        assert count == 2
        assert len(agent.knowledge_texts) == 2
        assert len(agent.knowledge_embeddings) == 2

    def test_add_knowledge_incremental(self, mock_384):
        agent = AECPEnabledAgent(embedder=mock_384, agent_id="kb")
        agent.add_knowledge(["fact one"])
        count = agent.add_knowledge(["fact two", "fact three"])

        assert count == 3
        assert len(agent.knowledge_texts) == 3

    def test_retrieve(self, mock_384):
        agent = AECPEnabledAgent(embedder=mock_384, agent_id="kb")
        agent.add_knowledge([
            "Dopamine is a neurotransmitter.",
            "Serotonin regulates mood.",
            "Python is a programming language.",
        ])

        results = agent.retrieve("neurotransmitter", top_k=2)

        assert len(results) == 2
        assert isinstance(results[0], tuple)
        assert isinstance(results[0][0], str)   # text
        assert isinstance(results[0][1], float)  # similarity score

    def test_retrieve_empty_knowledge(self, mock_384):
        agent = AECPEnabledAgent(embedder=mock_384, agent_id="kb")
        results = agent.retrieve("anything")

        assert results == []

    def test_build_context(self, mock_384):
        agent = AECPEnabledAgent(embedder=mock_384, agent_id="kb")
        agent.add_knowledge(["fact one", "fact two", "fact three"])

        context = agent.build_context("query", top_k=2)

        assert isinstance(context, str)
        assert len(context) > 0

    def test_clear_knowledge(self, mock_384):
        agent = AECPEnabledAgent(embedder=mock_384, agent_id="kb")
        agent.add_knowledge(["fact one", "fact two"])
        agent.clear_knowledge()

        assert len(agent.knowledge_texts) == 0
        assert len(agent.knowledge_embeddings) == 0

    def test_communicate_with(self, mock_384, mock_768, small_vocab):
        agent1 = AECPEnabledAgent(embedder=mock_384, agent_id="sender")
        agent2 = AECPEnabledAgent(embedder=mock_768, agent_id="receiver")

        agent1.calibrate_with(agent2, vocabulary=small_vocab, verbose=False)

        result = agent1.communicate_with(agent2, "dopamine pathways")

        assert "transferred_embedding" in result
        assert result["source_agent"] == "sender"
        assert result["target_agent"] == "receiver"
        assert isinstance(result["transferred_embedding"], np.ndarray)
        assert result["transferred_embedding"].shape == (768,)
        assert "query" in result
        assert "expected_similarity" in result

    def test_process_transferred_query(self, mock_384, mock_768, small_vocab):
        agent1 = AECPEnabledAgent(embedder=mock_384, agent_id="sender")
        agent2 = AECPEnabledAgent(embedder=mock_768, agent_id="receiver")

        # Agent 2 has knowledge
        agent2.add_knowledge([
            "Dopamine pathways in the brain.",
            "Serotonin function.",
        ])

        agent1.calibrate_with(agent2, vocabulary=small_vocab, verbose=False)

        # Transfer query
        comm = agent1.communicate_with(agent2, "dopamine")
        transferred_emb = comm["transferred_embedding"]

        # Process on receiver side
        results = agent2.process_transferred_query(transferred_emb, top_k=1)

        assert len(results) == 1
        assert isinstance(results[0][0], str)
        assert isinstance(results[0][1], float)

    def test_process_transferred_query_empty_kb(self, mock_768):
        agent = AECPEnabledAgent(embedder=mock_768, agent_id="empty")
        fake_emb = np.random.randn(768)

        results = agent.process_transferred_query(fake_emb)
        assert results == []

    def test_get_knowledge_stats(self, mock_384):
        agent = AECPEnabledAgent(embedder=mock_384, agent_id="stats_test")
        agent.add_knowledge(["fact one"])

        stats = agent.get_knowledge_stats()

        assert stats["knowledge_size"] == 1
        assert stats["supports_aecp"] is True
        assert "embedding_dimensions" in stats

    def test_repr(self, mock_384):
        agent = AECPEnabledAgent(embedder=mock_384, agent_id="repr_test")
        agent.add_knowledge(["fact one"])
        r = repr(agent)

        assert "AECPEnabledAgent" in r
        assert "repr_test" in r
        assert "1 items" in r


# ── AECPMCPServer Tests ──────────────────────────────────────────────


class TestAECPMCPServer:
    """Tests for AECPMCPServer."""

    def test_init(self, mock_768):
        server = AECPMCPServer(
            server_name="test-server",
            embedder=mock_768,
        )

        assert server.server_name == "test-server"
        assert server.supports_aecp is True
        assert len(server.documents) == 0
        assert len(server.document_embeddings) == 0

    def test_init_custom_agent_id(self, mock_768):
        server = AECPMCPServer(
            server_name="test",
            embedder=mock_768,
            agent_id="custom-server-id",
        )

        assert server.aecp.agent_id == "custom-server-id"

    def test_init_default_agent_id(self, mock_768):
        server = AECPMCPServer(server_name="my-server", embedder=mock_768)

        assert server.aecp.agent_id == "my-server"

    def test_add_documents(self, mock_768):
        server = AECPMCPServer(server_name="test", embedder=mock_768)
        count = server.add_documents(["doc1", "doc2", "doc3"])

        assert count == 3
        assert len(server.documents) == 3
        assert len(server.document_embeddings) == 3

    def test_add_documents_incremental(self, mock_768):
        server = AECPMCPServer(server_name="test", embedder=mock_768)
        server.add_documents(["doc1"])
        count = server.add_documents(["doc2", "doc3"])

        assert count == 3

    def test_semantic_search_text(self, mock_768):
        server = AECPMCPServer(server_name="test", embedder=mock_768)
        server.add_documents(["machine learning", "deep learning", "cooking recipes"])

        result = server.semantic_search(query="neural networks", top_k=2)

        assert result["method"] == "text"
        assert result["fidelity"] == "43%"
        assert len(result["results"]) == 2
        assert result["source_agent_id"] is None

        # Each result should be a dict with text and score
        for r in result["results"]:
            assert "text" in r
            assert "score" in r
            assert "index" in r

    def test_semantic_search_aecp(self, mock_768):
        server = AECPMCPServer(server_name="test", embedder=mock_768)
        server.add_documents(["machine learning", "deep learning"])

        fake_emb = np.random.randn(768).tolist()

        result = server.semantic_search(
            embedding=fake_emb,
            source_agent_id="client_agent",
            top_k=1,
        )

        assert result["method"] == "aecp"
        assert result["fidelity"] == "97%"
        assert result["source_agent_id"] == "client_agent"
        assert len(result["results"]) == 1

    def test_semantic_search_no_args_raises(self, mock_768):
        server = AECPMCPServer(server_name="test", embedder=mock_768)

        with pytest.raises(ValueError, match="Either"):
            server.semantic_search()

    def test_semantic_search_empty_index(self, mock_768):
        server = AECPMCPServer(server_name="test", embedder=mock_768)

        result = server.semantic_search(query="anything")
        assert result["results"] == []

    def test_aecp_negotiate(self, mock_768):
        server = AECPMCPServer(server_name="knowledge-server", embedder=mock_768)

        info = server.aecp_negotiate(client_agent_id="my_client")

        assert info["supports_aecp"] is True
        assert info["server_name"] == "knowledge-server"
        assert info["client_agent_id"] == "my_client"
        assert "dimensions" in info
        assert "embedding_model" in info
        assert "protocol_version" in info

    def test_get_stats(self, mock_768):
        server = AECPMCPServer(server_name="test", embedder=mock_768)
        server.add_documents(["doc1"])
        server.semantic_search(query="test")

        stats = server.get_stats()

        assert stats["documents_added"] == 1
        assert stats["text_searches"] == 1
        assert stats["aecp_searches"] == 0
        assert stats["total_documents"] == 1

    def test_get_stats_aecp_search(self, mock_768):
        server = AECPMCPServer(server_name="test", embedder=mock_768)
        server.add_documents(["doc1"])
        server.semantic_search(
            embedding=np.random.randn(768).tolist(),
            source_agent_id="cli",
        )

        stats = server.get_stats()
        assert stats["aecp_searches"] == 1

    def test_repr(self, mock_768):
        server = AECPMCPServer(server_name="test", embedder=mock_768)
        r = repr(server)

        assert "AECPMCPServer" in r
        assert "test" in r


# ── AECPMCPClient Tests ──────────────────────────────────────────────


class TestAECPMCPClient:
    """Tests for AECPMCPClient."""

    def test_init(self, mock_384):
        client = AECPMCPClient(embedder=mock_384, agent_id="test_client")

        assert client.aecp.agent_id == "test_client"
        assert len(client.calibrated_servers) == 0

    def test_init_default_agent_id(self, mock_384):
        client = AECPMCPClient(embedder=mock_384)

        assert client.aecp.agent_id == "aecp_mcp_client"

    def test_connect_and_calibrate(self, mock_384, mock_768, small_vocab):
        server = AECPMCPServer(server_name="srv", embedder=mock_768)
        client = AECPMCPClient(embedder=mock_384, agent_id="cli")

        result = client.connect_and_calibrate(
            server, vocabulary=small_vocab, verbose=False
        )

        assert result.success
        assert client.is_calibrated_with(server)

    def test_is_calibrated_with_false(self, mock_384, mock_768):
        server = AECPMCPServer(server_name="srv", embedder=mock_768)
        client = AECPMCPClient(embedder=mock_384, agent_id="cli")

        assert not client.is_calibrated_with(server)

    def test_semantic_search_aecp_path(self, mock_384, mock_768, small_vocab):
        server = AECPMCPServer(server_name="srv", embedder=mock_768)
        server.add_documents(["machine learning", "deep learning", "cooking"])

        client = AECPMCPClient(embedder=mock_384, agent_id="cli")
        client.connect_and_calibrate(server, vocabulary=small_vocab, verbose=False)

        result = client.semantic_search("neural networks", server, top_k=2)

        assert result["method"] == "aecp"
        assert result["fidelity"] == "97%"
        assert len(result["results"]) == 2

    def test_semantic_search_text_fallback(self, mock_384, mock_768):
        server = AECPMCPServer(server_name="srv", embedder=mock_768)
        server.add_documents(["machine learning", "deep learning"])

        client = AECPMCPClient(embedder=mock_384, agent_id="cli")
        # NOT calibrated

        result = client.semantic_search("neural networks", server, top_k=1)

        assert result["method"] == "text"
        assert result["fidelity"] == "43%"

    def test_get_stats_initial(self, mock_384):
        client = AECPMCPClient(embedder=mock_384, agent_id="cli")

        stats = client.get_stats()

        assert stats["aecp_queries"] == 0
        assert stats["text_queries"] == 0
        assert stats["calibrations"] == 0
        assert stats["calibrated_servers"] == []

    def test_get_stats_after_operations(self, mock_384, mock_768, small_vocab):
        server = AECPMCPServer(server_name="srv", embedder=mock_768)
        server.add_documents(["doc1"])

        client = AECPMCPClient(embedder=mock_384, agent_id="cli")
        client.connect_and_calibrate(server, vocabulary=small_vocab, verbose=False)
        client.semantic_search("test", server)

        stats = client.get_stats()

        assert stats["calibrations"] == 1
        assert stats["aecp_queries"] == 1
        assert stats["text_queries"] == 0
        assert len(stats["calibrated_servers"]) == 1

    def test_repr(self, mock_384):
        client = AECPMCPClient(embedder=mock_384, agent_id="repr_test")
        r = repr(client)

        assert "AECPMCPClient" in r
        assert "repr_test" in r


# ── End-to-End Integration Tests ─────────────────────────────────────


class TestEndToEnd:
    """Full scenario tests combining multiple components."""

    def test_scenario1_local_models(self):
        """Scenario 1: Local model weights with full control."""
        from tests.test_local_adapter import FakeSentenceTransformer

        model1 = FakeSentenceTransformer(dimensions=384)
        model2 = FakeSentenceTransformer(dimensions=768)

        agent1 = AECP(LocalModelAdapter(model1), agent_id="local_a")
        agent2 = AECP(LocalModelAdapter(model2), agent_id="local_b")

        vocab = ["hello", "world", "test", "example", "data"] * 4
        result = agent1.calibrate_with(agent2, vocabulary=vocab, verbose=False)
        assert result.success

        transfer = agent1.transfer_to(agent2.agent_id, "test")
        assert transfer.embedding.shape == (768,)

    def test_scenario2_decoupled_agents(self):
        """Scenario 2: Decoupled LLM + AECP architecture."""
        adapter1 = MockAdapter(dimensions=384)
        adapter2 = MockAdapter(dimensions=768)

        agent1 = AECPEnabledAgent(
            embedder=adapter1,
            llm_provider="openai:gpt-4",
            agent_id="expert_a",
        )
        agent2 = AECPEnabledAgent(
            embedder=adapter2,
            llm_provider="anthropic:claude-3",
            agent_id="expert_b",
        )

        # Build knowledge bases
        agent1.add_knowledge(["Dopamine is a neurotransmitter."])
        agent2.add_knowledge(["Serotonin regulates mood."])

        # Calibrate
        vocab = ["hello", "world", "test", "data", "model"] * 4
        result = agent1.calibrate_with(agent2, vocabulary=vocab, verbose=False)
        assert result.success

        # Cross-agent communication
        comm = agent1.communicate_with(agent2, "neurotransmitter function")
        assert comm["transferred_embedding"].shape == (768,)

        # Process transferred query on agent2
        results = agent2.process_transferred_query(comm["transferred_embedding"])
        assert len(results) > 0
        assert results[0][0] == "Serotonin regulates mood."

    def test_scenario3_mcp_server_client(self):
        """Scenario 3: MCP server + client with AECP."""
        server = AECPMCPServer(
            server_name="neuroscience-kb",
            embedder=MockAdapter(dimensions=768),
        )
        server.add_documents([
            "Dopamine pathways in the brain",
            "Serotonin and mood regulation",
            "Neural plasticity mechanisms",
        ])

        client = AECPMCPClient(
            embedder=MockAdapter(dimensions=384),
            agent_id="research_client",
        )

        # Calibrate
        vocab = ["hello", "world", "test", "data", "model"] * 4
        cal_result = client.connect_and_calibrate(
            server, vocabulary=vocab, verbose=False
        )
        assert cal_result.success

        # AECP search
        result = client.semantic_search("dopamine pathways", server, top_k=2)
        assert result["method"] == "aecp"
        assert result["fidelity"] == "97%"
        assert len(result["results"]) == 2

        # Verify stats
        server_stats = server.get_stats()
        assert server_stats["aecp_searches"] == 1

        client_stats = client.get_stats()
        assert client_stats["aecp_queries"] == 1

    def test_scenario3_text_fallback(self):
        """Scenario 3 without calibration: falls back to text."""
        server = AECPMCPServer(
            server_name="kb",
            embedder=MockAdapter(dimensions=768),
        )
        server.add_documents(["doc one", "doc two"])

        client = AECPMCPClient(
            embedder=MockAdapter(dimensions=384),
            agent_id="cli",
        )

        # No calibration — should fall back to text
        result = client.semantic_search("query", server, top_k=1)
        assert result["method"] == "text"
        assert result["fidelity"] == "43%"

        assert server.get_stats()["text_searches"] == 1
        assert client.get_stats()["text_queries"] == 1

    def test_full_pipeline_local_to_mcp(self):
        """Combine local model adapter with MCP server/client."""
        from tests.test_local_adapter import FakeSentenceTransformer

        server_model = FakeSentenceTransformer(dimensions=768)
        client_model = FakeSentenceTransformer(dimensions=384)

        server = AECPMCPServer(
            server_name="local-server",
            embedder=LocalModelAdapter(server_model),
        )
        server.add_documents(["machine learning", "deep learning"])

        client = AECPMCPClient(
            embedder=LocalModelAdapter(client_model),
            agent_id="local-client",
        )

        vocab = ["hello", "world", "test", "data", "model"] * 4
        cal = client.connect_and_calibrate(server, vocabulary=vocab, verbose=False)
        assert cal.success

        result = client.semantic_search("neural networks", server, top_k=1)
        assert result["method"] == "aecp"
        assert len(result["results"]) == 1
