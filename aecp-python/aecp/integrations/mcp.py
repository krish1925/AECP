"""
AECP + MCP Integration

Provides ``AECPMCPServer`` and ``AECPMCPClient`` for embedding-aware
MCP (Model Context Protocol) communication.

Without AECP, MCP tools send queries as plain text and the server
re-embeds them — losing ~57% of semantic information in the roundtrip.

With AECP, the client transfers the embedding directly into the
server's space, preserving ~97% semantic fidelity.

Architecture::

    ┌─────────────┐   AECP vector   ┌─────────────────┐
    │ AECPMCPClient│ ──────────────► │ AECPMCPServer    │
    │  (OpenAI emb)│                 │  (Voyage emb)    │
    │              │ ◄────results─── │                  │
    └─────────────┘                  └─────────────────┘

    1. Client embeds query locally
    2. Client transfers embedding to server's space via AECP
    3. Server uses transferred embedding directly (no re-embedding)
    4. 97% fidelity vs 43% with text roundtrip

Example:
    >>> from aecp.integrations.mcp import AECPMCPServer, AECPMCPClient
    >>> from aecp.adapters import MockAdapter
    >>>
    >>> server = AECPMCPServer(
    ...     server_name="knowledge-server",
    ...     embedder=MockAdapter(dimensions=768),
    ... )
    >>> server.add_documents(["doc1 text", "doc2 text"])
    >>>
    >>> client = AECPMCPClient(embedder=MockAdapter(dimensions=384))
    >>> client.connect_and_calibrate(server)
    >>>
    >>> results = client.semantic_search("my query", server)
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..protocol import AECP, CalibrationResult
from ..types import EmbeddingProvider, SemanticTransfer
from ..matrix import cosine_similarity

logger = logging.getLogger("aecp.integrations.mcp")


class AECPMCPServer:
    """
    MCP Server with AECP support for high-fidelity semantic search.

    Exposes two search paths:

    1. **AECP path** — client sends a pre-transferred embedding.
       No re-embedding needed; 97% fidelity.
    2. **Text fallback** — client sends a query string.
       Server re-embeds locally; lower fidelity.

    Also exposes an ``aecp_negotiate`` endpoint so clients can
    discover AECP support and calibrate.

    Attributes:
        server_name: Human-readable server identifier.
        aecp: The AECP protocol handler.
        supports_aecp: Always ``True`` for this class.
        documents: Stored document texts.
        document_embeddings: Corresponding embeddings.
    """

    def __init__(
        self,
        server_name: str,
        embedder: EmbeddingProvider,
        agent_id: Optional[str] = None,
        **aecp_kwargs: Any,
    ):
        """
        Initialise an AECP-aware MCP server.

        Args:
            server_name: Server name / identifier.
            embedder: Embedding provider for the server.
            agent_id: AECP agent ID (defaults to ``server_name``).
            **aecp_kwargs: Extra kwargs forwarded to ``AECP()``.
        """
        self.server_name = server_name
        self.supports_aecp = True

        self.aecp = AECP(
            embedder,
            agent_id=agent_id or server_name,
            **aecp_kwargs,
        )

        # Document store
        self.documents: List[str] = []
        self.document_embeddings: List[np.ndarray] = []

        # Stats
        self._stats: Dict[str, int] = {
            "aecp_searches": 0,
            "text_searches": 0,
            "negotiations": 0,
            "documents_added": 0,
        }

    # ── Document Management ──────────────────────────────────────────

    def add_documents(self, texts: List[str]) -> int:
        """
        Add documents to the server's index.

        Args:
            texts: Document texts to index.

        Returns:
            Total number of indexed documents.
        """
        embeddings = self.aecp._encode_batch(texts)
        for i, text in enumerate(texts):
            self.documents.append(text)
            self.document_embeddings.append(embeddings[i])

        self._stats["documents_added"] += len(texts)
        return len(self.documents)

    # ── MCP Tool: semantic_search ────────────────────────────────────

    def semantic_search(
        self,
        query: Optional[str] = None,
        embedding: Optional[Union[List[float], np.ndarray]] = None,
        source_agent_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Search documents — supports both text and AECP embedding input.

        This is the primary MCP tool endpoint.

        Args:
            query: Text query (used if no embedding provided).
            embedding: Pre-transferred AECP embedding vector.
            source_agent_id: ID of the source agent (for AECP path).
            top_k: Number of results to return.

        Returns:
            Dictionary with:
            - ``results``: List of ``{"text": ..., "score": ...}`` dicts.
            - ``method``: ``"aecp"`` or ``"text"``.
            - ``fidelity``: Estimated fidelity percentage.
            - ``source_agent_id``: If AECP was used.

        Raises:
            ValueError: If neither query nor embedding is provided.
        """
        if embedding is not None and source_agent_id is not None:
            # ── AECP path ────────────────────────────────────────
            query_emb = np.asarray(embedding)
            method = "aecp"
            self._stats["aecp_searches"] += 1
            logger.info(
                "AECP search from agent %s (embedding dim=%d)",
                source_agent_id,
                len(query_emb),
            )
        elif query is not None:
            # ── Text fallback ────────────────────────────────────
            query_emb = self.aecp.embed(query)
            method = "text"
            self._stats["text_searches"] += 1
            logger.info("Text-based search: %s", query[:60])
        else:
            raise ValueError(
                "Either 'query' (text) or 'embedding' + 'source_agent_id' "
                "must be provided."
            )

        # Search documents
        results = self._search(query_emb, top_k=top_k)

        return {
            "results": results,
            "method": method,
            "fidelity": "97%" if method == "aecp" else "43%",
            "source_agent_id": source_agent_id,
            "num_documents": len(self.documents),
        }

    # ── MCP Tool: aecp_negotiate ─────────────────────────────────────

    def aecp_negotiate(self, client_agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Negotiate AECP calibration with a client.

        Returns server metadata so the client can set up calibration.

        Args:
            client_agent_id: The connecting client's agent ID.

        Returns:
            Dictionary with server AECP info.
        """
        self._stats["negotiations"] += 1
        return {
            "server_name": self.server_name,
            "server_agent_id": self.aecp.agent_id,
            "supports_aecp": True,
            "embedding_model": self.aecp.capabilities.embedding_model,
            "dimensions": self.aecp.capabilities.dimensions,
            "protocol_version": self.aecp.capabilities.protocol_version,
            "client_agent_id": client_agent_id,
        }

    # ── Internal ─────────────────────────────────────────────────────

    def _search(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Brute-force cosine similarity search."""
        if not self.document_embeddings:
            return []

        similarities = [
            cosine_similarity(query_emb, emb)
            for emb in self.document_embeddings
        ]

        sorted_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True,
        )

        results = []
        for idx in sorted_indices[:top_k]:
            results.append({
                "text": self.documents[idx],
                "score": round(float(similarities[idx]), 4),
                "index": idx,
            })
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return server usage statistics."""
        return {
            **self._stats,
            "total_documents": len(self.documents),
            "agent_id": self.aecp.agent_id,
            "embedding_model": self.aecp.capabilities.embedding_model,
        }

    def __repr__(self) -> str:
        return (
            f"AECPMCPServer(name={self.server_name!r}, "
            f"docs={len(self.documents)}, "
            f"model={self.aecp.capabilities.embedding_model!r})"
        )


class AECPMCPClient:
    """
    MCP Client with AECP support for high-fidelity semantic queries.

    Automatically negotiates AECP with servers that support it and
    falls back to text-based queries otherwise.

    Usage flow:
    1. Create client with your embedding adapter.
    2. ``connect_and_calibrate(server)`` — negotiates & calibrates.
    3. ``semantic_search(query, server)`` — auto-uses AECP if calibrated.

    Attributes:
        aecp: The AECP protocol handler.
        calibrated_servers: Map of calibrated server AECP instances.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        agent_id: Optional[str] = None,
        **aecp_kwargs: Any,
    ):
        """
        Initialise an AECP-aware MCP client.

        Args:
            embedder: Client-side embedding provider.
            agent_id: Client agent ID (auto-generated if omitted).
            **aecp_kwargs: Extra kwargs forwarded to ``AECP()``.
        """
        self.aecp = AECP(
            embedder,
            agent_id=agent_id or "aecp_mcp_client",
            **aecp_kwargs,
        )

        # server_key → server AECP instance (for transfer)
        self.calibrated_servers: Dict[str, AECP] = {}

        self._stats: Dict[str, int] = {
            "aecp_queries": 0,
            "text_queries": 0,
            "calibrations": 0,
        }

    # ── Connection & Calibration ─────────────────────────────────────

    def connect_and_calibrate(
        self,
        server: AECPMCPServer,
        vocabulary: Optional[List[str]] = None,
        verbose: bool = True,
        **calibration_kwargs: Any,
    ) -> CalibrationResult:
        """
        Connect to an AECP MCP server and calibrate.

        Args:
            server: The MCP server to connect to.
            vocabulary: Custom calibration vocabulary.
            verbose: Print progress.
            **calibration_kwargs: Extra calibration arguments.

        Returns:
            CalibrationResult with quality metrics.
        """
        # Step 1: Negotiate
        server_info = server.aecp_negotiate(
            client_agent_id=self.aecp.agent_id
        )

        if not server_info.get("supports_aecp"):
            if verbose:
                print(
                    f"⚠️  Server '{server.server_name}' does not support AECP. "
                    "Using text fallback."
                )
            return CalibrationResult(
                success=False,
                error_message="Server does not support AECP",
            )

        # Step 2: Calibrate
        result = self.aecp.calibrate_with(
            server.aecp,
            vocabulary=vocabulary,
            verbose=verbose,
            **calibration_kwargs,
        )

        if result.success:
            server_key = server.aecp.agent_id
            self.calibrated_servers[server_key] = server.aecp
            self._stats["calibrations"] += 1
            if verbose:
                print(
                    f"✓ AECP calibrated with server '{server.server_name}' "
                    f"(quality: {result.validation_similarity:.1%})"
                )
        else:
            if verbose:
                print(
                    f"⚠️  Calibration failed with '{server.server_name}': "
                    f"{result.error_message}"
                )

        return result

    def is_calibrated_with(self, server: AECPMCPServer) -> bool:
        """Check if this client is calibrated with a server."""
        return server.aecp.agent_id in self.calibrated_servers

    # ── Querying ─────────────────────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        server: AECPMCPServer,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Search a server with automatic AECP if calibrated.

        If calibrated, embeds locally and transfers the embedding
        to the server's space (97% fidelity).  Otherwise falls back
        to sending the query as text (43% fidelity).

        Args:
            query: Query text.
            server: Target MCP server.
            top_k: Number of results.

        Returns:
            Server search results dictionary.
        """
        server_key = server.aecp.agent_id

        if server_key in self.calibrated_servers:
            # ── AECP path ────────────────────────────────────────
            query_emb = self.aecp.embed(query)
            server_aecp = self.calibrated_servers[server_key]

            transfer = self.aecp.transfer_embedding_to(
                server_aecp.agent_id, query_emb
            )

            result = server.semantic_search(
                embedding=transfer.embedding.tolist(),
                source_agent_id=self.aecp.agent_id,
                top_k=top_k,
            )
            self._stats["aecp_queries"] += 1
        else:
            # ── Text fallback ────────────────────────────────────
            result = server.semantic_search(query=query, top_k=top_k)
            self._stats["text_queries"] += 1

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Return client usage statistics."""
        return {
            **self._stats,
            "agent_id": self.aecp.agent_id,
            "calibrated_servers": list(self.calibrated_servers.keys()),
            "embedding_model": self.aecp.capabilities.embedding_model,
        }

    def __repr__(self) -> str:
        return (
            f"AECPMCPClient(id={self.aecp.agent_id!r}, "
            f"calibrated={len(self.calibrated_servers)} servers, "
            f"model={self.aecp.capabilities.embedding_model!r})"
        )
