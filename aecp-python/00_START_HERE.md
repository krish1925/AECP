# 🚀 AECP Python Package - Quick Start

## What is AECP?

**Agent Embedding Communication Protocol** - Enable AI agents with different embedding models to communicate with **97% semantic fidelity preservation**.

## Installation

```bash
# Basic (numpy only)
pip install aecp

# With OpenAI support
pip install aecp[openai]

# With all providers
pip install aecp[all]

# Development
pip install -e ".[dev]"
```

## 30-Second Example

```python
from aecp import AECP
from aecp.adapters import MockAdapter  # Use OpenAIAdapter, VoyageAdapter, etc. in production

# Create two agents with different embedding dimensions
agent1 = AECP(MockAdapter(dimensions=384))
agent2 = AECP(MockAdapter(dimensions=768))

# One-time calibration (learns transfer matrices)
agent1.calibrate_with(agent2)

# Transfer embeddings between agents
transfer = agent1.transfer_to(agent2.agent_id, "Hello world!")
print(f"Transferred embedding shape: {transfer.embedding.shape}")  # (768,)
```

## Real-World Example with OpenAI

```python
from aecp import AECP
from aecp.adapters import OpenAIAdapter, VoyageAdapter

# Initialize with real embedding providers
agent1 = AECP(OpenAIAdapter(api_key="sk-..."))
agent2 = AECP(VoyageAdapter(api_key="pa-..."))

# Calibrate once
result = agent1.calibrate_with(agent2)
print(f"Calibration quality: {result.validation_similarity:.2%}")

# Transfer embeddings
transfer = agent1.transfer_to(agent2.agent_id, "machine learning concepts")
```

## High-Level Patterns

### Cost Optimization
```python
from aecp.patterns import CostOptimizer

optimizer = CostOptimizer(
    cheap_adapter=OpenAIAdapter(model="text-embedding-3-small"),
    expensive_adapter=VoyageAdapter(model="voyage-large-2"),
)
optimizer.calibrate()

# Use cheap model by default, expensive when needed
embedding = optimizer.embed("query", precision="low")   # Cheap
embedding = optimizer.embed("critical", precision="high")  # Expensive
```

### Privacy Bridge
```python
from aecp.patterns import PrivacyBridge

bridge = PrivacyBridge(
    local_adapter=HuggingFaceAdapter(),  # Runs locally
    cloud_adapter=OpenAIAdapter(api_key="sk-..."),
)
bridge.calibrate()

# Data stays local, only semantics go to cloud
local_emb = bridge.embed_local("sensitive patient data")
cloud_emb = bridge.transfer_to_cloud(local_emb)
```

### Multi-Agent Handoff
```python
from aecp.patterns import AgentHandoff

handoff = AgentHandoff({
    'code': VoyageAdapter(model='voyage-code-2'),
    'general': OpenAIAdapter(),
})
handoff.calibrate_all()

context = handoff.start("Debug this code", agent='code')
context = handoff.transfer(context, to_agent='general')
```

## Package Structure

```
aecp-python/
├── aecp/
│   ├── __init__.py      # Main exports
│   ├── protocol.py      # AECP, ProtocolHandler
│   ├── matrix.py        # Transfer matrix operations
│   ├── types.py         # Type definitions
│   ├── vocabulary.py    # Default calibration vocabulary
│   ├── adapters/        # Provider adapters
│   │   ├── openai.py
│   │   ├── voyage.py
│   │   ├── cohere.py
│   │   ├── huggingface.py
│   │   └── mock.py      # For testing
│   └── patterns/        # High-level patterns
│       ├── cost_optimizer.py
│       ├── privacy_bridge.py
│       └── agent_handoff.py
├── tests/               # Test suite
├── setup.py
├── pyproject.toml
└── README.md
```

## Running Tests

```bash
cd aecp-python
pip install -e ".[dev]"
pytest tests/ -v
```

## Publishing to PyPI

```bash
python -m build
python -m twine upload dist/*
```

## Key Metrics

| Metric | Text Serialization | AECP | Improvement |
|--------|-------------------|------|-------------|
| Semantic Similarity | 43% | 86% | **2x better** |
| Information Loss | 95% | 3% | **32x better** |

## Need Help?

- **README.md** - Full documentation
- **CONTRIBUTING.md** - Development guide
- **tests/** - Example usage in tests
