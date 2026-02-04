# AECP NPM Package - Project Overview

## Mission

Enable seamless semantic communication between AI agents using different embedding models, achieving 2x better information preservation compared to text serialization.

## Key Metrics

- **Quality**: 97% semantic fidelity (validated on 300k vocabulary)
- **Performance**: < 1ms transfer latency
- **Improvement**: 2x better than text baseline (0.86 vs 0.43 similarity)
- **Production-Ready**: Zero overfitting, strong generalization

## Architecture

```
┌─────────────────────────────────────────┐
│   AECP Protocol Layer (@aecp/core)      │
│  - Handshake negotiation                │
│  - Calibration (least squares)          │
│  - Transfer matrices (W_AB, W_BA)       │
│  - Quality monitoring                   │
└─────────────────────────────────────────┘
              ↓ ↑
┌─────────────────────────────────────────┐
│   Adapter Layer (provider-specific)     │
│  - OpenAI adapter                       │
│  - Voyage adapter                       │
│  - Cohere adapter                       │
│  - HuggingFace adapter                  │
└─────────────────────────────────────────┘
              ↓ ↑
┌─────────────────────────────────────────┐
│   Provider APIs                         │
│  - OpenAI embeddings API                │
│  - Voyage AI                            │
│  - Cohere                               │
│  - HuggingFace (local)                  │
└─────────────────────────────────────────┘
```

## Package Structure

```
aecp-npm/
├── packages/
│   ├── core/                    # Core protocol (no dependencies)
│   │   ├── src/
│   │   │   ├── types.ts         # TypeScript interfaces
│   │   │   ├── matrix.ts        # Matrix operations
│   │   │   ├── vocabulary.ts    # Default calibration data
│   │   │   ├── protocol.ts      # Main AECP class
│   │   │   └── index.ts         # Public exports
│   │   ├── package.json
│   │   └── tsconfig.json
│   │
│   ├── adapters-openai/         # OpenAI adapter
│   │   ├── src/index.ts
│   │   └── package.json
│   │
│   ├── adapters-voyage/         # Voyage AI adapter
│   │   ├── src/index.ts
│   │   └── package.json
│   │
│   ├── adapters-cohere/         # Cohere adapter
│   │   ├── src/index.ts
│   │   └── package.json
│   │
│   └── adapters-huggingface/    # HuggingFace adapter
│       ├── src/index.ts
│       └── package.json
│
├── examples/
│   ├── basic-transfer/          # Simple transfer example
│   ├── multi-agent-chat/        # Multi-agent communication
│   └── custom-adapter/          # Custom adapter example
│
├── docs/
│   ├── getting-started.md       # Quick start guide
│   ├── api-reference.md         # Complete API docs
│   └── protocol-spec.md         # Protocol specification
│
├── package.json                 # Root package (monorepo)
├── tsconfig.json                # TypeScript config
├── README.md                    # Main documentation
├── SETUP.md                     # Setup instructions
├── CHANGELOG.md                 # Version history
├── CONTRIBUTING.md              # Contribution guide
└── LICENSE                      # MIT License
```

## Technical Implementation

### Core Algorithm

1. **Calibration Phase**:
   ```
   Input: Vocabulary V (N items)
   
   1. Encode with both models:
      E_A = embed_A(V)  // [N × D_A]
      E_B = embed_B(V)  // [N × D_B]
   
   2. Solve least squares:
      W_AB = argmin ||E_A @ W - E_B||²  // [D_A × D_B]
      W_BA = argmin ||E_B @ W - E_A||²  // [D_B × D_A]
   
   3. Validate on held-out set:
      quality = mean(cosine_similarity(E_A, (E_A @ W_AB) @ W_BA))
   ```

2. **Transfer Phase**:
   ```
   Input: Embedding e_A from agent A
   
   1. Transform to B's space:
      e_B = e_A @ W_AB
   
   2. Agent B uses e_B natively
   ```

### Key Features

- **Linear Transformation**: Simple matrix multiplication (no neural networks)
- **Least Squares**: Optimal linear mapping between spaces
- **Round-Trip Validation**: Quality measured on A → B → A
- **Generalization**: Validated on completely unseen data

## Design Principles

1. **Minimalism**: Every line of code serves a purpose
2. **Efficiency**: Optimized for production use
3. **Simplicity**: Clear, readable implementation
4. **Extensibility**: Easy to add new adapters
5. **Type Safety**: Full TypeScript support

## Usage Patterns

### Pattern 1: Same Provider, Different Models

```typescript
// OpenAI small → OpenAI large
const agent1 = new AECP({
  embedder: new OpenAIAdapter({ model: 'text-embedding-3-small' })
});

const agent2 = new AECP({
  embedder: new OpenAIAdapter({ model: 'text-embedding-3-large' })
});

await agent1.calibrateWith(agent2);
```

### Pattern 2: Different Providers

```typescript
// OpenAI → Voyage
const agent1 = new AECP({
  embedder: new OpenAIAdapter({ model: 'text-embedding-3-small' })
});

const agent2 = new AECP({
  embedder: new VoyageAdapter({ model: 'voyage-2' })
});

await agent1.calibrateWith(agent2);
```

### Pattern 3: Cloud → Local

```typescript
// OpenAI → HuggingFace (local)
const agent1 = new AECP({
  embedder: new OpenAIAdapter({ model: 'text-embedding-3-small' })
});

const agent2 = new AECP({
  embedder: new HuggingFaceAdapter({ model: 'Xenova/all-MiniLM-L6-v2' })
});

await agent1.calibrateWith(agent2);
```

## Performance Characteristics

| Operation | Complexity | Typical Time |
|-----------|-----------|--------------|
| Calibration | O(N × D²) | 2-5s (1000 items) |
| Transfer | O(D²) | < 1ms |
| Find Similar | O(K × D) | < 10ms (K=1000) |

| Metric | Value |
|--------|-------|
| Memory (matrices) | ~10MB (1536D × 1536D) |
| Memory (embeddings) | ~6MB (1000 × 1536D) |
| Calibration quality | 0.80-0.97 |
| Transfer quality | 0.75-0.95 |

## Security Considerations

1. **API Keys**: Store securely, never commit
2. **Vocabulary**: Don't include sensitive data
3. **Transfer Matrices**: Can reveal model relationships
4. **Embeddings**: May leak semantic information

## Real-World Applications

1. **Multi-Agent Systems**: Agents with different models communicate seamlessly
2. **Model Migration**: Transfer embeddings when upgrading models
3. **Hybrid Systems**: Combine cloud and local models
4. **Cost Optimization**: Use cheaper models with transfer to expensive ones
5. **Privacy**: Keep sensitive data local, transfer to cloud

## Validation Results

From Python proof-of-concept:

| Test | Vocabulary | Quality | Status |
|------|-----------|---------|--------|
| Training | 240k items | 0.9586 | Pass |
| Validation | 30k items | 0.9734 | Pass |
| Unseen Vocab | 30k items | 0.9735 | Pass |
| Unseen Corpus | 1k sentences | 0.8642 | Pass |
| Text Baseline | 1k sentences | 0.4306 | Baseline |

**Key Finding**: 2x improvement over text (0.86 vs 0.43)

## Roadmap

### v1.0 (Current)
- Core protocol
- 4 adapters (OpenAI, Voyage, Cohere, HuggingFace)
- Quality monitoring
- Comprehensive docs

### v1.1 (Planned)
- [ ] Matrix compression (8-bit quantization)
- [ ] Batch transfer optimization
- [ ] WebAssembly acceleration
- [ ] Additional adapters (Anthropic, Azure)

### v1.2 (Future)
- [ ] Incremental calibration
- [ ] Multi-hop transfer (A → B → C)
- [ ] Neural transfer functions
- [ ] Adaptive vocabulary selection

### v2.0 (Research)
- [ ] Non-linear transformations
- [ ] Domain adaptation
- [ ] Real-time quality prediction
- [ ] Adversarial robustness

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guide
- Adding new adapters
- Testing guidelines
- Pull request process

## Resources

- **Research**: [ENHANCED_SUMMARY.md](../ENHANCED_SUMMARY.md)
- **Python POC**: [protocol.py](../protocol.py)
- **Validation**: [reports/ENHANCED_REPORT.md](../reports/ENHANCED_REPORT.md)
- **Specification**: [protocol_spec.md](../protocol_spec.md)

## Support

- **Documentation**: [docs/](./docs/)
- **Examples**: [examples/](./examples/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

Based on research validating the Agent Embedding Communication Protocol (AECP):
- Validated on 300k vocabulary
- 97% semantic fidelity
- Zero overfitting
- Production-ready

---

**Status**: Production Ready  
**Version**: 1.0.0  
**Last Updated**: 2026-02-04
