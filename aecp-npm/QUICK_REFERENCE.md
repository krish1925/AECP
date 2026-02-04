# AECP Quick Reference

One-page reference for the AECP NPM package.

## Installation

```bash
npm install @aecp/core @aecp/adapters-openai
```

## Basic Usage

```typescript
import { AECP } from '@aecp/core';
import { OpenAIAdapter } from '@aecp/adapters-openai';

// Create agent
const agent = new AECP({
  embedder: new OpenAIAdapter({ apiKey: 'sk-...' })
});

// Embed text
const embedding = await agent.embed("text");

// Calibrate with another agent
await agent1.calibrateWith(agent2);

// Transfer embedding
const transfer = await agent1.transferTo(agent2, embedding);

// Find similar
const results = await agent.findSimilar(embedding, knowledgeBase, 5);
```

## API Cheatsheet

### AECP Class

| Method | Purpose | Returns |
|--------|---------|---------|
| `new AECP(config)` | Create agent | AECP |
| `getCapabilities()` | Get agent info | AgentCapabilities |
| `embed(text)` | Embed single text | Promise\<number[]\> |
| `embedBatch(texts)` | Embed multiple | Promise\<number[][]\> |
| `calibrateWith(agent, config?)` | Calibrate | Promise\<CalibrationResult\> |
| `transferTo(agent, embedding)` | Transfer | Promise\<SemanticTransfer\> |
| `findSimilar(emb, kb, k?)` | Search | Promise\<Match[]\> |
| `getQualityScore(agentId)` | Get quality | number \| null |
| `requiresRecalibration(agentId)` | Check status | boolean |

### Config Objects

```typescript
// AECP Config
{
  embedder: EmbeddingProvider;
  agentId?: string;
  minQualityThreshold?: number;  // default: 0.75
  maxBatchSize?: number;         // default: 1000
}

// Calibration Config
{
  vocabularySize?: number;       // default: 1000
  validationSize?: number;       // default: 100
  qualityThreshold?: number;     // default: 0.75
  customVocabulary?: string[];
}
```

## Adapters

### OpenAI

```typescript
import { OpenAIAdapter } from '@aecp/adapters-openai';

new OpenAIAdapter({
  apiKey: string;
  model?: 'text-embedding-3-small' | 'text-embedding-3-large';
  organization?: string;
})
```

### Voyage

```typescript
import { VoyageAdapter } from '@aecp/adapters-voyage';

new VoyageAdapter({
  apiKey: string;
  model?: 'voyage-2' | 'voyage-large-2';
})
```

### Cohere

```typescript
import { CohereAdapter } from '@aecp/adapters-cohere';

new CohereAdapter({
  apiKey: string;
  model?: 'embed-english-v3.0' | 'embed-multilingual-v3.0';
})
```

### HuggingFace

```typescript
import { HuggingFaceAdapter } from '@aecp/adapters-huggingface';

new HuggingFaceAdapter({
  model?: 'Xenova/all-MiniLM-L6-v2';
  quantized?: boolean;  // default: true
})
```

## Custom Adapter

```typescript
import { EmbeddingProvider } from '@aecp/core';

class MyAdapter implements EmbeddingProvider {
  async embed(text: string): Promise<number[]> { /* ... */ }
  async embedBatch(texts: string[]): Promise<number[][]> { /* ... */ }
  getDimensions(): number { /* ... */ }
  getModelId(): string { /* ... */ }
}
```

## Quality Metrics

| Score | Quality | Action |
|-------|---------|--------|
| > 0.90 | Excellent | Use confidently |
| 0.80-0.90 | Good | Production ready |
| 0.75-0.80 | Acceptable | Monitor closely |
| < 0.75 | Poor | Recalibrate |

## Common Patterns

### Pattern: Check Before Transfer

```typescript
if (agent1.requiresRecalibration(agent2.agentId)) {
  await agent1.calibrateWith(agent2);
}
const transfer = await agent1.transferTo(agent2, embedding);
```

### Pattern: Quality Monitoring

```typescript
const quality = agent1.getQualityScore(agent2.agentId);
if (quality && quality < 0.80) {
  console.warn('Low quality, recalibrating...');
  await agent1.calibrateWith(agent2);
}
```

### Pattern: Batch Processing

```typescript
const embeddings = await agent1.embedBatch(texts);
const transfers = await Promise.all(
  embeddings.map(emb => agent1.transferTo(agent2, emb))
);
```

### Pattern: Domain-Specific Calibration

```typescript
const medicalVocab = ['diagnosis', 'treatment', 'patient', ...];
await agent1.calibrateWith(agent2, {
  customVocabulary: medicalVocab,
  qualityThreshold: 0.85
});
```

## Error Handling

```typescript
try {
  const transfer = await agent1.transferTo(agent2, embedding);
} catch (error) {
  if (error.message.includes('No calibration')) {
    await agent1.calibrateWith(agent2);
  } else if (error.message.includes('expired')) {
    await agent1.calibrateWith(agent2);
  } else {
    throw error;
  }
}
```

## Performance Tips

1. **Cache calibrations**: Save transfer matrices to disk
2. **Use larger vocabularies**: 5000+ for production
3. **Batch operations**: Use `embedBatch()` when possible
4. **Recalibrate periodically**: Weekly or when quality drops
5. **Monitor quality**: Check scores regularly

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Low quality | Increase vocabulary size |
| Slow calibration | Reduce vocabulary size |
| Memory issues | Use smaller batches |
| API errors | Check API keys and quotas |
| Type errors | Ensure TypeScript >= 5.3 |

## Environment Variables

```bash
export OPENAI_API_KEY=sk-...
export VOYAGE_API_KEY=pa-...
export COHERE_API_KEY=...
```

## File Structure

```
node_modules/@aecp/
├── core/
│   ├── dist/
│   │   ├── index.js
│   │   ├── index.d.ts
│   │   └── ...
│   └── package.json
└── adapters-*/
    └── ...
```

## Resources

- **Docs**: [docs/getting-started.md](docs/getting-started.md)
- **API**: [docs/api-reference.md](docs/api-reference.md)
- **Spec**: [docs/protocol-spec.md](docs/protocol-spec.md)
- **Examples**: [examples/](examples/)

## Support

- GitHub Issues
- GitHub Discussions
- Documentation

---

**Version**: 1.0.0 | **License**: MIT | **Status**: Production Ready
