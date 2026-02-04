# @aecp/adapters-cohere

Cohere embeddings adapter for AECP.

## Installation

```bash
npm install @aecp/core @aecp/adapters-cohere
```

## Usage

```typescript
import { AECP } from '@aecp/core';
import { CohereAdapter } from '@aecp/adapters-cohere';

const agent = new AECP({
  embedder: new CohereAdapter({
    apiKey: process.env.COHERE_API_KEY,
    model: 'embed-english-v3.0',
  })
});
```

## Supported Models

- `embed-english-v3.0` (1024 dimensions)
- `embed-multilingual-v3.0` (1024 dimensions)
- `embed-english-light-v3.0` (384 dimensions)
- `embed-multilingual-light-v3.0` (384 dimensions)

## License

MIT
