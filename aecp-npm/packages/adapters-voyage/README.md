# @aecp/adapters-voyage

Voyage AI embeddings adapter for AECP.

## Installation

```bash
npm install @aecp/core @aecp/adapters-voyage
```

## Usage

```typescript
import { AECP } from '@aecp/core';
import { VoyageAdapter } from '@aecp/adapters-voyage';

const agent = new AECP({
  embedder: new VoyageAdapter({
    apiKey: process.env.VOYAGE_API_KEY,
    model: 'voyage-2',
  })
});
```

## Supported Models

- `voyage-2` (1024 dimensions)
- `voyage-large-2` (1536 dimensions)
- `voyage-code-2` (1536 dimensions)

## License

MIT
