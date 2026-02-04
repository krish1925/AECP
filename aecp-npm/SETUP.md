# AECP Setup Guide

Complete setup instructions for the AECP NPM package.

## Prerequisites

- Node.js >= 18.0.0
- npm >= 9.0.0
- TypeScript >= 5.3.0 (for development)

## Installation

### For Users

Install the packages you need:

```bash
# Core + OpenAI
npm install @aecp/core @aecp/adapters-openai

# Core + Voyage
npm install @aecp/core @aecp/adapters-voyage

# Core + Cohere
npm install @aecp/core @aecp/adapters-cohere

# Core + HuggingFace (local)
npm install @aecp/core @aecp/adapters-huggingface

# All adapters
npm install @aecp/core @aecp/adapters-openai @aecp/adapters-voyage @aecp/adapters-cohere @aecp/adapters-huggingface
```

### For Development

Clone and build from source:

```bash
# Clone repository
git clone https://github.com/your-org/aecp.git
cd aecp/aecp-npm

# Install dependencies
npm install

# Build all packages
npm run build

# Link for local development
cd packages/core
npm link

cd ../adapters-openai
npm link @aecp/core
npm link
```

## Building

### Build All Packages

```bash
npm run build
```

This compiles TypeScript to JavaScript in each package's `dist/` directory.

### Build Individual Package

```bash
cd packages/core
npm run build
```

### Clean Build

```bash
npm run clean
npm run build
```

## Testing

### Quick Test

Create a test file `test.ts`:

```typescript
import { AECP } from '@aecp/core';
import { OpenAIAdapter } from '@aecp/adapters-openai';

async function test() {
  const agent = new AECP({
    embedder: new OpenAIAdapter({
      apiKey: process.env.OPENAI_API_KEY!,
      model: 'text-embedding-3-small'
    })
  });

  const embedding = await agent.embed("Hello world");
  console.log(`Embedding dimensions: ${embedding.length}`);
  console.log('✓ AECP working!');
}

test().catch(console.error);
```

Run with:

```bash
npx tsx test.ts
```

### Run Examples

```bash
# Basic transfer
cd examples/basic-transfer
npm install
OPENAI_API_KEY=sk-... npm start

# Multi-agent chat
cd examples/multi-agent-chat
npm install
OPENAI_API_KEY=sk-... npm start

# Custom adapter
cd examples/custom-adapter
npm install
npm start
```

## Publishing

### Pre-publish Checklist

- [ ] All tests passing
- [ ] Version numbers updated
- [ ] CHANGELOG.md updated
- [ ] Documentation reviewed
- [ ] Examples working
- [ ] No TypeScript errors

### Publish to NPM

```bash
# Login to NPM
npm login

# Publish core package
cd packages/core
npm publish --access public

# Publish adapters
cd ../adapters-openai
npm publish --access public

cd ../adapters-voyage
npm publish --access public

cd ../adapters-cohere
npm publish --access public

cd ../adapters-huggingface
npm publish --access public
```

### Version Bumping

```bash
# Patch version (1.0.0 -> 1.0.1)
npm version patch

# Minor version (1.0.0 -> 1.1.0)
npm version minor

# Major version (1.0.0 -> 2.0.0)
npm version major
```

## Environment Variables

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_ORG_ID=org-...  # Optional
```

### Voyage

```bash
export VOYAGE_API_KEY=pa-...
```

### Cohere

```bash
export COHERE_API_KEY=...
```

### HuggingFace

No API key required (runs locally).

## Troubleshooting

### TypeScript Errors

```bash
# Clean and rebuild
npm run clean
npm install
npm run build
```

### Module Not Found

```bash
# Ensure all packages are built
npm run build

# Check symlinks (for development)
npm link @aecp/core
```

### Import Errors

Make sure you're importing from the correct package:

```typescript
// ✓ Correct
import { AECP } from '@aecp/core';
import { OpenAIAdapter } from '@aecp/adapters-openai';

// ✗ Incorrect
import { AECP } from 'aecp';
import { OpenAIAdapter } from '@aecp/core';
```

### API Key Issues

```bash
# Check environment variable
echo $OPENAI_API_KEY

# Set for current session
export OPENAI_API_KEY=sk-...

# Or use .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

### HuggingFace Model Download

First run downloads models (can take time):

```bash
# Models cached in ~/.cache/huggingface/
ls ~/.cache/huggingface/

# Clear cache if needed
rm -rf ~/.cache/huggingface/
```

## IDE Setup

### VS Code

Recommended extensions:
- ESLint
- TypeScript
- Prettier

`.vscode/settings.json`:
```json
{
  "typescript.tsdk": "node_modules/typescript/lib",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  }
}
```

### WebStorm

1. Enable TypeScript support
2. Set Node.js interpreter
3. Enable ESLint
4. Set code style to match project

## Performance Optimization

### Calibration

```typescript
// Use smaller vocabulary for testing
await agent1.calibrateWith(agent2, {
  vocabularySize: 200,
  validationSize: 20
});

// Use larger vocabulary for production
await agent1.calibrateWith(agent2, {
  vocabularySize: 5000,
  validationSize: 500
});
```

### Caching

```typescript
// Save calibration
const matrix = agent1.transferMatrices.get(agent2.agentId);
fs.writeFileSync('calibration.json', JSON.stringify(matrix));

// Load calibration
const saved = JSON.parse(fs.readFileSync('calibration.json'));
agent1.transferMatrices.set(agent2.agentId, saved);
```

### Batch Processing

```typescript
// Batch embeddings
const embeddings = await agent.embedBatch(texts);

// Batch transfers
const transfers = await Promise.all(
  embeddings.map(emb => agent.transferTo(partner, emb))
);
```

## Production Deployment

### Docker

```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --production

COPY dist/ ./dist/

CMD ["node", "dist/index.js"]
```

### Environment

```bash
# Production environment variables
NODE_ENV=production
OPENAI_API_KEY=sk-...
AECP_QUALITY_THRESHOLD=0.85
AECP_CALIBRATION_SIZE=5000
```

### Monitoring

```typescript
// Log quality metrics
const quality = agent.getQualityScore(partnerId);
console.log(`Quality: ${quality}`);

// Alert on low quality
if (quality && quality < 0.80) {
  alerting.send('AECP quality degraded');
}

// Recalibrate periodically
setInterval(async () => {
  if (agent.requiresRecalibration(partnerId)) {
    await agent.calibrateWith(partner);
  }
}, 24 * 60 * 60 * 1000); // Daily
```

## Support

- Documentation: [docs/](./docs/)
- Examples: [examples/](./examples/)
- Issues: GitHub Issues
- Discussions: GitHub Discussions

## Next Steps

1. Read [Getting Started Guide](./docs/getting-started.md)
2. Review [API Reference](./docs/api-reference.md)
3. Try [Examples](./examples/)
4. Read [Protocol Specification](./docs/protocol-spec.md)
