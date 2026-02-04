/**
 * Core type definitions for AECP protocol
 */

export interface EmbeddingProvider {
  /**
   * Generate embedding for text
   */
  embed(text: string): Promise<number[]>;

  /**
   * Generate embeddings for multiple texts (batch)
   */
  embedBatch(texts: string[]): Promise<number[][]>;

  /**
   * Get embedding dimensions
   */
  getDimensions(): number;

  /**
   * Get model identifier
   */
  getModelId(): string;
}

export interface AgentCapabilities {
  agentId: string;
  embeddingModel: {
    name: string;
    dimensions: number;
  };
  protocolVersion: string;
  maxBatchSize: number;
  minQualityThreshold: number;
}

export interface CalibrationConfig {
  vocabularySize?: number;
  validationSize?: number;
  qualityThreshold?: number;
  customVocabulary?: string[];
}

export interface TransferMatrix {
  matrixAB: number[][];
  matrixBA: number[][];
  trainingsimilarity: number;
  validationSimilarity: number;
  worstCaseSimilarity: number;
  validUntil: string;
}

export interface QualityMetrics {
  meanSimilarity: number;
  medianSimilarity: number;
  stdSimilarity: number;
  minSimilarity: number;
  maxSimilarity: number;
}

export interface SemanticTransfer {
  transferId: string;
  embedding: number[];
  sourceAgent: string;
  targetAgent: string;
  expectedSimilarity: number;
  timestamp: string;
}

export interface AECPConfig {
  embedder: EmbeddingProvider;
  agentId?: string;
  minQualityThreshold?: number;
  maxBatchSize?: number;
}

export interface CalibrationResult {
  success: boolean;
  transferMatrix: TransferMatrix;
  qualityMetrics: QualityMetrics;
  calibrationTime: number;
}
