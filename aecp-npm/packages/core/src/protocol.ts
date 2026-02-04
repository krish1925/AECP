/**
 * AECP Protocol Implementation
 */

import { randomBytes } from 'crypto';
import {
  AECPConfig,
  AgentCapabilities,
  CalibrationConfig,
  CalibrationResult,
  QualityMetrics,
  SemanticTransfer,
  TransferMatrix,
} from './types';
import {
  cosineSimilarity,
  computeTransferMatrices,
  vectorMatrixMultiply,
} from './matrix';
import { DEFAULT_VOCABULARY, generateExtendedVocabulary } from './vocabulary';

export class AECP {
  private embedder: AECPConfig['embedder'];
  private agentId: string;
  private minQualityThreshold: number;
  private maxBatchSize: number;
  private transferMatrices: Map<string, TransferMatrix> = new Map();
  private calibrationCache: Map<string, number[][]> = new Map();

  constructor(config: AECPConfig) {
    this.embedder = config.embedder;
    this.agentId = config.agentId || this.generateAgentId();
    this.minQualityThreshold = config.minQualityThreshold || 0.75;
    this.maxBatchSize = config.maxBatchSize || 1000;
  }

  /**
   * Get agent capabilities
   */
  getCapabilities(): AgentCapabilities {
    return {
      agentId: this.agentId,
      embeddingModel: {
        name: this.embedder.getModelId(),
        dimensions: this.embedder.getDimensions(),
      },
      protocolVersion: '1.0',
      maxBatchSize: this.maxBatchSize,
      minQualityThreshold: this.minQualityThreshold,
    };
  }

  /**
   * Calibrate with another AECP agent
   */
  async calibrateWith(
    otherAgent: AECP,
    config: CalibrationConfig = {}
  ): Promise<CalibrationResult> {
    const startTime = Date.now();

    // Generate or use provided vocabulary
    const vocabularySize = config.vocabularySize || 1000;
    const validationSize = config.validationSize || 100;
    const qualityThreshold = config.qualityThreshold || this.minQualityThreshold;

    const vocabulary = config.customVocabulary || 
      generateExtendedVocabulary(vocabularySize + validationSize);

    // Split into training and validation
    const trainVocab = vocabulary.slice(0, vocabularySize);
    const valVocab = vocabulary.slice(vocabularySize, vocabularySize + validationSize);

    console.log(`[AECP] Calibrating ${this.agentId} <-> ${otherAgent.agentId}`);
    console.log(`[AECP] Training: ${trainVocab.length} items, Validation: ${valVocab.length} items`);

    // Encode training vocabulary
    console.log('[AECP] Encoding training vocabulary...');
    const embA = await this.embedder.embedBatch(trainVocab);
    const embB = await otherAgent.embedder.embedBatch(trainVocab);

    // Compute transfer matrices
    console.log('[AECP] Computing transfer matrices...');
    const { forward, backward } = computeTransferMatrices(embA, embB);

    // Validate on held-out data
    console.log('[AECP] Validating on held-out data...');
    const valEmbA = await this.embedder.embedBatch(valVocab);
    const valEmbB = await otherAgent.embedder.embedBatch(valVocab);

    // Round-trip validation: A -> B -> A
    const roundtripSimilarities: number[] = [];
    for (let i = 0; i < valEmbA.length; i++) {
      const transferred = vectorMatrixMultiply(valEmbA[i], forward);
      const roundtrip = vectorMatrixMultiply(transferred, backward);
      const similarity = cosineSimilarity(valEmbA[i], roundtrip);
      roundtripSimilarities.push(similarity);
    }

    const qualityMetrics: QualityMetrics = {
      meanSimilarity: this.mean(roundtripSimilarities),
      medianSimilarity: this.median(roundtripSimilarities),
      stdSimilarity: this.std(roundtripSimilarities),
      minSimilarity: Math.min(...roundtripSimilarities),
      maxSimilarity: Math.max(...roundtripSimilarities),
    };

    console.log(`[AECP] Validation similarity: ${qualityMetrics.meanSimilarity.toFixed(4)}`);
    console.log(`[AECP] Worst-case similarity: ${qualityMetrics.minSimilarity.toFixed(4)}`);

    // Check quality threshold
    const success = qualityMetrics.meanSimilarity >= qualityThreshold;
    if (!success) {
      console.warn(
        `[AECP] Quality below threshold: ${qualityMetrics.meanSimilarity.toFixed(4)} < ${qualityThreshold}`
      );
    }

    // Store transfer matrices
    const validUntil = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString();
    const transferMatrix: TransferMatrix = {
      matrixAB: forward,
      matrixBA: backward,
      trainingsimilarity: 0, // Not computed for efficiency
      validationSimilarity: qualityMetrics.meanSimilarity,
      worstCaseSimilarity: qualityMetrics.minSimilarity,
      validUntil,
    };

    this.transferMatrices.set(otherAgent.agentId, transferMatrix);
    
    // Store reverse for other agent
    const reverseMatrix: TransferMatrix = {
      matrixAB: backward,
      matrixBA: forward,
      trainingsimilarity: 0,
      validationSimilarity: qualityMetrics.meanSimilarity,
      worstCaseSimilarity: qualityMetrics.minSimilarity,
      validUntil,
    };
    otherAgent.transferMatrices.set(this.agentId, reverseMatrix);

    const calibrationTime = Date.now() - startTime;
    console.log(`[AECP] Calibration complete in ${calibrationTime}ms`);

    return {
      success,
      transferMatrix,
      qualityMetrics,
      calibrationTime,
    };
  }

  /**
   * Embed text using this agent's embedder
   */
  async embed(text: string): Promise<number[]> {
    return this.embedder.embed(text);
  }

  /**
   * Embed multiple texts (batch)
   */
  async embedBatch(texts: string[]): Promise<number[][]> {
    return this.embedder.embedBatch(texts);
  }

  /**
   * Transfer embedding to another agent's space
   */
  async transferTo(targetAgent: AECP, embedding: number[]): Promise<SemanticTransfer> {
    const matrix = this.transferMatrices.get(targetAgent.agentId);
    if (!matrix) {
      throw new Error(
        `No calibration found for ${targetAgent.agentId}. Call calibrateWith() first.`
      );
    }

    // Check if recalibration needed
    if (new Date(matrix.validUntil) < new Date()) {
      throw new Error('Transfer matrix expired. Recalibration required.');
    }

    // Transform embedding
    const transferred = vectorMatrixMultiply(embedding, matrix.matrixAB);

    return {
      transferId: this.generateTransferId(),
      embedding: transferred,
      sourceAgent: this.agentId,
      targetAgent: targetAgent.agentId,
      expectedSimilarity: matrix.validationSimilarity,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Find similar embeddings in a knowledge base
   */
  async findSimilar(
    queryEmbedding: number[],
    knowledgeBase: number[][],
    topK: number = 5
  ): Promise<Array<{ index: number; similarity: number }>> {
    const similarities = knowledgeBase.map((emb, index) => ({
      index,
      similarity: cosineSimilarity(queryEmbedding, emb),
    }));

    return similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);
  }

  /**
   * Get quality score for a specific agent connection
   */
  getQualityScore(targetAgentId: string): number | null {
    const matrix = this.transferMatrices.get(targetAgentId);
    return matrix ? matrix.validationSimilarity : null;
  }

  /**
   * Check if recalibration is needed
   */
  requiresRecalibration(targetAgentId: string): boolean {
    const matrix = this.transferMatrices.get(targetAgentId);
    if (!matrix) return true;

    // Check expiration
    if (new Date(matrix.validUntil) < new Date()) return true;

    // Check quality
    if (matrix.validationSimilarity < this.minQualityThreshold) return true;

    return false;
  }

  // Utility methods
  private generateAgentId(): string {
    return `agent_${randomBytes(8).toString('hex')}`;
  }

  private generateTransferId(): string {
    return randomBytes(16).toString('hex');
  }

  private mean(arr: number[]): number {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  private median(arr: number[]): number {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }

  private std(arr: number[]): number {
    const avg = this.mean(arr);
    const squareDiffs = arr.map((value) => Math.pow(value - avg, 2));
    return Math.sqrt(this.mean(squareDiffs));
  }
}
