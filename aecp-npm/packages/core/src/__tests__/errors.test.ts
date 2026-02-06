/**
 * Tests for error handling and resilience features
 */

import {
  AECPError,
  CalibrationError,
  TransferError,
  CircuitBreaker,
  CircuitState,
  CircuitOpenError,
  RetryPolicy,
  GracefulDegradation,
} from '../errors';

describe('Error Hierarchy', () => {
  test('AECPError is base class', () => {
    const error = new AECPError('test', true);
    expect(error).toBeInstanceOf(Error);
    expect(error.recoverable).toBe(true);
  });

  test('CalibrationError extends AECPError', () => {
    const error = new CalibrationError('calibration failed');
    expect(error).toBeInstanceOf(AECPError);
    expect(error.recoverable).toBe(true);
  });

  test('TransferError extends AECPError', () => {
    const error = new TransferError('transfer failed', true);
    expect(error).toBeInstanceOf(AECPError);
    expect(error.recoverable).toBe(true);
  });
});

describe('CircuitBreaker', () => {
  test('starts in CLOSED state', () => {
    const cb = new CircuitBreaker();
    expect(cb.state).toBe(CircuitState.CLOSED);
  });

  test('opens after threshold failures', async () => {
    const cb = new CircuitBreaker({ failureThreshold: 2, resetTimeout: 1000 });
    
    // Fail twice
    try {
      await cb.call(() => {
        throw new Error('test');
      });
    } catch {}
    
    try {
      await cb.call(() => {
        throw new Error('test');
      });
    } catch {}
    
    expect(cb.state).toBe(CircuitState.OPEN);
  });

  test('throws CircuitOpenError when open', async () => {
    const cb = new CircuitBreaker({ failureThreshold: 1, resetTimeout: 1000 });
    
    // Trigger failure
    try {
      await cb.call(() => {
        throw new Error('test');
      });
    } catch {}
    
    // Next call should throw CircuitOpenError
    await expect(cb.call(() => 'success')).rejects.toThrow(CircuitOpenError);
  });

  test('resets on success', async () => {
    const cb = new CircuitBreaker({ failureThreshold: 2, resetTimeout: 1000 });
    
    // Fail once
    try {
      await cb.call(() => {
        throw new Error('test');
      });
    } catch {}
    
    // Success should reset
    await cb.call(() => 'success');
    const status = cb.getStatus();
    expect(status.failureCount).toBe(0);
  });

  test('manual reset works', () => {
    const cb = new CircuitBreaker({ failureThreshold: 1, resetTimeout: 1000 });
    cb.reset();
    expect(cb.state).toBe(CircuitState.CLOSED);
    const status = cb.getStatus();
    expect(status.failureCount).toBe(0);
  });
});

describe('RetryPolicy', () => {
  test('retries on failure', async () => {
    let attempts = 0;
    const policy = new RetryPolicy({ maxRetries: 2, baseDelay: 10 });
    
    const result = await policy.execute(() => {
      attempts++;
      if (attempts < 3) {
        throw new Error('retry');
      }
      return 'success';
    });
    
    expect(result).toBe('success');
    expect(attempts).toBe(3);
  });

  test('throws after max retries', async () => {
    const policy = new RetryPolicy({ maxRetries: 2, baseDelay: 10 });
    
    await expect(
      policy.execute(() => {
        throw new Error('always fails');
      })
    ).rejects.toThrow('always fails');
  });
});

describe('GracefulDegradation', () => {
  test('falls back to text when AECP fails', async () => {
    const gd = new GracefulDegradation('en');
    
    const result = await gd.send('Hello', () => {
      throw new Error('AECP failed');
    });
    
    expect(result.method).toBe('text');
    if (result.method === 'text') {
      expect(result.message).toBe('Hello');
      expect(result.language).toBe('en');
      expect(result.fallback).toBe(true);
    }
  });

  test('uses AECP when available', async () => {
    const gd = new GracefulDegradation('en');
    
    const result = await gd.send('Hello', () => ({
      transferId: 'test-123',
      embedding: [1, 2, 3],
    }));
    
    expect(result.method).toBe('aecp');
    if (result.method === 'aecp') {
      expect(result.data).toBeDefined();
    }
  });

  test('tracks stats', async () => {
    const gd = new GracefulDegradation('en');
    
    await gd.send('Hello', () => ({ data: 'aecp' }));
    await gd.send('World', () => {
      throw new Error('fail');
    });
    
    const stats = gd.getStats();
    expect(stats.aecpCount).toBe(1);
    expect(stats.fallbackCount).toBe(1);
    expect(stats.total).toBe(2);
  });
});
