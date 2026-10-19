/**
 * Experience Replay Buffer for preventing catastrophic forgetting.
 * Stores training samples from past cycles and mixes them with new data
 * during training to ensure the network retains previously learned patterns.
 */

export class ReplayBuffer {
  constructor(maxSize = 50000) {
    this.buffer = [];
    this.maxSize = maxSize;
  }

  /**
   * Add training samples to the buffer.
   * Oldest samples are evicted when buffer exceeds maxSize.
   */
  add(samples) {
    this.buffer.push(...samples);
    if (this.buffer.length > this.maxSize) {
      // Keep most recent samples (they reflect current policy)
      this.buffer = this.buffer.slice(-this.maxSize);
    }
  }

  /**
   * Sample n random items from the buffer.
   */
  sample(n) {
    if (n <= 0 || this.buffer.length === 0) return [];
    const count = Math.min(n, this.buffer.length);
    const sampled = [];
    const indices = new Set();
    while (indices.size < count) {
      indices.add(Math.floor(Math.random() * this.buffer.length));
    }
    for (const i of indices) {
      sampled.push(this.buffer[i]);
    }
    return sampled;
  }

  /**
   * Get current buffer size.
   */
  get size() {
    return this.buffer.length;
  }

  /**
   * Check if buffer has enough samples for meaningful replay.
   */
  get isReady() {
    return this.buffer.length >= 1000;
  }
}
