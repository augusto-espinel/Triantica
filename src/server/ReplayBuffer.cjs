/**
 * Experience Replay Buffer for Triantica AI Training
 * 
 * Stores game positions and trains on random samples from history.
 * This breaks temporal correlation between consecutive moves and
 * prevents the network from forgetting early games.
 * 
 * Why this matters:
 * - Without replay, the network sees games in chronological order.
 *   It overwrites what it learned from game 1 by the time it reaches game 100.
 * - With replay, the network samples from ALL past games every training step.
 *   Each position is seen many times, from different network states.
 * - This is the key difference between our current ~50% vs Random and 
 *   brain.js exp20's 91% (which used 50% replay data).
 * 
 * Optimized for M2 Ultra 64GB:
 * - Stores up to 1M samples in memory (uses ~3GB at peak)
 * - Circular buffer: oldest samples are overwritten when full
 * - Supports priority sampling: more surprising/recent samples get higher weight
 * 
 * Usage:
 *   const buffer = new ReplayBuffer({ maxSize: 500000 });
 *   buffer.add(inputs, outputs, chosenIndices);
 *   const batch = buffer.sample(256);
 */

class ReplayBuffer {
  constructor(config = {}) {
    this.maxSize = config.maxSize || 500000;
    this.alpha = config.alpha || 0.6;       // Priority exponent (0=uniform, 1=full priority)
    this.beta = config.beta || 0.4;         // Importance sampling exponent (annealed to 1.0)
    this.betaGrowth = config.betaGrowth || 0.001; // Beta growth per sample call
    this.epsilon = config.epsilon || 1e-6;  // Small constant to avoid zero priority
    
    // Storage arrays (flat for memory efficiency)
    this.inputs = [];
    this.outputs = [];
    this.chosenIndices = [];
    this.valueTargets = [];
    
    // Priority tree (SumTree for O(log n) sampling)
    this.priorities = [];
    this.maxPriority = 1.0;
    
    // Position pointer for circular buffer
    this.position = 0;
    this.size = 0;
    
    // Statistics
    this.totalAdded = 0;
    this.totalSampled = 0;
  }

  /**
   * Add training samples to the buffer.
   * If buffer is full, overwrite oldest samples (circular).
   * 
   * @param {number[][]} inputs - State vectors
   * @param {number[][]} outputs - Target output vectors  
   * @param {number[]} chosenIndices - Index of chosen move per sample
   */
  add(inputs, outputs, chosenIndices, valueTargets = null) {
    const count = inputs.length;
    
    for (let i = 0; i < count; i++) {
      const valTarget = valueTargets ? valueTargets[i] : 0;
      if (this.size < this.maxSize) {
        // Buffer not yet full — append
        this.inputs.push(inputs[i]);
        this.outputs.push(outputs[i]);
        this.chosenIndices.push(chosenIndices[i]);
        this.valueTargets.push(valTarget);
        this.priorities.push(this.maxPriority);
        this.size++;
      } else {
        // Buffer full — overwrite oldest (circular)
        this.inputs[this.position] = inputs[i];
        this.outputs[this.position] = outputs[i];
        this.chosenIndices[this.position] = chosenIndices[i];
        this.valueTargets[this.position] = valTarget;
        this.priorities[this.position] = this.maxPriority;
      }
      
      this.position = (this.position + 1) % this.maxSize;
      this.totalAdded++;
    }
  }

  /**
   * Sample a batch from the buffer using priority-based sampling.
   * 
   * High-priority samples (large TD-error / surprising) are sampled more often.
   * Importance sampling weights correct for the bias.
   * 
   * @param {number} batchSize - Number of samples to draw
   * @returns {{ inputs, outputs, chosenIndices, weights, indices }}
   */
  sample(batchSize) {
    if (this.size === 0) {
      throw new Error('Cannot sample from empty replay buffer');
    }
    
    const actualBatchSize = Math.min(batchSize, this.size);
    
    // Compute sum of priorities
    const prioritySum = this.priorities.slice(0, this.size).reduce((a, b) => a + b, 0);
    
    // Stratified sampling: divide total priority into batchSize segments
    // and sample one element from each segment (more uniform than pure random)
    const segmentSize = prioritySum / actualBatchSize;
    
    const indices = [];
    const weights = [];
    
    // Anneal beta towards 1.0 for importance sampling
    this.beta = Math.min(1.0, this.beta + this.betaGrowth);
    
    // Compute max weight for normalization
    const maxWeight = Math.pow(this.size * (this.epsilon / prioritySum), -this.beta);
    
    for (let i = 0; i < actualBatchSize; i++) {
      // Sample within this segment
      const low = i * segmentSize;
      const high = (i + 1) * segmentSize;
      const target = low + Math.random() * (high - low);
      
      // Find sample by cumulative priority
      let cumulative = 0;
      let idx = 0;
      for (let j = 0; j < this.size; j++) {
        cumulative += this.priorities[j];
        if (cumulative >= target) {
          idx = j;
          break;
        }
      }
      
      indices.push(idx);
      
      // Importance sampling weight
      const prob = this.priorities[idx] / prioritySum;
      const weight = Math.pow(this.size * prob, -this.beta);
      weights.push(weight / maxWeight); // Normalize
    }
    
    // Gather samples
    const sampledInputs = indices.map(i => this.inputs[i]);
    const sampledOutputs = indices.map(i => this.outputs[i]);
    const sampledChosen = indices.map(i => this.chosenIndices[i]);
    const sampledValueTargets = indices.map(i => this.valueTargets[i] ?? 0);
    
    this.totalSampled += actualBatchSize;
    
    return {
      inputs: sampledInputs,
      outputs: sampledOutputs,
      chosenIndices: sampledChosen,
      valueTargets: sampledValueTargets,
      weights,
      indices
    };
  }

  /**
   * Update priorities for previously sampled indices.
   * Higher TD-error → higher priority → sampled more often.
   * 
   * @param {number[]} indices - Sample indices from last sample() call
   * @param {number[]} tdErrors - TD-error for each sample
   */
  updatePriorities(indices, tdErrors) {
    for (let i = 0; i < indices.length; i++) {
      const priority = Math.pow(Math.abs(tdErrors[i]) + this.epsilon, this.alpha);
      this.priorities[indices[i]] = priority;
      this.maxPriority = Math.max(this.maxPriority, priority);
    }
  }

  /**
   * Get buffer utilization as a fraction.
   */
  get utilization() {
    return this.size / this.maxSize;
  }

  /**
   * Get buffer statistics for logging.
   */
  getStats() {
    const prios = this.priorities.slice(0, this.size);
    return {
      size: this.size,
      maxSize: this.maxSize,
      utilization: this.utilization,
      totalAdded: this.totalAdded,
      totalSampled: this.totalSampled,
      avgPriority: prios.length > 0 ? prios.reduce((a, b) => a + b, 0) / prios.length : 0,
      maxPriority: this.maxPriority,
      beta: this.beta
    };
  }

  /**
   * Serialize buffer state for checkpointing.
   * Only saves metadata — actual training data is ephemeral.
   */
  toCheckpoint() {
    return {
      size: this.size,
      position: this.position,
      totalAdded: this.totalAdded,
      totalSampled: this.totalSampled,
      beta: this.beta,
      maxSize: this.maxSize
    };
  }

  /**
   * Clear the buffer.
   */
  clear() {
    this.inputs = [];
    this.outputs = [];
    this.chosenIndices = [];
    this.valueTargets = [];
    this.priorities = [];
    this.position = 0;
    this.size = 0;
    this.maxPriority = 1.0;
  }
}

module.exports = { ReplayBuffer };
