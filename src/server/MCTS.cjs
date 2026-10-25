/**
 * Monte Carlo Tree Search for Triantica AI
 * 
 * Implements MCTS with neural network guidance following AlphaGo Zero principles:
 * - Policy network provides move priors (P) for UCT selection
 * - Value network provides position evaluation (v) at leaves
 * - Dirichlet noise at root for exploration
 * - Temperature schedule for move selection
 * - Batch evaluation for M2 Ultra optimization
 */

/**
 * Single node in the MCTS tree
 */
class MCTSNode {
  /**
   * @param {GridManager} grid - Current game state
   * @param {Object} move - Move that led to this state {x, y, rotation}
   * @param {MCTSNode|null} parent - Parent node (null for root)
   * @param {number} prior - Policy prior from neural network P(move)
   */
  constructor(grid, move, parent, prior) {
    this.grid = grid;
    this.move = move;
    this.parent = parent;
    this.prior = prior;
    
    // Tree statistics
    this.Q = 0;          // Average value from backpropagation
    this.N = 0;          // Visit count
    
    // Children: moveKey -> MCTSNode
    this.children = new Map();
    
    // Valid moves cache
    this._validMoves = null;
    
    // Player to move at this node
    this.player = grid.currentPlayer;
  }
  
  /**
   * Check if node is a leaf (hasn't been expanded)
   */
  isLeaf() {
    return this.children.size === 0;
  }
  
  /**
   * Check if node represents a terminal game state
   */
  isTerminal() {
    return this.getValidMoves().length === 0;
  }
  
  /**
   * Get all valid moves from this node (cached)
   */
  getValidMoves() {
    if (!this._validMoves) {
      this._validMoves = this.grid.getValidMoves();
    }
    return this._validMoves;
  }
  
  /**
   * Select best child using UCT formula
   * UCT = Q/N + c * P * sqrt(N_parent) / (1 + N)
   * 
   * @param {number} c - Exploration constant (default 1.5)
   * @returns {MCTSNode|null} Best child node
   */
  select(c = 1.5) {
    let bestScore = -Infinity;
    let bestChild = null;
    
    const sqrtParentN = Math.sqrt(this.N);
    
    for (const child of this.children.values()) {
      // UCT formula
      const uct = (child.Q / Math.max(1, child.N)) + 
                  c * child.prior * sqrtParentN / (1 + child.N);
      
      if (uct > bestScore) {
        bestScore = uct;
        bestChild = child;
      }
    }
    
    return bestChild;
  }
  
  /**
   * Expand node with policy priors from neural network
   * 
   * @param {number[]} policy - Policy vector (400 sigmoid outputs)
   */
  expand(policy) {
    if (!this.isLeaf()) return; // Already expanded
    
    if (!policy || policy.length < 400) {
      console.error(`⚠️  Expand called with invalid policy: length=${policy?.length || 'undefined'}`);
      return;
    }
    
    const validMoves = this.getValidMoves();
    
    for (const move of validMoves) {
      // Get policy prior for this move
      const idx = this._moveToIndex(move);
      const prior = policy[idx] || 0.01; // Small prior for unranked moves
      
      // Create child node
      const newGrid = this.grid.clone();
      newGrid.currentPlayer = this.player;
      newGrid.placeTriangle(move.x, move.y, move.rotation, this.player);
      
      // Switch player for child
      newGrid.currentPlayer = this.player === 1 ? 2 : 1;
      
      const moveKey = this._moveKey(move);
      this.children.set(moveKey, new MCTSNode(newGrid, move, this, prior));
    }
  }
  
  /**
   * Update node statistics during backpropagation
   * 
   * @param {number} value - Value from evaluation (-1 to +1)
   */
  update(value) {
    this.N += 1;
    // Running average
    this.Q += (value - this.Q) / this.N;
  }
  
  /**
   * Apply virtual loss for parallel search
   * Makes node temporarily look worse to other threads
   */
  applyVirtualLoss() {
    this._virtualLoss = true;
    this.Q -= 1;
    this.N += 1;
  }
  
  /**
   * Revert virtual loss after evaluation
   */
  revertVirtualLoss() {
    if (this._virtualLoss) {
      this.Q += 1;
      this.N -= 1;
      this._virtualLoss = false;
    }
  }
  
  /**
   * Convert move to policy vector index
   */
  _moveToIndex(move) {
    return (move.y * 10 + move.x) * 4 + move.rotation;
  }
  
  /**
   * Create unique key for a move
   */
  _moveKey(move) {
    return `${move.x},${move.y},${move.rotation}`;
  }
}

/**
 * Monte Carlo Tree Search controller
 */
class MCTS {
  /**
   * @param {TFConvNetwork} network - Neural network for evaluation
   * @param {Object} config - Configuration options
   */
  constructor(network, config = {}) {
    this.network = network;
    
    // Exploration constant
    this.c = config.c ?? 1.5;
    
    // Dirichlet noise parameters
    this.alpha = config.alpha ?? 0.03;
    this.epsilon = config.epsilon ?? 0.25;
    
    // Batch size for parallel evaluation
    this.batchSize = config.batchSize ?? 8;
    
    // Temperature for move selection
    this.temperature = config.temperature ?? 1.0;
    
    // Encode function (injected from training pipeline)
    this.encodeState = config.encodeState || null;
  }
  
  /**
   * Run MCTS search from a given position
   * 
   * @param {GridManager} grid - Starting position
   * @param {number} simulations - Number of simulations to run
   * @returns {Object} { move, policy, value } - Best move and improved policy
   */
  async search(grid, simulations = 50) {
    // Create root node
    const root = new MCTSNode(grid.clone(), null, null, 1.0);
    
    // Initial evaluation
    const state = this.encodeState ? 
      this.encodeState(grid, grid.currentPlayer) : 
      this._defaultEncode(grid, grid.currentPlayer);
    
    const { policy, value } = await this.network.run(state);
    
    // Add Dirichlet noise at root for exploration
    const noisyPolicy = this._addNoise(policy);
    root.expand(noisyPolicy);
    
    // Run simulations
    for (let i = 0; i < simulations; i++) {
      // Selection: traverse tree to leaf
      const path = this._selectPath(root);
      const leaf = path[path.length - 1];
      
      // Evaluation
      let leafValue;
      if (leaf.isTerminal()) {
        // Terminal: game over, use actual outcome
        leafValue = this._getTerminalValue(leaf, grid.currentPlayer);
      } else {
        // Non-terminal: use neural network
        const leafState = this.encodeState ?
          this.encodeState(leaf.grid, leaf.player) :
          this._defaultEncode(leaf.grid, leaf.player);
        
        const { policy: leafPolicy, value: nnValue } = await this.network.run(leafState);
        leaf.expand(leafPolicy);
        leafValue = nnValue;
      }
      
      // Backpropagation
      this._backpropagate(path, leafValue, grid.currentPlayer);
    }
    
    // Get improved policy from visit counts
    const improvedPolicy = this._getVisitPolicy(root);
    
    // Select move
    const move = this._selectMove(root, this.temperature);
    
    return {
      move,
      policy: improvedPolicy,
      value: root.Q / Math.max(1, root.N)
    };
  }
  
  /**
   * Run batched MCTS search (M2 Ultra optimization)
   * Evaluates multiple leaves in a single neural network call
   * 
   * @param {GridManager} grid - Starting position
   * @param {number} simulations - Number of simulations to run
   * @returns {Object} { move, policy, value }
   */
  async searchBatched(grid, simulations = 50) {
    const root = new MCTSNode(grid.clone(), null, null, 1.0);
    
    // Initial evaluation
    const state = this.encodeState ?
      this.encodeState(grid, grid.currentPlayer) :
      this._defaultEncode(grid, grid.currentPlayer);
    
    const { policy, value } = await this.network.run(state);
    const noisyPolicy = this._addNoise(policy);
    root.expand(noisyPolicy);
    
    // Batched simulation loop
    const batches = Math.floor(simulations / this.batchSize);
    const remainder = simulations % this.batchSize;
    
    // Run full batches
    for (let b = 0; b < batches; b++) {
      await this._runBatch(root, this.batchSize, grid.currentPlayer);
    }
    
    // Run remainder
    if (remainder > 0) {
      await this._runBatch(root, remainder, grid.currentPlayer);
    }
    
    const improvedPolicy = this._getVisitPolicy(root);
    const move = this._selectMove(root, this.temperature);
    
    return {
      move,
      policy: improvedPolicy,
      value: root.Q / Math.max(1, root.N)
    };
  }
  
  /**
   * Run a batch of simulations with parallel leaf evaluation
   */
  async _runBatch(root, batchSize, rootPlayer) {
    // Collect paths to leaves
    const paths = [];
    const leaves = [];
    
    for (let i = 0; i < batchSize; i++) {
      const path = this._selectPath(root);
      const leaf = path[path.length - 1];
      
      // Apply virtual loss
      for (const node of path) {
        node.applyVirtualLoss();
      }
      
      paths.push(path);
      leaves.push(leaf);
    }
    
    // Batch evaluate all leaves
    const states = [];
    const leafIndices = [];  // Track which leaves are non-terminal
    const terminalResults = new Map();  // Map leaf index -> terminal value
    
    for (let i = 0; i < leaves.length; i++) {
      const leaf = leaves[i];
      
      if (leaf.isTerminal()) {
        terminalResults.set(i, this._getTerminalValue(leaf, rootPlayer));
      } else {
        const state = this.encodeState ?
          this.encodeState(leaf.grid, leaf.player) :
          this._defaultEncode(leaf.grid, leaf.player);
        states.push(state);
        leafIndices.push(i);
      }
    }
    
    // Single batched NN call (only for non-terminal leaves)
    const { policies, values } = await this.network.batchPredict(states);
    
    // Process results
    for (let i = 0; i < leaves.length; i++) {
      const leaf = leaves[i];
      const path = paths[i];
      
      // Revert virtual loss
      for (const node of path) {
        node.revertVirtualLoss();
      }
      
      let leafValue;
      if (terminalResults.has(i)) {
        // Terminal node - use actual outcome
        leafValue = terminalResults.get(i);
      } else {
        // Non-terminal - expand with NN policy
        const stateIdx = leafIndices.indexOf(i);
        if (stateIdx >= 0 && policies[stateIdx]) {
          const policy = policies[stateIdx];
          if (policy.length !== 400) {
            console.error(`⚠️  Invalid policy length: ${policy.length} (expected 400)`);
            leafValue = 0;
          } else {
            leaf.expand(policy);
            leafValue = values[stateIdx];
          }
        } else {
          console.error(`⚠️  Missing policy for state ${i}, stateIdx=${stateIdx}`);
          leafValue = 0;  // Fallback
        }
      }
      
      this._backpropagate(path, leafValue, rootPlayer);
    }
  }
  
  /**
   * Select path from root to leaf using UCT
   */
  _selectPath(root) {
    const path = [root];
    let node = root;
    
    while (!node.isLeaf() && !node.isTerminal()) {
      node = node.select(this.c);
      if (node) {
        path.push(node);
      } else {
        break;
      }
    }
    
    return path;
  }
  
  /**
   * Backpropagate value up the tree
   */
  _backpropagate(path, value, rootPlayer) {
    // Flip value for opponent moves
    let currentValue = value;
    
    for (let i = path.length - 1; i >= 0; i--) {
      const node = path[i];
      
      // Value is from root player's perspective
      // Node.player is the player to move at that node
      // If node.player != rootPlayer, flip value
      if (node.player !== rootPlayer) {
        node.update(-currentValue);
      } else {
        node.update(currentValue);
      }
    }
  }
  
  /**
   * Get terminal game value
   */
  _getTerminalValue(node, rootPlayer) {
    // Count scores using getLargestContiguous
    const score1 = node.grid.getLargestContiguous(1);
    const score2 = node.grid.getLargestContiguous(2);
    
    if (score1 === score2) return 0; // Draw
    
    const winner = score1 > score2 ? 1 : 2;
    return winner === rootPlayer ? 1 : -1;
  }
  
  /**
   * Convert visit counts to policy distribution
   */
  _getVisitPolicy(root) {
    const policy = new Array(400).fill(0);
    const totalVisits = Math.max(1, root.N);
    
    for (const [moveKey, child] of root.children) {
      const idx = this._moveToIndex(child.move);
      policy[idx] = child.N / totalVisits;
    }
    
    return policy;
  }
  
  /**
   * Select move from root based on visit counts
   */
  _selectMove(root, temperature = 1.0) {
    const visits = [];
    
    for (const [moveKey, child] of root.children) {
      visits.push({
        move: child.move,
        count: child.N
      });
    }
    
    if (visits.length === 0) {
      return null;
    }
    
    if (temperature === 0) {
      // Greedy: pick most visited
      visits.sort((a, b) => b.count - a.count);
      return visits[0].move;
    }
    
    // Softmax with temperature
    const probs = visits.map(v => Math.pow(v.count, 1 / temperature));
    const sum = probs.reduce((a, b) => a + b, 0);
    const normalized = probs.map(p => p / sum);
    
    // Sample
    const r = Math.random();
    let acc = 0;
    for (let i = 0; i < visits.length; i++) {
      acc += normalized[i];
      if (r < acc) return visits[i].move;
    }
    
    return visits[visits.length - 1].move;
  }
  
  /**
   * Add Dirichlet noise to policy at root
   */
  _addNoise(policy) {
    // Sample from Dirichlet(alpha)
    const noise = this._sampleDirichlet(this.alpha, policy.length);
    
    // Mix with policy
    return policy.map((p, i) => 
      (1 - this.epsilon) * p + this.epsilon * noise[i]
    );
  }
  
  /**
   * Sample from Dirichlet distribution (approximation using Gamma)
   */
  _sampleDirichlet(alpha, size) {
    const samples = [];
    let sum = 0;
    
    for (let i = 0; i < size; i++) {
      // Gamma(alpha, 1) approximation
      const sample = this._sampleGamma(alpha, 1);
      samples.push(sample);
      sum += sample;
    }
    
    // Normalize
    return samples.map(s => s / sum);
  }
  
  /**
   * Sample from Gamma distribution (Marsaglia and Tsang's method)
   */
  _sampleGamma(shape, scale) {
    if (shape < 1) {
      return this._sampleGamma(shape + 1, scale) * Math.pow(Math.random(), 1 / shape);
    }
    
    const d = shape - 1/3;
    const c = 1 / Math.sqrt(9 * d);
    
    while (true) {
      let x, v;
      do {
        x = this._sampleNormal();
        v = 1 + c * x;
      } while (v <= 0);
      
      v = v * v * v;
      const u = Math.random();
      
      if (u < 1 - 0.0331 * (x * x) * (x * x)) {
        return d * v * scale;
      }
      
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
        return d * v * scale;
      }
    }
  }
  
  /**
   * Sample from standard normal distribution (Box-Muller)
   */
  _sampleNormal() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }
  
  /**
   * Default encoding if no encoder provided
   */
  _defaultEncode(grid, player) {
    // Simple canonical encoding
    const state = new Array(1800).fill(0);
    const opponent = player === 1 ? 2 : 1;
    
    for (let y = 0; y < 10; y++) {
      for (let x = 0; x < 10; x++) {
        const cell = grid.grid[y][x];
        const baseIdx = (y * 10 + x) * 18;
        
        // Encode each triangle
        for (let t = 0; t < Math.min(2, cell.triangles.length); t++) {
          const tri = cell.triangles[t];
          const channel = t * 9 + (tri.player === player ? 0 : 9);
          state[baseIdx + channel + tri.rotation] = 1;
        }
        
        // Encode empty slots
        for (let t = cell.triangles.length; t < 2; t++) {
          const channel = t * 9 + 8; // Empty channel
          state[baseIdx + channel] = 1;
        }
      }
    }
    
    return state;
  }
  
  /**
   * Convert move to policy index
   */
  _moveToIndex(move) {
    return (move.y * 10 + move.x) * 4 + move.rotation;
  }
}

module.exports = { MCTS, MCTSNode };
