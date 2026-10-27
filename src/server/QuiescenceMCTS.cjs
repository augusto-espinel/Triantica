/**
 * Quiescence MCTS for Triantica
 * 
 * Extends standard MCTS with enclosure-aware quiescence search.
 * Both single (`search`) and batched (`searchBatched` / `_runBatch`) paths
 * run tactical quiescence evaluation at every leaf and adjust root policy targets:
 * 
 * 1. Enclosing moves (`claimed > 0`) grant extra turns (`floor((1 + claimed) / 2)`),
 *    allowing the active player to immediately patch any newly created holes or
 *    continue cascading enclosures. Therefore, enclosing moves NEVER have opponent
 *    risk (`risk = 0`) and receive a positive enclosure bonus.
 * 2. Non-enclosing moves (`claimed === 0`) that pass the turn to the opponent
 *    and leave enclosure opportunities trigger greedy cascade rollouts in
 *    `_quiescenceEval()`, penalizing both leaf values (Q-values) and root
 *    policy targets (`_getRiskAdjustedPolicy`).
 * 3. Leaf nodes are always expanded using the policy of `leaf.grid` (pre-cascade),
 *    while leaf value evaluation uses the quiescent post-cascade board + tactical
 *    cascade adjustment.
 */

const { MCTS, MCTSNode } = require('./MCTS.cjs');

/**
 * Fast O(1) check of how many triangles a move (mx, my, mrot) by `player` would enclose,
 * without cloning the grid or rebuilding clusters.
 * Because `isTriangleEnclosed` only depends on (cx, cy) and its 4 orthogonal neighbors,
 * placing at (mx, my) can only enclose (mx, my) itself or its 4 orthogonal neighbors.
 */
function countWouldBeEnclosed(grid, mx, my, mrot, player) {
  const { rows, cols } = grid;
  const candidates = [
    [mx, my],
    [(mx + 1) % cols, my],
    [(mx - 1 + cols) % cols, my],
    [mx, (my + 1) % rows],
    [mx, (my - 1 + rows) % rows]
  ];
  let claimed = 0;
  const mapping = [2, 3, 0, 1];
  const cell = grid.grid[my][mx];
  cell.triangles.push({ player, rotation: mrot });

  for (let i = 0; i < 5; i++) {
    const [cx, cy] = candidates[i];
    const cCell = grid.grid[cy][cx];
    if (cCell.triangles.length === 1) {
      const oppRotation = mapping[cCell.triangles[0].rotation];
      if (grid.isTriangleEnclosed(cx, cy, oppRotation)) {
        claimed++;
      }
    }
  }

  cell.triangles.pop();
  return claimed;
}

/**
 * Enclosure opportunity detection
 * Returns moves that claim 1+ enclosed triangles for `player`
 */
function findEnclosureOpportunities(grid, player) {
  const savedPlayer = grid.currentPlayer;
  grid.currentPlayer = player;
  const opportunities = [];
  const { rows, cols } = grid;
  
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      if (grid.grid[y][x].triangles.length >= 2) continue;
      for (let rotation = 0; rotation < 4; rotation++) {
        if (!grid.isValidMove(x, y, rotation)) continue;
        
        const claimed = countWouldBeEnclosed(grid, x, y, rotation, player);
        if (claimed > 0) {
          opportunities.push({
            x, y, rotation,
            claimed,
            extraTurns: Math.floor((1 + claimed) / 2)
          });
        }
      }
    }
  }
  
  grid.currentPlayer = savedPlayer;
  // Sort by claimed count (most enclosures first)
  opportunities.sort((a, b) => b.claimed - a.claimed);
  return opportunities;
}

/**
 * Greedy enclosure cascade player
 * Takes ALL available enclosures in descending order of claims
 */
function playEnclosureCascade(grid, player, maxDepth = 10) {
  let currentGrid = grid.clone();
  let currentPlayer = player;
  let totalClaimed = 0;
  let moves = [];
  let depth = 0;
  
  while (depth < maxDepth && !currentGrid.isGridFull()) {
    const enclosures = findEnclosureOpportunities(currentGrid, currentPlayer);
    if (enclosures.length === 0) break;
    
    const best = enclosures[0];
    currentGrid.executeMove(best.x, best.y, best.rotation, currentPlayer);
    totalClaimed += best.claimed;
    moves.push({ ...best, player: currentPlayer });
    
    currentPlayer = currentGrid.currentPlayer;
    depth++;
  }
  
  return {
    grid: currentGrid,
    totalClaimed,
    moves,
    depth
  };
}

/**
 * Evaluate enclosure risk of a candidate move from `grid` by `player`
 * Enclosing moves (`claimed > 0`) or moves that retain the turn (`nextPlayer === player`)
 * have ZERO opponent enclosure risk.
 */
function calculateEnclosureRisk(grid, move, player) {
  const tempGrid = grid.clone();
  const execRes = tempGrid.executeMove(move.x, move.y, move.rotation, player);
  const claimed = execRes.claimed || 0;
  
  // Enclosing moves grant extra turns so the mover can immediately patch holes or cascade
  if (claimed > 0 || tempGrid.currentPlayer === player) {
    return {
      risk: 0,
      claimed,
      oppEnclosures: 0,
      maxOppClaim: 0
    };
  }
  
  const opponent = player === 1 ? 2 : 1;
  const oppEnclosures = findEnclosureOpportunities(tempGrid, opponent);
  
  let risk = 0;
  for (const opp of oppEnclosures) {
    risk += opp.claimed * 0.3;
    risk += opp.extraTurns * 0.5;
  }
  
  return {
    risk,
    claimed: 0,
    oppEnclosures: oppEnclosures.length,
    maxOppClaim: oppEnclosures[0]?.claimed || 0
  };
}

/**
 * Evaluate enclosure risk directly on an already-expanded child node (`child.grid`)
 */
function calculateChildNodeRisk(child, moverPlayer) {
  const claimed = child.claimed || 0;
  if (claimed > 0 || child.grid.currentPlayer === moverPlayer) {
    return {
      risk: 0,
      claimed,
      oppEnclosures: 0,
      maxOppClaim: 0
    };
  }
  
  const opponent = moverPlayer === 1 ? 2 : 1;
  const oppEnclosures = findEnclosureOpportunities(child.grid, opponent);
  
  let risk = 0;
  for (const opp of oppEnclosures) {
    risk += opp.claimed * 0.3;
    risk += opp.extraTurns * 0.5;
  }
  
  return {
    risk,
    claimed: 0,
    oppEnclosures: oppEnclosures.length,
    maxOppClaim: oppEnclosures[0]?.claimed || 0
  };
}

/**
 * Quiescence MCTS Node
 */
class QuiescenceNode extends MCTSNode {
  constructor(grid, move, parent, prior) {
    super(grid, move, parent, prior);
    this.enclosureRisk = null;
    this.quiescenceDepth = 0;
  }
}

/**
 * Quiescence MCTS Controller
 */
class QuiescenceMCTS extends MCTS {
  constructor(network, config = {}) {
    super(network, config);
    
    // Quiescence configuration
    this.quiescenceMode = config.quiescenceMode ?? 'greedy';  // 'none' | 'greedy' | 'heuristic' | 'hybrid'
    this.quiescenceDepth = config.quiescenceDepth ?? 5;
    this.riskPenaltyWeight = config.riskPenaltyWeight ?? 0.15;
    this.enclosureBonusWeight = config.enclosureBonusWeight ?? 0.2;
    
    // Heuristic player for opponent model (optional)
    this.heuristicPlayer = config.heuristicPlayer || null;
  }
  
  /**
   * Expand root node and compute tactical enclosure bonuses/risks on root children
   */
  _expandRootWithRisk(root, policy, rootPlayer) {
    root.expand(policy);
    
    if (this.quiescenceMode === 'none') return;
    
    let priorSum = 0;
    for (const child of root.children.values()) {
      child.enclosureRisk = calculateChildNodeRisk(child, rootPlayer);
      // Shape root priors so MCTS immediately explores enclosing moves and avoids hole-creating moves
      if (child.claimed > 0) {
        child.prior = child.prior * (1 + child.claimed * 1.5) + 0.05 * child.claimed;
      } else if (child.enclosureRisk.risk > 0) {
        const penaltyFactor = Math.max(0.05, 1 - child.enclosureRisk.risk * this.riskPenaltyWeight * 2);
        child.prior *= penaltyFactor;
      }
      priorSum += child.prior;
    }
    
    if (priorSum > 0) {
      for (const child of root.children.values()) {
        child.prior /= priorSum;
      }
    }
  }
  
  /**
   * Override: Run single-step MCTS search with quiescence extension
   */
  async search(grid, simulations = 50) {
    const rootPlayer = grid.currentPlayer;
    const root = new QuiescenceNode(grid.clone(), null, null, 1.0);
    
    const state = this.encodeState 
      ? this.encodeState(grid, rootPlayer)
      : this._defaultEncode(grid, rootPlayer);
    
    const { policy } = await this.network.run(state);
    const noisyPolicy = this._addNoise(policy);
    this._expandRootWithRisk(root, noisyPolicy, rootPlayer);
    
    for (let i = 0; i < simulations; i++) {
      const path = this._selectPath(root);
      const leaf = path[path.length - 1];
      
      let valueFromRootPerspective;
      
      if (leaf.isTerminal()) {
        valueFromRootPerspective = this._getTerminalValue(leaf, rootPlayer);
      } else {
        // 1. Expand leaf using pre-cascade leaf.grid policy
        const leafState = this.encodeState
          ? this.encodeState(leaf.grid, leaf.player)
          : this._defaultEncode(leaf.grid, leaf.player);
        const { policy: leafPolicy, value: rawLeafValue } = await this.network.run(leafState);
        leaf.expand(leafPolicy);
        
        // 2. Quiescence rollout for value evaluation (from rootPlayer's perspective)
        const qEval = this._runQuiescenceRollout(leaf, rootPlayer);
        leaf.quiescenceDepth = qEval.depth;
        
        let quietValueFromRoot;
        if (qEval.depth > 0) {
          if (qEval.grid.getValidMoves().length === 0) {
            quietValueFromRoot = this._getGridTerminalValue(qEval.grid, rootPlayer);
          } else {
            const qPlayer = qEval.grid.currentPlayer;
            const qState = this.encodeState
              ? this.encodeState(qEval.grid, qPlayer)
              : this._defaultEncode(qEval.grid, qPlayer);
            const { value: qNNValue } = await this.network.run(qState);
            quietValueFromRoot = qPlayer === rootPlayer ? qNNValue : -qNNValue;
          }
        } else {
          quietValueFromRoot = leaf.player === rootPlayer ? rawLeafValue : -rawLeafValue;
        }
        
        valueFromRootPerspective = Math.max(-1, Math.min(1, quietValueFromRoot + qEval.rootAdjustment));
      }
      
      this._backpropagate(path, valueFromRootPerspective, rootPlayer);
    }
    
    const improvedPolicy = this._getRiskAdjustedPolicy(root);
    const move = this._selectMove(root, this.temperature);
    
    return {
      move,
      policy: improvedPolicy,
      value: root.Q / Math.max(1, root.N)
    };
  }
  
  /**
   * Override: Run batched MCTS search with quiescence extension (used by train_tfjs.cjs!)
   */
  async searchBatched(grid, simulations = 50) {
    const rootPlayer = grid.currentPlayer;
    const root = new QuiescenceNode(grid.clone(), null, null, 1.0);
    
    const state = this.encodeState
      ? this.encodeState(grid, rootPlayer)
      : this._defaultEncode(grid, rootPlayer);
    
    const { policy } = await this.network.run(state);
    const noisyPolicy = this._addNoise(policy);
    this._expandRootWithRisk(root, noisyPolicy, rootPlayer);
    
    const batches = Math.floor(simulations / this.batchSize);
    const remainder = simulations % this.batchSize;
    
    for (let b = 0; b < batches; b++) {
      await this._runBatch(root, this.batchSize, rootPlayer);
    }
    if (remainder > 0) {
      await this._runBatch(root, remainder, rootPlayer);
    }
    
    const improvedPolicy = this._getRiskAdjustedPolicy(root);
    const move = this._selectMove(root, this.temperature);
    
    return {
      move,
      policy: improvedPolicy,
      value: root.Q / Math.max(1, root.N)
    };
  }
  
  /**
   * Override: Batched leaf expansion + batched quiescence value evaluation
   */
  async _runBatch(root, batchSize, rootPlayer) {
    const paths = [];
    const leaves = [];
    
    for (let i = 0; i < batchSize; i++) {
      const path = this._selectPath(root);
      const leaf = path[path.length - 1];
      
      for (const node of path) {
        node.applyVirtualLoss(rootPlayer);
      }
      
      paths.push(path);
      leaves.push(leaf);
    }
    
    const expandStates = [];
    const nonTerminalIndices = [];
    const terminalResults = new Map();
    const quiescenceRollouts = new Map();
    
    // Quiescence states that changed board (depth > 0 and non-terminal after cascade)
    const quietStates = [];
    const quietLeafMap = new Map(); // leaf index i -> index in quietStates
    
    for (let i = 0; i < leaves.length; i++) {
      const leaf = leaves[i];
      
      if (leaf.isTerminal()) {
        terminalResults.set(i, this._getTerminalValue(leaf, rootPlayer));
      } else {
        // State for expanding `leaf.grid` (pre-cascade)
        const state = this.encodeState
          ? this.encodeState(leaf.grid, leaf.player)
          : this._defaultEncode(leaf.grid, leaf.player);
        expandStates.push(state);
        nonTerminalIndices.push(i);
        
        // Run deterministic quiescence cascade from `leaf`
        const qEval = this._runQuiescenceRollout(leaf, rootPlayer);
        quiescenceRollouts.set(i, qEval);
        leaf.quiescenceDepth = qEval.depth;
        
        if (qEval.depth > 0 && !qEval.grid.isGridFull() && qEval.grid.getValidMoves().length > 0) {
          const qPlayer = qEval.grid.currentPlayer;
          const qState = this.encodeState
            ? this.encodeState(qEval.grid, qPlayer)
            : this._defaultEncode(qEval.grid, qPlayer);
          quietLeafMap.set(i, quietStates.length);
          quietStates.push(qState);
        }
      }
    }
    
    // Single combined batchPredict call for both pre-cascade policies and post-cascade quiet values
    const allStates = expandStates.concat(quietStates);
    const { policies, values } = await this.network.batchPredict(allStates);
    const quietOffset = expandStates.length;
    
    for (let i = 0; i < leaves.length; i++) {
      const leaf = leaves[i];
      const path = paths[i];
      
      for (const node of path) {
        node.revertVirtualLoss();
      }
      
      let valueFromRootPerspective;
      if (terminalResults.has(i)) {
        valueFromRootPerspective = terminalResults.get(i);
      } else {
        const stateIdx = nonTerminalIndices.indexOf(i);
        const policy = stateIdx >= 0 ? policies[stateIdx] : null;
        if (policy && policy.length === 400) {
          leaf.expand(policy);
        }
        
        const qEval = quiescenceRollouts.get(i);
        let baseValueFromRoot;
        
        if (qEval && qEval.depth > 0) {
          if (quietLeafMap.has(i)) {
            const qIdx = quietOffset + quietLeafMap.get(i);
            const qPlayer = qEval.grid.currentPlayer;
            const qNNVal = values[qIdx] ?? 0;
            baseValueFromRoot = qPlayer === rootPlayer ? qNNVal : -qNNVal;
          } else {
            // Quiescence cascade reached terminal board
            baseValueFromRoot = this._getGridTerminalValue(qEval.grid, rootPlayer);
          }
        } else {
          const rawVal = stateIdx >= 0 ? (values[stateIdx] ?? 0) : 0;
          baseValueFromRoot = leaf.player === rootPlayer ? rawVal : -rawVal;
        }
        
        const adj = qEval ? qEval.rootAdjustment : 0;
        valueFromRootPerspective = Math.max(-1, Math.min(1, baseValueFromRoot + adj));
      }
      
      this._backpropagate(path, valueFromRootPerspective, rootPlayer);
    }
  }
  
  /**
   * Run synchronous tactical quiescence cascade on a leaf node and return
   * the resulting quiet grid + value adjustment from `rootPlayer`'s perspective.
   * 
   * Also credits any enclosure already claimed by the move that created `leaf` (`leaf.claimed`).
   */
  _runQuiescenceRollout(leaf, rootPlayer) {
    if (this.quiescenceMode === 'none') {
      return { grid: leaf.grid, rootAdjustment: 0, depth: 0, totalClaimed: 0, cascade: [] };
    }
    
    let rootAdjustment = 0;
    const mover = leaf.parent ? leaf.parent.player : rootPlayer;
    
    // Credit any enclosure claimed by the move that led to `leaf`
    if (leaf.claimed > 0) {
      if (mover === rootPlayer) {
        rootAdjustment += leaf.claimed * this.enclosureBonusWeight;
      } else {
        rootAdjustment -= leaf.claimed * this.riskPenaltyWeight;
      }
    }
    
    const currentGrid = leaf.grid.clone();
    let currentPlayer = currentGrid.currentPlayer;
    let depth = 0;
    let totalClaimed = leaf.claimed || 0;
    const cascade = [];
    
    while (depth < this.quiescenceDepth && !currentGrid.isGridFull()) {
      let bestMove = null;
      let bestScore = 0;
      
      if (this.quiescenceMode === 'greedy') {
        const enclosures = findEnclosureOpportunities(currentGrid, currentPlayer);
        if (enclosures.length === 0) break;
        bestMove = enclosures[0];
        bestScore = enclosures[0].claimed;
      } else if (this.quiescenceMode === 'heuristic' && this.heuristicPlayer) {
        bestMove = this.heuristicPlayer.getMove(currentGrid);
        if (!bestMove) break;
        const tempGrid = currentGrid.clone();
        tempGrid.placeTriangle(bestMove.x, bestMove.y, bestMove.rotation, currentPlayer);
        bestScore = tempGrid.claimEnclosedTriangles(currentPlayer);
        if (bestScore === 0) break; // Stop quiescence once no more forced enclosures
      } else if (this.quiescenceMode === 'hybrid') {
        const enclosures = findEnclosureOpportunities(currentGrid, currentPlayer);
        if (enclosures.length > 0) {
          bestMove = enclosures[0];
          bestScore = enclosures[0].claimed;
        } else {
          break;
        }
      } else {
        break;
      }
      
      if (!bestMove) break;
      
      const actingPlayer = currentPlayer;
      currentGrid.executeMove(bestMove.x, bestMove.y, bestMove.rotation, actingPlayer);
      totalClaimed += bestScore;
      cascade.push({ ...bestMove, player: actingPlayer, score: bestScore });
      
      if (actingPlayer === rootPlayer && bestScore > 0) {
        rootAdjustment += bestScore * this.enclosureBonusWeight;
      } else if (actingPlayer !== rootPlayer && bestScore > 0) {
        rootAdjustment -= bestScore * this.riskPenaltyWeight;
      }
      
      currentPlayer = currentGrid.currentPlayer;
      depth++;
    }
    
    return {
      grid: currentGrid,
      rootAdjustment,
      depth,
      totalClaimed,
      cascade
    };
  }
  
  /**
   * Compute terminal value of an arbitrary grid from `rootPlayer`'s perspective
   */
  _getGridTerminalValue(grid, rootPlayer) {
    const score1 = grid.getLargestContiguous(1);
    const score2 = grid.getLargestContiguous(2);
    if (score1 === score2) return 0;
    const winner = score1 > score2 ? 1 : 2;
    return winner === rootPlayer ? 1 : -1;
  }
  
  /**
   * Get risk-adjusted policy target from root visit counts:
   * - Enclosing moves (`child.claimed > 0`) have 0 risk and receive an enclosure bonus.
   * - Non-enclosing moves that leave opponent enclosures (`child.enclosureRisk.risk > 0`)
   *   are Multiplicatively penalized so the Policy head learns never to open free cascades.
   */
  _getRiskAdjustedPolicy(root) {
    if (this.quiescenceMode === 'none') {
      return this._getVisitPolicy(root);
    }
    
    const policy = new Array(400).fill(0);
    const totalVisits = Math.max(1, root.N);
    
    for (const child of root.children.values()) {
      const idx = this._moveToIndex(child.move);
      let score = child.N / totalVisits;
      
      if (child.claimed > 0) {
        score = score * (1 + child.claimed * this.enclosureBonusWeight) + 0.02 * child.claimed;
      } else if (child.enclosureRisk && child.enclosureRisk.risk > 0) {
        const penaltyFactor = Math.max(0.02, 1 - child.enclosureRisk.risk * this.riskPenaltyWeight * 2);
        score *= penaltyFactor;
      }
      
      policy[idx] = Math.max(0, score);
    }
    
    const sum = policy.reduce((a, b) => a + b, 0);
    if (sum > 0) {
      for (let i = 0; i < policy.length; i++) {
        policy[i] /= sum;
      }
    }
    
    return policy;
  }
  
  /**
   * Select move from root using risk-adjusted visit scores
   */
  _selectMove(root, temperature = 1.0) {
    if (this.quiescenceMode === 'none' || root.children.size === 0) {
      return super._selectMove(root, temperature);
    }
    
    const policy = this._getRiskAdjustedPolicy(root);
    const entries = [];
    for (const child of root.children.values()) {
      const idx = this._moveToIndex(child.move);
      entries.push({ move: child.move, prob: policy[idx] || 0 });
    }
    
    if (temperature === 0) {
      entries.sort((a, b) => b.prob - a.prob);
      return entries[0].move;
    }
    
    const scaled = entries.map(e => Math.pow(Math.max(1e-6, e.prob), 1 / temperature));
    const sum = scaled.reduce((a, b) => a + b, 0);
    let r = Math.random() * sum;
    for (let i = 0; i < entries.length; i++) {
      r -= scaled[i];
      if (r <= 0) return entries[i].move;
    }
    return entries[entries.length - 1].move;
  }
}

module.exports = {
  QuiescenceMCTS,
  QuiescenceNode,
  findEnclosureOpportunities,
  playEnclosureCascade,
  calculateEnclosureRisk,
  calculateChildNodeRisk
};
