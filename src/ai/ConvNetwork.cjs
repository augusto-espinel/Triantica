/**
 * ConvNetwork — Pure-JS 2D/3D Toroidal Convolutional Network + Safety Filter
 *
 * Runs the dual-head (Policy + Value) convolutional network trained by `train_tfjs.cjs`
 * in both Node.js and browser environments with zero external dependencies, and includes
 * fast 1-ply tactical lookahead (`predictMove`) to prevent immediate enclosure cascades.
 */

class ConvNetwork {
  constructor(jsonData = null) {
    this.rows = 10;
    this.cols = 10;
    this.depth = 1;
    this.channels = 18;
    this.numFilters = [32, 64, 64];
    this.policySize = 400;
    this.paddingMode = 'circular';
    this.weights = [];
    if (jsonData) {
      this.fromJSON(jsonData);
    }
  }

  fromJSON(json) {
    const arch = json.architecture || {};
    this.rows = arch.rows || 10;
    this.cols = arch.cols || 10;
    this.depth = arch.depth || 1;
    this.channels = arch.channels || 18;
    this.numFilters = arch.numFilters || [32, 64, 64];
    this.policySize = arch.policySize || (this.rows * this.cols * 4);
    this.paddingMode = arch.paddingMode || 'circular';
    this.weights = json.weights || [];

    // Pre-resolve head weights for fast forward passes
    const lastFilters = this.numFilters[this.numFilters.length - 1];
    this._pConvW = this.weights.find(w => w.name.includes('conv') && w.data[0].length === lastFilters * 2);
    this._vConvW = this.weights.find(w => w.name.includes('conv') && w.data[0].length === lastFilters * 1);
    this._pDenseW = this.weights.find(w => w.name.includes('dense') && w.data[1].length === this.policySize);
    this._vDense1W = this.weights.find(w => w.name.includes('dense') && w.data[1].length === 64);
    this._vDense2W = this.weights.find(w => w.name.includes('dense') && w.data[1].length === 1);
    return this;
  }

  toJSON() {
    return {
      type: 'ConvNetwork',
      architecture: {
        rows: this.rows,
        cols: this.cols,
        depth: this.depth,
        channels: this.channels,
        numFilters: this.numFilters,
        policySize: this.policySize,
        paddingMode: this.paddingMode
      },
      weights: this.weights
    };
  }

  _conv2d3x3(inBuf, inC, outC, kernel, bias) {
    const H = this.rows;
    const W = this.cols;
    const outBuf = new Float32Array(H * W * outC);
    const circular = this.paddingMode === 'circular';

    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const outOffset = (r * W + c) * outC;
        for (let oc = 0; oc < outC; oc++) {
          outBuf[outOffset + oc] = bias[oc];
        }
        for (let kr = 0; kr < 3; kr++) {
          let ir = r + kr - 1;
          if (circular) {
            ir = (ir + H) % H;
          } else if (ir < 0 || ir >= H) {
            continue;
          }
          for (let kc = 0; kc < 3; kc++) {
            let ic = c + kc - 1;
            if (circular) {
              ic = (ic + W) % W;
            } else if (ic < 0 || ic >= W) {
              continue;
            }
            const inOffset = (ir * W + ic) * inC;
            const kBase = ((kr * 3 + kc) * inC) * outC;
            for (let icIdx = 0; icIdx < inC; icIdx++) {
              const val = inBuf[inOffset + icIdx];
              if (val === 0) continue;
              const kOffset = kBase + icIdx * outC;
              for (let oc = 0; oc < outC; oc++) {
                outBuf[outOffset + oc] += val * kernel[kOffset + oc];
              }
            }
          }
        }
      }
    }
    return outBuf;
  }

  _batchNormRelu(buf, C, gamma, beta, mean, variance) {
    const eps = 1e-3;
    const scale = new Float32Array(C);
    const shift = new Float32Array(C);
    for (let c = 0; c < C; c++) {
      scale[c] = gamma[c] / Math.sqrt(variance[c] + eps);
      shift[c] = beta[c] - mean[c] * scale[c];
    }
    const out = new Float32Array(buf.length);
    for (let i = 0; i < buf.length; i++) {
      const c = i % C;
      const v = buf[i] * scale[c] + shift[c];
      out[i] = v > 0 ? v : 0;
    }
    return out;
  }

  _conv2d1x1(inBuf, inC, outC, kernel, bias) {
    const N = this.rows * this.cols * this.depth;
    const outBuf = new Float32Array(N * outC);
    for (let i = 0; i < N; i++) {
      const inOffset = i * inC;
      const outOffset = i * outC;
      for (let oc = 0; oc < outC; oc++) {
        let sum = bias[oc];
        for (let ic = 0; ic < inC; ic++) {
          sum += inBuf[inOffset + ic] * kernel[ic * outC + oc];
        }
        outBuf[outOffset + oc] = sum;
      }
    }
    return outBuf;
  }

  _dense(inBuf, inSize, outSize, kernel, bias, activation) {
    const out = new Float32Array(outSize);
    for (let o = 0; o < outSize; o++) {
      let sum = bias[o];
      for (let i = 0; i < inSize; i++) {
        sum += inBuf[i] * kernel[i * outSize + o];
      }
      if (activation === 'sigmoid') {
        out[o] = 1 / (1 + Math.exp(-sum));
      } else if (activation === 'relu') {
        out[o] = sum > 0 ? sum : 0;
      } else if (activation === 'tanh') {
        out[o] = Math.tanh(sum);
      } else {
        out[o] = sum;
      }
    }
    return out;
  }

  /**
   * Run a forward pass on a flat canonical state vector (1800 values for 10x10x18).
   * Returns { policy: number[], value: number }.
   */
  run(input1d) {
    let x = new Float32Array(input1d);
    let inC = this.channels;
    let wIdx = 0;

    for (let i = 0; i < this.numFilters.length; i++) {
      const outC = this.numFilters[i];
      const convW = this.weights[wIdx++];
      const bnW = this.weights[wIdx++];
      x = this._conv2d3x3(x, inC, outC, convW.data[0], convW.data[1]);
      x = this._batchNormRelu(x, outC, bnW.data[0], bnW.data[1], bnW.data[2], bnW.data[3]);
      inC = outC;
    }

    const pConv = this._conv2d1x1(x, inC, 2, this._pConvW.data[0], this._pConvW.data[1]);
    const policy = this._dense(pConv, pConv.length, this.policySize, this._pDenseW.data[0], this._pDenseW.data[1], 'sigmoid');

    const vConv = this._conv2d1x1(x, inC, 1, this._vConvW.data[0], this._vConvW.data[1]);
    const vH = this._dense(vConv, vConv.length, 64, this._vDense1W.data[0], this._vDense1W.data[1], 'relu');
    const vOut = this._dense(vH, 64, 1, this._vDense2W.data[0], this._vDense2W.data[1], 'tanh');

    return {
      policy: Array.from(policy),
      value: vOut[0]
    };
  }

  /**
   * Count how many triangles would be enclosed if `player` places at (mx, my, mrot).
   * Uses O(1) push/pop mutation on the cell and checks the 5 affected cells.
   */
  static countWouldBeEnclosed(grid, mx, my, mrot, player) {
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
   * Check local 5x5 toroidal radius around (mx, my) for immediate opponent enclosure responses.
   */
  static evaluateOpponentReplyRisk(grid, mx, my, mrot, player) {
    const { rows, cols } = grid;
    const cell = grid.grid[my][mx];
    cell.triangles.push({ player, rotation: mrot });

    const opp = player === 1 ? 2 : 1;
    let maxOppClaim = 0;
    let totalOppClaims = 0;

    for (let dy = -2; dy <= 2; dy++) {
      for (let dx = -2; dx <= 2; dx++) {
        const ox = (mx + dx + cols) % cols;
        const oy = (my + dy + rows) % rows;
        if (grid.grid[oy][ox].triangles.length >= 2) continue;
        for (let r = 0; r < 4; r++) {
          if (!grid.isValidMove(ox, oy, r)) continue;
          const c = ConvNetwork.countWouldBeEnclosed(grid, ox, oy, r, opp);
          if (c > 0) {
            totalOppClaims += c;
            if (c > maxOppClaim) maxOppClaim = c;
          }
        }
      }
    }

    cell.triangles.pop();
    return { maxOppClaim, totalOppClaims };
  }

  /**
   * Select the best move using the ConvNetwork policy + optional 1-ply tactical safety filter.
   *
   * @param {GridManager} grid - Current game state
   * @param {number} playerNumber - AI player number (1 or 2)
   * @param {Object} options - { safetyFilter: boolean, topK: number }
   * @returns {Object|null} { x, y, rotation, score, rawPolicyScore, value }
   */
  predictMove(grid, playerNumber, options = {}) {
    const useSafety = options.safetyFilter !== undefined ? options.safetyFilter : true;
    const topK = options.topK || 12;

    const state = grid.getCanonicalState
      ? grid.getCanonicalState(playerNumber)
      : ConvNetwork.encodeCanonicalState(grid, playerNumber);

    const { policy, value } = this.run(state);
    const moves = [];

    for (let i = 0; i < policy.length; i++) {
      const rotation = i % 4;
      const cellIndex = Math.floor(i / 4);
      const x = cellIndex % grid.cols;
      const y = Math.floor(cellIndex / grid.cols);
      if (grid.isValidMove(x, y, rotation)) {
        moves.push({
          x,
          y,
          rotation,
          score: policy[i],
          rawPolicyScore: policy[i],
          value
        });
      }
    }

    if (moves.length === 0) return null;
    moves.sort((a, b) => b.score - a.score);

    if (!useSafety) {
      return moves[0];
    }

    const K = Math.min(topK, moves.length);
    let bestMove = moves[0];
    let bestAdjustedScore = -Infinity;

    for (let i = 0; i < K; i++) {
      const m = moves[i];
      const myClaimed = ConvNetwork.countWouldBeEnclosed(grid, m.x, m.y, m.rotation, playerNumber);
      let adjScore = m.rawPolicyScore;

      if (myClaimed > 0) {
        adjScore = m.rawPolicyScore * (1 + myClaimed * 0.5) + 0.25 * myClaimed;
      } else if ((grid.extraTurns || 0) === 0) {
        const { maxOppClaim, totalOppClaims } = ConvNetwork.evaluateOpponentReplyRisk(
          grid,
          m.x,
          m.y,
          m.rotation,
          playerNumber
        );
        if (maxOppClaim > 0) {
          adjScore = m.rawPolicyScore * Math.max(0.01, 1 - (maxOppClaim * 0.45 + totalOppClaims * 0.1));
        }
      }

      m.score = adjScore;
      if (adjScore > bestAdjustedScore) {
        bestAdjustedScore = adjScore;
        bestMove = m;
      }
    }

    return bestMove;
  }

  /**
   * Fallback canonical state encoder if GridManager instance does not expose getCanonicalState().
   */
  static encodeCanonicalState(grid, playerNumber) {
    const { rows, cols } = grid;
    const state = new Array(rows * cols * 18).fill(0);
    let offset = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cell = grid.grid[r][c];
        let slashTriangle = null;
        let backslashTriangle = null;
        for (const t of cell.triangles) {
          if (t.rotation % 2 === 0) slashTriangle = t;
          else backslashTriangle = t;
        }
        if (!slashTriangle) {
          state[offset + 0] = 1;
        } else if (slashTriangle.player === playerNumber) {
          state[offset + 1 + slashTriangle.rotation] = 1;
        } else {
          state[offset + 5 + slashTriangle.rotation] = 1;
        }
        if (!backslashTriangle) {
          state[offset + 9] = 1;
        } else if (backslashTriangle.player === playerNumber) {
          state[offset + 10 + backslashTriangle.rotation] = 1;
        } else {
          state[offset + 14 + backslashTriangle.rotation] = 1;
        }
        offset += 18;
      }
    }
    return state;
  }
}

if (typeof window !== 'undefined') {
  window.ConvNetwork = ConvNetwork;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ConvNetwork };
}
