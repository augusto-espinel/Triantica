/**
 * GridManager - Pure game logic for Triantica
 * 
 * Platform-agnostic: Works in browser and Node.js
 * No DOM or Phaser dependencies
 */

// Constants
export const GRID_ROWS = 10;
export const GRID_COLS = 10;
export const ROTATIONS = [0, 1, 2, 3]; // 0°, 90°, 180°, 270°
export const PLAYERS = [1, 2];

export class GridManager {
  constructor(rows = GRID_ROWS, cols = GRID_COLS) {
    this.rows = rows;
    this.cols = cols;
    this.grid = this.createGrid();
    this.currentPlayer = 1;
    this.selectedRotation = 0;
    this.extraTurns = 0;
    this.winRounds = { 1: 0, 2: 0 };
    this.clusterMap = this.createClusterMap();
    this.clusters = {};
    this.nextClusterId = 1;
  }

  // === Grid Initialization ===

  createClusterMap() {
    return new Map(); // Key: "y,x,rotation", Value: clusterId
  }

  createGrid() {
    return Array(this.rows).fill(null).map(() => 
      Array(this.cols).fill(null).map(() => ({
        triangles: [] // Array of { player: number, rotation: number }
      }))
    );
  }

  reset() {
    this.grid = this.createGrid();
    this.currentPlayer = 1;
    this.selectedRotation = 0;
    this.extraTurns = 0;
    this.clusterMap = this.createClusterMap();
    this.clusters = {};
    this.nextClusterId = 1;
  }

  isGridFull() {
    return this.grid.flat().every(cell => cell.triangles.length === 2);
  }

  // === Move Validation ===

  isFirstMove() {
    return !this.grid.flat().some(cell => cell.triangles.length > 0);
  }

  isValidPlacement(x, y) {
    if (this.isFirstMove()) return true;
    const corners = this.getTriangleCorners(x, y, this.selectedRotation);
    return corners.some(([cx, cy]) => this.hasCorner(cx, cy));
  }

  hasCorner(x, y) {
    for (let dy = -1; dy <= 0; dy++) {
      for (let dx = -1; dx <= 0; dx++) {
        const cellX = x + dx;
        const cellY = y + dy;
        if (cellX >= 0 && cellY >= 0 && cellX < this.cols && cellY < this.rows) {
          const cell = this.grid[cellY][cellX];
          if (cell.triangles.some(t => 
            this.getTriangleCorners(cellX, cellY, t.rotation)
              .some(([tx, ty]) => tx === x && ty === y))
          ) {
            return true;
          }
        }
      }
    }
    return false;
  }

  getTriangleCorners(x, y, rotation) {
    switch (rotation) {
      case 0: return [[x, y+1], [x+1, y], [x, y]];     // ╱ bottom-left
      case 1: return [[x, y], [x, y+1], [x+1, y+1]];   // ╲ top-left
      case 3: return [[x+1, y], [x, y], [x+1, y+1]];   // ╱ top-right
      case 2: return [[x+1, y+1], [x, y+1], [x+1, y]];  // ╲ bottom-right
      default: return [];
    }
  }

  getDiagonalGroup(rotation) {
    return rotation % 2 === 0 ? 0 : 1; // 0=╱-type, 1=╲-type
  }

  // === Move Execution ===

  rotateTriangle(direction = 1) {
    this.selectedRotation = (this.selectedRotation + direction + 4) % 4;
  }

  placeTriangle(x, y, rotation = null, player = null) {
    const rot = rotation !== null ? rotation : this.selectedRotation;
    const pl = player !== null ? player : this.currentPlayer;
    
    const cell = this.grid[y][x];
    
    // Check if cell is full
    if (cell.triangles.length >= 2) return false;

    // Check for overlap
    if (cell.triangles.length > 0) {
      const newGroup = this.getDiagonalGroup(rot);
      const existingGroup = this.getDiagonalGroup(cell.triangles[0].rotation);
      if (newGroup === existingGroup || cell.triangles[0].rotation === rot) {
        return false;
      }
    }

    // Check placement validity
    const originalRotation = this.selectedRotation;
    this.selectedRotation = rot;
    const valid = this.isValidPlacement(x, y);
    this.selectedRotation = originalRotation;
    
    if (!valid) return false;

    // Place the triangle
    cell.triangles.push({ player: pl, rotation: rot });
    this.updateClustersForNewTriangle(x, y, rot, pl);

    return true;
  }

  // === Cluster Management ===

  updateClustersForNewTriangle(x, y, rotation, player) {
    const triangleKey = `${y},${x},${rotation}`;
    const adjacentClusterIds = new Set();

    const neighbors = this.getAdjacentTriangles(x, y, rotation, player);
    
    neighbors.forEach(({ x: nx, y: ny, rotation: nr }) => {
      const neighborKey = `${ny},${nx},${nr}`;
      if (this.clusterMap.has(neighborKey)) {
        adjacentClusterIds.add(this.clusterMap.get(neighborKey));
      }
    });

    let finalClusterId;
    let newClusterSize = 1;

    if (adjacentClusterIds.size === 0) {
      finalClusterId = this.nextClusterId++;
      this.clusters[finalClusterId] = { size: 1, player: player };
      newClusterSize = 1;
    } else if (adjacentClusterIds.size === 1) {
      finalClusterId = adjacentClusterIds.values().next().value;
      this.clusters[finalClusterId].size++;
      newClusterSize = this.clusters[finalClusterId].size;
    } else {
      finalClusterId = adjacentClusterIds.values().next().value;
      newClusterSize = 1;

      adjacentClusterIds.forEach(clusterId => {
        if (this.clusters[clusterId]) {
          newClusterSize += this.clusters[clusterId].size;
          if (clusterId !== finalClusterId) {
            this.updateMapForMergedCluster(clusterId, finalClusterId);
            delete this.clusters[clusterId];
          }
        }
      });

      if (this.clusters[finalClusterId]) {
        this.clusters[finalClusterId].size = newClusterSize;
      } else {
        this.clusters[finalClusterId] = { size: newClusterSize, player: player };
      }
    }

    this.clusterMap.set(triangleKey, finalClusterId);

    const cellTriangle = this.grid[y][x].triangles.find(
      t => t.rotation === rotation && t.player === player
    );
    if (cellTriangle) {
      cellTriangle.calculatedClusterSize = newClusterSize;
    }
  }

  updateMapForMergedCluster(oldClusterId, newClusterId) {
    for (const [key, value] of this.clusterMap.entries()) {
      if (value === oldClusterId) {
        this.clusterMap.set(key, newClusterId);
      }
    }
  }

  getConnectedClusterSize(x, y, rotation) {
    const key = `${y},${x},${rotation}`;
    if (this.clusterMap.has(key)) {
      const clusterId = this.clusterMap.get(key);
      return this.clusters[clusterId]?.size || 0;
    }
    return 0;
  }

  getLargestContiguous(player) {
    let maxSize = 0;
    for (const clusterId in this.clusters) {
      if (Object.hasOwnProperty.call(this.clusters, clusterId)) {
        const clusterInfo = this.clusters[clusterId];
        if (clusterInfo && clusterInfo.player === player) {
          maxSize = Math.max(maxSize, clusterInfo.size);
        }
      }
    }
    return maxSize;
  }

  // === Adjacency ===

  getAdjacentTriangles(x, y, rotation, player) {
    const adjacents = [];
    const edges = this.getEdgesForRotation(rotation);

    edges.forEach(edge => {
      let adjX = x, adjY = y;
      switch (edge) {
        case 'left':
          adjX = (x - 1 + this.cols) % this.cols;
          break;
        case 'right':
          adjX = (x + 1) % this.cols;
          break;
        case 'top':
          adjY = (y - 1 + this.rows) % this.rows;
          break;
        case 'bottom':
          adjY = (y + 1) % this.rows;
          break;
      }

      const adjCell = this.grid[adjY][adjX];
      adjCell.triangles.forEach(t => {
        if (t.player === player && 
            this.getEdgesForRotation(t.rotation).includes(this.getOppositeEdge(edge))) {
          adjacents.push({ x: adjX, y: adjY, rotation: t.rotation });
        }
      });
    });

    // Same cell, other diagonal - triangles sharing a corner
    const currCell = this.grid[y][x];
    const myCorners = this.getTriangleCorners(x, y, rotation);
    currCell.triangles.forEach(t => {
      if (t.player === player && t.rotation !== rotation) {
        const otherCorners = this.getTriangleCorners(x, y, t.rotation);
        // Check if they share at least one corner
        const sharesCorner = myCorners.some(([mx, my]) => 
          otherCorners.some(([ox, oy]) => mx === ox && my === oy)
        );
        if (sharesCorner) {
          adjacents.push({ x, y, rotation: t.rotation });
        }
      }
    });

    return adjacents;
  }

  getEdgesForRotation(rotation) {
    switch (rotation) {
      case 0: return ['left', 'top'];
      case 1: return ['left', 'bottom'];
      case 2: return ['right', 'bottom'];
      case 3: return ['right', 'top'];
      default: return [];
    }
  }

  getOppositeEdge(edge) {
    switch (edge) {
      case 'left': return 'right';
      case 'right': return 'left';
      case 'top': return 'bottom';
      case 'bottom': return 'top';
      default: return edge;
    }
  }

  // === Enclosure ===

  isTriangleEnclosed(x, y, rotation) {
    if (rotation === 0) { // ╱ top-right slot
      const topBlocked = 
        this.grid[(y - 1 + this.rows) % this.rows][x].triangles.some(t => t.rotation === 1 || t.rotation === 2);
      const leftBlocked = 
        this.grid[y][(x - 1 + this.cols) % this.cols].triangles.some(t => t.rotation === 2 || t.rotation === 3);
      const diagBlocked = 
        this.grid[y][x].triangles.some(t => t.rotation === 2);
      return topBlocked && leftBlocked && diagBlocked;

    } else if (rotation === 1) { // ╲ left-bottom slot
      const bottomBlocked = 
        this.grid[(y + 1) % this.rows][x].triangles.some(t => t.rotation === 0 || t.rotation === 3);
      const leftBlocked = 
        this.grid[y][(x - 1 + this.cols) % this.cols].triangles.some(t => t.rotation === 2 || t.rotation === 3);
      const diagBlocked = 
        this.grid[y][x].triangles.some(t => t.rotation === 3);
      return bottomBlocked && leftBlocked && diagBlocked;

    } else if (rotation === 2) { // ╱ right-bottom slot
      const bottomBlocked = 
        this.grid[(y + 1) % this.rows][x].triangles.some(t => t.rotation === 0 || t.rotation === 3);
      const rightBlocked = 
        this.grid[y][(x + 1) % this.cols].triangles.some(t => t.rotation === 0 || t.rotation === 1);
      const diagBlocked = 
        this.grid[y][x].triangles.some(t => t.rotation === 0);
      return bottomBlocked && rightBlocked && diagBlocked;

    } else if (rotation === 3) { // ╲ right-top slot
      const topBlocked = 
        this.grid[(y - 1 + this.rows) % this.rows][x].triangles.some(t => t.rotation === 1 || t.rotation === 2);
      const rightBlocked = 
        this.grid[y][(x + 1) % this.cols].triangles.some(t => t.rotation === 0 || t.rotation === 1);
      const diagBlocked = 
        this.grid[y][x].triangles.some(t => t.rotation === 1);
      return topBlocked && rightBlocked && diagBlocked;
    }
    return false;
  }

  claimEnclosedTriangles(player = null) {
    const pl = player !== null ? player : this.currentPlayer;
    let claimed = 0;
    const newlyClaimed = [];

    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        const cell = this.grid[y][x];
        if (cell.triangles.length === 1) {
          const mapping = [2, 3, 0, 1];
          const oppRotation = mapping[cell.triangles[0].rotation];
          if (this.isTriangleEnclosed(x, y, oppRotation)) {
            const newTriangle = { player: pl, rotation: oppRotation };
            cell.triangles.push(newTriangle);
            claimed++;
            newlyClaimed.push({ x, y, rotation: oppRotation, player: pl });
          }
        }
      }
    }

    newlyClaimed.forEach(claim => {
      this.updateClustersForNewTriangle(claim.x, claim.y, claim.rotation, claim.player);
    });

    return claimed;
  }

  // === Game Flow ===

  switchPlayer() {
    if (this.extraTurns > 0) {
      this.extraTurns--;
      return false; // Same player continues
    }
    this.currentPlayer = this.currentPlayer === 1 ? 2 : 1;
    return true; // Player switched
  }

  getWinner() {
    if (!this.isGridFull()) return null;
    
    const p1Score = this.getLargestContiguous(1);
    const p2Score = this.getLargestContiguous(2);
    
    if (p1Score > p2Score) return { winner: 1, p1Score, p2Score };
    if (p2Score > p1Score) return { winner: 2, p1Score, p2Score };
    return { winner: 0, p1Score, p2Score }; // Draw
  }

  // === Utility ===

  getValidMoves() {
    const moves = [];
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        for (const rotation of ROTATIONS) {
          if (this.isValidMove(x, y, rotation)) {
            moves.push({ x, y, rotation });
          }
        }
      }
    }
    return moves;
  }

  isValidMove(x, y, rotation) {
    const cell = this.grid[y][x];
    
    if (cell.triangles.length >= 2) return false;
    
    if (cell.triangles.length > 0) {
      const newGroup = this.getDiagonalGroup(rotation);
      const existingGroup = this.getDiagonalGroup(cell.triangles[0].rotation);
      if (newGroup === existingGroup || cell.triangles[0].rotation === rotation) {
        return false;
      }
    }

    const originalRotation = this.selectedRotation;
    this.selectedRotation = rotation;
    const valid = this.isValidPlacement(x, y);
    this.selectedRotation = originalRotation;
    
    return valid;
  }

  clone() {
    const cloned = new GridManager(this.rows, this.cols);
    
    // Deep copy grid
    cloned.grid = this.grid.map(row => 
      row.map(cell => ({
        triangles: cell.triangles.map(t => ({ ...t }))
      }))
    );
    
    // Copy state
    cloned.currentPlayer = this.currentPlayer;
    cloned.selectedRotation = this.selectedRotation;
    cloned.extraTurns = this.extraTurns;
    
    // Copy cluster data
    cloned.clusterMap = new Map(this.clusterMap);
    cloned.clusters = { ...this.clusters };
    cloned.nextClusterId = this.nextClusterId;
    
    return cloned;
  }

  getStateRepresentation(forPlayer = this.currentPlayer) {
    const state = [];
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        const cell = this.grid[y][x];
        const t1 = cell.triangles[0];
        const t2 = cell.triangles[1];

        const normalizePlayer = (p) => {
          if (!p) return 0;
          return p === forPlayer ? 1 : -1;
        };

        const normalizeRotation = (r) => {
          return r === undefined || r === -1 ? -1 : r / 4.0;
        };

        state.push(normalizePlayer(t1?.player));
        state.push(normalizeRotation(t1?.rotation ?? -1));
        state.push(normalizePlayer(t2?.player));
        state.push(normalizeRotation(t2?.rotation ?? -1));
      }
    }
    state.push(this.currentPlayer === forPlayer ? 1 : -1);
    return state;
  }

  // Export for debugging
  toJSON() {
    return {
      rows: this.rows,
      cols: this.cols,
      currentPlayer: this.currentPlayer,
      extraTurns: this.extraTurns,
      clusters: this.clusters,
      grid: this.grid
    };
  }
}

// Also export as global for browser compatibility
if (typeof window !== 'undefined') {
  window.GridManager = GridManager;
  window.GRID_ROWS = GRID_ROWS;
  window.GRID_COLS = GRID_COLS;
}
