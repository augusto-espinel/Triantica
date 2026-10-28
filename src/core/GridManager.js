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

/**
 * Dual Face-Adjacency Manifold Graph Cache for Toroidal Grids G = (V_T, E_edge, E_vertex)
 * Avoids rebuilding static topology when cloning GridManager during MCTS search.
 */
const MANIFOLD_GRAPH_CACHE = new Map();

/**
 * Build Dual Face-Adjacency Manifold Graph G = (V_T, E_edge, E_vertex) for a rows × cols Torus
 * - V_T: 4 * rows * cols triangle face slots
 * - E_edge: 3 boundary edge-neighbor groups per face (governs enclosure & contiguous clusters)
 * - E_vertex: 3 mesh vertex IDs per face (governs corner-sharing placement legality)
 */
export function buildToroidalManifoldGraph(rows, cols, toroidalPlacement = true) {
  const cacheKey = `${rows}x${cols}:${toroidalPlacement ? 'T2' : 'hybrid2D'}`;
  if (MANIFOLD_GRAPH_CACHE.has(cacheKey)) {
    return MANIFOLD_GRAPH_CACHE.get(cacheKey);
  }

  const numCells = rows * cols;
  const numFaceSlots = numCells * 4;
  const numVertices = toroidalPlacement ? rows * cols : (rows + 1) * (cols + 1);

  const getVertexId = (vx, vy) => {
    if (toroidalPlacement) {
      const wx = ((vx % cols) + cols) % cols;
      const wy = ((vy % rows) + rows) % rows;
      return wy * cols + wx;
    }
    return vy * (cols + 1) + vx;
  };

  const getCorners = (x, y, rot) => {
    switch (rot) {
      case 0: return [[x, y + 1], [x + 1, y], [x, y]];
      case 1: return [[x, y], [x, y + 1], [x + 1, y + 1]];
      case 3: return [[x + 1, y], [x, y], [x + 1, y + 1]];
      case 2: return [[x + 1, y + 1], [x, y + 1], [x + 1, y]];
      default: return [];
    }
  };

  const faces = new Array(numFaceSlots);
  const vertexToFaces = Array.from({ length: numVertices }, () => []);

  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      for (let rot = 0; rot < 4; rot++) {
        const faceId = (y * cols + x) * 4 + rot;
        const vertices = getCorners(x, y, rot).map(([cx, cy]) => getVertexId(cx, cy));
        const leftX = (x - 1 + cols) % cols;
        const rightX = (x + 1) % cols;
        const topY = (y - 1 + rows) % rows;
        const bottomY = (y + 1) % rows;

        let edgeNeighbors;
        if (rot === 0) {
          edgeNeighbors = [
            [(topY * cols + x) * 4 + 1, (topY * cols + x) * 4 + 2],
            [(y * cols + leftX) * 4 + 2, (y * cols + leftX) * 4 + 3],
            [(y * cols + x) * 4 + 2]
          ];
        } else if (rot === 1) {
          edgeNeighbors = [
            [(bottomY * cols + x) * 4 + 0, (bottomY * cols + x) * 4 + 3],
            [(y * cols + leftX) * 4 + 2, (y * cols + leftX) * 4 + 3],
            [(y * cols + x) * 4 + 3]
          ];
        } else if (rot === 2) {
          edgeNeighbors = [
            [(bottomY * cols + x) * 4 + 0, (bottomY * cols + x) * 4 + 3],
            [(y * cols + rightX) * 4 + 0, (y * cols + rightX) * 4 + 1],
            [(y * cols + x) * 4 + 0]
          ];
        } else {
          edgeNeighbors = [
            [(topY * cols + x) * 4 + 1, (topY * cols + x) * 4 + 2],
            [(y * cols + rightX) * 4 + 0, (y * cols + rightX) * 4 + 1],
            [(y * cols + x) * 4 + 1]
          ];
        }

        faces[faceId] = {
          faceId,
          x,
          y,
          rotation: rot,
          cellId: y * cols + x,
          diagonalGroup: rot % 2,
          oppositeFaceId: (y * cols + x) * 4 + ((rot + 2) % 4),
          vertices,
          edgeNeighbors
        };

        for (const vId of vertices) {
          vertexToFaces[vId].push(faceId);
        }
      }
    }
  }

  const graph = {
    type: toroidalPlacement ? 'T2_torus' : 'hybrid_2d',
    rows,
    cols,
    numCells,
    numFaceSlots,
    numVertices,
    faces,
    vertexToFaces
  };
  MANIFOLD_GRAPH_CACHE.set(cacheKey, graph);
  return graph;
}

/**
 * Build Dual Face-Adjacency Manifold Graph G = (V_T, E_edge, E_vertex) for an arbitrary
 * closed 3D triangulated mesh M = (V, F), where each face f in F is a triple [v0, v1, v2].
 */
export function buildGeneralMeshManifoldGraph(meshFaces, numVertices) {
  const numFaceSlots = meshFaces.length;
  const vertexToFaces = Array.from({ length: numVertices }, () => []);
  const edgeMap = new Map();

  for (let fId = 0; fId < numFaceSlots; fId++) {
    const [v0, v1, v2] = meshFaces[fId];
    for (const v of [v0, v1, v2]) {
      vertexToFaces[v].push(fId);
    }
    const edges = [[v0, v1], [v1, v2], [v2, v0]];
    for (const [a, b] of edges) {
      const key = a < b ? `${a},${b}` : `${b},${a}`;
      if (!edgeMap.has(key)) edgeMap.set(key, []);
      edgeMap.get(key).push(fId);
    }
  }

  const faces = meshFaces.map((verts, fId) => {
    const [v0, v1, v2] = verts;
    const edges = [[v0, v1], [v1, v2], [v2, v0]];
    const edgeNeighbors = edges.map(([a, b]) => {
      const key = a < b ? `${a},${b}` : `${b},${a}`;
      return (edgeMap.get(key) || []).filter(other => other !== fId);
    });
    return {
      faceId: fId,
      x: fId,
      y: 0,
      rotation: 0,
      cellId: fId,
      diagonalGroup: 0,
      oppositeFaceId: fId,
      vertices: verts,
      edgeNeighbors
    };
  });

  return {
    type: 'general_3d_manifold',
    rows: 1,
    cols: numFaceSlots,
    numCells: numFaceSlots,
    numFaceSlots,
    numVertices,
    faces,
    vertexToFaces
  };
}

export class GridManager {
  constructor(rows = GRID_ROWS, cols = GRID_COLS, options = {}) {
    this.rows = rows;
    this.cols = cols;
    // Phase 3: Full Toroidal Edge Symmetry (toroidalPlacement = true by default on main)
    this.toroidalPlacement = options.toroidalPlacement !== undefined ? options.toroidalPlacement : true;
    this.manifoldGraph = options.manifoldGraph || buildToroidalManifoldGraph(this.rows, this.cols, this.toroidalPlacement);
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

  isValidPlacement(x, y, rotation = null) {
    if (this.isFirstMove()) return true;
    const rot = rotation !== null ? rotation : this.selectedRotation;
    const corners = this.getTriangleCorners(x, y, rot);
    return corners.some(([cx, cy]) => this.hasCorner(cx, cy));
  }

  hasCorner(x, y) {
    if (this.toroidalPlacement) {
      const wx = ((x % this.cols) + this.cols) % this.cols;
      const wy = ((y % this.rows) + this.rows) % this.rows;
      for (let dy = -1; dy <= 0; dy++) {
        for (let dx = -1; dx <= 0; dx++) {
          const cellX = (wx + dx + this.cols) % this.cols;
          const cellY = (wy + dy + this.rows) % this.rows;
          const cell = this.grid[cellY][cellX];
          if (cell.triangles.some(t =>
            this.getTriangleCorners(cellX, cellY, t.rotation)
              .some(([tx, ty]) => (tx % this.cols) === wx && (ty % this.rows) === wy)
          )) {
            return true;
          }
        }
      }
      return false;
    }

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

  /**
   * Graph-native manifold methods for arbitrary 3D triangulated manifolds G = (V_T, E_edge, E_vertex)
   */
  isFaceOccupied(faceId, player = null) {
    const f = this.manifoldGraph.faces[faceId];
    if (!f) return false;
    const cell = this.grid[f.y][f.x];
    return cell.triangles.some(t => t.rotation === f.rotation && (player === null || t.player === player));
  }

  isManifoldVertexOccupied(vertexId) {
    const incident = this.manifoldGraph.vertexToFaces[vertexId] || [];
    return incident.some(fId => this.isFaceOccupied(fId));
  }

  isManifoldFaceValidPlacement(faceId) {
    if (this.isFirstMove()) return true;
    const f = this.manifoldGraph.faces[faceId];
    if (!f) return false;
    return f.vertices.some(vId => this.isManifoldVertexOccupied(vId));
  }

  isManifoldFaceEnclosed(faceId) {
    const f = this.manifoldGraph.faces[faceId];
    if (!f) return false;
    return f.edgeNeighbors.every(edgeGroup =>
      edgeGroup.some(neighborFaceId => this.isFaceOccupied(neighborFaceId))
    );
  }

  getManifoldAdjacentFaces(faceId, player) {
    const f = this.manifoldGraph.faces[faceId];
    if (!f) return [];
    const adj = [];
    for (const edgeGroup of f.edgeNeighbors) {
      for (const nFaceId of edgeGroup) {
        if (this.isFaceOccupied(nFaceId, player)) {
          adj.push(this.manifoldGraph.faces[nFaceId]);
        }
      }
    }
    return adj;
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
    
    // Single source of truth for move legality
    if (!this.isValidMove(x, y, rot)) return false;

    const cell = this.grid[y][x];
    // Place the triangle
    cell.triangles.push({ player: pl, rotation: rot });
    this.updateClustersForNewTriangle(x, y, rot, pl);

    return true;
  }

  /**
   * Canonical full move execution: place triangle, claim enclosed triangles,
   * and advance turn / extraTurns state.
   */
  executeMove(x, y, rotation = null, player = null) {
    const rot = rotation !== null ? rotation : this.selectedRotation;
    const pl = player !== null ? player : this.currentPlayer;
    this.currentPlayer = pl;
    this.selectedRotation = rot;

    if (!this.placeTriangle(x, y, rot, pl)) {
      return { success: false, claimed: 0, nextPlayer: this.currentPlayer, extraTurns: this.extraTurns };
    }

    const claimed = this.claimEnclosedTriangles(pl);
    this.extraTurns += Math.floor((1 + claimed) / 2);

    if (this.extraTurns > 0) {
      this.extraTurns--;
    } else {
      this.currentPlayer = pl === 1 ? 2 : 1;
    }

    return {
      success: true,
      claimed,
      nextPlayer: this.currentPlayer,
      extraTurns: this.extraTurns
    };
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

  /**
   * Get the triangles belonging to the largest contiguous cluster for a player.
   * Returns an array of {x, y, rotation} for each triangle in that cluster.
   * Used for TD(λ) cluster-contribution reward.
   */
  getLargestContiguousCluster(player) {
    let maxSize = 0;
    let largestClusterId = null;
    for (const clusterId in this.clusters) {
      if (Object.hasOwnProperty.call(this.clusters, clusterId)) {
        const clusterInfo = this.clusters[clusterId];
        if (clusterInfo && clusterInfo.player === player && clusterInfo.size > maxSize) {
          maxSize = clusterInfo.size;
          largestClusterId = clusterId;
        }
      }
    }
    if (largestClusterId === null) return [];

    // Collect all triangles in this cluster from the clusterMap
    const triangles = [];
    for (const [key, cid] of this.clusterMap.entries()) {
      if (cid == largestClusterId) {  // == because clusterIds can be numeric strings
        const parts = key.split(',');
        triangles.push({
          y: parseInt(parts[0], 10),
          x: parseInt(parts[1], 10),
          rotation: parseInt(parts[2], 10),
        });
      }
    }
    return triangles;
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

    // Check current cell for triangles with matching diagonal (sharing the diagonal edge)
    const currCell = this.grid[y][x];
    currCell.triangles.forEach(t => {
      if (t.player === player && this.getDiagonalGroup(t.rotation) === this.getDiagonalGroup(rotation) && t.rotation !== rotation) {
        adjacents.push({ x, y, rotation: t.rotation });
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

  isValidMove(x, y, rotation = null) {
    if (x < 0 || x >= this.cols || y < 0 || y >= this.rows) return false;
    const rot = rotation !== null ? rotation : this.selectedRotation;
    const cell = this.grid[y][x];
    
    if (cell.triangles.length >= 2) return false;
    
    if (cell.triangles.length > 0) {
      const newGroup = this.getDiagonalGroup(rot);
      const existingGroup = this.getDiagonalGroup(cell.triangles[0].rotation);
      // Second triangle in a cell MUST share the same diagonal group (╱ with ╱, ╲ with ╲)
      // and have the opposite rotation (not identical)
      if (newGroup !== existingGroup || cell.triangles[0].rotation === rot) {
        return false;
      }
    }

    return this.isValidPlacement(x, y, rot);
  }

  clone() {
    const cloned = new GridManager(this.rows, this.cols, {
      toroidalPlacement: this.toroidalPlacement,
      manifoldGraph: this.manifoldGraph
    });
    
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
    
    // Copy cluster data (DEEP copy to prevent mutation of original)
    cloned.clusterMap = new Map(this.clusterMap);
    cloned.clusters = Object.fromEntries(
      Object.entries(this.clusters).map(([k, v]) => [k, { ...v }])
    );
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

  /**
   * Canonical 10x10x18 one-hot state representation from `forPlayer`'s perspective.
   * Each cell has 2 non-overlapping triangle slots:
   * - Slot 0 (channels 0..8): triangle with rotation < 2 (rot 0 or 1), or empty (ch 0 = 1)
   * - Slot 1 (channels 9..17): triangle with rotation >= 2 (rot 2 or 3), or empty (ch 9 = 1)
   * Within each 9-channel slot:
   *   ch 0: empty
   *   ch 1..4: my piece (rot 0..3)
   *   ch 5..8: opponent piece (rot 0..3)
   */
  getCanonicalState(forPlayer = this.currentPlayer) {
    const state = new Float32Array(this.rows * this.cols * 18);
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const cell = this.grid[r][c];
        const cellOffset = (r * this.cols + c) * 18;

        for (let slot = 0; slot < 2; slot++) {
          const slotOffset = cellOffset + slot * 9;
          const tri = cell.triangles.find(t => (slot === 0 ? t.rotation < 2 : t.rotation >= 2));

          if (!tri) {
            state[slotOffset] = 1;
          } else {
            const ownerOffset = tri.player === forPlayer ? 0 : 4;
            state[slotOffset + 1 + ownerOffset + tri.rotation] = 1;
          }
        }
      }
    }
    return Array.from(state);
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
