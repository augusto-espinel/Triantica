// Global debug mode flag: gates benchmark/debug modes (AI vs Random, AI vs Heuristic, AI vs AI)
// and on-screen debug overlays. Enabled by default during development.
const DEBUG_MODE = true;
if (typeof window !== 'undefined') {
  window.DEBUG_MODE = DEBUG_MODE;
}

class GridManager {
  constructor(rows, cols, options = {}) {
    this.rows = rows;
    this.cols = cols;
    this.toroidalPlacement = options.toroidalPlacement !== undefined ? options.toroidalPlacement : true;
    this.grid = this.createGrid();
    this.currentPlayer = 1;
    this.selectedRotation = 0; // 0-3 (0°, 90°, 180°, 270°)
    this.extraTurns = 0; // Track remaining extra turns
    this.winRounds = { 1: 0, 2: 0 }; // Track rounds won by each player
    this.clusterMap = this.createClusterMap();  //Maps triangle position/index to cluster ID
    this.clusters = {};  //Stores info per cluster ID, e.g., { size: number, player: number }
    this.nextClusterId = 1;  //Counter for assigning unique cluster IDs
  }

  // Add helper to create the initial empty map
  createClusterMap() {
    // A map where key is "y,x,rotation" and value is clusterId
    // Or a 3D array: 
    Array(this.rows).fill().map(() => Array(this.cols).fill().map(() => Array(4).fill(0))); // 0 = no cluster
    // Using a Map might be easier for sparse grids
    return new Map(); // Key: "y,x,rotation", Value: clusterId
  }

  reset() {
    this.grid = this.createGrid(); // 
    this.currentPlayer = 1; // 
    this.extraTurns = 0; // 
    // Reset cluster data
    this.clusterMap = this.createClusterMap();
    this.clusters = {};
    this.nextClusterId = 1;
  }

  isGridFull() {
    return this.grid.flat().every(cell => cell.triangles.length === 2);
  }

  getValidMoves() {
    const moves = [];
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        for (let rotation = 0; rotation < 4; rotation++) {
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

  createGrid() {
      return Array(this.rows).fill().map(() => 
        Array(this.cols).fill().map(() => ({
          triangles: [] // Array of { player: number, rotation: number }
        }))
      );
    }

  isValidPlacement(x, y, rotation = null) {
      if (this.isFirstMove()) return true;
      const rot = rotation !== null ? rotation : this.selectedRotation;
      // Check if any corner of the new triangle touches existing triangles
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
            if (cell.triangles.some(t => this.getTriangleCorners(cellX, cellY, t.rotation)
              .some(([tx, ty]) => (tx % this.cols) === wx && (ty % this.rows) === wy))) {
              return true;
            }
          }
        }
        return false;
      }
      // Check all cells around grid point (x,y) for overlapping corners
      for (let dy = -1; dy <= 0; dy++) {
        for (let dx = -1; dx <= 0; dx++) {
          const cellX = x + dx;
          const cellY = y + dy;
          if (cellX >= 0 && cellY >= 0 && cellX < this.cols && cellY < this.rows) {
            const cell = this.grid[cellY][cellX];
            if (cell.triangles.some(t => this.getTriangleCorners(cellX, cellY, t.rotation)
              .some(([tx, ty]) => tx === x && ty === y))) {
              return true;
            }
          }
        }
      }
      return false;
  }

  getTriangleCorners(x, y, rotation) {
      // Returns grid points (not pixel positions) for triangle corners
      switch (rotation) {
        case 0: return [[x, y+1], [x+1, y], [x, y]];    // ╱ bottom-left
        case 1: return [[x, y], [x, y+1], [x+1, y+1]];  // ╲ top-left
        case 3: return [[x+1, y], [x, y], [x+1, y+1]];  // ╱ top-right
        case 2: return [[x+1, y+1], [x, y+1], [x+1, y]]; // ╲ bottom-right
      }
  }

  rotateTriangle(direction = 1) {
    this.selectedRotation = (this.selectedRotation + direction + 4) % 4;
  }
  
  placeTriangle(x, y, rotation = null, player = null) {
    // Accept optional rotation and player args for compatibility with core GridManager
    const rot = rotation !== null ? rotation : this.selectedRotation;
    const pl = player !== null ? player : this.currentPlayer;
    
    // Single source of truth for move legality
    if (!this.isValidMove(x, y, rot)) return false;

    const cell = this.grid[y][x];
    cell.triangles.push({
      player: pl,
      rotation: rot
    });

    // --- Update Cluster Information ---
    this.updateClustersForNewTriangle(x, y, rot, pl);

    return true;
  }

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

  updateClustersForNewTriangle(x, y, rotation, player) {
    const triangleKey = `${y},${x},${rotation}`;
    const adjacentClusterIds = new Set(); // Use a Set to store unique adjacent cluster IDs

    // 1. Find adjacent triangles of the same player
    const neighbors = this.getAdjacentTriangles(x, y, rotation, player); // 

    // 2. Get the cluster IDs of these neighbors
    neighbors.forEach(({ x: nx, y: ny, rotation: nr }) => {
        const neighborKey = `${ny},${nx},${nr}`;
        if (this.clusterMap.has(neighborKey)) {
            adjacentClusterIds.add(this.clusterMap.get(neighborKey));
        }
    });

    let finalClusterId;
    let newClusterSize = 1; // Start with the new triangle itself

    if (adjacentClusterIds.size === 0) {
        // Case 1: New cluster
        finalClusterId = this.nextClusterId++;
        this.clusters[finalClusterId] = { size: 1, player: player };
        newClusterSize = 1;
        // console.log(`New Cluster ${finalClusterId} created.`);

    } else if (adjacentClusterIds.size === 1) {
        // Case 2: Add to existing cluster
        finalClusterId = adjacentClusterIds.values().next().value; // Get the single ID
        this.clusters[finalClusterId].size++;
        newClusterSize = this.clusters[finalClusterId].size;
        // console.log(`Triangle added to Cluster ${finalClusterId}, new size ${newClusterSize}.`);

    } else {
        // Case 3: Merge multiple clusters
        // console.log("Merging clusters:", [...adjacentClusterIds]);
        // Choose the cluster ID to keep (e.g., the first one encountered)
        finalClusterId = adjacentClusterIds.values().next().value;
        newClusterSize = 1; // Start with the new triangle

        // Merge sizes and update map for other clusters
        adjacentClusterIds.forEach(clusterId => {
            if (this.clusters[clusterId]) { // Check if cluster still exists (might have been merged already in complex cases)
                 newClusterSize += this.clusters[clusterId].size;
                 if (clusterId !== finalClusterId) {
                     // Mark cluster for merging/update map entries
                     this.updateMapForMergedCluster(clusterId, finalClusterId);
                     delete this.clusters[clusterId]; // Remove merged cluster info
                    //  console.log(`Cluster ${clusterId} merged into ${finalClusterId}.`);
                 }
             }
        });

        // Update the size of the final cluster
        if (this.clusters[finalClusterId]) { // Ensure the kept cluster wasn't deleted if it was the only one
             this.clusters[finalClusterId].size = newClusterSize;
        } else {
             // This case might happen if finalClusterId was somehow deleted; recreate it
            //  console.warn('Recreating cluster ${finalClusterId} during merge.');
             this.clusters[finalClusterId] = { size: newClusterSize, player: player };
             // Need to ensure all triangles previously mapped to finalClusterId are still correct (should be)
        }
        //  console.log(`Clusters merged into ${finalClusterId}, new size ${newClusterSize}.`);
    }

    // 3. Update the map for the newly placed triangle
    this.clusterMap.set(triangleKey, finalClusterId);

    // Store the size calculated during the update process
    // We'll attach it to the triangle object temporarily? Or return it?
    // Let's try attaching it. Note: This adds state to the grid object.
    const cellTriangle = this.grid[y][x].triangles.find(t => t.rotation === rotation && t.player === player);
    if(cellTriangle) {
        cellTriangle.calculatedClusterSize = newClusterSize; // Store size temporarily
    }
  } // End updateClustersForNewTriangle

  // NEW HELPER METHOD: Update map entries when merging clusters
  // NOTE: This is the potentially slow part if not optimized (e.g., using Union-Find)
  updateMapForMergedCluster(oldClusterId, newClusterId) {
    // Inefficiently iterates through the whole map.
    // A better approach would store triangles per cluster or use Union-Find.
    for (const [key, value] of this.clusterMap.entries()) {
        if (value === oldClusterId) {
            this.clusterMap.set(key, newClusterId);
        }
    }
  }
  
  getDiagonalGroup(rotation) {
      // Group rotations by their base diagonal (╱ or ╲)
      return rotation % 2 === 0 ? 0 : 1; // 0=╱-type, 1=╲-type
  }

  isFirstMove() {
      return !this.grid.flat().some(cell => cell.triangles.length > 0);
  }

  // Check if a specific triangle slot (slash or backslash) is enclosed
  isTriangleEnclosed(x, y, rotation) {
    if (rotation === 0) { // Slash (╱) top-right slot
      const topBlocked = 
        this.grid[(y - 1 + this.rows) % this.rows][x].triangles.some(t => t.rotation === 1) || 
        this.grid[(y - 1 + this.rows) % this.rows][x].triangles.some(t => t.rotation === 2);
      const leftBlocked = 
        this.grid[y][(x - 1 + this.cols) % this.cols].triangles.some(t => t.rotation === 2) || 
        this.grid[y][(x - 1 + this.cols) % this.cols].triangles.some(t => t.rotation === 3);
      const diagBlocked = 
        this.grid[y][x].triangles.some(t => t.rotation === 2);
      return topBlocked && leftBlocked && diagBlocked;

    } else if (rotation === 1)  { // Backslash (╲) left-bottom slot
      const bottomBlocked = 
        this.grid[(y + 1) % this.rows][x].triangles.some(t => t.rotation === 0) || 
        this.grid[(y + 1) % this.rows][x].triangles.some(t => t.rotation === 3);
      const leftBlocked = 
        this.grid[y][(x - 1 + this.cols) % this.cols].triangles.some(t => t.rotation === 2) || 
        this.grid[y][(x - 1 + this.cols) % this.cols].triangles.some(t => t.rotation === 3);
      const diagBlocked = 
        this.grid[y][x].triangles.some(t => t.rotation === 3);
      return bottomBlocked && leftBlocked && diagBlocked;

    } else if (rotation === 2)  { // Slash (╱) right-bottom slot
      const bottomBlocked = 
        this.grid[(y + 1) % this.rows][x].triangles.some(t => t.rotation === 0) || 
        this.grid[(y + 1) % this.rows][x].triangles.some(t => t.rotation === 3);
      const rightBlocked = 
        this.grid[y][(x + 1) % this.cols].triangles.some(t => t.rotation === 0) || 
        this.grid[y][(x + 1) % this.cols].triangles.some(t => t.rotation === 1);
      const diagBlocked = 
        this.grid[y][x].triangles.some(t => t.rotation === 0);
      return bottomBlocked && rightBlocked && diagBlocked;

    }else if (rotation === 3)  { // Backslash (╲) right-top slot
      const topBlocked = 
        this.grid[(y - 1 + this.rows) % this.rows][x].triangles.some(t => t.rotation === 1) || 
        this.grid[(y - 1 + this.rows) % this.rows][x].triangles.some(t => t.rotation === 2);
      const rightBlocked = 
        this.grid[y][(x + 1) % this.cols].triangles.some(t => t.rotation === 0) || 
        this.grid[y][(x + 1) % this.cols].triangles.some(t => t.rotation === 1);
      const diagBlocked = 
        this.grid[y][x].triangles.some(t => t.rotation === 1);
      return topBlocked && rightBlocked && diagBlocked;
    }
    return false;
  }

  /**
   * Gets the largest contiguous triangle group size for a player
   * by looking up pre-calculated cluster sizes.
   * Relies on the this.clusters object being accurately maintained.
   * @param {number} player - The player number (1 or 2).
   * @returns {number} The size of the largest cluster belonging to the player.
   */
  getLargestContiguous(player) {
    let maxSize = 0;

    // Iterate through the keys (cluster IDs) of the clusters object
    for (const clusterId in this.clusters) {
        // Standard check to ensure we are iterating over own properties
        if (Object.hasOwnProperty.call(this.clusters, clusterId)) {
            const clusterInfo = this.clusters[clusterId];

            // Check if this cluster belongs to the requested player
            if (clusterInfo && clusterInfo.player === player) {
                // Update maxSize if this cluster is larger
                maxSize = Math.max(maxSize, clusterInfo.size);
            }
        }
    }

    // Return the largest size found for that player
    return maxSize;
  }

  // Claim all enclosed triangles for the current player
  claimEnclosedTriangles() {
    let claimed = 0;
    const newlyClaimed = []; // Keep track of newly claimed triangles {x, y, rotation, player}
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        const cell = this.grid[y][x]
        if (cell.triangles.length == 1)  {
          // Check for unclaimed slot
          const mapping = [2, 3, 0, 1]; // Mapping of rotations to opposite
          if (this.isTriangleEnclosed(x, y, mapping[cell.triangles[0].rotation])) {
            const newTriangle = { player: this.currentPlayer, rotation: mapping[cell.triangles[0].rotation] };
            cell.triangles.push(newTriangle);
            claimed++;
            newlyClaimed.push({ x, y, rotation: newTriangle.rotation, player: newTriangle.player });
            // console.log(`Claimed triangle at ${x},${y} rot ${newTriangle.rotation}`);
          }
        }
      }
    }
    // After identifying all claims, update clusters for each claimed triangle
    // This prevents issues where claiming one triangle affects the neighbors of another claimed triangle in the same turn.
    newlyClaimed.forEach(claim => {
      this.updateClustersForNewTriangle(claim.x, claim.y, claim.rotation, claim.player);
    });
    return claimed;
  }


  /**
   * Gets the pre-calculated size of the cluster a specific triangle belongs to.
   * @param {number} x - The cell x-coordinate.
   * @param {number} y - The cell y-coordinate.
   * @param {number} rotation - The rotation of the target triangle.
   * @returns {number} The size of the connected cluster, or 0 if not found.
   */
  getConnectedClusterSize(x, y, rotation) {
    const key = `${y},${x},${rotation}`;
    if (this.clusterMap.has(key)) {
        const clusterId = this.clusterMap.get(key);
        return this.clusters[clusterId]?.size || 0; // Return size or 0 if cluster info is missing
    }
    return 0; // Triangle not part of any cluster
  }

  // Flood-fill to find connected triangles
  floodFill(x, y, rotation, player, visited) {
    const queue = [{ x, y, rotation }];
    let size = 0;
    const key = `${x},${y},${rotation}`;
    visited.add(key);

    while (queue.length > 0) {
      const { x: currX, y: currY, rotation: currRot } = queue.shift();
      size++;

      // Get adjacent triangles sharing a side (with wrap-around)
      const neighbors = this.getAdjacentTriangles(currX, currY, currRot, player);
      neighbors.forEach(({ x: adjX, y: adjY, rotation: adjRot }) => {
        const adjKey = `${adjX},${adjY},${adjRot}`;
        if (!visited.has(adjKey)) {
          visited.add(adjKey);
          queue.push({ x: adjX, y: adjY, rotation: adjRot });
        }
      });
    }
    return size;
  }

  // Get adjacent triangles sharing a side (with wrap-around)
  getAdjacentTriangles(x, y, rotation, player) {
    const adjacents = [];
    const edges = this.getEdgesForRotation(rotation);

    edges.forEach(edge => {
      // Calculate adjacent cell (wrapping around grid edges)
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

      // Check adjacent cell for triangles with matching edge
      const adjCell = this.grid[adjY][adjX];
      adjCell.triangles.forEach(t => {
        if (t.player === player && this.getEdgesForRotation(t.rotation).includes(this.getOppositeEdge(edge))) {
          adjacents.push({ x: adjX, y: adjY, rotation: t.rotation });
        }
      });
      
    });

    // Check current cell for triangles with matching diagonal
    const currCell = this.grid[y][x];
    currCell.triangles.forEach(t => {
      if (t.player === player && this.getDiagonalGroup(t.rotation) === this.getDiagonalGroup(rotation) && t.rotation !== rotation) {
        adjacents.push({ x, y, rotation: t.rotation });
      }
    });

    return adjacents;
  }

  getEdgesForRotation(rotation) {
    // Returns which edges (left/right/top/bottom) the triangle uses
    switch (rotation) {
      case 0: return ['left', 'top'];
      case 1: return ['left', 'bottom'];
      case 2: return ['right', 'bottom'];
      case 3: return ['right', 'top'];
      default: return [];
    }
  }

  getOppositeEdge(edge) {
    // Returns the edge that would connect across cells
    switch (edge) {
      case 'left': return 'right';
      case 'right': return 'left';
      case 'top': return 'bottom';
      case 'bottom': return 'top';
    }
  }

  // Clone method needed for HeuristicPlayer simulations
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
    
    // Copy cluster data (shallow copy for performance in UI)
    cloned.clusterMap = new Map(this.clusterMap);
    cloned.clusters = Object.fromEntries(
      Object.entries(this.clusters).map(([k, v]) => [k, { ...v }])
    );
    cloned.nextClusterId = this.nextClusterId;
    
    return cloned;
  }

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
}

class GameScene extends Phaser.Scene {
  constructor() {
      super({ key: 'Game' });
      this.cellSize = 50;
      this.gridOffset = { x: 50, y: 120 };
      this.gridManager = new GridManager(10, 10);
      this.cursorPos = { x: 0, y: 0 };
      this.isDragging = false;
      this.hasMovedBeyondThreshold = false;
      this.lastPinchDistance = 0;
      this.dragStart = { x: 0, y: 0 };
      this.panStart = { u: 0, v: 0 };
      this.cursorStart = { x: 0, y: 0 };
      this.gridBounds = { x: 0, y: 0, width: 0, height: 0 };
      // Step 1 View-Projection state (100% isolated from GridManager)
      this.panOffset = { u: 0, v: 0 }; // Continuous toroidal pan offset in cell units [0, 10)
      this.zoomLevel = 1.0; // Zoom factor: 1.0 (full 10x10 board) up to 2.5 (4x4 magnified view)
      this.minZoom = 1.0; // Max zoom out = default full 10x10 grid view
      this.maxZoom = 2.5; // Max zoom in = 10 / 2.5 = 4x4 grid area
      this.isPinching = false;
      this.gestureHadMultiTouch = false; // Locks out tap-to-move-cursor until all fingers are lifted
      this.pinchStartDist = 0;
      this.pinchStartZoom = 1.0;
      this.pinchAnchorToroidal = { x: 0, y: 0 };
      this.isTiltedView = true; // Default to tilted 2.5D tabletop slab
      this.showGhostRing = false; // Default to Mode A (Strict Zero-Duplication 10x10 Stencil Slab)
      this.gameMode = 'HvH'; // Default, or set based on selection ('HvH', 'HvAI', 'AvR', 'AvH', 'AvA')
      this.aiDifficulty = 'medium'; // 'easy' | 'medium' | 'hard'
      this.humanPlayerNumber = 1; // 1 (Red - First Move) | 2 (Blue - Second Move)
      this.players = { 1: { type: 'human' }, 2: { type: 'human' } }; // Player setup
      this.aiInstances = { 1: null, 2: null }; // To hold AI instances
      this.totalMovesPlayed = 0; // Track move count for opening diversity
  }

  preload() {
      // Load assets later
  }

  init(data) {
    console.log('GameScene init with data:', data);
    // Set the gameMode, difficulty, and human side based on data passed from MenuScene
    this.gameMode = data.mode || 'HvH';
    this.aiDifficulty = (data.difficulty || 'medium').toLowerCase();
    this.humanPlayerNumber = data.humanPlayerNumber === 2 ? 2 : 1;
    this.players = { 1: { type: 'human' }, 2: { type: 'human' } };
    this.aiInstances = { 1: null, 2: null };
    this.totalMovesPlayed = 0;
    this.panOffset = { u: 0, v: 0 };
    this.zoomLevel = 1.0;
    this.isPinching = false;
    this.gestureHadMultiTouch = false;
    this.isDragging = false;
    this.hasMovedBeyondThreshold = false;
    this.tutorialActive = false;
    this.tutorialStep = 0;
    this.tutorialRotateCount = 0;
    this.gridManager.reset();
    this.gridManager.winRounds = { 1: 0, 2: 0 };
  }

  async create() {
    console.log(`Creating GameScene in mode: ${this.gameMode} (difficulty: ${this.aiDifficulty}, human: P${this.humanPlayerNumber})`);

    // --- Function to handle AI initialization (including preload check) ---
    const initializeAI = async (playerNumber, difficultyOverride = null) => {
      const diff = (difficultyOverride || this.aiDifficulty || 'medium').toLowerCase();
      console.log(`Initializing AI Player ${playerNumber} (difficulty: ${diff})...`);
      this.players[playerNumber] = { type: 'ai', difficulty: diff };
      const newAI = new AIPlayer(this.gridManager, diff, playerNumber);
      this.aiInstances[playerNumber] = newAI;

      // Check if preloading was attempted
      const loadPromise = this.registry.get('networkLoadPromise');
      const tempAI = this.registry.get('tempAIInstanceForStatus');

      let loadedSuccessfully = false;
      if (loadPromise && tempAI) {
          console.log(`AI ${playerNumber}: Found preload promise. Waiting for it to settle...`);
          try {
              await loadPromise;

              if (tempAI.isNetworkLoaded) {
                  console.log(`AI ${playerNumber}: Preload successful. Copying network state...`);
                  const networkState = tempAI.exportNetwork();
                  if (networkState) {
                      newAI.loadNetworkJSON(networkState);
                      newAI.isNetworkLoaded = true;
                      loadedSuccessfully = true;
                  } else {
                      console.error(`AI ${playerNumber}: Preload seemed successful but failed to export network state from temporary AI.`);
                  }
              } else {
                    console.log(`AI ${playerNumber}: Preload attempt finished but was unsuccessful or network was not marked as loaded.`);
              }
          } catch (promiseError) {
                console.error(`AI ${playerNumber}: Error occurred while awaiting preload promise:`, promiseError);
          }
      } else {
            console.log(`AI ${playerNumber}: No preload attempt found in registry.`);
      }

      // Fallback: If preloading didn't succeed, try loading normally
      if (!loadedSuccessfully) {
            console.log(`AI ${playerNumber}: Attempting standard network load...`);
            await newAI.importNetwork();
            if (!newAI.isNetworkLoaded) {
                console.warn(`AI Player ${playerNumber} will start untrained.`);
            }
      }
    }; // --- End of initializeAI function ---

    // --- Initialize AI based on gameMode ---
    if (this.gameMode === 'HvAI') {
        const humanNum = this.humanPlayerNumber === 2 ? 2 : 1;
        const aiPlayerNumber = humanNum === 1 ? 2 : 1;
        this.players[humanNum] = { type: 'human' };
        await initializeAI(aiPlayerNumber, this.aiDifficulty);

    } else if (this.gameMode === 'AvR') {
        // AI vs Random: P1 is AI (hard), P2 is Random
        await initializeAI(1, 'hard');
        this.players[2] = { type: 'random' };

    } else if (this.gameMode === 'AvH') {
        // AI vs Heuristic: P1 is AI (hard), P2 is Heuristic
        await initializeAI(1, 'hard');
        this.players[2] = { type: 'heuristic' };

    } else if (this.gameMode === 'AvA') {
        await initializeAI(1, 'hard'); // Initialize P1
        await initializeAI(2, 'hard'); // Initialize P2
    }

    // --- Rest of your create method ---
    console.log("Finished AI initialization. Setting up visuals and input...");

    // Ensure at least 2 touch pointers are active for two-finger pinch-to-zoom on mobile
    this.input.addPointer(2);

    this.calculateGridBounds();
    this.setupBoardLayers();
    this.drawGrid();
    this.redrawTriangles();
    this.setupTouchControls();
    this.setupInput();
    this.updateCursor();
    this.createScoreboard();

    this.createPlaceButton();
    this.createRotateButton(); 
    this.createForfeitButton();
    this.createMenuReturnButton();
    this.createViewControlsHUD();
    this.gameOverText = null;

    // Make sure human input is enabled/disabled correctly at the start
    this.updateInputStateForCurrentPlayer();

    // Add on-screen debug text for mobile debugging (visible only when DEBUG_MODE is enabled)
    this.debugText = this.add.text(12, 102, 'Debug: Initializing...', {
      font: '12px Arial',
      fill: '#FFFF00',
      backgroundColor: '#000000',
      padding: { x: 5, y: 2 }
    }).setDepth(200).setScrollFactor(0).setVisible(Boolean(typeof DEBUG_MODE !== 'undefined' && DEBUG_MODE));
    const modeDetail = this.gameMode === 'HvAI' ? `${this.gameMode} (${this.aiDifficulty.toUpperCase()})` : this.gameMode;
    this.debugText.setText(`Mode: ${modeDetail} | P1: ${this.players[1]?.type || '?'} | P2: ${this.players[2]?.type || '?'}`);

    if (this.gameMode === 'Tutorial') {
      this.startTutorial();
      return;
    }

    // If Player 1 is non-human (e.g., HvAI with Human as P2, or AvR/AvH/AvA), trigger P1's first move
    const p1Type = this.players[this.gridManager.currentPlayer]?.type;
    if (p1Type === 'ai') {
      this.time.delayedCall(150, this.triggerAIMove, [], this);
    } else if (p1Type === 'random' || p1Type === 'heuristic') {
      this.time.delayedCall(150, this.triggerNonHumanMove, [], this);
    }
  }

  enableHumanInput() {
    console.log("Enabling human input");
    // Make buttons interactive again
    this.placeButton.setInteractive(); // [cite: 186]
    this.rotateButton.setInteractive(); // [cite: 177]
    // You might visually change the buttons back from a disabled state here

    // Enable keyboard listeners (Phaser doesn't have a simple scene-wide toggle,
    // so the checks within handlers are primary)
    this.input.keyboard.enabled = true; // Generally keep keyboard enabled, rely on checks

      // Enable direct grid interaction if you add it (e.g., clicking grid cells)
      // this.input.on('pointerdown', this.handleGridClick, this); // Re-add listener if removed
  }
  
  disableHumanInput() {
      console.log("Disabling human input");
      // Disable buttons
      this.placeButton.disableInteractive();
      this.rotateButton.disableInteractive();
      // Visually grey out or change buttons appearance here
  
      // Disable keyboard (less effective than checks in handlers)
      this.input.keyboard.enabled = false; // This might block all keys, be cautious
  
      // Disable direct grid interaction if used
      // this.input.off('pointerdown', this.handleGridClick, this); // Remove listener
  }
  
  updateInputStateForCurrentPlayer() {
      if (this.players[this.gridManager.currentPlayer].type === 'human') {
          this.enableHumanInput();
      } else {
          this.disableHumanInput();
      }
  }

  drawChamferedPanel(gfx, cx, cy, width, height, cut, fillColor, fillAlpha, strokeColor, strokeAlpha = 1.0, lineWidth = 2) {
    const x0 = cx - width / 2;
    const y0 = cy - height / 2;
    const x1 = cx + width / 2;
    const y1 = cy + height / 2;

    const tracePath = () => {
      gfx.beginPath();
      gfx.moveTo(x0 + cut, y0);
      gfx.lineTo(x1 - cut, y0);
      gfx.lineTo(x1, y0 + cut);
      gfx.lineTo(x1, y1 - cut);
      gfx.lineTo(x1 - cut, y1);
      gfx.lineTo(x0 + cut, y1);
      gfx.lineTo(x0, y1 - cut);
      gfx.lineTo(x0, y0 + cut);
      gfx.closePath();
    };

    gfx.clear();
    // Outer neon halo
    gfx.lineStyle(lineWidth + 4, strokeColor, strokeAlpha * 0.22);
    tracePath();
    gfx.strokePath();

    // Dark cyber-glass fill
    gfx.fillStyle(fillColor, fillAlpha);
    tracePath();
    gfx.fillPath();

    // Crisp neon perimeter
    gfx.lineStyle(lineWidth, strokeColor, strokeAlpha);
    tracePath();
    gfx.strokePath();
  }

  createForfeitButton() {
    const btnWidth = 76;
    const btnHeight = 56;
    const buttonX = this.cameras.main.width - 76;
    const buttonY = this.cameras.main.height - 62;

    this.forfeitButton = this.add.graphics().setScrollFactor(0).setDepth(150);
    const drawForfeit = (hover = false) => {
      this.drawChamferedPanel(
        this.forfeitButton,
        buttonX,
        buttonY,
        btnWidth,
        btnHeight,
        10,
        hover ? 0x2A0814 : 0x14060D,
        0.92,
        0xFF2A54,
        hover ? 1.0 : 0.85,
        2
      );
    };
    drawForfeit(false);

    this.add.text(buttonX, buttonY, '⚑', {
      fontFamily: '"Orbitron", "Share Tech Mono", Arial',
      fontSize: '24px',
      color: '#FF4D6D',
      align: 'center'
    }).setOrigin(0.5).setScrollFactor(0).setDepth(151);

    this.forfeitButton.setInteractive(
      new Phaser.Geom.Rectangle(buttonX - btnWidth / 2, buttonY - btnHeight / 2, btnWidth, btnHeight),
      Phaser.Geom.Rectangle.Contains
    );

    this.forfeitButton.on('pointerover', () => drawForfeit(true));
    this.forfeitButton.on('pointerout', () => drawForfeit(false));
    this.forfeitButton.on('pointerdown', () => {
      if (this.tutorialActive) return;
      this.handleForfeit();
    });
  }

  handleForfeit() {
    const winner = this.gridManager.currentPlayer === 1 ? 2 : 1; // [cite: 171]
    this.gridManager.winRounds[winner]++;
    const p1Score = this.gridManager.getLargestContiguous(1);
    const p2Score = this.gridManager.getLargestContiguous(2);
    this.showGameOver(`Player ${this.gridManager.currentPlayer} forfeits!\nPlayer ${winner} wins!\nFinal Score: P1: ${p1Score}, P2: ${p2Score}`); // [cite: 172]
    this.updateScore();
    this.disableHumanInput();
    this.time.delayedCall(3000, () => {
         this.resetGame();
         this.updateInputStateForCurrentPlayer();
         const pType = this.players[this.gridManager.currentPlayer]?.type;
         if (pType === 'ai') {
             this.time.delayedCall(100, this.triggerAIMove, [], this);
         } else if (pType === 'random' || pType === 'heuristic') {
             this.time.delayedCall(100, this.triggerNonHumanMove, [], this);
         }
    });
  }

  createMenuReturnButton() {
    const btnWidth = 114;
    const btnHeight = 40;
    const btnX = 72;
    const btnY = 40;

    this.menuReturnBg = this.add.graphics().setScrollFactor(0).setDepth(150);
    const drawMenuBtn = (hover = false) => {
      this.drawChamferedPanel(
        this.menuReturnBg,
        btnX,
        btnY,
        btnWidth,
        btnHeight,
        8,
        hover ? 0x0D2238 : 0x071120,
        0.92,
        0x00F0FF,
        hover ? 1.0 : 0.75,
        1.8
      );
    };
    drawMenuBtn(false);

    this.menuReturnText = this.add.text(btnX, btnY, '◀ MENU', {
      fontFamily: '"Share Tech Mono", monospace',
      fontSize: '18px',
      fontStyle: 'bold',
      color: '#E6F9FF',
      align: 'center'
    }).setOrigin(0.5).setScrollFactor(0).setDepth(151);

    this.menuReturnBg.setInteractive(
      new Phaser.Geom.Rectangle(btnX - btnWidth / 2, btnY - btnHeight / 2, btnWidth, btnHeight),
      Phaser.Geom.Rectangle.Contains
    );

    this.menuReturnBg.on('pointerdown', () => {
      this.time.removeAllEvents();
      this.scene.start('Menu');
    });

    this.menuReturnBg.on('pointerover', () => drawMenuBtn(true));
    this.menuReturnBg.on('pointerout', () => drawMenuBtn(false));
  }

  createRotateButton() {
    const btnWidth = 76;
    const btnHeight = 56;
    const buttonX = 76;
    const buttonY = this.cameras.main.height - 62;

    this.rotateButton = this.add.graphics().setScrollFactor(0).setDepth(150);
    const drawRotate = (hover = false) => {
      this.drawChamferedPanel(
        this.rotateButton,
        buttonX,
        buttonY,
        btnWidth,
        btnHeight,
        10,
        hover ? 0x0D2644 : 0x071324,
        0.92,
        0x00F0FF,
        hover ? 1.0 : 0.85,
        2
      );
    };
    drawRotate(false);

    this.add.text(buttonX, buttonY, '↻', {
      fontFamily: '"Orbitron", Arial',
      fontSize: '28px',
      color: '#00F0FF',
      align: 'center'
    }).setOrigin(0.5).setScrollFactor(0).setDepth(151);

    this.rotateButton.setInteractive(
      new Phaser.Geom.Rectangle(buttonX - btnWidth / 2, buttonY - btnHeight / 2, btnWidth, btnHeight),
      Phaser.Geom.Rectangle.Contains
    );

    this.rotateButton.on('pointerdown', () => {
      if (this.tutorialActive && !this.tutorialCanInteract('rotate')) return;
      this.gridManager.rotateTriangle();
      this.updateCursor();
      drawRotate(true);
      if (this.tutorialActive) this.onTutorialRotate();
    });

    this.rotateButton.on('pointerup', () => drawRotate(false));
    this.rotateButton.on('pointerover', () => drawRotate(true));
    this.rotateButton.on('pointerout', () => drawRotate(false));
  }

  createPlaceButton() {
    const buttonWidth = 248;
    const buttonHeight = 56;
    const buttonX = this.cameras.main.width / 2;
    const buttonY = this.cameras.main.height - 62;

    this.placeButton = this.add.graphics().setScrollFactor(0).setDepth(150);
    this.drawPlaceButtonState = (hover = false) => {
      const cur = this.gridManager?.currentPlayer || 1;
      const stroke = cur === 1 ? 0xFF2A54 : 0x00F0FF;
      const fill = cur === 1 ? (hover ? 0x2E0916 : 0x19060E) : (hover ? 0x0A2542 : 0x061326);
      this.drawChamferedPanel(
        this.placeButton,
        buttonX,
        buttonY,
        buttonWidth,
        buttonHeight,
        12,
        fill,
        0.94,
        stroke,
        hover ? 1.0 : 0.9,
        2.4
      );
    };
    this.drawPlaceButtonState(false);

    this.placeButtonText = this.add.text(buttonX, buttonY, 'PLACE TRIANGLE', {
      fontFamily: '"Orbitron", "Share Tech Mono", Arial',
      fontSize: '18px',
      fontStyle: 'bold',
      color: '#E6F9FF',
      align: 'center'
    }).setOrigin(0.5).setScrollFactor(0).setDepth(151);

    this.placeButton.setInteractive(
      new Phaser.Geom.Rectangle(
        buttonX - buttonWidth / 2,
        buttonY - buttonHeight / 2,
        buttonWidth,
        buttonHeight
      ),
      Phaser.Geom.Rectangle.Contains
    );

    this.placeButton.on('pointerdown', () => this.placeCurrentTriangle());
    this.placeButton.on('pointerover', () => this.drawPlaceButtonState(true));
    this.placeButton.on('pointerout', () => this.drawPlaceButtonState(false));
  }

  calculateGridBounds() {
    this.gridBounds.x = this.gridOffset.x;
    this.gridBounds.y = this.gridOffset.y;
    this.gridBounds.width = this.gridManager.cols * this.cellSize;
    this.gridBounds.height = this.gridManager.rows * this.cellSize;
  }

  // =========================================================================
  // STEP 1: VIEW-PROJECTION & DRAGGABLE TOROIDAL PLANE (100% Core Isolation)
  // =========================================================================

  /**
   * Forward Projection: maps continuous board space (u, v) in [0, 10] x [0, 10]
   * to screen pixel coordinates (sx, sy) with optional 2.5D tabletop tilt
   * and optional zoom magnification around the board center (5, 5).
   *
   * Note: Pass applyZoom = false when computing fixed viewport frame corners
   * (such as the 10x10 stencil mask and outer tabletop border) so the slab
   * frame stays locked on screen while its contents magnify from 10x10 down to 4x4.
   */
  boardToScreen(u, v, applyZoom = true) {
    const cx = this.gridOffset.x + 5 * this.cellSize;
    const cy = this.gridOffset.y + 5 * this.cellSize;
    const zoom = applyZoom ? (this.zoomLevel || 1.0) : 1.0;
    const du = (u - 5) * zoom;
    const dv = (v - 5) * zoom;

    if (!this.isTiltedView) {
      return {
        x: cx + du * this.cellSize,
        y: cy + dv * this.cellSize
      };
    }

    // 2.5D Tabletop Isometric Skew & Pitch Compression (fits comfortably in 600px width)
    const scaleX = 0.86;
    const skewX = -0.22;
    const skewY = 0.08;
    const scaleY = 0.76;

    return {
      x: cx + (du * scaleX + dv * skewX) * this.cellSize,
      y: cy + (du * skewY + dv * scaleY) * this.cellSize
    };
  }

  /**
   * Inverse Projection: maps screen pixel coordinates (sx, sy) back to
   * continuous board coordinates (u, v).
   * Pass applyZoom = false to query unzoomed viewport window coordinates [0, 10].
   */
  screenToBoard(sx, sy, applyZoom = true) {
    const cx = this.gridOffset.x + 5 * this.cellSize;
    const cy = this.gridOffset.y + 5 * this.cellSize;
    const zoom = applyZoom ? (this.zoomLevel || 1.0) : 1.0;
    const dx = (sx - cx) / this.cellSize;
    const dy = (sy - cy) / this.cellSize;

    if (!this.isTiltedView) {
      return { u: dx / zoom + 5, v: dy / zoom + 5 };
    }

    const scaleX = 0.86;
    const skewX = -0.22;
    const skewY = 0.08;
    const scaleY = 0.76;
    const det = scaleX * scaleY - skewX * skewY;

    const du = (dx * scaleY - dy * skewX) / (det * zoom);
    const dv = (-dx * skewY + dy * scaleX) / (det * zoom);

    return { u: du + 5, v: dv + 5 };
  }

  /**
   * Initializes layered graphics and the exact 10x10 Isometric Stencil Mask
   * (Mode A: Strict Zero Duplication) and optional Ghost Ring layer (Mode B).
   */
  setupBoardLayers() {
    if (this.slabBgGraphics) this.slabBgGraphics.destroy();
    if (this.ghostGraphics) this.ghostGraphics.destroy();
    if (this.triangleGraphics) this.triangleGraphics.destroy();
    if (this.graphics) this.graphics.destroy();
    if (this.borderGraphics) this.borderGraphics.destroy();
    if (this.cursorTriangle) this.cursorTriangle.destroy();
    if (this.maskShape) this.maskShape.destroy();

    // 1. Tabletop Slab Drop-Shadow & Surface Background
    this.slabBgGraphics = this.add.graphics().setDepth(1);
    // 2. Optional 1-Cell Ghost Ring Layer (outside mask, 28% alpha)
    this.ghostGraphics = this.add.graphics().setDepth(2);
    // 3. Primary 10x10 Clipped Triangles Layer (Zero Duplication)
    this.triangleGraphics = this.add.graphics().setDepth(5);
    // 4. Primary 10x10 Clipped Grid Lines & Origin Seam Guides
    this.graphics = this.add.graphics().setDepth(10);
    // 5. Outer Frame Border of the 10x10 Slab
    this.borderGraphics = this.add.graphics().setDepth(20);
    // 6. Placement Cursor Preview (Clipped to 10x10 Slab)
    this.cursorTriangle = this.add.graphics().setDepth(100);

    // Geometry Stencil Mask for exact 10x10 Fundamental Domain
    this.maskShape = this.make.graphics({ x: 0, y: 0, add: false });
    this.boardMask = this.maskShape.createGeometryMask();
    this.triangleGraphics.setMask(this.boardMask);
    this.graphics.setMask(this.boardMask);
    this.cursorTriangle.setMask(this.boardMask);

    this.updateBoardMask();
  }

  /**
   * Updates the 10x10 stencil mask polygon and tabletop slab frame whenever
   * the user toggles between 2.5D Tilt and 2D Flat projection.
   */
  updateBoardMask() {
    if (!this.maskShape) return;
    const p00 = this.boardToScreen(0, 0, false);
    const p10 = this.boardToScreen(10, 0, false);
    const p11 = this.boardToScreen(10, 10, false);
    const p01 = this.boardToScreen(0, 10, false);

    this.maskShape.clear();
    this.maskShape.fillStyle(0xffffff, 1);
    this.maskShape.beginPath();
    this.maskShape.moveTo(p00.x, p00.y);
    this.maskShape.lineTo(p10.x, p10.y);
    this.maskShape.lineTo(p11.x, p11.y);
    this.maskShape.lineTo(p01.x, p01.y);
    this.maskShape.closePath();
    this.maskShape.fillPath();

    // Draw 3D slab thickness / drop-shadow and board surface (Choice 2-A: Deep Cyber-Cyan & Obsidian)
    if (this.slabBgGraphics) {
      this.slabBgGraphics.clear();
      if (this.isTiltedView) {
        const thickness = 12;
        // Bottom-left 3D slab bevel
        this.slabBgGraphics.fillStyle(0x091526, 1);
        this.slabBgGraphics.beginPath();
        this.slabBgGraphics.moveTo(p01.x, p01.y);
        this.slabBgGraphics.lineTo(p11.x, p11.y);
        this.slabBgGraphics.lineTo(p11.x, p11.y + thickness);
        this.slabBgGraphics.lineTo(p01.x, p01.y + thickness);
        this.slabBgGraphics.closePath();
        this.slabBgGraphics.fillPath();

        // Bottom-right 3D slab bevel
        this.slabBgGraphics.fillStyle(0x050D1A, 1);
        this.slabBgGraphics.beginPath();
        this.slabBgGraphics.moveTo(p10.x, p10.y);
        this.slabBgGraphics.lineTo(p11.x, p11.y);
        this.slabBgGraphics.lineTo(p11.x, p11.y + thickness);
        this.slabBgGraphics.lineTo(p10.x, p10.y + thickness);
        this.slabBgGraphics.closePath();
        this.slabBgGraphics.fillPath();

        // Neon cyan bevel edge trace
        this.slabBgGraphics.lineStyle(1.5, 0x00F0FF, 0.45);
        this.slabBgGraphics.beginPath();
        this.slabBgGraphics.moveTo(p01.x, p01.y + thickness);
        this.slabBgGraphics.lineTo(p11.x, p11.y + thickness);
        this.slabBgGraphics.lineTo(p10.x, p10.y + thickness);
        this.slabBgGraphics.strokePath();
      }

      // Main 10x10 board surface fill (Obsidian Cyber-Glass)
      this.slabBgGraphics.fillStyle(0x060B14, 1);
      this.slabBgGraphics.beginPath();
      this.slabBgGraphics.moveTo(p00.x, p00.y);
      this.slabBgGraphics.lineTo(p10.x, p10.y);
      this.slabBgGraphics.lineTo(p11.x, p11.y);
      this.slabBgGraphics.lineTo(p01.x, p01.y);
      this.slabBgGraphics.closePath();
      this.slabBgGraphics.fillPath();
    }
  }

  /**
   * Top-right HUD controls for Step 1:
   * - [Tilt / 2D]: Toggles 2.5D tabletop tilt vs classic 2D flat view
   * - [Ghost]: Toggles Mode A (Strict 10x10 Zero-Duplication) vs Mode B (+1-Cell Ghost Ring)
   * - [Center]: Smoothly resets panOffset (u, v) back to (0, 0)
   */
  createViewControlsHUD() {
    const pillHeight = 40;
    const createPillBtn = (x, y, width, label, onClick) => {
      const bg = this.add.graphics().setScrollFactor(0).setDepth(150);
      const txt = this.add.text(x, y, label, {
        fontFamily: '"Share Tech Mono", monospace',
        fontSize: '16px',
        fontStyle: 'bold',
        color: '#E6F9FF',
        align: 'center'
      }).setOrigin(0.5).setScrollFactor(0).setDepth(151);

      const drawPill = (active, hover = false) => {
        const fill = active ? (hover ? 0x0E2A47 : 0x091D34) : (hover ? 0x0B1829 : 0x060E1A);
        const stroke = active ? 0x00F0FF : 0x1E4E68;
        this.drawChamferedPanel(bg, x, y, width, pillHeight, 7, fill, 0.92, stroke, active ? 0.95 : 0.7, 1.8);
        txt.setColor(active ? '#00F0FF' : '#8AB4C9');
      };

      bg.setInteractive(
        new Phaser.Geom.Rectangle(x - width / 2, y - pillHeight / 2, width, pillHeight),
        Phaser.Geom.Rectangle.Contains
      );
      bg.on('pointerdown', onClick);
      return { bg, txt, drawPill };
    };

    // 1. Tilt / 2D Toggle
    this.tiltBtn = createPillBtn(286, 40, 114, '', () => {
      this.isTiltedView = !this.isTiltedView;
      this.refreshViewControlsHUD();
      this.updateBoardMask();
      this.refreshBoardView();
    });

    // 2. Ghost Ring Toggle (Mode A vs Mode B)
    this.ghostBtn = createPillBtn(412, 40, 120, '', () => {
      this.showGhostRing = !this.showGhostRing;
      this.refreshViewControlsHUD();
      this.refreshBoardView();
    });

    // 3. Center / Reset Pan & Zoom Button
    this.centerBtn = createPillBtn(532, 40, 104, '⌖ CENTER', () => {
      const startU = this.panOffset.u;
      const startV = this.panOffset.v;
      const startZ = this.zoomLevel;
      this.tweens.addCounter({
        from: 0,
        to: 1,
        duration: 240,
        ease: 'Cubic.easeOut',
        onUpdate: (tween) => {
          const t = tween.getValue();
          this.panOffset.u = startU * (1 - t);
          this.panOffset.v = startV * (1 - t);
          this.zoomLevel = startZ + (1.0 - startZ) * t;
          this.refreshBoardView();
        },
        onComplete: () => {
          this.panOffset.u = 0;
          this.panOffset.v = 0;
          this.zoomLevel = 1.0;
          this.refreshBoardView();
        }
      });
    });

    // Helper hint above board (larger for mobile legibility)
    this.panHintText = this.add.text(
      this.cameras.main.width / 2,
      80,
      'DRAG: PAN TORUS  •  PINCH: ZOOM (4×4)  •  TAP: SELECT/ROTATE',
      {
        fontFamily: '"Share Tech Mono", monospace',
        fontSize: '15px',
        fontStyle: 'bold',
        color: '#5CE1E6',
        align: 'center'
      }
    ).setOrigin(0.5).setScrollFactor(0).setDepth(150);

    this.refreshViewControlsHUD();
  }

  refreshViewControlsHUD() {
    if (this.tiltBtn) {
      this.tiltBtn.txt.setText(this.isTiltedView ? '◈ TILT: ON' : '▢ TILT: OFF');
      this.tiltBtn.drawPill(this.isTiltedView);
    }
    if (this.ghostBtn) {
      this.ghostBtn.txt.setText(this.showGhostRing ? '⟳ WRAP: ON' : '⟳ WRAP: OFF');
      this.ghostBtn.drawPill(this.showGhostRing);
    }
    if (this.centerBtn) {
      const isPanned = Math.abs(this.panOffset.u) > 0.01 || Math.abs(this.panOffset.v) > 0.01;
      const isZoomed = Math.abs(this.zoomLevel - 1.0) > 0.01;
      this.centerBtn.txt.setText(isZoomed ? `⌖ ${this.zoomLevel.toFixed(1)}×` : '⌖ CENTER');
      this.centerBtn.drawPill(isPanned || isZoomed);
    }
  }

  refreshBoardView() {
    this.drawGrid();
    this.redrawTriangles();
    this.updateCursor();
    this.refreshViewControlsHUD();
    if (this.tutorialActive) {
      this.drawTutorialHighlights();
    }
  }

  drawGrid() {
    if (!this.graphics) return;
    this.graphics.clear();

    const cols = this.gridManager.cols;
    const rows = this.gridManager.rows;
    const fracU = ((this.panOffset.u % 1) + 1) % 1;
    const fracV = ((this.panOffset.v % 1) + 1) % 1;
    const originU = ((this.panOffset.u % cols) + cols) % cols;
    const originV = ((this.panOffset.v % rows) + rows) % rows;

    // 1. Draw subtle sub-grid diagonal circuit traces inside each cell
    this.graphics.lineStyle(1, 0x082433, 0.55);
    for (let i = -1; i <= cols; i++) {
      const u = i + fracU;
      for (let j = -1; j <= rows; j++) {
        const v = j + fracV;
        const p00 = this.boardToScreen(u, v);
        const p10 = this.boardToScreen(u + 1, v);
        const p11 = this.boardToScreen(u + 1, v + 1);
        const p01 = this.boardToScreen(u, v + 1);
        this.graphics.lineBetween(p00.x, p00.y, p11.x, p11.y);
        this.graphics.lineBetween(p01.x, p01.y, p10.x, p10.y);
      }
    }

    // 2. Draw primary Cyber-Cyan matrix grid lines (Choice 2-A: #0D4054)
    this.graphics.lineStyle(1.8, 0x0D4054, 0.92);
    for (let i = -1; i <= cols + 1; i++) {
      const u = i + fracU;
      const pTop = this.boardToScreen(u, -1);
      const pBot = this.boardToScreen(u, rows + 1);
      this.graphics.lineBetween(pTop.x, pTop.y, pBot.x, pBot.y);
    }
    for (let j = -1; j <= rows + 1; j++) {
      const v = j + fracV;
      const pLeft = this.boardToScreen(-1, v);
      const pRight = this.boardToScreen(cols + 1, v);
      this.graphics.lineBetween(pLeft.x, pLeft.y, pRight.x, pRight.y);
    }

    // 3. Draw glowing micro-nodes at matrix vertex intersections
    this.graphics.fillStyle(0x00F0FF, 0.38);
    for (let i = -1; i <= cols + 1; i++) {
      const u = i + fracU;
      for (let j = -1; j <= rows + 1; j++) {
        const v = j + fracV;
        const pt = this.boardToScreen(u, v);
        this.graphics.fillRect(pt.x - 1.5, pt.y - 1.5, 3, 3);
      }
    }

    // 4. Laser-cyan seam indicator showing where canonical (0,0) wrap boundary sits when panned
    if (Math.abs(this.panOffset.u) > 0.01 || Math.abs(this.panOffset.v) > 0.01) {
      this.graphics.lineStyle(5, 0x00FFCC, 0.16);
      const sTop = this.boardToScreen(originU, 0);
      const sBot = this.boardToScreen(originU, rows);
      this.graphics.lineBetween(sTop.x, sTop.y, sBot.x, sBot.y);
      const sLeft = this.boardToScreen(0, originV);
      const sRight = this.boardToScreen(cols, originV);
      this.graphics.lineBetween(sLeft.x, sLeft.y, sRight.x, sRight.y);

      this.graphics.lineStyle(1.8, 0x00FFCC, 0.65);
      this.graphics.lineBetween(sTop.x, sTop.y, sBot.x, sBot.y);
      this.graphics.lineBetween(sLeft.x, sLeft.y, sRight.x, sRight.y);
    }

    // 5. Draw double-pass neon cyan border (#00F0FF) around the fixed 10x10 viewport slab
    if (this.borderGraphics) {
      this.borderGraphics.clear();
      const p00 = this.boardToScreen(0, 0, false);
      const p10 = this.boardToScreen(cols, 0, false);
      const p11 = this.boardToScreen(cols, rows, false);
      const p01 = this.boardToScreen(0, rows, false);

      // If Ghost Ring is ON and at 1.0x full-board view, also draw the outer 12x12 ghost frame
      if (this.showGhostRing && this.zoomLevel <= 1.01) {
        const g00 = this.boardToScreen(-1, -1, false);
        const g10 = this.boardToScreen(cols + 1, -1, false);
        const g11 = this.boardToScreen(cols + 1, rows + 1, false);
        const g01 = this.boardToScreen(-1, rows + 1, false);
        this.borderGraphics.lineStyle(1.5, 0x00F0FF, 0.28);
        this.borderGraphics.beginPath();
        this.borderGraphics.moveTo(g00.x, g00.y);
        this.borderGraphics.lineTo(g10.x, g10.y);
        this.borderGraphics.lineTo(g11.x, g11.y);
        this.borderGraphics.lineTo(g01.x, g01.y);
        this.borderGraphics.closePath();
        this.borderGraphics.strokePath();
      }

      // Outer neon cyan halo
      this.borderGraphics.lineStyle(8, 0x00F0FF, 0.20);
      this.borderGraphics.beginPath();
      this.borderGraphics.moveTo(p00.x, p00.y);
      this.borderGraphics.lineTo(p10.x, p10.y);
      this.borderGraphics.lineTo(p11.x, p11.y);
      this.borderGraphics.lineTo(p01.x, p01.y);
      this.borderGraphics.closePath();
      this.borderGraphics.strokePath();

      // Core laser cyan frame
      this.borderGraphics.lineStyle(2.5, 0x00F0FF, 0.96);
      this.borderGraphics.beginPath();
      this.borderGraphics.moveTo(p00.x, p00.y);
      this.borderGraphics.lineTo(p10.x, p10.y);
      this.borderGraphics.lineTo(p11.x, p11.y);
      this.borderGraphics.lineTo(p01.x, p01.y);
      this.borderGraphics.closePath();
      this.borderGraphics.strokePath();
    }
  }

  setupInput() {
      // Keyboard controls
      this.keys = this.input.keyboard.addKeys('R,ENTER,LEFT,RIGHT,UP,DOWN');
      
      this.keys.LEFT.on('down', () => {
        if (this.players[this.gridManager.currentPlayer].type === 'human') {
          this.moveCursor(-1, 0)
        }
      });
      this.keys.RIGHT.on('down', () => {
        if (this.players[this.gridManager.currentPlayer].type === 'human') {
          this.moveCursor(1, 0)
        }
      });
      this.keys.UP.on('down', () => {
        if (this.players[this.gridManager.currentPlayer].type === 'human') {
          this.moveCursor(0, -1)
        }
      });
      this.keys.DOWN.on('down', () => {
        if (this.players[this.gridManager.currentPlayer].type === 'human') {
          this.moveCursor(0, 1)
        }
      });
      this.keys.R.on('down', () => {
      if (this.players[this.gridManager.currentPlayer].type === 'human') {
        if (this.tutorialActive && !this.tutorialCanInteract('rotate')) return;
        this.gridManager.rotateTriangle();
        this.updateCursor();
        if (this.tutorialActive) this.onTutorialRotate();
      }
      });
      this.keys.ENTER.on('down', () => {
        if (this.players[this.gridManager.currentPlayer].type === 'human') {
            this.placeCurrentTriangle();
        }
      });

      // Mouse wheel rotation
      this.input.on('wheel', (pointer, deltaX, deltaY) => {
        if (!pointer.isTouch) { // Only for mouse wheel
          if (this.tutorialActive && !this.tutorialCanInteract('rotate')) return;
          const direction = deltaY > 0 ? 1 : -1;
          this.gridManager.rotateTriangle(direction);
          this.updateCursor();
          if (this.tutorialActive) this.onTutorialRotate();
        }
      });
  }

  moveCursor(dx, dy) {
      if (this.tutorialActive && !this.tutorialCanInteract('grid')) return;
      const cols = this.gridManager.cols;
      const rows = this.gridManager.rows;
      this.cursorPos.x = ((this.cursorPos.x + dx) % cols + cols) % cols;
      this.cursorPos.y = ((this.cursorPos.y + dy) % rows + rows) % rows;
      this.updateCursor();
  }

  /**
   * Helper: returns the 3 projected screen vertices of a triangle at
   * continuous board cell (u, v) and rotation r in {0, 1, 2, 3}.
   */
  getProjectedTriangleVertices(u, v, rotation) {
      const p00 = this.boardToScreen(u, v);
      const p10 = this.boardToScreen(u + 1, v);
      const p11 = this.boardToScreen(u + 1, v + 1);
      const p01 = this.boardToScreen(u, v + 1);

      switch (rotation) {
        case 0: // Top-Left triangle (╱)
          return [p01, p10, p00];
        case 1: // Bottom-Left triangle (╲)
          return [p00, p01, p11];
        case 2: // Bottom-Right triangle (╱)
          return [p11, p01, p10];
        case 3: // Top-Right triangle (╲)
          return [p10, p00, p11];
        default:
          return [p00, p10, p01];
      }
  }

  updateCursor() {
      if (!this.cursorTriangle) return;
      this.cursorTriangle.clear();
      if (this.hideTutorialCursor) return;

      const isP1 = this.gridManager.currentPlayer === 1;
      const rimColor = isP1 ? 0xFF2A54 : 0x00F0FF;
      const coreColor = isP1 ? 0xFF99AA : 0xB8FFFF;

      const cols = this.gridManager.cols;
      const rows = this.gridManager.rows;
      const baseU = ((this.cursorPos.x + this.panOffset.u) % cols + cols) % cols;
      const baseV = ((this.cursorPos.y + this.panOffset.v) % rows + rows) % rows;

      // Render shifts [-10, 0, +10] inside the 10x10 stencil mask so a cell straddling
      // the boundary during smooth fractional panning is seamlessly completed on both sides
      for (let kx = -1; kx <= 1; kx++) {
        const u = baseU + kx * cols;
        if (u + 1 < 0 || u > cols) continue;
        for (let ky = -1; ky <= 1; ky++) {
          const v = baseV + ky * rows;
          if (v + 1 < 0 || v > rows) continue;
          const [v0, v1, v2] = this.getProjectedTriangleVertices(u, v, this.gridManager.selectedRotation);

          // 1. Subtle holographic target preview fill
          this.cursorTriangle.fillStyle(rimColor, 0.18);
          this.cursorTriangle.beginPath();
          this.cursorTriangle.moveTo(v0.x, v0.y);
          this.cursorTriangle.lineTo(v1.x, v1.y);
          this.cursorTriangle.lineTo(v2.x, v2.y);
          this.cursorTriangle.closePath();
          this.cursorTriangle.fillPath();

          // 2. Wide outer neon halo
          this.cursorTriangle.lineStyle(6.5, rimColor, 0.35);
          this.cursorTriangle.beginPath();
          this.cursorTriangle.moveTo(v0.x, v0.y);
          this.cursorTriangle.lineTo(v1.x, v1.y);
          this.cursorTriangle.lineTo(v2.x, v2.y);
          this.cursorTriangle.closePath();
          this.cursorTriangle.strokePath();

          // 3. Crisp laser-hot filament rim
          this.cursorTriangle.lineStyle(2.5, coreColor, 0.98);
          this.cursorTriangle.beginPath();
          this.cursorTriangle.moveTo(v0.x, v0.y);
          this.cursorTriangle.lineTo(v1.x, v1.y);
          this.cursorTriangle.lineTo(v2.x, v2.y);
          this.cursorTriangle.closePath();
          this.cursorTriangle.strokePath();
        }
      }
  }

  placeCurrentTriangle() {
    if (this.tutorialActive) {
      this.handleTutorialPlace();
      return;
    }

    if (this.players[this.gridManager.currentPlayer].type !== 'human') {
      console.log("Not human player's turn.");
      return;
    }
    const success = this.gridManager.placeTriangle(
      this.cursorPos.x,
      this.cursorPos.y
    );

    if (success) {
      this.totalMovesPlayed++;
      // Claim enclosed triangles and calculate extra turns
      const placedTriangles = 1; // Initial placement
      const enclosedTriangles = this.gridManager.claimEnclosedTriangles();
      const totalClaimed = placedTriangles + enclosedTriangles;

      if (enclosedTriangles > 0) {
        this.gridManager.extraTurns += Math.floor((1 + enclosedTriangles) / 2);
        console.log(`Player ${this.gridManager.currentPlayer} enclosed ${enclosedTriangles} triangle(s) and gets extra turn(s).`);
      }

      // Update UI and turn logic
      this.redrawTriangles();
      this.updateScore();
      if (this.gridManager.isGridFull()) { // [cite: 98, 219]
          this.handleGameOver();
      } else {
          this.handleTurnSwitch(); // Only switch turn if game is not over
      }

    } else {
      this.cameras.main.flash(200, 255, 0, 0); // Red flash on invalid
    }
  }

  handleGameOver() {
    const p1Score = this.gridManager.getLargestContiguous(1); // [cite: 138, 220]
    const p2Score = this.gridManager.getLargestContiguous(2); // [cite: 138, 220]
    let message = `Game Over!\nP1: ${p1Score}, P2: ${p2Score}\n`; // [cite: 221]

    if (p1Score > p2Score) {
        this.gridManager.winRounds[1]++; // [cite: 222]
        message += "Player 1 wins!"; // [cite: 222]
    } else if (p2Score > p1Score) {
        this.gridManager.winRounds[2]++; // [cite: 223]
        message += "Player 2 wins!"; // [cite: 223]
    } else {
        message += "It's a draw!"; // [cite: 221]
    }

    this.showGameOver(message); // [cite: 224, 226]
    this.updateScore(); // Ensure final scores are displayed [cite: 250]

    // Disable all input during game over message
    this.disableHumanInput();

    // Decide what happens next: Reset or go to menu?
    // For now, reset after delay [cite: 224]
    this.time.delayedCall(3000, () => {
      this.resetGame(); // [cite: 228]
      // After reset, need to re-check game mode and trigger AI if needed
      this.updateInputStateForCurrentPlayer();
      const pType = this.players[this.gridManager.currentPlayer]?.type;
      if (pType === 'ai') {
          this.time.delayedCall(100, this.triggerAIMove, [], this);
      } else if (pType === 'random' || pType === 'heuristic') {
          this.time.delayedCall(100, this.triggerNonHumanMove, [], this);
      }
    });
  }
  showGameOver(message) {
    if (this.gameOverText) this.gameOverText.destroy();
    if (this.gameOverPanel) this.gameOverPanel.destroy();

    const cx = this.cameras.main.width / 2;
    const cy = this.cameras.main.height / 2;

    this.gameOverPanel = this.add.graphics().setScrollFactor(0).setDepth(249);
    this.drawChamferedPanel(this.gameOverPanel, cx, cy, 440, 175, 16, 0x050C18, 0.96, 0x00F0FF, 1.0, 2.5);

    this.gameOverText = this.add.text(
      cx,
      cy,
      message,
      {
        fontFamily: '"Orbitron", "Share Tech Mono", Arial',
        fontSize: '24px',
        fontStyle: 'bold',
        color: '#E6F9FF',
        align: 'center',
        lineSpacing: 8
      }
    ).setOrigin(0.5).setScrollFactor(0).setDepth(250);
  }

  resetGame() {
    this.gridManager.reset();
    this.totalMovesPlayed = 0;
    this.redrawTriangles();
    this.updateScore();
    this.updateCursor();
    if (this.gameOverText) this.gameOverText.destroy();
    if (this.gameOverPanel) this.gameOverPanel.destroy();
  }

  handleTurnSwitch() {
    console.log(`[handleTurnSwitch] Current: ${this.gridManager.currentPlayer}, Extra turns: ${this.gridManager.extraTurns}`);
    this.debugText.setText(`Turn: P${this.gridManager.currentPlayer} | Extra: ${this.gridManager.extraTurns} | P1: ${this.players[1]?.type} | P2: ${this.players[2]?.type}`);
    
    if (this.gridManager.extraTurns > 0) {
        this.gridManager.extraTurns--; // [cite: 229]
        // Current player continues, check player type
        const pType = this.players[this.gridManager.currentPlayer].type;
        console.log(`[handleTurnSwitch] Same player (${this.gridManager.currentPlayer}), type: ${pType}`);
        this.debugText.setText(`P${this.gridManager.currentPlayer} (${pType}) extra turn`);
        if (pType === 'ai') {
             this.time.delayedCall(100, this.triggerAIMove, [], this); // AI takes another turn
        } else if (pType === 'random' || pType === 'heuristic') {
             this.time.delayedCall(100, this.triggerNonHumanMove, [], this);
        } else {
             this.enableHumanInput(); // Human player
        }
    } else {
        // Switch player
        this.gridManager.currentPlayer = this.gridManager.currentPlayer === 1 ? 2 : 1; // [cite: 230]
        this.updateCursor(); // Update cursor color [cite: 204]
        this.updateScore(); // Refresh turn indicator in HUD

        // Check the type of the NEW current player
        const pType = this.players[this.gridManager.currentPlayer].type;
        console.log(`[handleTurnSwitch] Switched to player ${this.gridManager.currentPlayer}, type: ${pType}`);
        this.debugText.setText(`SWITCHED: P${this.gridManager.currentPlayer} (${pType})`);
        if (pType === 'ai') {
             this.disableHumanInput(); // Disable human input while AI thinks
             this.time.delayedCall(100, this.triggerAIMove, [], this); // Start AI's turn
        } else if (pType === 'random' || pType === 'heuristic') {
             this.disableHumanInput();
             console.log(`[handleTurnSwitch] Scheduling triggerNonHumanMove for ${pType} player`);
             this.debugText.setText(`CALLING: triggerNonHumanMove for P${this.gridManager.currentPlayer} (${pType})`);
             this.time.delayedCall(100, this.triggerNonHumanMove, [], this);
        } else {
             this.enableHumanInput(); // Human player
        }
    }
  }

  redrawTriangles() {
    if (!this.triangleGraphics) return;
    this.triangleGraphics.clear();
    if (this.ghostGraphics) this.ghostGraphics.clear();

    const cols = this.gridManager.cols;
    const rows = this.gridManager.rows;

    this.gridManager.grid.forEach((row, y) => {
      row.forEach((cell, x) => {
        if (!cell.triangles || cell.triangles.length === 0) return;

        const baseU = ((x + this.panOffset.u) % cols + cols) % cols;
        const baseV = ((y + this.panOffset.v) % rows + rows) % rows;

        cell.triangles.forEach(triangle => {
          // 1. Draw inside the 10x10 Stencil Mask (Mode A: Strict Zero Duplication)
          for (let kx = -1; kx <= 1; kx++) {
            const u = baseU + kx * cols;
            if (u + 1 < 0 || u > cols) continue;
            for (let ky = -1; ky <= 1; ky++) {
              const v = baseV + ky * rows;
              if (v + 1 < 0 || v > rows) continue;
              this.drawProjectedTriangleOnGraphics(this.triangleGraphics, u, v, triangle.rotation, triangle.player, 1.0);
            }
          }

          // 2. Optional Mode B: Draw 1-Cell Translucent Ghost Ring [-1, 11) when at 1.0x full-board view
          if (this.showGhostRing && this.zoomLevel <= 1.01 && this.ghostGraphics) {
            for (let kx = -1; kx <= 1; kx++) {
              const u = baseU + kx * cols;
              if (u + 1 < -1 || u > cols + 1) continue;
              for (let ky = -1; ky <= 1; ky++) {
                const v = baseV + ky * rows;
                if (v + 1 < -1 || v > rows + 1) continue;
                // Only draw copies that extend into the outer 1-cell ghost border
                if (u < 0 || u + 1 > cols || v < 0 || v + 1 > rows) {
                  this.drawProjectedTriangleOnGraphics(this.ghostGraphics, u, v, triangle.rotation, triangle.player, 0.34);
                }
              }
            }
          }
        });
      });
    });
  }

  /**
   * Choice 3-B: High-Opacity Neon Solid Triangle Rendering
   * Slightly insets vertices (~3.5% toward centroid) so adjacent triangles form a crisp
   * cyber-circuit seam, filled at 82% neon saturation with a 100% glowing laser rim.
   */
  drawProjectedTriangleOnGraphics(gfx, u, v, rotation, playerOrColor, alphaScale = 1.0) {
    const [raw0, raw1, raw2] = this.getProjectedTriangleVertices(u, v, rotation);
    const cx = (raw0.x + raw1.x + raw2.x) / 3;
    const cy = (raw0.y + raw1.y + raw2.y) / 3;
    const inset = 0.038;

    const v0 = { x: raw0.x + (cx - raw0.x) * inset, y: raw0.y + (cy - raw0.y) * inset };
    const v1 = { x: raw1.x + (cx - raw1.x) * inset, y: raw1.y + (cy - raw1.y) * inset };
    const v2 = { x: raw2.x + (cx - raw2.x) * inset, y: raw2.y + (cy - raw2.y) * inset };

    const isP1 = playerOrColor === 1 || playerOrColor === 0xff2222;
    const fillColor = isP1 ? 0xE6193C : 0x0066FF;
    const rimColor = isP1 ? 0xFF4D6D : 0x00F0FF;

    // High-opacity neon solid interior (82% base alpha)
    gfx.fillStyle(fillColor, 0.82 * alphaScale);
    gfx.beginPath();
    gfx.moveTo(v0.x, v0.y);
    gfx.lineTo(v1.x, v1.y);
    gfx.lineTo(v2.x, v2.y);
    gfx.closePath();
    gfx.fillPath();

    // Crisp neon circuit rim
    gfx.lineStyle(1.6, rimColor, 0.96 * alphaScale);
    gfx.beginPath();
    gfx.moveTo(v0.x, v0.y);
    gfx.lineTo(v1.x, v1.y);
    gfx.lineTo(v2.x, v2.y);
    gfx.closePath();
    gfx.strokePath();
  }

  // Add this method to GameScene
triggerAIMove() {
  const currentPlayerNum = this.gridManager.currentPlayer;
  const aiInstance = this.aiInstances[currentPlayerNum];
  
  if (!aiInstance || this.players[currentPlayerNum].type !== 'ai') {
      console.error("triggerAIMove called for non-AI player or missing instance.");
      this.debugText.setText(`ERROR: P${currentPlayerNum} not AI!`);
      return;
  }
  
  this.debugText.setText(`P${currentPlayerNum} (AI) thinking...`);
  
  const state = aiInstance._getStateRepresentation();
  let move;
  // 1. Calculate IMMEDIATE rewards for all moves from this state
  // (Assumes calculateImmediateRewardVector returns rewards scaled approx 0-1)
  const immediateRewardVector = aiInstance.calculateImmediateRewardVector(state, currentPlayerNum, this.gridManager);

  if (!aiInstance || this.players[currentPlayerNum].type !== 'ai') {
      console.error("triggerAIMove called for non-AI player or missing instance.");
      return;
  }

  console.log(`AI Player ${currentPlayerNum} is thinking...`);
  // Use a short delay to simulate thinking and prevent visual glitches
  this.time.delayedCall(300, () => { // Delay can be adjusted

    // === Training-aligned move selection (80% network, 10% random, 10% greedy) ===
    // Opening diversity: 50% random for first 3 moves
    const isOpeningPhase = this.totalMovesPlayed < 3;
    const openingRandomFactor = isOpeningPhase ? 0.50 : 0;
    
    const rand = Math.random();
    if (this.totalMovesPlayed < 2) {
      // Diversify opening trajectories (first 1 move per player = 2 plies, matching eval games)
      console.log(`AI Player ${currentPlayerNum} opening random move (ply ${this.totalMovesPlayed + 1}/2)...`);
      const validMoves = this.gridManager.getValidMoves();
      move = validMoves.length > 0 ? validMoves[Math.floor(Math.random() * validMoves.length)] : null;
      this.debugText.setText(`Mode: ${this.gameMode} | P${currentPlayerNum} Opening Random (ply ${this.totalMovesPlayed + 1}/2)`);
    } else if (aiInstance.networkType === 'ConvNetwork') {
      // Respect difficulty-configured explorationFactor & useSafetyFilter inside aiInstance.predictMove()
      console.log(`AI Player ${currentPlayerNum} ConvNetwork (${aiInstance.difficulty}) prediction...`);
      move = aiInstance.predictMove();
      if (aiInstance.lastPredictedValue !== undefined) {
        this.debugText.setText(`Mode: ${this.gameMode} | P${currentPlayerNum} (${aiInstance.difficulty}) Val: ${aiInstance.lastPredictedValue.toFixed(2)}`);
      }
    } else if (rand < openingRandomFactor) {
      // Opening diversity: random move
      console.log(`AI Player ${currentPlayerNum} opening random move...`);
      const validMoves = this.gridManager.getValidMoves();
      move = validMoves.length > 0 ? validMoves[Math.floor(Math.random() * validMoves.length)] : null;
    } else if (rand < 0.10 + openingRandomFactor) {
      // Random exploration (10%)
      console.log(`AI Player ${currentPlayerNum} exploring randomly...`);
      const validMoves = this.gridManager.getValidMoves();
      move = validMoves.length > 0 ? validMoves[Math.floor(Math.random() * validMoves.length)] : null;
    } else if (rand < 0.20 + openingRandomFactor) {
      // Greedy immediate reward (10%)
      console.log(`AI Player ${currentPlayerNum} greedy heuristic move...`);
      let bestMoveIndex = -1;
      let maxImmediateReward = -Infinity;
      for (let i = 0; i < immediateRewardVector.length; i++) {
        if (immediateRewardVector[i] > 0.01 && immediateRewardVector[i] > maxImmediateReward) {
          maxImmediateReward = immediateRewardVector[i];
          bestMoveIndex = i;
        }
      }
      if (bestMoveIndex !== -1) {
        const rotation = bestMoveIndex % 4;
        const cellIndex = Math.floor(bestMoveIndex / 4);
        const x = cellIndex % this.gridManager.cols;
        const y = Math.floor(cellIndex / this.gridManager.cols);
        move = { x, y, rotation };
      } else {
        const validMoves = this.gridManager.getValidMoves();
        move = validMoves.length > 0 ? validMoves[Math.floor(Math.random() * validMoves.length)] : null;
      }
    } else {
      // Network prediction (80%)
      console.log(`AI Player ${currentPlayerNum} network prediction...`);
      move = aiInstance.predictMove();
    }

      if (move && typeof move.x !== 'undefined' && typeof move.y !== 'undefined' && typeof move.rotation !== 'undefined') {
          console.log(`AI Player ${currentPlayerNum} chose: (${move.x}, ${move.y}), Rot: ${move.rotation}`);
          // Apply the move
          this.gridManager.selectedRotation = move.rotation;
          // Optional: Move cursor for visual feedback
          this.cursorPos.x = move.x;
          this.cursorPos.y = move.y;
          this.updateCursor(); // Update visual cursor [cite: 204]

          // Place the triangle after another small delay for visual pacing
          this.time.delayedCall(150, () => {
              let success = this.gridManager.placeTriangle(move.x, move.y); // [cite: 114, 215]
              
              // FIX: Retry with random valid move if prediction failed
              if (!success) {
                console.warn(`AI: Predicted move invalid, retrying with random valid move`);
                const retryMoves = this.gridManager.getValidMoves();
                if (retryMoves.length > 0) {
                  move = retryMoves[Math.floor(Math.random() * retryMoves.length)];
                  this.gridManager.selectedRotation = move.rotation;
                  success = this.gridManager.placeTriangle(move.x, move.y);
                  console.log(`AI retry: (${move.x},${move.y}) r${move.rotation} = ${success}`);
                }
              }
              
              if (success) {
                  this.totalMovesPlayed++;
                  const enclosed = this.gridManager.claimEnclosedTriangles(); // [cite: 139, 217]
                  if (enclosed > 0) {
                    this.gridManager.extraTurns += Math.floor((1 + enclosed) / 2);
                    console.log(`AI Player ${this.gridManager.currentPlayer} enclosed ${enclosed} triangle(s) and gets extra turn(s).`);
                  }
                  this.redrawTriangles(); // [cite: 231]
                  this.updateScore(); // [cite: 250]

                  // Check for game over AFTER placing and scoring
                  if (this.gridManager.isGridFull()) { // [cite: 98, 219]
                      this.handleGameOver(); // Use a separate function for game over logic
                  } else {
                      this.handleTurnSwitch(); // Proceed to next turn/extra turn
                  }
              } else {
                  console.error(`AI Error: Predicted move (${move.x}, ${move.y}, ${move.rotation}) failed placement.`); // [cite: 72]
                  // Decide how to handle AI failure (e.g., skip turn, try random?)
                  this.gridManager.extraTurns = 0; // AI loses extra turns on invalid move
                  this.handleTurnSwitch(); // Move to the next player
              }
          }, [], this);

      } else {
          console.error(`AI Player ${currentPlayerNum} failed to produce a valid move object.`); // [cite: 52, 70]
          // Handle case where AI can't move (grid full or error?)
          if (this.gridManager.isGridFull()) {
                this.handleGameOver();
          } else {
                console.warn("AI forfeiting turn due to inability to move.");
                this.gridManager.extraTurns = 0;
                this.handleTurnSwitch(); // Skip turn
          }
      }
    }, [], this);
  }

  // Handle Random and Heuristic player moves
  triggerNonHumanMove() {
    const currentPlayerNum = this.gridManager.currentPlayer;
    
    if (!this.players[currentPlayerNum]) {
      console.error(`[triggerNonHumanMove] Player ${currentPlayerNum} not found in players object!`);
      console.log(`[triggerNonHumanMove] Players:`, JSON.stringify(this.players));
      this.debugText.setText(`ERROR: P${currentPlayerNum} not in players object!`);
      return;
    }
    
    const playerType = this.players[currentPlayerNum].type;
    let move;

    console.log(`[triggerNonHumanMove] ${playerType} Player ${currentPlayerNum} is thinking...`);
    this.debugText.setText(`P${currentPlayerNum} (${playerType}) thinking...`);

    // Use Phaser's delayedCall for timing consistency
    this.time.delayedCall(200, () => {
      try {
        this.debugText.setText(`P${currentPlayerNum} TIMEOUT FIRED`);
        
        // FIX: Compute valid moves INSIDE the timeout, right before use
        const validMoves = this.gridManager.getValidMoves();
        
        if (this.totalMovesPlayed < 2) {
          // Diversify opening trajectories (first 1 move per player = 2 plies, matching eval games)
          console.log(`${playerType} Player ${currentPlayerNum} opening random move (ply ${this.totalMovesPlayed + 1}/2)...`);
          if (validMoves && validMoves.length > 0) {
            move = validMoves[Math.floor(Math.random() * validMoves.length)];
            this.debugText.setText(`P${currentPlayerNum} (${playerType}) Opening Random (ply ${this.totalMovesPlayed + 1}/2)`);
          }
        } else if (playerType === 'random') {
          // Random player: select random valid move
          this.debugText.setText(`P${currentPlayerNum} random: ${validMoves?.length || 'NULL'} valid moves`);
          if (validMoves && validMoves.length > 0) {
            move = validMoves[Math.floor(Math.random() * validMoves.length)];
          } else {
            this.debugText.setText(`ERROR: No valid moves for random!`);
            console.error(`[random] No valid moves available!`);
            return; // Can't make a move
          }
        } else if (playerType === 'heuristic') {
        // Heuristic player: use greedy heuristic
        // Create a temporary HeuristicPlayer instance
        try {
          const heuristicPlayer = new HeuristicPlayer(currentPlayerNum, { explorationFactor: 0 });
          this.debugText.setText(`P${currentPlayerNum} heuristic: calling getMove...`);
          move = heuristicPlayer.getMove(this.gridManager);
          this.debugText.setText(`P${currentPlayerNum} heuristic: move=${JSON.stringify(move)}`);
        } catch (err) {
          this.debugText.setText(`ERROR heuristic: ${err.message}`);
          console.error('Heuristic error:', err);
        }
      }

      if (move && typeof move.x !== 'undefined' && typeof move.y !== 'undefined' && typeof move.rotation !== 'undefined') {
        console.log(`${playerType} Player ${currentPlayerNum} chose: (${move.x}, ${move.y}), Rot: ${move.rotation}`);
        this.debugText.setText(`P${currentPlayerNum} (${playerType}): (${move.x},${move.y}) rot=${move.rotation}`);

        // Apply the move
        this.gridManager.selectedRotation = move.rotation;
        this.cursorPos.x = move.x;
        this.cursorPos.y = move.y;
        this.updateCursor();

        this.debugText.setText(`P${currentPlayerNum} placing at (${move.x},${move.y})...`);
        
        this.time.delayedCall(100, () => {
          this.debugText.setText(`P${currentPlayerNum} calling placeTriangle...`);
          // FIX: Re-set selectedRotation right before placing (prevents race condition)
          this.gridManager.selectedRotation = move.rotation;
          let success = this.gridManager.placeTriangle(move.x, move.y);
          this.debugText.setText(`P${currentPlayerNum} placeTriangle=${success}`);
          
          // FIX: Retry with random valid move if first attempt failed
          if (!success) {
            console.warn(`${playerType}: Move (${move.x},${move.y}) r${move.rotation} invalid, retrying...`);
            const retryMoves = this.gridManager.getValidMoves();
            console.log(`${playerType}: ${retryMoves?.length || 0} valid moves available`);
            if (retryMoves && retryMoves.length > 0) {
              move = retryMoves[Math.floor(Math.random() * retryMoves.length)];
              this.gridManager.selectedRotation = move.rotation;
              success = this.gridManager.placeTriangle(move.x, move.y);
              console.log(`${playerType} retry: (${move.x},${move.y}) r${move.rotation} = ${success}`);
            }
          }
          
          if (success) {
            this.totalMovesPlayed++;
            const enclosed = this.gridManager.claimEnclosedTriangles();
            if (enclosed > 0) {
              this.gridManager.extraTurns += Math.floor((1 + enclosed) / 2);
              console.log(`${playerType} Player ${this.gridManager.currentPlayer} enclosed ${enclosed} triangle(s) and gets extra turn(s).`);
            }
            this.redrawTriangles();
            this.updateScore();

            if (this.gridManager.isGridFull()) {
              this.handleGameOver();
            } else {
              this.handleTurnSwitch();
            }
          } else {
            console.error(`${playerType} Error: Move (${move.x}, ${move.y}, ${move.rotation}) failed placement.`);
            this.debugText.setText(`ERROR: ${playerType} move failed!`);
            this.gridManager.extraTurns = 0;
            this.handleTurnSwitch();
          }
        }, [], this);
      } else {
        console.error(`${playerType} Player ${currentPlayerNum} failed to produce a valid move.`);
        this.debugText.setText(`ERROR: ${playerType} failed to move!`);
        if (this.gridManager.isGridFull()) {
          this.handleGameOver();
        } else {
          this.gridManager.extraTurns = 0;
          this.handleTurnSwitch();
        }
      }
    } catch (err) {
      this.debugText.setText(`ERROR in timeout: ${err.message}`);
      console.error('Error in triggerNonHumanMove timeout:', err);
    }
    }, [], this);
  }

  drawTriangle(x, y, rotation, player) {
      const cols = this.gridManager.cols;
      const rows = this.gridManager.rows;
      const u = ((x + this.panOffset.u) % cols + cols) % cols;
      const v = ((y + this.panOffset.v) % rows + rows) % rows;
      this.drawProjectedTriangleOnGraphics(this.triangleGraphics, u, v, rotation, player, 1.0);
  }

  createScoreboard() {
    this.scorePanelBg = this.add.graphics().setScrollFactor(0).setDepth(149);
    this.scoreText = this.add.text(
      this.cameras.main.width / 2,
      676,
      '',
      {
        fontFamily: '"Share Tech Mono", monospace',
        fontSize: '18px',
        color: '#E6F9FF',
        align: 'center',
        lineSpacing: 4
      }
    ).setOrigin(0.5).setScrollFactor(0).setDepth(150);
    this.updateScore();
  }

  updateScore() {
    if (!this.scoreText) return;
    const p1Score = this.gridManager.getLargestContiguous(1);
    const p2Score = this.gridManager.getLargestContiguous(2);
    const cur = this.gridManager.currentPlayer;
    const p1Turn = cur === 1 ? '▶ ' : '  ';
    const p2Turn = cur === 2 ? '▶ ' : '  ';
    this.scoreText.setText([
      `${p1Turn}P1 [ARES RED] : ${String(p1Score).padStart(3, ' ')}   | WINS: ${this.gridManager.winRounds[1]}`,
      `${p2Turn}P2 [TRON BLU] : ${String(p2Score).padStart(3, ' ')}   | WINS: ${this.gridManager.winRounds[2]}`
    ].join('\n'));

    if (this.scorePanelBg) {
      const borderStroke = cur === 1 ? 0xFF2A54 : 0x00F0FF;
      this.drawChamferedPanel(
        this.scorePanelBg,
        this.cameras.main.width / 2,
        676,
        430,
        58,
        10,
        0x060E1B,
        0.92,
        borderStroke,
        0.85,
        1.8
      );
    }
    if (this.drawPlaceButtonState) {
      this.drawPlaceButtonState(false);
    }
  }

  /**
   * Step 1 Input Disambiguation + Two-Finger Pinch-to-Zoom:
   * - Two-Finger Pinch on the board smoothly zooms between 1.0x (full 10x10 board)
   *   and 2.5x (4x4 grid area), anchoring the logical toroidal point under the
   *   midpoint of both fingers so you can pinch-zoom and pan simultaneously.
   * - Single-Finger Drag >= 8px on the board smoothly pans the continuous toroidal
   *   offset (u, v) like a conveyor belt without duplicating pieces inside the mask.
   * - Tap / Click (< 8px movement) selects the tapped logical cell (gx, gy) (or
   *   rotates if tapping the already-selected cell).
   * - Hovering with a mouse updates the cursor preview in real time.
   */
  setupTouchControls() {
    const DRAG_THRESHOLD_PX = 8;

    const getTwoTouchPointers = () => {
      const p1 = this.input.pointer1;
      const p2 = this.input.pointer2;
      if (p1 && p2 && p1.isDown && p2.isDown) {
        return [p1, p2];
      }
      return null;
    };

    const getRemainingDownPointer = (releasedPointer) => {
      const candidates = [this.input.pointer1, this.input.pointer2, this.input.pointer3];
      for (const p of candidates) {
        if (p && p !== releasedPointer && p.isDown) {
          return p;
        }
      }
      return null;
    };

    const handlePointerRelease = (pointer) => {
      // If any touch finger was part of a multi-touch/pinch gesture, suppress cursor tap
      // until EVERY finger has been lifted off the screen
      if (this.gestureHadMultiTouch) {
        this.isPinching = false;
        const remaining = getRemainingDownPointer(pointer);
        if (remaining) {
          // Allow the remaining finger to continue panning smoothly, but keep tap suppressed
          this.isDragging = true;
          this.hasMovedBeyondThreshold = true;
          this.dragStart.x = remaining.x;
          this.dragStart.y = remaining.y;
          this.panStart.u = this.panOffset.u;
          this.panStart.v = this.panOffset.v;
          this.updateCursor();
          return;
        }
        // All fingers are now up after a pinch/zoom gesture: reset state without moving cursor
        this.gestureHadMultiTouch = false;
        this.isDragging = false;
        this.hasMovedBeyondThreshold = false;
        this.updateCursor();
        return;
      }

      if (this.isDragging) {
        const wasPan = this.hasMovedBeyondThreshold;
        this.isDragging = false;
        this.isPinching = false;
        this.hasMovedBeyondThreshold = false;

        if (!wasPan && this.isWithinGrid(pointer.x, pointer.y)) {
          if (this.players[this.gridManager.currentPlayer]?.type === 'human') {
            const logical = this.screenToLogicalCell(pointer.x, pointer.y);
            if (logical.gx === this.cursorPos.x && logical.gy === this.cursorPos.y) {
              if (!this.tutorialActive || this.tutorialCanInteract('rotate')) {
                // Tapping the already-selected cell rotates the preview triangle
                this.gridManager.rotateTriangle(1);
                if (this.tutorialActive) this.onTutorialRotate();
              }
            } else if (!this.tutorialActive || this.tutorialCanInteract('grid')) {
              this.cursorPos.x = logical.gx;
              this.cursorPos.y = logical.gy;
            }
          }
        }
        this.updateCursor();
      }
    };

    this.input.on('pointerup', handlePointerRelease);
    this.input.on('pointerupoutside', handlePointerRelease);

    const startPinchGesture = (p1, p2) => {
      const dist = Phaser.Math.Distance.Between(p1.x, p1.y, p2.x, p2.y);
      this.gestureHadMultiTouch = true; // Mark entire touch sequence as multi-touch (never trigger tap on release)
      this.hasMovedBeyondThreshold = true;
      this.isDragging = true;
      if (dist < 10) return;
      this.isPinching = true;
      this.pinchStartDist = dist;
      this.pinchStartZoom = this.zoomLevel;

      const midX = (p1.x + p2.x) * 0.5;
      const midY = (p1.y + p2.y) * 0.5;
      const midBoard = this.screenToBoard(midX, midY, true);
      // Continuous toroidal coordinate under finger midpoint before zoom change
      this.pinchAnchorToroidal.x = midBoard.u - this.panOffset.u;
      this.pinchAnchorToroidal.y = midBoard.v - this.panOffset.v;
    };

    this.input.on('pointerdown', (pointer) => {
      const twoPointers = getTwoTouchPointers();
      if (twoPointers) {
        startPinchGesture(twoPointers[0], twoPointers[1]);
        return;
      }

      if (this.isWithinGrid(pointer.x, pointer.y)) {
        this.isDragging = true;
        this.isPinching = false;
        this.gestureHadMultiTouch = false;
        this.hasMovedBeyondThreshold = false;
        this.dragStart.x = pointer.x;
        this.dragStart.y = pointer.y;
        this.panStart.u = this.panOffset.u;
        this.panStart.v = this.panOffset.v;
      }
    });

    this.input.on('pointermove', (pointer) => {
      const twoPointers = getTwoTouchPointers();
      if (twoPointers) {
        if (!this.isPinching) {
          startPinchGesture(twoPointers[0], twoPointers[1]);
        } else {
          const [p1, p2] = twoPointers;
          const dist = Phaser.Math.Distance.Between(p1.x, p1.y, p2.x, p2.y);
          if (this.pinchStartDist > 10 && dist > 10) {
            const rawZoom = this.pinchStartZoom * (dist / this.pinchStartDist);
            this.zoomLevel = Phaser.Math.Clamp(rawZoom, this.minZoom, this.maxZoom);

            // Recompute board coordinate at current finger midpoint under new zoom
            const midX = (p1.x + p2.x) * 0.5;
            const midY = (p1.y + p2.y) * 0.5;
            const midBoardAfter = this.screenToBoard(midX, midY, true);

            // Adjust panOffset so the exact toroidal point stays locked under the fingers
            const cols = this.gridManager.cols;
            const rows = this.gridManager.rows;
            const newPanU = midBoardAfter.u - this.pinchAnchorToroidal.x;
            const newPanV = midBoardAfter.v - this.pinchAnchorToroidal.y;
            this.panOffset.u = ((newPanU % cols) + cols) % cols;
            this.panOffset.v = ((newPanV % rows) + rows) % rows;

            this.refreshBoardView();
          }
        }
        return;
      }

      if (this.isPinching) {
        // Transition cleanly from 2-finger pinch to 1-finger drag if one finger remains down
        const activePointer = (this.input.pointer1 && this.input.pointer1.isDown)
          ? this.input.pointer1
          : ((this.input.pointer2 && this.input.pointer2.isDown) ? this.input.pointer2 : null);
        this.isPinching = false;
        this.hasMovedBeyondThreshold = true;
        if (activePointer) {
          this.dragStart.x = activePointer.x;
          this.dragStart.y = activePointer.y;
          this.panStart.u = this.panOffset.u;
          this.panStart.v = this.panOffset.v;
        }
        return;
      }

      if (this.isDragging) {
        const dist = Phaser.Math.Distance.Between(
          this.dragStart.x,
          this.dragStart.y,
          pointer.x,
          pointer.y
        );

        if (dist >= DRAG_THRESHOLD_PX) {
          this.hasMovedBeyondThreshold = true;
          const startBoard = this.screenToBoard(this.dragStart.x, this.dragStart.y, true);
          const currBoard = this.screenToBoard(pointer.x, pointer.y, true);
          const du = currBoard.u - startBoard.u;
          const dv = currBoard.v - startBoard.v;

          const cols = this.gridManager.cols;
          const rows = this.gridManager.rows;
          this.panOffset.u = ((this.panStart.u + du) % cols + cols) % cols;
          this.panOffset.v = ((this.panStart.v + dv) % rows + rows) % rows;

          this.refreshBoardView();
        }
      } else if (!pointer.isTouch && this.isWithinGrid(pointer.x, pointer.y)) {
        if (this.players[this.gridManager.currentPlayer]?.type === 'human') {
          if (!this.tutorialActive || this.tutorialCanInteract('grid')) {
            this.updateCursorPosition(pointer);
          }
        }
      }
    });
  }

  screenToLogicalCell(sx, sy) {
    const { u, v } = this.screenToBoard(sx, sy, true);
    const cols = this.gridManager.cols;
    const rows = this.gridManager.rows;
    const gx = ((Math.floor(u - this.panOffset.u) % cols) + cols) % cols;
    const gy = ((Math.floor(v - this.panOffset.v) % rows) + rows) % rows;
    return { gx, gy, u, v };
  }

  updateCursorPosition(pointer) {
    if (!this.isWithinGrid(pointer.x, pointer.y)) return;
    const { gx, gy } = this.screenToLogicalCell(pointer.x, pointer.y);

    if (gx !== this.cursorPos.x || gy !== this.cursorPos.y) {
      this.cursorPos.x = gx;
      this.cursorPos.y = gy;
      this.updateCursor();
    }
  }

  isWithinGrid(x, y) {
    // Check against the fixed unzoomed viewport slab window [0, 10] x [0, 10]
    const { u, v } = this.screenToBoard(x, y, false);
    const margin = (this.showGhostRing && this.zoomLevel <= 1.01) ? 1 : 0;
    return (
      u >= -margin &&
      u <= this.gridManager.cols + margin &&
      v >= -margin &&
      v <= this.gridManager.rows + margin
    );
  }

  // =========================================================================
  // INTERACTIVE "HOW TO PLAY" TUTORIAL FLOW
  // =========================================================================

  startTutorial() {
    this.tutorialActive = true;
    this.tutorialRotateCount = 0;

    // Seed an opening Blue (P2) triangle at (5, 5, rot 1) so the player can learn legal corner contact
    this.gridManager.reset();
    this.gridManager.placeTriangle(5, 5, 1, 2);
    this.gridManager.currentPlayer = 1;
    this.gridManager.selectedRotation = 0;
    this.cursorPos = { x: 4, y: 5 };
    this.refreshBoardView();
    this.updateScore();

    this.tutorialSteps = [
      {
        type: 'info',
        title: 'WELCOME TO TRIANTICA',
        body: 'MAIN OBJECTIVE: Build the largest single contiguous cluster of connected triangles! When the 10×10 wrap-around board fills up, only your biggest connected territory counts.',
        prompt: 'TAP HERE TO START EXERCISES ▸',
        view: { u: 0.0, v: 0.4, zoom: 1.35 },
        highlights: [
          { x: 300, y: 676, w: 444, h: 70, color: 0x00F0FF }
        ]
      },
      {
        type: 'rotate',
        title: 'EXERCISE 1: ROTATE 4 TIMES',
        body: 'Tap any square on the board to move your red triangle there. Tap the SAME square again (or the ↻ button) to rotate it!\nRotate your red triangle 4 times to proceed.',
        prompt: 'ROTATE 4 TIMES TO PROCEED: 0 / 4',
        view: { u: -0.5, v: 0.5, zoom: 1.5 },
        highlights: [
          { x: 76, y: 738, w: 88, h: 68, color: 0x00FFCC }
        ]
      },
      {
        type: 'place1',
        title: 'EXERCISE 2: MOVE & PLACE',
        body: 'Your red cursor is in an illegal position (not touching any corner). Tap the green target square on the board to move there, rotate to fit inside the green outline, then tap PLACE TRIANGLE.',
        prompt: 'TAP GREEN TARGET → ROTATE TO FIT → TAP PLACE',
        view: { u: -0.5, v: 0.5, zoom: 1.5 },
        startCursor: { x: 3, y: 5, rot: 0 },
        targetCell: { x: 4, y: 5, rot: 2 },
        highlights: [
          { x: 300, y: 738, w: 260, h: 68, color: 0x00FFCC }
        ]
      },
      {
        type: 'place2',
        title: 'EXERCISE 3: CLOSE A GAP (EXTRA TURN!)',
        body: 'The yellow-outlined gap has 2 of 3 sides blocked! Tap the green target square above it to move there, rotate to fit inside the green outline, then tap PLACE TRIANGLE.',
        prompt: 'TAP GREEN TARGET → ROTATE TO FIT → TAP PLACE',
        view: { u: -0.5, v: 0.8, zoom: 1.5 },
        startCursor: { x: 3, y: 5, rot: 0 },
        targetCell: { x: 5, y: 4, rot: 1 },
        enclosedCell: { x: 5, y: 5 },
        highlights: [
          { x: 300, y: 738, w: 260, h: 68, color: 0x00FFCC }
        ]
      },
      {
        type: 'info',
        title: 'GAP CLOSED → EXTRA TURN!',
        body: 'You automatically claimed the enclosed triangle AND earned an Extra Turn! Look at the bottom Scoreboard (▶ marks the turn).',
        prompt: 'TAP HERE TO CONTINUE ▸',
        view: { u: -0.5, v: 0.6, zoom: 1.5 },
        highlights: [
          { x: 300, y: 676, w: 444, h: 70, color: 0x00FFCC }
        ]
      },
      {
        type: 'done',
        title: 'VIEW & GAME CONTROLS',
        body: 'Top-right buttons toggle TILT (3D/2D), WRAP (edge preview), or CENTER the board. ⚑ forfeits the current round; only ◀ MENU exits to the main menu.',
        prompt: 'TAP HERE TO FINISH & RETURN TO MENU ▸',
        view: { u: 0, v: 0, zoom: 1.0 },
        highlights: [
          { x: 410, y: 40, w: 368, h: 52, color: 0x00F0FF },
          { x: 72, y: 40, w: 126, h: 52, color: 0x00FFCC },
          { x: 524, y: 738, w: 88, h: 68, color: 0xFF2A54 }
        ]
      }
    ];

    this.createTutorialUI();
    this.setTutorialStep(0);
  }

  createTutorialUI() {
    this.tutorialHighlightGfx = this.add.graphics().setScrollFactor(0).setDepth(210);

    const cx = this.cameras.main.width / 2;
    const cardY = 200;
    const cardW = 556;
    const cardH = 192;
    this.tutorialCardBounds = { cx, cy: cardY, w: cardW, h: cardH };

    this.tutorialCardGfx = this.add.graphics().setScrollFactor(0).setDepth(220);

    this.tutorialTitleText = this.add.text(cx - cardW / 2 + 20, cardY - cardH / 2 + 15, '', {
      fontFamily: '"Orbitron", "Share Tech Mono", Arial',
      fontSize: '18px',
      fontStyle: 'bold',
      color: '#00FFCC'
    }).setOrigin(0, 0).setScrollFactor(0).setDepth(221);

    // Always-visible Exit Instructions button in top-right of the tutorial card
    const exitW = 108;
    const exitH = 32;
    const exitX = cx + cardW / 2 - exitW / 2 - 14;
    const exitY = cardY - cardH / 2 + 26;

    this.tutorialExitGfx = this.add.graphics().setScrollFactor(0).setDepth(222);
    const drawExitBtn = (hover = false) => {
      this.drawChamferedPanel(
        this.tutorialExitGfx,
        exitX,
        exitY,
        exitW,
        exitH,
        6,
        hover ? 0x380A1C : 0x1F0610,
        0.95,
        0xFF2A54,
        1.0,
        1.6
      );
    };
    drawExitBtn(false);

    this.tutorialExitText = this.add.text(exitX, exitY, '✕ EXIT', {
      fontFamily: '"Share Tech Mono", monospace',
      fontSize: '15px',
      fontStyle: 'bold',
      color: '#FF4D6D'
    }).setOrigin(0.5).setScrollFactor(0).setDepth(223);

    this.tutorialExitGfx.setInteractive(
      new Phaser.Geom.Rectangle(exitX - exitW / 2, exitY - exitH / 2, exitW, exitH),
      Phaser.Geom.Rectangle.Contains
    );
    this.tutorialExitGfx.on('pointerover', () => drawExitBtn(true));
    this.tutorialExitGfx.on('pointerout', () => drawExitBtn(false));
    this.tutorialExitGfx.on('pointerdown', () => {
      this.time.removeAllEvents();
      this.scene.start('Menu');
    });

    this.tutorialBodyText = this.add.text(cx, cardY + 4, '', {
      fontFamily: '"Share Tech Mono", monospace',
      fontSize: '18px',
      color: '#E6F9FF',
      align: 'center',
      wordWrap: { width: cardW - 36 },
      lineSpacing: 5
    }).setOrigin(0.5).setScrollFactor(0).setDepth(221);

    this.tutorialPromptText = this.add.text(cx, cardY + cardH / 2 - 20, '', {
      fontFamily: '"Orbitron", "Share Tech Mono", Arial',
      fontSize: '16px',
      fontStyle: 'bold',
      color: '#00F0FF',
      align: 'center'
    }).setOrigin(0.5).setScrollFactor(0).setDepth(221);

    // Interactive zone on the card for Tap-to-Continue steps
    this.tutorialCardZone = this.add.zone(cx, cardY + 14, cardW, cardH - 28)
      .setOrigin(0.5)
      .setScrollFactor(0)
      .setDepth(221)
      .setInteractive({ useHandCursor: true });

    this.tutorialCardZone.on('pointerdown', () => {
      const step = this.tutorialSteps?.[this.tutorialStep];
      if (!step) return;
      if (step.type === 'info') {
        this.setTutorialStep(this.tutorialStep + 1);
      } else if (step.type === 'done') {
        this.time.removeAllEvents();
        this.scene.start('Menu');
      }
    });

    // Subtle pulse animation for highlights & prompt
    this.tweens.add({
      targets: this.tutorialPromptText,
      alpha: { from: 1.0, to: 0.45 },
      duration: 650,
      yoyo: true,
      repeat: -1
    });

    // Continuous zoom-in + fade-in loop for the green target triangle socket
    this.targetPulseProgress = 0;
    this.targetPulseTween = this.tweens.addCounter({
      from: 0,
      to: 1,
      duration: 1250,
      repeat: -1,
      onUpdate: (tween) => {
        this.targetPulseProgress = tween.getValue();
        const step = this.tutorialSteps?.[this.tutorialStep];
        if (this.tutorialActive && step && step.targetCell) {
          this.drawTutorialHighlights();
        }
      }
    });
  }

  stopIntroClusterLoop() {
    if (this.introLoopEvent) {
      this.introLoopEvent.remove(false);
      this.introLoopEvent = null;
    }
    this.hideTutorialCursor = false;
  }

  startIntroClusterLoop() {
    this.stopIntroClusterLoop();
    this.hideTutorialCursor = true;

    const keyframes = [
      // Keyframe 1: Opening seed clusters (Red: 2, Blue: 2)
      [
        { x: 4, y: 5, r: 0, p: 1 }, { x: 4, y: 5, r: 2, p: 1 },
        { x: 5, y: 5, r: 1, p: 2 }, { x: 5, y: 5, r: 3, p: 2 }
      ],
      // Keyframe 2: Expanding contiguous clusters (Red: 7, Blue: 7)
      [
        { x: 4, y: 5, r: 0, p: 1 }, { x: 4, y: 5, r: 2, p: 1 },
        { x: 3, y: 5, r: 1, p: 1 }, { x: 3, y: 5, r: 3, p: 1 },
        { x: 4, y: 6, r: 0, p: 1 }, { x: 4, y: 6, r: 2, p: 1 },
        { x: 3, y: 6, r: 3, p: 1 },
        { x: 5, y: 5, r: 1, p: 2 }, { x: 5, y: 5, r: 3, p: 2 },
        { x: 6, y: 5, r: 0, p: 2 }, { x: 6, y: 5, r: 2, p: 2 },
        { x: 5, y: 6, r: 3, p: 2 }, { x: 5, y: 6, r: 1, p: 2 },
        { x: 6, y: 6, r: 0, p: 2 }
      ],
      // Keyframe 3: Large competing territories (Red: 15, Blue: 13)
      [
        { x: 4, y: 5, r: 0, p: 1 }, { x: 4, y: 5, r: 2, p: 1 },
        { x: 3, y: 5, r: 1, p: 1 }, { x: 3, y: 5, r: 3, p: 1 },
        { x: 4, y: 6, r: 0, p: 1 }, { x: 4, y: 6, r: 2, p: 1 },
        { x: 3, y: 6, r: 1, p: 1 }, { x: 3, y: 6, r: 3, p: 1 },
        { x: 2, y: 5, r: 0, p: 1 }, { x: 2, y: 5, r: 2, p: 1 },
        { x: 2, y: 6, r: 1, p: 1 }, { x: 2, y: 6, r: 3, p: 1 },
        { x: 4, y: 7, r: 0, p: 1 }, { x: 4, y: 7, r: 2, p: 1 },
        { x: 3, y: 7, r: 3, p: 1 },
        { x: 5, y: 5, r: 1, p: 2 }, { x: 5, y: 5, r: 3, p: 2 },
        { x: 6, y: 5, r: 0, p: 2 }, { x: 6, y: 5, r: 2, p: 2 },
        { x: 5, y: 6, r: 3, p: 2 }, { x: 5, y: 6, r: 1, p: 2 },
        { x: 6, y: 6, r: 0, p: 2 }, { x: 6, y: 6, r: 2, p: 2 },
        { x: 7, y: 5, r: 1, p: 2 }, { x: 7, y: 5, r: 3, p: 2 },
        { x: 7, y: 6, r: 0, p: 2 }, { x: 7, y: 6, r: 2, p: 2 },
        { x: 6, y: 7, r: 3, p: 2 }
      ]
    ];

    let frameIdx = 0;
    const applyKeyframe = (kf) => {
      this.gridManager.reset();
      for (const m of kf) {
        this.gridManager.placeTriangle(m.x, m.y, m.r, m.p);
      }
      this.gridManager.currentPlayer = 1;
      this.updateScore();
      this.refreshBoardView();
    };

    applyKeyframe(keyframes[0]);
    this.introLoopEvent = this.time.addEvent({
      delay: 1100,
      loop: true,
      callback: () => {
        if (!this.tutorialActive || this.tutorialStep !== 0) return;
        frameIdx = (frameIdx + 1) % keyframes.length;
        applyKeyframe(keyframes[frameIdx]);
      }
    });
  }

  setTutorialStep(index) {
    this.tutorialStep = index;
    const step = this.tutorialSteps[index];
    if (!step) return;

    this.stopIntroClusterLoop();

    const { cx, cy, w, h } = this.tutorialCardBounds;
    const borderColor = (step.type === 'info' || step.type === 'done') ? 0x00F0FF : 0x00FFCC;
    this.drawChamferedPanel(this.tutorialCardGfx, cx, cy, w, h, 12, 0x050D1A, 0.94, borderColor, 0.95, 2.2);

    this.tutorialTitleText.setText(`[${index + 1}/${this.tutorialSteps.length}] ${step.title}`);
    this.tutorialBodyText.setText(step.body);
    this.tutorialPromptText.setText(step.prompt);

    // Re-center and pre-zoom the board at the start of each exercise/step for optimal visibility
    if (step.view) {
      const cols = this.gridManager.cols;
      const rows = this.gridManager.rows;
      this.panOffset.u = ((step.view.u % cols) + cols) % cols;
      this.panOffset.v = ((step.view.v % rows) + rows) % rows;
      this.zoomLevel = step.view.zoom || 1.0;
    }

    if (index === 0) {
      this.startIntroClusterLoop();
      return;
    }

    if (step.type === 'rotate') {
      // Reset the board cleanly after the Step 0 cluster demo loop
      this.gridManager.reset();
      this.gridManager.placeTriangle(5, 5, 1, 2);
      this.gridManager.currentPlayer = 1;
      this.tutorialRotateCount = 0;
      this.cursorPos = { x: 4, y: 5 };
      this.gridManager.selectedRotation = 0;
      this.updateScore();
    } else if (step.type === 'place1') {
      // Start the red cursor behind/near the instruction card, away from the green target socket
      this.cursorPos = { x: step.startCursor.x, y: step.startCursor.y };
      this.gridManager.selectedRotation = step.startCursor.rot;
    } else if (step.type === 'place2') {
      // Ensure the setup triangle at (6, 5, rot 1) is present so (5, 5, rot 3) is 2/3 enclosed
      const cell65 = this.gridManager.grid[5][6];
      if (!cell65.triangles.some(t => t.rotation === 1)) {
        this.gridManager.placeTriangle(6, 5, 1, 2);
      }
      this.gridManager.currentPlayer = 1;
      // Start the red cursor behind/near the instruction card, away from the green target socket
      this.cursorPos = { x: step.startCursor.x, y: step.startCursor.y };
      this.gridManager.selectedRotation = step.startCursor.rot;
      this.updateScore();
    }

    this.refreshBoardView();
  }

  drawTutorialHighlights() {
    if (!this.tutorialHighlightGfx) return;
    this.tutorialHighlightGfx.clear();

    const step = this.tutorialSteps?.[this.tutorialStep];
    if (!step) return;

    // 1. Draw HUD button highlight boxes
    if (step.highlights) {
      for (const box of step.highlights) {
        this.tutorialHighlightGfx.lineStyle(6, box.color, 0.28);
        this.tutorialHighlightGfx.strokeRect(box.x - box.w / 2, box.y - box.h / 2, box.w, box.h);
        this.tutorialHighlightGfx.lineStyle(2.2, box.color, 0.95);
        this.tutorialHighlightGfx.strokeRect(box.x - box.w / 2, box.y - box.h / 2, box.w, box.h);
      }
    }

    // 2. Highlight target board cell + animated zooming-in green target triangle socket
    if (step.targetCell) {
      const cols = this.gridManager.cols;
      const rows = this.gridManager.rows;
      const u = ((step.targetCell.x + this.panOffset.u) % cols + cols) % cols;
      const v = ((step.targetCell.y + this.panOffset.v) % rows + rows) % rows;
      const [raw0, raw1, raw2] = this.getProjectedTriangleVertices(u, v, step.targetCell.rot);

      const midX = (raw0.x + raw1.x + raw2.x) / 3;
      const midY = (raw0.y + raw1.y + raw2.y) / 3;

      // Animated zoom-in from larger scale & zero transparency into the target socket in a loop
      const t = this.targetPulseProgress ?? 0.5;
      const easeOut = 1 - Math.pow(1 - t, 2.6);
      const animScale = 2.05 - 0.90 * easeOut; // Zooms in from 2.05x down to the 1.15x socket
      const animAlpha = t < 0.82
        ? Math.min(1, t / 0.35)
        : Math.max(0, (1 - t) / 0.18); // Starts at 0 alpha, ramps up while zooming in, fades out cleanly at end of loop

      const a0 = { x: midX + (raw0.x - midX) * animScale, y: midY + (raw0.y - midY) * animScale };
      const a1 = { x: midX + (raw1.x - midX) * animScale, y: midY + (raw1.y - midY) * animScale };
      const a2 = { x: midX + (raw2.x - midX) * animScale, y: midY + (raw2.y - midY) * animScale };

      // Steady inner socket reference so the player always sees the exact destination boundary
      const baseExpand = 1.15;
      const v0 = { x: midX + (raw0.x - midX) * baseExpand, y: midY + (raw0.y - midY) * baseExpand };
      const v1 = { x: midX + (raw1.x - midX) * baseExpand, y: midY + (raw1.y - midY) * baseExpand };
      const v2 = { x: midX + (raw2.x - midX) * baseExpand, y: midY + (raw2.y - midY) * baseExpand };

      this.tutorialHighlightGfx.fillStyle(0x00FFCC, 0.06 * animAlpha);
      this.tutorialHighlightGfx.beginPath();
      this.tutorialHighlightGfx.moveTo(a0.x, a0.y);
      this.tutorialHighlightGfx.lineTo(a1.x, a1.y);
      this.tutorialHighlightGfx.lineTo(a2.x, a2.y);
      this.tutorialHighlightGfx.closePath();
      this.tutorialHighlightGfx.fillPath();

      // Zooming outer green halo
      this.tutorialHighlightGfx.lineStyle(6, 0x00FFCC, 0.28 * animAlpha);
      this.tutorialHighlightGfx.beginPath();
      this.tutorialHighlightGfx.moveTo(a0.x, a0.y);
      this.tutorialHighlightGfx.lineTo(a1.x, a1.y);
      this.tutorialHighlightGfx.lineTo(a2.x, a2.y);
      this.tutorialHighlightGfx.closePath();
      this.tutorialHighlightGfx.strokePath();

      // Zooming bright green triangle outline
      this.tutorialHighlightGfx.lineStyle(2.8, 0x00FFCC, 0.92 * animAlpha);
      this.tutorialHighlightGfx.beginPath();
      this.tutorialHighlightGfx.moveTo(a0.x, a0.y);
      this.tutorialHighlightGfx.lineTo(a1.x, a1.y);
      this.tutorialHighlightGfx.lineTo(a2.x, a2.y);
      this.tutorialHighlightGfx.closePath();
      this.tutorialHighlightGfx.strokePath();

      // Subtle persistent socket outline at 1.15x
      this.tutorialHighlightGfx.lineStyle(1.6, 0x00FFCC, 0.32);
      this.tutorialHighlightGfx.beginPath();
      this.tutorialHighlightGfx.moveTo(v0.x, v0.y);
      this.tutorialHighlightGfx.lineTo(v1.x, v1.y);
      this.tutorialHighlightGfx.lineTo(v2.x, v2.y);
      this.tutorialHighlightGfx.closePath();
      this.tutorialHighlightGfx.strokePath();
    }

    // 3. Highlight the gap about to be enclosed in Step 3
    if (step.enclosedCell) {
      const cols = this.gridManager.cols;
      const rows = this.gridManager.rows;
      const u = ((step.enclosedCell.x + this.panOffset.u) % cols + cols) % cols;
      const v = ((step.enclosedCell.y + this.panOffset.v) % rows + rows) % rows;
      const [e0, e1, e2] = this.getProjectedTriangleVertices(u, v, 3);

      this.tutorialHighlightGfx.lineStyle(2, 0xFFFF00, 0.85);
      this.tutorialHighlightGfx.beginPath();
      this.tutorialHighlightGfx.moveTo(e0.x, e0.y);
      this.tutorialHighlightGfx.lineTo(e1.x, e1.y);
      this.tutorialHighlightGfx.lineTo(e2.x, e2.y);
      this.tutorialHighlightGfx.closePath();
      this.tutorialHighlightGfx.strokePath();
    }
  }

  tutorialCanInteract(action) {
    const step = this.tutorialSteps?.[this.tutorialStep];
    if (!step) return false;
    if (step.type === 'info' || step.type === 'done') return false;
    if (step.type === 'rotate') return action === 'rotate' || action === 'grid';
    if (step.type === 'place1' || step.type === 'place2') {
      return action === 'rotate' || action === 'grid' || action === 'place';
    }
    return false;
  }

  onTutorialRotate() {
    const step = this.tutorialSteps?.[this.tutorialStep];
    if (!step) return;
    if (step.type === 'rotate') {
      this.tutorialRotateCount++;
      this.tutorialPromptText.setText(`ROTATE 4 TIMES TO PROCEED: ${Math.min(this.tutorialRotateCount, 4)} / 4`);
      if (this.tutorialRotateCount >= 4) {
        this.time.delayedCall(280, () => this.setTutorialStep(this.tutorialStep + 1));
      }
    }
  }

  handleTutorialPlace() {
    const step = this.tutorialSteps?.[this.tutorialStep];
    if (!step || (step.type !== 'place1' && step.type !== 'place2')) return;

    const target = step.targetCell;
    if (this.cursorPos.x !== target.x || this.cursorPos.y !== target.y) {
      this.cameras.main.flash(180, 255, 42, 84);
      this.tutorialPromptText.setText('MOVE INTO THE GREEN TARGET CELL FIRST!');
      return;
    }

    if (this.gridManager.selectedRotation !== target.rot) {
      this.cameras.main.flash(180, 255, 42, 84);
      this.tutorialPromptText.setText('ROTATE (↻ OR TAP CELL) TO MATCH GREEN TARGET!');
      return;
    }

    this.gridManager.currentPlayer = 1;
    const placed = this.gridManager.placeTriangle(target.x, target.y, target.rot, 1);
    if (placed) {
      const enclosed = this.gridManager.claimEnclosedTriangles();
      if (enclosed > 0) {
        this.gridManager.extraTurns += Math.floor((1 + enclosed) / 2);
      }
      this.redrawTriangles();
      this.updateScore();
      this.setTutorialStep(this.tutorialStep + 1);
    }
  }
}

class MenuScene extends Phaser.Scene {
  constructor() {
      super({ key: 'Menu' });
  }

  preload() {
      // Procedural Neon Manifold visuals require zero external image assets
  }

  create() {
      this.bgGraphics = this.add.graphics().setDepth(1);
      this.drawCyberMatrixBackground();

      this.icosaGraphics = this.add.graphics().setDepth(10);
      this.menuContainer = this.add.container(0, 0).setDepth(20);

      this.initIcosahedron();
      this.renderMainMenu();
  }

  drawCyberMatrixBackground() {
      if (!this.bgGraphics) return;
      const w = this.cameras.main.width;
      const h = this.cameras.main.height;
      this.bgGraphics.clear();

      // Subtle background matrix grid
      this.bgGraphics.lineStyle(1, 0x092232, 0.45);
      const step = 40;
      for (let x = 0; x <= w; x += step) {
          this.bgGraphics.lineBetween(x, 0, x, h);
      }
      for (let y = 0; y <= h; y += step) {
          this.bgGraphics.lineBetween(0, y, w, y);
      }

      // Micro-crosshair nodes at every 2nd grid intersection
      this.bgGraphics.fillStyle(0x00F0FF, 0.16);
      for (let x = step; x < w; x += step * 2) {
          for (let y = step; y < h; y += step * 2) {
              this.bgGraphics.fillRect(x - 1, y - 1, 2, 2);
          }
      }

      // Outer HUD frame border
      this.bgGraphics.lineStyle(1.5, 0x0D4054, 0.7);
      this.bgGraphics.strokeRect(14, 14, w - 28, h - 28);
  }

  initIcosahedron() {
      const phi = (1 + Math.sqrt(5)) / 2;
      const rawVerts = [
          [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
          [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
          [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
      ];
      // Normalize vertices to unit sphere
      this.icosaVertices = rawVerts.map(([x, y, z]) => {
          const len = Math.hypot(x, y, z);
          return { x: x / len, y: y / len, z: z / len };
      });

      // 20 equilateral triangular faces (CCW winding)
      this.icosaFaces = [
          [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
          [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
          [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
          [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
      ];

      // Build face adjacency list (faces sharing an edge)
      this.faceNeighbors = this.icosaFaces.map((fA, idxA) => {
          const neighbors = [];
          this.icosaFaces.forEach((fB, idxB) => {
              if (idxA === idxB) return;
              let shared = 0;
              for (const v of fA) {
                  if (fB.includes(v)) shared++;
              }
              if (shared === 2) neighbors.push(idxB);
          });
          return neighbors;
      });

      // Initialize face states: mix of P1 Red (1), P2 Blue (2), and Neutral Cyan (0)
      this.faceOwners = this.icosaFaces.map((_, i) => (i % 3 === 0 ? 1 : (i % 3 === 1 ? 2 : 0)));
      this.icosaAngles = { x: 0.35, y: 0.0, z: 0.18 };
      this.nextColorSwapTime = 0;
      this.glitchState = {
          active: false,
          endTime: 0,
          nextTrigger: 1800,
          sliceY: 0,
          offsetX: 0
      };

      // Allow tapping the hologram to trigger an instant glitch + territory wave
      this.input.on('pointerdown', (pointer) => {
          const isDebug = typeof DEBUG_MODE !== 'undefined' && DEBUG_MODE;
          const stageCy = isDebug ? 225 : 262;
          const hitRadius = isDebug ? 78 : 102;
          const dist = Math.hypot(pointer.x - 300, pointer.y - stageCy);
          if (dist <= hitRadius) {
              this.triggerIcosaGlitch(this.time.now, 200);
              this.stepTerritoryAutomaton(true);
          }
      });
  }

  triggerIcosaGlitch(now, duration = 140) {
      this.glitchState.active = true;
      this.glitchState.endTime = now + duration;
      this.glitchState.sliceY = (Math.random() - 0.5) * 1.2;
      this.glitchState.offsetX = (Math.random() < 0.5 ? -1 : 1) * (7 + Math.random() * 7);
      this.glitchState.nextTrigger = now + 2200 + Math.random() * 1800;
  }

  stepTerritoryAutomaton(forceBurst = false) {
      const steps = forceBurst ? 4 : 1;
      for (let s = 0; s < steps; s++) {
          const srcIdx = Math.floor(Math.random() * this.icosaFaces.length);
          let owner = this.faceOwners[srcIdx];
          if (owner === 0 || Math.random() < 0.22) {
              owner = Math.random() < 0.5 ? 1 : 2;
              this.faceOwners[srcIdx] = owner;
          }
          const neighbors = this.faceNeighbors[srcIdx];
          if (neighbors && neighbors.length > 0) {
              const targetIdx = neighbors[Math.floor(Math.random() * neighbors.length)];
              // Occasionally leave a neutral wireframe circuit gap so wireframe structure shines
              this.faceOwners[targetIdx] = Math.random() < 0.14 ? 0 : owner;
          }
      }
  }

  update(time, delta) {
      if (!this.icosaGraphics) return;

      const dt = delta * 0.001;
      this.icosaAngles.y += dt * 0.85;
      this.icosaAngles.x += dt * 0.42;

      if (time > this.nextColorSwapTime) {
          this.stepTerritoryAutomaton(false);
          this.nextColorSwapTime = time + 240;
      }

      if (!this.glitchState.active && time > this.glitchState.nextTrigger) {
          this.triggerIcosaGlitch(time, 110 + Math.random() * 70);
      } else if (this.glitchState.active && time > this.glitchState.endTime) {
          this.glitchState.active = false;
          if (this.titleText) {
              this.titleText.x = this.cameras.main.width / 2;
          }
      }

      this.renderIcosahedron(time);
  }

  renderIcosahedron(time) {
      const gfx = this.icosaGraphics;
      gfx.clear();

      const isDebug = typeof DEBUG_MODE !== 'undefined' && DEBUG_MODE;
      const cx = this.cameras.main.width / 2;
      const cy = isDebug ? 225 : 262;
      const radius = isDebug ? 62 : 84;
      const isGlitching = this.glitchState.active;

      if (this.titleText && isGlitching) {
          this.titleText.x = cx + (Math.random() - 0.5) * 6;
      }

      // 1. Ambient outer hologram ring & telemetry ticks
      gfx.lineStyle(1, 0x0D4054, 0.55);
      gfx.strokeCircle(cx, cy, radius + 16);
      gfx.lineStyle(1.5, 0x00F0FF, 0.35);
      const tickLen = 8;
      gfx.lineBetween(cx - radius - 22, cy, cx - radius - 22 + tickLen, cy);
      gfx.lineBetween(cx + radius + 22 - tickLen, cy, cx + radius + 22, cy);
      gfx.lineBetween(cx, cy - radius - 22, cx, cy - radius - 22 + tickLen);
      gfx.lineBetween(cx, cy + radius + 22 - tickLen, cx, cy + radius + 22);

      // 2. Rotate & project 12 vertices
      const cosY = Math.cos(this.icosaAngles.y);
      const sinY = Math.sin(this.icosaAngles.y);
      const cosX = Math.cos(this.icosaAngles.x);
      const sinX = Math.sin(this.icosaAngles.x);

      const projected = this.icosaVertices.map((v) => {
          // Rotate around Y
          const x1 = v.x * cosY - v.z * sinY;
          const z1 = v.z * cosY + v.x * sinY;
          // Rotate around X
          const y2 = v.y * cosX - z1 * sinX;
          const z2 = z1 * cosX + v.y * sinX;

          const perspective = 3.2 / (3.2 - z2 * 0.75);
          let sx = cx + x1 * radius * perspective;
          let sy = cy + y2 * radius * perspective;

          if (isGlitching && Math.abs(y2 - this.glitchState.sliceY) < 0.45) {
              sx += this.glitchState.offsetX;
          }

          return { x: sx, y: sy, z: z2, rawY: y2 };
      });

      // 3. Compute face depths and sort back-to-front (Painter's algorithm)
      const faceList = this.icosaFaces.map((indices, fIdx) => {
          const v0 = projected[indices[0]];
          const v1 = projected[indices[1]];
          const v2 = projected[indices[2]];
          const avgZ = (v0.z + v1.z + v2.z) / 3;
          // 2D cross product in screen space to determine front vs back face
          const cross = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
          return { fIdx, v0, v1, v2, avgZ, isFront: cross > 0, owner: this.faceOwners[fIdx] };
      });

      faceList.sort((a, b) => a.avgZ - b.avgZ);

      // 4. Optional Chromatic Aberration (RGB Split) ghost wireframes during glitch burst
      if (isGlitching) {
          const shift = this.glitchState.offsetX * 0.85;
          for (const f of faceList) {
              if (!f.isFront) continue;
              // Red ghost shifted left
              gfx.lineStyle(1.4, 0xFF2A54, 0.55);
              gfx.beginPath();
              gfx.moveTo(f.v0.x - shift, f.v0.y);
              gfx.lineTo(f.v1.x - shift, f.v1.y);
              gfx.lineTo(f.v2.x - shift, f.v2.y);
              gfx.closePath();
              gfx.strokePath();

              // Cyan ghost shifted right
              gfx.lineStyle(1.4, 0x00F0FF, 0.55);
              gfx.beginPath();
              gfx.moveTo(f.v0.x + shift, f.v0.y);
              gfx.lineTo(f.v1.x + shift, f.v1.y);
              gfx.lineTo(f.v2.x + shift, f.v2.y);
              gfx.closePath();
              gfx.strokePath();
          }

          // Horizontal glitch scanline bars across the stage
          gfx.lineStyle(1.5, 0x00F0FF, 0.65);
          const scanY = cy + this.glitchState.sliceY * radius;
          gfx.lineBetween(cx - radius - 34, scanY, cx + radius + 34, scanY);
          gfx.lineStyle(1, 0xFF2A54, 0.65);
          gfx.lineBetween(cx - radius - 26, scanY + 5, cx + radius + 26, scanY + 5);
      }

      // 5. Render back-faces (translucent dark wireframe) then front-faces (Neon Manifold 3-B)
      for (const f of faceList) {
          const { v0, v1, v2, isFront, owner } = f;
          if (!isFront) {
              const backStroke = owner === 1 ? 0xFF2A54 : (owner === 2 ? 0x00F0FF : 0x0D4054);
              gfx.lineStyle(1, backStroke, 0.20);
              gfx.beginPath();
              gfx.moveTo(v0.x, v0.y);
              gfx.lineTo(v1.x, v1.y);
              gfx.lineTo(v2.x, v2.y);
              gfx.closePath();
              gfx.strokePath();
          } else {
              // Slight centroid inset so individual triangular power cells stand out clearly
              const fcx = (v0.x + v1.x + v2.x) / 3;
              const fcy = (v0.y + v1.y + v2.y) / 3;
              const inset = 0.05;
              const p0 = { x: v0.x + (fcx - v0.x) * inset, y: v0.y + (fcy - v0.y) * inset };
              const p1 = { x: v1.x + (fcx - v1.x) * inset, y: v1.y + (fcy - v1.y) * inset };
              const p2 = { x: v2.x + (fcx - v2.x) * inset, y: v2.y + (fcy - v2.y) * inset };

              if (owner === 1) {
                  gfx.fillStyle(0xE6193C, 0.78);
                  gfx.lineStyle(1.8, 0xFF4D6D, 0.98);
              } else if (owner === 2) {
                  gfx.fillStyle(0x0066FF, 0.78);
                  gfx.lineStyle(1.8, 0x00F0FF, 0.98);
              } else {
                  gfx.fillStyle(0x071828, 0.55);
                  gfx.lineStyle(1.5, 0x00F0FF, 0.65);
              }

              gfx.beginPath();
              gfx.moveTo(p0.x, p0.y);
              gfx.lineTo(p1.x, p1.y);
              gfx.lineTo(p2.x, p2.y);
              gfx.closePath();
              gfx.fillPath();
              gfx.strokePath();
          }
      }
  }

  clearMenu() {
      if (this.menuContainer) {
          this.menuContainer.removeAll(true);
      } else {
          this.menuContainer = this.add.container(0, 0).setDepth(20);
      }
  }

  drawChamferedMenuBtn(gfx, x, y, width, height, cut, fillHex, strokeHex, hover = false) {
      const x0 = x - width / 2;
      const y0 = y - height / 2;
      const x1 = x + width / 2;
      const y1 = y + height / 2;

      const trace = () => {
          gfx.beginPath();
          gfx.moveTo(x0 + cut, y0);
          gfx.lineTo(x1 - cut, y0);
          gfx.lineTo(x1, y0 + cut);
          gfx.lineTo(x1, y1 - cut);
          gfx.lineTo(x1 - cut, y1);
          gfx.lineTo(x0 + cut, y1);
          gfx.lineTo(x0, y1 - cut);
          gfx.lineTo(x0, y0 + cut);
          gfx.closePath();
      };

      gfx.clear();
      // Outer neon glow pass
      gfx.lineStyle(hover ? 7 : 5, strokeHex, hover ? 0.34 : 0.18);
      trace();
      gfx.strokePath();

      // Obsidian cyber-glass fill
      gfx.fillStyle(fillHex, hover ? 0.96 : 0.88);
      trace();
      gfx.fillPath();

      // Crisp laser rim
      gfx.lineStyle(hover ? 2.4 : 1.8, strokeHex, hover ? 1.0 : 0.88);
      trace();
      gfx.strokePath();
  }

  createStyledButton(x, y, text, baseFillHex, hoverFillHex, strokeHex, onClick, width = 350, height = 52, fontSize = '20px') {
      const bg = this.add.graphics();
      this.drawChamferedMenuBtn(bg, x, y, width, height, 11, baseFillHex, strokeHex, false);

      const label = this.add.text(x, y, text.toUpperCase(), {
          fontFamily: '"Orbitron", "Share Tech Mono", Arial',
          fontSize: fontSize,
          fontStyle: 'bold',
          color: '#E6F9FF',
          align: 'center'
      }).setOrigin(0.5);

      const hitZone = this.add.zone(x, y, width, height)
          .setOrigin(0.5)
          .setInteractive({ useHandCursor: true });

      hitZone.on('pointerover', () => {
          this.drawChamferedMenuBtn(bg, x, y, width, height, 11, hoverFillHex, strokeHex, true);
          label.setColor('#FFFFFF');
      });

      hitZone.on('pointerout', () => {
          this.drawChamferedMenuBtn(bg, x, y, width, height, 11, baseFillHex, strokeHex, false);
          label.setColor('#E6F9FF');
      });

      hitZone.on('pointerdown', onClick);

      this.menuContainer.add([bg, label, hitZone]);
      return { bg, label, hitZone };
  }

  renderHeader(titleStr, subtitleStr) {
      const centerX = this.cameras.main.width / 2;

      this.titleText = this.add.text(centerX, 78, titleStr, {
          fontFamily: '"Orbitron", Arial',
          fontSize: titleStr === 'TRIANTICA' ? '50px' : '32px',
          fontStyle: '900',
          color: '#E6F9FF',
          align: 'center'
      }).setOrigin(0.5);
      this.titleText.setShadow(0, 0, '#00F0FF', 14, true, true);

      const subtitle = this.add.text(centerX, 124, subtitleStr.toUpperCase(), {
          fontFamily: '"Share Tech Mono", monospace',
          fontSize: '15px',
          color: '#5CE1E6',
          letterSpacing: 2,
          align: 'center'
      }).setOrigin(0.5);

      this.menuContainer.add([this.titleText, subtitle]);
  }

  renderMainMenu() {
      this.clearMenu();
      const centerX = this.cameras.main.width / 2;
      const isDebug = typeof DEBUG_MODE !== 'undefined' && DEBUG_MODE;

      this.renderHeader('TRIANTICA', 'Toroidal Manifold Strategy // T²');

      if (!isDebug) {
          // Standard release layout: larger D20 hologram at y=262, 3 primary buttons placed lower
          this.createStyledButton(
              centerX,
              436,
              'Human vs AI',
              0x071A2E,
              0x0D2E4D,
              0x00F0FF,
              () => this.renderDifficultyMenu(),
              360,
              54,
              '21px'
          );

          this.createStyledButton(
              centerX,
              510,
              'Human vs Human',
              0x1A0714,
              0x2E0B22,
              0xFF2A54,
              () => this.scene.start('Game', { mode: 'HvH' }),
              360,
              54,
              '21px'
          );

          this.createStyledButton(
              centerX,
              584,
              'How to Play',
              0x061E1C,
              0x0B3330,
              0x00FFCC,
              () => this.scene.start('Game', { mode: 'Tutorial' }),
              360,
              54,
              '21px'
          );
      } else {
          // Compact debug layout to accommodate benchmark buttons below
          this.createStyledButton(
              centerX,
              342,
              'Human vs AI',
              0x071A2E,
              0x0D2E4D,
              0x00F0FF,
              () => this.renderDifficultyMenu(),
              360,
              48,
              '19px'
          );

          this.createStyledButton(
              centerX,
              402,
              'Human vs Human',
              0x1A0714,
              0x2E0B22,
              0xFF2A54,
              () => this.scene.start('Game', { mode: 'HvH' }),
              360,
              48,
              '19px'
          );

          this.createStyledButton(
              centerX,
              462,
              'How to Play',
              0x061E1C,
              0x0B3330,
              0x00FFCC,
              () => this.scene.start('Game', { mode: 'Tutorial' }),
              360,
              48,
              '19px'
          );

          const debugLabel = this.add.text(centerX, 518, '— AI BENCHMARK TELEMETRY (DEBUG) —', {
              fontFamily: '"Share Tech Mono", monospace',
              fontSize: '14px',
              color: '#5CE1E6',
              align: 'center'
          }).setOrigin(0.5);
          this.menuContainer.add(debugLabel);

          this.createStyledButton(
              centerX,
              564,
              'AI vs Random',
              0x081322,
              0x10243E,
              0x0D4054,
              () => this.scene.start('Game', { mode: 'AvR', difficulty: 'hard' }),
              330,
              42,
              '17px'
          );

          this.createStyledButton(
              centerX,
              618,
              'AI vs Heuristic',
              0x081322,
              0x10243E,
              0x0D4054,
              () => this.scene.start('Game', { mode: 'AvH', difficulty: 'hard' }),
              330,
              42,
              '17px'
          );

          this.createStyledButton(
              centerX,
              672,
              'AI vs AI (Network)',
              0x081322,
              0x10243E,
              0x0D4054,
              () => this.scene.start('Game', { mode: 'AvA', difficulty: 'hard' }),
              330,
              42,
              '17px'
          );
      }
  }

  renderDifficultyMenu() {
      this.clearMenu();
      const centerX = this.cameras.main.width / 2;

      this.renderHeader('SELECT DIFFICULTY', 'Choose Neural Opponent Strength');

      this.createStyledButton(
          centerX,
          412,
          'Easy // Casual',
          0x061E1C,
          0x0B3330,
          0x00FFCC,
          () => this.renderSideSelectionMenu('easy'),
          380,
          52,
          '19px'
      );

      this.createStyledButton(
          centerX,
          482,
          'Medium // Raw ConvNet',
          0x071A2E,
          0x0D2E4D,
          0x00F0FF,
          () => this.renderSideSelectionMenu('medium'),
          380,
          52,
          '19px'
      );

      this.createStyledButton(
          centerX,
          552,
          'Hard // Master + Safety',
          0x1F0610,
          0x360A1C,
          0xFF2A54,
          () => this.renderSideSelectionMenu('hard'),
          380,
          52,
          '19px'
      );

      this.createStyledButton(
          centerX,
          636,
          '◀ Back',
          0x071120,
          0x0E223D,
          0x5CE1E6,
          () => this.renderMainMenu(),
          220,
          44,
          '16px'
      );
  }

  renderSideSelectionMenu(difficulty) {
      this.clearMenu();
      const centerX = this.cameras.main.width / 2;

      this.renderHeader('CHOOSE YOUR SIDE', `AI Profile: ${difficulty.toUpperCase()}`);

      this.createStyledButton(
          centerX,
          422,
          'P1 [Ares Red] // 1st Move',
          0x220611,
          0x380A1D,
          0xFF2A54,
          () => this.scene.start('Game', {
              mode: 'HvAI',
              difficulty: difficulty,
              humanPlayerNumber: 1
          }),
          420,
          56,
          '19px'
      );

      this.createStyledButton(
          centerX,
          500,
          'P2 [Tron Blue] // 2nd Move',
          0x061830,
          0x0B2952,
          0x00F0FF,
          () => this.scene.start('Game', {
              mode: 'HvAI',
              difficulty: difficulty,
              humanPlayerNumber: 2
          }),
          420,
          56,
          '19px'
      );

      this.createStyledButton(
          centerX,
          598,
          '◀ Back to Difficulty',
          0x071120,
          0x0E223D,
          0x5CE1E6,
          () => this.renderDifficultyMenu(),
          270,
          44,
          '16px'
      );
  }
}


class BootScene extends Phaser.Scene {
  constructor() {
      super({ key: 'Boot' });
  }

  preload() {
      // In case you need to load assets common to all scenes later
      // For now, we just need classes to be available
  }

  create() {
      console.log("BootScene: Initiating background network load...");

      // --- Attempt to start the background load ---
      // Ensure necessary classes are globally available
      if (typeof GridManager !== 'undefined' && typeof AIPlayer !== 'undefined') {
          try {
              // Create temporary instances needed just to call importNetwork
              const tempGridManager = new GridManager(10, 10); // Use standard size
              const tempAIPlayer = new AIPlayer(tempGridManager, 'medium', 1); // Player number doesn't matter here

              // Start the import process BUT DO NOT await it here.
              // This lets the load happen in the background.
              const networkLoadPromise = tempAIPlayer.importNetwork();

              // Store the promise and the temporary AI instance in the registry
              // so GameScene can check their status later.
              this.registry.set('networkLoadPromise', networkLoadPromise);
              this.registry.set('tempAIInstanceForStatus', tempAIPlayer); // To check isNetworkLoaded later

              console.log("BootScene: Background network load initiated.");

          } catch (error) {
              console.error("BootScene: Error initiating background load -", error);
               // Ensure registry keys are not set if initiation failed
               this.registry.remove('networkLoadPromise');
               this.registry.remove('tempAIInstanceForStatus');
          }
      } else {
          console.warn("BootScene: GridManager or AIPlayer class not found. Skipping background load.");
      }

      // --- Immediately start the MenuScene ---
      console.log("BootScene: Starting MenuScene.");
      this.scene.start('Menu');
  }
}


const config = {
  type: Phaser.AUTO,
  parent: 'game-container',
  width: 600,
  height: 800,
  backgroundColor: '#03060C',
  input: {
    activePointers: 3
  },
  scale: {
    mode: Phaser.Scale.FIT,
    autoCenter: Phaser.Scale.CENTER_BOTH
  },
  scene: [BootScene, MenuScene, GameScene]
};

new Phaser.Game(config);