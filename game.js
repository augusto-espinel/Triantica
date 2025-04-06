class GridManager {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.grid = this.createGrid();
    this.currentPlayer = 1;
    this.selectedRotation = 0; // 0-3 (0°, 90°, 180°, 270°)
    this.extraTurns = 0; // Track remaining extra turns
    this.winRounds = { 1: 0, 2: 0 }; // Track rounds won by each player
  }

  reset() {
    this.grid = this.createGrid();
    this.currentPlayer = 1;
    this.extraTurns = 0;
  }

  isGridFull() {
    return this.grid.flat().every(cell => cell.triangles.length === 2);
  }

  createGrid() {
      return Array(this.rows).fill().map(() => 
        Array(this.cols).fill().map(() => ({
          triangles: [] // Array of { player: number, rotation: number }
        }))
      );
    }

  isValidPlacement(x, y) {
      if (this.isFirstMove()) return true;
      // Check if any corner of the new triangle touches existing triangles
      const corners = this.getTriangleCorners(x, y, this.selectedRotation);
      return corners.some(([cx, cy]) => this.hasCorner(cx, cy));
  }

  hasCorner(x, y) {
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
  
  placeTriangle(x, y) {
    const cell = this.grid[y][x];
    let hasOverlap = false;
    
    // Check if the cell is full (max 2 triangles)
    if (cell.triangles.length >= 2) return false;

    if (cell.triangles.length > 0)  {
      // Check if new triangle overlaps existing ones
      const newTriangleGroup = this.getDiagonalGroup(this.selectedRotation);
      hasOverlap = !(this.getDiagonalGroup(cell.triangles[0].rotation) === newTriangleGroup) || 
      cell.triangles[0].rotation === this.selectedRotation;
    } 

    // Check adjacency rule and overlaps
    if (!this.isValidPlacement(x, y) || hasOverlap) return false;

    cell.triangles.push({
      player: this.currentPlayer,
      rotation: this.selectedRotation
    });
    return true;
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

  // Get the largest contiguous triangle group size for a player
  getLargestContiguous(player) {
    const visited = new Set();
    let maxSize = 0;

    // Iterate over all cells and triangles
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        const cell = this.grid[y][x];
        cell.triangles.forEach(triangle => {
          if (triangle.player === player && !visited.has(`${x},${y},${triangle.rotation}`)) {
            const clusterSize = this.floodFill(x, y, triangle.rotation, player, visited);
            maxSize = Math.max(maxSize, clusterSize);
          }
        });
      }
    }
    return maxSize;
  }

  // Claim all enclosed triangles for the current player
  claimEnclosedTriangles() {
    let claimed = 0;
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        const cell = this.grid[y][x]
        if (cell.triangles.length == 1)  {
          // Check for unclaimed slot
          const mapping = [2, 3, 0, 1]; // Mapping of rotations to opposite
          if (this.isTriangleEnclosed(x, y, mapping[cell.triangles[0].rotation])) {
            cell.triangles.push({ player: this.currentPlayer, rotation: mapping[cell.triangles[0].rotation] });
            claimed++;
          }
        }
      }
    }
    return claimed;
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
}

class GameScene extends Phaser.Scene {
  constructor() {
      super({ key: 'Game' });
      this.cellSize = 50;
      this.gridOffset = { x: 50, y: 50 };
      this.gridManager = new GridManager(10, 10);
      this.cursorPos = { x: 0, y: 0 };
      this.isDragging = false;
      this.lastPinchDistance = 0;
      this.dragStart = { x: 0, y: 0 };
      this.cursorStart = { x: 0, y: 0 };
      this.gridBounds = { x: 0, y: 0, width: 0, height: 0 };
      this.gameMode = 'HvH'; // Default, or set based on selection ('HvH', 'HvAI', 'AvA')
      this.players = { 1: { type: 'human' }, 2: { type: 'human' } }; // Player setup
      this.aiInstances = { 1: null, 2: null }; // To hold AI instances
  }

  preload() {
      // Load assets later
  }

  init(data) {
    console.log('GameScene init with data:', data);
    // Set the gameMode based on data passed from MenuScene, otherwise use default
    this.gameMode = data.mode || 'HvH';
  }

  async create() {
    console.log(`Creating GameScene in mode: ${this.gameMode}`);

    // --- Function to handle AI initialization (including preload check) ---
    const initializeAI = async (playerNumber) => {
      console.log(`Initializing AI Player ${playerNumber}...`);
      this.players[playerNumber] = { type: 'ai' };
      const newAI = new AIPlayer(this.gridManager, 'medium', playerNumber);
      this.aiInstances[playerNumber] = newAI;

      // Check if preloading was attempted
      const loadPromise = this.registry.get('networkLoadPromise');
      const tempAI = this.registry.get('tempAIInstanceForStatus');

      let loadedSuccessfully = false;
      if (loadPromise && tempAI) {
          console.log(`AI ${playerNumber}: Found preload promise. Waiting for it to settle...`);
          try {
              // Wait for the promise initiated by BootScene to finish
              const preloadFinished = await loadPromise; // Waits if still pending, resolves quickly if already done

              // Check the status flag on the temporary AI instance used in BootScene
              if (tempAI.isNetworkLoaded) {
                  console.log(`AI ${playerNumber}: Preload successful. Copying network state...`);
                  const networkState = tempAI.exportNetwork();
                  if (networkState) {
                        // Apply the preloaded state to the actual game AI instance
                      newAI.network.fromJSON(networkState);
                      newAI.isNetworkLoaded = true; // Mark this instance as ready
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
            await newAI.importNetwork(); // Run the normal import logic
            if (!newAI.isNetworkLoaded) {
                console.warn(`AI Player ${playerNumber} will start untrained.`);
            }
      }
    }; // --- End of initializeAI function ---

    // --- Initialize AI based on gameMode ---
    if (this.gameMode === 'HvAI') {
        const aiPlayerNumber = 2;
        await initializeAI(aiPlayerNumber); // Use the helper function

    } else if (this.gameMode === 'AvA') {
        await initializeAI(1); // Initialize P1
        await initializeAI(2); // Initialize P2

          // Start AvA game automatically only after both AIs are initialized
          this.time.delayedCall(100, this.triggerAIMove, [], this);
    }

    // --- Rest of your create method ---
    console.log("Finished AI initialization. Setting up visuals and input...");

    this.drawGrid();
    this.setupTouchControls();
    this.setupInput();
    this.updateCursor();
    this.createScoreboard();
    this.calculateGridBounds();

    // Initialize cursor with high depth (top layer)
    this.cursorTriangle.setDepth(100); // Higher depth = rendered last
    
    // Initialize triangle graphics with lower depth
    this.triangleGraphics = this.add.graphics();
    this.triangleGraphics.setDepth(0); // Lower depth = rendered first

    this.createPlaceButton();
    this.createRotateButton(); 
    this.createForfeitButton();
    this.gameOverText = null;

    // Make sure human input is enabled/disabled correctly at the start
    this.updateInputStateForCurrentPlayer();

    // 'Train AI' button if mode is AvA
    if (this.gameMode === 'AvA') {
      this.trainButton = this.add.text(this.cameras.main.width - 250, 20, 'Run Training Game', { /* style */ })
          .setInteractive()
          .on('pointerdown', async () => {
              if (this.aiInstances[1]) {
                  console.log("Starting AI self-play training game...");
                  this.trainButton.setText('Training...'); // Provide feedback
                  await this.aiInstances[1].runSelfPlayGame(); // [cite: 62]
                  console.log("Training game finished.");
                  // Optionally trigger the network training process itself
                  // this.aiInstances[1].trainNetwork(50); [cite: 55, 88]
                  // Optionally save the updated network
                  // localStorage.setItem('aiNetworkP1', JSON.stringify(this.aiInstances[1].exportNetwork())); [cite: 89]
                    this.trainButton.setText('Run Training Game');
              }
          });
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

  createForfeitButton() {
    const buttonRadius = 40;
    const buttonX = this.cameras.main.width - 80;
    const buttonY = this.cameras.main.height - 80;

    this.forfeitButton = this.add.graphics()
      .fillStyle(0xFF5252, 1)
      .fillCircle(buttonX, buttonY, buttonRadius)
      .lineStyle(4, 0xFFFFFF)
      .strokeCircle(buttonX, buttonY, buttonRadius);

    this.add.text(buttonX, buttonY, '😞', {
      font: '30px Arial',
      fill: '#FFFFFF'
    }).setOrigin(0.5);

    this.forfeitButton.setInteractive(
      new Phaser.Geom.Circle(buttonX, buttonY, buttonRadius),
      Phaser.Geom.Circle.Contains
    );

    this.forfeitButton.on('pointerdown', () => this.handleForfeit());
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
          if (this.players[this.gridManager.currentPlayer].type === 'ai') {
               this.time.delayedCall(100, this.triggerAIMove, [], this);
          }
    });
  }

  createRotateButton() {
    const buttonRadius = 40;
    const buttonX = 80; // Left side of screen
    const buttonY = this.cameras.main.height - 80; // Same vertical position as place button

    // Create button graphics
    this.rotateButton = this.add.graphics()
      .fillStyle(0x2196F3, 1)
      .fillCircle(buttonX, buttonY, buttonRadius)
      .lineStyle(4, 0xFFFFFF)
      .strokeCircle(buttonX, buttonY, buttonRadius);

    // Create icon
    this.add.text(buttonX, buttonY, '↻', {
      font: '40px Arial',
      fill: '#FFFFFF',
      align: 'center'
    }).setOrigin(0.5);

    // Make interactive
    this.rotateButton.setInteractive(
      new Phaser.Geom.Circle(buttonX, buttonY, buttonRadius),
      Phaser.Geom.Circle.Contains
    );

    // Add click/touch handlers
    this.rotateButton.on('pointerdown', () => {
      this.gridManager.rotateTriangle();
      this.updateCursor();
      
      // Animate press
      this.rotateButton.fillStyle(0x1976D2);
      this.rotateButton.fillCircle(buttonX, buttonY, buttonRadius);
    });

    // Add release animation
    this.rotateButton.on('pointerup', () => {
      this.rotateButton.fillStyle(0x2196F3);
      this.rotateButton.fillCircle(buttonX, buttonY, buttonRadius);
    });

    this.rotateButton.on('pointerout', () => {
      this.rotateButton.fillStyle(0x2196F3);
      this.rotateButton.fillCircle(buttonX, buttonY, buttonRadius);
    });
  }

  createPlaceButton() {
    // Create button graphics
    const buttonWidth = 200;
    const buttonHeight = 60;
    const buttonX = this.cameras.main.width / 2;
    const buttonY = this.cameras.main.height - 80;

    // Button background
    this.placeButton = this.add.graphics()
      .fillStyle(0x4CAF50, 1)
      .fillRoundedRect(
        buttonX - buttonWidth/2, 
        buttonY - buttonHeight/2, 
        buttonWidth, 
        buttonHeight, 
        15
      )
      .lineStyle(4, 0xFFFFFF)
      .strokeRoundedRect(
        buttonX - buttonWidth/2, 
        buttonY - buttonHeight/2, 
        buttonWidth, 
        buttonHeight, 
        15
      );

    // Button text
    this.placeButtonText = this.add.text(buttonX, buttonY, 'PLACE TRIANGLE', {
      font: '24px Arial',
      fill: '#FFFFFF',
      align: 'center'
    }).setOrigin(0.5);

    // Make interactive
    this.placeButton.setInteractive(
      new Phaser.Geom.Rectangle(
        buttonX - buttonWidth/2,
        buttonY - buttonHeight/2,
        buttonWidth,
        buttonHeight
      ),
      Phaser.Geom.Rectangle.Contains
    );

    // Add click/touch handlers
    this.placeButton.on('pointerdown', () => this.placeCurrentTriangle());
    
    // Add hover effects
    this.placeButton.on('pointerover', () => {
      this.placeButton.fillStyle(0x45a049);
      this.placeButton.fillRoundedRect(
        buttonX - buttonWidth/2, 
        buttonY - buttonHeight/2, 
        buttonWidth, 
        buttonHeight, 
        15
      );
    });

    this.placeButton.on('pointerout', () => {
      this.placeButton.fillStyle(0x4CAF50);
      this.placeButton.fillRoundedRect(
        buttonX - buttonWidth/2, 
        buttonY - buttonHeight/2, 
        buttonWidth, 
        buttonHeight, 
        15
      );
    });
  }

  calculateGridBounds() {
    this.gridBounds.x = this.gridOffset.x;
    this.gridBounds.y = this.gridOffset.y;
    this.gridBounds.width = this.gridManager.cols * this.cellSize;
    this.gridBounds.height = this.gridManager.rows * this.cellSize;
  }

  drawGrid() {
    this.graphics = this.add.graphics();
    // Draw grid lines
    this.graphics.lineStyle(2, 0x000000);
    for (let y = 0; y <= this.gridManager.rows; y++) {
        this.graphics.lineBetween(
            this.gridOffset.x,
            this.gridOffset.y + y * this.cellSize,
            this.gridOffset.x + this.gridManager.cols * this.cellSize,
            this.gridOffset.y + y * this.cellSize
        );
    }
    for (let x = 0; x <= this.gridManager.cols; x++) {
        this.graphics.lineBetween(
            this.gridOffset.x + x * this.cellSize,
            this.gridOffset.y,
            this.gridOffset.x + x * this.cellSize,
            this.gridOffset.y + this.gridManager.rows * this.cellSize
        );
    }
    // Add white border around the entire board (new code)
    const borderGraphics = this.add.graphics();
    borderGraphics.lineStyle(4, 0xffffff); // White border, thickness 4px
    borderGraphics.strokeRect(
      this.gridOffset.x,
      this.gridOffset.y,
      this.gridManager.cols * this.cellSize,
      this.gridManager.rows * this.cellSize
    );
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
        this.gridManager.rotateTriangle();
        this.updateCursor();
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
          const direction = deltaY > 0 ? 1 : -1;
          this.gridManager.rotateTriangle(direction);
          this.updateCursor();
        }
      });

      this.input.on('pointerup', () => {
        this.isDragging = false;
      });
  }

  moveCursor(dx, dy) {
      this.cursorPos.x = Phaser.Math.Clamp(
          this.cursorPos.x + dx,
          0,
          this.gridManager.cols - 1
      );
      this.cursorPos.y = Phaser.Math.Clamp(
          this.cursorPos.y + dy,
          0,
          this.gridManager.rows - 1
      );
      this.updateCursor();
  }

  updateCursor() {
      if (!this.cursorTriangle) {
        this.cursorTriangle = this.add.graphics();
      }
      this.cursorTriangle.clear();
      
      const playerColor = this.gridManager.currentPlayer === 1 ? 0xff5555 : 0x5555ff;
      this.cursorTriangle.lineStyle(4, playerColor);
  
      const cellX = this.gridOffset.x + this.cursorPos.x * this.cellSize;
      const cellY = this.gridOffset.y + this.cursorPos.y * this.cellSize;
  
      // Draw preview based on rotation
      this.cursorTriangle.beginPath();
      switch (this.gridManager.selectedRotation) {
        case 0: // ╱ bottom-right angle
          this.cursorTriangle.moveTo(cellX, cellY + this.cellSize);
          this.cursorTriangle.lineTo(cellX + this.cellSize, cellY);
          this.cursorTriangle.lineTo(cellX, cellY);
          break;
        case 1: // ╲ top-right angle
          this.cursorTriangle.moveTo(cellX, cellY);
          this.cursorTriangle.lineTo(cellX, cellY + this.cellSize);
          this.cursorTriangle.lineTo(cellX + this.cellSize, cellY + this.cellSize);
          break;
        case 3: // ╱ top-left angle
          this.cursorTriangle.moveTo(cellX + this.cellSize, cellY);
          this.cursorTriangle.lineTo(cellX, cellY);
          this.cursorTriangle.lineTo(cellX + this.cellSize, cellY + this.cellSize);
          break;
        case 2: // ╲ bottom-left angle
          this.cursorTriangle.moveTo(cellX + this.cellSize, cellY + this.cellSize);
          this.cursorTriangle.lineTo(cellX, cellY + this.cellSize);
          this.cursorTriangle.lineTo(cellX + this.cellSize, cellY);
          break;
      }
      this.cursorTriangle.closePath();
      this.cursorTriangle.strokePath();
      this.cursorTriangle.setDepth(100);
  }

  placeCurrentTriangle() {

    if (this.players[this.gridManager.currentPlayer].type !== 'human') {
      console.log("Not human player's turn.");
      return;
    }
    const success = this.gridManager.placeTriangle(
      this.cursorPos.x,
      this.cursorPos.y
    );

    if (success) {
      // Claim enclosed triangles and calculate extra turns
      const placedTriangles = 1; // Initial placement
      const enclosedTriangles = this.gridManager.claimEnclosedTriangles();
      const totalClaimed = placedTriangles + enclosedTriangles;

      if (enclosedTriangles > 0) {
        this.gridManager.extraTurns += 1; // Add one extra turn for the successful enclosure
        console.log(`Player ${this.gridManager.currentPlayer} enclosed ${enclosedTriangles} triangle(s) and gets an extra turn.`);
      }

      // Update UI and turn logic
      this.redrawTriangles();
      this.updateScore();
      if (this.gridManager.isGridFull()) { // [cite: 98, 219]
          this.handleGameOver();
      } else {
          this.handleTurnSwitch(); // Only switch turn if game is not over
      }

      if (this.gridManager.isGridFull()) {
        const p1Score = this.gridManager.getLargestContiguous(1);
        const p2Score = this.gridManager.getLargestContiguous(2);
        
        let message = "It's a draw!";
        if (p1Score > p2Score) {
          this.gridManager.winRounds[1]++;
          message = "Player 1 wins!";
        } else if (p2Score > p1Score) {
          this.gridManager.winRounds[2]++;
          message = "Player 2 wins!";
        }
        
        this.showGameOver(message);
        this.time.delayedCall(2000, () => this.resetGame());
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
      if (this.players[this.gridManager.currentPlayer].type === 'ai') {
            this.time.delayedCall(100, this.triggerAIMove, [], this);
      }
    });
  }
  showGameOver(message) {
    if (this.gameOverText) this.gameOverText.destroy();
    
    this.gameOverText = this.add.text(
      this.cameras.main.width / 2,
      this.cameras.main.height / 2,
      message,
      { 
        font: '48px Arial',
        fill: '#FFFFFF',
        backgroundColor: '#000000',
        padding: { x: 20, y: 10 }
      }
    ).setOrigin(0.5).setScrollFactor(0);
  }

  resetGame() {
    this.gridManager.reset();
    this.redrawTriangles();
    this.updateScore();
    this.updateCursor();
    if (this.gameOverText) this.gameOverText.destroy();
  }

  handleTurnSwitch() {
    if (this.gridManager.extraTurns > 0) {
        this.gridManager.extraTurns--; // [cite: 229]
        console.log(`Player <span class="math-inline">\{this\.gridManager\.currentPlayer\} has an extra turn \(</span>{this.gridManager.extraTurns} remaining).`);
        // Current player continues, check if it's AI or Human
        if (this.players[this.gridManager.currentPlayer].type === 'ai') {
             this.time.delayedCall(100, this.triggerAIMove, [], this); // AI takes another turn
        } else {
             this.enableHumanInput(); // Ensure human input is enabled for the extra turn
        }
    } else {
        // Switch player
        this.gridManager.currentPlayer = this.gridManager.currentPlayer === 1 ? 2 : 1; // [cite: 230]
        console.log(`Switching to Player ${this.gridManager.currentPlayer}`);
        this.updateCursor(); // Update cursor color [cite: 204]

        // Check the type of the NEW current player
        if (this.players[this.gridManager.currentPlayer].type === 'ai') {
             this.disableHumanInput(); // Disable human input while AI thinks
             this.time.delayedCall(100, this.triggerAIMove, [], this); // Start AI's turn
        } else {
             this.enableHumanInput(); // Enable input for the human player
        }
    }
    // Consider adding a visual indicator for whose turn it is
  }

  redrawTriangles() {
    if (this.triangleGraphics) this.triangleGraphics.clear();
    this.triangleGraphics = this.add.graphics();
    this.triangleGraphics.setDepth(0); // Ensure triangles stay below cursor

    this.gridManager.grid.forEach((row, y) => {
      row.forEach((cell, x) => {
        cell.triangles.forEach(triangle => {
          this.drawTriangle(x, y, triangle.rotation, triangle.player);
        });
      });
    });
  }

  // Add this method to GameScene
triggerAIMove() {
  const currentPlayerNum = this.gridManager.currentPlayer;
  const aiInstance = this.aiInstances[currentPlayerNum];

  if (!aiInstance || this.players[currentPlayerNum].type !== 'ai') {
      console.error("triggerAIMove called for non-AI player or missing instance.");
      return;
  }

  console.log(`AI Player ${currentPlayerNum} is thinking...`);
    // Use a short delay to simulate thinking and prevent visual glitches
    this.time.delayedCall(300, () => { // Delay can be adjusted
      const move = aiInstance.predictMove(); // [cite: 29, 69]

      if (move && typeof move.x !== 'undefined' && typeof move.y !== 'undefined' && typeof move.rotation !== 'undefined') {
          console.log(`AI Player <span class="math-inline">\{currentPlayerNum\} chose\: \(</span>{move.x}, ${move.y}), Rot: ${move.rotation}`); // [cite: 42]
          // Apply the move
          this.gridManager.selectedRotation = move.rotation;
          // Optional: Move cursor for visual feedback
          this.cursorPos.x = move.x;
          this.cursorPos.y = move.y;
          this.updateCursor(); // Update visual cursor [cite: 204]

          // Place the triangle after another small delay for visual pacing
          this.time.delayedCall(150, () => {
              const success = this.gridManager.placeTriangle(move.x, move.y); // [cite: 114, 215]
              if (success) {
                  const enclosed = this.gridManager.claimEnclosedTriangles(); // [cite: 139, 217]
                  if (enclosed > 0) {
                    this.gridManager.extraTurns += 1;
                    console.log(`AI Player ${this.gridManager.currentPlayer} enclosed ${enclosed} triangle(s) and gets an extra turn.`);
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
  drawTriangle(x, y, rotation, player) {
      const color = player === 1 ? 0xff0000 : 0x0000ff;
      this.triangleGraphics.fillStyle(color);
      
      const cellX = this.gridOffset.x + x * this.cellSize;
      const cellY = this.gridOffset.y + y * this.cellSize;
  
      this.triangleGraphics.beginPath();
      switch (rotation) {
        case 0:
          this.triangleGraphics.moveTo(cellX, cellY + this.cellSize);
          this.triangleGraphics.lineTo(cellX + this.cellSize, cellY);
          this.triangleGraphics.lineTo(cellX, cellY);
          break;
        case 1:
          this.triangleGraphics.moveTo(cellX, cellY);
          this.triangleGraphics.lineTo(cellX, cellY + this.cellSize);
          this.triangleGraphics.lineTo(cellX + this.cellSize, cellY + this.cellSize);
          break;
        case 3:
          this.triangleGraphics.moveTo(cellX + this.cellSize, cellY);
          this.triangleGraphics.lineTo(cellX, cellY);
          this.triangleGraphics.lineTo(cellX + this.cellSize, cellY + this.cellSize);
          break;
        case 2:
          this.triangleGraphics.moveTo(cellX + this.cellSize, cellY + this.cellSize);
          this.triangleGraphics.lineTo(cellX, cellY + this.cellSize);
          this.triangleGraphics.lineTo(cellX + this.cellSize, cellY);
          break;
      }
      this.triangleGraphics.closePath();
      this.triangleGraphics.fill();
  }

  drawSlashTriangle(x, y, player) {
      const color = player === 1 ? 0xff0000 : 0x0000ff;
      this.triangleGraphics.fillStyle(color);
      this.triangleGraphics.beginPath();
      this.triangleGraphics.moveTo(
          this.gridOffset.x + x * this.cellSize,
          this.gridOffset.y + (y + 1) * this.cellSize
      );
      this.triangleGraphics.lineTo(
          this.gridOffset.x + (x + 1) * this.cellSize,
          this.gridOffset.y + y * this.cellSize
      );
      this.triangleGraphics.lineTo(
          this.gridOffset.x + x * this.cellSize,
          this.gridOffset.y + y * this.cellSize
      );
      this.triangleGraphics.closePath();
      this.triangleGraphics.fill();
  }

  drawBackslashTriangle(x, y, player) {
      const color = player === 1 ? 0xff0000 : 0x0000ff;
      this.triangleGraphics.fillStyle(color);
      this.triangleGraphics.beginPath();
      this.triangleGraphics.moveTo(
          this.gridOffset.x + x * this.cellSize,
          this.gridOffset.y + y * this.cellSize
      );
      this.triangleGraphics.lineTo(
          this.gridOffset.x + (x + 1) * this.cellSize,
          this.gridOffset.y + (y + 1) * this.cellSize
      );
      this.triangleGraphics.lineTo(
          this.gridOffset.x + (x + 1) * this.cellSize,
          this.gridOffset.y + y * this.cellSize
      );
      this.triangleGraphics.closePath();
      this.triangleGraphics.fill();
  }

  createScoreboard() {
    this.scoreText = this.add.text(
      180, 590,
      'Player 1: 0\nPlayer 2: 0',
      { font: '24px Arial', fill: '#ffffff', backgroundColor: '#000000' }
    ).setScrollFactor(0);
  }

  updateScore() {
    const p1Score = this.gridManager.getLargestContiguous(1);
    const p2Score = this.gridManager.getLargestContiguous(2);
    this.scoreText.setText([
      `Player 1: ${p1Score} (Wins: ${this.gridManager.winRounds[1]})`,
      `Player 2: ${p2Score} (Wins: ${this.gridManager.winRounds[2]})`
    ].join('\n'));
  }

  setupTouchControls() {
    // Single-touch drag to move cursor
    this.input.on('pointerdown', (pointer) => {
      // Check if touch started within grid bounds
      if (this.isWithinGrid(pointer.x, pointer.y)) {
        this.isDragging = true;
        this.dragStart.x = pointer.x;
        this.dragStart.y = pointer.y;
        this.cursorStart.x = this.cursorPos.x;
        this.cursorStart.y = this.cursorPos.y;
      }
    });

    this.input.on('pointermove', (pointer) => {
      if (this.isDragging) {
        // Calculate delta movement
        const deltaX = pointer.x - this.dragStart.x;
        const deltaY = pointer.y - this.dragStart.y;
        
        // Convert delta to grid cells
        const cellsX = Math.round(deltaX / this.cellSize);
        const cellsY = Math.round(deltaY / this.cellSize);
        
        // Update cursor position relative to start
        const newX = Phaser.Math.Clamp(
          this.cursorStart.x + cellsX,
          0,
          this.gridManager.cols - 1
        );
        const newY = Phaser.Math.Clamp(
          this.cursorStart.y + cellsY,
          0,
          this.gridManager.rows - 1
        );

        if (newX !== this.cursorPos.x || newY !== this.cursorPos.y) {
          this.cursorPos.x = newX;
          this.cursorPos.y = newY;
          this.updateCursor();
        }
      }
    });

    this.input.on('pointerup', () => {
      this.isDragging = false;
    });

  }

  updateCursorPosition(pointer) {
    // Convert to grid coordinates (works for both mouse/touch)
    const gridX = Phaser.Math.Clamp(
      Math.floor((pointer.x - this.gridOffset.x) / this.cellSize),
      0,
      this.gridManager.cols - 1
    );
    const gridY = Phaser.Math.Clamp(
      Math.floor((pointer.y - this.gridOffset.y) / this.cellSize),
      0,
      this.gridManager.rows - 1
    );

    if (gridX !== this.cursorPos.x || gridY !== this.cursorPos.y) {
      this.cursorPos.x = gridX;
      this.cursorPos.y = gridY;
      this.updateCursor();
    }
  }

  isWithinGrid(x, y) {
    return x >= this.gridBounds.x && 
            y >= this.gridBounds.y && 
            x <= this.gridBounds.x + this.gridBounds.width && 
            y <= this.gridBounds.y + this.gridBounds.height;
  }
}

class MenuScene extends Phaser.Scene {
  constructor() {
      super({ key: 'Menu' }); // Unique key for this scene
  }

  preload() {
      // No assets needed for this basic menu yet
  }

  create() {
      const centerX = this.cameras.main.width / 2;
      const centerY = this.cameras.main.height / 2;

      // --- Game Title ---
      this.add.text(centerX, centerY - 200, 'Triantica', {
          font: '48px Arial',
          fill: '#ffffff',
          align: 'center'
      }).setOrigin(0.5);

      // --- Button Styling ---
      const buttonStyle = {
          font: '32px Arial',
          fill: '#ffffff',
          backgroundColor: '#555555', // A distinct background color
          padding: { x: 20, y: 10 },
          align: 'center'
      };
      const buttonHoverStyle = {
          fill: '#ffffff', // Keep text white
          backgroundColor: '#777777' // Slightly lighter on hover
      };

      // --- Helper Function to Create Buttons ---
      const createMenuButton = (yPos, text, mode) => {
          const button = this.add.text(centerX, yPos, text, buttonStyle)
              .setOrigin(0.5)
              .setInteractive({ useHandCursor: true }); // Make it interactive and show hand cursor

          // Click Action
          button.on('pointerdown', () => {
              console.log(`Starting game mode: ${mode}`);
              // Start the GameScene and pass the selected mode as data
              this.scene.start('Game', { mode: mode });
          });

          // Hover Effects
          button.on('pointerover', () => {
              button.setStyle(buttonHoverStyle);
          });
          button.on('pointerout', () => {
              button.setStyle(buttonStyle);
          });

          return button;
      };

      // --- Create the Buttons ---
      createMenuButton(centerY - 50, 'Human vs Human', 'HvH');
      createMenuButton(centerY + 50, 'Human vs AI', 'HvAI');
      createMenuButton(centerY + 150, 'AI vs AI (Visual)', 'AvA');
      // Optional: Add a button for dedicated AI training if you implement that separately
      // createMenuButton(centerY + 250, 'Train AI (Fast)', 'TrainAI');
      // --- Create the Training Button ---
      const trainButtonY = centerY + 250; // Adjust Y position as needed
      const trainButton = this.add.text(centerX, trainButtonY, 'Train AI (Self-Play)', buttonStyle)
          .setOrigin(0.5)
          .setInteractive({ useHandCursor: true });

      // --- Training Button Action ---
      trainButton.on('pointerdown', async () => {
          // Prevent clicking again while training
          trainButton.disableInteractive();
          trainButton.setText('Training...'); // Update text
          trainButton.setStyle({ ...buttonStyle, backgroundColor: '#888888' }); // Visual feedback: busy

          console.log("--- Starting AI Training Process ---");

          try {
              // Ensure necessary classes are available (assuming global scope based on previous discussions)
              if (typeof GridManager === 'undefined' || typeof AIPlayer === 'undefined' || typeof brain === 'undefined') {
                  console.error("Error: GridManager, AIPlayer, or brain.js class not available globally!");
                  trainButton.setText('Error: Classes Missing');
                  // Re-enable after delay
                  this.time.delayedCall(3000, () => {
                      trainButton.setText('Train AI (Self-Play)');
                      trainButton.setStyle(buttonStyle);
                      trainButton.setInteractive();
                  });
                  return;
              }

              // Create temporary instances for training simulation
              const tempGridManager = new GridManager(10, 10); // Or use actual grid dimensions
              const trainingAI = new AIPlayer(tempGridManager, 'medium', 1); // Train AI as player 1 perspective

              // console.log(`Loading saved network for AI Player 1`);
              // const success = trainingAI.importNetwork(); // [cite: 90]
              // if (!success) { console.error("Failed to import network data."); }

              // Check if preloading was attempted
              const loadPromise = this.registry.get('networkLoadPromise');
              const tempAI = this.registry.get('tempAIInstanceForStatus');
        
              let loadedSuccessfully = false;
              if (loadPromise && tempAI) {
                  console.log(`AI: Found preload promise. Waiting for it to settle...`);
                  try {
                      // Wait for the promise initiated by BootScene to finish
                      const preloadFinished = await loadPromise; // Waits if still pending, resolves quickly if already done
        
                      // Check the status flag on the temporary AI instance used in BootScene
                      if (tempAI.isNetworkLoaded) {
                          console.log(`AI: Preload successful. Copying network state...`);
                          const networkState = tempAI.exportNetwork();
                          if (networkState) {
                                // Apply the preloaded state to the actual game AI instance
                              trainingAI.network.fromJSON(networkState);
                              trainingAI.isNetworkLoaded = true; // Mark this instance as ready
                              loadedSuccessfully = true;
                          } else {
                              console.error(`AI: Preload seemed successful but failed to export network state from temporary AI.`);
                          }
                      } else {
                            console.log(`AI: Preload attempt finished but was unsuccessful or network was not marked as loaded.`);
                      }
                  } catch (promiseError) {
                        console.error(`AI: Error occurred while awaiting preload promise:`, promiseError);
                  }
              } else {
                    console.log(`AI: No preload attempt found in registry.`);
              }

              // --- Step 1: Generate Training Data via Self-Play ---
              const numTrainingGames = 2000; // << CONFIGURABLE: Number of games to simulate for data
              console.log(`Simulating ${numTrainingGames} self-play game(s)...`);
              for (let i = 0; i < numTrainingGames; i++) {
                  await trainingAI.runSelfPlayGame(); // runSelfPlayGame is async [cite: 62]
                  console.log(`Finished self-play game ${i + 1} / ${numTrainingGames}`);
                  // Optional: Update button text periodically, e.g., trainButton.setText(`Simulating ${i+1}/${numTrainingGames}`);
              }
              trainButton.setText('Training Network...'); // Update status

              // --- Step 2: Train the Network ---
              if (trainingAI.trainingData.length > 0) {
                  console.log(`Training network with ${trainingAI.trainingData.length} samples...`);
                  // Train the network with collected data. Adjust iterations as needed.
                  trainingAI.trainNetwork(1000); // << CONFIGURABLE: Number of training iterations [cite: 54, 55]
                  console.log("Network training complete.");

                  // --- Step 3: Save the Trained Network ---
                  const trainedNetworkJSON = trainingAI.exportNetwork(); // Get network state [cite: 89]
                  if (trainedNetworkJSON) {
                      try {
                        const networkString = JSON.stringify(trainedNetworkJSON, null, 2);
                        // Action 1: Save to localStorage (Default)
                          localStorage.setItem('trainedAI_P1', networkString);
                          console.log("Trained network saved to localStorage key 'trainedAI_P1'.");

                        // Action 2: Trigger File Download
                          const blob = new Blob([networkString], { type: 'application/json' });
                          const url = URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = 'trainedAI_network.json'; // Filename
                          document.body.appendChild(a);
                          a.click();
                          document.body.removeChild(a);
                          URL.revokeObjectURL(url);
                          console.log("Trained network download initiated.");

                          trainButton.setText('Training Complete!'); // Success message
                      } catch (e) {
                          console.error("Error saving network to localStorage or initiating download:", e);
                          trainButton.setText('Save Error!');
                      }
                  } else {
                        console.log("Could not export network data after training.");
                        trainButton.setText('Export Error!');
                  }
              } else {
                  console.log("No training data was generated. Skipping training and saving.");
                  trainButton.setText('No Data Generated');
              }

          } catch (error) {
              console.error("An error occurred during the AI training process:", error);
              trainButton.setText('Training Error!'); // Error message
              trainButton.setStyle({ ...buttonStyle, backgroundColor: '#AA0000' }); // Error color
          } finally {
              // Re-enable the button after a delay, allowing user to see the final message
              this.time.delayedCall(3000, () => { // 3-second delay
                  trainButton.setText('Train AI (Self-Play)'); // Reset text
                  trainButton.setStyle(buttonStyle); // Reset style
                  trainButton.setInteractive(); // Re-enable
              });
              console.log("--- AI Training Process Finished ---");
          }
      });

      // Add hover effects (only when enabled)
      trainButton.on('pointerover', () => {
          if (trainButton.input.enabled) { // Check if interactive before changing style
                trainButton.setStyle(buttonHoverStyle);
          }
      });
      trainButton.on('pointerout', () => {
          if (trainButton.input.enabled) {
                trainButton.setStyle(buttonStyle);
          }
      });
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
  width: 600,
  height: 800,
  scene:  [BootScene, MenuScene, GameScene] 
};

new Phaser.Game(config);