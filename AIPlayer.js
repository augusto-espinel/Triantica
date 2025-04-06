// Import brain.js (make sure it's installed: npm install brain.js)
// Note: brain.js might work better when run in a Node.js environment for training,
// but we'll use it directly in the browser for prediction for now.
// If performance becomes an issue, consider server-side AI or Web Workers.
// Since we don't have direct access to 'require' or 'import' in this Phaser setup easily without a bundler,
// we'll assume brain.js is loaded globally via a <script> tag in index.html for simplicity in this context.
// <script src="https://cdn.jsdelivr.net/npm/brain.js@2.0.0-beta.23/dist/browser.min.js"></script>

class AIPlayer {
    constructor(gridManager, difficulty = 'medium', playerNumber = 2) {
        if (typeof brain === 'undefined') {
            console.error("Brain.js not loaded. Please include it via a script tag.");
            // Provide a fallback or throw an error
            this.network = null;
            this.playerNumber = playerNumber;
            this.gridManager = gridManager;
            this.difficulty = difficulty;
            return;
        }

        this.gridManager = gridManager;
        this.playerNumber = playerNumber;
        this.difficulty = difficulty;
        this.network = this._createNetwork(difficulty);
        this.isNetworkLoaded = false;

        // Placeholder for training data - ideally loaded/saved
        this.trainingData = [];
    }

    _createNetwork(difficulty) {
        const inputSize = this.gridManager.rows * this.gridManager.cols * 4 + 1; // 10x10 grid * 4 values per cell + 1 for current player
        const outputSize = this.gridManager.rows * this.gridManager.cols * 4; // 10x10 grid * 4 possible rotations

        let hiddenLayers;
        switch (difficulty) {
            case 'easy':
                hiddenLayers = [Math.floor(inputSize / 4)]; // Smaller network
                break;
            case 'hard':
                hiddenLayers = [Math.floor(inputSize / 1.5), Math.floor(inputSize / 2.5)]; // Larger network
                break;
            case 'medium':
            default:
                hiddenLayers = [Math.floor(inputSize / 2)]; // Default size
                break;
        }

        console.log(`Creating AI network (Difficulty: ${difficulty})`);
        console.log(`Input Size: ${inputSize}, Hidden Layers: ${hiddenLayers}, Output Size: ${outputSize}`);

        // Using a simple Feedforward network for now. LSTM might be better for sequential decisions.
        return new brain.NeuralNetwork({
            inputSize: inputSize,
            hiddenLayers: hiddenLayers,
            outputSize: outputSize,
            activation: 'sigmoid' // Common activation function
        });
    }

    // Convert the current grid state into a flat array for the network
    _getStateRepresentation() {
        const state = [];
        for (let y = 0; y < this.gridManager.rows; y++) {
            for (let x = 0; x < this.gridManager.cols; x++) {
                const cell = this.gridManager.grid[y][x];
                // Represent up to two triangles per cell
                const t1 = cell.triangles[0];
                const t2 = cell.triangles[1];

                // Normalize player: 0=empty, 0.5=P1, 1=P2 (or adjust based on current AI player?)
                // Let's use: 0=empty, 1=current AI player, -1=opponent
                const normalizePlayer = (p) => {
                    if (!p) return 0;
                    return p === this.playerNumber ? 1 : -1;
                };

                // Normalize rotation: 0, 0.25, 0.5, 0.75
                const normalizeRotation = (r) => (r === undefined || r === -1) ? -1 : r / 4.0;

                state.push(normalizePlayer(t1?.player));
                state.push(normalizeRotation(t1?.rotation ?? -1));
                state.push(normalizePlayer(t2?.player));
                state.push(normalizeRotation(t2?.rotation ?? -1));
            }
        }
        // Add current player perspective (e.g., 1 if it's AI's turn, -1 otherwise - though AI only predicts on its turn)
        state.push(this.gridManager.currentPlayer === this.playerNumber ? 1 : -1);
        // console.log("State Representation Size:", state.length);
        return state;
    }

    // Predict the best move based on the current state
    predictMove() {
        if (!this.network) {
            console.warn("AI Network not available. Making random valid move.");
            return this.getRandomValidMove();
        }

        const state = this._getStateRepresentation();
        const output = this.network.run(state); // Get probabilities/scores for each move

        // Map output index to move {x, y, rotation}
        const moves = [];
        for (let i = 0; i < output.length; i++) {
            const score = output[i];
            const rotation = i % 4;
            const cellIndex = Math.floor(i / 4);
            const x = cellIndex % this.gridManager.cols;
            const y = Math.floor(cellIndex / this.gridManager.cols);
            moves.push({ x, y, rotation, score });
        }

        // Sort moves by score in descending order
        moves.sort((a, b) => b.score - a.score);

        // Find the highest-scoring valid move
        for (const move of moves) {
            // Temporarily set selectedRotation to check validity
            const originalRotation = this.gridManager.selectedRotation;
            this.gridManager.selectedRotation = move.rotation;

            const cell = this.gridManager.grid[move.y][move.x];
            let isValid = false;
            if (cell.triangles.length < 2) {
                 let hasOverlap = false;
                 if (cell.triangles.length > 0)  {
                    const newTriangleGroup = this.gridManager.getDiagonalGroup(move.rotation);
                    hasOverlap = !(this.gridManager.getDiagonalGroup(cell.triangles[0].rotation) === newTriangleGroup) ||
                    cell.triangles[0].rotation === move.rotation;
                 }
                 isValid = this.gridManager.isValidPlacement(move.x, move.y) && !hasOverlap;
            }


            // Restore original rotation
            this.gridManager.selectedRotation = originalRotation;

            if (isValid) {
                // console.log(`AI Prediction: (${move.x}, ${move.y}), Rot: ${move.rotation}, Score: ${move.score.toFixed(4)}`);
                return { x: move.x, y: move.y, rotation: move.rotation };
            }
        }

        // If no valid move found by the network (shouldn't happen in a normal game state unless grid is full)
        console.warn("AI couldn't find a valid move from network output. Making random valid move.");
        return this.getRandomValidMove();
    }

    getRandomValidMove(tempGridManager) {
        const possibleMoves = [];
        for (let y = 0; y < tempGridManager.rows; y++) {
            for (let x = 0; x < tempGridManager.cols; x++) {
                for (let rotation = 0; rotation < 4; rotation++) {
                    // Temporarily set selectedRotation to check validity
                    const originalRotation = tempGridManager.selectedRotation;
                    tempGridManager.selectedRotation = rotation;

                    const cell = tempGridManager.grid[y][x];
                     let isValid = false;
                     if (cell.triangles.length < 2) {
                          let hasOverlap = false;
                          if (cell.triangles.length > 0)  {
                             const newTriangleGroup = tempGridManager.getDiagonalGroup(rotation);
                             hasOverlap = !(tempGridManager.getDiagonalGroup(cell.triangles[0].rotation) === newTriangleGroup) ||
                             cell.triangles[0].rotation === rotation;
                          }
                          isValid = tempGridManager.isValidPlacement(x, y) && !hasOverlap;
                     }

                    // Restore original rotation
                    tempGridManager.selectedRotation = originalRotation;

                    if (isValid) {
                        possibleMoves.push({ x, y, rotation });
                    }
                }
            }
        }

        if (possibleMoves.length === 0) {
            console.error("AI: No valid moves found at all!");
            return null; // Or handle game end condition
        }

        // Select a random move from the valid ones
        const randomIndex = Math.floor(Math.random() * possibleMoves.length);
        return possibleMoves[randomIndex];
    }


    // --- Self-Play Training (Basic Structure) ---

    // Call this periodically or after games to train
    trainNetwork(iterations = 500, learningRate = 0.1) {
        if (!this.network || this.trainingData.length === 0) {
            console.log("AI: No training data or network available.");
            return;
        }

        console.log(`AI: Starting training with ${this.trainingData.length} samples...`);
        const result = this.network.train(this.trainingData, {
            iterations: iterations,
            log: (details) => console.log(`Training - Iteration: ${details.iterations}, Error: ${details.error}`),
            logPeriod: 10,
            learningRate: learningRate,
            errorThresh: 0.01 // Stop if error is low enough
        });
        console.log("AI: Training complete.", result);

        // Check if training likely succeeded based on the result object
        // (brain.js train result might vary; adapt this condition if needed)
        if (result && result.error < 1) { // Example condition: error is below 1
            console.log("Setting network status to 'loaded/trained'.");
            this.isNetworkLoaded = true; // <<< SET FLAG TO TRUE AFTER SUCCESSFUL TRAINING
        } else {
            console.log("Training might not have reached desired threshold, network not marked as 'loaded'.");
            // Optionally set to false if you only want loading to count:
            this.isNetworkLoaded = false;
        }
        // Optional: Clear training data after training cycle? Or keep accumulating?
        // this.trainingData = [];
    }

    // Add data from a completed game or simulation
    addTrainingData(state, move, reward) {
        // Convert move {x, y, rotation} to the network's output format (one-hot encoding)
        const outputSize = this.gridManager.rows * this.gridManager.cols * 4; // Get output size from network
        const output = new Array(outputSize).fill(0); // Initialize output array
        const moveIndex = (move.y * this.gridManager.cols + move.x) * 4 + move.rotation;

        // Ensure moveIndex is valid
        if (moveIndex < 0 || moveIndex >= outputSize) {
            console.error(`Invalid move index calculated: ${moveIndex}. Move:`, move, `Output size: ${outputSize}`);
            return; // Skip adding invalid data
        }

        // --- New Reward Mapping to Target Output (e.g., 0 to 1) ---

        // Re-estimate reward range including cluster bonus/penalty
        const clusterRewardFactor = 0.05; // <<<< Must match the factor used in runSelfPlayGame
        // Estimate max possible cluster difference (theoretically grid size, practically maybe less)
        // Max possible cluster size is rows * cols * 2 triangles? Let's estimate based on grid area for simplicity.
        const maxPossibleClusterDiff = this.gridManager.rows * this.gridManager.cols;
        const maxClusterBonus = maxPossibleClusterDiff * clusterRewardFactor;
        const minClusterPenalty = -maxPossibleClusterDiff * clusterRewardFactor; // Symmetric penalty

        // Estimate max claim bonus (e.g., 4 claims * 0.3 reward/claim = 1.2) - adjust if needed
        const maxClaimBonus = 1.2;

        const minReward = -1.0 + minClusterPenalty; // Loss + max cluster penalty
        const maxReward = 1.0 + maxClaimBonus + maxClusterBonus; // Win + max claim + max cluster bonus

        // Scale the reward linearly to the range [0, 1]
        let targetOutput = 0; // Default for safety
        if ((maxReward - minReward) !== 0) { // Avoid division by zero if range is somehow 0
            targetOutput = (reward - minReward) / (maxReward - minReward);
        } else if (reward >= maxReward) { // Handle edge case where reward matches single possible value
            targetOutput = 1;
        }


        // Clamp the value strictly between 0.01 and 0.99 (good practice for sigmoid/similar activations)
        targetOutput = Math.max(0.01, Math.min(0.99, targetOutput));

        // --- End New Reward Mapping ---

        output[moveIndex] = targetOutput; // Set the target output for the specific move taken

        this.trainingData.push({ input: state, output: output });
    }

    // Simulate a game for training data generation (AI vs AI)
    async runSelfPlayGame() {
        console.log("Starting self-play game for training...");
        // const tempGridManager = new GridManager(this.gridManager.rows, this.gridManager.cols); // Use a temporary manager
        const gameHistory = []; // Store {state, move, player} for reward assignment later

        // Create opponent AI (could be same network or a different instance/difficulty)
        const opponentAI = new AIPlayer(this.gridManager, this.difficulty, this.playerNumber === 1 ? 2 : 1);
        opponentAI.network.fromJSON(this.network.toJSON()); // Clone network state if using same
        opponentAI.isNetworkLoaded = this.isNetworkLoaded; // Copy flag status

        let currentPlayerAI = this.playerNumber === 1 ? this : opponentAI;
        let movesCount = 0;
        const maxMoves = this.gridManager.rows * this.gridManager.cols * 2 * 1.5; // Safety break

        while (!this.gridManager.isGridFull() && movesCount < maxMoves) {
            const currentManager = currentPlayerAI.gridManager; // Ensure AI uses the temp manager
            const state = currentPlayerAI._getStateRepresentation(); // Get state BEFORE the move
            const currentPlayerNum = currentPlayerAI.playerNumber;
            this.gridManager.currentPlayer = currentPlayerNum; // Set current player in the grid manager
            const opponentPlayerNum = currentPlayerNum === 1 ? 2 : 1;
            // --- Conditionally choose move based on flag ---
            let move;
            if (this.isNetworkLoaded && movesCount>0) { // Avoid using network on first move (otherwise it will start at same plce)
                // Use prediction if network is loaded/trained
                // console.log(`Player ${currentPlayerAI.playerNumber} using predictMove (Network Ready: ${currentPlayerAI.isNetworkLoaded})`); // Optional detailed log
                move = currentPlayerAI.predictMove();
            } else {
                // Use random moves if network is not ready
                // console.log(`Player ${currentPlayerAI.playerNumber} using getRandomValidMove (Network Ready: ${currentPlayerAI.isNetworkLoaded})`); // Optional detailed log
                move = currentPlayerAI.getRandomValidMove(this.gridManager);
            }
            // --- End conditional move choice ---

            if (!move) {
                console.error("Self-play: AI failed to find a move. Ending game.");
                break; // Should not happen
            }

            // Apply the move to the temporary grid manager
            this.gridManager.selectedRotation = move.rotation;
            const success = this.gridManager.placeTriangle(move.x, move.y);

            if (!success) {
                console.error(`Self-play: AI predicted an invalid move (${move.x}, ${move.y}, ${move.rotation}). This shouldn't happen.`);
                 // If predictMove includes validation, this is unlikely. If it happens, maybe penalize?
                 // For now, just break or try random move.
                 break;
            }

            // Handle claims and turn switching within the temporary manager
            const claimed = this.gridManager.claimEnclosedTriangles();

            // --- Calculate cluster sizes AFTER the move and claims ---
            const currentPlayerClusterSize = currentManager.getLargestContiguous(currentPlayerNum);
            const opponentPlayerClusterSize = currentManager.getLargestContiguous(opponentPlayerNum);
            // --- End cluster size calculation ---

            // Store step information including cluster sizes
            gameHistory.push({
                state: state,
                move: move,
                player: currentPlayerNum, // Player who made the move
                claimed: claimed,
                clusterSize: currentPlayerClusterSize,          // << Player's cluster size AFTER move
                opponentClusterSize: opponentPlayerClusterSize // << Opponent's cluster size AFTER move
            });

            this.gridManager.extraTurns += Math.floor((1 + claimed) / 2);

            if (this.gridManager.extraTurns > 0) {
                this.gridManager.extraTurns--;
                // Same player goes again
            } else {
                this.gridManager.currentPlayer = this.gridManager.currentPlayer === 1 ? 2 : 1;
                currentPlayerAI = currentPlayerAI === this ? opponentAI : this;
            }
            movesCount++;
        }
        // const test = currentPlayerAI._getStateRepresentation(); // use for debugging

        // --- Game End: Assign Rewards ---
        let winner = 0; // 0 = draw, 1 = P1 wins, 2 = P2 wins
        if (this.gridManager.isGridFull()) {
            const p1Score = this.gridManager.getLargestContiguous(1);
            const p2Score = this.gridManager.getLargestContiguous(2);
            if (p1Score > p2Score) winner = 1;
            else if (p2Score > p1Score) winner = 2;
        } else {
            // Game ended prematurely (e.g., invalid move prediction, max moves)
            // Could assign penalty or default to draw/opponent win
            console.warn("Self-play game ended prematurely.");
            winner = 0; // Treat as draw for now
        }
        
        this.gridManager.reset();   // Reset the grid manager for next game
        console.log(`Self-play game finished. Winner: ${winner === 0 ? 'Draw' : `Player ${winner}`}`);

        // Add training data with rewards
        gameHistory.forEach(step => {
            // 1. Base reward for winning/losing the game
            let finalReward  = 0;
            if (winner !== 0) {
                finalReward  = step.player === winner ? 1 : -1; // Win = +1, Loss = -1
            }

            // 2. Intermediate reward for claiming triangles on THIS specific move
            let claimReward  = 0;
            if (step.claimed > 0) { claimReward = 0.3 * step.claimed; } // Example value
            // Adjust based on how many triangles were claimed (e.g., 0.3 for each triangle claimed)

            // 3. Intermediate reward for cluster size advantage
            // Reward based on the difference between the player's largest cluster
            // and the opponent's largest cluster AFTER the move was made.  
            let clusterReward = 0;
            const clusterDifference = step.clusterSize - step.opponentClusterSize;
            // Assign a small reward proportional to the size advantage. TUNE THIS FACTOR!
            const clusterRewardFactor = 0.05; // <<<< Example factor, adjust based on testing
            clusterReward = clusterDifference * clusterRewardFactor;
            
            // 4. Combine rewards
            let totalReward = finalReward + claimReward + clusterReward;

            // Could add intermediate rewards later (e.g., for claiming triangles)
            this.addTrainingData(step.state, step.move, totalReward);
        });

        console.log(`Added ${gameHistory.length} training samples from self-play.`);

        // Optional: Trigger training immediately after a game
        // this.trainNetwork(50); // Train for a few iterations
    }

     // --- Load/Save Network ---
     exportNetwork() {
        if (!this.network) return null;
        return this.network.toJSON();
    }

    /**
     * Imports network state, trying a relative file path first, then localStorage.
     * NOTE: Fetching from a relative path usually only works when served via HTTP/S,
     * not when opening files directly using file:///.
     * @returns {Promise<boolean>} True if loading was successful, false otherwise.
     */
    async importNetwork() { // Make the function async to use await for fetch
        if (!this.network) {
            console.error("Network not initialized, cannot import.");
            this.isNetworkLoaded = false; // Ensure flag is false
            return false;
        }
        console.log("Attempting to load network data...");

        let jsonData = null;
        let sourceUsed = 'none'; // Track where data came from
        this.isNetworkLoaded = false; // Reset flag at start of import attempt

        // --- Attempt 1: Fetch from relative JSON file ---
        const filePath = './trainedAI_network.json'; // Assumed filename in the same folder (relative)
        try {
            console.log(`Attempting to fetch network data from file: ${filePath}`);
            const response = await fetch(filePath, { cache: 'no-store' }); // Fetch the file, prevent caching

            if (response.ok) { // Check if status code is 2xx (e.g., 200 OK)
                jsonData = await response.json(); // Parse the response body as JSON
                console.log("Successfully fetched and parsed network data from file.");
                sourceUsed = 'file';
            } else {
                // Log non-critical failure (e.g., 404 Not Found), proceed to localStorage
                console.log(`Failed to fetch file '${filePath}': ${response.status} ${response.statusText}. Trying localStorage.`);
            }
        } catch (fetchError) {
            // Log errors like network issues or CORS problems if running locally
            console.log(`Error fetching file '${filePath}': ${fetchError.message}. Trying localStorage.`);
        }

        // --- Attempt 2: Fallback to localStorage (if file fetch failed) ---
        if (sourceUsed !== 'file') {
            console.log("Attempting to load network data from localStorage key 'trainedAI_P1'...");
            try {
                const savedData = localStorage.getItem('trainedAI_P1');
                if (savedData) {
                    jsonData = JSON.parse(savedData);
                    console.log("Successfully loaded network data from localStorage.");
                    sourceUsed = 'localStorage';
                } else {
                    console.log("No network data found in localStorage either.");
                }
            } catch (parseError) {
                console.error("Error parsing network data from localStorage:", parseError);
                // If localStorage data is corrupt, stop here.
                this.isNetworkLoaded = false; // Ensure flag is false on parse error
                return false;
            }
        }

        // --- If data was loaded from either source, apply it ---
        if (jsonData) {
            try {
                this.network.fromJSON(jsonData); // Apply the loaded data
                console.log(`AI Network state loaded successfully from ${sourceUsed}.`);
                this.isNetworkLoaded = true; // <<< SET FLAG TO TRUE ON SUCCESS
                return true; // Indicate success
            } catch (applyError) {
                console.error("Failed to apply loaded JSON data to AI Network state:", applyError);
                this.isNetworkLoaded = false; // Ensure flag is false on apply error
                return false; // Indicate failure
            }
        } else {
            // If jsonData is still null after trying both sources
            console.log("Failed to load network data from any source.");
            this.isNetworkLoaded = false; // Ensure flag is false if no data loaded
            return false; // Indicate failure
        }
    }

}
