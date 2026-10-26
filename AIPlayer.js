// Import brain.js (make sure it's installed: npm install brain.js)
// Note: brain.js might work better when run in a Node.js environment for training,
// but we'll use it directly in the browser for prediction for now.
// If performance becomes an issue, consider server-side AI or Web Workers.
// Since we don't have direct access to 'require' or 'import' in this Phaser setup easily without a bundler,
// we'll assume brain.js is loaded globally via a <script> tag in index.html for simplicity in this context.
// <script src="https://cdn.jsdelivr.net/npm/brain.js@2.0.0-beta.23/dist/browser.min.js"></script>

// Global TF.js Convolutional Network wrapper for browser inference
class BrowserConvNetwork {
    constructor(jsonData = null) {
        this.type = 'ConvNetwork';
        this.rows = 10;
        this.cols = 10;
        this.channels = 18;
        this.numFilters = [32, 64, 64];
        this.policySize = 400;
        this.model = null;
        this.rawJSON = null;
        if (jsonData) {
            this.fromJSON(jsonData);
        }
    }

    _buildModel() {
        const input = tf.input({ shape: [this.rows, this.cols, this.channels] });
        let x = input;
        for (let i = 0; i < this.numFilters.length; i++) {
            x = tf.layers.conv2d({
                filters: this.numFilters[i],
                kernelSize: 3,
                padding: 'same',
                kernelInitializer: 'glorotUniform',
                biasInitializer: 'zeros'
            }).apply(x);
            x = tf.layers.batchNormalization().apply(x);
            x = tf.layers.reLU().apply(x);
        }

        const policyConv = tf.layers.conv2d({
            filters: 2,
            kernelSize: 1,
            padding: 'same',
            kernelInitializer: 'glorotUniform'
        }).apply(x);
        const policyFlat = tf.layers.flatten().apply(policyConv);
        const policyDense = tf.layers.dense({
            units: this.policySize,
            activation: 'sigmoid',
            kernelInitializer: 'glorotUniform'
        }).apply(policyFlat);

        const valueConv = tf.layers.conv2d({
            filters: 1,
            kernelSize: 1,
            padding: 'same',
            kernelInitializer: 'glorotUniform'
        }).apply(x);
        const valueFlat = tf.layers.flatten().apply(valueConv);
        const valueDense1 = tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'glorotUniform'
        }).apply(valueFlat);
        const valueDense2 = tf.layers.dense({
            units: 1,
            activation: 'tanh',
            kernelInitializer: 'glorotUniform'
        }).apply(valueDense1);

        this.model = tf.model({
            inputs: input,
            outputs: [policyDense, valueDense2]
        });
    }

    fromJSON(json) {
        if (typeof tf === 'undefined') {
            throw new Error('TensorFlow.js (tf) is not loaded in the browser.');
        }
        this.rawJSON = json;
        tf.tidy(() => {
            this.rows = json.architecture?.rows || 10;
            this.cols = json.architecture?.cols || 10;
            this.channels = json.architecture?.channels || 18;
            this.numFilters = json.architecture?.numFilters || [32, 64, 64];
            this.policySize = json.architecture?.policySize || 400;

            this._buildModel();

            if (json.weights) {
                let weightIdx = 0;
                for (const layer of this.model.layers) {
                    const layerWeights = layer.getWeights();
                    if (layerWeights.length > 0 && weightIdx < json.weights.length) {
                        const saved = json.weights[weightIdx];
                        const restored = saved.data.map((arr, i) => {
                            const original = layerWeights[i];
                            return tf.tensor(arr, original.shape);
                        });
                        try {
                            layer.setWeights(restored);
                        } catch (e) {
                            console.warn('ConvNetwork layer weight shape mismatch:', e);
                        }
                        restored.forEach(t => t.dispose());
                        weightIdx++;
                    }
                }
            }
        });
    }

    toJSON() {
        return this.rawJSON;
    }

    run(input1d) {
        return tf.tidy(() => {
            const inputTensor = tf.tensor4d(new Float32Array(input1d), [1, this.rows, this.cols, this.channels]);
            const [policyOut, valueOut] = this.model.predict(inputTensor);
            return {
                policy: Array.from(policyOut.dataSync()),
                value: valueOut.dataSync()[0]
            };
        });
    }
}

class BrowserFFNetwork {
    constructor(jsonData = null) {
        this.type = 'TFNetwork';
        this.sizes = [401, 200, 400];
        this.model = null;
        this.rawJSON = null;
        if (jsonData) {
            this.fromJSON(jsonData);
        }
    }

    fromJSON(json) {
        if (typeof tf === 'undefined') {
            throw new Error('TensorFlow.js (tf) is not loaded in the browser.');
        }
        this.rawJSON = json;
        tf.tidy(() => {
            this.sizes = json.sizes || [401, 200, 400];
            this.model = tf.sequential();
            for (let i = 1; i < this.sizes.length; i++) {
                this.model.add(tf.layers.dense({
                    inputShape: i === 1 ? [this.sizes[0]] : undefined,
                    units: this.sizes[i],
                    activation: 'sigmoid'
                }));
            }
            if (json.layers) {
                for (let i = 0; i < this.model.layers.length; i++) {
                    const layerData = json.layers[i + 1];
                    if (layerData && layerData.weights && layerData.biases) {
                        const w = tf.tensor2d(layerData.weights).transpose();
                        const b = tf.tensor1d(layerData.biases);
                        this.model.layers[i].setWeights([w, b]);
                        w.dispose();
                        b.dispose();
                    }
                }
            }
        });
    }

    toJSON() {
        return this.rawJSON;
    }

    run(input1d) {
        return tf.tidy(() => {
            const inputTensor = tf.tensor2d([input1d]);
            const out = this.model.predict(inputTensor);
            return Array.from(out.dataSync());
        });
    }
}

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
        this.explorationFactor = 0.2; // Example: 10% chance to explore (choose random move)
        this.immediacyFactor = 0.9; // Example: 10% chance to choose based on neural network or short term cluster maximization
        this.isNetworkLoaded = false; // Flag to check if the network is loaded/trained

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

    // Convert current grid state into 10x10x18 canonical one-hot representation (1800 floats) for ConvNetwork
    _getCanonicalStateRepresentation() {
        const rows = this.gridManager.rows;
        const cols = this.gridManager.cols;
        const state = new Float32Array(rows * cols * 18);

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const cell = this.gridManager.grid[r][c];
                const cellOffset = (r * cols + c) * 18;

                for (let group = 0; group < 2; group++) {
                    const groupOffset = cellOffset + group * 9;
                    const tri = cell.triangles.find(t => this.gridManager.getDiagonalGroup(t.rotation) === group);

                    if (!tri) {
                        state[groupOffset] = 1;
                    } else {
                        const ownerOffset = tri.player === this.playerNumber ? 0 : 4;
                        state[groupOffset + 1 + ownerOffset + tri.rotation] = 1;
                    }
                }
            }
        }
        return Array.from(state);
    }

    loadNetworkJSON(jsonData) {
        if (!jsonData) return false;
        if (jsonData.type === 'ConvNetwork') {
            this.network = new BrowserConvNetwork(jsonData);
            this.networkType = 'ConvNetwork';
            this.isNetworkLoaded = true;
            return true;
        } else if (jsonData.type === 'TFNetwork') {
            this.network = new BrowserFFNetwork(jsonData);
            this.networkType = 'TFNetwork';
            this.isNetworkLoaded = true;
            return true;
        } else {
            if (!this.network || typeof this.network.fromJSON !== 'function' || this.networkType === 'ConvNetwork' || this.networkType === 'TFNetwork') {
                this.network = this._createNetwork(this.difficulty);
            }
            this.network.fromJSON(jsonData);
            this.networkType = 'brain';
            this.isNetworkLoaded = true;
            return true;
        }
    }

    // Predict the best move based on the current state
    predictMove() {
        if (!this.network) {
            console.warn("AI Network not available. Making random valid move.");
            return this.getRandomValidMove(this.gridManager);
        }

        // --- Epsilon-Greedy Exploration ---
        // Decide whether to explore (random move) or exploit (best predicted move)
        if (Math.random() < this.explorationFactor) {
            // console.log("AI Exploration: Choosing random valid move.");
            return this.getRandomValidMove(this.gridManager); // EXPLORE: Choose randomly
        } else {
            let output;
            if (this.networkType === 'ConvNetwork' || this.network.type === 'ConvNetwork') {
                const state = this._getCanonicalStateRepresentation();
                const res = this.network.run(state);
                output = res.policy;
                this.lastPredictedValue = res.value;
                console.log(`ConvNetwork Value Estimate (P${this.playerNumber}): ${res.value?.toFixed(4)}`);
            } else {
                const state = this._getStateRepresentation();
                output = this.network.run(state); // Get probabilities/scores for each move
            }

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
            return this.getRandomValidMove(this.gridManager);
        }
    }

    getRandomValidMove() {
        const possibleMoves = [];
        for (let y = 0; y < this.gridManager.rows; y++) {
            for (let x = 0; x < this.gridManager.cols; x++) {
                for (let rotation = 0; rotation < 4; rotation++) {
                    // Temporarily set selectedRotation to check validity
                    const originalRotation = this.gridManager.selectedRotation;
                    this.gridManager.selectedRotation = rotation;

                    const cell = this.gridManager.grid[y][x];
                     let isValid = false;
                     if (cell.triangles.length < 2) {
                          let hasOverlap = false;
                          if (cell.triangles.length > 0)  {
                             const newTriangleGroup = this.gridManager.getDiagonalGroup(rotation);
                             hasOverlap = !(this.gridManager.getDiagonalGroup(cell.triangles[0].rotation) === newTriangleGroup) ||
                             cell.triangles[0].rotation === rotation;
                          }
                          isValid = this.gridManager.isValidPlacement(x, y) && !hasOverlap;
                     }

                    // Restore original rotation
                    this.gridManager.selectedRotation = originalRotation;

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
            errorThresh: 0.00093 // Stop if error is low enough
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

        // // Re-estimate reward range including cluster bonus/penalty
        // const clusterRewardFactor = 0.05; // <<<< Must match the factor used in runSelfPlayGame
        // // Estimate max possible cluster difference (theoretically grid size, practically maybe less)
        // // Max possible cluster size is rows * cols * 2 triangles? Let's estimate based on grid area for simplicity.
        // const maxPossibleClusterDiff = this.gridManager.rows * this.gridManager.cols;
        // const maxClusterBonus = maxPossibleClusterDiff * clusterRewardFactor;
        // const minClusterPenalty = -maxPossibleClusterDiff * clusterRewardFactor; // Symmetric penalty

        // Re-estimate reward range including connected cluster bonus
        const connectedClusterRewardFactor = 0.1; // <<<< Must match the factor used in runSelfPlayGame
        // Max possible cluster size is roughly rows * cols * 2 (every slot filled)
        const maxPossibleClusterSize = this.gridManager.rows * this.gridManager.cols * 2;
        const maxConnectedClusterBonus = maxPossibleClusterSize * connectedClusterRewardFactor;


        // Estimate max claim bonus (e.g., 4 claims * 0.3 reward/claim = 1.2) - adjust if needed
        const maxClaimBonus = 1.2;

        const minReward = -1.0; // + minClusterPenalty; // Loss + max cluster penalty
        const maxReward = 1.0 + maxClaimBonus + maxConnectedClusterBonus; // + maxClusterBonus; // Win + max claim + max cluster bonus

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
        console.log("Starting self-play game with immediate shaping + discounted final outcome on path...");
        // const tempGridManager = new GridManager(this.gridManager.rows, this.gridManager.cols); // Use a temporary manager
        const gameHistory = []; // Store { state, move, player, claimed, connectedClusterSize } for reward assignment later

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

            // 1. Calculate IMMEDIATE rewards for all moves from this state
            // (Assumes calculateImmediateRewardVector returns rewards scaled approx 0-1)
            const immediateRewardVector = this.calculateImmediateRewardVector(state, currentPlayerNum, currentManager);

            // 2. Determine ACTUAL move to take to advance simulation state
            // (Using exploitation/exploration based on immediateRewardVector or random)
            let move;
            let actualMoveIndex = -1; // Initialize index
            if (this.isNetworkLoaded && movesCount>0) { // Avoid using network on first move (otherwise it will start at same plce)
                // Use prediction if network is loaded/trained
                // console.log(`Player ${currentPlayerAI.playerNumber} using predictMove (Network Ready: ${currentPlayerAI.isNetworkLoaded})`); // Optional detailed log
                if ( Math.random() >= this.immediacyFactor) {
                    move = currentPlayerAI.predictMove();
                } else {
                    // Exploit: Choose move with highest immediate reward
                    let bestMoveIndex = -1;
                    let maxImmediateReward = -Infinity;
                    for (let i = 0; i < immediateRewardVector.length; i++) {
                        // Check legality implicitly (reward > default low value, e.g., 0.01)
                        if (immediateRewardVector[i] > 0.01 && immediateRewardVector[i] > maxImmediateReward) {
                            maxImmediateReward = immediateRewardVector[i];
                            bestMoveIndex = i;
                        }
                    }
                    if (bestMoveIndex !== -1) {
                        const rotation = bestMoveIndex % 4;
                        const cellIndex = Math.floor(bestMoveIndex / 4);
                        const x = cellIndex % currentManager.cols; // Use currentManager dims
                        const y = Math.floor(cellIndex / currentManager.cols);
                        move = { x, y, rotation };
                    } else { move = currentPlayerAI.getRandomValidMove(); } // Fallback
                }
            } else {
                // Use random moves if network is not ready
                // console.log(`Player ${currentPlayerAI.playerNumber} using getRandomValidMove (Network Ready: ${currentPlayerAI.isNetworkLoaded})`); // Optional detailed log
                move = currentPlayerAI.getRandomValidMove();
            }
            // --- End conditional move choice ---

            if (!move) {
                console.error("Self-play: AI failed to find a move. Ending game.");
                break; // Should not happen
            }

            actualMoveIndex = (move.y * currentManager.cols + move.x) * 4 + move.rotation;

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

            // --- Calculate size of the cluster connected to the placed piece ---
            // Use the new GridManager method
            // const connectedClusterSize = currentManager.getConnectedClusterSize(
            //     move.x,
            //     move.y,
            //     move.rotation // Rotation of the placed piece
            //     // No player needed as input, cluster ID lookup is sufficient
            // );
            // --- End cluster size calculation ---

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
                immediateRewardVector: immediateRewardVector,
                actualMoveIndex: actualMoveIndex
                // connectedClusterSize: connectedClusterSize // Store size from lookup
                //clusterSize: currentPlayerClusterSize,          // << Player's cluster size AFTER move
                //opponentClusterSize: opponentPlayerClusterSize // << Opponent's cluster size AFTER move
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

        // Base final reward: +1 for win, -1 for loss, 0 for draw
        const finalOutcomeRewardValue = (winner === this.playerNumber) ? 1.0 : (winner === 0 ? 0.0 : -1.0);
        // Weighting factor for the final outcome reward (tune this!)
        const finalRewardWeight = 1.0; // Example: Give full weight to win/loss
        const totalMovesInGame = gameHistory.length;

        // --- Process Trajectory BACKWARDS to Create Final Training Data ---
        console.log(`Processing ${totalMovesInGame} steps for final training data...`);

        // Define scaling parameters for the COMBINED reward
        // Immediate rewards are already scaled 0-1 by calculateImmediateRewardVector
        const minImmediateReward = 0.01; // Min output from immediate calculation
        const maxImmediateReward = 0.99; // Max output from immediate calculation
        const minCombinedReward = minImmediateReward - (1.0 * finalRewardWeight); // Theoretical min (loss + min immediate)
        const maxCombinedReward = maxImmediateReward + (1.0 * finalRewardWeight); // Theoretical max (win + max immediate)
        const combinedRewardRange = maxCombinedReward - minCombinedReward;

        // Iterate backwards to potentially apply more complex discounting later if needed,
        // although simple linear scaling doesn't strictly require backward iteration.
        for (let t = totalMovesInGame - 1; t >= 0; t--) {
            const step = gameHistory[t];
            const state = step.state;
            const immediateRewardVector = step.immediateRewardVector;
            const actualMoveIndex = step.actualMoveIndex;

            // Create the final output vector, starting as a copy of immediate rewards
            const finalOutputVector = [...immediateRewardVector];

            // Calculate the discount factor - linear scaling based on step number
            // Moves closer to the end (higher t) get a larger portion of the final reward.
            const discountFactor = (t + 1) / totalMovesInGame;
            const scaledFinalReward = finalOutcomeRewardValue * discountFactor * finalRewardWeight;

            // Get the immediate reward component for the action actually taken
            const immediateRewardActualMove = immediateRewardVector[actualMoveIndex];

            // Calculate the combined reward ONLY for the action taken
            const combinedRewardActualMove = immediateRewardActualMove + scaledFinalReward;

            // Scale the combined reward for the taken action into the target range [0.01, 0.99]
            let scaledCombinedRewardActualMove = 0.01;
            if (combinedRewardRange !== 0) {
                scaledCombinedRewardActualMove = (combinedRewardActualMove - minCombinedReward) / combinedRewardRange;
            } else if (combinedRewardActualMove >= maxCombinedReward) {
                scaledCombinedRewardActualMove = 1.0;
            }
            scaledCombinedRewardActualMove = Math.max(0.01, Math.min(0.99, scaledCombinedRewardActualMove)); // Clamp

            // --- Update ONLY the target for the move that was actually taken ---
            finalOutputVector[actualMoveIndex] = scaledCombinedRewardActualMove;

            // Add the final input/output pair to training data
            // Note: Data for earlier states is added LAST here due to backward iteration
            this.trainingData.push({ input: state, output: finalOutputVector });

        } // End processing trajectory

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
     * Calculates a reward vector for all possible moves from a given state.
     * Reward is based on simulating immediate captures and resulting connected cluster size
     * using the OPTIMIZED cluster lookup from GridManager.
     * NOTE: Still computationally expensive due to simulating every legal move.
     * @param {Array<number>} stateRepresentation - The input state vector.
     * @param {number} player - The player whose turn it is.
     * @param {GridManager} currentManager - The *actual* GridManager instance for current state info.
     * @returns {Array<number>} - The output vector with scaled rewards for each possible move.
     */
    calculateImmediateRewardVector(stateRepresentation, player, currentManager) {
        const rows = currentManager.rows;
        const cols = currentManager.cols;
        const outputVector = new Array(rows * cols * 4).fill(0.01); // Default low value (illegal moves)

        // --- Define Reward Factors (tune these!) ---
        const claimRewardFactor = 0.3;
        const connectedClusterRewardFactor = 0.1; // Factor for size reward

        // --- Define Scaling Range (based only on these immediate rewards) ---
        // (Recalculate based on factors - same as previous step)
        const maxPossibleClaimBonus = 1.5; // Estimate
        const maxPossibleClusterSize = rows * cols * 2;
        const maxPossibleClusterBonus = maxPossibleClusterSize * connectedClusterRewardFactor;
        const minImmediateReward = 0;
        const maxImmediateReward = maxPossibleClaimBonus + maxPossibleClusterBonus;
        const rewardRange = maxImmediateReward - minImmediateReward;

        // --- Iterate through ALL possible moves ---
        for (let y = 0; y < rows; y++) {
            for (let x = 0; x < cols; x++) {
                for (let rotation = 0; rotation < 4; rotation++) {
                    const moveIndex = (y * cols + x) * 4 + rotation;

                    // --- 1. Check Legality (using currentManager's state) ---
                    const originalRotation = currentManager.selectedRotation;
                    currentManager.selectedRotation = rotation;
                    const cell = currentManager.grid[y]?.[x]; // Add safe navigation for grid access
                    let isLegal = false;
                    if (cell && cell.triangles.length < 2) {
                        let hasOverlap = false;
                        if (cell.triangles.length > 0) {
                            const newGroup = currentManager.getDiagonalGroup(rotation);
                            const existingGroup = currentManager.getDiagonalGroup(cell.triangles[0].rotation);
                            hasOverlap = !(existingGroup === newGroup) || cell.triangles[0].rotation === rotation;
                        }
                        isLegal = currentManager.isValidPlacement(x, y) && !hasOverlap;
                    }
                    currentManager.selectedRotation = originalRotation; // Restore

                    // --- 2. If Legal, Simulate on a Temporary Manager ---
                    if (isLegal) {
                        // Create a temporary manager instance FOR EACH legal move simulation
                        const tempManager = new GridManager(rows, cols);
                        try {
                            // Deep copy the necessary state from the *actual* manager
                            tempManager.grid = JSON.parse(JSON.stringify(currentManager.grid));
                            // Deep copy the cluster state (essential for the optimized methods)
                            tempManager.clusters = JSON.parse(JSON.stringify(currentManager.clusters));
                            tempManager.clusterMap = new Map(currentManager.clusterMap); // Shallow map copy is likely okay if cluster IDs don't change, but deep is safer if values change
                            tempManager.nextClusterId = currentManager.nextClusterId;

                        } catch (e) {
                            console.error("Failed to clone state for simulation", e);
                            continue; // Skip this move if cloning fails
                        }

                        tempManager.currentPlayer = player;
                        tempManager.selectedRotation = rotation;

                        // --- 3. Simulate the move using OPTIMIZED GridManager methods ---
                        // placeTriangle internally calls updateClustersForNewTriangle on tempManager
                        const simSuccess = tempManager.placeTriangle(x, y);

                        if (simSuccess) {
                            // claimEnclosedTriangles also calls updateClustersForNewTriangle on tempManager
                            const simClaimed = tempManager.claimEnclosedTriangles();

                            // --- 4. Get Cluster Size using the OPTIMIZED LOOKUP ---
                            // This reads the size calculated and stored by the *simulated*
                            // place/claim operations within tempManager's cluster data.
                            const simClusterSize = tempManager.getConnectedClusterSize(x, y, rotation);

                            // --- 5. Calculate and Scale Reward ---
                            const hypoClaimReward = simClaimed * claimRewardFactor;
                            const hypoClusterReward = simClusterSize * connectedClusterRewardFactor;
                            const hypotheticalReward = hypoClaimReward + hypoClusterReward;

                            let scaledReward = 0.01;
                            if (rewardRange !== 0) {
                                scaledReward = (hypotheticalReward - minImmediateReward) / rewardRange;
                            } else if (hypotheticalReward >= maxImmediateReward) {
                                scaledReward = 1.0;
                            }
                            scaledReward = Math.max(0.01, Math.min(0.99, scaledReward)); // Clamp

                            outputVector[moveIndex] = scaledReward;

                        } else {
                            // If the simulation placement failed (e.g., cloned state issue?)
                            outputVector[moveIndex] = 0.0;
                        }
                        // tempManager instance is discarded here
                    } // End if(isLegal)
                } // end rotation
            } // end x
        } // end y

        return outputVector;
    } // End calculateImmediateRewardVector

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
                this.loadNetworkJSON(jsonData); // Supports ConvNetwork, TFNetwork, and brain.js JSON
                console.log(`AI Network state (${jsonData.type || 'brain.js'}) loaded successfully from ${sourceUsed}.`);
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

// Global for browser
if (typeof window !== "undefined") {
  window.AIPlayer = AIPlayer;
}

export { AIPlayer };
