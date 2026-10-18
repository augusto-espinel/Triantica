/**
 * MetricsLogger - Performance tracking for AI training experiments
 * 
 * Usage:
 *   const metrics = new MetricsLogger();
 *   metrics.recordGame(winner, moveCount, claims);
 *   metrics.logTraining(iteration, error);
 *   metrics.getSummary();
 */

class MetricsLogger {
    constructor(experimentId = 'baseline') {
        this.experimentId = experimentId;
        this.startTime = Date.now();
        
        // Game metrics
        this.gamesPlayed = 0;
        this.wins = { 1: 0, 2: 0, draws: 0 };
        this.totalMoves = 0;
        this.totalClaims = 0;
        this.decisionTimes = [];
        
        // Training metrics
        this.trainingSessions = 0;
        this.trainingIterations = 0;
        this.trainingErrors = [];
        this.samplesProcessed = 0;
        
        // Per-game tracking
        this.currentGame = {
            moves: 0,
            claims: 0,
            times: []
        };
    }
    
    // === Game Tracking ===
    
    startGame() {
        this.currentGame = { moves: 0, claims: 0, times: [] };
    }
    
    recordMove(decisionTimeMs) {
        this.currentGame.moves++;
        this.currentGame.times.push(decisionTimeMs);
    }
    
    recordClaims(count) {
        this.currentGame.claims += count;
    }
    
    endGame(winner) {
        this.gamesPlayed++;
        this.wins[winner === 0 ? 'draws' : winner]++;
        this.totalMoves += this.currentGame.moves;
        this.totalClaims += this.currentGame.claims;
        this.decisionTimes.push(...this.currentGame.times);
    }
    
    // === Training Tracking ===
    
    startTraining(samples) {
        this.trainingSessions++;
        this.samplesProcessed += samples;
    }
    
    logIteration(iteration, error) {
        this.trainingIterations++;
        this.trainingErrors.push({ iteration, error, time: Date.now() });
    }
    
    endTraining(finalError, iterations) {
        console.log(`[Metrics] Training complete: ${iterations} iterations, final error: ${finalError.toFixed(6)}`);
    }
    
    // === Reporting ===
    
    getSummary() {
        const elapsed = (Date.now() - this.startTime) / 1000;
        const avgDecisionTime = this.decisionTimes.length > 0
            ? this.decisionTimes.reduce((a, b) => a + b, 0) / this.decisionTimes.length
            : 0;
        
        return {
            experimentId: this.experimentId,
            elapsedSeconds: elapsed,
            games: {
                played: this.gamesPlayed,
                wins: this.wins,
                winRate1: this.gamesPlayed > 0 
                    ? (this.wins[1] / this.gamesPlayed * 100).toFixed(1) + '%'
                    : 'N/A',
                winRate2: this.gamesPlayed > 0
                    ? (this.wins[2] / this.gamesPlayed * 100).toFixed(1) + '%'
                    : 'N/A',
                drawRate: this.gamesPlayed > 0
                    ? (this.wins.draws / this.gamesPlayed * 100).toFixed(1) + '%'
                    : 'N/A'
            },
            moves: {
                total: this.totalMoves,
                avgPerGame: this.gamesPlayed > 0
                    ? (this.totalMoves / this.gamesPlayed).toFixed(1)
                    : 'N/A'
            },
            claims: {
                total: this.totalClaims,
                avgPerGame: this.gamesPlayed > 0
                    ? (this.totalClaims / this.gamesPlayed).toFixed(1)
                    : 'N/A'
            },
            performance: {
                avgDecisionTimeMs: avgDecisionTime.toFixed(2),
                maxDecisionTimeMs: Math.max(...this.decisionTimes, 0).toFixed(2),
                minDecisionTimeMs: this.decisionTimes.length > 0
                    ? Math.min(...this.decisionTimes).toFixed(2)
                    : 'N/A'
            },
            training: {
                sessions: this.trainingSessions,
                totalIterations: this.trainingIterations,
                samplesProcessed: this.samplesProcessed,
                finalError: this.trainingErrors.length > 0
                    ? this.trainingErrors[this.trainingErrors.length - 1].error.toFixed(6)
                    : 'N/A'
            }
        };
    }
    
    printSummary() {
        const summary = this.getSummary();
        console.log('\n' + '='.repeat(50));
        console.log(`EXPERIMENT: ${summary.experimentId}`);
        console.log('='.repeat(50));
        console.log(`Elapsed: ${summary.elapsedSeconds}s`);
        console.log('\n📊 Games:');
        console.log(`  Played: ${summary.games.played}`);
        console.log(`  P1 Wins: ${summary.games.winRate1}`);
        console.log(`  P2 Wins: ${summary.games.winRate2}`);
        console.log(`  Draws: ${summary.games.drawRate}`);
        console.log('\n🎯 Moves:');
        console.log(`  Total: ${summary.moves.total}`);
        console.log(`  Avg/Game: ${summary.moves.avgPerGame}`);
        console.log('\n📦 Claims:');
        console.log(`  Total: ${summary.claims.total}`);
        console.log(`  Avg/Game: ${summary.claims.avgPerGame}`);
        console.log('\n⚡ Performance:');
        console.log(`  Avg Decision: ${summary.performance.avgDecisionTimeMs}ms`);
        console.log(`  Max Decision: ${summary.performance.maxDecisionTimeMs}ms`);
        console.log('\n🧠 Training:');
        console.log(`  Sessions: ${summary.training.sessions}`);
        console.log(`  Iterations: ${summary.training.totalIterations}`);
        console.log(`  Final Error: ${summary.training.finalError}`);
        console.log('='.repeat(50) + '\n');
    }
    
    exportJSON() {
        return JSON.stringify(this.getSummary(), null, 2);
    }
    
    downloadLog() {
        const data = this.exportJSON();
        const blob = new Blob([data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `metrics_${this.experimentId}_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }
}

// For use in Node.js (if needed)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MetricsLogger;
}
