# Triantica Quick Reference

## Game Rules Summary

### Setup
- 10×10 grid (100 cells)
- 2 players (or 1 player vs AI)
- Each cell can hold 2 triangles (one per diagonal)

### Turn Sequence
1. Rotate your triangle (optional)
2. Place on valid empty slot
3. Claim any enclosed triangles
4. If you claimed, take another turn

### Placement Rules
| Rule | Details |
|------|---------|
| First move | Anywhere on empty board |
| Subsequent | Must touch corner with existing piece |
| No overlap | Can't place same diagonal orientation |
| Wrap-around | Board edges connect (torus topology) |

### Enclosure
A triangle slot is **enclosed** when surrounded on all three sides:
- Example: If cell has a ╱ triangle, the ╲ slot is enclosed when:
  - Top neighbor has ╲ or ╱ (blocking top edge)
  - Left neighbor has ╲ or ╱ (blocking left edge)  
  - The ╱ in same cell (blocking diagonal)

### Winning
Game ends when grid is full (200 triangles).  
**Winner**: Player with largest contiguous cluster.

---

## Controls

| Input | Action |
|-------|--------|
| Click on empty slot | Place triangle |
| Mouse wheel / R key | Rotate preview |
| Space | Rotate preview |
| Game mode buttons | Switch modes (HvH, HvAI, AvA) |

---

## Triangle Orientations

```
Rotation 0 (0°):    Rotation 1 (90°):
  ┌───┐               ┌───┐
  │  /│               │\  │
  │ / │               │ \ │
  │/  │               │  \│
  └───┘               └───┘

Rotation 2 (270°):  Rotation 3 (180°):
  ┌───┐               ┌───┐
  │  \│               │/  │
  │ \ │               │ \ │
  │\  │               │  /│
  └───┘               └───┘
```

---

## Cluster Definition

Two triangles are **connected** if they:
1. Belong to the same player
2. Share a full edge (not just a corner)

Cluster size = count of connected triangles.

---

## AI Difficulty Levels

| Level | Network Size | Behavior |
|-------|--------------|----------|
| Easy | 401→100→400 | Simpler patterns, more random |
| Medium | 401→200→400 | Balanced play |
| Hard | 401→267→160→400 | Deeper strategy |

---

## Scoring Examples

### Enclosure Bonus
```
Before:          After:
  A   B            A   B
 ┌───┬───┐        ┌───┬───┐
 │\  │  /│        │\A │A/│  ← Player A claims
 │ \ │ / │        │ \ │ / │
 │  \│/  │        │A \│/A │
 └───┴───┘        └───┴───┘
```

### Cluster Growth
```
Connected cluster of 7:
  ┌───┬───┬───┐
  │\A │A/ │\A │
  │ \ │ / │ \ │
  │A \│/A │A \│
  └───┴───┴───┘
  ┌───┬───┐
  │\A │A/ │
  │ \ │ / │
  │A \│/A │
  └───┴───┘
```
