#pragma once
#include "Position.hpp"

/**
 * Pawn.hpp
 * Represents a pawn in Quoridor
 */
class Pawn {
public:
    int index;           // 0 = first player (moves first), 1 = second player
    Position position;
    int goalRow;         // The row the pawn needs to reach to win
    int wallsLeft;       // Number of walls remaining (starts at 10)
    
    Pawn() : index(0), goalRow(0), wallsLeft(10) {}
    
    Pawn(int idx, bool startsAtBottom) : index(idx), wallsLeft(10) {
        if (startsAtBottom) {
            // Starts at row 8 (bottom), goal is row 0 (top)
            position = Position(8, 4);
            goalRow = 0;
        } else {
            // Starts at row 0 (top), goal is row 8 (bottom)
            position = Position(0, 4);
            goalRow = 8;
        }
    }
    
    bool hasReachedGoal() const {
        return position.row == goalRow;
    }
    
    bool hasWalls() const {
        return wallsLeft > 0;
    }
    
    // Clone this pawn
    Pawn clone() const {
        Pawn p;
        p.index = index;
        p.position = position;
        p.goalRow = goalRow;
        p.wallsLeft = wallsLeft;
        return p;
    }
};
