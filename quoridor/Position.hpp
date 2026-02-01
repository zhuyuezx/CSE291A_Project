#pragma once

/**
 * Position.hpp
 * Represents a position on the Quoridor board (row, col)
 */
class Position {
public:
    int row;
    int col;
    
    Position() : row(0), col(0) {}
    Position(int r, int c) : row(r), col(c) {}
    
    bool operator==(const Position& other) const {
        return row == other.row && col == other.col;
    }
    
    bool operator!=(const Position& other) const {
        return !(*this == other);
    }
    
    // Add a move tuple (dr, dc) to this position
    Position addMove(int dr, int dc) const {
        return Position(row + dr, col + dc);
    }
    
    // Get displacement from another position to this one
    void getDisplacementFrom(const Position& from, int& dr, int& dc) const {
        dr = row - from.row;
        dc = col - from.col;
    }
    
    bool isValid() const {
        return row >= 0 && row <= 8 && col >= 0 && col <= 8;
    }
};

// Move directions as constants
namespace Direction {
    constexpr int UP[2] = {-1, 0};
    constexpr int DOWN[2] = {1, 0};
    constexpr int LEFT[2] = {0, -1};
    constexpr int RIGHT[2] = {0, 1};
}
