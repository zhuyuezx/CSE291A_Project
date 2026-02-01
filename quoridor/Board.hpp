#pragma once
#include "Pawn.hpp"
#include <array>

/**
 * Board.hpp
 * Represents the Quoridor board state
 * 
 * Wall coordinate system:
 * - Horizontal walls: 8x8 grid, wall at (r,c) blocks between rows r and r+1, columns c and c+1
 * - Vertical walls: 8x8 grid, wall at (r,c) blocks between columns c and c+1, rows r and r+1
 */
class Board {
public:
    std::array<Pawn, 2> pawns;
    
    // Wall arrays: true = wall present
    std::array<std::array<bool, 8>, 8> horizontalWalls;  // [row][col]
    std::array<std::array<bool, 8>, 8> verticalWalls;    // [row][col]
    
    Board() {
        // Initialize walls to false
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                horizontalWalls[i][j] = false;
                verticalWalls[i][j] = false;
            }
        }
    }
    
    Board(bool player0AtBottom) : Board() {
        // Player 0 always moves first
        // If player0AtBottom, they start at row 8, goal row 0
        pawns[0] = Pawn(0, player0AtBottom);
        pawns[1] = Pawn(1, !player0AtBottom);
    }
    
    // Clone the board
    Board clone() const {
        Board b;
        b.pawns[0] = pawns[0].clone();
        b.pawns[1] = pawns[1].clone();
        b.horizontalWalls = horizontalWalls;
        b.verticalWalls = verticalWalls;
        return b;
    }
};
