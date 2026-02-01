#pragma once
#include <variant>
#include <optional>

/**
 * Move.hpp
 * Represents a move in Quoridor
 */

// Move types
struct PawnMove {
    int row;
    int col;
    
    PawnMove() : row(0), col(0) {}
    PawnMove(int r, int c) : row(r), col(c) {}
};

struct HorizontalWall {
    int row;
    int col;
    
    HorizontalWall() : row(0), col(0) {}
    HorizontalWall(int r, int c) : row(r), col(c) {}
};

struct VerticalWall {
    int row;
    int col;
    
    VerticalWall() : row(0), col(0) {}
    VerticalWall(int r, int c) : row(r), col(c) {}
};

// A Move can be one of: PawnMove, HorizontalWall, or VerticalWall
using Move = std::variant<PawnMove, HorizontalWall, VerticalWall>;

// Helper functions for Move
inline bool isPawnMove(const Move& m) {
    return std::holds_alternative<PawnMove>(m);
}

inline bool isHorizontalWall(const Move& m) {
    return std::holds_alternative<HorizontalWall>(m);
}

inline bool isVerticalWall(const Move& m) {
    return std::holds_alternative<VerticalWall>(m);
}

inline const PawnMove& getPawnMove(const Move& m) {
    return std::get<PawnMove>(m);
}

inline const HorizontalWall& getHorizontalWall(const Move& m) {
    return std::get<HorizontalWall>(m);
}

inline const VerticalWall& getVerticalWall(const Move& m) {
    return std::get<VerticalWall>(m);
}
