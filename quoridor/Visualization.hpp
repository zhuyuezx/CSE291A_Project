#pragma once
#include "Game.hpp"
#include "Move.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

/**
 * Visualization.hpp
 * Terminal visualization for Quoridor
 */
class Visualization {
public:
    // ANSI color codes
    static constexpr const char* RESET = "\033[0m";
    static constexpr const char* BOLD = "\033[1m";
    static constexpr const char* RED = "\033[31m";
    static constexpr const char* GREEN = "\033[32m";
    static constexpr const char* YELLOW = "\033[33m";
    static constexpr const char* BLUE = "\033[34m";
    static constexpr const char* MAGENTA = "\033[35m";
    static constexpr const char* CYAN = "\033[36m";
    static constexpr const char* WHITE = "\033[37m";
    static constexpr const char* BG_WHITE = "\033[47m";
    static constexpr const char* BG_GRAY = "\033[100m";
    
    // Display the board state
    static void printBoard(const Game& game, bool useColors = true) {
        std::cout << "\n";
        printHeader(game, useColors);
        
        // Column labels
        std::cout << "     ";
        for (int c = 0; c < 9; c++) {
            std::cout << " " << c << "  ";
        }
        std::cout << "\n";
        
        // Top border
        std::cout << "   ";
        if (useColors) std::cout << CYAN;
        std::cout << "+";
        for (int c = 0; c < 9; c++) {
            std::cout << "---+";
        }
        if (useColors) std::cout << RESET;
        std::cout << "\n";
        
        for (int r = 0; r < 9; r++) {
            // Cell row
            std::cout << " " << r << " ";
            if (useColors) std::cout << CYAN;
            std::cout << "|";
            if (useColors) std::cout << RESET;
            
            for (int c = 0; c < 9; c++) {
                // Cell content
                std::string cell = getCellContent(game, r, c, useColors);
                std::cout << cell;
                
                // Right wall or separator
                if (c < 8) {
                    if (hasVerticalWall(game, r, c)) {
                        if (useColors) std::cout << YELLOW << BOLD;
                        std::cout << "#";
                        if (useColors) std::cout << RESET;
                    } else {
                        if (useColors) std::cout << CYAN;
                        std::cout << "|";
                        if (useColors) std::cout << RESET;
                    }
                } else {
                    if (useColors) std::cout << CYAN;
                    std::cout << "|";
                    if (useColors) std::cout << RESET;
                }
            }
            std::cout << "\n";
            
            // Horizontal separator row
            std::cout << "   ";
            if (useColors) std::cout << CYAN;
            std::cout << "+";
            if (useColors) std::cout << RESET;
            
            for (int c = 0; c < 9; c++) {
                if (r < 8 && hasHorizontalWall(game, r, c)) {
                    if (useColors) std::cout << YELLOW << BOLD;
                    std::cout << "===";
                    if (useColors) std::cout << RESET;
                } else {
                    if (useColors) std::cout << CYAN;
                    std::cout << "---";
                    if (useColors) std::cout << RESET;
                }
                
                // Corner/intersection
                if (c < 8) {
                    char corner = getCornerChar(game, r, c);
                    if (corner == '#') {
                        if (useColors) std::cout << YELLOW << BOLD;
                        std::cout << corner;
                        if (useColors) std::cout << RESET;
                    } else {
                        if (useColors) std::cout << CYAN;
                        std::cout << corner;
                        if (useColors) std::cout << RESET;
                    }
                } else {
                    if (useColors) std::cout << CYAN;
                    std::cout << "+";
                    if (useColors) std::cout << RESET;
                }
            }
            std::cout << "\n";
        }
        
        printFooter(game, useColors);
        std::cout << "\n";
    }
    
    // Print move description
    static std::string getMoveDescription(const Move& move) {
        std::stringstream ss;
        if (isPawnMove(move)) {
            const auto& pm = getPawnMove(move);
            ss << "Pawn to (" << pm.row << ", " << pm.col << ")";
        } else if (isHorizontalWall(move)) {
            const auto& hw = getHorizontalWall(move);
            ss << "Horizontal wall at (" << hw.row << ", " << hw.col << ")";
        } else if (isVerticalWall(move)) {
            const auto& vw = getVerticalWall(move);
            ss << "Vertical wall at (" << vw.row << ", " << vw.col << ")";
        }
        return ss.str();
    }
    
    // Print game status
    static void printStatus(const Game& game, bool useColors = true) {
        if (useColors) std::cout << BOLD;
        std::cout << "Turn " << game.turn << " - ";
        if (game.winner != -1) {
            if (useColors) std::cout << GREEN;
            std::cout << "Player " << game.winner << " WINS!";
        } else {
            std::cout << "Player " << game.getPawnIndexOfTurn() << "'s turn";
        }
        if (useColors) std::cout << RESET;
        std::cout << "\n";
    }
    
    // Print move with colors
    static void printMove(int player, const Move& move, double winRate = -1, bool useColors = true) {
        if (useColors) {
            std::cout << (player == 0 ? RED : BLUE) << BOLD;
        }
        std::cout << "Player " << player << ": " << getMoveDescription(move);
        if (winRate >= 0) {
            std::cout << " (Win rate: " << std::fixed << std::setprecision(1) << (winRate * 100) << "%)";
        }
        if (useColors) std::cout << RESET;
        std::cout << "\n";
    }
    
    // Clear screen
    static void clearScreen() {
        std::cout << "\033[2J\033[1;1H";
    }
    
private:
    static void printHeader(const Game& game, bool useColors) {
        if (useColors) std::cout << BOLD << CYAN;
        std::cout << "╔═══════════════════════════════════════════╗\n";
        std::cout << "║            Q U O R I D O R                ║\n";
        std::cout << "╚═══════════════════════════════════════════╝\n";
        if (useColors) std::cout << RESET;
    }
    
    static void printFooter(const Game& game, bool useColors) {
        // Player info
        std::cout << "\n";
        
        // Player 0
        if (useColors) std::cout << RED << BOLD;
        std::cout << "  ● Player 0";
        if (useColors) std::cout << RESET;
        std::cout << " - Walls: " << game.board.pawns[0].wallsLeft;
        std::cout << " | Goal: Row " << game.board.pawns[0].goalRow;
        if (game.getPawnIndexOfTurn() == 0) {
            if (useColors) std::cout << GREEN;
            std::cout << " ◄ TURN";
            if (useColors) std::cout << RESET;
        }
        std::cout << "\n";
        
        // Player 1
        if (useColors) std::cout << BLUE << BOLD;
        std::cout << "  ◆ Player 1";
        if (useColors) std::cout << RESET;
        std::cout << " - Walls: " << game.board.pawns[1].wallsLeft;
        std::cout << " | Goal: Row " << game.board.pawns[1].goalRow;
        if (game.getPawnIndexOfTurn() == 1) {
            if (useColors) std::cout << GREEN;
            std::cout << " ◄ TURN";
            if (useColors) std::cout << RESET;
        }
        std::cout << "\n";
        
        // Legend
        std::cout << "\n";
        if (useColors) std::cout << YELLOW << BOLD;
        std::cout << "  === : Horizontal Wall    # : Vertical Wall";
        if (useColors) std::cout << RESET;
        std::cout << "\n";
    }
    
    static std::string getCellContent(const Game& game, int r, int c, bool useColors) {
        std::stringstream ss;
        
        // Check if pawn is here
        if (game.board.pawns[0].position.row == r && game.board.pawns[0].position.col == c) {
            if (useColors) ss << RED << BOLD;
            ss << " ● ";
            if (useColors) ss << RESET;
        } else if (game.board.pawns[1].position.row == r && game.board.pawns[1].position.col == c) {
            if (useColors) ss << BLUE << BOLD;
            ss << " ◆ ";
            if (useColors) ss << RESET;
        } else {
            // Empty cell - show goal rows differently
            if (r == game.board.pawns[0].goalRow) {
                if (useColors) ss << BG_GRAY;
                ss << " · ";
                if (useColors) ss << RESET;
            } else if (r == game.board.pawns[1].goalRow) {
                if (useColors) ss << BG_GRAY;
                ss << " · ";
                if (useColors) ss << RESET;
            } else {
                ss << "   ";
            }
        }
        
        return ss.str();
    }
    
    static bool hasHorizontalWall(const Game& game, int r, int c) {
        // Check if there's a horizontal wall segment below (r, c)
        // Walls are stored by their left corner position
        if (r >= 8) return false;
        
        // Check if wall at (r, c) covers this segment
        if (c < 8 && game.board.horizontalWalls[r][c]) return true;
        // Check if wall at (r, c-1) covers this segment
        if (c > 0 && game.board.horizontalWalls[r][c-1]) return true;
        
        return false;
    }
    
    static bool hasVerticalWall(const Game& game, int r, int c) {
        // Check if there's a vertical wall segment to the right of (r, c)
        // Walls are stored by their top corner position
        if (c >= 8) return false;
        
        // Check if wall at (r, c) covers this segment
        if (r < 8 && game.board.verticalWalls[r][c]) return true;
        // Check if wall at (r-1, c) covers this segment
        if (r > 0 && game.board.verticalWalls[r-1][c]) return true;
        
        return false;
    }
    
    static char getCornerChar(const Game& game, int r, int c) {
        // Check if any wall touches this corner
        bool hasWall = false;
        
        // Horizontal walls
        if (r < 8 && c < 8 && game.board.horizontalWalls[r][c]) hasWall = true;
        if (r < 8 && c > 0 && game.board.horizontalWalls[r][c-1]) hasWall = true;
        
        // Vertical walls
        if (r < 8 && c < 8 && game.board.verticalWalls[r][c]) hasWall = true;
        if (r > 0 && c < 8 && game.board.verticalWalls[r-1][c]) hasWall = true;
        
        return hasWall ? '#' : '+';
    }
};
