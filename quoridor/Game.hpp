#pragma once
#include "Board.hpp"
#include "Move.hpp"
#include <array>
#include <vector>
#include <queue>
#include <functional>

/**
 * Game.hpp
 * Represents the Quoridor game state and rules
 */
class Game {
public:
    Board board;
    int winner;  // -1 = no winner, 0 or 1 = winning player index
    int turn;    // Current turn number (0-indexed)
    
    // Valid wall placement arrays (true = valid location)
    std::array<std::array<bool, 8>, 8> validHorizontalWalls;
    std::array<std::array<bool, 8>, 8> validVerticalWalls;
    
    // Open ways: whether paths are open between adjacent positions
    // upDown[r][c] = true means path is open between (r,c) and (r+1,c)
    std::array<std::array<bool, 9>, 8> openWaysUpDown;     // 8 rows, 9 cols
    // leftRight[r][c] = true means path is open between (r,c) and (r,c+1)
    std::array<std::array<bool, 8>, 9> openWaysLeftRight;  // 9 rows, 8 cols
    
    // Probable next walls (for heuristics)
    std::array<std::array<bool, 8>, 8> probableHorizontalWalls;
    std::array<std::array<bool, 8>, 8> probableVerticalWalls;
    
    // Valid next positions for current pawn
    std::array<std::array<bool, 9>, 9> validNextPositions;
    bool validNextPositionsUpdated;
    
    Game() : winner(-1), turn(0), validNextPositionsUpdated(false) {
        initArrays();
    }
    
    Game(bool player0AtBottom) : board(player0AtBottom), winner(-1), turn(0), validNextPositionsUpdated(false) {
        initArrays();
    }
    
    // Accessors
    int getPawnIndexOfTurn() const { return turn % 2; }
    int getPawnIndexOfNotTurn() const { return (turn + 1) % 2; }
    Pawn& getPawnOfTurn() { return board.pawns[getPawnIndexOfTurn()]; }
    Pawn& getPawnOfNotTurn() { return board.pawns[getPawnIndexOfNotTurn()]; }
    const Pawn& getPawnOfTurn() const { return board.pawns[getPawnIndexOfTurn()]; }
    const Pawn& getPawnOfNotTurn() const { return board.pawns[getPawnIndexOfNotTurn()]; }
    Pawn& getPawn0() { return board.pawns[0]; }
    Pawn& getPawn1() { return board.pawns[1]; }
    const Pawn& getPawn0() const { return board.pawns[0]; }
    const Pawn& getPawn1() const { return board.pawns[1]; }
    
    // Check if path is open in a direction from position (row, col)
    bool isOpenWay(int row, int col, int dr, int dc) const {
        if (dr == -1 && dc == 0) { // UP
            return row > 0 && openWaysUpDown[row - 1][col];
        } else if (dr == 1 && dc == 0) { // DOWN
            return row < 8 && openWaysUpDown[row][col];
        } else if (dr == 0 && dc == -1) { // LEFT
            return col > 0 && openWaysLeftRight[row][col - 1];
        } else if (dr == 0 && dc == 1) { // RIGHT
            return col < 8 && openWaysLeftRight[row][col];
        }
        return false;
    }
    
    // Check if move is valid not considering other pawn
    bool isValidMoveNotConsideringOtherPawn(const Position& pos, int dr, int dc) const {
        return isOpenWay(pos.row, pos.col, dr, dc);
    }
    
    // Get valid next positions for the pawn of turn
    const std::array<std::array<bool, 9>, 9>& getValidNextPositions() {
        if (validNextPositionsUpdated) {
            return validNextPositions;
        }
        validNextPositionsUpdated = true;
        
        // Reset
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                validNextPositions[i][j] = false;
            }
        }
        
        // Check all four directions
        setValidNextPositionsToward(-1, 0, 0, -1, 0, 1); // UP, LEFT, RIGHT
        setValidNextPositionsToward(1, 0, 0, -1, 0, 1);  // DOWN, LEFT, RIGHT
        setValidNextPositionsToward(0, -1, -1, 0, 1, 0); // LEFT, UP, DOWN
        setValidNextPositionsToward(0, 1, -1, 0, 1, 0);  // RIGHT, UP, DOWN
        
        return validNextPositions;
    }
    
    // Get array of valid next position coordinates
    std::vector<std::pair<int, int>> getArrOfValidNextPositionTuples() {
        const auto& positions = getValidNextPositions();
        std::vector<std::pair<int, int>> result;
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (positions[i][j]) {
                    result.push_back({i, j});
                }
            }
        }
        return result;
    }
    
    // Move pawn to position
    bool movePawn(int row, int col, bool needCheck = false) {
        if (needCheck && !getValidNextPositions()[row][col]) {
            return false;
        }
        getPawnOfTurn().position.row = row;
        getPawnOfTurn().position.col = col;
        if (getPawnOfTurn().hasReachedGoal()) {
            winner = getPawnIndexOfTurn();
        }
        turn++;
        validNextPositionsUpdated = false;
        return true;
    }
    
    // Place horizontal wall
    bool placeHorizontalWall(int row, int col, bool needCheck = false) {
        if (needCheck && !testIfExistPathsToGoalLinesAfterPlaceHorizontalWall(row, col)) {
            return false;
        }
        
        // Block paths
        openWaysUpDown[row][col] = false;
        openWaysUpDown[row][col + 1] = false;
        
        // Mark invalid wall positions
        validVerticalWalls[row][col] = false;
        validHorizontalWalls[row][col] = false;
        if (col > 0) validHorizontalWalls[row][col - 1] = false;
        if (col < 7) validHorizontalWalls[row][col + 1] = false;
        
        board.horizontalWalls[row][col] = true;
        adjustProbableWallsAfterHorizontal(row, col);
        getPawnOfTurn().wallsLeft--;
        turn++;
        validNextPositionsUpdated = false;
        return true;
    }
    
    // Place vertical wall
    bool placeVerticalWall(int row, int col, bool needCheck = false) {
        if (needCheck && !testIfExistPathsToGoalLinesAfterPlaceVerticalWall(row, col)) {
            return false;
        }
        
        // Block paths
        openWaysLeftRight[row][col] = false;
        openWaysLeftRight[row + 1][col] = false;
        
        // Mark invalid wall positions
        validHorizontalWalls[row][col] = false;
        validVerticalWalls[row][col] = false;
        if (row > 0) validVerticalWalls[row - 1][col] = false;
        if (row < 7) validVerticalWalls[row + 1][col] = false;
        
        board.verticalWalls[row][col] = true;
        adjustProbableWallsAfterVertical(row, col);
        getPawnOfTurn().wallsLeft--;
        turn++;
        validNextPositionsUpdated = false;
        return true;
    }
    
    // Execute a move
    bool doMove(const Move& move, bool needCheck = false) {
        if (winner != -1) return false;
        
        if (isPawnMove(move)) {
            const auto& pm = getPawnMove(move);
            return movePawn(pm.row, pm.col, needCheck);
        } else if (isHorizontalWall(move)) {
            const auto& hw = getHorizontalWall(move);
            return placeHorizontalWall(hw.row, hw.col, needCheck);
        } else if (isVerticalWall(move)) {
            const auto& vw = getVerticalWall(move);
            return placeVerticalWall(vw.row, vw.col, needCheck);
        }
        return false;
    }
    
    // Test if paths exist after placing horizontal wall
    bool testIfExistPathsToGoalLinesAfterPlaceHorizontalWall(int row, int col) {
        if (!validHorizontalWalls[row][col]) return false;
        if (!testIfConnectedOnTwoPointsForHorizontalWall(row, col)) return true;
        
        // Temporarily block paths
        openWaysUpDown[row][col] = false;
        openWaysUpDown[row][col + 1] = false;
        bool result = existPathsToGoalLines();
        openWaysUpDown[row][col] = true;
        openWaysUpDown[row][col + 1] = true;
        return result;
    }
    
    // Test if paths exist after placing vertical wall
    bool testIfExistPathsToGoalLinesAfterPlaceVerticalWall(int row, int col) {
        if (!validVerticalWalls[row][col]) return false;
        if (!testIfConnectedOnTwoPointsForVerticalWall(row, col)) return true;
        
        // Temporarily block paths
        openWaysLeftRight[row][col] = false;
        openWaysLeftRight[row + 1][col] = false;
        bool result = existPathsToGoalLines();
        openWaysLeftRight[row][col] = true;
        openWaysLeftRight[row + 1][col] = true;
        return result;
    }
    
    // Get valid wall positions that don't block all paths
    std::vector<std::pair<int, int>> getArrOfValidNoBlockHorizontalWalls() {
        std::vector<std::pair<int, int>> result;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (validHorizontalWalls[i][j] && testIfExistPathsToGoalLinesAfterPlaceHorizontalWall(i, j)) {
                    result.push_back({i, j});
                }
            }
        }
        return result;
    }
    
    std::vector<std::pair<int, int>> getArrOfValidNoBlockVerticalWalls() {
        std::vector<std::pair<int, int>> result;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (validVerticalWalls[i][j] && testIfExistPathsToGoalLinesAfterPlaceVerticalWall(i, j)) {
                    result.push_back({i, j});
                }
            }
        }
        return result;
    }
    
    // Get probable valid walls (for MCTS expansion heuristics)
    std::vector<std::pair<int, int>> getArrOfProbableValidNoBlockHorizontalWalls() {
        updateProbableWalls();
        std::vector<std::pair<int, int>> result;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (probableHorizontalWalls[i][j] && validHorizontalWalls[i][j] &&
                    testIfExistPathsToGoalLinesAfterPlaceHorizontalWall(i, j)) {
                    result.push_back({i, j});
                }
            }
        }
        return result;
    }
    
    std::vector<std::pair<int, int>> getArrOfProbableValidNoBlockVerticalWalls() {
        updateProbableWalls();
        std::vector<std::pair<int, int>> result;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (probableVerticalWalls[i][j] && validVerticalWalls[i][j] &&
                    testIfExistPathsToGoalLinesAfterPlaceVerticalWall(i, j)) {
                    result.push_back({i, j});
                }
            }
        }
        return result;
    }
    
    // Clone the game state
    Game clone() const {
        Game g;
        g.board = board.clone();
        g.winner = winner;
        g.turn = turn;
        g.validHorizontalWalls = validHorizontalWalls;
        g.validVerticalWalls = validVerticalWalls;
        g.openWaysUpDown = openWaysUpDown;
        g.openWaysLeftRight = openWaysLeftRight;
        g.probableHorizontalWalls = probableHorizontalWalls;
        g.probableVerticalWalls = probableVerticalWalls;
        g.validNextPositions = validNextPositions;
        g.validNextPositionsUpdated = validNextPositionsUpdated;
        return g;
    }
    
private:
    void initArrays() {
        // Initialize valid wall positions
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                validHorizontalWalls[i][j] = true;
                validVerticalWalls[i][j] = true;
                probableHorizontalWalls[i][j] = false;
                probableVerticalWalls[i][j] = false;
            }
        }
        
        // Initialize open ways
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 9; j++) {
                openWaysUpDown[i][j] = true;
            }
        }
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 8; j++) {
                openWaysLeftRight[i][j] = true;
            }
        }
        
        // Initialize valid next positions
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                validNextPositions[i][j] = false;
            }
        }
    }
    
    void setValidNextPositionsToward(int mainDr, int mainDc, int sub1Dr, int sub1Dc, int sub2Dr, int sub2Dc) {
        const Position& pos = getPawnOfTurn().position;
        const Position& otherPos = getPawnOfNotTurn().position;
        
        if (!isValidMoveNotConsideringOtherPawn(pos, mainDr, mainDc)) return;
        
        Position mainMovePos = pos.addMove(mainDr, mainDc);
        
        // If other pawn is there, check for jumping
        if (mainMovePos == otherPos) {
            // Try jumping straight over
            if (isValidMoveNotConsideringOtherPawn(mainMovePos, mainDr, mainDc)) {
                Position jumpPos = mainMovePos.addMove(mainDr, mainDc);
                validNextPositions[jumpPos.row][jumpPos.col] = true;
            } else {
                // Try diagonal jumps
                if (isValidMoveNotConsideringOtherPawn(mainMovePos, sub1Dr, sub1Dc)) {
                    Position diagPos = mainMovePos.addMove(sub1Dr, sub1Dc);
                    validNextPositions[diagPos.row][diagPos.col] = true;
                }
                if (isValidMoveNotConsideringOtherPawn(mainMovePos, sub2Dr, sub2Dc)) {
                    Position diagPos = mainMovePos.addMove(sub2Dr, sub2Dc);
                    validNextPositions[diagPos.row][diagPos.col] = true;
                }
            }
        } else {
            validNextPositions[mainMovePos.row][mainMovePos.col] = true;
        }
    }
    
    bool testIfConnectedOnTwoPointsForHorizontalWall(int row, int col) {
        bool left = (col == 0) || testIfAdjacentToOtherWallForHorizontalLeft(row, col);
        bool right = (col == 7) || testIfAdjacentToOtherWallForHorizontalRight(row, col);
        bool middle = testIfAdjacentToOtherWallForHorizontalMiddle(row, col);
        return (left && right) || (right && middle) || (middle && left);
    }
    
    bool testIfConnectedOnTwoPointsForVerticalWall(int row, int col) {
        bool top = (row == 0) || testIfAdjacentToOtherWallForVerticalTop(row, col);
        bool bottom = (row == 7) || testIfAdjacentToOtherWallForVerticalBottom(row, col);
        bool middle = testIfAdjacentToOtherWallForVerticalMiddle(row, col);
        return (top && bottom) || (bottom && middle) || (middle && top);
    }
    
    bool testIfAdjacentToOtherWallForHorizontalLeft(int row, int col) {
        if (col >= 1) {
            if (board.verticalWalls[row][col-1]) return true;
            if (row >= 1 && board.verticalWalls[row-1][col-1]) return true;
            if (row <= 6 && board.verticalWalls[row+1][col-1]) return true;
            if (col >= 2 && board.horizontalWalls[row][col-2]) return true;
        }
        return false;
    }
    
    bool testIfAdjacentToOtherWallForHorizontalRight(int row, int col) {
        if (col <= 6) {
            if (board.verticalWalls[row][col+1]) return true;
            if (row >= 1 && board.verticalWalls[row-1][col+1]) return true;
            if (row <= 6 && board.verticalWalls[row+1][col+1]) return true;
            if (col <= 5 && board.horizontalWalls[row][col+2]) return true;
        }
        return false;
    }
    
    bool testIfAdjacentToOtherWallForHorizontalMiddle(int row, int col) {
        if (row >= 1 && board.verticalWalls[row-1][col]) return true;
        if (row <= 6 && board.verticalWalls[row+1][col]) return true;
        return false;
    }
    
    bool testIfAdjacentToOtherWallForVerticalTop(int row, int col) {
        if (row >= 1) {
            if (board.horizontalWalls[row-1][col]) return true;
            if (col >= 1 && board.horizontalWalls[row-1][col-1]) return true;
            if (col <= 6 && board.horizontalWalls[row-1][col+1]) return true;
            if (row >= 2 && board.verticalWalls[row-2][col]) return true;
        }
        return false;
    }
    
    bool testIfAdjacentToOtherWallForVerticalBottom(int row, int col) {
        if (row <= 6) {
            if (board.horizontalWalls[row+1][col]) return true;
            if (col >= 1 && board.horizontalWalls[row+1][col-1]) return true;
            if (col <= 6 && board.horizontalWalls[row+1][col+1]) return true;
            if (row <= 5 && board.verticalWalls[row+2][col]) return true;
        }
        return false;
    }
    
    bool testIfAdjacentToOtherWallForVerticalMiddle(int row, int col) {
        if (col >= 1 && board.horizontalWalls[row][col-1]) return true;
        if (col <= 6 && board.horizontalWalls[row][col+1]) return true;
        return false;
    }
    
    bool existPathsToGoalLines() {
        return existPathToGoalLineFor(board.pawns[0]) && existPathToGoalLineFor(board.pawns[1]);
    }
    
    bool existPathToGoalLineFor(const Pawn& pawn) {
        std::array<std::array<bool, 9>, 9> visited;
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                visited[i][j] = false;
            }
        }
        
        // DFS
        const int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        std::function<bool(int, int)> dfs = [&](int row, int col) -> bool {
            for (int d = 0; d < 4; d++) {
                if (isOpenWay(row, col, dirs[d][0], dirs[d][1])) {
                    int nextRow = row + dirs[d][0];
                    int nextCol = col + dirs[d][1];
                    if (!visited[nextRow][nextCol]) {
                        visited[nextRow][nextCol] = true;
                        if (nextRow == pawn.goalRow) return true;
                        if (dfs(nextRow, nextCol)) return true;
                    }
                }
            }
            return false;
        };
        
        return dfs(pawn.position.row, pawn.position.col);
    }
    
    void adjustProbableWallsAfterHorizontal(int row, int col) {
        // Mark nearby positions as probable wall locations
        if (row >= 1) {
            probableVerticalWalls[row-1][col] = true;
        }
        if (row <= 6) {
            probableVerticalWalls[row+1][col] = true;
        }
        if (col >= 1) {
            probableVerticalWalls[row][col-1] = true;
            if (row >= 1) probableVerticalWalls[row-1][col-1] = true;
            if (row <= 6) probableVerticalWalls[row+1][col-1] = true;
            if (col >= 2) {
                probableHorizontalWalls[row][col-2] = true;
                probableVerticalWalls[row][col-2] = true;
                if (col >= 3) probableHorizontalWalls[row][col-3] = true;
            }
        }
        if (col <= 6) {
            probableVerticalWalls[row][col+1] = true;
            if (row >= 1) probableVerticalWalls[row-1][col+1] = true;
            if (row <= 6) probableVerticalWalls[row+1][col+1] = true;
            if (col <= 5) {
                probableHorizontalWalls[row][col+2] = true;
                probableVerticalWalls[row][col+2] = true;
                if (col <= 4) probableHorizontalWalls[row][col+3] = true;
            }
        }
    }
    
    void adjustProbableWallsAfterVertical(int row, int col) {
        // Mark nearby positions as probable wall locations
        if (col >= 1) {
            probableHorizontalWalls[row][col-1] = true;
        }
        if (col <= 6) {
            probableHorizontalWalls[row][col+1] = true;
        }
        if (row >= 1) {
            probableHorizontalWalls[row-1][col] = true;
            if (col >= 1) probableHorizontalWalls[row-1][col-1] = true;
            if (col <= 6) probableHorizontalWalls[row-1][col+1] = true;
            if (row >= 2) {
                probableVerticalWalls[row-2][col] = true;
                probableHorizontalWalls[row-2][col] = true;
                if (row >= 3) probableVerticalWalls[row-3][col] = true;
            }
        }
        if (row <= 6) {
            probableHorizontalWalls[row+1][col] = true;
            if (col >= 1) probableHorizontalWalls[row+1][col-1] = true;
            if (col <= 6) probableHorizontalWalls[row+1][col+1] = true;
            if (row <= 5) {
                probableVerticalWalls[row+2][col] = true;
                probableHorizontalWalls[row+2][col] = true;
                if (row <= 4) probableVerticalWalls[row+3][col] = true;
            }
        }
    }
    
    void updateProbableWalls() {
        // Add walls beside pawns after several turns
        if (turn >= 3) {
            setWallsBesidePawn(getPawnOfNotTurn());
        }
        if (turn >= 6) {
            setWallsBesidePawn(getPawnOfTurn());
            // Add leftmost and rightmost horizontal walls
            for (int i = 0; i < 8; i++) {
                probableHorizontalWalls[i][0] = true;
                probableHorizontalWalls[i][7] = true;
            }
        }
    }
    
    void setWallsBesidePawn(const Pawn& pawn) {
        int row = pawn.position.row;
        int col = pawn.position.col;
        
        if (row >= 1) {
            if (col >= 1) {
                probableHorizontalWalls[row-1][col-1] = true;
                probableVerticalWalls[row-1][col-1] = true;
                if (col >= 2) probableHorizontalWalls[row-1][col-2] = true;
            }
            if (col <= 7) {
                probableHorizontalWalls[row-1][col] = true;
                probableVerticalWalls[row-1][col] = true;
                if (col <= 6) probableHorizontalWalls[row-1][col+1] = true;
            }
            if (row >= 2) {
                if (col >= 1) probableVerticalWalls[row-2][col-1] = true;
                if (col <= 7) probableVerticalWalls[row-2][col] = true;
            }
        }
        if (row <= 7) {
            if (col >= 1) {
                probableHorizontalWalls[row][col-1] = true;
                probableVerticalWalls[row][col-1] = true;
                if (col >= 2) probableHorizontalWalls[row][col-2] = true;
            }
            if (col <= 7) {
                probableHorizontalWalls[row][col] = true;
                probableVerticalWalls[row][col] = true;
                if (col <= 6) probableHorizontalWalls[row][col+1] = true;
            }
            if (row <= 6) {
                if (col >= 1) probableVerticalWalls[row+1][col-1] = true;
                if (col <= 7) probableVerticalWalls[row+1][col] = true;
            }
        }
    }
};
