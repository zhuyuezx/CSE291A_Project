#pragma once
#include "Game.hpp"
#include <vector>
#include <queue>
#include <limits>
#include <random>
#include <algorithm>

/**
 * AI.hpp
 * AI utilities and heuristics for Quoridor
 * Contains pathfinding and strategic move selection
 */
class AI {
public:
    // Random number generator
    static std::mt19937& getRng() {
        static std::mt19937 rng(std::random_device{}());
        return rng;
    }
    
    // Check if pawns are adjacent
    static bool arePawnsAdjacent(const Game& game) {
        const Position& p1 = game.getPawnOfTurn().position;
        const Position& p2 = game.getPawnOfNotTurn().position;
        return (p1.row == p2.row && std::abs(p1.col - p2.col) == 1) ||
               (p1.col == p2.col && std::abs(p1.row - p2.row) == 1);
    }
    
    // Get shortest distance to goal for a pawn using BFS
    static int getShortestDistanceToGoal(const Pawn& pawn, const Game& game) {
        std::array<std::array<int, 9>, 9> dist;
        std::array<std::array<bool, 9>, 9> visited;
        
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                dist[i][j] = std::numeric_limits<int>::max();
                visited[i][j] = false;
            }
        }
        
        const int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        std::queue<Position> queue;
        
        visited[pawn.position.row][pawn.position.col] = true;
        dist[pawn.position.row][pawn.position.col] = 0;
        queue.push(pawn.position);
        
        while (!queue.empty()) {
            Position pos = queue.front();
            queue.pop();
            
            if (pos.row == pawn.goalRow) {
                return dist[pos.row][pos.col];
            }
            
            for (int d = 0; d < 4; d++) {
                if (game.isOpenWay(pos.row, pos.col, dirs[d][0], dirs[d][1])) {
                    Position next = pos.addMove(dirs[d][0], dirs[d][1]);
                    if (!visited[next.row][next.col]) {
                        dist[next.row][next.col] = dist[pos.row][pos.col] + 1;
                        visited[next.row][next.col] = true;
                        queue.push(next);
                    }
                }
            }
        }
        
        return std::numeric_limits<int>::max(); // No path found
    }
    
    // Get next positions on shortest path to goal
    struct PathInfo {
        std::array<std::array<Position*, 9>, 9> prev;  // Previous position on path
        std::array<std::array<Position*, 9>, 9> next;  // Next position toward goal
        int distanceToGoal;
        Position goalPos;
        
        PathInfo() {
            for (int i = 0; i < 9; i++) {
                for (int j = 0; j < 9; j++) {
                    prev[i][j] = nullptr;
                    next[i][j] = nullptr;
                }
            }
            distanceToGoal = std::numeric_limits<int>::max();
        }
        
        ~PathInfo() {
            for (int i = 0; i < 9; i++) {
                for (int j = 0; j < 9; j++) {
                    delete prev[i][j];
                    delete next[i][j];
                }
            }
        }
    };
    
    // BFS to find shortest path
    static void getShortestPathInfo(const Pawn& pawn, const Game& game, 
                                    std::array<std::array<int, 9>, 9>& dist,
                                    std::array<std::array<std::pair<int,int>, 9>, 9>& prev,
                                    int& goalRow, int& goalCol) {
        std::array<std::array<bool, 9>, 9> visited;
        
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                dist[i][j] = std::numeric_limits<int>::max();
                visited[i][j] = false;
                prev[i][j] = {-1, -1};
            }
        }
        
        // Shuffle directions for randomness
        std::vector<std::pair<int,int>> dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        std::shuffle(dirs.begin(), dirs.end(), getRng());
        
        std::queue<Position> queue;
        visited[pawn.position.row][pawn.position.col] = true;
        dist[pawn.position.row][pawn.position.col] = 0;
        queue.push(pawn.position);
        
        goalRow = -1;
        goalCol = -1;
        
        while (!queue.empty()) {
            Position pos = queue.front();
            queue.pop();
            
            if (pos.row == pawn.goalRow) {
                goalRow = pos.row;
                goalCol = pos.col;
                return;
            }
            
            for (const auto& dir : dirs) {
                if (game.isOpenWay(pos.row, pos.col, dir.first, dir.second)) {
                    Position next = pos.addMove(dir.first, dir.second);
                    if (!visited[next.row][next.col]) {
                        dist[next.row][next.col] = dist[pos.row][pos.col] + 1;
                        prev[next.row][next.col] = {pos.row, pos.col};
                        visited[next.row][next.col] = true;
                        queue.push(next);
                    }
                }
            }
        }
    }
    
    // Get next position array from prev array
    static void getNextFromPrev(const std::array<std::array<std::pair<int,int>, 9>, 9>& prev,
                                int goalRow, int goalCol,
                                std::array<std::array<std::pair<int,int>, 9>, 9>& next) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                next[i][j] = {-1, -1};
            }
        }
        
        int curRow = goalRow, curCol = goalCol;
        while (prev[curRow][curCol].first != -1) {
            int prevRow = prev[curRow][curCol].first;
            int prevCol = prev[curRow][curCol].second;
            next[prevRow][prevCol] = {curRow, curCol};
            curRow = prevRow;
            curCol = prevCol;
        }
    }
    
    // Choose next position on shortest path
    static Position chooseShortestPathNextPosition(Game& game) {
        if (arePawnsAdjacent(game)) {
            return chooseShortestPathNextPositionThoroughly(game);
        }
        
        std::array<std::array<int, 9>, 9> dist;
        std::array<std::array<std::pair<int,int>, 9>, 9> prev;
        std::array<std::array<std::pair<int,int>, 9>, 9> next;
        int goalRow, goalCol;
        
        getShortestPathInfo(game.getPawnOfTurn(), game, dist, prev, goalRow, goalCol);
        
        if (goalRow == -1) {
            // No path found, shouldn't happen in valid game
            return game.getPawnOfTurn().position;
        }
        
        getNextFromPrev(prev, goalRow, goalCol, next);
        
        const Position& curPos = game.getPawnOfTurn().position;
        if (next[curPos.row][curPos.col].first != -1) {
            return Position(next[curPos.row][curPos.col].first, next[curPos.row][curPos.col].second);
        }
        
        return curPos;
    }
    
    // Choose shortest path considering pawn adjacency
    static Position chooseShortestPathNextPositionThoroughly(Game& game) {
        auto validPositions = game.getArrOfValidNextPositionTuples();
        if (validPositions.empty()) {
            return game.getPawnOfTurn().position;
        }
        
        std::vector<int> distances;
        for (const auto& pos : validPositions) {
            Game cloned = game.clone();
            cloned.movePawn(pos.first, pos.second);
            int d = getShortestDistanceToGoal(cloned.getPawnOfNotTurn(), cloned);
            distances.push_back(d);
        }
        
        // Find minimum distance positions
        int minDist = *std::min_element(distances.begin(), distances.end());
        std::vector<std::pair<int,int>> bestPositions;
        for (size_t i = 0; i < distances.size(); i++) {
            if (distances[i] == minDist) {
                bestPositions.push_back(validPositions[i]);
            }
        }
        
        // Random choice among best
        std::uniform_int_distribution<> dis(0, bestPositions.size() - 1);
        const auto& chosen = bestPositions[dis(getRng())];
        return Position(chosen.first, chosen.second);
    }
    
    // Choose longest path next position (for simulation diversity)
    static Position chooseLongestPathNextPositionThoroughly(Game& game) {
        auto validPositions = game.getArrOfValidNextPositionTuples();
        if (validPositions.empty()) {
            return game.getPawnOfTurn().position;
        }
        
        std::vector<int> distances;
        for (const auto& pos : validPositions) {
            Game cloned = game.clone();
            cloned.movePawn(pos.first, pos.second);
            int d = getShortestDistanceToGoal(cloned.getPawnOfNotTurn(), cloned);
            distances.push_back(d);
        }
        
        // Find maximum distance positions (excluding infinity)
        int maxDist = -1;
        for (int d : distances) {
            if (d != std::numeric_limits<int>::max() && d > maxDist) {
                maxDist = d;
            }
        }
        
        std::vector<std::pair<int,int>> bestPositions;
        for (size_t i = 0; i < distances.size(); i++) {
            if (distances[i] == maxDist) {
                bestPositions.push_back(validPositions[i]);
            }
        }
        
        if (bestPositions.empty()) {
            bestPositions = validPositions;
        }
        
        std::uniform_int_distribution<> dis(0, bestPositions.size() - 1);
        const auto& chosen = bestPositions[dis(getRng())];
        return Position(chosen.first, chosen.second);
    }
    
    // Choose a random valid next position
    static Position chooseNextPositionRandomly(Game& game) {
        auto validPositions = game.getArrOfValidNextPositionTuples();
        if (validPositions.empty()) {
            return game.getPawnOfTurn().position;
        }
        
        std::uniform_int_distribution<> dis(0, validPositions.size() - 1);
        const auto& chosen = validPositions[dis(getRng())];
        return Position(chosen.first, chosen.second);
    }
    
    // Choose a random probable wall
    static std::optional<Move> chooseProbableNextWall(Game& game) {
        std::vector<Move> moves;
        
        auto hWalls = game.getArrOfProbableValidNoBlockHorizontalWalls();
        for (const auto& w : hWalls) {
            moves.push_back(HorizontalWall(w.first, w.second));
        }
        
        auto vWalls = game.getArrOfProbableValidNoBlockVerticalWalls();
        for (const auto& w : vWalls) {
            moves.push_back(VerticalWall(w.first, w.second));
        }
        
        if (moves.empty()) {
            return std::nullopt;
        }
        
        std::uniform_int_distribution<> dis(0, moves.size() - 1);
        return moves[dis(getRng())];
    }
    
    // Choose a random valid wall
    static std::optional<Move> chooseNextWallRandomly(Game& game) {
        std::vector<Move> moves;
        
        auto hWalls = game.getArrOfValidNoBlockHorizontalWalls();
        for (const auto& w : hWalls) {
            moves.push_back(HorizontalWall(w.first, w.second));
        }
        
        auto vWalls = game.getArrOfValidNoBlockVerticalWalls();
        for (const auto& w : vWalls) {
            moves.push_back(VerticalWall(w.first, w.second));
        }
        
        if (moves.empty()) {
            return std::nullopt;
        }
        
        std::uniform_int_distribution<> dis(0, moves.size() - 1);
        return moves[dis(getRng())];
    }
    
    // Get walls that disturb a pawn's path
    static void getWallsDisturbingPath(const Pawn& pawn, Game& game,
                                       std::vector<std::pair<int,int>>& hWalls,
                                       std::vector<std::pair<int,int>>& vWalls) {
        hWalls.clear();
        vWalls.clear();
        
        // Get all shortest paths
        std::array<std::array<int, 9>, 9> dist;
        std::array<std::array<std::vector<std::pair<int,int>>, 9>, 9> multiPrev;
        std::array<std::array<bool, 9>, 9> visited;
        std::array<std::array<bool, 9>, 9> searched;
        
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                dist[i][j] = std::numeric_limits<int>::max();
                visited[i][j] = false;
                searched[i][j] = false;
            }
        }
        
        const int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        std::queue<Position> queue;
        
        visited[pawn.position.row][pawn.position.col] = true;
        dist[pawn.position.row][pawn.position.col] = 0;
        queue.push(pawn.position);
        
        while (!queue.empty()) {
            Position pos = queue.front();
            queue.pop();
            
            for (int d = 0; d < 4; d++) {
                if (game.isOpenWay(pos.row, pos.col, dirs[d][0], dirs[d][1])) {
                    Position next = pos.addMove(dirs[d][0], dirs[d][1]);
                    if (!searched[next.row][next.col]) {
                        int alt = dist[pos.row][pos.col] + 1;
                        if (alt < dist[next.row][next.col]) {
                            dist[next.row][next.col] = alt;
                            multiPrev[next.row][next.col].clear();
                            multiPrev[next.row][next.col].push_back({pos.row, pos.col});
                        } else if (alt == dist[next.row][next.col]) {
                            multiPrev[next.row][next.col].push_back({pos.row, pos.col});
                        }
                        if (!visited[next.row][next.col]) {
                            visited[next.row][next.col] = true;
                            queue.push(next);
                        }
                    }
                }
            }
            searched[pos.row][pos.col] = true;
        }
        
        // Find goal positions with minimum distance
        int minDist = std::numeric_limits<int>::max();
        for (int c = 0; c < 9; c++) {
            if (dist[pawn.goalRow][c] < minDist) {
                minDist = dist[pawn.goalRow][c];
            }
        }
        
        // Trace back all shortest paths and mark disturbing walls
        std::array<std::array<bool, 8>, 8> hWallMarked, vWallMarked;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                hWallMarked[i][j] = false;
                vWallMarked[i][j] = false;
            }
        }
        
        std::queue<std::pair<int,int>> traceQueue;
        std::array<std::array<bool, 9>, 9> traced;
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                traced[i][j] = false;
            }
        }
        
        for (int c = 0; c < 9; c++) {
            if (dist[pawn.goalRow][c] == minDist) {
                traceQueue.push({pawn.goalRow, c});
            }
        }
        
        while (!traceQueue.empty()) {
            auto [row, col] = traceQueue.front();
            traceQueue.pop();
            
            for (const auto& prevPos : multiPrev[row][col]) {
                int prevRow = prevPos.first;
                int prevCol = prevPos.second;
                
                int dr = row - prevRow;
                int dc = col - prevCol;
                
                // Mark walls that would block this move
                if (dr == -1 && dc == 0) { // UP
                    if (prevCol < 8) hWallMarked[prevRow-1][prevCol] = true;
                    if (prevCol > 0) hWallMarked[prevRow-1][prevCol-1] = true;
                } else if (dr == 1 && dc == 0) { // DOWN
                    if (prevCol < 8) hWallMarked[prevRow][prevCol] = true;
                    if (prevCol > 0) hWallMarked[prevRow][prevCol-1] = true;
                } else if (dr == 0 && dc == -1) { // LEFT
                    if (prevRow < 8) vWallMarked[prevRow][prevCol-1] = true;
                    if (prevRow > 0) vWallMarked[prevRow-1][prevCol-1] = true;
                } else if (dr == 0 && dc == 1) { // RIGHT
                    if (prevRow < 8) vWallMarked[prevRow][prevCol] = true;
                    if (prevRow > 0) vWallMarked[prevRow-1][prevCol] = true;
                }
                
                if (!traced[prevRow][prevCol]) {
                    traced[prevRow][prevCol] = true;
                    traceQueue.push({prevRow, prevCol});
                }
            }
        }
        
        // Collect valid disturbing walls
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (hWallMarked[i][j] && game.validHorizontalWalls[i][j] &&
                    game.testIfExistPathsToGoalLinesAfterPlaceHorizontalWall(i, j)) {
                    hWalls.push_back({i, j});
                }
                if (vWallMarked[i][j] && game.validVerticalWalls[i][j] &&
                    game.testIfExistPathsToGoalLinesAfterPlaceVerticalWall(i, j)) {
                    vWalls.push_back({i, j});
                }
            }
        }
    }
};
