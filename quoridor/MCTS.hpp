#pragma once
#include "MNode.hpp"
#include "Game.hpp"
#include "AI.hpp"
#include <stack>
#include <random>

/**
 * MCTS.hpp
 * Monte Carlo Tree Search implementation for Quoridor
 */
class MonteCarloTreeSearch {
public:
    Game& game;
    double uctConst;
    std::unique_ptr<MNode> root;
    int totalSimulations;
    
    // Random number generator
    std::mt19937& rng;
    
    MonteCarloTreeSearch(Game& g, double uctConstant)
        : game(g), uctConst(uctConstant), totalSimulations(0), rng(AI::getRng()) {
        root = std::make_unique<MNode>(PawnMove(), nullptr);  // Dummy move for root
    }
    
    // Run MCTS for specified number of simulations
    void search(int numSimulations) {
        int limitSimulations = totalSimulations + numSimulations;
        
        while (totalSimulations < limitSimulations) {
            MNode* currentNode = root.get();
            
            // Selection phase
            while (!currentNode->isLeaf() && !currentNode->isTerminal) {
                currentNode = currentNode->getMaxUCTChild(uctConst);
            }
            
            if (currentNode->isTerminal) {
                rollout(currentNode);
            } else if (currentNode->isNew()) {
                rollout(currentNode);
            } else {
                // Expansion phase
                Game simGame = getSimulationGameAtNode(currentNode);
                expand(currentNode, simGame);
                
                if (!currentNode->children.empty()) {
                    // Pick random child for rollout
                    std::uniform_int_distribution<> dis(0, currentNode->children.size() - 1);
                    MNode* child = currentNode->children[dis(rng)].get();
                    rollout(child);
                } else {
                    // No valid moves - terminal node
                    currentNode->isTerminal = true;
                    rollout(currentNode);
                }
            }
        }
    }
    
    // Select the best move based on most simulations
    std::pair<Move, double> selectBestMove() {
        MNode* best = root->getMaxSimsChild();
        if (best) {
            return {best->move, best->getWinRate()};
        }
        // Fallback - shouldn't happen
        return {PawnMove(game.getPawnOfTurn().position.row, game.getPawnOfTurn().position.col), 0.0};
    }
    
private:
    // Get game state at a node by replaying moves from root
    Game getSimulationGameAtNode(MNode* node) {
        Game simGame = game.clone();
        std::stack<Move> moves;
        
        MNode* ancestor = node;
        while (ancestor->parent != nullptr) {
            moves.push(ancestor->move);
            ancestor = ancestor->parent;
        }
        
        while (!moves.empty()) {
            simGame.doMove(moves.top());
            moves.pop();
        }
        
        return simGame;
    }
    
    // Expand node by adding all valid children
    void expand(MNode* node, Game& simGame) {
        // If opponent has no walls left, use heuristic
        if (simGame.getPawnOfNotTurn().wallsLeft == 0) {
            // Only move along shortest path
            auto nextPositions = getShortestPathNextPositions(simGame);
            for (const auto& pos : nextPositions) {
                auto child = std::make_unique<MNode>(PawnMove(pos.row, pos.col), node);
                node->addChild(std::move(child));
            }
            
            // Place walls only to disturb opponent
            if (simGame.getPawnOfTurn().hasWalls()) {
                std::vector<std::pair<int,int>> hWalls, vWalls;
                AI::getWallsDisturbingPath(simGame.getPawnOfNotTurn(), simGame, hWalls, vWalls);
                
                for (const auto& w : hWalls) {
                    auto child = std::make_unique<MNode>(HorizontalWall(w.first, w.second), node);
                    node->addChild(std::move(child));
                }
                for (const auto& w : vWalls) {
                    auto child = std::make_unique<MNode>(VerticalWall(w.first, w.second), node);
                    node->addChild(std::move(child));
                }
            }
        } else {
            // Add all valid pawn moves
            auto positions = simGame.getArrOfValidNextPositionTuples();
            for (const auto& pos : positions) {
                auto child = std::make_unique<MNode>(PawnMove(pos.first, pos.second), node);
                node->addChild(std::move(child));
            }
            
            // Add probable valid wall moves
            if (simGame.getPawnOfTurn().hasWalls()) {
                auto hWalls = simGame.getArrOfProbableValidNoBlockHorizontalWalls();
                for (const auto& w : hWalls) {
                    auto child = std::make_unique<MNode>(HorizontalWall(w.first, w.second), node);
                    node->addChild(std::move(child));
                }
                
                auto vWalls = simGame.getArrOfProbableValidNoBlockVerticalWalls();
                for (const auto& w : vWalls) {
                    auto child = std::make_unique<MNode>(VerticalWall(w.first, w.second), node);
                    node->addChild(std::move(child));
                }
            }
        }
    }
    
    // Get positions on shortest path (for heuristic expansion)
    std::vector<Position> getShortestPathNextPositions(Game& simGame) {
        auto validPositions = simGame.getArrOfValidNextPositionTuples();
        if (validPositions.empty()) {
            return {};
        }
        
        std::vector<int> distances;
        for (const auto& pos : validPositions) {
            Game cloned = simGame.clone();
            cloned.movePawn(pos.first, pos.second);
            int d = AI::getShortestDistanceToGoal(cloned.getPawnOfNotTurn(), cloned);
            distances.push_back(d);
        }
        
        int minDist = *std::min_element(distances.begin(), distances.end());
        std::vector<Position> result;
        for (size_t i = 0; i < distances.size(); i++) {
            if (distances[i] == minDist) {
                result.push_back(Position(validPositions[i].first, validPositions[i].second));
            }
        }
        return result;
    }
    
    // Rollout (simulation) phase
    void rollout(MNode* node) {
        totalSimulations++;
        Game simGame = getSimulationGameAtNode(node);
        
        // The pawn index at this node (who moved to get here)
        int nodePawnIndex = simGame.getPawnIndexOfNotTurn();
        
        if (simGame.winner != -1) {
            node->isTerminal = true;
        }
        
        // Simulation with heuristics
        simulateGame(simGame);
        
        // Backpropagation
        backpropagate(node, simGame.winner, nodePawnIndex);
    }
    
    // Simulate game to completion
    void simulateGame(Game& simGame) {
        // Cache shortest path info for efficiency
        struct PathCache {
            std::array<std::array<std::pair<int,int>, 9>, 9> prev;
            std::array<std::array<std::pair<int,int>, 9>, 9> next;
            int distanceToGoal;
            bool updated;
            
            PathCache() : distanceToGoal(0), updated(false) {}
        };
        
        PathCache cache[2];
        bool pawnMoveFlag = false;
        int maxMoves = 200;  // Prevent infinite games
        int moves = 0;
        
        while (simGame.winner == -1 && moves < maxMoves) {
            moves++;
            
            int pawnIndex = simGame.getPawnIndexOfTurn();
            const Pawn& pawn = simGame.getPawnOfTurn();
            
            // Update path cache if needed
            if (!cache[pawnIndex].updated) {
                std::array<std::array<int, 9>, 9> dist;
                int goalRow, goalCol;
                AI::getShortestPathInfo(pawn, simGame, dist, cache[pawnIndex].prev, goalRow, goalCol);
                if (goalRow != -1) {
                    AI::getNextFromPrev(cache[pawnIndex].prev, goalRow, goalCol, cache[pawnIndex].next);
                    cache[pawnIndex].distanceToGoal = dist[goalRow][goalCol];
                }
                cache[pawnIndex].updated = true;
            }
            
            // Heuristic: with 70% probability, move toward goal
            std::uniform_real_distribution<> prob(0.0, 1.0);
            if (prob(rng) < 0.7) {
                // Move pawn toward goal
                pawnMoveFlag = false;
                const Position& curPos = pawn.position;
                auto& next = cache[pawnIndex].next;
                
                if (next[curPos.row][curPos.col].first != -1) {
                    Position nextPos(next[curPos.row][curPos.col].first, next[curPos.row][curPos.col].second);
                    
                    // Check for adjacent pawn jump
                    if (AI::arePawnsAdjacent(simGame)) {
                        auto& nextNext = next[nextPos.row][nextPos.col];
                        if (nextNext.first != -1) {
                            Position nnPos(nextNext.first, nextNext.second);
                            if (simGame.getValidNextPositions()[nnPos.row][nnPos.col]) {
                                simGame.movePawn(nnPos.row, nnPos.col);
                                cache[pawnIndex].distanceToGoal -= 2;
                                continue;
                            }
                        }
                        // Find shortest path position thoroughly
                        Position bestPos = AI::chooseShortestPathNextPositionThoroughly(simGame);
                        if (bestPos == nextPos) {
                            cache[pawnIndex].distanceToGoal -= 1;
                        } else {
                            nextPos = bestPos;
                            cache[pawnIndex].updated = false;
                        }
                    } else {
                        cache[pawnIndex].distanceToGoal -= 1;
                    }
                    simGame.movePawn(nextPos.row, nextPos.col);
                } else {
                    // Already at goal? Shouldn't happen
                    simGame.movePawn(curPos.row, curPos.col);
                }
            } else if (!pawnMoveFlag && pawn.wallsLeft > 0) {
                // Try to place a wall
                auto wallMove = AI::chooseProbableNextWall(simGame);
                if (wallMove.has_value()) {
                    simGame.doMove(*wallMove);
                    cache[0].updated = false;
                    cache[1].updated = false;
                } else {
                    pawnMoveFlag = true;
                }
            } else {
                // Move pawn backward or randomly
                pawnMoveFlag = false;
                Position prevPos = AI::chooseLongestPathNextPositionThoroughly(simGame);
                simGame.movePawn(prevPos.row, prevPos.col);
                cache[pawnIndex].updated = false;
            }
        }
        
        // If max moves reached, declare winner by distance
        if (simGame.winner == -1) {
            int d0 = AI::getShortestDistanceToGoal(simGame.getPawn0(), simGame);
            int d1 = AI::getShortestDistanceToGoal(simGame.getPawn1(), simGame);
            simGame.winner = (d0 <= d1) ? 0 : 1;
        }
    }
    
    // Backpropagate results up the tree
    void backpropagate(MNode* node, int winner, int nodePawnIndex) {
        MNode* ancestor = node;
        int ancestorPawnIndex = nodePawnIndex;
        
        while (ancestor != nullptr) {
            ancestor->numSims++;
            if (winner == ancestorPawnIndex) {
                ancestor->numWins += 1.0;
            }
            ancestor = ancestor->parent;
            ancestorPawnIndex = (ancestorPawnIndex + 1) % 2;
        }
    }
};
