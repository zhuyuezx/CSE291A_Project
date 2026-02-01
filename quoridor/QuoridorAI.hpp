#pragma once
#include "Game.hpp"
#include "MCTS.hpp"
#include "AI.hpp"
#include <chrono>
#include <iostream>

/**
 * QuoridorAI.hpp
 * AI player for Quoridor using MCTS
 */
class QuoridorAI {
public:
    int numSimulations;
    double uctConst;
    bool developMode;
    
    QuoridorAI(int sims = 5000, double uct = 1.414, bool devMode = false)
        : numSimulations(sims), uctConst(uct), developMode(devMode) {}
    
    // Choose the next move using MCTS
    std::pair<Move, double> chooseNextMove(Game& game) {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Heuristic: for first move of each pawn, just go forward
        if (game.turn < 2) {
            Position nextPos = AI::chooseShortestPathNextPosition(game);
            const Position& curPos = game.getPawnOfTurn().position;
            
            // If movement is purely vertical (forward), just do it
            if (nextPos.col == curPos.col) {
                return {PawnMove(nextPos.row, nextPos.col), 1.0};
            }
        }
        
        // Run MCTS
        MonteCarloTreeSearch mcts(game, uctConst);
        mcts.search(numSimulations);
        
        auto [bestMove, winRate] = mcts.selectBestMove();
        
        // Heuristic: for initial phase, help AI find shortest path
        if ((game.turn < 6 && game.getPawnOfTurn().position.col == 4) || winRate < 0.1) {
            if (isPawnMove(bestMove)) {
                const auto& pm = getPawnMove(bestMove);
                Position bestPos = AI::chooseShortestPathNextPositionThoroughly(game);
                if (pm.row != bestPos.row || pm.col != bestPos.col) {
                    if (developMode) {
                        std::cout << "AI heuristic correction applied\n";
                    }
                    bestMove = PawnMove(bestPos.row, bestPos.col);
                }
            }
        }
        
        // Common openings heuristic
        const Position& oppPos = game.getPawnOfNotTurn().position;
        if (game.turn < 5 && oppPos.col == 4) {
            std::uniform_real_distribution<> prob(0.0, 1.0);
            if (oppPos.row == 6 && prob(AI::getRng()) < 0.5) {
                std::vector<Move> openingMoves = {
                    HorizontalWall(5, 3), HorizontalWall(5, 4),
                    VerticalWall(4, 3), VerticalWall(4, 4)
                };
                std::uniform_int_distribution<> dis(0, openingMoves.size() - 1);
                bestMove = openingMoves[dis(AI::getRng())];
            } else if (oppPos.row == 2 && prob(AI::getRng()) < 0.5) {
                std::vector<Move> openingMoves = {
                    HorizontalWall(2, 3), HorizontalWall(2, 4),
                    VerticalWall(3, 3), VerticalWall(3, 4)
                };
                std::uniform_int_distribution<> dis(0, openingMoves.size() - 1);
                bestMove = openingMoves[dis(AI::getRng())];
            }
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        if (developMode) {
            std::cout << "MCTS: " << numSimulations << " simulations in " 
                      << duration.count() << " ms, UCT=" << uctConst 
                      << ", Win rate=" << (winRate * 100) << "%\n";
        }
        
        return {bestMove, winRate};
    }
};
