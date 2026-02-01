/**
 * main.cpp
 * Main entry point for Quoridor AI
 * 
 * Usage:
 *   ./quoridor                    - AI vs AI with visualization
 *   ./quoridor --human            - Human vs AI (terminal)
 *   ./quoridor --gui              - Human vs AI (graphical window)
 *   ./quoridor --train            - Run training mode
 *   ./quoridor --benchmark        - Benchmark AI performance
 * 
 * GUI requires SDL2:
 *   macOS: brew install sdl2
 *   Build with: make gui
 */

#include "Position.hpp"
#include "Pawn.hpp"
#include "Board.hpp"
#include "Move.hpp"
#include "Game.hpp"
#include "AI.hpp"
#include "MNode.hpp"
#include "MCTS.hpp"
#include "QuoridorAI.hpp"
#include "Visualization.hpp"
#include "GUIVisualization.hpp"

#include <iostream>
#include <string>
#include <thread>
#include <chrono>

// Game mode enumeration
enum class GameMode {
    AI_VS_AI,
    HUMAN_VS_AI,
    HUMAN_VS_AI_GUI,
    TRAINING,
    BENCHMARK
};

// Parse command line arguments
GameMode parseArgs(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--gui" || arg == "-g") return GameMode::HUMAN_VS_AI_GUI;
        if (arg == "--human" || arg == "-h") return GameMode::HUMAN_VS_AI;
        if (arg == "--train" || arg == "-t") return GameMode::TRAINING;
        if (arg == "--benchmark" || arg == "-b") return GameMode::BENCHMARK;
    }
    return GameMode::AI_VS_AI;
}

// Human input for move
Move getHumanMove(Game& game) {
    std::cout << "\nEnter your move:\n";
    std::cout << "  m <row> <col>  - Move pawn to position\n";
    std::cout << "  h <row> <col>  - Place horizontal wall\n";
    std::cout << "  v <row> <col>  - Place vertical wall\n";
    std::cout << "> ";
    
    char type;
    int row, col;
    std::cin >> type >> row >> col;
    
    switch (type) {
        case 'm': case 'M':
            return PawnMove(row, col);
        case 'h': case 'H':
            return HorizontalWall(row, col);
        case 'v': case 'V':
            return VerticalWall(row, col);
        default:
            std::cout << "Invalid move type. Try again.\n";
            return getHumanMove(game);
    }
}

// AI vs AI game
void runAIvsAI(int sims1 = 3000, int sims2 = 3000, bool visualize = true, int delayMs = 500) {
    Game game(true);  // Player 0 at bottom
    QuoridorAI ai0(sims1, 1.414, true);
    QuoridorAI ai1(sims2, 1.414, true);
    
    if (visualize) {
        Visualization::clearScreen();
        Visualization::printBoard(game);
        Visualization::printStatus(game);
    }
    
    while (game.winner == -1) {
        QuoridorAI& currentAI = (game.getPawnIndexOfTurn() == 0) ? ai0 : ai1;
        int currentPlayer = game.getPawnIndexOfTurn();
        
        auto [move, winRate] = currentAI.chooseNextMove(game);
        game.doMove(move);
        
        if (visualize) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
            Visualization::clearScreen();
            Visualization::printMove(currentPlayer, move, winRate);
            Visualization::printBoard(game);
            Visualization::printStatus(game);
        }
    }
    
    std::cout << "\n";
    std::cout << "=== GAME OVER ===\n";
    std::cout << "Player " << game.winner << " wins in " << game.turn << " turns!\n";
}

// Human vs AI game (terminal)
void runHumanVsAI(int aiSims = 5000) {
    std::cout << "Do you want to go first? (y/n): ";
    char choice;
    std::cin >> choice;
    
    bool humanFirst = (choice == 'y' || choice == 'Y');
    int humanPlayer = humanFirst ? 0 : 1;
    
    Game game(humanFirst);  // Human at bottom if going first
    QuoridorAI ai(aiSims, 1.414, true);
    
    Visualization::clearScreen();
    Visualization::printBoard(game);
    Visualization::printStatus(game);
    
    while (game.winner == -1) {
        int currentPlayer = game.getPawnIndexOfTurn();
        Move move;
        double winRate = -1;
        
        if (currentPlayer == humanPlayer) {
            // Human's turn
            bool validMove = false;
            while (!validMove) {
                move = getHumanMove(game);
                if (isPawnMove(move)) {
                    const auto& pm = getPawnMove(move);
                    validMove = game.getValidNextPositions()[pm.row][pm.col];
                } else if (isHorizontalWall(move)) {
                    const auto& hw = getHorizontalWall(move);
                    validMove = game.getPawnOfTurn().hasWalls() &&
                                game.testIfExistPathsToGoalLinesAfterPlaceHorizontalWall(hw.row, hw.col);
                } else if (isVerticalWall(move)) {
                    const auto& vw = getVerticalWall(move);
                    validMove = game.getPawnOfTurn().hasWalls() &&
                                game.testIfExistPathsToGoalLinesAfterPlaceVerticalWall(vw.row, vw.col);
                }
                if (!validMove) {
                    std::cout << "Invalid move! Try again.\n";
                }
            }
        } else {
            // AI's turn
            std::cout << "\nAI is thinking...\n";
            auto result = ai.chooseNextMove(game);
            move = result.first;
            winRate = result.second;
        }
        
        game.doMove(move);
        
        Visualization::clearScreen();
        Visualization::printMove(currentPlayer, move, winRate);
        Visualization::printBoard(game);
        Visualization::printStatus(game);
    }
    
    std::cout << "\n";
    if (game.winner == humanPlayer) {
        std::cout << "=== CONGRATULATIONS! You win! ===\n";
    } else {
        std::cout << "=== AI wins! Better luck next time! ===\n";
    }
}

// Human vs AI game (GUI)
void runHumanVsAIGUI(int aiSims = 5000) {
#ifdef USE_GUI
    try {
        GUIVisualization gui;
        
        int humanPlayer = 0;  // Human is always player 0 in GUI mode
        Game game(true);      // Human at bottom
        QuoridorAI ai(aiSims, 1.414, false);
        
        gui.run(game, ai, humanPlayer);
        
    } catch (const std::exception& e) {
        std::cerr << "GUI error: " << e.what() << std::endl;
        std::cerr << "Falling back to terminal mode...\n\n";
        runHumanVsAI(aiSims);
    }
#else
    std::cerr << "GUI mode not available.\n";
    std::cerr << "Please rebuild with: make gui\n";
    std::cerr << "Requires SDL2: brew install sdl2 sdl2_ttf\n";
    std::cerr << "\nFalling back to terminal mode...\n\n";
    runHumanVsAI(aiSims);
#endif
}

// Training mode - run multiple games with different parameters
void runTraining() {
    std::cout << "=== TRAINING MODE ===\n\n";
    
    const std::vector<int> simulations = {1000, 2000, 3000};
    const std::vector<double> uctConstants = {0.7, 1.0, 1.414, 2.0};
    const int gamesPerConfig = 10;
    
    std::cout << "Configuration: " << simulations.size() * uctConstants.size() 
              << " parameter sets, " << gamesPerConfig << " games each\n\n";
    
    for (double uct : uctConstants) {
        for (int sims : simulations) {
            int player0Wins = 0;
            int player1Wins = 0;
            int totalMoves = 0;
            
            std::cout << "Testing: UCT=" << uct << ", Sims=" << sims << " ... ";
            std::cout.flush();
            
            for (int g = 0; g < gamesPerConfig; g++) {
                Game game(true);
                QuoridorAI ai0(sims, uct);
                QuoridorAI ai1(sims, uct);
                
                while (game.winner == -1 && game.turn < 200) {
                    QuoridorAI& currentAI = (game.getPawnIndexOfTurn() == 0) ? ai0 : ai1;
                    auto [move, _] = currentAI.chooseNextMove(game);
                    game.doMove(move);
                }
                
                if (game.winner == 0) player0Wins++;
                else if (game.winner == 1) player1Wins++;
                totalMoves += game.turn;
            }
            
            std::cout << "P0:" << player0Wins << " P1:" << player1Wins 
                      << " Avg moves:" << (totalMoves / gamesPerConfig) << "\n";
        }
    }
    
    std::cout << "\n=== TRAINING COMPLETE ===\n";
}

// Benchmark mode
void runBenchmark() {
    std::cout << "=== BENCHMARK MODE ===\n\n";
    
    const std::vector<int> simulations = {500, 1000, 2000, 5000};
    
    for (int sims : simulations) {
        Game game(true);
        QuoridorAI ai(sims, 1.414);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto [move, winRate] = ai.chooseNextMove(game);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Sims: " << sims << " | Time: " << duration.count() 
                  << " ms | Rate: " << (sims * 1000.0 / duration.count()) << " sims/sec\n";
    }
    
    std::cout << "\n=== BENCHMARK COMPLETE ===\n";
}

void printUsage() {
    std::cout << "Usage:\n";
    std::cout << "  ./quoridor              AI vs AI with visualization\n";
    std::cout << "  ./quoridor --human      Human vs AI (terminal input)\n";
    std::cout << "  ./quoridor --gui        Human vs AI (graphical window)\n";
    std::cout << "  ./quoridor --train      Run training mode\n";
    std::cout << "  ./quoridor --benchmark  Benchmark AI performance\n";
    std::cout << "  ./quoridor --help       Show this help\n";
}

int main(int argc, char* argv[]) {
    // Check for help flag
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-?") {
            printUsage();
            return 0;
        }
    }
    
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════╗\n";
    std::cout << "║    Quoridor AI - MCTS Implementation      ║\n";
    std::cout << "║    Based on gorisanson/quoridor-ai        ║\n";
    std::cout << "╚═══════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    GameMode mode = parseArgs(argc, argv);
    
    switch (mode) {
        case GameMode::AI_VS_AI:
            std::cout << "Mode: AI vs AI\n";
            runAIvsAI(3000, 3000, true, 1000);
            break;
            
        case GameMode::HUMAN_VS_AI:
            std::cout << "Mode: Human vs AI (Terminal)\n";
            runHumanVsAI(5000);
            break;
            
        case GameMode::HUMAN_VS_AI_GUI:
            std::cout << "Mode: Human vs AI (GUI)\n";
            runHumanVsAIGUI(5000);
            break;
            
        case GameMode::TRAINING:
            runTraining();
            break;
            
        case GameMode::BENCHMARK:
            runBenchmark();
            break;
    }
    
    return 0;
}
