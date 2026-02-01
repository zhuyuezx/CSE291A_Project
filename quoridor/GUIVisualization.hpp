#pragma once

/**
 * GUIVisualization.hpp  
 * SDL2-based graphical interface for Quoridor with text rendering
 * 
 * Requires SDL2 and SDL2_ttf libraries:
 *   macOS: brew install sdl2 sdl2_ttf
 *   Linux: sudo apt-get install libsdl2-dev libsdl2-ttf-dev
 */

#ifdef USE_GUI

#include <SDL.h>
#include <SDL_ttf.h>
#include <iostream>
#include <string>
#include <vector>
#include <optional>
#include "Game.hpp"
#include "Move.hpp"

enum class InputMode {
    MOVE_PAWN,
    PLACE_H_WALL,
    PLACE_V_WALL
};

struct Color {
    uint8_t r, g, b, a;
};

class GUIVisualization {
public:
    // Window constants
    static constexpr int WINDOW_WIDTH = 800;
    static constexpr int WINDOW_HEIGHT = 900;
    static constexpr int CELL_SIZE = 60;
    static constexpr int WALL_THICKNESS = 8;
    static constexpr int BOARD_OFFSET_X = 80;
    static constexpr int BOARD_OFFSET_Y = 100;
    
    // Colors
    static constexpr Color COLOR_BG = {25, 25, 25, 255};
    static constexpr Color COLOR_BOARD = {139, 90, 43, 255};
    static constexpr Color COLOR_CELL = {210, 180, 140, 255};
    static constexpr Color COLOR_GRID = {100, 70, 40, 255};
    static constexpr Color COLOR_PLAYER0 = {51, 102, 204, 255};  // Blue
    static constexpr Color COLOR_PLAYER1 = {204, 51, 51, 255};   // Red
    static constexpr Color COLOR_WALL = {80, 50, 20, 255};
    static constexpr Color COLOR_VALID_MOVE = {100, 200, 100, 150};
    static constexpr Color COLOR_HOVER = {255, 255, 100, 100};
    static constexpr Color COLOR_TEXT = {240, 240, 240, 255};
    static constexpr Color COLOR_BUTTON = {70, 130, 180, 255};
    static constexpr Color COLOR_BUTTON_HOVER = {100, 160, 210, 255};
    
private:
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    TTF_Font* font = nullptr;
    TTF_Font* fontLarge = nullptr;
    bool initialized = false;
    
    InputMode currentMode = InputMode::MOVE_PAWN;
    std::optional<std::pair<int, int>> hoveredCell;
    std::optional<std::pair<int, int>> hoveredWall;
    bool wallPreviewValid = false;
    
    // Button rectangles
    SDL_Rect btnMovePawn = {50, 40, 210, 40};
    SDL_Rect btnHWall = {280, 40, 210, 40};
    SDL_Rect btnVWall = {510, 40, 210, 40};
    
public:
    GUIVisualization() {
        std::cout << "Initializing GUI..." << std::endl;
        if (!init()) {
            throw std::runtime_error("Failed to initialize GUI");
        }
        std::cout << "GUI initialized successfully" << std::endl;
    }
    
    ~GUIVisualization() {
        cleanup();
    }
    
    // Run the GUI game loop
    void run(Game& game, QuoridorAI& ai, int humanPlayer) {
        std::cout << "GUI window opened. Controls: Click buttons or press M/H/V to switch modes..." << std::endl;
        std::cout << "You are Player " << humanPlayer << " (" 
                  << (humanPlayer == 0 ? "Blue" : "Red") << "). Good luck!" << std::endl;
        
        bool running = true;
        
        while (running && game.winner == -1) {
            // Render current state
            render(game, humanPlayer);
            
            if (game.getPawnIndexOfTurn() == humanPlayer) {
                // Human player's turn
                auto move = handleInput(game, humanPlayer);
                
                if (move) {
                    // Check for quit signal
                    if (std::holds_alternative<PawnMove>(*move)) {
                        auto pm = std::get<PawnMove>(*move);
                        if (pm.row == -1) {
                            running = false;
                            break;
                        }
                    }
                    
                    // Try to apply the move
                    if (game.doMove(*move, true)) {
                        std::cout << "Human played: " << moveToString(*move) << std::endl;
                    }
                }
                
                SDL_Delay(16);  // ~60 FPS
            } else {
                // AI's turn
                auto [aiMove, winRate] = ai.chooseNextMove(game);
                game.doMove(aiMove);
                std::cout << "AI played: " << moveToString(aiMove) << " (win rate: " << winRate << ")" << std::endl;
                SDL_Delay(500);  // Brief pause to show AI move
            }
        }
        
        // Show final state
        if (running) {
            render(game, humanPlayer);
            std::cout << "\nGame Over! Player " << game.winner << " wins!" << std::endl;
            std::cout << "Press ESC or close window to exit..." << std::endl;
            
            // Wait for user to close window
            bool waiting = true;
            while (waiting) {
                SDL_Event event;
                while (SDL_PollEvent(&event)) {
                    if (event.type == SDL_QUIT ||
                        (event.type == SDL_KEYDOWN && 
                         (event.key.keysym.sym == SDLK_ESCAPE || event.key.keysym.sym == SDLK_q))) {
                        waiting = false;
                    }
                }
                render(game, humanPlayer);
                SDL_Delay(16);
            }
        }
    }
    
private:
    bool init() {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL init failed: " << SDL_GetError() << std::endl;
            return false;
        }
        
        if (TTF_Init() < 0) {
            std::cerr << "SDL_ttf init failed: " << TTF_GetError() << std::endl;
            return false;
        }
        
        window = SDL_CreateWindow(
            "Quoridor AI",
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            SDL_WINDOW_SHOWN
        );
        
        if (!window) {
            std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
            return false;
        }
        
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (!renderer) {
            std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
            return false;
        }
        
        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
        
        // Try to load system fonts
        const char* fontPaths[] = {
            "/System/Library/Fonts/Helvetica.ttc",           // macOS
            "/System/Library/Fonts/SFNS.ttf",                // macOS San Francisco
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", // Linux
            "/usr/share/fonts/TTF/DejaVuSans.ttf"            // Linux alt
        };
        
        for (const char* fontPath : fontPaths) {
            font = TTF_OpenFont(fontPath, 16);
            if (font) {
                fontLarge = TTF_OpenFont(fontPath, 24);
                break;
            }
        }
        
        if (!font) {
            std::cerr << "Warning: Could not load font. Text will not display.\n";
            std::cerr << "TTF Error: " << TTF_GetError() << std::endl;
        }
        
        initialized = true;
        return true;
    }
    
    void cleanup() {
        if (fontLarge) {
            TTF_CloseFont(fontLarge);
            fontLarge = nullptr;
        }
        if (font) {
            TTF_CloseFont(font);
            font = nullptr;
        }
        if (renderer) {
            SDL_DestroyRenderer(renderer);
            renderer = nullptr;
        }
        if (window) {
            SDL_DestroyWindow(window);
            window = nullptr;
        }
        if (initialized) {
            TTF_Quit();
            SDL_Quit();
            initialized = false;
        }
    }
    
    // Get cell from screen coordinates
    std::optional<std::pair<int, int>> getCellFromScreen(int x, int y) {
        int col = (x - BOARD_OFFSET_X) / CELL_SIZE;
        int row = (y - BOARD_OFFSET_Y) / CELL_SIZE;
        
        if (row >= 0 && row < 9 && col >= 0 && col < 9) {
            return std::make_pair(row, col);
        }
        return std::nullopt;
    }
    
    // Get wall position from screen coordinates
    std::optional<std::pair<int, int>> getWallFromScreen(int x, int y, bool horizontal) {
        if (horizontal) {
            // Horizontal wall detection - near bottom edge of cells
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    int wallX = BOARD_OFFSET_X + c * CELL_SIZE;
                    int wallY = BOARD_OFFSET_Y + (r + 1) * CELL_SIZE - WALL_THICKNESS / 2;
                    int wallW = CELL_SIZE * 2;
                    int wallH = WALL_THICKNESS;
                    
                    if (x >= wallX && x < wallX + wallW &&
                        y >= wallY && y < wallY + wallH) {
                        return std::make_pair(r, c);
                    }
                }
            }
        } else {
            // Vertical wall detection - near right edge of cells
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    int wallX = BOARD_OFFSET_X + (c + 1) * CELL_SIZE - WALL_THICKNESS / 2;
                    int wallY = BOARD_OFFSET_Y + r * CELL_SIZE;
                    int wallW = WALL_THICKNESS;
                    int wallH = CELL_SIZE * 2;
                    
                    if (x >= wallX && x < wallX + wallW &&
                        y >= wallY && y < wallY + wallH) {
                        return std::make_pair(r, c);
                    }
                }
            }
        }
        return std::nullopt;
    }
    
    // Check if point is in rectangle
    bool pointInRect(int x, int y, const SDL_Rect& rect) {
        return x >= rect.x && x < rect.x + rect.w &&
               y >= rect.y && y < rect.y + rect.h;
    }
    
    // Set color for drawing
    void setColor(const Color& c) {
        SDL_SetRenderDrawColor(renderer, c.r, c.g, c.b, c.a);
    }
    
    // Draw filled rectangle
    void drawFilledRect(int x, int y, int w, int h, const Color& color) {
        setColor(color);
        SDL_Rect rect = {x, y, w, h};
        SDL_RenderFillRect(renderer, &rect);
    }
    
    // Draw filled circle
    void drawFilledCircle(int cx, int cy, int radius, const Color& color) {
        setColor(color);
        for (int y = -radius; y <= radius; y++) {
            for (int x = -radius; x <= radius; x++) {
                if (x*x + y*y <= radius*radius) {
                    SDL_RenderDrawPoint(renderer, cx + x, cy + y);
                }
            }
        }
    }
    
    // Draw text
    void drawText(const std::string& text, int x, int y, const Color& color, bool large = false) {
        if (!font) return;
        
        TTF_Font* useFont = large && fontLarge ? fontLarge : font;
        SDL_Color sdlColor = {color.r, color.g, color.b, color.a};
        SDL_Surface* surface = TTF_RenderText_Blended(useFont, text.c_str(), sdlColor);
        if (!surface) return;
        
        SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
        if (texture) {
            SDL_Rect dstRect = {x, y, surface->w, surface->h};
            SDL_RenderCopy(renderer, texture, nullptr, &dstRect);
            SDL_DestroyTexture(texture);
        }
        SDL_FreeSurface(surface);
    }
    
    // Draw centered text
    void drawTextCentered(const std::string& text, int centerX, int centerY, const Color& color, bool large = false) {
        if (!font) return;
        
        TTF_Font* useFont = large && fontLarge ? fontLarge : font;
        int w, h;
        TTF_SizeText(useFont, text.c_str(), &w, &h);
        drawText(text, centerX - w/2, centerY - h/2, color, large);
    }
    
    // Draw button
    void drawButton(const SDL_Rect& rect, const std::string& label, bool selected, bool hovered) {
        Color color = selected ? COLOR_BUTTON_HOVER : (hovered ? COLOR_BUTTON_HOVER : COLOR_BUTTON);
        drawFilledRect(rect.x, rect.y, rect.w, rect.h, color);
        
        // Border
        setColor(selected ? COLOR_TEXT : COLOR_GRID);
        SDL_Rect borderRect = rect;
        if (selected) {
            borderRect.x -= 2;
            borderRect.y -= 2;
            borderRect.w += 4;
            borderRect.h += 4;
            SDL_RenderDrawRect(renderer, &borderRect);
        }
        SDL_RenderDrawRect(renderer, &rect);
        
        // Draw text label
        drawTextCentered(label, rect.x + rect.w/2, rect.y + rect.h/2, COLOR_TEXT);
    }
    
    // Main render function
    void render(const Game& game, int humanPlayer, const std::string& message = "") {
        if (!initialized) return;
        
        // Clear screen
        setColor(COLOR_BG);
        SDL_RenderClear(renderer);
        
        // Draw title area
        drawFilledRect(0, 0, WINDOW_WIDTH, 90, {30, 30, 30, 255});
        
        // Draw title
        drawTextCentered("QUORIDOR AI", WINDOW_WIDTH / 2, 15, COLOR_TEXT, true);
        
        // Draw mode buttons
        int mouseX, mouseY;
        SDL_GetMouseState(&mouseX, &mouseY);
        
        drawButton(btnMovePawn, "MOVE PAWN", currentMode == InputMode::MOVE_PAWN, 
                   pointInRect(mouseX, mouseY, btnMovePawn));
        drawButton(btnHWall, "HORIZONTAL WALL", currentMode == InputMode::PLACE_H_WALL,
                   pointInRect(mouseX, mouseY, btnHWall));
        drawButton(btnVWall, "VERTICAL WALL", currentMode == InputMode::PLACE_V_WALL,
                   pointInRect(mouseX, mouseY, btnVWall));
        
        // Draw board background
        drawFilledRect(BOARD_OFFSET_X - 10, BOARD_OFFSET_Y - 10, 
                       CELL_SIZE * 9 + 20, CELL_SIZE * 9 + 20, COLOR_BOARD);
        
        // Draw goal rows
        int p0GoalRow = game.board.pawns[0].goalRow;
        int p1GoalRow = game.board.pawns[1].goalRow;
        
        drawFilledRect(BOARD_OFFSET_X, BOARD_OFFSET_Y + p0GoalRow * CELL_SIZE,
                       CELL_SIZE * 9, CELL_SIZE, 
                       {COLOR_PLAYER0.r, COLOR_PLAYER0.g, COLOR_PLAYER0.b, 50});
        drawFilledRect(BOARD_OFFSET_X, BOARD_OFFSET_Y + p1GoalRow * CELL_SIZE,
                       CELL_SIZE * 9, CELL_SIZE,
                       {COLOR_PLAYER1.r, COLOR_PLAYER1.g, COLOR_PLAYER1.b, 50});
        
        // Draw cells
        for (int r = 0; r < 9; r++) {
            for (int c = 0; c < 9; c++) {
                int x = BOARD_OFFSET_X + c * CELL_SIZE;
                int y = BOARD_OFFSET_Y + r * CELL_SIZE;
                drawFilledRect(x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4, COLOR_CELL);
            }
        }
        
        // Draw valid moves for human player
        if (game.winner == -1 && game.getPawnIndexOfTurn() == humanPlayer) {
            if (currentMode == InputMode::MOVE_PAWN) {
                // Highlight valid pawn moves
                auto& validPos = const_cast<Game&>(game).getValidNextPositions();
                for (int r = 0; r < 9; r++) {
                    for (int c = 0; c < 9; c++) {
                        if (validPos[r][c]) {
                            int x = BOARD_OFFSET_X + c * CELL_SIZE;
                            int y = BOARD_OFFSET_Y + r * CELL_SIZE;
                            drawFilledRect(x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4, COLOR_VALID_MOVE);
                        }
                    }
                }
            }
        }
        
        // Draw hover effect
        if (hoveredCell) {
            auto [r, c] = *hoveredCell;
            int x = BOARD_OFFSET_X + c * CELL_SIZE;
            int y = BOARD_OFFSET_Y + r * CELL_SIZE;
            drawFilledRect(x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4, COLOR_HOVER);
        }
        
        // Draw walls
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                // Horizontal walls
                if (game.board.horizontalWalls[r][c]) {
                    int x = BOARD_OFFSET_X + c * CELL_SIZE;
                    int y = BOARD_OFFSET_Y + (r + 1) * CELL_SIZE - WALL_THICKNESS / 2;
                    drawFilledRect(x, y, CELL_SIZE * 2, WALL_THICKNESS, COLOR_WALL);
                }
                
                // Vertical walls
                if (game.board.verticalWalls[r][c]) {
                    int x = BOARD_OFFSET_X + (c + 1) * CELL_SIZE - WALL_THICKNESS / 2;
                    int y = BOARD_OFFSET_Y + r * CELL_SIZE;
                    drawFilledRect(x, y, WALL_THICKNESS, CELL_SIZE * 2, COLOR_WALL);
                }
            }
        }
        
        // Draw wall preview
        if (hoveredWall && wallPreviewValid && game.winner == -1 && 
            game.getPawnIndexOfTurn() == humanPlayer) {
            auto [r, c] = *hoveredWall;
            Color previewColor = COLOR_VALID_MOVE;
            
            if (currentMode == InputMode::PLACE_H_WALL) {
                int x = BOARD_OFFSET_X + c * CELL_SIZE;
                int y = BOARD_OFFSET_Y + (r + 1) * CELL_SIZE - WALL_THICKNESS / 2;
                drawFilledRect(x, y, CELL_SIZE * 2, WALL_THICKNESS, previewColor);
            } else if (currentMode == InputMode::PLACE_V_WALL) {
                int x = BOARD_OFFSET_X + (c + 1) * CELL_SIZE - WALL_THICKNESS / 2;
                int y = BOARD_OFFSET_Y + r * CELL_SIZE;
                drawFilledRect(x, y, WALL_THICKNESS, CELL_SIZE * 2, previewColor);
            }
        }
        
        // Draw pawns
        for (int p = 0; p < 2; p++) {
            const auto& pawn = game.board.pawns[p];
            Color pColor = (p == 0) ? COLOR_PLAYER0 : COLOR_PLAYER1;
            
            int x = BOARD_OFFSET_X + pawn.position.col * CELL_SIZE + CELL_SIZE / 2;
            int y = BOARD_OFFSET_Y + pawn.position.row * CELL_SIZE + CELL_SIZE / 2;
            
            drawFilledCircle(x, y, CELL_SIZE / 3, pColor);
            
            // Draw border
            setColor(COLOR_TEXT);
            for (int angle = 0; angle < 360; angle += 10) {
                float rad = angle * 3.14159f / 180.0f;
                int bx = x + static_cast<int>(std::cos(rad) * CELL_SIZE / 3);
                int by = y + static_cast<int>(std::sin(rad) * CELL_SIZE / 3);
                SDL_RenderDrawPoint(renderer, bx, by);
            }
        }
        
        // Draw grid lines
        setColor(COLOR_GRID);
        for (int i = 0; i <= 9; i++) {
            // Vertical lines
            int x = BOARD_OFFSET_X + i * CELL_SIZE;
            SDL_RenderDrawLine(renderer, x, BOARD_OFFSET_Y, x, BOARD_OFFSET_Y + 9 * CELL_SIZE);
            
            // Horizontal lines
            int y = BOARD_OFFSET_Y + i * CELL_SIZE;
            SDL_RenderDrawLine(renderer, BOARD_OFFSET_X, y, BOARD_OFFSET_X + 9 * CELL_SIZE, y);
        }
        
        // Draw info panel at bottom
        int infoY = BOARD_OFFSET_Y + 9 * CELL_SIZE + 20;
        
        // Player info boxes
        for (int p = 0; p < 2; p++) {
            int boxX = 50 + p * 350;
            Color pColor = (p == 0) ? COLOR_PLAYER0 : COLOR_PLAYER1;
            
            // Box background
            bool isCurrentTurn = (game.getPawnIndexOfTurn() == p && game.winner == -1);
            Color bgColor = isCurrentTurn ? Color{pColor.r, pColor.g, pColor.b, 100} : Color{60, 60, 60, 255};
            drawFilledRect(boxX, infoY, 300, 80, bgColor);
            
            // Border
            setColor(isCurrentTurn ? COLOR_VALID_MOVE : COLOR_GRID);
            SDL_Rect boxRect = {boxX, infoY, 300, 80};
            SDL_RenderDrawRect(renderer, &boxRect);
            
            // Player indicator circle
            drawFilledCircle(boxX + 30, infoY + 40, 20, pColor);
            
            // Player name
            std::string playerName = "Player " + std::to_string(p);
            if (p == humanPlayer) playerName += " (You)";
            drawText(playerName, boxX + 65, infoY + 15, COLOR_TEXT);
            
            // Walls remaining text
            std::string wallText = "Walls: " + std::to_string(game.board.pawns[p].wallsLeft);
            drawText(wallText, boxX + 65, infoY + 40, COLOR_TEXT);
            
            // Goal row
            std::string goalText = "Goal: Row " + std::to_string(game.board.pawns[p].goalRow);
            drawText(goalText, boxX + 65, infoY + 60, COLOR_TEXT);
            
            // Turn indicator
            if (isCurrentTurn) {
                drawText("◄ TURN", boxX + 220, infoY + 30, COLOR_VALID_MOVE);
            }
        }
        
        // Winner banner
        if (game.winner != -1) {
            // Semi-transparent overlay
            drawFilledRect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, {0, 0, 0, 180});
            
            // Winner box
            int boxW = 400, boxH = 150;
            int boxX = WINDOW_WIDTH / 2 - boxW / 2;
            int boxY = WINDOW_HEIGHT / 2 - boxH / 2;
            drawFilledRect(boxX, boxY, boxW, boxH, 
                          game.winner == 0 ? COLOR_PLAYER0 : COLOR_PLAYER1);
            
            // Border
            setColor(COLOR_TEXT);
            SDL_Rect winRect = {boxX, boxY, boxW, boxH};
            SDL_RenderDrawRect(renderer, &winRect);
            
            // Winner text
            std::string winText = "PLAYER " + std::to_string(game.winner) + " WINS!";
            drawTextCentered(winText, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2 - 20, COLOR_TEXT, true);
            
            std::string subText = (game.winner == humanPlayer) ? "Congratulations!" : "Better luck next time!";
            drawTextCentered(subText, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2 + 20, COLOR_TEXT);
        }
        
        SDL_RenderPresent(renderer);
    }
    
    // Handle input and return move if one was made
    std::optional<Move> handleInput(Game& game, int humanPlayer) {
        SDL_Event event;
        
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    return PawnMove(-1, -1);  // Signal to quit
                    
                case SDL_MOUSEMOTION: {
                    int x = event.motion.x;
                    int y = event.motion.y;
                    
                    if (currentMode == InputMode::MOVE_PAWN) {
                        hoveredCell = getCellFromScreen(x, y);
                        hoveredWall = std::nullopt;
                    } else if (currentMode == InputMode::PLACE_H_WALL) {
                        hoveredCell = std::nullopt;
                        auto wall = getWallFromScreen(x, y, true);
                        if (wall) {
                            auto [r, c] = *wall;
                            // Just show preview, validation happens on click
                            wallPreviewValid = (r < 8 && c < 8);
                            hoveredWall = wall;
                        } else {
                            hoveredWall = std::nullopt;
                        }
                    } else if (currentMode == InputMode::PLACE_V_WALL) {
                        hoveredCell = std::nullopt;
                        auto wall = getWallFromScreen(x, y, false);
                        if (wall) {
                            auto [r, c] = *wall;
                            // Just show preview, validation happens on click
                            wallPreviewValid = (r < 8 && c < 8);
                            hoveredWall = wall;
                        } else {
                            hoveredWall = std::nullopt;
                        }
                    }
                    break;
                }
                    
                case SDL_MOUSEBUTTONDOWN: {
                    if (event.button.button == SDL_BUTTON_LEFT) {
                        int x = event.button.x;
                        int y = event.button.y;
                        
                        // Check button clicks
                        if (pointInRect(x, y, btnMovePawn)) {
                            currentMode = InputMode::MOVE_PAWN;
                            hoveredWall = std::nullopt;
                            return std::nullopt;
                        }
                        if (pointInRect(x, y, btnHWall)) {
                            currentMode = InputMode::PLACE_H_WALL;
                            hoveredCell = std::nullopt;
                            return std::nullopt;
                        }
                        if (pointInRect(x, y, btnVWall)) {
                            currentMode = InputMode::PLACE_V_WALL;
                            hoveredCell = std::nullopt;
                            return std::nullopt;
                        }
                        
                        // Only allow input if it's human player's turn
                        if (game.getPawnIndexOfTurn() != humanPlayer || game.winner != -1) {
                            return std::nullopt;
                        }
                        
                        // Handle board clicks
                        if (currentMode == InputMode::MOVE_PAWN) {
                            auto cell = getCellFromScreen(x, y);
                            if (cell) {
                                auto [r, c] = *cell;
                                return PawnMove(r, c);
                            }
                        } else if (currentMode == InputMode::PLACE_H_WALL) {
                            auto wall = getWallFromScreen(x, y, true);
                            if (wall) {
                                auto [r, c] = *wall;
                                return HorizontalWall(r, c);
                            }
                        } else if (currentMode == InputMode::PLACE_V_WALL) {
                            auto wall = getWallFromScreen(x, y, false);
                            if (wall) {
                                auto [r, c] = *wall;
                                return VerticalWall(r, c);
                            }
                        }
                    }
                    break;
                }
                    
                case SDL_KEYDOWN: {
                    switch (event.key.keysym.sym) {
                        case SDLK_m:
                            currentMode = InputMode::MOVE_PAWN;
                            hoveredWall = std::nullopt;
                            break;
                        case SDLK_h:
                            currentMode = InputMode::PLACE_H_WALL;
                            hoveredCell = std::nullopt;
                            break;
                        case SDLK_v:
                            currentMode = InputMode::PLACE_V_WALL;
                            hoveredCell = std::nullopt;
                            break;
                        case SDLK_ESCAPE:
                        case SDLK_q:
                            return PawnMove(-1, -1);  // Signal to quit
                    }
                    break;
                }
            }
        }
        
        return std::nullopt;
    }
    
    std::string moveToString(const Move& move) {
        if (std::holds_alternative<PawnMove>(move)) {
            auto m = std::get<PawnMove>(move);
            return "Pawn(" + std::to_string(m.row) + "," + std::to_string(m.col) + ")";
        } else if (std::holds_alternative<HorizontalWall>(move)) {
            auto m = std::get<HorizontalWall>(move);
            return "HWall(" + std::to_string(m.row) + "," + std::to_string(m.col) + ")";
        } else {
            auto m = std::get<VerticalWall>(move);
            return "VWall(" + std::to_string(m.row) + "," + std::to_string(m.col) + ")";
        }
    }
};

#endif // USE_GUI
