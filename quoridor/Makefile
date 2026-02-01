# Quoridor AI Makefile
# Based on gorisanson/quoridor-ai

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
DEBUG_FLAGS = -std=c++17 -g -Wall -Wextra -DDEBUG

# SDL2 flags (for GUI mode)
SDL2_CFLAGS = $(shell sdl2-config --cflags 2>/dev/null || echo "")
SDL2_LIBS = $(shell sdl2-config --libs 2>/dev/null || echo "")

TARGET = quoridor
TARGET_GUI = quoridor_gui
SRCS = main.cpp
HEADERS = Position.hpp Pawn.hpp Board.hpp Move.hpp Game.hpp AI.hpp MNode.hpp MCTS.hpp QuoridorAI.hpp Visualization.hpp GUIVisualization.hpp

.PHONY: all clean debug gui run run-human run-gui run-train run-benchmark help

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

# GUI version with SDL2
gui: $(SRCS) $(HEADERS)
	@if [ -z "$(SDL2_LIBS)" ]; then \
		echo "SDL2 not found. Please install SDL2:"; \
		echo "  macOS: brew install sdl2 sdl2_ttf"; \
		echo "  Linux: sudo apt-get install libsdl2-dev libsdl2-ttf-dev"; \
		exit 1; \
	fi
	$(CXX) $(CXXFLAGS) -DUSE_GUI $(SDL2_CFLAGS) -o $(TARGET_GUI) $(SRCS) $(SDL2_LIBS) -lSDL2_ttf
	@echo "GUI version built successfully: $(TARGET_GUI)"

debug: $(SRCS) $(HEADERS)
	$(CXX) $(DEBUG_FLAGS) -o $(TARGET)_debug $(SRCS)

clean:
	rm -f $(TARGET) $(TARGET_GUI) $(TARGET)_debug

# Run targets
run: $(TARGET)
	./$(TARGET)

run-human: $(TARGET)
	./$(TARGET) --human

run-gui: gui
	./$(TARGET_GUI) --gui

run-train: $(TARGET)
	./$(TARGET) --train

run-benchmark: $(TARGET)
	./$(TARGET) --benchmark

# Help
help:
	@echo "Quoridor AI Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build terminal version (default)"
	@echo "  gui           - Build GUI version (requires SDL2)"
	@echo "  debug         - Build debug version"
	@echo "  clean         - Remove built files"
	@echo ""
	@echo "Run targets:"
	@echo "  run           - AI vs AI game"
	@echo "  run-human     - Human vs AI (terminal)"
	@echo "  run-gui       - Human vs AI (graphical window)"
	@echo "  run-train     - Training mode"
	@echo "  run-benchmark - Benchmark mode"
	@echo ""
	@echo "GUI Prerequisites:"
	@echo "  macOS: brew install sdl2"
	@echo "  Linux: sudo apt-get install libsdl2-dev"
