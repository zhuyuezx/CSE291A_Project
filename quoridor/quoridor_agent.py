"""
Quoridor Agent Core Module

This module encapsulates all core agent logic:
- ExperienceManager: Episodic memory with JSON storage
- LLMEvaluator: Transformer-based board evaluation
- HeuristicEvaluator: Fallback evaluator
- MCTSNode: MCTS tree node
- LLM_MCTS_Bot: Hybrid LLM + MCTS agent
- RandomBot: Baseline random agent
- Utility functions for state conversion

All other scripts (human_vs_ai.py, train_evolution.py) import from this module.

Author: CSE291A Research Project
"""

from open_spiel.python import games  # noqa: F401 - registers games
import pyspiel
import numpy as np
import math
import random
import sqlite3
import hashlib
import os
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================================
# Configuration
# ==============================================================================

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MAX_NEW_TOKENS = 32

# Unified memory configuration
DEFAULT_MEMORY_DB = "quoridor_memory.db"
MAX_RETRIEVED_MEMORIES = 5


# ==============================================================================
# Experience Manager (Episodic Memory with SQLite)
# ==============================================================================

class ExperienceManager:
    """
    Manages long-term memory storage using SQLite database.
    
    Features:
    - O(1) inserts (no full file rewrite)
    - Indexed lookups by board hash for fast similarity matching
    - Persistent storage with ACID guarantees
    - Memory-efficient (no need to load all records)
    
    For vector similarity search, consider upgrading to:
    - ChromaDB: pip install chromadb
    - FAISS: pip install faiss-cpu
    - sqlite-vec: SQLite extension for vector search
    """
    
    _instances: Dict[str, 'ExperienceManager'] = {}
    
    @classmethod
    def get_instance(cls, memory_file: str = DEFAULT_MEMORY_DB, 
                     player_id: int = 0) -> 'ExperienceManager':
        """
        Get or create a singleton instance for the given database file.
        Ensures unified memory access across all scripts.
        """
        # Normalize file extension
        if memory_file.endswith('.json'):
            memory_file = memory_file.replace('.json', '.db')
        
        if memory_file not in cls._instances:
            cls._instances[memory_file] = cls(memory_file, player_id)
        return cls._instances[memory_file]
    
    def __init__(self, memory_file: str = DEFAULT_MEMORY_DB, player_id: int = 0):
        """
        Initialize the experience manager with SQLite backend.
        
        Args:
            memory_file: Path to the SQLite database file
            player_id: ID of the player this manager tracks (0 or 1)
        """
        # Normalize extension
        if memory_file.endswith('.json'):
            memory_file = memory_file.replace('.json', '.db')
        
        self.memory_file = memory_file
        self.player_id = player_id
        self.current_game_buffer: List[Dict[str, Any]] = []
        
        # Initialize database
        self._init_db()
        
        count = self._get_count()
        print(f"[Memory] Connected to SQLite DB with {count} experiences")
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.memory_file)
        cursor = conn.cursor()
        
        # Create experiences table with indexes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                board_hash TEXT NOT NULL,
                p1_row INTEGER,
                p2_row INTEGER,
                p1_walls INTEGER,
                p2_walls INTEGER,
                state_text TEXT NOT NULL,
                mcts_confidence REAL NOT NULL,
                final_outcome REAL NOT NULL,
                player_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for fast lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_board_hash ON experiences(board_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions ON experiences(p1_row, p2_row)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_outcome ON experiences(final_outcome)')
        
        conn.commit()
        conn.close()
    
    def _get_count(self) -> int:
        """Get total number of experiences."""
        conn = sqlite3.connect(self.memory_file)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM experiences')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def _compute_board_hash(self, state_text: str) -> str:
        """
        Compute a hash signature of the board state.
        Uses only the grid portion for position-based matching.
        """
        board_lines = self._extract_board_lines(state_text)
        return hashlib.md5(board_lines.encode()).hexdigest()[:16]
    
    def _extract_board_lines(self, state_text: str) -> str:
        """Extract only grid lines from state text."""
        lines = state_text.split('\n')
        board_lines = [l for l in lines if '.' in l or '@' in l or '0' in l or '---' in l or '|' in l]
        return '\n'.join(board_lines)
    
    def _extract_positions(self, state_text: str) -> Tuple[int, int, int, int]:
        """Extract player positions and wall counts from state text."""
        p1_row, p2_row = 0, 8
        p1_walls, p2_walls = 10, 10
        
        lines = state_text.split('\n')
        for i, line in enumerate(lines):
            if '@' in line:
                p1_row = i // 2
            if '0' in line:
                p2_row = i // 2
            if 'walls:' in line.lower():
                try:
                    parts = line.split('walls:')[1].strip().split(',')
                    if len(parts) >= 2:
                        p1_walls = int(parts[0].strip())
                        p2_walls = int(parts[1].strip())
                except:
                    pass
        
        return p1_row, p2_row, p1_walls, p2_walls
    
    def record_step(self, state_text: str, mcts_confidence: float):
        """Record a step during gameplay (buffered until game ends)."""
        step = {
            "state_text": state_text,
            "mcts_confidence": mcts_confidence,
            "player_id": self.player_id,
            "board_hash": self._compute_board_hash(state_text),
            "positions": self._extract_positions(state_text),
        }
        self.current_game_buffer.append(step)
    
    def end_game(self, final_outcome: float):
        """
        Commit all buffered steps to database with final outcome.
        
        KEY INSIGHT: Quoridor is symmetric! We record experiences for BOTH perspectives:
        - Player 0's perspective: original outcome
        - Player 1's perspective: negated outcome (what's good for P0 is bad for P1)
        
        This doubles our training data and makes memory useful for both players.
        """
        if not self.current_game_buffer:
            return
        
        conn = sqlite3.connect(self.memory_file)
        cursor = conn.cursor()
        
        # Record from the current player's perspective
        for step in self.current_game_buffer:
            p1_row, p2_row, p1_walls, p2_walls = step["positions"]
            cursor.execute('''
                INSERT INTO experiences 
                (board_hash, p1_row, p2_row, p1_walls, p2_walls, 
                 state_text, mcts_confidence, final_outcome, player_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                step["board_hash"],
                p1_row, p2_row, p1_walls, p2_walls,
                step["state_text"],
                step["mcts_confidence"],
                final_outcome,
                step["player_id"]
            ))
            
            # Also record the OPPOSITE perspective
            # (same board state but outcome is inverted)
            opposite_outcome = -final_outcome
            opposite_player = 1 - step["player_id"]
            cursor.execute('''
                INSERT INTO experiences 
                (board_hash, p1_row, p2_row, p1_walls, p2_walls, 
                 state_text, mcts_confidence, final_outcome, player_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                step["board_hash"],
                p1_row, p2_row, p1_walls, p2_walls,
                step["state_text"],
                step["mcts_confidence"],
                opposite_outcome,
                opposite_player
            ))
        
        conn.commit()
        conn.close()
        
        print(f"[Memory] Committed {len(self.current_game_buffer) * 2} steps (both perspectives) with outcome: ±{abs(final_outcome)}")
        self.current_game_buffer = []
    
    def clear_game_buffer(self):
        """Clear the current game buffer without saving."""
        self.current_game_buffer = []
    
    def retrieve_similar_states(self, current_state_text: str, 
                                 top_k: int = MAX_RETRIEVED_MEMORIES,
                                 player_id: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve similar past states using indexed lookups.
        
        Uses a two-stage approach:
        1. Fast: Exact hash match (O(1) with index)
        2. Fallback: Position-based similarity (O(log n) with index)
        
        Args:
            current_state_text: The current board state as text
            top_k: Maximum number of results to return
            player_id: If specified, only return experiences from this player's perspective
        """
        conn = sqlite3.connect(self.memory_file)
        cursor = conn.cursor()
        
        results = []
        
        # Build player filter if specified
        player_filter = ""
        player_params = ()
        if player_id is not None:
            player_filter = "AND player_id = ?"
            player_params = (player_id,)
        
        # Stage 1: Exact board hash match
        board_hash = self._compute_board_hash(current_state_text)
        cursor.execute(f'''
            SELECT state_text, mcts_confidence, final_outcome, player_id
            FROM experiences
            WHERE board_hash = ? {player_filter}
            ORDER BY created_at DESC
            LIMIT ?
        ''', (board_hash,) + player_params + (top_k,))
        
        for row in cursor.fetchall():
            results.append({
                "state_text": row[0],
                "mcts_confidence": row[1],
                "final_outcome": row[2],
                "player_id": row[3],
                "similarity": 1.0  # Exact match
            })
        
        # Stage 2: If not enough exact matches, find position-similar states
        if len(results) < top_k:
            p1_row, p2_row, _, _ = self._extract_positions(current_state_text)
            
            # Find states with similar player positions (within 1 row)
            cursor.execute(f'''
                SELECT state_text, mcts_confidence, final_outcome, player_id,
                       ABS(p1_row - ?) + ABS(p2_row - ?) as distance
                FROM experiences
                WHERE board_hash != ?
                  AND ABS(p1_row - ?) <= 1
                  AND ABS(p2_row - ?) <= 1
                  {player_filter}
                ORDER BY distance ASC, created_at DESC
                LIMIT ?
            ''', (p1_row, p2_row, board_hash, p1_row, p2_row) + player_params + (top_k - len(results),))
            
            for row in cursor.fetchall():
                distance = row[4]
                # Convert distance to similarity score (0 distance = 0.9, 2 distance = 0.7)
                similarity = max(0.7, 0.9 - distance * 0.1)
                results.append({
                    "state_text": row[0],
                    "mcts_confidence": row[1],
                    "final_outcome": row[2],
                    "player_id": row[3],
                    "similarity": similarity
                })
        
        conn.close()
        return results
    
    def get_memory_context(self, current_state_text: str) -> str:
        """Generate memory context string for LLM prompt injection."""
        similar_states = self.retrieve_similar_states(current_state_text)
        
        if not similar_states:
            return ""
        
        total_memories = len(similar_states)
        wins = sum(1 for m in similar_states if m.get("final_outcome", 0) > 0)
        avg_outcome = np.mean([m.get("final_outcome", 0) for m in similar_states])
        avg_confidence = np.mean([m.get("mcts_confidence", 0) for m in similar_states])
        
        win_rate = (wins / total_memories) * 100 if total_memories > 0 else 0
        
        if win_rate >= 70:
            advice = "CONFIDENT - This type of position historically leads to wins."
        elif win_rate >= 40:
            advice = "NEUTRAL - Mixed historical results. Play carefully."
        else:
            advice = "CAUTIOUS - This type of position historically leads to losses."
        
        return f"""
MEMORY CONTEXT:
- Similar positions seen: {total_memories} times
- Historical Win Rate: {win_rate:.0f}%
- Average past MCTS confidence: {avg_confidence:.2f}
- Average outcome: {avg_outcome:.2f}
- ADVICE: {advice}"""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics using efficient SQL aggregation."""
        conn = sqlite3.connect(self.memory_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN final_outcome > 0 THEN 1 ELSE 0 END) as wins,
                AVG(final_outcome) as avg_outcome
            FROM experiences
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row[0] == 0:
            return {"total": 0, "win_rate": 0, "avg_outcome": 0}
        
        return {
            "total": row[0],
            "win_rate": (row[1] / row[0]) * 100 if row[0] > 0 else 0,
            "avg_outcome": row[2] or 0,
        }
    
    def get_recent_losses(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent losing experiences for analysis."""
        conn = sqlite3.connect(self.memory_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT state_text, mcts_confidence, final_outcome
            FROM experiences
            WHERE final_outcome < 0
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "state_text": row[0],
                "mcts_confidence": row[1],
                "final_outcome": row[2],
            })
        
        conn.close()
        return results


# ==============================================================================
# LLM Evaluator
# ==============================================================================

class LLMEvaluator:
    """
    Local transformer-based evaluator for Quoridor with episodic memory.
    """
    
    BASE_SYSTEM_PROMPT = """You are an expert Quoridor game analyst. Evaluate board positions and output a score.

QUORIDOR RULES:
- Two players race to reach the opposite side of a 9x9 board
- Player 1 (marked @) starts at row 1 and must reach row 9
- Player 2 (marked 0) starts at row 9 and must reach row 1
- Each turn: MOVE one step (N/S/E/W) or place a WALL
- Walls are 2-square barriers blocking movement (shown as ---+--- or |)
- Each player has 10 walls maximum
- CRITICAL: Cannot completely block opponent's path to goal

STRATEGIC PRINCIPLES:
1. **Path Length**: Shorter path = winning. Compare your path vs opponent's path.
2. **Wall Economy**: Walls are precious. Each placement should force opponent 2+ extra steps.
3. **Positional Advantage**: Center control = more movement options.
4. **Tempo**: When ahead in path, push forward. When behind, wall strategically.

OUTPUT FORMAT:
Output ONLY a single number between -1.0 and 1.0:
- Positive = Player 1 is winning
- Negative = Player 2 is winning
- Magnitude = confidence (0.0 = even, 1.0 = decisive)

Example outputs: 0.3, -0.5, 0.8, -0.2"""

    def __init__(self, device: Optional[str] = None, 
                 experience_manager: Optional[ExperienceManager] = None,
                 strategy_text: str = ""):
        """
        Initialize the LLM evaluator.
        
        Args:
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
            experience_manager: ExperienceManager for memory retrieval
            strategy_text: Additional strategy text to inject into prompt
        """
        print(f"[LLM] Loading model: {MODEL_ID}")
        
        self.experience_manager = experience_manager
        self.strategy_text = strategy_text
        
        # Determine best device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
                print("[LLM] Using MPS (Apple Silicon GPU)")
            elif torch.cuda.is_available():
                self.device = "cuda"
                print("[LLM] Using CUDA GPU")
            else:
                self.device = "cpu"
                print("[LLM] Using CPU")
        else:
            self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )
        
        # Load model
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        
        if self.device in ["mps", "cpu"]:
            self.model = self.model.to(self.device)
        
        print("[LLM] Model loaded successfully!")
    
    def _build_prompt_with_memory(self, state_text: str) -> str:
        """Build system prompt with memory context and strategy."""
        system_prompt = self.BASE_SYSTEM_PROMPT
        
        if self.strategy_text:
            system_prompt += f"\n\nSTRATEGY GUIDE:\n{self.strategy_text}"
        
        if self.experience_manager:
            memory_context = self.experience_manager.get_memory_context(state_text)
            if memory_context:
                system_prompt += memory_context
        
        return system_prompt
    
    def evaluate(self, state_text: str) -> float:
        """Evaluate a board position using the LLM."""
        system_prompt = self._build_prompt_with_memory(state_text)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Evaluate this position:\n\n{state_text}\n\nScore:"},
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_ids = output_ids[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        try:
            score = float(response.split()[0].replace(',', ''))
            score = np.clip(score, -1.0, 1.0)
        except (ValueError, IndexError):
            score = 0.0
        
        return float(score)


# ==============================================================================
# Heuristic Evaluator
# ==============================================================================

class HeuristicEvaluator:
    """
    Simple heuristic evaluator with optional memory integration.
    """
    
    def __init__(self, experience_manager: Optional[ExperienceManager] = None,
                 strategy_text: str = "", verbose: bool = False):
        self.experience_manager = experience_manager
        self.strategy_text = strategy_text
        self.strategy_biases = self._parse_strategy_biases(strategy_text)
        self.verbose = verbose
        self._memory_hit_count = 0
    
    def _parse_strategy_biases(self, text: str) -> Dict[str, float]:
        """Extract strategic biases from text."""
        biases = {}
        text_lower = text.lower()
        
        if "aggressive" in text_lower or "push forward" in text_lower:
            biases["aggression"] = 0.1
        if "defensive" in text_lower or "cautious" in text_lower:
            biases["defense"] = 0.1
        if "horizontal wall" in text_lower:
            biases["h_wall"] = 0.05
        if "vertical wall" in text_lower:
            biases["v_wall"] = 0.05
        
        return biases
    
    def evaluate(self, state_text: str, player_id: int = 0) -> float:
        """
        Evaluate using path length heuristic + memory bias.
        
        Returns score from the REQUESTING PLAYER's perspective:
        - Positive = good for requesting player
        - Negative = bad for requesting player
        
        Args:
            state_text: The current board state as text
            player_id: The player requesting evaluation (0 or 1)
        """
        heuristic_score = 0.0
        
        # Simple path-based heuristic:
        # - Player 0 controls '0' piece, measured by P2 path
        # - Player 1 controls '@' piece, measured by P1 path
        # - Shorter path = winning
        # - Score = (opponent_path - my_path) normalized
        
        p1_path, p2_path = 8, 8  # defaults
        
        if "P1 path:" in state_text and "P2 path:" in state_text:
            try:
                p1_idx = state_text.find("P1 path:") + 8
                p1_end = state_text.find(",", p1_idx)
                if p1_end == -1:
                    p1_end = state_text.find("\n", p1_idx)
                p1_path = int(state_text[p1_idx:p1_end].strip())
                
                p2_idx = state_text.find("P2 path:") + 8
                p2_end = state_text.find(",", p2_idx)
                if p2_end == -1:
                    p2_end = state_text.find("\n", p2_idx)
                if p2_end == -1:
                    p2_path = int(state_text[p2_idx:].strip())
                else:
                    p2_path = int(state_text[p2_idx:p2_end].strip())
                
            except (ValueError, IndexError):
                pass
        
        # Calculate score from requesting player's perspective
        # Positive = I'm ahead (my path shorter)
        if player_id == 0:
            # I control '0' (P2 path), opponent is '@' (P1 path)
            # Good for me = P2 path < P1 path = (P1 - P2) > 0
            path_advantage = p1_path - p2_path
        else:
            # I control '@' (P1 path), opponent is '0' (P2 path)
            # Good for me = P1 path < P2 path = (P2 - P1) > 0
            path_advantage = p2_path - p1_path
        
        # Normalize to [-1, 1] range
        heuristic_score = np.tanh(path_advantage * 0.4)
        
        score = heuristic_score
        
        # Memory bias - already in requesting player's perspective
        # (filtered by player_id, outcomes stored with correct signs)
        if self.experience_manager:
            similar_states = self.experience_manager.retrieve_similar_states(
                state_text, top_k=5, player_id=player_id
            )
            if similar_states:
                self._memory_hit_count += 1
                outcomes = [m.get("final_outcome", 0) for m in similar_states]
                avg_outcome = np.mean(outcomes)
                
                # Only apply memory bias if outcomes are meaningful
                outcome_variance = np.var(outcomes)
                if outcome_variance > 0.01 or abs(avg_outcome) > 0.1:
                    # Blend heuristic with memory - heuristic should dominate
                    memory_weight = 0.15  # Reduced from 0.4
                    score = (1 - memory_weight) * heuristic_score + memory_weight * avg_outcome
                    
                    if self.verbose and self._memory_hit_count % 50 == 1:
                        print(f"[Memory] P{player_id}: Found {len(similar_states)} states, "
                              f"avg={avg_outcome:.2f}, heur={heuristic_score:.2f}, final={score:.2f}")
        
        # Strategy bias
        bias_adjustment = sum(self.strategy_biases.values()) * 0.1
        score += bias_adjustment
        
        # Noise for variety
        noise = np.random.normal(0, 0.05)
        return float(np.clip(score + noise, -1.0, 1.0))


# ==============================================================================
# State Utilities
# ==============================================================================

def state_to_text(state: pyspiel.State) -> str:
    """Converts OpenSpiel Quoridor state to semantic text."""
    state_str = str(state)
    description_parts = []
    
    current_player = state.current_player()
    if current_player >= 0:
        description_parts.append(f"Current turn: Player {current_player + 1}")
    
    description_parts.append("\nBoard State:")
    description_parts.append(state_str)
    
    if not state.is_terminal():
        legal_actions = state.legal_actions()
        description_parts.append(f"\nLegal moves available: {len(legal_actions)}")
    
    p1_path, p2_path = estimate_path_lengths(state)
    description_parts.append(f"\nP1 path: {p1_path}, P2 path: {p2_path}")
    
    return '\n'.join(description_parts)


def estimate_path_lengths(state: pyspiel.State) -> Tuple[int, int]:
    """
    Estimate shortest path lengths for both players using BFS.
    
    This properly accounts for walls by parsing the board representation
    and finding actual shortest paths.
    """
    from collections import deque
    
    state_str = str(state)
    lines = state_str.strip().split('\n')
    
    # Parse board to find pieces and walls
    # Board is 9x9, displayed with spacing
    # Rows are numbered 1-9, columns a-i
    
    # Find piece positions
    p1_pos = None  # @ position (row, col)
    p2_pos = None  # 0 position (row, col)
    
    # Parse walls - horizontal walls block vertical movement, vertical walls block horizontal
    h_walls = set()  # (row, col) means wall below cell at (row, col)
    v_walls = set()  # (row, col) means wall right of cell at (row, col)
    
    board_row = -1
    for line_idx, line in enumerate(lines):
        # Skip header lines
        if 'Board size' in line or line.strip().startswith('a '):
            continue
        
        # Check if this is a board row (has position numbers at start/end)
        stripped = line.strip()
        if stripped and stripped[0].isdigit():
            board_row = int(stripped[0]) - 1  # Convert to 0-indexed
            
            # Find pieces in this line
            for col in range(9):
                # Each cell is 4 chars wide (like ".   " or "@   ")
                char_pos = 3 + col * 4  # Starting after row number
                if char_pos < len(line):
                    char = line[char_pos] if char_pos < len(line) else '.'
                    if char == '@':
                        p1_pos = (board_row, col)
                    elif char == '0':
                        p2_pos = (board_row, col)
        
        # Check for walls (--- for horizontal, | for vertical)
        if '---' in line and board_row >= 0:
            # Horizontal wall - blocks movement between this row and next
            for col in range(8):
                start = 3 + col * 4
                if start + 3 <= len(line) and '---' in line[start:start+6]:
                    h_walls.add((board_row, col))
        
        if '|' in line and board_row >= 0:
            # Vertical wall - blocks movement between columns
            for pos in range(len(line)):
                if line[pos] == '|':
                    col = (pos - 3) // 4
                    if 0 <= col < 8:
                        v_walls.add((board_row, col))
    
    # Default positions if not found
    if p1_pos is None:
        p1_pos = (0, 4)  # @ starts at top center
    if p2_pos is None:
        p2_pos = (8, 4)  # 0 starts at bottom center
    
    def can_move(from_pos, to_pos):
        """Check if movement is possible (no wall blocking)."""
        r1, c1 = from_pos
        r2, c2 = to_pos
        
        # Check bounds
        if r2 < 0 or r2 > 8 or c2 < 0 or c2 > 8:
            return False
        
        # Horizontal wall check (blocks vertical movement)
        if r2 > r1:  # Moving down
            if (r1, c1) in h_walls or (r1, c1-1) in h_walls:
                return False
        elif r2 < r1:  # Moving up
            if (r2, c2) in h_walls or (r2, c2-1) in h_walls:
                return False
        
        # Vertical wall check (blocks horizontal movement)
        if c2 > c1:  # Moving right
            if (r1, c1) in v_walls or (r1-1, c1) in v_walls:
                return False
        elif c2 < c1:  # Moving left
            if (r1, c2) in v_walls or (r1-1, c2) in v_walls:
                return False
        
        return True
    
    def bfs_path_length(start, goal_row):
        """BFS to find shortest path to goal row."""
        if start[0] == goal_row:
            return 0
        
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            pos, dist = queue.popleft()
            r, c = pos
            
            # Check all 4 directions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (r + dr, c + dc)
                
                if new_pos in visited:
                    continue
                
                if can_move(pos, new_pos):
                    if new_pos[0] == goal_row:
                        return dist + 1
                    
                    visited.add(new_pos)
                    queue.append((new_pos, dist + 1))
        
        return 99  # No path found (shouldn't happen in valid Quoridor)
    
    # Calculate actual shortest paths
    p1_path = bfs_path_length(p1_pos, 8)  # @ wants to reach row 8 (bottom)
    p2_path = bfs_path_length(p2_pos, 0)  # 0 wants to reach row 0 (top)
    
    return max(1, p1_path), max(1, p2_path)


def get_piece_positions(state_str: str) -> tuple:
    """Get piece row positions from state string."""
    lines = state_str.strip().split('\n')
    p1_row, p2_row = 0, 8
    for i, line in enumerate(lines):
        if '@' in line:
            p1_row = i // 2
        if '0' in line:
            p2_row = i // 2
    return p1_row, p2_row


def load_game() -> pyspiel.Game:
    """Load the Quoridor game."""
    return pyspiel.load_game("quoridor")


# ==============================================================================
# MCTS Node
# ==============================================================================

class MCTSNode:
    """Node in the Monte Carlo Tree Search tree."""
    
    def __init__(self, state: pyspiel.State, parent: Optional['MCTSNode'] = None,
                 action: Optional[int] = None):
        self.state = state.clone()
        self.parent = parent
        self.action = action
        
        self.children: List['MCTSNode'] = []
        self.untried_actions: List[int] = []
        
        self.visits = 0
        self.total_value = 0.0
        
        if not state.is_terminal():
            self.untried_actions = list(state.legal_actions())
            random.shuffle(self.untried_actions)  # Ensure fair exploration
    
    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal()
    
    def ucb1_score(self, exploration_constant: float = 1.41) -> float:
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_value / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration
    
    def best_child(self, exploration_constant: float = 1.41) -> 'MCTSNode':
        return max(self.children, key=lambda c: c.ucb1_score(exploration_constant))
    
    def expand(self) -> 'MCTSNode':
        action = self.untried_actions.pop()
        next_state = self.state.clone()
        next_state.apply_action(action)
        child = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child)
        return child
    
    def update(self, value: float):
        self.visits += 1
        self.total_value += value


# ==============================================================================
# LLM-MCTS Bot
# ==============================================================================

class LLM_MCTS_Bot(pyspiel.Bot):
    """
    Hybrid LLM + MCTS Bot with episodic memory recording.
    """
    
    def __init__(self, game: pyspiel.Game, player_id: int,
                 num_simulations: int = 50,
                 exploration_constant: float = 1.41,
                 evaluator=None,
                 experience_manager: Optional[ExperienceManager] = None):
        """
        Initialize the LLM-MCTS Bot.
        
        Args:
            game: OpenSpiel game instance
            player_id: Player ID (0 or 1)
            num_simulations: Number of MCTS simulations per move
            exploration_constant: UCB1 exploration parameter
            evaluator: Evaluator instance (LLMEvaluator or HeuristicEvaluator)
            experience_manager: ExperienceManager for recording experiences
        """
        super().__init__()
        self.game = game
        self.player_id = player_id
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.experience_manager = experience_manager
        
        if evaluator is not None:
            self.evaluator = evaluator
        else:
            self.evaluator = HeuristicEvaluator(experience_manager)
    
    def step(self, state: pyspiel.State) -> int:
        """Select the best action and record experience."""
        legal_actions = state.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        #  Early game heuristic: first move should always be forward
        if hasattr(self, '_move_count'):
            self._move_count += 1
        else:
            self._move_count = 1
        
        if self._move_count <= 2:
            # First 2 moves: just move forward
            forward_move = self._get_shortest_path_action(state)
            if forward_move is not None:
                return forward_move
        
        root = MCTSNode(state)
        
        for _ in range(self.num_simulations):
            node = self._select(root)
            
            if not node.is_terminal and not node.is_fully_expanded:
                node = node.expand()
            
            value = self._evaluate(node)
            self._backpropagate(node, value)
        
        if not root.children:
            return random.choice(legal_actions)
        
        # With low simulations relative to actions, use value-based selection
        # Otherwise use visit-based selection (more statistically robust)
        avg_visits = sum(c.visits for c in root.children) / len(root.children)
        if avg_visits < 3:
            # Low visits: use average value (exploitation only)
            best_child = max(root.children, 
                            key=lambda c: c.total_value / c.visits if c.visits > 0 else float('-inf'))
        else:
            # Enough visits: use visit count
            best_child = max(root.children, key=lambda c: c.visits)
        best_action = best_child.action
        
        # Record experience
        mcts_confidence = best_child.visits / self.num_simulations
        if self.experience_manager:
            state_text = state_to_text(state)
            self.experience_manager.record_step(state_text, mcts_confidence)
        
        return best_action
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.is_fully_expanded and not node.is_terminal:
            node = node.best_child(self.exploration_constant)
        return node
    
    def _evaluate(self, node: MCTSNode) -> float:
        """
        Evaluate a node by playing out to terminal state (rollout).
        
        Uses a smart rollout policy:
        - 70% probability: move along shortest path to goal
        - 30% probability: random action
        
        This is much more accurate than heuristic evaluation.
        """
        if node.is_terminal:
            returns = node.state.returns()
            return returns[self.player_id]
        
        # Rollout: play to terminal using semi-random policy
        sim_state = node.state.clone()
        
        max_moves = 200  # Prevent infinite loops
        moves = 0
        
        while not sim_state.is_terminal() and moves < max_moves:
            legal_actions = sim_state.legal_actions()
            
            if random.random() < 0.7:
                # 70%: Try to move toward goal along shortest path
                action = self._get_shortest_path_action(sim_state)
                if action is None:
                    action = random.choice(legal_actions)
            else:
                # 30%: Random action
                action = random.choice(legal_actions)
            
            sim_state.apply_action(action)
            moves += 1
        
        if sim_state.is_terminal():
            returns = sim_state.returns()
            return returns[self.player_id]
        else:
            # Timeout - use heuristic as fallback
            p1_path, p2_path = estimate_path_lengths(sim_state)
            if self.player_id == 0:
                return np.tanh((p1_path - p2_path) * 0.4)
            else:
                return np.tanh((p2_path - p1_path) * 0.4)
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """
        Backpropagate value up the tree.
        
        The value is from OUR perspective (self.player_id): positive = good for us.
        We track who made the move (parent's current_player) to determine value sign.
        """
        while node is not None:
            # Who made the move that led to this node?
            if node.parent is None:
                # Root node: just increment visits (needed for UCB1), no value update
                node.visits += 1
            else:
                mover = node.parent.state.current_player()
                if mover == self.player_id:
                    node_value = value  # We made this move: good values should boost
                else:
                    node_value = -value  # Opponent made this move: flip
                node.update(node_value)
            node = node.parent
    
    def _get_shortest_path_action(self, state: pyspiel.State) -> Optional[int]:
        """
        Get an action that moves toward the current player's goal.
        Returns None if no move actions available.
        """
        current_player = state.current_player()
        legal_actions = state.legal_actions()
        
        # Filter to only move actions (not walls)
        moves = []
        for action in legal_actions:
            action_str = state.action_to_string(current_player, action)
            # Move actions don't end with 'h' or 'v'
            if not (action_str.endswith('h') or action_str.endswith('v')):
                moves.append(action)
        
        if not moves:
            return None
        
        # Evaluate each move and pick the one that gets closest to goal
        best_action = None
        best_path_length = float('inf')
        
        for action in moves:
            test_state = state.clone()
            test_state.apply_action(action)
            
            # Get path length for the player who just moved
            p1_path, p2_path = estimate_path_lengths(test_state)
            
            # The current player just moved, so they are "not turn" in test_state
            if current_player == 0:
                path_length = p2_path  # P0 controls '0' piece (P2 path)
            else:
                path_length = p1_path  # P1 controls '@' piece (P1 path)
            
            if path_length < best_path_length:
                best_path_length = path_length
                best_action = action
        
        return best_action
    
    def step_with_policy(self, state: pyspiel.State) -> Tuple[List[Tuple[int, float]], int]:
        action = self.step(state)
        legal_actions = state.legal_actions()
        policy = [(a, 1.0 / len(legal_actions)) for a in legal_actions]
        return policy, action
    
    def restart(self):
        pass
    
    def inform_action(self, state: pyspiel.State, player_id: int, action: int):
        pass


# ==============================================================================
# Random Bot
# ==============================================================================

class RandomBot(pyspiel.Bot):
    """Simple random bot for baseline comparison."""
    
    def __init__(self, player_id: int, seed: Optional[int] = None):
        super().__init__()
        self.player_id = player_id
        self.rng = np.random.default_rng(seed)
    
    def step(self, state: pyspiel.State) -> int:
        return self.rng.choice(state.legal_actions())
    
    def restart(self):
        pass
    
    def inform_action(self, state: pyspiel.State, player_id: int, action: int):
        pass


# ==============================================================================
# Agent Factory
# ==============================================================================

class QuoridorAgentFactory:
    """
    Factory for creating configured Quoridor agents.
    Provides a simple interface for all scripts.
    """
    
    def __init__(self, memory_file: str = DEFAULT_MEMORY_DB,
                 use_llm: bool = False,
                 strategy_text: str = "",
                 verbose: bool = False):
        """
        Initialize the factory.
        
        Args:
            memory_file: Path to unified memory file
            use_llm: Whether to use LLM evaluation
            strategy_text: Optional strategy guide text
            verbose: Whether to print debug info about memory usage
        """
        self.memory_file = memory_file
        self.use_llm = use_llm
        self.strategy_text = strategy_text
        self.verbose = verbose
        
        # Unified experience manager
        self.experience_manager = ExperienceManager.get_instance(memory_file, player_id=0)
        
        # Evaluator (created lazily)
        self._evaluator = None
    
    @property
    def evaluator(self):
        """Get or create the evaluator (lazy initialization)."""
        if self._evaluator is None:
            if self.use_llm:
                try:
                    self._evaluator = LLMEvaluator(
                        experience_manager=self.experience_manager,
                        strategy_text=self.strategy_text
                    )
                except Exception as e:
                    print(f"[Factory] LLM failed: {e}, using heuristic")
                    self._evaluator = HeuristicEvaluator(
                        experience_manager=self.experience_manager,
                        strategy_text=self.strategy_text,
                        verbose=self.verbose
                    )
            else:
                self._evaluator = HeuristicEvaluator(
                    experience_manager=self.experience_manager,
                    strategy_text=self.strategy_text,
                    verbose=self.verbose
                )
        return self._evaluator
    
    def create_bot(self, game: pyspiel.Game, player_id: int,
                   num_simulations: int = 50,
                   record_experience: bool = True) -> LLM_MCTS_Bot:
        """
        Create a configured LLM-MCTS bot.
        
        Args:
            game: OpenSpiel game instance
            player_id: Player ID (0 or 1)
            num_simulations: MCTS simulations per move
            record_experience: Whether to record experiences
        """
        exp_manager = self.experience_manager if record_experience else None
        
        return LLM_MCTS_Bot(
            game=game,
            player_id=player_id,
            num_simulations=num_simulations,
            evaluator=self.evaluator,
            experience_manager=exp_manager,
        )
    
    def create_random_bot(self, player_id: int, seed: int = None) -> RandomBot:
        """Create a random bot."""
        return RandomBot(player_id=player_id, seed=seed)
    
    def end_game(self, outcome: float):
        """Record game outcome to memory."""
        self.experience_manager.end_game(outcome)
    
    def clear_buffer(self):
        """Clear the current game buffer."""
        self.experience_manager.clear_game_buffer()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self.experience_manager.get_statistics()
