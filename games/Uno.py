import pygame
import random
import sys

# --- Constants ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
CARD_WIDTH = 80
CARD_HEIGHT = 120
MARGIN = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
RED = (230, 40, 40)
GREEN = (40, 200, 40)
BLUE = (40, 80, 230)
YELLOW = (240, 220, 20)
ORANGE = (255, 127, 0) # For Wild cards background

COLORS = ["Red", "Green", "Blue", "Yellow"]
TYPES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Skip", "Reverse", "+2"]
WILD_TYPES = ["Wild", "Wild +4"]

COLOR_MAP = {
    "Red": RED, "Green": GREEN, "Blue": BLUE, "Yellow": YELLOW, "Wild": BLACK
}

class Card:
    def __init__(self, color, type_val):
        self.color = color
        self.type = type_val
        self.rect = None # For click detection

    def __repr__(self):
        return f"{self.color} {self.type}"

    def draw(self, surface, x, y, face_up=True):
        self.rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
        
        if face_up:
            # Draw Card Background
            bg_color = COLOR_MAP.get(self.color, BLACK)
            if self.color is None: bg_color = BLACK # Unset wild
            
            pygame.draw.rect(surface, bg_color, self.rect, border_radius=8)
            pygame.draw.rect(surface, WHITE, self.rect, 2, border_radius=8)
            
            # Draw Center Oval
            oval_rect = pygame.Rect(x + 5, y + 15, CARD_WIDTH - 10, CARD_HEIGHT - 30)
            pygame.draw.ellipse(surface, WHITE, oval_rect)
            
            # Draw Text
            font = pygame.font.SysFont("Arial", 24, bold=True)
            text_color = bg_color
            if self.color == "Wild" or self.color is None: text_color = BLACK
            
            label = self.type
            if label == "Wild +4": label = "+4"
            
            text_surf = font.render(label, True, text_color)
            text_rect = text_surf.get_rect(center=self.rect.center)
            surface.blit(text_surf, text_rect)
            
            # Small corner text
            small_font = pygame.font.SysFont("Arial", 14, bold=True)
            small_text = small_font.render(label, True, WHITE)
            surface.blit(small_text, (x + 5, y + 5))
            
        else:
            # Face Down (Opponent)
            pygame.draw.rect(surface, BLACK, self.rect, border_radius=8)
            pygame.draw.rect(surface, WHITE, self.rect, 2, border_radius=8)
            pygame.draw.circle(surface, RED, self.rect.center, 20)
            pygame.draw.circle(surface, YELLOW, self.rect.center, 15)

class UnoGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pygame UNO - 2D & Text State")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 16)
        
        self.deck = self.create_deck()
        self.discard_pile = []
        self.players = [[], []] # 0: Human, 1: AI
        self.turn = 0 # 0 for Human, 1 for AI
        self.direction = 1
        self.game_over = False
        self.message = "Game Start! Your Turn."
        self.waiting_for_color = False # For Wild card selection
        
        self.deal_initial_cards()
    
    def create_deck(self):
        deck = []
        for color in COLORS:
            for t in TYPES:
                deck.append(Card(color, t))
                if t != "0": deck.append(Card(color, t)) # Two of each except 0
        
        for _ in range(4):
            deck.append(Card("Wild", "Wild"))
            deck.append(Card("Wild", "Wild +4"))
            
        random.shuffle(deck)
        return deck

    def deal_initial_cards(self):
        for _ in range(7):
            self.players[0].append(self.deck.pop())
            self.players[1].append(self.deck.pop())
        
        # Start discard pile (retry if first card is Wild for simplicity)
        while True:
            card = self.deck.pop()
            if card.color != "Wild":
                self.discard_pile.append(card)
                break
            self.deck.append(card)
            random.shuffle(self.deck)

    def draw_card(self, player_idx, count=1):
        for _ in range(count):
            if not self.deck:
                # Reshuffle discard into deck
                top = self.discard_pile.pop()
                self.deck = self.discard_pile[:]
                self.discard_pile = [top]
                random.shuffle(self.deck)
                if not self.deck: break # No cards left
            
            self.players[player_idx].append(self.deck.pop())

    def is_valid_move(self, card):
        top_card = self.discard_pile[-1]
        
        # Wild is always valid
        if "Wild" in card.color:
            return True
            
        # Color match
        if card.color == top_card.color:
            return True
            
        # Type/Value match
        if card.type == top_card.type:
            return True
            
        return False

    def handle_special_card(self, card):
        if card.type == "Skip":
            self.turn = (self.turn + self.direction) % 2
            self.message = "Turn Skipped!"
        elif card.type == "Reverse":
            self.direction *= -1
            # In 2 player, reverse acts like skip
            if len(self.players) == 2:
                self.turn = (self.turn + self.direction) % 2
                self.message = "Reverse! (Turn Skipped)"
        elif card.type == "+2":
            next_p = (self.turn + self.direction) % 2
            self.draw_card(next_p, 2)
            self.turn = next_p # Skip after draw
            self.message = "Opponent drew 2 and was skipped!"
        elif card.type == "Wild +4":
            next_p = (self.turn + self.direction) % 2
            self.draw_card(next_p, 4)
            self.turn = next_p
            self.message = "Wild +4! Opponent drew 4 and skipped."

    def ai_turn(self):
        pygame.time.delay(1000) # Think time
        hand = self.players[1]
        valid_moves = [c for c in hand if self.is_valid_move(c)]
        
        if valid_moves:
            # Basic AI: Prioritize special cards, else random
            card = valid_moves[0] 
            hand.remove(card)
            
            # Handle Wild Color Selection (AI picks most common color in hand)
            if "Wild" in card.color:
                colors = [c.color for c in hand if c.color in COLORS]
                if not colors: chosen = "Red"
                else: chosen = max(set(colors), key=colors.count)
                card.color = chosen # Temporarily set color for pile
            
            self.discard_pile.append(card)
            self.message = f"AI played {card}"
            self.handle_special_card(card)
            
            if not hand:
                self.game_over = True
                self.message = "AI Wins!"
                return
        else:
            self.draw_card(1)
            self.message = "AI drew a card."
        
        self.turn = 0 # Back to human

    def render_text_state(self):
        """Draws the raw state as text for debugging/info."""
        state_info = [
            f"STATE LOG:",
            f"----------------",
            f"Turn: {'Human' if self.turn == 0 else 'AI'}",
            f"Direction: {self.direction}",
            f"Deck Count: {len(self.deck)}",
            f"Discard Top: {self.discard_pile[-1]}",
            f"Human Hand: {len(self.players[0])}",
            f"AI Hand: {len(self.players[1])}",
            f"Message: {self.message}"
        ]
        
        y_offset = 10
        pygame.draw.rect(self.screen, (30, 30, 30), (0, 0, 250, 200))
        for line in state_info:
            text = self.font.render(line, True, GREEN)
            self.screen.blit(text, (10, y_offset))
            y_offset += 20

    def draw_wild_selector(self):
        """Draws color buttons when human plays a wild."""
        panel_rect = pygame.Rect(300, 250, 400, 200)
        pygame.draw.rect(self.screen, GRAY, panel_rect, border_radius=10)
        
        title = self.font.render("Choose a Color:", True, WHITE)
        self.screen.blit(title, (420, 270))
        
        buttons = []
        colors = [(RED, "Red"), (GREEN, "Green"), (BLUE, "Blue"), (YELLOW, "Yellow")]
        
        for i, (col, name) in enumerate(colors):
            bx = 320 + (i * 90)
            by = 320
            rect = pygame.Rect(bx, by, 80, 80)
            pygame.draw.rect(self.screen, col, rect, border_radius=5)
            buttons.append((rect, name))
            
        return buttons

    def run(self):
        running = True
        while running:
            self.screen.fill((20, 100, 40)) # Felt green table
            
            # --- Event Handling ---
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                
                if self.turn == 0 and not self.game_over and not self.waiting_for_color:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        mx, my = pygame.mouse.get_pos()
                        
                        # Check Human Hand Clicks
                        hand_x_start = (SCREEN_WIDTH - (len(self.players[0]) * 50)) // 2
                        
                        clicked_card = None
                        clicked_index = -1
                        
                        # Iterate backwards to click top cards first
                        for i in range(len(self.players[0]) - 1, -1, -1):
                            c = self.players[0][i]
                            # Recalculate rect exactly as drawn
                            card_x = hand_x_start + (i * 50)
                            card_y = SCREEN_HEIGHT - CARD_HEIGHT - 20
                            temp_rect = pygame.Rect(card_x, card_y, CARD_WIDTH, CARD_HEIGHT)
                            
                            if temp_rect.collidepoint(mx, my):
                                clicked_card = c
                                clicked_index = i
                                break
                        
                        if clicked_card:
                            if self.is_valid_move(clicked_card):
                                if "Wild" in clicked_card.color:
                                    self.waiting_for_color = True
                                    self.temp_wild_card_index = clicked_index
                                else:
                                    # Play standard card
                                    self.players[0].pop(clicked_index)
                                    self.discard_pile.append(clicked_card)
                                    self.handle_special_card(clicked_card)
                                    if not self.players[0]:
                                        self.game_over = True
                                        self.message = "You Win!"
                                    elif self.turn == 0: # If turn didn't skip
                                        self.turn = 1
                            else:
                                self.message = "Invalid Move!"
                                
                        # Check Draw Deck Click
                        deck_rect = pygame.Rect(SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - 60, CARD_WIDTH, CARD_HEIGHT)
                        if deck_rect.collidepoint(mx, my):
                            self.draw_card(0)
                            self.turn = 1
                            self.message = "You drew a card."

                elif self.waiting_for_color:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        mx, my = pygame.mouse.get_pos()
                        buttons = self.draw_wild_selector() # Just to get rects
                        for rect, color_name in buttons:
                            if rect.collidepoint(mx, my):
                                # Execute Wild Play
                                card = self.players[0].pop(self.temp_wild_card_index)
                                card.color = color_name # Morph card color
                                self.discard_pile.append(card)
                                self.handle_special_card(card)
                                self.waiting_for_color = False
                                if not self.players[0]:
                                    self.game_over = True
                                    self.message = "You Win!"
                                elif self.turn == 0:
                                    self.turn = 1
                                break

            # --- Drawing ---
            
            # Draw Deck (Face down)
            deck_x, deck_y = SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - 60
            pygame.draw.rect(self.screen, BLACK, (deck_x, deck_y, CARD_WIDTH, CARD_HEIGHT), border_radius=8)
            pygame.draw.rect(self.screen, WHITE, (deck_x, deck_y, CARD_WIDTH, CARD_HEIGHT), 2, border_radius=8)
            font_draw = pygame.font.SysFont("Arial", 20, bold=True)
            text_draw = font_draw.render("Draw", True, WHITE)
            self.screen.blit(text_draw, (deck_x + 15, deck_y + 50))

            # Draw Discard Pile
            if self.discard_pile:
                self.discard_pile[-1].draw(self.screen, SCREEN_WIDTH//2 + 20, SCREEN_HEIGHT//2 - 60)

            # Draw AI Hand (Top, Face Down)
            ai_start_x = (SCREEN_WIDTH - (len(self.players[1]) * 40)) // 2
            for i, card in enumerate(self.players[1]):
                card.draw(self.screen, ai_start_x + (i * 40), 20, face_up=False)

            # Draw Human Hand (Bottom, Face Up)
            human_start_x = (SCREEN_WIDTH - (len(self.players[0]) * 50)) // 2
            for i, card in enumerate(self.players[0]):
                # Hover effect
                mx, my = pygame.mouse.get_pos()
                x = human_start_x + (i * 50)
                y = SCREEN_HEIGHT - CARD_HEIGHT - 20
                if pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT).collidepoint(mx, my):
                    y -= 20 # Pop up
                    
                card.draw(self.screen, x, y, face_up=True)

            # Draw Wild Selector Overlay
            if self.waiting_for_color:
                self.draw_wild_selector()

            # Render Text State (Debug View)
            self.render_text_state()

            # --- AI Logic ---
            if self.turn == 1 and not self.game_over:
                pygame.display.flip() # Force draw so we see AI thinking state
                self.ai_turn()

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    UnoGame().run()