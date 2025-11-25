# agent.py

from collections.abc import Callable
from typing import List, Tuple

# This imports board, enums, game_map, etc
from game import board as board_mod
from game.enums import Direction, MoveType, loc_after_direction
from game.board import manhattan_distance


class PlayerAgent:
    """
    You may add helper methods.
    But __init__ and play must keep this exact signature.
    """

    def __init__(self, board: board_mod.Board, time_left: Callable):
        # You can store info that persists across turns
        # For example, board size and a guess about trapdoors
        self.map_size = board.game_map.MAP_SIZE

        # Optional memory about where we think trapdoors might be
        # We start with no knowledge
        self.possible_trapdoors = set()

    def play(
        self,
        board: board_mod.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        """
        Decide and return a move as (Direction, MoveType).

        board: copy of current game state
        sensor_data: [(heardA, feltA), (heardB, feltB)]
        time_left: function returning seconds left for the whole game
        """

        # Keep it cheap in time each move so we never time out
        moves = board.get_valid_moves()

        # Safety: if engine ever gives no moves, just return something harmless
        if not moves:
            return Direction.UP, MoveType.PLAIN

        # Split moves by type
        egg_moves = [m for m in moves if m[1] == MoveType.EGG]
        turd_moves = [m for m in moves if m[1] == MoveType.TURD]
        plain_moves = [m for m in moves if m[1] == MoveType.PLAIN]

        # Basic preference:
        # 1. If we can lay an egg, do that
        if egg_moves:
            return self._choose_best_move(board, egg_moves)

        # 2. If no egg and we still have turds left,
        #    maybe place a turd when the enemy is nearby
        if turd_moves and board.chicken_player.turds_left > 0:
            # Only bother with turds when enemy is in the same half of the board
            my_loc = board.chicken_player.get_location()
            enemy_loc = board.chicken_enemy.get_location()
            if manhattan_distance(my_loc, enemy_loc) <= 4:
                return self._choose_best_move(board, turd_moves)

        # 3. Otherwise move plain
        if plain_moves:
            return self._choose_best_move(board, plain_moves)

        # Fallback: if something weird happens, just pick the first legal move
        return moves[0]

    def _choose_best_move(
        self,
        board: board_mod.Board,
        candidate_moves: List[Tuple[Direction, MoveType]],
    ) -> Tuple[Direction, MoveType]:
        """
        Simple heuristic:
        - Prefer eggs
        - Prefer being closer to center
        - Prefer being a little farther from the enemy
        """

        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()

        center = (self.map_size // 2, self.map_size // 2)

        best_move = candidate_moves[0]
        best_score = float("-inf")

        for direction, move_type in candidate_moves:
            # Where will we end up after this move
            new_loc = loc_after_direction(my_loc, direction)

            # Distances
            dist_center = manhattan_distance(new_loc, center)
            dist_enemy = manhattan_distance(new_loc, enemy_loc)

            # Score this move
            score = 0.0

            # Eggs are the main way to win
            if move_type == MoveType.EGG:
                score += 100.0

            # Turds are situationally useful
            if move_type == MoveType.TURD:
                score += 15.0

            # Prefer the center of the board
            # Smaller distance to center is better
            score -= dist_center

            # Slight preference to stay a bit away from the enemy
            score += 0.2 * dist_enemy

            # Optionally, you could add trapdoor logic using sensor_data in the future

            if score > best_score:
                best_score = score
                best_move = (direction, move_type)

        return best_move
