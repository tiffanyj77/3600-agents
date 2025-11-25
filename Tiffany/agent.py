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

    def _score_board(self, board: board_mod.Board) -> float:
        """Higher score means the position is better for us."""

        # egg difference is the most important thing
        my_eggs = len(board.eggs_player)
        enemy_eggs = len(board.eggs_enemy)
        egg_diff = my_eggs - enemy_eggs

        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()

        center = (board.game_map.MAP_SIZE // 2, board.game_map.MAP_SIZE // 2)
        dist_center = manhattan_distance(my_loc, center)
        dist_enemy = manhattan_distance(my_loc, enemy_loc)

        # mobility  how many moves will I have next turn
        my_moves_next = board.get_valid_moves()
        mobility = len(my_moves_next)

        score = 0.0

        # eggs are everything
        score += 40.0 * egg_diff

        # prefer to have more options next turn
        score += 0.5 * mobility

        # prefer being closer to the center
        score -= dist_center

        # small reward for being a bit away from the enemy
        score += 0.2 * dist_enemy

        # avoid enemy turd zones
        if board.is_cell_in_enemy_turd_zone(my_loc):
            score -= 50.0

        # avoid known trapdoors if any have been found
        if my_loc in board.found_trapdoors:
            score += board.game_map.TRAPDOOR_PENALTY

        return score


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

        if egg_moves:
            return self._choose_best_move(board, egg_moves)

        if turd_moves and board.chicken_player.turds_left > 0:
            return self._choose_best_move(board, turd_moves)

        if plain_moves:
            return self._choose_best_move(board, plain_moves)

        return moves[0]


    def _choose_best_move(
    self,
    board: board_mod.Board,
    candidate_moves: List[Tuple[Direction, MoveType]],
) -> Tuple[Direction, MoveType]:

        best_move = candidate_moves[0]
        best_score = float("-inf")

        for direction, move_type in candidate_moves:
            # ask the board what the world would look like after this move
            new_board = board.forecast_move(direction, move_type, check_ok=True)

            # forecast_move returns None if it is not valid
            if new_board is None:
                continue

            # give a small direct reward to egg and turd actions
            base = 0.0
            if move_type == MoveType.EGG:
                base += 5.0
            if move_type == MoveType.TURD:
                base += 2.0

            position_score = self._score_board(new_board)
            total_score = base + position_score

            if total_score > best_score:
                best_score = total_score
                best_move = (direction, move_type)

        return best_move

