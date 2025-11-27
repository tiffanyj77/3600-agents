from collections.abc import Callable
from typing import List, Set, Tuple
from collections import deque

from game import *
from game.enums import Direction, MoveType, loc_after_direction

# prob_hear table from the diagram.
# Keys are (abs(dx), abs(dy)), values are P(hear | that offset).
PROB_HEAR = {
    (0, 1): 0.50,
    (1, 0): 0.50,
    (1, 1): 0.25,
    (0, 2): 0.10,
    (2, 0): 0.10,
    (1, 2): 0.10,
    (2, 1): 0.10,
    # everything else => 0
}

# prob_feel table from the diagram.
PROB_FEEL = {
    (0, 1): 0.30,
    (1, 0): 0.30,
    (1, 1): 0.15,
    # everything else => 0
}

class PlayerAgent:
    """
    /you may add functions, however, __init__ and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        self.corners = {}
        self.other_corners = {}
        self.visited = set()
        self.last_location = None

        self.move_history = []

        # board size (should be 8)
        self.map_size = board.game_map.MAP_SIZE

        # trapdoor memory:
        # index 0 = white trapdoor (i + j even)
        # index 1 = black trapdoor (i + j odd)
        self.heard_locs = [set(), set()]
        self.felt_locs = [set(), set()]
        self.ever_heard = [False, False]
        self.ever_felt = [False, False]

        # candidate locations for each trapdoor color
        # restricted to the inner 4×4 square:
        # rows 2..5 and cols 2..5 (0 based)
        self.trap_candidates = [set(), set()]

        inner_row_min = 2
        inner_row_max = 5
        inner_col_min = 2
        inner_col_max = 5

        for x in range(inner_row_min, inner_row_max + 1):
            for y in range(inner_col_min, inner_col_max + 1):
                if (x + y) % 2 == 0:
                    # white square → candidate for white trapdoor
                    self.trap_candidates[0].add((x, y))
                else:
                    # black square → candidate for black trapdoor
                    self.trap_candidates[1].add((x, y))

                # belief weights for each candidate (start uniform at 1.0)
        self.trap_belief = [dict(), dict()]  # index 0 = white, 1 = black
        for idx in (0, 1):
            for c in self.trap_candidates[idx]:
                self.trap_belief[idx][c] = 1.0


    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        if len(self.corners) == 0:
            if board.chicken_player.is_player_a():
                self.corners = {(0, 0), (7, 7)}
                self.other_corners = {(0, 7), (7, 0)}
            else:
                self.corners = {(0, 7), (7, 0)}
                self.other_corners = {(0, 0), (7, 7)}
        location = board.chicken_player.get_location()
        print(f"I'm at {location}.")
        print(f"Trapdoor A: heard? {sensor_data[0][0]}, felt? {sensor_data[0][1]}")
        print(f"Trapdoor B: heard? {sensor_data[1][0]}, felt? {sensor_data[1][1]}")
        print(f"Starting to think with {time_left()} seconds left.")

        self.update_trap_senses(location, sensor_data)

        # print candidate trapdoor locations for debugging
        self.debug_print_trap_candidates()

        moves = board.get_valid_moves()
        result = moves[0]
        result_score = -1e18

        for move in moves:
            s = self.heuristic(board, move, location)
            if s > result_score:
                result_score = s
                result = move

        self.move_history.append((location, result))

        self.visited.add(location)
        self.last_location = location
        print(f"I have {time_left()} seconds left. Playing {result}.")

        if len(self.move_history) == 40:   # 40 turns per player from the assignment
            print("==== FULL MOVE HISTORY FOR THIS GAME ====")
            for turn_idx, (loc, mv) in enumerate(self.move_history):
                print(f"Turn {turn_idx}: at {loc}, played {mv}")

        return result

    def debug_print_trap_candidates(self):
        """
        Print current candidate squares for each trapdoor.
        Index 0 = white trapdoor, index 1 = black trapdoor.
        """
        white_candidates = sorted(self.trap_candidates[0])
        black_candidates = sorted(self.trap_candidates[1])

        print("Current candidate trapdoor locations:")
        print(f"  White trapdoor candidates: {white_candidates}")
        print(f"  Black trapdoor candidates: {black_candidates}")


    def _neighbors_radius1(self, loc: Tuple[int, int]) -> Set[Tuple[int, int]]:
        x, y = loc
        out = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    out.add((nx, ny))
        return out

    def update_trap_senses(
        self,
        location: Tuple[int, int],
        sensor_data: List[Tuple[bool, bool]],
    ) -> None:
        """
        Update memory and belief weights for each trapdoor color
        using the prob_hear / prob_feel tables.

        sensor_data[0] = (heard_white, felt_white)
        sensor_data[1] = (heard_black, felt_black)
        """
        for idx in (0, 1):
            heard, felt = sensor_data[idx]

            if heard:
                self.ever_heard[idx] = True
                self.heard_locs[idx].add(location)
            if felt:
                self.ever_felt[idx] = True
                self.felt_locs[idx].add(location)

            # Bayesian-style weight update for each candidate square
            new_belief = {}
            total_weight = 0.0

            for c, w in self.trap_belief[idx].items():
                p_hear = self.prob_hear_given_trap(c, location)
                p_feel = self.prob_feel_given_trap(c, location)

                # likelihood of what we actually observed at this location
                like_hear = p_hear if heard else (1.0 - p_hear)
                like_feel = p_feel if felt else (1.0 - p_feel)

                likelihood = like_hear * like_feel

                # multiply old weight by likelihood
                new_w = w * likelihood
                if new_w > 0.0:
                    new_belief[c] = new_w
                    total_weight += new_w

            # if everything got zeroed (numerical issues), keep old beliefs
            if total_weight > 0.0:
                # normalize so weights sum to 1 for stability
                for c in new_belief:
                    new_belief[c] /= total_weight
                self.trap_belief[idx] = new_belief
                # trap_candidates are just the keys with nonzero belief
                self.trap_candidates[idx] = set(new_belief.keys())


    def prob_hear_given_trap(self, trap_loc: Tuple[int, int], loc: Tuple[int, int]) -> float:
        """Return P(hear | trap at trap_loc, we are at loc)."""
        dx = abs(loc[0] - trap_loc[0])
        dy = abs(loc[1] - trap_loc[1])
        return PROB_HEAR.get((dx, dy), 0.0)

    def prob_feel_given_trap(self, trap_loc: Tuple[int, int], loc: Tuple[int, int]) -> float:
        """Return P(feel | trap at trap_loc, we are at loc)."""
        dx = abs(loc[0] - trap_loc[0])
        dy = abs(loc[1] - trap_loc[1])
        return PROB_FEEL.get((dx, dy), 0.0)


    def distance_corner(self, loc: Tuple[int, int]) -> int:
        x, y = loc
        if len(self.corners) != 0:
            return min(abs(x - cx) + abs(y - cy) for (cx, cy) in self.corners)
        return min(abs(x - cx) + abs(y - cy) for (cx, cy) in self.other_corners)

    def center_penalty(self, location: Tuple[int, int]) -> float:
        x, y = location
        cx, cy = 3.5, 3.5
        dist = abs(x - cx) + abs(y - cy)

        return max(0, 5 - dist)

    def reachable(self, board: board.Board, start: Tuple[int, int]) -> int:
        if board.is_cell_blocked(start):
            return 0

        visited = set()
        q = deque([start])

        while q:
            curr = q.popleft()
            if curr in visited:
                continue
            visited.add(curr)

            for d in Direction:
                next_move = loc_after_direction(curr, d)
                if not board.is_valid_cell(next_move):
                    continue
                if board.is_cell_blocked(next_move):
                    continue
                if next_move not in visited:
                    q.append(next_move)
        return len(visited)

    def heuristic(self, board: board.Board, move: Tuple[int, int], location: Tuple[int, int]):
        direction, move_type = move
        next_loc = loc_after_direction(location, direction)

        if location in self.corners:
            self.corners.remove(location)

        if not board.is_valid_cell(next_loc):
            return -1e9

        score = 0

        danger_cells = self.trap_candidates[0] | self.trap_candidates[1]
        if next_loc in danger_cells:
            score -= 500    # big penalty for stepping on possible trap

        score -= self.center_penalty(next_loc)
        if move_type == MoveType.EGG:
            score += 4
            if next_loc in self.corners:
                score += 4

        if next_loc in self.visited:
            #maybe changing the heuristic here will prevent it from looping?
            score -= 50
        if next_loc is not None and next_loc == self.last_location:
            score -= 50

        forecast = board.forecast_move(direction, move_type)
        if forecast is None:
            return -1e9

        post_loc = forecast.chicken_player.get_location()
        reach = self.reachable(forecast, post_loc)
        score += reach

        if reach == 0:
            score -= 100

        # Check reachability to each corner
        not_reachable = []
        for corner in self.corners:
            if not self.is_reachable(forecast, post_loc, corner):
                not_reachable.append(corner)
            else:
                if corner in self.visited:
                    not_reachable.append(corner)
                else:
                    dist = self.distance_corner(next_loc)
                    score -= dist
        for not_reach in not_reachable:
            self.corners.remove(not_reach)

        not_reachable1 = []
        if len(self.corners) == 0:
            for corner in self.other_corners:
                if not self.is_reachable(forecast, post_loc, corner):
                    not_reachable1.append(corner)
                else:
                    dist = self.distance_corner(next_loc)
                    score -= dist
        for not_reach1 in not_reachable1:
            self.other_corners.remove(not_reach1)


        return score

    def is_reachable(self, board, start, target):
        if board.is_cell_blocked(target):
            return False

        visited = set()
        q = deque([start])

        while q:
            cur = q.popleft()
            if cur == target:
                return True
            if cur in visited:
                continue
            visited.add(cur)

            for d in Direction:
                nxt = loc_after_direction(cur, d)
                if not board.is_valid_cell(nxt):
                    continue
                if board.is_cell_blocked(nxt):
                    continue
                if nxt not in visited:
                    q.append(nxt)

        return False

