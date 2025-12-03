#testtest
from collections.abc import Callable
from typing import List, Set, Tuple
from collections import deque
from time import sleep

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
        self.unreachable_corners = set()
        self.visited = set()
        self.last_location = None
        self.egg_direction = None

                # --- opponent + trapdoor observations ---
        self.last_enemy_loc = None               # enemy position on previous turn
        self.enemy_trap_events = []              # list of {"turn", "loc", "color_idx"}
        self.turn_index = 0                      # how many times play() has been called
        self.enemy_start = None

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

        if self.enemy_start is None:
            self.enemy_start = board.chicken_enemy.get_spawn()

        location = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()
        print(f"I'm at {location}.")
        print(f"Opponent at: {enemy_loc}")
        print(f"Trapdoor A: heard? {sensor_data[0][0]}, felt? {sensor_data[0][1]}")
        print(f"Trapdoor B: heard? {sensor_data[1][0]}, felt? {sensor_data[1][1]}")
        print(f"Starting to think with {time_left()} seconds left.")
        self.update_trap_senses(location, sensor_data)

        info = self.get_top_trapdoor_estimates()
        for idx in (0,1):
            color = "White" if idx == 0 else "Black"
            print(f"{color} trap belief:")
            print("  Top candidates:", info[idx]["top_candidates"])
            print(f"  BEST → {info[idx]['detected']} (conf={info[idx]['confidence']:.2f})\n")


        # print candidate trapdoor locations for debugging
        self.debug_print_trap_candidates()

        # choose an egg target & a direction for this turn
        best_egg_target, best_egg_path, egg_dir = self.closest_reachable_lay_egg(board)
        self.egg_direction = egg_dir

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

        # if len(self.move_history) == 40:   # 40 turns per player from the assignment
        #     print("==== FULL MOVE HISTORY FOR THIS GAME ====")
        #     for turn_idx, (loc, mv) in enumerate(self.move_history):
        #         print(f"Turn {turn_idx}: at {loc}, played {mv}")

                # update enemy memory and turn counter
        self.last_enemy_loc = enemy_loc
        self.turn_index += 1

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

    def get_top_trapdoor_estimates(self, k: int = 3, threshold: float = 0.75):
        """
        Returns:
            - top candidate squares for each color
            - if confident enough, returns the detected trapdoor cell
        """
        results = {}

        for idx in (0, 1):
            beliefs = self.trap_belief[idx]
            sorted_cells = sorted(beliefs.items(), key=lambda x: x[1], reverse=True)

            top_k = sorted_cells[:k]
            best_cell, best_prob = top_k[0]

            results[idx] = {
                "top_candidates": top_k,
                "detected" : best_cell if best_prob >= threshold else None,
                "confidence": best_prob
            }

        return results

    def distance_to(self, loc1: Tuple[int, int], loc2: Tuple[int, int]) -> int:
        x1, y1 = loc1
        x2, y2 = loc2
        return abs(x1 - x2) + abs(y1 - y2)

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

    def trap_danger_score(self, board: board.Board, loc: Tuple[int,int]) -> float:
        """Returns risk score based on trap probability at this square."""
        if loc in board.found_trapdoors:
            return 1e6

        danger = 0
        for idx in (0,1):     # white + black trapdoor
            if loc in self.trap_belief[idx]:
                danger += self.trap_belief[idx][loc] * 500  # scale penalty
        return danger


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

    def check_opponent_trap_fall(self, current_enemy_loc: Tuple[int, int] | None) -> None:
        """
        Detect if the opponent has just fallen into a trapdoor and record it.

        Assumes the game sets enemy location to None (or some invalid value)
        when they fall through a trap. If that is different in your engine,
        tweak the detection condition below.
        """
        # enemy "disappeared" this turn, but existed last turn
        if current_enemy_loc is None and self.last_enemy_loc is not None:
            fallen_cell = self.last_enemy_loc

            # trapdoor color: parity of (i + j)
            color_idx = 0 if (fallen_cell[0] + fallen_cell[1]) % 2 == 0 else 1

            # store in a log for later analysis / debugging
            event = {
                "turn": self.turn_index,
                "loc": fallen_cell,
                "color_idx": color_idx,       # 0 = white, 1 = black
            }
            self.enemy_trap_events.append(event)

            print("=== ENEMY TRAP EVENT DETECTED ===")
            print(f"  Turn: {self.turn_index}")
            print(f"  Enemy fell into trapdoor at {fallen_cell}, color_idx={color_idx}")
            print("=================================")

            # now we KNOW that square is the trapdoor for that color:
            self.trap_candidates[color_idx] = {fallen_cell}
            self.trap_belief[color_idx] = {fallen_cell: 1.0}


    def prob_feel_given_trap(self, trap_loc: Tuple[int, int], loc: Tuple[int, int]) -> float:
        """Return P(feel | trap at trap_loc, we are at loc)."""
        dx = abs(loc[0] - trap_loc[0])
        dy = abs(loc[1] - trap_loc[1])
        return PROB_FEEL.get((dx, dy), 0.0)


    def score_corner_progress(self, board: board.Board, post_loc: Tuple[int,int]) -> float:
        score = 0
        best_corner_dist = 999

        all_corners = list(self.corners) + list(self.other_corners)
        best_corner = None

        for corner in all_corners:
            if not self.is_reachable(board, post_loc, corner):
                continue

            dist = self.distance_to(post_loc, corner)

            if best_corner is None or dist < best_corner_dist:
                best_corner = corner

            best_corner_dist = min(best_corner_dist, dist)

        if best_corner is None:
            return 0

        if board.can_lay_egg_at_loc(best_corner):
            score += 80 / (best_corner_dist + 1)
        else:
            score += 40 / (best_corner_dist + 1)

        return score

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
                if next_move in board.found_trapdoors:
                    continue
                if next_move not in visited:
                    q.append(next_move)
        return len(visited)

    def heuristic(self, board: board.Board, move: Tuple[int, int], location: Tuple[int, int]):
        direction, move_type = move
        next_loc = loc_after_direction(location, direction)

        if location in self.corners:
            self.corners.remove(location)

        if location in self.other_corners:
            self.other_corners.remove(location)

        if not board.is_valid_cell(next_loc):
            return -1e9

        score = 0

        danger = self.trap_danger_score(board, next_loc)
        score -= danger

        if move_type == MoveType.EGG:
            score += 50
            if next_loc in self.corners:
                score += 20

        if move_type == MoveType.TURD:
            if not board.can_lay_egg():
                x, y = location
                if board.turns_left_player < 15:
                    if x != 0 and y != 0 and x != 7 and y != 7:
                        score += 4

        if next_loc in self.visited:
            #maybe changing the heuristic here will prevent it from looping?
            score -= 100
        if next_loc is not None and next_loc == self.last_location:
            score -= 50

        if next_loc in self.other_corners:
            score -= 50
            self.other_corners.remove(next_loc)

        forecast = board.forecast_move(direction, move_type)
        if forecast is None:
            return -1e9

        post_loc = forecast.chicken_player.get_location()
        reach = self.reachable(forecast, post_loc)
        score += reach

        if reach == 0:
            score -= 500

        opp_loc = forecast.chicken_enemy.get_location()
        opp_dist = abs(post_loc[0] - opp_loc[0]) + abs(post_loc[1] - opp_loc[1])

        if opp_dist <= 3:
            if move_type == MoveType.TURD:
                score += 50

        if post_loc[0] in (0,7) or post_loc[1] in (0,7):
            if opp_dist <= 3:
                score -= 50
            elif opp_dist <= 2:
                score -= 150

        dist_opp_start = self.distance_to(post_loc, self.enemy_start)
        if dist_opp_start <= 2:
            score -= 30

        # Check reachability to each corner
        not_reachable = []
        for corner in self.corners:
            if not self.is_reachable(forecast, post_loc, corner):
                not_reachable.append(corner)
                self.unreachable_corners.add(corner)
        for not_reach in not_reachable:
            self.corners.remove(not_reach)

        not_reachable1 = []
        if len(self.corners) == 0:
            for corner in self.other_corners:
                if not self.is_reachable(forecast, post_loc, corner):
                    not_reachable1.append(corner)
                    self.unreachable_corners.add(corner)
        for not_reach1 in not_reachable1:
            self.other_corners.remove(not_reach1)

        for unreachable in self.unreachable_corners:
            if self.distance_to(next_loc, unreachable) <= 2:
                score -= 20

        # ---- CORNER vs EGG PRIORITY ----
        corner_score = self.score_corner_progress(forecast, post_loc)
        score += corner_score

        # If corners aren't giving us any progress (blocked/unreachable),
        # follow the egg plan instead.
        if corner_score == 0 and self.egg_direction is not None:
            if move_type != MoveType.EGG and direction == self.egg_direction:
                score += 200   # guidance bonus towards egg target

        return score

    def closest_reachable_lay_egg(self, board: board.Board):
        """
        Returns:
            (best_target_square, best_path, first_direction)

        best_path is a list of coordinates (excluding start).
        first_direction is a Direction enum.
        """

        start = board.chicken_player.get_location()

        # All legal egg-lay squares
        egg_squares = [
            (x, y)
            for x in range(8)
            for y in range(8)
            if board.can_lay_egg_at_loc((x, y))
        ]
        if not egg_squares:
            return None, None, None

        # ---------- TRAP PROSPECT MAP ----------
        trap_score_map = {}
        for sq in egg_squares:
            risk = 0
            for idx in (0, 1):
                for trap_loc, belief in self.trap_belief[idx].items():
                    d = self.distance_to(sq, trap_loc)
                    risk += belief * max(0, 6 - d) * 50
            trap_score_map[sq] = risk

        # ---------- BFS ----------
        queue = deque([(start, [])])   # (position, path)
        visited = {start}

        best_target = None
        best_path = None
        best_cost = 1e18  # minimize (distance + trap risk)

        while queue:
            pos, path = queue.popleft()

            # If this position is a legal egg square:
            if pos in egg_squares:
                base_dist = len(path)
                trap_penalty = trap_score_map[pos]
                total_cost = base_dist + trap_penalty

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_target = pos
                    best_path = path[:]  # copy

            # Continue BFS expansion
            for d in Direction:
                nxt = loc_after_direction(pos, d)

                if not board.is_valid_cell(nxt):
                    continue
                if nxt in visited:
                    continue
                if board.is_cell_blocked(nxt):
                    continue

                # Hard avoid high-trap probability squares
                if self.trap_danger_score(board, nxt) > 400:
                    continue

                visited.add(nxt)
                queue.append((nxt, path + [nxt]))

        # Nothing reachable safely
        if best_target is None:
            return None, None, None

        # Extract direction from first step on the path
        if len(best_path) == 0:
            return best_target, [], None

        first_step = best_path[0]

        for d in Direction:
            if loc_after_direction(start, d) == first_step:
                return best_target, best_path, d

        # Fallback: path but no clear direction (shouldn't really happen)
        return best_target, best_path, None


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
            if cur in board.found_trapdoors:
                continue
            visited.add(cur)

            for d in Direction:
                nxt = loc_after_direction(cur, d)
                if not board.is_valid_cell(nxt):
                    continue
                if board.is_cell_blocked(nxt):
                    continue
                if nxt in board.found_trapdoors:
                    continue
                if nxt not in visited:
                    q.append(nxt)

        return False

