from collections.abc import Callable
from typing import List, Set, Tuple
from collections import deque

from game import *
from game.enums import Direction, MoveType, loc_after_direction

class PlayerAgent:
    """
    /you may add functions, however, __init__ and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        self.corners = {}
        self.other_corners = {}
        self.visited = set()
        self.last_location = []

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
    
        moves = board.get_valid_moves()
        result = moves[0]
        result_score = -1e18

        for move in moves:
            s = self.heuristic(board, move, location)
            if s > result_score:
                result_score = s
                result = move

        self.visited.add(location)
        self.last_location = location
        print(f"I have {time_left()} seconds left. Playing {result}.")
        return result
    
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

        score -= self.center_penalty(next_loc)
        if move_type == MoveType.EGG:
            score += 4
            if next_loc in self.corners:
                score += 4

        if next_loc in self.visited:
            score -= 20
        if next_loc == self.last_location:
            score -= 20
        
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

            