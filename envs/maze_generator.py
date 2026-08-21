import numpy as np
import random


class MazeGenerator:
    def __init__(self, width, height):
        # Ensure dimensions are odd for the wall/path generation to work cleanly
        self.width = width if width % 2 != 0 else width + 1
        self.height = height if height % 2 != 0 else height + 1
        # 1 = Wall, 0 = Free Path
        self.grid = np.ones((self.width, self.height), dtype=np.int8)

    def generate(self):
        """Generates a maze using Randomized Depth-First Search."""
        start_x, start_y = 1, 1
        self.grid[start_x, start_y] = 0
        walls = [(start_x, start_y, start_x + 2, start_y),
                 (start_x, start_y, start_x, start_y + 2)]

        while walls:
            # Pick a random wall
            wx, wy, nx, ny = walls.pop(random.randint(0, len(walls) - 1))

            if 0 < nx < self.width - 1 and 0 < ny < self.height - 1:
                if self.grid[nx, ny] == 1:
                    # Knock down the wall and mark the new cell as a path
                    self.grid[nx, ny] = 0
                    self.grid[wx + (nx - wx) // 2, wy + (ny - wy) // 2] = 0

                    # Add neighboring walls
                    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                        if 0 < nx + dx < self.width - 1 and 0 < ny + dy < self.height - 1:
                            walls.append((nx, ny, nx + dx, ny + dy))

        return self.grid

    def place_entities(self, grid, min_distance=2):
        """Randomly places the nest and food at a minimum Manhattan distance."""
        free_cells = np.argwhere(grid == 0)

        nest_idx = random.choice(range(len(free_cells)))
        nest_pos = tuple(free_cells[nest_idx])

        # Filter for valid food positions based on curriculum distance
        valid_food_cells = [
            tuple(cell) for cell in free_cells
            if abs(cell[0] - nest_pos[0]) + abs(cell[1] - nest_pos[1]) >= min_distance
        ]

        if not valid_food_cells:
            valid_food_cells = [tuple(cell) for cell in free_cells if tuple(cell) != nest_pos]

        food_pos = random.choice(valid_food_cells)
        return nest_pos, food_pos