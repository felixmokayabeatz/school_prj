import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
import random

EMPTY, OBSTACLE, START, GOAL, PATH = 0, 1, 2, 3, 4

GRID_SIZE = 10

class Grid:
    def __init__(self, size, start, goal):
        self.size = size
        self.start = start
        self.goal = goal
        self.grid = np.zeros((size, size), dtype=int)
        self.grid[start] = START
        self.grid[goal] = GOAL
        self.obstacles = self.generate_obstacles()

    def generate_obstacles(self, num_obstacles=15):
        obstacles = set()
        while len(obstacles) < num_obstacles:
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if (x, y) != self.start and (x, y) != self.goal:
                obstacles.add((x, y))
                self.grid[x, y] = OBSTACLE
        return obstacles

    def move_obstacles(self):
        """Randomly move each obstacle to an adjacent cell."""
        new_obstacles = set()
        self.grid[self.grid == OBSTACLE] = EMPTY
        for (x, y) in self.obstacles:
            new_x, new_y = x + random.choice([-1, 0, 1]), y + random.choice([-1, 0, 1])
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                if (new_x, new_y) != self.start and (new_x, new_y) != self.goal:
                    new_obstacles.add((new_x, new_y))
                    self.grid[new_x, new_y] = OBSTACLE
        self.obstacles = new_obstacles

    def is_obstacle(self, position):
        return self.grid[position] == OBSTACLE

    def in_bounds(self, position):
        x, y = position
        return 0 <= x < self.size and 0 <= y < self.size

class GBFS:
    def __init__(self, grid):
        self.grid = grid
        self.start = grid.start
        self.goal = grid.goal
        self.path = []

    def heuristic(self, position):
        """Calculate Manhattan distance to the goal."""
        return abs(position[0] - self.goal[0]) + abs(position[1] - self.goal[1])

    def get_neighbors(self, position):
        """Get valid neighboring positions."""
        x, y = position
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        return [(nx, ny) for nx, ny in neighbors if self.grid.in_bounds((nx, ny)) and not self.grid.is_obstacle((nx, ny))]

    def search(self):
        """Perform the Greedy Best-First Search."""
        frontier = []
        heapq.heappush(frontier, (self.heuristic(self.start), self.start))
        came_from = {self.start: None}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == self.goal:
                self.reconstruct_path(came_from)
                return True

            for neighbor in self.get_neighbors(current):
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    heapq.heappush(frontier, (self.heuristic(neighbor), neighbor))

            self.grid.move_obstacles()
            yield came_from

        return False

    def reconstruct_path(self, came_from):
        """Reconstruct path from goal to start."""
        current = self.goal
        while current != self.start:
            self.path.append(current)
            current = came_from[current]
        self.path.reverse()

def visualize_search(grid, gbfs):
    fig, ax = plt.subplots()

    def update(came_from):
        ax.clear()
        ax.imshow(grid.grid, cmap="tab20c")
        
        for position in came_from:
            if came_from[position] is not None and position != grid.start:
                ax.plot([position[1], came_from[position][1]], [position[0], came_from[position][0]], "c-")

        for (x, y) in gbfs.path:
            ax.plot(y, x, "bs")

        ax.plot(grid.start[1], grid.start[0], "go")
        ax.plot(grid.goal[1], grid.goal[0], "ro")

        ax.set_title("Treasure Hunt with Greedy Best-First Search")
        ax.grid(False)

    ani = animation.FuncAnimation(fig, update, frames=gbfs.search, repeat=False, interval=500)
    plt.show()

start_position = (0, 0)
goal_position = (9, 9)
grid = Grid(GRID_SIZE, start_position, goal_position)
gbfs = GBFS(grid)

visualize_search(grid, gbfs)
