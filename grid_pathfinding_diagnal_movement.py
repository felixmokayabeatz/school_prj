import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
import math

class DiagonalPathfinder:
    def __init__(self, grid):
        self.grid = np.array(grid)
        self.height = len(grid)
        self.width = len(grid[0])
        self.explored = set()
        
    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_neighbors(self, current):
        """Get valid neighboring cells including diagonals."""
        neighbors = []
        # All possible movements (including diagonals)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dx, dy in directions:
            new_x, new_y = current[0] + dx, current[1] + dy
            if (0 <= new_x < self.height and 
                0 <= new_y < self.width and 
                self.grid[new_x][new_y] == 0):
                neighbors.append((new_x, new_y))
        return neighbors
    
    def greedy_best_first_search(self, start, goal):
        """Implement Greedy Best-First Search algorithm."""
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = {start: None}
        
        while not frontier.empty():
            current = frontier.get()[1]
            self.explored.add(current)
            
            if current == goal:
                break
                
            for next_pos in self.get_neighbors(current):
                if next_pos not in came_from:
                    priority = self.euclidean_distance(next_pos, goal)
                    frontier.put((priority, next_pos))
                    came_from[next_pos] = current
        
        return self.reconstruct_path(came_from, start, goal)
    
    def reconstruct_path(self, came_from, start, goal):
        """Reconstruct the path from start to goal."""
        current = goal
        path = []
        
        while current is not None:
            path.append(current)
            current = came_from[current]
        
        path.reverse()
        return path if path[0] == start else []
    
    def visualize(self, path=None):
        """Visualize the grid, explored nodes, and path."""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap='binary')
        
        # Plot explored nodes
        if self.explored:
            explored_x, explored_y = zip(*self.explored)
            plt.scatter(explored_y, explored_x, color='yellow', alpha=0.3, s=100, label='Explored')
        
        # Plot path
        if path:
            path_x, path_y = zip(*path)
            plt.plot(path_y, path_x, color='red', linewidth=3, label='Path')
            plt.scatter([path_y[0]], [path_x[0]], color='green', s=200, label='Start')
            plt.scatter([path_y[-1]], [path_x[-1]], color='red', s=200, label='Goal')
        
        plt.grid(True)
        plt.legend()
        plt.title('Grid Pathfinding with Diagonal Movement')
        plt.show()

# Example usage
def main():
    # Example grid (0 = path, 1 = obstacle)
    grid = [
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0]
    ]
    
    start = (0, 0)
    goal = (6, 6)
    
    pathfinder = DiagonalPathfinder(grid)
    path = pathfinder.greedy_best_first_search(start, goal)
    pathfinder.visualize(path)

if __name__ == "__main__":
    main()