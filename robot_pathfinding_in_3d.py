import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from queue import PriorityQueue
import math

class Robot3DPathfinder:
    def __init__(self, grid_3d):
        self.grid = np.array(grid_3d)
        self.depth, self.height, self.width = self.grid.shape
        self.explored = set()
        
    def euclidean_distance_3d(self, point1, point2):
        """Calculate 3D Euclidean distance."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def get_neighbors(self, current):
        """Get valid neighboring positions in 3D space."""
        neighbors = []
        directions = [
            (0, 0, 1), (0, 0, -1),  # forward/back
            (0, 1, 0), (0, -1, 0),  # up/down
            (1, 0, 0), (-1, 0, 0)   # right/left
        ]
        
        for dx, dy, dz in directions:
            new_x = current[0] + dx
            new_y = current[1] + dy
            new_z = current[2] + dz
            
            if (0 <= new_x < self.depth and 
                0 <= new_y < self.height and 
                0 <= new_z < self.width and 
                self.grid[new_x, new_y, new_z] == 0):
                neighbors.append((new_x, new_y, new_z))
        
        return neighbors
    
    def greedy_best_first_search(self, start, goal):
        """Implement 3D Greedy Best-First Search."""
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
                    priority = self.euclidean_distance_3d(next_pos, goal)
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
        """Visualize the 3D environment and path."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot obstacles
        obstacle_positions = np.where(self.grid == 1)
        ax.scatter(obstacle_positions[0], obstacle_positions[1], obstacle_positions[2], 
                  c='gray', marker='s', alpha=0.1, label='Obstacles')
        
        # Plot explored positions
        if self.explored:
            explored_x, explored_y, explored_z = zip(*self.explored)
            ax.scatter(explored_x, explored_y, explored_z, 
                      c='yellow', alpha=0.3, label='Explored')
        
        # Plot path
        if path:
            path_x, path_y, path_z = zip(*path)
            ax.plot(path_x, path_y, path_z, 
                   c='red', linewidth=2, label='Path')
            
            # Mark start and goal
            ax.scatter([path[0][0]], [path[0][1]], [path[0][2]], 
                      c='green', s=100, label='Start')
            ax.scatter([path[-1][0]], [path[-1][1]], [path[-1][2]], 
                      c='red', s=100, label='Goal')
        
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.legend()
        plt.title('3D Robot Pathfinding')
        plt.show()

# Example usage
def main():
    # Create a 5x5x5 3D grid with some obstacles
    grid_size = 5
    grid_3d = np.zeros((grid_size, grid_size, grid_size))
    
    # Add some obstacles
    obstacles = [
        (1, 1, 1), (1, 2, 1), (2, 2, 2),
        (3, 3, 3), (2, 3, 2), (3, 2, 3)
    ]
    for obs in obstacles:
        grid_3d[obs] = 1
    
    start = (0, 0, 0)
    goal = (4, 4, 4)
    
    pathfinder = Robot3DPathfinder(grid_3d)
    path = pathfinder.greedy_best_first_search(start, goal)
    pathfinder.visualize(path)

if __name__ == "__main__":
    main()