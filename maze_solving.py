import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

class MazeSolver:
    def __init__(self, maze):
        self.maze = np.array(maze)
        self.height = len(maze)
        self.width = len(maze[0])
        self.explored = set()
        
    def manhattan_distance(self, point1, point2):
        """Calculate Manhattan distance between two points."""
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
    
    def get_neighbors(self, current):
        """Get valid neighboring cells."""
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for dx, dy in directions:
            new_x, new_y = current[0] + dx, current[1] + dy
            if (0 <= new_x < self.height and 
                0 <= new_y < self.width and 
                self.maze[new_x][new_y] == 0):
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
                    priority = self.manhattan_distance(next_pos, goal)
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
        """Visualize the maze, explored nodes, and path."""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.maze, cmap='binary')
        
        explored_x, explored_y = zip(*self.explored)
        plt.scatter(explored_y, explored_x, color='yellow', alpha=0.3, s=100, label='Explored')

        if path:
            path_x, path_y = zip(*path)
            plt.plot(path_y, path_x, color='red', linewidth=3, label='Path')
            plt.scatter([path_y[0]], [path_x[0]], color='green', s=200, label='Start')
            plt.scatter([path_y[-1]], [path_x[-1]], color='red', s=200, label='Goal')
        
        plt.grid(True)
        plt.legend()
        plt.title('Maze Solution using Greedy Best-First Search')
        plt.show()

def main():
    maze = [
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    
    start = (0, 0)
    goal = (6, 6)
    
    solver = MazeSolver(maze)
    path = solver.greedy_best_first_search(start, goal)
    solver.visualize(path)

if __name__ == "__main__":
    main()