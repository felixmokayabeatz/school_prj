import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue
import math

class CityNavigator:
    def __init__(self):
        self.graph = nx.Graph()
        self.positions = {}
        self.explored = set()
        
    def add_city(self, name, x, y):
        """Add a city with its coordinates."""
        self.graph.add_node(name)
        self.positions[name] = (x, y)
        
    def add_road(self, city1, city2):
        """Add a road (edge) between two cities."""
        dist = self.calculate_distance(city1, city2)
        self.graph.add_edge(city1, city2, weight=dist)
        
    def calculate_distance(self, city1, city2):
        """Calculate straight-line distance between two cities."""
        x1, y1 = self.positions[city1]
        x2, y2 = self.positions[city2]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
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
                
            for next_city in self.graph.neighbors(current):
                if next_city not in came_from:
                    priority = self.calculate_distance(next_city, goal)
                    frontier.put((priority, next_city))
                    came_from[next_city] = current
        
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
        """Visualize the city graph, explored cities, and path."""
        plt.figure(figsize=(12, 8))
        
        # Draw the base graph
        nx.draw_networkx_edges(self.graph, self.positions, alpha=0.2)
        nx.draw_networkx_nodes(self.graph, self.positions, 
                             node_color='lightgray', node_size=500)
        nx.draw_networkx_labels(self.graph, self.positions)
        
        # Draw explored cities
        explored_nodes = list(self.explored)
        nx.draw_networkx_nodes(self.graph, self.positions, 
                             nodelist=explored_nodes,
                             node_color='yellow', node_size=500)
        
        # Draw path
        if path:
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(self.graph, self.positions,
                                 edgelist=path_edges,
                                 edge_color='r', width=2)
            
            # Highlight start and goal
            nx.draw_networkx_nodes(self.graph, self.positions,
                                 nodelist=[path[0]],
                                 node_color='green', node_size=500)
            nx.draw_networkx_nodes(self.graph, self.positions,
                                 nodelist=[path[-1]],
                                 node_color='red', node_size=500)
        
        plt.title("City Navigation using Greedy Best-First Search")
        plt.axis('equal')
        plt.show()

# Example usage
def main():
    navigator = CityNavigator()
    
    # Add cities with their coordinates
    cities = {
        'New York': (0, 0),
        'Boston': (2, 2),
        'Philadelphia': (1, -1),
        'Washington': (2, -2),
        'Pittsburgh': (3, 0),
        'Chicago': (5, 1),
        'Detroit': (4, 2)
    }
    
    for city, coords in cities.items():
        navigator.add_city(city, coords[0], coords[1])
    
    # Add roads
    roads = [
        ('New York', 'Boston'),
        ('New York', 'Philadelphia'),
        ('Philadelphia', 'Washington'),
        ('Philadelphia', 'Pittsburgh'),
        ('Boston', 'Pittsburgh'),
        ('Pittsburgh', 'Chicago'),
        ('Pittsburgh', 'Detroit'),
        ('Chicago', 'Detroit')
    ]
    
    for city1, city2 in roads:
        navigator.add_road(city1, city2)
    
    # Find and visualize path
    path = navigator.greedy_best_first_search('New York', 'Chicago')
    navigator.visualize(path)

if __name__ == "__main__":
    main()