import heapq
import math

# Define the heuristic function using diagonal distance
def diagonal_distance(current, goal):
    dx = abs(current[0] - goal[0])
    dy = abs(current[1] - goal[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

# Define the A* algorithm function
def a_star(start, goal, grid):
    # Initialize the open and closed sets
    open_set = [(0, start)]
    closed_set = set()

    # Initialize the cost and parent dictionaries
    cost = {start: 0}
    parent = {start: None}

    # Loop until the open set is empty
    while open_set:
        # Pop the node with the lowest cost from the open set
        current_cost, current = heapq.heappop(open_set)

        # If the current node is the goal, we have found a path
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1]

        # Add the current node to the closed set
        closed_set.add(current)

        # Loop through the neighbors of the current node
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            # Skip if the neighbor is not in the grid or is an obstacle or is in the closed set
            if not (0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0])) or grid[neighbor[0]][neighbor[1]] == 1 or neighbor in closed_set:
                continue

            # Calculate the tentative cost to reach the neighbor
            tentative_cost = cost[current] + math.sqrt(dx ** 2 + dy ** 2)

            # If the neighbor is not in the open set or the tentative cost is lower than the existing cost,
            # update the cost and parent dictionaries and add the neighbor to the open set
            if neighbor not in cost or tentative_cost < cost[neighbor]:
                cost[neighbor] = tentative_cost
                priority = tentative_cost + diagonal_distance(neighbor, goal)
                heapq.heappush(open_set, (priority, neighbor))
                parent[neighbor] = current

    # If we reach here, there is no path to the goal
    return None

def update_grid(grid, coord, length):
    try:
        row, col = coord
        # print(grid,coord)
        for i in range(row - length, row + length + 1):
            # print("THIS IS THE GRID TYPE",type(grid))
            for j in range(col - length, col + length + 1):
                if (i != row or j != col) and (i - row) ** 2 + (j - col) ** 2 <= length ** 2:
                    # print("THIS IS 0th INDEX OF GRID",grid[0])
                    if (0 <= i < len(grid)) and (0 <= j < len(grid[0])):
                        grid[i][j] = 1
        return grid            
    except:
        return grid



def path_planning(obs):
    grid_size = 500                
    grid = [[0]*grid_size for i in range(grid_size)]
    for i in obs:
        grid=update_grid( grid , (i[ 0 ],i[ 1 ]) , 50 )
    start = (grid_size - 1, math.floor( grid_size/2))
    goal = (0, math.floor( grid_size/2 ) )
    # grid[start[ 0 ]][start[ 1 ]]=2
    # grid[goal[ 0 ]][goal[  1] ]=3
    path = a_star( start, goal, grid )
    # for i in path:
    #     grid[i[0]][i[1]]=7
    # for i in grid:
    #     print(i)
    # if path:
    #     print("Path found:", path)
    # else:
    #     print("error")
    return path


