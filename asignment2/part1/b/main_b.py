import matplotlib.pyplot as plt
import csv
from collections import deque

"""
TRIWIZARD TOURNAMENT LABYRINTH SOLVER

This program simulates the third task of the Triwizard Tournament from Harry Potter,
where wizards must navigate through a labyrinth to reach the exit. The program:

1. Models a 2D grid-based labyrinth with walls (#), paths (.), and exit (E)
2. Uses BFS (Breadth-First Search) to find shortest paths for each wizard
3. Calculates time to complete based on wizard speed and path length
4. Determines the winner and visualizes results

Main Algorithms Used:
- Breadth-First Search (BFS) for shortest path finding
- 2D Grid traversal and representation
- CSV file I/O for data persistence
- Data visualization using matplotlib
"""

# Fixed labyrinth - 2D grid representation
"""
Labyrinth Data Structure:
- 2D array of strings representing the maze
- '#' = walls (impassable)
- '.' = open paths (passable)  
- 'E' = exit (goal position)

This is a standard grid-based maze representation used in pathfinding algorithms.
Each cell can be accessed using (row, col) coordinates.
"""
labyrinth = [
    "###############",
    "#.#.#.........#",
    "#.#.#.#####.#.#",
    "#...#.#...#.#.#",
    "###.#.#.#.#.#.#",
    "#...#.#.#...#.#",
    "#.#.#.#.#####.#",
    "#.#...#.......#",
    "#.#####.#####.#",
    "#.....#...E...#",
    "###############"
]

# Wizard configuration - each wizard has different starting position and movement speed
"""
Wizard Data Structure:
- Dictionary containing wizard properties
- name: identifier string
- start: (row, col) tuple representing starting coordinates
- speed: floating point number representing movement rate (cells per minute)

This allows for different wizard characteristics affecting their performance.
"""
wizards = [
    {"name": "Harry Potter", "start": (1, 1), "speed": 2.5},
    {"name": "Cedric Diggory", "start": (3, 5), "speed": 1.8},
    {"name": "Viktor Krum", "start": (5, 7), "speed": 2.2},
]

# Grid dimensions calculation
"""
Grid Bounds Calculation:
- rows: number of rows in the labyrinth
- cols: number of columns (characters) in each row

These are used for boundary checking during pathfinding to prevent
array index out of bounds errors.
"""
rows = len(labyrinth)
cols = len(labyrinth[0])

# Find the exit position using linear search
"""
Exit Position Finding Algorithm:
- Linear search through 2D grid
- Nested loop iteration: O(rows × cols) time complexity
- Early termination when 'E' is found
- Returns (row, col) coordinates of exit

Time Complexity: O(n×m) where n=rows, m=cols
Space Complexity: O(1)

This could be optimized by storing exit position as a constant,
but for small grids, the performance impact is negligible.
"""
exit_pos = None
for i in range(rows):
    for j in range(cols):
        if labyrinth[i][j] == "E":
            exit_pos = (i, j)
            break
    if exit_pos:
        break

def bfs(start):
    """
    Breadth-First Search (BFS) Algorithm for Shortest Path Finding
    
    Algorithm: BFS on 2D Grid
    - Uses queue (FIFO) to explore nodes level by level
    - Guarantees shortest path in unweighted graphs
    - Explores all nodes at distance k before exploring nodes at distance k+1
    - Uses visited array to prevent revisiting nodes
    - Uses distance array to track shortest distance to each cell
    
    Data Structures Used:
    1. Queue (deque): For BFS traversal - O(1) append/popleft operations
    2. Visited 2D array: Boolean matrix to track explored cells - prevents cycles
    3. Distance 2D array: Integer matrix storing shortest distance from start
    
    Time Complexity: O(V + E) where V = rows×cols, E = number of valid moves
    - In grid: E ≤ 4V (each cell has max 4 neighbors)
    - Overall: O(rows × cols)
    
    Space Complexity: O(rows × cols) for visited, distance arrays, and queue
    
    Why BFS for this problem:
    - Unweighted graph (each move costs 1 unit)
    - Need shortest path (minimum number of steps)
    - BFS guarantees optimality for unweighted shortest path
    
    Alternative algorithms considered:
    - DFS: Doesn't guarantee shortest path
    - Dijkstra: Overkill for unweighted graph (BFS is simpler and faster)
    - A*: Could be faster with heuristic, but BFS is sufficient for small grids
    """
    
    # Initialize BFS data structures
    queue = deque()  # Queue for BFS traversal
    visited = [[False for _ in range(cols)] for _ in range(rows)]  # Track visited cells
    distance = [[-1 for _ in range(cols)] for _ in range(rows)]   # Track distances (-1 = unreachable)

    # Starting position setup
    sr, sc = start
    queue.append((sr, sc))
    visited[sr][sc] = True
    distance[sr][sc] = 0

    # BFS main loop
    """
    BFS Traversal Process:
    1. Dequeue front element (current position)
    2. Explore all 4 adjacent cells (up, down, left, right)
    3. For each valid, unvisited neighbor:
       - Mark as visited
       - Set distance = current_distance + 1
       - Add to queue for future exploration
    4. Repeat until queue is empty
    
    The 4-directional movement array [(-1,0),(1,0),(0,-1),(0,1)] represents:
    - (-1,0): Up (decrease row)
    - (1,0):  Down (increase row)  
    - (0,-1): Left (decrease column)
    - (0,1):  Right (increase column)
    """
    while queue:
        r, c = queue.popleft()
        
        # Explore 4-connected neighbors (up, down, left, right)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            
            # Boundary and validity checks
            """
            Cell Validation Conditions:
            1. 0 <= nr < rows: Within vertical bounds
            2. 0 <= nc < cols: Within horizontal bounds  
            3. labyrinth[nr][nc] != '#': Not a wall
            4. not visited[nr][nc]: Not already explored
            
            All conditions must be true to proceed with this neighbor.
            """
            if 0 <= nr < rows and 0 <= nc < cols:
                if labyrinth[nr][nc] != '#' and not visited[nr][nc]:
                    visited[nr][nc] = True
                    distance[nr][nc] = distance[r][c] + 1
                    queue.append((nr, nc))
    
    # Return shortest distance to exit
    """
    Result Extraction:
    - Access distance array at exit position
    - Returns -1 if exit is unreachable (never visited during BFS)
    - Returns positive integer representing minimum steps to exit
    """
    er, ec = exit_pos
    return distance[er][ec]

# Process each wizard and calculate their performance
"""
Wizard Processing Algorithm:
For each wizard:
1. Extract starting position and speed
2. Handle special case: already at exit (path_length = 0)
3. Otherwise: compute shortest path using BFS
4. Calculate time = path_length / speed
5. Store results for analysis and output

This demonstrates the practical application of the BFS algorithm
in a real-world scenario with additional business logic.
"""
results = []

# Note: There appears to be duplicate processing code in the original - cleaning this up
for wizard in wizards:
    start = wizard["start"]
    speed = wizard["speed"]
    
    # Special case: wizard starts at exit
    """
    Edge Case Handling:
    If wizard starts at exit position, no movement is needed.
    This prevents unnecessary BFS computation and handles the
    mathematical edge case where distance = 0.
    """
    if start == exit_pos:
        path_length = 0  # Already at the exit
    else:
        path_length = bfs(start)  # Compute the shortest path using BFS
    
    # Result calculation and storage
    """
    Performance Calculation:
    - If path_length == -1: Exit is unreachable from this starting position
    - Otherwise: time_to_exit = path_length / speed
    
    The division gives time in minutes, assuming:
    - path_length is in grid cells
    - speed is in cells per minute
    
    Results are stored in a list of dictionaries for easy CSV export
    and further analysis.
    """
    if path_length == -1:
        results.append({
            "Wizard": wizard["name"],
            "Start Row": start[0],
            "Start Col": start[1],
            "Speed": speed,
            "Path Length": "Unreachable",
            "Time to Exit": "Unreachable"
        })
    else:
        time_to_exit = round(path_length / speed, 2)  # Round to 2 decimal places
        results.append({
            "Wizard": wizard["name"],
            "Start Row": start[0],
            "Start Col": start[1],
            "Speed": speed,
            "Path Length": path_length,
            "Time to Exit": time_to_exit
        })

# CSV Export
"""
CSV File Output Algorithm:
- Uses Python's csv.DictWriter for structured data export
- Creates header row with field names
- Writes each result dictionary as a row
- Maintains data integrity and enables external analysis

File I/O Operations:
- Opens file in write mode with newline='' to prevent extra blank rows
- Uses context manager (with statement) for automatic file closure
- Structured format allows easy import into spreadsheets or databases

Time Complexity: O(n) where n = number of wizards
Space Complexity: O(1) additional space (file operations)
"""
with open("./part1/b/triwizard_results.csv", "w", newline="") as csvfile:
    fieldnames = ["Wizard", "Start Row", "Start Col", "Speed", "Path Length", "Time to Exit"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

# Winner Determination
"""
Winner Selection Algorithm:
1. Filter out wizards who cannot reach the exit
2. Find minimum time among reachable wizards
3. Use Python's min() function with custom key

Algorithm: Linear Search for Minimum
- Filter operation: O(n) to create reachable_wizards list
- min() with key function: O(n) to find minimum time
- Overall: O(n) where n = number of wizards

The key parameter allows min() to compare by "Time to Exit" field
rather than the entire dictionary object.
"""
reachable_wizards = [r for r in results if r["Path Length"] != "Unreachable"]
if reachable_wizards:
    winner = min(reachable_wizards, key=lambda x: x["Time to Exit"])
    print(f"🏆 Winner: {winner['Wizard']} in {winner['Time to Exit']} minutes!")
else:
    print("No wizard could reach the exit!")

# Labyrinth Visualization
"""
2D Grid Visualization Algorithm using Matplotlib:
- Creates figure with specified dimensions
- Iterates through each grid cell
- Maps cell types to colors:
  * '#' (walls) → black rectangles
  * 'E' (exit) → green rectangle  
  * '.' (paths) → white rectangles with gray borders
- Adds wizard starting positions as colored text markers

Coordinate System Transformation:
- Grid coordinates: (row, col) where (0,0) is top-left
- Matplotlib coordinates: (x, y) where (0,0) is bottom-left
- Transformation: matplotlib_y = rows - grid_row - 1

This creates an intuitive top-down view where the labyrinth
appears correctly oriented for human interpretation.

Rendering Complexity: O(rows × cols) for grid + O(wizards) for markers
"""
fig, ax = plt.subplots(figsize=(8, 6))

# Draw labyrinth cells
for r in range(rows):
    for c in range(cols):
        # Coordinate transformation for proper visual orientation
        matplotlib_y = rows - r - 1
        
        if labyrinth[r][c] == '#':
            # Walls - solid black rectangles
            ax.add_patch(plt.Rectangle((c, matplotlib_y), 1, 1, color='black'))
        elif labyrinth[r][c] == 'E':
            # Exit - green rectangle (goal visualization)
            ax.add_patch(plt.Rectangle((c, matplotlib_y), 1, 1, color='green'))
        else:
            # Paths - white with gray borders for visibility
            ax.add_patch(plt.Rectangle((c, matplotlib_y), 1, 1, color='white', edgecolor='gray'))

# Add wizard starting positions
"""
Wizard Position Markers:
- Uses first letter of wizard name as identifier
- Positions text at cell center using offset (+0.5)
- Red color for visibility against white/gray background
- Bold font for emphasis
"""
for wizard in wizards:
    sr, sc = wizard["start"]
    matplotlib_y = rows - sr - 0.5  # Center the text in the cell
    ax.text(sc + 0.5, matplotlib_y, wizard["name"][0], 
            ha='center', va='center', color='red', fontsize=12, fontweight='bold')

# Configure plot appearance
plt.xlim(0, cols)
plt.ylim(0, rows)
plt.gca().set_aspect('equal')  # Ensures square cells
plt.axis('off')  # Removes axis labels and ticks for cleaner look
plt.title("Labyrinth Map")
plt.savefig("./part1/b/labyrinth_visualization.png")
plt.show()

# Results Bar Chart Visualization
"""
Bar Chart Visualization Algorithm:
- Extracts wizard names and completion times
- Handles unreachable cases by setting height to 0 with red color
- Uses color coding: green for successful completion, red for failure
- Adds text annotations showing exact times

Data Preparation:
- Creates parallel arrays for names and times
- Converts "Unreachable" to None for proper handling
- Uses list comprehension for efficient data transformation

Matplotlib Bar Chart:
- bar() function creates rectangular bars with heights proportional to times
- Color mapping based on reachability status
- Annotations placed above bars using annotate() with offset positioning

Visual Design Principles:
- Clear color distinction (green/red) for quick status identification
- Numeric labels for precise value reading
- Meaningful title and axis labels for context
"""
names = [r["Wizard"] for r in results]
times = [r["Time to Exit"] if r["Time to Exit"] != "Unreachable" else None for r in results]

fig, ax = plt.subplots(figsize=(8, 5))

# Create bars with conditional coloring
bars = ax.bar(names, [t if t is not None else 0 for t in times], 
              color=['green' if t is not None else 'red' for t in times])

ax.set_ylabel('Time to Exit (minutes)')
ax.set_title('Triwizard Tournament Results')

# Add value annotations on bars
"""
Annotation Algorithm:
- Iterates through bars and corresponding time values
- Calculates text position: center of bar horizontally, top of bar vertically
- Uses different text and colors for reachable vs unreachable cases
- Offset positioning prevents text overlap with bar tops

Text Positioning:
- xy: anchor point (center of bar, height of bar)
- xytext: offset from anchor (0 pixels horizontal, 3 pixels up)
- textcoords="offset points": uses point-based offset system
"""
for bar, time in zip(bars, times):
    height = bar.get_height()
    if time is not None:
        # Successful completion - show time in default color
        ax.annotate(f'{time}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    else:
        # Failed attempt - show "Unreachable" in red
        ax.annotate('Unreachable', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', color='red')

plt.savefig("./part1/b/triwizard_results.png")
plt.show()

"""
SUMMARY OF ALGORITHMS AND COMPLEXITY ANALYSIS:

1. Breadth-First Search (BFS):
   - Time: O(V + E) = O(rows × cols) for grid graphs
   - Space: O(V) = O(rows × cols) for visited and distance arrays
   - Guarantees shortest path in unweighted graphs

2. Linear Search (Exit Finding):
   - Time: O(rows × cols) worst case
   - Space: O(1)
   - Could be optimized to O(1) by storing exit position as constant

3. Winner Selection:
   - Time: O(n) where n = number of wizards
   - Space: O(n) for filtered list
   - Uses min() with key function for efficient comparison

4. Visualization Rendering:
   - Time: O(rows × cols + wizards) for labyrinth + O(wizards) for results
   - Space: O(1) additional space (matplotlib handles internal storage)

5. File I/O Operations:
   - Time: O(wizards) for CSV writing
   - Space: O(1) additional space (streaming write)

OVERALL PROGRAM COMPLEXITY:
- Time: O(wizards × rows × cols) - BFS dominates for multiple wizards
- Space: O(rows × cols) - Grid storage and BFS arrays dominate

The program efficiently solves the pathfinding problem and provides
comprehensive analysis and visualization of the results.
"""