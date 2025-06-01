import matplotlib.pyplot as plt
import csv
from collections import deque

# Fixed labyrinth
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

# Wizards
wizards = [
    {"name": "Harry Potter", "start": (1, 1), "speed": 2.5},
    {"name": "Cedric Diggory", "start": (3, 5), "speed": 1.8},
    {"name": "Viktor Krum", "start": (5, 7), "speed": 2.2},
]

rows = len(labyrinth)
cols = len(labyrinth[0])

# Find the exit
exit_pos = None
for i in range(rows):
    for j in range(cols):
        if labyrinth[i][j] == "E":
            exit_pos = (i, j)
            break
    if exit_pos:
        break

def bfs(start):
    queue = deque()
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    distance = [[-1 for _ in range(cols)] for _ in range(rows)]

    sr, sc = start
    queue.append((sr, sc))
    visited[sr][sc] = True
    distance[sr][sc] = 0

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if labyrinth[nr][nc] != '#' and not visited[nr][nc]:
                    visited[nr][nc] = True
                    distance[nr][nc] = distance[r][c] + 1
                    queue.append((nr, nc))
    
    er, ec = exit_pos
    return distance[er][ec]

# Process wizards
# Process wizards
results = []
for wizard in wizards:
    start = wizard["start"]
    speed = wizard["speed"]
    
    # Check if the wizard starts directly at or next to the exit
    if start == exit_pos:
        path_length = 0  # Already at the exit
    else:
        path_length = bfs(start)  # Compute the shortest path
    
    # If path_length is -1, it's unreachable, otherwise, calculate time
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
        time_to_exit = round(path_length / speed, 2)  # Calculate the time to exit
        results.append({
            "Wizard": wizard["name"],
            "Start Row": start[0],
            "Start Col": start[1],
            "Speed": speed,
            "Path Length": path_length,
            "Time to Exit": time_to_exit
        })

    start = wizard["start"]
    speed = wizard["speed"]
    
    if start == exit_pos:
        path_length = 0
    else:
        path_length = bfs(start)
    
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
        time_to_exit = round(path_length / speed, 2)
        results.append({
            "Wizard": wizard["name"],
            "Start Row": start[0],
            "Start Col": start[1],
            "Speed": speed,
            "Path Length": path_length,
            "Time to Exit": time_to_exit
        })

# Write CSV
with open("./part1/b/triwizard_results.csv", "w", newline="") as csvfile:
    fieldnames = ["Wizard", "Start Row", "Start Col", "Speed", "Path Length", "Time to Exit"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

# Predict winner
reachable_wizards = [r for r in results if r["Path Length"] != "Unreachable"]
if reachable_wizards:
    winner = min(reachable_wizards, key=lambda x: x["Time to Exit"])
    print(f"🏆 Winner: {winner['Wizard']} in {winner['Time to Exit']} minutes!")
else:
    print("No wizard could reach the exit!")

# Visualize labyrinth
fig, ax = plt.subplots(figsize=(8, 6))
for r in range(rows):
    for c in range(cols):
        if labyrinth[r][c] == '#':
            ax.add_patch(plt.Rectangle((c, rows-r-1), 1, 1, color='black'))
        elif labyrinth[r][c] == 'E':
            ax.add_patch(plt.Rectangle((c, rows-r-1), 1, 1, color='green'))
        else:
            ax.add_patch(plt.Rectangle((c, rows-r-1), 1, 1, color='white', edgecolor='gray'))

for wizard in wizards:
    sr, sc = wizard["start"]
    ax.text(sc + 0.5, rows - sr - 0.5, wizard["name"][0], ha='center', va='center', color='red', fontsize=12, fontweight='bold')

plt.xlim(0, cols)
plt.ylim(0, rows)
plt.gca().set_aspect('equal')
plt.axis('off')
plt.title("Labyrinth Map")
plt.savefig("./part1/b/labyrinth_visualization.png")
plt.show()

# Visualize results
names = [r["Wizard"] for r in results]
times = [r["Time to Exit"] if r["Time to Exit"] != "Unreachable" else None for r in results]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(names, [t if t is not None else 0 for t in times], color=['green' if t is not None else 'red' for t in times])
ax.set_ylabel('Time to Exit (minutes)')
ax.set_title('Triwizard Tournament Results')

for bar, time in zip(bars, times):
    height = bar.get_height()
    if time is not None:
        ax.annotate(f'{time}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    else:
        ax.annotate('Unreachable', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', color='red')

plt.savefig("./part1/b/triwizard_results.png")
plt.show()
