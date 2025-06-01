import time
import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import pandas as pd


class AuntPetuniaSeatingPlanner:
    def __init__(self):
        self.guests = []
        self.animosities = []
        self.seating_solution = None
        self.enemy_graph = None
        
        # Performance tracking
        self.performance_metrics = {
            'guest_count': 0,
            'animosity_count': 0,
            'algorithm_time': 0.0,
            'graph_creation_time': 0.0,
            'verification_time': 0.0,
            'total_time': 0.0,
            'memory_usage': 0,
            'iterations': 0,
            'connected_components': 0,
            'solution_found': False
        }
    
    def load_guest_list(self, filename):
        """Load the list of invited guests from file."""
        try:
            with open(filename, 'r') as file:
                self.guests = [name.strip() for name in file.readlines() if name.strip()]
            self.performance_metrics['guest_count'] = len(self.guests)
            print(f"Loaded {len(self.guests)} guests for Aunt Petunia's party:")
            for guest in self.guests:
                print(f"   • {guest}")
            print()
            return True
        except FileNotFoundError:
            print(f"Could not find guest list file: {filename}")
            return False
    
    def load_animosity_list(self, filename):
        """Load Aunt Petunia's suggestions on 'who doesn't like whom'."""
        try:
            with open(filename, 'r') as file:
                for line in file:
                    if ',' in line:
                        person1, person2 = line.strip().split(',', 1)
                        self.animosities.append((person1.strip(), person2.strip()))
            
            self.performance_metrics['animosity_count'] = len(self.animosities)
            print(f"Loaded {len(self.animosities)} animosity relationships:")
            for person1, person2 in self.animosities:
                print(f"   • {person1} dislikes {person2}")
            print()
            return True
        except FileNotFoundError:
            print(f"Could not find animosity list file: {filename}")
            return False
    
    def create_animosity_graph(self):
        """
        Create adjacency list representation of the animosity graph.
        Each person maps to list of people they dislike (bidirectional relationships).
        """
        start_time = time.time()
        
        self.enemy_graph = defaultdict(list)
        
        # Add all guests as nodes (including those with no animosities)
        for guest in self.guests:
            self.enemy_graph[guest] = []
        
        # Add animosity edges (bidirectional - if A dislikes B, then B dislikes A)
        for person1, person2 in self.animosities:
            if person1 in self.guests and person2 in self.guests:
                self.enemy_graph[person1].append(person2)
                self.enemy_graph[person2].append(person1)
        
        self.performance_metrics['graph_creation_time'] = time.time() - start_time
        print("Animosity graph created!")
        return self.enemy_graph
    
    def solve_with_non_recursive_dfs(self):
        """
        NON-RECURSIVE DFS ALGORITHM to determine if guests can be split into two tables.
        
        This is the core algorithm requested by Aunt Petunia!
        Uses stack-based DFS to check if the animosity graph is bipartite.
        
        Bipartite = can be colored with 2 colors so no adjacent nodes have same color
        Two colors = Two tables!
        """
        if not self.enemy_graph:
            self.create_animosity_graph()
        
        print("Starting non-recursive DFS algorithm...")
        start_time = time.time()
        
        # Color assignment: -1 = uncolored, 0 = Table A, 1 = Table B
        table_assignments = {guest: -1 for guest in self.guests}
        
        iterations = 0
        connected_components = 0
        
        # Process each connected component using non-recursive DFS
        for starting_guest in self.guests:
            if table_assignments[starting_guest] != -1:
                continue  # Already processed this guest
            
            connected_components += 1
            print(f"   Processing connected component #{connected_components} starting with {starting_guest}")
            
            # NON-RECURSIVE DFS using explicit stack
            dfs_stack = [(starting_guest, 0)]  # (guest, table_assignment)
            table_assignments[starting_guest] = 0
            
            while dfs_stack:
                current_guest, current_table = dfs_stack.pop()
                iterations += 1
                
                # Check all enemies of current guest
                for enemy in self.enemy_graph[current_guest]:
                    if table_assignments[enemy] == -1:
                        # Unvisited enemy - assign to opposite table
                        opposite_table = 1 - current_table
                        table_assignments[enemy] = opposite_table
                        dfs_stack.append((enemy, opposite_table))
                        print(f"     Assigning {enemy} to Table {'A' if opposite_table == 0 else 'B'}")
                    
                    elif table_assignments[enemy] == current_table:
                        # CONFLICT! Enemy at same table - impossible arrangement
                        end_time = time.time()
                        print(f"   Conflict detected: {current_guest} and {enemy} both need Table {'A' if current_table == 0 else 'B'}")
                        print(f"   Algorithm completed in {end_time - start_time:.4f} seconds")
                        
                        # Update performance metrics
                        self.performance_metrics['algorithm_time'] = end_time - start_time
                        self.performance_metrics['iterations'] = iterations
                        self.performance_metrics['connected_components'] = connected_components
                        self.performance_metrics['solution_found'] = False
                        
                        return None
        
        end_time = time.time()
        print(f"Non-recursive DFS completed successfully in {end_time - start_time:.4f} seconds!")
        
        # Update performance metrics
        self.performance_metrics['algorithm_time'] = end_time - start_time
        self.performance_metrics['iterations'] = iterations
        self.performance_metrics['connected_components'] = connected_components
        self.performance_metrics['solution_found'] = True
        
        self.seating_solution = table_assignments
        return table_assignments
    
    def display_seating_arrangement(self):
        """Display Aunt Petunia's final seating plan."""
        if not self.seating_solution:
            print("\nNo seating arrangement possible with just two tables!")
            print("Aunt Petunia's options:")
            print("   1. Add a third table")
            print("   2. Uninvite some problematic guests")
            print("   3. Have a 'conflict resolution' session before the party")
            return
        
        # Separate guests by table
        table_a = [guest for guest, table in self.seating_solution.items() if table == 0]
        table_b = [guest for guest, table in self.seating_solution.items() if table == 1]
        
        print("\n" + "=" * 60)
        print("AUNT PETUNIA'S NAMESDAY PARTY SEATING ARRANGEMENT")
        print("=" * 60)
        
        print(f"\nTABLE A ({len(table_a)} guests):")
        print("   " + "─" * 40)
        for guest in sorted(table_a):
            print(f"   │ {guest:<35} │")
        print("   " + "─" * 40)
        
        print(f"\nTABLE B ({len(table_b)} guests):")
        print("   " + "─" * 40)
        for guest in sorted(table_b):
            print(f"   │ {guest:<35} │")
        print("   " + "─" * 40)
        
        print(f"\nSuccess! All {len(self.animosities)} animosities resolved!")
        print("Aunt Petunia will have a wonderful, conflict-free namesday party!")
        print("=" * 60)
    
    def verify_solution(self):
        """Double-check that no enemies are at the same table."""
        if not self.seating_solution:
            return False
        
        print("\nVerifying seating arrangement...")
        start_time = time.time()
        
        conflicts = 0
        
        for person1, person2 in self.animosities:
            if (person1 in self.seating_solution and person2 in self.seating_solution):
                table1 = self.seating_solution[person1]
                table2 = self.seating_solution[person2]
                
                if table1 == table2:
                    print(f"   CONFLICT: {person1} and {person2} both at Table {'A' if table1 == 0 else 'B'}")
                    conflicts += 1
                else:
                    print(f"   OK: {person1} (Table {'A' if table1 == 0 else 'B'}) and {person2} (Table {'A' if table2 == 0 else 'B'}) separated")
        
        self.performance_metrics['verification_time'] = time.time() - start_time
        
        if conflicts == 0:
            print("Verification PASSED! No conflicts detected!")
            return True
        else:
            print(f"Verification FAILED! {conflicts} conflicts detected!")
            return False
    
    def export_to_csv(self, save_filename="./asignment2/part2/seating_results.csv"):
        """Export seating arrangement and performance metrics to CSV."""
        
        # Create guest seating data
        guest_data = []
        if self.seating_solution:
            for guest in self.guests:
                table_assignment = self.seating_solution[guest]
                table_name = 'Table A' if table_assignment == 0 else 'Table B'
                
                # Count animosities for this guest
                animosity_count = len([enemy for enemy in self.enemy_graph.get(guest, [])])
                
                guest_data.append({
                    'Guest_Name': guest,
                    'Table_Assignment': table_name,
                    'Table_Number': table_assignment,
                    'Animosity_Count': animosity_count
                })
        else:
            # No solution found
            for guest in self.guests:
                animosity_count = len([enemy for enemy in self.enemy_graph.get(guest, [])])
                guest_data.append({
                    'Guest_Name': guest,
                    'Table_Assignment': 'No Solution',
                    'Table_Number': -1,
                    'Animosity_Count': animosity_count
                })
        
        # Save guest seating data
        guest_csv_filename = save_filename.replace('.csv', '_guests.csv')
        with open(guest_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Guest_Name', 'Table_Assignment', 'Table_Number', 'Animosity_Count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(guest_data)
        
        # Create animosity relationship data
        animosity_data = []
        for person1, person2 in self.animosities:
            if self.seating_solution and person1 in self.seating_solution and person2 in self.seating_solution:
                table1 = 'Table A' if self.seating_solution[person1] == 0 else 'Table B'
                table2 = 'Table A' if self.seating_solution[person2] == 0 else 'Table B'
                resolved = table1 != table2
            else:
                table1 = table2 = 'Unknown'
                resolved = False
            
            animosity_data.append({
                'Person_1': person1,
                'Person_2': person2,
                'Person_1_Table': table1,
                'Person_2_Table': table2,
                'Conflict_Resolved': resolved
            })
        
        # Save animosity data
        animosity_csv_filename = save_filename.replace('.csv', '_animosities.csv')
        with open(animosity_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Person_1', 'Person_2', 'Person_1_Table', 'Person_2_Table', 'Conflict_Resolved']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(animosity_data)
        
        # Create performance metrics data
        performance_data = [self.performance_metrics]
        
        # Save performance data
        performance_csv_filename = save_filename.replace('.csv', '_performance.csv')
        with open(performance_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(self.performance_metrics.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(performance_data)
        
        print(f"CSV files exported:")
        print(f"   • Guest seating: {guest_csv_filename}")
        print(f"   • Animosities: {animosity_csv_filename}")
        print(f"   • Performance: {performance_csv_filename}")
        
        return guest_csv_filename, animosity_csv_filename, performance_csv_filename
    
    def visualize_party_arrangement(self, save_filename="./asignment2/part2/aunt_petunia_seating.png"):
        """Create a visual representation of the party seating arrangement."""
        if not self.guests:
            print("No guests to visualize!")
            return
        
        # Create circular layout for guests
        import math
        num_guests = len(self.guests)
        positions = {}
        
        for i, guest in enumerate(self.guests):
            angle = 2 * math.pi * i / num_guests
            radius = 6
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions[guest] = (x, y)
        
        # Create the visualization
        plt.figure(figsize=(14, 12))
        
        # Draw animosity relationships as red lines
        for person1, person2 in self.animosities:
            if person1 in positions and person2 in positions:
                x1, y1 = positions[person1]
                x2, y2 = positions[person2]
                plt.plot([x1, x2], [y1, y2], 'red', alpha=0.6, linewidth=2, linestyle='--')
        
        # Draw guests as colored dots based on table assignment
        if self.seating_solution:
            table_a_guests = []
            table_b_guests = []
            
            for guest in self.guests:
                x, y = positions[guest]
                if self.seating_solution[guest] == 0:
                    table_a_guests.append((x, y))
                    plt.scatter(x, y, s=1200, c='lightblue', edgecolor='navy', linewidth=3, alpha=0.8)
                else:
                    table_b_guests.append((x, y))
                    plt.scatter(x, y, s=1200, c='lightgreen', edgecolor='darkgreen', linewidth=3, alpha=0.8)
                
                # Add guest names with background
                plt.annotate(guest, (x, y), ha='center', va='center', 
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                   edgecolor='gray', alpha=0.9))
            
            # Create legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                          markersize=15, label=f'Table A ({len([g for g in self.guests if self.seating_solution[g] == 0])} guests)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                          markersize=15, label=f'Table B ({len([g for g in self.guests if self.seating_solution[g] == 1])} guests)'),
                plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.6, 
                          label='Animosity (enemies)')
            ]
            plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
            
            title = "Aunt Petunia's Namesday Party - SEATING SUCCESS!"
        else:
            # All guests same color if no solution
            for guest in self.guests:
                x, y = positions[guest]
                plt.scatter(x, y, s=1200, c='lightcoral', edgecolor='darkred', linewidth=3, alpha=0.8)
                plt.annotate(guest, (x, y), ha='center', va='center', 
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                   edgecolor='gray', alpha=0.9))
            
            title = "Aunt Petunia's Party - No 2-Table Solution Possible"
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Party visualization saved as {save_filename}")
        plt.show()
    
    def run_complete_party_planning(self, guest_file, animosity_file):
        """Run the complete party planning process for Aunt Petunia."""
        print("Starting Aunt Petunia's namesday party planning process...\n")
        
        total_start_time = time.time()
        
        # Step 1: Load guest data
        if not self.load_guest_list(guest_file):
            return False
        if not self.load_animosity_list(animosity_file):
            return False
        
        # Step 2: Create the animosity graph
        self.create_animosity_graph()
        
        # Step 3: Solve using non-recursive DFS
        solution = self.solve_with_non_recursive_dfs()
        
        # Step 4: Display results
        self.display_seating_arrangement()
        
        # Step 5: Verify the solution
        if solution:
            self.verify_solution()
        
        # Calculate total time
        self.performance_metrics['total_time'] = time.time() - total_start_time
        
        # Step 6: Export to CSV
        print("\nExporting results to CSV...")
        self.export_to_csv()
        
        # Step 7: Create visualization
        self.visualize_party_arrangement()
        
        # Step 8: Display final performance summary
        self.display_performance_summary()
        
        return solution is not None
    
    def display_performance_summary(self):
        """Display a summary of performance metrics."""
        metrics = self.performance_metrics
        
        print("\n" + "=" * 60)
        print("           PERFORMANCE ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"\nPROBLEM COMPLEXITY:")
        print(f"   • Total guests: {metrics['guest_count']}")
        print(f"   • Animosity relationships: {metrics['animosity_count']}")
        print(f"   • Connected components: {metrics['connected_components']}")
        print(f"   • Graph density: {(2 * metrics['animosity_count']) / max(metrics['guest_count'] * (metrics['guest_count'] - 1), 1):.3f}")
        
        print(f"\nALGORITHM PERFORMANCE:")
        print(f"   • Total processing time: {metrics['total_time']:.4f} seconds")
        print(f"   • DFS algorithm time: {metrics['algorithm_time']:.4f} seconds")
        print(f"   • DFS iterations: {metrics['iterations']}")
        print(f"   • Time per guest: {(metrics['algorithm_time'] / max(metrics['guest_count'], 1)) * 1000:.3f} ms")
        print(f"   • Solution found: {'YES' if metrics['solution_found'] else 'NO'}")
        
        print(f"\nVERIFICATION:")
        print(f"   • Verification time: {metrics['verification_time']:.4f} seconds")
        
        efficiency_rating = "EXCELLENT" if metrics['algorithm_time'] < 0.001 else \
                          "GOOD" if metrics['algorithm_time'] < 0.01 else \
                          "ACCEPTABLE" if metrics['algorithm_time'] < 0.1 else "NEEDS OPTIMIZATION"
        
        print(f"\nEFFICIENCY RATING: {efficiency_rating}")
        print("=" * 60)


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("  AUNT PETUNIA'S NAMESDAY PARTY SEATING PLANNER")
    print("    Using Non-Recursive DFS Algorithm")
    print("    With Performance Analysis & CSV Export")
    print("=" * 60)
    print()
    
    # Create planner instance
    planner = AuntPetuniaSeatingPlanner()
    
    # Check if files exist, create samples if not
    guest_file = "./asignment2/part2/guest_list.txt"
    animosity_file = "./asignment2/part2/animosities.txt"
    
    if not os.path.exists(guest_file) or not os.path.exists(animosity_file):
        print("Sample party files not found. Creating them now...")
        print()
    
    # Run the complete party planning process
    success = planner.run_complete_party_planning(guest_file, animosity_file)
    
    if success:
        print("\nParty planning completed successfully!")
        print("Aunt Petunia can now enjoy her conflict-free namesday celebration!")
        print("Performance data and CSV exports are ready for analysis!")
    else:
        print("\nParty planning encountered issues.")
        print("Aunt Petunia may need to reconsider her guest list or seating options.")
        print("Check the performance analysis for insights!")