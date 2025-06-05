import time
import random
import string
import matplotlib.pyplot as plt
from collections import deque
import os

class WildcardStringMatching:
    """
    String matching with wildcard support using two different algorithms:
    
    1. Brute Force with Stack-based Backtracking
       - Uses explicit stack to handle '*' wildcard's multiple possibilities
       - Time Complexity: O(2^n) worst case due to exponential branching
       - Space Complexity: O(n) for stack storage
    
    2. Sunday Algorithm with Wildcard Support
       - Extends Sunday algorithm to handle wildcards
       - Uses greedy matching for '*' wildcards
       - Time Complexity: O(n*m) worst case, better average performance
       - Space Complexity: O(1)
    
    Wildcard Semantics:
    - '?' matches exactly one character
    - '*' matches zero or more characters
    - '\' escapes special characters (treats next char as literal)
    """
    
    def __init__(self):
        # Storage for benchmark results and test data
        self.results_brute_force = []    # Timing results for brute force
        self.results_sunday = []         # Timing results for Sunday algorithm
        self.patterns_brute_force = []   # Patterns used in brute force tests
        self.patterns_sunday = []        # Patterns used in Sunday tests
        self.texts_brute_force = []      # Texts used in brute force tests
        self.texts_sunday = []           # Texts used in Sunday tests

    def preprocess_pattern(self, pattern):
        """
        Parse pattern string and convert to structured format.
        
        Handles escape sequences and categorizes each element as:
        - ('literal', char): Regular character that must match exactly
        - ('wildcard', '?'): Single character wildcard
        - ('wildcard', '*'): Multi-character wildcard
        
        Example: "a?b*c" → [('literal','a'), ('wildcard','?'), ('literal','b'), 
                             ('wildcard','*'), ('literal','c')]
        """
        processed = []
        escape = False  # Flag to track if we're in escape mode
        
        for char in pattern:
            if escape:
                # Previous character was '\', treat this as literal
                processed.append(('literal', char))
                escape = False
            elif char == "\\":
                # Start escape sequence
                escape = True
            elif char == "?":
                # Single character wildcard
                processed.append(('wildcard', '?'))
            elif char == "*":
                # Multi-character wildcard
                processed.append(('wildcard', '*'))
            else:
                # Regular literal character
                processed.append(('literal', char))
        
        return processed

    def char_match(self, text_char, pattern_item):
        """
        Check if a single character matches a pattern element.
        
        Args:
            text_char: Character from text to match
            pattern_item: Tuple of (type, value) from processed pattern
            
        Returns:
            True if character matches pattern element, False otherwise
        """
        kind, value = pattern_item
        
        if kind == 'wildcard' and value == '?':
            # '?' matches any single character
            return True
        elif kind == 'literal':
            # Literal character must match exactly
            return text_char == value
        
        return False

    def brute_force_wildcard(self, text, pattern):
        """
        Brute force wildcard matching using explicit stack for backtracking.
        
        Algorithm:
        1. Convert pattern to structured format
        2. Use stack to explore all possible matches
        3. For '*' wildcards, try both:
           - Match zero characters (advance pattern only)
           - Match one+ characters (advance text, keep pattern)
        4. For other elements, advance both text and pattern if match
        
        Time Complexity: O(2^n) worst case - exponential due to '*' branching
        Space Complexity: O(n) for stack storage
        
        The stack stores (text_position, pattern_position) tuples representing
        all possible matching states to explore.
        """
        start_time = time.time()
        processed_pattern = self.preprocess_pattern(pattern)
        n, m = len(text), len(processed_pattern)
        found = False

        # Empty pattern matches empty text
        if m == 0:
            return True, time.time() - start_time

        # Quick optimization: if pattern has more non-wildcard chars than text,
        # it cannot possibly match
        effective_len = len([item for item in processed_pattern if item != ('wildcard', '*')])
        if effective_len > n:
            return False, time.time() - start_time

        # Stack for backtracking: stores (text_pos, pattern_pos) states
        stack = deque([(0, 0)])

        while stack:
            text_pos, pattern_pos = stack.pop()

            # Successfully matched entire pattern
            if pattern_pos == m:
                found = True
                break

            # Exceeded text bounds without completing pattern
            if text_pos > n:
                continue

            # Handle current pattern element
            if pattern_pos < m and processed_pattern[pattern_pos] == ('wildcard', '*'):
                # '*' wildcard: explore both possibilities
                # Option 1: '*' matches zero characters (advance pattern only)
                stack.append((text_pos, pattern_pos + 1))
                
                # Option 2: '*' matches one+ characters (advance text, keep pattern)
                if text_pos < n:
                    stack.append((text_pos + 1, pattern_pos))
                    
            elif (text_pos < n and pattern_pos < m and 
                  self.char_match(text[text_pos], processed_pattern[pattern_pos])):
                # Current character matches pattern element
                stack.append((text_pos + 1, pattern_pos + 1))

        return found, time.time() - start_time

    def sunday_wildcard(self, text, pattern):
        """
        Sunday algorithm extended to support wildcards with greedy matching.
        
        Algorithm:
        1. Process pattern to handle wildcards
        2. Try to match pattern at each position using greedy strategy
        3. For '*' wildcards, greedily consume characters until next pattern element matches
        4. Use Sunday's bad character heuristic for shifting when mismatch occurs
        
        Time Complexity: O(n*m) worst case, better average performance
        Space Complexity: O(1)
        
        Greedy Strategy for '*':
        - When encountering '*', look ahead to next pattern element
        - Consume text characters until finding one that matches next element
        - This is greedy (takes first match) but efficient
        """
        start_time = time.time()
        processed_pattern = self.preprocess_pattern(pattern)
        n, m = len(text), len(processed_pattern)
        i = 0  # Current position in text

        while i <= n:
            j = 0   # Current position in pattern
            ti = i  # Current text position for this match attempt
            
            # Try to match pattern starting at position i
            while j < m:
                if ti > n:
                    break
                    
                if processed_pattern[j] == ('wildcard', '*'):
                    # Handle '*' wildcard with greedy matching
                    if j == m - 1:
                        # '*' is last element in pattern - matches rest of text
                        ti = n
                        j = m
                        break
                    
                    # Look ahead to next pattern element after '*'
                    next_pattern = processed_pattern[j + 1]
                    
                    # Greedily consume characters until finding match for next element
                    while ti <= n:
                        if ti < n and self.char_match(text[ti], next_pattern):
                            break
                        ti += 1
                    j += 1  # Move past '*' wildcard
                    
                elif ti < n and self.char_match(text[ti], processed_pattern[j]):
                    # Current character matches current pattern element
                    ti += 1
                    j += 1
                else:
                    # Mismatch - break out of matching loop
                    break

            # Check if we successfully matched entire pattern
            if j == m:
                return True, time.time() - start_time

            # Check if we can continue searching
            if i + m >= n:
                break

            # Apply Sunday algorithm's bad character heuristic for shifting
            if i + m < n:
                next_char = text[i + m]  # Character after current window
                shift = 1  # Default shift
                
                # Look for rightmost occurrence of next_char in pattern
                for idx in reversed(range(m)):
                    kind, val = processed_pattern[idx]
                    if kind == 'literal' and val == next_char:
                        shift = m - idx
                        break
                i += shift
            else:
                break

        return False, time.time() - start_time

    def generate_random_text(self, length):
        """Generate random text of specified length using ASCII letters."""
        return ''.join(random.choices(string.ascii_letters, k=length))

    def generate_pattern_with_wildcards(self, base_pattern, num_wildcards):
        """
        Insert random wildcards into a base pattern.
        
        Args:
            base_pattern: Original pattern string
            num_wildcards: Number of characters to replace with wildcards
            
        Returns:
            Pattern string with some characters replaced by '*' or '?'
        """
        pattern = list(base_pattern)
        
        # Replace random positions with wildcards
        for _ in range(num_wildcards):
            if pattern:
                index = random.randint(0, len(pattern) - 1)
                pattern[index] = random.choice(["*", "?"])
                
        return ''.join(pattern)

    def benchmark(self, text_lengths, base_pattern, num_tests=10, max_wildcards=5):
        """
        Comprehensive benchmark of both algorithms across different text sizes.
        
        Args:
            text_lengths: List of text lengths to test
            base_pattern: Base pattern to modify with wildcards
            num_tests: Number of test runs per text length (for averaging)
            max_wildcards: Maximum number of wildcards to insert
            
        Process:
        1. For each text length, run multiple tests
        2. Generate random text and pattern with wildcards
        3. Time both algorithms on same inputs
        4. Store results for analysis and plotting
        """
        for length in text_lengths:
            avg_time_bf = 0   # Average time for brute force
            avg_time_sun = 0  # Average time for Sunday algorithm

            # Run multiple tests for statistical reliability
            for _ in range(num_tests):
                # Generate test data
                text = self.generate_random_text(length)
                pattern_with_wildcards = self.generate_pattern_with_wildcards(
                    base_pattern, random.randint(0, max_wildcards)
                )

                # Test both algorithms on same data
                found_bf, duration_bf = self.brute_force_wildcard(text, pattern_with_wildcards)
                found_sun, duration_sun = self.sunday_wildcard(text, pattern_with_wildcards)

                # Accumulate timing results
                avg_time_bf += duration_bf
                avg_time_sun += duration_sun

            # Calculate averages
            avg_time_bf /= num_tests
            avg_time_sun /= num_tests

            # Store results for plotting and analysis
            self.results_brute_force.append(avg_time_bf)
            self.results_sunday.append(avg_time_sun)
            self.patterns_brute_force.append(pattern_with_wildcards)
            self.patterns_sunday.append(pattern_with_wildcards)
            self.texts_brute_force.append(text)
            self.texts_sunday.append(text)

            print(f"Completed benchmarking for text length {length}")

    def save_results_to_csv(self, folder="./results"):
        """
        Save benchmark results to CSV files for further analysis.
        
        Creates separate files for each algorithm with columns:
        - Pattern: The wildcard pattern used
        - Text Length: Length of the test text
        - Time: Execution time in seconds
        """
        # Create results directory if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        # Save brute force results
        brute_file = os.path.join(folder, "brute_force.csv")
        with open(brute_file, "w") as f:
            f.write("Pattern,Text Length,Time\n")
            for p, t, r in zip(self.patterns_brute_force, self.texts_brute_force, self.results_brute_force):
                f.write(f"{p},{len(t)},{r}\n")
        print(f"Saved Brute-force results to {brute_file}")

        # Save Sunday algorithm results
        sunday_file = os.path.join(folder, "sunday.csv")
        with open(sunday_file, "w") as f:
            f.write("Pattern,Text Length,Time\n")
            for p, t, r in zip(self.patterns_sunday, self.texts_sunday, self.results_sunday):
                f.write(f"{p},{len(t)},{r}\n")
        print(f"Saved Sunday results to {sunday_file}")

    def plot_results(self, save_path="./results/benchmark_plot.png"):
        """
        Create visualization comparing performance of both algorithms.
        
        Plots execution time vs text length for both algorithms,
        showing how each scales with input size.
        """
        plt.figure(figsize=(10, 6))
        
        # Extract text lengths for x-axis
        text_lengths = [len(t) for t in self.texts_brute_force]
        
        # Plot both algorithms
        plt.plot(text_lengths, self.results_brute_force, 
                label="Brute-force", marker='o', linewidth=2)
        plt.plot(text_lengths, self.results_sunday, 
                label="Sunday", marker='x', linewidth=2)
        
        # Formatting
        plt.xlabel("Text Length")
        plt.ylabel("Average Time (seconds)")
        plt.title("Wildcard String Matching Benchmark")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved benchmark plot to {save_path}")
        plt.show()

if __name__ == "__main__":
    """
    Main execution: Benchmark wildcard string matching algorithms.
    
    Test Configuration:
    - Text lengths: 10, 50, 100, 200, 500 characters
    - Base pattern: "a?b*c" (contains both wildcard types)
    - 10 test runs per length for statistical reliability
    - Up to 5 random wildcards inserted per test
    """
    wsm = WildcardStringMatching()
    print("Starting benchmark...")
    
    # Run comprehensive benchmark
    wsm.benchmark(
        text_lengths=[10, 50, 100, 200, 500], 
        base_pattern="a?b*c", 
        num_tests=10
    )
    
    # Save and visualize results
    wsm.save_results_to_csv()
    wsm.plot_results()
    print("Benchmark complete.")