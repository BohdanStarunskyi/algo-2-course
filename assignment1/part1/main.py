import time
import string
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class StringMatchingAlgorithms:
    """
    A comprehensive class implementing 6 different string matching algorithms:
    1. Brute Force - O(nm) worst case
    2. Sunday Algorithm - O(nm) worst case, but good average performance
    3. Knuth-Morris-Pratt (KMP) - O(n+m) time complexity
    4. Finite State Machine (FSM) - O(n) after preprocessing
    5. Rabin-Karp - O(n+m) average case with rolling hash
    6. Gusfield Z Algorithm - O(n+m) using Z-array
    
    Where n = text length, m = pattern length
    """
    
    def __init__(self):
        # Dictionary to store results from different algorithms
        self.results = defaultdict(dict)

    def brute_force(self, text, pattern):
        """
        Brute Force string matching algorithm.
        
        Algorithm: Check every possible position in text for pattern match
        Time Complexity: O(n*m) where n=text length, m=pattern length
        Space Complexity: O(1)
        
        How it works:
        1. Try matching pattern at each position in text
        2. For each position, compare characters one by one
        3. If mismatch found, move to next position
        4. Continue until pattern fully matches or end of text
        """
        start_time = time.time()
        n, m = len(text), len(pattern)
        occurrences = []

        # Try every possible starting position in text
        for i in range(n - m + 1):
            j = 0
            # Compare pattern with text starting at position i
            while j < m and text[i + j] == pattern[j]:
                j += 1
            # If we matched the entire pattern
            if j == m:
                occurrences.append(i)

        return occurrences, time.time() - start_time

    def sunday(self, text, pattern):
        """
        Sunday string matching algorithm (Boyer-Moore variant).
        
        Algorithm: Uses bad character heuristic to skip characters
        Time Complexity: O(n*m) worst case, O(n) average case
        Space Complexity: O(alphabet_size)v
        
        How it works:
        1. Preprocess pattern to create shift table
        2. When mismatch occurs, look at character AFTER current window
        3. Skip ahead based on last occurrence of that character in pattern
        4. This allows skipping multiple characters at once
        """
        start_time = time.time()
        n, m = len(text), len(pattern)
        occurrences = []

        if m > n:
            return occurrences, time.time() - start_time

        # Preprocessing: compute shift table for all possible characters
        shift = {}
        # Initialize all characters to maximum shift (pattern length + 1)
        for i in range(256):  # ASCII characters
            shift[chr(i)] = m + 1

        # Calculate actual shift values for characters in pattern
        # shift[c] = distance from rightmost occurrence of c to end of pattern
        for i in range(m):
            shift[pattern[i]] = m - i

        # Searching phase
        i = 0
        while i <= n - m:
            j = 0
            # Try to match pattern at current position
            while j < m and text[i + j] == pattern[j]:
                j += 1
            
            if j == m:  # Full match found
                occurrences.append(i)
                
            # Calculate shift based on character AFTER current window
            # This is key difference from Boyer-Moore
            if i + m < n:
                i += shift.get(text[i + m], m + 1)
            else:
                break

        return occurrences, time.time() - start_time

    def kmp(self, text, pattern):
        """
        Knuth-Morris-Pratt string matching algorithm.
        
        Algorithm: Uses failure function to avoid redundant comparisons
        Time Complexity: O(n+m) - linear time
        Space Complexity: O(m) for LPS array
        
        How it works:
        1. Preprocess pattern to build LPS (Longest Proper Prefix Suffix) array
        2. LPS[i] = length of longest proper prefix of pattern[0..i] that is also suffix
        3. When mismatch occurs, use LPS to determine how far to shift pattern
        4. Never need to backtrack in text, only in pattern
        """
        start_time = time.time()
        n, m = len(text), len(pattern)
        occurrences = []

        if m == 0:
            return occurrences, time.time() - start_time

        # Preprocessing: compute LPS array (Longest Proper Prefix Suffix)
        # LPS[i] = length of longest proper prefix of pattern[0..i] which is also suffix
        lps = [0] * m
        length = 0  # Length of previous longest prefix suffix
        i = 1

        # Build LPS array
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    # Use previously computed LPS value
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1

        # Searching phase using LPS array
        i = j = 0  # i for text, j for pattern
        while i < n:
            if pattern[j] == text[i]:
                i += 1
                j += 1
            
            if j == m:  # Found complete match
                occurrences.append(i - j)
                j = lps[j - 1]  # Get next position using LPS
            elif i < n and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j - 1]  # Use LPS to avoid redundant comparisons
                else:
                    i += 1

        return occurrences, time.time() - start_time

    def build_fsm(self, pattern, alphabet):
        """
        Build Finite State Machine for FSM algorithm.
        
        Creates transition table where:
        - States represent how many characters of pattern have been matched
        - Each state has transitions for every character in alphabet
        - fsm[state][char] = next_state after reading char in current state
        """
        m = len(pattern)
        # Create FSM with m+1 states (0 to m, where m is accepting state)
        fsm = [{} for _ in range(m + 1)]
        
        for state in range(m + 1):
            for char in alphabet:
                # Calculate next state for each character from current state
                # Try longest possible suffix that matches prefix
                k = min(m, state + 1)
                while k > 0 and pattern[:state] + char != pattern[k-state:k]:
                    k -= 1
                fsm[state][char] = k
                
        return fsm

    def fsm(self, text, pattern):
        """
        Finite State Machine string matching algorithm.
        
        Algorithm: Build automaton that recognizes pattern, then run text through it
        Time Complexity: O(n) after O(m^3 * alphabet_size) preprocessing
        Space Complexity: O(m * alphabet_size)
        
        How it works:
        1. Build FSM where each state represents partial match progress
        2. State 0 = no match, state m = complete match
        3. Process each character of text through FSM
        4. When reach accepting state (m), pattern found
        """
        start_time = time.time()
        n, m = len(text), len(pattern)
        occurrences = []

        if m == 0:
            return occurrences, time.time() - start_time

        # Create alphabet from pattern and text for FSM construction
        alphabet = set(pattern + text)
        
        # Build FSM transition table
        fsm = self.build_fsm(pattern, alphabet)
        
        # Searching phase: run text through FSM
        state = 0
        for i in range(n):
            if text[i] in fsm[state]:
                state = fsm[state][text[i]]
            else:
                state = 0  # Reset to start state
            
            # If reached accepting state, found match
            if state == m:
                occurrences.append(i - m + 1)

        return occurrences, time.time() - start_time

    def rabin_karp(self, text, pattern):
        """
        Rabin-Karp string matching algorithm.
        
        Algorithm: Uses rolling hash to quickly filter potential matches
        Time Complexity: O(n+m) average case, O(n*m) worst case
        Space Complexity: O(1)
        
        How it works:
        1. Compute hash of pattern and first window of text
        2. Slide window across text, updating hash incrementally
        3. When hashes match, verify with character-by-character comparison
        4. Rolling hash allows O(1) hash updates per position
        """
        start_time = time.time()
        n, m = len(text), len(pattern)
        occurrences = []

        if m == 0 or m > n:
            return occurrences, time.time() - start_time

        # Hash function parameters
        prime = 101  # Prime number for modular arithmetic
        base = 256   # Base for polynomial hash (alphabet size)
        
        def hash_value(s, length):
            """Compute polynomial hash of string s with given length"""
            result = 0
            for i in range(length):
                result = (base * result + ord(s[i])) % prime
            return result
        
        # Calculate hash values for pattern and first window of text
        pattern_hash = hash_value(pattern, m)
        text_hash = hash_value(text, m)
        
        # Calculate base^(m-1) % prime for rolling hash formula
        h = 1
        for _ in range(m - 1):
            h = (h * base) % prime
        
        # Slide pattern over text using rolling hash
        for i in range(n - m + 1):
            # Check if hash values match (quick filter)
            if pattern_hash == text_hash:
                # Hash collision possible, verify with actual comparison
                match = True
                for j in range(m):
                    if text[i + j] != pattern[j]:
                        match = False
                        break
                if match:
                    occurrences.append(i)
            
            # Calculate rolling hash for next window
            # Remove leftmost character and add new rightmost character
            if i < n - m:
                text_hash = (base * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
                # Handle negative values
                if text_hash < 0:
                    text_hash += prime

        return occurrences, time.time() - start_time

    def compute_z_array(self, s):
        """
        Compute Z array for Gusfield Z algorithm.
        
        Z[i] = length of longest substring starting from s[i] that matches prefix of s
        Uses Z-box optimization to compute efficiently.
        """
        n = len(s)
        z = [0] * n
        
        # Z[0] is undefined (entire string matches itself)
        # l, r define the Z-box: rightmost segment where Z[i] was computed
        l, r = 0, 0
        
        for i in range(1, n):
            if i <= r:
                # i is within current Z-box, use previously computed information
                # Z[i-l] is the corresponding position in the prefix
                z[i] = min(r - i + 1, z[i - l])
            
            # Explicitly extend match as far as possible
            while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                z[i] += 1
            
            # Update Z-box if current match extends further right
            if i + z[i] - 1 > r:
                l, r = i, i + z[i] - 1
                
        return z

    def gusfield_z(self, text, pattern):
        """
        Gusfield Z algorithm for string matching.
        
        Algorithm: Uses Z-array on concatenated string to find matches
        Time Complexity: O(n+m) - linear time
        Space Complexity: O(n+m) for concatenated string and Z-array
        
        How it works:
        1. Create string: pattern + "$" + text ($ is separator)
        2. Compute Z-array for this concatenated string
        3. Z[i] = m means pattern matches text starting at position i-m-1
        4. Separator ensures pattern doesn't match across boundary
        """
        start_time = time.time()
        occurrences = []
        
        # Concatenate pattern, separator, and text
        # Separator prevents matches across pattern-text boundary
        concat = pattern + "$" + text
        
        # Compute Z array for concatenated string
        z = self.compute_z_array(concat)
        
        # Check for matches: Z[i] = m means pattern matches at position i-m-1 in text
        for i in range(len(pattern) + 1, len(concat)):
            if z[i] == len(pattern):
                # Found match at position i-len(pattern)-1 in original text
                occurrences.append(i - len(pattern) - 1)
                
        return occurrences, time.time() - start_time

    def run_comparison(self, text, pattern, algorithms=None):
        """
        Run comparison of specified algorithms on given text and pattern.
        Returns results dictionary with occurrences and runtime for each algorithm.
        """
        if algorithms is None:
            algorithms = ["brute_force", "sunday", "kmp", "fsm", "rabin_karp", "gusfield_z"]
            
        results = {}
        
        for algo_name in algorithms:
            algo = getattr(self, algo_name)
            occurrences, runtime = algo(text, pattern)
            results[algo_name] = {"occurrences": occurrences, "runtime": runtime}
            print(f"{algo_name}: Found {len(occurrences)} occurrences in {runtime:.6f} seconds")
            
        return results

    def generate_text(self, size_kb):
        """Generate random text of specified size in KB for testing."""
        chars = string.ascii_letters + string.digits + string.punctuation + ' \n\t'
        size_chars = size_kb * 1024  # Convert KB to characters
        return ''.join(random.choice(chars) for _ in range(size_chars))

    def extract_book_chapters(self, filepath, num_chapters=20):
        """
        Extract chapters from a book file for testing.
        Simple implementation that splits on 'CHAPTER' markers.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple chapter extraction (may need customization for different formats)
            chapters = content.split('\n\nCHAPTER ')[1:]  # Skip content before first chapter
            
            # Return up to num_chapters or all available chapters
            return ['CHAPTER ' + chapter for chapter in chapters[:num_chapters]]
        except Exception as e:
            print(f"Error reading book file: {e}")
            return []

    def run_and_save_timing_tests(self, text_samples, patterns, output_filename="timing_results.csv"):
        """
        Run comprehensive timing tests and save results to CSV.
        Tests all algorithms on all text samples with all patterns.
        """
        algorithms = ["brute_force", "sunday", "kmp", "fsm", "rabin_karp", "gusfield_z"]
        results = []
        
        # Test each text sample
        for i, text in enumerate(text_samples):
            text_size = len(text) / 1024  # Convert to KB
            
            # Test each pattern on current text
            for pattern_name, pattern in patterns.items():
                print(f"\nRunning test for text #{i+1} ({text_size:.2f}KB) with pattern '{pattern_name}'")
                
                # Test each algorithm
                for algo_name in algorithms:
                    algo = getattr(self, algo_name)
                    occurrences, runtime = algo(text, pattern)
                    
                    # Store result
                    result = {
                        "text_id": i + 1,
                        "text_size_kb": text_size,
                        "pattern": pattern_name,
                        "algorithm": algo_name,
                        "occurrences": len(occurrences),
                        "runtime": runtime
                    }
                    
                    results.append(result)
                    print(f"{algo_name}: Found {len(occurrences)} occurrences in {runtime:.6f} seconds")
        
        # Save all results to CSV file
        with open(output_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
        return results

    def plot_results(self, results, output_filename="comparison_plot.png"):
        """
        Create visualization of algorithm performance comparison.
        Plots runtime vs text size for each algorithm and pattern combination.
        """
        # Group results by algorithm and pattern
        grouped_data = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            algo = result["algorithm"]
            text_size = result["text_size_kb"]
            runtime = result["runtime"]
            pattern = result["pattern"]
            
            grouped_data[(algo, pattern)][text_size].append(runtime)
        
        # Calculate average runtime for each algorithm/pattern at each text size
        avg_data = defaultdict(lambda: {"sizes": [], "runtimes": []})
        
        for (algo, pattern), size_data in grouped_data.items():
            key = f"{algo} ({pattern})"
            sizes = sorted(size_data.keys())
            avg_data[key]["sizes"] = sizes
            avg_data[key]["runtimes"] = [np.mean(size_data[size]) for size in sizes]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot each algorithm/pattern combination
        for key, data in avg_data.items():
            plt.plot(data["sizes"], data["runtimes"], marker='o', linestyle='-', label=key)
        
        plt.xlabel('Text Size (KB)')
        plt.ylabel('Runtime (seconds)')
        plt.title('String Matching Algorithm Performance Comparison')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Use logarithmic scale if runtimes vary significantly
        max_runtime = max(max(data["runtimes"]) for data in avg_data.values())
        min_runtime = min(min(data["runtimes"]) for data in avg_data.values())
        
        if max_runtime / min_runtime > 100:
            plt.yscale('log')  # Log scale for better visualization
            
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.show()
        
        return output_filename

    def wacky_races(self):
        """
        Design specific test cases to demonstrate performance relationships.
        Creates scenarios where certain algorithms outperform others by at least 2x.
        """
        print("\n=== WACKY RACES: Empirical Performance Comparison ===")
        
        # Case 1: Sunday vs Gusfield Z
        # Sunday excels when it can skip many characters
        # Create text where Sunday's bad character heuristic is very effective
        print("\nCase 1: Sunday vs Gusfield Z")
        text_sunday_vs_z = "a" * 50000 + "b" * 50000 + "c" * 1000
        pattern_sunday_vs_z = "cb" * 10  # Pattern appears only at end
        
        _, sunday_time = self.sunday(text_sunday_vs_z, pattern_sunday_vs_z)
        _, z_time = self.gusfield_z(text_sunday_vs_z, pattern_sunday_vs_z)
        
        print(f"Sunday algorithm: {sunday_time:.6f} seconds")
        print(f"Gusfield Z algorithm: {z_time:.6f} seconds")
        print(f"Ratio (Z/Sunday): {z_time/sunday_time:.2f}x")
        
        # Case 2: KMP vs Rabin-Karp
        # KMP excels with patterns that have self-repeating prefixes
        # Create scenario where KMP's failure function provides major advantage
        print("\nCase 2: KMP vs Rabin-Karp")
        text_kmp_vs_rk = "ab" * 50000 + "c" + "ab" * 1000
        pattern_kmp_vs_rk = "ab" * 15 + "c"  # Long repeating prefix
        
        _, kmp_time = self.kmp(text_kmp_vs_rk, pattern_kmp_vs_rk)
        _, rk_time = self.rabin_karp(text_kmp_vs_rk, pattern_kmp_vs_rk)
        
        print(f"KMP algorithm: {kmp_time:.6f} seconds")
        print(f"Rabin-Karp algorithm: {rk_time:.6f} seconds")
        print(f"Ratio (RK/KMP): {rk_time/kmp_time:.2f}x")
        
        # Case 3: Rabin-Karp vs Sunday
        # Rabin-Karp excels when hash filtering eliminates many comparisons
        # Create random text where Sunday's skip table isn't very effective
        print("\nCase 3: Rabin-Karp vs Sunday")
        alphabet = string.ascii_lowercase
        text_rk_vs_sunday = ''.join(random.choice(alphabet) for _ in range(100000))
        pattern_rk_vs_sunday = ''.join(random.choice(alphabet) for _ in range(20)) + "xyz"
        
        _, rk_time = self.rabin_karp(text_rk_vs_sunday, pattern_rk_vs_sunday)
        _, sunday_time = self.sunday(text_rk_vs_sunday, pattern_rk_vs_sunday)
        
        print(f"Rabin-Karp algorithm: {rk_time:.6f} seconds")
        print(f"Sunday algorithm: {sunday_time:.6f} seconds")
        print(f"Ratio (Sunday/RK): {sunday_time/rk_time:.2f}x")
        
        # Save test cases to files for reproducibility
        with open("test_case_sunday_vs_z.txt", "w") as f:
            f.write(f"Text: {text_sunday_vs_z[:100]}... (length: {len(text_sunday_vs_z)})\n")
            f.write(f"Pattern: {pattern_sunday_vs_z}\n")
            
        with open("test_case_kmp_vs_rk.txt", "w") as f:
            f.write(f"Text: {text_kmp_vs_rk[:100]}... (length: {len(text_kmp_vs_rk)})\n")
            f.write(f"Pattern: {pattern_kmp_vs_rk}\n")
            
        with open("test_case_rk_vs_sunday.txt", "w") as f:
            f.write(f"Text: {text_rk_vs_sunday[:100]}... (length: {len(text_rk_vs_sunday)})\n")
            f.write(f"Pattern: {pattern_rk_vs_sunday}\n")
            
        return {
            "sunday_vs_z": {"text": text_sunday_vs_z, "pattern": pattern_sunday_vs_z, 
                           "sunday_time": sunday_time, "z_time": z_time},
            "kmp_vs_rk": {"text": text_kmp_vs_rk, "pattern": pattern_kmp_vs_rk,
                         "kmp_time": kmp_time, "rk_time": rk_time},
            "rk_vs_sunday": {"text": text_rk_vs_sunday, "pattern": pattern_rk_vs_sunday,
                           "rk_time": rk_time, "sunday_time": sunday_time}
        }

def main():
    """
    Main function that demonstrates the string matching algorithm comparison.
    
    Part A: Performance comparison across different text sizes and patterns
    Part B: "Wacky Races" - specific scenarios showing algorithm strengths
    """
    matcher = StringMatchingAlgorithms()
    
    # Part A: Comprehensive performance testing
    print("=== PART A: 'Mom bought me a new computer' ===")
    
    # Generate test texts of increasing sizes
    texts = []
    sizes_kb = [1, 5, 10, 20, 50, 100]
    
    for size in sizes_kb:
        text = matcher.generate_text(size)
        texts.append(text)
    
    # Define test patterns
    small_pattern = "computer"  # Short pattern
    large_pattern = "The quick brown fox jumps over the lazy dog. " * 3  # Long pattern
    
    patterns = {
        "small": small_pattern,
        "large": large_pattern
    }
    
    # Run comprehensive timing tests
    results = matcher.run_and_save_timing_tests(texts, patterns)
    
    # Create and save performance visualization
    plot_file = matcher.plot_results(results)
    print(f"Comparison plot saved to {plot_file}")
    
    # Part B: Targeted performance demonstrations
    print("\n=== PART B: 'Wacky Races' ===")
    wacky_results = matcher.wacky_races()
    
    return results, wacky_results

if __name__ == "__main__":
    main()