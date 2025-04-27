import time
import string
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class StringMatchingAlgorithms:
    def __init__(self):
        self.results = defaultdict(dict)

    def brute_force(self, text, pattern):
        """Brute Force string matching algorithm."""
        start_time = time.time()
        n, m = len(text), len(pattern)
        occurrences = []

        for i in range(n - m + 1):
            j = 0
            while j < m and text[i + j] == pattern[j]:
                j += 1
            if j == m:
                occurrences.append(i)

        return occurrences, time.time() - start_time

    def sunday(self, text, pattern):
        """Sunday string matching algorithm."""
        start_time = time.time()
        n, m = len(text), len(pattern)
        occurrences = []

        if m > n:
            return occurrences, time.time() - start_time

        # Preprocessing: compute shift table
        shift = {}
        for i in range(256):  # For all possible characters
            shift[chr(i)] = m + 1  # Default shift is pattern length + 1

        for i in range(m):
            shift[pattern[i]] = m - i  # Shift value for each character in pattern

        # Searching phase
        i = 0
        while i <= n - m:
            j = 0
            while j < m and text[i + j] == pattern[j]:
                j += 1
            
            if j == m:
                occurrences.append(i)
                
            # Calculate shift based on character after the current window
            if i + m < n:
                i += shift.get(text[i + m], m + 1)
            else:
                break

        return occurrences, time.time() - start_time

    def kmp(self, text, pattern):
        """Knuth-Morris-Pratt string matching algorithm."""
        start_time = time.time()
        n, m = len(text), len(pattern)
        occurrences = []

        if m == 0:
            return occurrences, time.time() - start_time

        # Preprocessing: compute LPS array (longest proper prefix which is also suffix)
        lps = [0] * m
        length = 0
        i = 1

        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1

        # Searching phase
        i = j = 0
        while i < n:
            if pattern[j] == text[i]:
                i += 1
                j += 1
            
            if j == m:
                occurrences.append(i - j)
                j = lps[j - 1]
            elif i < n and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1

        return occurrences, time.time() - start_time

    def build_fsm(self, pattern, alphabet):
        """Build Finite State Machine for FSM algorithm."""
        m = len(pattern)
        fsm = [{} for _ in range(m + 1)]
        
        for state in range(m + 1):
            for char in alphabet:
                # Calculate next state for each character from current state
                k = min(m, state + 1)
                while k > 0 and pattern[:state] + char != pattern[k-state:k]:
                    k -= 1
                fsm[state][char] = k
                
        return fsm

    def fsm(self, text, pattern):
        """Finite State Machine string matching algorithm."""
        start_time = time.time()
        n, m = len(text), len(pattern)
        occurrences = []

        if m == 0:
            return occurrences, time.time() - start_time

        # Create alphabet from pattern and text
        alphabet = set(pattern + text)
        
        # Build FSM transition table
        fsm = self.build_fsm(pattern, alphabet)
        
        # Searching phase
        state = 0
        for i in range(n):
            if text[i] in fsm[state]:
                state = fsm[state][text[i]]
            else:
                state = 0
            
            if state == m:
                occurrences.append(i - m + 1)

        return occurrences, time.time() - start_time

    def rabin_karp(self, text, pattern):
        """Rabin-Karp string matching algorithm."""
        start_time = time.time()
        n, m = len(text), len(pattern)
        occurrences = []

        if m == 0 or m > n:
            return occurrences, time.time() - start_time

        # Choose a prime number for the hash function
        prime = 101
        # Choose a base for the hash function (alphabet size)
        base = 256
        
        # Function to compute the hash value
        def hash_value(s, length):
            result = 0
            for i in range(length):
                result = (base * result + ord(s[i])) % prime
            return result
        
        # Calculate the hash value for the pattern and first window of text
        pattern_hash = hash_value(pattern, m)
        text_hash = hash_value(text, m)
        
        # Calculate base^(m-1) % prime for rolling hash
        h = 1
        for _ in range(m - 1):
            h = (h * base) % prime
        
        # Slide the pattern over the text
        for i in range(n - m + 1):
            # Check if the hash values match
            if pattern_hash == text_hash:
                # Verify character by character
                match = True
                for j in range(m):
                    if text[i + j] != pattern[j]:
                        match = False
                        break
                if match:
                    occurrences.append(i)
            
            # Calculate hash value for the next window
            if i < n - m:
                text_hash = (base * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
                if text_hash < 0:
                    text_hash += prime

        return occurrences, time.time() - start_time

    def compute_z_array(self, s):
        """Compute Z array for Gusfield Z algorithm."""
        n = len(s)
        z = [0] * n
        
        # Initialize: Z[0] is meaningless
        l, r = 0, 0
        for i in range(1, n):
            if i <= r:
                # If i is within the current Z-box, use precomputed values
                z[i] = min(r - i + 1, z[i - l])
            
            # Explicitly compute Z[i] by comparing characters
            while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                z[i] += 1
            
            # Update Z-box if needed
            if i + z[i] - 1 > r:
                l, r = i, i + z[i] - 1
                
        return z

    def gusfield_z(self, text, pattern):
        """Gusfield Z algorithm for string matching."""
        start_time = time.time()
        occurrences = []
        
        # Concatenate pattern, a special character, and text
        concat = pattern + "$" + text
        
        # Compute Z array
        z = self.compute_z_array(concat)
        
        # Check for matches
        for i in range(len(pattern) + 1, len(concat)):
            if z[i] == len(pattern):
                occurrences.append(i - len(pattern) - 1)
                
        return occurrences, time.time() - start_time

    def run_comparison(self, text, pattern, algorithms=None):
        """Run comparison of specified algorithms."""
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
        """Generate random text of specified size in KB."""
        chars = string.ascii_letters + string.digits + string.punctuation + ' \n\t'
        size_chars = size_kb * 1024
        return ''.join(random.choice(chars) for _ in range(size_chars))

    def extract_book_chapters(self, filepath, num_chapters=20):
        """Extract chapters from a book file for testing."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple chapter extraction (this may need customization based on the book format)
            chapters = content.split('\n\nCHAPTER ')[1:]  # Skip first split which is before Chapter 1
            
            # Take up to num_chapters or all available chapters
            return ['CHAPTER ' + chapter for chapter in chapters[:num_chapters]]
        except Exception as e:
            print(f"Error reading book file: {e}")
            return []

    def run_and_save_timing_tests(self, text_samples, patterns, output_filename="timing_results.csv"):
        """Run timing tests and save results to CSV."""
        algorithms = ["brute_force", "sunday", "kmp", "fsm", "rabin_karp", "gusfield_z"]
        
        results = []
        
        for i, text in enumerate(text_samples):
            text_size = len(text) / 1024  # Size in KB
            
            for pattern_name, pattern in patterns.items():
                print(f"\nRunning test for text #{i+1} ({text_size:.2f}KB) with pattern '{pattern_name}'")
                
                for algo_name in algorithms:
                    algo = getattr(self, algo_name)
                    occurrences, runtime = algo(text, pattern)
                    
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
        
        # Save results to CSV
        with open(output_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
        return results

    def plot_results(self, results, output_filename="comparison_plot.png"):
        """Plot comparison results."""
        # Group by algorithm and text size
        grouped_data = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            algo = result["algorithm"]
            text_size = result["text_size_kb"]
            runtime = result["runtime"]
            pattern = result["pattern"]
            
            grouped_data[(algo, pattern)][text_size].append(runtime)
        
        # Calculate average runtime for each algorithm and text size
        avg_data = defaultdict(lambda: {"sizes": [], "runtimes": []})
        
        for (algo, pattern), size_data in grouped_data.items():
            key = f"{algo} ({pattern})"
            sizes = sorted(size_data.keys())
            avg_data[key]["sizes"] = sizes
            avg_data[key]["runtimes"] = [np.mean(size_data[size]) for size in sizes]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
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
            plt.yscale('log')
            
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.show()
        
        return output_filename

    def wacky_races(self):
        """Design specific test cases to demonstrate specific performance relationships."""
        print("\n=== WACKY RACES: Empirical Performance Comparison ===")
        
        # Case 1: Binary Sunday is at least twice as fast as Gusfield Z
        print("\nCase 1: Binary Sunday vs Gusfield Z")
        # Create a text with many repeating characters but few matches
        # Sunday algorithm performs well when it can skip many characters
        text_sunday_vs_z = "a" * 50000 + "b" * 50000 + "c" * 1000
        pattern_sunday_vs_z = "cb" * 10  # Pattern that appears only near the end
        
        _, sunday_time = self.sunday(text_sunday_vs_z, pattern_sunday_vs_z)
        _, z_time = self.gusfield_z(text_sunday_vs_z, pattern_sunday_vs_z)
        
        print(f"Sunday algorithm: {sunday_time:.6f} seconds")
        print(f"Gusfield Z algorithm: {z_time:.6f} seconds")
        print(f"Ratio (Z/Sunday): {z_time/sunday_time:.2f}x")
        
        # Case 2: KMP is at least twice as fast as Rabin-Karp
        print("\nCase 2: KMP vs Rabin-Karp")
        # Create a text where hash collisions might slow down Rabin-Karp
        # KMP can skip comparisons efficiently using its prefix function
        text_kmp_vs_rk = "ab" * 50000 + "c" + "ab" * 1000
        pattern_kmp_vs_rk = "ab" * 15 + "c"  # Pattern with repeating prefix
        
        _, kmp_time = self.kmp(text_kmp_vs_rk, pattern_kmp_vs_rk)
        _, rk_time = self.rabin_karp(text_kmp_vs_rk, pattern_kmp_vs_rk)
        
        print(f"KMP algorithm: {kmp_time:.6f} seconds")
        print(f"Rabin-Karp algorithm: {rk_time:.6f} seconds")
        print(f"Ratio (RK/KMP): {rk_time/kmp_time:.2f}x")
        
        # Case 3: Rabin-Karp is at least twice as fast as Sunday
        print("\nCase 3: Rabin-Karp vs Sunday")
        # Create a text where Sunday's skip table isn't very effective
        # Rabin-Karp can quickly filter out non-matches using hash comparison
        alphabet = string.ascii_lowercase
        text_rk_vs_sunday = ''.join(random.choice(alphabet) for _ in range(100000))
        pattern_rk_vs_sunday = ''.join(random.choice(alphabet) for _ in range(20)) + "xyz"
        
        _, rk_time = self.rabin_karp(text_rk_vs_sunday, pattern_rk_vs_sunday)
        _, sunday_time = self.sunday(text_rk_vs_sunday, pattern_rk_vs_sunday)
        
        print(f"Rabin-Karp algorithm: {rk_time:.6f} seconds")
        print(f"Sunday algorithm: {sunday_time:.6f} seconds")
        print(f"Ratio (Sunday/RK): {sunday_time/rk_time:.2f}x")
        
        # Save the test cases to files for reproducibility
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
    matcher = StringMatchingAlgorithms()
    
    # Part A: "Mom bought me a new computer"
    print("=== PART A: 'Mom bought me a new computer' ===")
    
    # Generate texts of different sizes
    texts = []
    sizes_kb = [1, 5, 10, 20, 50, 100]
    
    for size in sizes_kb:
        text = matcher.generate_text(size)
        texts.append(text)
    
    # Define patterns for testing
    small_pattern = "computer"  # Small pattern
    large_pattern = "The quick brown fox jumps over the lazy dog. " * 3  # Larger pattern (paragraph)
    
    patterns = {
        "small": small_pattern,
        "large": large_pattern
    }
    
    # Run timing tests and save results
    results = matcher.run_and_save_timing_tests(texts, patterns)
    
    # Plot and save the comparison graph
    plot_file = matcher.plot_results(results)
    print(f"Comparison plot saved to {plot_file}")
    
    # Part B: "Wacky Races"
    print("\n=== PART B: 'Wacky Races' ===")
    wacky_results = matcher.wacky_races()
    
    return results, wacky_results

if __name__ == "__main__":
    main()