import time
import os
import matplotlib.pyplot as plt
from collections import defaultdict


class SpellChecker:
    """
    A comprehensive spell checker that implements and benchmarks four different data structures:
    1. Naive List - Simple linear search through a list
    2. BBST (Balanced Binary Search Tree) - Using sortedcontainers.SortedSet
    3. Trie - Prefix tree structure for efficient string searching
    4. HashMap - Hash table using Python's built-in set
    
    Each implementation is benchmarked for both dictionary building time and spell checking time.
    """
    
    def __init__(self):
        """
        Initialize the SpellChecker with empty data structures to store benchmark results.
        These will be used to track performance metrics across different text sizes.
        """
        # Dictionary to store build times for each data structure
        # Key: algorithm name, Value: list of build times for different tests
        self.build_times = {
            "naive": [],      # Linear list build times
            "bbst": [],       # Balanced BST build times  
            "trie": [],       # Trie construction times
            "hashmap": []     # Hash table build times
        }
        
        # Dictionary to store spell checking times for each data structure
        # Key: algorithm name, Value: list of check times for different text sizes
        self.check_times = {
            "naive": [],      # Linear search times
            "bbst": [],       # BST search times
            "trie": [],       # Trie traversal times
            "hashmap": []     # Hash lookup times
        }
        
        # List to store the length (word count) of each test text
        # Used for plotting performance vs. input size
        self.text_lengths = []

    def load_dictionary(self, dictionary_file):
        """
        Load dictionary from a text file and return a list of words.
        
        Algorithm: Simple file I/O with error handling
        Time Complexity: O(n) where n is the number of words in the file
        Space Complexity: O(n) to store all words in memory
        
        Args:
            dictionary_file (str): Path to the dictionary file
            
        Returns:
            list: List of dictionary words in lowercase, or empty list if error
        """
        try:
            with open(dictionary_file, 'r') as f:
                # Read all lines, strip whitespace, convert to lowercase
                # This normalization ensures consistent comparisons later
                return [word.strip().lower() for word in f.readlines()]
        except FileNotFoundError:
            print(f"Error: Dictionary file '{dictionary_file}' not found.")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Please ensure the file exists at the specified location.")
            return []
        except Exception as e:
            print(f"Error loading dictionary file: {e}")
            return []

    # ==================== Naive List Implementation ====================
    # Algorithm: Linear Search
    # Build Time: O(1) - just assigns reference
    # Search Time: O(n) - must check every word in worst case
    # Space: O(n) - stores all dictionary words
    
    def build_naive_list(self, dictionary_words):
        """
        Build a naive list dictionary - simply uses the input list as-is.
        
        This is the simplest approach but has poor search performance.
        The "build" time is essentially zero since we just assign the reference.
        
        Time Complexity: O(1) - no actual building required
        Space Complexity: O(1) additional space (shares reference with input)
        """
        start_time = time.time()
        # No actual building required - just use the list directly
        naive_dict = dictionary_words
        end_time = time.time()
        return naive_dict, end_time - start_time

    def check_word_naive(self, word, naive_dict):
        """
        Check if a word exists using linear search through the list.
        
        Algorithm: Linear Search
        - Iterate through entire list until word is found or end is reached
        - Python's 'in' operator on lists uses linear search internally
        
        Time Complexity: O(n) where n is dictionary size
        Space Complexity: O(1)
        
        Args:
            word (str): Word to check
            naive_dict (list): Dictionary as a list
            
        Returns:
            bool: True if word exists, False otherwise
        """
        return word.lower() in naive_dict

    def check_text_naive(self, text, naive_dict):
        """
        Check all words in a text using the naive linear search approach.
        
        Algorithm: For each word in text, perform linear search
        Time Complexity: O(m * n) where m is text length, n is dictionary size
        
        Args:
            text (str): Text to spell check
            naive_dict (list): Dictionary as a list
            
        Returns:
            tuple: (list of boolean results, time taken)
        """
        start_time = time.time()
        words = text.split()  # Split text into individual words
        
        # Check each word, removing common punctuation first
        # strip() removes punctuation that commonly appears at word boundaries
        results = [self.check_word_naive(word.strip('.,!?"\'():;-'), naive_dict) for word in words]
        
        end_time = time.time()
        return results, end_time - start_time

    # ==================== BBST Implementation ====================
    # Algorithm: Balanced Binary Search Tree
    # Build Time: O(n log n) - insert each word maintaining balance
    # Search Time: O(log n) - binary search through balanced tree
    # Space: O(n) - stores all dictionary words in tree structure
    
    def build_bbst(self, dictionary_words):
        """
        Build a balanced binary search tree using Python's sortedcontainers.SortedSet.
        
        Algorithm: Self-balancing BST (typically Red-Black or AVL tree internally)
        - Maintains sorted order while keeping tree balanced
        - Guarantees O(log n) operations by preventing degenerate cases
        
        Time Complexity: O(n log n) - insert n words, each taking O(log n)
        Space Complexity: O(n) - tree nodes plus overhead
        """
        try:
            import sortedcontainers
            start_time = time.time()
            
            # SortedSet maintains elements in sorted order with balanced tree
            # Provides O(log n) search, insert, delete operations
            bbst_dict = sortedcontainers.SortedSet(dictionary_words)
            
            end_time = time.time()
            return bbst_dict, end_time - start_time
        except ImportError:
            print("Error: sortedcontainers package not found.")
            print("Please install it with: pip install sortedcontainers")
            # Fallback to regular set (which uses hashing, not BST)
            return set(dictionary_words), 0

    def check_word_bbst(self, word, bbst_dict):
        """
        Check if a word exists using balanced binary search tree lookup.
        
        Algorithm: Binary Search Tree Traversal
        - Start at root, compare word with current node
        - Go left if word < current, right if word > current
        - Continue until word found or null node reached
        
        Time Complexity: O(log n) - tree height is log n when balanced
        Space Complexity: O(1) - only uses stack space for recursion
        """
        return word.lower() in bbst_dict

    def check_text_bbst(self, text, bbst_dict):
        """
        Check all words in text using BBST lookup.
        
        Time Complexity: O(m log n) where m is text length, n is dictionary size
        Much better than naive O(m * n) approach for large dictionaries
        """
        start_time = time.time()
        words = text.split()
        results = [self.check_word_bbst(word.strip('.,!?"\'():;-'), bbst_dict) for word in words]
        end_time = time.time()
        return results, end_time - start_time

    # ==================== Trie Implementation ====================
    # Algorithm: Prefix Tree (Trie)
    # Build Time: O(sum of word lengths) - insert each character once
    # Search Time: O(word length) - follow path character by character
    # Space: O(sum of word lengths) - but shares common prefixes
    
    class TrieNode:
        """
        Node structure for Trie (Prefix Tree).
        
        Each node represents a character position and contains:
        - children: dictionary mapping characters to child nodes
        - is_end_of_word: boolean flag indicating if this node ends a valid word
        
        This structure allows sharing of common prefixes, making it space-efficient
        for dictionaries with many words sharing prefixes.
        """
        def __init__(self):
            # Dictionary mapping character -> TrieNode
            # Only creates child nodes as needed (sparse representation)
            self.children = {}
            
            # Flag indicating whether this node represents the end of a valid word
            # Distinguishes between prefixes and complete words
            self.is_end_of_word = False

    def build_trie(self, dictionary_words):
        """
        Build a Trie (Prefix Tree) from dictionary words.
        
        Algorithm: Trie Construction
        - Start with empty root node
        - For each word, traverse/create path character by character
        - Mark final node as end-of-word
        
        Advantages:
        - Excellent for prefix matching and autocomplete
        - Shares storage for common prefixes
        - Search time depends only on word length, not dictionary size
        
        Time Complexity: O(sum of all word lengths)
        Space Complexity: O(sum of all word lengths) worst case, 
                         but often much better due to prefix sharing
        """
        start_time = time.time()
        root = self.TrieNode()  # Create empty root node
        
        # Insert each word into the trie
        for word in dictionary_words:
            current = root  # Start at root for each word
            
            # Traverse/create path for each character in word
            for char in word:
                if char not in current.children:
                    # Create new node if path doesn't exist
                    current.children[char] = self.TrieNode()
                
                # Move to next node in path
                current = current.children[char]
            
            # Mark the final node as end of a valid word
            current.is_end_of_word = True
            
        end_time = time.time()
        return root, end_time - start_time

    def check_word_trie(self, word, trie_root):
        """
        Check if a word exists in the Trie.
        
        Algorithm: Trie Traversal
        - Start at root node
        - Follow path character by character
        - If any character path doesn't exist, word not in dictionary
        - If reach end of word, check if node is marked as end-of-word
        
        Time Complexity: O(word length) - independent of dictionary size!
        Space Complexity: O(1) - only uses a few variables
        
        This is the key advantage of tries: search time depends only on 
        the word length, not the size of the dictionary.
        """
        current = trie_root
        word = word.lower()
        
        # Follow the path character by character
        for char in word:
            if char not in current.children:
                # Path doesn't exist, word not in dictionary
                return False
            current = current.children[char]
            
        # Reached end of word - check if it's a valid complete word
        # This distinguishes "cat" from "ca" if only "cat" is in dictionary
        return current.is_end_of_word

    def check_text_trie(self, text, trie_root):
        """
        Check all words in text using Trie lookup.
        
        Time Complexity: O(sum of word lengths in text)
        This is often better than other approaches for long texts with short words.
        """
        start_time = time.time()
        words = text.split()
        results = [self.check_word_trie(word.strip('.,!?"\'():;-'), trie_root) for word in words]
        end_time = time.time()
        return results, end_time - start_time

    # ==================== HashMap Implementation ====================
    # Algorithm: Hash Table
    # Build Time: O(n) - hash each word and insert
    # Search Time: O(1) average case - direct hash lookup
    # Space: O(n) - stores all words with hash table overhead
    
    def build_hashmap(self, dictionary_words):
        """
        Build a hashmap dictionary using Python's built-in set.
        
        Algorithm: Hash Table Construction
        - Compute hash value for each word
        - Store in hash table with collision resolution
        - Python's set uses open addressing with random probing
        
        Advantages:
        - Fastest average-case lookup time O(1)
        - Simple to implement and understand
        - Good performance for most practical applications
        
        Time Complexity: O(n) average case for building
        Space Complexity: O(n) with some overhead for hash table structure
        """
        start_time = time.time()
        
        # Python's set implementation uses hash table internally
        # Provides O(1) average case lookup with good hash function
        hashmap_dict = set(dictionary_words)
        
        end_time = time.time()
        return hashmap_dict, end_time - start_time

    def check_word_hashmap(self, word, hashmap_dict):
        """
        Check if word exists using hash table lookup.
        
        Algorithm: Hash Table Lookup
        - Compute hash value of word
        - Look up hash table at computed index
        - Handle collisions using probing/chaining
        
        Time Complexity: O(1) average case, O(n) worst case (poor hash function)
        Space Complexity: O(1)
        
        In practice, this is usually the fastest method for spell checking.
        """
        return word.lower() in hashmap_dict

    def check_text_hashmap(self, text, hashmap_dict):
        """
        Check all words in text using hash table lookup.
        
        Time Complexity: O(m) average case where m is text length
        This is typically the fastest approach for large texts.
        """
        start_time = time.time()
        words = text.split()
        results = [self.check_word_hashmap(word.strip('.,!?"\'():;-'), hashmap_dict) for word in words]
        end_time = time.time()
        return results, end_time - start_time

    # ==================== Benchmarking ====================
    
    def benchmark(self, dictionary_file, text_files):
        """
        Comprehensive benchmark comparing all four data structures.
        
        This method:
        1. Loads the dictionary file
        2. Builds each data structure and measures build time
        3. Tests spell checking on various text files
        4. Records all timing data for analysis
        
        The benchmark tests both construction time and query time,
        which have different trade-offs for each data structure.
        """
        print("Loading dictionary...")
        dictionary_words = self.load_dictionary(dictionary_file)
        
        if not dictionary_words:
            print("Error: No words loaded from dictionary. Aborting benchmark.")
            return
            
        print(f"Successfully loaded {len(dictionary_words)} words from dictionary.")
        
        # ===== Dictionary Building Phase =====
        # Build each data structure and measure construction time
        print("Building dictionaries...")
        
        # Build naive list (essentially O(1) - just reference assignment)
        _, naive_build_time = self.build_naive_list(dictionary_words)
        naive_dict, _ = self.build_naive_list(dictionary_words)
        print(f"Naive list built successfully.")
        
        # Build balanced BST (O(n log n) construction time)
        _, bbst_build_time = self.build_bbst(dictionary_words)
        bbst_dict, _ = self.build_bbst(dictionary_words)
        print(f"BBST built successfully.")
        
        # Build trie (O(sum of word lengths) construction time)
        _, trie_build_time = self.build_trie(dictionary_words)
        trie_dict, _ = self.build_trie(dictionary_words)
        print(f"Trie built successfully.")
        
        # Build hashmap (O(n) average construction time)
        _, hashmap_build_time = self.build_hashmap(dictionary_words)
        hashmap_dict, _ = self.build_hashmap(dictionary_words)
        print(f"HashMap built successfully.")
        
        # Display build time results
        print("Dictionary build times:")
        print(f"Naive List: {naive_build_time:.6f} seconds")
        print(f"BBST: {bbst_build_time:.6f} seconds")
        print(f"Trie: {trie_build_time:.6f} seconds")
        print(f"HashMap: {hashmap_build_time:.6f} seconds")
        
        # ===== Text Checking Phase =====
        # Test spell checking performance on various text sizes
        print("\nRunning spell check benchmarks...")
        for text_file in text_files:
            try:
                with open(text_file, 'r') as f:
                    text = f.read()
                
                text_word_count = len(text.split())
                self.text_lengths.append(text_word_count)
                print(f"\nChecking text with {text_word_count} words from {text_file}:")
                
                # Test naive approach (O(m*n) where m=text length, n=dictionary size)
                _, naive_check_time = self.check_text_naive(text, naive_dict)
                self.check_times["naive"].append(naive_check_time)
                print(f"Naive List: {naive_check_time:.6f} seconds")
                
                # Test BBST approach (O(m*log n))
                _, bbst_check_time = self.check_text_bbst(text, bbst_dict)
                self.check_times["bbst"].append(bbst_check_time)
                print(f"BBST: {bbst_check_time:.6f} seconds")
                
                # Test trie approach (O(sum of word lengths in text))
                _, trie_check_time = self.check_text_trie(text, trie_dict)
                self.check_times["trie"].append(trie_check_time)
                print(f"Trie: {trie_check_time:.6f} seconds")
                
                # Test hashmap approach (O(m) average case)
                _, hashmap_check_time = self.check_text_hashmap(text, hashmap_dict)
                self.check_times["hashmap"].append(hashmap_check_time)
                print(f"HashMap: {hashmap_check_time:.6f} seconds")
                
                # Store build times (same for all text lengths, but needed for plotting)
                self.build_times["naive"].append(naive_build_time)
                self.build_times["bbst"].append(bbst_build_time)
                self.build_times["trie"].append(trie_build_time)
                self.build_times["hashmap"].append(hashmap_build_time)
                
            except FileNotFoundError:
                print(f"Error: Text file '{text_file}' not found. Skipping this file.")
            except Exception as e:
                print(f"Error processing text file '{text_file}': {e}. Skipping this file.")

    def plot_results(self):
        """
        Create visualizations of benchmark results.
        
        Generates two plots:
        1. Build times vs text length (shows construction overhead)
        2. Check times vs text length (shows query performance scaling)
        
        This helps visualize the trade-offs between different approaches:
        - Naive: Fast build, slow queries
        - HashMap: Fast build, fast queries  
        - BBST: Medium build, medium queries
        - Trie: Slow build, fast queries (especially for short words)
        """
        if not self.text_lengths:
            print("Error: No benchmark data available for plotting.")
            return
            
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Dictionary building times
            # Shows one-time cost of constructing each data structure
            plt.subplot(2, 1, 1)
            plt.plot(self.text_lengths, self.build_times["naive"], 'o-', label="Naive List")
            plt.plot(self.text_lengths, self.build_times["bbst"], 's-', label="BBST")
            plt.plot(self.text_lengths, self.build_times["trie"], '^-', label="Trie") 
            plt.plot(self.text_lengths, self.build_times["hashmap"], 'D-', label="HashMap")
            plt.title("Dictionary Building Times vs. Text Length")
            plt.xlabel("Text Length (words)")
            plt.ylabel("Time (seconds)")
            plt.legend()
            plt.grid(True)
            
            # Plot 2: Spell checking times  
            # Shows how query performance scales with text size
            plt.subplot(2, 1, 2)
            plt.plot(self.text_lengths, self.check_times["naive"], 'o-', label="Naive List")
            plt.plot(self.text_lengths, self.check_times["bbst"], 's-', label="BBST") 
            plt.plot(self.text_lengths, self.check_times["trie"], '^-', label="Trie")
            plt.plot(self.text_lengths, self.check_times["hashmap"], 'D-', label="HashMap")
            plt.title("Spell Checking Times vs. Text Length")
            plt.xlabel("Text Length (words)")
            plt.ylabel("Time (seconds)")
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig("./part1/a/spell_checker_benchmark.png")
            print("Plot saved as 'spell_checker_benchmark.png'")
            plt.show()
        except Exception as e:
            print(f"Error plotting results: {e}")

    def save_results_to_csv(self, filename="./part1/a/hecker_benchmark.csv"):
        """
        Save benchmark results to CSV file for further analysis.
        
        Creates a structured dataset with:
        - Text length (independent variable)
        - Build times for each algorithm
        - Check times for each algorithm
        
        This allows for detailed performance analysis and comparison.
        """
        if not self.text_lengths:
            print("Error: No benchmark data available to save.")
            return
            
        try:
            with open(filename, 'w') as f:
                # CSV header with all measured metrics
                f.write("Text Length,Naive Build,BBST Build,Trie Build,HashMap Build,Naive Check,BBST Check,Trie Check,HashMap Check\n")
                
                # Write data rows
                for i in range(len(self.text_lengths)):
                    f.write(f"{self.text_lengths[i]},{self.build_times['naive'][i]},{self.build_times['bbst'][i]},{self.build_times['trie'][i]},{self.build_times['hashmap'][i]},{self.check_times['naive'][i]},{self.check_times['bbst'][i]},{self.check_times['trie'][i]},{self.check_times['hashmap'][i]}\n")
                    
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")


# ==================== Main Execution ====================
if __name__ == "__main__":
    """
    Main execution block that sets up and runs the benchmark.
    
    This creates test files if they don't exist and runs comprehensive
    benchmarks comparing all four spell checking approaches.
    """
    checker = SpellChecker()
    
    # Get the directory where the script is located for relative file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths relative to the script directory
    dictionary_file = os.path.join(script_dir, "english_words.txt")
    
    # Test files of different sizes to analyze scaling behavior
    text_files = [
        os.path.join(script_dir, "text_sample_small.txt"),   # Small text (~20 words)
        os.path.join(script_dir, "text_sample_medium.txt"),  # Medium text (~400 words)  
        os.path.join(script_dir, "text_sample_large.txt")    # Large text (~1300 words)
    ]
    
    # Create sample text files if they don't exist
    # This ensures the benchmark can run even without pre-existing test files
    if not os.path.exists(text_files[0]):
        print("Creating sample text files for testing...")
        
        # Small sample: ~15 words
        sample_text_small = "This is a small sample text file for spell checking. It contains relatively few words."
        
        # Medium sample: ~400 words (20 repetitions of 20-word sentence)
        sample_text_medium = " ".join(["This is a medium sample text file."] * 20)
        
        # Large sample: ~1300 words (100 repetitions of 13-word sentence)
        sample_text_large = " ".join(["This is a large sample text file with many words to check."] * 100)
        
        try:
            # Write each sample file
            with open(text_files[0], 'w') as f:
                f.write(sample_text_small)
            with open(text_files[1], 'w') as f:
                f.write(sample_text_medium)
            with open(text_files[2], 'w') as f:
                f.write(sample_text_large)
            print("Sample text files created successfully.")
        except Exception as e:
            print(f"Error creating sample text files: {e}")
    
    # Run the comprehensive benchmark
    print("Starting spell checker benchmarking...")
    checker.benchmark(dictionary_file, text_files)
    
    # Generate visualizations and save results
    checker.plot_results()
    checker.save_results_to_csv()
    print("Benchmarking complete.")

    """
    Expected Performance Characteristics:
    
    BUILD TIMES:
    - Naive List: Fastest (O(1)) - just assigns reference
    - HashMap: Fast (O(n)) - hash each word once
    - BBST: Medium (O(n log n)) - insert with rebalancing
    - Trie: Slowest (O(sum of word lengths)) - character-by-character insertion
    
    SEARCH TIMES:
    - HashMap: Fastest (O(1) average) - direct hash lookup
    - Trie: Fast (O(word length)) - path traversal
    - BBST: Medium (O(log n)) - binary search through tree
    - Naive List: Slowest (O(n)) - linear search through entire list
    
    SPACE USAGE:
    - Naive List: Most efficient - just stores word list
    - HashMap: Efficient with some hash table overhead
    - BBST: Moderate overhead for tree structure
    - Trie: Can be very efficient due to prefix sharing, or expensive for diverse words
    
    PRACTICAL RECOMMENDATIONS:
    - For few lookups: Use naive list (simple, no build cost)
    - For many lookups: Use hashmap (best overall performance)
    - For prefix matching: Use trie (supports autocomplete, etc.)
    - For sorted access: Use BBST (maintains order, good for range queries)
    """