import time
import os
import matplotlib.pyplot as plt
from collections import defaultdict


class SpellChecker:
    def __init__(self):
        # For storing benchmark results
        self.build_times = {
            "naive": [],
            "bbst": [],
            "trie": [],
            "hashmap": []
        }
        self.check_times = {
            "naive": [],
            "bbst": [],
            "trie": [],
            "hashmap": []
        }
        self.text_lengths = []

    def load_dictionary(self, dictionary_file):
        """Load dictionary file and return a list of words."""
        try:
            with open(dictionary_file, 'r') as f:
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
    
    def build_naive_list(self, dictionary_words):
        """Build a naive list dictionary."""
        start_time = time.time()
        naive_dict = dictionary_words
        end_time = time.time()
        return naive_dict, end_time - start_time

    def check_word_naive(self, word, naive_dict):
        """Check if a word exists in the naive list."""
        return word.lower() in naive_dict

    def check_text_naive(self, text, naive_dict):
        """Check all words in a text using naive approach."""
        start_time = time.time()
        words = text.split()
        results = [self.check_word_naive(word.strip('.,!?"\'():;-'), naive_dict) for word in words]
        end_time = time.time()
        return results, end_time - start_time

    # ==================== BBST Implementation ====================
    
    def build_bbst(self, dictionary_words):
        """Build a balanced binary search tree (using Python's sorted set)."""
        try:
            import sortedcontainers
            start_time = time.time()
            bbst_dict = sortedcontainers.SortedSet(dictionary_words)
            end_time = time.time()
            return bbst_dict, end_time - start_time
        except ImportError:
            print("Error: sortedcontainers package not found.")
            print("Please install it with: pip install sortedcontainers")
            return set(dictionary_words), 0

    def check_word_bbst(self, word, bbst_dict):
        """Check if a word exists in the BBST."""
        return word.lower() in bbst_dict

    def check_text_bbst(self, text, bbst_dict):
        """Check all words in a text using BBST approach."""
        start_time = time.time()
        words = text.split()
        results = [self.check_word_bbst(word.strip('.,!?"\'():;-'), bbst_dict) for word in words]
        end_time = time.time()
        return results, end_time - start_time

    # ==================== Trie Implementation ====================
    
    class TrieNode:
        """Node for Trie data structure."""
        def __init__(self):
            self.children = {}
            self.is_end_of_word = False

    def build_trie(self, dictionary_words):
        """Build a Trie from dictionary words."""
        start_time = time.time()
        root = self.TrieNode()
        
        for word in dictionary_words:
            current = root
            for char in word:
                if char not in current.children:
                    current.children[char] = self.TrieNode()
                current = current.children[char]
            current.is_end_of_word = True
            
        end_time = time.time()
        return root, end_time - start_time

    def check_word_trie(self, word, trie_root):
        """Check if a word exists in the Trie."""
        current = trie_root
        word = word.lower()
        
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
            
        return current.is_end_of_word

    def check_text_trie(self, text, trie_root):
        """Check all words in a text using Trie approach."""
        start_time = time.time()
        words = text.split()
        results = [self.check_word_trie(word.strip('.,!?"\'():;-'), trie_root) for word in words]
        end_time = time.time()
        return results, end_time - start_time

    # ==================== HashMap Implementation ====================
    
    def build_hashmap(self, dictionary_words):
        """Build a hashmap dictionary."""
        start_time = time.time()
        hashmap_dict = set(dictionary_words)
        end_time = time.time()
        return hashmap_dict, end_time - start_time

    def check_word_hashmap(self, word, hashmap_dict):
        """Check if a word exists in the hashmap."""
        return word.lower() in hashmap_dict

    def check_text_hashmap(self, text, hashmap_dict):
        """Check all words in a text using hashmap approach."""
        start_time = time.time()
        words = text.split()
        results = [self.check_word_hashmap(word.strip('.,!?"\'():;-'), hashmap_dict) for word in words]
        end_time = time.time()
        return results, end_time - start_time

    # ==================== Benchmarking ====================
    
    def benchmark(self, dictionary_file, text_files):
        """Benchmark different spell checking approaches."""
        print("Loading dictionary...")
        dictionary_words = self.load_dictionary(dictionary_file)
        
        if not dictionary_words:
            print("Error: No words loaded from dictionary. Aborting benchmark.")
            return
            
        print(f"Successfully loaded {len(dictionary_words)} words from dictionary.")
        
        # Dictionary building benchmarks
        print("Building dictionaries...")
        _, naive_build_time = self.build_naive_list(dictionary_words)
        naive_dict, _ = self.build_naive_list(dictionary_words)
        print(f"Naive list built successfully.")
        
        _, bbst_build_time = self.build_bbst(dictionary_words)
        bbst_dict, _ = self.build_bbst(dictionary_words)
        print(f"BBST built successfully.")
        
        _, trie_build_time = self.build_trie(dictionary_words)
        trie_dict, _ = self.build_trie(dictionary_words)
        print(f"Trie built successfully.")
        
        _, hashmap_build_time = self.build_hashmap(dictionary_words)
        hashmap_dict, _ = self.build_hashmap(dictionary_words)
        print(f"HashMap built successfully.")
        
        print("Dictionary build times:")
        print(f"Naive List: {naive_build_time:.6f} seconds")
        print(f"BBST: {bbst_build_time:.6f} seconds")
        print(f"Trie: {trie_build_time:.6f} seconds")
        print(f"HashMap: {hashmap_build_time:.6f} seconds")
        
        # Text checking benchmarks
        print("\nRunning spell check benchmarks...")
        for text_file in text_files:
            try:
                with open(text_file, 'r') as f:
                    text = f.read()
                
                text_word_count = len(text.split())
                self.text_lengths.append(text_word_count)
                print(f"\nChecking text with {text_word_count} words from {text_file}:")
                
                # Naive benchmark
                _, naive_check_time = self.check_text_naive(text, naive_dict)
                self.check_times["naive"].append(naive_check_time)
                print(f"Naive List: {naive_check_time:.6f} seconds")
                
                # BBST benchmark
                _, bbst_check_time = self.check_text_bbst(text, bbst_dict)
                self.check_times["bbst"].append(bbst_check_time)
                print(f"BBST: {bbst_check_time:.6f} seconds")
                
                # Trie benchmark
                _, trie_check_time = self.check_text_trie(text, trie_dict)
                self.check_times["trie"].append(trie_check_time)
                print(f"Trie: {trie_check_time:.6f} seconds")
                
                # HashMap benchmark
                _, hashmap_check_time = self.check_text_hashmap(text, hashmap_dict)
                self.check_times["hashmap"].append(hashmap_check_time)
                print(f"HashMap: {hashmap_check_time:.6f} seconds")
                
                # Store build times (same for all text lengths)
                self.build_times["naive"].append(naive_build_time)
                self.build_times["bbst"].append(bbst_build_time)
                self.build_times["trie"].append(trie_build_time)
                self.build_times["hashmap"].append(hashmap_build_time)
                
            except FileNotFoundError:
                print(f"Error: Text file '{text_file}' not found. Skipping this file.")
            except Exception as e:
                print(f"Error processing text file '{text_file}': {e}. Skipping this file.")

    def plot_results(self):
        """Plot the benchmark results."""
        if not self.text_lengths:
            print("Error: No benchmark data available for plotting.")
            return
            
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot build times
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
            
            # Plot check times
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
        """Save the benchmark results to a CSV file."""
        if not self.text_lengths:
            print("Error: No benchmark data available to save.")
            return
            
        try:
            with open(filename, 'w') as f:
                # Write header
                f.write("Text Length,Naive Build,BBST Build,Trie Build,HashMap Build,Naive Check,BBST Check,Trie Check,HashMap Check\n")
                
                # Write data
                for i in range(len(self.text_lengths)):
                    f.write(f"{self.text_lengths[i]},{self.build_times['naive'][i]},{self.build_times['bbst'][i]},{self.build_times['trie'][i]},{self.build_times['hashmap'][i]},{self.check_times['naive'][i]},{self.check_times['bbst'][i]},{self.check_times['trie'][i]},{self.check_times['hashmap'][i]}\n")
                    
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")


# Example usage
if __name__ == "__main__":
    checker = SpellChecker()
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths relative to the script directory
    dictionary_file = os.path.join(script_dir, "english_words.txt")
    
    # For testing with different text sizes
    text_files = [
        os.path.join(script_dir, "text_sample_small.txt"),
        os.path.join(script_dir, "text_sample_medium.txt"),
        os.path.join(script_dir, "text_sample_large.txt")
    ]
    
    # If text files don't exist, create them for testing
    if not os.path.exists(text_files[0]):
        print("Creating sample text files for testing...")
        
        sample_text_small = "This is a small sample text file for spell checking. It contains relatively few words."
        sample_text_medium = " ".join(["This is a medium sample text file."] * 20)
        sample_text_large = " ".join(["This is a large sample text file with many words to check."] * 100)
        
        try:
            with open(text_files[0], 'w') as f:
                f.write(sample_text_small)
            with open(text_files[1], 'w') as f:
                f.write(sample_text_medium)
            with open(text_files[2], 'w') as f:
                f.write(sample_text_large)
            print("Sample text files created successfully.")
        except Exception as e:
            print(f"Error creating sample text files: {e}")
    
    print("Starting spell checker benchmarking...")
    checker.benchmark(dictionary_file, text_files)
    checker.plot_results()
    checker.save_results_to_csv()
    print("Benchmarking complete.")