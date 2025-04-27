import time
import random
import string
import matplotlib.pyplot as plt
from collections import deque


class WildcardStringMatching:
    def __init__(self):
        self.results_brute_force = []
        self.results_sunday = []
        self.patterns_brute_force = []
        self.patterns_sunday = []
        self.texts_brute_force = []
        self.texts_sunday = []

    def preprocess_pattern(self, pattern):
        """Preprocess the pattern by handling escape sequences and converting to list."""
        processed = []
        escape = False
        for char in pattern:
            if escape:
                processed.append(char)
                escape = False
            elif char == "\\":
                escape = True
            else:
                processed.append(char)
        return processed

    def char_match(self, char, pattern_char):
        """Check if a single character matches a pattern character."""
        return pattern_char == "?" or char == pattern_char

    def brute_force_wildcard(self, text, pattern):
        """Brute Force algorithm with wildcard support (iterative)."""
        start_time = time.time()
        processed_pattern = self.preprocess_pattern(pattern)
        n, m = len(text), len(processed_pattern)
        found = False

        if processed_pattern == ["*"]:
            return True, time.time() - start_time

        effective_pattern_len = len([c for c in processed_pattern if c != "*"])
        if effective_pattern_len > n:
            return False, time.time() - start_time

        stack = deque([(0, 0)])  # (text_pos, pattern_pos)
        while stack:
            text_pos, pattern_pos = stack.pop()
            if pattern_pos == m:
                found = True
                break

            if text_pos == n:
                continue

            if processed_pattern[pattern_pos] == "*":
                # '*' matches zero or more characters, push next possible positions
                stack.append((text_pos + 1, pattern_pos))  # Skip one character in text
                stack.append((text_pos, pattern_pos + 1))  # Skip '*' in pattern
            elif self.char_match(text[text_pos], processed_pattern[pattern_pos]):
                # Match the character, continue with next positions
                stack.append((text_pos + 1, pattern_pos + 1))
            else:
                continue

        return found, time.time() - start_time

    def sunday_wildcard(self, text, pattern):
        """Sunday algorithm with wildcard support."""
        start_time = time.time()
        processed_pattern = self.preprocess_pattern(pattern)
        n, m = len(text), len(processed_pattern)
        i = 0

        while i <= n - m:
            j = 0
            while j < m and (i + j < n) and (self.char_match(text[i + j], processed_pattern[j]) or processed_pattern[j] == "*"):
                if processed_pattern[j] == "*":
                    # Handle '*' by matching zero or more characters
                    matched = False
                    for k in range(i + j, n):
                        if j + 1 < m and (self.char_match(text[k], processed_pattern[j + 1]) or processed_pattern[j + 1] == "*"):
                            i = k  # Move i to the next character after the match
                            j += 1
                            matched = True
                            break
                    if not matched:
                        break  # If no match was found for '*', exit the loop
                else:
                    j += 1

            if j == m:
                return True, time.time() - start_time  # Pattern matched

            # Move i to the next possible position (following Sunday algorithm)
            i += 1

        return False, time.time() - start_time

    def generate_random_text(self, length):
        """Generate a random text of the specified length."""
        return "".join(
            random.choices(string.ascii_lowercase + string.ascii_uppercase, k=length)
        )

    def generate_pattern_with_wildcards(self, pattern, num_wildcards):
        """Generate a pattern with random wildcards inserted at random positions."""
        pattern_list = list(pattern)
        for _ in range(num_wildcards):
            index = random.randint(0, len(pattern_list) - 1)
            pattern_list[index] = "*"
        return "".join(pattern_list)

    def benchmark(self, text_lengths, pattern, num_tests=100, max_wildcards=5):
        """Benchmark matching performance with varying text lengths."""
        # Brute Force Benchmark
        print("Running Brute-force Benchmark...")
        for length in text_lengths:
            avg_time = 0
            for _ in range(num_tests):
                text = self.generate_random_text(length)
                pattern_with_wildcards = self.generate_pattern_with_wildcards(
                    pattern, random.randint(0, max_wildcards)
                )
                found, duration = self.brute_force_wildcard(
                    text, pattern_with_wildcards
                )
                avg_time += duration
            avg_time /= num_tests
            self.results_brute_force.append(avg_time)
            self.patterns_brute_force.append(pattern_with_wildcards)
            self.texts_brute_force.append(text)
            print(f"Brute-force: Completed for text length {length}")

        # Sunday Benchmark
        print("Running Sunday Benchmark...")
        for length in text_lengths:
            avg_time = 0
            for _ in range(num_tests):
                text = self.generate_random_text(length)
                pattern_with_wildcards = self.generate_pattern_with_wildcards(
                    pattern, random.randint(0, max_wildcards)
                )
                found, duration = self.sunday_wildcard(text, pattern_with_wildcards)
                avg_time += duration
            avg_time /= num_tests
            self.results_sunday.append(avg_time)
            self.patterns_sunday.append(pattern_with_wildcards)
            self.texts_sunday.append(text)
            print(f"Sunday: Completed for text length {length}")

    def plot_results(self):
        """Plot the results of the benchmarking for both algorithms."""
        print("Plotting results...")
        plt.figure(figsize=(12, 6))

        # Plot Brute Force results
        plt.subplot(1, 2, 1)
        plt.plot(
            range(len(self.results_brute_force)),
            self.results_brute_force,
            label="Brute-force",
        )
        plt.title("Brute-force Wildcard Matching Performance")
        plt.xlabel("Test Case Index")
        plt.ylabel("Time (seconds)")
        plt.grid(True)

        # Plot Sunday results
        plt.subplot(1, 2, 2)
        plt.plot(
            range(len(self.results_sunday)),
            self.results_sunday,
            label="Sunday",
            color="orange",
        )
        plt.title("Sunday Wildcard Matching Performance")
        plt.xlabel("Test Case Index")
        plt.ylabel("Time (seconds)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def save_results_to_csv(
        self,
        filename_brute_force="brute_force_benchmark_results.csv",
        filename_sunday="sunday_benchmark_results.csv",
    ):
        """Save the benchmark results to CSV files."""
        print("Saving results to CSV files...")
        # Save Brute Force results
        with open(filename_brute_force, "w") as file:
            file.write("Pattern,Text Length,Time\n")
            for pattern, text, time in zip(
                self.patterns_brute_force,
                self.texts_brute_force,
                self.results_brute_force,
            ):
                file.write(f"{pattern},{len(text)},{time}\n")
        print(f"Brute-force results saved to {filename_brute_force}")

        # Save Sunday results
        with open(filename_sunday, "w") as file:
            file.write("Pattern,Text Length,Time\n")
            for pattern, text, time in zip(
                self.patterns_sunday, self.texts_sunday, self.results_sunday
            ):
                file.write(f"{pattern},{len(text)},{time}\n")
        print(f"Sunday results saved to {filename_sunday}")


# Example usage of the class
if __name__ == "__main__":
    wsm = WildcardStringMatching()
    print("Starting benchmarking...")
    wsm.benchmark(text_lengths=[10, 50, 100, 200, 500], pattern="a?b*c", num_tests=10)
    wsm.plot_results()
    wsm.save_results_to_csv()
    print("Benchmarking complete.")
