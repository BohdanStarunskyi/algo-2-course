import time
import random
import string
import matplotlib.pyplot as plt
from collections import deque
import os

class WildcardStringMatching:
    def __init__(self):
        self.results_brute_force = []
        self.results_sunday = []
        self.patterns_brute_force = []
        self.patterns_sunday = []
        self.texts_brute_force = []
        self.texts_sunday = []

    def preprocess_pattern(self, pattern):
        """Handle escape sequences and return processed pattern."""
        processed = []
        escape = False
        for char in pattern:
            if escape:
                processed.append(('literal', char))
                escape = False
            elif char == "\\":
                escape = True
            elif char == "?":
                processed.append(('wildcard', '?'))
            elif char == "*":
                processed.append(('wildcard', '*'))
            else:
                processed.append(('literal', char))
        return processed

    def char_match(self, text_char, pattern_item):
        """Match a single character with pattern element (wildcard or literal)."""
        kind, value = pattern_item
        if kind == 'wildcard' and value == '?':
            return True
        if kind == 'literal':
            return text_char == value
        return False

    def brute_force_wildcard(self, text, pattern):
        start_time = time.time()
        processed_pattern = self.preprocess_pattern(pattern)
        n, m = len(text), len(processed_pattern)
        found = False

        if m == 0:
            return True, time.time() - start_time

        effective_len = len([item for item in processed_pattern if item != ('wildcard', '*')])
        if effective_len > n:
            return False, time.time() - start_time

        stack = deque([(0, 0)])

        while stack:
            text_pos, pattern_pos = stack.pop()

            if pattern_pos == m:
                found = True
                break

            if text_pos > n:
                continue

            if pattern_pos < m and processed_pattern[pattern_pos] == ('wildcard', '*'):
                # '*' matches zero or more characters
                stack.append((text_pos, pattern_pos + 1))   # '*' matches zero chars
                if text_pos < n:
                    stack.append((text_pos + 1, pattern_pos))  # '*' matches one more char
            elif text_pos < n and pattern_pos < m and self.char_match(text[text_pos], processed_pattern[pattern_pos]):
                stack.append((text_pos + 1, pattern_pos + 1))

        return found, time.time() - start_time

    def sunday_wildcard(self, text, pattern):
        start_time = time.time()
        processed_pattern = self.preprocess_pattern(pattern)
        n, m = len(text), len(processed_pattern)
        i = 0

        while i <= n:
            j = 0
            ti = i
            while j < m:
                if ti > n:
                    break
                if processed_pattern[j] == ('wildcard', '*'):
                    # Greedy match: try to consume as many as needed
                    if j == m - 1:
                        ti = n
                        j = m
                        break
                    next_pattern = processed_pattern[j + 1]
                    while ti <= n:
                        if ti < n and self.char_match(text[ti], next_pattern):
                            break
                        ti += 1
                    j += 1
                elif ti < n and self.char_match(text[ti], processed_pattern[j]):
                    ti += 1
                    j += 1
                else:
                    break

            if j == m:
                return True, time.time() - start_time

            if i + m >= n:
                break

            # Standard Sunday shift: look at the next character
            if i + m < n:
                next_char = text[i + m]
                shift = 1
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
        return ''.join(random.choices(string.ascii_letters, k=length))

    def generate_pattern_with_wildcards(self, base_pattern, num_wildcards):
        """Insert random wildcards into the pattern."""
        pattern = list(base_pattern)
        for _ in range(num_wildcards):
            if pattern:
                index = random.randint(0, len(pattern) - 1)
                pattern[index] = random.choice(["*", "?"])
        return ''.join(pattern)

    def benchmark(self, text_lengths, base_pattern, num_tests=10, max_wildcards=5):
        """Benchmark brute-force and Sunday implementations."""
        for length in text_lengths:
            avg_time_bf = 0
            avg_time_sun = 0

            for _ in range(num_tests):
                text = self.generate_random_text(length)
                pattern_with_wildcards = self.generate_pattern_with_wildcards(
                    base_pattern, random.randint(0, max_wildcards)
                )

                found_bf, duration_bf = self.brute_force_wildcard(text, pattern_with_wildcards)
                found_sun, duration_sun = self.sunday_wildcard(text, pattern_with_wildcards)

                avg_time_bf += duration_bf
                avg_time_sun += duration_sun

            avg_time_bf /= num_tests
            avg_time_sun /= num_tests

            self.results_brute_force.append(avg_time_bf)
            self.results_sunday.append(avg_time_sun)
            self.patterns_brute_force.append(pattern_with_wildcards)
            self.patterns_sunday.append(pattern_with_wildcards)
            self.texts_brute_force.append(text)
            self.texts_sunday.append(text)

            print(f"Completed benchmarking for text length {length}")

    def save_results_to_csv(self, folder="./results"):
        os.makedirs(folder, exist_ok=True)
        brute_file = os.path.join(folder, "brute_force.csv")
        sunday_file = os.path.join(folder, "sunday.csv")

        with open(brute_file, "w") as f:
            f.write("Pattern,Text Length,Time\n")
            for p, t, r in zip(self.patterns_brute_force, self.texts_brute_force, self.results_brute_force):
                f.write(f"{p},{len(t)},{r}\n")
        print(f"Saved Brute-force results to {brute_file}")

        with open(sunday_file, "w") as f:
            f.write("Pattern,Text Length,Time\n")
            for p, t, r in zip(self.patterns_sunday, self.texts_sunday, self.results_sunday):
                f.write(f"{p},{len(t)},{r}\n")
        print(f"Saved Sunday results to {sunday_file}")

    def plot_results(self, save_path="./results/benchmark_plot.png"):
        plt.figure(figsize=(10, 6))
        text_lengths = [len(t) for t in self.texts_brute_force]
        plt.plot(text_lengths, self.results_brute_force, label="Brute-force", marker='o')
        plt.plot(text_lengths, self.results_sunday, label="Sunday", marker='x')
        plt.xlabel("Text Length")
        plt.ylabel("Average Time (seconds)")
        plt.title("Wildcard String Matching Benchmark")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved benchmark plot to {save_path}")
        plt.show()

if __name__ == "__main__":
    wsm = WildcardStringMatching()
    print("Starting benchmark...")
    wsm.benchmark(text_lengths=[10, 50, 100, 200, 500], base_pattern="a?b*c", num_tests=10)
    wsm.save_results_to_csv()
    wsm.plot_results()
    print("Benchmark complete.")
