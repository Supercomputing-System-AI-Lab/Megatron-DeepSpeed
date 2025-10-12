#!/usr/bin/env python3
import sys
import re

def parse_tflops_in_range(log_file_path, iteration_range):
    """
    Parses the training log and extracts TFLOPs for the iterations in 'iteration_range'.
    Returns a tuple: (max_TFLOPs, avg_TFLOPs).
    """

    # Regex to match lines with this pattern:
    #   iteration        1/      50 | ... | TFLOPs: 3.46 |
    #
    # We'll capture:
    #   1) The current iteration (group 1)
    #   2) The total iterations (group 2) - not really used for calculations
    #   3) The TFLOPs number (group 3)
    pattern = re.compile(
        r'iteration\s+(\d+)\s*/\s*(\d+).*?TFLOPs:\s*([\d.]+)',
        re.IGNORECASE
    )

    tflops_list = []

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # current iteration
                iteration = int(match.group(1))
                # total_iterations = int(match.group(2))  # we don't need it if you only want iteration
                tflops_val = float(match.group(3))

                # If the iteration is within the user-specified range/list, capture the TFLOPs.
                # e.g., if iteration_range is range(1,10), it will accept iteration 1..9
                if iteration in iteration_range:
                    tflops_list.append(tflops_val)

    if not tflops_list:
        return None, None  # or (0, 0), or raise an exception

    max_tflops = max(tflops_list)
    avg_tflops = sum(tflops_list) / len(tflops_list)

    return max_tflops, avg_tflops


def main():
    """
    Example usage:
        python parse_tf_ops.py /path/to/training_log.txt 1 10

    That would parse the file for iterations in range(1, 10) (i.e., iterations 1 through 9)
    and print the max and average TFLOPs in that range.
    """
    if len(sys.argv) < 2:
        print("Usage: python parse_tf_ops.py <log-file>")
        sys.exit(1)

    log_file_path = sys.argv[1]

    # If you want to include 'end_iter' in the range, use range(start_iter, end_iter+1)
    iteration_range = range(4, 30)

    max_tflops, avg_tflops = parse_tflops_in_range(log_file_path, iteration_range)

    if max_tflops is None:
        print("No matching iterations found in the log file for the specified range.")
    else:
        print(f"Range of iterations analyzed: {iteration_range}")
        print(f"Max TFLOPs: {max_tflops}")
        print(f"Average TFLOPs: {avg_tflops}")


if __name__ == "__main__":
    main()

