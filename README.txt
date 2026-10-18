ARTI 503 - PROJECT STAGE 1
Parallel Data Analysis for Large Datasets

FILES
-----
train.csv
    The common dataset used by every group member.

sequential_benchmark.py
    The Stage 1 sequential code. It performs:
    - Average / minimum / maximum of trip_duration
    - Frequency counts
    - Grouped aggregation
    - Sequential execution-time measurement
    - CPU/core information collection

requirements.txt
    Python packages required by the code.

HOW EACH GROUP MEMBER RUNS THE SAME TEST
----------------------------------------
1. Install Python 3 if it is not already installed.
2. Keep train.csv and sequential_benchmark.py in this same folder.
3. Open Command Prompt / PowerShell in this folder.
4. Install the required packages once:

   pip install -r requirements.txt

5. Run the benchmark:

   python sequential_benchmark.py

6. The default test uses these adjustable input sizes:
   10,000 / 100,000 / 500,000 / 1,000,000 rows

   The script repeats each measurement 3 times so the result is not based on one run.

7. When the script finishes, it creates three files for that computer:
   benchmark_summary_<computer>.csv
   benchmark_runs_<computer>.csv
   system_info_<computer>.txt

8. Send those three generated files to the group leader.

OPTIONAL: CHANGE INPUT SIZES OR REPEATS
---------------------------------------
Example:

   python sequential_benchmark.py --rows 10000,100000,500000 --repeats 4

IMPORTANT FOR STAGE 1
---------------------
- Do NOT modify the code between group members.
- Do NOT use multiprocessing or parallel code in Stage 1.
- Everyone should use the same train.csv file.
- The CSV loading time is shown separately and is NOT included in the analysis benchmark, so the measurements focus on the analytical work rather than disk I/O.
