AI Lab - Exercise 3: CVRP & Rush Hour (GP/GEP)

This project contains solutions for Exercise 3 of the Artificial Intelligence Laboratory. It includes the Capacitated Vehicle Routing Problem (CVRP) and an evolutionary approach (GP & GEP) to solving Rush Hour puzzles.

 Important Prerequisite

Before running the program, you must ensure that all relevant files are located in the exact same directory.

CVRP.py / CVRP.exe: Source code / Executable for Part A.

RH_GP.py / RH_GP.exe: Source code / Executable for Part B.

rh.txt: Input file containing 40 Rush Hour puzzles for comparison.

*.vrp files: All 6 instance files for the CVRP problem.
 How to Run

Method 1: Running from Source Code (Python)

Requirements:

Python 3.13

External libraries: numpy, matplotlib

Part A: CVRP
The script will automatically scan the directory for the presence of the 6 .vrp files. Make sure they are in the same folder.

python CVRP.py


Part B: Rush Hour GP
Ensure that the rh.txt file is located in the same directory as the script.

python RH_GP.py


Method 2: Running from Executables (EXE)

Note: The executable files do not require Python or any external libraries to be installed on your machine. You can find them in the "Releases" section of this repository.

Just like the source code, you must verify that all input files (*.vrp, rh.txt) are in the same directory as the .exe files.

Part A: Simply run CVRP.exe

Part B: Simply run RH_GP.exe

 Outputs & Results

Part A: CVRP

When running this part, the program will generate the following:

Progress Table: Prints a table showing the best results, average, and standard deviation across 3 different seeds for each algorithm.

Visualizations: Generates and saves two .png files for each instance in the directory:

A map displaying the best vehicle routes along with an output matrix.

A bar chart comparing the performance of the different algorithms.

Part B: Rush Hour GP

When running this part, the program will generate the following:

Evolutionary Process: Prints the step-by-step evolutionary development process of the GP.

Comparison Table: Prints a detailed table comparing the performance of the basic H2 heuristic against the heuristics evolved by GP and GEP, tested on the 40 puzzles from rh.txt.

The table compares: Number of developed states, execution time, and solution path length.

Concludes with a comparison of the overall averages throughout the test.

Diversity Check: Includes an examination between 2 separate runs of GP and GEP to demonstrate the diversity between them.