"""
Extended rush hour, assignpyinstaller --onefile --clean CVRP.pyment 3, part 2 (GP/GEP)
"""

import heapq
import time
import random
import os
import sys
import copy
import numpy as np
from collections import deque


#original RH code
#board & state

class Car:
    """struct for a vehicle on the board"""
    def __init__(self, char_id, orientation, length, row, col):
        self.char_id = char_id
        self.orientation = orientation   # 'H' or 'V'
        self.length = length
        self.row = row
        self.col = col


class State:
    """
    board state, keeps the flat 36-char string as the string form and a 2D view for indexing.
    """
    def __init__(self, board_string, g_cost, parent=None, move_from_parent=""):
        self.board_string = board_string
        self.g_cost = g_cost
        self.h_cost = 0
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.board_2d = [list(board_string[i:i+6]) for i in range(0, 36, 6)]
        self.cars = self.parse_cars()

    def f_cost(self):
        return self.g_cost + self.h_cost

    def __lt__(self, other):
        #tie breaker on f-cost for heapq for A*
        return self.f_cost() < other.f_cost()

    def __eq__(self, other):
        return self.board_string == other.board_string

    def __hash__(self):
        #keep state usable (was overridden in __eq__)
        return hash(self.board_string)

    def parse_cars(self):
        """walk the board string once and pull out a car for every letter."""
        cars = []
        seen = set()
        for i, ch in enumerate(self.board_string):
            if ch == '.' or ch in seen:
                continue
            seen.add(ch)
            cells = [j for j, c in enumerate(self.board_string) if c == ch]
            length = len(cells)
            row, col = divmod(cells[0], 6)
            # In the flat string, horizontal-neighbours differ by 1 and  vertical-neighbours differ by 6.
            if length == 1:
                orientation = 'H'
            else:
                orientation = 'H' if cells[1] - cells[0] == 1 else 'V'
            cars.append(Car(ch, orientation, length, row, col))
        return cars

    def is_goal(self):
        #the exit is to the right of column 5 in row 2, so the goal is X parked in cells (2,4)-(2,5).
        return self.board_2d[2][4] == 'X' and self.board_2d[2][5] == 'X'

    def get_successors(self):
        """
        for every car, slide it as far as it can go in each direction and
        eject one successor per number of steps. So if a car can move 1, 2,
        or 3 cells, we produce three successors (R1, R2, R3) and
        each one counts as a single move
        """
        out = []
        for car in self.cars:
            if car.orientation == 'H':
                #rightward
                steps = 1
                while (car.col + car.length - 1 + steps < 6
                       and self.board_2d[car.row][car.col + car.length - 1 + steps] == '.'):
                    out.append(self.slide(car, 'R', steps))
                    steps += 1
                #leftward
                steps = 1
                while car.col - steps >= 0 and self.board_2d[car.row][car.col - steps] == '.':
                    out.append(self.slide(car, 'L', steps))
                    steps += 1
            else:
                #downward
                steps = 1
                while (car.row + car.length - 1 + steps < 6
                       and self.board_2d[car.row + car.length - 1 + steps][car.col] == '.'):
                    out.append(self.slide(car, 'D', steps))
                    steps += 1
                #upward
                steps = 1
                while car.row - steps >= 0 and self.board_2d[car.row - steps][car.col] == '.':
                    out.append(self.slide(car, 'U', steps))
                    steps += 1
        return out

    def slide(self, car, direction, steps):
        """build the State that results from sliding `car` by `steps` cells."""
        grid = [row[:] for row in self.board_2d]
        # erase the car
        for r in range(6):
            for c in range(6):
                if grid[r][c] == car.char_id:
                    grid[r][c] = '.'
        # write it at the new position
        new_r, new_c = car.row, car.col
        if direction == 'R': new_c += steps
        elif direction == 'L': new_c -= steps
        elif direction == 'D': new_r += steps
        elif direction == 'U': new_r -= steps
        if car.orientation == 'H':
            for k in range(car.length):
                grid[new_r][new_c + k] = car.char_id
        else:
            for k in range(car.length):
                grid[new_r + k][new_c] = car.char_id
        new_string = "".join("".join(row) for row in grid)
        move_label = f"{car.char_id}{direction}{steps}"
        return State(new_string, self.g_cost + 1, self, move_label)


#heuristics
def heuristic_1_blocking_cars(state):
    """
    H1- blocking cars: 1 + (number of distinct cars sitting between X and the exit on X's row)
    """
    if state.is_goal():
        return 0
    x_col = -1
    for c in range(6):
        if state.board_2d[2][c] == 'X':
            x_col = c
    #defected board
    if x_col == -1:
        return float('inf')
    blockers = set()
    for col in range(x_col + 2, 6):
        cell = state.board_2d[2][col]
        if cell != '.':
            blockers.add(cell)
    return 1 + len(blockers)


def heuristic_2_blockers_of_blockers(state):
    """
    H2- blockers of blockers: H1 plus 1 for each blocker that has no room to move on its own
    column (blocked from above and below)
    """
    if state.is_goal():
        return 0
    base = heuristic_1_blocking_cars(state)
    #defected board
    if base == float('inf'):
        return base

    extra = 0
    x_col = -1
    for c in range(6):
        if state.board_2d[2][c] == 'X':
            x_col = c

    for col in range(x_col + 2, 6):
        cell = state.board_2d[2][col]
        if cell == '.':
            continue
        # checks if vertical blocker has any room to slide- if both above and below are blocked
        # (by another car or by the bound), so its stuck and clearing it will cost more
        stuck_up = False
        for r in range(2, -1, -1):
            v = state.board_2d[r][col]
            if v != cell and v != '.':
                stuck_up = True
                break
            if v != cell and r == 0:
                stuck_up = True
        stuck_down = False
        for r in range(3, 6):
            v = state.board_2d[r][col]
            if v != cell and v != '.':
                stuck_down = True
                break
            if v != cell and r == 5:
                stuck_down = True
        if stuck_up and stuck_down:
            extra += 1

    return base + extra


#searches

def calculate_ebf(n_nodes, depth, tolerance=0.001):
    """
    effective branching factor: solve  N+1 = b + b^2 + ... + b^d  for b.
    a binary search
    """
    if depth == 0 or n_nodes == 0:
        return 1.0
    lo, hi = 1.0, float(n_nodes)
    while hi - lo > tolerance:
        mid = (lo + hi) / 2.0
        if mid == 1.0:
            n_estimate = depth + 1
        else:
            n_estimate = (mid ** (depth + 1) - 1) / (mid - 1)
        if n_estimate > n_nodes + 1:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def a_star_search(initial_board_string, heuristic_func, time_limit=60):
    """
    graph search A*: open list is a min-heap on f = g + h,
    closed list is a set of board strings.
    """
    start = State(initial_board_string, 0)
    start.h_cost = heuristic_func(start)

    open_list = []
    heapq.heappush(open_list, start)
    closed = set()

    n_expanded = 0
    sum_h = 0
    sum_g = 0
    max_depth = 0
    # min-depth- the smallest depth reached before the search was cut off and moved to a different branch.
    # A* backtracks every time it pops a node shallower than the previous one
    min_depth = float('inf')
    last_depth = -1

    t0 = time.time()
    while open_list:
        if time.time() - t0 > time_limit:
            return {"Success (Y/N)": "N", "Time (ms)": (time.time() - t0) * 1000}

        cur = heapq.heappop(open_list)

        if cur.is_goal():
            # rebuild the move sequence
            moves = []
            node = cur
            while node.parent:
                moves.append(node.move_from_parent)
                node = node.parent
            moves.reverse()

            d = cur.g_cost
            return {
                #stats for results table
                "Success (Y/N)": "Y",
                "solution": " ".join(moves),
                "N": n_expanded,
                "d": d,
                "d/N": d / n_expanded if n_expanded else 0,
                "Time (ms)": (time.time() - t0) * 1000,
                "EBF": calculate_ebf(n_expanded, d),
                "avg H value": sum_h / n_expanded if n_expanded else 0,
                "Min": min_depth if min_depth != float('inf') else d,
                "Avg": sum_g / n_expanded if n_expanded else 0,
                "Max": max_depth,
            }

        if cur.board_string in closed:
            continue
        closed.add(cur.board_string)

        n_expanded += 1
        sum_h += cur.h_cost
        sum_g += cur.g_cost
        max_depth = max(max_depth, cur.g_cost)

        #detect a jump back to a shallower branch
        if last_depth != -1 and cur.g_cost < last_depth:
            min_depth = min(min_depth, cur.g_cost)
        last_depth = cur.g_cost

        for succ in cur.get_successors():
            if succ.board_string not in closed:
                succ.h_cost = heuristic_func(succ)
                heapq.heappush(open_list, succ)

    return {"Success (Y/N)": "N", "Reason": "Queue Empty"}


def bfs_search(initial_board_string, time_limit=60):
    """
    BFS - the uninformed baseline the assignment asks for.
    BFS finds the optimal depth but expands for more nodes than A*.
    """
    start = State(initial_board_string, 0)
    if start.is_goal():
        return {"Success (Y/N)": "Y", "solution": "", "N": 0, "d": 0,
                "d/N": 0, "Time (ms)": 0, "EBF": 1.0,
                "avg H value": 0, "Min": 0, "Avg": 0, "Max": 0}

    queue = deque([start])
    visited = {start.board_string}
    n_expanded = 0
    sum_g = 0
    max_depth = 0

    t0 = time.time()
    while queue:
        if time.time() - t0 > time_limit:
            return {"Success (Y/N)": "N", "Time (ms)": (time.time() - t0) * 1000}

        cur = queue.popleft()
        n_expanded += 1
        sum_g += cur.g_cost
        max_depth = max(max_depth, cur.g_cost)

        for succ in cur.get_successors():
            if succ.board_string in visited:
                continue
            visited.add(succ.board_string)
            if succ.is_goal():
                moves = []
                node = succ
                while node.parent:
                    moves.append(node.move_from_parent)
                    node = node.parent
                moves.reverse()
                d = succ.g_cost
                return {
                    "Success (Y/N)": "Y",
                    "solution": " ".join(moves),
                    "N": n_expanded,
                    "d": d,
                    "d/N": d / n_expanded if n_expanded else 0,
                    "Time (ms)": (time.time() - t0) * 1000,
                    "EBF": calculate_ebf(n_expanded, d),
                    "avg H value": 0,   # BFS uses no heuristic
                    "Min": 0,           # BFS expands level by level
                    "Avg": sum_g / n_expanded if n_expanded else 0,
                    "Max": max_depth,
                }
            queue.append(succ)

    return {"Success (Y/N)": "N", "Reason": "Queue Empty"}


#puzzle generator (assignment requirement 7)

#letters matching the convention in the PDF: cars are A-K, trucks are O-R, X is the red car.
CAR_LETTERS   = list("ABCDEFGHIJK")
TRUCK_LETTERS = list("OPQR")


def try_place(board, char, length, orientation, row, col):
    """
    return new board string with the vehicle placed, or none on conflict
    """
    cells = list(board)
    targets = []
    for i in range(length):
        r = row + (i if orientation == 'V' else 0)
        c = col + (i if orientation == 'H' else 0)
        if r >= 6 or c >= 6:
            return None
        idx = r * 6 + c
        if cells[idx] != '.':
            return None
        targets.append(idx)
    for idx in targets:
        cells[idx] = char
    return "".join(cells)


def generate_puzzle(target_depth, max_attempts=2000, time_limit=10):
    """
    generate a random Rush Hour puzzle whose A* solution depth is almost
    target_depth (accepts anything within +-2). Constraints from the assignment:
    - X lives on row 2
    - at least one vertical blocker in front of X
    - 1-3 vertical blockers in the exit corridor when possible
    - board is not too sparse
    Returns (board_string, depth, solution_moves) or (None, None, None).
    """
    t0 = time.time()
    for _ in range(max_attempts):
        if time.time() - t0 > time_limit:
            break

        board = '.' * 36
        x_col = random.choice([0, 1, 2])
        board = try_place(board, 'X', 2, 'H', 2, x_col)
        if board is None:
            continue

        pool = CAR_LETTERS[:] + TRUCK_LETTERS[:]
        random.shuffle(pool)
        ptr = iter(pool)

        #vertical blockers in the exit
        blocker_cols = list(range(x_col + 2, 6))
        random.shuffle(blocker_cols)
        n_blockers = random.randint(1, min(3, len(blocker_cols)))
        placed = 0
        for bcol in blocker_cols[:n_blockers]:
            ch = next(ptr, None)
            if ch is None:
                break
            length = random.choice([2, 3])
            #force the vehicle to span row 2 so it's actually in the way
            row_min = max(0, 2 - length + 1)
            row_max = min(2, 6 - length)
            if row_min > row_max:
                continue
            row = random.randint(row_min, row_max)
            new_board = try_place(board, ch, length, 'V', row, bcol)
            if new_board:
                board = new_board
                placed += 1

        if placed == 0:
            continue

        #add a few extra cars to fill the board
        for _ in range(random.randint(2, 5)):
            ch = next(ptr, None)
            if ch is None:
                break
            length = random.choice([2, 2, 3])
            ori = random.choice(['H', 'V'])
            r = random.randint(0, 5)
            c = random.randint(0, 5)
            nb = try_place(board, ch, length, ori, r, c)
            if nb:
                board = nb

        #check if the resulting puzzle has the right difficulty
        res = a_star_search(board, heuristic_1_blocking_cars, time_limit=3)
        if res.get("Success (Y/N)") == "Y":
            d = res["d"]
            if abs(d - target_depth) <= 2:
                return board, d, res["solution"]

    return None, None, None


def board_to_grid(board_string):
    """formats a 36-char board string as a readable 6x6 grid"""
    rows = []
    for i in range(0, 36, 6):
        rows.append("    " + " ".join(board_string[i:i+6]))
    return "\n".join(rows)


#input parsing

def parse_rh_file(filename):
    """
    pull just the 36-char board lines out of the file
    which we skip the start by matching exact length and the dot or letter character set.
    """
    boards = []
    with open(filename, 'r') as f:
        for line in f:
            s = line.strip()
            if len(s) == 36 and all(c == '.' or c.isalpha() for c in s):
                boards.append(s)
    return boards


# extentions for GP
def t_dist_to_exit(state):
    """
    terminal feature- distance of the red car to the exit
    """
    if state.is_goal(): return 0
    for c in range(6):
        if state.board_2d[2][c] == 'X':
            return 5 - c
    return 100

# function set- inside nodes & terminal set- leaves
terminals = [heuristic_1_blocking_cars, heuristic_2_blockers_of_blockers, t_dist_to_exit, 1.0, 2.0]
functions = ['ADD', 'SUB', 'MUL', 'MAX']


class GPNode:
    """
    Node in the abstract syntax tree- AST
    represents operator (ADD,MAX...) if is an internal node,
    or a feature/constant (H1, 2.0...) if is a terminal leaf
    """
    def __init__(self, is_terminal, value, left=None, right=None):
        self.is_terminal = is_terminal
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, state):
        """
        recursively evaluates the AST to compute the heuristic value for a state
        """
        if self.is_terminal:
            if callable(self.value):
                return self.value(state)
            return self.value

        left_val = self.left.evaluate(state)
        right_val = self.right.evaluate(state)

        if self.value == 'ADD':
            return left_val + right_val
        elif self.value == 'SUB':
            # Bounded to 0 to prevent negative heuristics, keeping A* well-behaved.
            return max(0.0, left_val - right_val)
        elif self.value == 'MUL':
            return left_val * right_val
        elif self.value == 'MAX':
            return max(left_val, right_val)
        return 0

    def get_size(self):
        """
        calculate the computational complexity- number of nodes in the tree
        """
        if self.is_terminal: return 1
        return 1 + self.left.get_size() + self.right.get_size()

    def __str__(self):
        if self.is_terminal:
            if self.value == heuristic_1_blocking_cars: return "H1"
            if self.value == heuristic_2_blockers_of_blockers: return "H2"
            if self.value == t_dist_to_exit: return "DistExit"
            return str(self.value)
        return f"{self.value}({str(self.left)}, {str(self.right)})"


def generate_random_tree(max_depth, current_depth=0):
    """
    build a random AST using grow method we learned,
    which allows creating asymmetrical trees of variable depth- adds diversity
    """
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.3):
        return GPNode(is_terminal=True, value=random.choice(terminals))

    op = random.choice(functions)
    left = generate_random_tree(max_depth, current_depth + 1)
    right = generate_random_tree(max_depth, current_depth + 1)
    return GPNode(is_terminal=False, value=op, left=left, right=right)


def mutate_tree(node, max_depth, current_depth=0):
    """
    subtree mutation- replace a random branch with a new random tree
    """
    if random.random() < 0.15 or current_depth >= max_depth:
        return generate_random_tree(max_depth - current_depth)

    new_node = copy.deepcopy(node)
    if not new_node.is_terminal:
        if random.random() < 0.5:
            new_node.left = mutate_tree(new_node.left, max_depth, current_depth + 1)
        else:
            new_node.right = mutate_tree(new_node.right, max_depth, current_depth + 1)
    return new_node


def get_random_node(tree):
    """
    extract a random node from the tree to be used as a donor in crossover
    """
    nodes = []
    def traverse(node):
        nodes.append(node)
        if not node.is_terminal:
            traverse(node.left)
            traverse(node.right)
    traverse(tree)
    return random.choice(nodes)


def crossover_trees(parent1, parent2):
    """
    subtree crossover- swap a random branch of parent1 with a random branch of parent2
    """
    child = copy.deepcopy(parent1)
    nodes_in_child = []

    def traverse_child(n):
        nodes_in_child.append(n)
        if not n.is_terminal:
            traverse_child(n.left)
            traverse_child(n.right)

    traverse_child(child)
    target_node = random.choice(nodes_in_child)
    donor_node = copy.deepcopy(get_random_node(parent2))

    # swap the contents of the target node with the donor
    target_node.is_terminal = donor_node.is_terminal
    target_node.value = donor_node.value
    target_node.left = donor_node.left
    target_node.right = donor_node.right

    return child


def evaluate_gp_fitness(tree, training_boards):
    """
    multi objective fitness:
    objective = average nodes expanded (N) + complexity penalty
    A* timeouts are penalized- lower fitness is better
    """
    total_nodes_expanded = 0

    def h_func(state):
        return tree.evaluate(state)

    for board in training_boards:
        #time limit to 1 second, to kill bloated/infinite heuristics
        res = a_star_search(board, h_func, time_limit=1.0)
        # using the keys from original a_star_search
        if res.get("Success (Y/N)") == "Y":
            total_nodes_expanded += res["N"]
        else:
            # big penalty for failing to solve
            total_nodes_expanded += 5000

    avg_nodes = total_nodes_expanded / len(training_boards)

    # bonus for low computational complexity, so deep or complex trees receive a penalty
    complexity_penalty = tree.get_size() * 5.0

    return avg_nodes + complexity_penalty


def run_genetic_programming(training_boards, pop_size=20, generations=10, max_depth=4):
    """
    the evolutionary loop for generating heuristics
    """
    print(f"GP population- size: {pop_size}")
    population = [generate_random_tree(max_depth) for _ in range(pop_size)]

    best_overall_tree = None
    best_overall_fitness = float('inf')

    for gen in range(generations):
        fitness_scores = []
        for tree in population:
            fit = evaluate_gp_fitness(tree, training_boards)
            fitness_scores.append((fit, tree))

            if fit < best_overall_fitness:
                best_overall_fitness = fit
                best_overall_tree = copy.deepcopy(tree)

        fitness_scores.sort(key=lambda x: x[0])
        print(f"Gen {gen + 1:02d}/{generations} | Best Fitness: {fitness_scores[0][0]:<6.1f} | Formula: {fitness_scores[0][1]}")

        new_population = []

        # elitism: Keep top 2 individuals exactly as they are
        new_population.append(copy.deepcopy(fitness_scores[0][1]))
        new_population.append(copy.deepcopy(fitness_scores[1][1]))

        #fill the rest with tournament selection crossover or mutation
        while len(new_population) < pop_size:
            if random.random() < 0.7:
                p1 = min(random.sample(fitness_scores, 3), key=lambda x: x[0])[1]
                p2 = min(random.sample(fitness_scores, 3), key=lambda x: x[0])[1]
                child = crossover_trees(p1, p2)
            else:
                p = min(random.sample(fitness_scores, 3), key=lambda x: x[0])[1]
                child = mutate_tree(p, max_depth)
            new_population.append(child)

        population = new_population

    return best_overall_tree



# GEP
# instead of a tree- GEP individual is a fixed length linear string split to head and tail
# terminals and functions are shared from GP

GEP_MAX_ARITY = 2

def gep_random_head_symbol():
    """
    a head can hold a function or a terminal
    """
    if random.random() < 0.5:
        # a function symbol
        return random.choice(functions)
    # a terminal
    return random.choice(terminals)


def gep_create_genome(head_length):
    """
    build a random GEP genome: a head- functions or terminals, and a tail- terminals.
    The tail length guarantees the gene always decodes into a complete tree
    """
    tail_length = head_length * (GEP_MAX_ARITY - 1) + 1
    head = [gep_random_head_symbol() for _ in range(head_length)]
    tail = [random.choice(terminals) for _ in range(tail_length)]
    return head + tail


def gep_translate(genome):
    """
    fit a linear GEP genome into a GPNode expression tree.
    genome is read left to right: the first symbol is the root, and every function symbol pulls its
    required number of arguments from the next unused positions in the genome. Terminals become leaves.
    Because of the head/tail length rule, this always consumes a valid tree, and leftover tail symbols aren't used.
    """
    # create the root node from the first symbol
    def make_node(symbol):
        #function symbol
        if isinstance(symbol, str):
            return GPNode(is_terminal=False, value=symbol)
        return GPNode(is_terminal=True, value=symbol)

    root = make_node(genome[0])
    #queue of nodes still waiting for their children
    queue = [root]
    idx = 1
    while queue:
        node = queue.pop(0)
        if not node.is_terminal:
            # binary function needs exactly two children
            for _ in range(GEP_MAX_ARITY):
                if idx >= len(genome):
                    # for safety- fill with a terminal
                    child = GPNode(is_terminal=True, value=random.choice(terminals))
                else:
                    child = make_node(genome[idx])
                    idx += 1
                if node.left is None:
                    node.left = child
                else:
                    node.right = child
                queue.append(child)
    return root


def gep_mutate(genome, head_length):
    """
    point mutation- each position may change. A head position can mutate into any  symbol (function or terminal),
    a tail position may mutate into a terminal only, so the gene stays valid
    """
    new_genome = genome[:]
    for i in range(len(new_genome)):
        if random.random() < 0.15:
            if i < head_length:
                new_genome[i] = gep_random_head_symbol()
            else:
                new_genome[i] = random.choice(terminals)
    return new_genome


def gep_crossover(parent1, parent2):
    """
    one point crossover on the linear genome: pick a cut point and swap the tails of the two parents.
    Because both parents share the same length and head/tail structure, the children are valid genomes.
    """
    if len(parent1) < 2:
        return parent1[:], parent2[:]
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def evaluate_gep_fitness(genome, training_boards):
    """
    translate the genome to a tree, then score it with evaluate_gp_fitness
    """
    tree = gep_translate(genome)
    return evaluate_gp_fitness(tree, training_boards)


def run_gene_expression_programming(training_boards, pop_size=20, generations=10, head_length=6):
    """
    GEP evolutionary loop- similar to GP loop (tournament selection, elitism, crossover/mutation)
    but operates on linear genomes instead of trees, and decodes to a tree only for evaluation and reporting
    """
    print(f"GEP population- size: {pop_size}, head length: {head_length}")
    population = [gep_create_genome(head_length) for _ in range(pop_size)]

    best_overall_genome = None
    best_overall_fitness = float('inf')

    for gen in range(generations):
        fitness_scores = []
        for genome in population:
            fit = evaluate_gep_fitness(genome, training_boards)
            fitness_scores.append((fit, genome))

            if fit < best_overall_fitness:
                best_overall_fitness = fit
                best_overall_genome = genome[:]

        fitness_scores.sort(key=lambda x: x[0])
        best_tree_str = gep_translate(fitness_scores[0][1])
        print(f"Gen {gen + 1:02d}/{generations} | Best Fitness: {fitness_scores[0][0]:<6.1f} | Formula: {best_tree_str}")

        new_population = []

        # elitism: keep the top 2 genomes unchanged
        new_population.append(fitness_scores[0][1][:])
        new_population.append(fitness_scores[1][1][:])

        # fill the rest with tournament selection + crossover or mutation
        while len(new_population) < pop_size:
            if random.random() < 0.7:
                p1 = min(random.sample(fitness_scores, 3), key=lambda x: x[0])[1]
                p2 = min(random.sample(fitness_scores, 3), key=lambda x: x[0])[1]
                c1, c2 = gep_crossover(p1, p2)
                new_population.append(c1)
                if len(new_population) < pop_size:
                    new_population.append(c2)
            else:
                p = min(random.sample(fitness_scores, 3), key=lambda x: x[0])[1]
                new_population.append(gep_mutate(p, head_length))

        population = new_population

    # return the best genome and its decoded tree- for evaluation
    return best_overall_genome, gep_translate(best_overall_genome)


def gep_diversity_runs(training_boards, n_runs=3):
    """
    for the GP & GEP comparison:
    runs a method several times and returns the set of distinct best formulas found,
    which is a measure of the diversity of solutions a method produces across independent runs.
    """
    gp_formulas = []
    gep_formulas = []
    for _ in range(n_runs):
        gp_tree = run_genetic_programming(training_boards, pop_size=20, generations=10)
        gp_formulas.append(str(gp_tree))
        _, gep_tree = run_gene_expression_programming(training_boards, pop_size=20, generations=10)
        gep_formulas.append(str(gep_tree))
    return gp_formulas, gep_formulas


def main():

    #generate fast training boards
    print("6 boards for GP/GEP training")
    training_boards = []
    for d in [6, 7, 8, 9, 10, 11]:
        b, _, _ = generate_puzzle(target_depth=d, time_limit=5)
        if b: training_boards.append(b)

    if not training_boards:
        print("FAIL")
        return

    # run GP to evolve heuristic
    print("\n Evolving heuristic (GP)")
    t_start = time.time()
    best_gp_tree = run_genetic_programming(training_boards, pop_size=20, generations=10)
    gp_evolution_time = time.time() - t_start
    print(f"\n- GP completed in {gp_evolution_time:.1f}s")
    print(f"- GP best formula: {best_gp_tree}\n")

    #run GEP to evolve heuristic
    print(" Evolving heuristic (GEP)")
    t_start = time.time()
    best_gep_genome, best_gep_tree = run_gene_expression_programming(training_boards, pop_size=20, generations=10)
    gep_evolution_time = time.time() - t_start
    print(f"\n- GEP completed in {gep_evolution_time:.1f}s")
    print(f"- GEP best formula: {best_gep_tree}\n")

    def evolved_heuristic(state):
        return best_gp_tree.evaluate(state)

    def evolved_gep_heuristic(state):
        return best_gep_tree.evaluate(state)

    #load the full set (rh.txt) from assignment 1
    rh_file = sys.argv[1] if len(sys.argv) > 1 else "rh.txt"
    full_boards = parse_rh_file(rh_file)

    print("Comparison on full set (rh.txt)")
    print(f" {len(full_boards)} puzzles, comparison between H2 vs evolved GP vs evolved GEP\n")

    accum_h2 = {"N": [], "d": [], "time": []}
    accum_gp = {"N": [], "d": [], "time": []}
    accum_gep = {"N": [], "d": [], "time": []}

    # print header for detailed output
    header = f"{'Board':<7} | {'Method':<15} | {'Expanded (N)':<15} | {'Time (ms)':<10} | {'Path (d)':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for idx, board in enumerate(full_boards, 1):
        # evaluate H2
        t0 = time.time()
        r2 = a_star_search(board, heuristic_2_blockers_of_blockers, time_limit=10)
        t2 = (time.time() - t0) * 1000

        # evaluate GP evolved
        t0 = time.time()
        rgp = a_star_search(board, evolved_heuristic, time_limit=10)
        tgp = (time.time() - t0) * 1000

        # evaluate GEP evolved
        t0 = time.time()
        rgep = a_star_search(board, evolved_gep_heuristic, time_limit=10)
        tgep = (time.time() - t0) * 1000

        #keys from a* search- only count boards all three solved for a fair average
        if (r2.get("Success (Y/N)") == "Y" and rgp.get("Success (Y/N)") == "Y"
                and rgep.get("Success (Y/N)") == "Y"):
            accum_h2["N"].append(r2["N"])
            accum_h2["time"].append(t2)
            accum_h2["d"].append(r2["d"])

            accum_gp["N"].append(rgp["N"])
            accum_gp["time"].append(tgp)
            accum_gp["d"].append(rgp["d"])

            accum_gep["N"].append(rgep["N"])
            accum_gep["time"].append(tgep)
            accum_gep["d"].append(rgep["d"])

            #print status every 5 boards
            if idx % 5 == 0 or idx == len(full_boards):
                print(f"Test {idx:02d} | {'H2 (Advanced)':<15} | {r2['N']:<15} | {t2:<10.1f} | {r2['d']:<8}")
                print(f"Test {idx:02d} | {'GP (Evolved)':<15} | {rgp['N']:<15} | {tgp:<10.1f} | {rgp['d']:<8}")
                print(f"Test {idx:02d} | {'GEP (Evolved)':<15} | {rgep['N']:<15} | {tgep:<10.1f} | {rgep['d']:<8}")
                print("-" * len(header))

    # final summary table (quality + solution time on the test set)
    print("\n" + "=" * 60)
    print("Final summary on the full set (averages)")
    print("=" * 60)
    summary_header = f"{'Heuristic':<20} | {'Avg Nodes (N)':<15} | {'Avg Time (ms)':<15}"
    print(summary_header)
    print("-" * 60)
    if accum_h2["N"]:
        print(f"{'H2 (Original)':<20} | {np.mean(accum_h2['N']):<15.1f} | {np.mean(accum_h2['time']):<15.2f}")
        print(f"{'GP (Evolved)':<20} | {np.mean(accum_gp['N']):<15.1f} | {np.mean(accum_gp['time']):<15.2f}")
        print(f"{'GEP (Evolved)':<20} | {np.mean(accum_gep['N']):<15.1f} | {np.mean(accum_gep['time']):<15.2f}")
    else:
        print("No successful runs completed to average")
    print("=" * 60)

    #GP vs GEP comparison
    print("\n" + "=" * 60)
    print("GP vs GEP comparison")
    print("=" * 60)

    #measure diversity: run each method few extra times and count the distinct best formulas found
    print("\n formula diversity in independent runs")
    gp_formulas, gep_formulas = gep_diversity_runs(training_boards, n_runs=2)
    gp_diversity = len(set(gp_formulas))
    gep_diversity = len(set(gep_formulas))

    if accum_gp["N"]:
        gp_quality = np.mean(accum_gp["N"])
        gep_quality = np.mean(accum_gep["N"])
    else:
        gp_quality = gep_quality = float('nan')

    cmp_header = f"{'Method':<8} | {'Quality (avg N)':<16} | {'Diversity (distinct/2)':<22} | {'Gen. time (s)':<14}"
    print(cmp_header)
    print("-" * len(cmp_header))
    print(f"{'GP':<8} | {gp_quality:<16.1f} | {gp_diversity:<22} | {gp_evolution_time:<14.1f}")
    print(f"{'GEP':<8} | {gep_quality:<16.1f} | {gep_diversity:<22} | {gep_evolution_time:<14.1f}")
    print("=" * len(cmp_header))
    print("\nGP formulas across runs:")
    for f in gp_formulas:
        print(f"   {f}")
    print("GEP formulas across runs:")
    for f in gep_formulas:
        print(f"   {f}")


if __name__ == "__main__":
    main()