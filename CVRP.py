"""
AI lab- assignment 3, part 1
"""
import os
import time
import random
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# data structures
class Node:
    """
    struct of a location on the map
    can represent either the depot (demand = 0) or a customer
    """
    def __init__(self, node_id, x, y, demand=0):
        self.id = node_id
        self.x = x
        self.y = y
        self.demand = demand

class CVRPProblem:
    """
    holds the data of the CVRP instance: nodes, demands, and vehicle capacity.
    pre computes the distance matrix, which allows local search algorithms (SA, tabu) to get distances in O(1) time
    """
    def __init__(self, name, num_vehicles, capacity, nodes):
        self.name = name
        self.num_vehicles = num_vehicles
        self.capacity = capacity
        self.nodes = nodes
        # 0 is the warehouse
        self.depot = nodes[0]
        self.customers = nodes[1:]

        self.num_nodes = len(nodes)
        self.distance_matrix = np.zeros((self.num_nodes, self.num_nodes))
        self._calculate_distance_matrix()

    def _calculate_distance_matrix(self):
        """calculate euclidean distance between all nodes once"""
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    node_i = self.nodes[i]
                    node_j = self.nodes[j]
                    dist = np.sqrt((node_i.x - node_j.x)**2 + (node_i.y - node_j.y)**2)
                    self.distance_matrix[i][j] = dist

    def get_distance(self, from_id, to_id):
        """O(1) lookup for distance between two nodes"""
        return self.distance_matrix[from_id][to_id]

class CVRPSolution:
    """
    represent a feasible solution for the CVRP
    maintains a list of routes,one for each vehicle, and calculates the total cost
    """
    def __init__(self, problem):
        self.problem = problem
        # a list of lists- sub list is a route containing node objects
        self.routes = [[] for _ in range(problem.num_vehicles)]
        self.total_cost = 0.0

    def calculate_cost(self):
        """
        calculates the total distance of all routes,depot-first costumer-next costumers-depot
        """
        total = 0.0
        for route in self.routes:
            #skip empty routes
            if not route:
                continue
            dist = self.problem.get_distance(self.problem.depot.id, route[0].id)
            for i in range(len(route) - 1):
                dist += self.problem.get_distance(route[i].id, route[i+1].id)
            dist += self.problem.get_distance(route[-1].id, self.problem.depot.id)
            total += dist
        self.total_cost = total
        return total

def parse_cvrp_file(filename, default_vehicles=None):
    """
    parse a CVRP file into a CVRPProblem object, extracts vehicle capacity, node coordinates, and demands.
    """
    if not os.path.exists(filename):
        return None

    name = os.path.basename(filename)
    capacity = 0
    nodes_dict = {}
    demands_dict = {}
    parsing_coords = False
    parsing_demands = False

    with open(filename, 'r') as f:
        for line in f:
            s = line.strip()
            if not s: continue

            #detect which section we are entering
            if s.startswith("CAPACITY"):
                capacity = int(s.split()[-1])
            elif s.startswith("NODE_COORD_SECTION"):
                parsing_coords = True
                parsing_demands = False
                continue
            elif s.startswith("DEMAND_SECTION"):
                parsing_coords = False
                parsing_demands = True
                continue
            elif s.startswith("DEPOT_SECTION") or s == "EOF":
                break

            #parse data based on the current section
            if parsing_coords:
                parts = s.split()
                #convert to 0-indexed
                node_id = int(parts[0]) - 1
                x, y = float(parts[1]), float(parts[2])
                nodes_dict[node_id] = {'x': x, 'y': y}

            if parsing_demands:
                parts = s.split()
                node_id = int(parts[0]) - 1
                demand = int(parts[1])
                demands_dict[node_id] = demand

    # construct final list of node objects
    nodes = []
    for node_id in sorted(nodes_dict.keys()):
        nodes.append(Node(node_id, nodes_dict[node_id]['x'], nodes_dict[node_id]['y'], demands_dict.get(node_id, 0)))

    #parse vehicle count from filename "-kN" (A-n32-k5 has 5 vehicles)
    if default_vehicles:
        num_vehicles = default_vehicles
    else:
        match = re.search(r'-k(\d+)', name)
        if match:
            num_vehicles = int(match.group(1))
        else:
            #fallback
            num_vehicles = len(nodes)

    return CVRPProblem(name, num_vehicles, capacity, nodes)

#ACKLEY

def ackley_function(x):
    """
    compute the ackley function for d dimension array x, has many local min but one global min at 0.0, when all x_i = 0
    """
    d = len(x)
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x)) / d)
    return sum_sq_term + cos_term + a + np.exp(1)

    """
    SA to find the global minimum of the ackley function,
    has a progress based step size- large early, tiny later, which is the key to reach ackley global basin
    """

def simulated_annealing_ackley(d=10, initial_temp=10.0, cooling_rate=0.9995, max_iterations=100000):
    current_x = np.random.uniform(-32.768, 32.768, size=d)
    current_energy = ackley_function(current_x)

    best_x = np.copy(current_x)
    best_energy = current_energy
    temp = initial_temp

    for it in range(max_iterations):
        #step lowered linearly with search progress- exploration early, fine tuning later
        # floor of 0.001 lets it settle into basin
        frac = it / max_iterations
        step = 2.0 * (1 - frac) + 0.001
        neighbor_x = np.clip(current_x + np.random.uniform(-step, step, size=d), -32.768, 32.768)
        neighbor_energy = ackley_function(neighbor_x)

        diff = neighbor_energy - current_energy
        if diff < 0 or random.random() < np.exp(-diff / max(temp, 1e-9)):
            current_x = neighbor_x
            current_energy = neighbor_energy
            if current_energy < best_energy:
                best_x = np.copy(current_x)
                best_energy = current_energy

        temp *= cooling_rate
        if best_energy < 1e-4:
            break

    return best_x, best_energy

# CVRP helpers

def clone_routes(routes):
    """
    copy of the routes list to prevent reference modifications
    """
    copied_routes = []
    for route in routes:
        copied_routes.append(route[:])
    return copied_routes

def get_route_demand(route):
    """
    return the total capacity used in a route
    """
    total = 0
    for node in route:
        total += node.demand
    return total

def calculate_single_route_cost(problem, route):
    """
    calculate the distance of a single route (depot-customers-depot)
    """
    if not route: return 0.0
    cost = problem.get_distance(problem.depot.id, route[0].id)
    for i in range(len(route) - 1):
        cost += problem.get_distance(route[i].id, route[i + 1].id)
    cost += problem.get_distance(route[-1].id, problem.depot.id)
    return cost


def calculate_fitness(problem, routes, penalty_weight=10000.0):
    """
    added to prevent invalid solutions after X-n101-k25 got all invalid, because needed 26 vehicles instead 25 given.
    strategic oscillation, by the lecture of feasibility vs optimality:
    objective = total distance + penalty_weight*total_overload

    instead of blocking any move that would overload a truck which causes gridlock on tight instances,
    we allow the search to charge a heavy penalty for every capacity overload,
    this lets the meta heuristics shuffle customers between almost full trucks to escape tight local optima,
    while the penalty pulls the search to feasability.
    the penalty is big enough, 10000 per unit, that any feasible solution always beats not feasible one
    """
    total_dist = 0.0
    total_overload = 0.0

    for route in routes:
        if not route:
            continue
        total_dist += calculate_single_route_cost(problem, route)
        load = get_route_demand(route)
        if load > problem.capacity:
            total_overload += (load - problem.capacity)

    return total_dist + (penalty_weight * total_overload)


def verify_solution(problem, solution):
    """
    check solution validity:
    - every customer is served exactly once
    - no route exceeds vehicle capacity
    """
    served = []
    for route in solution.routes:
        for node in route:
            served.append(node.id)

    expected = set(c.id for c in problem.customers)
    served_set = set(served)

    if len(served) != len(served_set):
        return False, "Some customer served more than once"
    if served_set != expected:
        missing = expected - served_set
        extra = served_set - expected
        return False, f"Missing: {missing}, Extra: {extra}"

    for v, route in enumerate(solution.routes):
        load = get_route_demand(route)
        if load > problem.capacity:
            return False, f"Route {v} overloaded: {load} > {problem.capacity}"

    #check the number of vehicles used, because solution that uses more routes than the available fleet
    #(in tight instance like X-n101-k25) was reported as "valid".
    used_vehicles = 0
    for r in solution.routes:
        if r:
            used_vehicles += 1
    if used_vehicles > problem.num_vehicles:
        return False, f"Too many vehicles: {used_vehicles} > {problem.num_vehicles}"

    return True, "Valid"

# CVRP algorithms


def greedy_multi_stage_heuristic(problem):
    """
    multi stage heuristic:
    1 (like knapsack): sort customers by distance from depot (most far first)
    2 (routing): insert into route using nearest neighbor logic without violating capacity

    which all create a fast and feasible initial solution
    """
    solution = CVRPSolution(problem)
    unassigned = sorted(problem.customers, key=lambda c: problem.get_distance(problem.depot.id, c.id),reverse=True)

    vehicle_loads = [0] * problem.num_vehicles

    for customer in unassigned:
        best_vehicle = -1
        best_insert_cost = float('inf')

        for v in range(problem.num_vehicles):
            #check capacity constraint
            if vehicle_loads[v] + customer.demand <= problem.capacity:
                if not solution.routes[v]:
                    #cost of opening a new route
                    cost = problem.get_distance(problem.depot.id, customer.id) * 2
                else:
                    #cost of adding to the end of an existing route
                    last_customer = solution.routes[v][-1]
                    cost = problem.get_distance(last_customer.id, customer.id)

                if cost < best_insert_cost:
                    best_insert_cost = cost
                    best_vehicle = v

        if best_vehicle != -1:
            solution.routes[best_vehicle].append(customer)
            vehicle_loads[best_vehicle] += customer.demand
        else:
            #no vehicle has room- put in least loaded vehicle, meta heuristic repairs overloads
            least_loaded = min(range(problem.num_vehicles), key=lambda v: vehicle_loads[v])
            solution.routes[least_loaded].append(customer)
            vehicle_loads[least_loaded] += customer.demand

    solution.calculate_cost()
    return solution


def tabu_search(problem, initial_solution, max_iter=100, tabu_tenure=10):
    """
    tabu Search:
    explores the neighborhood using a relocate operator- moving a customer between routes
    maintains a tabu list to prevent returning to visited states so no cycles happen

    uses strategic oscillation (as seen in the lectures)-
    neighbours are evaluated with calculate_fitness (distance + capacity penalty)
    instead of being blocked, so the search can pass infeasiblility to escape gridlock
    """
    current_routes = clone_routes(initial_solution.routes)
    # pad up to num_vehicles with empty routes so a relocation can target an empty truck-
    # the initial solution may have fewer routes than the fleet, and strategic
    # oscillation should be free to open a new truck if useful
    while len(current_routes) < problem.num_vehicles:
        current_routes.append([])
    current_fitness = calculate_fitness(problem, current_routes)
    best_routes = clone_routes(current_routes)
    best_fitness = current_fitness

    #for safety,the best feasible solution is tracked by distance,
    #so we never return an overloaded solution when a feasible one was found during oscillation
    best_feasible_routes = None
    best_feasible_cost = float('inf')
    initial_feasible = True
    for r in current_routes:
        if get_route_demand(r) > problem.capacity:
            initial_feasible = False
            break
    if initial_feasible:
        best_feasible_routes = clone_routes(current_routes)
        best_feasible_cost = 0.0
        for r in current_routes:
            best_feasible_cost += calculate_single_route_cost(problem, r)

    #stores (customer_id, route_idx)
    tabu_list = {}

    for iter in range(max_iter):
        best_neighbor_routes = None
        best_neighbor_fitness = float('inf')
        best_move = None

        #sample 30 random neighbors to keep computation time low- good for large N
        for _ in range(30):
            non_empty = []
            for i, r in enumerate(current_routes):
                if len(r) > 0:
                    non_empty.append(i)
            if not non_empty: break

            from_idx = random.choice(non_empty)
            to_idx = random.choice(range(problem.num_vehicles))
            if from_idx == to_idx: continue

            #pick a customer to move
            cust_idx = random.randrange(len(current_routes[from_idx]))
            customer = current_routes[from_idx][cust_idx]

            # moving into a full truck is allowed- strategic oscillation- penalty prevents overload
            #create neighbor state
            neighbor_routes = clone_routes(current_routes)
            cust_obj = neighbor_routes[from_idx].pop(cust_idx)
            insert_pos = random.randint(0, len(neighbor_routes[to_idx]))
            neighbor_routes[to_idx].insert(insert_pos, cust_obj)

            neighbor_fitness = calculate_fitness(problem, neighbor_routes)

            # tabu status
            move_signature = (customer.id, to_idx)
            is_tabu = tabu_list.get(move_signature, 0) > iter

            #aspiration criterion- override tabu if this is the best solution
            if is_tabu and neighbor_fitness < best_fitness:
                is_tabu = False

            if not is_tabu and neighbor_fitness < best_neighbor_fitness:
                best_neighbor_fitness = neighbor_fitness
                best_neighbor_routes = neighbor_routes
                best_move = move_signature

        #move to the best non tabu neighbor
        if best_neighbor_routes is not None:
            current_routes = best_neighbor_routes
            current_fitness = best_neighbor_fitness
            #add to tabu list
            tabu_list[best_move] = iter + tabu_tenure

            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_routes = clone_routes(current_routes)

            # update best feasible if this state is feasible and cheaper
            current_feasible = True
            for r in current_routes:
                if get_route_demand(r) > problem.capacity:
                    current_feasible = False
                    break
            if current_feasible:
                feas_cost = 0.0
                for r in current_routes:
                    feas_cost += calculate_single_route_cost(problem, r)
                if feas_cost < best_feasible_cost:
                    best_feasible_routes = clone_routes(current_routes)
                    best_feasible_cost = feas_cost

    final_solution = CVRPSolution(problem)
    final_solution.routes = best_feasible_routes if best_feasible_routes is not None else best_routes
    #distance only for report
    final_solution.calculate_cost()
    return final_solution


def simulated_annealing(problem, initial_solution, initial_temp=100.0, cooling_rate=0.95, max_iter=200):
    """
    SA for CVRP:
    cools down the acceptance probability of worse moves

    also uses strategic oscillation
    """
    current_routes = clone_routes(initial_solution.routes)
    #pad up to num_vehicles with empty routes
    while len(current_routes) < problem.num_vehicles:
        current_routes.append([])
    current_fitness = calculate_fitness(problem, current_routes)
    # best_routes tracks the best by fitness
    best_routes = clone_routes(current_routes)
    best_fitness = current_fitness

    #track best feasible by distance
    best_feasible_routes = None
    best_feasible_cost = float('inf')
    initial_feasible = True
    for r in current_routes:
        if get_route_demand(r) > problem.capacity:
            initial_feasible = False
            break
    if initial_feasible:
        best_feasible_routes = clone_routes(current_routes)
        best_feasible_cost = 0.0
        for r in current_routes:
            best_feasible_cost += calculate_single_route_cost(problem, r)

    temp = initial_temp

    for _ in range(max_iter):
        non_empty = []
        for i, r in enumerate(current_routes):
            if len(r) > 0:
                non_empty.append(i)
        if not non_empty: break

        from_idx = random.choice(non_empty)
        to_idx = random.choice(range(problem.num_vehicles))
        if from_idx == to_idx: continue

        cust_idx = random.randrange(len(current_routes[from_idx]))
        customer = current_routes[from_idx][cust_idx]

        # moving into a full truck is allowed- strategic oscillation- penalty prevents overload
        #relocate operator
        neighbor_routes = clone_routes(current_routes)
        cust_obj = neighbor_routes[from_idx].pop(cust_idx)
        insert_pos = random.randint(0, len(neighbor_routes[to_idx]))
        neighbor_routes[to_idx].insert(insert_pos, cust_obj)

        neighbor_fitness = calculate_fitness(problem, neighbor_routes)
        diff = neighbor_fitness - current_fitness

        #metropolis acceptance criterion
        if diff < 0 or random.random() < np.exp(-diff / max(temp, 1e-9)):
            current_routes = neighbor_routes
            current_fitness = neighbor_fitness
            if current_fitness < best_fitness:
                best_routes = clone_routes(current_routes)
                best_fitness = current_fitness

            # update best feasible if this state is feasible and cheaper
            current_feasible = True
            for r in current_routes:
                if get_route_demand(r) > problem.capacity:
                    current_feasible = False
                    break
            if current_feasible:
                feas_cost = 0.0
                for r in current_routes:
                    feas_cost += calculate_single_route_cost(problem, r)
                if feas_cost < best_feasible_cost:
                    best_feasible_routes = clone_routes(current_routes)
                    best_feasible_cost = feas_cost
        #cool down
        temp *= cooling_rate

    sol = CVRPSolution(problem)
    # prefer the best feasible solution- considers best by fitness if none was feasible
    sol.routes = best_feasible_routes if best_feasible_routes is not None else best_routes
    sol.calculate_cost()
    return sol


def alns_search(problem, initial_solution, max_iter=50):
    """
    ALNS:
    uses destroy operator to take out parts of the solution,
    and repair operator to greedily put them back in optimal spots.

    uses also strategic oscillation
    """
    current_routes = clone_routes(initial_solution.routes)
    #pad up to num_vehicles with empty routes
    while len(current_routes) < problem.num_vehicles:
        current_routes.append([])
    current_fitness = calculate_fitness(problem, current_routes)
    best_routes = clone_routes(current_routes)
    best_fitness = current_fitness

    #track best feasible by distance
    best_feasible_routes = None
    best_feasible_cost = float('inf')
    initial_feasible = True
    for r in current_routes:
        if get_route_demand(r) > problem.capacity:
            initial_feasible = False
            break
    if initial_feasible:
        best_feasible_routes = clone_routes(current_routes)
        best_feasible_cost = 0.0
        for r in current_routes:
            best_feasible_cost += calculate_single_route_cost(problem, r)

    temp = 100.0

    for _ in range(max_iter):
        # copied so a rejected move wont corrupt the current state
        candidate_routes = clone_routes(current_routes)

        # destroy
        num_to_destroy = random.randint(2, max(3, len(problem.customers) // 4))
        positions = []
        for r, route in enumerate(candidate_routes):
            for c in range(len(route)):
                positions.append((r, c))
        if len(positions) <= num_to_destroy: continue

        to_remove = random.sample(positions, num_to_destroy)
        to_remove.sort(key=lambda x: x[1], reverse=True)

        removed_customers = []
        for r_idx, c_idx in to_remove:
            removed_customers.append(candidate_routes[r_idx].pop(c_idx))

        # repair- cheapest insertion- uses a soft penalty to prefer feasible slots,
        # which allows overloads to escape gridlock without overloading into one truck.
        for customer in removed_customers:
            best_r, best_i, best_score = -1, -1, float('inf')

            for r_idx, route in enumerate(candidate_routes):
                # soft over capacity bias- cheaper to insert where there is room
                route_load = get_route_demand(route)
                over_bias = 0.0
                if route_load + customer.demand > problem.capacity:
                    over_bias = 1000.0 * (route_load + customer.demand - problem.capacity)

                for i in range(len(route) + 1):
                    prev_node = problem.depot if i == 0 else route[i - 1]
                    next_node = problem.depot if i == len(route) else route[i]

                    #cost variation- add two new edges, remove the original one
                    inc = (problem.get_distance(prev_node.id, customer.id) +
                           problem.get_distance(customer.id, next_node.id) -
                           problem.get_distance(prev_node.id, next_node.id))

                    score = inc + over_bias
                    if score < best_score:
                        best_score = score
                        best_r, best_i = r_idx, i

            #execute best insertion- best_r is always set cause routes exist
            candidate_routes[best_r].insert(best_i, customer)

        candidate_fitness = calculate_fitness(problem, candidate_routes)

        # acceptance- like SA on fitness- allows strategic oscillation through penalized overloads,
        # and on rejection keeps the current state to prevent search freezing
        diff = candidate_fitness - current_fitness
        if diff < 0 or random.random() < np.exp(-diff / max(temp, 1e-9)):
            current_routes = candidate_routes
            current_fitness = candidate_fitness
            if current_fitness < best_fitness:
                best_routes = clone_routes(current_routes)
                best_fitness = current_fitness

            # update best feasible if this state is feasible and cheaper
            current_feasible = True
            for r in current_routes:
                if get_route_demand(r) > problem.capacity:
                    current_feasible = False
                    break
            if current_feasible:
                feas_cost = 0.0
                for r in current_routes:
                    feas_cost += calculate_single_route_cost(problem, r)
                if feas_cost < best_feasible_cost:
                    best_feasible_routes = clone_routes(current_routes)
                    best_feasible_cost = feas_cost

        temp *= 0.95

    sol = CVRPSolution(problem)
    sol.routes = best_feasible_routes if best_feasible_routes is not None else best_routes
    sol.calculate_cost()
    return sol


def aco_search(problem, num_ants=10, max_iter=20):
    """
    ACO:
    ants build solutions with probability based on pheromones (tau) and heuristic info (eta)
    """

    # initialize pheromone matrix
    pheromones = np.full((problem.num_nodes, problem.num_nodes), 0.1)
    best_sol = None

    for _ in range(max_iter):
        solutions = []
        for _ in range(num_ants):
            sol = CVRPSolution(problem)
            unvisited = set(node.id for node in problem.customers)

            #build routes for each vehicle
            for v in range(problem.num_vehicles):
                if not unvisited: break
                curr_node_id = problem.depot.id
                curr_cap = 0

                while unvisited:
                    probs = {}
                    #calculate probabilities for valid next steps
                    for target_id in unvisited:
                        target_node = problem.nodes[target_id]
                        if curr_cap + target_node.demand <= problem.capacity:
                            dist = problem.get_distance(curr_node_id, target_id)
                            # heuristic- 1/distance
                            eta = 1.0 / dist if dist > 0 else 0
                            tau = pheromones[curr_node_id][target_id]
                            probs[target_id] = (tau ** 1.0) * (eta ** 2.0)

                    if not probs: break

                    #selection based on probabilities
                    total_prob = sum(probs.values())
                    rand_val = random.uniform(0, total_prob)
                    cumulative = 0.0
                    next_node_id = None
                    for tid, p in probs.items():
                        cumulative += p
                        if cumulative >= rand_val:
                            next_node_id = tid
                            break

                    #move the ant
                    sol.routes[v].append(problem.nodes[next_node_id])
                    unvisited.remove(next_node_id)
                    curr_cap += problem.nodes[next_node_id].demand
                    curr_node_id = next_node_id

            # on tight instance the num_vehicles loop above can end with unvisited customers,
            # so before I noticed, the customers were dropped.
            # the ants solution was not good- best_sol stayed none, and returned inf, which raised RuntimeWarning
            # so now I put the remaining customers into extra routes so every ant returns a full solution
            while unvisited:
                extra_route = []
                extra_cap = 0
                for cid in sorted(unvisited):
                    if extra_cap + problem.nodes[cid].demand <= problem.capacity:
                        extra_route.append(problem.nodes[cid])
                        extra_cap += problem.nodes[cid].demand
                for node in extra_route:
                    unvisited.discard(node.id)
                if not extra_route:
                    #single customer whose demand alone exceeds capacity- give him his own route to avoid infinite loop
                    cid = next(iter(unvisited))
                    extra_route.append(problem.nodes[cid])
                    unvisited.discard(cid)
                sol.routes.append(extra_route)

            sol.calculate_cost()
            solutions.append(sol)

            #keep track of global best valid solution
            if not unvisited:
                if best_sol is None or sol.total_cost < best_sol.total_cost:
                    best_sol = sol

        # pdate pheromone- evaporation & deposit
        #evaporation rate rho=0.1
        pheromones *= 0.9
        for s in solutions:
            if s.total_cost == 0: continue
            # better routes leave more pheromone
            deposit = 100.0 / s.total_cost
            for r in s.routes:
                if not r: continue
                pheromones[problem.depot.id][r[0].id] += deposit
                for i in range(len(r) - 1):
                    pheromones[r[i].id][r[i + 1].id] += deposit
                pheromones[r[-1].id][problem.depot.id] += deposit

    return best_sol


def lds_route_builder(problem, vehicle_idx, unvisited, current_route, current_cost, max_disc, curr_disc):
    """
    recursive LDS to build a single route-
    allows deviating from greedy choice (nearest neighbor) up to max_disc times
    """
    if not unvisited:
        return current_route, current_cost

    last_id = current_route[-1].id if current_route else problem.depot.id
    curr_cap = 0
    for n in current_route:
        curr_cap += n.demand

    valid_customers = []
    for c in unvisited:
        if curr_cap + c.demand <= problem.capacity:
            valid_customers.append(c)
    if not valid_customers:
        return current_route, current_cost

    #sort choices greedily by distance
    valid_customers.sort(key=lambda c: problem.get_distance(last_id, c.id))

    best_route = current_route
    best_cost = float('inf')

    for i, customer in enumerate(valid_customers):
        # i is the number of discrepancies taken, i=0 is the greedy choice
        if curr_disc + i <= max_disc:
            next_route = current_route + [customer]
            next_unvisited = unvisited.copy()
            next_unvisited.remove(customer)
            next_cost = current_cost + problem.get_distance(last_id, customer.id)

            #traverse branch
            res_route, res_cost = lds_route_builder(
                problem, vehicle_idx, next_unvisited, next_route, next_cost, max_disc, curr_disc + i)

            # evaluate total cost
            final_cost = res_cost + problem.get_distance(res_route[-1].id, problem.depot.id)
            if final_cost < best_cost:
                best_cost = final_cost
                best_route = res_route

    return best_route, best_cost


def branch_and_bound_lds(problem, max_discrepancies=1):
    """
    branch and bound/LDS: Sequential route building-
    B&B explodes on CVRP, but LDS gives an approximation by exploring a limited tree
    """
    solution = CVRPSolution(problem)
    unvisited = set(problem.customers)

    for v in range(problem.num_vehicles):
        if not unvisited: break
        route, _ = lds_route_builder(
            problem, v, unvisited, current_route=[],
            current_cost=0.0, max_disc=max_discrepancies, curr_disc=0
        )
        solution.routes[v] = route
        for cust in route:
            unvisited.remove(cust)

    solution.calculate_cost()
    return solution


# GA with island model- memetic
# from assignment 2- permutation GA (OX crossover,swap mutation,memetic 2-opt) and extended with the island model

def ga_two_opt_route(problem, route):
    """
    2-opt local search on a single route- memetic step
    reverses segments to remove crossing edges and returns an improved route
    same 2-opt idea from assignment 2, applied per route for CVRP
    """
    if len(route) < 4:
        return route

    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue
                #nodes around the two edges being swapped
                a = problem.depot if i == 0 else best[i - 1]
                b = best[i]
                c = best[j]
                d = problem.depot if j == len(best) - 1 else best[j + 1]

                d_current = (problem.get_distance(a.id, b.id) +
                             problem.get_distance(c.id, d.id))
                d_new = (problem.get_distance(a.id, c.id) +
                         problem.get_distance(b.id, d.id))

                if d_new < d_current - 1e-9:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    improved = True
    return best


def ga_flatten(routes):
    """flatten route solution into a single ordered list of node objects"""
    flat = []
    for route in routes:
        for node in route:
            flat.append(node)
    return flat


def ga_split_into_routes(problem, sequence):
    """
    split a flat sequence of customers into capacity feasible routes

    greedy sequential split- keep adding to the current route until the next customer would overflow capacity, then start a new route.
    respects the vehicle count by allowing at most num_vehicles routes, if the
    sequence needs more, the overflow is packed to the last route, in which the feasibility gate/repair handles
    """
    routes = []
    current = []
    load = 0
    for node in sequence:
        if load + node.demand <= problem.capacity:
            current.append(node)
            load += node.demand
        else:
            if current:
                routes.append(current)
            current = [node]
            load = node.demand
    if current:
        routes.append(current)

    # padding num_vehicles with empty routes, so CVRPSolution shape is stable
    while len(routes) < problem.num_vehicles:
        routes.append([])
    return routes


def ga_create_random_individual(problem):
    """create a random feasible list of routes"""
    customers = problem.customers[:]
    random.shuffle(customers)
    return ga_split_into_routes(problem, customers)


def ga_order_crossover(problem, parent1, parent2):
    """
    order crossover OX adapted for CVRP- the same from assignment 2
    - flatten both parents into customer sequences
    - copy a random part from parent1
    - fill remaining positions from parent2 in order
    - resplit the child sequence into capacity feasible routes
    """
    seq1 = ga_flatten(parent1)
    seq2 = ga_flatten(parent2)
    n = len(seq1)
    if n < 2:
        return clone_routes(parent1)

    start, end = sorted(random.sample(range(n), 2))

    #slice of parent1, by node id for membership testing
    slice_ids = set(node.id for node in seq1[start:end])

    child_seq = [None] * n
    child_seq[start:end] = seq1[start:end]

    # fill the rest from parent2 in order, skipping ids already placed
    fill_pos = end % n
    p2_pos = end % n
    filled = 0
    needed = n - (end - start)
    while filled < needed:
        candidate = seq2[p2_pos % n]
        if candidate.id not in slice_ids:
            child_seq[fill_pos % n] = candidate
            fill_pos += 1
            filled += 1
        p2_pos += 1

    return ga_split_into_routes(problem, child_seq)


def ga_swap_mutation(problem, routes, mutation_rate=0.15):
    """
    swap mutation- same operator from assignment 2
    with probability mutation_rate, swaps two random customers in the flattened sequence,
    then resplits into feasible routes
    """
    if random.random() > mutation_rate:
        return routes

    seq = ga_flatten(routes)
    if len(seq) < 2:
        return routes

    i, j = random.sample(range(len(seq)), 2)
    seq[i], seq[j] = seq[j], seq[i]
    return ga_split_into_routes(problem, seq)


def ga_fitness(problem, routes):
    """
    fitness=1/(cost+penalty), penalty for capacity violations, so the GA is pushed to feasibility
    """
    cost = 0.0
    for r in routes:
        cost += calculate_single_route_cost(problem, r)
    penalty = 0
    for r in routes:
        load = get_route_demand(r)
        if load > problem.capacity:
            penalty += (load - problem.capacity) * 10000
    return 1.0 / (cost + penalty + 1e-6)


def ga_tournament_select(population, fitnesses, k=5):
    """
    tournament selection- pick the best of k random individuals
    """
    picks = random.sample(range(len(population)), min(k, len(population)))
    best_idx = picks[0]
    for idx in picks:
        if fitnesses[idx] > fitnesses[best_idx]:
            best_idx = idx
    return population[best_idx]


def genetic_algorithm_island(problem, pop_size=40, generations=100,
                             num_islands=4, migration_freq=10, migration_size=2,
                             mutation_rate=0.15, elitism=2, memetic_freq=5):
    """
    GA with island model & memetic 2-opt

    island model: num_islands sub populations evolve independently
    every migration_freq generations, the best migration_size individuals from each
    island replace the worst in the next island (ring topology), which preserves diversity

    memetic: every memetic_freq generations, children are refined with 2-opt
    """
    #initialize islands
    islands = []
    for _ in range(num_islands):
        pop = []
        for _ in range(pop_size):
            pop.append(ga_create_random_individual(problem))
        islands.append(pop)

    global_best_routes = None
    global_best_cost = float('inf')

    for gen in range(generations):
        for isl in range(num_islands):
            pop = islands[isl]
            fitnesses = []
            for ind in pop:
                fitnesses.append(ga_fitness(problem, ind))

            #track global best- feasible only
            for ind, fit in zip(pop, fitnesses):
                ind_feasible = True
                for r in ind:
                    if get_route_demand(r) > problem.capacity:
                        ind_feasible = False
                        break
                if ind_feasible:
                    cost = 0.0
                    for r in ind:
                        cost += calculate_single_route_cost(problem, r)
                    if cost < global_best_cost:
                        global_best_cost = cost
                        global_best_routes = clone_routes(ind)

            #build next generation
            order = sorted(range(len(pop)), key=lambda idx: fitnesses[idx], reverse=True)
            new_pop = []
            for e in range(elitism):
                new_pop.append(clone_routes(pop[order[e]]))

            while len(new_pop) < pop_size:
                p1 = ga_tournament_select(pop, fitnesses)
                p2 = ga_tournament_select(pop, fitnesses)
                child = ga_order_crossover(problem, p1, p2)
                child = ga_swap_mutation(problem, child, mutation_rate)

                if gen % memetic_freq == 0:
                    improved_child = []
                    for r in child:
                        improved_child.append(ga_two_opt_route(problem, r))
                    child = improved_child

                new_pop.append(child)

            islands[isl] = new_pop

        #ring topology
        if gen > 0 and gen % migration_freq == 0:
            migrants = []
            for isl in range(num_islands):
                pop = islands[isl]
                fits = []
                for ind in pop:
                    fits.append(ga_fitness(problem, ind))
                order = sorted(range(len(pop)), key=lambda idx: fits[idx], reverse=True)
                chosen_migrants = []
                for mig in range(migration_size):
                    chosen_migrants.append(clone_routes(pop[order[mig]]))
                migrants.append(chosen_migrants)

            for isl in range(num_islands):
                target = (isl + 1) % num_islands
                target_pop = islands[target]
                target_fits = []
                for ind in target_pop:
                    target_fits.append(ga_fitness(problem, ind))
                #replace the worst individuals in the target island
                worst_order = sorted(range(len(target_pop)), key=lambda idx: target_fits[idx])
                for m, migrant in enumerate(migrants[isl]):
                    islands[target][worst_order[m]] = migrant

    # build final solution object
    sol = CVRPSolution(problem)
    if global_best_routes is not None:
        # trim empty routes but keep the list shape valid
        sol.routes = []
        for r in global_best_routes:
            if r:
                sol.routes.append(r)
        # padding to num_vehicles
        while len(sol.routes) < problem.num_vehicles:
            sol.routes.append([])
    sol.calculate_cost()
    return sol

def plot_routes(problem, solution, method_name, filename=None):
    """
    route map:
    shows depot, numbered customer nodes, colored vehicle routes, and
    output matrix- cost & per vehicle routes.
    the route map and the output matrix are drawn in two side-by-side panels
    so the matrix text never overlaps the customers, even on dense instances.
    """
    # two panels: wide route map on the left, narrow matrix panel on the right
    fig, (ax, ax_txt) = plt.subplots(
        1, 2, figsize=(13, 10),
        gridspec_kw={'width_ratios': [4, 1]}
    )

    # pepot- circle with id
    dx, dy = problem.depot.x, problem.depot.y
    ax.scatter(dx, dy, c='black', s=300, marker='o',
               edgecolors='white', linewidths=2, zorder=6)
    ax.annotate(str(problem.depot.id), (dx, dy), color='white', fontsize=10,
                ha='center', va='center', zorder=7, fontweight='bold')

    # customer nodes- white circles with id numbers
    for c in problem.customers:
        ax.scatter(c.x, c.y, c='white', s=300, marker='o',
                   edgecolors='black', linewidths=2, zorder=5)
        ax.annotate(str(c.id), (c.x, c.y), fontsize=9, ha='center',
                    va='center', zorder=6, fontweight='bold')

    # draw each vehicle's route with different color
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(solution.routes))))
    for v, route in enumerate(solution.routes):
        if not route:
            continue
        path_x = [problem.depot.x] + [n.x for n in route] + [problem.depot.x]
        path_y = [problem.depot.y] + [n.y for n in route] + [problem.depot.y]
        ax.plot(path_x, path_y, '-', color=colors[v % len(colors)],
                linewidth=2.5, alpha=0.85, zorder=3, label=f'vehicle {v}')

    ax.set_title(f"CVRP Routes - {method_name} (cost = {solution.total_cost:.1f})",
                 fontsize=13)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # output matrix text
    matrix_lines = [f"{solution.total_cost:.1f} 0"]
    for route in solution.routes:
        if route:
            matrix_lines.append("0 " + " ".join(str(n.id) for n in route) + " 0")
        else:
            matrix_lines.append("0 0")
    matrix_text = "Output\n" + "\n".join(matrix_lines)

    # the right panel holds ONLY the output matrix (no axes/ticks), so the text
    # has its own dedicated space and can never cover the routes on the left
    ax_txt.axis('off')
    # scale the font down a little when there are many routes so it always fits
    n_lines = len(matrix_lines)
    font_size = 9 if n_lines <= 26 else max(5, int(9 * 26 / n_lines))
    ax_txt.text(0.0, 1.0, matrix_text, transform=ax_txt.transAxes,
                fontsize=font_size, family='monospace', va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison_bars(accum, instance_name, optimal=None, filename=None):
    """
    comparison bar chart of algorithm performance.
    bars show mean cost across seeds with std error bars, a red marker shows the minimum cost of each algorithm
    """
    #sort algorithms by mean cost- best on the left
    algos = sorted(accum.keys(), key=lambda a: np.mean(accum[a]["costs"]))
    bests = [min(accum[a]["costs"]) for a in algos]
    means = [np.mean(accum[a]["costs"]) for a in algos]
    stds = [np.std(accum[a]["costs"]) for a in algos]

    x = np.arange(len(algos))
    fig, ax = plt.subplots(figsize=(11, 6))

    #mean bars with std error bars
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.75,
                  color='steelblue', edgecolor='navy',
                  label='Mean cost (bar), ± std (error bar)')

    # best cost markers
    ax.scatter(x, bests, color='red', marker='D', zorder=5, s=55,
               edgecolors='darkred', label='Best cost (min over seeds)')

    # numeric labels above each bar
    for i, (m, b) in enumerate(zip(means, bests)):
        ax.annotate(f"{m:.0f}", (i, m + max(means) * 0.01),
                    ha='center', va='bottom', fontsize=8, color='navy')

    if optimal:
        ax.axhline(y=optimal, color='green', linestyle='--', linewidth=1.5,
                   label=f'Known optimum ({optimal})')

    ax.set_title(f"Algorithm Comparison - {instance_name}", fontsize=13)
    ax.set_xlabel("Algorithm (sorted best to worst by mean cost)")
    ax.set_ylabel("Total route cost")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=30, ha='right')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # zoom y axis
    lo = min(bests) * 0.95
    hi = (max(means) + max(stds)) * 1.02
    ax.set_ylim(lo, hi)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def print_solution(solution, method_name, elapsed_time):
    """
    prints a summary line for the algorithm run
    """
    print(f"{method_name:<20} | Time: {elapsed_time * 1000:7.2f} ms | Cost: {solution.total_cost:.2f}")


def print_detailed_output(solution):
    """
    prints route matrix
    """
    print(f"\n{solution.total_cost:.1f} 0")
    for route in solution.routes:
        if route:
            route_str = " ".join(str(node.id) for node in route)
            print(f"0 {route_str} 0")
        else:
            print("0 0")


def run_all_on_problem(problem, seed):
    """runs all 6 algorithms once on a problem with a given seed"""
    random.seed(seed)
    np.random.seed(seed)

    results = {}
    #collect the solution objects for plotting- results holds (cost, time, valid),
    #solutions holds CVRPSolution objects- key is the algorithm name
    solutions = {}
    #time- elapsed and CPU
    t0, c0 = time.time(), time.process_time()
    initial_sol = greedy_multi_stage_heuristic(problem)
    t1, c1 = time.time(), time.process_time()
    valid, _ = verify_solution(problem, initial_sol)
    results["Multi stage"] = (initial_sol.total_cost, t1 - t0, c1 - c0, valid)
    solutions["Multi stage"] = initial_sol

    t0, c0 = time.time(), time.process_time()
    ts_sol = tabu_search(problem, initial_sol, max_iter=100)
    t1, c1 = time.time(), time.process_time()
    valid, _ = verify_solution(problem, ts_sol)
    results["Tabu search"] = (ts_sol.total_cost, t1 - t0, c1 - c0, valid)
    solutions["Tabu search"] = ts_sol

    t0, c0 = time.time(), time.process_time()
    sa_sol = simulated_annealing(problem, initial_sol, max_iter=2000)
    t1, c1 = time.time(), time.process_time()
    valid, _ = verify_solution(problem, sa_sol)
    results["SA"] = (sa_sol.total_cost, t1 - t0, c1 - c0, valid)
    solutions["SA"] = sa_sol

    t0, c0 = time.time(), time.process_time()
    alns_sol = alns_search(problem, initial_sol, max_iter=200)
    t1, c1 = time.time(), time.process_time()
    valid, _ = verify_solution(problem, alns_sol)
    results["ALNS"] = (alns_sol.total_cost, t1 - t0, c1 - c0, valid)
    solutions["ALNS"] = alns_sol

    t0, c0 = time.time(), time.process_time()
    aco_sol = aco_search(problem, num_ants=10, max_iter=50)
    t1, c1 = time.time(), time.process_time()
    if aco_sol:
        valid, _ = verify_solution(problem, aco_sol)
        results["ACO"] = (aco_sol.total_cost, t1 - t0, c1 - c0, valid)
        solutions["ACO"] = aco_sol
    else:
        results["ACO"] = (float('inf'), t1 - t0, c1 - c0, False)

    #GA with island model- memetic
    t0, c0 = time.time(), time.process_time()
    ga_sol = genetic_algorithm_island(problem, pop_size=40, generations=100)
    t1, c1 = time.time(), time.process_time()
    valid, _ = verify_solution(problem, ga_sol)
    results["GA island"] = (ga_sol.total_cost, t1 - t0, c1 - c0, valid)
    solutions["GA island"] = ga_sol

    if len(problem.customers) <= 31:
        t0, c0 = time.time(), time.process_time()
        lds_sol = branch_and_bound_lds(problem, max_discrepancies=2)
        t1, c1 = time.time(), time.process_time()
        valid, _ = verify_solution(problem, lds_sol)
        results["B&B(LDS)"] = (lds_sol.total_cost, t1 - t0, c1 - c0, valid)
        solutions["B&B(LDS)"] = lds_sol

    #return the solution objects with results
    return results, solutions


def get_example_problem():
    """
    create and return the example problem from the assignment
    """
    nodes = [
        #depot
        Node(0, 0, 0, 0),
        Node(1, 0, 10, 3),
        Node(2, -10, 10, 3),
        Node(3, 0, -10, 3),
        Node(4, 10, -10, 3),
    ]
    return CVRPProblem("example-n5-k4", num_vehicles=4, capacity=10, nodes=nodes)


def run_ackley_all_algorithms():
    """
    sanity check for the algorithms: verify that our search strategies can find the global minimum of the continuous
    ackley function (min=0). ackley is continuous while CVRP methods are discrete, so adapting the search of each
    algorithm to a continuous vector is needed.
    - SA: simulated annealing
    - GA: real valued genetic algorithm- blend crossover, gaussian mutation
    - Tabu Search: neighborhood of perturbed vectors & a short tabu memory
    - ALNS: destroy/repair on vector components
    - ACO: continuous ACO- sample new points around an archive of good ones
    - Greedy: greedy local search- continuous analog of multi stage greedy heuristic
     B&B/LDS is a discrete tree search- so it is excluded here
    run over 3 seeds and we report best/mean/std
    """
    print("ACKLEY")

    d = 10
    bound = 32.768

    # using the ackley function defined once at the top of the file
    def ackley(x):
        return ackley_function(np.asarray(x, dtype=float))

    def random_point():
        return [random.uniform(-bound, bound) for _ in range(d)]

    def clamp(x):
        return [max(-bound, min(bound, xi)) for xi in x]

    #using early made ackley for sa
    def sa_ackley():
        return simulated_annealing_ackley()[1]

    #GA
    def ga_ackley(pop_size=60, generations=400):
        pop = [random_point() for _ in range(pop_size)]
        best_val = float('inf')
        for _ in range(generations):
            scored = sorted(pop, key=ackley)
            best_val = min(best_val, ackley(scored[0]))
            #elitism
            new_pop = [scored[0][:], scored[1][:]]
            while len(new_pop) < pop_size:
                p1 = min(random.sample(pop, 3), key=ackley)
                p2 = min(random.sample(pop, 3), key=ackley)
                #crossover
                child = [(a + b) / 2.0 for a, b in zip(p1, p2)]
                for i in range(d):
                    if random.random() < 0.2:
                        # gaussian mutation
                        child[i] += random.gauss(0, 1.0)
                new_pop.append(clamp(child))
            pop = new_pop
        return best_val

    #Tabu search: perturbed vector neighborhood & short tabu memory
    def tabu_ackley(max_iter=4000, step=1.0):
        current = random_point()
        best = current[:]
        best_val = ackley(best)
        # recent visited points to avoid revisiting
        tabu = []
        for _ in range(max_iter):
            neighbors = []
            for _ in range(20):
                cand = clamp([xi + random.gauss(0, step) for xi in current])
                key = tuple(round(c, 1) for c in cand)
                if key not in tabu:
                    neighbors.append((ackley(cand), cand, key))
            if not neighbors:
                continue
            neighbors.sort(key=lambda t: t[0])
            val, current, key = neighbors[0]
            tabu.append(key)
            if len(tabu) > 50:
                tabu.pop(0)
            if val < best_val:
                best_val, best = val, current[:]
            # gradually shrink the neighborhood
            step *= 0.999
        return best_val

    #ALNS: destroy/repair on vector components
    def alns_ackley(max_iter=5000):
        current = random_point()
        best_val = ackley(current)
        temp = 5.0
        for _ in range(max_iter):
            cand = current[:]
            #destroy: pick a random subset of coordinates
            k = random.randint(1, d)
            idxs = random.sample(range(d), k)
            #repair: resample those coordinates near the current value
            for i in idxs:
                cand[i] = max(-bound, min(bound, cand[i] + random.gauss(0, 1.0)))
            cur_val, cand_val = ackley(current), ackley(cand)
            if cand_val < cur_val or random.random() < np.exp(-(cand_val - cur_val) / max(temp, 1e-9)):
                current = cand
            best_val = min(best_val, ackley(current))
            temp *= 0.999
        return best_val

    #ACO continuous: archive of good solutions
    def aco_ackley(archive_size=20, iterations=300, ants=20):
        archive = [random_point() for _ in range(archive_size)]
        archive.sort(key=ackley)
        best_val = ackley(archive[0])
        for _ in range(iterations):
            new_solutions = []
            for _ in range(ants):
                #pick a guide solution from archive and sample around it
                guide = archive[random.randint(0, archive_size - 1)]
                #spread= mean distance of the archive on each dimension
                cand = []
                for i in range(d):
                    spread = sum(abs(guide[i] - s[i]) for s in archive) / archive_size
                    spread = max(spread, 0.01)
                    cand.append(max(-bound, min(bound, random.gauss(guide[i], spread))))
                new_solutions.append(cand)
            #keep the best archive_size solutions overall
            archive = sorted(archive + new_solutions, key=ackley)[:archive_size]
            best_val = min(best_val, ackley(archive[0]))
        return best_val


    def greedy_ackley(restarts=15, iters=1500, step=0.5):
        """
        Greedy with random restarts: analog of multi stage greedy heuristic- always steps to a better neighbour and
        never accepts a worse one, so it gets trapped by ackley's many local minima
        """
        best_val = float('inf')
        for _ in range(restarts):
            current = random_point()
            cur_val = ackley(current)
            for _ in range(iters):
                cand = clamp([xi + random.gauss(0, step) for xi in current])
                if ackley(cand) < cur_val:
                    current = cand
                    cur_val = ackley(cand)
            best_val = min(best_val, cur_val)
        return best_val

    algorithms = [
        ("SA", sa_ackley),
        ("GA", ga_ackley),
        ("Tabu", tabu_ackley),
        ("ALNS", alns_ackley),
        ("ACO", aco_ackley),
        ("Greedy", greedy_ackley),
    ]

    seeds = [42, 123, 999]
    print(f"{'Algorithm':<12} {'Best':>10} {'Mean':>10} {'Std':>8}")
    for name, fn in algorithms:
        vals = []
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            vals.append(fn())
        print(f"{name:<12} {min(vals):>10.6f} {np.mean(vals):>10.6f} {np.std(vals):>8.6f}")
    print("  (target global minimum = 0.0)\n")


def main():
    run_ackley_all_algorithms()

    print("CVRP MH")
    instance_files = [
        "P-n16-k8.vrp",
        "E-n22-k4.vrp",
        "A-n32-k5.vrp",
        "A-n80-k10.vrp",
        "X-n101-k25.vrp",
        "M-n200-k17.vrp",
    ]

    #best known solution
    bks = {
        "example-n5-k4": 80.645,
        "P-n16-k8": 450,
        "E-n22-k4": 375,
        "A-n32-k5":784,
        "A-n80-k10": 1763,
        "M-n200-k17": 1275,
        "X-n101-k25": 27591,
    }
    seeds = [42, 123, 999]

    all_problems = [get_example_problem()]

    for filename in instance_files:
        prob = parse_cvrp_file(filename)
        if prob is not None:
            all_problems.append(prob)
        else:
            print(f"\nfile not found: {filename}")

    for problem in all_problems:
        print(f"\nproblem: {problem.name} " f"(customers={len(problem.customers)}, vehicles={problem.num_vehicles}, "
              f"capacity={problem.capacity}) ")

        accum = {}
        best_cost_seen = {}
        best_solution_seen = {}
        for seed in seeds:
            res, sols = run_all_on_problem(problem, seed)
            for alg, (cost, t0, t1, valid) in res.items():
                if alg not in accum:
                    accum[alg] = {"costs": [], "time_elapsed": [], "time_cpu": [], "valids": []}
                accum[alg]["costs"].append(cost)
                accum[alg]["time_elapsed"].append(t0)
                accum[alg]["time_cpu"].append(t1)
                accum[alg]["valids"].append(valid)
                if alg in sols and (alg not in best_cost_seen or cost < best_cost_seen[alg]):
                    best_cost_seen[alg] = cost
                    best_solution_seen[alg] = sols[alg]

        base = problem.name.replace('.vrp', '')
        instance_bks = bks.get(base)
        bks_label = f"(BKS={instance_bks})" if instance_bks else ""

        print(f"{'Algorithm':<15} {'Best':>10} {'Mean':>10} {'Std':>8} {'Elapsed(s)':>11} {'CPU(s)':>8} {'Gap%':>8} {'Valid':>6} {bks_label}")
        for alg, data in accum.items():
            costs = [c for c in data["costs"] if c != float('inf')]
            if not costs:
                print(f"{alg:<16} {'FAILED (no valid solution)':>44}")
                continue
            best = min(costs)
            mean = np.mean(costs)
            std = np.std(costs)
            mean_time = np.mean(data["time_elapsed"])
            cpu_time = np.mean(data["time_cpu"])
            all_valid = all(data["valids"])
            if instance_bks:
                gap = 100.0 * (best - instance_bks) / instance_bks
                gap_str = f"{gap:>7.1f}%"
            else:
                gap_str = f"{'-':>8}"

            print(f"{alg:<15} {best:>10.2f} {mean:>10.2f} {std:>8.2f} {mean_time:>11.2f} {cpu_time:>8.2f} {gap_str:>8} {str(all_valid):>6}")

        if best_cost_seen:
            best_algo = min(best_cost_seen, key=best_cost_seen.get)
            plot_routes(problem, best_solution_seen[best_algo], best_algo,
                        filename=f"plot_{base}_routes.png")

            print(f"\ndetailed output for best algorithm ({best_algo}):")
            print_detailed_output(best_solution_seen[best_algo])

        plot_comparison_bars(accum, problem.name,
                             filename=f"plot_{base}_comparison.png")

        print(f"saved plots: plot_{base}_routes.png, plot_{base}_comparison.png")


if __name__ == "__main__":
    main()