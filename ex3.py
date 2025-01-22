# # ex3.py
# ids = ['123456789']
# import math
# import random
# from collections import deque, defaultdict
# from heapq import heappush, heappop
# from typing import Tuple, Dict, List, Set
#
# # Define the WizardAgent and OptimalWizardAgent classes
#
# class WizardAgent:
#     """
#     A simple wizard agent that follows a heuristic-based approach to collect horcruxes
#     and avoid death eaters.
#     """
#
#     def __init__(self, initial):
#         """
#         Initialize the agent with the initial state.
#         """
#         self.map = initial['map']
#         self.wizards = initial['wizards']
#         self.horcruxes = initial['horcrux']
#         self.death_eaters = initial['death_eaters']
#         self.turns_to_go = initial['turns_to_go']
#         self.width = len(self.map[0])
#         self.height = len(self.map)
#         self.passable = self._compute_passable_positions()
#         self.targets = list(self.horcruxes.keys())
#         self.visited = set()
#         for wiz in self.wizards.values():
#             self.visited.add(wiz['location'])
#
#     def act(self, state):
#         """
#         Decide on the next action based on the current state.
#         """
#         actions = []
#         for wizard_name, wizard_info in state['wizards'].items():
#             current_pos = wizard_info['location']
#             # If on a horcrux, destroy it
#             horcrux_at_pos = self._find_horcrux_at_position(current_pos, state['horcrux'])
#             if horcrux_at_pos:
#                 actions.append(("destroy", wizard_name, horcrux_at_pos))
#                 continue
#
#             # Find the nearest horcrux
#             nearest_horcrux, path = self._find_nearest_horcrux(current_pos, state['horcrux'])
#             if nearest_horcrux and path:
#                 next_step = path[1]  # Move towards the horcrux
#                 actions.append(("move", wizard_name, next_step))
#                 self.visited.add(next_step)
#                 continue
#
#             # If no horcruxes left, move towards Voldemort if possible
#             if not state['horcrux']:
#                 voldemort_pos = self._find_voldemort_position(state['map'])
#                 if voldemort_pos and current_pos != voldemort_pos:
#                     path = self._find_path(current_pos, voldemort_pos, state['map'])
#                     if path and len(path) > 1:
#                         next_step = path[1]
#                         actions.append(("move", wizard_name, next_step))
#                         self.visited.add(next_step)
#                         continue
#                     elif current_pos == voldemort_pos:
#                         actions.append(("kill", "Harry Potter"))
#                         continue
#
#             # If no specific action, wait
#             actions.append(("wait", wizard_name))
#
#         # If no actions decided, wait
#         if not actions:
#             for wizard_name in state['wizards'].keys():
#                 actions.append(("wait", wizard_name))
#
#         return tuple(actions)
#
#     def _compute_passable_positions(self):
#         """Precompute passable positions on the map."""
#         passable = set()
#         for i in range(len(self.map)):
#             for j in range(len(self.map[0])):
#                 if self.map[i][j] != 'I':
#                     passable.add((i, j))
#         return passable
#
#     def _find_horcrux_at_position(self, position, horcruxes):
#         """Check if there's a horcrux at the given position."""
#         for name, info in horcruxes.items():
#             if info['location'] == position:
#                 return name
#         return None
#
#     def _find_nearest_horcrux(self, current_pos, horcruxes):
#         """Find the nearest horcrux and return its name and path."""
#         min_distance = float('inf')
#         nearest_horcrux = None
#         nearest_path = None
#         for name, info in horcruxes.items():
#             path = self._find_path(current_pos, info['location'], self.map)
#             if path and len(path) < min_distance:
#                 min_distance = len(path)
#                 nearest_horcrux = name
#                 nearest_path = path
#         return nearest_horcrux, nearest_path
#
#     def _find_voldemort_position(self, map_grid):
#         """Find Voldemort's position on the map."""
#         for i, row in enumerate(map_grid):
#             for j, cell in enumerate(row):
#                 if cell == 'V':
#                     return (i, j)
#         return None
#
#     def _find_path(self, start, goal, map_grid):
#         """Use BFS to find the shortest path from start to goal."""
#         queue = deque()
#         queue.append((start, [start]))
#         visited = set()
#         visited.add(start)
#         while queue:
#             current, path = queue.popleft()
#             if current == goal:
#                 return path
#             neighbors = self._get_neighbors(current, map_grid)
#             for neighbor in neighbors:
#                 if neighbor not in visited:
#                     visited.add(neighbor)
#                     queue.append((neighbor, path + [neighbor]))
#         return None
#
#     def _get_neighbors(self, position, map_grid):
#         """Get passable neighbors for a given position."""
#         x, y = position
#         neighbors = []
#         for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#             new_x, new_y = x + dx, y + dy
#             if 0 <= new_x < len(map_grid) and 0 <= new_y < len(map_grid[0]):
#                 if map_grid[new_x][new_y] != 'I':
#                     neighbors.append((new_x, new_y))
#         return neighbors
#
#
# class OptimalWizardAgent:
#     """
#     An optimal wizard agent that uses A* search to plan actions considering
#     the stochastic behavior of death eaters and horcruxes.
#     """
#
#     def __init__(self, initial):
#         """
#         Initialize the agent with the initial state.
#         """
#         self.map = initial['map']
#         self.wizards = initial['wizards']
#         self.horcruxes = initial['horcrux']
#         self.death_eaters = initial['death_eaters']
#         self.turns_to_go = initial['turns_to_go']
#         self.width = len(self.map[0])
#         self.height = len(self.map)
#         self.passable = self._compute_passable_positions()
#         self.target_horcruxes = set(self.horcruxes.keys())
#         self.visited = set()
#         for wiz in self.wizards.values():
#             self.visited.add(wiz['location'])
#         self.current_plan = deque()
#
#     def act(self, state):
#         """
#         Decide on the next action based on the current state using A* planning.
#         """
#         wizard_name = list(state['wizards'].keys())[0]  # Assuming single wizard
#         wizard_info = state['wizards'][wizard_name]
#         current_pos = wizard_info['location']
#
#         # If there's a current plan, follow it
#         if self.current_plan:
#             next_action = self.current_plan.popleft()
#             return (next_action,)
#
#         # If on a horcrux, destroy it
#         horcrux_at_pos = self._find_horcrux_at_position(current_pos, state['horcrux'])
#         if horcrux_at_pos:
#             return (("destroy", wizard_name, horcrux_at_pos),)
#
#         # Plan a path to the nearest horcrux
#         nearest_horcrux, path = self._find_nearest_horcrux(current_pos, state['horcrux'])
#         if nearest_horcrux and path:
#             for step in path[1:]:
#                 self.current_plan.append(("move", wizard_name, step))
#             if path[-1] == nearest_horcrux:
#                 self.current_plan.append(("destroy", wizard_name, nearest_horcrux))
#             next_action = self.current_plan.popleft()
#             self.visited.add(next_action[2]) if next_action[0] == "move" else None
#             return (next_action,)
#
#         # If no horcruxes left, plan to kill Voldemort
#         if not state['horcrux']:
#             voldemort_pos = self._find_voldemort_position(state['map'])
#             if voldemort_pos and current_pos != voldemort_pos:
#                 path = self._find_path(current_pos, voldemort_pos, state['map'])
#                 if path:
#                     for step in path[1:]:
#                         self.current_plan.append(("move", wizard_name, step))
#                     self.current_plan.append(("kill", wizard_name))
#                     next_action = self.current_plan.popleft()
#                     self.visited.add(next_action[2]) if next_action[0] == "move" else None
#                     return (next_action,)
#             elif voldemort_pos == current_pos:
#                 return (("kill", wizard_name),)
#
#         # If no specific action, wait
#         return (("wait", wizard_name),)
#
#     def _compute_passable_positions(self):
#         """Precompute passable positions on the map."""
#         passable = set()
#         for i in range(len(self.map)):
#             for j in range(len(self.map[0])):
#                 if self.map[i][j] != 'I':
#                     passable.add((i, j))
#         return passable
#
#     def _find_horcrux_at_position(self, position, horcruxes):
#         """Check if there's a horcrux at the given position."""
#         for name, info in horcruxes.items():
#             if info['location'] == position:
#                 return name
#         return None
#
#     def _find_nearest_horcrux(self, current_pos, horcruxes):
#         """Find the nearest horcrux and return its name and path."""
#         min_distance = float('inf')
#         nearest_horcrux = None
#         nearest_path = None
#         for name, info in horcruxes.items():
#             path = self._find_path(current_pos, info['location'], self.map)
#             if path and len(path) < min_distance:
#                 min_distance = len(path)
#                 nearest_horcrux = name
#                 nearest_path = path
#         return nearest_horcrux, nearest_path
#
#     def _find_voldemort_position(self, map_grid):
#         """Find Voldemort's position on the map."""
#         for i, row in enumerate(map_grid):
#             for j, cell in enumerate(row):
#                 if cell == 'V':
#                     return (i, j)
#         return None
#
#     def _find_path(self, start, goal, map_grid):
#         """Use BFS to find the shortest path from start to goal."""
#         queue = deque()
#         queue.append((start, [start]))
#         visited = set()
#         visited.add(start)
#         while queue:
#             current, path = queue.popleft()
#             if current == goal:
#                 return path
#             neighbors = self._get_neighbors(current, map_grid)
#             for neighbor in neighbors:
#                 if neighbor not in visited:
#                     visited.add(neighbor)
#                     queue.append((neighbor, path + [neighbor]))
#         return None
#
#     def _get_neighbors(self, position, map_grid):
#         """Get passable neighbors for a given position."""
#         x, y = position
#         neighbors = []
#         for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#             new_x, new_y = x + dx, y + dy
#             if 0 <= new_x < len(map_grid) and 0 <= new_y < len(map_grid[0]):
#                 if map_grid[new_x][new_y] != 'I':
#                     neighbors.append((new_x, new_y))
#         return neighbors


########################################

########################################

# # ex3.py
#
# import itertools
# import math
# import random
# from collections import deque, defaultdict
# from heapq import heappush, heappop
# from typing import Tuple, Dict, List, Set
# from functools import lru_cache
# import logging
#
# # Configure logging
# logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logs
#
# # Define your IDs
# ids = ['207476763']  # Replace with your actual ID(s)
#
# # Define a hashable State class
# class State:
#     def __init__(self, wizard_positions, horcrux_states, death_eater_indices, turns_left):
#         """
#         Initialize the state.
#
#         :param wizard_positions: Tuple of (wizard_name, (x, y))
#         :param horcrux_states: Tuple of (horcrux_name, (x, y), exists)
#         :param death_eater_indices: Tuple of (death_eater_name, path_index)
#         :param turns_left: Integer representing remaining turns
#         """
#         self.wizard_positions = tuple(sorted(wizard_positions))
#         self.horcrux_states = tuple(sorted(horcrux_states))
#         self.death_eater_indices = tuple(sorted(death_eater_indices))
#         self.turns_left = turns_left
#
#     def __hash__(self):
#         return hash((self.wizard_positions, self.horcrux_states, self.death_eater_indices, self.turns_left))
#
#     def __eq__(self, other):
#         return (self.wizard_positions == other.wizard_positions and
#                 self.horcrux_states == other.horcrux_states and
#                 self.death_eater_indices == other.death_eater_indices and
#                 self.turns_left == other.turns_left)
#
#     def __repr__(self):
#         return f"State(wizards={self.wizard_positions}, horcruxes={self.horcrux_states}, death_eaters={self.death_eater_indices}, turns_left={self.turns_left})"
#
# class OptimalWizardAgent:
#     """
#     An optimal wizard agent that uses Value Iteration to compute the optimal policy.
#     """
#
#     def __init__(self, initial):
#         """
#         Initialize the agent with the initial state.
#         Perform Value Iteration to compute the optimal policy.
#         """
#         # Convert map to tuple of tuples for hashability
#         self.map = tuple(tuple(row) for row in initial['map'])
#         self.wizards_initial = initial['wizards']
#         self.horcruxes_initial = initial['horcrux']
#         self.death_eaters_initial = initial['death_eaters']
#         self.turns_to_go = initial['turns_to_go']
#         self.width = len(self.map[0])
#         self.height = len(self.map)
#         self.passable = self._compute_passable_positions()
#
#         # Initialize death eater paths
#         self.death_eaters_paths = {}
#         for de_name, de_info in self.death_eaters_initial.items():
#             self.death_eaters_paths[de_name] = de_info['path']
#
#         # Initialize horcrux states
#         self.horcrux_names = list(self.horcruxes_initial.keys())
#
#         # Initialize wizards
#         self.wizard_names = list(self.wizards_initial.keys())
#
#         # Precompute all possible horcrux locations and movement probabilities
#         self.horcrux_info = {}
#         for h_name, h_info in self.horcruxes_initial.items():
#             self.horcrux_info[h_name] = {
#                 'current_location': tuple(h_info['location']),
#                 'possible_locations': tuple(tuple(loc) for loc in h_info['possible_locations']),
#                 'prob_change_location': h_info['prob_change_location']
#             }
#
#         # Precompute death eater movement probabilities
#         self.death_eater_info = {}
#         for de_name, de_info in self.death_eaters_initial.items():
#             path = de_info['path']
#             self.death_eater_info[de_name] = {
#                 'path': path,
#                 'current_index': de_info['index']  # Ensure consistent key naming
#             }
#
#         # Initialize the initial state
#         initial_wizard_positions = tuple(
#             (name, tuple(info['location'])) for name, info in self.wizards_initial.items()
#         )
#
#         initial_horcrux_states = tuple(
#             (name, tuple(info['location']), True) for name, info in self.horcruxes_initial.items()
#         )
#
#         initial_death_eater_indices = tuple(
#             (name, info['current_index']) for name, info in self.death_eater_info.items()  # Fixed key name
#         )
#
#         self.initial_state = State(
#             wizard_positions=initial_wizard_positions,
#             horcrux_states=initial_horcrux_states,
#             death_eater_indices=initial_death_eater_indices,
#             turns_left=self.turns_to_go
#         )
#
#         # Initialize Value Function and Policy
#         self.value_function = defaultdict(lambda: 0)
#         self.value_function[self.initial_state] = 0
#         self.policy = {}
#
#         # Perform Value Iteration
#         self.value_iteration()
#
#     def act(self, state):
#         """
#         Decide the next action based on the current state using the precomputed policy.
#         """
#         # Extract current state information
#         wizard_positions = tuple(
#             sorted((name, tuple(info['location'])) for name, info in state['wizards'].items())
#         )
#
#         horcrux_states = tuple(
#             sorted(
#                 (name, tuple(info['location']), name in self.horcrux_info and info['location'] != (-1, -1))
#                 for name, info in state['horcrux'].items()
#             )
#         )
#
#         death_eater_indices = tuple(
#             sorted((name, info['index']) for name, info in state['death_eaters'].items())
#         )
#
#         turns_left = state['turns_to_go']
#
#         current_state = State(
#             wizard_positions=wizard_positions,
#             horcrux_states=horcrux_states,
#             death_eater_indices=death_eater_indices,
#             turns_left=turns_left
#         )
#
#         # Retrieve the optimal action from the policy
#         action = self.policy.get(current_state, (("wait", self.wizard_names[0]),))
#
#         logging.info(f"Current State: {current_state}")
#         logging.info(f"Chosen Action: {action}")
#
#         # Return the action as a tuple
#         return action  # Changed from (action,) to action
#
#     def value_iteration(self):
#         """
#         Perform Value Iteration to compute the optimal value function and policy.
#         """
#         gamma = 0.9  # Discount factor
#         epsilon = 1e-3  # Convergence threshold
#         max_iterations = 1000  # To prevent infinite loops
#
#         # Initialize all states with value 0
#         self.value_function = defaultdict(lambda: 0)
#         self.value_function[self.initial_state] = 0
#         self.policy = {}
#
#         # Initialize a queue with the initial state
#         queue = deque()
#         queue.append(self.initial_state)
#
#         # To keep track of states already in the queue
#         in_queue = set()
#         in_queue.add(self.initial_state)
#
#         for iteration in range(max_iterations):
#             if not queue:
#                 break  # No more states to process
#
#             state = queue.popleft()
#             in_queue.remove(state)
#
#             if state.turns_left == 0:
#                 continue  # Terminal state
#
#             # Generate possible actions
#             actions = self._generate_actions(state)
#
#             max_action_value = float('-inf')
#             best_action = None
#             best_resulting_states = []
#
#             for action in actions:
#                 # Compute expected value
#                 expected_value, resulting_states = self._compute_expected_value(state, action, gamma)
#
#                 if expected_value > max_action_value:
#                     max_action_value = expected_value
#                     best_action = action
#                     best_resulting_states = resulting_states  # Capture resulting states of the best action
#
#             # Calculate change in value
#             delta = abs(max_action_value - self.value_function[state])
#
#             # Update value function and policy
#             self.value_function[state] = max_action_value
#             self.policy[state] = best_action
#
#             logging.debug(f"Value Iteration - Iteration: {iteration}, State: {state}, New Value: {max_action_value}, Delta: {delta}")
#
#             # If the change is significant, enqueue successor states from the best action
#             if delta > epsilon:
#                 for new_state in best_resulting_states:
#                     if new_state not in in_queue:
#                         queue.append(new_state)
#                         in_queue.add(new_state)
#
#         logging.info("Value Iteration completed.")
#
#     def _generate_actions(self, state):
#         """
#         Generate all possible actions from the current state.
#         Each action is a tuple of atomic actions, one per wizard.
#         """
#         actions_per_wizard = []
#         for wizard in state.wizard_positions:
#             wizard_name, position = wizard
#             possible_actions = self._get_possible_actions(wizard_name, position, state)
#             actions_per_wizard.append(possible_actions)
#         # Cartesian product of actions for all wizards
#         all_possible_actions = list(itertools.product(*actions_per_wizard))
#         return all_possible_actions
#
#     def _get_possible_actions(self, wizard_name, position, state):
#         """
#         Get possible actions for a single wizard based on the current state.
#         """
#         actions = []
#
#         # If on a horcrux, can destroy it
#         horcrux_at_pos = self._find_horcrux_at_position(position, state.horcrux_states)
#         if horcrux_at_pos:
#             actions.append(("destroy", wizard_name, horcrux_at_pos))
#
#         # Move actions: up, down, left, right
#         for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#             new_x, new_y = position[0] + dx, position[1] + dy
#             if (new_x, new_y) in self.passable:
#                 actions.append(("move", wizard_name, (new_x, new_y)))
#
#         # Wait action
#         actions.append(("wait", wizard_name))
#
#         return actions
#
#     def _find_horcrux_at_position(self, position, horcrux_states):
#         """
#         Check if there's a horcrux at the given position.
#         Returns the horcrux name if present, else None.
#         """
#         for h_name, h_pos, exists in horcrux_states:
#             if exists and h_pos == position:
#                 return h_name
#         return None
#
#     def _compute_expected_value(self, state, action, gamma):
#         """
#         Compute the expected value of taking an action in a state.
#         Returns the expected value and a list of resulting states.
#         """
#         # Initialize reward
#         reward = 0
#         actions = action  # Tuple of atomic actions
#
#         # Apply actions to get new wizard positions and update horcruxes
#         new_wizard_positions = list(state.wizard_positions)
#         new_horcrux_states = list(state.horcrux_states)
#
#         for atomic_action in actions:
#             if atomic_action[0] == "destroy":
#                 _, wizard_name, horcrux_name = atomic_action
#                 reward += 10  # Increased reward for destruction
#                 # Remove horcrux
#                 for idx, (h_name, h_pos, exists) in enumerate(new_horcrux_states):
#                     if h_name == horcrux_name and exists:
#                         new_horcrux_states[idx] = (h_name, h_pos, False)
#                         break
#             elif atomic_action[0] == "move":
#                 _, wizard_name, new_pos = atomic_action
#                 # Update wizard position
#                 for idx, (w_name, w_pos) in enumerate(new_wizard_positions):
#                     if w_name == wizard_name:
#                         new_wizard_positions[idx] = (w_name, new_pos)
#                         break
#             elif atomic_action[0] == "wait":
#                 pass  # No change
#
#         # Simulate death eaters' movements
#         death_eater_movements = {}
#         for de_name, de_info in self.death_eater_info.items():
#             path = de_info['path']
#             current_index = None
#             for de in state.death_eater_indices:
#                 if de[0] == de_name:
#                     current_index = de[1]
#                     break
#             possible_indices = []
#             probabilities = []
#             if len(path) == 1:
#                 possible_indices = [0]
#                 probabilities = [1.0]
#             else:
#                 # Possible moves: stay, move forward, move backward
#                 moves = []
#                 if current_index > 0:
#                     moves.append(current_index - 1)
#                 moves.append(current_index)
#                 if current_index < len(path) - 1:
#                     moves.append(current_index + 1)
#                 possible_indices = moves
#                 probabilities = [1/len(moves)] * len(moves)  # Uniform probability
#
#             death_eater_movements[de_name] = list(zip(possible_indices, probabilities))
#
#         # Simulate horcruxes' movements
#         horcrux_movements = {}
#         for h_name, h_pos, exists in new_horcrux_states:
#             if exists:
#                 prob_move = self.horcrux_info[h_name]['prob_change_location']
#                 possible_locs = self.horcrux_info[h_name]['possible_locations']
#                 num_new_locs = len(possible_locs)
#                 prob_per_loc = prob_move / num_new_locs
#                 # With probability prob_move, horcrux moves to one of possible_locs
#                 # With probability (1 - prob_move), it stays
#                 movements = [(loc, prob_per_loc) for loc in possible_locs]
#                 movements.append((h_pos, 1 - prob_move))
#                 horcrux_movements[h_name] = movements
#             else:
#                 horcrux_movements[h_name] = [(h_pos, 1.0)]  # No movement if destroyed
#
#         # Generate all possible combinations of death eater movements
#         death_eater_names = list(death_eater_movements.keys())
#         death_eater_move_lists = [death_eater_movements[name] for name in death_eater_names]
#         all_de_moves = list(itertools.product(*death_eater_move_lists))
#
#         # Generate all possible combinations of horcrux movements
#         horcrux_names = list(horcrux_movements.keys())
#         horcrux_move_lists = [horcrux_movements[name] for name in horcrux_names]
#         all_h_moves = list(itertools.product(*horcrux_move_lists))
#
#         expected_value = 0
#         resulting_states = []
#
#         # Enumerate all combinations of death eaters' moves and horcruxes' moves
#         for de_move in all_de_moves:
#             # Calculate death eater movement probability
#             de_prob = 1.0
#             new_death_eater_indices = []
#             for de_name, (de_new_index, de_move_prob) in zip(death_eater_names, de_move):
#                 de_prob *= de_move_prob
#                 new_death_eater_indices.append((de_name, de_new_index))
#
#             for h_move in all_h_moves:
#                 # Calculate horcrux movement probability
#                 h_prob = 1.0
#                 new_horcrux_states_updated = list(new_horcrux_states)
#                 for idx, (h_name, h_move_entry) in enumerate(zip(horcrux_names, h_move)):
#                     h_new_loc, h_move_prob = h_move_entry
#                     h_prob *= h_move_prob
#                     # Update horcrux location
#                     new_horcrux_states_updated[idx] = (h_name, h_new_loc, True)
#
#                 # Total probability for this combination
#                 total_prob = de_prob * h_prob
#
#                 # Check for encounters
#                 penalty = 0
#                 wizard_locations = {w[0]: w[1] for w in new_wizard_positions}
#                 death_eater_positions = set()
#                 for de_idx, de_name in enumerate(death_eater_names):
#                     path = self.death_eaters_paths[de_name]
#                     de_current_index = new_death_eater_indices[de_idx][1]
#                     de_pos = path[de_current_index]
#                     death_eater_positions.add(de_pos)
#                 for wiz_name, wiz_pos in wizard_locations.items():
#                     if wiz_pos in death_eater_positions:
#                         penalty -= 2  # Adjusted penalty per encounter
#
#                 # Create new state
#                 new_state = State(
#                     wizard_positions=tuple(new_wizard_positions),
#                     horcrux_states=tuple(new_horcrux_states_updated),
#                     death_eater_indices=tuple(new_death_eater_indices),
#                     turns_left=state.turns_left - 1
#                 )
#
#                 # Accumulate expected value
#                 future_value = self.value_function.get(new_state, 0)
#                 expected_value += total_prob * (reward + penalty + gamma * future_value)
#                 resulting_states.append(new_state)
#
#                 logging.debug(f"Action: {action}, Total Prob: {total_prob}, New State: {new_state}, Reward: {reward}, Penalty: {penalty}, Future Value: {future_value}")
#
#         return expected_value, resulting_states
#
#     def _compute_passable_positions(self):
#         """Precompute passable positions on the map."""
#         passable = set()
#         for i in range(len(self.map)):
#             for j in range(len(self.map[0])):
#                 if self.map[i][j] != 'I':
#                     passable.add((i, j))
#         return passable
#
#     @lru_cache(maxsize=None)
#     def _find_path(self, start, goal, map_grid):
#         """Use BFS to find the shortest path from start to goal."""
#         queue = deque()
#         queue.append((start, [start]))
#         visited = set()
#         visited.add(start)
#         while queue:
#             current, path = queue.popleft()
#             if current == goal:
#                 return path
#             neighbors = self._get_neighbors(current, map_grid)
#             for neighbor in neighbors:
#                 if neighbor not in visited:
#                     visited.add(neighbor)
#                     queue.append((neighbor, path + [neighbor]))
#         return None
#
#     def _get_neighbors(self, position, map_grid):
#         """Get passable neighbors for a given position."""
#         x, y = position
#         neighbors = []
#         for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#             new_x, new_y = x + dx, y + dy
#             if (0 <= new_x < len(map_grid)) and (0 <= new_y < len(map_grid[0])):
#                 if map_grid[new_x][new_y] != 'I':
#                     neighbors.append((new_x, new_y))
#         return neighbors
#
# class WizardAgent:
#     """
#     A simple wizard agent that follows a heuristic-based approach to collect horcruxes
#     and avoid death eaters.
#     """
#
#     def __init__(self, initial):
#         """
#         Initialize the agent with the initial state.
#         """
#         # Convert map to tuple of tuples for hashability
#         self.map = tuple(tuple(row) for row in initial['map'])
#         self.wizards = initial['wizards']
#         self.horcruxes = initial['horcrux']
#         self.death_eaters = initial['death_eaters']
#         self.turns_to_go = initial['turns_to_go']
#         self.width = len(self.map[0])
#         self.height = len(self.map)
#         self.passable = self._compute_passable_positions()
#         self.targets = list(self.horcruxes.keys())
#         self.visited = set()
#         for wiz in self.wizards.values():
#             self.visited.add(wiz['location'])
#
#     def act(self, state):
#         """
#         Decide on the next action based on the current state.
#         """
#         actions = []
#         for wizard_name, wizard_info in state['wizards'].items():
#             current_pos = wizard_info['location']
#             # If on a horcrux, destroy it
#             horcrux_at_pos = self._find_horcrux_at_position(current_pos, state['horcrux'])
#             if horcrux_at_pos:
#                 actions.append(("destroy", wizard_name, horcrux_at_pos))
#                 continue
#
#             # Find the nearest horcrux
#             nearest_horcrux, path = self._find_nearest_horcrux(current_pos, state['horcrux'])
#             if nearest_horcrux and path:
#                 if len(path) > 1:
#                     next_step = path[1]  # Move towards the horcrux
#                     actions.append(("move", wizard_name, next_step))
#                     self.visited.add(next_step)
#                 else:
#                     # Already at the horcrux location
#                     actions.append(("destroy", wizard_name, nearest_horcrux))
#                 continue
#
#             # If no horcruxes left, move towards Voldemort if possible
#             if not state['horcrux']:
#                 voldemort_pos = self._find_voldemort_position(self.map)
#                 if voldemort_pos and current_pos != voldemort_pos:
#                     path = self._find_path(current_pos, voldemort_pos, self.map)
#                     if path and len(path) > 1:
#                         next_step = path[1]
#                         actions.append(("move", wizard_name, next_step))
#                         self.visited.add(next_step)
#                         continue
#                     elif current_pos == voldemort_pos:
#                         actions.append(("kill", "Harry Potter"))
#                         continue
#
#             # If no specific action, wait
#             actions.append(("wait", wizard_name))
#
#         # If no actions decided, wait
#         if not actions:
#             for wizard_name in state['wizards'].keys():
#                 actions.append(("wait", wizard_name))
#
#         return tuple(actions)
#
#     def _compute_passable_positions(self):
#         """Precompute passable positions on the map."""
#         passable = set()
#         for i in range(len(self.map)):
#             for j in range(len(self.map[0])):
#                 if self.map[i][j] != 'I':
#                     passable.add((i, j))
#         return passable
#
#     def _find_horcrux_at_position(self, position, horcruxes):
#         """Check if there's a horcrux at the given position."""
#         for name, info in horcruxes.items():
#             if info['location'] == position:
#                 return name
#         return None
#
#     def _find_nearest_horcrux(self, current_pos, horcruxes):
#         """Find the nearest horcrux and return its name and path."""
#         min_distance = float('inf')
#         nearest_horcrux = None
#         nearest_path = None
#         for name, info in horcruxes.items():
#             path = self._find_path(current_pos, tuple(info['location']), self.map)
#             if path and len(path) < min_distance:
#                 min_distance = len(path)
#                 nearest_horcrux = name
#                 nearest_path = path
#         return nearest_horcrux, nearest_path
#
#     @lru_cache(maxsize=None)
#     def _find_path(self, start, goal, map_grid):
#         """Use BFS to find the shortest path from start to goal."""
#         queue = deque()
#         queue.append((start, [start]))
#         visited = set()
#         visited.add(start)
#         while queue:
#             current, path = queue.popleft()
#             if current == goal:
#                 return path
#             neighbors = self._get_neighbors(current, map_grid)
#             for neighbor in neighbors:
#                 if neighbor not in visited:
#                     visited.add(neighbor)
#                     queue.append((neighbor, path + [neighbor]))
#         return None
#
#     def _find_voldemort_position(self, map_grid):
#         """Find Voldemort's position on the map."""
#         for i, row in enumerate(map_grid):
#             for j, cell in enumerate(row):
#                 if cell == 'V':
#                     return (i, j)
#         return None
#
#     def _get_neighbors(self, position, map_grid):
#         """Get passable neighbors for a given position."""
#         x, y = position
#         neighbors = []
#         for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#             new_x, new_y = x + dx, y + dy
#             if (0 <= new_x < len(map_grid)) and (0 <= new_y < len(map_grid[0])):
#                 if map_grid[new_x][new_y] != 'I':
#                     neighbors.append((new_x, new_y))
#         return neighbors

####################

####################


# ex3.py

import itertools
import math
import random
from collections import deque, defaultdict
from heapq import heappush, heappop
from typing import Tuple, Dict, List, Set
from functools import lru_cache


# Define your IDs
ids = ['207476763']  # Replace with your actual ID(s)


# Define a hashable State class
class State:
    def __init__(self, wizard_positions, horcrux_states, death_eater_indices, turns_left):
        """
        Initialize the state.

        :param wizard_positions: Tuple of (wizard_name, (x, y))
        :param horcrux_states: Tuple of (horcrux_name, (x, y), exists)
        :param death_eater_indices: Tuple of (death_eater_name, path_index)
        :param turns_left: Integer representing remaining turns
        """
        self.wizard_positions = tuple(sorted(wizard_positions))
        self.horcrux_states = tuple(sorted(horcrux_states))
        self.death_eater_indices = tuple(sorted(death_eater_indices))
        self.turns_left = turns_left

    def __hash__(self):
        return hash((self.wizard_positions, self.horcrux_states, self.death_eater_indices, self.turns_left))

    def __eq__(self, other):
        return (self.wizard_positions == other.wizard_positions and
                self.horcrux_states == other.horcrux_states and
                self.death_eater_indices == other.death_eater_indices and
                self.turns_left == other.turns_left)

    def __repr__(self):
        return f"State(wizards={self.wizard_positions}, horcruxes={self.horcrux_states}, death_eaters={self.death_eater_indices}, turns_left={self.turns_left})"


class OptimalWizardAgent:
    """
    An optimal wizard agent that uses Value Iteration to compute the optimal policy.
    """

    def __init__(self, initial):
        """
        Initialize the agent with the initial state.
        Perform Value Iteration to compute the optimal policy.
        """
        # Convert map to tuple of tuples for hashability
        self.map = tuple(tuple(row) for row in initial['map'])
        self.wizards_initial = initial['wizards']
        self.horcruxes_initial = initial['horcrux']
        self.death_eaters_initial = initial['death_eaters']
        self.turns_to_go = initial['turns_to_go']
        self.width = len(self.map[0])
        self.height = len(self.map)
        self.passable = self._compute_passable_positions()

        # Initialize death eater paths
        self.death_eaters_paths = {}
        for de_name, de_info in self.death_eaters_initial.items():
            self.death_eaters_paths[de_name] = de_info['path']

        # Initialize horcrux states
        self.horcrux_names = list(self.horcruxes_initial.keys())

        # Initialize wizards
        self.wizard_names = list(self.wizards_initial.keys())

        # Precompute all possible horcrux locations and movement probabilities
        self.horcrux_info = {}
        for h_name, h_info in self.horcruxes_initial.items():
            self.horcrux_info[h_name] = {
                'current_location': tuple(h_info['location']),
                'possible_locations': tuple(tuple(loc) for loc in h_info['possible_locations']),
                'prob_change_location': h_info['prob_change_location']
            }

        # Precompute death eater movement probabilities
        self.death_eater_info = {}
        for de_name, de_info in self.death_eaters_initial.items():
            path = de_info['path']
            self.death_eater_info[de_name] = {
                'path': path,
                'current_index': de_info['index']  # Ensure consistent key naming
            }

        # Initialize the initial state
        initial_wizard_positions = tuple(
            (name, tuple(info['location'])) for name, info in self.wizards_initial.items()
        )

        initial_horcrux_states = tuple(
            (name, tuple(info['location']), True) for name, info in self.horcruxes_initial.items()
        )

        initial_death_eater_indices = tuple(
            (name, info['current_index']) for name, info in self.death_eater_info.items()  # Fixed key name
        )

        self.initial_state = State(
            wizard_positions=initial_wizard_positions,
            horcrux_states=initial_horcrux_states,
            death_eater_indices=initial_death_eater_indices,
            turns_left=self.turns_to_go
        )

        # Initialize Value Function and Policy
        self.value_function = defaultdict(lambda: 0)
        self.value_function[self.initial_state] = 0
        self.policy = {}

        # Perform Value Iteration
        self.value_iteration()

    def act(self, state):
        """
        Decide the next action based on the current state using the precomputed policy.
        """
        # Extract current state information
        wizard_positions = tuple(
            sorted((name, tuple(info['location'])) for name, info in state['wizards'].items())
        )

        # Correctly determine the 'exists' status from horcrux_states
        horcrux_states = tuple(
            sorted(
                (name, tuple(info['location']), info['location'] != (-1, -1))
                for name, info in state['horcrux'].items()
            )
        )

        death_eater_indices = tuple(
            sorted((name, info['index']) for name, info in state['death_eaters'].items())
        )

        turns_left = state['turns_to_go']

        current_state = State(
            wizard_positions=wizard_positions,
            horcrux_states=horcrux_states,
            death_eater_indices=death_eater_indices,
            turns_left=turns_left
        )

        # Retrieve the optimal action from the policy
        action = self.policy.get(current_state, (("wait", self.wizard_names[0]),))

        # Return the action as a tuple
        return action  # Changed from (action,) to action

    def value_iteration(self):
        """
        Perform Value Iteration to compute the optimal value function and policy.
        """
        gamma = 0.9  # Discount factor
        epsilon = 1e-3  # Convergence threshold
        max_iterations = 1000  # To prevent infinite loops

        # Initialize all states with value 0
        self.value_function = defaultdict(lambda: 0)
        self.value_function[self.initial_state] = 0
        self.policy = {}

        # Initialize a queue with the initial state
        queue = deque()
        queue.append(self.initial_state)

        # To keep track of states already in the queue
        in_queue = set()
        in_queue.add(self.initial_state)

        for iteration in range(max_iterations):
            if not queue:
                break  # No more states to process

            state = queue.popleft()
            in_queue.remove(state)

            if state.turns_left == 0:
                # Assign utility based on remaining horcruxes
                remaining_horcruxes = sum(1 for h in state.horcrux_states if h[2])
                if remaining_horcruxes == 0:
                    self.value_function[state] = 100  # High reward for success
                else:
                    self.value_function[state] = -10 * remaining_horcruxes  # Penalty for each remaining horcrux
                continue  # Terminal state, no further actions

            # Generate possible actions
            actions = self._generate_actions(state)

            max_action_value = float('-inf')
            best_action = None
            best_resulting_states = []

            for action in actions:
                # Compute expected value
                expected_value, resulting_states = self._compute_expected_value(state, action, gamma)

                if expected_value > max_action_value:
                    max_action_value = expected_value
                    best_action = action
                    best_resulting_states = resulting_states  # Capture resulting states of the best action

            # Calculate change in value
            delta = abs(max_action_value - self.value_function[state])

            # Update value function and policy
            self.value_function[state] = max_action_value
            self.policy[state] = best_action


            # If the change is significant, enqueue successor states from the best action
            if delta > epsilon:
                for new_state in best_resulting_states:
                    if new_state not in in_queue:
                        queue.append(new_state)
                        in_queue.add(new_state)


    def _generate_actions(self, state):
        """
        Generate all possible actions from the current state.
        Each action is a tuple of atomic actions, one per wizard.
        """
        actions_per_wizard = []
        for wizard in state.wizard_positions:
            wizard_name, position = wizard
            possible_actions = self._get_possible_actions(wizard_name, position, state)
            actions_per_wizard.append(possible_actions)
        # Cartesian product of actions for all wizards
        all_possible_actions = list(itertools.product(*actions_per_wizard))
        return all_possible_actions

    def _get_possible_actions(self, wizard_name, position, state):
        """
        Get possible actions for a single wizard based on the current state.
        """
        actions = []

        # If on a horcrux, can destroy it
        horcrux_at_pos = self._find_horcrux_at_position(position, state.horcrux_states)
        if horcrux_at_pos:
            actions.append(("destroy", wizard_name, horcrux_at_pos))

        # Move actions: up, down, left, right
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = position[0] + dx, position[1] + dy
            if (new_x, new_y) in self.passable:
                actions.append(("move", wizard_name, (new_x, new_y)))

        # Wait action
        actions.append(("wait", wizard_name))

        return actions

    def _find_horcrux_at_position(self, position, horcrux_states):
        """
        Check if there's a horcrux at the given position.
        Returns the horcrux name if present, else None.
        """
        for h_name, h_pos, exists in horcrux_states:
            if exists and h_pos == position:
                return h_name
        return None

    def _compute_expected_value(self, state, action, gamma):
        """
        Compute the expected value of taking an action in a state.
        Returns the expected value and a list of resulting states.
        """
        # Initialize reward
        reward = 0
        actions = action  # Tuple of atomic actions

        # Apply actions to get new wizard positions and update horcruxes
        new_wizard_positions = list(state.wizard_positions)
        new_horcrux_states = list(state.horcrux_states)

        for atomic_action in actions:
            if atomic_action[0] == "destroy":
                _, wizard_name, horcrux_name = atomic_action
                reward += 10  # Increased reward for destruction
                # Remove horcrux
                for idx, (h_name, h_pos, exists) in enumerate(new_horcrux_states):
                    if h_name == horcrux_name and exists:
                        new_horcrux_states[idx] = (h_name, h_pos, False)
                        break
            elif atomic_action[0] == "move":
                _, wizard_name, new_pos = atomic_action
                # Update wizard position
                for idx, (w_name, w_pos) in enumerate(new_wizard_positions):
                    if w_name == wizard_name:
                        new_wizard_positions[idx] = (w_name, new_pos)
                        break
            elif atomic_action[0] == "wait":
                pass  # No change

        # Simulate death eaters' movements
        death_eater_movements = {}
        for de_name, de_info in self.death_eater_info.items():
            path = de_info['path']
            current_index = None
            for de in state.death_eater_indices:
                if de[0] == de_name:
                    current_index = de[1]
                    break
            possible_indices = []
            probabilities = []
            if len(path) == 1:
                possible_indices = [0]
                probabilities = [1.0]
            else:
                # Possible moves: stay, move forward, move backward
                moves = []
                if current_index > 0:
                    moves.append(current_index - 1)
                moves.append(current_index)
                if current_index < len(path) - 1:
                    moves.append(current_index + 1)
                possible_indices = moves
                probabilities = [1 / len(moves)] * len(moves)  # Uniform probability

            death_eater_movements[de_name] = list(zip(possible_indices, probabilities))

        # Simulate horcruxes' movements
        horcrux_movements = {}
        for h_name, h_pos, exists in new_horcrux_states:
            if exists:
                prob_move = self.horcrux_info[h_name]['prob_change_location']
                possible_locs = self.horcrux_info[h_name]['possible_locations']
                num_new_locs = len(possible_locs)
                prob_per_loc = prob_move / num_new_locs
                # With probability prob_move, horcrux moves to one of possible_locs
                # With probability (1 - prob_move), it stays
                movements = [(loc, prob_per_loc) for loc in possible_locs]
                movements.append((h_pos, 1 - prob_move))
                horcrux_movements[h_name] = movements
            else:
                horcrux_movements[h_name] = [(h_pos, 1.0)]  # No movement if destroyed

        # Generate all possible combinations of death eater movements
        death_eater_names = list(death_eater_movements.keys())
        death_eater_move_lists = [death_eater_movements[name] for name in death_eater_names]
        all_de_moves = list(itertools.product(*death_eater_move_lists))

        # Generate all possible combinations of horcrux movements
        horcrux_names = list(horcrux_movements.keys())
        horcrux_move_lists = [horcrux_movements[name] for name in horcrux_names]
        all_h_moves = list(itertools.product(*horcrux_move_lists))

        expected_value = 0
        resulting_states = []

        # Enumerate all combinations of death eaters' moves and horcruxes' moves
        for de_move in all_de_moves:
            # Calculate death eater movement probability
            de_prob = 1.0
            new_death_eater_indices = []
            for de_name, (de_new_index, de_move_prob) in zip(death_eater_names, de_move):
                de_prob *= de_move_prob
                new_death_eater_indices.append((de_name, de_new_index))

            for h_move in all_h_moves:
                # Calculate horcrux movement probability
                h_prob = 1.0
                new_horcrux_states_updated = list(new_horcrux_states)
                for idx, (h_name, h_move_entry) in enumerate(zip(horcrux_names, h_move)):
                    h_new_loc, h_move_prob = h_move_entry
                    h_prob *= h_move_prob
                    # Update horcrux location
                    new_horcrux_states_updated[idx] = (h_name, h_new_loc, True)

                # Total probability for this combination
                total_prob = de_prob * h_prob

                # Check for encounters
                penalty = 0
                wizard_locations = {w[0]: w[1] for w in new_wizard_positions}
                death_eater_positions = set()
                for de_idx, de_name in enumerate(death_eater_names):
                    path = self.death_eaters_paths[de_name]
                    de_current_index = new_death_eater_indices[de_idx][1]
                    de_pos = path[de_current_index]
                    death_eater_positions.add(de_pos)
                for wiz_name, wiz_pos in wizard_locations.items():
                    if wiz_pos in death_eater_positions:
                        penalty -= 2  # Adjusted penalty per encounter

                # Create new state
                new_state = State(
                    wizard_positions=tuple(new_wizard_positions),
                    horcrux_states=tuple(new_horcrux_states_updated),
                    death_eater_indices=tuple(new_death_eater_indices),
                    turns_left=state.turns_left - 1
                )

                # Accumulate expected value
                future_value = self.value_function.get(new_state, 0)
                expected_value += total_prob * (reward + penalty + gamma * future_value)
                resulting_states.append(new_state)

        return expected_value, resulting_states

    def _compute_passable_positions(self):
        """Precompute passable positions on the map."""
        passable = set()
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] != 'I':
                    passable.add((i, j))
        return passable

    @lru_cache(maxsize=None)
    def _find_path(self, start, goal, map_grid):
        """Use BFS to find the shortest path from start to goal."""
        queue = deque()
        queue.append((start, [start]))
        visited = set()
        visited.add(start)
        while queue:
            current, path = queue.popleft()
            if current == goal:
                return path
            neighbors = self._get_neighbors(current, map_grid)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def _get_neighbors(self, position, map_grid):
        """Get passable neighbors for a given position."""
        x, y = position
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < len(map_grid)) and (0 <= new_y < len(map_grid[0])):
                if map_grid[new_x][new_y] != 'I':
                    neighbors.append((new_x, new_y))
        return neighbors


class WizardAgent:
    """
    A simple wizard agent that follows a heuristic-based approach to collect horcruxes
    and avoid death eaters.
    """

    def __init__(self, initial):
        """
        Initialize the agent with the initial state.
        """
        # Convert map to tuple of tuples for hashability
        self.map = tuple(tuple(row) for row in initial['map'])
        self.wizards = initial['wizards']
        self.horcruxes = initial['horcrux']
        self.death_eaters = initial['death_eaters']
        self.turns_to_go = initial['turns_to_go']
        self.width = len(self.map[0])
        self.height = len(self.map)
        self.passable = self._compute_passable_positions()
        self.targets = list(self.horcruxes.keys())
        self.visited = set()
        for wiz in self.wizards.values():
            self.visited.add(wiz['location'])

    def act(self, state):
        """
        Decide on the next action based on the current state.
        """
        actions = []
        for wizard_name, wizard_info in state['wizards'].items():
            current_pos = wizard_info['location']
            # If on a horcrux, destroy it
            horcrux_at_pos = self._find_horcrux_at_position(current_pos, state['horcrux'])
            if horcrux_at_pos:
                actions.append(("destroy", wizard_name, horcrux_at_pos))
                continue

            # Find the nearest horcrux
            nearest_horcrux, path = self._find_nearest_horcrux(current_pos, state['horcrux'])
            if nearest_horcrux and path:
                if len(path) > 1:
                    next_step = path[1]  # Move towards the horcrux
                    actions.append(("move", wizard_name, next_step))
                    self.visited.add(next_step)
                else:
                    # Already at the horcrux location
                    actions.append(("destroy", wizard_name, nearest_horcrux))
                continue

            # If no horcruxes left, move towards Voldemort if possible
            if not state['horcrux']:
                voldemort_pos = self._find_voldemort_position(self.map)
                if voldemort_pos and current_pos != voldemort_pos:
                    path = self._find_path(current_pos, voldemort_pos, self.map)
                    if path and len(path) > 1:
                        next_step = path[1]
                        actions.append(("move", wizard_name, next_step))
                        self.visited.add(next_step)
                        continue
                    elif current_pos == voldemort_pos:
                        actions.append(("kill", "Harry Potter"))
                        continue

            # If no specific action, wait
            actions.append(("wait", wizard_name))

        # If no actions decided, wait
        if not actions:
            for wizard_name in state['wizards'].keys():
                actions.append(("wait", wizard_name))

        return tuple(actions)

    def _compute_passable_positions(self):
        """Precompute passable positions on the map."""
        passable = set()
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] != 'I':
                    passable.add((i, j))
        return passable

    def _find_horcrux_at_position(self, position, horcruxes):
        """Check if there's a horcrux at the given position."""
        for name, info in horcruxes.items():
            if info['location'] == position:
                return name
        return None

    def _find_nearest_horcrux(self, current_pos, horcruxes):
        """Find the nearest horcrux and return its name and path."""
        min_distance = float('inf')
        nearest_horcrux = None
        nearest_path = None
        for name, info in horcruxes.items():
            if info['location'] == (-1, -1):
                continue  # Horcrux already destroyed
            path = self._find_path(current_pos, tuple(info['location']), self.map)
            if path and len(path) < min_distance:
                min_distance = len(path)
                nearest_horcrux = name
                nearest_path = path
        return nearest_horcrux, nearest_path

    @lru_cache(maxsize=None)
    def _find_path(self, start, goal, map_grid):
        """Use BFS to find the shortest path from start to goal."""
        queue = deque()
        queue.append((start, [start]))
        visited = set()
        visited.add(start)
        while queue:
            current, path = queue.popleft()
            if current == goal:
                return path
            neighbors = self._get_neighbors(current, map_grid)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def _find_voldemort_position(self, map_grid):
        """Find Voldemort's position on the map."""
        for i, row in enumerate(map_grid):
            for j, cell in enumerate(row):
                if cell == 'V':
                    return (i, j)
        return None

    def _get_neighbors(self, position, map_grid):
        """Get passable neighbors for a given position."""
        x, y = position
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < len(map_grid)) and (0 <= new_y < len(map_grid[0])):
                if map_grid[new_x][new_y] != 'I':
                    neighbors.append((new_x, new_y))
        return neighbors

#############################
# Value Iteration
#############################

# # ex3.py
#
# import itertools
# import math
# import random
# from collections import deque, defaultdict
# from heapq import heappush, heappop
# from typing import Tuple, Dict, List, Set
# from functools import lru_cache
#
# # Define your IDs
# ids = ['207476763']  # Replace with your actual ID(s)
#
# # Define a hashable State class
# class State:
#     def __init__(self, wizard_positions, horcrux_states, death_eater_indices, turns_left):
#         """
#         Initialize the state.
#
#         :param wizard_positions: Tuple of (wizard_name, (x, y))
#         :param horcrux_states: Tuple of (horcrux_name, (x, y), exists)
#         :param death_eater_indices: Tuple of (death_eater_name, path_index)
#         :param turns_left: Integer representing remaining turns
#         """
#         self.wizard_positions = tuple(sorted(wizard_positions))
#         self.horcrux_states = tuple(sorted(horcrux_states))
#         self.death_eater_indices = tuple(sorted(death_eater_indices))
#         self.turns_left = turns_left
#
#     def __hash__(self):
#         return hash((self.wizard_positions, self.horcrux_states, self.death_eater_indices, self.turns_left))
#
#     def __eq__(self, other):
#         return (self.wizard_positions == other.wizard_positions and
#                 self.horcrux_states == other.horcrux_states and
#                 self.death_eater_indices == other.death_eater_indices and
#                 self.turns_left == other.turns_left)
#
#     def __repr__(self):
#         return f"State(wizards={self.wizard_positions}, horcruxes={self.horcrux_states}, death_eaters={self.death_eater_indices}, turns_left={self.turns_left})"
#
#
# class OptimalWizardAgent:
#     """
#     An optimal wizard agent that uses Synchronous Value Iteration to compute the optimal policy.
#     """
#
#     def __init__(self, initial):
#         """
#         Initialize the agent with the initial state.
#         Perform Synchronous Value Iteration to compute the optimal policy.
#         """
#         # Convert map to tuple of tuples for hashability
#         self.map = tuple(tuple(row) for row in initial['map'])
#         self.wizards_initial = initial['wizards']
#         self.horcruxes_initial = initial['horcrux']
#         self.death_eaters_initial = initial['death_eaters']
#         self.turns_to_go = initial['turns_to_go']
#         self.width = len(self.map[0])
#         self.height = len(self.map)
#         self.passable = self._compute_passable_positions()
#
#         # Initialize death eater paths
#         self.death_eaters_paths = {}
#         for de_name, de_info in self.death_eaters_initial.items():
#             self.death_eaters_paths[de_name] = de_info['path']
#
#         # Initialize horcrux states
#         self.horcrux_names = list(self.horcruxes_initial.keys())
#
#         # Initialize wizards
#         self.wizard_names = list(self.wizards_initial.keys())
#
#         # Precompute all possible horcrux locations and movement probabilities
#         self.horcrux_info = {}
#         for h_name, h_info in self.horcruxes_initial.items():
#             self.horcrux_info[h_name] = {
#                 'current_location': tuple(h_info['location']),
#                 'possible_locations': tuple(tuple(loc) for loc in h_info['possible_locations']),
#                 'prob_change_location': h_info['prob_change_location']
#             }
#
#         # Precompute death eater movement probabilities
#         self.death_eater_info = {}
#         for de_name, de_info in self.death_eaters_initial.items():
#             path = de_info['path']
#             self.death_eater_info[de_name] = {
#                 'path': path,
#                 'current_index': de_info['index']  # Ensure consistent key naming
#             }
#
#         # Initialize all possible states
#         self.all_states = self._enumerate_all_states()
#
#         # Initialize the initial state
#         initial_wizard_positions = tuple(
#             (name, tuple(info['location'])) for name, info in self.wizards_initial.items()
#         )
#
#         initial_horcrux_states = tuple(
#             (name, tuple(info['location']), True) for name, info in self.horcruxes_initial.items()
#         )
#
#         initial_death_eater_indices = tuple(
#             (name, info['current_index']) for name, info in self.death_eater_info.items()
#         )
#
#         self.initial_state = State(
#             wizard_positions=initial_wizard_positions,
#             horcrux_states=initial_horcrux_states,
#             death_eater_indices=initial_death_eater_indices,
#             turns_left=self.turns_to_go
#         )
#
#         # Initialize Value Function and Policy
#         self.value_function = defaultdict(lambda: 0)
#         self.policy = {}
#
#         # Perform Synchronous Value Iteration
#         self.value_iteration()
#
#     def _enumerate_all_states(self):
#         """
#         Enumerate all possible states based on the initial configurations.
#         """
#         all_states = set()
#
#         # Possible wizard positions
#         wizard_positions = []
#         for wizard_name, wizard_info in self.wizards_initial.items():
#             wizard_positions.append([(wizard_name, pos) for pos in self.passable])
#
#         # Possible horcrux states
#         horcrux_states = []
#         for h_name, h_info in self.horcruxes_initial.items():
#             horcrux_states.append([
#                 (h_name, loc, True) for loc in [h_info['location']] + list(h_info['possible_locations'])
#             ] + [
#                 (h_name, (-1, -1), False)
#             ])  # Include destroyed state
#
#         # Possible death eater indices
#         death_eater_indices = []
#         for de_name, de_info in self.death_eater_info.items():
#             path = de_info['path']
#             death_eater_indices.append([(de_name, idx) for idx in range(len(path))])
#
#         # Possible turns_left
#         turns_left = list(range(self.turns_to_go + 1))
#
#         # Cartesian product of all components
#         for w_pos in itertools.product(*wizard_positions):
#             for h_state in itertools.product(*horcrux_states):
#                 for de_idx in itertools.product(*death_eater_indices):
#                     for t_left in turns_left:
#                         state = State(
#                             wizard_positions=w_pos,
#                             horcrux_states=h_state,
#                             death_eater_indices=de_idx,
#                             turns_left=t_left
#                         )
#                         all_states.add(state)
#
#         return all_states
#
#     def act(self, state):
#         """
#         Decide the next action based on the current state using the precomputed policy.
#         """
#         # Construct the State object
#         wizard_positions = tuple(
#             (name, tuple(info['location'])) for name, info in state['wizards'].items()
#         )
#
#         horcrux_states = tuple(
#             (name, tuple(info['location']), info['location'] != (-1, -1))
#             for name, info in state['horcrux'].items()
#         )
#
#         death_eater_indices = tuple(
#             (name, info['index']) for name, info in state['death_eaters'].items()
#         )
#
#         turns_left = state['turns_to_go']
#
#         current_state = State(
#             wizard_positions=wizard_positions,
#             horcrux_states=horcrux_states,
#             death_eater_indices=death_eater_indices,
#             turns_left=turns_left
#         )
#
#         # Retrieve the optimal action from the policy
#         action = self.policy.get(current_state, (("wait", self.wizard_names[0]),))
#
#         # Return the action as a tuple
#         return action  # Changed from (action,) to action
#
#     def value_iteration(self):
#         """
#         Perform Synchronous Value Iteration to compute the optimal value function and policy.
#         """
#         gamma = 0.9  # Discount factor
#         epsilon = 2  # Convergence threshold
#
#         # Initialize all state values to 0
#         for state in self.all_states:
#             self.value_function[state] = 0
#
#         # Iterate until convergence
#         iteration = 0
#         while True:
#             delta = 0
#             new_value_function = self.value_function.copy()
#             new_policy = self.policy.copy()
#
#             for state in self.all_states:
#                 if state.turns_left == 0:
#                     # Terminal state
#                     remaining_horcruxes = sum(1 for h in state.horcrux_states if h[2])
#                     if remaining_horcruxes == 0:
#                         new_value_function[state] = 100  # High reward for success
#                     else:
#                         new_value_function[state] = -10 * remaining_horcruxes  # Penalty for each remaining horcrux
#                     continue  # No actions from terminal state
#
#                 # Generate possible actions
#                 actions = self._generate_actions(state)
#
#                 if not actions:
#                     # No possible actions, assign a minimal utility
#                     best_action = ("wait", self.wizard_names[0]),
#                     best_utility = 0
#                 else:
#                     best_utility = float('-inf')
#                     best_action = None
#
#                     for action in actions:
#                         expected_value = self._compute_expected_value(state, action, gamma)
#                         if expected_value > best_utility:
#                             best_utility = expected_value
#                             best_action = action
#
#                 # Update the value function and policy
#                 if abs(best_utility - self.value_function[state]) > delta:
#                     delta = abs(best_utility - self.value_function[state])
#
#                 new_value_function[state] = best_utility
#                 new_policy[state] = best_action
#
#             # Update the value function and policy
#             self.value_function = new_value_function
#             self.policy = new_policy
#
#             print(f"Iteration {iteration}: max delta = {delta}")
#             iteration += 1
#
#             # Check for convergence
#             if delta < epsilon:
#                 break
#
#             # Optional: Limit the number of iterations to prevent infinite loops
#             if iteration >= 1000:
#                 print("Value Iteration did not converge within the iteration limit.")
#                 break
#
#         print(f"Value Iteration converged in {iteration} iterations.\n")
#
#     def _generate_actions(self, state):
#         """
#         Generate all possible actions from the current state.
#         Each action is a tuple of atomic actions, one per wizard.
#         """
#         actions_per_wizard = []
#         for wizard in state.wizard_positions:
#             wizard_name, position = wizard
#             possible_actions = self._get_possible_actions(wizard_name, position, state)
#             actions_per_wizard.append(possible_actions)
#         # Cartesian product of actions for all wizards
#         all_possible_actions = list(itertools.product(*actions_per_wizard))
#         return all_possible_actions
#
#     def _get_possible_actions(self, wizard_name, position, state):
#         """
#         Get possible actions for a single wizard based on the current state.
#         """
#         actions = []
#
#         # If on a horcrux, can destroy it
#         horcrux_at_pos = self._find_horcrux_at_position(position, state.horcrux_states)
#         if horcrux_at_pos:
#             actions.append(("destroy", wizard_name, horcrux_at_pos))
#
#         # Move actions: up, down, left, right
#         for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#             new_x, new_y = position[0] + dx, position[1] + dy
#             if (new_x, new_y) in self.passable:
#                 actions.append(("move", wizard_name, (new_x, new_y)))
#
#         # Wait action
#         actions.append(("wait", wizard_name))
#
#         return actions
#
#     def _find_horcrux_at_position(self, position, horcrux_states):
#         """
#         Check if there's a horcrux at the given position.
#         Returns the horcrux name if present, else None.
#         """
#         for h_name, h_pos, exists in horcrux_states:
#             if exists and h_pos == position:
#                 return h_name
#         return None
#
#     def _compute_expected_value(self, state, action, gamma):
#         """
#         Compute the expected value of taking an action in a state.
#         Returns the expected value.
#         """
#         # Initialize reward
#         reward = 0
#         actions = action  # Tuple of atomic actions
#
#         # Apply actions to get new wizard positions and update horcruxes
#         new_wizard_positions = list(state.wizard_positions)
#         new_horcrux_states = list(state.horcrux_states)
#
#         for atomic_action in actions:
#             if atomic_action[0] == "destroy":
#                 _, wizard_name, horcrux_name = atomic_action
#                 reward += 10  # Increased reward for destruction
#                 # Remove horcrux
#                 for idx, (h_name, h_pos, exists) in enumerate(new_horcrux_states):
#                     if h_name == horcrux_name and exists:
#                         new_horcrux_states[idx] = (h_name, h_pos, False)
#                         break
#             elif atomic_action[0] == "move":
#                 _, wizard_name, new_pos = atomic_action
#                 # Update wizard position
#                 for idx, (w_name, w_pos) in enumerate(new_wizard_positions):
#                     if w_name == wizard_name:
#                         new_wizard_positions[idx] = (w_name, new_pos)
#                         break
#             elif atomic_action[0] == "wait":
#                 pass  # No change
#
#         # Simulate death eaters' movements
#         death_eater_movements = {}
#         for de_name, de_info in self.death_eater_info.items():
#             path = de_info['path']
#             current_index = None
#             for de in state.death_eater_indices:
#                 if de[0] == de_name:
#                     current_index = de[1]
#                     break
#             possible_indices = []
#             probabilities = []
#             if len(path) == 1:
#                 possible_indices = [0]
#                 probabilities = [1.0]
#             else:
#                 # Possible moves: stay, move forward, move backward
#                 moves = []
#                 if current_index > 0:
#                     moves.append(current_index - 1)
#                 moves.append(current_index)
#                 if current_index < len(path) - 1:
#                     moves.append(current_index + 1)
#                 possible_indices = moves
#                 probabilities = [1 / len(moves)] * len(moves)  # Uniform probability
#
#             death_eater_movements[de_name] = list(zip(possible_indices, probabilities))
#
#         # Simulate horcruxes' movements
#         horcrux_movements = {}
#         for h_name, h_pos, exists in new_horcrux_states:
#             if exists:
#                 prob_move = self.horcrux_info[h_name]['prob_change_location']
#                 possible_locs = self.horcrux_info[h_name]['possible_locations']
#                 num_new_locs = len(possible_locs)
#                 prob_per_loc = prob_move / num_new_locs
#                 # With probability prob_move, horcrux moves to one of possible_locs
#                 # With probability (1 - prob_move), it stays
#                 movements = [(loc, prob_per_loc) for loc in possible_locs]
#                 movements.append((h_pos, 1 - prob_move))
#                 horcrux_movements[h_name] = movements
#             else:
#                 horcrux_movements[h_name] = [(h_pos, 1.0)]  # No movement if destroyed
#
#         # Generate all possible combinations of death eater movements
#         death_eater_names = list(death_eater_movements.keys())
#         death_eater_move_lists = [death_eater_movements[name] for name in death_eater_names]
#         all_de_moves = list(itertools.product(*death_eater_move_lists))
#
#         # Generate all possible combinations of horcrux movements
#         horcrux_names = list(horcrux_movements.keys())
#         horcrux_move_lists = [horcrux_movements[name] for name in horcrux_names]
#         all_h_moves = list(itertools.product(*horcrux_move_lists))
#
#         expected_value = 0
#
#         # Enumerate all combinations of death eater moves and horcrux moves
#         for de_move in all_de_moves:
#             # Calculate death eater movement probability
#             de_prob = 1.0
#             new_death_eater_indices = []
#             for de_name, (de_new_index, de_move_prob) in zip(death_eater_names, de_move):
#                 de_prob *= de_move_prob
#                 new_death_eater_indices.append((de_name, de_new_index))
#
#             for h_move in all_h_moves:
#                 # Calculate horcrux movement probability
#                 h_prob = 1.0
#                 new_horcrux_states_updated = list(new_horcrux_states)
#                 for idx, (h_name, h_move_entry) in enumerate(zip(horcrux_names, h_move)):
#                     h_new_loc, h_move_prob = h_move_entry
#                     h_prob *= h_move_prob
#                     # Update horcrux location and existence
#                     new_horcrux_states_updated[idx] = (h_name, h_new_loc, h_new_loc != (-1, -1))
#
#                 # Total probability for this combination
#                 total_prob = de_prob * h_prob
#
#                 # Check for encounters
#                 penalty = 0
#                 wizard_locations = {w[0]: w[1] for w in new_wizard_positions}
#                 death_eater_positions = set()
#                 for de_idx, de_name in enumerate(death_eater_names):
#                     path = self.death_eaters_paths[de_name]
#                     de_current_index = new_death_eater_indices[de_idx][1]
#                     de_pos = path[de_current_index]
#                     death_eater_positions.add(de_pos)
#                 for wiz_name, wiz_pos in wizard_locations.items():
#                     if wiz_pos in death_eater_positions:
#                         penalty -= 2  # Adjusted penalty per encounter
#
#                 # Create new state
#                 new_state = State(
#                     wizard_positions=tuple(new_wizard_positions),
#                     horcrux_states=tuple(new_horcrux_states_updated),
#                     death_eater_indices=tuple(new_death_eater_indices),
#                     turns_left=state.turns_left - 1
#                 )
#
#                 # Accumulate expected value
#                 future_value = self.value_function.get(new_state, 0)
#                 expected_value += total_prob * (reward + penalty + gamma * future_value)
#
#         return expected_value
#
#     def _compute_passable_positions(self):
#         """Precompute passable positions on the map."""
#         passable = set()
#         for i in range(len(self.map)):
#             for j in range(len(self.map[0])):
#                 if self.map[i][j] != 'I':
#                     passable.add((i, j))
#         return passable
#
#     @lru_cache(maxsize=None)
#     def _find_path(self, start, goal, map_grid):
#         """Use BFS to find the shortest path from start to goal."""
#         queue = deque()
#         queue.append((start, [start]))
#         visited = set()
#         visited.add(start)
#         while queue:
#             current, path = queue.popleft()
#             if current == goal:
#                 return path
#             neighbors = self._get_neighbors(current, map_grid)
#             for neighbor in neighbors:
#                 if neighbor not in visited:
#                     visited.add(neighbor)
#                     queue.append((neighbor, path + [neighbor]))
#         return None
#
#     def _get_neighbors(self, position, map_grid):
#         """Get passable neighbors for a given position."""
#         x, y = position
#         neighbors = []
#         for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#             new_x, new_y = x + dx, y + dy
#             if (0 <= new_x < len(map_grid)) and (0 <= new_y < len(map_grid[0])):
#                 if map_grid[new_x][new_y] != 'I':
#                     neighbors.append((new_x, new_y))
#         return neighbors
#
#
# class WizardAgent:
#     """
#     A simple wizard agent that follows a heuristic-based approach to collect horcruxes
#     and avoid death eaters.
#     """
#
#     def __init__(self, initial):
#         """
#         Initialize the agent with the initial state.
#         """
#         # Convert map to tuple of tuples for hashability
#         self.map = tuple(tuple(row) for row in initial['map'])
#         self.wizards = initial['wizards']
#         self.horcruxes = initial['horcrux']
#         self.death_eaters = initial['death_eaters']
#         self.turns_to_go = initial['turns_to_go']
#         self.width = len(self.map[0])
#         self.height = len(self.map)
#         self.passable = self._compute_passable_positions()
#         self.targets = list(self.horcruxes.keys())
#         self.visited = set()
#         for wiz in self.wizards.values():
#             self.visited.add(wiz['location'])
#
#     def act(self, state):
#         """
#         Decide on the next action based on the current state.
#         """
#         actions = []
#         for wizard_name, wizard_info in state['wizards'].items():
#             current_pos = wizard_info['location']
#             # If on a horcrux, destroy it
#             horcrux_at_pos = self._find_horcrux_at_position(current_pos, state['horcrux'])
#             if horcrux_at_pos:
#                 actions.append(("destroy", wizard_name, horcrux_at_pos))
#                 continue
#
#             # Find the nearest horcrux
#             nearest_horcrux, path = self._find_nearest_horcrux(current_pos, state['horcrux'])
#             if nearest_horcrux and path:
#                 if len(path) > 1:
#                     next_step = path[1]  # Move towards the horcrux
#                     actions.append(("move", wizard_name, next_step))
#                     self.visited.add(next_step)
#                 else:
#                     # Already at the horcrux location
#                     actions.append(("destroy", wizard_name, nearest_horcrux))
#                 continue
#
#             # If no horcruxes left, move towards Voldemort if possible
#             if not state['horcrux']:
#                 voldemort_pos = self._find_voldemort_position(self.map)
#                 if voldemort_pos and current_pos != voldemort_pos:
#                     path = self._find_path(current_pos, voldemort_pos, self.map)
#                     if path and len(path) > 1:
#                         next_step = path[1]
#                         actions.append(("move", wizard_name, next_step))
#                         self.visited.add(next_step)
#                         continue
#                     elif current_pos == voldemort_pos:
#                         actions.append(("kill", "Harry Potter"))
#                         continue
#
#             # If no specific action, wait
#             actions.append(("wait", wizard_name))
#
#         # If no actions decided, wait
#         if not actions:
#             for wizard_name in state['wizards'].keys():
#                 actions.append(("wait", wizard_name))
#
#         return tuple(actions)
#
#     def _compute_passable_positions(self):
#         """Precompute passable positions on the map."""
#         passable = set()
#         for i in range(len(self.map)):
#             for j in range(len(self.map[0])):
#                 if self.map[i][j] != 'I':
#                     passable.add((i, j))
#         return passable
#
#     def _find_horcrux_at_position(self, position, horcruxes):
#         """Check if there's a horcrux at the given position."""
#         for name, info in horcruxes.items():
#             if info['location'] == position:
#                 return name
#         return None
#
#     def _find_nearest_horcrux(self, current_pos, horcruxes):
#         """Find the nearest horcrux and return its name and path."""
#         min_distance = float('inf')
#         nearest_horcrux = None
#         nearest_path = None
#         for name, info in horcruxes.items():
#             if info['location'] == (-1, -1):
#                 continue  # Horcrux already destroyed
#             path = self._find_path(current_pos, tuple(info['location']), self.map)
#             if path and len(path) < min_distance:
#                 min_distance = len(path)
#                 nearest_horcrux = name
#                 nearest_path = path
#         return nearest_horcrux, nearest_path
#
#     @lru_cache(maxsize=None)
#     def _find_path(self, start, goal, map_grid):
#         """Use BFS to find the shortest path from start to goal."""
#         queue = deque()
#         queue.append((start, [start]))
#         visited = set()
#         visited.add(start)
#         while queue:
#             current, path = queue.popleft()
#             if current == goal:
#                 return path
#             neighbors = self._get_neighbors(current, map_grid)
#             for neighbor in neighbors:
#                 if neighbor not in visited:
#                     visited.add(neighbor)
#                     queue.append((neighbor, path + [neighbor]))
#         return None
#
#     def _find_voldemort_position(self, map_grid):
#         """Find Voldemort's position on the map."""
#         for i, row in enumerate(map_grid):
#             for j, cell in enumerate(row):
#                 if cell == 'V':
#                     return (i, j)
#         return None
#
#     def _get_neighbors(self, position, map_grid):
#         """Get passable neighbors for a given position."""
#         x, y = position
#         neighbors = []
#         for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#             new_x, new_y = x + dx, y + dy
#             if (0 <= new_x < len(map_grid)) and (0 <= new_y < len(map_grid[0])):
#                 if map_grid[new_x][new_y] != 'I':
#                     neighbors.append((new_x, new_y))
#         return neighbors



#############################
# Policy Iteration
#############################

# ex3.py

import itertools
import math
import random
from collections import deque, defaultdict
from heapq import heappush, heappop
from typing import Tuple, Dict, List, Set
from functools import lru_cache

# Define your IDs
ids = ['207476763']  # Replace with your actual ID(s)

# Define a hashable State class
class State:
    def __init__(self, wizard_positions, horcrux_states, death_eater_indices, turns_left):
        """
        Initialize the state.

        :param wizard_positions: Tuple of (wizard_name, (x, y))
        :param horcrux_states: Tuple of (horcrux_name, (x, y), exists)
        :param death_eater_indices: Tuple of (death_eater_name, path_index)
        :param turns_left: Integer representing remaining turns
        """
        self.wizard_positions = tuple(sorted(wizard_positions))
        self.horcrux_states = tuple(sorted(horcrux_states))
        self.death_eater_indices = tuple(sorted(death_eater_indices))
        self.turns_left = turns_left

    def __hash__(self):
        return hash((self.wizard_positions, self.horcrux_states, self.death_eater_indices, self.turns_left))

    def __eq__(self, other):
        return (self.wizard_positions == other.wizard_positions and
                self.horcrux_states == other.horcrux_states and
                self.death_eater_indices == other.death_eater_indices and
                self.turns_left == other.turns_left)

    def __repr__(self):
        return f"State(wizards={self.wizard_positions}, horcruxes={self.horcrux_states}, death_eaters={self.death_eater_indices}, turns_left={self.turns_left})"


class OptimalWizardAgent:
    """
    An optimal wizard agent that uses Policy Iteration to compute the optimal policy.
    """

    def __init__(self, initial):
        """
        Initialize the agent with the initial state.
        Perform Policy Iteration to compute the optimal policy.
        """
        # Convert map to tuple of tuples for hashability
        self.map = tuple(tuple(row) for row in initial['map'])
        self.wizards_initial = initial['wizards']
        self.horcruxes_initial = initial['horcrux']
        self.death_eaters_initial = initial['death_eaters']
        self.turns_to_go = initial['turns_to_go']
        self.width = len(self.map[0])
        self.height = len(self.map)
        self.passable = self._compute_passable_positions()

        # Initialize death eater paths
        self.death_eaters_paths = {}
        for de_name, de_info in self.death_eaters_initial.items():
            self.death_eaters_paths[de_name] = de_info['path']

        # Initialize horcrux states
        self.horcrux_names = list(self.horcruxes_initial.keys())

        # Initialize wizards
        self.wizard_names = list(self.wizards_initial.keys())

        # Precompute all possible horcrux locations and movement probabilities
        self.horcrux_info = {}
        for h_name, h_info in self.horcruxes_initial.items():
            self.horcrux_info[h_name] = {
                'current_location': tuple(h_info['location']),
                'possible_locations': tuple(tuple(loc) for loc in h_info['possible_locations']),
                'prob_change_location': h_info['prob_change_location']
            }

        # Precompute death eater movement probabilities
        self.death_eater_info = {}
        for de_name, de_info in self.death_eaters_initial.items():
            path = de_info['path']
            self.death_eater_info[de_name] = {
                'path': path,
                'current_index': de_info['index']  # Ensure consistent key naming
            }

        # Initialize all possible states
        self.all_states = self._enumerate_all_states()

        # Initialize the initial state
        initial_wizard_positions = tuple(
            (name, tuple(info['location'])) for name, info in self.wizards_initial.items()
        )

        initial_horcrux_states = tuple(
            (name, tuple(info['location']), True) for name, info in self.horcruxes_initial.items()
        )

        initial_death_eater_indices = tuple(
            (name, info['current_index']) for name, info in self.death_eater_info.items()
        )

        self.initial_state = State(
            wizard_positions=initial_wizard_positions,
            horcrux_states=initial_horcrux_states,
            death_eater_indices=initial_death_eater_indices,
            turns_left=self.turns_to_go
        )

        # Initialize Value Function and Policy
        self.value_function = defaultdict(lambda: 0)
        self.policy = {}

        # Perform Policy Iteration
        self.policy_iteration()

    def _enumerate_all_states(self):
        """
        Enumerate all possible states based on the initial configurations.
        """
        all_states = set()

        # Possible wizard positions
        wizard_positions = []
        for wizard_name, wizard_info in self.wizards_initial.items():
            wizard_positions.append([(wizard_name, pos) for pos in self.passable])

        # Possible horcrux states
        horcrux_states = []
        for h_name, h_info in self.horcruxes_initial.items():
            horcrux_states.append([
                (h_name, loc, True) for loc in [h_info['location']] + list(h_info['possible_locations'])
            ] + [
                (h_name, (-1, -1), False)
            ])  # Include destroyed state

        # Possible death eater indices
        death_eater_indices = []
        for de_name, de_info in self.death_eater_info.items():
            path = de_info['path']
            death_eater_indices.append([(de_name, idx) for idx in range(len(path))])

        # Possible turns_left
        turns_left = list(range(self.turns_to_go + 1))

        # Cartesian product of all components
        for w_pos in itertools.product(*wizard_positions):
            for h_state in itertools.product(*horcrux_states):
                for de_idx in itertools.product(*death_eater_indices):
                    for t_left in turns_left:
                        state = State(
                            wizard_positions=w_pos,
                            horcrux_states=h_state,
                            death_eater_indices=de_idx,
                            turns_left=t_left
                        )
                        all_states.add(state)

        return all_states

    def act(self, state):
        """
        Decide the next action based on the current state using the precomputed policy.
        """
        # Construct the State object
        wizard_positions = tuple(
            (name, tuple(info['location'])) for name, info in state['wizards'].items()
        )

        horcrux_states = tuple(
            (name, tuple(info['location']), info['location'] != (-1, -1))
            for name, info in state['horcrux'].items()
        )

        death_eater_indices = tuple(
            (name, info['index']) for name, info in state['death_eaters'].items()
        )

        turns_left = state['turns_to_go']

        current_state = State(
            wizard_positions=wizard_positions,
            horcrux_states=horcrux_states,
            death_eater_indices=death_eater_indices,
            turns_left=turns_left
        )

        # Retrieve the optimal action from the policy
        action = self.policy.get(current_state, (("wait", self.wizard_names[0]),))

        # Return the action as a tuple
        return action  # Changed from (action,) to action

    def policy_iteration(self):
        """
        Perform Policy Iteration to compute the optimal value function and policy.
        """
        gamma = 0.9  # Discount factor
        epsilon = 1e-3  # Convergence threshold for policy evaluation

        # Initialize a random policy
        for state in self.all_states:
            if state.turns_left == 0:
                continue  # No actions in terminal states
            actions = self._generate_actions(state)
            if actions:
                self.policy[state] = actions[0]  # Assign the first action as the initial policy
            else:
                self.policy[state] = (("wait", self.wizard_names[0]),)

        is_policy_stable = False
        iteration = 0

        while not is_policy_stable:
            print(f"Policy Iteration Step {iteration}: Policy Evaluation")
            # Policy Evaluation
            self.policy_evaluation(gamma, epsilon)

            print(f"Policy Iteration Step {iteration}: Policy Improvement")
            # Policy Improvement
            is_policy_stable = self.policy_improvement(gamma)

            iteration += 1

            # Optional: Limit the number of iterations to prevent infinite loops
            if iteration >= 1000:
                print("Policy Iteration did not converge within the iteration limit.")
                break

        print(f"Policy Iteration converged in {iteration} iterations.\n")

    def policy_evaluation(self, gamma, epsilon):
        """
        Evaluate the current policy by computing the value function V(s) for all states.
        """
        # Initialize value function to 0 for all states
        for state in self.all_states:
            self.value_function[state] = 0

        while True:
            delta = 0
            new_value_function = self.value_function.copy()

            for state in self.all_states:
                if state.turns_left == 0:
                    # Terminal state
                    remaining_horcruxes = sum(1 for h in state.horcrux_states if h[2])
                    if remaining_horcruxes == 0:
                        new_value_function[state] = 100  # High reward for success
                    else:
                        new_value_function[state] = -10 * remaining_horcruxes  # Penalty for each remaining horcrux
                    continue  # No actions from terminal state

                # Get the action from the current policy
                action = self.policy.get(state, (("wait", self.wizard_names[0]),))

                # Compute the expected value for this action
                expected_value = self._compute_expected_value(state, action, gamma)

                # Update the value function
                if abs(expected_value - self.value_function[state]) > delta:
                    delta = abs(expected_value - self.value_function[state])

                new_value_function[state] = expected_value

            self.value_function = new_value_function

            print(f"Policy Evaluation: max delta = {delta}")

            if delta < epsilon:
                break

    def policy_improvement(self, gamma):
        """
        Improve the policy based on the current value function.
        Returns True if the policy is stable (no changes), False otherwise.
        """
        is_policy_stable = True

        for state in self.all_states:
            if state.turns_left == 0:
                continue  # No actions in terminal states

            old_action = self.policy.get(state, (("wait", self.wizard_names[0]),))
            actions = self._generate_actions(state)

            if not actions:
                continue  # No possible actions to improve

            # Find the best action based on the current value function
            best_action = None
            best_utility = float('-inf')

            for action in actions:
                expected_value = self._compute_expected_value(state, action, gamma)
                if expected_value > best_utility:
                    best_utility = expected_value
                    best_action = action

            # Update the policy if a better action is found
            if best_action != old_action:
                self.policy[state] = best_action
                is_policy_stable = False

        return is_policy_stable

    def _generate_actions(self, state):
        """
        Generate all possible actions from the current state.
        Each action is a tuple of atomic actions, one per wizard.
        """
        actions_per_wizard = []
        for wizard in state.wizard_positions:
            wizard_name, position = wizard
            possible_actions = self._get_possible_actions(wizard_name, position, state)
            actions_per_wizard.append(possible_actions)
        # Cartesian product of actions for all wizards
        all_possible_actions = list(itertools.product(*actions_per_wizard))
        return all_possible_actions

    def _get_possible_actions(self, wizard_name, position, state):
        """
        Get possible actions for a single wizard based on the current state.
        """
        actions = []

        # If on a horcrux, can destroy it
        horcrux_at_pos = self._find_horcrux_at_position(position, state.horcrux_states)
        if horcrux_at_pos:
            actions.append(("destroy", wizard_name, horcrux_at_pos))

        # Move actions: up, down, left, right
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = position[0] + dx, position[1] + dy
            if (new_x, new_y) in self.passable:
                actions.append(("move", wizard_name, (new_x, new_y)))

        # Wait action
        actions.append(("wait", wizard_name))

        return actions

    def _find_horcrux_at_position(self, position, horcrux_states):
        """
        Check if there's a horcrux at the given position.
        Returns the horcrux name if present, else None.
        """
        for h_name, h_pos, exists in horcrux_states:
            if exists and h_pos == position:
                return h_name
        return None

    def _compute_expected_value(self, state, action, gamma):
        """
        Compute the expected value of taking an action in a state under the current policy.
        Returns the expected value.
        """
        # Initialize reward
        reward = 0
        actions = action  # Tuple of atomic actions

        # Apply actions to get new wizard positions and update horcruxes
        new_wizard_positions = list(state.wizard_positions)
        new_horcrux_states = list(state.horcrux_states)

        for atomic_action in actions:
            if atomic_action[0] == "destroy":
                _, wizard_name, horcrux_name = atomic_action
                reward += 10  # Increased reward for destruction
                # Remove horcrux
                for idx, (h_name, h_pos, exists) in enumerate(new_horcrux_states):
                    if h_name == horcrux_name and exists:
                        new_horcrux_states[idx] = (h_name, h_pos, False)
                        break
            elif atomic_action[0] == "move":
                _, wizard_name, new_pos = atomic_action
                # Update wizard position
                for idx, (w_name, w_pos) in enumerate(new_wizard_positions):
                    if w_name == wizard_name:
                        new_wizard_positions[idx] = (w_name, new_pos)
                        break
            elif atomic_action[0] == "wait":
                pass  # No change

        # Simulate death eaters' movements
        death_eater_movements = {}
        for de_name, de_info in self.death_eater_info.items():
            path = de_info['path']
            current_index = None
            for de in state.death_eater_indices:
                if de[0] == de_name:
                    current_index = de[1]
                    break
            possible_indices = []
            probabilities = []
            if len(path) == 1:
                possible_indices = [0]
                probabilities = [1.0]
            else:
                # Possible moves: stay, move forward, move backward
                moves = []
                if current_index > 0:
                    moves.append(current_index - 1)
                moves.append(current_index)
                if current_index < len(path) - 1:
                    moves.append(current_index + 1)
                possible_indices = moves
                probabilities = [1 / len(moves)] * len(moves)  # Uniform probability

            death_eater_movements[de_name] = list(zip(possible_indices, probabilities))

        # Simulate horcruxes' movements
        horcrux_movements = {}
        for h_name, h_pos, exists in new_horcrux_states:
            if exists:
                prob_move = self.horcrux_info[h_name]['prob_change_location']
                possible_locs = self.horcrux_info[h_name]['possible_locations']
                num_new_locs = len(possible_locs)
                prob_per_loc = prob_move / num_new_locs
                # With probability prob_move, horcrux moves to one of possible_locs
                # With probability (1 - prob_move), it stays
                movements = [(loc, prob_per_loc) for loc in possible_locs]
                movements.append((h_pos, 1 - prob_move))
                horcrux_movements[h_name] = movements
            else:
                horcrux_movements[h_name] = [(h_pos, 1.0)]  # No movement if destroyed

        # Generate all possible combinations of death eater movements
        death_eater_names = list(death_eater_movements.keys())
        death_eater_move_lists = [death_eater_movements[name] for name in death_eater_names]
        all_de_moves = list(itertools.product(*death_eater_move_lists))

        # Generate all possible combinations of horcrux movements
        horcrux_names = list(horcrux_movements.keys())
        horcrux_move_lists = [horcrux_movements[name] for name in horcrux_names]
        all_h_moves = list(itertools.product(*horcrux_move_lists))

        expected_value = 0

        # Enumerate all combinations of death eater moves and horcrux moves
        for de_move in all_de_moves:
            # Calculate death eater movement probability
            de_prob = 1.0
            new_death_eater_indices = []
            for de_name, (de_new_index, de_move_prob) in zip(death_eater_names, de_move):
                de_prob *= de_move_prob
                new_death_eater_indices.append((de_name, de_new_index))

            for h_move in all_h_moves:
                # Calculate horcrux movement probability
                h_prob = 1.0
                new_horcrux_states_updated = list(new_horcrux_states)
                for idx, (h_name, h_move_entry) in enumerate(zip(horcrux_names, h_move)):
                    h_new_loc, h_move_prob = h_move_entry
                    h_prob *= h_move_prob
                    # Update horcrux location and existence
                    new_horcrux_states_updated[idx] = (h_name, h_new_loc, h_new_loc != (-1, -1))

                # Total probability for this combination
                total_prob = de_prob * h_prob

                # Check for encounters
                penalty = 0
                wizard_locations = {w[0]: w[1] for w in new_wizard_positions}
                death_eater_positions = set()
                for de_idx, de_name in enumerate(death_eater_names):
                    path = self.death_eaters_paths[de_name]
                    de_current_index = new_death_eater_indices[de_idx][1]
                    de_pos = path[de_current_index]
                    death_eater_positions.add(de_pos)
                for wiz_name, wiz_pos in wizard_locations.items():
                    if wiz_pos in death_eater_positions:
                        penalty -= 2  # Adjusted penalty per encounter

                # Create new state
                new_state = State(
                    wizard_positions=tuple(new_wizard_positions),
                    horcrux_states=tuple(new_horcrux_states_updated),
                    death_eater_indices=tuple(new_death_eater_indices),
                    turns_left=state.turns_left - 1
                )

                # Accumulate expected value
                future_value = self.value_function.get(new_state, 0)
                expected_value += total_prob * (reward + penalty + gamma * future_value)

        return expected_value

    def _compute_passable_positions(self):
        """Precompute passable positions on the map."""
        passable = set()
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] != 'I':
                    passable.add((i, j))
        return passable

    @lru_cache(maxsize=None)
    def _find_path(self, start, goal, map_grid):
        """Use BFS to find the shortest path from start to goal."""
        queue = deque()
        queue.append((start, [start]))
        visited = set()
        visited.add(start)
        while queue:
            current, path = queue.popleft()
            if current == goal:
                return path
            neighbors = self._get_neighbors(current, map_grid)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def _get_neighbors(self, position, map_grid):
        """Get passable neighbors for a given position."""
        x, y = position
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < len(map_grid)) and (0 <= new_y < len(map_grid[0])):
                if map_grid[new_x][new_y] != 'I':
                    neighbors.append((new_x, new_y))
        return neighbors


#############################
# Value Iteration
#############################

# ex3_refactored.py

import itertools
import math
import random
from collections import deque, defaultdict
from heapq import heappush, heappop
from typing import Tuple, Dict, List, Set
from functools import lru_cache
import time

# Define your IDs
ids = ['207476763']  # Replace with your actual ID(s)

# Define a hashable State class
class State:
    def __init__(self, wizard_positions, horcrux_states, death_eater_indices, turns_left):
        """
        Initialize the state.

        :param wizard_positions: Tuple of (wizard_name, (x, y))
        :param horcrux_states: Tuple of (horcrux_name, (x, y), exists)
        :param death_eater_indices: Tuple of (death_eater_name, path_index)
        :param turns_left: Integer representing remaining turns
        """
        self.wizard_positions = tuple(sorted(wizard_positions))
        self.horcrux_states = tuple(sorted(horcrux_states))
        self.death_eater_indices = tuple(sorted(death_eater_indices))
        self.turns_left = turns_left

    def __hash__(self):
        return hash((self.wizard_positions, self.horcrux_states, self.death_eater_indices, self.turns_left))

    def __eq__(self, other):
        return (self.wizard_positions == other.wizard_positions and
                self.horcrux_states == other.horcrux_states and
                self.death_eater_indices == other.death_eater_indices and
                self.turns_left == other.turns_left)

    def __repr__(self):
        return f"State(wizards={self.wizard_positions}, horcruxes={self.horcrux_states}, death_eaters={self.death_eater_indices}, turns_left={self.turns_left})"


class OptimalWizardAgent:
    """
    An optimal wizard agent that uses a Hybrid Policy and Value Iteration to compute the optimal policy.
    """

    def __init__(self, initial):
        """
        Initialize the agent with the initial state.
        Perform Hybrid Policy and Value Iteration to compute the optimal policy.
        """
        # Record the start time for initialization
        self.init_start_time = time.time()

        # Convert map to tuple of tuples for hashability
        self.map = tuple(tuple(row) for row in initial['map'])
        self.wizards_initial = initial['wizards']
        self.horcruxes_initial = initial['horcrux']
        self.death_eaters_initial = initial['death_eaters']
        self.turns_to_go = initial['turns_to_go']
        self.width = len(self.map[0])
        self.height = len(self.map)
        self.passable = self._compute_passable_positions()

        # Initialize death eater paths
        self.death_eaters_paths = {}
        for de_name, de_info in self.death_eaters_initial.items():
            self.death_eaters_paths[de_name] = de_info['path']

        # Initialize horcrux states
        self.horcrux_names = list(self.horcruxes_initial.keys())

        # Initialize wizards
        self.wizard_names = list(self.wizards_initial.keys())

        # Precompute all possible horcrux locations and movement probabilities
        self.horcrux_info = {}
        for h_name, h_info in self.horcruxes_initial.items():
            self.horcrux_info[h_name] = {
                'current_location': tuple(h_info['location']),
                'possible_locations': tuple(tuple(loc) for loc in h_info['possible_locations']),
                'prob_change_location': h_info['prob_change_location']
            }

        # Precompute death eater movement probabilities
        self.death_eater_info = {}
        for de_name, de_info in self.death_eaters_initial.items():
            path = de_info['path']
            self.death_eater_info[de_name] = {
                'path': path,
                'current_index': de_info['index']  # Ensure consistent key naming
            }

        # Initialize all possible states
        # Note: Enumerating all states is infeasible for large state spaces.
        # Instead, we'll generate states on-the-fly as needed.
        self.all_states = self._enumerate_all_states()

        # Initialize the initial state
        initial_wizard_positions = tuple(
            (name, tuple(info['location'])) for name, info in self.wizards_initial.items()
        )

        initial_horcrux_states = tuple(
            (name, tuple(info['location']), True) for name, info in self.horcruxes_initial.items()
        )

        initial_death_eater_indices = tuple(
            (name, info['current_index']) for name, info in self.death_eater_info.items()
        )

        self.initial_state = State(
            wizard_positions=initial_wizard_positions,
            horcrux_states=initial_horcrux_states,
            death_eater_indices=initial_death_eater_indices,
            turns_left=self.turns_to_go
        )

        # Initialize Value Function and Policy
        self.value_function = defaultdict(lambda: 0)
        self.policy = {}

        # Perform Hybrid Policy and Value Iteration
        self.hybrid_policy_value_iteration()

        # Record the end time for initialization
        self.init_end_time = time.time()
        self.init_duration = self.init_end_time - self.init_start_time
        print(f"Initialization completed in {self.init_duration:.2f} seconds.\n")

    def _enumerate_all_states(self):
        """
        Enumerate all possible states based on the initial configurations.
        Note: For large state spaces, this is not feasible. Instead, states are generated on-the-fly.
        """
        # Placeholder for compatibility; actual state generation is handled dynamically.
        return set()

    def act(self, state):
        """
        Decide the next action based on the current state using the precomputed policy.
        """
        # Construct the State object
        wizard_positions = tuple(
            (name, tuple(info['location'])) for name, info in state['wizards'].items()
        )

        horcrux_states = tuple(
            (name, tuple(info['location']), info['location'] != (-1, -1))
            for name, info in state['horcrux'].items()
        )

        death_eater_indices = tuple(
            (name, info['index']) for name, info in state['death_eaters'].items()
        )

        turns_left = state['turns_to_go']

        current_state = State(
            wizard_positions=wizard_positions,
            horcrux_states=horcrux_states,
            death_eater_indices=death_eater_indices,
            turns_left=turns_left
        )

        # Retrieve the optimal action from the policy
        action = self.policy.get(current_state, (("wait", self.wizard_names[0]),))

        # Return the action as a tuple
        return action  # Changed from (action,) to action

    def hybrid_policy_value_iteration(self):
        """
        Perform a hybrid of Policy Iteration and Value Iteration to compute the optimal policy and value function.
        """
        gamma = 0.9  # Discount factor
        epsilon = 1e-3  # Convergence threshold for policy evaluation
        max_iterations = 1000  # To prevent infinite loops

        # Step 1: Initialize Policy Iteration
        print("Step 1: Starting Policy Iteration")
        self.initialize_policy()

        is_policy_stable = False
        iteration = 0

        while not is_policy_stable and iteration < max_iterations:
            print(f"\nPolicy Iteration Iteration {iteration + 1}")
            # Policy Evaluation
            self.policy_evaluation(gamma, epsilon)

            # Policy Improvement
            is_policy_stable = self.policy_improvement(gamma)

            print(f"Policy Iteration Iteration {iteration + 1} {'Converged' if is_policy_stable else 'Continues'}")
            iteration += 1

            # Time Check: Ensure initialization does not exceed 300 seconds
            if time.time() - self.init_start_time > 290:  # Leave buffer for other operations
                print("Initialization time limit reached during Policy Iteration.")
                break

        print(f"\nPolicy Iteration completed in {iteration} iterations.")

        # Step 2: Value Iteration Refinement
        print("\nStep 2: Starting Value Iteration Refinement")
        self.value_iteration_refinement(gamma, epsilon, max_iterations=1000)

        print("Hybrid Policy and Value Iteration completed.\n")

    def initialize_policy(self):
        """
        Initialize the policy with arbitrary actions (e.g., first available action).
        """
        # Initialize a random policy
        for wizard_pos in self._get_initial_wizard_positions():
            initial_policy = self._get_initial_policy_for_wizard(wizard_pos)
            for state, action in initial_policy.items():
                self.policy[state] = action

    def _get_initial_wizard_positions(self):
        """
        Generator to yield initial wizard positions for policy initialization.
        """
        # Only the initial state is considered for policy initialization
        yield self.initial_state

    def _get_initial_policy_for_wizard(self, state):
        """
        Assign the first available action for each wizard in the state.
        """
        policy = {}
        if state.turns_left == 0:
            return policy  # No actions in terminal states

        actions = self._generate_actions(state)
        if actions:
            # Assign the first action as the initial policy for the state
            policy[state] = actions[0]
        else:
            # If no actions are available, assign 'wait'
            policy[state] = (("wait", self.wizard_names[0]),)

        return policy

    def policy_evaluation(self, gamma, epsilon):
        """
        Evaluate the current policy by computing the value function V(s) for all states.
        Uses Iterative Policy Evaluation.
        """
        delta = float('inf')
        while delta > epsilon:
            delta = 0
            new_value_function = self.value_function.copy()

            # Iterate over all states that have been encountered (dynamic state generation)
            for state in self.policy.keys():
                if state.turns_left == 0:
                    # Terminal state
                    remaining_horcruxes = sum(1 for h in state.horcrux_states if h[2])
                    if remaining_horcruxes == 0:
                        new_value = 100  # High reward for success
                    else:
                        new_value = -10 * remaining_horcruxes  # Penalty for each remaining horcrux
                    if abs(new_value - self.value_function[state]) > delta:
                        delta = abs(new_value - self.value_function[state])
                    new_value_function[state] = new_value
                    continue

                action = self.policy[state]
                expected_value = self._compute_expected_value(state, action, gamma)
                if abs(expected_value - self.value_function[state]) > delta:
                    delta = abs(expected_value - self.value_function[state])
                new_value_function[state] = expected_value

            self.value_function = new_value_function
            print(f"Policy Evaluation: max delta = {delta}")

            # Time Check: Ensure initialization does not exceed 300 seconds
            if time.time() - self.init_start_time > 290:
                print("Initialization time limit reached during Policy Evaluation.")
                break

    def policy_improvement(self, gamma):
        """
        Improve the policy based on the current value function.
        Returns True if the policy is stable (no changes), False otherwise.
        """
        policy_stable = True
        new_policy = {}

        for state in self.policy.keys():
            if state.turns_left == 0:
                continue  # No actions in terminal states

            old_action = self.policy[state]
            actions = self._generate_actions(state)
            if not actions:
                continue  # No possible actions to improve

            best_action = None
            best_utility = float('-inf')

            for action in actions:
                utility = self._compute_expected_value(state, action, gamma)
                if utility > best_utility:
                    best_utility = utility
                    best_action = action

            # Check if the best action is different from the current policy
            if best_action != old_action:
                policy_stable = False
                new_policy[state] = best_action
                print(f"Policy Improvement: State {state} changed action from {old_action} to {best_action}")
            else:
                new_policy[state] = old_action

        self.policy = new_policy
        return policy_stable

    def value_iteration_refinement(self, gamma, epsilon, max_iterations=1000):
        """
        Perform Value Iteration to refine the value function after Policy Iteration.
        """
        iteration = 0
        while iteration < max_iterations:
            delta = 0
            new_value_function = self.value_function.copy()

            # Iterate over all states in the policy
            for state in self.policy.keys():
                if state.turns_left == 0:
                    continue  # Skip terminal states

                actions = self._generate_actions(state)
                if not actions:
                    continue  # No actions available

                max_utility = float('-inf')
                for action in actions:
                    utility = self._compute_expected_value(state, action, gamma)
                    if utility > max_utility:
                        max_utility = utility

                if abs(max_utility - self.value_function[state]) > delta:
                    delta = abs(max_utility - self.value_function[state])
                new_value_function[state] = max_utility

            self.value_function = new_value_function
            print(f"Value Iteration Refinement: Iteration {iteration + 1}, max delta = {delta}")

            if delta < epsilon:
                print("Value Iteration Refinement converged.")
                break

            iteration += 1

            # Time Check: Ensure initialization does not exceed 300 seconds
            if time.time() - self.init_start_time > 290:
                print("Initialization time limit reached during Value Iteration Refinement.")
                break

    def _generate_actions(self, state):
        """
        Generate all possible actions from the current state.
        Each action is a tuple of atomic actions, one per wizard.
        """
        actions_per_wizard = []
        for wizard in state.wizard_positions:
            wizard_name, position = wizard
            possible_actions = self._get_possible_actions(wizard_name, position, state)
            actions_per_wizard.append(possible_actions)
        # Cartesian product of actions for all wizards
        all_possible_actions = list(itertools.product(*actions_per_wizard))
        return all_possible_actions

    def _get_possible_actions(self, wizard_name, position, state):
        """
        Get possible actions for a single wizard based on the current state.
        """
        actions = []

        # If on a horcrux, can destroy it
        horcrux_at_pos = self._find_horcrux_at_position(position, state.horcrux_states)
        if horcrux_at_pos:
            actions.append(("destroy", wizard_name, horcrux_at_pos))

        # Move actions: up, down, left, right
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = position[0] + dx, position[1] + dy
            if (new_x, new_y) in self.passable:
                actions.append(("move", wizard_name, (new_x, new_y)))

        # Wait action
        actions.append(("wait", wizard_name))

        return actions

    def _find_horcrux_at_position(self, position, horcrux_states):
        """
        Check if there's a horcrux at the given position.
        Returns the horcrux name if present, else None.
        """
        for h_name, h_pos, exists in horcrux_states:
            if exists and h_pos == position:
                return h_name
        return None

    def _compute_expected_value(self, state, action, gamma):
        """
        Compute the expected value of taking an action in a state.
        Returns the expected value.
        """
        # Initialize reward
        reward = 0
        actions = action  # Tuple of atomic actions

        # Apply actions to get new wizard positions and update horcruxes
        new_wizard_positions = list(state.wizard_positions)
        new_horcrux_states = list(state.horcrux_states)

        for atomic_action in actions:
            if atomic_action[0] == "destroy":
                _, wizard_name, horcrux_name = atomic_action
                reward += 10  # Increased reward for destruction
                # Remove horcrux
                for idx, (h_name, h_pos, exists) in enumerate(new_horcrux_states):
                    if h_name == horcrux_name and exists:
                        new_horcrux_states[idx] = (h_name, h_pos, False)
                        break
            elif atomic_action[0] == "move":
                _, wizard_name, new_pos = atomic_action
                # Update wizard position
                for idx, (w_name, w_pos) in enumerate(new_wizard_positions):
                    if w_name == wizard_name:
                        new_wizard_positions[idx] = (w_name, new_pos)
                        break
            elif atomic_action[0] == "wait":
                pass  # No change

        # Simulate death eaters' movements
        death_eater_movements = {}
        for de_name, de_info in self.death_eater_info.items():
            path = de_info['path']
            current_index = None
            for de in state.death_eater_indices:
                if de[0] == de_name:
                    current_index = de[1]
                    break
            possible_indices = []
            probabilities = []
            if len(path) == 1:
                possible_indices = [0]
                probabilities = [1.0]
            else:
                # Possible moves: stay, move forward, move backward
                moves = []
                if current_index > 0:
                    moves.append(current_index - 1)
                moves.append(current_index)
                if current_index < len(path) - 1:
                    moves.append(current_index + 1)
                possible_indices = moves
                probabilities = [1 / len(moves)] * len(moves)  # Uniform probability

            death_eater_movements[de_name] = list(zip(possible_indices, probabilities))

        # Simulate horcruxes' movements
        horcrux_movements = {}
        for h_name, h_pos, exists in new_horcrux_states:
            if exists:
                prob_move = self.horcrux_info[h_name]['prob_change_location']
                possible_locs = self.horcrux_info[h_name]['possible_locations']
                num_new_locs = len(possible_locs)
                prob_per_loc = prob_move / num_new_locs
                # With probability prob_move, horcrux moves to one of possible_locs
                # With probability (1 - prob_move), it stays
                movements = [(loc, prob_per_loc) for loc in possible_locs]
                movements.append((h_pos, 1 - prob_move))
                horcrux_movements[h_name] = movements
            else:
                horcrux_movements[h_name] = [(h_pos, 1.0)]  # No movement if destroyed

        # Generate all possible combinations of death eater movements
        death_eater_names = list(death_eater_movements.keys())
        death_eater_move_lists = [death_eater_movements[name] for name in death_eater_names]
        all_de_moves = list(itertools.product(*death_eater_move_lists))

        # Generate all possible combinations of horcrux movements
        horcrux_names = list(horcrux_movements.keys())
        horcrux_move_lists = [horcrux_movements[name] for name in horcrux_names]
        all_h_moves = list(itertools.product(*horcrux_move_lists))

        expected_value = 0

        # Enumerate all combinations of death eater moves and horcrux moves
        for de_move in all_de_moves:
            # Calculate death eater movement probability
            de_prob = 1.0
            new_death_eater_indices = []
            for de_name, (de_new_index, de_move_prob) in zip(death_eater_names, de_move):
                de_prob *= de_move_prob
                new_death_eater_indices.append((de_name, de_new_index))

            for h_move in all_h_moves:
                # Calculate horcrux movement probability
                h_prob = 1.0
                new_horcrux_states_updated = list(new_horcrux_states)
                for idx, (h_name, h_move_entry) in enumerate(zip(horcrux_names, h_move)):
                    h_new_loc, h_move_prob = h_move_entry
                    h_prob *= h_move_prob
                    # Update horcrux location and existence
                    new_horcrux_states_updated[idx] = (h_name, h_new_loc, h_new_loc != (-1, -1))

                # Total probability for this combination
                total_prob = de_prob * h_prob

                # Check for encounters
                penalty = 0
                wizard_locations = {w[0]: w[1] for w in new_wizard_positions}
                death_eater_positions = set()
                for de_idx, de_name in enumerate(death_eater_names):
                    path = self.death_eaters_paths[de_name]
                    de_current_index = new_death_eater_indices[de_idx][1]
                    de_pos = path[de_current_index]
                    death_eater_positions.add(de_pos)
                for wiz_name, wiz_pos in wizard_locations.items():
                    if wiz_pos in death_eater_positions:
                        penalty -= 2  # Adjusted penalty per encounter

                # Create new state
                new_state = State(
                    wizard_positions=tuple(new_wizard_positions),
                    horcrux_states=tuple(new_horcrux_states_updated),
                    death_eater_indices=tuple(new_death_eater_indices),
                    turns_left=state.turns_left - 1
                )

                # Accumulate expected value
                future_value = self.value_function.get(new_state, 0)
                expected_value += total_prob * (reward + penalty + gamma * future_value)

        return expected_value

    def value_iteration_refinement(self, gamma, epsilon, max_iterations=1000):
        """
        Perform Value Iteration to refine the value function after Policy Iteration.
        """
        iteration = 0
        while iteration < max_iterations:
            delta = 0
            new_value_function = self.value_function.copy()

            # Iterate over all states in the policy
            for state in self.policy.keys():
                if state.turns_left == 0:
                    continue  # Skip terminal states

                actions = self._generate_actions(state)
                if not actions:
                    continue  # No actions available

                max_utility = float('-inf')
                for action in actions:
                    utility = self._compute_expected_value(state, action, gamma)
                    if utility > max_utility:
                        max_utility = utility

                if abs(max_utility - self.value_function[state]) > delta:
                    delta = abs(max_utility - self.value_function[state])
                new_value_function[state] = max_utility

            self.value_function = new_value_function
            print(f"Value Iteration Refinement: Iteration {iteration + 1}, max delta = {delta}")

            if delta < epsilon:
                print("Value Iteration Refinement converged.")
                break

            iteration += 1

            # Time Check: Ensure initialization does not exceed 300 seconds
            if time.time() - self.init_start_time > 290:  # Leave buffer for other operations
                print("Initialization time limit reached during Value Iteration Refinement.")
                break

    def _compute_passable_positions(self):
        """Precompute passable positions on the map."""
        passable = set()
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] != 'I':
                    passable.add((i, j))
        return passable

    @lru_cache(maxsize=None)
    def _find_path(self, start, goal, map_grid):
        """Use BFS to find the shortest path from start to goal."""
        queue = deque()
        queue.append((start, [start]))
        visited = set()
        visited.add(start)
        while queue:
            current, path = queue.popleft()
            if current == goal:
                return path
            neighbors = self._get_neighbors(current, map_grid)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def _get_neighbors(self, position, map_grid):
        """Get passable neighbors for a given position."""
        x, y = position
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < len(map_grid)) and (0 <= new_y < len(map_grid[0])):
                if map_grid[new_x][new_y] != 'I':
                    neighbors.append((new_x, new_y))
        return neighbors

