# # ex3.py
#
# import itertools
# from collections import deque, defaultdict
# from typing import Tuple, Dict, List
# from functools import lru_cache
#
# # Define your IDs
# ids = ['207476763']  # Replace with your actual ID(s)
#
# # Define a hashable State class
# class State:
#     """
#     State represents the current configuration of the game, including wizard positions,
#     horcrux states, death eater indices, and remaining turns.
#     """
#
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
#     An optimal wizard agent that uses Finite-Horizon Value Iteration to compute the optimal policy.
#     """
#
#     def __init__(self, initial):
#         """
#         Initialize the agent with the initial state.
#         Perform Finite-Horizon Value Iteration to compute the optimal policy.
#         """
#         # 1. Creating the map and initializing wizards, horcruxes, death eaters, and passable positions
#         self.map = tuple(tuple(row) for row in initial['map'])
#         self.wizards_initial = initial['wizards']
#         self.horcruxes_initial = initial['horcrux']
#         self.death_eaters_initial = initial['death_eaters']
#         self.turns_to_go = initial['turns_to_go']
#         self.width = len(self.map[0])
#         self.height = len(self.map)
#         self.passable = self._compute_passable_positions()
#
#         # 2. Saving death eater paths, horcruxes, and wizards names
#         self.death_eaters_paths = {de_name: de_info['path'] for de_name, de_info in self.death_eaters_initial.items()}
#         self.horcrux_names = list(self.horcruxes_initial.keys())
#         self.wizard_names = list(self.wizards_initial.keys())
#
#         # 3. Saving horcrux movement information
#         self.horcrux_info = {}
#         for h_name, h_info in self.horcruxes_initial.items():
#             self.horcrux_info[h_name] = {
#                 'current_location': tuple(h_info['location']),
#                 'possible_locations': tuple(tuple(loc) for loc in h_info['possible_locations']),
#                 'prob_change_location': h_info['prob_change_location']
#             }
#
#         # 4. Saving death eater movement probabilities
#         self.death_eater_info = {}
#         for de_name, de_info in self.death_eaters_initial.items():
#             self.death_eater_info[de_name] = {
#                 'path': de_info['path'],
#                 'current_index': de_info['index']  # Ensure consistent key naming
#             }
#
#         # 5. Initializing the initial state
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
#         # 6. Initialize Value Function and Policy
#         self.value_prev = defaultdict(float)  # V_prev[s]
#         self.value_current = defaultdict(float)  # V_current[s]
#         self.policy = {}  # policy[s]
#
#         # 7. Perform Finite-Horizon Value Iteration using Backward Induction
#         self.value_iteration()
#
#     def act(self, state):
#         """
#         Decide the next action based on the current state using the precomputed policy.
#         """
#         # Construct the State object from the current state
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
#         action = self.policy.get(current_state, tuple(("wait", wizard) for wizard in self.wizard_names))
#
#         return action  # Return as a tuple of atomic actions
#
#     def value_iteration(self):
#         """
#         Perform Finite-Horizon Value Iteration to compute the optimal value function and policy.
#         Uses Backward Induction from t=1 to t=T.
#         """
#         gamma = 1  # No discount factor for finite-horizon
#
#         # Iterate backward from t=1 to t=T (turns_to_go)
#         for t in range(1, self.turns_to_go + 1):
#             self.value_current = defaultdict(float)
#             policy_current = {}
#
#             print(f"Starting Value Iteration for turn {t}")
#
#             # Enumerate all possible states with turns_left = t
#             # Note: For large t and state spaces, this is computationally intensive
#
#             # Enumerate wizard positions
#             wizard_positions_options = []
#             for wizard_name in self.wizard_names:
#                 # Wizards can be in any passable position
#                 wizard_pos_options = list(self.passable)
#                 wizard_positions_options.append([(wizard_name, pos) for pos in wizard_pos_options])
#
#             # Enumerate horcrux states
#             horcrux_states_options = []
#             for h_name in self.horcrux_names:
#                 h_info = self.horcrux_info[h_name]
#                 # Each horcrux can be in one of its possible new locations or stay, or be destroyed
#                 possible_new_locs = h_info['possible_locations']
#                 prob_change = h_info['prob_change_location']
#                 num_new_locs = len(possible_new_locs)
#
#                 # Exclude current location from possible new locations if not intended
#                 # If current location is included, it represents staying in place
#                 # Based on your description, staying is handled separately
#                 options = [(h_name, loc, True) for loc in possible_new_locs]
#                 options.append((h_name, h_info['current_location'], True))  # Stay in current location
#                 options.append((h_name, (-1, -1), False))  # Destroyed state
#                 horcrux_states_options.append(options)
#
#             # Enumerate death eater indices
#             death_eater_indices_options = []
#             for de_name in self.death_eater_info.keys():
#                 path_length = len(self.death_eater_info[de_name]['path'])
#                 death_eater_indices_options.append([(de_name, idx) for idx in range(path_length)])
#
#             # Cartesian product to generate all possible states with turns_left = t
#             total_states = 0
#             for wizard_pos in itertools.product(*wizard_positions_options):
#                 for horcrux_state in itertools.product(*horcrux_states_options):
#                     for de_idx in itertools.product(*death_eater_indices_options):
#                         state = State(
#                             wizard_positions=wizard_pos,
#                             horcrux_states=horcrux_state,
#                             death_eater_indices=de_idx,
#                             turns_left=t
#                         )
#
#                         total_states += 1
#
#                         # Generate possible actions
#                         actions = self._generate_actions(state)
#
#                         if not actions:
#                             # If no actions, assign minimal utility
#                             best_action = tuple(("wait", wizard) for wizard in self.wizard_names)
#                             best_utility = 0
#                             if best_utility > self.value_current[state]:
#                                 self.value_current[state] = best_utility
#                                 policy_current[state] = best_action
#                             continue
#
#                         # Find the best action
#                         best_utility = float('-inf')
#                         best_action = None
#
#                         for action in actions:
#                             expected_value = self._compute_expected_value(state, action, t, gamma)
#                             if expected_value > best_utility:
#                                 best_utility = expected_value
#                                 best_action = action
#
#                         # Update value and policy
#                         if best_action:
#                             if best_utility > self.value_current[state]:
#                                 self.value_current[state] = best_utility
#                                 policy_current[state] = best_action
#
#             # Update V_prev and policy
#             self.value_prev = self.value_current
#             self.policy.update(policy_current)
#
#             print(f"Completed Value Iteration for turn {t} with {total_states} states processed.")
#
#         print("Value Iteration completed.")
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
#     def _compute_expected_value(self, state, action, t, gamma=1):
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
#                 reward += 2  # 2 points for destroying a horcrux
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
#                 # Ensure unique moves
#                 moves = list(set(moves))
#                 prob = 1.0 / len(moves)
#                 possible_indices = moves
#                 probabilities = [prob] * len(moves)
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
#         all_de_moves = itertools.product(*death_eater_move_lists)
#
#         # Generate all possible combinations of horcrux movements
#         horcrux_names = list(horcrux_movements.keys())
#         horcrux_move_lists = [horcrux_movements[name] for name in horcrux_names]
#         all_h_moves = itertools.product(*horcrux_move_lists)
#
#         expected_value = 0
#
#         # Enumerate all combinations of death eater moves and horcrux moves
#         for de_move in all_de_moves:
#             # Calculate death eater movement probability
#             de_prob = 1.0
#             new_death_eater_indices = []
#             death_eater_positions = []
#             for de_name, (de_new_index, de_move_prob) in zip(death_eater_names, de_move):
#                 de_prob *= de_move_prob
#                 new_death_eater_indices.append((de_name, de_new_index))
#                 # Get new position
#                 path = self.death_eaters_paths[de_name]
#                 de_pos = path[de_new_index]
#                 death_eater_positions.append(de_pos)
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
#                 death_eater_set = set(death_eater_positions)
#                 for wiz_name, wiz_pos in wizard_locations.items():
#                     if wiz_pos in death_eater_set:
#                         penalty -= 1  # -1 point per encounter
#
#                 # Compute immediate reward and penalty
#                 immediate_reward = reward + penalty
#
#                 # Create new state
#                 new_state = State(
#                     wizard_positions=tuple(new_wizard_positions),
#                     horcrux_states=tuple(new_horcrux_states_updated),
#                     death_eater_indices=tuple(new_death_eater_indices),
#                     turns_left=state.turns_left - 1
#                 )
#
#                 # Get future value from V_prev
#                 future_value = self.value_prev.get(new_state, 0)
#
#                 # Accumulate expected value
#                 expected_value += total_prob * (immediate_reward + gamma * future_value)
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
#             # If no horcruxes left, wait
#             if not any(info['location'] != (-1, -1) for info in state['horcrux'].values()):
#                 # Optionally, implement actions to terminate the game
#                 # Here, we'll wait
#                 actions.append(("wait", wizard_name))
#                 continue
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
#
#
# def create_harrypotter_problem(game):
#     if game['optimal']:
#         return OptimalWizardAgent(game)
#     else:
#         return WizardAgent(game)


# ex3.py

import itertools
import heapq
from collections import deque, defaultdict
from functools import lru_cache

# Replace with your actual ID(s)
ids = ['207476763']


##########################################
# STATE REPRESENTATION
##########################################

class State:
    """
    A hashable state that contains:
      - wizard_positions: tuple of (wizard_name, (x, y))
      - horcrux_states: tuple of (horcrux_name, (x, y), exists)
      - death_eater_indices: tuple of (death_eater_name, path_index)
      - turns_left: integer
    """

    def __init__(self, wizard_positions, horcrux_states, death_eater_indices, turns_left):
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
        return (f"State(wizards={self.wizard_positions}, "
                f"horcruxes={self.horcrux_states}, "
                f"death_eaters={self.death_eater_indices}, "
                f"turns_left={self.turns_left})")


##########################################
# IMPROVED OPTIMAL WIZARD AGENT
##########################################

class OptimalWizardAgent:
    """
    This agent computes the optimal finite–horizon policy using full Bellman backups with
    expectation over stochastic transitions. It uses recursion with memoization over
    reachable states.
    """

    def __init__(self, initial):
        # Read the map and time information.
        self.map = tuple(tuple(row) for row in initial['map'])
        self.turns_to_go = initial['turns_to_go']
        self.width = len(self.map[0])
        self.height = len(self.map)

        # Get wizard, horcrux, and death eater info.
        self.wizards_initial = initial['wizards']  # e.g. { 'Harry Potter': {"location": (2,0)} }
        self.horcruxes_initial = initial[
            'horcrux']  # e.g. { 'Nagini': {"location": (0,3), "possible_locations": [(0,3), (1,3), (2,2)], "prob_change_location": 0.9} }
        self.death_eaters_initial = initial[
            'death_eaters']  # e.g. { 'Lucius Malfoy': {"index": 0, "path": [(1,1), (1,0)]} }

        # Compute passable positions.
        self.passable = self._compute_passable_positions()

        # Save death eater paths.
        self.death_eaters_paths = {name: info['path'] for name, info in self.death_eaters_initial.items()}

        # Build the initial state.
        initial_wizard_positions = tuple((name, tuple(info['location'])) for name, info in self.wizards_initial.items())
        initial_horcrux_states = tuple(
            (name, tuple(info['location']), True) for name, info in self.horcruxes_initial.items())
        initial_death_eater_indices = tuple((name, info['index']) for name, info in self.death_eaters_initial.items())
        self.initial_state = State(initial_wizard_positions, initial_horcrux_states, initial_death_eater_indices,
                                   self.turns_to_go)

        # Dictionaries for memoization and storing the optimal policy.
        # Keys are (state, remaining_turns).
        self.memo = {}
        self.optimal_policy = {}

        # Compute the optimal value from the initial state.
        self._compute_optimal_value(self.initial_state, self.turns_to_go)

    def act(self, state_dict):
        """
        Given the current environment state as a dictionary, convert it to our State object and
        return the pre–computed optimal joint action.
        """
        wizard_positions = tuple((name, tuple(info['location'])) for name, info in state_dict['wizards'].items())
        horcrux_states = tuple((name, tuple(info['location']), info['location'] != (-1, -1))
                               for name, info in state_dict['horcrux'].items())
        death_eater_indices = tuple((name, info['index']) for name, info in state_dict['death_eaters'].items())
        turns_left = state_dict['turns_to_go']
        current_state = State(wizard_positions, horcrux_states, death_eater_indices, turns_left)
        key = (current_state, turns_left)
        if key in self.optimal_policy:
            return self.optimal_policy[key]
        else:
            # Fallback: if the state was not reached during offline computation.
            wizard_names = list(state_dict['wizards'].keys())
            return tuple(("wait", name) for name in wizard_names)

    def _compute_optimal_value(self, state, t):
        """
        Recursively compute the optimal expected value for a given state and horizon t.
        Also record the optimal joint action in self.optimal_policy.
        """
        if t == 0:
            return 0  # No reward if no turns remain.
        key = (state, t)
        if key in self.memo:
            return self.memo[key]
        best_val = float('-inf')
        best_action = None
        for action in self._generate_actions(state):
            exp_val = 0
            # Sum over all stochastic outcomes.
            for (p, next_state, immediate) in self._get_transitions(state, action):
                exp_val += p * (immediate + self._compute_optimal_value(next_state, t - 1))
            if exp_val > best_val:
                best_val = exp_val
                best_action = action
        self.memo[key] = best_val
        self.optimal_policy[key] = best_action
        return best_val

    def _generate_actions(self, state):
        """
        Generate all possible joint actions from the given state.
        For each wizard, generate atomic actions:
          - "destroy" (if the wizard is on an active horcrux),
          - "move" in each of the four cardinal directions (if the resulting cell is passable),
          - "wait"
        Then take the Cartesian product.
        """
        actions_per_wizard = []
        for wiz_name, pos in state.wizard_positions:
            wizard_actions = []
            # Check for horcrux destruction.
            for (h_name, h_pos, exists) in state.horcrux_states:
                if exists and pos == h_pos:
                    wizard_actions.append(("destroy", wiz_name, h_name))
            # Movement actions.
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (pos[0] + dx, pos[1] + dy)
                if new_pos in self.passable:
                    wizard_actions.append(("move", wiz_name, new_pos))
            wizard_actions.append(("wait", wiz_name))
            actions_per_wizard.append(wizard_actions)
        return list(itertools.product(*actions_per_wizard))

    def _get_transitions(self, state, action):
        """
        Given a state and a joint action, return a list of (p, new_state, immediate_reward)
        tuples representing possible outcomes.

        The wizard actions are applied deterministically; then we enumerate the stochastic moves of
        the death eaters and horcruxes.
        """
        # 1. Apply wizard actions.
        new_wizard_positions = list(state.wizard_positions)
        new_horcrux_states = list(state.horcrux_states)
        immediate_reward = 0
        for atomic in action:
            act_type = atomic[0]
            wiz_name = atomic[1]
            if act_type == "destroy":
                horcrux_name = atomic[2]
                immediate_reward += 2  # Reward for destroying a horcrux.
                for i, (h_name, h_pos, exists) in enumerate(new_horcrux_states):
                    if h_name == horcrux_name and exists:
                        new_horcrux_states[i] = (h_name, h_pos, False)
                        break
            elif act_type == "move":
                new_pos = atomic[2]
                for i, (w_name, pos) in enumerate(new_wizard_positions):
                    if w_name == wiz_name:
                        new_wizard_positions[i] = (w_name, new_pos)
                        break
            # "wait" leaves the wizard’s position unchanged.

        # 2. Enumerate death eater outcomes.
        de_outcomes_list = []
        for de_name, index in state.death_eater_indices:
            path = self.death_eaters_paths[de_name]
            L = len(path)
            outcomes = []
            if L == 1:
                outcomes.append((index, 1.0))
            else:
                if index == 0:
                    outcomes.append((0, 0.5))
                    outcomes.append((1, 0.5))
                elif index == L - 1:
                    outcomes.append((L - 1, 0.5))
                    outcomes.append((L - 2, 0.5))
                else:
                    outcomes.append((index - 1, 1 / 3))
                    outcomes.append((index, 1 / 3))
                    outcomes.append((index + 1, 1 / 3))
            de_outcomes_list.append((de_name, outcomes))

        # 3. Enumerate horcrux outcomes.
        hor_outcomes_list = []
        for (h_name, pos, exists) in new_horcrux_states:
            if not exists:
                hor_outcomes_list.append((h_name, [(pos, 1.0, False)]))
            else:
                p_change = self.horcruxes_initial[h_name]['prob_change_location']
                outcomes = []
                outcomes.append((pos, 1 - p_change, True))
                possible = self.horcruxes_initial[h_name]['possible_locations']
                num_possible = len(possible)
                if num_possible > 0:
                    for loc in possible:
                        outcomes.append((tuple(loc), p_change / num_possible, True))
                hor_outcomes_list.append((h_name, outcomes))

        # 4. Combine outcomes over death eaters and horcruxes.
        de_combos = list(itertools.product(*[outs for (_, outs) in de_outcomes_list]))
        hor_combos = list(itertools.product(*[outs for (_, outs) in hor_outcomes_list]))
        transitions = []
        for de_combo in de_combos:
            prob_de = 1.0
            new_de_indices = []
            de_positions = []
            for idx, (new_index, p) in enumerate(de_combo):
                prob_de *= p
                de_name = de_outcomes_list[idx][0]
                new_de_indices.append((de_name, new_index))
                de_positions.append(tuple(self.death_eaters_paths[de_name][new_index]))
            for hor_combo in hor_combos:
                prob_hor = 1.0
                new_hor_states = []
                for idx, (new_pos, p, exists) in enumerate(hor_combo):
                    prob_hor *= p
                    h_name = hor_outcomes_list[idx][0]
                    new_hor_states.append((h_name, new_pos, exists))
                total_prob = prob_de * prob_hor
                # Compute penalty: for each wizard that ends up in the same cell as a death eater, subtract 1.
                penalty = 0
                for w_name, w_pos in new_wizard_positions:
                    if w_pos in de_positions:
                        penalty -= 1
                total_immediate = immediate_reward + penalty
                new_state = State(tuple(new_wizard_positions), tuple(new_hor_states), tuple(new_de_indices),
                                  state.turns_left - 1)
                transitions.append((total_prob, new_state, total_immediate))
        return transitions

    def _compute_passable_positions(self):
        """Return a set of coordinates (i,j) for cells that are passable (i.e. 'P' or 'V')."""
        passable = set()
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] in ("P", "V"):
                    passable.add((i, j))
        return passable


##########################################
# IMPROVED HEURISTIC WIZARD AGENT
##########################################

class WizardAgent:
    """
    A heuristic–based wizard agent that uses a complex, multi–criteria evaluation.

    For each wizard, the agent computes a safe path toward an active horcrux using an enhanced
    Dijkstra–like search that adds extra cost for:
      • Each step (base cost 1)
      • Cells adjacent (in 8 directions) to death eaters (×3 cost per adjacent death eater)
      • Revisited cells (extra penalty of 2)

    It then scores candidate targets by combining:
      • A fixed bonus (10) for eventually destroying a horcrux,
      • Minus the safe–path cost,
      • Minus twice the cumulative risk (number of adjacent death eater counts along the path),
      • Plus a bonus proportional to the wizard’s current safety (Manhattan distance to the closest death eater).

    If no candidate safe path is found, the agent falls back on moving to the neighbor that maximizes safety.
    """

    def __init__(self, initial):
        self.map = tuple(tuple(row) for row in initial['map'])
        self.wizards = initial['wizards']
        self.horcruxes = initial['horcrux']
        self.death_eaters = initial['death_eaters']
        self.turns_to_go = initial['turns_to_go']
        self.width = len(self.map[0])
        self.height = len(self.map)
        self.passable = self._compute_passable_positions()
        # For each wizard, track visited positions to discourage loops.
        self.visited = {wiz: set() for wiz in self.wizards.keys()}

    def act(self, state):
        actions = []
        de_positions = self._get_death_eater_positions(state)
        # Build active horcrux dictionary (ignore those destroyed, assumed to have location (-1,-1))
        active_horcruxes = {}
        for h_name, info in state['horcrux'].items():
            if info['location'] != (-1, -1):
                active_horcruxes[h_name] = info
        for wiz_name, wiz_info in state['wizards'].items():
            current_pos = tuple(wiz_info['location'])
            # If wizard is on a horcrux, destroy it.
            target_here = self._horcrux_at_position(current_pos, active_horcruxes)
            if target_here is not None:
                actions.append(("destroy", wiz_name, target_here))
                continue
            best_score = float('-inf')
            best_path = None
            best_target = None
            safety = self._safety_distance(current_pos, de_positions)
            # Evaluate each active horcrux as a target.
            for h_name, h_info in active_horcruxes.items():
                target_pos = tuple(h_info['location'])
                path, cost, risk = self._find_safe_path_enhanced(current_pos, target_pos, state, wiz_name)
                if path is None:
                    continue
                # Combine criteria: bonus for destroy (10), minus cost, minus 2×risk, plus 0.5×current safety.
                score = 10 - cost - 2 * risk + 0.5 * safety
                if score > best_score:
                    best_score = score
                    best_path = path
                    best_target = h_name
            if best_path is not None and len(best_path) > 1:
                next_step = best_path[1]
                actions.append(("move", wiz_name, next_step))
                self.visited[wiz_name].add(next_step)
            elif best_path is not None and len(best_path) == 1:
                actions.append(("destroy", wiz_name, best_target))
            else:
                fallback = self._choose_safer_move(current_pos, state)
                if fallback is not None:
                    actions.append(("move", wiz_name, fallback))
                else:
                    actions.append(("wait", wiz_name))
        return tuple(actions)

    #####################
    # Helper Methods
    #####################

    def _compute_passable_positions(self):
        passable = set()
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] in ("P", "V"):
                    passable.add((i, j))
        return passable

    def _is_passable(self, pos, state):
        x, y = pos
        if 0 <= x < self.height and 0 <= y < self.width:
            return state['map'][x][y] in ("P", "V")
        return False

    def _get_neighbors(self, pos, state):
        x, y = pos
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (x + dx, y + dy)
            if self._is_passable(new_pos, state):
                neighbors.append(new_pos)
        return neighbors

    def _get_death_eater_positions(self, state):
        positions = []
        for de in state['death_eaters'].values():
            idx = de['index']
            path = de['path']
            positions.append(tuple(path[idx]))
        return positions

    def _count_adjacent_de(self, pos, de_positions):
        count = 0
        x, y = pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if (x + dx, y + dy) in de_positions:
                    count += 1
        return count

    def _find_safe_path_enhanced(self, start, goal, state, wiz_name):
        """
        Enhanced safe–path search (Dijkstra–like). The cost for a step is:
          cost = 1 + 3*(# adjacent death eaters) + (2 if the cell was visited before by this wizard)
        Returns a tuple (path, total_cost, cumulative_risk). If no path is found, returns (None, inf, inf).
        """
        de_positions = self._get_death_eater_positions(state)
        frontier = []
        heapq.heappush(frontier, (0, 0, [start]))  # (cumulative cost, cumulative risk, path)
        visited = {}
        while frontier:
            cost, risk, path = heapq.heappop(frontier)
            current = path[-1]
            if current == goal:
                return path, cost, risk
            if current in visited and visited[current] <= cost:
                continue
            visited[current] = cost
            for neighbor in self._get_neighbors(current, state):
                if not self._is_passable(neighbor, state):
                    continue
                add_cost = 1
                add_risk = self._count_adjacent_de(neighbor, de_positions)
                add_cost += 3 * add_risk
                if neighbor in self.visited[wiz_name]:
                    add_cost += 2  # discourage cycles
                new_cost = cost + add_cost
                new_risk = risk + add_risk
                new_path = path + [neighbor]
                heapq.heappush(frontier, (new_cost, new_risk, new_path))
        return None, float('inf'), float('inf')

    def _horcrux_at_position(self, pos, horcruxes):
        for h_name, h_info in horcruxes.items():
            if tuple(h_info['location']) == pos:
                return h_name
        return None

    def _safety_distance(self, pos, de_positions):
        if not de_positions:
            return 10
        return min(abs(pos[0] - de[0]) + abs(pos[1] - de[1]) for de in de_positions)

    def _choose_safer_move(self, pos, state):
        de_positions = self._get_death_eater_positions(state)
        neighbors = self._get_neighbors(pos, state)
        safe_neighbors = [n for n in neighbors if self._count_adjacent_de(n, de_positions) == 0]
        if safe_neighbors:
            return max(safe_neighbors, key=lambda n: self._safety_distance(n, de_positions))
        if neighbors:
            return min(neighbors, key=lambda n: self._count_adjacent_de(n, de_positions))
        return None

##########################################
# END OF FILE
##########################################
