# ex3.py

import itertools
from collections import deque, defaultdict
from typing import Tuple, Dict, List
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
    An optimal wizard agent that uses Finite-Horizon Value Iteration to compute the optimal policy.
    """

    def __init__(self, initial):
        """
        Initialize the agent with the initial state.
        Perform Finite-Horizon Value Iteration to compute the optimal policy.
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
            (name, info['current_index']) for name, info in self.death_eater_info.items()
        )

        self.initial_state = State(
            wizard_positions=initial_wizard_positions,
            horcrux_states=initial_horcrux_states,
            death_eater_indices=initial_death_eater_indices,
            turns_left=self.turns_to_go
        )

        # Initialize Value Function and Policy
        # We will use two dictionaries: V_prev and V_current
        # V_prev holds values for t-1
        # V_current holds values for t

        self.value_prev = defaultdict(float)  # V_prev[s]
        self.value_current = defaultdict(float)  # V_current[s]
        self.policy = {}  # policy[s]

        # Perform Finite-Horizon Value Iteration
        self.value_iteration()

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
        action = self.policy.get(current_state, tuple(("wait", wizard) for wizard in self.wizard_names))

        return action  # Return as a tuple of atomic actions

    def value_iteration(self):
        """
        Perform Finite-Horizon Value Iteration to compute the optimal value function and policy.
        """
        gamma = 1  # No discount factor for finite-horizon

        # Iterate from t=1 to t=T (turns_to_go)
        for t in range(1, self.turns_to_go + 1):

            self.value_current = defaultdict(float)
            policy_current = {}

            # Enumerate all possible states with turns_left = t

            # Enumerate wizard positions
            wizard_positions_options = []

            for wizard_name, wizard_info in self.wizards_initial.items():
                # Assuming each wizard can be in any passable position
                # Note: In a more optimized version, you should consider wizard positions based on previous states
                wizard_pos_options = list(self.passable)
                wizard_positions_options.append([(wizard_name, pos) for pos in wizard_pos_options])

            # Enumerate horcrux states
            horcrux_states_options = []

            for h_name, h_info in self.horcrux_info.items():
                # Each horcrux can be in one of its possible locations or destroyed
                options = [(h_name, loc, True) for loc in [h_info['current_location']] + list(h_info['possible_locations'])]
                options.append((h_name, (-1, -1), False))  # Destroyed state
                horcrux_states_options.append(options)

            # Enumerate death eater indices
            death_eater_indices_options = []

            for de_name, de_info in self.death_eater_info.items():
                path_length = len(de_info['path'])
                death_eater_indices_options.append([(de_name, idx) for idx in range(path_length)])

            # Possible turns_left = t
            turns_left = t

            # Cartesian product to generate all possible states with turns_left = t
            for wizard_pos in itertools.product(*wizard_positions_options):
                for horcrux_state in itertools.product(*horcrux_states_options):
                    for de_idx in itertools.product(*death_eater_indices_options):
                        state = State(
                            wizard_positions=wizard_pos,
                            horcrux_states=horcrux_state,
                            death_eater_indices=de_idx,
                            turns_left=turns_left
                        )

                        # Generate possible actions
                        actions = self._generate_actions(state)

                        if not actions:
                            # Assign minimal utility
                            best_action = tuple(("wait", wizard) for wizard in self.wizard_names)
                            best_utility = 0
                            if best_utility > self.value_current[state]:
                                self.value_current[state] = best_utility
                                policy_current[state] = best_action
                            continue

                        # Find the best action
                        best_utility = float('-inf')
                        best_action = None

                        for action in actions:
                            expected_value = self._compute_expected_value(state, action, t)
                            if expected_value > best_utility:
                                best_utility = expected_value
                                best_action = action

                        # Update value and policy
                        if best_action:
                            if best_utility > self.value_current[state]:
                                self.value_current[state] = best_utility
                                policy_current[state] = best_action

            # Update V_prev and policy
            self.value_prev = self.value_current
            self.policy.update(policy_current)

            print(f"Completed value iteration for turn {t}")

        print("Value Iteration completed.")

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

    def _compute_expected_value(self, state, action, t, gamma=1):
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
                reward += 2  # 2 points for destroying a horcrux
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
                # Ensure unique moves
                moves = list(set(moves))
                prob = 1.0 / len(moves)
                possible_indices = moves
                probabilities = [prob] * len(moves)

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
            death_eater_positions = []
            for de_name, (de_new_index, de_move_prob) in zip(death_eater_names, de_move):
                de_prob *= de_move_prob
                new_death_eater_indices.append((de_name, de_new_index))
                # Get new position
                path = self.death_eaters_paths[de_name]
                de_pos = path[de_new_index]
                death_eater_positions.append(de_pos)

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
                death_eater_set = set(death_eater_positions)
                for wiz_name, wiz_pos in wizard_locations.items():
                    if wiz_pos in death_eater_set:
                        penalty -= 1  # -1 point per encounter

                # Compute immediate reward and penalty
                immediate_reward = reward + penalty

                # Create new state
                new_state = State(
                    wizard_positions=tuple(new_wizard_positions),
                    horcrux_states=tuple(new_horcrux_states_updated),
                    death_eater_indices=tuple(new_death_eater_indices),
                    turns_left=state.turns_left - 1
                )

                # Get future value from V_prev
                future_value = self.value_prev.get(new_state, 0)

                # Accumulate expected value
                expected_value += total_prob * (immediate_reward + gamma * future_value)

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

            # If no horcruxes left, wait
            if not any(info['location'] != (-1, -1) for info in state['horcrux'].values()):
                # Optionally, implement actions to terminate the game
                # Here, we'll wait
                actions.append(("wait", wizard_name))
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


def create_harrypotter_problem(game):
    if game['optimal']:
        return OptimalWizardAgent(game)
    else:
        return WizardAgent(game)
