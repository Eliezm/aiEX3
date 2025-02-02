import itertools
from utils import PriorityQueue, argmin_random_tie, argmax_random_tie, shuffled, clip

ids = ['207476763']


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


class OptimalWizardAgent:
    def __init__(self, initial):
        self.map = tuple(tuple(row) for row in initial['map'])
        self.turns_to_go = initial['turns_to_go']
        self.width = len(self.map[0])
        self.height = len(self.map)

        self.wizards_initial = initial['wizards']
        self.horcruxes_initial = initial['horcrux']
        self.death_eaters_initial = initial['death_eaters']

        # Compute passable positions.
        self.passable = self._compute_passable_positions()

        self.death_eaters_paths = {name: info['path'] for name, info in self.death_eaters_initial.items()}

        initial_wizard_positions = tuple((name, tuple(info['location'])) for name, info in self.wizards_initial.items())
        initial_horcrux_states = tuple(
            (name, tuple(info['location']), True) for name, info in self.horcruxes_initial.items())
        initial_death_eater_indices = tuple((name, info['index']) for name, info in self.death_eaters_initial.items())
        self.initial_state = State(initial_wizard_positions, initial_horcrux_states, initial_death_eater_indices,
                                   self.turns_to_go)

        self.memo = {}
        self.optimal_policy = {}

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
            wizard_names = list(state_dict['wizards'].keys())
            return tuple(("wait", name) for name in wizard_names)

    def _compute_optimal_value(self, state, t):
        """
        Recursively compute the optimal expected value for a given state and horizon t.
        Also record the optimal joint action in self.optimal_policy.
        """
        if t == 0:
            return 0
        key = (state, t)
        if key in self.memo:
            return self.memo[key]
        best_val = float('-inf')
        best_action = None
        for action in self._generate_actions(state):
            exp_val = 0
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
# HEURISTIC WIZARD AGENT
##########################################

class WizardAgent:

    def __init__(self, initial):
        self.map = tuple(tuple(row) for row in initial['map'])
        self.wizards = initial['wizards']
        self.horcruxes = initial['horcrux']
        self.death_eaters = initial['death_eaters']
        self.turns_to_go = initial['turns_to_go']
        self.width = len(self.map[0])
        self.height = len(self.map)
        self.passable = self._compute_passable_positions()
        self.visited = {wiz: set() for wiz in self.wizards.keys()}

    def act(self, state):
        actions = []
        de_positions = self._get_death_eater_positions(state)
        active_horcruxes = {}
        for h_name, info in state['horcrux'].items():
            if info['location'] != (-1, -1):
                active_horcruxes[h_name] = info
        for wiz_name, wiz_info in state['wizards'].items():
            current_pos = tuple(wiz_info['location'])
            target_here = self._horcrux_at_position(current_pos, active_horcruxes)
            if target_here is not None:
                actions.append(("destroy", wiz_name, target_here))
                continue
            best_score = float('-inf')
            best_path = None
            best_target = None
            safety = self._safety_distance(current_pos, de_positions)
            for h_name, h_info in active_horcruxes.items():
                target_pos = tuple(h_info['location'])
                path, cost, risk = self._find_safe_path_enhanced(current_pos, target_pos, state, wiz_name)
                if path is None:
                    continue
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
        de_positions = self._get_death_eater_positions(state)
        # The frontier items are tuples: (cumulative_cost, cumulative_risk, path)
        frontier = PriorityQueue(order=min, f=lambda x: x[0])
        frontier.append((0, 0, [start]))
        visited = {}
        while len(frontier) > 0:
            cost, risk, path = frontier.pop()
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
                frontier.append((new_cost, new_risk, new_path))
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
        """
        Fallback method: choose a neighbor that is safest.
        Here we use the random–tie–breaking helpers from utils.
        """
        de_positions = self._get_death_eater_positions(state)
        neighbors = self._get_neighbors(pos, state)
        safe_neighbors = [n for n in neighbors if self._count_adjacent_de(n, de_positions) == 0]
        if safe_neighbors:
            return argmax_random_tie(safe_neighbors, key=lambda n: self._safety_distance(n, de_positions))
        if neighbors:
            return argmin_random_tie(neighbors, key=lambda n: self._count_adjacent_de(n, de_positions))
        return None


