"""Define random policy."""

from typing import Any, Dict, Tuple

from agent.abstract_agent import AbstractAgent

STATE = Tuple[int, int]


class RandomPolicy(AbstractAgent):
    """Define random policy for grid-world."""

    def get_action(self, state: STATE, epsilon: float = 0.0) -> int:
        """Get random action."""
        return self.action_space.sample()

    def update_policy(
        self, update_info: Dict[str, Any]
    ) -> None:  # pylint: disable=unused-argument
        """Pass because it is random policy."""
        pass

    def print_results(self) -> None:
        """Print results like policy, values.

        Notes:
            - Use methods below
                - print_policy
                - print_state_value
                - print_action_value
        """
        pass
