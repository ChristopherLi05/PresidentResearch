"""Named ExperiencedAgent4 search configurations, importable as module:ClassName.

tune_search.py builds variants on the fly; arena/compare_agent need a real
class, so promising configurations are pinned here.
"""

from src.agent import ExperiencedAgent4


class ExperiencedAgent4Full64M2(ExperiencedAgent4):
    """tune_search variant full64m2: 64 worlds, rollouts played to the end
    (search_depth=None, through own turns), override margin 2, danger=0."""

    NEXT_RESPONSE_PENALTY = 0.0
    NEXT_FINISH_PENALTY = 0.0
    NEXT_COMPLETION_PENALTY = 0.0
    NEXT_BOMB_PENALTY = 0.0
    SEARCH = {"search_worlds": 64, "search_margin": 2.0, "search_depth": None,
              "search_through_own_turns": True}

    def __init__(self, rng=None, **kwargs):
        super().__init__(rng, **{**self.SEARCH, **kwargs})


class ExperiencedAgent4Full64M2Modeled(ExperiencedAgent4Full64M2):
    """full64m2 with modeled opponent hands (search_sampling="modeled"):
    HumanRulesAgent3-likelihood worlds, 25% still dealt uniformly.
    See docs/agents/experienced-agent4.md, "Modeled opponent hands"."""

    SEARCH = {**ExperiencedAgent4Full64M2.SEARCH, "search_sampling": "modeled",
              "search_random_fraction": 0.25}
