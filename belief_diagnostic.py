"""How close are sampled opponent hands to the truth? Uniform vs modeled.

Plays later-round games (Python engine, so true hands are visible to this
harness only). At every ExperiencedAgent4 search decision it draws worlds from
both samplers for the same position and compares them with the real hands:

* ``l1``: mean over opponents of sum_rank |sampled count - true count| (lower is better);
* ``marg``: same, using the mean sampled count per rank (the belief's marginal);
* ``hit``: share of (opponent, rank) cells where sampled count == true count.

    .venv/Scripts/python.exe belief_diagnostic.py --games 40 --worlds 32
"""

import argparse
from collections import defaultdict
import random
import time

from src.agent import ExperiencedAgent3Tuned, ExperiencedAgent4, HumanRulesAgent3
from src.agent_belief import BeliefSampler, R, _idx
from src.agent_search import prepare, sample_world
from src.president import Round
from src.simulate import PLAYERS, role_assignments

OPPONENTS = {"tuned": ExperiencedAgent3Tuned, "rules": lambda: HumanRulesAgent3(heads_up_solver=True)}


def counts(cards):
    vec = [0] * R
    for card in cards:
        vec[_idx(card.rank)] += 1
    return vec


def score(worlds, truth, opponents):
    l1 = hit = cells = marg = 0.0
    for p in opponents:
        true = counts(truth[p])
        mean = [0.0] * R
        for world in worlds:
            c = counts(world.hands[p])
            l1 += sum(abs(a - b) for a, b in zip(c, true)) / len(worlds)
            hit += sum(a == b for a, b in zip(c, true)) / len(worlds)
            mean = [m + x / len(worlds) for m, x in zip(mean, c)]
        marg += sum(abs(a - b) for a, b in zip(mean, true))
        cells += R
    n = len(opponents)
    return {"l1": l1 / n, "marg": marg / n, "hit": hit / cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--opponent", choices=sorted(OPPONENTS), default="tuned")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--burn", type=float, nargs="+", default=[20.0],
                        help="burn-in steps per free card (several values: sweep on the same decisions)")
    parser.add_argument("--native", action="store_true", help="run the walk in the C search library")
    parser.add_argument("--thin", type=float, nargs="+", default=[0.5],
                        help="steps between worlds per free card")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    native = None
    if args.native:
        from src import agent_search_c
        native = agent_search_c.backend(ExperiencedAgent4(search_backend="c"))
    roles = list(role_assignments(args.games, random.Random(args.seed + 1)))
    totals = defaultdict(lambda: defaultdict(float))
    stages = defaultdict(lambda: defaultdict(float))
    decisions = 0
    belief_ms, steps = defaultdict(float), defaultdict(float)
    for g in range(args.games):
        round_ = Round(PLAYERS, first_round=False, roles=roles[g], rng=random.Random(rng.getrandbits(64)))
        seat = PLAYERS[g % 4]
        clients = {p: ExperiencedAgent4(random.Random(g), search_worlds=0) if p == seat
                   else OPPONENTS[args.opponent]() for p in PLAYERS}
        while not round_.done:
            actor = round_._actor()
            message = round_.message_for(actor)
            if (actor == seat and message["request"]["type"] in {"play", "completion"}
                    and len(message["active_players"]) > 2):
                prepared = prepare(clients[seat], message)
                opponents = [p for p in message["active_players"] if p != seat]
                if prepared is not None:
                    draw = random.Random(decisions)
                    uniform = [sample_world(prepared, seat, draw) for _ in range(args.worlds)]
                    samplers = {"uniform": uniform}
                    for burn in args.burn:
                        for thin in args.thin:
                            start = time.perf_counter()
                            belief = BeliefSampler(prepared, seat, message, random.Random(decisions),
                                                   noise=args.noise, returned=clients[seat].returned_cards(message),
                                                   burn_in_per_card=burn, thin_per_card=thin,
                                                   native=native)
                            samplers[(burn, thin)] = [belief.sample() for _ in range(args.worlds)]
                            belief_ms[(burn, thin)] += (time.perf_counter() - start) * 1000
                            steps[(burn, thin)] += belief.steps
                    stage = "early" if len(message["events"]) < 40 else "mid" if len(message["events"]) < 80 else "late"
                    for name, worlds in samplers.items():
                        for k, v in score(worlds, round_.hands, opponents).items():
                            totals[name][k] += v
                            stages[(stage, name)][k] += v
                    stages[(stage, "n")]["n"] += 1
                    decisions += 1
            round_.apply_action(actor, clients[actor].respond(message))
    print(f"{decisions} decisions, {args.worlds} worlds each, opponents={args.opponent}, noise={args.noise}")
    names = ["uniform"] + [(b, t) for b in args.burn for t in args.thin]
    print(f"{'sampler':22}{'l1':>8}{'marg':>8}{'hit':>8}{'steps':>8}{'ms':>8}   early l1  mid l1")
    for name in names:
        t = totals[name]
        label = name if name == "uniform" else f"burn {name[0]:g}/card thin {name[1]:g}/card"
        extra = "" if name == "uniform" else f"{steps[name] / decisions:8.0f}{belief_ms[name] / decisions:8.1f}"
        early = [stages[(s, name)]["l1"] / stages[(s, "n")]["n"] for s in ("early", "mid") if stages[(s, "n")]["n"]]
        print(f"{label:22}{t['l1'] / decisions:8.3f}{t['marg'] / decisions:8.3f}{t['hit'] / decisions:8.3f}"
              f"{extra or ' ' * 16}   " + "  ".join(f"{x:7.3f}" for x in early))

if __name__ == "__main__":
    main()
