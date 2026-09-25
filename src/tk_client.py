"""A playable local Tk client for the President JSON protocol.

Run ``python -m src.tk_client`` from the repository root to play as ``you``
against three :class:`~src.client.BasicClient` opponents. ``HumanClient`` is
also usable with any host that calls :meth:`src.client.Client.respond`.
"""

from __future__ import annotations

from copy import deepcopy
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Mapping, Optional

from src.client import BasicClient, Client
from src.president import Game


SUIT_SYMBOLS = {"C": "♣", "D": "♦", "H": "♥", "S": "♠"}
RED_SUITS = {"D", "H"}


class HumanClient(Client):
    """A blocking Tk implementation of the engine's request/response protocol.

    Each call to :meth:`respond` displays the supplied private state and waits
    until the user picks one of the exact legal actions provided by the engine.
    Tk is initialized lazily, allowing hosts and tests to construct this client
    without opening a window.
    """

    TABLE_COLOR = "#176b3a"
    CARD_WIDTH = 72
    CARD_HEIGHT = 104

    def __init__(self, *, title: str = "President") -> None:
        self.title = title
        self._root: Optional[tk.Tk] = None
        self._canvas: Optional[tk.Canvas] = None
        self._actions: Optional[ttk.Frame] = None
        self._status: Optional[tk.StringVar] = None
        self._message: Optional[Mapping[str, Any]] = None
        self._selected: list[dict[str, str]] = []
        self._answer: Optional[dict[str, Any]] = None
        self._waiting: Optional[tk.BooleanVar] = None
        self._closed = False

    def respond(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """Render a requested state and wait for a legal human choice."""
        request = message.get("request")
        if not request or not request.get("legal_actions"):
            raise ValueError("no legal action requested")
        # The engine deliberately polls every active player during a completion
        # window so it does not reveal who holds matching cards.  A human has
        # no decision to make when decline is the only legal response, so keep
        # that protocol exchange invisible in the local UI.
        actions = request["legal_actions"]
        if (request.get("type") == "completion" and len(actions) == 1
                and actions[0].get("type") == "decline_completion"):
            return deepcopy(actions[0])
        self.observe(message)
        assert self._root is not None
        self._answer = None
        self._waiting = tk.BooleanVar(self._root, value=False)
        self._render()
        self._root.deiconify()
        self._root.lift()
        self._root.wait_variable(self._waiting)
        if self._closed or self._answer is None:
            raise RuntimeError("human client window was closed")
        return deepcopy(self._answer)

    def observe(self, message: Mapping[str, Any]) -> None:
        """Refresh the window without asking the player for an action."""
        self._ensure_window()
        if self._closed or self._root is None:
            raise RuntimeError("human client window was closed")
        self._message = deepcopy(message)
        self._selected = []
        self._render()
        # ``observe`` runs before any seat is waiting for input. Give this
        # independent Tk root one event-loop turn so all four seat windows are
        # mapped immediately, rather than only when their turn first arrives.
        self._root.deiconify()
        self._root.update_idletasks()
        self._root.update()

    def play_again(self, rankings: Mapping[str, int]) -> bool:
        """Show final placements and let the local host start another round."""
        self._ensure_window()
        if self._closed or self._root is None:
            raise RuntimeError("human client window was closed")
        placements = "\n".join(
            f"{place}. {player}" for player, place in sorted(rankings.items(), key=lambda item: item[1])
        )
        return messagebox.askyesno(
            self.title,
            f"Round complete!\n\n{placements}\n\nStart the next round?",
            parent=self._root,
        )

    def _ensure_window(self) -> None:
        if self._root is not None:
            return
        root = tk.Tk()
        root.title(self.title)
        root.geometry("960x700")
        root.minsize(740, 540)
        root.configure(bg="#102418")
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._root = root

        toolbar = ttk.Frame(root, padding=(12, 10))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="You are playing President", font=("TkDefaultFont", 12, "bold")).pack(side="left")
        ttk.Button(toolbar, text="Clear selection", command=self._clear_selection).pack(side="right")

        self._canvas = tk.Canvas(root, bg=self.TABLE_COLOR, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self._canvas.bind("<Configure>", lambda _event: self._draw_table())
        self._actions = ttk.Frame(root, padding=(12, 4))
        self._actions.pack(fill="x")
        self._status = tk.StringVar(root, value="Waiting for the game.")
        ttk.Label(root, textvariable=self._status, anchor="w", padding=(14, 8)).pack(fill="x")

    def _render(self) -> None:
        assert self._message is not None and self._actions is not None and self._status is not None
        request = self._message.get("request")
        top = self._message["top"]
        top_text = "cleared board" if top is None else f"{top['count']} × {top['rank']} (stack {top['stack_count']})"
        if request:
            self._status.set(f"{request['type'].replace('_', ' ').title()}: choose an action. Table: {top_text}")
        else:
            self._status.set(f"Waiting for {self._message.get('turn')}. Table: {top_text}")
        for child in self._actions.winfo_children():
            child.destroy()
        self._draw_table()
        self._draw_action_buttons()

    def _draw_table(self) -> None:
        if self._canvas is None:
            return
        canvas = self._canvas
        canvas.delete("all")
        width, height = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
        canvas.create_oval(55, 35, width - 55, height - 35, outline="#d7b65b", width=3)
        message = self._message
        if message is None:
            return
        canvas.create_text(width // 2, 70, text="PRESIDENT", fill="white", font=("TkDefaultFont", 15, "bold"))
        canvas.create_text(width // 2, 103, text=f"Current response: {message.get('turn')}", fill="#d9f0dd")
        top = message.get("top")
        table_text = "TABLE CLEARED" if top is None else f"TOP: {top['count']} × {top['rank']}"
        canvas.create_text(width // 2, 135, text=table_text, fill="#c8e6c9", font=("TkDefaultFont", 16, "bold"))
        canvas.create_text(width // 2, 158, text="Active: " + ", ".join(message["active_players"]), fill="white")
        all_hands = message.get("all_hands")
        if all_hands:
            history_top = self._draw_omniscient_state(canvas, width, all_hands)
        else:
            history_top = 178
        self._draw_card_history(canvas, width, height, message, history_top)

        hand = message["hand"]
        spacing = min(self.CARD_WIDTH + 8, max(25, (width - 32) // max(len(hand), 1)))
        total = (len(hand) - 1) * spacing + self.CARD_WIDTH
        start_x, y = max(16, (width - total) // 2), height - self.CARD_HEIGHT - 28
        for index, card in enumerate(hand):
            x = start_x + index * spacing
            offset = -14 if card in self._selected else 0
            tag = f"card-{index}"
            canvas.create_rectangle(x, y + offset, x + self.CARD_WIDTH, y + self.CARD_HEIGHT + offset,
                                    fill="#fff9e6" if card in self._selected else "white", outline="#283618", width=2, tags=(tag,))
            color = "#ba1a1a" if card["suit"] in RED_SUITS else "#1c1c1c"
            canvas.create_text(x + 10, y + 14 + offset, text=self._card_label(card), anchor="nw", fill=color,
                               font=("TkDefaultFont", 14, "bold"), tags=(tag,))
            canvas.tag_bind(tag, "<Button-1>", lambda _event, value=card: self._toggle_card(value))

    def _draw_omniscient_state(self, canvas: tk.Canvas, width: int,
                                all_hands: Mapping[str, list[Mapping[str, str]]]) -> int:
        """Draw the intentionally non-private hands/piles used by the debug client."""
        entries = list(all_hands.items())
        split = (len(entries) + 1) // 2
        for index, (owner, cards) in enumerate(entries):
            column, row = (0, index) if index < split else (1, index - split)
            x = 14 if column == 0 else width // 2 + 14
            y = 18 + row * 74
            canvas.create_text(x, y, text=f"{owner} ({len(cards)})", anchor="nw",
                               fill="#e6f4e8", font=("TkDefaultFont", 8, "bold"))
            for card_index, card in enumerate(cards):
                self._draw_card_face(canvas, x + card_index * 21, y + 14, 24, 36, card, font_size=7)
        return max(178, 18 + ((len(entries) + 1) // 2) * 74 + 8)

    def _draw_card_history(self, canvas: tk.Canvas, width: int, height: int,
                           message: Mapping[str, Any], history_top: int) -> None:
        """Draw the recent public card events in the center of the table."""
        history = self.card_history(message)
        if not history:
            canvas.create_text(width // 2, height // 2 + 50, text="No cards played yet",
                               fill="#c8e6c9", font=("TkDefaultFont", 11, "italic"))
            return

        # The vertical overlap makes room for a useful timeline while keeping
        # every card face readable at a glance.
        row_height, card_width, card_height = 26, 30, 42
        history_bottom = height - self.CARD_HEIGHT - 48
        max_rows = max(1, (history_bottom - history_top - card_height) // row_height + 1)
        visible = history[-min(10, max_rows):]
        earlier = len(history) - len(visible)
        history_height = card_height + (len(visible) - 1) * row_height
        first_y = max(history_top, history_bottom - history_height)
        if earlier:
            canvas.create_text(width // 2, first_y - 26, text=f"{earlier} earlier card event(s)",
                               fill="#c8e6c9", font=("TkDefaultFont", 9, "italic"))
        for index, event in enumerate(visible):
            cards = event["cards"]
            spacing = min(card_width - 8, 24)
            y = first_y + index * row_height
            label = f"{event['player']}: {event['label']}"
            if not cards:
                canvas.create_text(width // 2, y + card_height // 2, text=label,
                                   fill="#e6f4e8", font=("TkDefaultFont", 10, "italic"))
                continue
            total_width = card_width + (len(cards) - 1) * spacing
            start_x = width // 2 - total_width // 2
            canvas.create_text(start_x - 8, y + card_height // 2, text=label, anchor="e",
                               fill="#e6f4e8", font=("TkDefaultFont", 9, "bold"))
            for card_index, card in enumerate(cards):
                x = start_x + card_index * spacing
                self._draw_card_face(canvas, x, y, card_width, card_height, card, font_size=9)

    def _draw_card_face(self, canvas: tk.Canvas, x: int, y: int, width: int, height: int,
                        card: Mapping[str, str], *, font_size: int) -> None:
        color = "#ba1a1a" if card["suit"] in RED_SUITS else "#1c1c1c"
        canvas.create_rectangle(x, y, x + width, y + height, fill="white", outline="#283618", width=1)
        canvas.create_text(x + 4, y + 4, text=self._card_label(card), anchor="nw", fill=color,
                           font=("TkDefaultFont", font_size, "bold"))

    @staticmethod
    def card_history(message: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Return public card events formatted for the centered table history."""
        labels = {
            "opening_play": "opened",
            "played": "played",
            "completed": "completed",
            "bombed": "bombed",
            "passed": "passed",
        }
        result = []
        for event in message.get("events", []):
            if event.get("type") not in labels:
                continue
            cards = event.get("cards", [])
            if event["type"] == "bombed":
                cards = [event["card"]]
            result.append({"player": event["player"], "label": labels[event["type"]],
                           "cards": deepcopy(cards)})
        return result

    def _draw_action_buttons(self) -> None:
        assert self._message is not None and self._actions is not None
        request = self._message.get("request")
        if not request:
            ttk.Label(self._actions, text="Waiting for another player.").pack(side="left")
            return
        actions = request["legal_actions"]
        kinds = {action["type"] for action in actions}
        if "choose_pile" in kinds:
            for action in actions:
                ttk.Button(self._actions, text=f"Choose pile {action['pile'] + 1} ({self._face_up(action['pile'])})",
                           command=lambda value=action: self._submit(value)).pack(side="left", padx=3)
        elif "trade_request" in kinds:
            for action in actions:
                ttk.Button(self._actions, text=f"Request {action['rank']}",
                           command=lambda value=action: self._submit(value)).pack(side="left", padx=2)
        else:
            selected_action = self._selected_action(actions)
            if selected_action is not None:
                ttk.Button(self._actions, text=self._action_label(selected_action),
                           command=lambda value=selected_action: self._submit(value)).pack(side="left", padx=3)
            if "pass" in kinds:
                self._add_first_button(actions, "pass", "Pass")
            if "decline_completion" in kinds:
                self._add_first_button(actions, "decline_completion", "Decline completion")
            if "bomb" in kinds:
                bomb = next((a for a in actions if a["type"] == "bomb" and [a["card"]] == self._selected), None)
                if bomb is not None:
                    ttk.Button(self._actions, text="Bomb with selected 2", command=lambda: self._submit(bomb)).pack(side="left", padx=3)
            ttk.Label(self._actions, text="Select cards, then use the matching action button.").pack(side="right")

    def _add_first_button(self, actions: list[dict[str, Any]], kind: str, label: str) -> None:
        assert self._actions is not None
        action = next(action for action in actions if action["type"] == kind)
        ttk.Button(self._actions, text=label, command=lambda: self._submit(action)).pack(side="left", padx=3)

    def _selected_action(self, actions: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        for action in actions:
            if self._same_cards(action.get("cards"), self._selected):
                return action
            if action.get("type") == "trade_return" and self._same_cards([action["card"]], self._selected):
                return action
        return None

    @staticmethod
    def _same_cards(left: Any, right: Any) -> bool:
        """Compare card groups by identity, not the order they were clicked."""
        if not isinstance(left, list) or not isinstance(right, list):
            return False
        try:
            return sorted((card["rank"], card["suit"]) for card in left) == sorted(
                (card["rank"], card["suit"]) for card in right
            )
        except (KeyError, TypeError):
            return False

    def _toggle_card(self, card: dict[str, str]) -> None:
        if card in self._selected:
            self._selected.remove(card)
        else:
            self._selected.append(card)
        self._render()

    def _clear_selection(self) -> None:
        self._selected = []
        if self._message is not None:
            self._render()

    def _submit(self, action: Mapping[str, Any]) -> None:
        self._answer = deepcopy(dict(action))
        if self._waiting is not None:
            self._waiting.set(True)

    def _close(self) -> None:
        self._closed = True
        if self._waiting is not None:
            self._waiting.set(True)
        if self._root is not None:
            self._root.destroy()
            self._root = None

    def _face_up(self, pile: int) -> str:
        assert self._message is not None
        start = next(event for event in reversed(self._message["events"]) if event["type"] == "round_started")
        return self._card_label(start["face_up"][pile])

    @staticmethod
    def _card_label(card: Mapping[str, str]) -> str:
        return card["rank"] + SUIT_SYMBOLS[card["suit"]]

    @staticmethod
    def _action_label(action: Mapping[str, Any]) -> str:
        if action["type"] == "complete":
            return "Complete four of a kind"
        if action["type"] == "trade_return":
            return "Return selected card"
        return "Play selected cards"


class OmniscientHumanClient(HumanClient):
    """A local-only human client whose host supplies every hand and pile.

    This is for game-engine inspection and testing only. Normal protocol
    messages deliberately do not contain the ``all_hands`` field that enables
    this display.
    """


def main() -> None:
    """Start successive local rounds: one human player and three basic opponents."""
    players = ("you", "north", "west", "east")
    game = Game(players)
    human = HumanClient()
    clients: dict[str, Client] = {"you": human}
    clients.update({player: BasicClient() for player in players if player != "you"})
    try:
        while True:
            game.start_round()
            game.run(clients)
            if not human.play_again(game.finish_round()):
                return
    except RuntimeError as error:
        if str(error) != "human client window was closed":
            raise


if __name__ == "__main__":
    main()
