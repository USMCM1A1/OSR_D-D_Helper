"""Combat-encounter simulator — Monte Carlo with sample trace.

Pre-tests how a planned encounter will play out before the table:

    party    : list of PC dicts (the schema produced by character_ingester)
    monsters : list of statblock_parser.ParsedMonster (from a room's
               SRD-enriched statblocks, or rolled off a WM table)

`run_one_combat` plays a single fight to completion with seeded RNG.
`monte_carlo` runs N independent fights and aggregates win-rate, TPK
rate, average rounds, average remaining party HP, and an MVP attacker.
The trace from trial #0 is captured as a representative sample.

Abstractions (deliberately not 5e-faithful):

  - Positioning is binary: every combatant is `melee` or `ranged`.
    Anyone in melee can attack any melee target; ranged attackers can
    target anyone. Ranged PCs are not pulled into melee — monsters
    just attack whichever target is closest (which we resolve as
    "anyone in melee first; ranged only if no melee target").
  - No opportunity attacks, no movement, no cover, no flanking, no
    line of sight, no concentration, no conditions (paralysis, fear,
    etc.). Save-rider effects on attacks are recorded in the trace
    but ignored mechanically.
  - At 0 HP a PC is `down` and out of the fight; no death saves.
  - One round of multiattack collapses to N successive attack rolls
    using the monster's attack list in order.

Combat ends when all combatants on one side are `down`, or after the
round cap is hit (declared a draw — which is rare; the cap is mostly a
safety net against pathological states).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import statblock_parser as sp


# Rounds before we call it a draw. Real fights end well before this; the
# cap exists to prevent an infinite loop if a future bug makes both
# sides unable to deal damage to each other.
ROUND_CAP = 30

# How many trials the Monte Carlo runs by default. Tunable by caller.
DEFAULT_TRIALS = 100


# ---------------------------------------------------------------------------
# Action — the verb a combatant chooses on their turn
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """One decision returned by the tactics module.

    `kind` values:
      attack       — weapon attack; needs target_id + attack_index
      heal         — single-target heal spell; needs target_id, spell_name, spell_level
      spell_save   — save-DC attack spell (Sacred Flame, Burning Hands, ...);
                     needs target_id (single target) OR aoe handled by tactics
                     pre-picking all targets; spell_name; spell_level
      second_wind  — fighter self-heal; consumes the once-per-rest flag
      pass         — combatant has no useful action this turn
    """
    kind: str
    target_id: str | None = None
    attack_index: int | None = None
    spell_name: str | None = None
    spell_level: int = 0
    aoe_target_ids: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Combatant — uniform PC/monster representation during combat
# ---------------------------------------------------------------------------


@dataclass
class Combatant:
    id: str
    side: str                 # "pc" | "monster"
    name: str
    ac: int
    hp: int
    hp_max: int
    init_bonus: int
    speed: int
    position: str             # "melee" | "ranged"
    attacks: list[dict]       # MonsterAttack-shaped dicts
    spells: dict = field(default_factory=lambda: {"slots": {}, "memorized": []})
    features: dict = field(default_factory=dict)
    class_: str = ""
    creature_type: str = ""
    saves: dict = field(default_factory=dict)   # {"dex": 2, "con": 4, ...}
    used_second_wind: bool = False
    is_down: bool = False
    damage_dealt: int = 0

    @property
    def alive(self) -> bool:
        return self.hp > 0 and not self.is_down


# ---------------------------------------------------------------------------
# Combat state
# ---------------------------------------------------------------------------


@dataclass
class CombatState:
    combatants: list[Combatant]
    rng: random.Random
    trace: list[str] = field(default_factory=list)
    round_no: int = 0

    def hostiles_of(self, c: Combatant) -> list[Combatant]:
        return [x for x in self.combatants if x.side != c.side and x.alive]

    def allies_of(self, c: Combatant) -> list[Combatant]:
        return [x for x in self.combatants
                if x.side == c.side and x.id != c.id and x.alive]

    def by_id(self, cid: str) -> Combatant | None:
        for c in self.combatants:
            if c.id == cid:
                return c
        return None

    def side_alive(self, side: str) -> bool:
        return any(c.side == side and c.alive for c in self.combatants)


# ---------------------------------------------------------------------------
# Outcome types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CombatOutcome:
    winner: str                 # "pc" | "monster" | "draw"
    rounds: int
    pc_hp_remaining: dict[str, int]    # id → hp
    pc_hp_max: dict[str, int]
    tpk: bool
    trace: tuple[str, ...]
    damage_by_id: dict[str, int]       # any combatant who dealt damage


@dataclass(frozen=True)
class MonteCarloReport:
    trials: int
    party_win_pct: float
    tpk_pct: float
    avg_rounds: float
    avg_party_hp_pct: float
    mvp_id: str | None
    mvp_name: str
    mvp_avg_damage: float
    sample_trace: tuple[str, ...]      # trial #0


# ---------------------------------------------------------------------------
# Building combatants from inputs
# ---------------------------------------------------------------------------


def combatant_from_pc(pc: dict, *, idx: int) -> Combatant:
    """Build a Combatant from the JSON shape character_ingester produces.

    The shape is documented in CHARACTER_SCHEMA; keys we don't find
    fall back to plausible defaults so a partial sheet still produces a
    runnable PC.
    """
    name = str(pc.get("name") or f"PC{idx + 1}")
    attacks = [_normalise_pc_attack(a) for a in (pc.get("attacks") or [])]
    primary_range = "ranged" if (
        attacks and all(a["range"] == "ranged" for a in attacks)
    ) else "melee"
    return Combatant(
        id=f"pc-{idx}-{_slug(name)}",
        side="pc",
        name=name,
        ac=int(pc.get("ac") or 10),
        hp=int(pc.get("hp_max") or 8),
        hp_max=int(pc.get("hp_max") or 8),
        init_bonus=int(pc.get("init_bonus") or 0),
        speed=int(pc.get("speed") or 30),
        position=primary_range,
        attacks=attacks,
        spells=_normalise_spells(pc.get("spells")),
        features=dict(pc.get("features") or {}),
        class_=str(pc.get("class") or "").lower(),
        saves=dict(pc.get("saves") or {}),
    )


def _normalise_pc_attack(raw: dict) -> dict:
    return {
        "name": str(raw.get("name") or "Strike"),
        "to_hit": int(raw.get("to_hit") or 0),
        "damage_dice": str(raw.get("damage") or "1d4"),
        "damage_type": str(raw.get("damage_type") or "bludgeoning"),
        "range": "ranged" if str(raw.get("range") or "melee").lower() == "ranged"
                 else "melee",
        "save": None,
    }


def _normalise_spells(raw: Any) -> dict:
    if not isinstance(raw, dict):
        return {"slots": {}, "memorized": []}
    raw_slots = raw.get("slots") or {}
    slots: dict[int, int] = {}
    for k, v in raw_slots.items():
        try:
            slots[int(k)] = int(v)
        except (ValueError, TypeError):
            continue
    memorized = list(raw.get("memorized") or [])
    return {"slots": slots, "memorized": memorized}


def combatant_from_monster(monster: sp.ParsedMonster, *, idx: int,
                           label: str | None = None) -> Combatant:
    """Build a Combatant from a ParsedMonster. `label` lets the caller
    distinguish multiple instances ("Skeleton #1", "Skeleton #2")."""
    attacks = [
        {
            "name": a.name,
            "to_hit": a.to_hit,
            "damage_dice": a.damage_dice,
            "damage_type": a.damage_type,
            "range": a.range,
            "save": dict(a.save) if a.save else None,
        }
        for a in monster.attacks
    ]
    primary_range = "ranged" if (
        attacks and all(a["range"] == "ranged" for a in attacks)
    ) else "melee"
    name = label or monster.name
    # Pre-compute the multiattack indices once so the turn loop just
    # iterates a list rather than re-parsing the prose every round.
    if monster.multiattack and monster.attacks:
        multiattack_indices = sp.select_multiattack_indices(
            monster.attacks, monster.multiattack,
        )
    else:
        multiattack_indices = ()
    return Combatant(
        id=f"mn-{idx}-{_slug(name)}",
        side="monster",
        name=name,
        ac=monster.ac,
        hp=monster.hp_avg,
        hp_max=monster.hp_avg,
        init_bonus=0,                     # SRD doesn't expose it; Dex mod-ish, fine to skip
        speed=monster.speed,
        position=primary_range,
        attacks=attacks,
        creature_type=monster.creature_type,
        features={
            "multiattack": monster.multiattack,
            "multiattack_indices": multiattack_indices,
        },
    )


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s.strip().lower())[:24]


# ---------------------------------------------------------------------------
# Initiative
# ---------------------------------------------------------------------------


def roll_initiative(combatants: list[Combatant], rng: random.Random) -> list[Combatant]:
    """Sort combatants by `d20 + init_bonus` descending. Stable: PCs win
    ties (so a PC with init_bonus 0 acts before a monster also at 0)."""
    rolled = []
    for c in combatants:
        roll = rng.randint(1, 20) + c.init_bonus
        # Negative side ordering puts "pc" (alphabetically before "monster"
        # via the - sign trick: pc → 0, monster → 1).
        side_priority = 0 if c.side == "pc" else 1
        rolled.append((roll, -side_priority, c))
    rolled.sort(key=lambda t: (-t[0], -t[1]))
    return [t[2] for t in rolled]


# ---------------------------------------------------------------------------
# Attack / save resolution
# ---------------------------------------------------------------------------


def resolve_attack(actor: Combatant, target: Combatant, attack: dict,
                   state: CombatState, *, bonus_damage: str = "") -> bool:
    """Roll d20 + to_hit vs target.AC. On hit, roll damage and apply.

    `bonus_damage` adds an extra dice expression (sneak attack, smite)
    on top of base damage. Returns True if the attack hit.

    Crit on natural 20 doubles dice. Nat 1 always misses.
    """
    nat = state.rng.randint(1, 20)
    total = nat + attack["to_hit"]
    crit = nat == 20
    miss = nat == 1
    hit = (not miss) and (crit or total >= target.ac)

    if not hit:
        state.trace.append(
            f"{actor.name} → {target.name} with {attack['name']}: "
            f"d20={nat}+{attack['to_hit']}={total} vs AC {target.ac} — miss."
        )
        return False

    base = sp.roll_dice(attack["damage_dice"], state.rng)
    extra = 0
    if crit:
        # Crit doubles the dice rolled (mod stays the same). We approximate
        # by rolling the dice portion again and adding it.
        extra += _reroll_dice_only(attack["damage_dice"], state.rng)
    bonus = sp.roll_dice(bonus_damage, state.rng) if bonus_damage else 0
    total_damage = max(0, base + extra + bonus)

    applied = min(total_damage, target.hp)
    target.hp = max(0, target.hp - total_damage)
    actor.damage_dealt += applied

    if target.hp <= 0:
        target.is_down = True

    bonus_clause = f" (+{bonus_damage}={bonus})" if bonus_damage else ""
    crit_clause = " CRIT" if crit else ""
    down = " — DOWN" if target.is_down else ""
    state.trace.append(
        f"{actor.name} hits {target.name} with {attack['name']}: "
        f"d20={nat}+{attack['to_hit']}={total} vs AC {target.ac}{crit_clause} → "
        f"{total_damage} {attack['damage_type']}{bonus_clause} → "
        f"{target.name} {target.hp}/{target.hp_max}{down}"
    )

    if attack.get("save") and not target.is_down:
        # Save rider on the attack — capture in trace, ignore mechanical
        # effect in v1 (out of scope per plan).
        save = attack["save"]
        state.trace.append(
            f"  (rider: DC {save['dc']} {save['ability'].upper()} save "
            f"vs {attack['name']} effect — not simulated)"
        )

    return True


def _reroll_dice_only(expr: str, rng: random.Random) -> int:
    """For crits: re-roll just the dice portion of NdN±M, ignoring the
    flat modifier. Used to add an extra dice roll on a crit."""
    m = sp._DICE_RE.search(expr)
    if not m:
        return 0
    count = int(m.group("count"))
    sides = int(m.group("sides"))
    return sum(rng.randint(1, sides) for _ in range(count))


def resolve_spell_save(actor: Combatant, targets: list[Combatant],
                       spell: dict, state: CombatState) -> None:
    """Resolve a save-DC damage spell against one or more targets.

    Spell shape:
      {"name": "Sacred Flame", "level": 0, "type": "save_attack",
       "save": "dex", "dc": 13, "damage": "1d8",
       "half_on_save": True, "aoe": False}
    """
    dc = int(spell.get("dc") or 10)
    save_ab = str(spell.get("save") or "dex").lower()
    half = bool(spell.get("half_on_save"))
    damage_expr = str(spell.get("damage") or "1d4")

    # AoE spells share one damage roll across all targets (5e RAW).
    is_aoe = bool(spell.get("aoe"))
    rolled_damage = sp.roll_dice(damage_expr, state.rng) if is_aoe else None

    for target in targets:
        save_bonus = int(target.saves.get(save_ab, 0))
        roll = state.rng.randint(1, 20) + save_bonus
        success = roll >= dc
        damage = rolled_damage if rolled_damage is not None else \
                 sp.roll_dice(damage_expr, state.rng)
        if success:
            damage = damage // 2 if half else 0
        applied = min(damage, target.hp)
        target.hp = max(0, target.hp - damage)
        actor.damage_dealt += applied
        if target.hp <= 0:
            target.is_down = True
        outcome = "save" if success else "fail"
        down = " — DOWN" if target.is_down else ""
        state.trace.append(
            f"{actor.name} casts {spell['name']} on {target.name}: "
            f"DC {dc} {save_ab.upper()} save = {roll} ({outcome}) → "
            f"{damage} damage → {target.name} {target.hp}/{target.hp_max}{down}"
        )


def resolve_heal(actor: Combatant, target: Combatant, spell: dict,
                 state: CombatState) -> None:
    """Apply a healing spell. `amount` is a dice expression like 1d8+3."""
    amount = sp.roll_dice(str(spell.get("amount") or "1d4"), state.rng)
    healed_to = min(target.hp_max, target.hp + amount)
    real_heal = healed_to - target.hp
    target.hp = healed_to
    state.trace.append(
        f"{actor.name} casts {spell['name']} on {target.name}: "
        f"+{real_heal} HP → {target.name} {target.hp}/{target.hp_max}"
    )


# ---------------------------------------------------------------------------
# Spell-slot accounting
# ---------------------------------------------------------------------------


def consume_slot(combatant: Combatant, level: int) -> bool:
    """Decrement a spell slot. Returns True if a slot was consumed.
    Cantrips (level 0) return True without modifying slots."""
    if level <= 0:
        return True
    slots = combatant.spells.get("slots") or {}
    n = int(slots.get(level, 0))
    if n <= 0:
        return False
    slots[level] = n - 1
    return True


# ---------------------------------------------------------------------------
# Round / combat loop
# ---------------------------------------------------------------------------


def _take_turn(actor: Combatant, state: CombatState, pick_action) -> None:
    """One combatant takes their turn — pick an action, resolve it.

    `pick_action(actor, state) -> Action` is injected so the simulator
    doesn't import tactics directly (avoids a circular import; tactics
    imports types from this module).
    """
    if not actor.alive:
        return

    # Monster multiattack: take only the swings the monster's prose
    # actually authorises. `multiattack_indices` is pre-computed at
    # combatant construction (see combatant_from_monster) from the
    # SRD prose: 'makes two morningstar attacks' → (0, 0), 'two melee
    # attacks, each with a different weapon' → (0, 1), and so on.
    indices = actor.features.get("multiattack_indices") if actor.side == "monster" else None
    if indices and actor.attacks:
        primary = pick_action(actor, state)
        if primary.kind != "attack":
            _resolve_action(actor, primary, state)
            return
        for i in indices:
            target = state.by_id(primary.target_id) if primary.target_id else None
            if target is None or not target.alive:
                hostiles = state.hostiles_of(actor)
                if not hostiles:
                    return
                target = hostiles[0]
                primary = Action(kind="attack", target_id=target.id)
            if i >= len(actor.attacks):
                continue
            resolve_attack(actor, target, actor.attacks[i], state)
            if not state.side_alive(target.side):
                return
        return

    action = pick_action(actor, state)
    _resolve_action(actor, action, state)

    # PC Extra Attack (Fighter / Paladin / Ranger / Barbarian, L≥5):
    # take a second attack on the same turn. Tactics re-pick the
    # target so a downed primary doesn't waste the second swing.
    if (
        actor.side == "pc"
        and action.kind == "attack"
        and actor.features.get("extra_attack")
    ):
        target = state.by_id(action.target_id) if action.target_id else None
        if target is not None and target.alive:
            second = pick_action(actor, state)
            if second.kind == "attack":
                _resolve_action(actor, second, state)


def _resolve_action(actor: Combatant, action: Action, state: CombatState) -> None:
    if action.kind == "pass":
        state.trace.append(f"{actor.name} holds.")
        return
    if action.kind == "second_wind":
        # Fighter feature: 1d10 + level (we approximate level via hp_max
        # bucket — close enough for this abstraction).
        if actor.used_second_wind:
            state.trace.append(f"{actor.name} second wind already spent.")
            return
        amount = sp.roll_dice("1d10+3", state.rng)
        new_hp = min(actor.hp_max, actor.hp + amount)
        diff = new_hp - actor.hp
        actor.hp = new_hp
        actor.used_second_wind = True
        state.trace.append(
            f"{actor.name} uses Second Wind: +{diff} HP → {actor.hp}/{actor.hp_max}"
        )
        return
    if action.kind == "attack":
        target = state.by_id(action.target_id) if action.target_id else None
        if target is None or not target.alive:
            state.trace.append(f"{actor.name} has no target.")
            return
        if action.attack_index is None or not actor.attacks:
            state.trace.append(f"{actor.name} has no attack.")
            return
        try:
            atk = actor.attacks[action.attack_index]
        except IndexError:
            state.trace.append(f"{actor.name} attack index out of range.")
            return
        bonus = ""
        # Sneak attack rider: tactics decides whether conditions are met
        # (set via attack_index sentinels would be uglier); we read it
        # from features when present and tactics flagged via spell_name
        # piggyback ("__sneak_attack__").
        if action.spell_name == "__sneak_attack__":
            dice = int(actor.features.get("sneak_attack_dice") or 0)
            if dice > 0:
                bonus = f"{dice}d6"
        if action.spell_name == "__divine_smite__":
            level = max(1, action.spell_level)
            consumed = consume_slot(actor, level)
            if consumed:
                # Smite damage scales: 2d8 at lvl 1, +1d8 per slot up to 5d8.
                # Extra die vs undead/fiend.
                base_dice = min(5, 1 + level)
                if target.creature_type in ("undead", "fiend"):
                    base_dice = min(6, base_dice + 1)
                bonus = f"{base_dice}d8"
        resolve_attack(actor, target, atk, state, bonus_damage=bonus)
        return
    if action.kind == "heal":
        target = state.by_id(action.target_id) if action.target_id else None
        if target is None or not target.alive:
            state.trace.append(f"{actor.name} heal target unavailable.")
            return
        if not consume_slot(actor, action.spell_level):
            state.trace.append(f"{actor.name} cannot heal — no slot.")
            return
        spell = _find_spell(actor, action.spell_name) or {}
        resolve_heal(actor, target, spell, state)
        return
    if action.kind == "spell_save":
        if not consume_slot(actor, action.spell_level):
            state.trace.append(f"{actor.name} cannot cast — no slot.")
            return
        spell = _find_spell(actor, action.spell_name) or {}
        target_ids = action.aoe_target_ids or (
            (action.target_id,) if action.target_id else ()
        )
        targets = [t for t in (state.by_id(tid) for tid in target_ids) if t and t.alive]
        if not targets:
            state.trace.append(f"{actor.name} spell has no targets.")
            return
        resolve_spell_save(actor, targets, spell, state)
        return
    state.trace.append(f"{actor.name} chose unknown action {action.kind!r}.")


def _find_spell(actor: Combatant, name: str | None) -> dict | None:
    if not name:
        return None
    for spell in actor.spells.get("memorized") or []:
        if str(spell.get("name", "")).lower() == name.lower():
            return spell
    return None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_one_combat(party: list[Combatant], monsters: list[Combatant],
                   rng: random.Random, *, pick_action=None) -> CombatOutcome:
    """Play one combat to completion. `pick_action` is the tactics
    dispatcher; if None, the default tactics module is imported and
    used. Combatants are modified in place — pass copies if you need to
    re-run."""
    if pick_action is None:
        import tactics
        pick_action = tactics.pick_action

    combatants = party + monsters
    state = CombatState(combatants=combatants, rng=rng)
    state.trace.append(
        f"=== Combat begins: {len(party)} PCs vs {len(monsters)} monsters ==="
    )

    init_order = roll_initiative(combatants, rng)
    state.trace.append(
        "Initiative: " + ", ".join(c.name for c in init_order)
    )

    for round_no in range(1, ROUND_CAP + 1):
        state.round_no = round_no
        state.trace.append(f"-- Round {round_no} --")
        for actor in init_order:
            if not state.side_alive("pc"):
                break
            if not state.side_alive("monster"):
                break
            _take_turn(actor, state, pick_action)
        if not state.side_alive("pc") or not state.side_alive("monster"):
            break

    pc_alive = state.side_alive("pc")
    monster_alive = state.side_alive("monster")
    if pc_alive and not monster_alive:
        winner = "pc"
    elif monster_alive and not pc_alive:
        winner = "monster"
    else:
        winner = "draw"
    tpk = not pc_alive

    pc_hp_remaining = {c.id: c.hp for c in party}
    pc_hp_max = {c.id: c.hp_max for c in party}
    damage_by_id = {c.id: c.damage_dealt for c in combatants if c.damage_dealt > 0}

    state.trace.append(
        f"=== Combat ends in round {state.round_no}: winner = {winner} ==="
    )

    return CombatOutcome(
        winner=winner,
        rounds=state.round_no,
        pc_hp_remaining=pc_hp_remaining,
        pc_hp_max=pc_hp_max,
        tpk=tpk,
        trace=tuple(state.trace),
        damage_by_id=damage_by_id,
    )


def monte_carlo(party_proto: list[dict], monsters_proto: list[sp.ParsedMonster],
                *, trials: int = DEFAULT_TRIALS, base_seed: int = 0,
                pick_action=None) -> MonteCarloReport:
    """Run `trials` independent combats with seeded RNGs.

    `party_proto`, `monsters_proto` are immutable templates — a fresh
    Combatant list is built per trial so HP starts at max each time.
    Returns aggregated statistics plus the trace from trial #0 as the
    sample.
    """
    wins = 0
    tpks = 0
    rounds_total = 0
    party_hp_pct_total = 0.0
    damage_totals: dict[str, float] = {}
    name_by_id: dict[str, str] = {}
    sample_trace: tuple[str, ...] = ()

    for i in range(trials):
        rng = random.Random(base_seed + i)
        party = [combatant_from_pc(pc, idx=j) for j, pc in enumerate(party_proto)]
        monsters: list[Combatant] = []
        m_idx = 0
        for monster in monsters_proto:
            label = monster.name if len([m for m in monsters_proto
                                         if m.name == monster.name]) == 1 \
                    else f"{monster.name} #{m_idx + 1}"
            monsters.append(combatant_from_monster(monster, idx=m_idx, label=label))
            m_idx += 1
        outcome = run_one_combat(party, monsters, rng, pick_action=pick_action)

        if outcome.winner == "pc":
            wins += 1
        if outcome.tpk:
            tpks += 1
        rounds_total += outcome.rounds

        if outcome.pc_hp_max:
            pct = sum(outcome.pc_hp_remaining.values()) / sum(outcome.pc_hp_max.values())
            party_hp_pct_total += pct

        for cid, dmg in outcome.damage_by_id.items():
            damage_totals[cid] = damage_totals.get(cid, 0) + dmg
        for c in party + monsters:
            name_by_id[c.id] = c.name

        if i == 0:
            sample_trace = outcome.trace

    mvp_id = max(damage_totals, key=damage_totals.get) if damage_totals else None
    mvp_name = name_by_id.get(mvp_id, "—") if mvp_id else "—"
    mvp_avg = damage_totals[mvp_id] / trials if mvp_id else 0.0

    return MonteCarloReport(
        trials=trials,
        party_win_pct=100.0 * wins / trials if trials else 0.0,
        tpk_pct=100.0 * tpks / trials if trials else 0.0,
        avg_rounds=rounds_total / trials if trials else 0.0,
        avg_party_hp_pct=100.0 * party_hp_pct_total / trials if trials else 0.0,
        mvp_id=mvp_id,
        mvp_name=mvp_name,
        mvp_avg_damage=mvp_avg,
        sample_trace=sample_trace,
    )
