# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Compiled hot core of the corridors game engine.

Pure C data structures behind the same semantics as game.py's pure-Python
implementations. game.py imports this module when it is built, feeds it the
board topology once via init(), and dispatches its hot functions here; when
the extension is missing, game.py silently falls back to pure Python.

Representation:
  - cells: index = row * 9 + col, 99 cells (11 rows x 9 cols).
  - blockable edges: 144 bits, carried as 3 x uint64 words in C and as an
    arbitrary-size Python int (the same value game.py uses) at the boundary.
  - walls: index into game.ALL_WALLS order, 128 slots; a placed-wall set is a
    128-bit Python int at the boundary, two uint64 words in C.

Ordering matters for parity with the pure code: pawn directions iterate
N,S,W,E (game._DIRS order), full adjacency preserves game._ADJ order (this
fixes shortest-path tie-breaking), and wall candidates iterate in ALL_WALLS
order. Parity tests compare exact lists, not sets.
"""

from libc.stdint cimport uint8_t, int16_t, uint64_t
from libc.string cimport memset

DEF NCELLS = 99
DEF NCOLS = 9
DEF NWALLS = 128
DEF NW = 3            # uint64 words per edge mask (144 bits)
DEF MAXADJ = 5        # playable cell: <=4 playable edges + 1 end-zone entry
DEF UNREACH = 255
DEF P1_END_ROW = 10
DEF P2_END_ROW = 0
DEF DCACHE = 64       # direct-mapped dist-table cache slots

_M64 = (1 << 64) - 1

# Per-direction neighbor tables (dir order: N,S,W,E to match game._DIRS).
cdef int16_t _DNBR[NCELLS][4]
cdef uint64_t _DMASK[NCELLS][4][NW]
# Full adjacency in game._ADJ order (drives BFS and path tie-breaking).
cdef int _ADJN[NCELLS]
cdef int16_t _ADJI[NCELLS][MAXADJ]
cdef uint64_t _ADJM[NCELLS][MAXADJ][NW]
# Per-cell OR of touching blockable-edge masks (pawn mobility test).
cdef uint64_t _CMASK[NCELLS][NW]
# Walls: edge masks and conflict sets (128-bit wall-index masks).
cdef uint64_t _WMASK[NWALLS][NW]
cdef uint64_t _WCONF[NWALLS][2]
# Goal-directed DFS neighbor order: permutation of adjacency slots per
# (goal, cell), most-goal-ward slot last so a LIFO stack pops it first.
cdef uint8_t _GORDER[NCELLS][NCELLS][MAXADJ]
cdef uint8_t _GREADY[NCELLS]
# Direct-mapped cache of BFS dist tables keyed on (goal, blocked mask).
cdef uint64_t _DC_KEY[DCACHE][4]
cdef uint8_t _DC_DIST[DCACHE][NCELLS]
cdef uint8_t _DC_VALID[DCACHE]

cdef bint _READY = False


def init(adj_dir, adj_full, cell_masks, wall_data):
    """Load board topology. Called once by game.py at import.

    adj_dir:    per cell, 4 entries (N,S,W,E) of (nb_idx or -1, edge_mask_int)
    adj_full:   per cell, list of (nb_idx, edge_mask_int) in game._ADJ order
    cell_masks: per cell, OR of touching blockable-edge masks (int)
    wall_data:  per wall in ALL_WALLS order, (edge_mask_int, conflict_bits_int)
    """
    global _READY
    cdef int i, d, k
    cdef object m
    for i in range(NCELLS):
        for d in range(4):
            nb, m = adj_dir[i][d]
            _DNBR[i][d] = nb
            _DMASK[i][d][0] = m & _M64
            _DMASK[i][d][1] = (m >> 64) & _M64
            _DMASK[i][d][2] = m >> 128
        entries = adj_full[i]
        _ADJN[i] = len(entries)
        for k in range(len(entries)):
            nb, m = entries[k]
            _ADJI[i][k] = nb
            _ADJM[i][k][0] = m & _M64
            _ADJM[i][k][1] = (m >> 64) & _M64
            _ADJM[i][k][2] = m >> 128
        m = cell_masks[i]
        _CMASK[i][0] = m & _M64
        _CMASK[i][1] = (m >> 64) & _M64
        _CMASK[i][2] = m >> 128
    for i in range(NWALLS):
        m, conf = wall_data[i]
        _WMASK[i][0] = m & _M64
        _WMASK[i][1] = (m >> 64) & _M64
        _WMASK[i][2] = m >> 128
        _WCONF[i][0] = conf & _M64
        _WCONF[i][1] = (conf >> 64) & _M64
    memset(_GREADY, 0, sizeof(_GREADY))
    memset(_DC_VALID, 0, sizeof(_DC_VALID))
    _READY = True


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

cdef inline void _split(object mask, uint64_t* out):
    out[0] = <uint64_t> (mask & _M64)
    out[1] = <uint64_t> ((mask >> 64) & _M64)
    out[2] = <uint64_t> (mask >> 128)


cdef inline object _join(uint64_t* m):
    return int(m[0]) | (int(m[1]) << 64) | (int(m[2]) << 128)


cdef inline bint _hit(uint64_t* a, uint64_t* b) noexcept:
    return ((a[0] & b[0]) | (a[1] & b[1]) | (a[2] & b[2])) != 0


cdef inline void _or3(uint64_t* dst, uint64_t* a, uint64_t* b) noexcept:
    dst[0] = a[0] | b[0]
    dst[1] = a[1] | b[1]
    dst[2] = a[2] | b[2]


cdef inline bint _zero3(uint64_t* a) noexcept:
    return (a[0] | a[1] | a[2]) == 0


# ---------------------------------------------------------------------------
# BFS distance tables (with direct-mapped cache)
# ---------------------------------------------------------------------------

cdef void _bfs(int goal, uint64_t* blk, uint8_t* dist) noexcept:
    cdef int16_t q[NCELLS]
    cdef int head = 0, tail = 0, cell, nb, k
    cdef uint8_t d
    memset(dist, UNREACH, NCELLS)
    dist[goal] = 0
    q[tail] = goal
    tail += 1
    while head < tail:
        cell = q[head]
        head += 1
        d = dist[cell] + 1
        for k in range(_ADJN[cell]):
            nb = _ADJI[cell][k]
            if dist[nb] != UNREACH:
                continue
            if _hit(_ADJM[cell][k], blk):
                continue
            dist[nb] = d
            q[tail] = nb
            tail += 1


cdef uint8_t* _dist_cached(int goal, uint64_t* blk) noexcept:
    """Dist table for (goal, blocked), via a direct-mapped cache.

    The returned pointer is valid only until the next _dist_cached call (a
    colliding key recomputes in place). Callers must finish reading before
    triggering another BFS.
    """
    cdef uint64_t h = (<uint64_t> goal) * <uint64_t> 0x9E3779B97F4A7C15
    h ^= blk[0] * <uint64_t> 0xC2B2AE3D27D4EB4F
    h ^= blk[1] * <uint64_t> 0x165667B19E3779F9
    h ^= blk[2] * <uint64_t> 0x27D4EB2F165667C5
    h ^= h >> 29
    cdef int slot = <int> (h & (DCACHE - 1))
    if (_DC_VALID[slot] and _DC_KEY[slot][0] == <uint64_t> goal
            and _DC_KEY[slot][1] == blk[0] and _DC_KEY[slot][2] == blk[1]
            and _DC_KEY[slot][3] == blk[2]):
        return _DC_DIST[slot]
    _bfs(goal, blk, _DC_DIST[slot])
    _DC_KEY[slot][0] = <uint64_t> goal
    _DC_KEY[slot][1] = blk[0]
    _DC_KEY[slot][2] = blk[1]
    _DC_KEY[slot][3] = blk[2]
    _DC_VALID[slot] = 1
    return _DC_DIST[slot]


# ---------------------------------------------------------------------------
# Goal-directed reachability (DFS ordered by unwalled distance)
# ---------------------------------------------------------------------------

cdef void _build_goal_order(int goal) noexcept:
    cdef uint8_t dist[NCELLS]
    cdef uint64_t none[NW]
    cdef int cell, k, j, n
    cdef uint8_t order[MAXADJ]
    cdef uint8_t key
    cdef uint8_t dk[MAXADJ]
    none[0] = 0
    none[1] = 0
    none[2] = 0
    _bfs(goal, none, dist)
    for cell in range(NCELLS):
        n = _ADJN[cell]
        # Stable insertion sort of adjacency slots, descending by unwalled
        # distance (unreachable sorts first / visited last).
        for k in range(n):
            order[k] = k
            dk[k] = dist[_ADJI[cell][k]]
        for k in range(1, n):
            key = order[k]
            j = k - 1
            while j >= 0 and dk[order[j]] < dk[key]:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = key
        for k in range(n):
            _GORDER[goal][cell][k] = order[k]
    _GREADY[goal] = 1


cdef bint _has_path(int start, int goal, uint64_t* blk) noexcept:
    if start == goal:
        return True
    if not _GREADY[goal]:
        _build_goal_order(goal)
    cdef uint64_t seen0 = 0, seen1 = 0
    cdef uint64_t one = 1
    cdef int16_t stack[NCELLS]
    cdef int sp = 0, cell, nb, k, slot
    if start >= 64:
        seen1 = one << (start - 64)
    else:
        seen0 = one << start
    stack[0] = start
    sp = 1
    while sp:
        sp -= 1
        cell = stack[sp]
        for k in range(_ADJN[cell]):
            slot = _GORDER[goal][cell][k]
            nb = _ADJI[cell][slot]
            if nb >= 64:
                if seen1 & (one << (nb - 64)):
                    continue
            elif seen0 & (one << nb):
                continue
            if _hit(_ADJM[cell][slot], blk):
                continue
            if nb == goal:
                return True
            if nb >= 64:
                seen1 |= one << (nb - 64)
            else:
                seen0 |= one << nb
            stack[sp] = nb
            sp += 1
    return False


# ---------------------------------------------------------------------------
# Shortest-path edge masks
# ---------------------------------------------------------------------------

cdef void _spath_mask(int pos, int goal, uint64_t* blk, uint64_t* out) noexcept:
    """Blockable-edge mask of ONE shortest path pos->goal (zeros if none).

    Walks the dist table greedily in adjacency order — identical tie-breaking
    to game._shortest_path_mask.
    """
    cdef uint8_t* dist
    cdef int cur, k, nb
    cdef uint8_t d
    cdef bint advanced
    out[0] = 0
    out[1] = 0
    out[2] = 0
    dist = _dist_cached(goal, blk)
    if dist[pos] == UNREACH:
        return
    cur = pos
    while cur != goal:
        d = dist[cur]
        advanced = False
        for k in range(_ADJN[cur]):
            nb = _ADJI[cur][k]
            if dist[nb] == UNREACH:
                continue
            if _hit(_ADJM[cur][k], blk):
                continue
            if dist[nb] == d - 1:
                out[0] |= _ADJM[cur][k][0]
                out[1] |= _ADJM[cur][k][1]
                out[2] |= _ADJM[cur][k][2]
                cur = nb
                advanced = True
                break
        if not advanced:
            break


# ---------------------------------------------------------------------------
# Pawn moves
# ---------------------------------------------------------------------------

cdef bint _side_has_pawn_move(int me, int opp, int own_start_row,
                              int opp_end_row, int goal,
                              uint64_t* blk) noexcept:
    cdef int d, step, straight
    for d in range(4):
        step = _DNBR[me][d]
        if step < 0:
            continue
        if _hit(_DMASK[me][d], blk):
            continue
        if step == opp:
            straight = _DNBR[step][d]
            if straight < 0 or _hit(_DMASK[step][d], blk):
                continue
            step = straight
        if step // NCOLS == own_start_row:
            continue
        if step // NCOLS == opp_end_row and step != goal:
            continue
        return True
    return False


cdef int _pawn_targets(int me, int opp, int own_start_row, int opp_end_row,
                       int goal, uint64_t* blk, int16_t* out) noexcept:
    """Legal-destination pawn targets in N,S,W,E order; returns count."""
    cdef int d, step, straight, n = 0
    for d in range(4):
        step = _DNBR[me][d]
        if step < 0:
            continue
        if _hit(_DMASK[me][d], blk):
            continue
        if step == opp:
            straight = _DNBR[step][d]
            if straight < 0 or _hit(_DMASK[step][d], blk):
                continue
            step = straight
        if step // NCOLS == own_start_row:
            continue
        if step // NCOLS == opp_end_row and step != goal:
            continue
        out[n] = step
        n += 1
    return n


def legal_pawn_moves(int me, int opp, int turn, int p1_goal, int p2_goal,
                     object mask):
    """Legal pawn destinations (cell indices) for the side to move.

    Caller (game.py) has already handled the game-over case.
    """
    cdef uint64_t blk[NW]
    cdef int16_t targets[4]
    cdef int n, i, t
    cdef int own_row, opp_end, my_goal, opp_own, opp_end2, opp_goal
    _split(mask, blk)
    if turn == 1:
        own_row = P1_END_ROW
        opp_end = P2_END_ROW
        my_goal = p1_goal
        opp_own = P2_END_ROW
        opp_end2 = P1_END_ROW
        opp_goal = p2_goal
    else:
        own_row = P2_END_ROW
        opp_end = P1_END_ROW
        my_goal = p2_goal
        opp_own = P1_END_ROW
        opp_end2 = P2_END_ROW
        opp_goal = p1_goal
    n = _pawn_targets(me, opp, own_row, opp_end, my_goal, blk, targets)
    out = []
    for i in range(n):
        t = targets[i]
        # Winning ends the game; otherwise the move may not leave the
        # opponent without any physical pawn move.
        if t == my_goal or _side_has_pawn_move(opp, t, opp_own, opp_end2,
                                               opp_goal, blk):
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Wall moves
# ---------------------------------------------------------------------------

def legal_wall_moves(int p1, int p2, int p1_goal, int p2_goal, int turn,
                     object placed_bits, object mask):
    """Legal wall placements as indices into ALL_WALLS order.

    Caller has already handled game-over and walls_left <= 0.
    placed_bits: 128-bit int, bit i set iff ALL_WALLS[i] is on the board.
    """
    cdef uint64_t m[NW]
    cdef uint64_t mw[NW]
    cdef uint64_t placed[2]
    cdef uint64_t conflict[2]
    cdef uint64_t my_path[NW]
    cdef uint64_t opp_path[NW]
    cdef uint64_t my_alt[NW]
    cdef uint64_t opp_alt[NW]
    cdef uint64_t alt_blk[NW]
    cdef uint64_t p1_mob[NW]
    cdef uint64_t p2_mob[NW]
    cdef uint64_t tmp[NW]
    cdef bint my_alt_done = False, opp_alt_done = False
    cdef int me, my_goal, opp, opp_goal
    cdef int w, d
    cdef bint adjacent
    _split(mask, m)
    placed[0] = <uint64_t> (placed_bits & _M64)
    placed[1] = <uint64_t> (placed_bits >> 64)
    conflict[0] = 0
    conflict[1] = 0
    for w in range(NWALLS):
        if (placed[w >> 6] >> (w & 63)) & 1:
            conflict[0] |= _WCONF[w][0]
            conflict[1] |= _WCONF[w][1]
    if turn == 1:
        me = p1
        my_goal = p1_goal
        opp = p2
        opp_goal = p2_goal
    else:
        me = p2
        my_goal = p2_goal
        opp = p1
        opp_goal = p1_goal
    # An already-immobilized position cannot be repaired by another wall.
    if not _side_has_pawn_move(p1, p2, P1_END_ROW, P2_END_ROW, p1_goal, m):
        return []
    if not _side_has_pawn_move(p2, p1, P2_END_ROW, P1_END_ROW, p2_goal, m):
        return []
    _spath_mask(me, my_goal, m, my_path)
    _spath_mask(opp, opp_goal, m, opp_path)
    # Mobility masks: candidate walls not touching these cannot change the
    # respective side's pawn moves. Jumps put the other pawn's edges in play
    # when the pawns are graph-adjacent.
    adjacent = False
    for d in range(4):
        if _DNBR[p1][d] == p2:
            adjacent = True
            break
    p1_mob[0] = _CMASK[p1][0]
    p1_mob[1] = _CMASK[p1][1]
    p1_mob[2] = _CMASK[p1][2]
    p2_mob[0] = _CMASK[p2][0]
    p2_mob[1] = _CMASK[p2][1]
    p2_mob[2] = _CMASK[p2][2]
    if adjacent:
        _or3(p1_mob, p1_mob, _CMASK[p2])
        _or3(p2_mob, p2_mob, _CMASK[p1])
    out = []
    for w in range(NWALLS):
        if ((placed[w >> 6] | conflict[w >> 6]) >> (w & 63)) & 1:
            continue
        mw[0] = _WMASK[w][0]
        mw[1] = _WMASK[w][1]
        mw[2] = _WMASK[w][2]
        _or3(tmp, m, mw)
        # Walls may not immobilize either pawn.
        if _hit(mw, p1_mob) and not _side_has_pawn_move(
                p1, p2, P1_END_ROW, P2_END_ROW, p1_goal, tmp):
            continue
        if _hit(mw, p2_mob) and not _side_has_pawn_move(
                p2, p1, P2_END_ROW, P1_END_ROW, p2_goal, tmp):
            continue
        # Path pair shortcut: a candidate leaving either kept path untouched
        # provably keeps that player connected; otherwise run reachability.
        if _hit(mw, my_path):
            if not my_alt_done:
                _or3(alt_blk, m, my_path)
                _spath_mask(me, my_goal, alt_blk, my_alt)
                my_alt_done = True
            if (_zero3(my_alt) or _hit(mw, my_alt)) and not _has_path(
                    me, my_goal, tmp):
                continue
        if _hit(mw, opp_path):
            if not opp_alt_done:
                _or3(alt_blk, m, opp_path)
                _spath_mask(opp, opp_goal, alt_blk, opp_alt)
                opp_alt_done = True
            if (_zero3(opp_alt) or _hit(mw, opp_alt)) and not _has_path(
                    opp, opp_goal, tmp):
                continue
        out.append(w)
    return out


# ---------------------------------------------------------------------------
# Evaluation / solver support
# ---------------------------------------------------------------------------

cdef void _dist_alt(int pos, int goal, uint64_t* blk,
                    int* dout, int* aout) noexcept:
    cdef uint8_t* dist = _dist_cached(goal, blk)
    cdef int k, nb, cnt = 0
    cdef uint8_t d = dist[pos]
    if d == UNREACH:
        dout[0] = 10_000
        aout[0] = 0
        return
    for k in range(_ADJN[pos]):
        nb = _ADJI[pos][k]
        if dist[nb] == UNREACH:
            continue
        if _hit(_ADJM[pos][k], blk):
            continue
        if dist[nb] == d - 1:
            cnt += 1
    dout[0] = d
    aout[0] = cnt


def dist_and_alt(int pos, int goal, object mask):
    """(shortest distance, shortest-path branching count) for pos -> goal.

    Distance is 10_000 when unreachable, matching solver semantics.
    """
    cdef uint64_t blk[NW]
    cdef int d, a
    _split(mask, blk)
    _dist_alt(pos, goal, blk, &d, &a)
    return d, a


cdef long long _W_DIST = 100
cdef long long _W_WALLS = 6
cdef long long _W_ALT = 2


def set_eval_weights(int w_dist, int w_walls, int w_alt):
    """Register the solver's evaluation weights (solver.py owns the values)."""
    global _W_DIST, _W_WALLS, _W_ALT
    _W_DIST = w_dist
    _W_WALLS = w_walls
    _W_ALT = w_alt


def evaluate(object p1, object p2, object g1, object g2,
             int walls_diff, int turn, object mask):
    """Full leaf evaluation from the side to move's perspective.

    Takes the Pos tuples raw (converted here, in C) — this runs once per
    leaf node, so boundary arithmetic in Python is worth avoiding.
    """
    cdef uint64_t blk[NW]
    cdef int d1, a1, d2, a2
    cdef int p1i = (<int> p1[0]) * NCOLS + <int> p1[1]
    cdef int p2i = (<int> p2[0]) * NCOLS + <int> p2[1]
    cdef int g1i = (<int> g1[0]) * NCOLS + <int> g1[1]
    cdef int g2i = (<int> g2[0]) * NCOLS + <int> g2[1]
    cdef long long score
    _split(mask, blk)
    _dist_alt(p1i, g1i, blk, &d1, &a1)
    _dist_alt(p2i, g2i, blk, &d2, &a2)
    # from P1's perspective: P1 wants small d1, opp large d2
    score = (_W_DIST * (d2 - d1) + _W_WALLS * walls_diff
             + _W_ALT * (a1 - a2))
    return score if turn == 1 else -score


def dist_and_alt_pair(int p1, int g1, int p2, int g2, object mask):
    """dist_and_alt for both players in one boundary crossing:
    (d1, alt1, d2, alt2)."""
    cdef uint64_t blk[NW]
    cdef int d1, a1, d2, a2
    _split(mask, blk)
    _dist_alt(p1, g1, blk, &d1, &a1)
    _dist_alt(p2, g2, blk, &d2, &a2)
    return d1, a1, d2, a2


def order_moves(list moves, object tt_move, object k0, object k1,
                dict history, int me, int my_goal, object opp_path_mask,
                object mask):
    """Solver move ordering: same keys and stable descending order as the
    pure-Python sorted(key=score) in solver._order_moves.

    Wall indices are derived arithmetically from the (r, c, o) tuple, which
    relies on ALL_WALLS enumeration order (r, then c, then H before V).
    """
    cdef uint64_t blk[NW]
    cdef uint64_t opm[NW]
    cdef int n = len(moves)
    cdef long long packed[160]
    cdef int order[160]
    cdef int i, j, idx, r, c, dr, dc, w, tmpi
    cdef long long key, tmpv
    cdef uint8_t* dist
    cdef int my_dist_now, new_d, me_r, me_c
    cdef object mv, arg, hval
    if n == 0:
        return []
    if n > 160:  # 5 pawn + 128 wall max; guard against future rule changes
        raise ValueError("too many moves for order_moves")
    _split(mask, blk)
    _split(opp_path_mask, opm)
    dist = _dist_cached(my_goal, blk)
    my_dist_now = 10_000 if dist[me] == UNREACH else dist[me]
    me_r = me // NCOLS
    me_c = me % NCOLS
    for i in range(n):
        mv = moves[i]
        if tt_move is not None and mv == tt_move:
            key = 10 ** 9
        elif k0 is not None and mv == k0:
            key = 10 ** 8
        elif k1 is not None and mv == k1:
            key = 10 ** 8 - 1
        else:
            hval = history.get(mv)
            key = 0 if hval is None else <long long> hval
            arg = mv[1]
            if mv[0] == "m":
                # Pawn moves: prefer advancing along shortest path; jumps get
                # a modest bonus (selective-extension candidates).
                r = arg[0]
                c = arg[1]
                idx = r * NCOLS + c
                new_d = 10_000 if dist[idx] == UNREACH else dist[idx]
                key += 500 * (my_dist_now - new_d)
                dr = r - me_r
                dc = c - me_c
                if (dr if dr >= 0 else -dr) + (dc if dc >= 0 else -dc) >= 2:
                    key += 200
            else:
                # Walls: prefer walls that touch opponent's shortest path.
                r = arg[0]
                c = arg[1]
                w = ((r - 1) * 8 + c) * 2 + (1 if arg[2] == "V" else 0)
                if _hit(_WMASK[w], opm):
                    key += 300
        # Unique sort value: descending by key, ties by original position
        # (replicates Python's stable sort).
        packed[i] = key * 256 - i
    for i in range(n):
        order[i] = i
    for i in range(1, n):
        tmpi = order[i]
        tmpv = packed[tmpi]
        j = i - 1
        while j >= 0 and packed[order[j]] < tmpv:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = tmpi
    return [moves[order[i]] for i in range(n)]


def dist_table_bytes(int goal, object mask):
    """BFS dist table as 99 bytes (255 = unreachable), cell-index order."""
    cdef uint64_t blk[NW]
    _split(mask, blk)
    cdef uint8_t* dist = _dist_cached(goal, blk)
    return (<const char*> dist)[:NCELLS]


def shortest_dist(int pos, int goal, object mask):
    """Shortest distance pos -> goal, or None when unreachable."""
    cdef uint64_t blk[NW]
    _split(mask, blk)
    cdef uint8_t* dist = _dist_cached(goal, blk)
    if dist[pos] == UNREACH:
        return None
    return <int> dist[pos]


def has_path(int start, int goal, object mask):
    cdef uint64_t blk[NW]
    _split(mask, blk)
    return _has_path(start, goal, blk)


def shortest_path_mask(int pos, int goal, object mask):
    """Blockable-edge bitmask (Python int) of one shortest path pos -> goal."""
    cdef uint64_t blk[NW]
    cdef uint64_t out[NW]
    _split(mask, blk)
    _spath_mask(pos, goal, blk, out)
    return _join(out)
