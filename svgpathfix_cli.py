"""
Reads an SVG, finds path sub-segments whose endpoints nearly touch,
joins them by replacing close M (moveTo) commands with L (lineTo),
and simplifies jittery micro-segments.

Usage:
  python svgpathfix_cli.py [input.svg] [output.svg] [threshold] [--simplify N] [--tolerance N]

  threshold     max gap to auto-join (default 0.5)
  --simplify N  collapse runs of line segments shorter than N into single
                straight lines (default: off; try 0.15 - 0.5)
"""

import re
import xml.etree.ElementTree as ET
import math
import os
import sys

CMD_ARGS = {
    'M': 2, 'm': 2, 'L': 2, 'l': 2,
    'H': 1, 'h': 1, 'V': 1, 'v': 1,
    'C': 6, 'c': 6, 'S': 4, 's': 4,
    'Q': 4, 'q': 4, 'T': 2, 't': 2,
    'A': 7, 'a': 7, 'Z': 0, 'z': 0,
}
IMPLICIT_NEXT = {'M': 'L', 'm': 'l'}


def tokenize(d):
    return re.findall(
        r'[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', d
    )


def parse_commands(d):
    tokens = tokenize(d)
    cmds = []
    i, cur = 0, None
    while i < len(tokens):
        tok = tokens[i]
        if tok in CMD_ARGS:
            cur = tok
            i += 1
            n = CMD_ARGS[cur]
            args = [float(tokens[j]) for j in range(i, i + n)]
            i += n
            cmds.append((cur, args))
            if cur in IMPLICIT_NEXT:
                cur = IMPLICIT_NEXT[cur]
        else:
            n = CMD_ARGS[cur]
            args = [float(tokens[j]) for j in range(i, i + n)]
            i += n
            cmds.append((cur, args))
    return cmds


def get_endpoint(cmd, args, cx, cy):
    if cmd in ('M', 'L', 'T'):      return args[0], args[1]
    if cmd in ('m', 'l', 't'):      return cx + args[0], cy + args[1]
    if cmd == 'H':                   return args[0], cy
    if cmd == 'h':                   return cx + args[0], cy
    if cmd == 'V':                   return cx, args[0]
    if cmd == 'v':                   return cx, cy + args[0]
    if cmd == 'C':                   return args[4], args[5]
    if cmd == 'c':                   return cx + args[4], cy + args[5]
    if cmd in ('S', 'Q'):            return args[2], args[3]
    if cmd in ('s', 'q'):            return cx + args[2], cy + args[3]
    if cmd == 'A':                   return args[5], args[6]
    if cmd == 'a':                   return cx + args[5], cy + args[6]
    return cx, cy


def dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def fmt(v):
    s = f'{v:.4f}'.rstrip('0').rstrip('.')
    return s


def cmds_to_str(cmds):
    parts = []
    for cmd, args in cmds:
        if args:
            parts.append(cmd + ' ' + ' '.join(fmt(a) for a in args))
        else:
            parts.append(cmd)
    return ''.join(parts)


# ── Step 1: threshold-based join (fix micro-gaps) ───────────────────────

def join_by_threshold(cmds, threshold):
    """Replace M with L when the gap from current pos is <= threshold."""
    cx, cy = 0.0, 0.0
    out = []
    joined = 0

    for i, (cmd, args) in enumerate(cmds):
        if cmd == 'M':
            mx, my = args[0], args[1]
            if i == 0:
                out.append(('M', args))
            elif dist(cx, cy, mx, my) <= threshold:
                out.append(('L', args))
                joined += 1
            else:
                out.append(('M', args))
            cx, cy = mx, my
        elif cmd == 'm':
            mx, my = cx + args[0], cy + args[1]
            if i == 0:
                out.append(('m', args))
            elif dist(cx, cy, mx, my) <= threshold:
                out.append(('l', args))
                joined += 1
            else:
                out.append(('m', args))
            cx, cy = mx, my
        elif cmd in ('Z', 'z'):
            out.append((cmd, args))
        else:
            out.append((cmd, args))
            cx, cy = get_endpoint(cmd, args, cx, cy)

    return out, joined


# ── Step 2: trace subpaths as graph edges (reverse where needed) ────────

def split_into_subpaths(cmds):
    subpaths, current = [], []
    for cmd, args in cmds:
        if cmd in ('M', 'm') and current:
            subpaths.append(current)
            current = []
        current.append((cmd, args))
    if current:
        subpaths.append(current)
    return subpaths


def subpath_endpoints(sp):
    cx, cy, sx, sy = 0.0, 0.0, 0.0, 0.0
    for i, (cmd, args) in enumerate(sp):
        if cmd == 'M':
            cx, cy = args[0], args[1]
            if i == 0: sx, sy = cx, cy
        elif cmd == 'm':
            cx, cy = cx + args[0], cy + args[1]
            if i == 0: sx, sy = cx, cy
        elif cmd in ('Z', 'z'):
            cx, cy = sx, sy
        else:
            cx, cy = get_endpoint(cmd, args, cx, cy)
    return (sx, sy), (cx, cy)


def reverse_subpath(sp):
    """
    Reverse a subpath so it goes from its old end to its old start.
    Handles L/H/V/C/S/Q/A commands in absolute form.
    Returns a new command list starting with M at the old endpoint.
    """
    # First, compute all the waypoints (absolute positions after each command)
    points = []
    cx, cy = 0.0, 0.0
    prev_c2x, prev_c2y = 0.0, 0.0  # second control point of previous C (for S)

    for i, (cmd, args) in enumerate(sp):
        if cmd == 'M':
            cx, cy = args[0], args[1]
            prev_c2x, prev_c2y = cx, cy
            points.append(('M', cx, cy, None))
        elif cmd in ('L', 'H', 'V'):
            ex, ey = get_endpoint(cmd, args, cx, cy)
            prev_c2x, prev_c2y = ex, ey
            points.append(('L', ex, ey, None))
            cx, cy = ex, ey
        elif cmd == 'C':
            # C x1 y1 x2 y2 x y — cubic bezier
            points.append(('C', args[4], args[5],
                          (cx, cy, args[0], args[1], args[2], args[3], args[4], args[5])))
            prev_c2x, prev_c2y = args[2], args[3]
            cx, cy = args[4], args[5]
        elif cmd == 'S':
            # S x2 y2 x y — smooth cubic, first control point is reflection of prev C's c2
            c1x = 2 * cx - prev_c2x
            c1y = 2 * cy - prev_c2y
            c2x, c2y = args[0], args[1]
            ex, ey = args[2], args[3]
            # Store as full cubic data for correct reversal
            points.append(('C', ex, ey,
                          (cx, cy, c1x, c1y, c2x, c2y, ex, ey)))
            prev_c2x, prev_c2y = c2x, c2y
            cx, cy = ex, ey
        elif cmd == 'Q':
            # Q x1 y1 x y — quadratic bezier
            qx, qy = args[0], args[1]
            ex, ey = args[2], args[3]
            points.append(('Q', ex, ey,
                          (cx, cy, qx, qy, ex, ey)))
            prev_c2x, prev_c2y = ex, ey
            cx, cy = ex, ey
        elif cmd == 'A':
            # A rx ry x-rot large-arc sweep x y
            ex, ey = args[5], args[6]
            points.append(('A', ex, ey,
                          (args[0], args[1], args[2], args[3], args[4], ex, ey)))
            prev_c2x, prev_c2y = ex, ey
            cx, cy = ex, ey
        else:
            # Relative or other commands: convert to absolute line
            ex, ey = get_endpoint(cmd, args, cx, cy)
            prev_c2x, prev_c2y = ex, ey
            points.append(('L', ex, ey, None))
            cx, cy = ex, ey

    if len(points) < 2:
        return sp

    # Reverse: start with M at old endpoint, then trace backwards
    rev = [('M', [points[-1][1], points[-1][2]])]

    for j in range(len(points) - 1, 0, -1):
        typ = points[j][0]
        prev_x, prev_y = points[j - 1][1], points[j - 1][2]

        if typ == 'C' and points[j][3] is not None:
            # Reverse cubic: swap control points
            # Original: from (sx,sy) via (c1x,c1y) (c2x,c2y) to (ex,ey)
            # Reversed: from (ex,ey) via (c2x,c2y) (c1x,c1y) to (sx,sy)
            sx, sy, c1x, c1y, c2x, c2y, ex, ey = points[j][3]
            rev.append(('C', [c2x, c2y, c1x, c1y, prev_x, prev_y]))
        elif typ == 'Q' and points[j][3] is not None:
            # Reverse quadratic: same control point, swap start/end
            sx, sy, qx, qy, ex, ey = points[j][3]
            rev.append(('Q', [qx, qy, prev_x, prev_y]))
        elif typ == 'A' and points[j][3] is not None:
            # Reverse arc: same radii and rotation, flip sweep flag
            rx, ry, x_rot, large_arc, sweep, ex, ey = points[j][3]
            rev.append(('A', [rx, ry, x_rot, large_arc, 1 - sweep, prev_x, prev_y]))
        else:
            rev.append(('L', [prev_x, prev_y]))

    return rev


def trace_graph(cmds, tol=0.1):
    """
    Treat each subpath as an undirected edge between its start and end vertices.
    Build a graph using union-find to merge nearby vertices, then trace continuous
    paths through it, reversing subpaths as needed.
    """
    subpaths = split_into_subpaths(cmds)
    if len(subpaths) <= 1:
        return cmds, 0

    # Collect all endpoint coordinates
    raw_pts = []  # list of (x, y, subpath_index, is_end)
    for i, sp in enumerate(subpaths):
        start, end = subpath_endpoints(sp)
        raw_pts.append((start[0], start[1], i, False))
        raw_pts.append((end[0], end[1], i, True))

    # Union-find to merge points within tolerance
    parent = list(range(len(raw_pts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    # Merge nearby points (O(n^2) but n is small ~1000)
    for i in range(len(raw_pts)):
        for j in range(i + 1, len(raw_pts)):
            if raw_pts[i][2] == raw_pts[j][2]:
                continue  # skip same subpath's start/end
            d = dist(raw_pts[i][0], raw_pts[i][1], raw_pts[j][0], raw_pts[j][1])
            if d <= tol:
                union(i, j)

    # Build vertex IDs from union-find roots
    # Each subpath has a start_vertex and end_vertex
    sp_start_v = {}  # subpath_index -> vertex_id
    sp_end_v = {}
    for idx, (x, y, sp_i, is_end) in enumerate(raw_pts):
        v = find(idx)
        if is_end:
            sp_end_v[sp_i] = v
        else:
            sp_start_v[sp_i] = v

    # Build adjacency: vertex -> [(edge_index, other_vertex)]
    adj = {}
    for i in range(len(subpaths)):
        sv = sp_start_v[i]
        ev = sp_end_v[i]
        adj.setdefault(sv, []).append((i, ev))
        adj.setdefault(ev, []).append((i, sv))

    # Trace paths greedily through the graph
    used_edges = set()
    traces = []

    # Start from degree-1 vertices (chain endpoints) first
    degree = {v: len(neighbors) for v, neighbors in adj.items()}
    start_vertices = sorted([v for v, d in degree.items() if d == 1])
    all_vertices = sorted(adj.keys())

    for start_v in start_vertices + all_vertices:
        if all(ei in used_edges for ei, _ in adj.get(start_v, [])):
            continue

        trace = []
        current_v = start_v

        while True:
            found = False
            for edge_idx, other_v in adj.get(current_v, []):
                if edge_idx not in used_edges:
                    used_edges.add(edge_idx)
                    sv = sp_start_v[edge_idx]
                    # Determine direction
                    if sv == current_v:
                        trace.append((edge_idx, False))
                        current_v = sp_end_v[edge_idx]
                    else:
                        trace.append((edge_idx, True))
                        current_v = sp_start_v[edge_idx]
                    found = True
                    break
            if not found:
                break

        if trace:
            traces.append(trace)

    # Build output commands
    out = []
    merged = 0

    for trace in traces:
        for ti, (sp_idx, rev) in enumerate(trace):
            sp = subpaths[sp_idx]
            if rev:
                sp = reverse_subpath(sp)

            if ti == 0:
                out.extend(sp)
            else:
                first_cmd, first_args = sp[0]
                out.append(('L', first_args))
                out.extend(sp[1:])
                merged += 1

    return out, merged


# ── Step 3: deduplicate near-coincident endpoints ────────────────────────

LINE_CMDS = {'L', 'l', 'H', 'h', 'V', 'v'}


def deduplicate_endpoints(cmds, tol=0.01):
    """
    Remove line segments that move the pen by less than `tol`.
    These are coordinate-rounding artifacts where two consecutive commands
    end at nearly the same point.

    When a near-zero segment is found, its endpoint (the "corrected" position)
    replaces the previous command's endpoint, and the tiny segment is dropped.
    """
    cx, cy = 0.0, 0.0
    out = []
    removed = 0

    for cmd, args in cmds:
        if cmd in LINE_CMDS:
            ex, ey = get_endpoint(cmd, args, cx, cy)
            seg_len = dist(cx, cy, ex, ey)

            if seg_len < tol and out:
                prev_cmd = out[-1][0]
                if prev_cmd in LINE_CMDS:
                    # Update previous line's endpoint to this corrected position
                    out[-1] = ('L', [ex, ey])
                    removed += 1
                elif prev_cmd == 'C':
                    # Update cubic's endpoint (last 2 args) keeping control points
                    prev_args = list(out[-1][1])
                    prev_args[4] = ex
                    prev_args[5] = ey
                    out[-1] = ('C', prev_args)
                    removed += 1
                elif prev_cmd == 'M':
                    # Update the M command's position
                    out[-1] = ('M', [ex, ey])
                    removed += 1
                else:
                    # Can't easily patch other commands; keep the segment
                    out.append(('L', [ex, ey]))
                cx, cy = ex, ey
            else:
                out.append((cmd, args))
                cx, cy = ex, ey
        else:
            out.append((cmd, args))
            if cmd == 'M':
                cx, cy = args[0], args[1]
            elif cmd == 'm':
                cx, cy = cx + args[0], cy + args[1]
            elif cmd not in ('Z', 'z'):
                cx, cy = get_endpoint(cmd, args, cx, cy)

    return out, removed


# ── Step 4: simplify runs of short line segments ────────────────────────


def simplify_short_runs(cmds, max_seg_len):
    """
    Find consecutive runs of short line segments (L/H/V with length <= max_seg_len).
    Replace each run of 1+ segments with a single L from the run's start to end.
    A single short segment is also absorbed (its endpoint replaces the previous
    long segment's endpoint, eliminating tiny jitter after long lines).
    """
    cx, cy = 0.0, 0.0
    out = []
    run_end_x, run_end_y = 0.0, 0.0
    run_count = 0
    segments_removed = 0

    def flush_run():
        nonlocal run_count, segments_removed
        if run_count >= 2:
            out.append(('L', [run_end_x, run_end_y]))
            segments_removed += run_count - 1
        elif run_count == 1:
            # Single short segment after a long one: absorb by updating
            # the previous command's endpoint if it was a line
            if out and out[-1][0] in LINE_CMDS:
                # Replace previous line's endpoint with this one
                out[-1] = ('L', [run_end_x, run_end_y])
                segments_removed += 1
            else:
                out.append(('L', [run_end_x, run_end_y]))
        run_count = 0

    for cmd, args in cmds:
        if cmd in LINE_CMDS:
            ex, ey = get_endpoint(cmd, args, cx, cy)
            seg_len = dist(cx, cy, ex, ey)

            if seg_len <= max_seg_len:
                run_end_x, run_end_y = ex, ey
                run_count += 1
                cx, cy = ex, ey
                continue
            else:
                flush_run()
                out.append((cmd, args))
                cx, cy = ex, ey
        else:
            flush_run()
            out.append((cmd, args))
            if cmd == 'M':
                cx, cy = args[0], args[1]
            elif cmd == 'm':
                cx, cy = cx + args[0], cy + args[1]
            elif cmd not in ('Z', 'z'):
                cx, cy = get_endpoint(cmd, args, cx, cy)

    flush_run()
    return out, segments_removed


# ── Step 5: close near-closed subpaths ────────────────────────────────


def close_subpaths(cmds, tol=0.01):
    """
    If a subpath's end point is within `tol` of its start point,
    snap the M to match the endpoint and append Z to formally close it.
    """
    subpaths = split_into_subpaths(cmds)
    out = []
    closed = 0

    for sp in subpaths:
        start, end = subpath_endpoints(sp)
        gap = dist(start[0], start[1], end[0], end[1])

        if gap > 0 and gap < tol:
            # Snap M to the endpoint
            sp[0] = ('M', [end[0], end[1]])
            # Append Z if not already there
            last_cmd = sp[-1][0]
            if last_cmd not in ('Z', 'z'):
                sp.append(('Z', []))
            closed += 1

        out.extend(sp)

    return out, closed


# ── Pipeline ────────────────────────────────────────────────────────────

def process_path(d, threshold=0.5, simplify=None, tol=0.1):
    cmds = parse_commands(d)
    total_m = sum(1 for c, _ in cmds[1:] if c in ('M', 'm'))
    print(f"  Total sub-path breaks (M commands after first): {total_m}")

    # Step 1: fix micro-gaps
    cmds, micro_joined = join_by_threshold(cmds, threshold)
    print(f"  Micro-gaps joined (dist <= {threshold}): {micro_joined}")

    remaining = sum(1 for c, _ in cmds[1:] if c in ('M', 'm'))
    print(f"  Remaining separate subpaths: {remaining}")

    # Step 2: trace graph — merge subpaths sharing endpoints (reversing as needed)
    cmds, traced = trace_graph(cmds, tol=tol)
    remaining2 = sum(1 for c, _ in cmds[1:] if c in ('M', 'm'))
    print(f"  Graph-traced (reversing where needed): {traced} merged ({remaining} -> {remaining2})")

    # Step 3: deduplicate near-coincident endpoints (rounding artifacts)
    before3 = len(cmds)
    cmds, deduped = deduplicate_endpoints(cmds)
    if deduped:
        print(f"  Near-duplicate endpoints removed: {deduped} "
              f"({before3} -> {len(cmds)} commands)")

    # Step 4: simplify jittery short segments
    if simplify is not None and simplify > 0:
        before = len(cmds)
        cmds, removed = simplify_short_runs(cmds, simplify)
        print(f"  Short-segment runs simplified (seg <= {simplify}): "
              f"{removed} segments removed ({before} -> {len(cmds)} commands)")

    # Step 5: close near-closed subpaths
    cmds, closed_count = close_subpaths(cmds)
    if closed_count:
        print(f"  Subpaths closed (start~=end): {closed_count}")

    return cmds_to_str(cmds)


def _parse_flag(flags, name, default=None):
    """Parse a --name N or --name=N flag from the flag list."""
    for f in flags:
        if f.startswith(f'--{name}'):
            if '=' in f:
                return float(f.split('=')[1])
            else:
                idx = sys.argv.index(f)
                if idx + 1 < len(sys.argv):
                    try:
                        return float(sys.argv[idx + 1])
                    except ValueError:
                        return default
                else:
                    return default
    return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]

    if len(args) < 1:
        print("Usage: python svgpathfix_cli.py <input.svg> [output.svg] [threshold]"
              " [--simplify N] [--tolerance N]")
        sys.exit(1)

    src = args[0]
    dst = args[1] if len(args) > 1 else os.path.splitext(src)[0] + '_fixed.svg'
    threshold = float(args[2]) if len(args) > 2 else 0.5

    simplify = _parse_flag(flags, 'simplify', default=1.0)
    tolerance = _parse_flag(flags, 'tolerance', default=0.1)
    if tolerance is None:
        tolerance = 0.1

    if not os.path.isfile(src):
        print(f"Error: file not found: {src}")
        sys.exit(1)

    print(f"Input    : {src}")
    print(f"Output   : {dst}")
    print(f"Threshold: {threshold}")
    print(f"Simplify : {simplify}")
    print(f"Tolerance: {tolerance}\n")

    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    tree = ET.parse(src)
    root = tree.getroot()

    paths = root.findall('.//{http://www.w3.org/2000/svg}path')
    if not paths:
        paths = root.findall('.//path')

    print(f"Found {len(paths)} <path> element(s)\n")

    for idx, path_el in enumerate(paths):
        d = path_el.get('d', '')
        if not d:
            continue
        print(f"Path #{idx + 1}:")
        new_d = process_path(d, threshold, simplify, tol=tolerance)
        path_el.set('d', new_d)
        print()

    tree.write(dst, xml_declaration=False, encoding='unicode')
    print(f"Wrote: {dst}")


if __name__ == '__main__':
    main()
