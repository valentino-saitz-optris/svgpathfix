"""
Tkinter GUI for SVG Path Fixer.

Loads an SVG, auto-analyzes gap, segment, and endpoint distributions to
recommend parameters, then processes the path using fix_path's 5-step pipeline.
Displays interactive histograms with log-scale sliders for picking values.
"""

import math
import multiprocessing
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import xml.etree.ElementTree as ET

# Ensure fix_path.py can be imported from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fix_path import (
    parse_commands, get_endpoint, dist, cmds_to_str,
    join_by_threshold, trace_graph, deduplicate_endpoints,
    simplify_short_runs, close_subpaths, split_into_subpaths,
    subpath_endpoints,
)

# Optional drag & drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

# Optional matplotlib for interactive charts
try:
    import numpy as np
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

LINE_CMDS = {'L', 'l', 'H', 'h', 'V', 'v'}

SLIDER_STEPS = 1000
COLOR_BELOW = '#4A90D9'
COLOR_ABOVE = '#E8913A'
COLOR_LINE = '#D32F2F'

PREVIEW_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

# Modern UI theme constants
BG_MAIN = '#F0F2F5'
BG_TOOLBAR = '#2C3E50'
FG_TOOLBAR = '#ECF0F1'
ACCENT = '#3498DB'
BG_STATUS = '#34495E'
FG_STATUS = '#BDC3C7'
FG_WELCOME = '#7F8C8D'
FONT_MAIN = ('Segoe UI', 10)
FONT_MONO = ('Consolas', 9)
FONT_SMALL = ('Segoe UI', 8)
FONT_WELCOME_LARGE = ('Segoe UI', 22)
FONT_WELCOME_SUB = ('Segoe UI', 11)


# ── Preview subprocess helpers ────────────────────────────────────────

def _cubic_bezier_points(x0, y0, x1, y1, x2, y2, x3, y3, n=8):
    """Approximate a cubic bezier with n line segments via De Casteljau."""
    pts = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        x = u*u*u*x0 + 3*u*u*t*x1 + 3*u*t*t*x2 + t*t*t*x3
        y = u*u*u*y0 + 3*u*u*t*y1 + 3*u*t*t*y2 + t*t*t*y3
        pts.append((x, y))
    return pts


def _flatten_subpath(cmds):
    """Convert a subpath's commands into a list of (x, y) points for drawing."""
    pts = []
    cx, cy = 0.0, 0.0
    sx, sy = 0.0, 0.0  # subpath start

    for cmd, args in cmds:
        if cmd == 'M':
            cx, cy = args[0], args[1]
            sx, sy = cx, cy
            pts.append((cx, cy))
        elif cmd == 'm':
            cx, cy = cx + args[0], cy + args[1]
            sx, sy = cx, cy
            pts.append((cx, cy))
        elif cmd == 'L':
            cx, cy = args[0], args[1]
            pts.append((cx, cy))
        elif cmd == 'l':
            cx, cy = cx + args[0], cy + args[1]
            pts.append((cx, cy))
        elif cmd == 'H':
            cx = args[0]
            pts.append((cx, cy))
        elif cmd == 'h':
            cx = cx + args[0]
            pts.append((cx, cy))
        elif cmd == 'V':
            cy = args[0]
            pts.append((cx, cy))
        elif cmd == 'v':
            cy = cy + args[0]
            pts.append((cx, cy))
        elif cmd == 'C':
            bezier = _cubic_bezier_points(
                cx, cy, args[0], args[1], args[2], args[3], args[4], args[5])
            pts.extend(bezier[1:])  # skip first (=current pos)
            cx, cy = args[4], args[5]
        elif cmd == 'c':
            bezier = _cubic_bezier_points(
                cx, cy, cx+args[0], cy+args[1],
                cx+args[2], cy+args[3], cx+args[4], cy+args[5])
            pts.extend(bezier[1:])
            cx, cy = cx + args[4], cy + args[5]
        elif cmd == 'S':
            # Smooth cubic — use endpoint as control point (simplified)
            bezier = _cubic_bezier_points(
                cx, cy, cx, cy, args[0], args[1], args[2], args[3])
            pts.extend(bezier[1:])
            cx, cy = args[2], args[3]
        elif cmd == 's':
            bezier = _cubic_bezier_points(
                cx, cy, cx, cy, cx+args[0], cy+args[1], cx+args[2], cy+args[3])
            pts.extend(bezier[1:])
            cx, cy = cx + args[2], cy + args[3]
        elif cmd == 'Q':
            # Quadratic → approximate as cubic
            qx1, qy1, qx2, qy2 = args[0], args[1], args[2], args[3]
            cx1 = cx + 2/3 * (qx1 - cx)
            cy1 = cy + 2/3 * (qy1 - cy)
            cx2 = qx2 + 2/3 * (qx1 - qx2)
            cy2 = qy2 + 2/3 * (qy1 - qy2)
            bezier = _cubic_bezier_points(cx, cy, cx1, cy1, cx2, cy2, qx2, qy2)
            pts.extend(bezier[1:])
            cx, cy = qx2, qy2
        elif cmd == 'q':
            qx1, qy1, qx2, qy2 = cx+args[0], cy+args[1], cx+args[2], cy+args[3]
            cx1 = cx + 2/3 * (qx1 - cx)
            cy1 = cy + 2/3 * (qy1 - cy)
            cx2 = qx2 + 2/3 * (qx1 - qx2)
            cy2 = qy2 + 2/3 * (qy1 - qy2)
            bezier = _cubic_bezier_points(cx, cy, cx1, cy1, cx2, cy2, qx2, qy2)
            pts.extend(bezier[1:])
            cx, cy = qx2, qy2
        elif cmd in ('Z', 'z'):
            if (cx, cy) != (sx, sy):
                pts.append((sx, sy))
            cx, cy = sx, sy
        else:
            # T, t, A, a — approximate with line to endpoint
            ex, ey = get_endpoint(cmd, args, cx, cy)
            pts.append((ex, ey))
            cx, cy = ex, ey

    return pts


def _compute_bounds(subpaths_points):
    """Compute (min_x, min_y, max_x, max_y) from list of point lists."""
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    for pts in subpaths_points:
        for x, y in pts:
            if x < min_x: min_x = x
            if y < min_y: min_y = y
            if x > max_x: max_x = x
            if y > max_y: max_y = y
    if min_x == float('inf'):
        return 0, 0, 1, 1
    # Ensure non-zero dimensions
    if max_x == min_x: max_x = min_x + 1
    if max_y == min_y: max_y = min_y + 1
    return min_x, min_y, max_x, max_y


def _process_cmds(group_cmds_list, threshold, simplify, graph_tol):
    """Run 5-step pipeline on each group's commands. Returns list of subpath cmd lists.

    group_cmds_list: list of cmd lists (one per visual-attribute group).
    This function runs in the subprocess.
    """
    all_subpaths = []
    for cmds in group_cmds_list:
        cmds, _ = join_by_threshold(cmds, threshold)
        cmds, _ = trace_graph(cmds, tol=graph_tol)
        cmds, _ = deduplicate_endpoints(cmds)
        if simplify is not None and simplify > 0:
            cmds, _ = simplify_short_runs(cmds, simplify)
        cmds, _ = close_subpaths(cmds)
        all_subpaths.extend(split_into_subpaths(cmds))
    return all_subpaths


def _preview_worker(in_q, out_q):
    """Long-lived worker thread for preview pipeline.

    Blocks until a request arrives, then drains the queue to skip to the
    latest request before processing. Only the most recent request is processed.
    """
    while True:
        msg = in_q.get()
        if msg is None:
            break
        # Drain to latest — discard everything except the newest
        while True:
            try:
                newer = in_q.get_nowait()
                if newer is None:
                    return
                msg = newer
            except Exception:
                break
        group_cmds_list, threshold, simplify, graph_tol = msg
        try:
            result = _process_cmds(group_cmds_list, threshold, simplify, graph_tol)
            out_q.put(result)
        except Exception:
            pass


# ── Auto-analysis functions ───────────────────────────────────────────

def _find_cluster_boundary(values):
    """Find the index of the largest gap between consecutive sorted values.

    Uses log-scale jumps so it works across orders of magnitude.
    Falls back to absolute jumps if all values are zero/negative.
    Returns the index i such that the split is values[:i+1] | values[i+1:].
    """
    min_nonzero = None
    for v in values:
        if v > 0:
            min_nonzero = v
            break

    best_jump, best_idx = 0, 0
    for i in range(len(values) - 1):
        a, b = values[i], values[i + 1]
        if min_nonzero and a > 0 and b > 0:
            jump = math.log(b) - math.log(a)
        else:
            jump = b - a
        if jump > best_jump:
            best_jump = jump
            best_idx = i

    return best_idx


def analyze_gap_distribution(cmds):
    """Analyze M-command gap distances and recommend a join threshold."""
    gaps = []
    cx, cy = 0.0, 0.0
    for i, (cmd, args) in enumerate(cmds):
        if cmd == 'M' and i > 0:
            gaps.append(dist(cx, cy, args[0], args[1]))
            cx, cy = args[0], args[1]
        elif cmd == 'm' and i > 0:
            mx, my = cx + args[0], cy + args[1]
            gaps.append(dist(cx, cy, mx, my))
            cx, cy = mx, my
        elif cmd in ('Z', 'z'):
            pass
        else:
            cx, cy = get_endpoint(cmd, args, cx, cy)

    if len(gaps) < 2:
        return None

    gaps.sort()
    boundary_idx = _find_cluster_boundary(gaps)
    boundary = (gaps[boundary_idx] + gaps[boundary_idx + 1]) / 2
    rec = gaps[boundary_idx] * 1.05

    micro = [g for g in gaps if g <= boundary]
    structural = [g for g in gaps if g > boundary]

    return {
        'gaps': gaps,
        'micro_count': len(micro),
        'structural_count': len(structural),
        'micro_max': micro[-1] if micro else 0,
        'structural_min': structural[0] if structural else 0,
        'boundary': boundary,
        'recommended_threshold': round(rec, 6),
    }


def analyze_segment_distribution(cmds):
    """Analyze original line segment lengths and recommend simplify."""
    segments = []
    cx, cy = 0.0, 0.0
    for cmd, args in cmds:
        if cmd in LINE_CMDS:
            ex, ey = get_endpoint(cmd, args, cx, cy)
            segments.append(dist(cx, cy, ex, ey))
            cx, cy = ex, ey
        elif cmd == 'M':
            cx, cy = args[0], args[1]
        elif cmd == 'm':
            cx, cy = cx + args[0], cy + args[1]
        elif cmd not in ('Z', 'z'):
            cx, cy = get_endpoint(cmd, args, cx, cy)

    if len(segments) < 2:
        return None

    segments.sort()
    boundary_idx = _find_cluster_boundary(segments)
    boundary = (segments[boundary_idx] + segments[boundary_idx + 1]) / 2
    rec = segments[boundary_idx] * 1.05

    jitter = [s for s in segments if s <= boundary]
    structural = [s for s in segments if s > boundary]

    return {
        'segments': segments,
        'jitter_count': len(jitter),
        'structural_count': len(structural),
        'jitter_max': jitter[-1] if jitter else 0,
        'structural_min': structural[0] if structural else 0,
        'boundary': boundary,
        'recommended_simplify': round(rec, 6),
    }


def analyze_endpoint_distribution(cmds):
    """Nearest-neighbor distances between subpath endpoints.

    For each subpath endpoint, find the distance to the closest endpoint
    on a different subpath. This measures how well trace_graph can merge
    subpaths at a given tolerance.
    """
    subpaths = split_into_subpaths(cmds)
    if len(subpaths) < 2:
        return None

    # Collect all endpoints: (x, y, subpath_index)
    pts = []
    for i, sp in enumerate(subpaths):
        start, end = subpath_endpoints(sp)
        pts.append((start[0], start[1], i))
        pts.append((end[0], end[1], i))

    # For each point, find min distance to any point from a different subpath
    nn_dists = []
    for i, (x1, y1, sp1) in enumerate(pts):
        min_d = float('inf')
        for j, (x2, y2, sp2) in enumerate(pts):
            if sp1 == sp2:
                continue
            d = dist(x1, y1, x2, y2)
            if d < min_d:
                min_d = d
        if min_d < float('inf'):
            nn_dists.append(min_d)

    if len(nn_dists) < 2:
        return None

    nn_dists.sort()
    boundary_idx = _find_cluster_boundary(nn_dists)
    boundary = (nn_dists[boundary_idx] + nn_dists[boundary_idx + 1]) / 2
    rec = nn_dists[boundary_idx] * 1.05

    matched = [d for d in nn_dists if d <= boundary]
    unmatched = [d for d in nn_dists if d > boundary]

    return {
        'distances': nn_dists,
        'matched_count': len(matched),
        'unmatched_count': len(unmatched),
        'matched_max': matched[-1] if matched else 0,
        'unmatched_min': unmatched[0] if unmatched else 0,
        'boundary': boundary,
        'recommended_tolerance': round(rec, 6),
    }


def _group_paths(paths):
    """Group path elements by visual attributes (everything except 'd').

    Returns list of (attrs_dict, combined_cmds, path_elements).
    Paths with identical non-d attributes are merged into one command list.
    """
    groups = {}  # key -> (attrs_dict, [cmds...], [elements...])
    for p in paths:
        attrs = dict(p.attrib)
        d = attrs.pop('d', '')
        if not d:
            continue
        key = tuple(sorted(attrs.items()))
        if key not in groups:
            groups[key] = (attrs, [], [])
        groups[key][1].extend(parse_commands(d))
        groups[key][2].append(p)
    return list(groups.values())


def auto_analyze_svg(svg_path):
    """Full auto-analysis on an SVG file. Returns analysis dict."""
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    tree = ET.parse(svg_path)
    root = tree.getroot()

    paths = root.findall('.//{http://www.w3.org/2000/svg}path')
    if not paths:
        paths = root.findall('.//path')

    if not paths:
        return {'error': 'No <path> elements found in SVG.'}

    # Group paths by visual attributes and merge commands within each group
    groups = _group_paths(paths)
    if not groups:
        return {'error': 'No <path> elements with d attributes found.'}

    # Combine all commands across all groups for analysis
    cmds = []
    for _attrs, group_cmds, _elems in groups:
        cmds.extend(group_cmds)

    subpaths = split_into_subpaths(cmds)

    gap_analysis = analyze_gap_distribution(cmds)
    seg_analysis = analyze_segment_distribution(cmds)

    # Endpoint analysis runs on gap-joined commands (trace_graph's input)
    if gap_analysis:
        joined_cmds, _ = join_by_threshold(cmds, gap_analysis['recommended_threshold'])
    else:
        joined_cmds = cmds
    endpoint_analysis = analyze_endpoint_distribution(joined_cmds)

    # Extract raw cmd lists per group for preview subprocess
    group_cmds_list = [group_cmds for _attrs, group_cmds, _elems in groups]

    return {
        'tree': tree,
        'path_count': len(paths),
        'command_count': len(cmds),
        'subpath_count': len(subpaths),
        'group_count': len(groups),
        'group_cmds': group_cmds_list,
        'gap_analysis': gap_analysis,
        'segment_analysis': seg_analysis,
        'endpoint_analysis': endpoint_analysis,
    }


# ── GUI ───────────────────────────────────────────────────────────────

class SVGPathFixerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SVG Path Fixer")
        if MPL_AVAILABLE:
            self.root.geometry("1200x850")
            self.root.minsize(900, 650)
        else:
            self.root.geometry("820x750")
            self.root.minsize(600, 550)

        self.current_file = None
        self.analysis = None
        # Current parameter values (set by sliders)
        self.threshold = 0.5
        self.simplify = 0.5
        self.graph_tol = 0.1
        self.simplify_enabled = True

        # Chart state
        self.gap_data = None
        self.seg_data = None
        self.tol_data = None
        self.gap_slider_range = (1e-6, 200.0)
        self.seg_slider_range = (1e-6, 200.0)
        self.tol_slider_range = (1e-6, 200.0)
        self._updating_threshold = False
        self._updating_simplify = False
        self._updating_tol = False

        # Chart artist references
        self.gap_bars = None
        self.gap_vline = None
        self.gap_left_txt = None
        self.gap_right_txt = None
        self.gap_bin_centers = None

        self.seg_bars = None
        self.seg_vline = None
        self.seg_left_txt = None
        self.seg_right_txt = None
        self.seg_bin_centers = None

        self.tol_bars = None
        self.tol_vline = None
        self.tol_left_txt = None
        self.tol_right_txt = None
        self.tol_bin_centers = None

        # Preview state
        self._raw_group_cmds = None   # list of cmd lists per group (set on analysis)
        self._preview_last_params = None
        self._preview_poll_id = None
        self._preview_pts = None      # list of [(x,y)...] per subpath
        self._preview_bounds = None   # (min_x, min_y, max_x, max_y)
        self._view_zoom = 1.0
        self._view_pan_x = 0.0
        self._view_pan_y = 0.0
        self._drag_start = None
        self._drag_pan_start = None
        self._preview_in_q = queue.Queue()
        self._preview_out_q = queue.Queue()
        self._preview_thread = threading.Thread(
            target=_preview_worker,
            args=(self._preview_in_q, self._preview_out_q),
            daemon=True,
        )
        self._preview_thread.start()

        self.setup_ui()

        # Start polling for preview results
        self._poll_preview()

        # Clean shutdown
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Log-scale slider helpers ─────────────────────────────────

    @staticmethod
    def _slider_to_value(pos, vmin, vmax):
        """Convert slider int position [0..SLIDER_STEPS] to log-scale value."""
        log_min = math.log10(max(vmin, 1e-12))
        log_max = math.log10(max(vmax, 1e-12))
        log_val = log_min + (log_max - log_min) * (pos / SLIDER_STEPS)
        return 10 ** log_val

    @staticmethod
    def _value_to_slider(value, vmin, vmax):
        """Convert real value to slider int position [0..SLIDER_STEPS]."""
        log_min = math.log10(max(vmin, 1e-12))
        log_max = math.log10(max(vmax, 1e-12))
        log_val = math.log10(max(value, 1e-12))
        if log_max == log_min:
            return SLIDER_STEPS // 2
        pos = (log_val - log_min) / (log_max - log_min) * SLIDER_STEPS
        return int(max(0, min(SLIDER_STEPS, pos)))

    # ── UI setup ──────────────────────────────────────────────────

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background=BG_MAIN, font=FONT_MAIN)
        style.configure('Toolbar.TFrame', background=BG_TOOLBAR)
        style.configure('Toolbar.TLabel', background=BG_TOOLBAR,
                        foreground=FG_TOOLBAR, font=FONT_MAIN)
        style.configure('Toolbar.TButton', font=FONT_MAIN, padding=(10, 4))
        style.configure('TLabelframe', background=BG_MAIN)
        style.configure('TLabelframe.Label', background=BG_MAIN,
                        font=('Segoe UI', 9, 'bold'))
        style.configure('Status.TFrame', background=BG_STATUS)
        style.configure('Status.TLabel', background=BG_STATUS,
                        foreground=FG_STATUS, font=FONT_MONO)
        self.root.configure(bg=BG_MAIN)

    def setup_ui(self):
        self._setup_style()
        pad = {'padx': 8, 'pady': 4}

        self._build_toolbar()
        self._build_status_bar()

        # Main content: charts left, preview right
        self.main_paned = ttk.PanedWindow(self.root, orient='horizontal')
        self.main_paned.pack(fill='both', expand=True, **pad)

        if MPL_AVAILABLE:
            self._setup_charts_panel()
        else:
            self._setup_text_analysis_panel()

        self._build_preview_panel()
        self._build_welcome_overlay()

    def _build_toolbar(self):
        toolbar = ttk.Frame(self.root, style='Toolbar.TFrame')
        toolbar.pack(fill='x')

        inner = ttk.Frame(toolbar, style='Toolbar.TFrame')
        inner.pack(fill='x', padx=8, pady=6)

        self.file_label = ttk.Label(inner, text="No file loaded",
                                    style='Toolbar.TLabel')
        self.file_label.pack(side='left', fill='x', expand=True)

        self.save_btn = ttk.Button(inner, text="Save As\u2026",
                                   command=self.on_save,
                                   style='Toolbar.TButton')
        self.save_btn.pack(side='right', padx=(6, 0))
        self.save_btn.state(['disabled'])

        browse_btn = ttk.Button(inner, text="Browse\u2026",
                                command=self.on_browse,
                                style='Toolbar.TButton')
        browse_btn.pack(side='right')

        if DND_AVAILABLE:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop)

    def _build_status_bar(self):
        status_frame = ttk.Frame(self.root, style='Status.TFrame')
        status_frame.pack(fill='x', side='bottom')

        self.status_label = ttk.Label(status_frame, text="Ready",
                                      style='Status.TLabel')
        self.status_label.pack(side='left', padx=8, pady=3)

        self.status_detail = ttk.Label(status_frame, text="",
                                       style='Status.TLabel')
        self.status_detail.pack(side='right', padx=8, pady=3)

    def _build_preview_panel(self):
        preview_frame = ttk.LabelFrame(self.main_paned, text="Preview", padding=4)
        self.preview_canvas = tk.Canvas(preview_frame, bg='white',
                                         highlightthickness=0)
        self.preview_canvas.pack(fill='both', expand=True)
        self.preview_canvas.bind('<MouseWheel>', self._on_preview_scroll)
        self.preview_canvas.bind('<Button-4>', self._on_preview_scroll)
        self.preview_canvas.bind('<Button-5>', self._on_preview_scroll)
        self.preview_canvas.bind('<ButtonPress-1>', self._on_preview_drag_start)
        self.preview_canvas.bind('<B1-Motion>', self._on_preview_drag)
        self.preview_canvas.bind('<Double-Button-1>', self._on_preview_reset)
        self.main_paned.add(preview_frame, weight=3)

    def _build_welcome_overlay(self):
        self.welcome_frame = tk.Frame(self.root, bg=BG_MAIN)
        self.welcome_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        center = tk.Frame(self.welcome_frame, bg=BG_MAIN)
        center.place(relx=0.5, rely=0.45, anchor='center')

        icon_label = tk.Label(center, text='\u2B07', font=('Segoe UI', 48),
                              fg=ACCENT, bg=BG_MAIN)
        icon_label.pack(pady=(0, 12))

        main_text = tk.Label(center, text='Drop SVG file here',
                             font=FONT_WELCOME_LARGE, fg='#2C3E50', bg=BG_MAIN)
        main_text.pack()

        sub_text = tk.Label(center,
                            text='or click anywhere to browse',
                            font=FONT_WELCOME_SUB, fg=FG_WELCOME, bg=BG_MAIN)
        sub_text.pack(pady=(6, 20))

        welcome_browse = ttk.Button(center, text='Open SVG File\u2026',
                                    command=self.on_browse,
                                    style='Toolbar.TButton')
        welcome_browse.pack(ipadx=16, ipady=4)

        if not DND_AVAILABLE:
            hint = tk.Label(center,
                            text='(Drag-and-drop not available \u2014 use button above)',
                            font=FONT_SMALL, fg='#BDC3C7', bg=BG_MAIN)
            hint.pack(pady=(12, 0))

        for widget in [self.welcome_frame, center, icon_label, main_text, sub_text]:
            widget.bind('<Button-1>', lambda e: self.on_browse())

    def _dismiss_welcome(self):
        if hasattr(self, 'welcome_frame') and self.welcome_frame.winfo_exists():
            self.welcome_frame.place_forget()
            self.welcome_frame.destroy()

    def _make_chart_block(self, parent, placeholder, slider_label, slider_cmd):
        """Create one chart + slider pair. Returns (fig, ax, slider, slider_label_widget)."""
        fig = Figure(figsize=(4, 2.2), dpi=96)
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.22, top=0.88)
        ax.text(0.5, 0.5, f'Load an SVG to see\n{placeholder}',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=9, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        mpl_canvas = FigureCanvasTkAgg(fig, master=parent)
        mpl_canvas.get_tk_widget().pack(fill='both', expand=True)
        mpl_canvas.draw()

        sf = ttk.Frame(parent)
        sf.pack(fill='x', pady=(0, 4))
        ttk.Label(sf, text=slider_label, font=FONT_SMALL, width=10).pack(side='left')
        slider = tk.Scale(sf, from_=0, to=SLIDER_STEPS,
                          orient='horizontal', showvalue=False,
                          command=slider_cmd)
        slider.pack(side='left', fill='x', expand=True)
        val_label = ttk.Label(sf, text="--", width=10, font=FONT_SMALL)
        val_label.pack(side='left', padx=(4, 0))

        return fig, ax, slider, val_label

    def _setup_charts_panel(self):
        """Create the left panel with 3 vertically stacked charts, each with its slider."""
        panel = ttk.LabelFrame(self.main_paned, text="Distribution Analysis",
                               padding=6)

        # Gap chart + slider
        self.fig_gap, self.ax_gap, self.gap_slider, self.gap_slider_label = \
            self._make_chart_block(panel, 'gap distribution',
                                   'Threshold:', self._on_gap_slider_move)

        # Segment chart + slider + enable checkbox
        self.fig_seg, self.ax_seg, self.seg_slider, self.seg_slider_label = \
            self._make_chart_block(panel, 'segment distribution',
                                   'Simplify:', self._on_seg_slider_move)
        # Insert enable checkbox after the segment slider
        self.enable_simplify_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(panel, text="Enable simplify",
                         variable=self.enable_simplify_var,
                         command=self._on_enable_simplify_toggle).pack(anchor='w')

        # Endpoint chart + slider
        self.fig_tol, self.ax_tol, self.tol_slider, self.tol_slider_label = \
            self._make_chart_block(panel, 'endpoint distances',
                                   'Tolerance:', self._on_tol_slider_move)

        # Summary label at bottom
        self.summary_label = ttk.Label(panel, text="Load an SVG to begin analysis",
                                        foreground='gray', font=FONT_MONO,
                                        wraplength=350)
        self.summary_label.pack(fill='x', pady=(4, 0))

        self.main_paned.add(panel, weight=1)

    def _setup_text_analysis_panel(self):
        """Fallback left panel when matplotlib is not available."""
        ana = ttk.LabelFrame(self.main_paned, text="Auto-Analysis", padding=6)

        self.analysis_text = tk.Text(ana, height=12, wrap='word', state='disabled',
                                     bg='#f5f5f5', relief='flat', font=FONT_MONO)
        self.analysis_text.pack(fill='both', expand=True)

        btn_frame = ttk.Frame(ana)
        btn_frame.pack(fill='x', pady=(6, 0))

        self.enable_simplify_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(btn_frame, text="Enable simplify",
                         variable=self.enable_simplify_var).pack(side='left')

        self.main_paned.add(ana, weight=1)

    # ── Histogram drawing ────────────────────────────────────────

    def _draw_histogram(self, ax, values, threshold, title, left_label, right_label):
        """Draw a log-scale histogram with a threshold line. Returns artist refs."""
        ax.clear()

        vmin = max(min(values), 1e-6)
        vmax = max(values) * 1.1
        bins = np.logspace(np.log10(vmin), np.log10(vmax), 51)
        counts, _ = np.histogram(values, bins=bins)

        lefts = bins[:-1]
        widths = np.diff(bins)
        centers = np.sqrt(bins[:-1] * bins[1:])  # geometric mean

        colors = [COLOR_BELOW if c < threshold else COLOR_ABOVE for c in centers]
        bars = ax.bar(lefts, counts, width=widths, align='edge',
                      color=colors, edgecolor='white', linewidth=0.3)

        ax.set_xscale('log')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Distance (log scale)', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.tick_params(labelsize=7)

        vline = ax.axvline(threshold, color=COLOR_LINE, linestyle='--',
                            linewidth=1.5, zorder=5)

        below = sum(1 for v in values if v <= threshold)
        above = len(values) - below

        ltxt = ax.text(0.02, 0.95, f'{below} {left_label}',
                       transform=ax.transAxes, fontsize=8, color=COLOR_BELOW,
                       fontweight='bold', verticalalignment='top')
        rtxt = ax.text(0.98, 0.95, f'{above} {right_label}',
                       transform=ax.transAxes, fontsize=8, color=COLOR_ABOVE,
                       fontweight='bold', verticalalignment='top',
                       horizontalalignment='right')

        return list(bars), vline, ltxt, rtxt, centers

    def _update_chart(self, bars, vline, left_txt, right_txt, bin_centers,
                      data, threshold, left_label, right_label):
        """Generic partial redraw for any chart."""
        if bars is None or data is None:
            return
        vline.set_xdata([threshold, threshold])
        for bar, center in zip(bars, bin_centers):
            bar.set_facecolor(COLOR_BELOW if center < threshold else COLOR_ABOVE)
        below = sum(1 for v in data if v <= threshold)
        above = len(data) - below
        left_txt.set_text(f'{below} {left_label}')
        right_txt.set_text(f'{above} {right_label}')
        vline.figure.canvas.draw_idle()

    # ── Slider callbacks ─────────────────────────────────────────

    def _on_gap_slider_move(self, pos):
        if self._updating_threshold or self.gap_data is None:
            return
        self._updating_threshold = True
        value = self._slider_to_value(int(float(pos)), *self.gap_slider_range)
        self.threshold = value
        self.gap_slider_label.configure(text=f"{value:.4g}")
        self._update_chart(self.gap_bars, self.gap_vline, self.gap_left_txt,
                           self.gap_right_txt, self.gap_bin_centers,
                           self.gap_data, value, 'micro', 'structural')
        self._updating_threshold = False
        self._submit_preview()

    def _on_seg_slider_move(self, pos):
        if self._updating_simplify or self.seg_data is None:
            return
        self._updating_simplify = True
        value = self._slider_to_value(int(float(pos)), *self.seg_slider_range)
        self.simplify = value
        self.seg_slider_label.configure(text=f"{value:.4g}")
        self._update_chart(self.seg_bars, self.seg_vline, self.seg_left_txt,
                           self.seg_right_txt, self.seg_bin_centers,
                           self.seg_data, value, 'jitter', 'structural')
        self._updating_simplify = False
        self._submit_preview()

    def _on_tol_slider_move(self, pos):
        if self._updating_tol or self.tol_data is None:
            return
        self._updating_tol = True
        value = self._slider_to_value(int(float(pos)), *self.tol_slider_range)
        self.graph_tol = value
        self.tol_slider_label.configure(text=f"{value:.4g}")
        self._update_chart(self.tol_bars, self.tol_vline, self.tol_left_txt,
                           self.tol_right_txt, self.tol_bin_centers,
                           self.tol_data, value, 'matched', 'unmatched')
        self._updating_tol = False
        self._submit_preview()

    # ── Enable checkbox callback ─────────────────────────────────

    def _on_enable_simplify_toggle(self):
        enabled = self.enable_simplify_var.get()
        self.simplify_enabled = enabled
        if self.seg_bars is not None:
            alpha = 1.0 if enabled else 0.25
            for bar in self.seg_bars:
                bar.set_alpha(alpha)
            if self.seg_vline:
                self.seg_vline.set_alpha(alpha)
            if self.seg_left_txt:
                self.seg_left_txt.set_alpha(alpha)
            if self.seg_right_txt:
                self.seg_right_txt.set_alpha(alpha)
            self.ax_seg.figure.canvas.draw_idle()

        if enabled:
            self.seg_slider.configure(state='normal')
        else:
            self.seg_slider.configure(state='disabled')

        self._submit_preview()

    # ── Preview ──────────────────────────────────────────────────

    def _submit_preview(self):
        """Submit a preview request if parameters changed.

        Drains the input queue first so at most one pending request waits
        behind whatever the worker is currently processing.
        """
        if self._raw_group_cmds is None:
            return
        simplify = self.simplify if self.enable_simplify_var.get() else None
        params = (self.threshold, simplify, self.graph_tol)
        if params == self._preview_last_params:
            return
        self._preview_last_params = params
        # Drain any stale pending request so only the latest waits
        try:
            while True:
                self._preview_in_q.get_nowait()
        except Exception:
            pass
        self._preview_in_q.put((
            self._raw_group_cmds,
            self.threshold,
            simplify,
            self.graph_tol,
        ))

    def _poll_preview(self):
        """Poll the output queue for completed preview results."""
        latest = None
        try:
            while True:
                latest = self._preview_out_q.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            self._render_preview(latest)
        self._preview_poll_id = self.root.after(50, self._poll_preview)

    def _render_preview(self, subpaths):
        """Store processed subpath data and redraw the preview canvas."""
        all_pts = [_flatten_subpath(sp) for sp in subpaths]
        all_pts = [pts for pts in all_pts if len(pts) >= 2]
        self._preview_pts = all_pts
        self._preview_bounds = _compute_bounds(all_pts) if all_pts else None
        self._redraw_preview()

    def _redraw_preview(self):
        """Draw stored point data with current zoom/pan transform."""
        c = self.preview_canvas
        c.delete('all')

        if not self._preview_pts or not self._preview_bounds:
            c.create_text(c.winfo_width() // 2, c.winfo_height() // 2,
                          text="No paths to display", fill='gray', font=('', 10))
            return

        bx, by, bx2, by2 = self._preview_bounds
        bw, bh = bx2 - bx, by2 - by

        cw = max(c.winfo_width(), 100)
        ch = max(c.winfo_height(), 100)
        margin = 12
        base_scale = min((cw - 2 * margin) / bw, (ch - 2 * margin) / bh)
        base_ox = margin + ((cw - 2 * margin) - bw * base_scale) / 2 - bx * base_scale
        base_oy = margin + ((ch - 2 * margin) - bh * base_scale) / 2 - by * base_scale

        # Apply zoom (centered on canvas center) + pan
        cx, cy = cw / 2, ch / 2
        scale = base_scale * self._view_zoom
        ox = cx + (base_ox - cx) * self._view_zoom + self._view_pan_x
        oy = cy + (base_oy - cy) * self._view_zoom + self._view_pan_y

        # Determine if zoom is high enough to show point markers
        # At base zoom, avg segment ≈ a few pixels; show dots when segments are ≥ 8px apart
        show_dots = self._view_zoom >= 5.0
        dot_r = min(max(scale * 0.3, 1.5), 6)  # radius scales with zoom, clamped

        for i, pts in enumerate(self._preview_pts):
            color = PREVIEW_COLORS[i % len(PREVIEW_COLORS)]
            coords = []
            for x, y in pts:
                coords.append(x * scale + ox)
                coords.append(y * scale + oy)
            if len(coords) >= 4:
                c.create_line(*coords, fill=color, width=1)

            if show_dots and len(pts) >= 2:
                # Draw circles at each intermediate point
                for j in range(1, len(pts) - 1):
                    sx = pts[j][0] * scale + ox
                    sy = pts[j][1] * scale + oy
                    c.create_oval(sx - dot_r, sy - dot_r, sx + dot_r, sy + dot_r,
                                  fill=color, outline='')

                # Check if subpath is closed (first point ≈ last point)
                fx, fy = pts[0]
                lx, ly = pts[-1]
                closed = (abs(fx - lx) + abs(fy - ly)) * scale < 2.0

                if closed:
                    # Closed: just a dot at the start
                    sx = fx * scale + ox
                    sy = fy * scale + oy
                    c.create_oval(sx - dot_r, sy - dot_r, sx + dot_r, sy + dot_r,
                                  fill=color, outline='')
                else:
                    # Start marker: green square
                    sr = dot_r * 1.4
                    sx = fx * scale + ox
                    sy = fy * scale + oy
                    c.create_rectangle(sx - sr, sy - sr, sx + sr, sy + sr,
                                       fill='#2ca02c', outline='white', width=1)
                    # End marker: red diamond
                    ex = lx * scale + ox
                    ey = ly * scale + oy
                    c.create_polygon(ex, ey - sr, ex + sr, ey, ex, ey + sr, ex - sr, ey,
                                     fill='#d62728', outline='white', width=1)

        # Summary text
        zoom_pct = self._view_zoom * 100
        c.create_text(4, ch - 4, anchor='sw', fill='gray', font=('Consolas', 8),
                       text=f"{len(self._preview_pts)} paths  |  {zoom_pct:.0f}%")

    def _on_preview_scroll(self, event):
        """Zoom preview at cursor position on mouse wheel."""
        if not self._preview_pts:
            return
        # Determine scroll direction
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            factor = 1.25
        else:
            factor = 1 / 1.25
        mx, my = event.x, event.y
        cw = max(self.preview_canvas.winfo_width(), 100)
        ch = max(self.preview_canvas.winfo_height(), 100)
        cx, cy = cw / 2, ch / 2
        # Adjust pan so the world point under cursor stays fixed
        self._view_pan_x -= (mx - cx - self._view_pan_x) * (factor - 1)
        self._view_pan_y -= (my - cy - self._view_pan_y) * (factor - 1)
        self._view_zoom *= factor
        self._redraw_preview()

    def _on_preview_drag_start(self, event):
        """Start dragging to pan the preview."""
        self._drag_start = (event.x, event.y)
        self._drag_pan_start = (self._view_pan_x, self._view_pan_y)

    def _on_preview_drag(self, event):
        """Pan the preview while dragging."""
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._view_pan_x = self._drag_pan_start[0] + dx
        self._view_pan_y = self._drag_pan_start[1] + dy
        self._redraw_preview()

    def _on_preview_reset(self, event):
        """Reset preview zoom/pan to fit-all on double-click."""
        self._view_zoom = 1.0
        self._view_pan_x = 0.0
        self._view_pan_y = 0.0
        self._redraw_preview()

    def _on_close(self):
        """Clean shutdown: stop preview thread, destroy window."""
        if self._preview_poll_id is not None:
            self.root.after_cancel(self._preview_poll_id)
        try:
            self._preview_in_q.put(None)
            self._preview_thread.join(timeout=2)
        except Exception:
            pass
        self.root.destroy()

    # ── Status bar ─────────────────────────────────────────────────

    def _set_status(self, msg, detail=''):
        self.status_label.configure(text=msg)
        self.status_detail.configure(text=detail)

    def _set_status_safe(self, msg, detail=''):
        self.root.after(0, self._set_status, msg, detail)

    def _set_analysis_text(self, text):
        """Set text in the fallback analysis widget (no matplotlib)."""
        if not hasattr(self, 'analysis_text'):
            return
        self.analysis_text.configure(state='normal')
        self.analysis_text.delete('1.0', 'end')
        self.analysis_text.insert('1.0', text)
        self.analysis_text.configure(state='disabled')

    # ── File loading ──────────────────────────────────────────────

    def on_browse(self):
        path = filedialog.askopenfilename(
            title="Select SVG file",
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
        )
        if path:
            self.load_file(path)

    def on_drop(self, event):
        path = event.data.strip()
        if path.startswith('{') and path.endswith('}'):
            path = path[1:-1]
        if path.lower().endswith('.svg'):
            self.load_file(path)
        else:
            messagebox.showwarning("Invalid file", "Please drop an SVG file.")

    def load_file(self, path):
        self._dismiss_welcome()
        self.current_file = path
        self.save_btn.state(['disabled'])
        name = os.path.basename(path)
        self.file_label.configure(text=name)
        self._set_status(f"Analyzing {name}...")

        if MPL_AVAILABLE:
            self.summary_label.configure(text="Analyzing...", foreground='gray')
        else:
            self._set_analysis_text("Analyzing\u2026")

        t = threading.Thread(target=self._run_analysis, args=(path,), daemon=True)
        t.start()

    def _run_analysis(self, path):
        try:
            result = auto_analyze_svg(path)
            self.root.after(0, self._on_analysis_done, result)
        except Exception as e:
            self.root.after(0, self._on_analysis_error, str(e))

    def _on_analysis_error(self, err):
        if MPL_AVAILABLE:
            self.summary_label.configure(text=f"Error: {err}", foreground='red')
        else:
            self._set_analysis_text(f"Error: {err}")
        self._set_status(f"Error: {err}")

    def _on_analysis_done(self, result):
        if 'error' in result:
            self._on_analysis_error(result['error'])
            return

        self.analysis = result
        self.save_btn.state(['!disabled'])

        # Store raw commands for preview subprocess
        self._raw_group_cmds = result.get('group_cmds')

        # Reset view for new file
        self._view_zoom = 1.0
        self._view_pan_x = 0.0
        self._view_pan_y = 0.0

        if MPL_AVAILABLE:
            self._on_analysis_done_charts(result)
        else:
            self._on_analysis_done_text(result)

        self._set_status("Analysis complete", "Ready to save")

        # Trigger initial preview with recommended values
        self._preview_last_params = None  # force submit
        self._submit_preview()

    def _set_slider(self, slider, label, value, slider_range, updating_attr):
        """Helper to set a slider position without triggering callbacks."""
        setattr(self, updating_attr, True)
        slider_pos = self._value_to_slider(value, *slider_range)
        slider.set(slider_pos)
        label.configure(text=f"{value:.4g}")
        setattr(self, updating_attr, False)

    def _on_analysis_done_charts(self, result):
        """Populate all three charts from analysis results."""
        pc = result['path_count']
        gc = result.get('group_count', 1)
        merged_note = f" ({pc} paths merged)" if pc > 1 else ""
        self.summary_label.configure(
            text=f"Groups: {gc}{merged_note}   "
                 f"Commands: {result['command_count']}   "
                 f"Subpaths: {result['subpath_count']}",
            foreground='black')

        # Gap chart
        ga = result['gap_analysis']
        if ga and len(ga['gaps']) >= 2:
            self.gap_data = ga['gaps']
            vmin = max(min(self.gap_data) * 0.5, 1e-6)
            vmax = max(self.gap_data) * 2.0
            self.gap_slider_range = (vmin, vmax)
            self.threshold = ga['recommended_threshold']

            self.gap_bars, self.gap_vline, self.gap_left_txt, self.gap_right_txt, \
                self.gap_bin_centers = self._draw_histogram(
                    self.ax_gap, self.gap_data, self.threshold,
                    'Gap Distribution', 'micro', 'structural')
            self._set_slider(self.gap_slider, self.gap_slider_label,
                             self.threshold, self.gap_slider_range, '_updating_threshold')
        else:
            self._show_placeholder(self.ax_gap, 'No gap data')

        # Segment chart
        sa = result['segment_analysis']
        if sa and len(sa['segments']) >= 2:
            self.seg_data = sa['segments']
            vmin = max(min(self.seg_data) * 0.5, 1e-6)
            vmax = max(self.seg_data) * 2.0
            self.seg_slider_range = (vmin, vmax)
            self.simplify = sa['recommended_simplify']

            self.seg_bars, self.seg_vline, self.seg_left_txt, self.seg_right_txt, \
                self.seg_bin_centers = self._draw_histogram(
                    self.ax_seg, self.seg_data, self.simplify,
                    'Segment Distribution', 'jitter', 'structural')
            self._set_slider(self.seg_slider, self.seg_slider_label,
                             self.simplify, self.seg_slider_range, '_updating_simplify')

            if not self.enable_simplify_var.get():
                for artist in (self.seg_bars or []):
                    artist.set_alpha(0.25)
                for artist in [self.seg_vline, self.seg_left_txt, self.seg_right_txt]:
                    if artist:
                        artist.set_alpha(0.25)
        else:
            self._show_placeholder(self.ax_seg, 'No segment data')

        # Endpoint / tolerance chart
        ea = result['endpoint_analysis']
        if ea and len(ea['distances']) >= 2:
            self.tol_data = ea['distances']
            vmin = max(min(self.tol_data) * 0.5, 1e-6)
            vmax = max(self.tol_data) * 2.0
            self.tol_slider_range = (vmin, vmax)
            self.graph_tol = ea['recommended_tolerance']

            self.tol_bars, self.tol_vline, self.tol_left_txt, self.tol_right_txt, \
                self.tol_bin_centers = self._draw_histogram(
                    self.ax_tol, self.tol_data, self.graph_tol,
                    'Endpoint Distance', 'matched', 'unmatched')
            self._set_slider(self.tol_slider, self.tol_slider_label,
                             self.graph_tol, self.tol_slider_range, '_updating_tol')
        else:
            self._show_placeholder(self.ax_tol, 'No endpoint data')

        for fig in (self.fig_gap, self.fig_seg, self.fig_tol):
            fig.canvas.draw()

    def _show_placeholder(self, ax, message):
        """Show placeholder text in an empty chart."""
        ax.clear()
        ax.text(0.5, 0.5, message, transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    def _on_analysis_done_text(self, result):
        """Fallback: populate text widget when matplotlib is unavailable."""
        lines = []
        pc = result['path_count']
        gc = result.get('group_count', 1)
        merged_note = f" ({pc} paths merged)" if pc > 1 else ""
        lines.append(f"Groups: {gc}{merged_note}   "
                      f"Commands: {result['command_count']}   "
                      f"Subpaths: {result['subpath_count']}")

        ga = result['gap_analysis']
        if ga:
            lines.append(f"Gaps:     {ga['micro_count']} micro "
                          f"(<= {ga['micro_max']:.4g}),  "
                          f"{ga['structural_count']} structural "
                          f"(>= {ga['structural_min']:.4g})")
            lines.append(f"  -> Recommended threshold: {ga['recommended_threshold']:.6g}")
            self.threshold = ga['recommended_threshold']
        else:
            lines.append("Gaps: none detected")

        sa = result['segment_analysis']
        if sa:
            lines.append(f"Segments: {sa['jitter_count']} jitter "
                          f"(<= {sa['jitter_max']:.4g}),  "
                          f"{sa['structural_count']} structural "
                          f"(>= {sa['structural_min']:.4g})")
            lines.append(f"  -> Recommended simplify: {sa['recommended_simplify']:.6g}")
            self.simplify = sa['recommended_simplify']
        else:
            lines.append("Segments: no line segments found")

        ea = result['endpoint_analysis']
        if ea:
            lines.append(f"Endpoints: {ea['matched_count']} matched "
                          f"(<= {ea['matched_max']:.4g}),  "
                          f"{ea['unmatched_count']} unmatched "
                          f"(>= {ea['unmatched_min']:.4g})")
            lines.append(f"  -> Recommended tolerance: {ea['recommended_tolerance']:.6g}")
            self.graph_tol = ea['recommended_tolerance']
        else:
            lines.append("Endpoints: insufficient data")

        self._set_analysis_text('\n'.join(lines))

    # ── Save ──────────────────────────────────────────────────────

    def on_save(self):
        if not self.current_file or not self.analysis:
            return

        base, ext = os.path.splitext(self.current_file)
        suggestion = base + '_fixed' + ext

        path = filedialog.asksaveasfilename(
            title="Save fixed SVG",
            initialfile=os.path.basename(suggestion),
            initialdir=os.path.dirname(suggestion),
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
            defaultextension=".svg",
        )
        if not path:
            return

        threshold = self.threshold
        simplify = self.simplify if self.enable_simplify_var.get() else None
        graph_tol = self.graph_tol

        self.save_btn.state(['disabled'])
        self._set_status("Saving...", f"threshold={threshold:.4g}")

        t = threading.Thread(
            target=self._run_save,
            args=(path, threshold, simplify, graph_tol),
            daemon=True,
        )
        t.start()

    def _run_save(self, save_path, threshold, simplify, graph_tol):
        try:
            ET.register_namespace('', 'http://www.w3.org/2000/svg')
            tree = ET.parse(self.current_file)
            root = tree.getroot()
            paths = root.findall('.//{http://www.w3.org/2000/svg}path')
            if not paths:
                paths = root.findall('.//path')

            parent_map = {child: parent for parent in root.iter() for child in parent}
            groups = _group_paths(paths)
            total_output_paths = 0

            for gi, (attrs, cmds, elements) in enumerate(groups):
                total_m = sum(1 for c, _ in cmds[1:] if c in ('M', 'm'))
                print(f"Group #{gi + 1} ({len(elements)} paths merged, "
                      f"{len(cmds)} commands, {total_m + 1} subpaths):")

                cmds, micro_joined = join_by_threshold(cmds, threshold)
                print(f"  Micro-gaps joined (dist <= {threshold:.6g}): {micro_joined}")
                remaining = sum(1 for c, _ in cmds[1:] if c in ('M', 'm'))
                print(f"  Remaining separate subpaths: {remaining + 1}")

                cmds, traced = trace_graph(cmds, tol=graph_tol)
                remaining2 = sum(1 for c, _ in cmds[1:] if c in ('M', 'm'))
                print(f"  Graph-traced: {traced} merged "
                      f"({remaining + 1} -> {remaining2 + 1} subpaths)")

                before3 = len(cmds)
                cmds, deduped = deduplicate_endpoints(cmds)
                if deduped:
                    print(f"  Near-duplicate endpoints removed: {deduped} "
                          f"({before3} -> {len(cmds)} commands)")

                if simplify is not None and simplify > 0:
                    before = len(cmds)
                    cmds, removed = simplify_short_runs(cmds, simplify)
                    print(f"  Simplified: {removed} segments removed "
                          f"({before} -> {len(cmds)} commands)")

                cmds, closed_count = close_subpaths(cmds)
                if closed_count:
                    print(f"  Subpaths closed (start~=end): {closed_count}")

                result_subpaths = split_into_subpaths(cmds)
                print(f"  Output: {len(result_subpaths)} continuous path(s)")
                total_output_paths += len(result_subpaths)

                insert_parent = parent_map[elements[0]]
                for el in elements:
                    parent_map[el].remove(el)

                tag = elements[0].tag
                for sp in result_subpaths:
                    new_el = ET.SubElement(insert_parent, tag)
                    new_el.set('d', cmds_to_str(sp))
                    for k, v in attrs.items():
                        new_el.set(k, v)

            tree.write(save_path, xml_declaration=False, encoding='unicode')
            print(f"Saved {total_output_paths} path(s) to: {save_path}")
            self._set_status_safe(
                f"Saved {total_output_paths} path(s)",
                os.path.basename(save_path))
            self.root.after(0, lambda: self.save_btn.state(['!disabled']))
        except Exception as e:
            print(f"Save error: {e}")
            self._set_status_safe(f"Save error: {e}")
            self.root.after(0, lambda: self.save_btn.state(['!disabled']))


# ── Entry point ───────────────────────────────────────────────────────

def main():
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    app = SVGPathFixerGUI(root)

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        root.after(100, app.load_file, sys.argv[1])

    root.mainloop()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
