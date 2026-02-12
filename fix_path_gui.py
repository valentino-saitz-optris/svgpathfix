"""
Tkinter GUI for SVG Path Fixer.

Loads an SVG, auto-analyzes gap and segment distributions to recommend
parameters, then processes the path using fix_path's 5-step pipeline.
Displays interactive histograms with log-scale sliders for picking values.
"""

import math
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import xml.etree.ElementTree as ET

# Ensure fix_path.py can be imported from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fix_path import (
    parse_commands, get_endpoint, dist, cmds_to_str,
    join_by_threshold, trace_graph, deduplicate_endpoints,
    simplify_short_runs, close_subpaths, split_into_subpaths,
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


# ── Auto-analysis functions ───────────────────────────────────────────

def _find_cluster_boundary(values):
    """Find the index of the largest gap between consecutive sorted values.

    Uses log-scale jumps so it works across orders of magnitude
    (e.g. gaps of 0.005 vs 1.5 vs 200 all get detected correctly).
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

    d = paths[0].get('d', '')
    if not d:
        return {'error': 'First <path> has no d attribute.'}

    cmds = parse_commands(d)
    total_m = sum(1 for c, _ in cmds[1:] if c in ('M', 'm'))
    subpaths = split_into_subpaths(cmds)

    gap_analysis = analyze_gap_distribution(cmds)
    seg_analysis = analyze_segment_distribution(cmds)

    return {
        'tree': tree,
        'path_count': len(paths),
        'command_count': len(cmds),
        'subpath_count': len(subpaths),
        'm_breaks': total_m,
        'gap_analysis': gap_analysis,
        'segment_analysis': seg_analysis,
    }


# ── GUI ───────────────────────────────────────────────────────────────

class SVGPathFixerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SVG Path Fixer")
        if MPL_AVAILABLE:
            self.root.geometry("1050x780")
            self.root.minsize(800, 600)
        else:
            self.root.geometry("820x680")
            self.root.minsize(600, 500)

        self.current_file = None
        self.analysis = None
        self.processed_tree = None

        # Chart state
        self.gap_data = None
        self.seg_data = None
        self.gap_slider_range = (1e-6, 200.0)
        self.seg_slider_range = (1e-6, 200.0)
        self._updating_threshold = False
        self._updating_simplify = False

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

        self.setup_ui()

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

    def setup_ui(self):
        pad = {'padx': 8, 'pady': 4}

        # -- Input frame --
        inp = ttk.LabelFrame(self.root, text="Input", padding=6)
        inp.pack(fill='x', **pad)

        self.file_label = ttk.Label(inp, text="No file loaded", foreground='gray')
        self.file_label.pack(side='left', fill='x', expand=True)

        browse_btn = ttk.Button(inp, text="Browse\u2026", command=self.on_browse)
        browse_btn.pack(side='right')

        if DND_AVAILABLE:
            self.file_label.configure(text="Drop SVG here or click Browse\u2026")
            inp.drop_target_register(DND_FILES)
            inp.dnd_bind('<<Drop>>', self.on_drop)

        # -- Charts / Analysis frame --
        if MPL_AVAILABLE:
            self._setup_charts_frame(pad)
        else:
            self._setup_text_analysis_frame(pad)

        # -- Parameters frame --
        par = ttk.LabelFrame(self.root, text="Parameters", padding=6)
        par.pack(fill='x', **pad)

        grid = ttk.Frame(par)
        grid.pack(fill='x')

        # Threshold
        ttk.Label(grid, text="Threshold (gap join):").grid(row=0, column=0, sticky='w')
        self.threshold_var = tk.StringVar(value="0.5")
        self.threshold_entry = ttk.Entry(grid, textvariable=self.threshold_var, width=12)
        self.threshold_entry.grid(row=0, column=1, padx=4)

        # Simplify
        ttk.Label(grid, text="Simplify (segment):").grid(row=1, column=0, sticky='w', pady=2)
        self.simplify_var = tk.StringVar(value="0.5")
        self.simplify_entry = ttk.Entry(grid, textvariable=self.simplify_var, width=12)
        self.simplify_entry.grid(row=1, column=1, padx=4, pady=2)
        self.enable_simplify_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grid, text="Enable", variable=self.enable_simplify_var).grid(row=1, column=2, pady=2)

        # Graph tolerance
        ttk.Label(grid, text="Graph tolerance:").grid(row=2, column=0, sticky='w', pady=2)
        self.graph_tol_var = tk.StringVar(value="0.1")
        ttk.Entry(grid, textvariable=self.graph_tol_var, width=12).grid(row=2, column=1, padx=4, pady=2)

        # Buttons
        btn_frame = ttk.Frame(par)
        btn_frame.pack(fill='x', pady=(8, 0))

        self.process_btn = ttk.Button(btn_frame, text="Process", command=self.on_process)
        self.process_btn.pack(side='left', padx=(0, 6))
        self.process_btn.state(['disabled'])

        self.save_btn = ttk.Button(btn_frame, text="Save As\u2026", command=self.on_save)
        self.save_btn.pack(side='right')
        self.save_btn.state(['disabled'])

        # Wire up trace callbacks for bidirectional sync
        if MPL_AVAILABLE:
            self.threshold_var.trace_add('write', self._on_threshold_entry_change)
            self.simplify_var.trace_add('write', self._on_simplify_entry_change)
            self.enable_simplify_var.trace_add('write', self._on_enable_simplify_toggle)

        # -- Log frame --
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=6)
        log_frame.pack(fill='both', expand=True, **pad)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap='word',
                                                   state='disabled', font=('Consolas', 9))
        self.log_text.pack(fill='both', expand=True)

    def _setup_charts_frame(self, pad):
        """Create the matplotlib charts frame with sliders and summary label."""
        charts = ttk.LabelFrame(self.root, text="Distribution Analysis", padding=6)
        charts.pack(fill='x', **pad)

        # Matplotlib figure with two side-by-side axes
        self.fig = Figure(figsize=(10, 3.0), dpi=96)
        self.ax_gap = self.fig.add_subplot(1, 2, 1)
        self.ax_seg = self.fig.add_subplot(1, 2, 2)
        self.fig.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.88, wspace=0.30)

        # Initial placeholder text
        self.ax_gap.text(0.5, 0.5, 'Load an SVG to see\ngap distribution',
                         transform=self.ax_gap.transAxes, ha='center', va='center',
                         fontsize=10, color='gray')
        self.ax_gap.set_xticks([])
        self.ax_gap.set_yticks([])
        self.ax_seg.text(0.5, 0.5, 'Load an SVG to see\nsegment distribution',
                         transform=self.ax_seg.transAxes, ha='center', va='center',
                         fontsize=10, color='gray')
        self.ax_seg.set_xticks([])
        self.ax_seg.set_yticks([])

        self.canvas = FigureCanvasTkAgg(self.fig, master=charts)
        self.canvas.get_tk_widget().pack(fill='x')
        self.canvas.draw()

        # Slider frame
        slider_frame = ttk.Frame(charts)
        slider_frame.pack(fill='x', pady=(2, 0))

        # Gap slider (left half)
        gap_sf = ttk.Frame(slider_frame)
        gap_sf.pack(side='left', fill='x', expand=True, padx=(0, 10))
        ttk.Label(gap_sf, text="Threshold:", font=('', 8)).pack(side='left')
        self.gap_slider = tk.Scale(gap_sf, from_=0, to=SLIDER_STEPS,
                                    orient='horizontal', showvalue=False,
                                    command=self._on_gap_slider_move)
        self.gap_slider.pack(side='left', fill='x', expand=True)
        self.gap_slider_label = ttk.Label(gap_sf, text="--", width=10,
                                           font=('Consolas', 8))
        self.gap_slider_label.pack(side='left', padx=(4, 0))

        # Segment slider (right half)
        seg_sf = ttk.Frame(slider_frame)
        seg_sf.pack(side='left', fill='x', expand=True, padx=(10, 0))
        ttk.Label(seg_sf, text="Simplify:", font=('', 8)).pack(side='left')
        self.seg_slider = tk.Scale(seg_sf, from_=0, to=SLIDER_STEPS,
                                    orient='horizontal', showvalue=False,
                                    command=self._on_seg_slider_move)
        self.seg_slider.pack(side='left', fill='x', expand=True)
        self.seg_slider_label = ttk.Label(seg_sf, text="--", width=10,
                                           font=('Consolas', 8))
        self.seg_slider_label.pack(side='left', padx=(4, 0))

        # Summary label
        self.summary_label = ttk.Label(charts, text="Load an SVG to begin analysis",
                                        foreground='gray', font=('Consolas', 9))
        self.summary_label.pack(fill='x', pady=(4, 0))

    def _setup_text_analysis_frame(self, pad):
        """Fallback: plain text analysis when matplotlib is not available."""
        ana = ttk.LabelFrame(self.root, text="Auto-Analysis", padding=6)
        ana.pack(fill='x', **pad)

        self.analysis_text = tk.Text(ana, height=5, wrap='word', state='disabled',
                                     bg='#f5f5f5', relief='flat', font=('Consolas', 9))
        self.analysis_text.pack(fill='x')

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

    def _update_gap_chart(self, threshold):
        """Partial redraw: move line, recolor bars, update count labels."""
        if self.gap_bars is None or self.gap_data is None:
            return
        self.gap_vline.set_xdata([threshold, threshold])
        for bar, center in zip(self.gap_bars, self.gap_bin_centers):
            bar.set_facecolor(COLOR_BELOW if center < threshold else COLOR_ABOVE)
        below = sum(1 for v in self.gap_data if v <= threshold)
        above = len(self.gap_data) - below
        self.gap_left_txt.set_text(f'{below} micro')
        self.gap_right_txt.set_text(f'{above} structural')
        self.canvas.draw_idle()

    def _update_seg_chart(self, simplify):
        """Partial redraw: move line, recolor bars, update count labels."""
        if self.seg_bars is None or self.seg_data is None:
            return
        self.seg_vline.set_xdata([simplify, simplify])
        for bar, center in zip(self.seg_bars, self.seg_bin_centers):
            bar.set_facecolor(COLOR_BELOW if center < simplify else COLOR_ABOVE)
        below = sum(1 for v in self.seg_data if v <= simplify)
        above = len(self.seg_data) - below
        self.seg_left_txt.set_text(f'{below} jitter')
        self.seg_right_txt.set_text(f'{above} structural')
        self.canvas.draw_idle()

    # ── Slider callbacks ─────────────────────────────────────────

    def _on_gap_slider_move(self, pos):
        if self._updating_threshold:
            return
        if self.gap_data is None:
            return
        self._updating_threshold = True
        value = self._slider_to_value(int(float(pos)), *self.gap_slider_range)
        self.threshold_var.set(f"{value:.6g}")
        self.gap_slider_label.configure(text=f"{value:.4g}")
        self._update_gap_chart(value)
        self._updating_threshold = False

    def _on_seg_slider_move(self, pos):
        if self._updating_simplify:
            return
        if self.seg_data is None:
            return
        self._updating_simplify = True
        value = self._slider_to_value(int(float(pos)), *self.seg_slider_range)
        self.simplify_var.set(f"{value:.6g}")
        self.seg_slider_label.configure(text=f"{value:.4g}")
        self._update_seg_chart(value)
        self._updating_simplify = False

    def _on_threshold_entry_change(self, *_args):
        if self._updating_threshold:
            return
        if self.gap_data is None:
            return
        try:
            value = float(self.threshold_var.get())
        except ValueError:
            return
        self._updating_threshold = True
        slider_pos = self._value_to_slider(value, *self.gap_slider_range)
        self.gap_slider.set(slider_pos)
        self.gap_slider_label.configure(text=f"{value:.4g}")
        self._update_gap_chart(value)
        self._updating_threshold = False

    def _on_simplify_entry_change(self, *_args):
        if self._updating_simplify:
            return
        if self.seg_data is None:
            return
        try:
            value = float(self.simplify_var.get())
        except ValueError:
            return
        self._updating_simplify = True
        slider_pos = self._value_to_slider(value, *self.seg_slider_range)
        self.seg_slider.set(slider_pos)
        self.seg_slider_label.configure(text=f"{value:.4g}")
        self._update_seg_chart(value)
        self._updating_simplify = False

    # ── Enable checkbox callback ─────────────────────────────────

    def _on_enable_simplify_toggle(self, *_args):
        enabled = self.enable_simplify_var.get()
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
            self.canvas.draw_idle()

        if enabled:
            self.seg_slider.configure(state='normal')
            self.simplify_entry.configure(state='normal')
        else:
            self.seg_slider.configure(state='disabled')
            self.simplify_entry.configure(state='disabled')

    # ── Logging ───────────────────────────────────────────────────

    def log(self, msg):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', msg + '\n')
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def clear_log(self):
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.configure(state='disabled')

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
        self.current_file = path
        self.processed_tree = None
        self.save_btn.state(['disabled'])
        name = os.path.basename(path)
        self.file_label.configure(text=name, foreground='black')
        self.clear_log()
        self.log(f"Loaded: {path}")

        if MPL_AVAILABLE:
            self.summary_label.configure(text="Analyzing...", foreground='gray')
        else:
            self._set_analysis_text("Analyzing\u2026")

        self.process_btn.state(['disabled'])

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
        self.log(f"Analysis error: {err}")

    def _on_analysis_done(self, result):
        if 'error' in result:
            self._on_analysis_error(result['error'])
            return

        self.analysis = result
        self.process_btn.state(['!disabled'])

        if MPL_AVAILABLE:
            self._on_analysis_done_charts(result)
        else:
            self._on_analysis_done_text(result)

        self.log("Analysis complete.")

    def _on_analysis_done_charts(self, result):
        """Populate charts from analysis results."""
        self.summary_label.configure(
            text=f"Paths: {result['path_count']}   "
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

            threshold = ga['recommended_threshold']
            self._updating_threshold = True
            self.threshold_var.set(f"{threshold:.6g}")
            self._updating_threshold = False

            self.gap_bars, self.gap_vline, self.gap_left_txt, self.gap_right_txt, \
                self.gap_bin_centers = self._draw_histogram(
                    self.ax_gap, self.gap_data, threshold,
                    'Gap Distribution', 'micro', 'structural')

            self._updating_threshold = True
            slider_pos = self._value_to_slider(threshold, *self.gap_slider_range)
            self.gap_slider.set(slider_pos)
            self.gap_slider_label.configure(text=f"{threshold:.4g}")
            self._updating_threshold = False
        else:
            self.ax_gap.clear()
            self.ax_gap.text(0.5, 0.5, 'No gap data', transform=self.ax_gap.transAxes,
                             ha='center', va='center', fontsize=10, color='gray')
            self.ax_gap.set_xticks([])
            self.ax_gap.set_yticks([])

        # Segment chart
        sa = result['segment_analysis']
        if sa and len(sa['segments']) >= 2:
            self.seg_data = sa['segments']
            vmin = max(min(self.seg_data) * 0.5, 1e-6)
            vmax = max(self.seg_data) * 2.0
            self.seg_slider_range = (vmin, vmax)

            simplify = sa['recommended_simplify']
            self._updating_simplify = True
            self.simplify_var.set(f"{simplify:.6g}")
            self._updating_simplify = False

            self.seg_bars, self.seg_vline, self.seg_left_txt, self.seg_right_txt, \
                self.seg_bin_centers = self._draw_histogram(
                    self.ax_seg, self.seg_data, simplify,
                    'Segment Distribution', 'jitter', 'structural')

            self._updating_simplify = True
            slider_pos = self._value_to_slider(simplify, *self.seg_slider_range)
            self.seg_slider.set(slider_pos)
            self.seg_slider_label.configure(text=f"{simplify:.4g}")
            self._updating_simplify = False

            if not self.enable_simplify_var.get():
                for bar in self.seg_bars:
                    bar.set_alpha(0.25)
                if self.seg_vline:
                    self.seg_vline.set_alpha(0.25)
                if self.seg_left_txt:
                    self.seg_left_txt.set_alpha(0.25)
                if self.seg_right_txt:
                    self.seg_right_txt.set_alpha(0.25)
        else:
            self.ax_seg.clear()
            self.ax_seg.text(0.5, 0.5, 'No segment data', transform=self.ax_seg.transAxes,
                             ha='center', va='center', fontsize=10, color='gray')
            self.ax_seg.set_xticks([])
            self.ax_seg.set_yticks([])

        self.canvas.draw()

    def _on_analysis_done_text(self, result):
        """Fallback: populate text widget when matplotlib is unavailable."""
        lines = []
        lines.append(f"Paths: {result['path_count']}   "
                      f"Commands: {result['command_count']}   "
                      f"Subpaths: {result['subpath_count']}")

        ga = result['gap_analysis']
        if ga:
            lines.append(f"Gaps:     {ga['micro_count']} micro "
                          f"(<= {ga['micro_max']:.4g}),  "
                          f"{ga['structural_count']} structural "
                          f"(>= {ga['structural_min']:.4g})")
            lines.append(f"  -> Recommended threshold: {ga['recommended_threshold']:.6g}")
            self.threshold_var.set(f"{ga['recommended_threshold']:.6g}")
        else:
            lines.append("Gaps: none detected")

        sa = result['segment_analysis']
        if sa:
            lines.append(f"Segments: {sa['jitter_count']} jitter "
                          f"(<= {sa['jitter_max']:.4g}),  "
                          f"{sa['structural_count']} structural "
                          f"(>= {sa['structural_min']:.4g})")
            lines.append(f"  -> Recommended simplify: {sa['recommended_simplify']:.6g}")
            self.simplify_var.set(f"{sa['recommended_simplify']:.6g}")
        else:
            lines.append("Segments: no line segments found")

        self._set_analysis_text('\n'.join(lines))

    # ── Processing ────────────────────────────────────────────────

    def on_process(self):
        if not self.analysis or 'tree' not in self.analysis:
            messagebox.showwarning("No file", "Load an SVG file first.")
            return

        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror("Invalid", "Threshold must be a number.")
            return

        simplify = None
        if self.enable_simplify_var.get():
            try:
                simplify = float(self.simplify_var.get())
            except ValueError:
                messagebox.showerror("Invalid", "Simplify must be a number.")
                return

        try:
            graph_tol = float(self.graph_tol_var.get())
        except ValueError:
            messagebox.showerror("Invalid", "Graph tolerance must be a number.")
            return

        self.process_btn.state(['disabled'])
        self.save_btn.state(['disabled'])
        self.log(f"\nProcessing with threshold={threshold}, simplify={simplify}, graph_tol={graph_tol}")

        t = threading.Thread(
            target=self._run_process,
            args=(threshold, simplify, graph_tol),
            daemon=True,
        )
        t.start()

    def _run_process(self, threshold, simplify, graph_tol):
        try:
            tree = self.analysis['tree']
            root = tree.getroot()
            paths = root.findall('.//{http://www.w3.org/2000/svg}path')
            if not paths:
                paths = root.findall('.//path')

            for idx, path_el in enumerate(paths):
                d = path_el.get('d', '')
                if not d:
                    continue

                self._log_safe(f"\nPath #{idx + 1}:")
                cmds = parse_commands(d)
                total_m = sum(1 for c, _ in cmds[1:] if c in ('M', 'm'))
                self._log_safe(f"  Sub-path breaks (M after first): {total_m}")

                # Step 1
                cmds, micro_joined = join_by_threshold(cmds, threshold)
                self._log_safe(f"  Micro-gaps joined (dist <= {threshold}): {micro_joined}")
                remaining = sum(1 for c, _ in cmds[1:] if c in ('M', 'm'))
                self._log_safe(f"  Remaining separate subpaths: {remaining}")

                # Step 2
                cmds, traced = trace_graph(cmds, tol=graph_tol)
                remaining2 = sum(1 for c, _ in cmds[1:] if c in ('M', 'm'))
                self._log_safe(f"  Graph-traced: {traced} merged ({remaining} -> {remaining2})")

                # Step 3: deduplicate near-coincident endpoints
                before3 = len(cmds)
                cmds, deduped = deduplicate_endpoints(cmds)
                if deduped:
                    self._log_safe(f"  Near-duplicate endpoints removed: {deduped} "
                                   f"({before3} -> {len(cmds)} commands)")

                # Step 4
                if simplify is not None and simplify > 0:
                    before = len(cmds)
                    cmds, removed = simplify_short_runs(cmds, simplify)
                    self._log_safe(f"  Simplified: {removed} segments removed "
                                   f"({before} -> {len(cmds)} commands)")

                # Step 5: close near-closed subpaths
                cmds, closed_count = close_subpaths(cmds)
                if closed_count:
                    self._log_safe(f"  Subpaths closed (start~=end): {closed_count}")

                path_el.set('d', cmds_to_str(cmds))

            self.root.after(0, self._on_process_done, tree)
        except Exception as e:
            self.root.after(0, self._on_process_error, str(e))

    def _log_safe(self, msg):
        """Thread-safe logging."""
        self.root.after(0, self.log, msg)

    def _on_process_error(self, err):
        self.log(f"Processing error: {err}")
        self.process_btn.state(['!disabled'])

    def _on_process_done(self, tree):
        self.processed_tree = tree
        self.log("\nProcessing complete.")
        self.process_btn.state(['!disabled'])
        self.save_btn.state(['!disabled'])

    # ── Save ──────────────────────────────────────────────────────

    def on_save(self):
        if self.processed_tree is None:
            return

        if self.current_file:
            base, ext = os.path.splitext(self.current_file)
            suggestion = base + '_fixed' + ext
        else:
            suggestion = 'output.svg'

        path = filedialog.asksaveasfilename(
            title="Save fixed SVG",
            initialfile=os.path.basename(suggestion),
            initialdir=os.path.dirname(suggestion),
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
            defaultextension=".svg",
        )
        if not path:
            return

        ET.register_namespace('', 'http://www.w3.org/2000/svg')
        self.processed_tree.write(path, xml_declaration=False, encoding='unicode')
        self.log(f"Saved: {path}")


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
    main()
