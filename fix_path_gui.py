"""
Tkinter GUI for SVG Path Fixer.

Loads an SVG, auto-analyzes gap and segment distributions to recommend
parameters, then processes the path using fix_path's 3-step pipeline.
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

LINE_CMDS = {'L', 'l', 'H', 'h', 'V', 'v'}


# ── Auto-analysis functions ───────────────────────────────────────────

def _find_cluster_boundary(values):
    """Find the index of the largest gap between consecutive sorted values.

    Uses log-scale jumps so it works across orders of magnitude
    (e.g. gaps of 0.005 vs 1.5 vs 200 all get detected correctly).
    Falls back to absolute jumps if all values are zero/negative.
    Returns the index i such that the split is values[:i+1] | values[i+1:].
    """
    # Filter out exact zeros for log-scale (keep them in lower cluster)
    min_nonzero = None
    for v in values:
        if v > 0:
            min_nonzero = v
            break

    best_jump, best_idx = 0, 0
    for i in range(len(values) - 1):
        a, b = values[i], values[i + 1]
        if min_nonzero and a > 0 and b > 0:
            jump = math.log(b) - math.log(a)  # log-ratio
        else:
            jump = b - a  # absolute fallback
        if jump > best_jump:
            best_jump = jump
            best_idx = i

    return best_idx

def analyze_gap_distribution(cmds):
    """Analyze M-command gap distances and recommend a join threshold.

    Returns dict with keys: gaps, micro_count, structural_count,
    boundary, recommended_threshold  (or None if <2 gaps).
    """
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
            pass  # position resets handled elsewhere
        else:
            cx, cy = get_endpoint(cmd, args, cx, cy)

    if len(gaps) < 2:
        return None

    gaps.sort()

    boundary_idx = _find_cluster_boundary(gaps)

    boundary = (gaps[boundary_idx] + gaps[boundary_idx + 1]) / 2
    rec = gaps[boundary_idx] * 1.05  # 5% above upper edge of micro cluster

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
    """Analyze original line segment lengths and recommend simplify.

    Analyzes the raw L/H/V segments (before gap-joining) so that
    micro-gap joins don't pollute the distribution.

    Returns dict with keys: segments, jitter_count, structural_count,
    boundary, recommended_simplify  (or None if <2 segments).
    """
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
        self.root.geometry("820x680")
        self.root.minsize(600, 500)

        self.current_file = None
        self.analysis = None
        self.processed_tree = None

        self.setup_ui()

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

        # Drag & drop zone
        if DND_AVAILABLE:
            self.file_label.configure(text="Drop SVG here or click Browse\u2026")
            inp.drop_target_register(DND_FILES)
            inp.dnd_bind('<<Drop>>', self.on_drop)

        # -- Analysis frame --
        ana = ttk.LabelFrame(self.root, text="Auto-Analysis", padding=6)
        ana.pack(fill='x', **pad)

        self.analysis_text = tk.Text(ana, height=5, wrap='word', state='disabled',
                                     bg='#f5f5f5', relief='flat', font=('Consolas', 9))
        self.analysis_text.pack(fill='x')

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
        self.auto_threshold_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grid, text="Auto", variable=self.auto_threshold_var).grid(row=0, column=2)

        # Simplify
        ttk.Label(grid, text="Simplify (segment):").grid(row=1, column=0, sticky='w', pady=2)
        self.simplify_var = tk.StringVar(value="0.5")
        self.simplify_entry = ttk.Entry(grid, textvariable=self.simplify_var, width=12)
        self.simplify_entry.grid(row=1, column=1, padx=4, pady=2)
        self.auto_simplify_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grid, text="Auto", variable=self.auto_simplify_var).grid(row=1, column=2, pady=2)
        self.enable_simplify_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grid, text="Enable", variable=self.enable_simplify_var).grid(row=1, column=3, pady=2)

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

        # -- Log frame --
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=6)
        log_frame.pack(fill='both', expand=True, **pad)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, wrap='word',
                                                   state='disabled', font=('Consolas', 9))
        self.log_text.pack(fill='both', expand=True)

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

    def set_analysis_text(self, text):
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
        # Windows wraps paths with spaces in {}
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
        self.set_analysis_text("Analyzing\u2026")
        self.process_btn.state(['disabled'])

        # Run analysis in background
        t = threading.Thread(target=self._run_analysis, args=(path,), daemon=True)
        t.start()

    def _run_analysis(self, path):
        try:
            result = auto_analyze_svg(path)
            self.root.after(0, self._on_analysis_done, result)
        except Exception as e:
            self.root.after(0, self._on_analysis_error, str(e))

    def _on_analysis_error(self, err):
        self.set_analysis_text(f"Error: {err}")
        self.log(f"Analysis error: {err}")

    def _on_analysis_done(self, result):
        if 'error' in result:
            self.set_analysis_text(result['error'])
            self.log(result['error'])
            return

        self.analysis = result
        self.process_btn.state(['!disabled'])

        lines = []
        lines.append(f"Paths: {result['path_count']}   "
                      f"Commands: {result['command_count']}   "
                      f"Subpaths: {result['subpath_count']}")

        ga = result['gap_analysis']
        if ga:
            lines.append(f"Gaps:     {ga['micro_count']} micro "
                          f"(\u2264 {ga['micro_max']:.4g}),  "
                          f"{ga['structural_count']} structural "
                          f"(\u2265 {ga['structural_min']:.4g})")
            lines.append(f"  \u2192 Recommended threshold: {ga['recommended_threshold']:.6g}")
            if self.auto_threshold_var.get():
                self.threshold_var.set(f"{ga['recommended_threshold']:.6g}")
        else:
            lines.append("Gaps: none detected")

        sa = result['segment_analysis']
        if sa:
            lines.append(f"Segments: {sa['jitter_count']} jitter "
                          f"(\u2264 {sa['jitter_max']:.4g}),  "
                          f"{sa['structural_count']} structural "
                          f"(\u2265 {sa['structural_min']:.4g})")
            lines.append(f"  \u2192 Recommended simplify: {sa['recommended_simplify']:.6g}")
            if self.auto_simplify_var.get():
                self.simplify_var.set(f"{sa['recommended_simplify']:.6g}")
        else:
            lines.append("Segments: no line segments found")

        self.set_analysis_text('\n'.join(lines))
        self.log("Analysis complete.")

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

        # Suggest output name
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

    # Accept file path as CLI argument for convenience
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        root.after(100, app.load_file, sys.argv[1])

    root.mainloop()


if __name__ == '__main__':
    main()
