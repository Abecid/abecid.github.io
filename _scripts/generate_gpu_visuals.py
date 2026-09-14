"""Generate the GPU post's responsive SVG diagrams (Python standard library).

    python3 _scripts/generate_gpu_visuals.py

All timing and roofline values are illustrative, not benchmarks. The GPU
layout is a logical schematic, not a physical floorplan. Coalescing assumes
32 active lanes, aligned FP32 loads, and 32-byte sectors (CUDA CC >= 6.0).
"""

from html import escape
from math import log10
from pathlib import Path


OUT = Path(__file__).resolve().parents[1] / "assets/img/blogs/gpu-systems"
BG, PANEL, INK, MUTED = "#131d2e", "#1c2a40", "#edf3fc", "#b5c4d9"
BLUE, TEAL, ORANGE, GRID = "#89b4ff", "#65dfc0", "#ffbd7a", "#3e506a"


class SVG:
    def __init__(self, name, height, mobile, title, desc):
        self.name, self.w, self.h = name, 390 if mobile else 900, height
        self.mobile = mobile
        self.font = 18 if mobile else 24
        self.items = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
            f'height="{height}" viewBox="0 0 {self.w} {height}" role="img" '
            'aria-labelledby="title desc">',
            f"<title id=\"title\">{escape(title)}</title>",
            f"<desc id=\"desc\">{escape(desc)}</desc>",
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" '
            'refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 Z" fill="context-stroke"/></marker></defs>',
            '<g font-family="Arial, Helvetica, sans-serif">',
        ]
        self.rect(0, 0, self.w, height, BG, radius=14)

    def rect(self, x, y, w, h, fill=PANEL, stroke="none", radius=0, sw=1.5):
        self.items.append(
            f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" '
            f'rx="{radius:g}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}"/>'
        )

    def text(self, x, y, value, size=None, color=INK, anchor="start", weight=400):
        self.items.append(
            f'<text x="{x:g}" y="{y:g}" fill="{color}" font-size="{size or self.font:g}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>'
        )

    def line(self, x1, y1, x2, y2, color=GRID, sw=1.5, dash=None, arrow=False):
        attrs = f' stroke-dasharray="{dash}"' if dash else ""
        if arrow:
            attrs += ' marker-end="url(#arrow)"'
        self.items.append(
            f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
            f'stroke="{color}" stroke-width="{sw:g}"{attrs}/>'
        )

    def circle(self, x, y, radius, color):
        self.items.append(f'<circle cx="{x:g}" cy="{y:g}" r="{radius:g}" fill="{color}"/>')

    def path(self, points, color, sw=3):
        d = "M " + " L ".join(f"{x:g} {y:g}" for x, y in points)
        self.items.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw:g}"/>')

    def heading(self, value, note=None):
        self.text(22 if self.mobile else 32, 36 if self.mobile else 44, value,
                  size=21 if self.mobile else 28, weight=600)
        if note:
            self.text(22 if self.mobile else 32, 63 if self.mobile else 75, note,
                      color=MUTED, size=17 if self.mobile else 21)

    def save(self):
        OUT.mkdir(parents=True, exist_ok=True)
        suffix = "-mobile" if self.mobile else ""
        (OUT / f"{self.name}{suffix}.svg").write_text("\n".join(self.items + ["</g>", "</svg>"]) + "\n")


def gpu_layout(mobile=False):
    s = SVG("gpu-layout", 626 if mobile else 530, mobile, "Inside a GPU",
            "Logical GPU package schematic: HBM stacks sit outside the GPU die. "
            "The die contains streaming multiprocessors and shared L2 cache. "
            "Each SM contains registers, compute units, and L1 cache/shared memory.")
    s.heading("Inside a GPU", "Logical layout · not to scale")
    px, py, pw, ph = (16, 85, 358, 520) if mobile else (26, 96, 848, 406)
    s.rect(px, py, pw, ph, PANEL, GRID, 14)
    s.text(px + 15, py + 28, "GPU package", color=MUTED, size=17 if mobile else 21)
    if mobile:
        dx, dy, dw, dh = 35, 138, 320, 360
        memories = [(67, 539, 105, 44), (218, 539, 105, 44)]
        sms = [(51, 188, 136, 207), (203, 188, 136, 207)]
        lx, ly, lw, lh = 51, 431, 288, 45
    else:
        dx, dy, dw, dh = 181, 145, 538, 322
        memories = [(51, 194, 99, 78), (51, 352, 99, 78),
                    (750, 194, 99, 78), (750, 352, 99, 78)]
        sms = [(201, 195, 157, 168), (372, 195, 157, 168), (543, 195, 157, 168)]
        lx, ly, lw, lh = 201, 395, 499, 49
    s.rect(dx, dy, dw, dh, BG, BLUE, 12, 2)
    s.text(dx + 16, dy + 29, "GPU die", color=BLUE, size=18 if mobile else 23, weight=600)
    for x, y, w, h in memories:
        if mobile:
            s.line(x + w / 2, y, x + w / 2, dy + dh, ORANGE, 3)
        else:
            edge = x + w if x < dx else x
            s.line(edge, y + h / 2, dx if x < dx else dx + dw, y + h / 2, ORANGE, 3)
        for off in [8, 4, 0]:
            s.rect(x, y - off, w, h, BG, ORANGE, 4)
        s.text(x + w / 2, y + h / 2 + 7, "HBM", color=ORANGE, anchor="middle", weight=600)
    for i, (x, y, w, h) in enumerate(sms):
        s.rect(x, y, w, h, PANEL, TEAL, 8)
        s.text(x + w / 2, y + 29, f"SM {i}", color=TEAL, anchor="middle", weight=600)
        labels = ["Registers", "Compute", "L1 / shared"]
        if mobile:
            labels = ["Registers", "Compute", "L1 / shared"]
        for j, label in enumerate(labels):
            yy = y + 45 + j * (49 if mobile else 37)
            s.rect(x + 9, yy, w - 18, 39 if mobile else 29, BG, radius=4)
            s.text(x + w / 2, yy + (25 if mobile else 21), label,
                   size=17 if mobile else 20, anchor="middle")
        s.line(x + w / 2, y + h, x + w / 2, ly, TEAL, 2.5)
    s.rect(lx, ly, lw, lh, BLUE, radius=5)
    s.text(lx + lw / 2, ly + lh / 2 + 7, "L2 · shared across SMs", color=BG,
           size=18 if mobile else 24, anchor="middle", weight=600)
    s.save()


def amdahl(mobile=False):
    s = SVG("amdahl", 399 if mobile else 344, mobile, "A small kernel has a small ceiling",
            "Illustrative 100 ms workload: 95 ms other work and 5 ms target kernel. "
            "A three-times faster kernel gives 96.67 ms total; removing it entirely gives 95 ms.")
    s.heading("Only 5 ms can shrink")
    x, width = (22, 265) if mobile else (204, 572)
    top, gap, bh = (114, 95, 30) if mobile else (111, 77, 36)
    for i, (label, target, total) in enumerate([
        ("Original", 5, "100"), ("3× kernel", 5 / 3, "96.67"), ("∞× kernel", 0, "95")
    ]):
        y = top + i * gap
        s.text(x if mobile else 32, y - 12 if mobile else y + 25, label)
        s.rect(x, y, width * .95, bh, BLUE)
        if target:
            s.rect(x + width * .95, y, width * target / 100, bh, ORANGE)
        s.text(x + width + 10, y + bh / 2 + 7, total, size=18 if mobile else 24, weight=600)
        if i == 0:
            s.text(x + width * .475, y + bh / 2 + 7, "95 ms", color=BG, anchor="middle", weight=600)
    # The legend is separate from the 5 ms bar, which must remain proportional.
    ly = 65 if mobile else 69
    s.rect(22 if mobile else 32, ly - 13, 13, 13, BLUE)
    s.text(43 if mobile else 53, ly, "Other work", size=17 if mobile else 21, color=MUTED)
    s.rect(179 if mobile else 221, ly - 13, 13, 13, ORANGE)
    s.text(200 if mobile else 242, ly, "Target", size=17 if mobile else 21, color=MUTED)
    s.text(s.w - 22, s.h - 18, "Total: ms", color=MUTED, size=17 if mobile else 21, anchor="end")
    s.save()


def roofline(mobile=False):
    s = SVG("roofline", 439 if mobile else 451, mobile, "Bandwidth and compute set the roofline",
            "Hypothetical GPU: 2 TB/s memory bandwidth and 120 TFLOP/s compute. "
            "The roofline is min(2 times arithmetic intensity, 120) TFLOP/s; "
            "the ridge is 60 FLOP per byte. Both axes use logarithmic scales.")
    s.heading("Bandwidth → compute", "Example: 2 TB/s · 120 TFLOP/s")
    x0, x1, y0, y1 = (59, 361, 354, 111) if mobile else (95, 851, 358, 114)
    xx = lambda value: x0 + (log10(value) + 1) / 4 * (x1 - x0)
    yy = lambda value: y0 - (log10(value) + 1) / (log10(200) + 1) * (y0 - y1)
    for value in [.1, 1, 10, 120]:
        y = yy(value)
        s.line(x0, y, x1, y, GRID, 1)
        s.text(x0 - 10, y + 6, f"{value:g}", size=17 if mobile else 21, color=MUTED, anchor="end")
    for value in [.1, 1, 10, 1000]:
        x = xx(value)
        s.line(x, y0, x, y0 + 6, MUTED)
        s.text(x, y0 + 28, f"{value:g}", size=17 if mobile else 21, color=MUTED, anchor="middle")
    s.line(x0, y0, x1, y0, MUTED)
    s.line(x0, y0, x0, y1, MUTED)
    s.line(xx(60), yy(120), xx(60), y0, MUTED, 1.5, "4 5")
    s.path([(xx(.1), yy(.2)), (xx(60), yy(120)), (xx(1000), yy(120))], TEAL, 4)
    s.circle(xx(60), yy(120), 5, TEAL)
    s.text(xx(60), y0 + 28, "60", size=17 if mobile else 21, color=TEAL, anchor="middle")
    s.text(x0, y1 - 14, "TFLOP/s", color=MUTED, size=17 if mobile else 21)
    s.text(xx(.4), yy(4), "2 TB/s", color=TEAL, weight=600)
    s.text(x1 - 3, yy(120) - 15, "120", color=TEAL, anchor="end", weight=600)
    s.text((x0 + x1) / 2, s.h - 19, "FLOP / byte · log scales", color=MUTED,
           size=17 if mobile else 21, anchor="middle")
    s.save()


def coalescing(mobile=False):
    s = SVG("coalescing", 538 if mobile else 438, mobile, "Contiguous lanes use fewer memory sectors",
            "For 32 aligned FP32 loads, contiguous lanes use four 32-byte sectors: 128 bytes. "
            "Stride-eight loads use 32 sectors: 1024 bytes. Each load requests the same 128 useful bytes.")
    s.heading("Same 128 B requested", "32 lanes × 4 B · aligned loads")
    x, width = (22, 346) if mobile else (32, 836)
    for index, (title, color, stride) in enumerate([
        ("Contiguous", TEAL, False), ("Stride: 8 floats", ORANGE, True)
    ]):
        y = (108 + index * 207) if mobile else (109 + index * 153)
        s.text(x, y, title, weight=600)
        # One colored cell = one lane's four useful bytes.
        if not stride:
            gap = 5 if mobile else 12
            groupw = (width - 3 * gap) / 4
            cellw = groupw / 8
            for group in range(4):
                gx = x + group * (groupw + gap)
                s.rect(gx, y + 20, groupw, 48, BG, color, 3)
                for lane in range(8):
                    s.rect(gx + lane * cellw + 1, y + 21, cellw - 2, 46, color)
            s.text(x, y + 96, "4 sectors · 128 B", color=color, weight=600)
        else:
            cols = 8 if mobile else 16
            gap = 5 if mobile else 7
            bw = (width - (cols - 1) * gap) / cols
            rows = 32 // cols
            for lane in range(32):
                gx = x + (lane % cols) * (bw + gap)
                gy = y + 20 + (lane // cols) * 25
                s.rect(gx, gy, bw, 19, PANEL, color, 2, 1)
                s.rect(gx + 1, gy + 1, (bw - 2) / 8, 17, color)
            s.text(x, y + 20 + rows * 25 + 24, "32 sectors · 1,024 B", color=color, weight=600)
        if index == 0:
            s.text(x, y + (130 if mobile else 122), "Each outlined group = 32 B", color=MUTED,
                   size=17 if mobile else 21)
    s.save()


def fusion(mobile=False):
    s = SVG("fusion", 465 if mobile else 476, mobile, "Fusion keeps intermediates on chip",
            "Three elementwise kernels perform multiply, add, and activation with two "
            "intermediate arrays in global memory. One fused kernel keeps those intermediates "
            "on chip, reducing ideal array traffic from 24N to 8N bytes for FP32.")
    s.heading("Keep intermediates on chip")
    x0, x1 = (35, 355) if mobile else (65, 835)
    for panel, fused in enumerate([False, True]):
        top = 89 + panel * 181
        s.text(22 if mobile else 32, top, "1 fused kernel" if fused else "3 kernels", weight=600)
        s.text(s.w - 22 if mobile else s.w - 32, top, "8N B" if fused else "24N B",
               color=TEAL if fused else ORANGE, anchor="end", weight=600)
        cy, my = top + 53, top + 108
        step = (x1 - x0) / 3
        if fused:
            s.rect(x0 + step / 2 - 27, cy - 28, (x1 - x0) - step + 54, 57,
                   PANEL, TEAL, 8)
        s.rect(x0 - 12, my - 5, x1 - x0 + 24, 36, PANEL, radius=4)
        for i in range(3):
            cx = x0 + step * (i + .5)
            s.rect(cx - 24, cy - 22, 48, 44, BG, TEAL if fused else BLUE, 6)
            s.text(cx, cy + 7, ["×a", "+b", "φ"][i], anchor="middle", weight=600)
            if not fused or i == 0:
                s.line(x0 + step * i, my - 7, cx - 16, cy + 25,
                       BLUE if i == 0 else ORANGE, 2.5, arrow=True)
            if not fused or i == 2:
                s.line(cx + 16, cy + 25, x0 + step * (i + 1), my - 7,
                       TEAL if i == 2 else ORANGE, 2.5, arrow=True)
            if fused and i < 2:
                s.line(cx + 28, cy, cx + step - 31, cy, TEAL, 2.5, arrow=True)
        for i, label in enumerate(["x", "z₁", "z₂", "y"]):
            if not fused or i in [0, 3]:
                s.text(x0 + i * step, my + 19, label,
                       color=BLUE if i == 0 else TEAL if i == 3 else ORANGE, anchor="middle")
        s.text(s.w / 2, my + 52, "Global memory", color=MUTED,
               size=17 if mobile else 21, anchor="middle")
    s.text(s.w - 22, s.h - 14, "FP32 · N elements · ideal traffic", color=MUTED,
           size=17 if mobile else 21, anchor="end")
    s.save()


def overlap(mobile=False):
    s = SVG("overlap", 447 if mobile else 414, mobile, "Overlap hides communication time",
            "Illustrative schedule: eight milliseconds compute and six milliseconds collective "
            "communication take fourteen milliseconds serialized. Starting communication at "
            "four milliseconds overlaps four milliseconds and finishes at ten milliseconds.")
    s.heading("14 ms → 10 ms", "4 ms of communication hidden")
    x0, width = (101, 265) if mobile else (182, 674)
    scale = width / 14
    for i, (title, start, end) in enumerate([("Serialized", 8, 14), ("Overlapped", 4, 10)]):
        y = 112 + i * (145 if mobile else 135)
        s.text(22 if mobile else 32, y, title, weight=600)
        for row, label, time, duration, color in [
            (0, "Compute", 0, 8, BLUE), (1, "Comm.", start, 6, ORANGE)
        ]:
            by = y + 20 + row * 44
            s.text(x0 - 10, by + 24, label, size=17 if mobile else 23, anchor="end", color=MUTED)
            s.rect(x0 + time * scale, by, duration * scale, 34, color, radius=3)
            s.text(x0 + (time + duration / 2) * scale, by + 24, f"{duration} ms", color=BG,
                   size=18 if mobile else 24, anchor="middle", weight=600)
        s.line(x0 + end * scale, y + 15, x0 + end * scale, y + 103, TEAL, 1.5, "4 4")
    axisy = 398 if mobile else 365
    s.line(x0, axisy, x0 + width, axisy, MUTED)
    for value in [0, 4, 8, 10, 14]:
        x = x0 + value * scale
        s.line(x, axisy, x, axisy + 5, MUTED)
        s.text(x, axisy + 27, value, color=MUTED, size=17 if mobile else 21, anchor="middle")
    s.text(22 if mobile else 32, axisy + 27, "ms" if mobile else "Time (ms)",
           color=MUTED, size=17 if mobile else 21)
    s.save()


def batching(mobile=False):
    s = SVG("batching", 447 if mobile else 414, mobile, "Refill a batch slot when a request finishes",
            "Illustrative two-slot schedule with A taking four iterations, B two, and C two. "
            "Static batching starts C after A finishes and ends at iteration six. Continuous "
            "batching replaces B with C at iteration two and ends at iteration four.")
    s.heading("Refill an empty slot", "A: 4 iterations · B: 2 · C: 2")
    x0, width = (92, 274) if mobile else (167, 689)
    scale = width / 6
    schedules = [
        ("Static", 6, [(0, 0, 4, "A", BLUE), (1, 0, 2, "B", ORANGE), (0, 4, 2, "C", TEAL)]),
        ("Continuous", 4, [(0, 0, 4, "A", BLUE), (1, 0, 2, "B", ORANGE), (1, 2, 2, "C", TEAL)]),
    ]
    for i, (title, end, jobs) in enumerate(schedules):
        y = 112 + i * (145 if mobile else 135)
        s.text(22 if mobile else 32, y, title, weight=600)
        for row in [0, 1]:
            by = y + 20 + row * 44
            s.text(x0 - 10, by + 24, f"Slot {row + 1}", color=MUTED,
                   size=17 if mobile else 23, anchor="end")
            s.rect(x0, by, end * scale, 34, PANEL, GRID, 3)
        for row, start, duration, label, color in jobs:
            by = y + 20 + row * 44
            s.rect(x0 + start * scale + 1, by, duration * scale - 2, 34, color, radius=3)
            s.text(x0 + (start + duration / 2) * scale, by + 24, label,
                   color=BG, anchor="middle", weight=600)
        s.line(x0 + end * scale, y + 15, x0 + end * scale, y + 103, TEAL, 1.5, "4 4")
    axisy = 398 if mobile else 365
    s.line(x0, axisy, x0 + width, axisy, MUTED)
    for value in [0, 2, 4, 6]:
        x = x0 + value * scale
        s.line(x, axisy, x, axisy + 5, MUTED)
        s.text(x, axisy + 27, value, color=MUTED, size=17 if mobile else 21, anchor="middle")
    s.text(14 if mobile else 32, axisy + 27, "Iteration", color=MUTED, size=17 if mobile else 21)
    s.save()


def denoising_budget(mobile=False):
    s = SVG("denoising-budget", 411 if mobile else 356, mobile, "Fewer denoising steps expose fixed work",
            "Illustrative latency is 40 milliseconds per denoising step plus 200 milliseconds fixed. "
            "Twenty steps take 1000 milliseconds, four take 360, and one takes 240.")
    s.heading("40K + 200 ms", "K denoising steps + fixed work")
    x0, width = (22, 263) if mobile else (146, 617)
    top, gap, bh = (117, 94, 32) if mobile else (111, 74, 38)
    for i, (k, label) in enumerate([(20, "1,000"), (4, "360"), (1, "240")]):
        y = top + i * gap
        s.text(x0 if mobile else 32, y - 12 if mobile else y + 26, f"K = {k}")
        dynamic = width * 40 * k / 1000
        fixed = width * .2
        s.rect(x0, y, dynamic, bh, BLUE)
        s.rect(x0 + dynamic, y, fixed, bh, ORANGE)
        s.text(x0 + dynamic + fixed + 11, y + bh / 2 + 7, label,
               size=18 if mobile else 24, weight=600)
    ly = s.h - 27
    s.rect(22 if mobile else 32, ly - 13, 13, 13, BLUE)
    s.text(43 if mobile else 53, ly, "Denoiser", color=MUTED, size=17 if mobile else 21)
    s.rect(179 if mobile else 242, ly - 13, 13, 13, ORANGE)
    s.text(200 if mobile else 263, ly, "Fixed", color=MUTED, size=17 if mobile else 21)
    s.text(s.w - 22, ly, "ms", color=MUTED, size=17 if mobile else 21, anchor="end")
    s.save()


def tiling(mobile=False):
    s = SVG("tiling", 638 if mobile else 618, mobile, "Matrix tiles reuse each input four times",
            "A four-by-four A tile and a four-by-four B tile contribute to a four-by-four C tile. "
            "One highlighted A element contributes to four columns of its C row; one highlighted "
            "B element contributes to four rows of its C column. Both inputs contribute at the "
            "intersection. Loading the two FP32 input tiles once transfers 128 bytes and enables "
            "64 multiply-adds, counted as 128 FLOPs. Output C traffic is excluded.")
    s.heading("Load once; reuse on chip", "4 × 4 tiles · FP32")
    size = 128 if mobile else 160
    cell = size / 4
    ax, bx, cx = (18, 236, 236) if mobile else (220, 590, 590)
    ay, by, cy = (355, 123, 355) if mobile else (320, 100, 320)
    row, k, col = 1, 2, 1
    s.text(22 if mobile else 220, 157 if mobile else 180, "C += A × B",
           size=23 if mobile else 30, weight=600)
    s.text(22 if mobile else 220, 184 if mobile else 211, "one K tile",
           size=17 if mobile else 21, color=MUTED)

    def tile(x, y, kind):
        for i in range(4):
            for j in range(4):
                color = PANEL
                if kind == "A" and (i, j) == (row, k):
                    color = BLUE
                elif kind == "B" and (i, j) == (k, col):
                    color = ORANGE
                elif kind == "C":
                    if i == row:
                        color = BLUE
                    if j == col:
                        color = ORANGE
                    if (i, j) == (row, col):
                        color = INK
                s.rect(x + j * cell, y + i * cell, cell, cell,
                       color, GRID, radius=1, sw=1.5)
        if kind in ["A", "B"]:
            i, j = (row, k) if kind == "A" else (k, col)
            s.text(x + (j + .5) * cell, y + (i + .5) * cell + 6,
                   kind.lower(), size=18 if mobile else 23,
                   color=BG, anchor="middle", weight=600)
        else:
            s.text(x + (col + .5) * cell, y + (row + .5) * cell + 6,
                   "ab", size=17 if mobile else 21,
                   color=BG, anchor="middle", weight=600)

    tile(ax, ay, "A")
    tile(bx, by, "B")
    tile(cx, cy, "C")
    s.text(ax + size / 2, ay - 16, "A tile", color=BLUE, anchor="middle", weight=600)
    s.text(bx + size / 2, by - 16, "B tile", color=ORANGE, anchor="middle", weight=600)
    s.text(cx + size / 2, cy + size + 28, "C tile", anchor="middle", weight=600)

    # Match the A element's row and the B element's column to their C outputs.
    ry = cy + (row + .5) * cell
    ccol = cx + (col + .5) * cell
    s.line(ax + size + 8, ry, cx - 12, ry, BLUE, 2.5, arrow=True)
    s.text((ax + size + cx) / 2, ry - 19, "reuse 4×", color=BLUE,
           size=17 if mobile else 23, anchor="middle", weight=600)
    s.line(ccol, by + size + 8, ccol, cy - 14, ORANGE, 2.5, arrow=True)
    s.text(ccol + 15, (by + size + cy) / 2 + 6, "reuse 4×", color=ORANGE,
           size=17 if mobile else 23, weight=600)

    # The white intersection is one partial sum, not the final C value.
    s.text(s.w / 2, 555 if mobile else 535, "Each C entry sums 4 products", color=MUTED,
           size=17 if mobile else 21, anchor="middle")
    s.text(s.w / 2, 590 if mobile else 566, "128 input bytes → 128 FLOPs",
           size=20 if mobile else 26, anchor="middle", weight=600)
    s.text(s.w / 2, 616 if mobile else 589, "C traffic excluded", color=MUTED,
           size=17 if mobile else 20, anchor="middle")
    s.save()


if __name__ == "__main__":
    for draw in [gpu_layout, amdahl, roofline, coalescing, fusion, overlap, batching, denoising_budget, tiling]:
        draw()
        draw(mobile=True)
