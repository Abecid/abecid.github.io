#!/usr/bin/env python3
"""Generate the Flow Map post’s deterministic desktop and mobile SVG diagrams.

Run from any directory: python3 _scripts/generate_flowmap_visuals.py
Only Python’s standard library is required.
"""
from pathlib import Path
import math
import random
import re
from html import escape
from xml.dom import minidom
root = Path(__file__).resolve().parents[1] / 'assets/img/blogs/flowmap'
root.mkdir(parents=True, exist_ok=True)
BG = '#131d2e'
FG = '#e8edf6'
MUT = '#aab9cf'
TEAL = '#65dfc0'
ORANGE = '#ffbd7a'
BLUE = '#89b4ff'
LINE = '#3c506a'


def text(x, y, s, size=24, color=FG, anchor='start', weight=400):
    return f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{escape(s)}</text>'


def line(x1, y1, x2, y2, color=LINE, width=2, dash='', arrow=False):
    return f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{color}" stroke-width="{width}" fill="none"' + (f' stroke-dasharray="{dash}"' if dash else '') + (f' marker-end="url(#{color[1:]})"' if arrow else '') + '/>'


def path(d, color=BLUE, width=4, fill='none', dash='', arrow=False):
    return f'<path d="{d}" stroke="{color}" stroke-width="{width}" fill="{fill}"' + (f' stroke-dasharray="{dash}"' if dash else '') + (f' marker-end="url(#{color[1:]})"' if arrow else '') + '/>'


def circle(x, y, r=7, color=FG, fill=None):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill or color}" stroke="{color}" stroke-width="2"/>'


def rect(x, y, w, h, fill=BG, stroke=LINE):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'


def svg(name, w, h, body, title, desc):
    defs = ''.join((f'<marker id="{c[1:]}" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M1,1 L9,5 L1,9" fill="none" stroke="{c}" stroke-width="2"/></marker>' for c in [TEAL, ORANGE, BLUE, MUT]))
    document = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><defs>{defs}</defs><rect width="{w}" height="{h}" rx="16" fill="{BG}"/><g font-family="Arial, Helvetica, sans-serif" stroke-linecap="round" stroke-linejoin="round">{body}</g></svg>'
    document = document.replace('sₚ − sᵩ', 's<tspan baseline-shift="sub" font-size="75%">p</tspan> − s<tspan baseline-shift="sub" font-size="75%">q</tspan>')
    formatted = minidom.parseString(document).toprettyxml(indent='  ')
    # Keep mixed text/tspan content inline: XML indentation changes SVG spacing.
    text_nodes = iter(re.findall(r'<text\b.*?</text>', document, flags=re.S))
    def preserve_text(_):
        node = next(text_nodes)
        return ('<!-- prettier-ignore -->\n    ' if '<tspan' in node else '') + node
    formatted = re.sub(r'<text\b.*?</text>', preserve_text, formatted, flags=re.S)
    (root / name).write_text(formatted)
# Fixed seed keeps the two-cluster interpolation reproducible.
rng = random.Random(21)
samples = [((-1 if i % 2 else 1) + rng.gauss(0, 0.14), rng.gauss(0, 0.22), rng.gauss(0, 0.85), rng.gauss(0, 0.85), i % 2) for i in range(80)]
# Generate a separate layout for each viewport.
for mobile in [False, True]:
    w, h = (400, 810) if mobile else (960, 310)
    b = ''
    fs = 20 if mobile else 24
    for k, t in enumerate([1, 0.5, 0]):
        cx, cy = (200, 120 + k * 254) if mobile else (160 + 320 * k, 135)
        b += text(cx, cy - 82, ['Noise · t = 1', 'Mixed · t = ½', 'Data · t = 0'][k], fs, FG, 'middle', 700)
        b += rect(cx - 113, cy - 60, 226, 152, stroke=LINE)
        for x0, y0, x1, y1, c in samples:
            px = cx + ((1 - t) * x0 + t * x1) * 43
            py = cy + 16 + ((1 - t) * y0 + t * y1) * 26
            b += circle(round(px, 2), round(py, 2), 2.8, TEAL if c else BLUE)
        if k < 2:
            if mobile:
                b += line(200, cy + 110, 200, cy + 142, MUT, 2, arrow=True)
            else:
                b += line(cx + 128, cy + 16, cx + 185, cy + 16, MUT, 2, arrow=True)
    b += text(w / 2, h - 20, 'Generation: 1 → 0', fs, TEAL, 'middle')
    svg('diffusion' + ('-mobile' if mobile else '') + '.svg', w, h, b, 'Noise to data', 'Toy linear interpolation marginals at t=1, one half, and zero. Two data clusters emerge as noise decreases. Colors track data cluster membership.')
for mobile in [False, True]:
    w, h = (400, 420) if mobile else (960, 440)
    b = ''
    fs = 20 if mobile else 24
    x0, x1 = (55, 345) if mobile else (140, 820)
    y0, y1 = (130, 275) if mobile else (155, 330)
    b += text(24 if mobile else 40, 42, 'A tangent cannot predict the whole turn.', fs, FG, weight=700) if not mobile else text(24, 38, 'Local direction ≠ destination', 20, FG, weight=700)
    b += path(f'M{x0},{y0} C{x0 + (x1 - x0) * 0.54},{y0} {x0 + (x1 - x0) * 0.5},{y1} {x1},{y1}', BLUE, 4)
    b += line(x0, y0, x1, y0, ORANGE, 3, '9 8', True)
    b += line(x0, y0 + 8, x1 - 6, y1 - 3, TEAL, 4, arrow=True)
    b += circle(x0, y0, 8)
    b += circle(x1, y1, 8, TEAL)
    b += circle(x1, y0, 8, ORANGE, BG)
    b += text(x0, y0 - 25, 'Start', fs, FG)
    b += text(x1, y0 - 25, 'Euler misses', fs, ORANGE, 'end')
    b += text(x1, y1 + 37, 'Destination', fs, TEAL, 'end')
    b += text(x0 + 25, y1 + 65 if mobile else y1 + 62, 'Map: start → destination', 18 if mobile else 24, TEAL)
    b += text(x0 + 45, y1 - 12, 'ODE path', fs, BLUE, 'middle')
    svg('transport' + ('-mobile' if mobile else '') + '.svg', w, h, b, 'Velocity versus flow map', 'Orange extrapolates the initial tangent and misses the endpoint of the blue curved ODE path. The teal chord is the flow map displacement.')
for mobile in [False, True]:
    w, h = (400, 660) if mobile else (960, 345)
    b = ''
    fs = 20 if mobile else 24
    for k in [0, 1]:
        ox, oy = (58, 244 + k * 310) if mobile else (74 + 480 * k, 252)
        pw, scale = (284, 70) if mobile else (330, 75)
        b += text(ox, oy - 190, 'Changing velocity v' if k == 0 else 'Average velocity u', fs, FG, weight=700)
        b += line(ox, oy, ox + pw + 12, oy, MUT, 2)
        b += line(ox, oy, ox, oy - 166, MUT, 2)
        b += text(ox, oy + 29, 'r', fs, MUT, 'middle')
        b += text(ox + pw, oy + 29, 't', fs, MUT, 'middle')
        if k == 0:
            pts = [(ox + pw * i / 100, oy - scale * (1 + (i / 100) ** 2)) for i in range(101)]
            d = 'M' + f'{ox},{oy} ' + ' '.join(('L' + f'{x:.2f},{y:.2f}' for x, y in pts)) + f' L{ox + pw},{oy} Z'
            b += path(d, BLUE, 3, '#273d5b')
            b += text(ox + pw * 0.5, oy - 27, '∫ v dτ', fs, BLUE, 'middle')
        else:
            b += f'<rect x="{ox}" y="{oy - scale * 4 / 3}" width="{pw}" height="{scale * 4 / 3}" fill="#1c4948" stroke="{TEAL}" stroke-width="3"/>'
            b += text(ox + pw * 0.5, oy - 27, '(t − r) u', fs, TEAL, 'middle')
        if k == 0:
            b += text(w / 2 if mobile else 472, oy + 60 if mobile else oy - 56, '=', 32, FG, 'middle', 700)
    b += text(w / 2, h - 17, 'Equal area = equal displacement', 20 if mobile else 24, MUT, 'middle')
    svg('meanflow-jvp' + ('-mobile' if mobile else '') + '.svg', w, h, b, 'MeanFlow learns the average velocity', 'Area under the varying velocity curve equals a rectangle of average height u over the same time interval.')
for mobile in [False, True]:
    w, h = (400, 670) if mobile else (960, 365)
    b = ''
    fs = 20 if mobile else 24
    for k in [0, 1]:
        ox, oy = (38, 122 + k * 315) if mobile else (60 + 480 * k, 136)
        span = 320 if mobile else 350
        b += text(ox, oy - 80, 'MeanFlow' if k == 0 else 'TVM', fs, FG, weight=700)
        b += text(ox, oy - 48, 'Move the start' if k == 0 else 'Move the destination', fs, ORANGE)
        b += path(f'M{ox},{oy + 40} C{ox + span * 0.45},{oy - 35} {ox + span * 0.65},{oy + 125} {ox + span},{oy + 65}', BLUE, 3)
        sx, sy = (ox, oy + 40)
        ex, ey = (ox + span, oy + 65)
        b += line(sx + 5, sy + 5, ex - 7, ey + 1, TEAL, 3, arrow=True)
        b += circle(sx, sy, 6)
        b += circle(ex, ey, 6)
        if k == 0:
            nx, ny = (ox + 58, oy + 25)
            b += circle(ex, ey, 13, FG, BG) + circle(ex, ey, 5, FG)
            b += circle(nx, ny, 6, ORANGE)
            b += line(sx + 5, sy - 4, nx - 6, ny - 2, ORANGE, 3, arrow=True)
            b += line(nx + 4, ny + 5, ex - 13, ey - 4, TEAL, 2, '5 5', True)
            b += text(ex, ey + 40, 'fixed r', fs, FG, 'end')
            b += text(sx, sy + 58, 't changes', fs, ORANGE)
        else:
            nx, ny = (ox + span - 52, oy + 81)
            b += circle(sx, sy, 13, FG, BG) + circle(sx, sy, 5, FG)
            b += circle(nx, ny, 6, ORANGE)
            b += line(ex - 6, ey + 5, nx + 8, ny + 1, ORANGE, 3, arrow=True)
            b += line(sx + 13, sy - 3, nx - 5, ny - 5, TEAL, 2, '5 5', True)
            b += text(sx, sy + 58, 'fixed (x, t)', fs, FG)
            b += text(ex, ey + 67, 's changes', fs, ORANGE, 'end')
    b += text(w / 2, h - 20, '○ = held fixed', fs, MUT, 'middle')
    svg('endpoint-derivatives' + ('-mobile' if mobile else '') + '.svg', w, h, b, 'Two ways to constrain a jump', 'MeanFlow varies the starting state and start time along an ODE path while holding destination time fixed. TVM holds the input fixed and varies the destination time. A ring marks the fixed endpoint.')
for mobile in [False, True]:
    w, h = (400, 395) if mobile else (960, 360)
    b = ''
    fs = 20 if mobile else 24
    left, right = (35, 365) if mobile else (95, 865)
    base = 278 if mobile else 265

    def X(x):
        return left + (x + 4) / 8 * (right - left)
    b += text(24 if mobile else 40, 38, 'Correct the generated distribution', 19 if mobile else 25, FG, weight=700)
    for mean, color, label in [(-1, TEAL, 'Target p'), (1, ORANGE, 'Student q')]:
        pts = [(X(-4 + 8 * i / 180), base - 140 * math.exp(-(-4 + 8 * i / 180 - mean) ** 2 / 2)) for i in range(181)]
        b += path('M' + ' L'.join((f'{x:.2f},{y:.2f}' for x, y in pts)), color, 4)
        b += text((120 if mean < 0 else 280) if mobile else X(mean), base - 157, label, fs, color, 'middle', 700)
    b += line(left, base, right, base, MUT, 2)
    for i in [-2, 0, 2]:
        b += text(X(i), base + 29, str(i), fs, MUT, 'middle')
    b += line(X(1.3), base - 53, X(-0.4), base - 53, BLUE, 3, arrow=True)
    b += text(w / 2, h - 19, 'Descent direction: sₚ − sᵩ = −2', 18 if mobile else 24, BLUE, 'middle')
    svg('dmd' + ('-mobile' if mobile else '') + '.svg', w, h, b, 'Distribution matching direction', 'At a fixed noise level, the target is N(-1,1) and the student is N(1,1). Their score difference sp minus sq is minus two, moving generated samples left toward the target. This is a toy distribution, not an empirical result.')
for mobile in [False, True]:
    w, h = (400, 520) if mobile else (960, 300)
    b = ''
    fs = 20 if mobile else 24
    for k in [0, 1]:
        ox, oy = (35, 94 + 245 * k) if mobile else (42 + 480 * k, 110)
        b += text(ox, oy - 55, 'Deterministic map' if k == 0 else 'Meta Flow Map', fs, FG, weight=700)
        b += circle(ox + 35, oy + 32, 9, BLUE)
        b += text(ox + 35, oy + 75, 'xₜ', fs, BLUE, 'middle')
        if k == 0:
            b += line(ox + 50, oy + 32, ox + 297, oy + 32, TEAL, 3, arrow=True) + circle(ox + 307, oy + 32, 9, TEAL)
            b += text(ox + 168, oy + 116, 'One endpoint', fs, MUT, 'middle')
        else:
            for j, y in enumerate([oy - 8, oy + 48, oy + 104]):
                b += path(f'M{ox + 50},{oy + 32} Q{ox + 164},{oy + 32} {ox + 289},{y}', TEAL, 2, arrow=True) + circle(ox + 303, y, 8, TEAL)
            b += text(ox + 168, oy - 3, 'fresh noise η', 18 if mobile else 24, ORANGE, 'middle')
            b += text(ox + 168, oy + 153, 'Posterior samples', fs, MUT, 'middle')
    svg('posterior' + ('-mobile' if mobile else '') + '.svg', w, h, b, 'One endpoint versus posterior samples', 'A deterministic flow map returns one endpoint from a fixed state. A Meta Flow Map uses fresh auxiliary noise to sample multiple clean completions conditioned on that same state.')
print('Generated 12 Flow Map SVGs.')
