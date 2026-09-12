"""Rebuild both posts' figures with matplotlib and numpy.

    MPLCONFIGDIR=/tmp/post-plot-cache python _scripts/generate_post_explainers.py

Values are analytic examples or explicit schedules, not model/hardware
measurements. SVGs are site assets; PNG review copies go to /tmp.
If canvas sizes change, update the dimensions in the posts' figure includes.
"""

from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "assets/img/blogs"
BG, INK, MUTED = "#131d2e", "#e8edf6", "#aebbd0"
BLUE, TEAL, ORANGE = "#89b4ff", "#65dfc0", "#ffbd7a"
PURPLE, GRID = "#d4a9ff", "#344159"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 13,
    "font.family": "DejaVu Sans", "svg.hashsalt": "post-explainers-v2",
})


def save(fig, folder, name, mobile=False):
    if mobile:
        name += "-mobile"
    destination = ROOT / folder
    destination.mkdir(parents=True, exist_ok=True)
    svg_path = destination / f"{name}.svg"
    fig.savefig(svg_path, facecolor=BG, metadata={"Date": None})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig(Path("/tmp") / f"{name}.png", dpi=150, facecolor=BG)
    plt.close(fig)


def clean(ax, grid=True):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(alpha=0.16, color=MUTED)
        ax.set_axisbelow(True)


def arrow(ax, start, end, color, **kwargs):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "->", "color": color, "lw": 2.3, **kwargs})


def noising(mobile=False):
    # Fixed four-cluster samples paired with independent Gaussian noise.
    # This plots marginals of a cosine schedule, not reverse sampling paths.
    rng = np.random.default_rng(29)
    n = 800
    groups = np.arange(n) % 4
    centers = np.array([[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]])
    data = centers[groups] + rng.normal(0, 0.18, (n, 2))
    noise = rng.normal(size=(n, 2))
    colors = np.array([BLUE, TEAL, ORANGE, PURPLE])[groups]
    fig, axes = plt.subplots(2 if mobile else 1, 2 if mobile else 4,
                             figsize=(5.2, 5.4) if mobile else (10, 3.05))
    fig.subplots_adjust(left=.025, right=.985, bottom=.025 if mobile else .05,
                        top=.91 if mobile else .83, wspace=.06, hspace=.25)
    axes = axes.flat
    for ax, t in zip(axes, [0, .35, .7, 1]):
        state = np.cos(t * np.pi / 2) * data + np.sin(t * np.pi / 2) * noise
        ax.scatter(*state.T, c=colors, s=5, alpha=.6, linewidths=0)
        ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), aspect="equal")
        ax.set_title(f"$t = {t:g}$", fontsize=15, pad=12)
        ax.set_axis_off()
    save(fig, "flowmap", "diffusion", mobile)


def path(t):
    """A smooth state-space trajectory; all endpoint diagrams share it."""
    t = np.asarray(t)
    return np.array([4 * t, 1.7 * np.sin(np.pi * t)])


def tangent(t):
    return np.array([4, 1.7 * np.pi * np.cos(np.pi * t)])


def transport(mobile=False):
    t, s = .8, .2
    start, end = path(t), path(s)
    euler = start + (s - t) * tangent(t)
    fig, ax = plt.subplots(figsize=(5.2, 4.6) if mobile else (7.2, 4.6))
    fig.subplots_adjust(left=.06, right=.94, bottom=.08, top=.91)
    ax.plot(*path(np.linspace(s, t, 200)), color=BLUE, lw=3)
    ax.plot(*np.array([start, euler]).T, color=ORANGE, ls="--", lw=2)
    arrow(ax, start, start - .18 * tangent(t), ORANGE)
    arrow(ax, start, end, TEAL)
    ax.scatter(*np.array([start, end]).T, s=65, color=INK, zorder=4)
    ax.scatter(*euler, s=90, marker="x", color=ORANGE, zorder=4)
    ax.text(*(start + [.12, -.06]), "$x_t$", fontsize=17)
    ax.text(*(end + [-.2, -.3]), "$x_s$", fontsize=17)
    ax.text(*(euler + [-.3, .16]), "Euler", color=ORANGE, fontsize=14)
    ax.text(1.65, .72, "Flow map", color=TEAL, fontsize=14)
    ax.text(1.03, 1.96, "ODE path", color=BLUE, fontsize=14)
    ax.set(xlim=(.2, 3.8), ylim=(.3, 4.05), aspect="equal")
    ax.set_axis_off()
    save(fig, "flowmap", "transport", mobile)


def integration(mobile=False):
    # Explicit Euler with a negative step on dx/dt=x, x(1)=1.
    fig, axes = plt.subplots(2 if mobile else 1, 1 if mobile else 2,
                             figsize=(5.2, 6.4) if mobile else (10, 4.1))
    fig.subplots_adjust(left=.18 if mobile else .085, right=.96,
                        bottom=.09 if mobile else .2, top=.96 if mobile else .89,
                        wspace=.36, hspace=.53)
    t = np.linspace(0, 1, 200)
    axes[0].plot(t, np.exp(t - 1), color=TEAL, lw=3, label="Exact")
    for n, color in [(1, ORANGE), (2, BLUE), (4, PURPLE)]:
        ts = np.linspace(1, 0, n + 1)
        xs = (1 - 1 / n) ** np.arange(n + 1)
        axes[0].plot(ts, xs, "o--", color=color, ms=4, label=f"Euler: {n}")
    axes[0].set(xlabel="Time $t$", ylabel="State $x(t)$", xlim=(1.02, -.02), ylim=(-.03, 1.05))
    axes[0].legend(frameon=False, fontsize=11, loc="lower left")
    ns = np.array([1, 2, 4, 8, 16])
    errors = math.exp(-1) - (1 - 1 / ns) ** ns
    axes[1].plot(ns, errors, "o-", color=ORANGE, lw=2)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(ns, labels=[str(n) for n in ns])
    axes[1].set(xlabel="Euler steps", ylabel="Endpoint error", ylim=(0, .4))
    for ax in axes:
        clean(ax)
    save(fig, "flowmap", "integration-example", mobile)


def meanflow_area(mobile=False):
    # v(tau)=1+4*tau^2 along a scalar orbit. The equal-area rectangle
    # represents the time average, which differs from the endpoint average.
    r, t = .15, .85
    velocity = lambda tau: 1 + 4 * tau**2
    u = 1 + 4 * (t * t + t * r + r * r) / 3
    tau = np.linspace(r, t, 250)
    fig, ax = plt.subplots(figsize=(5.2, 3.9) if mobile else (8.4, 3.9))
    fig.subplots_adjust(left=.095, right=.95, bottom=.19, top=.9)
    ax.fill_between(tau, velocity(tau), color=BLUE, alpha=.22)
    ax.plot(tau, velocity(tau), color=BLUE, lw=3)
    ax.add_patch(Rectangle((r, 0), t - r, u, fc="none", ec=TEAL, lw=2, ls="--"))
    ax.text(.41, u + .16, "$u$", color=TEAL, fontsize=20)
    ax.text(.48, 3.35, r"$v(x_\tau,\tau)$", color=BLUE, fontsize=16)
    extra = np.linspace(t, t + .045, 25)
    ax.fill_between(extra, velocity(extra), facecolor=ORANGE, alpha=.5)
    ax.plot(extra, velocity(extra), color=ORANGE, lw=3)
    ax.text(t + .0225, 4.45, "$dt$", color=ORANGE, fontsize=15, ha="center")
    ax.set_xticks([r, t], labels=["$r$", "$t$"])
    ax.set_yticks([])
    ax.set(xlim=(.1, .94), ylim=(0, 4.7), xlabel="Time", ylabel="Velocity")
    clean(ax, grid=False)
    save(fig, "flowmap", "meanflow-jvp", mobile)


def score_difference(mobile=False):
    # At one noise level, p=N(-1,1), q=N(1,1): s_q-s_p=2 everywhere.
    # Sample-space descent shifts left; this is not a general neural network
    # parameter update or an empirical DMD training trajectory.
    x = np.linspace(-4, 4, 400)
    density = lambda mean: np.exp(-.5 * (x - mean)**2) / np.sqrt(2 * np.pi)
    fig, ax = plt.subplots(figsize=(5.5, 4.2) if mobile else (8.4, 4))
    fig.subplots_adjust(left=.15 if mobile else .1, right=.97, bottom=.2, top=.83)
    for mean, color, label in [(-1, TEAL, "Target $p_t$"), (1, BLUE, "Student $q_t$")]:
        ax.plot(x, density(mean), color=color, lw=2.8, label=label.split()[0] if mobile else label)
        ax.fill_between(x, density(mean), color=color, alpha=.09)
    ax.plot(x, density(0), color=ORANGE, lw=2, ls="--", label="Shift" if mobile else "After shift")
    for sample in [-.2, 1.25, 2.7]:
        arrow(ax, (sample, -.048), (sample - 1, -.048), ORANGE, lw=1.8)
        ax.scatter(sample, -.048, color=BLUE, s=24, zorder=4)
    ax.text(-3.6, -.057, r"$-\,(s_q-s_p)$", color=ORANGE, fontsize=14)
    ax.set(xlim=(-4, 4), ylim=(-.08, .43), xlabel="Noisy state $x_t$", ylabel="Density")
    ax.set_yticks([0, .2, .4])
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.03), ncol=3, fontsize=12,
              columnspacing=.8, handlelength=1.5, handletextpad=.5)
    clean(ax, grid=False)
    save(fig, "flowmap", "dmd", mobile)


def endpoint_derivatives(mobile=False):
    fig, axes = plt.subplots(2 if mobile else 1, 1 if mobile else 2,
                             figsize=(5.2, 5.7) if mobile else (10, 3.6))
    fig.subplots_adjust(left=.04, right=.97, bottom=.025 if mobile else .06,
                        top=.88 if mobile else .83, wspace=.2, hspace=.32)
    s, t, dt = .2, .76, .13
    for ax, title in zip(axes, ["MeanFlow: move the start", "TVM: move the destination"]):
        ax.plot(*path(np.linspace(.1, .94, 250)), color=GRID, lw=3)
        arrow(ax, path(t), path(s), BLUE)
        ax.scatter(*np.array([path(t), path(s)]).T, color=BLUE, s=50, zorder=4)
        ax.set_title(title, fontsize=14, pad=12)
        ax.set(xlim=(.1, 3.95), ylim=(.1, 2.05), aspect="equal")
        ax.set_axis_off()
    # MF holds r fixed while (x_t,t) moves along the same orbit.
    ax = axes[0]
    moved = path(t + dt)
    arrow(ax, moved, path(s), TEAL)
    arrow(ax, path(t), moved, ORANGE, connectionstyle="arc3,rad=.12")
    ax.scatter(*moved, color=TEAL, s=50, zorder=4)
    ax.text(*(path(s) + [-.32, -.31]), "$x_r$", fontsize=17)
    ax.text(*(path(t) + [-.12, .24]), "$x_t$", fontsize=17)
    ax.text(*(moved + [-.12, -.33]), "$x_{t+dt}$", fontsize=17)
    ax.add_patch(Circle(path(s), .12, ec=INK, fc="none", lw=1.3))
    # TVM holds (x_t,t) fixed while destination time s moves.
    ax = axes[1]
    moved = path(s + dt)
    arrow(ax, path(t), moved, TEAL)
    arrow(ax, path(s), moved, ORANGE, connectionstyle="arc3,rad=-.13")
    ax.scatter(*moved, color=TEAL, s=50, zorder=4)
    ax.text(*(path(t) + [.08, -.27]), "$x_t$", fontsize=17)
    ax.text(*(path(s) + [-.32, -.31]), "$x_s$", fontsize=17)
    ax.text(*(moved + [-.15, .24]), "$x_{s+ds}$", fontsize=17)
    ax.add_patch(Circle(path(t), .12, ec=INK, fc="none", lw=1.3))
    save(fig, "flowmap", "endpoint-derivatives", mobile)


def amdahl(mobile=False):
    fig, ax = plt.subplots(figsize=(5.5, 3.8) if mobile else (8.4, 3.5))
    fig.subplots_adjust(left=.22 if mobile else .17, right=.97, bottom=.22, top=.82)
    for y, target in enumerate([5, 5 / 3, 0]):
        ax.barh(y, 95, height=.5, color=BLUE, label="Other work" if y == 0 else None)
        ax.barh(y, target, left=95, height=.5, color=ORANGE, label="Target kernel" if y == 0 else None)
        total = "96.67" if y == 1 else f"{95 + target:g}"
        ax.text(95 + target + 1, y, total, va="center", fontsize=13)
    ax.axvline(95, color=INK, lw=1, ls=":", alpha=.65)
    ax.set_yticks(range(3), labels=["Original", "3× kernel", "∞× kernel"])
    ax.invert_yaxis()
    ax.set(xlim=(0, 111), xlabel="Total time (ms)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2, fontsize=12)
    ax.xaxis.grid(alpha=.12)
    ax.set_axisbelow(True)
    save(fig, "gpu-systems", "amdahl", mobile)


def roofline(mobile=False):
    intensity = np.geomspace(.05, 1000, 500)
    fig, ax = plt.subplots(figsize=(5.5, 4.4) if mobile else (8.4, 4.4))
    fig.subplots_adjust(left=.17 if mobile else .12, right=.97, bottom=.19, top=.89)
    ax.loglog(intensity, np.minimum(120, 2 * intensity), color=BLUE, lw=3)
    ax.axvline(60, color=MUTED, lw=1, ls=":")
    ax.text(75, 150, "120 TFLOP/s", color=BLUE, fontsize=13)
    ax.text(.26, 2.2, "2 TB/s", color=BLUE, fontsize=13, rotation=30)
    ax.text(65, .1, "60", color=MUTED, fontsize=12)
    # Illustrative kernel states below the bound, not measurements.
    ax.scatter([2, 2, 12], [1, 4, 24], color=[ORANGE, ORANGE, TEAL], s=45, zorder=5)
    arrow(ax, (2, 1), (2, 4), ORANGE)
    ax.text(2.5, .65, "Fewer stalls", color=ORANGE, fontsize=12)
    arrow(ax, (2, 4), (12, 24), TEAL)
    ax.text(5, 3.1, "Reuse", color=TEAL, fontsize=13)
    ax.scatter(1 / 12, 1 / 6, marker="s", s=40, color=INK, zorder=5)
    ax.text(.1, .12, "Vector add", fontsize=11)
    ax.set(xlim=(.05, 1000), ylim=(.07, 230),
           xlabel="Arithmetic intensity (FLOP/byte)", ylabel="Throughput (TFLOP/s)")
    clean(ax)
    save(fig, "gpu-systems", "roofline", mobile)


def tensor(ax, x, y, label, color):
    for i in range(4):
        ax.add_patch(Rectangle((x - .31 + .155 * i, y - .13), .145, .26, fc=color, ec=BG, lw=.6))
    ax.text(x, y + .21, label, color=color, ha="center", fontsize=15)


def operator(ax, x, label):
    # Keep the symbol's outline circular when the mobile canvas narrows.
    bounds = ax.get_window_extent()
    pixels_x = bounds.width / np.diff(ax.get_xlim())[0]
    pixels_y = bounds.height / np.diff(ax.get_ylim())[0]
    ax.add_patch(Ellipse((x, 1.27), .68, .68 * pixels_x / pixels_y,
                         fc=BG, ec=INK, lw=1.5))
    ax.text(x, 1.27, label, ha="center", va="center", fontsize=15)


def fusion(mobile=False):
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.2) if mobile else (9, 5.1))
    fig.subplots_adjust(left=.18 if mobile else .14, right=.97, bottom=.06, top=.92, hspace=.44)
    for ax, title in zip(axes, ["3 kernels", "1 fused kernel"]):
        ax.set(xlim=(-.4, 8.2), ylim=(-.18, 1.75))
        ax.set_axis_off()
        ax.add_patch(Rectangle((-.35, -.1), 8.5, .62, fc=GRID, alpha=.65))
        ax.text(-.55, .13, "HBM", ha="right", va="center", fontsize=12)
        ax.text(-.55, 1.26, "On chip", ha="right", va="center", fontsize=12)
        ax.set_title(title, loc="left", fontsize=14, pad=6)
    ax = axes[0]
    for x, label, color in [(0, "$x$", BLUE), (2.6, "$z_1$", ORANGE),
                             (5.2, "$z_2$", ORANGE), (7.8, "$y$", TEAL)]:
        tensor(ax, x, .12, label, color)
    for i, operation in enumerate(["×a", "+b", r"$\phi$"]):
        x = 1.3 + i * 2.6
        operator(ax, x, operation)
        arrow(ax, (x - 1.3, .57), (x - .23, 1.07), BLUE if i == 0 else ORANGE)
        arrow(ax, (x + .23, 1.07), (x + 1.3, .57), TEAL if i == 2 else ORANGE)
    ax = axes[1]
    tensor(ax, 0, .12, "$x$", BLUE)
    tensor(ax, 7.8, .12, "$y$", TEAL)
    ax.add_patch(Rectangle((.65, .87), 6.5, .8, fc=TEAL, ec=TEAL, alpha=.07))
    for i, operation in enumerate(["×a", "+b", r"$\phi$"]):
        x = 1.3 + i * 2.6
        operator(ax, x, operation)
        if i < 2:
            arrow(ax, (x + .35, 1.27), (x + 2.25, 1.27), TEAL)
    arrow(ax, (0, .57), (1.07, 1.07), BLUE)
    arrow(ax, (6.73, 1.07), (7.8, .57), TEAL)
    save(fig, "gpu-systems", "fusion", mobile)


def overlap(mobile=False):
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.4) if mobile else (8.4, 4.4), sharex=True)
    fig.subplots_adjust(left=.22 if mobile else .16, right=.97, bottom=.16, top=.89, hspace=.6)
    for ax, start, title in zip(axes, [8, 4], ["Serialized", "Overlapped"]):
        ax.barh(1, 8, color=BLUE, height=.58)
        ax.barh(0, 6, left=start, color=ORANGE, height=.58)
        ax.text(4, 1, "8 ms", ha="center", va="center", color=BG)
        ax.text(start + 3, 0, "6 ms", ha="center", va="center", color=BG)
        ax.set_yticks([1, 0], labels=["Compute", "Collective"])
        ax.set_title(title, loc="left", fontsize=14)
        ax.set(xlim=(0, 14.5), ylim=(-.65, 1.6))
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.xaxis.grid(alpha=.13)
        ax.set_axisbelow(True)
    axes[1].axvline(4, color=MUTED, ls=":", lw=1)
    axes[1].axvspan(8, 10, color=ORANGE, alpha=.13)
    axes[1].text(9, 1, "2 ms", color=ORANGE, ha="center", va="center")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_xticks(range(0, 15, 2))
    save(fig, "gpu-systems", "overlap", mobile)


def batching(mobile=False):
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.3) if mobile else (8.4, 4.3), sharex=True)
    fig.subplots_adjust(left=.17 if mobile else .13, right=.96, bottom=.18, top=.89, hspace=.6)
    schedules = [
        [(1, 0, 4, "A", BLUE), (0, 0, 2, "B", ORANGE), (1, 4, 2, "C", TEAL)],
        [(1, 0, 4, "A", BLUE), (0, 0, 2, "B", ORANGE), (0, 2, 2, "C", TEAL)],
    ]
    for ax, jobs, end, title in zip(axes, schedules, [6, 4], ["Static", "Continuous"]):
        for slot in [0, 1]:
            ax.add_patch(Rectangle((0, slot - .29), end, .58, fc=BG, ec=GRID, hatch="///", lw=0))
        for slot, start, duration, label, color in jobs:
            ax.barh(slot, duration, left=start, height=.58, color=color, edgecolor=BG, linewidth=1.4)
            ax.text(start + duration / 2, slot, label, ha="center", va="center", fontsize=15, color=BG)
        ax.axvline(end, color=TEAL, ls=":", lw=1.5)
        ax.set_yticks([1, 0], labels=["Slot 1", "Slot 2"])
        ax.set_title(title, loc="left", fontsize=14)
        ax.set(xlim=(0, 6.15), ylim=(-.65, 1.6))
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[1].set_xticks(range(7))
    axes[1].set_xlabel("Iteration boundary")
    save(fig, "gpu-systems", "batching", mobile)


def denoising_budget(mobile=False):
    fig, ax = plt.subplots(figsize=(5.5, 3.7) if mobile else (8.4, 3.7))
    fig.subplots_adjust(left=.17 if mobile else .13, right=.97, bottom=.21, top=.81)
    for y, k in enumerate([20, 4, 1]):
        dynamic, fixed = 40 * k, 200
        ax.barh(y, dynamic, color=BLUE, height=.52, label="Denoiser" if y == 0 else None)
        ax.barh(y, fixed, left=dynamic, color=ORANGE, height=.52, label="Fixed work" if y == 0 else None)
        ax.text(dynamic + fixed + 15, y, f"{dynamic + fixed:,}", va="center", fontsize=13)
    ax.set_yticks(range(3), labels=["K = 20", "K = 4", "K = 1"])
    ax.invert_yaxis()
    ax.set(xlabel="End-to-end latency (ms)", xlim=(0, 1140))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(alpha=.12)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.04), ncol=2, fontsize=12)
    save(fig, "gpu-systems", "denoising-budget", mobile)


if __name__ == "__main__":
    for draw in [
        noising, transport, integration, meanflow_area, score_difference,
        endpoint_derivatives, amdahl, roofline, fusion, overlap, batching,
        denoising_budget,
    ]:
        draw()
        draw(mobile=True)
