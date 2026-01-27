"""Visualization utilities for Schrödinger Bridges.

Provides plotting and animation capabilities for visualizing
transport paths, marginals, and diagnostics.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np

from .core.types import Array, TrajectoryBatch

# Lazy imports for matplotlib
_plt = None
_animation = None


def _get_plt():
    """Lazy import matplotlib."""
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def _get_animation():
    """Lazy import matplotlib.animation."""
    global _animation
    if _animation is None:
        from matplotlib import animation
        _animation = animation
    return _animation


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    figsize: Tuple[int, int] = (10, 8)
    dpi: int = 100
    cmap: str = 'viridis'
    alpha: float = 0.5
    point_size: int = 10
    line_width: float = 0.5
    fps: int = 10
    interval: int = 100


DEFAULT_VIS_CONFIG = VisualizationConfig()


def plot_marginals(
    source_samples: Array,
    target_samples: Array,
    generated_samples: Optional[Array] = None,
    config: Optional[VisualizationConfig] = None,
    title: str = "Marginal Distributions",
    save_path: Optional[str] = None,
) -> Any:
    """Plot source, target, and optionally generated marginals.
    
    Args:
        source_samples: Samples from source distribution.
        target_samples: Samples from target distribution.
        generated_samples: Optional samples from learned bridge.
        config: Visualization configuration.
        title: Plot title.
        save_path: Path to save figure.
        
    Returns:
        Matplotlib figure.
    """
    plt = _get_plt()
    config = config or DEFAULT_VIS_CONFIG
    
    fig, axes = plt.subplots(1, 3 if generated_samples is not None else 2,
                             figsize=config.figsize)
    
    # Source
    axes[0].scatter(
        np.array(source_samples[:, 0]),
        np.array(source_samples[:, 1]),
        c='blue', alpha=config.alpha, s=config.point_size, label='Source'
    )
    axes[0].set_title('Source (μ₀)')
    axes[0].set_aspect('equal')
    
    # Target
    axes[1].scatter(
        np.array(target_samples[:, 0]),
        np.array(target_samples[:, 1]),
        c='red', alpha=config.alpha, s=config.point_size, label='Target'
    )
    axes[1].set_title('Target (μ₁)')
    axes[1].set_aspect('equal')
    
    # Generated
    if generated_samples is not None:
        axes[2].scatter(
            np.array(target_samples[:, 0]),
            np.array(target_samples[:, 1]),
            c='red', alpha=0.2, s=config.point_size, label='Target'
        )
        axes[2].scatter(
            np.array(generated_samples[:, 0]),
            np.array(generated_samples[:, 1]),
            c='green', alpha=config.alpha, s=config.point_size, label='Generated'
        )
        axes[2].set_title('Generated vs Target')
        axes[2].set_aspect('equal')
        axes[2].legend()
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
    
    return fig


def plot_trajectories(
    trajectories: Union[TrajectoryBatch, Array],
    times: Optional[Array] = None,
    num_show: int = 50,
    config: Optional[VisualizationConfig] = None,
    title: str = "Transport Trajectories",
    save_path: Optional[str] = None,
    colorby: str = 'time',  # 'time' or 'trajectory'
) -> Any:
    """Plot sample trajectories.
    
    Args:
        trajectories: Trajectory batch or array [batch, time, dim].
        times: Time points (extracted from batch if TrajectoryBatch).
        num_show: Number of trajectories to show.
        config: Visualization configuration.
        title: Plot title.
        save_path: Path to save figure.
        colorby: Color trajectories by 'time' or 'trajectory'.
        
    Returns:
        Matplotlib figure.
    """
    plt = _get_plt()
    config = config or DEFAULT_VIS_CONFIG
    
    if isinstance(trajectories, TrajectoryBatch):
        paths = np.array(trajectories.paths)
        times = np.array(trajectories.times)
    else:
        paths = np.array(trajectories)
        times = np.array(times) if times is not None else np.linspace(0, 1, paths.shape[1])
    
    fig, ax = plt.subplots(figsize=config.figsize)
    
    n_show = min(num_show, paths.shape[0])
    
    for i in range(n_show):
        traj = paths[i]
        
        if colorby == 'time':
            # Color by time
            points = traj.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            from matplotlib.collections import LineCollection
            lc = LineCollection(segments, cmap=config.cmap, alpha=config.alpha)
            lc.set_array(times[:-1])
            lc.set_linewidth(config.line_width)
            ax.add_collection(lc)
        else:
            # Single color per trajectory
            ax.plot(traj[:, 0], traj[:, 1], '-', alpha=config.alpha,
                   linewidth=config.line_width)
    
    # Mark endpoints
    ax.scatter(paths[:n_show, 0, 0], paths[:n_show, 0, 1],
              c='green', s=config.point_size * 2, marker='o', label='Start', zorder=5)
    ax.scatter(paths[:n_show, -1, 0], paths[:n_show, -1, 1],
              c='red', s=config.point_size * 2, marker='x', label='End', zorder=5)
    
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_aspect('equal')
    ax.autoscale()
    
    if colorby == 'time':
        plt.colorbar(lc, ax=ax, label='Time')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
    
    return fig


def plot_diagnostics(
    loss_history: Array,
    diagnostics: Optional[Dict] = None,
    config: Optional[VisualizationConfig] = None,
    title: str = "Training Diagnostics",
    save_path: Optional[str] = None,
) -> Any:
    """Plot training diagnostics.
    
    Args:
        loss_history: Loss values over training.
        diagnostics: Additional diagnostic data.
        config: Visualization configuration.
        title: Plot title.
        save_path: Path to save figure.
        
    Returns:
        Matplotlib figure.
    """
    plt = _get_plt()
    config = config or DEFAULT_VIS_CONFIG
    
    n_plots = 1 + (2 if diagnostics else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(config.figsize[0], config.figsize[1] // 2))
    
    if n_plots == 1:
        axes = [axes]
    
    # Loss curve
    axes[0].plot(np.array(loss_history))
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_yscale('log')
    
    if diagnostics:
        # Mass conservation
        if hasattr(diagnostics, 'mass_conservation') and diagnostics.mass_conservation is not None:
            axes[1].plot(np.array(diagnostics.mass_conservation))
            axes[1].axhline(y=1.0, color='r', linestyle='--', label='Expected')
            axes[1].set_xlabel('Time index')
            axes[1].set_ylabel('Mass')
            axes[1].set_title('Mass Conservation')
            axes[1].legend()
        
        # Entropy evolution
        if hasattr(diagnostics, 'entropy_evolution') and diagnostics.entropy_evolution is not None:
            axes[2].plot(np.array(diagnostics.entropy_evolution))
            axes[2].set_xlabel('Time index')
            axes[2].set_ylabel('Entropy')
            axes[2].set_title('Entropy Evolution')
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
    
    return fig


def create_transport_gif(
    trajectories: Union[TrajectoryBatch, Array],
    times: Optional[Array] = None,
    source_samples: Optional[Array] = None,
    target_samples: Optional[Array] = None,
    save_path: str = "transport.gif",
    config: Optional[VisualizationConfig] = None,
    title: str = "Schrödinger Bridge Transport",
) -> str:
    """Create animated GIF of transport evolution.
    
    Args:
        trajectories: Trajectory data.
        times: Time points.
        source_samples: Reference source samples.
        target_samples: Reference target samples.
        save_path: Path to save GIF.
        config: Visualization configuration.
        title: Animation title.
        
    Returns:
        Path to saved GIF.
    """
    plt = _get_plt()
    animation = _get_animation()
    config = config or DEFAULT_VIS_CONFIG
    
    if isinstance(trajectories, TrajectoryBatch):
        paths = np.array(trajectories.paths)
        times = np.array(trajectories.times)
    else:
        paths = np.array(trajectories)
        times = np.array(times) if times is not None else np.linspace(0, 1, paths.shape[1])
    
    num_frames = paths.shape[1]
    
    # Compute axis limits
    all_x = paths[:, :, 0].flatten()
    all_y = paths[:, :, 1].flatten()
    
    x_min, x_max = np.min(all_x) - 0.5, np.max(all_x) + 0.5
    y_min, y_max = np.min(all_y) - 0.5, np.max(all_y) + 0.5
    
    fig, ax = plt.subplots(figsize=config.figsize)
    
    # Background: source and target
    if source_samples is not None:
        ax.scatter(np.array(source_samples[:, 0]), np.array(source_samples[:, 1]),
                  c='blue', alpha=0.1, s=config.point_size, label='Source')
    if target_samples is not None:
        ax.scatter(np.array(target_samples[:, 0]), np.array(target_samples[:, 1]),
                  c='red', alpha=0.1, s=config.point_size, label='Target')
    
    scatter = ax.scatter([], [], c='green', s=config.point_size * 2, alpha=config.alpha)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=12, verticalalignment='top')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return scatter, time_text
    
    def animate(frame):
        # Current positions
        positions = paths[:, frame, :2]
        scatter.set_offsets(positions)
        
        # Update time display
        t = times[frame]
        time_text.set_text(f't = {t:.3f}')
        
        # Color by progress
        colors = plt.cm.viridis(np.full(len(positions), frame / num_frames))
        scatter.set_facecolors(colors)
        
        return scatter, time_text
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=num_frames, interval=config.interval,
        blit=True
    )
    
    # Save as GIF
    anim.save(save_path, writer='pillow', fps=config.fps, dpi=config.dpi)
    plt.close(fig)
    
    return save_path


def create_comparison_gif(
    trajectories_dict: Dict[str, Union[TrajectoryBatch, Array]],
    source_samples: Optional[Array] = None,
    target_samples: Optional[Array] = None,
    save_path: str = "comparison.gif",
    config: Optional[VisualizationConfig] = None,
) -> str:
    """Create side-by-side comparison GIF of multiple methods.
    
    Args:
        trajectories_dict: Dictionary mapping method names to trajectories.
        source_samples: Reference source samples.
        target_samples: Reference target samples.
        save_path: Path to save GIF.
        config: Visualization configuration.
        
    Returns:
        Path to saved GIF.
    """
    plt = _get_plt()
    animation = _get_animation()
    config = config or DEFAULT_VIS_CONFIG
    
    n_methods = len(trajectories_dict)
    
    # Convert to arrays
    methods = list(trajectories_dict.keys())
    all_paths = []
    times = None
    
    for name, traj in trajectories_dict.items():
        if isinstance(traj, TrajectoryBatch):
            all_paths.append(np.array(traj.paths))
            if times is None:
                times = np.array(traj.times)
        else:
            all_paths.append(np.array(traj))
    
    if times is None:
        times = np.linspace(0, 1, all_paths[0].shape[1])
    
    num_frames = all_paths[0].shape[1]
    
    # Compute global axis limits
    all_data = np.concatenate([p.reshape(-1, 2) for p in all_paths], axis=0)
    x_min, x_max = np.min(all_data[:, 0]) - 0.5, np.max(all_data[:, 0]) + 0.5
    y_min, y_max = np.min(all_data[:, 1]) - 0.5, np.max(all_data[:, 1]) + 0.5
    
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    
    scatters = []
    for i, (name, ax) in enumerate(zip(methods, axes)):
        if source_samples is not None:
            ax.scatter(np.array(source_samples[:, 0]), np.array(source_samples[:, 1]),
                      c='blue', alpha=0.1, s=5)
        if target_samples is not None:
            ax.scatter(np.array(target_samples[:, 0]), np.array(target_samples[:, 1]),
                      c='red', alpha=0.1, s=5)
        
        sc = ax.scatter([], [], c='green', s=config.point_size, alpha=config.alpha)
        scatters.append(sc)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(name)
    
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12)
    
    def animate(frame):
        for i, sc in enumerate(scatters):
            positions = all_paths[i][:, frame, :2]
            sc.set_offsets(positions)
        
        time_text.set_text(f't = {times[frame]:.3f}')
        return scatters + [time_text]
    
    anim = animation.FuncAnimation(
        fig, animate, frames=num_frames,
        interval=config.interval, blit=True
    )
    
    anim.save(save_path, writer='pillow', fps=config.fps, dpi=config.dpi)
    plt.close(fig)
    
    return save_path


def plot_velocity_field(
    velocity_fn: Callable[[Array, float], Array],
    t: float,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    resolution: int = 20,
    config: Optional[VisualizationConfig] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    samples: Optional[Array] = None,
) -> Any:
    """Plot velocity field at a given time.
    
    Args:
        velocity_fn: Velocity function v(x, t).
        t: Time to evaluate.
        xlim: X-axis limits.
        ylim: Y-axis limits.
        resolution: Grid resolution.
        config: Visualization configuration.
        title: Plot title.
        save_path: Path to save figure.
        samples: Optional samples to overlay.
        
    Returns:
        Matplotlib figure.
    """
    plt = _get_plt()
    config = config or DEFAULT_VIS_CONFIG
    
    # Create grid
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    points = np.stack([X.flatten(), Y.flatten()], axis=-1)
    
    # Evaluate velocity
    velocities = np.array(velocity_fn(jnp.array(points), t))
    U = velocities[:, 0].reshape(X.shape)
    V = velocities[:, 1].reshape(X.shape)
    
    fig, ax = plt.subplots(figsize=config.figsize)
    
    # Quiver plot
    magnitude = np.sqrt(U**2 + V**2)
    ax.quiver(X, Y, U, V, magnitude, cmap=config.cmap, alpha=0.8)
    
    # Overlay samples
    if samples is not None:
        ax.scatter(np.array(samples[:, 0]), np.array(samples[:, 1]),
                  c='red', s=config.point_size, alpha=config.alpha)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title or f'Velocity Field at t={t:.3f}')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
    
    return fig


__all__ = [
    'VisualizationConfig',
    'plot_marginals',
    'plot_trajectories',
    'plot_diagnostics',
    'create_transport_gif',
    'create_comparison_gif',
    'plot_velocity_field',
]
