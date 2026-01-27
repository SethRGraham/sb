"""Device utilities for Schrödinger Bridge solvers.

Provides utilities for:
- Device detection (CPU/GPU/TPU)
- Memory management
- Device placement of computations
- Multi-device support (data parallelism)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax

from .core.types import Array, PRNGKey


# =============================================================================
# Device Detection
# =============================================================================

class DeviceKind(Enum):
    """Types of accelerator devices."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


@dataclass
class DeviceInfo:
    """Information about available compute devices."""
    kind: DeviceKind
    count: int
    device_ids: List[int]
    platform: str
    memory_stats: Optional[Dict[str, int]] = None
    
    def __repr__(self) -> str:
        return f"DeviceInfo({self.kind.value}, count={self.count}, platform={self.platform})"


def get_device_info() -> DeviceInfo:
    """Get information about available compute devices.
    
    Returns:
        DeviceInfo with device type, count, and capabilities.
    """
    devices = jax.devices()
    
    if not devices:
        return DeviceInfo(
            kind=DeviceKind.CPU,
            count=1,
            device_ids=[0],
            platform="cpu",
        )
    
    # Get device type from first device
    device = devices[0]
    platform = device.platform.lower()
    
    if "gpu" in platform or "cuda" in platform:
        kind = DeviceKind.GPU
    elif "tpu" in platform:
        kind = DeviceKind.TPU
    else:
        kind = DeviceKind.CPU
    
    device_ids = list(range(len(devices)))
    
    # Try to get memory stats (GPU only)
    memory_stats = None
    if kind == DeviceKind.GPU:
        try:
            memory_stats = jax.devices()[0].memory_stats()
        except (AttributeError, NotImplementedError):
            pass
    
    return DeviceInfo(
        kind=kind,
        count=len(devices),
        device_ids=device_ids,
        platform=platform,
        memory_stats=memory_stats,
    )


def print_device_info():
    """Print formatted device information."""
    info = get_device_info()
    print(f"JAX Device Configuration")
    print(f"  Backend: {jax.default_backend()}")
    print(f"  Device type: {info.kind.value.upper()}")
    print(f"  Device count: {info.count}")
    print(f"  Platform: {info.platform}")
    
    if info.memory_stats:
        bytes_in_use = info.memory_stats.get('bytes_in_use', 0)
        peak_bytes = info.memory_stats.get('peak_bytes_in_use', 0)
        print(f"  Memory in use: {bytes_in_use / 1e9:.2f} GB")
        print(f"  Peak memory: {peak_bytes / 1e9:.2f} GB")


# =============================================================================
# Device Placement
# =============================================================================

def place_on_device(x: Array, device_id: int = 0) -> Array:
    """Place array on specified device.
    
    Args:
        x: Array to place.
        device_id: Target device index.
        
    Returns:
        Array on target device.
    """
    devices = jax.devices()
    if device_id >= len(devices):
        device_id = 0
    
    return jax.device_put(x, devices[device_id])


def get_default_device():
    """Get the default compute device."""
    devices = jax.devices()
    return devices[0] if devices else None


def ensure_on_device(x: Array) -> Array:
    """Ensure array is materialized on a device.
    
    This triggers computation if x is a lazy value.
    """
    return jax.device_get(jax.device_put(x))


# =============================================================================
# Memory Management
# =============================================================================

def clear_cache():
    """Clear JAX compilation cache to free memory."""
    # Clear compilation cache
    jax.clear_backends()


def estimate_memory_usage(shape: Tuple[int, ...], dtype=jnp.float32) -> int:
    """Estimate memory usage for an array in bytes."""
    dtype_size = jnp.dtype(dtype).itemsize
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    return num_elements * dtype_size


def check_memory_for_batch(
    batch_size: int,
    dim: int,
    num_steps: int,
    safety_factor: float = 0.8,
) -> Tuple[bool, int]:
    """Check if there's enough memory for a batch of trajectories.
    
    Args:
        batch_size: Number of samples.
        dim: Dimension of state space.
        num_steps: Number of time steps.
        safety_factor: Fraction of available memory to use.
        
    Returns:
        (can_fit, recommended_batch_size)
    """
    info = get_device_info()
    
    if info.memory_stats is None:
        # CPU or unknown - assume it fits
        return True, batch_size
    
    # Estimate memory for trajectories + gradients + optimizer state
    traj_bytes = batch_size * num_steps * dim * 4  # float32
    total_bytes = traj_bytes * 4  # Rough multiplier for gradients, etc.
    
    available = info.memory_stats.get('bytes_limit', float('inf'))
    in_use = info.memory_stats.get('bytes_in_use', 0)
    free = (available - in_use) * safety_factor
    
    if total_bytes <= free:
        return True, batch_size
    else:
        # Compute recommended batch size
        recommended = int(batch_size * free / total_bytes)
        recommended = max(1, recommended)
        return False, recommended


# =============================================================================
# Data Parallelism
# =============================================================================

def shard_batch(x: Array, num_devices: Optional[int] = None) -> Array:
    """Shard a batch across devices for data parallelism.
    
    Args:
        x: Batch array with shape [batch, ...].
        num_devices: Number of devices (defaults to all available).
        
    Returns:
        Sharded array with shape [num_devices, batch_per_device, ...].
    """
    if num_devices is None:
        num_devices = len(jax.devices())
    
    batch_size = x.shape[0]
    
    # Ensure batch_size is divisible by num_devices
    if batch_size % num_devices != 0:
        # Pad to make divisible
        pad_size = num_devices - (batch_size % num_devices)
        padding = [(0, pad_size)] + [(0, 0)] * (x.ndim - 1)
        x = jnp.pad(x, padding, mode='edge')
    
    # Reshape for sharding
    batch_per_device = x.shape[0] // num_devices
    new_shape = (num_devices, batch_per_device) + x.shape[1:]
    
    return x.reshape(new_shape)


def unshard_batch(x: Array, original_batch_size: Optional[int] = None) -> Array:
    """Unshard a batch from devices back to single array.
    
    Args:
        x: Sharded array with shape [num_devices, batch_per_device, ...].
        original_batch_size: If provided, truncate to this size.
        
    Returns:
        Unsharded array with shape [batch, ...].
    """
    num_devices, batch_per_device = x.shape[:2]
    new_shape = (num_devices * batch_per_device,) + x.shape[2:]
    
    result = x.reshape(new_shape)
    
    if original_batch_size is not None:
        result = result[:original_batch_size]
    
    return result


def pmap_with_devices(
    fn: Callable,
    in_axes: Union[int, Tuple] = 0,
    out_axes: Union[int, Tuple] = 0,
    devices: Optional[List] = None,
) -> Callable:
    """Wrapper for jax.pmap with explicit device placement.
    
    Args:
        fn: Function to parallelize.
        in_axes: Input axis specification for pmap.
        out_axes: Output axis specification.
        devices: Specific devices to use.
        
    Returns:
        Parallelized function.
    """
    if devices is None:
        devices = jax.devices()
    
    return jax.pmap(fn, in_axes=in_axes, out_axes=out_axes, devices=devices)


# =============================================================================
# JIT Compilation Utilities
# =============================================================================

def jit_with_device(
    fn: Callable,
    device: Optional[Any] = None,
    donate_argnums: Tuple[int, ...] = (),
) -> Callable:
    """JIT compile a function with specific device placement.
    
    Args:
        fn: Function to compile.
        device: Target device (defaults to first available).
        donate_argnums: Argument indices to donate (transfer ownership).
        
    Returns:
        JIT-compiled function.
    """
    if device is None:
        device = get_default_device()
    
    return jax.jit(fn, device=device, donate_argnums=donate_argnums)


def static_argnums_for_config(config_argnum: int = 1) -> Tuple[int, ...]:
    """Get static_argnums for functions with config objects.
    
    Configs should typically be static to enable proper tracing.
    """
    return (config_argnum,)


# =============================================================================
# Batch Processing Utilities
# =============================================================================

def process_in_batches(
    fn: Callable[[Array], Array],
    data: Array,
    batch_size: int,
    show_progress: bool = False,
) -> Array:
    """Process data in batches to manage memory.
    
    Args:
        fn: Function to apply to each batch.
        data: Full dataset with shape [N, ...].
        batch_size: Size of each batch.
        show_progress: Whether to print progress.
        
    Returns:
        Concatenated results.
    """
    n = data.shape[0]
    results = []
    
    num_batches = (n + batch_size - 1) // batch_size
    
    for i in range(0, n, batch_size):
        batch = data[i:min(i + batch_size, n)]
        result = fn(batch)
        results.append(result)
        
        if show_progress:
            batch_idx = i // batch_size + 1
            print(f"  Batch {batch_idx}/{num_batches}", end='\r')
    
    if show_progress:
        print()
    
    return jnp.concatenate(results, axis=0)


def split_key_for_devices(key: PRNGKey, num_devices: Optional[int] = None) -> Array:
    """Split a PRNG key across devices.
    
    Args:
        key: Base PRNG key.
        num_devices: Number of devices (defaults to all available).
        
    Returns:
        Array of keys with shape [num_devices, 2].
    """
    if num_devices is None:
        num_devices = len(jax.devices())
    
    return jax.random.split(key, num_devices)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Device info
    'DeviceKind',
    'DeviceInfo',
    'get_device_info',
    'print_device_info',
    # Placement
    'place_on_device',
    'get_default_device',
    'ensure_on_device',
    # Memory
    'clear_cache',
    'estimate_memory_usage',
    'check_memory_for_batch',
    # Parallelism
    'shard_batch',
    'unshard_batch',
    'pmap_with_devices',
    # JIT
    'jit_with_device',
    'static_argnums_for_config',
    # Batching
    'process_in_batches',
    'split_key_for_devices',
]
