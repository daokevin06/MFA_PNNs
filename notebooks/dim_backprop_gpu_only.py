"""Exact GPU based computation of polynomial-network neurovariety dimensions. This module has no SageMath or NumPy backend.   

It requires CuPy and a CUDA-capable NVIDIA GPU. CuPy replaces Numpy and provides GPU-accelerated array operations.

Comment: I used an ASUS Dual GeForce RTX 4070 Super.

The public function keeps the original calling convention of the Sage implementation: compute_dimension(network_widths, network_exponent) so that I can use the previous notebooks.

The returned tuple like before is: (sizes, exponent, ambient_dim, expected_dim, dimension, defect).

The computation is exact over the finite fields listed in ``DEFAULT_PRIMES``. One might want to pick larger primes for large runs.

All samples and all output-coordinate pullbacks are differentiated in one batched GPU computation.  

The final Jacobian rank is computed modulo each prime by GPU-parallel row elimination.
"""

from __future__ import annotations

from math import comb
from typing import Iterator, Sequence

try:
    import cupy as cp
except ImportError as exc:
    raise RuntimeError(
        "This GPU-only module requires CuPy. Install the CuPy package that "
        "matches your CUDA version, for example `pip install cupy-cuda12x`."
    ) from exc


DEFAULT_PRIMES = (100003, 100153)
_INT64_MAX = (1 << 63) - 1


def _require_cuda() -> None:
    """Raise a clear error unless a usable CUDA device is visible."""
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise RuntimeError(
            "CuPy is installed, but CUDA could not be initialized. Check the "
            "NVIDIA driver, CUDA/CuPy compatibility, and notebook kernel."
        ) from exc

    if device_count < 1:
        raise RuntimeError("No CUDA-capable GPU is visible to CuPy.")


def gpu_information() -> dict[str, object]:
    """Return basic information about the CUDA device used by this module."""
    _require_cuda()
    device_id = int(cp.cuda.runtime.getDevice())
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return {
        "device_id": device_id,
        "name": name,
        "device_count": int(cp.cuda.runtime.getDeviceCount()),
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
    }


def _is_prime(value: int) -> bool:
    """Deterministic Miller--Rabin test for unsigned 64-bit integers to check for primality."""
    if value < 2:
        return False

    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if value in small_primes:
        return True
    if any(value % prime == 0 for prime in small_primes):
        return False

    odd_part = value - 1
    power_of_two = 0
    while odd_part % 2 == 0:
        power_of_two += 1
        odd_part //= 2

    # Deterministic for every n < 2^64.
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, odd_part, value)
        if witness in (1, value - 1):
            continue
        for _ in range(power_of_two - 1):
            witness = (witness * witness) % value
            if witness == value - 1:
                break
        else:
            return False

    return True


def _weak_compositions(total: int, length: int) -> Iterator[tuple[int, ...]]:
    """Yield nonnegative ``length``-tuples whose entries sum to ``total``. 
    These will be used to form a unisolvent evaluation set for 
    homogeneous polynomials of total degree ``total`` in 
    ``length``-many variables. Tuples should be generated in lexicographic order."""
    if length == 0:
        if total == 0:
            yield ()
        return
    if length == 1:
        yield (total,)
        return

    for first in range(total + 1):
        for rest in _weak_compositions(total - first, length - 1):
            yield (first,) + rest


def _unisolvent_samples(input_dim: int, degree: int, prime: int):
    r"""Construct the homogeneous interpolation points directly on the GPU.

    On the chart ``x_0 = 1``, homogeneous degree-``degree`` forms become
    polynomials of total degree at most ``degree`` in ``input_dim - 1``
    variables.  The integer simplex is an unisolvent evaluation set when
    ``prime > degree``.
    """
    if input_dim < 1:
        raise ValueError("the input layer must have positive width")
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if prime <= degree:
        raise ValueError(
            f"prime {prime} must exceed polynomial degree {degree}"
        )

    expected = comb(degree + input_dim - 1, input_dim - 1)
    if input_dim == 1:
        return cp.ones((1, 1), dtype=cp.int64)

    points: list[tuple[int, ...]] = []
    for total in range(degree + 1):
        for alpha in _weak_compositions(total, input_dim - 1):
            points.append((1,) + alpha)

    if len(points) != expected:
        raise RuntimeError(
            f"internal sample-count error: got {len(points)}, expected {expected}"
        )

    return cp.asarray(points, dtype=cp.int64) % prime


def _mod_pow(base, exponent: int, prime: int):
    """Elementwise matrix + mod p exponentiation on the GPU."""
    if exponent < 0:
        raise ValueError("exponent must be nonnegative")

    result = cp.ones_like(base, dtype=cp.int64)
    if exponent == 0:
        return result

    power = base.astype(cp.int64, copy=False) % prime
    remaining = int(exponent)
    while remaining:
        if remaining & 1:
            result = (result * power) % prime
        remaining >>= 1
        if remaining:
            power = (power * power) % prime

    return result


def _check_dot_product_safety(widths: Sequence[int], prime: int) -> None:
    """Prevent signed int64 overflow before a matrix product is reduced."""
    largest_inner_dimension = max(int(width) for width in widths)
    worst_case = largest_inner_dimension * (prime - 1) ** 2
    if worst_case > _INT64_MAX:
        raise OverflowError(
            "An int64 GPU matrix product may overflow before reduction modulo "
            "the prime. Use a smaller prime or narrower layers."
        )


def _random_weights(
    widths: Sequence[int], prime: int, seed: int
) -> list[cp.ndarray]:
    """Generate all network weight matrices directly in GPU memory.
    The random number generator is seeded for reproducibility. 
    The generation is uniform over ``0, 1, ..., prime - 1``.
    """
    rng = cp.random.RandomState(seed)
    return [
        rng.randint(
            0,
            prime,
            size=(out_width, in_width),
            dtype=cp.int64,
        )
        for in_width, out_width in zip(widths[:-1], widths[1:])
    ]


def _parameter_offsets(
    weights: Sequence[cp.ndarray],
) -> tuple[list[int], int]:
    offsets: list[int] = []
    total = 0
    for weight in weights:
        offsets.append(total)
        total += int(weight.shape[0] * weight.shape[1])
    return offsets, total


def _batched_weight_jacobian(
    weights: Sequence[cp.ndarray],
    samples: cp.ndarray,
    exponent: int,
    prime: int,
) -> cp.ndarray:
    """Evaluate every output/weight derivative at every sample on the GPU.

    The result has shape ``(number_of_samples, output_width, num_parameters)``.
    Weight matrices are flattened layer by layer in row-major order, matching
    the ordering used by Sage's matrix ``list()`` method in the old code.
    """
    if exponent < 1:
        raise ValueError("network exponent must be at least 1")
    if not weights:
        raise ValueError("the network must contain at least one weight layer")

    activations = [samples]
    preactivations: list[cp.ndarray] = []
    activation = samples

    # Hidden layers use z -> z^exponent; the final layer is linear.
    for weight in weights[:-1]:
        preactivation = cp.matmul(activation, weight.T) % prime
        preactivations.append(preactivation)
        activation = _mod_pow(preactivation, exponent, prime)
        activations.append(activation)

    batch_size = int(samples.shape[0])
    output_width = int(weights[-1].shape[0])
    offsets, num_parameters = _parameter_offsets(weights)

    jacobian = cp.zeros(
        (batch_size, output_width, num_parameters), dtype=cp.int64
    )

    # Final linear layer.
    final_input = activations[-1]
    final_offset = offsets[-1]
    final_input_width = int(weights[-1].shape[1])
    for output_index in range(output_width):
        start = final_offset + output_index * final_input_width
        stop = start + final_input_width
        jacobian[:, output_index, start:stop] = final_input

    if len(weights) == 1:
        return jacobian

    # Derivatives of all output coordinates with respect to the last hidden
    # preactivation: shape (sample, output, hidden neuron).
    delta = cp.broadcast_to(
        weights[-1][None, :, :],
        (batch_size, output_width, int(weights[-1].shape[1])),
    ).copy()
    derivative = (
        (exponent % prime)
        * _mod_pow(preactivations[-1], exponent - 1, prime)
    ) % prime
    delta = (delta * derivative[:, None, :]) % prime

    # Hidden layers from last to first.
    for layer_index in range(len(weights) - 2, -1, -1):
        weight = weights[layer_index]
        layer_input = activations[layer_index]

        gradient = (
            delta[:, :, :, None] * layer_input[:, None, None, :]
        ) % prime

        start = offsets[layer_index]
        stop = start + int(weight.shape[0] * weight.shape[1])
        jacobian[:, :, start:stop] = gradient.reshape(
            batch_size, output_width, stop - start
        )

        if layer_index > 0:
            delta = cp.matmul(delta, weight) % prime
            derivative = (
                (exponent % prime)
                * _mod_pow(
                    preactivations[layer_index - 1], exponent - 1, prime
                )
            ) % prime
            delta = (delta * derivative[:, None, :]) % prime

    return jacobian


def _rank_mod_prime_gpu(
    matrix: cp.ndarray,
    prime: int,
    workspace_bytes: int = 512 * 1024**2,
) -> int:
    """Compute exact matrix rank over GF(prime) using GPU row operations.
    
    Question for future: Can we replace this with a predefined CuPy function?
    """
    if matrix.ndim != 2:
        raise ValueError("rank input must be a matrix")
    if workspace_bytes <= 0:
        raise ValueError("workspace_bytes must be positive")

    reduced = matrix.astype(cp.int64, copy=True) % prime

    # Eliminate along the smaller dimension.
    if reduced.shape[1] > reduced.shape[0]:
        reduced = reduced.T.copy()

    nrows, ncols = map(int, reduced.shape)
    pivot_row = 0

    for column in range(ncols):
        if pivot_row == nrows:
            break

        nonzero = reduced[pivot_row:, column] != 0
        if not bool(cp.any(nonzero).item()):
            continue

        pivot = pivot_row + int(cp.argmax(nonzero).item())
        if pivot != pivot_row:
            temporary = reduced[pivot_row, :].copy()
            reduced[pivot_row, :] = reduced[pivot, :]
            reduced[pivot, :] = temporary

        pivot_value = int(reduced[pivot_row, column].item())
        inverse = pow(pivot_value, prime - 2, prime)
        reduced[pivot_row, column:] = (
            reduced[pivot_row, column:] * inverse
        ) % prime

        # The row updates are parallel CUDA kernels. Chunking bounds temporary
        # memory usage for large Jacobians.
        remaining_columns = ncols - column
        bytes_per_row = max(1, 3 * remaining_columns * 8)
        rows_per_chunk = max(1, workspace_bytes // bytes_per_row)

        start = pivot_row + 1
        while start < nrows:
            stop = min(nrows, start + rows_per_chunk)
            factors = reduced[start:stop, column].copy()
            reduced[start:stop, column:] = (
                reduced[start:stop, column:]
                - factors[:, None] * reduced[pivot_row, column:][None, :]
            ) % prime
            start = stop

        pivot_row += 1

    return pivot_row


def compute_dimension(
    network_widths: Sequence[int],
    network_exponent: int,
    *,
    primes: Sequence[int] = DEFAULT_PRIMES,
    seed: int = 20260630,
    rank_workspace_bytes: int = 512 * 1024**2,
    verbose: bool = False,
):
    """Compute the neurovariety dimension entirely with the CUDA backend.

    Parameters
    ----------
    network_widths:
        Layer widths ``[d0, d1, ..., dL]``.
    network_exponent:
        Common hidden-layer activation exponent.
    primes:
        Prime moduli used to cross-check the generic rank.
    seed:
        Base random seed for the network weights.
    rank_workspace_bytes:
        Approximate upper bound for temporary elimination workspace.
    verbose:
        Print GPU and rank information.

    Returns
    -------
    tuple
        ``(sizes, exponent, ambient_dim, expected_dim, dimension, defect)``.
    """
    _require_cuda()

    widths = tuple(int(width) for width in network_widths)
    exponent = int(network_exponent)

    if len(widths) < 2:
        raise ValueError("network_widths must contain input and output widths")
    if any(width <= 0 for width in widths):
        raise ValueError("all network widths must be positive")
    if exponent < 1:
        raise ValueError("network_exponent must be at least 1")
    if not primes:
        raise ValueError("at least one prime is required")

    degree = exponent ** (len(widths) - 2)
    ambient_per_output = comb(degree + widths[0] - 1, widths[0] - 1)
    ambient_dim = ambient_per_output * widths[-1]
    num_parameters = sum(
        in_width * out_width
        for in_width, out_width in zip(widths[:-1], widths[1:])
    )

    dimensions: list[int] = []

    if verbose:
        info = gpu_information()
        print(
            f"GPU {info['device_id']}: {info['name']} | "
            f"CuPy {info['cupy_version']} | "
            f"CUDA runtime {info['cuda_runtime_version']}"
        )

    for prime_index, prime_value in enumerate(primes):
        prime = int(prime_value)
        if not _is_prime(prime):
            raise ValueError(f"modulus {prime} is not prime")
        _check_dot_product_safety(widths, prime)

        samples = _unisolvent_samples(widths[0], degree, prime)
        weights = _random_weights(
            widths,
            prime,
            seed + 1_000_003 * prime_index + prime,
        )

        jacobian = _batched_weight_jacobian(
            weights,
            samples,
            exponent,
            prime,
        )
        rank_matrix = jacobian.reshape(
            ambient_per_output * widths[-1], num_parameters
        )
        dimension = _rank_mod_prime_gpu(
            rank_matrix,
            prime,
            workspace_bytes=rank_workspace_bytes,
        )
        dimensions.append(dimension)

        # Ensure kernels for this prime have completed before reporting and
        # releasing memory.
        cp.cuda.Stream.null.synchronize()

        if verbose:
            print(
                f"prime={prime}, samples={ambient_per_output}, "
                f"rank_matrix={tuple(rank_matrix.shape)}, rank={dimension}"
            )

        del rank_matrix, jacobian, weights, samples
        cp.get_default_memory_pool().free_all_blocks()

    if not all(dimension == dimensions[0] for dimension in dimensions):
        raise ValueError(
            "different dimensions over finite fields: " + str(dimensions)
        )

    naive_bound = sum(
        (in_width - 1) * out_width
        for in_width, out_width in zip(widths[:-1], widths[1:])
    ) + widths[-1]
    expected_dim = min(ambient_dim, naive_bound)
    dimension = dimensions[0]

    return (
        list(widths),
        exponent,
        ambient_dim,
        expected_dim,
        dimension,
        expected_dim - dimension,
    )
