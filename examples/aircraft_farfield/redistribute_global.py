"""Redistribute the complete 50 m sphere with edge, area, and angle regularity."""
from pathlib import Path
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

root = Path(__file__).resolve().parent
base = np.load(root / "sphere_balanced.npz")
U0 = base["unit_directions"]
Q = base["quads"]
edges = np.unique(np.sort(np.concatenate([Q[:, [0,1]], Q[:, [1,2]],
                                           Q[:, [2,3]], Q[:, [3,0]]]), axis=1), axis=0)
QQ = jnp.array(Q)
EE = jnp.array(edges)
REF = jnp.array(U0)

def metrics(z):
    U = z.reshape(-1, 3)
    U = U / jnp.linalg.norm(U, axis=1)[:, None]
    v = U[QQ]
    side = jnp.roll(v, -1, axis=1) - v
    length = jnp.linalg.norm(side, axis=2)
    # Four normalized corner sines; positive values preserve orientation.
    prev = jnp.roll(v, 1, axis=1) - v
    sine = jnp.sum(jnp.cross(side, prev) * v, axis=2) / (
        jnp.linalg.norm(side, axis=2) * jnp.linalg.norm(prev, axis=2))
    area = jnp.sum(jnp.sum(jnp.cross(side, prev) * v, axis=2), axis=1) / 8
    return U, length, sine, area

def objective(z, tether):
    U, length, sine, area = metrics(z)
    all_edges = jnp.linalg.norm(U[EE[:, 1]] - U[EE[:, 0]], axis=1)
    loge = jnp.log(jnp.maximum(all_edges, 1e-9))
    loga = jnp.log(jnp.maximum(area, 1e-9))
    # Global edge and face equalization, plus within-cell aspect regularity.
    uniform = 3.0 * jnp.var(loge) + 2.0 * jnp.var(loga)
    aspect = jnp.mean(jnp.var(jnp.log(jnp.maximum(length, 1e-9)), axis=1))
    angle = jnp.mean((1.0 - sine) ** 2)
    barrier = 2e5 * jnp.mean(jnp.minimum(sine - 0.35, 0.0) ** 2)
    move = tether * jnp.mean(jnp.sum((U - REF) ** 2, axis=1))
    return uniform + 2.0 * aspect + 0.5 * angle + barrier + move

fg_jit = jax.jit(jax.value_and_grad(objective))
z = U0.ravel()
for tether in (2.0, 0.7, 0.2):
    def fg(x):
        value, grad = fg_jit(jnp.array(x), tether)
        return float(value), np.asarray(grad)
    result = minimize(fg, z, jac=True, method="L-BFGS-B",
                      options={"maxiter": 2500, "ftol": 1e-14, "gtol": 1e-9,
                               "maxls": 40, "maxcor": 15})
    z = result.x
    U, length, sine, area = map(np.asarray, metrics(z))
    print("tether", tether, "energy", result.fun, "min sine", sine.min(),
          "edge p95/p05", np.percentile(length, 95) / np.percentile(length, 5),
          "area p95/p05", np.percentile(area, 95) / np.percentile(area, 5),
          "max motion deg", np.rad2deg(np.arccos(np.clip(np.sum(U * U0, axis=1), -1, 1))).max(),
          flush=True)
np.savez_compressed(root / "sphere_global.npz", unit_directions=U, quads=Q,
                    area=area, corner_sj=sine)
