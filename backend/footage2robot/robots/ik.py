"""Task-priority damped least-squares inverse kinematics with joint limits, warm-started per frame."""

from __future__ import annotations

import numpy as np

from .urdf import Chain


def solve_ik(chain: Chain, pos: np.ndarray, approach: np.ndarray | None, q0: np.ndarray,
             approach_axis: np.ndarray, second: np.ndarray | None = None,
             second_axis: np.ndarray | None = None, *, second_weight: float = 0.3,
             posture_gain: float = 0.05, iters: int = 60, damping: float = 0.02,
             rot_damping: float = 0.15, tol: float = 1e-5) -> tuple[np.ndarray, float]:
    """Find joint angles placing the end-effector at ``pos`` (base frame, metres).

    Task-priority scheme: position is solved first; aligning the tool's
    ``approach_axis`` (EE frame) with the world direction ``approach`` — and a
    second tool axis with ``second``, up to sign, since a parallel gripper is
    symmetric — only uses the motion left over (the null space), and a weak
    posture term keeps any remaining freedom near ``q0`` for smooth motion.
    Returns ``(q, position_error_m)``.
    """
    q = np.clip(q0.astype(np.float64).copy(), chain.lower, chain.upper)
    n = chain.dof
    eye = np.eye(n)

    def step_for(T, J, free):
        J = J * free                      # locked joints contribute no motion
        R = T[:3, :3]
        Jp = J[:3]
        Jp_pinv = Jp.T @ np.linalg.inv(Jp @ Jp.T + damping ** 2 * np.eye(3))
        dq = Jp_pinv @ (pos - T[:3, 3])
        N = eye - Jp_pinv @ Jp
        if approach is not None:
            e_rot = np.cross(R @ approach_axis, approach)
            if second is not None and second_axis is not None:
                s = R @ second_axis
                e_rot = e_rot + second_weight * np.cross(s, second if np.dot(s, second) >= 0 else -second)
            Jo = J[3:] @ N
            dq = dq + N @ (Jo.T @ np.linalg.solve(Jo @ Jo.T + rot_damping ** 2 * np.eye(3), e_rot - J[3:] @ dq))
            N = N @ (eye - np.linalg.pinv(Jo, rcond=1e-2) @ Jo)
        dq = (dq + posture_gain * (N @ (q0 - q))) * free
        norm = np.linalg.norm(dq)
        return dq * (0.4 / norm) if norm > 0.4 else dq

    for _ in range(iters):
        T, J = chain.jacobian(q)
        free = np.ones(n)
        dq = step_for(T, J, free)
        # joints that would leave their limits are locked and the step re-solved without them
        blocked = ((q + dq > chain.upper) & (q >= chain.upper - 1e-9)) | ((q + dq < chain.lower) & (q <= chain.lower + 1e-9))
        if blocked.any() and not blocked.all():
            free = (~blocked).astype(np.float64)
            dq = step_for(T, J, free)
        q = np.clip(q + dq, chain.lower, chain.upper)
        if np.linalg.norm(dq) < tol:
            break
    return q, float(np.linalg.norm(pos - chain.fk(q)[:3, 3]))


def solve_trajectory(chain: Chain, positions: np.ndarray, approaches: np.ndarray | None,
                     rest: np.ndarray, approach_axis: np.ndarray, seconds: np.ndarray | None = None,
                     second_axis: np.ndarray | None = None, *, max_step: float | None = None,
                     retry_above_m: float = 0.01, restarts: int = 8, **kw) -> tuple[np.ndarray, np.ndarray]:
    """IK for every frame as one continuous motion.

    Each frame is warm-started from its neighbour and may move each joint by at
    most ``max_step`` radians (the robot's speed limit per frame), so the result
    never jumps between arm configurations. If a sweep from the first frame gets
    stuck — typically against a joint limit — the hardest frame is solved from
    fresh starts and the sweep is redone outwards from there; the better of the
    two is kept. NaN targets yield NaN rows. Returns ``(q (T, dof), error (T,))``.
    """
    T = positions.shape[0]
    valid = np.where(np.isfinite(positions).all(axis=1))[0]
    qs = np.full((T, chain.dof), np.nan)
    errs = np.full(T, np.nan)
    if len(valid) == 0:
        return qs, errs
    rng = np.random.default_rng(0)

    def targets(t):
        a = approaches[t] if approaches is not None and np.isfinite(approaches[t]).all() else None
        s = seconds[t] if seconds is not None and a is not None and np.isfinite(seconds[t]).all() else None
        return a, s

    def fresh(t, extra=()):
        a, s = targets(t)
        starts = [rest, 0.5 * (chain.lower + chain.upper), *extra,
                  *rng.uniform(chain.lower, chain.upper, size=(restarts, chain.dof))]
        return min((solve_ik(chain, positions[t], a, q0, approach_axis, s, second_axis, iters=200, **kw)
                    for q0 in starts), key=lambda c: c[1])

    def sweep(seed_frame, seed_q):
        q_out = np.full((T, chain.dof), np.nan)
        e_out = np.full(T, np.nan)
        k = int(np.searchsorted(valid, seed_frame))
        q_out[seed_frame] = seed_q
        e_out[seed_frame] = np.linalg.norm(positions[seed_frame] - chain.fk(seed_q)[:3, 3])
        for order in (valid[k + 1:], valid[:k][::-1]):
            q_prev = seed_q
            for t in order:
                a, s = targets(t)
                q, err = solve_ik(chain, positions[t], a, q_prev, approach_axis, s, second_axis, **kw)
                if max_step is not None:
                    q = np.clip(q, q_prev - max_step, q_prev + max_step)
                    err = float(np.linalg.norm(positions[t] - chain.fk(q)[:3, 3]))
                q_out[t], e_out[t], q_prev = q, err, q
        return q_out, e_out

    first = int(valid[0])
    best = sweep(first, fresh(first)[0])
    for _ in range(2):
        e = best[1][valid]
        if (e > retry_above_m).mean() <= 0.02:
            break
        worst = int(valid[np.argmax(e)])
        candidate = sweep(worst, fresh(worst, extra=[best[0][worst]])[0])
        if np.mean(candidate[1][valid]) < np.mean(e):
            best = candidate
        else:
            break
    return best
