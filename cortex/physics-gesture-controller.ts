/**
 * physics-gesture-controller.ts
 *
 * Biomechanical gesture control with full physics simulation.
 *
 * Pipeline per frame:
 *
 *  HandInput
 *      │
 *  ┌───▼──────────────────────────────────────────┐
 *  │  GESTURE LAYER                               │
 *  │  A. FingerCircleTracker  → 2D angular vel.   │
 *  │  B. WristQuatTracker     → 3D delta quat.    │
 *  │  C. HandDragTracker      → linear velocity   │
 *  └───┬──────────────────────────────────────────┘
 *      │ raw signals + confidence scores
 *  ┌───▼──────────────────────────────────────────┐
 *  │  MICRO-INTERACTION LAYER                     │
 *  │  dead zone → speed curve → clamp             │
 *  └───┬──────────────────────────────────────────┘
 *      │ conditioned signals
 *  ┌───▼──────────────────────────────────────────┐
 *  │  DISAMBIGUATION ENGINE                       │
 *  │  winner-takes-all + hysteresis + hybrid      │
 *  └───┬──────────────────────────────────────────┘
 *      │ active mode + blended force
 *  ┌───▼──────────────────────────────────────────┐
 *  │  PHYSICS ENGINE                              │
 *  │  inertia, exponential friction, clamping     │
 *  └───┬──────────────────────────────────────────┘
 *      │ angular / linear velocity
 *  ┌───▼──────────────────────────────────────────┐
 *  │  POST-PROCESSING                             │
 *  │  snap magnets, elastic bounds, feedback      │
 *  └───┬──────────────────────────────────────────┘
 *      │
 *  GestureOutput
 */

// ═══════════════════════════════════════════════════════════════════════════════
// Public Types
// ═══════════════════════════════════════════════════════════════════════════════

export interface Vec3 { x: number; y: number; z: number }
export interface Quat { w: number; x: number; y: number; z: number }

export interface FingerData {
  id: number;           // 0=thumb 1=index 2=middle 3=ring 4=pinky
  tip_position: Vec3;
  direction: Vec3;
}

export interface HandInput {
  hand: {
    position: Vec3;
    velocity: Vec3;     // m/s
    palm_normal: Vec3;  // unit vector perpendicular to palm
    wrist_position: Vec3;
  };
  fingers: FingerData[];
}

export type GestureMode = "2D_rotation" | "3D_rotation" | "translation" | "hybrid" | "idle";

export interface GestureOutput {
  /** Dominant gesture mode this frame. */
  mode: GestureMode;

  /**
   * Per-frame rotation delta in Euler angles (radians, XYZ world order).
   * Apply directly: object.rotation.x += output.rotation.x
   */
  rotation: Vec3;

  /**
   * Per-frame translation delta (m/s × dt, scaled).
   * Apply: object.position.x += output.translation.x
   */
  translation: Vec3;

  /** Physics inertia state — use `strength` for visual trail effects. */
  inertia: { active: boolean; strength: number };

  /** Dominant gesture confidence [0–1]. */
  confidence: number;

  /** Accumulated orientation quaternion — use for direct camera/object.quaternion. */
  orientation: Quat;

  /** Non-null when snap magnet locks to an axis. Useful for UI feedback. */
  snapFeedback: { axis: "z"; targetAngle: number } | null;
}

export interface PhysicsGestureConfig {
  // ── Physics ──────────────────────────────────────────────────────────────
  /** Friction coefficient (1/s). Higher = faster deceleration. Default 3 */
  angularFriction?: number;
  /** Default 4 */
  linearFriction?: number;
  /** Max angular speed cap (rad/s). Default 12 */
  maxAngularSpeed?: number;
  /** Max translation speed (m/s). Default 1.5 */
  maxLinearSpeed?: number;
  /** How fast physics velocity tracks gesture input [0-1]. Default 0.35 */
  inputResponsiveness?: number;

  // ── Dead zone & Speed curve ───────────────────────────────────────────────
  /** Angular inputs below this are discarded (rad/s). Default 0.06 */
  deadZoneAngular?: number;
  /** Linear inputs below this are discarded (m/s). Default 0.008 */
  deadZoneLinear?: number;
  /**
   * Exponent for non-linear speed mapping.
   * >1 = small moves are slower (precision), large moves are faster.
   * Default 1.8
   */
  speedCurveExponent?: number;

  // ── Snap magnets ──────────────────────────────────────────────────────────
  /** Enable snap-to-90° for 2D rotation. Default true */
  snapEnabled?: boolean;
  /** Proximity (rad) to trigger snap attraction. Default 0.18 (~10°) */
  snapRadius?: number;
  /** Max angular speed for snap to engage (rad/s). Default 1.2 */
  snapVelocityThreshold?: number;
  /** Spring constant for snap attraction. Default 8 */
  snapSpringK?: number;

  // ── Elastic bounds ────────────────────────────────────────────────────────
  /** Max accumulated rotation on X axis (rad). Default Infinity */
  maxRotationX?: number;
  /** Max accumulated rotation on Y axis (rad). Default Infinity */
  maxRotationY?: number;
  /**
   * Soft zone depth before the hard limit (rad).
   * Resistance starts here and reaches 100% at the limit. Default 0.26 (15°)
   */
  elasticSoftZone?: number;

  // ── Gesture detection ─────────────────────────────────────────────────────
  /** Sliding window size for finger circle analysis. Default 20 */
  historyWindow?: number;
  /** Min circle radius to consider intentional rotation (m). Default 0.02 */
  minCircleRadius?: number;
  /** Tip-to-wrist distance to classify a finger as "extended" (m). Default 0.06 */
  fingerExtendDist?: number;

  // ── Disambiguation ────────────────────────────────────────────────────────
  /** Frames before mode switch is allowed. Default 8 */
  hysteresisFrames?: number;
  /** Confidence needed to enter a new mode. Default 0.42 */
  entryThreshold?: number;
  /** Confidence drop-to before mode exits. Default 0.22 */
  exitThreshold?: number;
  /** Both finger + wrist must exceed this for hybrid mode. Default 0.38 */
  hybridThreshold?: number;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal Math — Vec3
// ═══════════════════════════════════════════════════════════════════════════════

const v3 = {
  add:   (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }),
  sub:   (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }),
  scale: (a: Vec3, s: number): Vec3 => ({ x: a.x * s, y: a.y * s, z: a.z * s }),
  dot:   (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z,
  cross: (a: Vec3, b: Vec3): Vec3 => ({
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  }),
  len:  (a: Vec3): number => Math.sqrt(a.x ** 2 + a.y ** 2 + a.z ** 2),
  norm: (a: Vec3): Vec3 => {
    const l = Math.sqrt(a.x ** 2 + a.y ** 2 + a.z ** 2);
    return l > 1e-9 ? { x: a.x / l, y: a.y / l, z: a.z / l } : { x: 0, y: 0, z: 0 };
  },
  lerp: (a: Vec3, b: Vec3, t: number): Vec3 => ({
    x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t,
  }),
  zero:  (): Vec3 => ({ x: 0, y: 0, z: 0 }),
  clone: (a: Vec3): Vec3 => ({ x: a.x, y: a.y, z: a.z }),
};

// ═══════════════════════════════════════════════════════════════════════════════
// Internal Math — Quaternion
// ═══════════════════════════════════════════════════════════════════════════════

const Q = {
  identity: (): Quat => ({ w: 1, x: 0, y: 0, z: 0 }),

  fromAxisAngle(axis: Vec3, angle: number): Quat {
    const n = v3.norm(axis);
    const s = Math.sin(angle / 2);
    return { w: Math.cos(angle / 2), x: n.x * s, y: n.y * s, z: n.z * s };
  },

  /** Hamilton product a × b (b applied first in world-space convention). */
  mul(a: Quat, b: Quat): Quat {
    return {
      w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
      x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
      y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
      z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    };
  },

  norm(qi: Quat): Quat {
    const l = Math.sqrt(qi.w ** 2 + qi.x ** 2 + qi.y ** 2 + qi.z ** 2);
    if (l < 1e-9) return Q.identity();
    return { w: qi.w / l, x: qi.x / l, y: qi.y / l, z: qi.z / l };
  },

  conj: (qi: Quat): Quat => ({ w: qi.w, x: -qi.x, y: -qi.y, z: -qi.z }),

  slerp(a: Quat, b: Quat, t: number): Quat {
    let dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
    let { w: bw, x: bx, y: by, z: bz } = b;
    if (dot < 0) { dot = -dot; bw = -bw; bx = -bx; by = -by; bz = -bz; }
    if (dot > 0.9995) {
      return Q.norm({ w: a.w + (bw - a.w) * t, x: a.x + (bx - a.x) * t,
                      y: a.y + (by - a.y) * t, z: a.z + (bz - a.z) * t });
    }
    const theta0 = Math.acos(dot);
    const theta = theta0 * t;
    const sinT = Math.sin(theta);
    const sinT0 = Math.sin(theta0);
    const s0 = Math.cos(theta) - dot * sinT / sinT0;
    const s1 = sinT / sinT0;
    return { w: s0 * a.w + s1 * bw, x: s0 * a.x + s1 * bx,
             y: s0 * a.y + s1 * by, z: s0 * a.z + s1 * bz };
  },

  /**
   * Convert angular velocity vector (rad/s) × dt → delta quaternion.
   * For small angles this is exact; for large angles (> π) it stays stable.
   */
  fromAngularVelocityDt(omega: Vec3, dt: number): Quat {
    const angle = v3.len(omega) * dt;
    if (angle < 1e-9) return Q.identity();
    return Q.fromAxisAngle(omega, angle);
  },

  /** Decompose to Euler XYZ (radians) for downstream consumption. */
  toEuler(qi: Quat): Vec3 {
    // Roll (X)
    const sinR = 2 * (qi.w * qi.x + qi.y * qi.z);
    const cosR = 1 - 2 * (qi.x ** 2 + qi.y ** 2);
    const rx = Math.atan2(sinR, cosR);

    // Pitch (Y)
    const sinP = 2 * (qi.w * qi.y - qi.z * qi.x);
    const ry = Math.abs(sinP) >= 1 ? Math.sign(sinP) * Math.PI / 2 : Math.asin(sinP);

    // Yaw (Z)
    const sinY = 2 * (qi.w * qi.z + qi.x * qi.y);
    const cosY = 1 - 2 * (qi.y ** 2 + qi.z ** 2);
    const rz = Math.atan2(sinY, cosY);

    return { x: rx, y: ry, z: rz };
  },

  clone: (qi: Quat): Quat => ({ ...qi }),
};

// ═══════════════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════════════

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

class CircularBuffer<T> {
  private buf: T[];
  private head = 0;
  private _size = 0;

  constructor(private cap: number) { this.buf = new Array(cap); }

  push(item: T): void {
    this.buf[this.head] = item;
    this.head = (this.head + 1) % this.cap;
    if (this._size < this.cap) this._size++;
  }

  get size(): number { return this._size; }

  toArray(): T[] {
    const out: T[] = [];
    for (let i = 0; i < this._size; i++)
      out.push(this.buf[(this.head - this._size + i + this.cap) % this.cap]);
    return out;
  }

  get latest(): T | undefined { return this.buf[(this.head - 1 + this.cap) % this.cap]; }
  reset(): void { this._size = 0; this.head = 0; }
}

/** Exponential moving average. */
class EMA {
  private v = 0; private init = false;
  constructor(private a: number) {}
  update(x: number): number {
    this.v = this.init ? this.v + this.a * (x - this.v) : (this.init = true, x);
    return this.v;
  }
  get value() { return this.v; }
  reset() { this.init = false; this.v = 0; }
}

/** Unwrap angles so consecutive values don't jump ±π. */
function unwrap(angles: number[]): number[] {
  if (!angles.length) return [];
  const out = [angles[0]];
  for (let i = 1; i < angles.length; i++) {
    let d = angles[i] - angles[i - 1];
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    out.push(out[i - 1] + d);
  }
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Layer 1 — Gesture Trackers
// ═══════════════════════════════════════════════════════════════════════════════

// ── A. Finger Circle Tracker ──────────────────────────────────────────────────

interface CircleResult { angularVelocity: number; confidence: number }

class FingerCircleTracker {
  private history: CircularBuffer<{ pos: Vec3; t: number }>;
  private confEMA = new EMA(0.10);
  private velEMA  = new EMA(0.18);

  constructor(private window: number, private minRadius: number) {
    this.history = new CircularBuffer(window);
  }

  update(fingers: FingerData[], tsMs: number): CircleResult {
    const tip = fingers.find(f => f.id === 1) ?? fingers.find(f => f.id !== 0);
    if (!tip) return { angularVelocity: 0, confidence: 0 };

    this.history.push({ pos: tip.tip_position, t: tsMs });
    if (this.history.size < 6) return { angularVelocity: 0, confidence: 0 };

    const samples = this.history.toArray();
    const cx = samples.reduce((s, p) => s + p.pos.x, 0) / samples.length;
    const cy = samples.reduce((s, p) => s + p.pos.y, 0) / samples.length;

    // Angular velocity from unwrapped angles
    const rawAngles = samples.map(s => Math.atan2(s.pos.y - cy, s.pos.x - cx));
    const angles = unwrap(rawAngles);
    const dt = (samples[samples.length - 1].t - samples[0].t) / 1000;
    const sweep = angles[angles.length - 1] - angles[0];
    const angVel = dt > 0.001 ? sweep / dt : 0;

    // Circularity: coefficient of variation of radii (low = perfect circle)
    const radii = samples.map(s => Math.sqrt((s.pos.x - cx) ** 2 + (s.pos.y - cy) ** 2));
    const meanR = radii.reduce((a, b) => a + b, 0) / radii.length;
    const varR  = radii.reduce((a, r) => a + (r - meanR) ** 2, 0) / radii.length;
    const cvR   = meanR > 1e-6 ? Math.sqrt(varR) / meanR : 1;

    const cRadius  = clamp(meanR / this.minRadius, 0, 1);
    const cArc     = clamp(Math.abs(sweep) / (Math.PI / 4), 0, 1); // ≥45° arc
    const cShape   = clamp(1 - cvR * 2, 0, 1);

    const rawConf  = cRadius * cArc * cShape;
    return {
      angularVelocity: this.velEMA.update(angVel),
      confidence:      clamp(this.confEMA.update(rawConf), 0, 1),
    };
  }

  reset() { this.history.reset(); this.confEMA.reset(); this.velEMA.reset(); }
}

// ── B. Wrist Quaternion Tracker ───────────────────────────────────────────────

interface WristResult { deltaQuat: Quat; confidence: number }

class WristQuatTracker {
  private prev: Vec3 | null = null;
  private confEMA = new EMA(0.10);
  // Noise floor: ignore normal changes smaller than ~0.5°
  private readonly NOISE = Math.sin(0.5 * Math.PI / 180);
  private readonly SATURATION = Math.sin(6 * Math.PI / 180); // 6°/frame = full confidence

  update(palmNormal: Vec3): WristResult {
    const n = v3.norm(palmNormal);

    if (!this.prev) {
      this.prev = n;
      return { deltaQuat: Q.identity(), confidence: 0 };
    }

    const cross = v3.cross(this.prev, n);
    const crossMag = v3.len(cross);
    const dot  = clamp(v3.dot(this.prev, n), -1, 1);
    const angle = Math.acos(dot);

    // Axis-angle → quaternion delta
    const deltaQuat = crossMag > 1e-9
      ? Q.fromAxisAngle(v3.scale(cross, 1 / crossMag), angle)
      : Q.identity();

    // Confidence: ramp between noise floor and saturation point
    const rawConf = clamp((crossMag - this.NOISE) / (this.SATURATION - this.NOISE), 0, 1);

    this.prev = n;
    return {
      deltaQuat,
      confidence: clamp(this.confEMA.update(rawConf), 0, 1),
    };
  }

  reset() { this.prev = null; this.confEMA.reset(); }
}

// ── C. Hand Drag Tracker ──────────────────────────────────────────────────────

interface DragResult { velocity: Vec3; confidence: number }

class HandDragTracker {
  private confEMA = new EMA(0.12);
  private readonly OPEN_FINGERS = 3; // min extended to be "open hand"

  constructor(private extendDist: number) {}

  update(handVel: Vec3, fingers: FingerData[], wrist: Vec3): DragResult {
    const extended = fingers.filter(f => v3.len(v3.sub(f.tip_position, wrist)) > this.extendDist).length;

    const cOpen  = clamp(extended / this.OPEN_FINGERS, 0, 1);
    const speed  = v3.len(handVel);
    const cSpeed = clamp(speed / 0.08, 0, 1); // saturates at 8 cm/s

    const rawConf = cOpen * 0.65 + cSpeed * 0.35;
    return {
      velocity:   handVel,
      confidence: clamp(this.confEMA.update(rawConf), 0, 1),
    };
  }

  reset() { this.confEMA.reset(); }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Layer 2 — Micro-Interaction Conditioning
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Apply dead zone + non-linear speed curve to a scalar.
 *
 * After the dead zone, the value is re-normalized so output starts at 0 (no
 * discontinuity). The speed curve exponent (> 1) compresses slow movements
 * (precision mode) and expands fast ones (navigation mode).
 *
 *   exponent 1.0 → linear
 *   exponent 1.8 → small inputs ≈ 60% of linear, large inputs ≈ 130% of linear
 *   exponent 2.5 → even more precision at low speed
 */
function conditionScalar(raw: number, deadZone: number, maxVal: number, exponent: number): number {
  const abs = Math.abs(raw);
  if (abs < deadZone) return 0;
  // Remap [deadZone, maxVal] → [0, 1]
  const n = clamp((abs - deadZone) / (maxVal - deadZone), 0, 1);
  // Apply speed curve
  const curved = Math.pow(n, exponent);
  return Math.sign(raw) * curved * maxVal;
}

function conditionVec3(raw: Vec3, deadZone: number, maxVal: number, exponent: number): Vec3 {
  const speed = v3.len(raw);
  if (speed < deadZone) return v3.zero();
  const n = clamp((speed - deadZone) / (maxVal - deadZone), 0, 1);
  const curved = Math.pow(n, exponent) * maxVal;
  return v3.scale(v3.norm(raw), curved);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Layer 3 — Gesture Disambiguation
// ═══════════════════════════════════════════════════════════════════════════════

interface Scores { finger: number; wrist: number; translation: number }

class Disambiguator {
  private mode: GestureMode = "idle";
  private stability = 0;

  constructor(
    private hysteresisFrames: number,
    private entryThreshold: number,
    private exitThreshold: number,
    private hybridThreshold: number,
  ) {}

  update(s: Scores): GestureMode {
    const desired = this.vote(s);

    // Hysteresis: require mode to stay dominant for N frames before switching
    if (desired === this.mode) {
      this.stability = Math.min(this.stability + 1, this.hysteresisFrames);
    } else {
      this.stability = Math.max(this.stability - 2, 0);
      if (this.stability === 0) this.mode = desired;
    }

    return this.mode;
  }

  private vote(s: Scores): GestureMode {
    // Hybrid: both finger and wrist exceed threshold
    if (s.finger > this.hybridThreshold && s.wrist > this.hybridThreshold) return "hybrid";

    // Find dominant score above entry threshold
    const candidates: [GestureMode, number][] = [
      ["3D_rotation",  s.wrist],
      ["2D_rotation",  s.finger],
      ["translation",  s.translation],
    ];

    // Check if current mode should EXIT first (uses lower exit threshold)
    const currentScore = this.scoreFor(this.mode, s);
    if (this.mode !== "idle" && currentScore < this.exitThreshold) {
      // Find new winner at entry threshold
      const winner = candidates.find(([, v]) => v >= this.entryThreshold);
      return winner ? winner[0] : "idle";
    }

    // Try to stay in current mode if still above exit threshold
    if (this.mode !== "idle" && currentScore >= this.exitThreshold) return this.mode;

    // Enter new mode if any candidate exceeds entry threshold
    const winner = candidates.find(([, v]) => v >= this.entryThreshold);
    return winner ? winner[0] : "idle";
  }

  private scoreFor(mode: GestureMode, s: Scores): number {
    switch (mode) {
      case "2D_rotation": return s.finger;
      case "3D_rotation": return s.wrist;
      case "translation": return s.translation;
      case "hybrid":      return Math.min(s.finger, s.wrist);
      default:            return 0;
    }
  }

  get currentMode() { return this.mode; }
  reset() { this.mode = "idle"; this.stability = 0; }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Layer 4 — Physics Engine
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Manages angular + linear momentum with exponential friction.
 *
 * Friction model: v(t) = v0 × e^(-k×t)
 * Per frame: v_new = v_old × e^(-k×dt)  — stable regardless of dt size.
 *
 * Input force blends toward target velocity at `responsiveness` rate (EMA on
 * velocity), so gesture input feels "sticky" rather than snapping instantly.
 */
class PhysicsEngine {
  angularVel: Vec3 = v3.zero(); // rad/s
  linearVel:  Vec3 = v3.zero(); // m/s

  constructor(
    private angFriction: number,
    private linFriction: number,
    private maxAngSpeed: number,
    private maxLinSpeed: number,
    private responsiveness: number,
  ) {}

  /** Feed a desired angular velocity (gesture intent). */
  applyAngularForce(target: Vec3): void {
    this.angularVel = v3.lerp(this.angularVel, target, this.responsiveness);
  }

  /** Feed a desired linear velocity. */
  applyLinearForce(target: Vec3): void {
    this.linearVel = v3.lerp(this.linearVel, target, this.responsiveness);
  }

  /** Integrate one frame. Returns delta angle (rad) and delta position (m). */
  integrate(dt: number): { deltaAngle: Vec3; deltaPos: Vec3 } {
    // Exponential friction: v *= e^(-k*dt)
    const angDecay = Math.exp(-this.angFriction * dt);
    const linDecay = Math.exp(-this.linFriction * dt);
    this.angularVel = v3.scale(this.angularVel, angDecay);
    this.linearVel  = v3.scale(this.linearVel,  linDecay);

    // Speed clamping
    const angSpeed = v3.len(this.angularVel);
    if (angSpeed > this.maxAngSpeed)
      this.angularVel = v3.scale(this.angularVel, this.maxAngSpeed / angSpeed);

    const linSpeed = v3.len(this.linearVel);
    if (linSpeed > this.maxLinSpeed)
      this.linearVel = v3.scale(this.linearVel, this.maxLinSpeed / linSpeed);

    return {
      deltaAngle: v3.scale(this.angularVel, dt),
      deltaPos:   v3.scale(this.linearVel,  dt),
    };
  }

  get speed(): number { return v3.len(this.angularVel); }

  reset(): void { this.angularVel = v3.zero(); this.linearVel = v3.zero(); }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Layer 5 — Post-Processing
// ═══════════════════════════════════════════════════════════════════════════════

// ── Snap Magnet (2D Z-axis) ───────────────────────────────────────────────────

interface SnapState {
  active: boolean;
  target: number;
}

class SnapMagnet {
  private state: SnapState = { active: false, target: 0 };
  private accumZ = 0;

  constructor(
    private radius: number,
    private velThreshold: number,
    private springK: number,
  ) {}

  /**
   * Given the current accumulated Z angle and angular velocity,
   * returns an additive correction to steer toward the nearest 90° snap.
   */
  update(
    accumZ: number,
    omegaZ: number,
    dt: number,
    enabled: boolean,
  ): { correction: number; feedback: GestureOutput["snapFeedback"] } {
    if (!enabled) return { correction: 0, feedback: null };

    const nearest = Math.round(accumZ / (Math.PI / 2)) * (Math.PI / 2);
    const dist = nearest - accumZ;
    const absOmega = Math.abs(omegaZ);

    // Enter snap state
    if (!this.state.active && Math.abs(dist) < this.radius && absOmega < this.velThreshold) {
      this.state = { active: true, target: nearest };
    }

    // Exit snap state if rotated away
    if (this.state.active && Math.abs(this.state.target - accumZ) > this.radius * 1.6) {
      this.state.active = false;
    }

    if (!this.state.active) return { correction: 0, feedback: null };

    // Spring attraction toward target
    const springForce = (this.state.target - accumZ) * this.springK;
    const correction = springForce * dt;

    return {
      correction,
      feedback: { axis: "z", targetAngle: this.state.target },
    };
  }

  reset() { this.state.active = false; this.accumZ = 0; }
}

// ── Elastic Bounds ────────────────────────────────────────────────────────────

/**
 * Returns a resistance multiplier [0, 1] for an axis.
 * 1.0 = free movement, 0.0 = fully blocked (at hard limit).
 * Resistance ramps linearly through the soft zone.
 */
function elasticResistance(angle: number, maxAngle: number, softZone: number): number {
  if (!isFinite(maxAngle)) return 1;
  const excess = Math.abs(angle) - (maxAngle - softZone);
  if (excess <= 0) return 1;
  return clamp(1 - excess / softZone, 0.05, 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Controller
// ═══════════════════════════════════════════════════════════════════════════════

const DEFAULTS: Required<PhysicsGestureConfig> = {
  angularFriction:       3.0,
  linearFriction:        4.0,
  maxAngularSpeed:       12,
  maxLinearSpeed:        1.5,
  inputResponsiveness:   0.35,
  deadZoneAngular:       0.06,
  deadZoneLinear:        0.008,
  speedCurveExponent:    1.8,
  snapEnabled:           true,
  snapRadius:            0.18,
  snapVelocityThreshold: 1.2,
  snapSpringK:           8,
  maxRotationX:          Infinity,
  maxRotationY:          Infinity,
  elasticSoftZone:       0.26,
  historyWindow:         20,
  minCircleRadius:       0.02,
  fingerExtendDist:      0.06,
  hysteresisFrames:      8,
  entryThreshold:        0.42,
  exitThreshold:         0.22,
  hybridThreshold:       0.38,
};

/**
 * PhysicsGestureController
 *
 * Drop-in replacement for GestureController with full physics simulation.
 *
 * @example
 * const gc = new PhysicsGestureController({ snapEnabled: true });
 *
 * function onFrame(input: HandInput, tsMs: number) {
 *   const out = gc.process(input, tsMs);
 *   // Euler delta:
 *   graph.rotation.x += out.rotation.x;
 *   graph.rotation.y += out.rotation.y;
 *   graph.rotation.z += out.rotation.z;
 *   // Or use quaternion directly:
 *   graph.quaternion.copy(out.orientation);
 *   // Translation:
 *   graph.position.x += out.translation.x;
 *   graph.position.y += out.translation.y;
 *   graph.position.z += out.translation.z;
 * }
 */
export class PhysicsGestureController {
  private cfg: Required<PhysicsGestureConfig>;

  // Gesture trackers
  private fingerTracker: FingerCircleTracker;
  private wristTracker:  WristQuatTracker;
  private dragTracker:   HandDragTracker;

  // Disambiguation
  private disambiguator: Disambiguator;

  // Physics
  private physics: PhysicsEngine;

  // Post-processing
  private snap: SnapMagnet;

  // Accumulated state (quaternion-based, no gimbal lock)
  private orientation: Quat = Q.identity();

  // Accumulated Euler angles for elastic bounds tracking
  private accumAngle: Vec3 = v3.zero();

  private lastTs = 0;

  constructor(config: PhysicsGestureConfig = {}) {
    this.cfg = { ...DEFAULTS, ...config };
    const c = this.cfg;

    this.fingerTracker = new FingerCircleTracker(c.historyWindow, c.minCircleRadius);
    this.wristTracker  = new WristQuatTracker();
    this.dragTracker   = new HandDragTracker(c.fingerExtendDist);

    this.disambiguator = new Disambiguator(
      c.hysteresisFrames, c.entryThreshold, c.exitThreshold, c.hybridThreshold,
    );

    this.physics = new PhysicsEngine(
      c.angularFriction, c.linearFriction,
      c.maxAngularSpeed, c.maxLinearSpeed,
      c.inputResponsiveness,
    );

    this.snap = new SnapMagnet(c.snapRadius, c.snapVelocityThreshold, c.snapSpringK);
  }

  /**
   * Process one hand-tracking frame.
   * @param input  Hand tracking snapshot
   * @param tsMs   Monotonic timestamp in milliseconds
   */
  process(input: HandInput, tsMs: number): GestureOutput {
    const { hand, fingers } = input;

    // Guard: compute dt; treat first frame or gaps > 500ms as fresh start
    const rawDt = (tsMs - this.lastTs) / 1000;
    const dt = this.lastTs === 0 || rawDt > 0.5 ? 0.016 : clamp(rawDt, 0.001, 0.1);
    this.lastTs = tsMs;
    const active = rawDt < 0.5;

    // ─── Step 1: Raw gesture signals ────────────────────────────────────────

    const finger = this.fingerTracker.update(fingers, tsMs);
    const wrist  = this.wristTracker.update(hand.palm_normal);
    const drag   = this.dragTracker.update(hand.velocity, fingers, hand.wrist_position);

    // ─── Step 2: Dead zone + speed curve conditioning ────────────────────────

    const condFingerVel = conditionScalar(
      finger.angularVelocity,
      this.cfg.deadZoneAngular,
      this.cfg.maxAngularSpeed,
      this.cfg.speedCurveExponent,
    );

    const condDragVel = conditionVec3(
      drag.velocity,
      this.cfg.deadZoneLinear,
      this.cfg.maxLinearSpeed,
      this.cfg.speedCurveExponent,
    );

    // Wrist: extract angular velocity from delta quaternion
    // angle of deltaQuat = 2*acos(w), axis = (x,y,z) / sin(angle/2)
    const wAngle = 2 * Math.acos(clamp(wrist.deltaQuat.w, -1, 1));
    const wSin   = Math.sqrt(1 - wrist.deltaQuat.w ** 2);
    const wAxis  = wSin > 1e-9
      ? { x: wrist.deltaQuat.x / wSin, y: wrist.deltaQuat.y / wSin, z: wrist.deltaQuat.z / wSin }
      : { x: 0, y: 0, z: 1 };
    const wristOmegaRaw: Vec3 = dt > 0 ? v3.scale(wAxis, wAngle / dt) : v3.zero();
    const condWristOmega = conditionVec3(
      wristOmegaRaw,
      this.cfg.deadZoneAngular,
      this.cfg.maxAngularSpeed,
      this.cfg.speedCurveExponent,
    );

    // ─── Step 3: Disambiguation ──────────────────────────────────────────────

    const scores: Scores = {
      finger:      finger.confidence,
      wrist:       wrist.confidence,
      translation: drag.confidence,
    };

    const mode = active ? this.disambiguator.update(scores) : "idle";

    // ─── Step 4: Route signals through physics based on mode ─────────────────

    let targetAngular: Vec3 = v3.zero();
    let targetLinear:  Vec3 = v3.zero();
    let modeConfidence = 0;

    switch (mode) {
      case "2D_rotation": {
        // Finger controls Z only
        targetAngular = { x: 0, y: 0, z: condFingerVel };
        modeConfidence = finger.confidence;
        break;
      }
      case "3D_rotation": {
        // Wrist controls full 3D
        targetAngular = condWristOmega;
        modeConfidence = wrist.confidence;
        break;
      }
      case "translation": {
        targetLinear = condDragVel;
        modeConfidence = drag.confidence;
        break;
      }
      case "hybrid": {
        // Compose: finger Z + wrist XYZ
        // Priority: wrist drives XY plane, finger adds Z spin on top
        const blendW = wrist.confidence / (wrist.confidence + finger.confidence + 1e-9);
        targetAngular = {
          x: condWristOmega.x * blendW,
          y: condWristOmega.y * blendW,
          z: condWristOmega.z * blendW + condFingerVel * (1 - blendW),
        };
        modeConfidence = Math.sqrt(finger.confidence * wrist.confidence); // geometric mean
        break;
      }
      default:
        // idle: let friction drain velocity naturally (inertia)
        break;
    }

    // ─── Step 5: Elastic resistance ──────────────────────────────────────────

    const rx = elasticResistance(this.accumAngle.x, this.cfg.maxRotationX, this.cfg.elasticSoftZone);
    const ry = elasticResistance(this.accumAngle.y, this.cfg.maxRotationY, this.cfg.elasticSoftZone);

    targetAngular = {
      x: targetAngular.x * rx,
      y: targetAngular.y * ry,
      z: targetAngular.z,
    };

    // ─── Step 6: Physics integration ─────────────────────────────────────────

    this.physics.applyAngularForce(targetAngular);
    this.physics.applyLinearForce(targetLinear);

    const { deltaAngle, deltaPos } = this.physics.integrate(dt);

    // ─── Step 7: Snap magnet (Z axis, 2D mode) ───────────────────────────────

    const { correction, feedback: snapFeedback } = this.snap.update(
      this.accumAngle.z,
      this.physics.angularVel.z,
      dt,
      this.cfg.snapEnabled && (mode === "2D_rotation" || mode === "idle"),
    );

    const finalDeltaZ = deltaAngle.z + correction;
    const finalDelta: Vec3 = { x: deltaAngle.x, y: deltaAngle.y, z: finalDeltaZ };

    // ─── Step 8: Accumulate orientation quaternion ────────────────────────────

    const dLen = v3.len(finalDelta);
    if (dLen > 1e-9) {
      const Q_delta = Q.fromAxisAngle(finalDelta, dLen);
      // World-space: new rotation applied on the left
      this.orientation = Q.norm(Q.mul(Q_delta, this.orientation));
    }

    // Track Euler angles for elastic bounds
    this.accumAngle = v3.add(this.accumAngle, finalDelta);

    // ─── Step 9: Inertia state ───────────────────────────────────────────────

    const angSpeed = this.physics.speed;
    const linSpeed = v3.len(this.physics.linearVel);
    const inertiaStrength = clamp(Math.max(angSpeed / this.cfg.maxAngularSpeed,
                                           linSpeed / this.cfg.maxLinearSpeed), 0, 1);

    return {
      mode,
      rotation:    finalDelta,        // euler delta to apply this frame
      translation: deltaPos,
      inertia: {
        active:   mode === "idle" && inertiaStrength > 0.01,
        strength: inertiaStrength,
      },
      confidence:  modeConfidence,
      orientation: Q.clone(this.orientation),
      snapFeedback,
    };
  }

  /** Reset all state (hand lost, scene change, etc.). */
  reset(): void {
    this.fingerTracker.reset();
    this.wristTracker.reset();
    this.dragTracker.reset();
    this.disambiguator.reset();
    this.physics.reset();
    this.snap.reset();
    this.orientation = Q.identity();
    this.accumAngle  = v3.zero();
    this.lastTs = 0;
  }

  /** Override the accumulated orientation (e.g. restore from saved state). */
  setOrientation(q: Quat): void {
    this.orientation = Q.norm(q);
  }

  /** Read current accumulated orientation without processing a frame. */
  get currentOrientation(): Quat { return Q.clone(this.orientation); }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Integration Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Three.js-style adapter.
 * Call inside your animation loop after `gc.process()`.
 *
 * @example
 * const out = gc.process(input, performance.now());
 * applyToThreeObject(out, mesh);
 */
export function applyToThreeObject(
  out: GestureOutput,
  obj: {
    rotation: { x: number; y: number; z: number };
    position: { x: number; y: number; z: number };
  },
): void {
  obj.rotation.x += out.rotation.x;
  obj.rotation.y += out.rotation.y;
  obj.rotation.z += out.rotation.z;
  obj.position.x += out.translation.x;
  obj.position.y += out.translation.y;
  obj.position.z += out.translation.z;
}

/**
 * Visualize confidence scores as a debug string (e.g. for overlay HUD).
 */
export function debugConfidence(
  scores: GestureOutput,
  raw: { finger: number; wrist: number; translation: number },
): string {
  const bar = (v: number) => "█".repeat(Math.round(v * 10)).padEnd(10, "░");
  return [
    `mode: ${scores.mode.padEnd(12)} conf: ${scores.confidence.toFixed(2)}`,
    `finger: ${bar(raw.finger)} ${raw.finger.toFixed(2)}`,
    `wrist:  ${bar(raw.wrist)} ${raw.wrist.toFixed(2)}`,
    `transl: ${bar(raw.translation)} ${raw.translation.toFixed(2)}`,
    scores.inertia.active ? `inertia: coasting (${(scores.inertia.strength * 100).toFixed(0)}%)` : "",
    scores.snapFeedback ? `SNAP → z=${(scores.snapFeedback.targetAngle * 180 / Math.PI).toFixed(0)}°` : "",
  ].filter(Boolean).join("\n");
}
