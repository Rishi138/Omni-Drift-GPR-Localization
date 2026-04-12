# Omni-Drift GPR Localization

A pre-computation localization correction pipeline for VEX robots.
Trains on a full sensor suite, models odometry error by movement type,
and bakes corrective biases directly into motion functions —
replacing complex match-time hardware with a single offline pipeline.

---

## Motivation

High-end VEX localization typically requires tracker pods, IMUs, and
distance sensors. The hardware cost alone is significant, but the deeper
cost is everything that comes with it. Every localization component must
be designed into the chassis from scratch — mounting points, structural
cutouts, wiring runs, sensor positioning — constraining the entire build
around it. Then it must be rebuilt for every new bot, retuned for every
new bot, redesigned when the chassis changes, maintained throughout the
season as components shift or fail, and continuously built around as the
rest of the bot evolves. Coefficients get retuned. Wiring gets redone.
Sensor positions get revisited. The localization suite is not a one-time
investment — it is a recurring burden that compounds across every bot,
every redesign, and every season.

The V5 brain adds a hard technical constraint on top of this. Its compute
limitations make real-time ML inference infeasible, it has no wireless
communication component, and VEX regulations prohibit external microchips.
Any correction system must operate entirely offline, before the match begins.

This pipeline addresses both problems simultaneously.

Rather than mounting a full localization suite on the competition bot,
a dedicated pipeline bot carries the hardware once, permanently. It trains
a GPR model for each movement type — curves, straight lines, swings, etc.
The resulting model produces a pre-computed corrective bias that is hardcoded
directly into the corresponding motion function on the competition bot.
At match time, the competition bot runs with zero localization hardware
and zero real-time ML. The correction is already in the code.

The tradeoff is explicit and accepted: fine-grained accuracy within a
movement segment is handled by PID, which runs on the brain in real time
as it always has. GPR targets the large, structured drift that accumulates
over full movement executions — the kind PID cannot correct because it has
no signal that the bot has drifted globally, only that current velocity is
off. These two systems are complementary, not competing. GPR corrects
accumulated positional error before the match. PID handles precision during it.

---

## Core Idea

The primary problem this pipeline solves is not odometry error in isolation —
it is the recurring mechanical, electrical, and tuning burden that standard
localization imposes on every bot every season. Odometry error correction is
what makes replacing that hardware possible, not the end goal in itself.

Standard odometry combined with a full localization suite means the
competition bot carries tracker pods, distance sensors, an IMU, and all
the wiring, mounting, and structural accommodation each requires. This
constrains chassis design, adds mass, and creates a maintenance surface
that persists for the entire season. The pipeline eliminates this entirely
from the competition bot.

    Standard approach  →  full localization suite on every competition bot,
                          rebuilt, retuned, and maintained every season

    This pipeline      →  full sensor suite on one dedicated pipeline bot,
                          trained once per movement type, reused indefinitely,
                          zero localization hardware on competition bot at match time

The pipeline bot is a one-time investment. Once built, it is never rebuilt.
The GPR model it produces for a given movement type is reused across bots
and across seasons with a single scalar coefficient adjustment — no
retraining, no new data collection, no kernel retuning. The trained model
transfers directly. One pipeline bot. One person to run it. Roughly one week
per movement type after the pipeline exists, running asynchronously to the
rest of the team. The competition bot is never touched for localization purposes.

A natural objection is that different bots have different error profiles —
different wheel sizes, different RPM ranges, different weight distributions —
so a model trained on one bot cannot transfer to another. This is addressed
directly in the architecture. The model is deliberately scoped to omni wheel
drift, which is physically universal across all bots running omni wheels.
The shape of that error — how it scales with RPM, heading, and velocity —
is the same across bots. Only its magnitude varies with wheel size and
bot configuration. A single scalar coefficient outside the GPR model accounts
for this magnitude difference. The GPR itself, and its kernel hyperparameters,
remain unchanged. This separation is not a convenience — it is physically
justified by the nature of omni drift, and the architecture is designed around it.

---

## Architecture

### 1. Physics Model — Omni Drift Correction

The first correction layer targets omni wheel drift specifically.
Omni wheel drift is the single largest and most generalizable source
of odometry error in VEX robots running omni wheels. It is physically
deterministic: the lateral slip of an omni wheel is a direct consequence
of its geometry and rotational velocity, not a stochastic or bot-specific
phenomenon. This determinism is what makes it modelable in a way that
transfers across bots.

The physics are straightforward. An omni wheel generates lateral slip
proportional to the component of velocity perpendicular to the wheel's
rolling direction. For a given wheel radius r and angular velocity ω,
the slip error scales predictably with RPM and heading. The model
captures this relationship directly.

**Deliberate scope:**
In any regression model, the model explains X% of output variance
and the remainder lives in the MSE as unexplained residual. This model
is explicitly scoped to the variance attributable to omni wheel drift.
Bot-specific nuances — motor variance, drivetrain asymmetry, weight
distribution effects — are not generalizable across bots and are not
claimed to be explained here. They live in the MSE by design.

This is not a weakness. A model that claims to explain everything
explains nothing precisely. By scoping to omni drift specifically —
physically motivated, deterministic, universal — it is possible to know
exactly when and why the model transfers, and exactly what the residual
contains. Precision of scope is what makes the model trustworthy
and the transfer claim defensible.

A possible objection is that error varies across field surfaces — different
tiles, different wear patterns, different friction characteristics between
venues produce different drift profiles, so a model trained on one field
cannot transfer reliably to another. This is true, and it is not a problem
the pipeline claims to solve. Field-surface variation is bot-and-environment-
specific, inconsistent across venues, and not physically generalizable in
the way omni drift is. It lives in the MSE by design, alongside motor variance
and other bot-specific nuances. The pipeline's transfer claim is scoped
exclusively to omni drift — the component of error that is physically
determined by wheel geometry and RPM regardless of surface. That claim
is defensible precisely because it does not overreach. A system that claimed
to also correct for field-surface variation would require retraining at every
venue, eliminating the reusability that makes the pipeline valuable in the
first place. By accepting field-surface error as unexplained variance, the
pipeline remains a one-time training investment that transfers everywhere.


**Tunable scaling coefficient:**
The physics model is general across bots because omni drift is general.
However, during training the pipeline bot carries a full sensor suite
that will not be present on the competition bot, changing its mass,
center of mass, and friction characteristics. A single scalar coefficient
k is introduced outside the GPR model:

    corrected_error = k · physics_model_output

k is tuned once per bot configuration to bridge the training environment
to the match environment. It absorbs the magnitude difference introduced
by wheel size, RPM range, and the weight delta from training hardware.
It does not alter the model's structure or its learned error shape —
it scales the output. This is why one coefficient is sufficient: the
shape of omni drift error is universal, only the magnitude varies,
and a scalar captures magnitude difference entirely.

---

### 2. Training Data Pipeline

Training runs on a clean field with no game elements. The pipeline bot
executes the target movement type repeatedly across varied parameter
ranges — different headings, velocities, and RPMs — building a dataset
that covers the input space the GPR will be queried over. No game
elements are present because the model is learning drivetrain error
characteristics, not environmental interactions. Field elements would
contaminate the training signal with error sources outside the model's
scope.

**Sensor suite — training phase only:**
- Tracker pods — high-frequency relative position via wheel odometry
- IMU — heading with drift
- 8 distance sensors — absolute position via wall triangulation

**Heading-ranked sensor selection:**
Modulus on the current heading deterministically ranks all 8 distance
sensors by geometric flatness to the nearest wall. The top 4 are selected
at each update step. Each selected reading is trig-corrected for its
small angular offset from true perpendicular:

    d_corrected = d_raw · cos(θ_offset)

The final position fix is a weighted average of the 4 corrected readings,
biased toward the flattest sensors. Best-4 selection is stable at all
headings — ranking shifts smoothly as heading changes with no discontinuous
jumps at transition boundaries.

**EKF state estimation:**
An Extended Kalman Filter fuses tracker pods, IMU, and distance sensor
triangulation into tight state estimates at each timestep. Standard Kalman
assumes linear dynamics, which is inappropriate for the nonlinear kinematics
of a VEX drivetrain. EKF linearizes around the current state estimate using
a Jacobian of the motion model, making it well-suited to the actual system.

Distance sensors are the critical input. Pure odometry and IMU both
accumulate drift with no absolute correction. Distance sensors provide
position anchors against field walls — each wall reading is a hard
absolute correction that resets accumulated drift regardless of how much
odometry has diverged. With three independent sensor streams including
absolute position anchors, the EKF state estimates are reliable enough
to treat as ground truth for training purposes.

The training signal for GPR is the residual between the physics model's
predicted position and the EKF's estimated position at each timestep.
This residual is clean: physics-subtracted, well-anchored, and sourced
from a sensor fusion pipeline with genuine absolute position references.
GPR learns from this signal.

All sensor hardware exists only during this phase. After training, it is
removed. It is never mounted on the competition bot.

---

### 3. GPR — Offline Error Modeling

After the physics model removes omni drift, the remaining residual contains
error from multiple sources. Not all of it is learnable or worth learning.
Bot-specific nuances that are inconsistent across runs live in the MSE
by the same deliberate scoping logic as the physics layer — accepted as
unexplained variance by design. GPR targets only the portion of the residual
that is consistent and repeatable for this movement type on this bot:
same input parameters, same error, reliably. That is the variance GPR
claims to explain. The rest is not claimed.

**Why the relationship is nonlinear:**

The relationship between movement parameters and odometry error is deeply
nonlinear, and this is not an edge case — it is the norm. Doubling movement
duration does not double drift, because drift accumulates as the integral
of slip over the trajectory, which depends nonlinearly on heading changes,
velocity profile, and motor torque curves throughout the movement. Doubling
RPM does not double lateral slip, because at higher RPMs inertia and wheel
deformation introduce additional nonlinear terms. A curve at velocity v
produces a fundamentally different error profile than the same curve at 2v
not because the geometry changed but because the dynamic interactions between
motors, wheels, and surface scale differently. The error function
e(heading, RPM, velocity) is not well-approximated by any low-degree
polynomial over the full operating range. GPR makes no assumption about
functional form and learns the true structure of the error surface directly
from data, which is why it is the right tool here.

**Uncertainty and why it matters:**

GPR does not produce a point prediction. It produces a full probability
distribution over predictions, characterized by a predicted mean correction
μ* and a prediction variance σ²*:

    μ* = K(x*, X) · K(XX)⁻¹ · y
    σ²* = K(x*, x*) - K(x*, X) · K(XX)⁻¹ · K(X, x*)

σ²* directly reflects how well the training data supports the prediction
at that input. In regions where training data is dense, σ²* is low and
the correction is trusted. In regions where data is sparse, σ²* is high
and the model is explicitly flagging that it has not seen enough of that
movement configuration to be confident.

This uncertainty is not just a validation mechanism — it is actionable
information that directly improves autonomous design. A high σ²* on a
given movement tells the programmer that movement is poorly characterized
and should not be relied on in a high-stakes autonomous. The options are
clear: collect more training data for that configuration, split the movement
into shorter segments that fall in well-characterized regions of the input
space, or redesign the autonomous to avoid that movement entirely in favor
of one GPR is confident about. Low σ²* across an entire autonomous path
is a quantitative guarantee that every movement in that path is well-modeled
and the corrections are trustworthy. This turns autonomous design from
intuition-driven iteration into a process with an explicit confidence signal
at every step.

**Kernel hyperparameters and transfer:**
The kernel hyperparameters — length scale, output scale, and noise variance —
are tuned once during training on the pipeline bot and then locked. They encode
the learned structure of omni drift error for that movement type: how quickly
error decorrelates across the input space, how large the error magnitudes are,
and how much noise is present in the training signal. These do not change when
the model transfers to a new bot. The error structure — the shape of how drift
scales with heading, RPM, and velocity — is physically the same across bots
because omni drift is universal. What changes between bots is magnitude,
handled entirely by k outside the GPR. Kernel hyperparameters capture shape.
k captures magnitude. Shape is universal. Magnitude is bot-specific.
The GPR transfers with zero retraining and zero kernel retuning.

**Inputs:**
Each training example and query is a vector of movement parameters:
heading, RPM, and velocity components. No sensor input is required at
query time. The model is queried entirely from motion function parameters.

**Output and deployment:**
For a given input vector, GPR outputs μ* and σ²*. If σ²* is within
acceptable bounds, μ* is used to construct a counter-bias hardcoded
into the motion function. The competition bot executes that function
with the correction already embedded. No sensors. No inference.
No additional compute on the brain.

---

## Why This Works

The architecture is built around two principles that reinforce each other:
**model only what is generalizable** and **deploy only what is necessary.**

Omni drift is physically universal — the physics model explains the
portion of error attributable to it, claims nothing beyond it, and
transfers across bots via a single scalar. The consistent, repeatable
residual within a movement type is learnable — GPR captures its shape
from sparse training data, locks its kernel hyperparameters, and transfers
without retraining. Everything else is accepted as unexplained variance
by deliberate design. The full sensor suite exists only to generate the
clean training signal that makes this possible. At match time, none of
it is present on the competition bot.

The V5 brain's compute limitations, lack of wireless capability, and VEX
legal restrictions on external hardware make real-time ML correction
impossible regardless. This pipeline treats those constraints as a design
requirement. The correction does not need to happen in real time because
it has already happened before the match begins.

The scaling argument is the strongest case for the pipeline. Manual PID
tuning by feel is repeated from scratch for every bot, every movement
type, every season. It does not transfer. It does not improve with reuse.
It scales with neither team size nor season length. This pipeline runs
once per movement type, produces a model that transfers across bots with
one coefficient, and costs nothing to reuse. One pipeline bot. One person
to run it. One week per movement type. The trained model then exists
permanently, improving in coverage as more movement types are added,
and transferring to every future bot the team builds.
