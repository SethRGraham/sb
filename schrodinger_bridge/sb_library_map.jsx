import React, { useState } from "react";

const COLORS = {
  bg: "#0a0e17",
  card: "#111827",
  cardHover: "#1a2236",
  border: "#1e293b",
  accent: "#f59e0b",
  accentDim: "#b45309",
  blue: "#3b82f6",
  green: "#10b981",
  purple: "#a78bfa",
  red: "#ef4444",
  pink: "#ec4899",
  text: "#e2e8f0",
  textDim: "#94a3b8",
  textMuted: "#64748b",
  codeBg: "#0d1117",
  codeText: "#c9d1d9",
  keyword: "#ff7b72",
  string: "#a5d6ff",
  func: "#d2a8ff",
  comment: "#8b949e",
};

function Code({ children }) {
  return (
    <code
      style={{
        background: COLORS.codeBg,
        color: COLORS.accent,
        padding: "2px 6px",
        borderRadius: 4,
        fontSize: 13,
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      }}
    >
      {children}
    </code>
  );
}

function CodeBlock({ code, highlight = [] }) {
  const lines = code.split("\n");
  return (
    <pre
      style={{
        background: COLORS.codeBg,
        border: `1px solid ${COLORS.border}`,
        borderRadius: 8,
        padding: "16px 20px",
        overflow: "auto",
        fontSize: 12.5,
        lineHeight: 1.6,
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        color: COLORS.codeText,
        margin: "12px 0",
      }}
    >
      {lines.map((line, i) => (
        <div
          key={i}
          style={{
            background: highlight.includes(i)
              ? "rgba(245, 158, 11, 0.08)"
              : "transparent",
            borderLeft: highlight.includes(i)
              ? `3px solid ${COLORS.accent}`
              : "3px solid transparent",
            paddingLeft: 12,
            marginLeft: -12,
          }}
        >
          {line}
        </div>
      ))}
    </pre>
  );
}

function MathBlock({ children, label }) {
  return (
    <div
      style={{
        background: "rgba(245, 158, 11, 0.06)",
        border: `1px solid rgba(245, 158, 11, 0.2)`,
        borderRadius: 8,
        padding: "16px 20px",
        margin: "12px 0",
        textAlign: "center",
      }}
    >
      {label && (
        <div
          style={{
            fontSize: 11,
            color: COLORS.accent,
            fontWeight: 600,
            letterSpacing: 1,
            textTransform: "uppercase",
            marginBottom: 8,
          }}
        >
          {label}
        </div>
      )}
      <div
        style={{
          fontSize: 17,
          color: COLORS.text,
          fontFamily: "'Georgia', serif",
          fontStyle: "italic",
          lineHeight: 1.8,
        }}
      >
        {children}
      </div>
    </div>
  );
}

function Takeaway({ children }) {
  return (
    <div
      style={{
        background: "rgba(16, 185, 129, 0.08)",
        border: `1px solid rgba(16, 185, 129, 0.25)`,
        borderRadius: 8,
        padding: "14px 18px",
        margin: "16px 0",
        display: "flex",
        gap: 12,
        alignItems: "flex-start",
      }}
    >
      <span style={{ fontSize: 18, flexShrink: 0 }}>🎯</span>
      <div style={{ fontSize: 14, color: COLORS.green, lineHeight: 1.6 }}>
        {children}
      </div>
    </div>
  );
}

function Arrow({ from, to }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        margin: "8px 0",
        padding: "6px 12px",
        fontSize: 13,
        color: COLORS.textDim,
      }}
    >
      <Code>{from}</Code>
      <span style={{ color: COLORS.accent }}>→</span>
      <span>{to}</span>
    </div>
  );
}

// ─── Section Components ───

function SectionProblem() {
  return (
    <div>
      <h2
        style={{
          color: COLORS.text,
          fontSize: 22,
          fontWeight: 700,
          marginBottom: 8,
        }}
      >
        Layer 1: Problem Definition
      </h2>
      <p style={{ color: COLORS.textDim, lineHeight: 1.7, marginBottom: 16 }}>
        Every SB computation begins with an <Code>SBProblem</Code>. This object
        encodes the <em>entire</em> optimization problem mathematically — you're
        handing the library three things, and they map one-to-one to the math.
      </p>

      <MathBlock label="The Schrödinger Bridge Problem">
        P* = argmin KL(P ‖ P_ref) &nbsp;&nbsp; subject to &nbsp;&nbsp; P₀ = μ₀
        , P₁ = μ₁
      </MathBlock>

      <p style={{ color: COLORS.textDim, lineHeight: 1.7, margin: "16px 0" }}>
        The three ingredients you provide, and what math they represent:
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: 12,
          margin: "16px 0",
        }}
      >
        {[
          {
            code: "source: MarginalDistribution",
            math: "μ₀",
            desc: "Where particles start. Defines sample() so you can draw x₀ ~ μ₀.",
            color: COLORS.blue,
          },
          {
            code: "target: MarginalDistribution",
            math: "μ₁",
            desc: "Where particles must arrive. Defines sample() so you can draw x₁ ~ μ₁.",
            color: COLORS.red,
          },
          {
            code: "reference: ReferenceDynamics",
            math: "P_ref (the SDE)",
            desc: "The 'prior' process. Defines drift b(x,t) and diffusion σ(x,t).",
            color: COLORS.purple,
          },
        ].map((item) => (
          <div
            key={item.math}
            style={{
              background: COLORS.card,
              border: `1px solid ${COLORS.border}`,
              borderTop: `3px solid ${item.color}`,
              borderRadius: 8,
              padding: 16,
            }}
          >
            <div
              style={{
                fontFamily: "Georgia, serif",
                fontSize: 20,
                color: item.color,
                marginBottom: 6,
                fontStyle: "italic",
              }}
            >
              {item.math}
            </div>
            <div
              style={{
                fontSize: 12,
                color: COLORS.accent,
                fontFamily: "monospace",
                marginBottom: 8,
              }}
            >
              {item.code}
            </div>
            <div style={{ fontSize: 13, color: COLORS.textDim, lineHeight: 1.5 }}>
              {item.desc}
            </div>
          </div>
        ))}
      </div>

      <CodeBlock
        code={`# This IS the math, written in Python.
problem = SBProblem(
    reference=BrownianMotion(sigma=0.5, dim=2),   # P_ref: dX = σ dW
    source=GaussianDistribution(mean=[-2, 0]),     # μ₀
    target=TwoMoonsDistribution(noise=0.05),       # μ₁
    time_grid=TimeGrid(t0=0, t1=1, num_steps=100), # discretization
)`}
        highlight={[1, 2, 3, 4]}
      />

      <p style={{ color: COLORS.textDim, lineHeight: 1.7, marginTop: 16 }}>
        The <Code>ReferenceDynamics</Code> base class is abstract. Subclasses
        like <Code>BrownianMotion</Code> and <Code>OrnsteinUhlenbeck</Code>{" "}
        implement the SDE coefficients:{" "}
      </p>

      <MathBlock label="Reference SDE">
        dX_t = b(X_t, t) dt + σ(X_t, t) dW_t
      </MathBlock>

      <div style={{ margin: "12px 0" }}>
        <Arrow from="drift(x, t)" to="Returns b(x,t) — the deterministic pull" />
        <Arrow from="diffusion(x, t)" to="Returns σ(x,t) — the noise magnitude" />
        <Arrow
          from="sample_source(key, n)"
          to="Draws n samples from μ₀"
        />
        <Arrow
          from="sample_target(key, n)"
          to="Draws n samples from μ₁"
        />
      </div>

      <Takeaway>
        <strong>Math takeaway:</strong> The SBProblem doesn't solve anything — it
        just <em>states</em> the variational problem. Think of it like writing
        down "minimize f(x) subject to g(x)=0" on a whiteboard. The solver
        classes are what actually do the minimization.
      </Takeaway>
    </div>
  );
}

function SectionAbstractSolver() {
  return (
    <div>
      <h2
        style={{
          color: COLORS.text,
          fontSize: 22,
          fontWeight: 700,
          marginBottom: 8,
        }}
      >
        Layer 2: The Abstract Solver — <Code>SBSolver</Code>
      </h2>
      <p style={{ color: COLORS.textDim, lineHeight: 1.7, marginBottom: 16 }}>
        <Code>SBSolver</Code> is the abstract base class that every solver
        inherits from. It enforces a contract: "If you want to solve an SB, you
        must implement these methods." The base class then provides a complete
        training loop that <em>calls</em> those methods.
      </p>

      <h3
        style={{ color: COLORS.accent, fontSize: 16, fontWeight: 600, marginTop: 20 }}
      >
        The 3 Abstract Methods (What Subclasses Must Provide)
      </h3>

      <div style={{ margin: "12px 0" }}>
        {[
          {
            method: "init_params(key) → Params",
            math: "Initialize θ₀ (random network weights)",
            why: "Every solver parameterizes its solution differently — scores, controls, potentials — so each needs its own initialization.",
          },
          {
            method: "train_step(key, params, opt_state, batch_size) → (params, opt_state, metrics)",
            math: "θ_{k+1} = θ_k − η ∇_θ L(θ_k)",
            why: "This is ONE gradient step. The loss L differs per solver. ScoreBased minimizes score-matching loss; FBSDE minimizes a BSDE terminal+running cost.",
          },
          {
            method: "extract_drift(params) → DriftFn",
            math: "b*(x,t) = b_ref(x,t) + σ²(t) · (learned term)",
            why: "After training, converts learned parameters into the actual SDE drift you can simulate. This is how math becomes trajectories.",
          },
        ].map((item, i) => (
          <div
            key={i}
            style={{
              background: COLORS.card,
              border: `1px solid ${COLORS.border}`,
              borderRadius: 8,
              padding: 16,
              marginBottom: 10,
            }}
          >
            <div
              style={{
                fontFamily: "monospace",
                fontSize: 13,
                color: COLORS.accent,
                marginBottom: 6,
              }}
            >
              {item.method}
            </div>
            <div
              style={{
                fontFamily: "Georgia, serif",
                fontSize: 14,
                color: COLORS.blue,
                fontStyle: "italic",
                marginBottom: 6,
              }}
            >
              {item.math}
            </div>
            <div style={{ fontSize: 13, color: COLORS.textDim, lineHeight: 1.5 }}>
              {item.why}
            </div>
          </div>
        ))}
      </div>

      <h3
        style={{ color: COLORS.accent, fontSize: 16, fontWeight: 600, marginTop: 24 }}
      >
        The Concrete Training Loop (Provided by Base Class)
      </h3>
      <p style={{ color: COLORS.textDim, lineHeight: 1.7, marginBottom: 12 }}>
        The <Code>train()</Code> method is <em>not</em> abstract — it's provided
        for free. Every solver gets the same outer loop. The magic is that
        <Code>train_step</Code> is polymorphic: each solver plugs in its own
        loss function.
      </p>

      <CodeBlock
        code={`def train(self, key, training_config, callback):
    config = training_config or TrainingConfig()

    # ① Initialize parameters (ABSTRACT — solver-specific)
    params = self.init_params(k1)
    opt_state = self._init_optimizer(params)

    for step in range(config.num_iterations):
        # ② One gradient step (ABSTRACT — solver-specific)
        params, opt_state, metrics = self.train_step(
            step_key, params, opt_state, config.batch_size
        )

        loss = metrics['loss']
        loss_history.append(loss)

        # ③ Early stopping
        if loss < best_loss - config.min_delta:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.patience:
            break

    # ④ Post-training diagnostics
    self._params = params
    diagnostics = self._run_diagnostics(key, params)
    return SolverResult(params, loss_history, diagnostics, ...)`}
        highlight={[3, 4, 9, 10, 11, 26, 27]}
      />

      <p style={{ color: COLORS.textDim, lineHeight: 1.7, marginTop: 16 }}>
        Notice the numbered steps — here's the flow:
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "auto 1fr",
          gap: "6px 16px",
          margin: "12px 0",
          fontSize: 14,
        }}
      >
        {[
          ["①", "init_params → randomly initialize neural net weights θ₀"],
          ["②", "train_step → sample data, compute loss, backprop, update θ"],
          ["③", "Early stopping — standard patience-based convergence check"],
          ["④", "Diagnostics — sample 500 trajectories, check MMD to μ₀ and μ₁"],
        ].map(([num, desc]) => (
          <>
            <div
              key={num}
              style={{ color: COLORS.accent, fontWeight: 700, textAlign: "right" }}
            >
              {num}
            </div>
            <div key={num + "d"} style={{ color: COLORS.textDim }}>
              {desc}
            </div>
          </>
        ))}
      </div>

      <Takeaway>
        <strong>Math takeaway:</strong> The base class implements a standard
        optimization loop: θ* = argmin_θ L(θ). Each solver defines a different
        loss L(θ). That's the <em>only</em> difference between solvers — the loss
        function and what θ represents.
      </Takeaway>
    </div>
  );
}

function SectionRepresentations() {
  return (
    <div>
      <h2
        style={{
          color: COLORS.text,
          fontSize: 22,
          fontWeight: 700,
          marginBottom: 8,
        }}
      >
        The Representation Classes — Different Ways to Encode the Solution
      </h2>
      <p style={{ color: COLORS.textDim, lineHeight: 1.7, marginBottom: 16 }}>
        All solvers learn the same thing — the optimal drift b*(x,t) — but they
        parameterize it differently. The <Code>Representation</Code> classes
        encode this choice. Each has a <Code>to_drift()</Code> method that
        converts back to the universal drift form.
      </p>

      <MathBlock label="The Universal Drift Formula (All Solvers Converge Here)">
        b*(x, t) = b_ref(x, t) + σ²(t) · ∇ log h(x, t)
      </MathBlock>

      <div style={{ margin: "16px 0" }}>
        {[
          {
            name: "ScoreRepresentation",
            learns: "∇ log p_t(x)  (the score function)",
            formula: "b* = b_ref + σ² · score_network(x, t)",
            solver: "ScoreBasedSolver",
            color: COLORS.blue,
          },
          {
            name: "ControlRepresentation",
            learns: "Z(x, t)  (the optimal control)",
            formula: "b* = b_ref + σ² · Z_network(x, t)",
            solver: "FBSDESolver",
            color: COLORS.green,
          },
          {
            name: "PotentialRepresentation",
            learns: "∇ log ψ(x, t)  (Schrödinger potential gradient)",
            formula: "b* = b_ref + σ² · ∇ log ψ(x, t)",
            solver: "IPFSolver",
            color: COLORS.purple,
          },
        ].map((r) => (
          <div
            key={r.name}
            style={{
              background: COLORS.card,
              border: `1px solid ${COLORS.border}`,
              borderLeft: `4px solid ${r.color}`,
              borderRadius: 8,
              padding: "14px 18px",
              marginBottom: 10,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontFamily: "monospace", color: r.color, fontSize: 14, fontWeight: 600 }}>
                {r.name}
              </span>
              <span
                style={{
                  fontSize: 11,
                  color: COLORS.textMuted,
                  background: COLORS.codeBg,
                  padding: "2px 8px",
                  borderRadius: 4,
                }}
              >
                used by {r.solver}
              </span>
            </div>
            <div
              style={{
                fontSize: 13,
                color: COLORS.textDim,
                margin: "6px 0",
              }}
            >
              Learns: <em style={{ color: COLORS.text }}>{r.learns}</em>
            </div>
            <div
              style={{
                fontFamily: "Georgia, serif",
                fontSize: 14,
                color: r.color,
                fontStyle: "italic",
              }}
            >
              {r.formula}
            </div>
          </div>
        ))}
      </div>

      <Takeaway>
        <strong>Math takeaway:</strong> Score, control, and potential are all
        different names for the same correction term ∇ log h(x,t). It's like
        velocity vs. momentum vs. kinetic energy — different representations of
        the same underlying physics. The <Code>to_drift()</Code> method is the
        universal converter.
      </Takeaway>
    </div>
  );
}

function SectionScoreSolver() {
  return (
    <div>
      <h2
        style={{
          color: COLORS.text,
          fontSize: 22,
          fontWeight: 700,
          marginBottom: 8,
        }}
      >
        Deep Dive: ScoreBasedSolver — The Training Step
      </h2>
      <p style={{ color: COLORS.textDim, lineHeight: 1.7, marginBottom: 16 }}>
        Let's trace exactly what happens in one call to{" "}
        <Code>train_step()</Code> for the score-based solver. This is where the
        math meets the gradient.
      </p>

      <h3 style={{ color: COLORS.accent, fontSize: 16, fontWeight: 600 }}>
        Step-by-step: One Training Iteration
      </h3>

      <div style={{ position: "relative", marginLeft: 20, borderLeft: `2px solid ${COLORS.border}`, paddingLeft: 24, marginTop: 16 }}>
        {[
          {
            step: "1. Sample data",
            math: "x₀ ~ μ₀,  x₁ ~ μ₁",
            code: `x0 = self.problem.sample_source(k1, batch_size)
x1 = self.problem.sample_target(k2, batch_size)`,
            detail: "Draw independent batches from source and target. These are unpaired — the solver will pair them via the bridge.",
          },
          {
            step: "2. (Optional) OT coupling",
            math: "Reorder x₁ so (x₀ⁱ, x₁ⁱ) are paired via optimal transport",
            code: `coupling = self._compute_ot_coupling(x0, x1, reg)
x1 = x1[coupling]  # reorder`,
            detail: "Sinkhorn algorithm finds the coupling that minimizes total squared distance. Better pairing → faster convergence.",
          },
          {
            step: "3. Sample random times",
            math: "t ~ Uniform(0.01, 0.99)",
            code: `t = jax.random.uniform(k1, (batch_size,),
        minval=0.01, maxval=0.99)`,
            detail: "We avoid t=0 and t=1 exactly because the bridge variance σ²t(1−t) → 0 at endpoints, causing numerical blow-up.",
          },
          {
            step: "4. Sample bridge points",
            math: "x_t = (1−t)x₀ + t·x₁ + σ√(t(1−t))·z,   z ~ N(0, I)",
            code: `bridge_mean = (1 - t) * x0 + t * x1
bridge_std = sigma * sqrt(t * (1 - t))
x_t = bridge_mean + bridge_std * z`,
            detail: "This is a Brownian bridge interpolation. At t=0 it equals x₀, at t=1 it equals x₁, and in between it adds noise proportional to σ²t(1−t).",
          },
          {
            step: "5. Compute true score (the target we train against)",
            math: "∇ log p(x_t | x₀, x₁) = −(x_t − μ_t) / σ²_t",
            code: `true_score = -(x_t - bridge_mean) / bridge_var`,
            detail: 'The conditional score of a Gaussian is always -(x - mean)/variance. This is the "teacher signal" — we know the exact answer because the bridge is Gaussian.',
          },
          {
            step: "6. Forward pass through network",
            math: "s_θ(x_t, t) ≈ ∇ log p_t(x_t)",
            code: `pred_score = self._factory.forward(params, x_t, t)`,
            detail: "The neural network takes (x_t, t) and predicts the score. The factory pattern means this could be an MLP, U-Net, Transformer, etc.",
          },
          {
            step: "7. Compute loss",
            math: "L = E[ σ²_t · ‖s_θ(x_t, t) − ∇ log p(x_t|x₀,x₁)‖² ]",
            code: `diff = pred_score - true_score
weights = bridge_std ** 2  # σ²t(1-t)
loss = mean(weights * diff ** 2)`,
            detail: "Denoising score matching: compare predicted vs. true conditional score. The σ² weighting stabilizes training across noise levels (analogous to the noise-conditioning in diffusion models).",
          },
          {
            step: "8. Backprop + update",
            math: "θ ← θ − η ∇_θ L(θ)",
            code: `(loss, metrics), grads = jax.value_and_grad(
    self._loss_fn, has_aux=True)(params, k3, x0, x1)
new_params, new_opt_state = adam_update(
    opt_state, grads, params, lr=lr)`,
            detail: "JAX's value_and_grad computes both the loss value and its gradient w.r.t. params in one pass (reverse-mode AD). Adam applies the standard adaptive learning rate update.",
          },
          {
            step: "9. EMA update",
            math: "θ_EMA ← α · θ_EMA + (1−α) · θ",
            code: `self._ema_params = decay * ema + (1-decay) * new`,
            detail: "Exponential Moving Average of parameters. At inference time, EMA params give smoother, more stable predictions than the raw training params. Same trick used in diffusion models.",
          },
        ].map((s, i) => (
          <div key={i} style={{ marginBottom: 24, position: "relative" }}>
            <div
              style={{
                position: "absolute",
                left: -33,
                top: 4,
                width: 12,
                height: 12,
                borderRadius: "50%",
                background: COLORS.accent,
              }}
            />
            <div
              style={{
                fontSize: 15,
                fontWeight: 700,
                color: COLORS.text,
                marginBottom: 4,
              }}
            >
              {s.step}
            </div>
            <div
              style={{
                fontFamily: "Georgia, serif",
                fontSize: 14,
                color: COLORS.blue,
                fontStyle: "italic",
                margin: "4px 0 8px",
              }}
            >
              {s.math}
            </div>
            <CodeBlock code={s.code} />
            <div
              style={{
                fontSize: 13,
                color: COLORS.textDim,
                lineHeight: 1.6,
                marginTop: 4,
              }}
            >
              {s.detail}
            </div>
          </div>
        ))}
      </div>

      <Takeaway>
        <strong>Math takeaway:</strong> The entire training loop is denoising
        score matching — you corrupt data with known noise (the Brownian bridge),
        compute the exact score of that corruption analytically, then train a
        network to predict it. After training, the network can predict the score
        at <em>any</em> point, not just bridge points — that's the
        generalization.
      </Takeaway>
    </div>
  );
}

function SectionFBSDE() {
  return (
    <div>
      <h2
        style={{
          color: COLORS.text,
          fontSize: 22,
          fontWeight: 700,
          marginBottom: 8,
        }}
      >
        Deep Dive: FBSDESolver — The Control Formulation
      </h2>
      <p style={{ color: COLORS.textDim, lineHeight: 1.7, marginBottom: 16 }}>
        The FBSDE solver views the SB as a stochastic optimal control problem.
        Instead of learning a score, it learns a <em>control</em> Z(x,t) that
        steers particles toward the target.
      </p>

      <MathBlock label="Forward-Backward SDE System">
        <div>Forward: dX_t = [b_ref + σ² Z_t] dt + σ dW_t</div>
        <div style={{ marginTop: 4 }}>
          Backward: dY_t = −½‖Z_t‖² dt + Z_t · dW_t, &nbsp; Y_T = g(X_T)
        </div>
      </MathBlock>

      <p style={{ color: COLORS.textDim, lineHeight: 1.7, margin: "16px 0" }}>
        The solver uses <strong>two networks</strong>:
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, margin: "12px 0" }}>
        <div
          style={{
            background: COLORS.card,
            border: `1px solid ${COLORS.border}`,
            borderTop: `3px solid ${COLORS.green}`,
            borderRadius: 8,
            padding: 16,
          }}
        >
          <div style={{ fontFamily: "monospace", fontSize: 13, color: COLORS.green }}>
            Z network (control)
          </div>
          <div style={{ fontSize: 13, color: COLORS.textDim, marginTop: 6, lineHeight: 1.5 }}>
            <Code>_z_factory</Code>: maps (x, t) → ℝᵈ. This is the control signal 
            that steers the forward SDE. After training, this <em>is</em> the 
            drift correction.
          </div>
        </div>
        <div
          style={{
            background: COLORS.card,
            border: `1px solid ${COLORS.border}`,
            borderTop: `3px solid ${COLORS.purple}`,
            borderRadius: 8,
            padding: 16,
          }}
        >
          <div style={{ fontFamily: "monospace", fontSize: 13, color: COLORS.purple }}>
            Y network (value function)
          </div>
          <div style={{ fontSize: 13, color: COLORS.textDim, marginTop: 6, lineHeight: 1.5 }}>
            <Code>_y_factory</Code>: maps (x, t) → ℝ. This tracks the "cost-to-go" 
            from the current state. Used in the BSDE consistency loss but not
            at inference time.
          </div>
        </div>
      </div>

      <h3 style={{ color: COLORS.accent, fontSize: 16, fontWeight: 600, marginTop: 20 }}>
        The Deep BSDE Loss (What It Optimizes)
      </h3>

      <p style={{ color: COLORS.textDim, lineHeight: 1.7, marginBottom: 12 }}>
        The loss has three terms that enforce different aspects of the FBSDE
        solution:
      </p>

      <CodeBlock
        code={`# 1. Terminal matching: does Y propagated via BSDE equal g(X_T)?
Y_0 = self._y_fn(params, x0, t=0)
Y_current = Y_0
for i in range(num_steps):
    f_t = 0.5 * ||Z_t||²       # running cost
    Y_current = Y - f_t*dt + Z_t · dW_t   # BSDE step
terminal_loss = mean((Y_T - g(X_T))²)

# 2. Endpoint loss: did particles reach the target?
endpoint_loss = mean(min_j ||X_T^i - x1^j||²)

# 3. Control regularization: don't use too much energy
control_cost = mean(||Z||²)

loss = w_terminal * terminal_loss + w_running * control_cost + endpoint_loss`}
        highlight={[6, 9, 12, 14]}
      />

      <Takeaway>
        <strong>Math takeaway:</strong> The FBSDE solver is solving a stochastic
        optimal control problem: find the minimum-energy control Z(x,t) that
        steers Brownian particles from μ₀ to μ₁. The terminal cost penalizes
        missing the target; the running cost penalizes excessive control effort.
        The optimal balance gives you the Schrödinger Bridge.
      </Takeaway>
    </div>
  );
}

function SectionIPF() {
  return (
    <div>
      <h2 style={{ color: COLORS.text, fontSize: 22, fontWeight: 700, marginBottom: 8 }}>
        Deep Dive: IPF — The Potential / Sinkhorn View
      </h2>
      <p style={{ color: COLORS.textDim, lineHeight: 1.7, marginBottom: 12 }}>
        The IPF solver (iterative proportional fitting) alternates between
        updating Schrödinger potentials that match the marginals. It's the
        classic Schrödinger solution expressed as log-potentials.
      </p>

      <CodeBlock
        code={`# IPF high-level loop
for k in range(K):
    psi = update_psi(phi)
    phi = update_phi(psi)
# after convergence psi, phi define the bridge via h(x,t) = exp(psi+phi)`}
      />

      <Takeaway>
        <strong>Math takeaway:</strong> IPF finds the pair of potentials
        (φ, ψ) whose entropic coupling has the prescribed marginals. The
        potentials convert to a drift via ∇ log ψ.
      </Takeaway>
    </div>
  );
}

function SectionIMF() {
  return (
    <div>
      <h2 style={{ color: COLORS.text, fontSize: 22, fontWeight: 700, marginBottom: 8 }}>
        Deep Dive: IMF — Iterative Mean-Field
      </h2>
      <p style={{ color: COLORS.textDim, lineHeight: 1.7, marginBottom: 12 }}>
        IMF methods cast the bridge as a mean-field fixed point and update
        densities or control maps iteratively to reach consistency.
      </p>

      <CodeBlock
        code={`# IMF pseudocode
repeat:
    compute density ρ_t from current drift
    update control / potential to reduce KL(ρ_t || target)
until converged`}
      />

      <Takeaway>
        <strong>Math takeaway:</strong> IMF blends ideas from IPF and control,
        iterating on mean-field consistency rather than explicit potentials.
      </Takeaway>
    </div>
  );
}

function SectionDoob() {
  return (
    <div>
      <h2 style={{ color: COLORS.text, fontSize: 22, fontWeight: 700, marginBottom: 8 }}>
        Deep Dive: Doob Transform — The Analytical Bridge
      </h2>
      <p style={{ color: COLORS.textDim, lineHeight: 1.7, marginBottom: 12 }}>
        The Doob solver constructs the bridge when the Doob h-transform is
        available analytically or via eigenfunction approximations.
      </p>

      <CodeBlock
        code={`# Doob form
b*(x,t) = b_ref(x,t) + σ² ∇ log h(x,t)
# where h solves the backward Kolmogorov / Schrödinger PDE`}
      />

      <Takeaway>
        <strong>Math takeaway:</strong> Doob gives the exact transformed drift
        when the potential h is known or approximated — it's the gold standard.
      </Takeaway>
    </div>
  );
}

function SectionRKHS() {
  return (
    <div>
      <h2 style={{ color: COLORS.text, fontSize: 22, fontWeight: 700, marginBottom: 8 }}>
        Deep Dive: RKHS Solvers — Kernelized Potential Estimation
      </h2>
      <p style={{ color: COLORS.textDim, lineHeight: 1.7, marginBottom: 12 }}>
        RKHS-based solvers estimate potentials in a reproducing kernel Hilbert
        space, enabling closed-form updates for small-to-medium problems.
      </p>

      <CodeBlock
        code={`# RKHS potential solve (schematic)
K = kernel_matrix(X, X)
alpha = solve(K + λI, targets)
psi(x) = sum_i alpha_i k(x, x_i)`}
      />

      <Takeaway>
        <strong>Math takeaway:</strong> RKHS methods trade scalability for
        analytical convenience — useful for diagnostics and small experiments.
      </Takeaway>
    </div>
  );
}

function SectionMirrorIPF() {
  return (
    <div>
      <h2 style={{ color: COLORS.text, fontSize: 22, fontWeight: 700, marginBottom: 8 }}>
        Deep Dive: Mirror-Descent IPF
      </h2>
      <p style={{ color: COLORS.textDim, lineHeight: 1.7, marginBottom: 12 }}>
        Mirror-descent variants accelerate IPF by taking proximal / mirror
        steps in potential space, improving convergence for noisy / high-dim data.
      </p>

      <CodeBlock
        code={`# Mirror-descent IPF step
psi_next = mirror_step(psi, grad_KL)
phi_next = mirror_step(phi, grad_KL)`}
      />

      <Takeaway>
        <strong>Math takeaway:</strong> Replace simple coordinate updates with
        geometry-aware mirror steps to stabilize and speed up IPF updates.
      </Takeaway>
    </div>
  );
}

// Simple error boundary to prevent whole app from crashing on render errors
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  componentDidCatch(error, info) {
    // could log to remote here
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 20, background: COLORS.card, borderRadius: 8 }}>
          <div style={{ color: COLORS.red, fontWeight: 700, marginBottom: 8 }}>Render error</div>
          <div style={{ color: COLORS.textDim, fontSize: 13 }}>{String(this.state.error)}</div>
        </div>
      );
    }
    return this.props.children;
  }
}

function SectionExtractDrift() {
  return (
    <div>
      <h2
        style={{
          color: COLORS.text,
          fontSize: 22,
          fontWeight: 700,
          marginBottom: 8,
        }}
      >
        Layer 3: From Trained Params → Trajectories
      </h2>
      <p style={{ color: COLORS.textDim, lineHeight: 1.7, marginBottom: 16 }}>
        After training, every solver calls <Code>extract_drift(params)</Code> to
        produce a drift function. Then the integrator simulates the SDE to
        produce actual particle trajectories.
      </p>

      <MathBlock label="What extract_drift() Returns">
        b*(x, t) = b_ref(x, t) + σ²(t) · network(params, x, t)
      </MathBlock>

      <div
        style={{
          background: COLORS.card,
          border: `1px solid ${COLORS.border}`,
          borderRadius: 12,
          padding: 20,
          margin: "20px 0",
        }}
      >
        <div
          style={{
            fontSize: 12,
            color: COLORS.textMuted,
            letterSpacing: 1,
            textTransform: "uppercase",
            marginBottom: 16,
          }}
        >
          Full Pipeline
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            flexWrap: "wrap",
            justifyContent: "center",
          }}
        >
          {[
            { label: "SBProblem", color: COLORS.purple, sub: "(defines μ₀, μ₁, P_ref)" },
            { label: "→" },
            { label: "Solver.train()", color: COLORS.blue, sub: "(minimizes L(θ))" },
            { label: "→" },
            { label: "extract_drift(θ*)", color: COLORS.green, sub: "(returns b*)" },
            { label: "→" },
            { label: "EulerMaruyama", color: COLORS.accent, sub: "(simulates SDE)" },
            { label: "→" },
            { label: "TrajectoryBatch", color: COLORS.pink, sub: "(paths + times)" },
          ].map((item, i) =>
            item.color ? (
              <div
                key={i}
                style={{
                  background: `${item.color}15`,
                  border: `1px solid ${item.color}40`,
                  borderRadius: 8,
                  padding: "10px 14px",
                  textAlign: "center",
                }}
              >
                <div
                  style={{
                    fontFamily: "monospace",
                    fontSize: 13,
                    color: item.color,
                    fontWeight: 600,
                  }}
                >
                  {item.label}
                </div>
                <div style={{ fontSize: 11, color: COLORS.textMuted, marginTop: 4 }}>
                  {item.sub}
                </div>
              </div>
            ) : (
              <span key={i} style={{ color: COLORS.textMuted, fontSize: 20 }}>
                {item.label}
              </span>
            )
          )}
        </div>
      </div>

      <h3 style={{ color: COLORS.accent, fontSize: 16, fontWeight: 600, marginTop: 20 }}>
        The Euler-Maruyama Integrator
      </h3>

      <MathBlock label="One Step of the Euler-Maruyama Discretization">
        {"X_{t+Δt} = X_t + b*(X_t, t)·Δt + σ(X_t, t)·√Δt · z,   z ~ N(0,I)"}
      </MathBlock>

      <p style={{ color: COLORS.textDim, lineHeight: 1.7, margin: "12px 0" }}>
        This is the stochastic equivalent of Euler's method for ODEs. Each step
        adds a deterministic push (drift × dt) and a random kick (diffusion ×
        √dt × noise). The <Code>sample()</Code> method does exactly this loop:
      </p>

      <CodeBlock
        code={`# SBSolver.sample() — uses extract_drift + integrator
def sample(self, key, num_samples, params):
    x0 = self.problem.sample_source(k1, num_samples)  # x₀ ~ μ₀
    drift = self.extract_drift(params)                 # b*(x,t)
    diffusion = self.problem.sigma                     # σ(x,t)
    return self.integrator.integrate(
        k2, x0, time_grid, drift, diffusion            # simulate!
    )`}
        highlight={[2, 3, 4, 5, 6]}
      />

      <Takeaway>
        <strong>Math takeaway:</strong> The full pipeline is: (1) define the
        variational problem, (2) solve it numerically via gradient descent to get
        θ*, (3) convert θ* into a drift function b*, (4) simulate the SDE
        dX = b* dt + σ dW using Euler-Maruyama. The output is a batch of
        stochastic trajectories that transform μ₀ into μ₁.
      </Takeaway>
    </div>
  );
}

function SectionDiagnostics() {
  return (
    <div>
      <h2
        style={{
          color: COLORS.text,
          fontSize: 22,
          fontWeight: 700,
          marginBottom: 8,
        }}
      >
        Diagnostics: How You Know It Worked
      </h2>
      <p style={{ color: COLORS.textDim, lineHeight: 1.7, marginBottom: 16 }}>
        After training, <Code>_run_diagnostics()</Code> checks four invariants
        that any valid SB solution must satisfy. These are your "unit tests" for
        the math.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {[
          {
            name: "Marginal Consistency",
            math: "MMD²(ρ₀, μ₀) ≈ 0  and  MMD²(ρ₁, μ₁) ≈ 0",
            what: "Do the endpoints of your trajectories match the source and target distributions? Measured by Maximum Mean Discrepancy — a kernel-based distance between distributions.",
            severity: "Most important check",
            color: COLORS.green,
          },
          {
            name: "Mass Conservation",
            math: "∫ ρ_t(x) dx = 1  for all t",
            what: "Particles shouldn't appear or vanish. Checked by tracking how the spread of particles changes — if the standard deviation explodes, particles are 'leaking'.",
            severity: "Catches integrator blow-up",
            color: COLORS.blue,
          },
          {
            name: "Path Regularity",
            math: "max |ΔX / Δt| < threshold",
            what: "Velocity shouldn't explode. If particles move too fast, the discretization is too coarse or the learned drift has blow-up regions.",
            severity: "Catches numerical instability",
            color: COLORS.accent,
          },
          {
            name: "Entropy Evolution",
            math: "H(ρ_t) should vary smoothly",
            what: "Entropy of the particle cloud at each time should change smoothly. Wild jumps indicate the solver found a degenerate solution.",
            severity: "Catches mode collapse",
            color: COLORS.purple,
          },
        ].map((d) => (
          <div
            key={d.name}
            style={{
              background: COLORS.card,
              border: `1px solid ${COLORS.border}`,
              borderTop: `3px solid ${d.color}`,
              borderRadius: 8,
              padding: 16,
            }}
          >
            <div style={{ fontWeight: 700, color: d.color, fontSize: 14, marginBottom: 4 }}>
              {d.name}
            </div>
            <div
              style={{
                fontFamily: "Georgia, serif",
                fontSize: 13,
                color: COLORS.text,
                fontStyle: "italic",
                marginBottom: 8,
              }}
            >
              {d.math}
            </div>
            <div style={{ fontSize: 13, color: COLORS.textDim, lineHeight: 1.5 }}>
              {d.what}
            </div>
            <div
              style={{
                fontSize: 11,
                color: COLORS.textMuted,
                marginTop: 8,
                fontStyle: "italic",
              }}
            >
              {d.severity}
            </div>
          </div>
        ))}
      </div>

      <Takeaway>
        <strong>Math takeaway:</strong> The MMD metric is key —
        MMD²(P, Q) = E[k(X,X')] + E[k(Y,Y')] − 2E[k(X,Y)] where k is a
        Gaussian kernel. It's zero if and only if P = Q. If your target MMD² is
        small (say &lt; 0.01), your bridge has successfully learned to transport
        μ₀ to μ₁.
      </Takeaway>
    </div>
  );
}

// ─── Main App ───

const SECTIONS = [
  { id: "problem", label: "1. Problem Definition", component: SectionProblem },
  { id: "solver", label: "2. Abstract Solver", component: SectionAbstractSolver },
  { id: "reps", label: "3. Representations", component: SectionRepresentations },
  { id: "score", label: "4. ScoreBased (deep dive)", component: SectionScoreSolver },
  { id: "fbsde", label: "5. FBSDE (deep dive)", component: SectionFBSDE },
  { id: "ipf", label: "5a. IPF (deep dive)", component: SectionIPF },
  { id: "mirror_ipf", label: "5b. Mirror-Descent IPF", component: SectionMirrorIPF },
  { id: "imf", label: "5c. IMF (deep dive)", component: SectionIMF },
  { id: "doob", label: "5d. Doob Transform", component: SectionDoob },
  { id: "rkhs", label: "5e. RKHS Solvers", component: SectionRKHS },
  { id: "drift", label: "6. Params → Trajectories", component: SectionExtractDrift },
  { id: "diag", label: "7. Diagnostics", component: SectionDiagnostics },
];

export default function SBLibraryMap() {
  const [activeSection, setActiveSection] = useState("problem");

  const ActiveComponent = SECTIONS.find(
    (s) => s.id === activeSection
  )?.component;

  return (
    <div
      style={{
        background: COLORS.bg,
        minHeight: "100vh",
        color: COLORS.text,
        fontFamily:
          "'Söhne', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "32px 32px 20px",
          borderBottom: `1px solid ${COLORS.border}`,
        }}
      >
        <h1
          style={{
            fontSize: 28,
            fontWeight: 800,
            color: COLORS.text,
            margin: 0,
            letterSpacing: -0.5,
          }}
        >
          <span style={{ color: COLORS.accent }}>SB Library</span> — Math ↔
          Code Map
        </h1>
        <p
          style={{
            color: COLORS.textDim,
            fontSize: 15,
            marginTop: 6,
            lineHeight: 1.5,
          }}
        >
          How the optimization problem, abstract classes, training loop, and
          diagnostics all connect.
        </p>
      </div>

      <div style={{ display: "flex" }}>
        {/* Nav */}
        <nav
          style={{
            width: 240,
            flexShrink: 0,
            borderRight: `1px solid ${COLORS.border}`,
            padding: "16px 0",
            position: "sticky",
            top: 0,
            height: "calc(100vh - 100px)",
            overflowY: "auto",
          }}
        >
          {SECTIONS.map((s) => (
            <button
              key={s.id}
              onClick={() => setActiveSection(s.id)}
              style={{
                display: "block",
                width: "100%",
                textAlign: "left",
                padding: "10px 20px",
                background:
                  activeSection === s.id
                    ? `${COLORS.accent}15`
                    : "transparent",
                border: "none",
                borderLeft:
                  activeSection === s.id
                    ? `3px solid ${COLORS.accent}`
                    : "3px solid transparent",
                color:
                  activeSection === s.id ? COLORS.accent : COLORS.textDim,
                fontSize: 13,
                fontWeight: activeSection === s.id ? 600 : 400,
                cursor: "pointer",
                transition: "all 0.15s",
                fontFamily: "inherit",
              }}
            >
              {s.label}
            </button>
          ))}
        </nav>

        {/* Content */}
        <main
          style={{
            flex: 1,
            padding: "28px 36px",
            maxWidth: 860,
            overflowY: "auto",
          }}
        >
          {ActiveComponent && (
            <ErrorBoundary>
              <ActiveComponent />
            </ErrorBoundary>
          )}
        </main>
      </div>
    </div>
  );
}
