# Notebook Review Framework — v1

**We will review each notebook as both executable software and a learning experience.** The central question is:

> **Can the intended learner complete the promised activity, understand what happened, and draw appropriate conclusions from the results?**

“Run all succeeds” is necessary, but it is not the entire acceptance test.

[`NOTEBOOK_SPEC.md`](https://github.com/kurtvalcorza/ml-worker/blob/main/integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md) will serve as the requirements baseline. It already defines notebooks as executable reference pipelines and guided technical tutorials; our framework will assess whether the implementation meaningfully delivers both—not merely whether the required sections are present.

## 1. Establish the review contract

Before judging a notebook, record its **revision, applicable specification version, profile, teaching mode, intended audience, prerequisites, supported runtime, and promised outcomes**.

This prevents us from applying the wrong expectations. An inference-only notebook should not be criticized for omitting training; a beginner-oriented guided notebook should be examined for assumptions that would be reasonable in an expert reference notebook.

The review covers the **complete learner-facing artifact**, not just changed code: markdown, code, forms and controls, outputs, visualizations, exercises, links, exported files, and failure messages.

## 2. Assess eight dimensions

| Dimension | What we assess | What would constitute a meaningful problem? |
|---|---|---|
| **1. Promise fulfillment** | Whether the title, introduction, objectives, and instructions accurately describe what the notebook actually executes and demonstrates. | A promised capability is absent, silently skipped, replaced by a fallback, or described more broadly than the implementation supports. |
| **2. Technical correctness** | Code behavior, dependencies, state management, validation, security, reproducibility, and artifact handling. | Hidden execution-order dependencies, stale results after reruns, unsafe handling of files or credentials, incorrect computations, or exports that cannot be reconstructed as claimed. |
| **3. Scientific and experimental validity** | Task definitions, data semantics, preprocessing, splits, baselines, comparisons, metrics, and conclusions. | Leakage, inappropriate metrics, unequal comparison conditions presented as equivalent, incorrect probability interpretation, or conclusions unsupported by the experiment. |
| **4. Learner orientation and progression** | Whether the stated audience can understand the purpose, starting point, sequence, and completion criteria without undocumented help. | Important prerequisites are concealed; concepts appear before explanation; infrastructure overwhelms the lesson; or the learner cannot tell what is essential versus optional. |
| **5. Explanations and result interpretation** | Whether prose explains the actual code and helps learners read its outputs correctly. | Unexplained tables, ambiguous labels, misleading simplifications, sample answers inconsistent with results, or instructions that assume a predetermined winning model. |
| **6. Meaningful learner activity** | Whether learners practice the stated objectives through prediction, modification, comparison, interpretation, or diagnosis. | The notebook promises learning outcomes but only asks learners to run cells, copy numbers, or repeat definitions without applying them. |
| **7. Interaction, pacing, and recovery** | Configuration controls, progress feedback, waiting time, output volume, rerun instructions, and recovery from expected failures. | A control does not affect execution, long operations give no useful feedback, an exercise leaves the notebook in an inconsistent state, or an error offers no actionable recovery. |
| **8. Completion and transfer** | Whether learners finish with an identifiable result, an appropriate conclusion, and a practical route to reuse the capability. | No clear completion point; unclear output files; a promised user-data path does not work; or learners cannot identify what changes when applying the workflow elsewhere. |

These dimensions overlap deliberately where necessary. A misleading metric explanation, for example, is both a content problem and a threat to the validity of the learner’s conclusion.

**Learner-facing defects are not automatically lower priority than code defects.**

## 3. Make promise-to-evidence tracing the central test

For each material promise, trace:

**Claim → implementation → observable result → learner interpretation**

For each learning objective, also identify:

**Objective → learner activity → evidence that the objective was exercised**

The following are hypothetical review tests—not findings about a particular notebook:

| Stated promise or objective | Evidence we should require |
|---|---|
| **“Compare several models.”** | The named models genuinely execute under documented comparison conditions; failed or skipped runs are visible; learners receive guidance on interpreting the comparison. |
| **“Explain which classes the model confuses.”** | Class-level results are available and readable, with an activity that asks learners to identify and explain an error pattern—not merely display a confusion matrix. |
| **“Change a parameter and observe its effect.”** | The parameter reaches the relevant computation, the necessary cells are rerun, and the resulting output is not stale. An unchanged result is acceptable when the notebook helps learners investigate it. |
| **“Use your own data.”** | Input requirements are clear, representative compatible data reaches the promised downstream stages, and incompatible data receives an actionable rejection. |

A heading, function name, or printed success message is not sufficient evidence. Conversely, we should not demand a particular numerical result when legitimate variation is expected.

## 4. Review the notebook through distinct user journeys

We should examine more than one execution path.

**The first-time learner journey.** Read from the beginning using only the stated prerequisites. At each major transition, ask: *Do I know what to do, why I am doing it, what normal output looks like, and how to interpret it?* This is consistent with the specification’s pedagogical UX requirements.

**The clean default journey.** Verify the documented default workflow from a fresh supported runtime. Check that the final summary reflects what actually completed, and that exported outputs belong to the current run.

**The active-learning journey.** Perform a documented exercise or meaningful configuration change. Follow the learner-facing rerun instructions exactly and check that the experiment remains valid and understandable.

**The reuse and recovery journey.** Where applicable, exercise documented alternatives such as a fast path, user-supplied data, or artifact reload. Include at least one representative invalid input and examine the recovery guidance.

These are separate checks. A successful default run does not establish that an exercise, fast path, or user-data option works.

## 5. Keep findings separate from evidence gaps

Every conclusion should state its evidence basis.

| Evidence basis | What it establishes |
|---|---|
| **Source inspection** | What the code and content appear to implement; contradictions, missing logic, and identifiable risks. |
| **Documented execution evidence** | What a recorded run demonstrates for its particular revision, configuration, and environment. |
| **Direct execution** | What was observed during the review in the explicitly identified environment and configuration. |
| **Learner observation** | What representative learners actually understood, misunderstood, or struggled to complete. |
| **Not verified** | A claim or behavior for which the available evidence is insufficient. |

**Untested does not mean broken—but it also does not mean passed.** Local execution must not be presented as verified Colab execution, and saved outputs must not be treated as proof that the current revision runs cleanly. The specification expressly distinguishes static checks from execution evidence and requires revision/runtime information for manual verification.

Likewise, our instructional review can identify likely barriers and misleading explanations; it should not claim measured learning effectiveness without observing learners.

## 6. Classify findings by consequence

| Severity | Definition | Expected treatment |
|---|---|---|
| **Blocker** | Prevents the core workflow, creates a serious safety/security risk, or invalidates the central demonstration or conclusion. | Must be resolved before intended use or release. |
| **Major** | The notebook runs, but a core promise or learning objective is not delivered, or the intended learner is likely to become materially confused or misled. | Must be resolved for the intended audience and use. |
| **Minor** | Localized friction or imprecision that does not undermine the core workflow or learning outcome. | Fix when practical; document remaining limitations. |
| **Suggestion** | A useful enhancement without an established defect or unmet requirement. | Optional; do not present preference as a release requirement. |

Each finding should contain:

**Cell/section → observed issue → learner or technical consequence → supporting evidence → recommended correction → acceptance check**

Where relevant, attach the corresponding specification requirement ID. Keep **specification conformance** separate from **severity**: the spec treats an unresolved applicable `MUST` as a release failure, even when the finding is not the most serious learner-facing issue.

## 7. Produce a consistent review report

Each review should deliver:

1. **Review scope and evidence:** revision, audience, runtime, paths examined, and verification limitations.
2. **Separate judgments:** technical correctness, promise fulfillment, learner experience, and specification conformance.
3. **Prioritized findings:** concrete issues with corrections and acceptance checks.
4. **Readiness decision:** **Ready for intended use**, **Needs revision**, or **Verification pending**, with the specific reasons and remaining gates.

“Ready” should mean no unresolved blockers or major findings, required execution evidence is available, and applicable mandatory requirements are satisfied. Minor improvements may remain.

**We should not use an averaged numerical score.** Strong formatting or extensive documentation must not compensate for an invalid experiment, an unfulfilled promise, or a lesson that teaches the wrong conclusion.
