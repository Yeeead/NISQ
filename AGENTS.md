# AGENTS.md

## 1. Project Goal

This project prioritizes correctness, clarity, reproducibility, and maintainability.

When modifying or generating code:

* Implement the requested behavior with the smallest reasonable change.
* Prefer straightforward, explicit code over generalized frameworks.
* Preserve the existing architecture unless the requested behavior cannot be implemented cleanly within it.
* Do not redesign the project while solving a local problem.
* Do not introduce abstractions for hypothetical future requirements.
* Keep the implementation easy to trace, debug, and verify.

The primary objective is to solve the current task correctly, not to make the project appear more architecturally sophisticated.

---

## 2. Instruction Priority

Follow instructions in this order:

1. The user's current request.
2. Correctness and experimental validity.
3. Existing project behavior and public interfaces.
4. This `AGENTS.md`.
5. Existing local coding style and conventions.

When instructions conflict:

* Preserve correctness first.
* Prefer the interpretation that changes less code.
* State any material assumption in the final response.
* Do not silently reinterpret the user's requested behavior.

---

## 3. Requirement Analysis

Before editing code, first inspect the relevant implementation and reduce the task to a concrete set of requirements.

Determine:

* The exact requested behavior.
* The files directly responsible for that behavior.
* Existing interfaces that must remain unchanged.
* Inputs, outputs, side effects, and data flow.
* Acceptance criteria.
* Explicit non-goals.
* The narrowest validation needed to confirm correctness.

For non-trivial tasks, organize the request into:

### Objective

One concise sentence describing the required outcome.

### Required Changes

A concrete list of behaviors that must be added, removed, or corrected.

### Constraints

Existing APIs, experiment protocols, file structures, command-line arguments, output formats, or compatibility requirements that must be preserved.

### Non-goals

Related changes that may appear useful but were not requested.

### Validation

The commands, tests, assertions, or smoke checks that demonstrate the task is complete.

Do not expand a narrow request into a broad redesign.

If a request contains ambiguity but a safe, local interpretation is available:

* State the assumption briefly.
* Proceed with the smallest reversible implementation.
* Do not block the task with unnecessary clarification.

Ask for clarification only when different interpretations would cause materially different behavior, data loss, incompatible APIs, or invalid experimental results.

---

## 4. Inspect Before Editing

Before writing new code:

1. Locate the current execution path.
2. Read the directly related files.
3. Search for existing implementations of similar behavior.
4. Identify current naming, configuration, logging, and testing conventions.
5. Reuse existing project mechanisms when they are adequate.

Do not create a new subsystem before confirming that the project does not already contain an appropriate implementation.

Do not infer architecture solely from filenames. Read the actual call chain.

For behavior reached through multiple files, trace the real path from entry point to output before modifying it.

---

## 5. Scope Control

Make the smallest change that fully satisfies the request.

Default behavior:

* Modify only files directly related to the task.
* Preserve unrelated code.
* Preserve existing directory structure.
* Preserve public function signatures.
* Preserve command-line interfaces.
* Preserve configuration names and defaults.
* Preserve output schemas, metric names, log keys, and checkpoint formats.
* Preserve behavior outside the requested scope.

The following are warning signs that the implementation may be exceeding scope:

* More than five source files must be modified for a local behavior change.
* A new top-level package or subsystem is introduced.
* Multiple public APIs are renamed.
* A new configuration framework is added.
* Existing working code is rewritten rather than locally adapted.
* Unrelated formatting or cleanup appears in the diff.
* The change requires explaining many architectural concepts unrelated to the request.

When one of these occurs, reassess whether a smaller solution exists.

If the broader change is truly necessary, explain why each additional file or abstraction is required.

---

## 6. Anti-Overengineering Rules

### 6.1 Default to Direct Implementation

Prefer:

* A direct conditional.
* A focused function.
* A small local helper.
* An explicit loop.
* Existing arguments and return values.
* Existing configuration objects.
* Existing module boundaries.

Do not replace simple logic with an extensible framework unless the task explicitly requires extensibility.

### 6.2 Do Not Add Speculative Abstractions

Do not introduce an abstraction because it might be useful later.

Abstractions must solve a current, demonstrated problem.

Do not add:

* Plugin systems for fixed implementations.
* Generic backends for a single backend.
* Strategy patterns for one or two simple branches.
* Registries for a fixed list already handled clearly.
* Factories that only call constructors.
* Managers that merely forward method calls.
* Service layers around local functions.
* Context objects that only group a few existing arguments.
* Wrappers that add no validation, transformation, or meaningful policy.
* Base classes with only one concrete subclass.
* Interfaces whose only purpose is to mirror an existing function.
* Generic configuration layers over existing configuration.
* Callback systems for direct sequential execution.
* Dependency injection containers for ordinary Python objects.
* Event buses for local function calls.
* Custom result objects when a current dictionary or tuple is already sufficient.
* New domain-specific languages or schema systems for simple parameters.

### 6.3 Names That Require Extra Justification

Do not create classes or modules with generic architectural names unless they represent a real, established responsibility:

* `Base*`
* `Abstract*`
* `Manager`
* `Service`
* `Handler`
* `Controller`
* `Coordinator`
* `Orchestrator`
* `Processor`
* `Engine`
* `Factory`
* `Registry`
* `Provider`
* `Adapter`
* `Wrapper`
* `Context`
* `Framework`

These names are not prohibited, but their use must be justified by actual state, lifecycle, polymorphism, or repeated behavior—not by a desire to organize a small amount of code.

### 6.4 Do Not Create Pass-through Layers

Avoid functions or methods that only:

* Forward all arguments unchanged.
* Rename another function without adding semantics.
* Read one configuration value and call another function.
* Return another function's result unchanged.
* Wrap one line solely to create another abstraction boundary.

A wrapper is justified only when it provides meaningful behavior such as:

* Validation.
* Normalization.
* Error translation.
* Resource ownership.
* Compatibility handling.
* Stable public API isolation.
* Repeated policy used by multiple call sites.

### 6.5 Do Not Generalize One-off Logic

One implementation is not a pattern.

Two superficially similar implementations are not automatically a pattern.

Generalize only when:

* The same meaningful logic already exists in multiple places.
* The shared behavior is stable and clearly defined.
* The abstraction reduces total complexity.
* Callers become easier to understand.
* The abstraction does not require many flags or mode switches.

If a proposed helper needs several booleans such as:

```python
run_task(
    use_new_mode=True,
    legacy_mode=False,
    special_case=True,
    skip_validation=False,
)
```

prefer separate explicit logic or a smaller helper instead.

---

## 7. Abstraction Thresholds

### Extract a Function When

Extract a function when at least one of the following is true:

* The logic is reused.
* The logic has a clear independent responsibility.
* The logic is difficult to understand inline.
* The logic can be tested independently.
* Extraction makes the main execution path easier to read.

Do not extract tiny expressions merely to reduce line count.

Do not split a readable function into many single-use helpers that force the reader to jump between files.

### Create a Class When

Create a class only when the behavior requires:

* Persistent related state.
* A meaningful lifecycle.
* Resource ownership.
* Multiple operations over the same invariant state.
* Genuine polymorphism already required by the project.
* Compatibility with an existing class-based project interface.

Do not create a class as a namespace for static functions.

Do not create a class only to hold configuration.

Do not create a class when a function with explicit parameters is clearer.

### Create a New Module When

Create a new module only when:

* The code represents a distinct cohesive responsibility.
* The existing file would otherwise become materially harder to navigate.
* The code is reused across modules.
* The project already separates that type of responsibility into modules.

Do not create a new module for a single small helper.

Do not create generic dumping grounds such as:

```text
common.py
helpers.py
misc.py
shared.py
utils2.py
```

Place logic near the domain that owns it.

### Create a Configuration Option When

Add a configuration option only when:

* The user requested selectable behavior.
* The value differs across legitimate experiment modes.
* The value cannot reasonably remain a local constant.
* Existing configuration mechanisms cannot express it.

Do not make every local constant configurable.

Do not add a configuration field merely to avoid making a concrete implementation decision.

---

## 8. Prefer Explicit Data Flow

Data flow should be visible from function parameters, local variables, and return values.

Prefer explicit:

```python
poisoned_images = apply_trigger(images, trigger, epsilon)
metrics = evaluate_model(model, poisoned_images, labels)
```

over hidden mutation or implicit global state.

Avoid:

* Hidden module-level mutable state.
* Runtime monkey patching.
* Implicit registration on import.
* Configuration discovered through unrelated environment variables.
* Functions whose output depends on undocumented global variables.
* Side effects inside property accessors.
* Import-time execution beyond definitions and constants.

Do not conceal important experiment behavior behind decorators, callbacks, or registries when direct calls are clearer.

---

## 9. Preserve Existing Interfaces

Unless explicitly requested or strictly necessary, do not change:

* Public function names.
* Function parameter order.
* Return types.
* Dictionary keys.
* Configuration field names.
* CLI flag names.
* Dataset formats.
* Checkpoint formats.
* Output file locations.
* Log message prefixes used by scripts.
* Metric names consumed by evaluation code.
* Import paths used elsewhere in the repository.

When an interface change is necessary:

1. Identify all callers.
2. Update them consistently.
3. Preserve backward compatibility when practical.
4. Explain the change in the final response.
5. Add or update validation for the changed interface.

Do not rename APIs merely for stylistic preference.

---

## 10. Refactoring Rules

Refactoring is allowed only when it directly supports:

* Correctness.
* Readability of the modified execution path.
* Removal of clear duplication encountered during the task.
* Testability required for the requested change.
* Elimination of a bug caused by the existing structure.

Do not perform broad refactoring as a side effect of a feature or bug fix.

Do not:

* Rewrite unrelated functions.
* Move files without necessity.
* Rename unrelated variables across the repository.
* Reformat entire files for a small change.
* Replace working libraries or frameworks.
* Convert procedural code to object-oriented code without need.
* Convert object-oriented code to functional code without need.
* Introduce a new architectural pattern for consistency alone.
* Modernize syntax unrelated to the task.
* Change all similar code when only one execution path is relevant.
* Remove apparently unused code without verifying all dynamic callers.

Keep refactoring local to the code being changed.

A refactor should reduce cognitive complexity, not merely redistribute it across more files.

---

## 11. Duplication Policy

Do not remove duplication mechanically.

Some duplication is preferable to a poorly defined abstraction.

Before consolidating duplicated code, verify:

* The duplicated behavior is semantically identical.
* It is expected to evolve together.
* The shared parameters have the same meaning.
* Error handling requirements are compatible.
* Combining the code does not introduce mode flags or branching complexity.

Prefer a small amount of obvious duplication over a generic helper with many conditionals.

Do not consolidate code merely because lines look similar.

---

## 12. Dependency Rules

Do not add new dependencies unless the requested behavior cannot reasonably be implemented with:

* The Python standard library.
* Existing project dependencies.
* Existing project utilities.

Before adding a dependency:

1. Verify that it is not already available.
2. Explain why existing tools are insufficient.
3. Use the smallest appropriate dependency.
4. Avoid adding overlapping packages.
5. Update dependency metadata consistently.
6. Validate imports inside the project environment.

Do not install packages globally.

Do not change package versions unrelated to the task.

Do not regenerate lockfiles unless dependency changes require it.

---

## 13. Error Handling

Handle errors at the layer that has enough context to act on them.

Prefer:

* Clear validation near external inputs.
* Specific exceptions.
* Error messages containing actionable context.
* Existing project error-handling conventions.

Avoid:

* Broad `except Exception` blocks without re-raising or meaningful handling.
* Silent fallback behavior.
* Returning `None` for unexpected failures when callers expect valid data.
* Suppressing warnings merely to make output look clean.
* Catching errors only to print them and continue with invalid state.
* Retry loops without bounded attempts and a clear reason.
* Replacing an error with a default value that changes experiment semantics.

Do not hide failures that may invalidate results.

---

## 14. Comments and Documentation

Comments should explain:

* Non-obvious reasoning.
* Mathematical meaning.
* Important invariants.
* Compatibility constraints.
* Why an unusual implementation is necessary.
* Assumptions that cannot be expressed through types or names.

Comments should not restate obvious code.

Avoid comments such as:

```python
# Increment i
i += 1
```

Prefer descriptive names and clear control flow.

Do not add large explanatory comment blocks to compensate for overly complex code. Simplify the code instead.

Update docstrings only when behavior, parameters, return values, or assumptions change.

Do not generate extensive documentation unrelated to the requested change.

---

## 15. Type Hints

Follow the existing project's type-hinting style.

Add type hints when they improve:

* Public API clarity.
* Non-obvious data structures.
* Tensor or array shape expectations.
* Optional values.
* Complex return values.

Do not add elaborate generic type hierarchies for local implementation details.

Do not introduce protocols, type variables, or custom generic containers unless they solve a real typing problem.

Do not rewrite an entire untyped module solely to type a small change.

---

## 16. Experimental Integrity

This is a research and experiment-oriented project. Experimental behavior must remain explicit and reproducible.

Unless requested, do not change:

* Dataset splits.
* Random seeds.
* Poisoning rates.
* Target labels.
* Training epochs.
* Batch sizes.
* Learning rates.
* Model initialization.
* Trigger constraints.
* Evaluation protocols.
* Defense thresholds.
* Resolution handling.
* Sampling or shot settings.
* Metric definitions.
* Logging field names.
* Checkpoint selection rules.

When adding a new experiment mode:

* Keep existing modes unchanged.
* Make the new mode explicit.
* Avoid changing defaults unless requested.
* Reuse the same data ordering when paired evaluations require it.
* Preserve labels and sample correspondence across evaluation variants.
* Keep incompatible tensor shapes in separate forward passes.
* Do not silently resize, normalize, clamp, detach, or cast data unless the protocol requires it.
* Document any transformation that changes the evaluated input.

Do not improve reported metrics by changing the evaluation protocol.

Do not weaken a baseline implementation to simplify comparison.

Do not share implementation state between methods when doing so changes their independent behavior.

---

## 17. Tensor and Numerical Code

For tensor operations:

* Preserve device placement.
* Preserve expected dtype.
* Preserve batch order.
* Preserve gradient flow unless detachment is intentional.
* Make shape transformations explicit.
* Validate assumptions about batch, channel, height, and width.
* Avoid unnecessary tensor copies.
* Avoid implicit broadcasting when it obscures semantics.
* Do not mix tensors from incompatible resolutions in one stacked tensor.
* Use separate forward passes when shapes differ.
* Keep paired samples in the same order across evaluations.

Before adding `.detach()`, `.cpu()`, `.numpy()`, `.item()`, or in-place operations, verify that gradients and device behavior remain correct.

Do not use in-place operations when they can interfere with autograd.

For stochastic code:

* Preserve intended randomness.
* Do not add a fixed seed merely to hide instability.
* Do not remove an existing seed without request.
* Distinguish expected stochastic variation from implementation bugs.
* Report nondeterministic validation appropriately.

For numerical safeguards:

* Add epsilon terms only where mathematically justified.
* Do not silently clamp values unless required by the data domain or algorithm.
* Make NaN and infinity handling explicit.
* Do not replace invalid values with zeros without diagnosing the source.

---

## 18. Training and Evaluation Separation

Keep training and evaluation responsibilities distinct.

Do not:

* Update model parameters during evaluation.
* Change model mode implicitly without restoring it when needed.
* Reuse training augmentations in evaluation unless requested.
* Compute evaluation metrics from training-only data.
* Modify checkpoints during pure evaluation tasks.
* Mix clean and poisoned labels unintentionally.
* Recompute triggers with different semantics between training and testing without an explicit protocol.

Use existing `train()` and `eval()` conventions consistently.

Use `torch.no_grad()` or inference mode for evaluation when gradients are not required.

Do not add gradient suppression to code paths that optimize inputs, triggers, or generators.

---

## 19. Logging and Results

Preserve existing machine-readable output.

Do not rename or remove log keys consumed by other scripts.

When adding output:

* Use existing logging conventions.
* Keep important values explicit.
* Avoid excessive per-batch logging.
* Do not print large tensors or model objects.
* Include enough context to distinguish methods, protocols, resolutions, seeds, and checkpoints.
* Keep human-readable and machine-readable output consistent.

Do not claim success based only on the absence of runtime errors.

Validate the actual behavior or metric requested.

---

## 20. Testing Environment

Before running Python commands, tests, smoke tests, or experiment scripts, activate the project virtual environment.

Run activation and the command in the same shell invocation:

```bash
source .qiskit/bin/activate && python ...
```

For pytest:

```bash
source .qiskit/bin/activate && pytest ...
```

For module execution:

```bash
source .qiskit/bin/activate && python -m package.module ...
```

Do not assume that environment activation persists across separate shell executions.

Do not install packages into the system Python environment.

Do not create a second virtual environment unless explicitly requested.

---

## 21. Validation Strategy

Use the narrowest validation that proves the change works.

Preferred order:

1. Syntax or import check.
2. Focused unit test.
3. Focused smoke test.
4. Relevant integration test.
5. Broader test suite only when necessary.

Examples:

```bash
source .qiskit/bin/activate && python -m py_compile path/to/changed_file.py
```

```bash
source .qiskit/bin/activate && pytest path/to/relevant_test.py -q
```

```bash
source .qiskit/bin/activate && python exp_run.py --relevant-options
```

Before inventing a new test command, inspect:

* Existing tests.
* Project documentation.
* Current scripts.
* Continuous integration configuration.
* Existing command examples.

Do not run expensive full experiments when a small smoke test is sufficient, unless the user requested full results.

Do not alter implementation behavior merely to make an existing test pass without understanding the test's contract.

Do not modify or weaken tests to conceal a defect.

Do not delete assertions because they expose a failure.

If validation cannot be completed:

* State exactly what was run.
* State what prevented further validation.
* Distinguish verified behavior from unverified behavior.
* Do not claim that all tests pass.

---

## 22. Test Code Quality

Test observable behavior rather than internal implementation details.

Prefer tests that check:

* Correct outputs.
* Preserved shapes.
* Preserved sample order.
* Expected error conditions.
* Configuration behavior.
* Compatibility with existing modes.
* Regression of the reported bug.

Avoid tests that:

* Depend on exact internal helper names.
* Duplicate the implementation.
* Pass only because values are hard-coded.
* Require unrelated external services.
* Are flaky without a documented stochastic tolerance.
* Assert meaningless facts solely to increase coverage.

For stochastic behavior, use statistical or bounded assertions appropriate to the algorithm rather than demanding exact equality.

---

## 23. File and Repository Hygiene

Do not commit or generate unnecessary artifacts such as:

* Model checkpoints.
* Dataset copies.
* Temporary logs.
* Cache directories.
* Large plots.
* Debug dumps.
* Compiled Python files.
* Environment directories.
* IDE-specific settings.
* Unrequested result files.

Do not modify:

* User data.
* Existing experiment outputs.
* Git history.
* Remote branches.
* Environment secrets.
* Credentials.
* Unrelated configuration files.

Do not run destructive Git commands unless explicitly requested.

Do not overwrite uncommitted user changes.

Before modifying a file with existing changes, preserve those changes and avoid replacing the entire file unnecessarily.

---

## 24. Formatting

Follow the formatting already used in the file.

Do not reformat an entire file for a small functional change.

Do not reorder all imports unless required by the project's formatter.

Do not change quote style, naming style, or whitespace throughout unrelated code.

Keep diffs focused so that behavioral changes are easy to review.

---

## 25. Completion Criteria

A task is complete when:

* The requested behavior is implemented.
* Acceptance criteria are satisfied.
* Existing behavior outside the scope is preserved.
* Relevant validation has been run.
* The diff contains no unrelated changes.
* No unnecessary dependency or abstraction was introduced.
* Known limitations are reported clearly.

Stop when the task is complete.

Do not continue with optional cleanup, architecture redesign, additional features, or speculative improvements unless explicitly requested.

Do not turn a completed local fix into a repository-wide refactor.

---

## 26. Final Response Format

After completing a coding task, provide a concise report containing:

### Summary

What behavior was changed.

### Files Changed

The files modified and the purpose of each modification.

### Validation

The exact commands run and their outcomes.

### Preserved Behavior

Important interfaces or experiment behavior intentionally left unchanged.

### Limitations

Anything not validated or any assumption that may affect the result.

Do not provide a long narrative of every step taken.

Do not claim tests passed unless they were actually run successfully.

Do not describe planned work as completed work.

---

## 27. Decision Examples

### Example: Adding One Evaluation Mode

Bad approach:

* Create an abstract evaluator base class.
* Add an evaluator registry.
* Add a factory.
* Move all existing evaluation code.
* Replace direct calls across the repository.
* Introduce a new configuration framework.

Preferred approach:

* Add the new mode to the existing evaluation entry point.
* Reuse existing focused helpers.
* Add one local helper only if it removes meaningful duplication.
* Preserve existing modes and outputs.
* Add a focused validation for the new mode.

### Example: Supporting Several Resolutions

Bad approach:

* Create a generic multi-dimensional batch container.
* Add automatic shape coercion.
* Introduce a resolution adapter hierarchy.
* Stack incompatible image sizes into one tensor.

Preferred approach:

* Construct paired views from the same source batch.
* Preserve sample and label order.
* Evaluate each resolution separately.
* Aggregate only scalar metrics or compatible outputs.

### Example: Fixing Repeated Training Code

Bad approach:

* Immediately replace all trainers with a universal `BaseTrainer`.
* Add callbacks, hooks, registries, and lifecycle objects.
* Force methods with different optimization behavior into one generic loop.

Preferred approach:

* First identify the exact duplicated operations.
* Extract only stable, semantically identical steps.
* Keep method-specific optimization logic explicit.
* Accept limited duplication when methods have genuinely different behavior.

### Example: Adding a Configuration Value

Bad approach:

* Add a new configuration class.
* Add a parser wrapper.
* Add a schema registry.
* Add automatic environment-variable discovery.

Preferred approach:

* Add the value to the existing configuration structure.
* Pass it explicitly to the relevant function.
* Preserve the existing default behavior.

### Example: Fixing a Local Bug

Bad approach:

* Rename related APIs.
* Move the module.
* Rewrite neighboring functions.
* Format the entire package.
* Add speculative validation layers.

Preferred approach:

* Reproduce or identify the failing path.
* Fix the narrow cause.
* Add a regression check.
* Leave unrelated code untouched.

---

## 28. Mandatory Self-Review

Before finalizing a change, check:

### Scope

* Did I change only what the request requires?
* Did I modify unrelated files?
* Did I accidentally alter defaults or public behavior?
* Can any changed file be removed from the diff?

### Design

* Did I introduce a class where a function would be clearer?
* Did I introduce a wrapper that only forwards calls?
* Did I create a registry, factory, manager, service, or base class without a demonstrated need?
* Did I generalize logic that currently has only one use?
* Did I add configuration for a value that does not need to vary?
* Is the new code easier to understand than a direct implementation?

### Correctness

* Did I trace the actual execution path?
* Are shapes, devices, dtypes, labels, and sample order preserved?
* Did I unintentionally change gradient flow?
* Did I preserve experiment protocols and metric definitions?
* Are error cases visible rather than silently ignored?

### Validation

* Did I run the most relevant focused check?
* Did I activate `.qiskit` in the same shell command?
* Did I report the exact command and result?
* Am I claiming anything that was not verified?

### Cleanup

* Did I add unused helpers, imports, options, or files?
* Did I leave debug output or temporary artifacts?
* Did I perform unrelated formatting?
* Did I continue changing code after the acceptance criteria were already met?

If any answer indicates unnecessary complexity, simplify the implementation before finishing.
