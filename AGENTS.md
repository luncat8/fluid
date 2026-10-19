# AGENTS.md

these are not strict rules, use best. feel free to stop any task and improve something if you found really good way: system was developed by different agents and some was not very smart.

## style

- use a single tab indentation. LF end.
const $ = id => document.getElementById(id);

- avoid deep nesting of braces { } and long if-else.
- flatten with early returns, helper functions, or flat data tables.

- avoid duplication of code.

- avoid allocations in the hot path (per-frame loop, sim, render).
- no new {}, [], object literals, closures, or string concat in the frame loop.
- reuse preallocated buffers / typed arrays / scratch objects.
- allocate once at setup, mutate in place per frame.


- plan*.md is NOT the implementation log. if need - update/improve plan, but keep final plan as artifact for possible fork or reimplementation without referring of what was and what done, without referring chat, etc.
- plan number i.e. 0.1 - is not version sequence. it is ok to continue old numbers by subject and implement in any useful order.

- only essential concise comments (in code and in worklogs) that really helpful i.e. explain why and decision. prefer descriptive naming.

- no legacy support, no old versions, no outdated browsers, no leftovers and no over protecting from unreal edge cases. we need clean architecture.

## runtime

- file:// friendly, classic <script> tags, no modules, no build.
- guard module.exports so files also run under node.
- no internet links: vendor any lib as a local js file.

## concepts

source brief: draft.txt. architecture, model and decisions: 0.0.0-plan.md (read first).
the scheme spec that both the cpu reference and the wgsl implement is 0.0.0-plan.md section 5.

- one substance. rgb = three conserved color tracers, not three materials.
- aggregate phases: no free surface, no surface tension. vacuum < gas < liquid = solid by density.
- conserved quantities: substance mass m, dye dr/dg/db, internal energy E. temperature and
	phase are derived from E (latent heat becomes a plateau in T(E)).
- conservation method: a face volume flux computed once per step, read by both neighbouring
	cells with opposite signs. never interpolate a conserved quantity.
- liquid/solid incompressible, gas weakly compressible through a divergence source S in the
	pressure projection. one solver, no separate gas pipeline.

## files

findings-pitfalls-skills.md - notes and pitfalls for LLM agents. write here if found good way to do something.

archive/ - for implemented plans

experiments/ - measurement scripts (node), not loaded by the page.
experiments/logs/ - keep useful;

## sandbox

git push returns "Invalid username or token" is ok, no need to investigate or report - i will apply manually

## Workflow

start work with review previous task: it was done by another agent and you can find really better way to improve it or find some issues. dont need to run tests again because already done in previous stage.
then continue with next: review next step plan, if found good ideas - improve plan, implement.
record development steps and its validation in `archive/*-worklog.md`, end job with a next step suggestion.

example of file structure:
0.1.x-draft.md - human write thoughts about goals and design
archive/0.1.0-plan.md
archive/0.1.0-worklog.md
0.1.1-plan.md
gui.1-plan.md

