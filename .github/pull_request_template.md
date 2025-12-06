## Summary

<!-- Briefly explain what this PR does in 2–3 sentences. -->
- 

## Context / Problem

<!-- Why is this change needed? Link any relevant issues, tickets, or docs. -->
- 

## Changes

<!-- High-level list of what you changed. Try to group logically. -->
- 
- 
- 

---

## ML / Model Changes

<!-- Fill this out if the PR touches training, data, or model code. -->

**Does this PR change the model or training pipeline?**
- [ ] Yes
- [ ] No

If **Yes**, complete the details below:

**1. Data**
- [ ] New dataset added
- [ ] Existing dataset modified
- [ ] Data preprocessing / feature engineering changed

Details (source, shape, key columns, filters, etc.):
- 

**2. Training / Code**
- Entry point(s) affected (e.g. `src/train_model.py`, config files):
  - 
- Hyperparameters changed (if any):
  - 

**3. Metrics & Evaluation**
- Baseline metrics (before this PR), e.g. accuracy / F1 / AUC / loss:
  - 
- New metrics (after this PR), with comparison:
  - 
- Evaluation dataset(s) used:
  - 

**4. Model Artifacts**
- Expected model output path(s):  
  - `models/model_<version>.pkl`  
- Any changes to how `VERSION` is generated or used?
  - [ ] Yes
  - [ ] No

If **Yes**, explain:
- 

**5. Backward Compatibility**
- [ ] Old models remain usable
- [ ] Inference API / interface unchanged
- [ ] Breaking changes (explain below)

Details (if breaking):
- 

---

## CI / Testing

**Local checks run before opening this PR**

- [ ] `pytest tests/ -v`
- [ ] `pytest` with coverage (if run locally)
- [ ] `black src/ tests/`
- [ ] `flake8 src/ tests/`
- [ ] `mypy` (if applicable)
- [ ] I have run the model training command locally (if relevant):  
  `python src/train_model.py --version <test-version>`

Notes on test results (if anything flaky, slow, or skipped):
- 

**GitHub Actions CI**

- [ ] I understand that this PR will trigger the `CI` workflow:
  - Lint (flake8)
  - Formatting check (black)
  - Tests + coverage (pytest + coverage.xml)
  - Codecov upload
  - Model training with auto-generated version
  - Model artifact upload

---

## Risks & Rollback Plan

**Risks**
- 

**Rollback plan (if something goes wrong)**  
<!-- e.g. revert PR, redeploy previous model artifact, disable a feature flag, etc. -->
- 

---

## Documentation & Communication

- [ ] README / docs updated (if behavior or usage changed)
- [ ] Example commands / notebooks still work
- [ ] Stakeholders informed (if needed) – e.g. DS/ML, product, ops

Links to updated docs or notebooks:
- 

---

## Additional Notes

<!-- Anything reviewers should pay special attention to (tricky logic, edge cases, design decisions, etc.). -->
- 

