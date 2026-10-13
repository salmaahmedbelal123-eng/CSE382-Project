# Team Git workflow
1. Never commit directly to `main`.
2. `git checkout main && git pull`
3. `git checkout -b feature/<short-name>`  (e.g. feature/eda-plots)
4. Work, then `git add <files>` and `git commit -m "type: short message"`  (commit often, small steps)
5. `git push -u origin feature/<short-name>`
6. Open a Pull Request on GitHub, a teammate reviews, then "Merge".
7. Each person commits with THEIR OWN GitHub account (check `git config user.name` / `user.email`).

Commit message types: feat, fix, docs, test, refactor, data, exp (experiment).
Examples: `feat: add lead-time feature`, `exp: baseline random forest ADR`, `docs: leakage audit for clustering`.
