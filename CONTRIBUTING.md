# How to contribute to this project

This repository uses the issue-branch-PR workflow.
The `master` branch is protected, meaning new code is incorporated into the `master` branch via pull requests (PRs) from other branches after they are reviewed by at least one collaborator.
Each branch should correspond to an issue scoping out the specific goals of the new code we aim to write.
It is customary to name new branches `issue/X/brief-summary`, where `X` is the issue number and `brief-summary` is a very short description of the scope of the issue.

Some issues are _Epic_ meaning they're a place to collect lots of small issues that break apart a large problem.
It's possible to have a branch for the Epic issue, into which branches from the smaller issues are merged before the Epic branch is merged into the `master` branch.
If the Epic branch isn't protected, PRs aren't necessary, but they might be useful if multiple people are working on the Epic branch.
