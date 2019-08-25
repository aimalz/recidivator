# How to contribute to this project

This repository uses an issue-branch-PR workflow.

The `master` branch is protected, meaning new code is incorporated into the `master` branch via pull requests (PRs) from other branches after they are reviewed by at least one collaborator.
Each branch should correspond to an issue scoping out the specific goals of the new code we aim to write, with a name of the form `issue/N/brief-summary`, where `N` is the issue number and `brief-summary` is a just a few words, usually from the issue title, describing the scope of the branch.

A good issue includes a short title as well as a longer description of what needs to be done, including the conditions under which it can be closed (e.g. "This issue can be closed when the bug in function X of module Y is fixed" or "This issue can be closed when the demo includes sample code for algorithm Z").
To contribute to resolving an issue, please assign yourself to it so others know you're working on it.
A single issue can have multiple assignees, so it's best to also leave a comment on the issue to notify other potential contributors, especially if there's already a branch for it.

Some issues are _Epic_ meaning they're a place to collect lots of small issues that break apart a larger problem.
It's possible to have a branch for the Epic issue, into which branches from the smaller issues are merged before the Epic branch is merged into the `master` branch.
If the Epic branch isn't protected, PRs aren't necessary, but they might be useful, particularly if multiple people are working on the Epic issue.

To review a pull request, please make sure that Jupyter notebooks have cleared outputs and that code actually runs.
When approving a pull request and merging a branch into `master`, close the issue it corresponds to.
In most cases, it's appropriate to delete the branch when closing an issue as well.
