# Agent_Manager_Skills

[简体中文](README.md) | English

[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)
![Domain](https://img.shields.io/badge/Domain-CS_Research-informational)
![Qoder](https://img.shields.io/badge/Qoder-supported-7B61FF)
![Claude Code](https://img.shields.io/badge/Claude_Code-supported-D97757)
![Codex](https://img.shields.io/badge/Codex-supported-10A37F)
![Skill Docs](https://img.shields.io/badge/Skill_Docs-Chinese-red)

A collection of skills that guide how coding agents work. Each skill lives in its own directory and can be installed independently.

These skills are intended for general computing research workflows, such as running experiments on local machines or GPU servers, training models, replacing methods, and organizing experiment records and results.

This README is available in English and Chinese. The skill instructions and reference files are currently written in Chinese.

## Skills

| Skill | Invocation | Purpose |
|---|---|---|
| [clean-replace](clean-replace/SKILL.md) | `/clean-replace` | Completely remove an old implementation when replacing a method, model, interface, or implementation approach, and update all references, configuration, tests, and documentation. Prohibits versioned copies, backups, compatibility layers, silent fallbacks, and verbose defensive code such as repeated field-by-field `isinstance` checks. Keeps script arguments minimal: new behavior switches require prior user approval, and the user sets the default behavior. Reviews project files individually before making changes and maintains a change index to narrow future reviews. Records each fact in one place, keeps state files current, avoids manually copying hashes into documentation, and puts temporary files in `.agents/tmp/`. Also helps organize projects with accumulated versions, duplicate records, and caches. |
| [env-first](env-first/SKILL.md) | Automatically applies to installation, download, and GPU-related tasks; can also be explicitly requested by name | Checks the machine environment before taking action, including GPU, CUDA, Python environments, package mirrors and proxies, disk space, server workspace locations, internet access, and shared data directories, then records the findings in the project's `AGENTS.md` or `CLAUDE.md`. Claims about the machine must be backed by command output. After a download failure, first checks whether the target exists and whether the version matches. Avoids unnecessary environments, damaging edits with `sed`, global configuration changes, caches in artifact directories, and reused container image tags. |
| [train-preflight](train-preflight/SKILL.md) | `/train-preflight` | Prepares for a full training run: estimates GPU memory and disk usage; measures peak GPU memory and throughput during training, evaluation, and checkpoint saving with smoke tests; maximizes throughput within the GPU memory target. Establishes baseline settings such as mixed precision, then finds the per-device batch size. When memory limits throughput, progressively considers gradient checkpointing, gradient accumulation, ZeRO/FSDP sharding, and offloading instead of relying only on batch size changes. Uses available GPU memory while retaining a safety margin agreed with the user. Sets up checkpoint recovery and verifies that interrupted training can resume. Parameters affecting only speed or memory may be adjusted independently; changes affecting training results require user approval. Prohibits silent OOM recovery that automatically reduces the batch size. |

## Installation

Copy the entire directory of each skill you need into your agent's skills directory. For example, to install `clean-replace`:

```bash
# Qoder: available across all projects
cp -r clean-replace ~/.qoder/skills/

# Qoder: available only in the current project
cp -r clean-replace <project-root>/.qoder/skills/

# Claude Code
cp -r clean-replace ~/.claude/skills/
```

Keep the entire directory intact. `SKILL.md` references other files in the same directory, so copying only `SKILL.md` is insufficient.

## Directory Structure

```text
Agent_Manager_Skills/
├── README.md               # Chinese overview
├── README.en.md            # English overview
├── LICENSE
├── clean-replace/
│   ├── SKILL.md            # Triggers, confirmation points, principles, workflow, checklist
│   ├── failure-modes.md    # Common failure modes
│   ├── rules.md            # Prohibitions, naming, tests, file growth
│   ├── scanning.md         # Phase B: file-by-file review; Phase C: change-list format
│   ├── change-index.md     # Change index (docs/agent-index/)
│   ├── artifacts.md        # Experiment artifact management
│   ├── records.md          # Record responsibilities, hashes, image tags, temporary files, cleanup
│   ├── asking-user.md      # When and how to ask the user
│   ├── verification.md     # Residual checks and final report format
│   └── examples.md         # Complete examples
├── env-first/
│   └── SKILL.md            # Environment checks, GPU/CUDA, Python, downloads, caches, servers
└── train-preflight/
    ├── SKILL.md            # Triggers, memory targets, principles, confirmations, workflow, report, checklist
    ├── estimation.md       # GPU memory components, three memory readings, multiple GPUs, disk budget
    ├── smoke-test.md       # Execution checks, memory peaks, throughput, artifact locations
    ├── tuning.md           # Batch-size tuning, related parameters, learning-rate relationships, guidance
    └── resume.md           # What to save, how to save and restore, how to verify recovery
```

## Adding a Skill

1. Create a directory at the repository root. Its name is the skill name and must use only lowercase letters, digits, and hyphens.
2. Add a `SKILL.md` file with YAML frontmatter at the beginning:

```markdown
---
name: skill-name
description: Describe what the skill does and when to use it, in the third person, within 1024 characters.
---
```

3. Keep the body of `SKILL.md` under 500 lines when possible. Move longer explanations into reference files in the same directory and link to them directly from `SKILL.md`, without nested reference chains.
4. Commands that need to be executed consistently can be placed in a `scripts/` subdirectory.
5. Add the skill to the table above.

## License

[Apache License 2.0](LICENSE)
