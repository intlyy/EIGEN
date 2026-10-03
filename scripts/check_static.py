"""Parse source, profile/project schemas and local imports; execute no experiments/tests."""

from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

from eigen.config import ProjectConfig, TuningConfig
from eigen.models import EngineProfile
from eigen.rendering import ExperimentRenderer
from eigen.search_space import SearchSpace
from eigen.study import StudyConfig
from eigen.tuning.partitioning import ProfilePartitioner


def main():
    root = Path(__file__).resolve().parents[1]
    sources = [
        *root.glob("*.py"),
        *root.glob("src/eigen/**/*.py"),
        *root.glob("src/eigen/**/*.py.txt"),
        *root.glob("scripts/*.py"),
        *root.glob("tests/*.py"),
    ]
    errors = []
    paper_objectives = [{"metric": "qps", "direction": "maximize"}]
    default_objectives = [o.model_dump() for o in TuningConfig().objectives]
    if default_objectives != paper_objectives:
        errors.append("TuningConfig: paper default must only maximize QPS")
    for path in sources:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                module = node.module if isinstance(node, ast.ImportFrom) else None
                if module and module.startswith("eigen"):
                    target = root / "src" / module.replace(".", "/")
                    if (
                        not target.with_suffix(".py").exists()
                        and not (target / "__init__.py").exists()
                    ):
                        errors.append(f"{path.relative_to(root)}: missing local module {module}")
        except (SyntaxError, UnicodeError) as error:
            errors.append(f"{path.relative_to(root)}: {error}")
    profiles = list(root.glob("src/eigen/resources/profiles/*.json"))
    rendered_regions = 0
    for path in profiles:
        try:
            profile = EngineProfile.model_validate_json(path.read_text(encoding="utf-8"))
            runtime = (
                {"vector_size": 384} if "vector_size" in profile.experiment.runtime_context else {}
            )
            partitioner = ProfilePartitioner(SearchSpace(profile), runtime=runtime)
            for region in partitioner.regions:
                ExperimentRenderer(profile).render(region.fixed, runtime)
                rendered_regions += 1
        except ValueError as error:
            errors.append(f"{path.relative_to(root)}: {error}")
    projects = [p for p in root.glob("examples/**/*.json") if "study" not in p.name]
    for path in projects:
        try:
            project = ProjectConfig.model_validate_json(path.read_text(encoding="utf-8"))
            if (
                path.parent.name == "paper"
                and [o.model_dump() for o in project.tuning.objectives] != paper_objectives
            ):
                errors.append(f"{path.relative_to(root)}: paper example must only maximize QPS")
        except ValueError as error:
            errors.append(f"{path.relative_to(root)}: {error}")
    studies = list(root.glob("examples/**/*study.json"))
    for path in studies:
        try:
            study = StudyConfig.model_validate_json(path.read_text(encoding="utf-8"))
            for reference in [*study.minidbs, study.full_database]:
                if not (path.parent / reference).is_file():
                    errors.append(f"{path.relative_to(root)}: missing project {reference}")
        except ValueError as error:
            errors.append(f"{path.relative_to(root)}: {error}")
    tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    report = {
        "python_sources_parsed": len(sources),
        "profiles_validated": len(profiles),
        "declarative_regions_rendered": rendered_regions,
        "project_schemas_validated": len(projects),
        "study_schemas_validated": len(studies),
        "default_objectives": default_objectives,
        "errors": errors,
        "experiments_executed": False,
        "unit_tests_executed": False,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
