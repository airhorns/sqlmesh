from pathlib import Path
from textwrap import dedent

import pytest

from sqlmesh.core.config import Config, ModelDefaultsConfig
from sqlmesh.core.context import Context


pytestmark = pytest.mark.slow


def _write(path: Path, sql: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(sql))


def _model_sql(name: str, value: int) -> str:
    return f"""
        MODEL (
            name {name},
            kind FULL
        );

        SELECT {value} AS value
    """


def _audit_sql(
    name: str,
    model: str,
    *,
    predicate: str = "FALSE",
    blocking: bool = True,
) -> str:
    return f"""
        AUDIT (
            name {name},
            standalone true,
            blocking {str(blocking).lower()}
        );

        SELECT * FROM {model} WHERE {predicate}
    """


def _environment_snapshots(context: Context) -> dict[str, object]:
    environment = context.state_reader.get_environment("prod")
    assert environment is not None
    return {snapshot.name: snapshot for snapshot in environment.snapshots}


def test_scoped_plan_preserves_unselected_standalone_audit_mutations(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    audits = tmp_path / "audits"
    baseball_model = models / "baseball.sql"
    hockey_model = models / "hockey.sql"
    _write(baseball_model, _model_sql("baseball.core", 1))
    _write(hockey_model, _model_sql("hockey.core", 1))

    initial_audits = {
        "baseball_modified": ("baseball.core", True),
        "baseball_removed": ("baseball.core", True),
        "hockey_modified": ("hockey.core", True),
        "hockey_removed": ("hockey.core", True),
        "hockey_metadata": ("hockey.core", True),
    }
    for name, (model, blocking) in initial_audits.items():
        _write(
            audits / f"{name}.sql",
            _audit_sql(name, model, blocking=blocking),
        )

    context = Context(
        paths=tmp_path,
        config=Config(model_defaults=ModelDefaultsConfig(dialect="duckdb")),
    )
    context.apply(context.plan_builder("prod", skip_tests=True).build())
    before = _environment_snapshots(context)

    # Selected baseball changes cover model/audit modification, addition, and
    # deliberate deletion.
    _write(baseball_model, _model_sql("baseball.core", 2))
    _write(
        audits / "baseball_modified.sql",
        _audit_sql("baseball_modified", "baseball.core", predicate="value < 0"),
    )
    (audits / "baseball_removed.sql").unlink()
    _write(
        audits / "baseball_new.sql",
        _audit_sql("baseball_new", "baseball.core"),
    )

    # Foreign local state contains every mutation class. A baseball-scoped plan
    # must preserve the target environment's exact hockey snapshots.
    _write(hockey_model, _model_sql("hockey.core", 2))
    _write(
        audits / "hockey_modified.sql",
        _audit_sql("hockey_modified", "hockey.core", predicate="value < 0"),
    )
    (audits / "hockey_removed.sql").unlink()
    _write(
        audits / "hockey_metadata.sql",
        _audit_sql("hockey_metadata", "hockey.core", blocking=False),
    )
    _write(
        audits / "hockey_new.sql",
        _audit_sql("hockey_new", "hockey.core"),
    )
    context.load()

    plan = context.plan_builder(
        "prod",
        select_models=["baseball.*"],
        select_standalone_audits=["(baseball.*)+"],
        skip_tests=True,
    ).build()

    mutated_names = {
        *(snapshot.name for snapshot in plan.new_snapshots),
        *(snapshot.name for snapshot in plan.modified_snapshots.values()),
        *(snapshot.name for snapshot in plan.context_diff.removed_snapshots.values()),
        *(
            plan.context_diff.snapshots[snapshot_id].name
            for snapshot_id in plan.metadata_updated
        ),
    }
    assert mutated_names == {
        '"memory"."baseball"."core"',
        "baseball_modified",
        "baseball_new",
        "baseball_removed",
    }

    context.apply(plan)
    after = _environment_snapshots(context)

    assert "baseball_new" in after
    assert "baseball_removed" not in after
    assert (
        after["baseball_modified"].fingerprint
        != before["baseball_modified"].fingerprint
    )
    assert (
        after['"memory"."baseball"."core"'].fingerprint
        != before['"memory"."baseball"."core"'].fingerprint
    )
    assert context.engine_adapter.fetchone("SELECT value FROM baseball.core") == (2,)
    assert context.engine_adapter.fetchone("SELECT value FROM hockey.core") == (1,)

    assert "hockey_new" not in after
    for name in (
        '"memory"."hockey"."core"',
        "hockey_modified",
        "hockey_removed",
        "hockey_metadata",
    ):
        assert after[name].snapshot_id == before[name].snapshot_id
        assert after[name].fingerprint == before[name].fingerprint

    # Preserving foreign audit nodes from state must also preserve their parent
    # dependency identifiers, not merely their own query/metadata fingerprint.
    assert after["hockey_modified"].parents == before["hockey_modified"].parents
