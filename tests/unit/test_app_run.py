"""MiniApp's API for a run of many engines and components, against fakes: what it refuses at
the call, so that a run is never assembled from parts, components or checks it cannot verify."""

import atexit

import pytest

from wfMiniAPI.api import Component, Composition, MiniApp, PartRef


@pytest.fixture(autouse=True)
def no_exit_hooks(monkeypatch):
    """Record atexit calls instead of registering hooks for the pytest process."""
    monkeypatch.setattr(atexit, "register", lambda f: None)
    monkeypatch.setattr(atexit, "unregister", lambda f: None)


class FakeFlow:
    uid = "asyncflow.session.run"
    backend = None
    telemetry = None
    is_shutting_down = False


class FakeManager:
    session_id = "asyncflow.session.run"
    _running = True

    def __init__(self):
        self.emitted = []

    def emit(self, event):
        self.emitted.append(event)


def entered(tmp_path, part_of=None):
    """A MiniApp as after `async with app` attached it to a checkpoint in `tmp_path`."""
    app = MiniApp(FakeFlow(), FakeManager(), part_of=part_of)
    app.telemetry.checkpoint_dir = tmp_path
    app.telemetry._checkpoint_abs = tmp_path.resolve()  # what attach records
    return app


def verdict(files):
    return True


def check(run):
    return True


def test_a_part_is_handed_out_once_the_block_is_entered_and_once_per_name(tmp_path):
    app = MiniApp(FakeFlow(), FakeManager())
    with pytest.raises(RuntimeError, match="enter the MiniApp"):
        app.part("p0")
    app = entered(tmp_path)
    ref = app.part("p0")
    assert ref == PartRef(
        run_id="asyncflow.session.run",
        name="p0",
        parts_dir=str(tmp_path.resolve() / "asyncflow.session.run.parts"),
    )
    with pytest.raises(ValueError, match="already handed out"):
        app.part("p0")
    with pytest.raises(ValueError, match="letters, digits"):
        app.part("p 1")


def test_a_part_reports_into_its_run_and_hands_out_no_parts(tmp_path):
    ref = PartRef(run_id="asyncflow.session.run", name="p0", parts_dir=str(tmp_path))
    part = entered(tmp_path / "p0", part_of=ref)
    assert part.telemetry.run_id == "asyncflow.session.run"
    with pytest.raises(RuntimeError, match="cannot hand out parts"):
        part.part("p1")
    with pytest.raises(TypeError, match="PartRef"):
        MiniApp(FakeFlow(), FakeManager(), part_of={"name": "p0"})


def test_components_have_one_name_each_and_a_verifier_that_resolves_now(tmp_path):
    app = entered(tmp_path)
    app.component("agents", tmp_path, verify=f"{__name__}:verdict")
    with pytest.raises(ValueError, match="already registered"):
        app.component("agents", tmp_path, verify=verdict)
    with pytest.raises(ModuleNotFoundError):
        app.component("model", tmp_path, verify="no_such_module:verify")
    with pytest.raises(ValueError, match="module:function"):
        app.component("model", tmp_path, verify="no_colon")
    assert app.composition.components == {
        "agents": Component("agents", str(tmp_path.resolve()), f"{__name__}:verdict")
    }


def test_checks_have_one_name_each_and_either_a_verifier_or_a_result(tmp_path):
    app = entered(tmp_path)
    app.check("rows_match", verify=check)
    app.check("computed", ok=True, detail={"n": 1})
    with pytest.raises(ValueError, match="its own name"):
        app.check("rows_match", ok=False)
    with pytest.raises(ValueError, match="verify= .* or ok="):
        app.check("both", verify=check, ok=True)
    with pytest.raises(ValueError, match="verify= .* or ok="):
        app.check("neither")
    with pytest.raises(ValueError, match="the parts' checks"):
        app.check("part p0", ok=True)
    with pytest.raises(AttributeError):
        app.check("missing", verify=f"{__name__}:no_such_check")
    assert [c["name"] for c in app.composition.checks] == ["rows_match", "computed"]


def test_the_run_is_composed_on_its_miniapp_and_telemetry_only_records_it(tmp_path):
    """The run's parts, components, checks and parameters belong to the MiniApp; its
    telemetry reads the same composition when it verifies the run."""
    app = entered(tmp_path)
    app.parameters({"agents": 2})
    app.part("p0")
    app.component("agents", tmp_path, verify=verdict)
    app.check("rows_match", verify=check)
    assert isinstance(app.composition, Composition)
    assert app.composition.parameters == {"agents": 2}
    assert app.composition.parts == ["p0"]
    assert list(app.composition.components) == ["agents"]
    assert [c["name"] for c in app.composition.checks] == ["rows_match"]
    assert app.telemetry.composition is app.composition
    assert not Composition().declared and app.composition.declared


def test_parameters_given_at_construction_go_through_app_parameters(tmp_path):
    """Declared with the MiniApp, as the run's configuration: recorded as `app.parameters()`
    records them, and emitted only once telemetry runs (at the start of the block)."""
    manager = FakeManager()
    app = MiniApp(FakeFlow(), manager, parameters={"agents": 2, "tags": {"b", "a"}})
    assert app.composition.parameters == {"agents": 2, "tags": ["a", "b"]}
    assert manager.emitted == []  # not started: RunStarted comes first, then Parameters
    app.telemetry.started = True
    app.parameters({"agents": 3})
    assert app.composition.parameters == {"agents": 3}
    assert [e.event_type for e in manager.emitted] == ["wfminiapi.Parameters"]


def test_the_run_id_is_fixed_when_the_miniapp_is_built(tmp_path):
    """The run's engine session, or the run a part reports into: known at construction."""
    app = MiniApp(FakeFlow(), FakeManager())
    assert vars(app.telemetry)["run_id"] == "asyncflow.session.run"
    ref = PartRef(run_id="asyncflow.session.other", name="p0", parts_dir=str(tmp_path))
    assert vars(MiniApp(FakeFlow(), FakeManager(), part_of=ref).telemetry)["run_id"] == (
        "asyncflow.session.other"
    )
