"""MiniApp's exit and verify() against fakes (no backend, no telemetry)."""

import asyncio
import atexit
import logging
import types

import pytest

from wfMiniAPI.api import AttachError, MiniApp
from wfMiniAPI.api import app as app_module
from wfMiniAPI.telemetry import capability


@pytest.fixture(autouse=True)
def no_exit_hooks(monkeypatch):
    """Record atexit calls instead of registering hooks for the pytest process."""
    calls = {"register": [], "unregister": []}
    monkeypatch.setattr(atexit, "register", lambda f: calls["register"].append(f))
    monkeypatch.setattr(atexit, "unregister", lambda f: calls["unregister"].append(f))
    return calls


class FakeFlow:
    uid = "flow.fake"
    backend = None
    telemetry = None

    def __init__(self, shutting_down=False):
        self.is_shutting_down = shutting_down


class FakeManager:
    """What MiniAppTelemetry uses of a RHAPSODY TelemetryManager."""

    session_id = "flow.fake"

    def __init__(self, running=True):
        self._running = running
        self.emitted = []

    def emit(self, event):
        self.emitted.append(event.event_type)


def attached(flow=None, running=True, **state):
    """A MiniApp whose telemetry passed the attach checks (check/start not run)."""
    flow = flow or FakeFlow()
    app = MiniApp(flow, FakeManager(running))
    t = app.telemetry
    t.started = True
    t._t0 = 0.0
    t.checkpoint_dir = None  # nothing is written by these tests
    for key, value in state.items():
        setattr(t, key, value)
    return app


def in_task(coro_fn):
    async def go():
        return await asyncio.create_task(coro_fn())

    return asyncio.run(go())


# ------------------------------------------------------------------ ended_by
@pytest.mark.parametrize(
    "exc_type,shutting_down,expected",
    [
        (None, False, "normal"),
        (None, True, "normal"),  # the core shut the engine down itself (or caught a signal)
        (ValueError, False, "exception"),
        (KeyboardInterrupt, True, "exception"),
        (ExceptionGroup, False, "exception"),
        (asyncio.CancelledError, True, "engine_shutdown"),  # AsyncFlow cancelled the futures
        (asyncio.CancelledError, False, "cancelled"),
    ],
)
def test_ended_by(exc_type, shutting_down, expected):
    async def go():
        return attached(FakeFlow(shutting_down))._ended_by(exc_type)

    assert in_task(go) == expected


def test_outer_cancel_after_the_cores_own_shutdown_is_cancelled():
    # a core that awaits flow.shutdown() in its finally: the engine is shutting down,
    # but the run was cancelled from outside (a timeout calls task.cancel())
    async def go():
        app = attached(FakeFlow(shutting_down=True))

        async def body():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                return app._ended_by(asyncio.CancelledError)

        task = asyncio.create_task(body())
        await asyncio.sleep(0)
        task.cancel()
        return await task

    assert asyncio.run(go()) == "cancelled"


# ------------------------------------------------------------------ the exit
@pytest.mark.parametrize("exc_type", [None, ValueError, asyncio.CancelledError, KeyboardInterrupt])
def test_the_exit_never_pauses(exc_type):
    """__aexit__ awaits nothing, so a cancel cannot interrupt it (run inside a task)."""

    async def go():
        app = attached()
        coro = app.__aexit__(exc_type, None, None)
        with pytest.raises(StopIteration) as stop:
            coro.send(None)  # completes in one step
        return stop.value.value, app.telemetry

    returned, telemetry = in_task(go)
    assert returned is False and telemetry._run_ended_queued


def test_the_exit_queues_run_ended_only_while_telemetry_runs():
    async def go():
        running, stopped = attached(running=True), attached(running=False)
        for app in (running, stopped):
            await app.__aexit__(None, None, None)
        return running, stopped

    running, stopped = in_task(go)
    assert running.telemetry.manager.emitted == ["wfminiapi.RunEnded"]
    assert stopped.telemetry.manager.emitted == [] and not stopped.telemetry._run_ended_queued
    assert running.telemetry.ended_by == stopped.telemetry.ended_by == "normal"


def test_an_error_in_the_exit_becomes_a_reason_not_an_exception():
    async def go():
        app = attached()

        def boom(*args, **kwargs):
            raise RuntimeError("emit failed")

        app.telemetry.manager.emit = boom
        assert await app.__aexit__(ValueError, ValueError("core"), None) is False
        return app.telemetry.reasons

    assert in_task(go) == ["run-end event failed: RuntimeError: emit failed"]


# ------------------------------------------------------------------ verify()
@pytest.fixture
def fake_checkpoint(monkeypatch):
    """verify_checkpoint() replaced by a recorder; returns what it was called with."""
    calls = {}

    def fake(directory, session_id, **kwargs):
        calls.update(kwargs)
        return capability.MiniAppReport(
            session_id,
            "ckpt",
            not kwargs["reasons"],
            kwargs["ended_by"],
            reasons=kwargs["reasons"],
            notes=kwargs["notes"],
        )

    monkeypatch.setattr(capability, "verify_checkpoint", fake)
    return calls


def stopped_app(engine_down_at_stop, run_ended_queued, session_ended_seen=True):
    return attached(
        running=False,
        _session_ended_seen=session_ended_seen,
        _engine_down_at_stop=engine_down_at_stop,
        _run_ended_queued=run_ended_queued,
        ended_by="normal",
    )


def test_verify_complete_when_run_ended_was_queued(fake_checkpoint):
    report = stopped_app(engine_down_at_stop=False, run_ended_queued=True).verify()
    assert report.telemetry_complete and report.notes == () and report.reasons == ()


def test_verify_notes_run_ended_lost_to_the_engine_shutdown(fake_checkpoint):
    report = stopped_app(engine_down_at_stop=True, run_ended_queued=False).verify()
    assert report.telemetry_complete and "engine shutdown stopped telemetry" in report.notes[0]


def test_verify_a_stop_by_hand_while_the_engine_ran_is_incomplete(fake_checkpoint):
    report = stopped_app(engine_down_at_stop=False, run_ended_queued=False).verify()
    assert not report.telemetry_complete
    assert report.reasons[0].startswith("stopped_before_exit")


def test_verify_a_stop_that_did_not_finish_is_incomplete(fake_checkpoint):
    report = stopped_app(None, run_ended_queued=True, session_ended_seen=False).verify()
    assert not report.telemetry_complete and "did not finish writing" in report.reasons[0]


def test_verify_while_telemetry_runs_reports_only_that(fake_checkpoint):
    report = attached(running=True).verify()
    assert not report.telemetry_complete and len(report.reasons) == 1
    assert report.reasons[0].startswith("telemetry still running")
    assert fake_checkpoint == {}  # nothing counted: the file is still buffered


def test_verify_not_attached_and_never_entered():
    app = MiniApp(FakeFlow(), FakeManager())
    assert app.verify().reasons == ("MiniApp was never entered",)
    app.telemetry.attach_error = "AttachError: wrong manager"
    report = app.verify()
    assert report.session_id == "flow.fake" and report.checkpoint is None
    assert report.reasons == ("not attached: AttachError: wrong manager",)
    assert app.report is report


def test_verify_turns_a_failing_count_into_a_reason(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(capability, "verify_checkpoint", broken)
    report = stopped_app(engine_down_at_stop=False, run_ended_queued=True).verify()
    assert report.reasons[-1] == "verification failed: OSError: disk gone"


def test_verify_never_raises(monkeypatch):
    app = stopped_app(engine_down_at_stop=False, run_ended_queued=True)

    def broken(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(app.telemetry, "_verify", broken)  # the last-resort path
    report = app.verify()
    assert report.reasons == ("verification failed: RuntimeError: bug",)


# ------------------------------------------------------------------ the reminder
def test_the_reminder_is_registered_on_entry_and_removed_by_verify(no_exit_hooks):
    class OldFlow:  # makes check() fail fast: entry still registers the reminder
        uid = "old"
        backend = None

    app = MiniApp(OldFlow(), types.SimpleNamespace())  # a manager without session_id
    with pytest.raises(AttachError):
        asyncio.run(app.__aenter__())
    assert no_exit_hooks["register"] == [app._reminder]
    assert app._reminder.args == ("old",)  # holds only a string, not the run's objects
    app.verify()
    assert no_exit_hooks["unregister"] == [app._reminder]
    assert app.telemetry.attach_error.startswith("AttachError: this AsyncFlow has no")


def test_the_reminder_logs_a_warning(caplog):
    app_module._warn_unverified("flow.x")
    assert "MiniApp flow.x: verify() was never called" in caplog.text


def test_the_reminder_prints_when_only_null_handlers_would_see_it(monkeypatch, capsys):
    # the Dragon runtime adds a NullHandler to the root logger, which hides warnings
    quiet = logging.getLogger("wfMiniAPI.test.quiet")
    quiet.propagate = False
    quiet.handlers = [logging.NullHandler()]
    monkeypatch.setattr(app_module, "log", quiet)
    app_module._warn_unverified("flow.y")
    assert "WARNING: MiniApp flow.y: verify() was never called" in capsys.readouterr().err


def test_an_asyncflow_without_joinable_shutdown_is_refused():
    class OldFlow:
        uid = "old"
        backend = None

    with pytest.raises(AttachError, match="no joinable shutdown"):
        MiniApp(OldFlow(), object()).telemetry.check()
