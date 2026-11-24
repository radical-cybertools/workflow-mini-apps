"""wfMiniAPI telemetry events (schema ``wfminiapi/1``), defined with RHAPSODY's define_event."""

from rhapsody.telemetry.events import define_event

SCHEMA = "wfminiapi/1"
PREFIX = "wfminiapi."

RunStarted = define_event(
    "wfminiapi.RunStarted",
    schema=(str, SCHEMA),
    wfminiapi_version=str,
    asyncflow_version=str,
    rhapsody_version=str,
    backends=str,  # comma-separated backend class names
    run_id=str,  # the run's engine session id; its own for the run's engine
    part=str,  # the part this engine is, in a run of many engines; "" for the run's engine
)

# The run's parameters (its configuration), as JSON
Parameters = define_event("wfminiapi.Parameters", schema=(str, SCHEMA), parameters=str)

RunEnded = define_event(
    "wfminiapi.RunEnded",
    schema=(str, SCHEMA),
    ended_by=str,  # normal | exception | cancelled | engine_shutdown
    exception_type=str,
    duration_s=float,
)

# Internal FIFO marker: once the subscriber sees it, every earlier event was processed.
Barrier = define_event("wfminiapi.Barrier", schema=(str, SCHEMA), purpose=str)
