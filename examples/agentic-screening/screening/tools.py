"""The simulation tool's arguments, as the model sees them, and the budget that refuses an agent's
simulations past `Benchmark.max_sims`."""

from __future__ import annotations

from flowgentic.tools import TaskInfo
from pydantic import BaseModel, Field


class SimulateArgs(BaseModel):
    """Run a simulation of the material at this synthesis temperature (K) and dopant fraction. Returns a result_key."""

    temperature: float = Field(description="synthesis temperature, K")
    doping: float = Field(description="dopant fraction")


class Budget:
    """Refuses an agent's simulations past `most`: the agent reads the refusal as the tool's error."""

    def __init__(self, most: int) -> None:
        self.most = most
        self.used: dict[str, int] = {}
        self.refused = 0

    async def admit(self, info: TaskInfo) -> None:
        n = self.used.get(info.agent_id, 0)
        if n >= self.most:
            self.refused += 1
            raise RuntimeError(
                f"simulation budget spent: at most {self.most} simulations per material"
            )
        self.used[info.agent_id] = n + 1

    def done(self, info: TaskInfo) -> None:
        return None
