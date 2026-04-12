"""Compatibility layer for OpenEnv server/client primitives.

This project targets the current Meta OpenEnv API shape, but falls back to a
minimal local implementation when the official package is unavailable. That
keeps the repo runnable on older local Python environments while the Docker
image installs the official dependency on Python 3.11.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar
from uuid import uuid4

import requests
from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:  # pragma: no cover - exercised in Docker / official runtime
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State

    OPENENV_AVAILABLE = True
except Exception:  # pragma: no cover - local fallback path tested
    OPENENV_AVAILABLE = False

    class Action(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)

        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)

        done: bool = False
        reward: float = 0.01
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        model_config = ConfigDict(extra="allow", validate_assignment=True)

        episode_id: Optional[str] = None
        step_count: int = 0

    class Environment(ABC):
        SUPPORTS_CONCURRENT_SESSIONS: bool = False

        @abstractmethod
        def reset(
            self,
            seed: Optional[int] = None,
            episode_id: Optional[str] = None,
            **kwargs: Any,
        ) -> Observation:
            raise NotImplementedError

        @abstractmethod
        def step(
            self,
            action: Action,
            timeout_s: Optional[float] = None,
            **kwargs: Any,
        ) -> Observation:
            raise NotImplementedError

        @property
        @abstractmethod
        def state(self) -> State:
            raise NotImplementedError

        def get_metadata(self) -> Dict[str, Any]:
            return {
                "name": self.__class__.__name__,
                "description": f"{self.__class__.__name__} environment",
                "version": "1.0.0",
            }

        def close(self) -> None:
            return None

    ObsT = TypeVar("ObsT")
    ActT = TypeVar("ActT")
    StateT = TypeVar("StateT")

    @dataclass
    class StepResult(Generic[ObsT]):
        observation: ObsT
        reward: float
        done: bool

    class EnvClient(ABC, Generic[ActT, ObsT, StateT]):
        """Minimal sync HTTP client fallback."""

        def __init__(self, base_url: str, **_: Any):
            self.base_url = base_url.rstrip("/")

        def _request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
            response = requests.request(method, f"{self.base_url}{path}", json=payload, timeout=30)
            response.raise_for_status()
            return response.json()

        @abstractmethod
        def _step_payload(self, action: Any) -> Dict[str, Any]:
            raise NotImplementedError

        @abstractmethod
        def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
            raise NotImplementedError

        @abstractmethod
        def _parse_state(self, payload: Dict[str, Any]) -> Any:
            raise NotImplementedError

        def reset(self, **kwargs: Any) -> StepResult:
            return self._parse_result(self._request("POST", "/reset", kwargs))

        def step(self, action: Any, **_: Any) -> StepResult:
            return self._parse_result(self._request("POST", "/step", {"action": self._step_payload(action)}))

        def state(self) -> Any:
            return self._parse_state(self._request("GET", "/state"))

        def close(self) -> None:
            return None

    class _HealthStatus(str, Enum):
        HEALTHY = "healthy"

    class _HealthResponse(BaseModel):
        status: _HealthStatus = _HealthStatus.HEALTHY

    class _SchemaResponse(BaseModel):
        action: Dict[str, Any]
        observation: Dict[str, Any]
        state: Dict[str, Any]

    def _serialize_observation(observation: Observation) -> Dict[str, Any]:
        obs_dict = observation.model_dump(exclude={"reward", "done"})
        return {
            "observation": obs_dict,
            "reward": observation.reward,
            "done": observation.done,
        }

    def create_app(
        env_factory: Callable[[], Environment] | Type[Environment],
        action_cls: Type[Action],
        observation_cls: Type[Observation],
        env_name: str,
    ) -> FastAPI:
        app = FastAPI(title=env_name, version="1.0.0")
        shared_env = None

        def make_env() -> Environment:
            if isinstance(env_factory, type):
                return env_factory()
            return env_factory()

        def get_shared_env() -> Environment:
            nonlocal shared_env
            if shared_env is None:
                shared_env = make_env()
            return shared_env

        def state_schema() -> Dict[str, Any]:
            env = get_shared_env()
            try:
                return type(env.state).model_json_schema()
            except Exception:
                return State.model_json_schema()

        @app.get("/health")
        async def health() -> _HealthResponse:
            return _HealthResponse()

        @app.get("/metadata")
        async def metadata() -> Dict[str, Any]:
            env = get_shared_env()
            try:
                meta = env.get_metadata()
                if isinstance(meta, BaseModel):
                    return meta.model_dump()
                return dict(meta)
            except Exception:
                return {"name": env_name, "description": f"{env_name} environment"}

        @app.get("/schema")
        async def schema() -> _SchemaResponse:
            return _SchemaResponse(
                action=action_cls.model_json_schema(),
                observation=observation_cls.model_json_schema(),
                state=state_schema(),
            )

        @app.get("/state")
        async def state() -> Dict[str, Any]:
            return get_shared_env().state.model_dump()

        @app.post("/reset")
        async def reset(request: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
            observation = get_shared_env().reset(**request)
            return _serialize_observation(observation)

        @app.post("/step")
        async def step(request: Dict[str, Any]) -> Dict[str, Any]:
            if "action" not in request:
                raise HTTPException(status_code=422, detail="Field required: action")
            env = get_shared_env()
            action = action_cls.model_validate(request["action"])
            observation = env.step(action)
            return _serialize_observation(observation)

        @app.post("/mcp")
        async def mcp(request: Request) -> Dict[str, Any]:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            return {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "error": {
                    "code": -32601,
                    "message": "MCP tools are not implemented for this simulation environment.",
                },
            }

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            env = make_env()
            try:
                while True:
                    raw = await websocket.receive_text()
                    message = json.loads(raw)
                    msg_type = message.get("type")
                    if msg_type == "reset":
                        observation = env.reset(**message.get("data", {}))
                        await websocket.send_json(
                            {"type": "observation", "data": _serialize_observation(observation)}
                        )
                    elif msg_type == "step":
                        action = action_cls.model_validate(message.get("data", {}))
                        observation = env.step(action)
                        await websocket.send_json(
                            {"type": "observation", "data": _serialize_observation(observation)}
                        )
                    elif msg_type == "state":
                        await websocket.send_json({"type": "state", "data": env.state.model_dump()})
                    elif msg_type == "close":
                        break
                    else:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "data": {
                                    "message": f"Unknown message type: {msg_type}",
                                    "code": "UNKNOWN_TYPE",
                                },
                            }
                        )
            except WebSocketDisconnect:
                pass
            finally:
                env.close()

        return app


ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
StateT = TypeVar("StateT")
