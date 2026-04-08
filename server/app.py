"""FastAPI app entrypoint for the tabular cleaning environment."""

from __future__ import annotations

from tabular_cleaning_env.models import TabularCleaningAction, TabularCleaningObservation
from tabular_cleaning_env.openenv_compat import create_app

from .environment import TabularCleaningEnvironment

app = create_app(
    TabularCleaningEnvironment,
    TabularCleaningAction,
    TabularCleaningObservation,
    env_name="tabular_cleaning_env",
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
