"""FastAPI app entrypoint for the tabular cleaning environment."""

from __future__ import annotations

from fastapi.responses import HTMLResponse

from tabular_cleaning_env.models import TabularCleaningAction, TabularCleaningObservation
from tabular_cleaning_env.openenv_compat import create_app
from tabular_cleaning_env.tasks import TASKS

from .environment import TabularCleaningEnvironment

app = create_app(
    TabularCleaningEnvironment,
    TabularCleaningAction,
    TabularCleaningObservation,
    env_name="tabular_cleaning_env",
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    task_cards = "".join(
        f"""
        <article class="card">
          <div class="eyebrow">{task.difficulty.title()} • {task.domain.title()}</div>
          <h2>{task.task_id}</h2>
          <p>{task.description}</p>
          <div class="meta">Max steps: {task.max_steps}</div>
        </article>
        """
        for task in TASKS.values()
    )
    html = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>tabular_cleaning_env</title>
        <style>
          :root {{
            color-scheme: light;
            --ink: #102a43;
            --muted: #486581;
            --bg: #f4f7fb;
            --panel: #ffffff;
            --accent: #1f7a8c;
            --accent-soft: #d9f0f4;
            --border: #d9e2ec;
          }}
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
            color: var(--ink);
            background:
              radial-gradient(circle at top right, #e7f7fb, transparent 32%),
              linear-gradient(180deg, #fdfefe 0%, var(--bg) 100%);
          }}
          main {{ max-width: 1080px; margin: 0 auto; padding: 48px 20px 72px; }}
          .hero {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 28px;
            box-shadow: 0 18px 40px rgba(16, 42, 67, 0.08);
          }}
          .badge {{
            display: inline-block;
            background: var(--accent-soft);
            color: var(--accent);
            border-radius: 999px;
            padding: 8px 14px;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
          }}
          h1 {{
            margin: 16px 0 12px;
            font-size: clamp(2rem, 4vw, 3.3rem);
            line-height: 1.05;
          }}
          p {{
            margin: 0;
            color: var(--muted);
            line-height: 1.65;
            font-size: 1rem;
          }}
          .links, .grid {{
            display: grid;
            gap: 16px;
          }}
          .links {{
            margin-top: 24px;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
          }}
          .grid {{
            margin-top: 28px;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          }}
          .card, .link-card, .code-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 20px;
          }}
          .link-card {{
            text-decoration: none;
            color: inherit;
            transition: transform 120ms ease, box-shadow 120ms ease;
          }}
          .link-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 24px rgba(16, 42, 67, 0.08);
          }}
          .eyebrow {{
            color: var(--accent);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 10px;
          }}
          .meta {{
            margin-top: 14px;
            color: var(--ink);
            font-weight: 600;
          }}
          pre {{
            margin: 14px 0 0;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-word;
            background: #0f172a;
            color: #e2e8f0;
            border-radius: 16px;
            padding: 16px;
            font-size: 13px;
            line-height: 1.6;
          }}
          .section-title {{
            margin: 32px 0 14px;
            font-size: 1.15rem;
            font-weight: 700;
          }}
        </style>
      </head>
      <body>
        <main>
          <section class="hero">
            <span class="badge">OpenEnv Ops Workbench</span>
            <h1>tabular_cleaning_env</h1>
            <p>
              A deterministic human-in-the-loop cleanup workbench for messy operational exports. Agents profile
              source data, apply structured cleanup actions, review risky changes, run validation gates, and
              export or publish audited tables for downstream systems.
            </p>
            <div class="links">
              <a class="link-card" href="/docs"><div class="eyebrow">Explore</div><strong>API Docs</strong><p>Interactive FastAPI and OpenEnv schema explorer.</p></a>
              <a class="link-card" href="/metadata"><div class="eyebrow">Inspect</div><strong>Metadata</strong><p>Benchmark metadata, versioning, and description.</p></a>
              <a class="link-card" href="/schema"><div class="eyebrow">Inspect</div><strong>Schema</strong><p>Action, observation, and state JSON schemas.</p></a>
              <a class="link-card" href="/health"><div class="eyebrow">Monitor</div><strong>Health</strong><p>Container health endpoint for validators and deployments.</p></a>
            </div>
          </section>
          <div class="section-title">Bundled Tasks</div>
          <section class="grid">{task_cards}</section>
          <div class="section-title">Example Action</div>
          <section class="code-card">
            <p>Workflow actions and cleanup actions use the same typed interface.</p>
            <pre>{{"action_type":"approve_changes","change_id":"chg-002"}}</pre>
          </section>
          <div class="section-title">Example Inference Logs</div>
          <section class="code-card">
            <p>The baseline emits parser-safe logs in the required START / STEP / END format.</p>
            <pre>[START] task=easy_contacts_cleanup env=tabular_cleaning_env model=gpt-4.1-mini
[STEP] step=1 action={{"action_type":"profile_table"}} reward=0.00 done=false error=null
[STEP] step=2 action={{"action_type":"rename_column","column":"full_name","new_name":"name"}} reward=0.05 done=false error=null
[STEP] step=3 action={{"action_type":"approve_changes","change_id":"chg-001"}} reward=0.00 done=false error=null
[END] success=true steps=13 rewards=0.00,0.05,0.00,0.08,0.06,0.04,0.04,0.14,0.03,0.00,0.00,0.00,0.00</pre>
          </section>
        </main>
      </body>
    </html>
    """
    return HTMLResponse(html)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
