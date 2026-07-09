# Hermes Agent Endpoint Setup

BashGym can connect a canvas node to a local or remote Hermes Agent API server.
The node stores API keys server-side, probes the Hermes capability surfaces, and
sends workspace context to Hermes without baking in any private machine names.

## Hermes side

Use the official Hermes docs as the source of truth:

- Docs home: https://hermes-agent.nousresearch.com/docs
- API server: https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server
- MCP: https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp
- Security: https://hermes-agent.nousresearch.com/docs/user-guide/security

Typical local setup:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
hermes setup --portal
hermes config set API_SERVER_ENABLED true
# Store API_SERVER_KEY in the path printed by: hermes config env-path
hermes gateway
```

Hermes defaults to `http://127.0.0.1:8642`. BashGym normalizes endpoint URLs to
the OpenAI-compatible `/v1` surface, so either of these are accepted:

```text
http://127.0.0.1:8642
http://127.0.0.1:8642/v1
```

## BashGym side

Open the Workspace canvas and add the **Hermes Agent** node.

For a fresh local install, click **Setup**. BashGym will:

- Ask Hermes for the real config and env paths with `hermes config path` and
  `hermes config env-path`.
- Enable the API server via `hermes config set API_SERVER_ENABLED true`.
- Generate or reuse an API key, write it directly to Hermes' discovered `.env`
  path, and save the same key in BashGym's server-side secret store.
- Save the canvas endpoint profile and start `hermes gateway run`.

This path discovery matters on Windows: Hermes may use
`%LOCALAPPDATA%\hermes\.env`, not `~/.hermes/.env`.

Manual configuration is still available:

- Endpoint id: stable local profile id such as `hermes` or `lab-hermes`.
- URL: Hermes API server URL, usually `http://127.0.0.1:8642/v1`.
- Model: usually `hermes-agent`, or the profile/model name exposed by Hermes.
- Session key: optional memory scope passed as `X-Hermes-Session-Key`.
- API key: stored server-side, never returned to the renderer.

Click **Save**, then **Test**. The test calls:

- `GET /health`
- `GET /v1/capabilities`
- `GET /v1/models`
- `GET /v1/skills`
- `GET /v1/toolsets`

When the node is linked to terminals or other canvas nodes, its context handoff
includes current panel connections, training run state, capability probe counts,
and useful BashGym API handles.

## Tool Kit node

The **Tool Kit** canvas node is the capability inventory surface for BashGym and
connected agents. It is intentionally read-first: it shows what exists and builds
handoff context for linked terminals, while execution still happens through the
agent's own approval model.

It collects:

- Local skills from configured skill roots.
- BashGym/Peony tools and skill-manifest tools.
- Connected Hermes-compatible endpoint models, skills, and toolsets.
- A skill workshop prompt for creating or revising skills with Claude, Codex, or
  Hermes in a linked terminal.

The backend endpoint is:

```text
GET /api/agent/toolkit
GET /api/agent/toolkit?include_remote=false
GET /api/agent/toolkit?refresh=true
```

Remote endpoint probing is cached for 60 seconds. Endpoints without configured
API keys are listed but not probed.

Skill roots are discovered from:

- `BASHGYM_SKILL_DIRS` (path-separated override)
- `assistant/workspace/skills`
- `~/.agents/skills`
- `~/.codex/skills`
- `~/.codex/skills/.system`
- `~/.claude/skills`

## Environment defaults

For unattended local setups, BashGym also reads:

```bash
export HERMES_API_BASE="http://127.0.0.1:8642/v1"
export HERMES_API_KEY="replace-me"
export HERMES_MODEL="hermes-agent"
export HERMES_SESSION_KEY="bashgym"
```

For custom profiles, the saved secret key is:

```text
AGENT_ENDPOINT_<PROFILE_ID>_API_KEY
```

Example for `lab-hermes`:

```bash
export AGENT_ENDPOINT_LAB_HERMES_API_KEY="replace-me"
```

## Security posture

Hermes API server access can expose powerful agent tools, including terminal
actions depending on the Hermes profile. Keep it bound to localhost unless you
have an explicit network security model, use a strong API key, and keep Hermes
approval settings enabled for dangerous commands.

For open-source deployments, prefer:

- User-configured endpoint URLs instead of hardcoded hostnames.
- Server-side secret storage instead of browser-local keys.
- Session keys for memory scoping between projects.
- Capability probing before exposing higher-level workflow buttons.
- Context handoff first, then explicit execution through Hermes' own approval
  model.
