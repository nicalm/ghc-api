# Local image tools and Codex request byte budgets

Large inline images can exceed an HTTP request limit before a model reaches its
token context limit. This optional feature addresses image ingress and final
outbound bytes separately. It does not change model usage, compact conversations,
remove reasoning or automatically delete historical images.

Both features are disabled by default. They are part of the same ghc-api package
and service. Pillow is the only additional runtime dependency; the MCP SDK is
used only by the optional acceptance script.

## Configure the service

Merge these sections into the existing private `config.yaml`:

```yaml
image_tools:
  enabled: true
  allowed_roots:
    - /absolute/path/to/your/images
  # Empty uses <ghc-api-config-dir>/image-tools. Keep outside your Git checkout.
  data_dir: ""
  max_source_bytes: 33554432
  max_pixels: 20000000
  preview_max_bytes: 200000
  preview_max_edge: 1024
  context_budget_bytes: 4194304
  store_max_bytes: 268435456
  asset_ttl_seconds: 86400

codex_request_budget:
  enabled: true
  warn_bytes: 12582912
  max_bytes: 20971520
  rejected_ttl_seconds: 300
```

On Windows, use an absolute path such as `C:/MyImages` with forward slashes or a
YAML single-quoted Windows path. Allow only specific source directories, not an
entire drive. Invalid settings abort startup. Existing files are read only;
registration stores an immutable snapshot outside the repository. Sources must
be PNG, JPEG or WebP still images within the configured byte/pixel limits.

Without user-token authentication, `/mcp/images` accepts only loopback peer
addresses and loopback Host names. Cross-origin browser requests are rejected.
With `enable_auth: true`, it requires the existing approved user token and scopes
assets/budgets by that authenticated caller. A configured source directory is
available to all approved image-tool callers; separate service instances or
directory policies are needed if their filesystem access must differ. Do not
expose an unauthenticated local service through a reverse proxy.

## Register the MCP with Codex

```sh
codex mcp add ghc_images --url http://127.0.0.1:8313/mcp/images
```

For an authenticated deployment, also use
`--bearer-token-env-var GHC_IMAGES_TOKEN`, with that variable containing an
approved **ghc-api caller token**, not an upstream provider credential.

The equivalent Codex user configuration is:

```toml
[mcp_servers.ghc_images]
url = "http://127.0.0.1:8313/mcp/images"
```

Start a new task or reload its MCP connections after adding the configuration.
No Codex binary patch is needed. The existing model provider remains unchanged.
Codex's built-in `view_image` and other screenshot tools are still available;
they do not automatically become this preview tool. The outbound guard covers
the final size of their native Codex Responses requests too.

The endpoint implements stateless Streamable HTTP MCP with JSON responses:
`initialize`, `notifications/initialized`, `ping`, `tools/list` and `tools/call`.
It negotiates MCP `2025-03-26`, `2025-06-18` or `2025-11-25`, otherwise offers
`2025-06-18`. POST requires `Accept: application/json, text/event-stream`.
There is no standalone server-initiated stream or session ID; GET and DELETE
return 405. The existing Web IQ `/v3/mcp` proxy is independent.

## Tool workflow

1. `register_image(path)` returns an asset ID, SHA-256, oriented dimensions,
   source byte size and alpha/transparency metadata. It returns no binary image
   content. Resolve/read operations are limited to configured directories.
2. `describe_image(asset_id)` reads the caller's metadata without adding another
   image to context.
3. `preview_images(context_id, items)` returns one or two JPEG previews. Each
   item contains `asset_id`, optionally `crop: [left, top, right, bottom]` and
   `background: white | checkerboard`. Crop coordinates use EXIF-oriented source
   pixels. Each result includes actual preview dimensions, bytes and SHA-256.

Use the **same context ID for the entire current task**, including repeated
reads. Do not rotate IDs to bypass the budget. The server persists a cumulative
byte counter per authenticated caller/context ID; it cannot infer a Codex task
ID, prove that the caller supplied it correctly, or count images from other
tools. The final request byte guard is separate and does not rely on this ID.

Defaults are at most 200,000 decoded JPEG bytes per preview and at most two
previews per call. The cumulative context limit is 4 MiB of returned image
bytes, including repeated reads; Base64 in requests adds approximately one
third. Text and other tools also consume request capacity. No original is ever
returned as an over-budget fallback. If shrinking cannot preserve at least a
320-pixel long edge (or the original/crop is smaller), select a more useful crop.

JPEG previews composite transparency onto an explicit white/checkerboard
background. They are not alpha-channel evidence. Use the original metadata and
the original download for transparency checks. Originals on disk remain intact.
Registration snapshots expire after the configured TTL and the store has a
global byte/count cap. Expiry cleanup happens on the next registration/preview
budget operation. Unexpired assets are not silently evicted to satisfy a new
registration; capacity errors are explicit. Snapshots can be re-registered from
their original sources. No originals are deleted by cache cleanup.

Save visual observations in a small text progress file, linked to asset IDs and
hashes. Present complete image galleries as HTML or attachments for the user,
and inspect only the previews needed for the current decision. A new task can
continue from that text and the originals without inheriting every old image.

## Exact outbound request budget

Only direct `/v1/responses` and `/responses` calls carrying
`X-Codex-Turn-Metadata` are guarded when enabled. Other clients and configured
`/proxy/...` routes retain their existing behavior. The marker is a scope
selector, not an authentication guarantee. It may change in a future Codex
release; verify it during client upgrades.

The guard prepares JSON using Requests' actual encoder, counts the prepared
body bytes, and sends those same bytes. This includes JSON escaping, Unicode,
Base64, instructions, tool definitions and reasoning. Every native direct
Responses retry/recovery submission is checked again. It changes no fields in
the payload and preserves the existing truthful `x-reasoning-included`
accounting header behavior.

At the warning threshold, requests continue normally. The response includes
`X-GHC-Request-Bytes`, `X-GHC-Request-Budget` and `X-GHC-Budget-Warning`.
Above the hard limit, the client gets HTTP 400 with
`error.code=ghc_request_budget_exceeded`, byte statistics and recovery advice;
no upstream request is made. This is deliberately neither a context-window
error nor a successful completion. The default 12/20 MiB thresholds are local
engineering budgets, not a statement of upstream limits.

If an allowed request still gets upstream HTTP 413, the first response preserves
that HTTP status. For a short TTL, identical JSON content for the same
caller/task/upstream returns a local HTTP 400 with
`error.code=ghc_request_previous_413` instead of uploading again. JSON object
key order is ignored for identity; all field values, arrays, tools, images and
client metadata remain part of the fingerprint. Changed requests still pass
through normal checks. This in-memory cache is capped and clears on service
restart. Other HTTP errors are not cached by this feature.

The installed Codex CLI acceptance checks distinguish client attempts from
upstream calls: an over-budget request is terminal after one local attempt;
an upstream 413 may induce one client retry, then the local permanent error
stops it. Do not assume every client/version handles errors identically.

Budget events use a small rotating `request-budget.jl` metadata log (2 MiB plus
one backup). It stores size/count information and hashes, never prompts,
image data, paths, auth tokens or full response bodies. The first guarded 413
does not add the original large payload to the legacy error log/cache. Existing
ordinary request cache/logging settings still govern other requests.

## Verify and maintain

```sh
python -m pip install -r requirements.txt pytest
python -m pytest -q

# Optional acceptance: current Codex binary + MCP SDK + local synthetic upstream.
python -m pip install mcp
python scripts/verify_image_tools_codex.py --codex /path/to/codex --output acceptance.json
```

The acceptance script creates local synthetic images, an isolated Codex data
directory, temporary loopback services and fixed synthetic model responses.
It verifies MCP initialization/discovery/preview, a Codex register/preview tool
round, the exact byte guard and upstream 413 suppression. It does not contact a
real model, use the user's credentials or reload an existing conversation.

Keep this change in the fork's version history alongside prior fixes. On a
ghc-api upgrade, merge upstream, run the tests and rebuild the package. On a
Codex upgrade, repeat the client acceptance. Retain the old installation and
config backups until live health, accounting headers and image tools have been
verified. Switch a shared gateway only after its active and background upstream
requests have drained; do not interrupt unrelated clients.
