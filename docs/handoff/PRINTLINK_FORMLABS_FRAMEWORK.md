# PrintLink — Blender → Formlabs printers (framework & findings)
Status: PARKED until Tessera 0.7.0 ships. Prototype exists and works against a stand-in server.
Written 2026-11-26.

## Idea
Blender becomes the prep + dispatch front end: connect, sign in to Formlabs Dashboard, see the whole printer
fleet, prepare parts (orient / support / layout / validate / print-time) and queue the job, all from the sidebar.

## What Formlabs actually provides (verified from docs + SDK repo)
- **PreForm Local API** (v1.0, PreForm 3.64+): HTTP/JSON server on localhost:44388, run by PreForm
  ("File > Start PreForm Server", Insider build) or headless PreForm Server (Windows/macOS; Linux only via
  Docker+Wine). Scenes, import, orient, support, layout, pack, hollow, labels, validate (manifold, cups,
  minima, thin features, bounds, interferences), print time, devices, print_scene, .form save/open, screenshots.
- Long operations: POST ?async=true -> 202 + operation_id -> poll /v1/operation/get/.
- **Login**: POST /v1/login/ {username,password} or access_token; GET /v1/user/ (404 = server up, nobody logged in);
  POST /v1/logout/. Login lives in PreForm; Blender never needs to keep tokens.
- **Web API** (cloud, api.formlabs.com, OAuth client_id/secret from dashboard.formlabs.com/#developer):
  printer status, consumables, print history. Separate from the Local API.
- Python SDK `formlabs-sdk` is proprietary and depends on a wheel from downloads.formlabs.com (not on PyPI).
  Recommendation: do NOT bundle it; the extension speaks plain HTTP (stdlib urllib) to the local server.

## Architecture
Blender extension (stdlib only) --HTTP--> PreForm Server on the user's machine --> Formlabs Dashboard / printers
- Connection panel: host/port, Connect (probe GET /v1/user/), status.
- Account panel: email/password sign-in via /v1/login/; sign-out; "Create account" button (see open items).
- Printers panel: fleet list from devices endpoint, pick target, status/material/tank/queue.
- Job panel: machine type, material, layer, job name, queue-only vs start, build-volume fit check, progress + log.
- Send: export selection to STL (mm) -> create scene -> import -> auto-orient -> auto-support -> auto-layout ->
  validate -> print time/resin -> print_scene (print_now=False by default).
- Network work runs on a worker thread; UI redraws from a timer.
- Password property is SKIP_SAVE so it is never written to the .blend.

## Universal-user flow (requirement from Justin)
1. Install extension (+ PreForm/PreForm Server, which the user installs themselves).
2. Extension menu: Connect -> Sign in (or Create account) -> pick printer -> work in the Blender scene -> Send.

## Prototype (this session)
Files (saved with this doc): printlink/__init__.py, printlink/client.py, mock_preform_server.py, test_headless.py,
capture_demo.py.
- Headless Blender 5.3 test PASSED against the stand-in server: connect -> sign in -> 3 printers listed ->
  scene/import/orient/support/layout/validate/print-time (0.6 h, 20.2 mL) -> job queued on chosen printer.
- The stand-in server is my own mock, NOT Formlabs software and not the real API.
- Screen recording was started but interrupted; not produced.

## Verified vs assumed
- VERIFIED paths: /v1/login/, /v1/logout/, /v1/user/, /v1/operation/get/, /v1/scene/model/auto-orient/,
  /v1/scene/save/, /v1/scene/open/, /v1/print-setting/list/, /v1/scene/get/, /v1/scene/model/list/.
- ASSUMED (inferred from SDK operation names; client.py marks each): scene create, import, auto-support,
  auto-layout, validate, print-time, devices list, print, request/response field names, machine codes other than
  FORM-4-0, material codes other than FLGPGR05.
- The real OpenAPI spec lives inside the wheel formlabs_preform_api (downloads.formlabs.com returned 403 here),
  so it must be fetched on a machine that can, or against a running PreForm Server.

## Could not do here
- Real PreForm Server not run: the Windows installer (PreForm 3.66.0.1163, WiX bundle -> 360 MB MSI) unpacked fine,
  but Wine 9 crashed while creating its prefix ("free(): invalid pointer"). Not retried. Options: full wine with
  wow64, Formlabs' Docker+Wine instructions, or run the test on the user's own PC/Mac.
- Mac DMG too large to upload; not needed if PreForm Server runs on the user's own machine.
- No live Dashboard login attempted.

## Open items / risks
1. Account creation in-menu: no documented API for sign-up. Realistic options: button opens Formlabs sign-up in the
   browser, then user returns and signs in. In-extension form would need Formlabs' approval/an endpoint.
2. License (Formlabs API License Agreement, in the SDK repo): limited, revocable, non-sublicensable; no competing
   products; cannot distribute Formlabs offerings stand-alone (don't bundle PreForm/SDK); distribution needs an EULA
   at least as protective and no print-outcome guarantees; no implying partnership/endorsement without written
   approval; trademarks only to promote the integration, per their guidelines; Formlabs may monitor use, change the API,
   and require updates (critical patches within 30 days). Get Justin's read / Formlabs contact before publishing.
3. Blender extension store review: network permission needed, reason <=64 chars (e.g. "Talk to local PreForm Server").
4. Printers not on Dashboard / local-network only: Local API can address printers by IP or serial too.
5. Linux users: PreForm Server only via Docker+Wine.
6. Scale: Blender units -> mm via scene unit settings; verify with real server.

## Next steps (after 0.7.0)
1. Run PreForm Server on a real Windows/Mac machine; dump its OpenAPI spec; replace ASSUMED routes/fields.
2. Test login/devices with the real test account (do not store credentials in repo or extension).
3. Add: scene preview screenshot back into Blender, per-object print settings, multi-object builds,
   fleet status refresh, job history, material/print-setting pickers from the server, error panel.
4. Record the demo against the real server; then package as extension `printlink` and review license/naming.
