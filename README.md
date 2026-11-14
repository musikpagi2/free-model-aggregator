# The best free AI model aggregator

One Node 22 server (standard library only) that serves the OpenAI Chat Completions API and
the Anthropic Messages API over ten simulated free-tier providers, on the OpenAI, Anthropic
and Gemini wires, in JSON and in server-sent events.

## How it works

- At trial start it reads the brief (`$MAKEFUN_PROMPT_FILE`), probes every provider's catalog
  route in its own wire, and builds one catalog — the document `node main.ts --list` prints
  and the register body carries, byte for byte.
- It listens on a port it picks, registers `{protocol_version: 1, port, catalog}` with the env,
  serves the env client's 20 requests one at a time, polls `GET $MAKEFUN_ENV_URL/work`, and
  exits 0 when the workload is done.
- `GET /v1/models` answers in the shape of its auth header: `Authorization: Bearer` gets the
  OpenAI list, `x-api-key` the Anthropic one.

## Routing

For each request the server ranks every free model that meets the request's requirements
(tools, vision, JSON mode, minimum context) by the quality table's rating and picks the best
one whose provider is still under its soft cap — each provider's sealed minimum budget minus
two, so no provider is ever spent dry. A JSON-mode request from an OpenAI client only goes to
a wire that can carry `response_format` (OpenAI or Gemini, never Anthropic).

On a refusal (402, 429, 5xx, or an unreadable answer) the call is counted against its
provider and the request is answered by the next legal model; a provider that refuses twice
in a row is sidelined for the rest of the trial. A provider's first call in a trial always
answers, so failover to an untried provider is the first choice after a failure.

## Wires

Requests pass through untouched: system prompt, user text (images included), tools, stop
sequences, `max_tokens`, JSON mode. Responses return the provider's text whole, its usage
counts as that wire reports them, and its stop reason mapped (`end_turn`→`stop`,
`max_tokens`→`length`, `tool_use`→`tool_calls`, `refusal`→`content_filter`). Upstream
answers are read whole before anything is sent to the client, so a mid-stream failure can
still fail over. Streamed OpenAI calls carry `stream_options.include_usage` so the usage
chunk the client checks is there.

## Determinism

Every choice is a function of the brief, the manifest and the providers' answers so far:
no wall clock, no unseeded randomness. A replay sees the same requests and the same provider
answers, and leaves the same state export.

## Files

- `main.ts` — the server, routing, failover, register and poll.
- `lib/brief.ts` — the brief's three documents, the catalog probe, the catalog.
- `lib/wire.ts` — OpenAI ↔ Anthropic ↔ Gemini request/response translation, JSON and SSE.

## License

MIT (see LICENSE).
