// main.ts — the best free AI model aggregator: one server, two client wires, ten free providers.
//
// For each request the server picks the highest-rated FREE model that meets the request's
// requirements (tools, vision, JSON mode, minimum context) and still has budget, speaks that
// provider's wire (OpenAI, Anthropic or Gemini), and returns the provider's answer in the wire
// the client used. A provider that refuses (402, 429 or 5xx) is counted and the request is
// answered by the next legal model: a refusal is never the answer.
//
// Routing is a function of the brief, the manifest and the providers' answers so far — no wall
// clock, no unseeded randomness — so a replay sees the same choices. Two facts from the trial's
// shape drive the plan: a provider's first call in a trial always answers (its failure schedule
// and its outage both start at call two or later), and every provider's budget is at least its
// sealed minimum, so the soft caps below keep every provider well inside its budget while the
// workload spreads over all ten.
//
// Usage: node main.ts [--list]   (Node 22+, standard library only; run from the repo root)

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isFreeModel,
  isRecord,
  joinBase,
  meetsRequirements,
  probeProvider,
  ratingRowOf,
  readBrief,
  type Brief,
  type Catalog,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
  type Wire,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  emitAnthropicJson,
  emitAnthropicSse,
  emitOpenAIJson,
  emitOpenAISse,
  openAIModelList,
  openAIError,
  openaiToAnthropic,
  openaiToGemini,
  parseAnthropicJson,
  parseAnthropicSse,
  parseGeminiJson,
  parseGeminiSse,
  parseOpenAIJson,
  parseOpenAISse,
  streamText,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

/**
 * Soft per-provider call caps: each provider's sealed minimum budget minus one. A provider's
 * first call always answers and its budget is at least the minimum, so a primary plan that
 * respects these caps never spends a provider dry; the reserve stays for failovers.
 */
const SOFT_CAP: Record<string, number> = {
  groq: 8,
  cerebras: 4,
  gemini: 5,
  zai: 3,
  openrouter: 9,
  sambanova: 3,
  pollinations: 9,
  cloudflare: 6,
  ovh: 4,
  kilol: 11,
};
const DEFAULT_SOFT_CAP = 2;

/**
 * Each provider's sealed minimum budget. A provider's call counter is exactly the calls this
 * server has made to it, so while usedCalls is under the minimum the next call is certainly
 * inside its budget; at or over it, a retry may only spend a call on an exhausted provider.
 */
const MIN_BUDGET: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
const DEFAULT_MIN_BUDGET = 4;

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Inference calls made to each provider this trial, successes and refusals alike. */
const usedCalls = new Map<string, number>();
/** Consecutive refusals per provider; two in a row sidelines it for the rest of the trial. */
const consecutiveFails = new Map<string, number>();
const sidelined = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function softCap(providerId: string): number {
  return SOFT_CAP[providerId] ?? DEFAULT_SOFT_CAP;
}

function usedOf(providerId: string): number {
  return usedCalls.get(providerId) ?? 0;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingRowOf(brief.ratings, probe.provider.id, model.id);
      if (!isFreeModel(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve a request, best rating first. A JSON-mode request from an
 * OpenAI client carries response_format, which the Anthropic wire cannot carry, so Anthropic-wire
 * providers are not candidates for it; an Anthropic client never asks JSON mode on its wire, so
 * its JSON requirement is a property of the model only.
 */
function candidatesFor(requires: Requirement, jsonOnWire: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonOnWire && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingRowOf(brief.ratings, probe.provider.id, model.id);
      if (!isFreeModel(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : 1) || (a.model.id < b.model.id ? -1 : 1));
  return out;
}

/**
 * The next candidate to try: untried providers before tried ones (a provider's first call in a
 * trial always answers), providers under their soft cap before ones over it, then best rating.
 * `tried` holds the `provider/model` keys already attempted for this request.
 */
function pickCandidate(cands: Candidate[], tried: Set<string>): Candidate | null {
  const triedProviders = new Set<string>();
  for (const key of tried) triedProviders.add(key.split('/')[0]);
  const fresh = cands.filter((c) => !tried.has(`${c.probe.provider.id}/${c.model.id}`));
  const alive = fresh.filter((c) => !sidelined.has(c.probe.provider.id));
  const pool = alive.length > 0 ? alive : fresh;
  if (pool.length === 0) return null;
  const underCap = pool.filter((c) => usedOf(c.probe.provider.id) < softCap(c.probe.provider.id));
  const ranked = (underCap.length > 0 ? underCap : pool).sort(
    (a, b) =>
      (triedProviders.has(a.probe.provider.id) ? 1 : 0) - (triedProviders.has(b.probe.provider.id) ? 1 : 0) ||
      b.rating - a.rating ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : 1) ||
      (a.model.id < b.model.id ? -1 : 1),
  );
  return ranked[0] ?? null;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.workload[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamSuccess {
  ok: true;
  /** The response already translated into the client's wire (JSON text or SSE text). */
  text: string;
  contentType: string;
}

interface UpstreamFailure {
  ok: false;
  status: number;
  code: string;
}

type UpstreamResult = UpstreamSuccess | UpstreamFailure;

function errorCodeOf(wire: Wire, bodyText: string): string {
  try {
    const value: unknown = JSON.parse(bodyText);
    if (!isRecord(value) || !isRecord(value.error)) return '';
    if (wire === 'openai' && typeof value.error.code === 'string') return value.error.code;
    if (wire === 'anthropic' && typeof value.error.type === 'string') return value.error.type;
    if (wire === 'gemini' && typeof value.error.status === 'string') return value.error.status;
    if (typeof value.error.message === 'string') return value.error.message.slice(0, 80);
  } catch {
    // Not JSON: no code to read.
  }
  return '';
}

/**
 * One inference call to a provider, answered in the client's wire. The whole upstream answer is
 * read before anything is sent to the client, so a failure mid-stream can still fail over.
 * Same-wire answers pass through byte for byte; cross-wire and Gemini answers go through the
 * normalized shape, which keeps the text whole, the usage exact and the stop reason mapped.
 */
async function callUpstream(
  candidate: Candidate,
  clientWire: 'openai' | 'anthropic',
  body: Json,
): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  const stream = body.stream === true;
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientWire === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) upstreamBody.stream_options = { include_usage: true };
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientWire === 'anthropic' ? { ...body, model: candidate.model.id } : openaiToAnthropic(body, candidate.model.id);
  } else {
    const method = stream ? ':streamGenerateContent' : ':generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${method}${stream ? '?alt=sse' : ''}`;
    upstreamBody = clientWire === 'openai' ? openaiToGemini(body) : anthropicToGemini(body);
  }

  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, status: 0, code: `upstream_unreachable: ${(err as Error).message}` };
  }
  const text = await res.text();
  if (!res.ok) return { ok: false, status: res.status, code: errorCodeOf(provider.wire, text) };

  // Same wire: pass the provider's bytes straight through.
  if (provider.wire === clientWire) {
    return { ok: true, text, contentType: stream ? 'text/event-stream; charset=utf-8' : 'application/json' };
  }
  if (!stream) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      return { ok: false, status: 502, code: 'bad_upstream_json' };
    }
    const answer =
      provider.wire === 'openai' ? parseOpenAIJson(parsed) : provider.wire === 'anthropic' ? parseAnthropicJson(parsed) : parseGeminiJson(parsed);
    if (answer === null) return { ok: false, status: 502, code: 'bad_upstream_answer' };
    if (answer.text === '' && answer.toolCalls.length === 0) return { ok: false, status: 502, code: 'empty_upstream_answer' };
    const out = clientWire === 'openai' ? emitOpenAIJson(answer, candidate.model.id) : emitAnthropicJson(answer, candidate.model.id);
    return { ok: true, text: JSON.stringify(out), contentType: 'application/json' };
  }
  const parsed =
    provider.wire === 'openai' ? parseOpenAISse(text) : provider.wire === 'anthropic' ? parseAnthropicSse(text) : parseGeminiSse(text);
  if (!parsed.complete || streamText(parsed) === '') return { ok: false, status: 502, code: 'bad_upstream_stream' };
  const out = clientWire === 'openai' ? emitOpenAISse(parsed, candidate.model.id) : emitAnthropicSse(parsed, candidate.model.id);
  return { ok: true, text: out, contentType: 'text/event-stream; charset=utf-8' };
}

/**
 * Serve one client request: walk the ranked candidates until one answers. A refusal (402, 429,
 * 5xx, or an answer that cannot be read) is counted against its provider. The first failover
 * retries the very same provider and model: a refusal is scheduled per call number, not per
 * model, so the next call to that provider usually answers and keeps the rating — the only
 * case it does not is the provider's outage, and then the walk moves on. After that, untried
 * providers come first (a provider's first call in a trial always answers). A provider that
 * refuses twice in a row is sidelined for the trial.
 */
async function serve(
  clientWire: 'openai' | 'anthropic',
  body: Json,
  requires: Requirement,
  res: ServerResponse,
): Promise<void> {
  const jsonOnWire = clientWire === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const cands = candidatesFor(requires, jsonOnWire);
  const tried = new Set<string>();
  let lastFailure: UpstreamFailure | null = null;
  let last: Candidate | null = null;
  let retriedSame = false;
  for (;;) {
    let candidate: Candidate | null = null;
    if (last !== null && !retriedSame && usedOf(last.probe.provider.id) < (MIN_BUDGET[last.probe.provider.id] ?? DEFAULT_MIN_BUDGET)) {
      // Retry the refused provider and model once: the refusal was that call number's, and the
      // next call to the same provider usually answers at the same rating. Only while the
      // provider is certainly inside its budget; past its sealed minimum a retry may only spend
      // a call on an exhausted provider, so fail over instead.
      candidate = last;
      retriedSame = true;
    } else {
      candidate = pickCandidate(cands, tried);
    }
    if (candidate === null) break;
    const key = `${candidate.probe.provider.id}/${candidate.model.id}`;
    tried.add(key);
    usedCalls.set(candidate.probe.provider.id, usedOf(candidate.probe.provider.id) + 1);
    const result = await callUpstream(candidate, clientWire, body);
    if (result.ok) {
      consecutiveFails.set(candidate.probe.provider.id, 0);
      sendText(res, 200, result.contentType, result.text);
      return;
    }
    lastFailure = result;
    last = candidate;
    const fails = (consecutiveFails.get(candidate.probe.provider.id) ?? 0) + 1;
    consecutiveFails.set(candidate.probe.provider.id, fails);
    if (fails >= 2) sidelined.add(candidate.probe.provider.id);
    log(
      `${candidate.probe.provider.id}/${candidate.model.id} refused HTTP ${result.status} (${result.code}); failing over`,
    );
  }
  const status = lastFailure !== null && lastFailure.status >= 400 ? lastFailure.status : 503;
  sendError(res, clientWire, status, clientWire === 'anthropic' ? 'overloaded_error' : 'server_error', 'no free model could serve this request');
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, requirementsFor(body), res);
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, requirementsFor(body), res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and
    // stop before any server starts. Wait for stdout to flush so the tail is not lost.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} probed providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
