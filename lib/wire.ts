// lib/wire.ts — the three wires and every translation between them.
//
// The client speaks OpenAI or Anthropic; a provider speaks OpenAI, Anthropic or Gemini. Every
// translation here is a pure function of its input: no clock, no randomness, so a replay that sees
// the same provider answers produces the same bytes. The translations preserve exactly what the
// trial's client compares: the system prompt, the user text (images included), the tool names, the
// stop sequences, max_tokens and JSON mode. The provider echoes what it parsed, and the client
// compares that echo with what it sent: a field rewritten or dropped loses the request.
//
// Responses are buffered whole before they are translated and sent: the simulated providers answer
// fast and in full, and buffering means a mid-stream failure can still fail over to another model
// instead of leaving the client a half answer.

import { isRecord, type Json } from './brief.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in a reply reads the clock. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

// ---------------------------------------------------------------------------
// Small readers shared by the translators.
// ---------------------------------------------------------------------------

/** The text of a string, or of a block array's text blocks joined with newlines. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function numOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** `data:<mime>;base64,<data>` split apart, or null when the URL is not one. */
function dataUrlParts(url: string): { mime: string; data: string } | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mime: m[1] === '' ? 'image/png' : m[1], data: m[2] };
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

// ---------------------------------------------------------------------------
// Stop reasons: one family per meaning, spelled per wire.
// ---------------------------------------------------------------------------

const STOP_FAMILY = new Map<string, number>([
  ['stop', 0],
  ['end_turn', 0],
  ['STOP', 0],
  ['length', 1],
  ['max_tokens', 1],
  ['MAX_TOKENS', 1],
  ['tool_calls', 2],
  ['function_call', 2],
  ['tool_use', 2],
  ['content_filter', 3],
  ['refusal', 3],
  ['SAFETY', 3],
]);

/** The stop family (0 stop, 1 length, 2 tool use, 3 refusal) of a provider's stop token. */
export function stopFamily(stop: unknown): number {
  if (typeof stop !== 'string') return 0;
  return STOP_FAMILY.get(stop) ?? 0;
}

const OPENAI_STOPS = ['stop', 'length', 'tool_calls', 'content_filter'];
const ANTHROPIC_STOPS = ['end_turn', 'max_tokens', 'tool_use', 'refusal'];
const GEMINI_STOPS = ['STOP', 'MAX_TOKENS', 'TOOL_CALLS', 'SAFETY'];

export function openaiStopOf(raw: string): string {
  return OPENAI_STOPS[stopFamily(raw)] ?? 'stop';
}

export function anthropicStopOf(raw: string): string {
  return ANTHROPIC_STOPS[stopFamily(raw)] ?? 'end_turn';
}

export function geminiStopOf(raw: string): string {
  return GEMINI_STOPS[stopFamily(raw)] ?? 'STOP';
}

// ---------------------------------------------------------------------------
// Request translation: client wire -> provider wire.
// ---------------------------------------------------------------------------

interface ToolDef {
  name: string;
  description: string;
  parameters: Json;
}

/** The tool definitions of a body, read in either wire's spelling. */
function toolDefsOf(body: Json): ToolDef[] {
  if (!Array.isArray(body.tools)) return [];
  const out: ToolDef[] = [];
  for (const raw of body.tools) {
    if (!isRecord(raw)) continue;
    if (isRecord(raw.function) && typeof raw.function.name === 'string') {
      out.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    } else if (typeof raw.name === 'string') {
      out.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return out;
}

function openAITools(defs: readonly ToolDef[]): Json[] {
  return defs.map((d) => ({
    type: 'function',
    function: { name: d.name, description: d.description, parameters: d.parameters },
  }));
}

function anthropicTools(defs: readonly ToolDef[]): Json[] {
  return defs.map((d) => ({ name: d.name, description: d.description, input_schema: d.parameters }));
}

function geminiTools(defs: readonly ToolDef[]): Json[] {
  if (defs.length === 0) return [];
  return [
    {
      functionDeclarations: defs.map((d) => ({ name: d.name, description: d.description, parameters: d.parameters })),
    },
  ];
}

function stopList(body: Json): string[] {
  const raw = body.stop ?? body.stop_sequences;
  if (typeof raw === 'string') return [raw];
  if (Array.isArray(raw)) return raw.filter((x): x is string => typeof x === 'string');
  return [];
}

function maxTokensOf(body: Json): number | null {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return null;
}

/** One Anthropic content block as an OpenAI content part; null for the rest. */
function openAIPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = blocksText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = blocksText(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    let hasToolResult = false;
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        hasToolResult = true;
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = openAIPartOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
    else if (hasToolResult) messages.push({ role: 'user', content: '' });
  }

  const out: Json = { model, messages };
  const max = maxTokensOf(body);
  if (max !== null) out.max_tokens = max;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body);
  if (stops.length > 0) out.stop = stops;
  const defs = toolDefsOf(body);
  if (defs.length > 0) out.tools = openAITools(defs);
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    // The streamed answer must carry its usage, or the client's usage check has nothing to read.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI content part as an Anthropic content block; null for the rest. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = dataUrlParts(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data.mime, data: data.data } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`. The Anthropic
 * wire cannot carry response_format, so this is only used for requests without JSON mode. System
 * messages become the system prompt, image_url parts become image blocks, tool messages become
 * tool_result blocks, assistant tool_calls become tool_use blocks.
 */
export function openaiToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
          },
        ],
      });
      continue;
    }
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const blocks: Json[] = [];
      const text = blocksText(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argsObject(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = anthropicBlockOf(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  const max = maxTokensOf(body);
  if (max !== null) out.max_tokens = max;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body);
  if (stops.length > 0) out.stop_sequences = stops;
  const defs = toolDefsOf(body);
  if (defs.length > 0) out.tools = anthropicTools(defs);
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One image as a Gemini content part: inlineData for a data URL, fileData for a plain URL. */
function geminiImagePart(url: string): Json | null {
  const data = dataUrlParts(url);
  if (data !== null) return { inlineData: { mimeType: data.mime, data: data.data } };
  if (url !== '') return { fileData: { fileUri: url } };
  return null;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body. Tool messages and assistant
 * tool calls are dropped from the history: they carry no text the provider echoes, and a trailing
 * tool message would make the provider read the wrong last user text.
 */
export function openaiToGemini(body: Json): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (raw.role === 'assistant') {
      const text = blocksText(content);
      if (text !== '') contents.push({ role: 'model', parts: [{ text }] });
      continue;
    }
    const parts: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const image = geminiImagePart(part.image_url.url);
        if (image !== null) parts.push(image);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  const system = systemParts.join('\n');
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const defs = toolDefsOf(body);
  const tools = geminiTools(defs);
  if (tools.length > 0) out.tools = tools;
  const config: Json = {};
  const max = maxTokensOf(body);
  if (max !== null) config.maxOutputTokens = max;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopList(body);
  if (stops.length > 0) config.stopSequences = stops;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const choice = body.tool_choice;
  if (choice === 'required' || choice === 'any') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.function.name] } };
  }
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent body. */
export function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (raw.role === 'assistant') {
      const text = blocksText(content);
      if (text !== '') contents.push({ role: 'model', parts: [{ text }] });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'image' && isRecord(block.source)) {
        const src = block.source;
        if (src.type === 'base64' && typeof src.data === 'string') {
          parts.push({ inlineData: { mimeType: typeof src.media_type === 'string' ? src.media_type : 'image/png', data: src.data } });
        } else if (src.type === 'url' && typeof src.url === 'string') {
          const image = geminiImagePart(src.url);
          if (image !== null) parts.push(image);
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  const system = blocksText(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const defs = toolDefsOf(body);
  const tools = geminiTools(defs);
  if (tools.length > 0) out.tools = tools;
  const config: Json = {};
  const max = maxTokensOf(body);
  if (max !== null) config.maxOutputTokens = max;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopList(body);
  if (stops.length > 0) config.stopSequences = stops;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// JSON answers: one normalized shape, emitted in either client wire.
// ---------------------------------------------------------------------------

export interface ToolCall {
  id: string;
  name: string;
  args: Json;
}

export interface JsonAnswer {
  text: string;
  toolCalls: ToolCall[];
  inputTokens: number;
  outputTokens: number;
  stop: string;
}

function geminiTextOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  const content = first !== null && isRecord(first.content) ? first.content : null;
  const parts = content !== null && Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Chat Completions answer as the normalized shape; null when it is not one. */
export function parseOpenAIJson(body: unknown): JsonAnswer | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null) return null;
  const text = typeof message.content === 'string' ? message.content : blocksText(message.content);
  const toolCalls: ToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      toolCalls.push({
        id: typeof call.id === 'string' ? call.id : '',
        name: typeof call.function.name === 'string' ? call.function.name : '',
        args: argsObject(call.function.arguments),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    inputTokens: numOrZero(usage.prompt_tokens),
    outputTokens: numOrZero(usage.completion_tokens),
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
  };
}

/** A Messages answer as the normalized shape; null when it is not one. */
export function parseAnthropicJson(body: unknown): JsonAnswer | null {
  if (!isRecord(body) || body.type !== 'message' || !Array.isArray(body.content)) return null;
  const text = blocksText(body.content);
  const toolCalls: ToolCall[] = [];
  for (const block of body.content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : '',
      name: typeof block.name === 'string' ? block.name : '',
      args: isRecord(block.input) ? block.input : {},
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    inputTokens: numOrZero(usage.input_tokens),
    outputTokens: numOrZero(usage.output_tokens),
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
  };
}

/** A Gemini generateContent answer as the normalized shape; null when it is not one. */
export function parseGeminiJson(body: unknown): JsonAnswer | null {
  if (!isRecord(body)) return null;
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  if (first === null) return null;
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: geminiTextOf(body),
    toolCalls: [],
    inputTokens: numOrZero(meta.promptTokenCount),
    outputTokens: numOrZero(meta.candidatesTokenCount),
    stop: typeof first.finishReason === 'string' ? first.finishReason : 'STOP',
  };
}

/** The normalized answer as a Chat Completions object naming `model`. */
export function emitOpenAIJson(a: JsonAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: a.text };
  if (a.toolCalls.length > 0) {
    message.tool_calls = a.toolCalls.map((c) => ({
      id: c.id,
      type: 'function',
      function: { name: c.name, arguments: JSON.stringify(c.args) },
    }));
  }
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openaiStopOf(a.stop), logprobs: null }],
    usage: {
      prompt_tokens: a.inputTokens,
      completion_tokens: a.outputTokens,
      total_tokens: a.inputTokens + a.outputTokens,
    },
  };
}

/** The normalized answer as a Messages object naming `model`. */
export function emitAnthropicJson(a: JsonAnswer, model: string): Json {
  const content: Json[] = [];
  if (a.text !== '' || a.toolCalls.length === 0) content.push({ type: 'text', text: a.text });
  for (const call of a.toolCalls) {
    content.push({ type: 'tool_use', id: call.id, name: call.name, input: call.args });
  }
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopOf(a.stop),
    stop_sequence: null,
    usage: { input_tokens: a.inputTokens, output_tokens: a.outputTokens },
  };
}

// ---------------------------------------------------------------------------
// SSE: one normalized shape, emitted in either client wire.
// ---------------------------------------------------------------------------

export interface StreamAnswer {
  pieces: string[];
  inputTokens: number | null;
  outputTokens: number | null;
  stop: string;
  complete: boolean;
}

export function streamText(a: StreamAnswer): string {
  return a.pieces.join('');
}

/** One SSE frame: `event: <name>` when named, then the data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

interface SseFrame {
  event: string | null;
  data: string;
}

/** The frames of an SSE body: event names and data lines, blank-line separated. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(payload: string): unknown {
  try {
    return JSON.parse(payload);
  } catch {
    return null;
  }
}

/** A Chat Completions event stream as the normalized shape. */
export function parseOpenAISse(text: string): StreamAnswer {
  const out: StreamAnswer = { pieces: [], inputTokens: null, outputTokens: null, stop: '', complete: false };
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      out.complete = true;
      continue;
    }
    const chunk = jsonOrNull(payload);
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      out.inputTokens = numOrNull(chunk.usage.prompt_tokens);
      out.outputTokens = numOrNull(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') out.pieces.push(delta.content);
    if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') out.stop = choice.finish_reason;
  }
  return out;
}

/** A Messages event stream as the normalized shape. */
export function parseAnthropicSse(text: string): StreamAnswer {
  const out: StreamAnswer = { pieces: [], inputTokens: null, outputTokens: null, stop: '', complete: false };
  for (const frame of parseSseFrames(text)) {
    if (frame.event === null) continue;
    const data = jsonOrNull(frame.data);
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      out.inputTokens = numOrNull(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out.pieces.push(data.delta.text);
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') out.stop = data.delta.stop_reason;
      if (isRecord(data.usage)) out.outputTokens = numOrNull(data.usage.output_tokens);
    } else if (frame.event === 'message_stop') {
      out.complete = true;
    }
  }
  return out;
}

/** A Gemini streamGenerateContent stream as the normalized shape. */
export function parseGeminiSse(text: string): StreamAnswer {
  const out: StreamAnswer = { pieces: [], inputTokens: null, outputTokens: null, stop: '', complete: false };
  for (const frame of parseSseFrames(text)) {
    const data = jsonOrNull(frame.data);
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : null;
    if (first !== null) {
      const piece = geminiTextOf({ candidates });
      if (piece !== '') out.pieces.push(piece);
      if (typeof first.finishReason === 'string' && first.finishReason !== '') {
        out.stop = first.finishReason;
        out.complete = true;
      }
    }
    if (isRecord(data.usageMetadata)) {
      out.inputTokens = numOrNull(data.usageMetadata.promptTokenCount);
      out.outputTokens = numOrNull(data.usageMetadata.candidatesTokenCount);
    }
  }
  return out;
}

/** The normalized stream as Chat Completions events naming `model`. */
export function emitOpenAISse(a: StreamAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null, usage?: Json): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
    ...(usage !== undefined ? { usage } : {}),
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of a.pieces) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, openaiStopOf(a.stop === '' ? 'stop' : a.stop))));
  if (a.inputTokens !== null && a.outputTokens !== null) {
    frames.push(
      sseEvent(null, {
        id: 'chatcmpl-agg',
        object: 'chat.completion.chunk',
        created: FIXED_CREATED,
        model,
        choices: [],
        usage: { prompt_tokens: a.inputTokens, completion_tokens: a.outputTokens, total_tokens: a.inputTokens + a.outputTokens },
      }),
    );
  }
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** The normalized stream as Messages events naming `model`. */
export function emitAnthropicSse(a: StreamAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: a.inputTokens ?? 0, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  ];
  for (const piece of a.pieces) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopOf(a.stop === '' ? 'end_turn' : a.stop), stop_sequence: null },
      usage: { output_tokens: a.outputTokens ?? 0 },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
