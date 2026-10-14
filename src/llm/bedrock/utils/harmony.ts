/**
 * Plain-text OpenAI harmony control tokens in Bedrock requests to OpenAI models.
 *
 * OpenAI models on Bedrock render the conversation into the harmony chat format, and a
 * literal header such as `<|channel|>analysis` inside message text is not treated as text:
 * Bedrock answers `200 OK` with an event stream that closes without a single event (no
 * `messageStart`, no `messageStop`, no exception). It is deterministic for the request, so
 * every later turn that still carries the text fails the same way. Pasted transcripts,
 * attached text files, and fetched pages that document the format all carry such headers.
 *
 * The ASCII bars of each token are swapped for fullwidth bars (`<｜channel｜>`): the model
 * still reads the token, but nothing renders it as a control token. Zero-width characters
 * and spaces between `<`, `|` and the name do not help (verified live), so they are
 * tolerated inside the match and kept.
 */
import type {
  BedrockMessage,
  BedrockContentBlock,
  BedrockSystemContentBlock,
} from '../types';

/** The JSON document type Converse uses for `toolUse.input` and `toolResult` JSON. */
type DocumentType = Exclude<
  NonNullable<BedrockContentBlock['toolUse']>['input'],
  undefined
>;

const GAP = '[\\s\\u200B-\\u200D\\u2060\\uFEFF]*';
const HARMONY_CONTROL_TOKEN = new RegExp(
  `<${GAP}\\|${GAP}[A-Za-z0-9_]+${GAP}\\|${GAP}>`,
  'g'
);
const ASCII_BAR = /\|/g;
const FULLWIDTH_BAR = '｜';

type ToolResultContentBlock = NonNullable<
  NonNullable<BedrockContentBlock['toolResult']>['content']
>[number];

/** The text with every harmony control token rendered inert; the same string when none. */
export function neutralizeHarmonyControlTokens(text: string): string {
  if (!text.includes('|')) {
    return text;
  }
  return text.replace(HARMONY_CONTROL_TOKEN, (token) =>
    token.replace(ASCII_BAR, FULLWIDTH_BAR)
  );
}

/** A JSON document with every string neutralized; the same reference when nothing changed. */
function neutralizeDocument(value: DocumentType): DocumentType {
  if (typeof value === 'string') {
    return neutralizeHarmonyControlTokens(value);
  }
  if (Array.isArray(value)) {
    let next: DocumentType[] | undefined;
    for (let i = 0; i < value.length; i++) {
      const item = neutralizeDocument(value[i]);
      if (item !== value[i]) {
        next ??= value.slice();
        next[i] = item;
      }
    }
    return next ?? value;
  }
  if (value == null || typeof value !== 'object') {
    return value;
  }
  let next: Record<string, DocumentType> | undefined;
  for (const key of Object.keys(value)) {
    const entry = value[key];
    const item = neutralizeDocument(entry);
    if (item !== entry) {
      next ??= { ...value };
      next[key] = item;
    }
  }
  return next ?? value;
}

function neutralizeToolResultBlock(
  block: ToolResultContentBlock
): ToolResultContentBlock {
  if (typeof block.text === 'string') {
    const text = neutralizeHarmonyControlTokens(block.text);
    return text === block.text ? block : { ...block, text };
  }
  if (block.json !== undefined) {
    const json = neutralizeDocument(block.json);
    return json === block.json ? block : { ...block, json };
  }
  return block;
}

/** Maps each item, returning the same array when no item changed. */
function mapChanged<T>(items: T[], map: (item: T) => T): T[] {
  let next: T[] | undefined;
  for (let i = 0; i < items.length; i++) {
    const item = map(items[i]);
    if (item !== items[i]) {
      next ??= items.slice();
      next[i] = item;
    }
  }
  return next ?? items;
}

function neutralizeContentBlock(
  block: BedrockContentBlock
): BedrockContentBlock {
  if (typeof block.text === 'string') {
    const text = neutralizeHarmonyControlTokens(block.text);
    return text === block.text ? block : { ...block, text };
  }
  if (block.toolUse != null) {
    const { input } = block.toolUse;
    const next = input === undefined ? input : neutralizeDocument(input);
    return next === input
      ? block
      : { ...block, toolUse: { ...block.toolUse, input: next } };
  }
  const content = block.toolResult?.content;
  if (block.toolResult != null && content != null) {
    const next = mapChanged(content, neutralizeToolResultBlock);
    return next === content
      ? block
      : { ...block, toolResult: { ...block.toolResult, content: next } };
  }
  return block;
}

function neutralizeSystemBlock(
  block: BedrockSystemContentBlock
): BedrockSystemContentBlock {
  if (typeof block.text !== 'string') {
    return block;
  }
  const text = neutralizeHarmonyControlTokens(block.text);
  return text === block.text ? block : { ...block, text };
}

/**
 * Neutralizes harmony control tokens across a converted request. Messages are updated in
 * place (the converter owns them); a block that changes is replaced by a copy, so content
 * shared with the LangChain input is never mutated.
 */
export function neutralizeHarmonyRequest(
  messages: BedrockMessage[],
  system: BedrockSystemContentBlock[]
): BedrockSystemContentBlock[] {
  for (const message of messages) {
    if (message.content != null) {
      message.content = mapChanged(message.content, neutralizeContentBlock);
    }
  }
  return mapChanged(system, neutralizeSystemBlock);
}
