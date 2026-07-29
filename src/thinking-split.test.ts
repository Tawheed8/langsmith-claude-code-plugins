/**
 * Thinking→generation boundary extraction.
 *
 * Fixtures are lifted verbatim from a real Claude Code transcript (message ids
 * and timestamps unchanged), which showed thinking blocks arriving in their own
 * streamed chunk, separate from the chunk carrying visible output:
 *
 *   yDQwzuQc  thinking 07:07:34.945 → tool_use 07:07:37.207   (2.26s thinking)
 *   H7S67Txt  thinking 07:08:08.309 → tool_use 07:08:13.110   (4.80s thinking)
 *   i8A59FoH  text     07:07:12.128 → tool_use 07:07:12.258   (no thinking)
 *
 * That separation is what makes the split recoverable at all; if a future
 * Claude Code version collapses thinking and output into one chunk, these
 * assertions are the early warning.
 */
import { describe, it, expect } from "vitest";
import { groupIntoTurns } from "./transcript.js";
import type { TranscriptMessage, AssistantMessage } from "./types.js";

function userMsg(timestamp: string): TranscriptMessage {
  return {
    type: "user",
    message: { role: "user", content: "go" },
    timestamp,
  } as TranscriptMessage;
}

function chunk(
  id: string,
  timestamp: string,
  content: AssistantMessage["message"]["content"],
  stop_reason?: string,
): TranscriptMessage {
  return {
    type: "assistant",
    message: {
      id,
      role: "assistant",
      model: "claude-sonnet-4-5-20250929",
      content,
      usage: { input_tokens: 10, output_tokens: 5 },
      ...(stop_reason ? { stop_reason } : {}),
    },
    timestamp,
  } as TranscriptMessage;
}

describe("thinking→generation boundary", () => {
  it("reports the boundary when thinking precedes visible output (real fixture)", () => {
    const messages: TranscriptMessage[] = [
      userMsg("2026-07-29T07:07:34.000Z"),
      chunk("yDQwzuQc", "2026-07-29T07:07:34.945Z", [
        { type: "thinking", thinking: "deciding which tool to use" },
      ]),
      chunk(
        "yDQwzuQc",
        "2026-07-29T07:07:37.207Z",
        [{ type: "tool_use", id: "t1", name: "Bash", input: {} }],
        "end_turn",
      ),
    ];

    const [turn] = groupIntoTurns(messages);
    const call = turn.llmCalls[0];

    // Span still spans the whole response...
    expect(call.startTime).toBe("2026-07-29T07:07:34.945Z");
    expect(call.endTime).toBe("2026-07-29T07:07:37.207Z");
    // ...and the boundary isolates the 2.26s that was thinking.
    expect(call.thinkingEndTime).toBe("2026-07-29T07:07:37.207Z");

    const thinkingMs =
      new Date(call.thinkingEndTime!).getTime() - new Date(call.startTime).getTime();
    expect(thinkingMs).toBe(2262);
  });

  it("isolates a longer thinking stretch (real fixture)", () => {
    const messages: TranscriptMessage[] = [
      userMsg("2026-07-29T07:08:08.000Z"),
      chunk("H7S67Txt", "2026-07-29T07:08:08.309Z", [
        { type: "thinking", thinking: "working out the expression" },
      ]),
      chunk(
        "H7S67Txt",
        "2026-07-29T07:08:13.110Z",
        [{ type: "tool_use", id: "t2", name: "Bash", input: {} }],
        "end_turn",
      ),
    ];

    const call = groupIntoTurns(messages)[0].llmCalls[0];
    const thinkingMs =
      new Date(call.thinkingEndTime!).getTime() - new Date(call.startTime).getTime();
    expect(thinkingMs).toBe(4801);
  });

  it("omits the boundary entirely when the response had no thinking (real fixture)", () => {
    const messages: TranscriptMessage[] = [
      userMsg("2026-07-29T07:07:12.000Z"),
      chunk("i8A59FoH", "2026-07-29T07:07:12.128Z", [{ type: "text", text: "on it" }]),
      chunk(
        "i8A59FoH",
        "2026-07-29T07:07:12.258Z",
        [{ type: "tool_use", id: "t3", name: "Read", input: {} }],
        "end_turn",
      ),
    ];

    const call = groupIntoTurns(messages)[0].llmCalls[0];
    // Undefined, never a fabricated value — a missing field must read as
    // "no split available", not "zero thinking".
    expect(call.thinkingEndTime).toBeUndefined();
  });

  it("does not treat thinking that never reached visible output as a boundary", () => {
    const messages: TranscriptMessage[] = [
      userMsg("2026-07-29T07:09:00.000Z"),
      chunk(
        "OnlyThink",
        "2026-07-29T07:09:01.000Z",
        [{ type: "thinking", thinking: "interrupted mid-thought" }],
        "end_turn",
      ),
    ];

    const call = groupIntoTurns(messages)[0].llmCalls[0];
    expect(call.thinkingEndTime).toBeUndefined();
  });

  it("uses the chunk timestamp when thinking and output share one chunk", () => {
    // Closest boundary the transcript records — the true switch is inside it.
    const messages: TranscriptMessage[] = [
      userMsg("2026-07-29T07:10:00.000Z"),
      chunk(
        "Mixed",
        "2026-07-29T07:10:02.500Z",
        [
          { type: "thinking", thinking: "brief" },
          { type: "text", text: "answer" },
        ],
        "end_turn",
      ),
    ];

    const call = groupIntoTurns(messages)[0].llmCalls[0];
    expect(call.thinkingEndTime).toBe("2026-07-29T07:10:02.500Z");
  });
});
