/**
 * Thinking→generation boundary extraction.
 *
 * A chunk's timestamp is when that content block *completed*. Verified against
 * real traces: the gap between a thinking chunk and the next visible chunk
 * tracks the size of the visible block (68 chars → 0.014s, 3572 chars →
 * 12.282s, ~290 chars/s), i.e. it measures generation throughput, not
 * deliberation. So the last thinking block's timestamp marks where thinking
 * ended and generation began.
 *
 * Fixtures use real transcript timestamps:
 *   yDQwzuQc  thinking 07:07:34.945 → tool_use 07:07:37.207   (2.262s generating)
 *   i8A59FoH  text     07:07:12.128 → tool_use 07:07:12.258   (no thinking)
 */
import { describe, it, expect } from "vitest";
import { groupIntoTurns } from "./transcript.js";
import type { TranscriptMessage, AssistantMessage } from "./types.js";

function userMsg(timestamp: string): TranscriptMessage {
  return { type: "user", message: { role: "user", content: "go" }, timestamp } as TranscriptMessage;
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
  it("marks where thinking ended, so the remainder is generation (real fixture)", () => {
    const messages: TranscriptMessage[] = [
      userMsg("2026-07-29T07:07:34.000Z"),
      // Empty text is how Claude Code actually persists thinking blocks.
      chunk("yDQwzuQc", "2026-07-29T07:07:34.945Z", [{ type: "thinking", thinking: "" }]),
      chunk(
        "yDQwzuQc",
        "2026-07-29T07:07:37.207Z",
        [{ type: "tool_use", id: "t1", name: "Bash", input: {} }],
        "end_turn",
      ),
    ];

    const call = groupIntoTurns(messages)[0].llmCalls[0];

    // Boundary is the thinking block's own completion, NOT the later visible
    // chunk — that later gap is time spent generating the tool call.
    expect(call.thinkingEndTime).toBe("2026-07-29T07:07:34.945Z");

    const generatingMs =
      new Date(call.endTime).getTime() - new Date(call.thinkingEndTime!).getTime();
    expect(generatingMs).toBe(2262);
  });

  it("uses the last thinking block when several precede the output", () => {
    const messages: TranscriptMessage[] = [
      userMsg("2026-07-29T07:09:00.000Z"),
      chunk("Multi", "2026-07-29T07:09:01.000Z", [{ type: "thinking", thinking: "" }]),
      chunk("Multi", "2026-07-29T07:09:03.500Z", [{ type: "thinking", thinking: "" }]),
      chunk(
        "Multi",
        "2026-07-29T07:09:05.000Z",
        [{ type: "text", text: "answer" }],
        "end_turn",
      ),
    ];

    const call = groupIntoTurns(messages)[0].llmCalls[0];
    // Thinking ran until 03.500; only the final 1.5s was generating.
    expect(call.thinkingEndTime).toBe("2026-07-29T07:09:03.500Z");
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
    // Undefined, never fabricated — means "no split available", not "zero".
    expect(call.thinkingEndTime).toBeUndefined();
  });

  it("does not report a boundary when thinking never reached visible output", () => {
    const messages: TranscriptMessage[] = [
      userMsg("2026-07-29T07:09:00.000Z"),
      chunk(
        "OnlyThink",
        "2026-07-29T07:09:01.000Z",
        [{ type: "thinking", thinking: "" }],
        "end_turn",
      ),
    ];

    expect(groupIntoTurns(messages)[0].llmCalls[0].thinkingEndTime).toBeUndefined();
  });

  it("collapses the generation window when one chunk holds thinking and output", () => {
    // The switch happened inside the chunk; only its timestamp is recorded, so
    // generation is unmeasurable (zero) rather than guessed at.
    const messages: TranscriptMessage[] = [
      userMsg("2026-07-29T07:10:00.000Z"),
      chunk(
        "Mixed",
        "2026-07-29T07:10:02.500Z",
        [
          { type: "thinking", thinking: "" },
          { type: "text", text: "answer" },
        ],
        "end_turn",
      ),
    ];

    const call = groupIntoTurns(messages)[0].llmCalls[0];
    expect(call.thinkingEndTime).toBe("2026-07-29T07:10:02.500Z");
    expect(new Date(call.endTime).getTime() - new Date(call.thinkingEndTime!).getTime()).toBe(0);
  });
});
