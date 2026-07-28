/**
 * Regression test for the LLM-node span-timing fix.
 *
 * Reconstructs a real 3-step slice from a Claude Code run (VibeCraft commission-plan
 * skill, steps building/validating expression ASTs) whose think/gen/tool split was
 * independently measured by cross-referencing this plugin's COT event log against its
 * own LangSmith trace: (112.43s think, 27.84s gen, 4.07s tool), (3.73s, 0s, 0.09s),
 * (16.73s, 2.41s, 0.08s). Before the fix, each LLM span reported only `gen + tool`,
 * with `think` silently missing from every span. This test drives the real traceTurn()
 * logic (LangSmith client mocked, no network) and asserts the corrected span
 * boundaries reconstruct exactly the measured think/gen/tool split.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import type { Turn } from "./types.js";

const mockCreateRun = vi.fn().mockResolvedValue(undefined);
const mockUpdateRun = vi.fn().mockResolvedValue(undefined);

let allRunTreeInstances: Array<{ params: Record<string, unknown>; ops: string[] }> = [];

vi.mock("langsmith", () => {
  class MockClient {
    createRun = mockCreateRun;
    updateRun = mockUpdateRun;
    awaitPendingTraceBatches = vi.fn().mockResolvedValue(undefined);
  }
  class MockRunTree {
    client: MockClient | undefined;
    params: Record<string, unknown>;
    _tracker: { params: Record<string, unknown>; ops: string[] };
    constructor(params: { client?: MockClient; id: string } & Record<string, unknown>) {
      this.client = params.client;
      this.params = params;
      this._tracker = { params, ops: [] };
      allRunTreeInstances.push(this._tracker);
    }
    postRun() {
      this._tracker.ops.push("postRun");
      if (this.client) this.client.createRun(this.params);
    }
    patchRun() {
      this._tracker.ops.push("patchRun");
      if (this.client) this.client.updateRun(this.params.id, this.params);
    }
  }
  return {
    RunTree: MockRunTree,
    Client: MockClient,
    uuid7: () => `test-uuid-${Math.random().toString(36).slice(2, 15)}`,
  };
});

vi.mock("./logger.js", () => ({
  log: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
  debug: vi.fn(),
  initLogger: vi.fn(),
}));

import { initTracing, traceTurn } from "./langsmith.js";

describe("motivating case: real think/gen/tool split reconstructed from a live run", () => {
  beforeEach(() => {
    mockCreateRun.mockClear();
    mockUpdateRun.mockClear();
    allRunTreeInstances = [];
    initTracing("test-api-key", "https://test.api.com");
  });

  it("recovers thinking time and excludes tool time for a 3-step tool-calling turn", async () => {
    const turn: Turn = {
      userContent: "create plan and criterias from given doc",
      userTimestamp: "2026-01-01T00:00:00.000Z",
      llmCalls: [
        {
          // think=112.43s (missing before the fix), gen=27.84s, tool=4.07s
          content: [{ type: "text", text: "Building all 3 expression ASTs..." }],
          model: "claude-sonnet-4-5",
          usage: { input_tokens: 114700, output_tokens: 800 },
          startTime: "2026-01-01T00:01:52.430Z",
          endTime: "2026-01-01T00:02:20.270Z",
          toolCalls: [
            {
              tool_use: { id: "tool-1", name: "validate_criteria", input: {} },
              result: { content: "validated", timestamp: "2026-01-01T00:02:24.340Z" },
            },
          ],
        },
        {
          // think=3.73s (missing before the fix), gen=0s (silent tool call), tool=0.09s
          content: [],
          model: "claude-sonnet-4-5",
          usage: { input_tokens: 114900, output_tokens: 5 },
          startTime: "2026-01-01T00:02:28.070Z",
          endTime: "2026-01-01T00:02:28.070Z",
          toolCalls: [
            {
              tool_use: { id: "tool-2", name: "re_inspect_result", input: {} },
              result: { content: "status: None", timestamp: "2026-01-01T00:02:28.160Z" },
            },
          ],
        },
        {
          // think=16.73s (missing before the fix), gen=2.41s, tool=0.08s
          content: [{ type: "text", text: "Extracting serCriterias..." }],
          model: "claude-sonnet-4-5",
          usage: { input_tokens: 116100, output_tokens: 200 },
          startTime: "2026-01-01T00:02:44.890Z",
          endTime: "2026-01-01T00:02:47.300Z",
          toolCalls: [
            {
              tool_use: { id: "tool-3", name: "extract_ser_criterias", input: {} },
              result: { content: "extracted", timestamp: "2026-01-01T00:02:47.380Z" },
            },
          ],
        },
      ],
      isComplete: true,
    };

    await traceTurn({ turn, sessionId: "session-real", turnNum: 1, project: "test-project" });

    const llmPatches = allRunTreeInstances
      .filter((i) => i.ops.includes("patchRun") && i.params.run_type === "llm")
      .map((i) => i.params);
    expect(llmPatches.length).toBe(3);

    // Step 1: first call in the turn — start boundary is the turn's own start,
    // NOT the model's first visible token (which would hide the 112.43s of thinking).
    expect(llmPatches[0].start_time).toBe("2026-01-01T00:00:00.000Z");
    // end_time is the model's own last chunk — NOT stretched to the tool result
    // 4.07s later.
    expect(llmPatches[0].end_time).toBe("2026-01-01T00:02:20.270Z");

    // Step 2: start boundary is step 1's tool result — recovers the 3.73s of
    // thinking that used to vanish between "tool finished" and "next first token".
    expect(llmPatches[1].start_time).toBe("2026-01-01T00:02:24.340Z");
    expect(llmPatches[1].end_time).toBe("2026-01-01T00:02:28.070Z");

    // Step 3: start boundary is step 2's tool result — recovers 16.73s of thinking.
    expect(llmPatches[2].start_time).toBe("2026-01-01T00:02:28.160Z");
    expect(llmPatches[2].end_time).toBe("2026-01-01T00:02:47.300Z");

    // Cross-check against the independently measured split: reconstructed
    // (end_time - start_time) per LLM span must equal think + gen, never gen + tool.
    const expected = [
      { think: 112.43, gen: 27.84 },
      { think: 3.73, gen: 0.0 },
      { think: 16.73, gen: 2.41 },
    ];
    llmPatches.forEach((p, i) => {
      const durationSec =
        (new Date(p.end_time as string).getTime() - new Date(p.start_time as string).getTime()) /
        1000;
      expect(durationSec).toBeCloseTo(expected[i].think + expected[i].gen, 2);
    });
  });
});
