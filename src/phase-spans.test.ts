/**
 * Opt-in timing phase spans (CC_LANGSMITH_PHASE_SPANS=true).
 *
 * The flag is read once at module load, so the env var is set before the
 * dynamic import below — a plain top-level import would capture the default.
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

process.env.CC_LANGSMITH_PHASE_SPANS = "true";
const { initTracing, traceTurn } = await import("./langsmith.js");

/** Phase runs are the `chain` children posted beneath the llm run. */
function phaseRuns() {
  const llm = allRunTreeInstances.find((i) => i.params.run_type === "llm");
  const llmId = llm?.params.id;
  return allRunTreeInstances
    .filter((i) => i.params.run_type === "chain" && i.params.parent_run_id === llmId)
    .map((i) => i.params);
}

describe("timing phase spans (opt-in)", () => {
  beforeEach(() => {
    mockCreateRun.mockClear();
    mockUpdateRun.mockClear();
    allRunTreeInstances = [];
    initTracing("test-api-key", "https://test.api.com");
  });

  it("splits a thinking response into waiting / thinking / generating", async () => {
    const turn: Turn = {
      userContent: "go",
      userTimestamp: "2026-07-29T07:07:34.000Z",
      llmCalls: [
        {
          content: [{ type: "text", text: "done" }],
          model: "claude-sonnet-4-5",
          usage: { input_tokens: 10, output_tokens: 5 },
          // Real fixture shape: 0.945s waiting, 2.262s thinking, 1.0s generating.
          startTime: "2026-07-29T07:07:34.945Z",
          thinkingEndTime: "2026-07-29T07:07:37.207Z",
          endTime: "2026-07-29T07:07:38.207Z",
          toolCalls: [],
        },
      ],
      isComplete: true,
    };

    await traceTurn({ turn, sessionId: "s", turnNum: 1, project: "p" });

    const phases = phaseRuns();
    expect(phases.map((p) => p.name)).toEqual([
      "Waiting (queue + prompt)",
      "Thinking",
      "Generating",
    ]);

    // Phases tile the span exactly: no gaps, no overlaps.
    expect(phases[0].start_time).toBe("2026-07-29T07:07:34.000Z"); // span start
    expect(phases[0].end_time).toBe("2026-07-29T07:07:34.945Z"); // first token
    expect(phases[1].start_time).toBe("2026-07-29T07:07:34.945Z");
    expect(phases[1].end_time).toBe("2026-07-29T07:07:37.207Z"); // thinking end
    expect(phases[2].start_time).toBe("2026-07-29T07:07:37.207Z");
    expect(phases[2].end_time).toBe("2026-07-29T07:07:38.207Z"); // span end

    expect(phases[1].outputs).toEqual({ duration_ms: 2262 });

    // Timing annotations only — must not add tokens/cost to trace rollups.
    for (const p of phases) {
      expect(p.run_type).toBe("chain");
      expect((p.extra as { metadata?: Record<string, unknown> })?.metadata).not.toHaveProperty(
        "usage_metadata",
      );
    }
  });

  it("emits no Thinking phase when the response had no thinking", async () => {
    const turn: Turn = {
      userContent: "go",
      userTimestamp: "2026-07-29T07:07:12.000Z",
      llmCalls: [
        {
          content: [{ type: "text", text: "ok" }],
          model: "claude-sonnet-4-5",
          usage: { input_tokens: 10, output_tokens: 5 },
          startTime: "2026-07-29T07:07:12.128Z",
          endTime: "2026-07-29T07:07:12.958Z",
          toolCalls: [],
        },
      ],
      isComplete: true,
    };

    await traceTurn({ turn, sessionId: "s", turnNum: 1, project: "p" });

    expect(phaseRuns().map((p) => p.name)).toEqual([
      "Waiting (queue + prompt)",
      "Generating",
    ]);
  });

  it("skips zero-length phases rather than drawing an empty bar", async () => {
    // Single-chunk response: start == end, so generating has no measurable
    // window. Emitting a 0ms bar would imply a measurement never taken.
    const turn: Turn = {
      userContent: "go",
      userTimestamp: "2026-07-29T07:07:34.000Z",
      llmCalls: [
        {
          content: [{ type: "tool_use", id: "t", name: "Bash", input: {} }],
          model: "claude-sonnet-4-5",
          usage: { input_tokens: 10, output_tokens: 5 },
          startTime: "2026-07-29T07:07:34.945Z",
          thinkingEndTime: "2026-07-29T07:07:37.207Z",
          endTime: "2026-07-29T07:07:37.207Z",
          toolCalls: [],
        },
      ],
      isComplete: true,
    };

    await traceTurn({ turn, sessionId: "s", turnNum: 1, project: "p" });

    const names = phaseRuns().map((p) => p.name);
    expect(names).toContain("Thinking");
    expect(names).not.toContain("Generating");
  });
});
