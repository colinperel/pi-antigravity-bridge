// Compaction & silent-overflow lifecycle tests for pi-antigravity-bridge:
//
//   1. Phantom silent-overflow prevention (follow-up to issue #12):
//      pi-ai's isContextOverflow() Case 2 checks `usage.input + usage.cacheRead
//      > contextWindow` on stopReason === "stop". Multi-step billing sums can
//      exceed contextWindow even when live occupancy (totalTokens) is well
//      inside the window; clampOverflowUsage scales input + cacheRead down
//      proportionally while leaving normal turns and genuine overflows alone.
//   2. Ephemeral helper turn isolation (options.cacheRetention === "none"):
//      pi's compaction and branch summarization pass cacheRetention: "none".
//      Those calls must run in a fresh agy conversation without reading,
//      mutating, or deleting the live SessionStore binding, claiming stranded
//      tool results, or arming background-task watchers.
//   3. Post-compaction rotation & handoff:
//      rotateSessionOnCompaction drops the bound agy conversation so backend
//      context is freed alongside pi's transcript, guarantees a one-shot
//      compaction handoff even when config.digest is disabled, and handles
//      mid-tool-loop compacted tails (toolResult/assistant) via
//      COMPACTION_RESUME_PROMPT + includeLast instead of hard-erroring.
//
// Run: npm test

import assert from "node:assert/strict";
import { getEventListeners } from "node:events";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterAll, describe, test, vi } from "vitest";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import factory from "../extensions/index.js";
import { isContextOverflow } from "@earendil-works/pi-ai/utils/overflow";
import {
	normalizeContext,
	type Api,
	type AssistantMessage,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type TranscriptContext,
	type Usage,
} from "@earendil-works/pi-ai";
import {
	COMPACTION_RESUME_PROMPT,
	CompactionHandoffs,
	DIGEST_PREAMBLE,
	ToolRoundTrips,
	buildContextDigest,
	createStreamSimple,
	rotateSessionOnCompaction,
	sessionKey,
} from "../src/provider.js";
import { clampOverflowUsage } from "../src/stream-events.js";
import { SessionStore } from "../src/sessions.js";
import { StreamDriver, type DriverTurnRequest } from "../src/driver.js";
import type { DriverActivity, TurnHandle, TurnOutcome } from "../src/driver-types.js";
import type { McpBridgeDeps } from "../src/mcp-server.js";
import { __resetRealmForTests, bridgeRealm } from "../src/runtime.js";

// Isolate loadConfig() from any live agent dir on the test runner's machine
// (matches tests/provider-streaming.test.ts).
const prevAgentDir = process.env.PI_CODING_AGENT_DIR;
const fakeAgentRoot = fs.mkdtempSync(path.join(os.tmpdir(), "agy-compact-home-"));
process.env.PI_CODING_AGENT_DIR = path.join(fakeAgentRoot, ".pi", "agent");
afterAll(() => {
	if (prevAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
	else process.env.PI_CODING_AGENT_DIR = prevAgentDir;
	fs.rmSync(fakeAgentRoot, { recursive: true, force: true });
});

const COMP_PREFIX =
	"The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
const COMP_SUFFIX = "\n</summary>";

const model: Model<Api> = {
	id: "gemini-flash",
	name: "Gemini 3.8 Flash",
	api: "agy-bridge" as Api,
	provider: "antigravity",
	baseUrl: "agy-bridge://antigravity",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 1_000_000,
	maxTokens: 65_536,
};

function zeroUsage(): Usage {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function user(content: string, timestamp = 1000): Message {
	return { role: "user", content, timestamp };
}

function compaction(summary: string, timestamp = 2000): Message {
	return user(COMP_PREFIX + summary + COMP_SUFFIX, timestamp);
}

function assistantMsg(text: string, provider = "antigravity", timestamp = 3000): Message {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "agy-bridge" as never,
		provider: provider as never,
		model: model.id,
		usage: zeroUsage(),
		stopReason: "stop",
		timestamp,
	};
}

function toolResultMsg(toolCallId: string, toolName: string, text: string, timestamp = 4000): Message {
	return {
		role: "toolResult",
		toolCallId,
		toolName,
		content: [{ type: "text", text }],
		isError: false,
		timestamp,
	};
}

interface TurnScript {
	outcome: TurnOutcome;
	activities?: DriverActivity[];
	syncBind?: boolean;
}

function makeFakeDriver(scripts: TurnScript[], requests: DriverTurnRequest[]): StreamDriver {
	let idx = 0;
	const driver = {
		get activeHandle() {
			return null;
		},
		kickIdle: () => {},
		reentry: () => null,
		snapshot: () => ({ state: "ready", conversationId: undefined, stats: {} }),
		run: async (opts: DriverTurnRequest) => {
			requests.push(opts);
			const script = scripts[Math.min(idx, scripts.length - 1)]!;
			idx += 1;
			if (script.syncBind && script.outcome.conversationId) {
				const sessionId = script.outcome.conversationId;
				const fresh = sessionId !== opts.conversationId;
				await opts.onSessionBound?.({ sessionId, fresh, reused: !fresh });
			}
			const activities = [...(script.activities ?? [])];
			return {
				id: `fake-turn-${idx}`,
				outcome: Promise.resolve(script.outcome),
				next: async () => activities.shift() ?? null,
				pushExternal: () => {},
			} satisfies TurnHandle as unknown as TurnHandle;
		},
	};
	return driver as unknown as StreamDriver;
}

// run() waits for the previous outcome, just like both real drivers. There
// is no natural settlement while the first tool call is parked.
function makeQueuedDriver(requests: DriverTurnRequest[], summaryOutcome = okOutcome("conv-summary", "summary")) {
	let active: TurnHandle | null = null;
	let conversationId: string | undefined;
	let finish: ((outcome: TurnOutcome) => void) | undefined;
	const driver = {
		get activeHandle() { return active; },
		kickIdle: () => {},
		reentry: () => active,
		snapshot: () => ({ conversationId }),
		close: async () => finish?.({ ...okOutcome(conversationId!), status: "ERROR", aborted: true }),
		run: async (opts: DriverTurnRequest) => {
			if (active) await active.outcome;
			requests.push(opts);
			if (requests.length === 1) {
				conversationId = "conv-old";
				const outcome = new Promise<TurnOutcome>((resolve) => {
					finish = (settled) => {
						active = null;
						finish = undefined;
						roundTrips.failAll("turn ended");
						roundTrips.stampConversation("conv-old");
						resolve(settled);
					};
				});
				let pushExternal!: (activity: DriverActivity) => void;
				const activity = new Promise<DriverActivity>((resolve) => { pushExternal = resolve; });
				active = { id: "park", outcome, next: () => activity, pushExternal };
				return active;
			}
			const outcome = opts.prompt === "summarize" ? summaryOutcome : okOutcome(opts.conversationId ?? "conv-fresh");
			return { id: `turn-${requests.length}`, outcome: Promise.resolve(outcome), next: async () => null, pushExternal: () => {} };
		},
	};
	const roundTrips = new ToolRoundTrips(driver as unknown as StreamDriver, undefined, { escalateAfterMs: 0 });
	return { driver: driver as unknown as StreamDriver, roundTrips };
}

function okOutcome(conversationId: string, response = "ok"): TurnOutcome {
	return {
		status: "OK",
		response,
		finished: true,
		aborted: false,
		conversationId,
		modelOutputSeen: true,
		sawResult: true,
	};
}

async function drainTerminal(
	stream: ReturnType<ReturnType<typeof createStreamSimple>>,
): Promise<{ type: "done" | "error"; message: AssistantMessage }> {
	for await (const ev of stream) {
		if (ev.type === "done") return { type: "done", message: ev.message };
		if (ev.type === "error") return { type: "error", message: ev.error };
	}
	throw new Error("stream ended without a terminal event");
}

function withEnv(values: Record<string, string>, fn: () => Promise<void>): Promise<void> {
	const prev: Record<string, string | undefined> = {};
	for (const [k, v] of Object.entries(values)) {
		prev[k] = process.env[k];
		process.env[k] = v;
	}
	return fn().finally(() => {
		for (const [k, v] of Object.entries(prev)) {
			if (v === undefined) delete process.env[k];
			else process.env[k] = v;
		}
	});
}

describe("clampOverflowUsage and pi-ai silent-overflow parity", () => {
	test("proportionally clamps multi-inference billing sums that exceed contextWindow", () => {
		const usage = { input: 400_000, output: 20_000, cacheRead: 3_600_000, totalTokens: 280_000 };
		clampOverflowUsage(usage, 1_000_000);

		assert.equal(usage.input, 26_000);
		assert.equal(usage.cacheRead, 234_000);
		assert.equal(usage.input + usage.cacheRead, 260_000);
		assert.equal(usage.input + usage.cacheRead + usage.output, 280_000);
		assert.equal(usage.totalTokens, 280_000);

		const msg = {
			role: "assistant",
			stopReason: "stop",
			usage: { ...zeroUsage(), ...usage },
		} as AssistantMessage;
		assert.equal(isContextOverflow(msg, 1_000_000), false);
	});

	test("leaves normal turns below contextWindow untouched", () => {
		const usage = { input: 1000, output: 50, cacheRead: 800, totalTokens: 1450 };
		clampOverflowUsage(usage, 1_000_000);
		assert.deepEqual(usage, { input: 1000, output: 50, cacheRead: 800, totalTokens: 1450 });
	});

	test("preserves genuine silent overflow when totalTokens exceeds contextWindow", () => {
		const usage = { input: 300_000, output: 10_000, cacheRead: 900_000, totalTokens: 1_150_000 };
		clampOverflowUsage(usage, 1_000_000);
		assert.deepEqual(usage, { input: 300_000, output: 10_000, cacheRead: 900_000, totalTokens: 1_150_000 });

		const msg = {
			role: "assistant",
			stopReason: "stop",
			usage: { ...zeroUsage(), ...usage },
		} as AssistantMessage;
		assert.equal(isContextOverflow(msg, 1_000_000), true);
	});

	test("handles exact boundary (totalTokens === contextWindow) without rounding overshoot", () => {
		const usage = { input: 333_333, output: 0, cacheRead: 1_666_667, totalTokens: 1_000_000 };
		clampOverflowUsage(usage, 1_000_000);
		assert.equal(usage.input + usage.cacheRead, 1_000_000);

		const msg = {
			role: "assistant",
			stopReason: "stop",
			usage: { ...zeroUsage(), ...usage },
		} as AssistantMessage;
		assert.equal(isContextOverflow(msg, 1_000_000), false);
	});

	test("handles output >= totalTokens without producing negative values", () => {
		const usage = { input: 800_000, output: 50_000, cacheRead: 800_000, totalTokens: 40_000 };
		clampOverflowUsage(usage, 1_000_000);
		assert.equal(usage.input, 20_000);
		assert.equal(usage.cacheRead, 20_000);
		assert.equal(usage.input + usage.cacheRead, 40_000);
	});

	test("provider clamps multi-step usage on finalized stop turns (ACP & stream-json)", async () => {
		for (const engine of ["acp", "stream-json"] as const) {
			const dir = fs.mkdtempSync(path.join(os.tmpdir(), "agy-overflow-"));
			try {
				const requests: DriverTurnRequest[] = [];
				const driver = makeFakeDriver(
					[
						{
							outcome: okOutcome("conv-1"),
							activities: [
								{
									type: "usage",
									usage: {
										input_tokens: 500_000,
										output_tokens: 15_000,
										cache_read_tokens: 4_500_000,
										total_tokens: 215_000,
									},
								},
								{ type: "text", delta: "done" },
							],
						},
					],
					requests,
				);
				const store = new SessionStore(path.join(dir, "sessions.json"));
				const streamSimple = createStreamSimple({
					entries: [{ id: model.id, full: "gemini-3.8-flash" }],
					store,
					driver,
					...(engine === "acp" ? { acpDriver: driver, engine } : { engine }),
					roundTrips: new ToolRoundTrips(driver),
				});
				const terminal = await drainTerminal(
					streamSimple(
						model,
						normalizeContext({ messages: [user("run tests")] }),
						{ cwd: dir, sessionId: "s-overflow" } as unknown as SimpleStreamOptions,
					),
				);
				assert.equal(terminal.type, "done");
				assert.equal(isContextOverflow(terminal.message, model.contextWindow), false);
				assert.equal(terminal.message.usage.totalTokens, 215_000);
				assert.equal(terminal.message.usage.input + terminal.message.usage.cacheRead, 200_000);
			} finally {
				fs.rmSync(dir, { recursive: true, force: true });
			}
		}
	});
});

describe("ephemeral turn isolation (cacheRetention: 'none')", () => {
	test("does not persist throwaway summarization session IDs to SessionStore", async () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "agy-ephemeral-"));
		try {
			const requests: DriverTurnRequest[] = [];
			const driver = makeFakeDriver([{ outcome: okOutcome("conv-throwaway", "summary") }], requests);
			const store = new SessionStore(path.join(dir, "sessions.json"));
			const streamSimple = createStreamSimple({
				entries: [{ id: model.id, full: "gemini-3.8-flash" }],
				store,
				driver,
				acpDriver: driver,
				engine: "acp",
				roundTrips: new ToolRoundTrips(driver),
			});

			const terminal = await drainTerminal(
				streamSimple(
					model,
					normalizeContext({ systemPrompt: "Summarize conversation.", messages: [user("transcript")] }),
					{ cwd: dir, sessionId: "random-uuidv7", cacheRetention: "none" } as unknown as SimpleStreamOptions,
				),
			);
			assert.equal(terminal.type, "done");
			assert.equal(requests[0]?.conversationId, null);
			assert.equal(store.size, 0);
		} finally {
			fs.rmSync(dir, { recursive: true, force: true });
		}
	});

	test("runs fresh without reading, mutating, or deleting a live bound session or arming background watchers", async () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "agy-ephemeral-live-"));
		try {
			const requests: DriverTurnRequest[] = [];
			const backgroundArms: Array<{ conversationId: string; turnStartMs: number }> = [];
			const driver = makeFakeDriver(
				[
					{ outcome: okOutcome("conv-live", "turn 1") },
					{
						outcome: okOutcome("conv-summary", "compacted summary"),
						activities: [
							{
								type: "tool_start",
								name: "run_command",
								args: { CommandLine: "sleep 1", WaitMsBeforeAsync: 500 },
							},
							{ type: "text", delta: "compacted summary" },
						],
					},
					// Simulate an ephemeral silent-drop failure: must not delete the live binding.
					{
						outcome: {
							status: "OK",
							response: "",
							finished: true,
							aborted: false,
							conversationId: "conv-drop",
							modelOutputSeen: false,
							sawResult: true,
						},
					},
					{ outcome: okOutcome("conv-live", "turn 2") },
				],
				requests,
			);
			const store = new SessionStore(path.join(dir, "sessions.json"));
			const rt = new ToolRoundTrips(driver);
			const streamSimple = createStreamSimple({
				entries: [{ id: model.id, full: "gemini-3.8-flash" }],
				store,
				driver,
				acpDriver: driver,
				engine: "acp",
				roundTrips: rt,
				onNativeBackground: (info) => backgroundArms.push(info),
			});
			const baseOpts = { cwd: dir, sessionId: "live-sess" } as unknown as SimpleStreamOptions;
			const key = sessionKey(baseOpts, dir, "acp");

			// Turn 1 binds conv-live.
			await drainTerminal(streamSimple(model, normalizeContext({ messages: [user("hello")] }), baseOpts));
			assert.equal(store.get(key)?.conversationId, "conv-live");
			const originalWatermark = store.get(key)?.lastMessageCount;

			// Ephemeral summarization with the SAME sessionId must start fresh and leave the binding alone.
			const sumTerminal = await drainTerminal(
				streamSimple(
					model,
					normalizeContext({ systemPrompt: "Summarize", messages: [user("summarize this")] }),
					{ ...baseOpts, cacheRetention: "none" } as unknown as SimpleStreamOptions,
				),
			);
			assert.equal(sumTerminal.type, "done");
			assert.equal(requests[1]?.conversationId, null);
			assert.equal(store.get(key)?.conversationId, "conv-live");
			assert.equal(store.get(key)?.lastMessageCount, originalWatermark);
			assert.equal(backgroundArms.length, 0);

			// Ephemeral silent-drop (tested via disableContinuation: true) must NOT clear the live binding.
			const dropTerminal = await drainTerminal(
				streamSimple(
					model,
					normalizeContext({ messages: [user("huge summarization prompt")] }),
					{ ...baseOpts, disableContinuation: true } as unknown as SimpleStreamOptions,
				),
			);
			assert.equal(dropTerminal.type, "error");
			assert.equal(store.get(key)?.conversationId, "conv-live");

			// Next real turn still resumes conv-live cleanly.
			await drainTerminal(
				streamSimple(model, normalizeContext({ messages: [user("hello"), user("follow-up")] }), baseOpts),
			);
			assert.equal(requests[3]?.conversationId, "conv-live");
		} finally {
			fs.rmSync(dir, { recursive: true, force: true });
		}
	});

	test("ephemeral turn does not steal a live session's dead park tombstone or stranded escalation", async () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "agy-ephemeral-late-"));
		try {
			const requests: DriverTurnRequest[] = [];
			const activeHandle = {
				id: "parked-turn",
				outcome: new Promise<never>(() => {}),
				next: async () => null,
				pushExternal: () => {},
			};
			const driver = {
				get activeHandle() {
					return activeHandle;
				},
				kickIdle: () => {},
				reentry: () => null,
				snapshot: () => ({ state: "ready", conversationId: "conv-live", stats: {} }),
				run: async (opts: DriverTurnRequest) => {
					requests.push(opts);
					return {
						id: `fake-turn-${requests.length}`,
						outcome: Promise.resolve(okOutcome(opts.conversationId ?? "conv-new", "ok")),
						next: async () => null,
						pushExternal: () => {},
					} satisfies TurnHandle as unknown as TurnHandle;
				},
			} as unknown as StreamDriver;

			const store = new SessionStore(path.join(dir, "sessions.json"));
			const rt = new ToolRoundTrips(driver, undefined, { escalateAfterMs: 1 });
			const opts = { cwd: dir, sessionId: "late-sess" } as unknown as SimpleStreamOptions;
			const key = sessionKey(opts, dir, "acp");
			store.set(key, { conversationId: "conv-live", lastStepIdx: -1, lastMessageCount: 1 });

			// Create a dead park tombstone (aborted/expired park whose tool result hasn't arrived yet).
			const deadSettled = rt
				.onToolCall("dead-call-1", "AskClaude", {}, new AbortController().signal)
				.then(
					() => "resolved",
					() => "rejected",
				);
			rt.failAll("expired");
			await deadSettled;
			assert.deepEqual(rt.deadIds, ["dead-call-1"]);

			// Create a done-but-undelivered escalated call.
			const esc = await rt.onToolCall("esc-call-1", "ask_user_question", {}, new AbortController().signal);
			assert.equal("escalated" in esc && esc.escalated, true);
			rt.stampConversation("conv-live");
			rt.resolve("esc-call-1", "user answered yes", false);

			const streamSimple = createStreamSimple({
				entries: [{ id: model.id, full: "gemini-3.8-flash" }],
				store,
				driver,
				acpDriver: driver,
				engine: "acp",
				roundTrips: rt,
			});

			// Ephemeral summarization turn must NOT claim either the dead tombstone or the stranded escalation.
			const ephTerminal = await drainTerminal(
				streamSimple(
					model,
					normalizeContext({
						systemPrompt: "Summarize",
						messages: [toolResultMsg("dead-call-1", "AskClaude", "LATE"), user("summarize")],
					}),
					{ ...opts, cacheRetention: "none" } as unknown as SimpleStreamOptions,
				),
			);
			assert.equal(ephTerminal.type, "done");
			assert.equal(requests[0]?.conversationId, null);
			assert.equal(requests[0]?.prompt.includes("Late tool delivery"), false);
			assert.deepEqual(rt.deadIds, ["dead-call-1"]);

			// The next real turn claims and delivers both cleanly into conv-live.
			const realTerminal = await drainTerminal(
				streamSimple(
					model,
					normalizeContext({
						messages: [toolResultMsg("dead-call-1", "AskClaude", "LATE"), user("continue")],
					}),
					opts,
				),
			);
			assert.equal(realTerminal.type, "done");
			assert.equal(requests[1]?.conversationId, "conv-live");
			assert.match(requests[1]?.prompt ?? "", /Late tool delivery/);
			assert.match(requests[1]?.prompt ?? "", /LATE/);
			assert.match(requests[1]?.prompt ?? "", /user answered yes/);
			assert.deepEqual(rt.deadIds, []);
		} finally {
			fs.rmSync(dir, { recursive: true, force: true });
		}
	});
});

describe.each(["acp", "stream-json"] as const)("%s: compaction during a queued bridge park", (engine) => {
	test.each(["success", "failure", "aborted"] as const)("summarizer %s releases the queue without rotating the binding", async (result) => {
		await withEnv({ AGY_DIGEST: "0", AGY_SYSTEM_PROMPT: "off" }, async () => {
			const dir = fs.mkdtempSync(path.join(fakeAgentRoot, "queued-"));
			const requests: DriverTurnRequest[] = [];
			const summary = okOutcome("conv-summary", "compacted progress");
			if (result !== "success") {
				summary.status = "ERROR";
				summary.aborted = result === "aborted";
			}
			const { driver, roundTrips } = makeQueuedDriver(requests, summary);
			try {
				const store = new SessionStore(path.join(dir, "sessions.json"));
				const handoff = new CompactionHandoffs();
				const stream = createStreamSimple({
					entries: [], store, driver, roundTrips, engine,
					acpDriver: engine === "acp" ? driver : undefined,
					compactionHandoffs: handoff,
				});
				const opts = { cwd: dir, sessionId: "queued-park" } as unknown as SimpleStreamOptions;
				const key = sessionKey(opts, dir, engine);
				const firstPrompt = user("initial task");
				const first = drainTerminal(stream(model, normalizeContext({ messages: [firstPrompt] }), opts));
				// The first run acquires the driver before requesting a Pi tool.
				await Promise.resolve();
				const park = roundTrips.onToolCall("park-1", "read", {}, new AbortController().signal);
				// Handle the rejection immediately, but assert its reason below.
				const rejected = assert.rejects(park, /turn ended/);
				const turn = await first;
				assert.equal(turn.message.stopReason, "toolUse");
				const oldBinding = store.get(key);
				assert.equal(oldBinding?.conversationId, "conv-old");
				const parkedHandle = driver.activeHandle;
				assert.ok(parkedHandle);

				const ephemeralOptions = result === "aborted"
					? { ...opts, disableContinuation: true }
					: { ...opts, cacheRetention: "none" as const };
				const summarized = await beforeDeadline(drainTerminal(stream(model, normalizeContext({ messages: [user("summarize")] }), ephemeralOptions)));
				assert.equal(summarized.type, result === "success" ? "done" : "error");
				assert.equal(summarized.message.stopReason, result === "aborted" ? "aborted" : result === "failure" ? "error" : "stop");
				assert.equal(requests[1]?.conversationId, null);
				assert.equal((await parkedHandle.outcome).aborted, true);
				await rejected;
				assert.deepEqual(roundTrips.deadIds, ["park-1"]);
				assert.deepEqual(store.get(key), oldBinding);
				assert.equal(handoff.hasPending(key), false);

				const resultMessage = toolResultMsg("park-1", "read", "retained tool output");
				if (result === "success") {
					assert.equal(await rotateSessionOnCompaction(store, key, handoff, driver, roundTrips), "conv-old");
					const next = await drainTerminal(stream(model, normalizeContext({
						messages: [compaction(summary.response), turn.message, resultMessage],
					}), opts));
					assert.equal(next.type, "done");
					assertCompactionResume(requests[2], summary.response, "retained tool output");
					assert.equal(store.get(key)?.conversationId, "conv-fresh");
					assert.equal(handoff.hasPending(key), false);
				} else {
					const next = await drainTerminal(stream(model, normalizeContext({
						messages: [firstPrompt, turn.message, resultMessage],
					}), opts));
					assert.equal(next.type, "done");
					assert.equal(requests[2]?.conversationId, "conv-old");
					assert.match(requests[2]?.prompt ?? "", /Late tool delivery[\s\S]*retained tool output/);
					assert.equal(store.get(key)?.conversationId, "conv-old");
				}
				assert.deepEqual(roundTrips.deadIds, []);
			} finally {
				await driver.close("shutdown");
				fs.rmSync(dir, { recursive: true, force: true });
			}
		});
	});
});

function beforeDeadline<T>(result: Promise<T>): Promise<T> {
	let timer: ReturnType<typeof setTimeout> | undefined;
	const deadline = new Promise<never>((_, reject) => {
		timer = setTimeout(() => reject(new Error("summary is stuck behind the parked turn")), 2000);
	});
	return Promise.race([result, deadline]).finally(() => clearTimeout(timer));
}

function assertCompactionResume(request: DriverTurnRequest | undefined, summary: string, toolOutput: string): void {
	assert.ok(request);
	assert.equal(request.conversationId, null);
	assert.ok(request.prompt.endsWith(COMPACTION_RESUME_PROMPT));
	assert.doesNotMatch(request.prompt, /Late tool delivery/);
	const history = request.contextBlock?.text ?? request.prompt;
	assert.ok(history.includes(summary), `summary missing from ${history}`);
	assert.ok(history.includes(toolOutput), `tool result missing from ${history}`);
}

test.each(["the same model", "another model"])("session_compact with %s resumes fresh without warning toasts", async (summarizer) => {
	const dir = fs.mkdtempSync(path.join(fakeAgentRoot, "extension-"));
	const bin = path.join(import.meta.dirname, "helpers", "fake-agy-bin", "agy");
	await withEnv({
		HOME: dir, TMPDIR: dir, PI_CODING_AGENT_DIR: path.join(dir, "agent"), AGY_BIN: bin,
		AGY_ENGINE: "stream-json", AGY_AGENTS_ROOT: dir, AGY_APPROVALS: "off", AGY_BRIDGE_TOOLS: "all",
		AGY_ASK_TOOL: "off", AGY_WEB_TOOLS: "off", AGY_DIGEST: "0", AGY_SYSTEM_PROMPT: "off",
	}, async () => {
		const realm = bridgeRealm();
		const originalJoin = realm.join.bind(realm);
		let bridge: McpBridgeDeps | undefined;
		let mcpLog: ((event: string, data: unknown) => void) | undefined;
		const join = vi.spyOn(realm, "join").mockImplementation((port) => {
			const member = originalJoin(port);
			return { ...member, attachBridge: async (deps, log) => {
				bridge = deps;
				mcpLog = log;
				return { ok: true, first: false, handle: {
					port: 1, token: "test", realmToken: "test", close: async () => {},
					approvals: { has: () => false, resolve: () => false },
				} };
			} };
		});
		const run = vi.spyOn(StreamDriver.prototype, "run");
		const handlers = new Map<string, Array<(event: unknown, ctx: unknown) => void | Promise<void>>>();
		const notifications: string[] = [];
		const opts = { cwd: dir, sessionId: "ext-compact" } as unknown as SimpleStreamOptions;
		const key = sessionKey(opts, dir, "stream-json");
		const ctx = {
			cwd: dir, hasUI: true, mode: "print", isProjectTrusted: () => false,
			sessionManager: { getSessionId: () => opts.sessionId, getEntries: () => [] },
			ui: {
				notify: (msg: string, level: string) => { if (level === "warning") notifications.push(msg); },
				setStatus: () => {},
			},
		};
		const emit = async (name: string, event: unknown) => {
			for (const h of handlers.get(name) ?? []) await h(event, ctx);
		};
		const api = {
			on: (name: string, handler: (event: unknown, ctx: unknown) => void | Promise<void>) => {
				if (!handlers.has(name)) handlers.set(name, []);
				handlers.get(name)!.push(handler);
			},
			getAllTools: () => [{ name: "fixture_tool", parameters: { type: "object" } }],
			getActiveTools: () => ["fixture_tool"],
			registerProvider: () => {}, registerCommand: () => {}, registerTool: () => {},
		} as unknown as ExtensionAPI;
		try {
			await factory(api);
			await emit("session_start", { reason: "startup" });
			assert.ok(bridge);
			const dispatch = (messages: Message[], options = opts) =>
				drainTerminal(realm.streamSimple(model, normalizeContext({ messages }), options));
			const turn = dispatch([user("HANG")]);
			const driver = run.mock.contexts[0] as StreamDriver;
			await vi.waitFor(() => assert.equal(driver.snapshot().conversationId, "conv-hang"));
			const park = bridge.onToolCall("ext-park", "fixture_tool", {}, new AbortController().signal);
			const rejected = park.then(() => false, (err: Error) => {
				// The MCP server logs the rejection before converting it into a tool error.
				mcpLog?.("call-tool-fail", { name: "fixture_tool", msg: err.message });
				return true;
			});
			const first = await turn;
			assert.equal(first.message.stopReason, "toolUse");
			assert.equal(realm.store.get(key)?.conversationId, "conv-hang");

			if (summarizer === "the same model") {
				const result = await beforeDeadline(dispatch([user("summarize")], { ...opts, cacheRetention: "none" }));
				assert.equal(result.type, "done");
				assert.equal(realm.store.get(key)?.conversationId, "conv-hang");
			} else {
				// Also cover a record removed before the compaction event fires.
				realm.store.delete(key);
			}
			await emit("session_compact", { reason: "threshold" });
			assert.equal(driver.activeHandle, null);
			assert.equal(await rejected, true);
			assert.equal(realm.store.get(key), null);

			const next = await dispatch([
				compaction("compact the completed work"), first.message, toolResultMsg("ext-park", "fixture_tool", "completed tool output"),
			]);
			assert.equal(next.type, "done");
			assertCompactionResume(run.mock.lastCall?.[0], "compact the completed work", "completed tool output");
			assert.equal(realm.store.get(key)?.conversationId, "conv-777");
			assert.deepEqual(notifications, []);
		} finally {
			await emit("session_shutdown", { reason: "quit" });
			await __resetRealmForTests();
			run.mockRestore();
			join.mockRestore();
			fs.rmSync(dir, { recursive: true, force: true });
		}
	});
});

describe("post-compaction session rotation and handoff", () => {
	test("rotateSessionOnCompaction clears bound key and returns rotated conversation id", async () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "agy-rotate-"));
		try {
			const store = new SessionStore(path.join(dir, "sessions.json"));
			const handoff = new CompactionHandoffs();
			const key = "sid:s1@acp";
			const driver = makeFakeDriver([], []);
			const rt = new ToolRoundTrips(driver);

			assert.equal(await rotateSessionOnCompaction(store, key, handoff, driver, rt), null);
			assert.equal(handoff.hasPending(key), false);

			store.set(key, { conversationId: "conv-old", lastStepIdx: -1, lastMessageCount: 10 });
			assert.equal(await rotateSessionOnCompaction(store, key, handoff, driver, rt), "conv-old");
			assert.equal(store.get(key), null);
			assert.equal(handoff.hasPending(key), true);
		} finally {
			fs.rmSync(dir, { recursive: true, force: true });
		}
	});

	test.each(["unrotated", "delivered handoff", "pending handoff"])("clearing bindings mid-stream lets the turn complete (%s)", async (state) => {
		await withEnv({ AGY_HANDOFF: "digest", AGY_SYSTEM_PROMPT: "off" }, async () => {
			const dir = fs.mkdtempSync(path.join(fakeAgentRoot, "clear-mid-stream-"));
			const store = new SessionStore(path.join(dir, "sessions.json"));
			try {
				const handoff = new CompactionHandoffs();
				const driver = makeFakeDriver([{
					outcome: okOutcome("conv-streaming", "before after"),
					activities: [{ type: "text", delta: "before " }, { type: "text", delta: "after" }],
				}], []);
				const roundTrips = new ToolRoundTrips(driver);
				const opts = { cwd: dir, sessionId: "clear-mid-stream" } as unknown as SimpleStreamOptions;
				const key = sessionKey(opts, dir, "stream-json");
				const stream = createStreamSimple({
					entries: [], store, driver, roundTrips, engine: "stream-json", compactionHandoffs: handoff,
				});
				const dispatch = (messages: Message[]) => drainTerminal(stream(model, normalizeContext({ messages }), opts));
				store.set(key, { conversationId: "conv-streaming", lastStepIdx: -1 });
				if (state !== "unrotated") await rotateSessionOnCompaction(store, key, handoff, driver, roundTrips);
				if (state === "delivered handoff") {
					assert.equal((await dispatch([compaction("prior summary"), user("continue")])).type, "done");
					assert.equal(handoff.hasPending(key), false);
				}

				let pendingAfterClear: boolean | undefined;
				const active = vi.spyOn(driver, "activeHandle", "get");
				const run = driver.run.bind(driver);
				vi.spyOn(driver, "run").mockImplementationOnce(async (request) => {
					const handle = await run(request);
					active.mockReturnValue(handle);
					let textChunks = 0;
					return {
						...handle,
						next: async () => {
							const activity = await handle.next();
							if (activity?.type === "text" && ++textChunks === 2) {
								// /agy clear does not close or cancel the active driver.
								store.clear();
								handoff.clear();
								pendingAfterClear = handoff.hasPending(key);
							}
							if (activity === null) active.mockReturnValue(null);
							return activity;
						},
					};
				});
				const result = await dispatch([compaction("prior summary"), user("keep going")]);
				assert.equal(result.message.stopReason, "stop");
				assert.deepEqual(result.message.content, [{ type: "text", text: "before after" }]);
				assert.equal(driver.activeHandle, null);
				assert.equal(pendingAfterClear, false);
			} finally {
				await store.flush();
				fs.rmSync(dir, { recursive: true, force: true });
			}
		});
	});

	test("ACP sync retries a quota-rejected handoff with the summary and commits its new checkpoint", async () => {
		await withEnv({ AGY_HANDOFF: "sync", AGY_DIGEST: "0" }, async () => {
			const dir = fs.mkdtempSync(path.join(fakeAgentRoot, "sync-quota-"));
			try {
				const store = new SessionStore(path.join(dir, "sessions.json"));
				const handoff = new CompactionHandoffs();
				const requests: DriverTurnRequest[] = [];
				const driver = makeFakeDriver([
					{ syncBind: true, outcome: { ...okOutcome("conv-quota", "Quota exhausted; resets in 10s"), promptTerminal: true } },
					{ syncBind: true, outcome: { ...okOutcome("conv-retry"), promptTerminal: true } },
				], requests);
				const roundTrips = new ToolRoundTrips(driver);
				const opts = { cwd: dir, sessionId: "sync-quota" } as unknown as SimpleStreamOptions;
				const key = sessionKey(opts, dir, "acp");
				store.set(key, { conversationId: "conv-old", lastStepIdx: -1 });
				await rotateSessionOnCompaction(store, key, handoff, driver, roundTrips);
				const stream = createStreamSimple({
					entries: [], store, driver, acpDriver: driver, roundTrips, engine: "acp", compactionHandoffs: handoff,
				});
				const summary = "Keep the compacted task history";
				const context = normalizeContext({ messages: [compaction(summary), user("continue")] });
				const rejected = await drainTerminal(stream(model, context, opts));
				assert.equal(rejected.type, "error");
				assert.match(rejected.message.errorMessage ?? "", /rate limit/);
				assert.equal(store.get(key)?.conversationId, "conv-quota");
				assert.ok(store.get(key)?.sync?.inFlight);
				assert.equal(handoff.hasPending(key), true);
				const retried = await drainTerminal(stream(model, context, opts));
				assert.equal(retried.type, "done");
				for (const request of requests) {
					assert.equal(request.conversationId, null);
					assert.ok(request.contextBlock?.text.includes(summary));
				}
				const record = store.get(key);
				assert.equal(record?.conversationId, "conv-retry");
				assert.ok(record?.sync?.checkpoint);
				assert.equal(record?.sync?.inFlight, undefined);
				assert.equal(handoff.hasPending(key), false);
			} finally {
				fs.rmSync(dir, { recursive: true, force: true });
			}
		});
	});

	test.each(["ordinary turn", "earlier handoff", "already replaced handoff", "handoff with buffered tool call"])("an in-flight ACP sync %s cannot restore a rotated binding", async (kind) => {
		await withEnv({ AGY_HANDOFF: "sync" }, async () => {
			const dir = fs.mkdtempSync(path.join(fakeAgentRoot, "rotate-sync-"));
			try {
				const store = new SessionStore(path.join(dir, "sessions.json"));
				const handoff = new CompactionHandoffs();
				const bufferedCall = kind === "handoff with buffered tool call";
				const driver = makeFakeDriver([
					{
						syncBind: true,
						outcome: { ...okOutcome("conv-in-flight"), promptTerminal: true },
						activities: bufferedCall ? [{ type: "bridge_call", callId: "old-call", name: "read", args: {} }] : [],
					},
					{ syncBind: true, outcome: { ...okOutcome("conv-new"), promptTerminal: true } },
				], []);
				const roundTrips = new ToolRoundTrips(driver);
				const opts = { cwd: dir, sessionId: "rotate-sync" } as unknown as SimpleStreamOptions;
				const key = sessionKey(opts, dir, "acp");
				store.set(key, { conversationId: "conv-old", lastStepIdx: -1 });
				if (kind !== "ordinary turn") await rotateSessionOnCompaction(store, key, handoff, driver, roundTrips);
				const stream = createStreamSimple({
					entries: [], store, driver, acpDriver: driver, roundTrips, engine: "acp", compactionHandoffs: handoff,
				});
				const alreadyReplaced = kind === "already replaced handoff";
				const run = driver.run.bind(driver);
				vi.spyOn(driver, "run").mockImplementationOnce(async (request) => {
					const handle = await run(request);
					assert.equal(await rotateSessionOnCompaction(store, key, handoff, driver, roundTrips), "conv-in-flight");
					if (alreadyReplaced) {
						const newest = normalizeContext({ messages: [compaction("new summary"), user("continue anew")] });
						assert.equal((await drainTerminal(stream(model, newest, opts))).type, "done");
						assert.equal(store.get(key)?.conversationId, "conv-new");
					}
					return handle;
				});
				const superseded = await drainTerminal(stream(model, normalizeContext({ messages: [user("continue")] }), opts));
				if (bufferedCall) assert.equal(superseded.message.stopReason, "aborted");
				assert.equal(store.get(key)?.conversationId ?? null, alreadyReplaced ? "conv-new" : null);
				assert.equal(handoff.hasPending(key), !alreadyReplaced);
			} finally {
				fs.rmSync(dir, { recursive: true, force: true });
			}
		});
	});

	test("rotation discards orphaned bridge state even without a binding or live turn", async () => {
		const dir = fs.mkdtempSync(path.join(fakeAgentRoot, "orphaned-"));
		vi.useFakeTimers();
		try {
			const store = new SessionStore(path.join(dir, "sessions.json"));
			const handoff = new CompactionHandoffs();
			const key = "sid:orphaned@acp";
			const driver = makeFakeDriver([], []);
			const active = vi.spyOn(driver, "activeHandle", "get").mockReturnValue({
				id: "parked", outcome: new Promise<never>(() => {}), next: async () => null, pushExternal: () => {},
			});
			const rt = new ToolRoundTrips(driver, undefined, { escalateAfterMs: (name) => name === "slow" ? 10 : 0 });
			const deny = vi.fn(() => true);
			rt.approvalPark = { has: () => true, resolve: deny };
			const signal = new AbortController().signal;
			const running = rt.onToolCall("running", "slow", {}, signal);
			const stranded = rt.onToolCall("stranded", "slow", {}, signal);
			await vi.advanceTimersByTimeAsync(10);
			assert.equal("escalated" in (await running), true);
			assert.equal("escalated" in (await stranded), true);
			rt.stampConversation("conv-old");
			rt.resolve("stranded", "never polled", false);
			const parked = rt.onToolCall("parked", "read", {}, signal).then(() => "resolved", (err: Error) => err.message);
			rt.onApproval("approval", { toolCall: { name: "run_command", args: { CommandLine: "pwd" } } });
			assert.equal(getEventListeners(signal, "abort").length, 2);

			active.mockReturnValue(null);
			assert.equal(await rotateSessionOnCompaction(store, key, handoff, driver, rt), null);
			assert.equal(handoff.hasPending(key), true);
			assert.match(await parked, /compacted/);
			assert.deepEqual(rt.pendingIds, []);
			assert.deepEqual(rt.deadIds, []);
			assert.equal(rt.poll("running"), undefined);
			assert.equal(rt.poll("stranded"), undefined);
			assert.equal(getEventListeners(signal, "abort").length, 0);
			assert.equal(vi.getTimerCount(), 0);
			assert.equal(deny.mock.calls.length, 1);
			assert.deepEqual(deny.mock.lastCall, ["approval", { allow: false, reason: "antigravity conversation was compacted" }]);
			assert.equal(rt.resolve("approval", "approved", false), false);
		} finally {
			vi.useRealTimers();
			fs.rmSync(dir, { recursive: true, force: true });
		}
	});

	test("rotates to a fresh agy conversation and delivers compaction summary once even with AGY_DIGEST=0", async () => {
		await withEnv({ AGY_DIGEST: "0", AGY_SYSTEM_PROMPT: "off" }, async () => {
			const dir = fs.mkdtempSync(path.join(os.tmpdir(), "agy-compact-handoff-"));
			try {
				const requests: DriverTurnRequest[] = [];
				const driver = makeFakeDriver(
					[
						{ outcome: okOutcome("conv-old", "before compact") },
						// First post-compaction attempt fails: handoff must survive for the retry.
						{
							outcome: {
								status: "ERROR",
								response: "",
								error: "transient failure",
								finished: true,
								aborted: false,
								conversationId: "conv-failed",
							},
						},
						{ outcome: okOutcome("conv-fresh", "after compact") },
						{ outcome: okOutcome("conv-fresh", "steady state") },
					],
					requests,
				);
				const store = new SessionStore(path.join(dir, "sessions.json"));
				const handoff = new CompactionHandoffs();
				const roundTrips = new ToolRoundTrips(driver);
				const streamSimple = createStreamSimple({
					entries: [{ id: model.id, full: "gemini-3.8-flash" }],
					store,
					driver,
					acpDriver: driver,
					engine: "acp",
					roundTrips,
					compactionHandoffs: handoff,
				});
				const opts = { cwd: dir, sessionId: "compact-sess" } as unknown as SimpleStreamOptions;
				const key = sessionKey(opts, dir, "acp");

				// Turn 1 binds conv-old.
				await drainTerminal(streamSimple(model, normalizeContext({ messages: [user("initial task")] }), opts));
				assert.equal(store.get(key)?.conversationId, "conv-old");

				// Simulate pi session_compact hook.
				assert.equal(await rotateSessionOnCompaction(store, key, handoff, driver, roundTrips), "conv-old");

				const compactedCtx: TranscriptContext = normalizeContext({
					messages: [compaction("Key decision: keep SQLite schema v2."), user("continue task")],
				});

				// Failed attempt starts fresh with the summary and preserves handoff for retry.
				const failed = await drainTerminal(streamSimple(model, compactedCtx, opts));
				assert.equal(failed.type, "error");
				assert.equal(requests[1]?.conversationId, null);
				assert.match(requests[1]?.contextBlock?.text ?? "", /keep SQLite schema v2/);
				assert.equal(handoff.hasPending(key), true);

				// Retry succeeds in a fresh conversation, binds conv-fresh, and clears the one-shot handoff.
				const recovered = await drainTerminal(streamSimple(model, compactedCtx, opts));
				assert.equal(recovered.type, "done");
				assert.equal(requests[2]?.conversationId, null);
				assert.match(requests[2]?.contextBlock?.text ?? "", /keep SQLite schema v2/);
				assert.equal(store.get(key)?.conversationId, "conv-fresh");
				assert.equal(handoff.hasPending(key), false);

				// Steady-state turn with AGY_DIGEST=0 resumes conv-fresh with no digest.
				await drainTerminal(
					streamSimple(
						model,
						normalizeContext({
							messages: [
								compaction("Key decision: keep SQLite schema v2."),
								user("continue task"),
								assistantMsg("after compact"),
								user("next question"),
							],
						}),
						opts,
					),
				);
				assert.equal(requests[3]?.conversationId, "conv-fresh");
				assert.equal(requests[3]?.contextBlock, undefined);
			} finally {
				fs.rmSync(dir, { recursive: true, force: true });
			}
		});
	});

	test("buildContextDigest with includeLast: true preserves trailing toolResult after compaction", () => {
		const msgs = [
			compaction("Finished step 1"),
			assistantMsg("Running build check"),
			toolResultMsg("call-9", "bash", "tests passed: 42"),
		];
		const digest = buildContextDigest(msgs, 0, { bound: false, includeLast: true });
		assert.match(digest, /Finished step 1/);
		assert.match(digest, /Running build check/);
		assert.match(digest, /\[tool result: bash\]/);
		assert.match(digest, /tests passed: 42/);
	});

	test("post-compaction mid-tool-loop tail resumes into a fresh conversation instead of hard-erroring", async () => {
		for (const engine of ["acp", "stream-json"] as const) {
			await withEnv({ AGY_DIGEST: "0", AGY_SYSTEM_PROMPT: "off" }, async () => {
				const dir = fs.mkdtempSync(path.join(os.tmpdir(), "agy-compact-resume-"));
				try {
					const requests: DriverTurnRequest[] = [];
					const driver = makeFakeDriver(
						[{ outcome: okOutcome("conv-old") }, { outcome: okOutcome("conv-fresh") }],
						requests,
					);
					const store = new SessionStore(path.join(dir, "sessions.json"));
					const handoff = new CompactionHandoffs();
					const roundTrips = new ToolRoundTrips(driver);
					const streamSimple = createStreamSimple({
						entries: [{ id: model.id, full: "gemini-3.8-flash" }],
						store,
						driver,
						...(engine === "acp" ? { acpDriver: driver, engine } : { engine }),
						roundTrips,
						compactionHandoffs: handoff,
					});
					const opts = { cwd: dir, sessionId: `mid-loop-${engine}` } as unknown as SimpleStreamOptions;
					const key = sessionKey(opts, dir, engine);

					await drainTerminal(streamSimple(model, normalizeContext({ messages: [user("start")] }), opts));
					await rotateSessionOnCompaction(store, key, handoff, driver, roundTrips);

					const terminal = await drainTerminal(
						streamSimple(
							model,
							normalizeContext({
								messages: [
									compaction("Compacted progress up to tool call."),
									assistantMsg("Inspecting test output"),
									toolResultMsg("call-42", "bash", "3 failures in parser.test.ts"),
								],
							}),
							opts,
						),
					);
					assert.equal(terminal.type, "done");
					assert.equal(requests[1]?.conversationId, null);
					assert.equal(store.get(key)?.conversationId, "conv-fresh");
					assert.equal(handoff.hasPending(key), false);

					if (engine === "acp") {
						assert.equal(requests[1]?.prompt, COMPACTION_RESUME_PROMPT);
						assert.match(requests[1]?.contextBlock?.text ?? "", /Compacted progress up to tool call/);
						assert.match(requests[1]?.contextBlock?.text ?? "", /3 failures in parser\.test\.ts/);
					} else {
						assert.ok(requests[1]?.prompt.includes(DIGEST_PREAMBLE));
						assert.ok(requests[1]?.prompt.includes("Compacted progress up to tool call"));
						assert.ok(requests[1]?.prompt.includes("3 failures in parser.test.ts"));
						assert.ok(requests[1]?.prompt.endsWith(COMPACTION_RESUME_PROMPT));
					}
				} finally {
					fs.rmSync(dir, { recursive: true, force: true });
				}
			});
		}
	});

	test("unbound toolResult tail WITHOUT compaction still errors with 'No user message to send to agy.'", async () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "agy-unbound-tail-"));
		try {
			const requests: DriverTurnRequest[] = [];
			const driver = makeFakeDriver([{ outcome: okOutcome("conv-unused") }], requests);
			const store = new SessionStore(path.join(dir, "sessions.json"));
			const streamSimple = createStreamSimple({
				entries: [{ id: model.id, full: "gemini-3.8-flash" }],
				store,
				driver,
				acpDriver: driver,
				engine: "acp",
				roundTrips: new ToolRoundTrips(driver),
			});
			const terminal = await drainTerminal(
				streamSimple(
					model,
					normalizeContext({
						messages: [user("do task"), assistantMsg("calling"), toolResultMsg("call-1", "bash", "ok")],
					}),
					{ cwd: dir, sessionId: "never-bound" } as unknown as SimpleStreamOptions,
				),
			);
			assert.equal(terminal.type, "error");
			assert.equal(terminal.message.errorMessage, "No user message to send to agy.");
			assert.equal(requests.length, 0);
		} finally {
			fs.rmSync(dir, { recursive: true, force: true });
		}
	});

	test("an armed handoff does not re-enter a stale park", async () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "agy-rotate-midpark-"));
		try {
			const requests: DriverTurnRequest[] = [];
			// The parked agy turn lives in the OLD conversation and is still
			// running: a live activeHandle parks the bridge call, and a live
			// reentry handle whose outcome still reports that id is the mid-park
			// continuation shape.
			const parkedHandle = {
				id: "original-turn",
				outcome: new Promise<never>(() => {}),
				next: async () => null,
				pushExternal: () => {},
			};
			const reentryHandle = {
				id: "parked-turn",
				outcome: Promise.resolve(okOutcome("conv-old", "parked work finished")),
				next: async () => null,
				pushExternal: () => {},
			};
			const driver = {
				get activeHandle() {
					return parkedHandle;
				},
				kickIdle: () => {},
				reentry: () => reentryHandle,
				snapshot: () => ({ state: "ready", conversationId: "conv-old", stats: {} }),
				run: async (opts: DriverTurnRequest) => {
					// First run binds conv-old; later fresh runs land conv-fresh.
					const conv = requests.length === 0 ? "conv-old" : "conv-fresh";
					requests.push(opts);
					return {
						id: `fake-turn-${requests.length}`,
						outcome: Promise.resolve(okOutcome(conv, "ok")),
						next: async () => null,
						pushExternal: () => {},
					} satisfies TurnHandle as unknown as TurnHandle;
				},
			} as unknown as StreamDriver;
			const store = new SessionStore(path.join(dir, "sessions.json"));
			const rt = new ToolRoundTrips(driver);
			const handoff = new CompactionHandoffs();
			const streamSimple = createStreamSimple({
				entries: [{ id: model.id, full: "gemini-3.8-flash" }],
				store,
				driver,
				acpDriver: driver,
				engine: "acp",
				roundTrips: rt,
				compactionHandoffs: handoff,
			});
			const opts = { cwd: dir, sessionId: "midpark-sess" } as unknown as SimpleStreamOptions;
			const key = sessionKey(opts, dir, "acp");

			await drainTerminal(streamSimple(model, normalizeContext({ messages: [user("initial task")] }), opts));
			assert.equal(store.get(key)?.conversationId, "conv-old");

			// Simulate a handoff armed without cleanup to exercise the provider guard.
			store.delete(key);
			handoff.rotate(key, true);
			const parkedPromise = rt
				.onToolCall("parked-call-1", "AskClaude", {}, new AbortController().signal)
				.then(
					() => "resolved",
					() => "rejected",
				);

			// Even if stale state survives rotation, the first tail pass must not
			// resolve it into its old conversation.
			const resumed = await drainTerminal(
				streamSimple(
					model,
					normalizeContext({
						messages: [
							compaction("Key decision: keep SQLite schema v2."),
							toolResultMsg("parked-call-1", "AskClaude", "late answer"),
						],
					}),
					opts,
				),
			);
			assert.equal(resumed.type, "done");
			assertCompactionResume(requests[1], "keep SQLite schema v2", "late answer");
			assert.equal(await parkedPromise, "rejected");
			assert.deepEqual(rt.deadIds, []);
			assert.equal(store.get(key)?.conversationId, "conv-fresh");
			assert.equal(handoff.hasPending(key), false);
		} finally {
			fs.rmSync(dir, { recursive: true, force: true });
		}
	});

	test("an armed handoff does not redeliver stale strands or dead parks", async () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "agy-rotate-strand-"));
		try {
			const requests: DriverTurnRequest[] = [];
			const parkedHandle = {
				id: "original-turn",
				outcome: new Promise<never>(() => {}),
				next: async () => null,
				pushExternal: () => {},
			};
			const convByRequest = ["conv-old", "conv-fresh"];
			const driver = {
				get activeHandle() {
					return parkedHandle;
				},
				kickIdle: () => {},
				reentry: () => null,
				snapshot: () => ({ state: "ready", conversationId: "conv-old", stats: {} }),
				run: async (opts: DriverTurnRequest) => {
					const conv = convByRequest[Math.min(requests.length, convByRequest.length - 1)];
					requests.push(opts);
					return {
						id: `fake-turn-${requests.length}`,
						outcome: Promise.resolve(okOutcome(conv, "ok")),
						next: async () => null,
						pushExternal: () => {},
					} satisfies TurnHandle as unknown as TurnHandle;
				},
			} as unknown as StreamDriver;
			const store = new SessionStore(path.join(dir, "sessions.json"));
			const rt = new ToolRoundTrips(driver, undefined, { escalateAfterMs: 1 });
			const handoff = new CompactionHandoffs();
			const streamSimple = createStreamSimple({
				entries: [{ id: model.id, full: "gemini-3.8-flash" }],
				store,
				driver,
				acpDriver: driver,
				engine: "acp",
				roundTrips: rt,
				compactionHandoffs: handoff,
			});
			const opts = { cwd: dir, sessionId: "strand-sess" } as unknown as SimpleStreamOptions;
			const key = sessionKey(opts, dir, "acp");

			await drainTerminal(streamSimple(model, normalizeContext({ messages: [user("initial task")] }), opts));
			assert.equal(store.get(key)?.conversationId, "conv-old");

			// A bridge call escalated, the model ended its turn without polling,
			// and the answer landed as a strand stamped with the old conversation.
			const esc = await rt.onToolCall("strand-call-1", "ask_user_question", {}, new AbortController().signal);
			assert.equal("escalated" in esc && esc.escalated, true);
			rt.stampConversation("conv-old");
			rt.resolve("strand-call-1", "user answered yes", false);

			const dead = assert.rejects(rt.onToolCall("dead-call", "read", {}, new AbortController().signal), /park expired/);
			rt.failAll("park expired");
			await dead;
			assert.deepEqual(rt.deadIds, ["dead-call"]);

			// Arm a handoff without clearing the round-trips, to exercise the guard
			// independently of the session_compact handler's own cleanup.
			store.delete(key);
			handoff.rotate(key, true);
			const next = await drainTerminal(streamSimple(model, normalizeContext({ messages: [
				compaction("Key decision: keep SQLite schema v2."),
				assistantMsg("reading"),
				toolResultMsg("dead-call", "read", "retained result"),
			] }), opts));
			assert.equal(next.type, "done");
			assertCompactionResume(requests[1], "keep SQLite schema v2", "retained result");
			assert.equal(rt.poll("strand-call-1"), undefined);
			assert.deepEqual(rt.deadIds, []);
			assert.equal(store.get(key)?.conversationId, "conv-fresh");
			assert.equal(handoff.hasPending(key), false);
		} finally {
			fs.rmSync(dir, { recursive: true, force: true });
		}
	});
});
