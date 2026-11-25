// Unit tests for SessionStore load-time narrowing (src/sessions.ts).
//
// Proves a corrupt or hand-edited sessions.json can't plant wrong-typed fields
// into the cache: malformed entries are dropped and bad lastStepIdx values fall
// back to -1. Written as raw JSON text (not JSON.stringify) so the 1e999 token
// survives as Infinity through JSON.parse.
// Run: npm test

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import lockfile from "proper-lockfile";
import { test, vi } from "vitest";
import { SessionStore } from "../src/sessions.js";

function tmpStorePath(): string {
	return path.join(fs.mkdtempSync(path.join(os.tmpdir(), "agy-sessions-")), "sessions.json");
}

test("SessionStore: missing file starts empty (no throw)", () => {
	const store = new SessionStore(tmpStorePath());
	assert.equal(store.size, 0);
	assert.equal(store.get("anything"), null);
});

test("SessionStore: load drops malformed entries, keeps valid ones", () => {
	const p = tmpStorePath();
	fs.writeFileSync(
		p,
		JSON.stringify({
			good: { conversationId: "abc-123", lastStepIdx: 5 },
			badId: { conversationId: 123, lastStepIdx: 2 }, // id not a string -> drop
			badIdx: { conversationId: "xyz", lastStepIdx: "no" }, // idx not a number -> keep, -1
			nonObj: "string-value", // not an object -> drop
			missing: { lastStepIdx: 3 }, // no conversationId -> drop
		}) + "\n",
	);
	const store = new SessionStore(p);
	assert.deepEqual(store.get("good"), { conversationId: "abc-123", lastStepIdx: 5, lastMessageCount: 0 });
	assert.equal(store.get("badId"), null);
	assert.deepEqual(store.get("badIdx"), { conversationId: "xyz", lastStepIdx: -1, lastMessageCount: 0 });
	assert.equal(store.get("nonObj"), null);
	assert.equal(store.get("missing"), null);
});

test("SessionStore: non-finite lastStepIdx (Infinity from 1e999) falls back to -1", () => {
	const p = tmpStorePath();
	// Raw text: 1e999 parses to Infinity, which Number.isFinite rejects.
	fs.writeFileSync(p, `{"huge": {"conversationId": "big", "lastStepIdx": 1e999}}\n`);
	const store = new SessionStore(p);
	assert.deepEqual(store.get("huge"), { conversationId: "big", lastStepIdx: -1, lastMessageCount: 0 });
});

test("SessionStore: set/get round-trip in memory (incl. lastMessageCount watermark)", () => {
	const store = new SessionStore(tmpStorePath());
	store.set("k", { conversationId: "c-1", lastStepIdx: 9, lastMessageCount: 42 });
	assert.deepEqual(store.get("k"), { conversationId: "c-1", lastStepIdx: 9, lastMessageCount: 42 });
	assert.equal(store.size, 1);
});

test("SessionStore: digest bookkeeping survives a reload (fingerprint + compaction index)", async () => {
	const p = tmpStorePath();
	const store = new SessionStore(p);
	store.set("k", {
		conversationId: "c-1",
		lastStepIdx: -1,
		lastMessageCount: 3,
		digestBoundary: "user:1700000000000",
		digestSummaryId: "8301:ff2a91c7",
	});
	// The write is queued; wait for it to land before a fresh instance reads.
	await vi.waitFor(() => {
		JSON.parse(fs.readFileSync(p, "utf8"));
	});
	const reloaded = new SessionStore(p);
	assert.deepEqual(reloaded.get("k"), {
		conversationId: "c-1",
		lastStepIdx: -1,
		lastMessageCount: 3,
		digestBoundary: "user:1700000000000",
		digestSummaryId: "8301:ff2a91c7",
	});
});

test("SessionStore: top-level non-object file is treated as empty", () => {
	const p = tmpStorePath();
	fs.writeFileSync(p, `[1, 2, 3]`); // array, not an object
	const store = new SessionStore(p);
	assert.equal(store.size, 0);
});

test("SessionStore: delete drops only the target key, in memory and on disk", async () => {
	const p = tmpStorePath();
	const store = new SessionStore(p);
	store.set("a", { conversationId: "conv-a", lastStepIdx: 1, lastMessageCount: 5 });
	store.set("b", { conversationId: "conv-b", lastStepIdx: 2, lastMessageCount: 7 });
	store.delete("a");
	assert.equal(store.get("a"), null);
	assert.equal(store.get("b")?.conversationId, "conv-b");
	assert.equal(store.size, 1);
	// persist() overlays dirty keys on the on-disk state; the deleted key must
	// not resurrect from it.
	await vi.waitFor(() => {
		const disk = JSON.parse(fs.readFileSync(p, "utf8"));
		assert.ok(!("a" in disk), "deleted key must not persist");
		assert.equal(disk.b?.conversationId, "conv-b");
	});
});

test("SessionStore: set after delete resurrects the key", async () => {
	const p = tmpStorePath();
	const store = new SessionStore(p);
	store.set("k", { conversationId: "c-1", lastStepIdx: 0, lastMessageCount: 3 });
	store.delete("k");
	store.set("k", { conversationId: "c-2", lastStepIdx: -1, lastMessageCount: 0 });
	assert.deepEqual(store.get("k"), { conversationId: "c-2", lastStepIdx: -1, lastMessageCount: 0 });
	await vi.waitFor(() => {
		const disk = JSON.parse(fs.readFileSync(p, "utf8"));
		assert.equal(disk.k?.conversationId, "c-2");
	});
});

test("SessionStore: a removed key stops being deleted once its persist lands (multi-process)", async () => {
	const p = tmpStorePath();
	const store = new SessionStore(p);
	store.set("a", { conversationId: "conv-a", lastStepIdx: 0, lastMessageCount: 1 });
	store.delete("a");
	await vi.waitFor(() => {
		const disk = JSON.parse(fs.readFileSync(p, "utf8"));
		assert.ok(!("a" in disk), "delete must land first");
	});
	// Another process reclaims the key (the cwd-fallback key is shared across
	// pi processes in one directory). Our NEXT unrelated persist must honor
	// its write, not replay our stale deletion.
	fs.writeFileSync(
		p,
		JSON.stringify({ a: { conversationId: "conv-other-process", lastStepIdx: 3, lastMessageCount: 9 } }) + "\n",
	);
	store.set("b", { conversationId: "conv-b", lastStepIdx: 0, lastMessageCount: 2 });
	await vi.waitFor(() => {
		const disk = JSON.parse(fs.readFileSync(p, "utf8"));
		assert.equal(disk.a?.conversationId, "conv-other-process", "unrelated persist must not re-delete a reclaimed key");
		assert.equal(disk.b?.conversationId, "conv-b");
	});
});

test("flush resolves only after queued sets and deletes reach disk", async () => {
	const p = tmpStorePath();

	const store = new SessionStore(p);
	store.set("sid:a", { conversationId: "conv-a", lastStepIdx: -1, lastMessageCount: 1 });
	store.set("sid:b", { conversationId: "conv-b", lastStepIdx: 2, lastMessageCount: 4 });
	store.delete("sid:a");

	await store.flush();

	const disk = JSON.parse(fs.readFileSync(p, "utf8"));
	assert.equal(disk["sid:a"], undefined);
	assert.equal(disk["sid:b"]?.conversationId, "conv-b");
});

const syncState = {
	version: 1 as const,
	epoch: 1,
	checkpoint: { entryCount: 2, hash: "a".repeat(64), entryId: "m1-0123456789abcdef" },
	inFlight: { turnId: "t1", target: { entryCount: 2, hash: "a".repeat(64) }, submission: "prepared" as const },
};

test("SessionStore: sync journal survives a reload and get() carries it", async () => {
	const p = tmpStorePath();
	const store = new SessionStore(p);
	store.set("s1", { conversationId: "conv-1", lastStepIdx: 0, lastMessageCount: 0, sync: structuredClone(syncState) });
	await store.flush();
	const reloaded = new SessionStore(p);
	const record = reloaded.get("s1");
	assert.ok(record?.sync);
	assert.equal(record.sync.epoch, 1);
	assert.equal(record.sync.checkpoint?.entryCount, 2);
	assert.equal(record.sync.inFlight?.turnId, "t1");
});

test("SessionStore: a malformed sync field drops the field but keeps the entry", async () => {
	const p = tmpStorePath();
	fs.writeFileSync(p, JSON.stringify({
		"s1": { conversationId: "conv-1", lastStepIdx: 0, sync: { version: 1, checkpoint: { entryCount: 1, hash: "not-a-hash" } } },
	}));
	const store = new SessionStore(p);
	const record = store.get("s1");
	assert.ok(record);
	assert.equal(record.conversationId, "conv-1");
	// malformed journals narrow to the invalid marker, never drop
	assert.equal(record.sync?.invalidReason, "invalid-sync-state");
	// the disk copy normalizes on the next write (an idle store never rewrites)
	store.set("s1", { ...record, lastStepIdx: 1 });
	await store.flush();
	const raw = JSON.parse(fs.readFileSync(p, "utf8")) as Record<string, { sync?: { invalidReason?: string } }>;
	assert.equal(raw.s1?.sync?.invalidReason, "invalid-sync-state");
});

test("SessionStore: concurrent stores racing on one sessions.json preserve every key", async () => {
	const p = tmpStorePath();
	const stores = Array.from({ length: 12 }, () => new SessionStore(p));
	await Promise.all(
		stores.map(async (s, i) => {
			s.set(`session-${i}`, { conversationId: `conv-${i}`, lastStepIdx: i, lastMessageCount: i + 1 });
			await s.flush();
		}),
	);
	const disk = JSON.parse(fs.readFileSync(p, "utf8")) as Record<string, { conversationId: string; lastStepIdx: number }>;
	assert.equal(Object.keys(disk).length, 12);
	for (let i = 0; i < 12; i++) {
		assert.equal(disk[`session-${i}`]?.conversationId, `conv-${i}`);
		assert.equal(disk[`session-${i}`]?.lastStepIdx, i);
	}
	assert.equal(fs.existsSync(`${p}.lock`), false, "lockfile must be removed after flush");
});

test("SessionStore: recovers a stale lock directory left by a crashed process", async () => {
	const p = tmpStorePath();
	fs.mkdirSync(`${p}.lock`);
	const staleTime = new Date(Date.now() - 60_000);
	fs.utimesSync(`${p}.lock`, staleTime, staleTime);
	const store = new SessionStore(p, { lockStaleMs: 2_000 });
	store.set("recovered", { conversationId: "conv-ok", lastStepIdx: 1, lastMessageCount: 2 });
	await store.flush();
	const disk = JSON.parse(fs.readFileSync(p, "utf8")) as Record<string, { conversationId: string }>;
	assert.equal(disk.recovered?.conversationId, "conv-ok");
	assert.equal(fs.existsSync(`${p}.lock`), false);
});

test("SessionStore: waits for a live proper-lockfile holder until released and then persists", async () => {
	const p = tmpStorePath();
	const canonicalDir = fs.realpathSync(path.dirname(p));
	const canonicalTarget = path.join(canonicalDir, path.basename(p));
	const release = await lockfile.lock(canonicalTarget, { realpath: false, stale: 10_000 });
	const store = new SessionStore(p);
	store.set("waited", { conversationId: "conv-wait", lastStepIdx: 0, lastMessageCount: 1 });
	setTimeout(() => {
		void release();
	}, 45);
	await store.flush();
	const disk = JSON.parse(fs.readFileSync(p, "utf8")) as Record<string, { conversationId: string }>;
	assert.equal(disk.waited?.conversationId, "conv-wait");
	assert.equal(fs.existsSync(`${p}.lock`), false);
});

test("SessionStore: set -> clear -> set keeps only post-clear mutations on disk", async () => {
	const p = tmpStorePath();
	const store = new SessionStore(p);
	store.set("before", { conversationId: "conv-before", lastStepIdx: 1, lastMessageCount: 2 });
	store.clear();
	store.set("after", { conversationId: "conv-after", lastStepIdx: 0, lastMessageCount: 1 });
	await store.flush();
	const disk = JSON.parse(fs.readFileSync(p, "utf8")) as Record<string, { conversationId: string }>;
	assert.equal(disk.before, undefined);
	assert.equal(disk.after?.conversationId, "conv-after");
});

test("SessionStore: synchronous bursts under a held lock coalesce into one timeout instead of stacking", async () => {
	const p = tmpStorePath();
	const canonicalDir = fs.realpathSync(path.dirname(p));
	const canonicalTarget = path.join(canonicalDir, path.basename(p));
	const release = await lockfile.lock(canonicalTarget, { realpath: false, stale: 10_000 });
	try {
		const store = new SessionStore(p, { lockTimeoutMs: 80 });
		const start = Date.now();
		for (let i = 0; i < 8; i++) {
			store.set(`k-${i}`, { conversationId: `c-${i}`, lastStepIdx: i, lastMessageCount: 1 });
		}
		await assert.rejects(() => store.flush(), /Timed out waiting for session store lock/);
		const elapsed = Date.now() - start;
		assert.ok(elapsed < 400, `expected single coalesced timeout (~80ms), got ${elapsed}ms`);
		// Once released, a follow-up flush persists all 8 coalesced keys.
		await release();
		await store.flush();
		const disk = JSON.parse(fs.readFileSync(p, "utf8")) as Record<string, { conversationId: string }>;
		assert.equal(Object.keys(disk).length, 8);
		assert.equal(disk["k-7"]?.conversationId, "c-7");
	} finally {
		await release().catch(() => {});
	}
});

test("SessionStore: non-directory lock paths (file, dangling symlink, FIFO) fail promptly", async () => {
	const pFile = tmpStorePath();
	fs.writeFileSync(`${pFile}.lock`, "stray-file\n");
	const storeFile = new SessionStore(pFile, { lockTimeoutMs: 2_000 });
	storeFile.set("k", { conversationId: "c", lastStepIdx: 0, lastMessageCount: 1 });
	await assert.rejects(() => storeFile.flush(), /lock path is not a directory/);

	if (process.platform !== "win32") {
		const pSym = tmpStorePath();
		fs.symlinkSync(path.join(path.dirname(pSym), "does-not-exist"), `${pSym}.lock`);
		const storeSym = new SessionStore(pSym, { lockTimeoutMs: 2_000 });
		storeSym.set("k", { conversationId: "c", lastStepIdx: 0, lastMessageCount: 1 });
		await assert.rejects(() => storeSym.flush(), /lock path is not a directory/);

		const pFifo = tmpStorePath();
		execFileSync("mkfifo", [`${pFifo}.lock`]);
		const storeFifo = new SessionStore(pFifo, { lockTimeoutMs: 2_000 });
		storeFifo.set("k", { conversationId: "c", lastStepIdx: 0, lastMessageCount: 1 });
		const start = Date.now();
		await assert.rejects(() => storeFifo.flush(), /lock path is not a directory/);
		assert.ok(Date.now() - start < 500, "FIFO lock path must fail immediately without blocking");
	}
});

test("SessionStore: unreadable sessions.json throws on flush instead of overwriting unrelated bindings", async () => {
	const p = tmpStorePath();
	fs.writeFileSync(p, JSON.stringify({ existing: { conversationId: "keep-me", lastStepIdx: 2, lastMessageCount: 3 } }) + "\n");
	const useChmod = process.platform !== "win32" && process.getuid?.() !== 0;
	const origReadFile = fs.promises.readFile.bind(fs.promises);
	const readSpy = useChmod
		? null
		: vi.spyOn(fs.promises, "readFile").mockImplementation(((filePath: Parameters<typeof fs.promises.readFile>[0], ...rest: any[]) => {
				if (String(filePath) === p) {
					return Promise.reject(Object.assign(new Error(`EACCES: permission denied, open '${p}'`), { code: "EACCES" }));
				}
				return origReadFile(filePath, ...(rest as []));
			}) as typeof fs.promises.readFile);
	if (useChmod) fs.chmodSync(p, 0o000);
	try {
		const store = new SessionStore(p);
		store.set("new-key", { conversationId: "c-new", lastStepIdx: 0, lastMessageCount: 1 });
		await assert.rejects(() => store.flush(), /EACCES/);
	} finally {
		readSpy?.mockRestore();
		if (useChmod) fs.chmodSync(p, 0o600);
	}
	const disk = JSON.parse(fs.readFileSync(p, "utf8")) as Record<string, { conversationId: string }>;
	assert.equal(disk.existing?.conversationId, "keep-me");
	assert.equal(disk["new-key"], undefined);
});

test("SessionStore: failed clearNow() leaves no destructive retry queued for a later flush()", async () => {
	const p = tmpStorePath();
	const storeA = new SessionStore(p, { lockTimeoutMs: 2_000 });
	storeA.set("initial", { conversationId: "c-initial", lastStepIdx: 0, lastMessageCount: 1 });
	await storeA.flush();

	// Break locking so clearNow() fails.
	fs.writeFileSync(`${p}.lock`, "stray-file\n");
	await assert.rejects(() => storeA.clearNow(), /lock path is not a directory/);
	// In-memory state stays intact on failed clear.
	assert.equal(storeA.get("initial")?.conversationId, "c-initial");

	// Clear the lock obstruction and let another process persist a new binding.
	fs.unlinkSync(`${p}.lock`);
	const storeB = new SessionStore(p);
	storeB.set("later", { conversationId: "c-later", lastStepIdx: 1, lastMessageCount: 2 });
	await storeB.flush();

	// A later flush() on storeA (e.g. session_shutdown) must not replay the failed clear.
	await storeA.flush();
	const disk = JSON.parse(fs.readFileSync(p, "utf8")) as Record<string, { conversationId: string }>;
	assert.equal(disk.initial?.conversationId, "c-initial");
	assert.equal(disk.later?.conversationId, "c-later");

	// Ordinary pending writes still retry after a failed clearNow().
	fs.writeFileSync(`${p}.lock`, "stray-file\n");
	storeA.set("queued", { conversationId: "c-queued", lastStepIdx: 2, lastMessageCount: 3 });
	await assert.rejects(() => storeA.flush(), /lock path is not a directory/);
	await assert.rejects(() => storeA.clearNow(), /lock path is not a directory/);
	fs.unlinkSync(`${p}.lock`);
	await storeA.flush();
	const diskAfter = JSON.parse(fs.readFileSync(p, "utf8")) as Record<string, { conversationId: string }>;
	assert.equal(diskAfter.initial?.conversationId, "c-initial");
	assert.equal(diskAfter.later?.conversationId, "c-later");
	assert.equal(diskAfter.queued?.conversationId, "c-queued");
});

test("SessionStore: clearNow() establishes clear boundary at invocation and preserves later writes", async () => {
	const p = tmpStorePath();
	const store = new SessionStore(p);
	store.set("before", { conversationId: "c-before", lastStepIdx: 0, lastMessageCount: 1 });
	const clearPromise = store.clearNow();
	store.set("after", { conversationId: "c-after", lastStepIdx: 1, lastMessageCount: 2 });
	await clearPromise;
	await store.flush();

	assert.equal(store.get("before"), null);
	assert.equal(store.get("after")?.conversationId, "c-after");
	const disk = JSON.parse(fs.readFileSync(p, "utf8")) as Record<string, { conversationId: string }>;
	assert.equal(disk.before, undefined);
	assert.equal(disk.after?.conversationId, "c-after");
});

test("/agy clear reports flush failure to stderr without UI, via ui.notify with UI, and keeps shutdown warnings on stderr", async () => {
	const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "agy-clear-"));
	const home = path.join(scratch, "home");
	const agentDir = path.join(scratch, "agent");
	const bridgeRoot = path.join(agentDir, "antigravity-bridge");
	fs.mkdirSync(home, { recursive: true });
	fs.mkdirSync(bridgeRoot, { recursive: true });
	const sessionsFile = path.join(bridgeRoot, "sessions.json");
	fs.writeFileSync(sessionsFile, JSON.stringify({ keep: { conversationId: "c-1", lastStepIdx: 0, lastMessageCount: 1 } }) + "\n");
	fs.writeFileSync(`${sessionsFile}.lock`, "stray-file\n");

	const envKeys = ["PI_CODING_AGENT_DIR", "HOME", "USERPROFILE", "AGY_BIN", "AGY_AGENTS_ROOT", "AGY_ASK_TOOL"] as const;
	const prevEnvs = envKeys.map((k) => [k, process.env[k]] as const);
	const realmKey = Symbol.for("pi-antigravity-bridge.runtime");
	const prevRealm = (globalThis as Record<symbol, unknown>)[realmKey];

	process.env.PI_CODING_AGENT_DIR = agentDir;
	process.env.HOME = home;
	process.env.USERPROFILE = home;
	process.env.AGY_BIN = path.join(scratch, "missing-agy");
	process.env.AGY_AGENTS_ROOT = path.join(scratch, "agents");
	process.env.AGY_ASK_TOOL = "off";
	delete (globalThis as Record<symbol, unknown>)[realmKey];

	const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
	try {
		vi.resetModules();
		const commands = new Map<string, (args: string, ctx: any) => Promise<void>>();
		const handlers = new Map<string, Array<(event: any, ctx: any) => unknown>>();
		const pi: any = new Proxy(
			{
				registerCommand: (name: string, opts: any) => commands.set(name, opts.handler),
				on: (event: string, fn: any) => {
					const list = handlers.get(event) ?? [];
					list.push(fn);
					handlers.set(event, list);
				},
			},
			{
				get(target, prop) {
					return prop in target ? (target as any)[prop] : () => {};
				},
			},
		);
		const { default: register } = await import("../extensions/index.js");
		await register(pi);
		const agyHandler = commands.get("agy");
		assert.ok(agyHandler, "expected /agy command to be registered");

		// Print-mode /agy clear fails on stray lock: writes to stderr and must not cache no-op UI.
		const headlessNotify = vi.fn();
		await agyHandler("clear", { hasUI: false, mode: "print", ui: { notify: headlessNotify } });
		assert.equal(headlessNotify.mock.calls.length, 0);
		assert.ok(
			errSpy.mock.calls.some(([msg]) =>
				String(msg).includes("[antigravity-bridge] Failed to clear antigravity session bindings:"),
			),
			`expected stderr error, got: ${JSON.stringify(errSpy.mock.calls)}`,
		);

		// A later shutdown flush failure after a print-mode command must still reach stderr.
		const realm = (globalThis as Record<symbol, unknown>)[realmKey] as { store: SessionStore } | undefined;
		assert.ok(realm, "expected bridge realm");
		errSpy.mockClear();
		realm.store.set("unsaved", { conversationId: "c-unsaved", lastStepIdx: 0, lastMessageCount: 1 });
		for (const fn of handlers.get("session_shutdown") ?? []) {
			await fn({}, { hasUI: false, ui: { notify: headlessNotify } });
		}
		assert.equal(headlessNotify.mock.calls.length, 0);
		assert.ok(
			errSpy.mock.calls.some(([msg]) =>
				String(msg).includes("[antigravity-bridge] Failed to persist Antigravity session bindings:"),
			),
			`expected shutdown stderr warning, got: ${JSON.stringify(errSpy.mock.calls)}`,
		);

		// UI-capable /agy clear reports via ui.notify("error") instead of stderr.
		errSpy.mockClear();
		const uiNotify = vi.fn();
		await agyHandler("clear", { hasUI: true, mode: "interactive", ui: { notify: uiNotify } });
		assert.equal(errSpy.mock.calls.length, 0);
		assert.equal(uiNotify.mock.calls.length, 1);
		assert.match(String(uiNotify.mock.calls[0]?.[0]), /^Failed to clear antigravity session bindings:/);
		assert.equal(uiNotify.mock.calls[0]?.[1], "error");
	} finally {
		errSpy.mockRestore();
		if (prevRealm === undefined) delete (globalThis as Record<symbol, unknown>)[realmKey];
		else (globalThis as Record<symbol, unknown>)[realmKey] = prevRealm;
		for (const [k, prev] of prevEnvs) {
			if (prev === undefined) delete process.env[k];
			else process.env[k] = prev;
		}
		fs.rmSync(scratch, { recursive: true, force: true });
	}
});
