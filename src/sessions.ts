// Persisted map: pi session key -> agy conversation id + last streamed step.
//
// agy holds its own conversation history in the DB keyed by conversation id.
// To resume a multi-turn pi conversation, we thread the same agy id across
// streamSimple calls and skip steps we already streamed last turn.
//
// The key is options.sessionId when pi provides it, else the cwd (single-
// conversation-per-process fallback). Stored at
// <agentDir>/antigravity-bridge/sessions.json with cross-process locking and
// atomic rename.

import { randomBytes } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import lockfile from "proper-lockfile";
import { bridgeDir } from "./config.js";
import { narrowSessionSync, type SessionSyncState } from "./history-sync.js";

const LOCK_TIMEOUT_MS = 15_000;
const LOCK_STALE_MS = 10_000;

export interface SessionStoreOptions {
	lockTimeoutMs?: number;
	lockStaleMs?: number;
	onError?: (err: Error) => void;
}

type PendingOp =
	| { kind: "set"; key: string; session: AgySession }
	| { kind: "delete"; key: string }
	| { kind: "clear" }
	| { kind: "clearNow"; resolve: () => void; reject: (err: Error) => void };

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Cross-process lock via proper-lockfile (atomic mkdir + mtime heartbeat +
 *  stale recovery + exit cleanup) guarding read-merge-rename on sessions.json. */
async function withFileLock<T>(
	targetPath: string,
	fn: (assertHeld: () => void) => Promise<T>,
	opts: { timeoutMs?: number; staleMs?: number } = {},
): Promise<T> {
	const timeoutMs = opts.timeoutMs ?? LOCK_TIMEOUT_MS;
	const staleMs = opts.staleMs ?? LOCK_STALE_MS;
	const dir = path.dirname(targetPath);
	await fs.promises.mkdir(dir, { recursive: true });
	const canonicalDir = await fs.promises.realpath(dir);
	const canonicalTarget = path.join(canonicalDir, path.basename(targetPath));
	const lockPath = `${canonicalTarget}.lock`;
	const deadline = Date.now() + timeoutMs;

	let compromisedError: Error | null = null;
	let release: (() => Promise<void>) | undefined;

	while (true) {
		try {
			release = await lockfile.lock(canonicalTarget, {
				realpath: false,
				stale: staleMs,
				retries: 0,
				onCompromised: (err) => {
					compromisedError = err instanceof Error ? err : new Error(String(err));
				},
			});
			break;
		} catch (err) {
			const code = (err as NodeJS.ErrnoException).code;
			if (code !== "ELOCKED") throw err;
			// proper-lockfile uses a directory lock; reject non-directory entries
			// (stray files, symlinks, FIFOs) promptly instead of waiting out the timeout.
			try {
				const st = await fs.promises.lstat(lockPath);
				if (!st.isDirectory()) {
					throw new Error(`Session store lock path is not a directory: ${lockPath}`);
				}
			} catch (statErr) {
				if ((statErr as NodeJS.ErrnoException).code !== "ENOENT") throw statErr;
			}
			if (Date.now() >= deadline) {
				throw new Error(`Timed out waiting for session store lock: ${lockPath}`);
			}
			await sleep(8 + Math.floor(Math.random() * 16));
		}
	}

	try {
		return await fn(() => {
			if (compromisedError) throw compromisedError;
		});
	} finally {
		try {
			await release();
		} catch (releaseErr) {
			const code = (releaseErr as NodeJS.ErrnoException).code;
			if (!compromisedError && code !== "ENOENT" && code !== "ERELEASED") {
				// Retry transient rmdir failures (e.g. EBUSY) briefly while we know the
				// lock wasn't compromised; proper-lockfile's stale check backs up anything longer.
				for (let attempt = 0; attempt < 3; attempt++) {
					await sleep(10 * (attempt + 1));
					try {
						await fs.promises.rmdir(lockPath);
						break;
					} catch (retryErr) {
						if ((retryErr as NodeJS.ErrnoException).code === "ENOENT") break;
					}
				}
			}
		}
	}
}

function defaultStorePath(): string {
	return path.join(bridgeDir(), "sessions.json");
}

export interface AgySession {
	conversationId: string;
	lastStepIdx: number;
	/** pi context.messages length captured at the start of the last agy turn.
	 *  Used by the provider to compute a delta digest of pi-side context agy
	 *  was not spawned for (compaction summaries, other-provider turns). 0 on a
	 *  fresh or pre-watermark session. Optional for backward compat with older
	 *  persisted sessions that predate the watermark. */
	lastMessageCount?: number;
	/** role:timestamp fingerprint of messages[lastMessageCount - 1] at write
	 *  time. The provider compares it against the live boundary message: a
	 *  mismatch (branch, fork, context-hook rewrite) resets the digest window
	 *  to the full delta. Optional: absent on records predating the
	 *  fingerprint, which the provider treats as a mismatch. */
	digestBoundary?: string;
	/** Compaction index already delivered in a previous digest, so the
	 *  builder neither re-injects that summary nor restarts its window on
	 *  it. Optional/absent when no compaction existed at write time. */
	digestSummaryId?: string;
	/** History-sync protocol journal (handoff = "sync"). Absent on digest
	 *  records and on records predating the sync mode. Zeroed digest
	 *  watermark fields accompany it (the sync persist contract), so a
	 *  later flip back to digest re-offers the backlog once, safely. */
	sync?: SessionSyncState;
}

type StoreMap = Record<string, AgySession>;

/** Narrow untrusted parsed JSON into a clean StoreMap, validating each entry's
 *  shape instead of trusting the whole structure (the cast previously let a
 *  hand-edited or corrupt file plant wrong-typed fields into the cache).
 *  Drops any entry whose `conversationId` isn't a string; falls back to -1 when
 *  `lastStepIdx` is missing or not a finite number. Returns {} for non-object
 *  input. */
function narrowStoreMap(parsed: unknown): StoreMap {
	if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
	const clean: StoreMap = {};
	for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
		if (!value || typeof value !== "object") continue;
		const conversationId = (value as { conversationId?: unknown }).conversationId;
		if (typeof conversationId !== "string") continue;
		const idx = (value as { lastStepIdx?: unknown }).lastStepIdx;
		const mc = (value as { lastMessageCount?: unknown }).lastMessageCount;
		const db = (value as { digestBoundary?: unknown }).digestBoundary;
		const dci = (value as { digestSummaryId?: unknown }).digestSummaryId;
		// Malformed sync journals narrow to an invalid-sync-state marker
		// (never throws; self-heals via a named rebuild), so the field is
		// kept but flagged.
		const sync = narrowSessionSync((value as { sync?: unknown }).sync);
		clean[key] = {
			conversationId,
			lastStepIdx: typeof idx === "number" && Number.isFinite(idx) ? idx : -1,
			lastMessageCount: typeof mc === "number" && Number.isFinite(mc) ? mc : 0,
			...(typeof db === "string" ? { digestBoundary: db } : {}),
			...(typeof dci === "string" ? { digestSummaryId: dci } : {}),
			...(sync ? { sync } : {}),
		};
	}
	return clean;
}

/** Atomic-write JSON session store. Queued mutations coalesce into one
 *  serialized persist loop so concurrent turns or rapid synchronous writes
 *  don't interleave renames or stack lock timeouts.
 *
 *  Multi-process safety: persist() holds a cross-process lock via
 *  proper-lockfile, re-reads sessions.json fresh, and replays only this
 *  instance's queued mutation batch before atomic rename. */
export class SessionStore {
	private readonly path: string;
	private readonly lockOpts: { timeoutMs?: number; staleMs?: number };
	private readonly onError?: (err: Error) => void;
	private cache: StoreMap = {};
	private readonly pendingOps: PendingOp[] = [];
	private activePersist: Promise<void> | null = null;
	private needsAnotherPersist = false;
	private lastError: Error | null = null;

	constructor(storePath: string = defaultStorePath(), opts: SessionStoreOptions = {}) {
		this.path = storePath;
		this.lockOpts = {
			timeoutMs: opts.lockTimeoutMs,
			staleMs: opts.lockStaleMs,
		};
		this.onError = opts.onError;
		this.load();
	}

	private load(): void {
		try {
			const raw = fs.readFileSync(this.path, "utf8");
			this.cache = narrowStoreMap(JSON.parse(raw));
		} catch {
			// missing or corrupt  -  start empty. The first set() will create it.
			this.cache = {};
		}
	}

	private schedulePersist(): void {
		if (this.activePersist) {
			this.needsAnotherPersist = true;
			return;
		}
		this.activePersist = this.runPersistLoop();
	}

	private rejectPendingClearNows(error: Error): void {
		for (let i = this.pendingOps.length - 1; i >= 0; i--) {
			const op = this.pendingOps[i]!;
			if (op.kind === "clearNow") {
				this.pendingOps.splice(i, 1);
				op.reject(error);
			}
		}
	}

	private async runPersistLoop(): Promise<void> {
		// Yield one microtask so synchronous bursts (e.g. set/clear/set) coalesce
		// into a single lock acquisition and rename.
		await Promise.resolve();
		let failed = false;
		try {
			while (this.pendingOps.length > 0) {
				this.needsAnotherPersist = false;
				try {
					await this.persist();
					this.lastError = null;
				} catch (err) {
					failed = true;
					const error = err instanceof Error ? err : new Error(String(err));
					this.lastError = error;
					this.rejectPendingClearNows(error);
					this.onError?.(error);
					break;
				}
			}
		} finally {
			this.activePersist = null;
			if (!failed && this.needsAnotherPersist && this.pendingOps.length > 0) {
				this.schedulePersist();
			}
		}
	}

	get(key: string): AgySession | null {
		const s = this.cache[key];
		if (!s || typeof s.conversationId !== "string") return null;
		return {
			conversationId: s.conversationId,
			lastStepIdx: s.lastStepIdx ?? -1,
			lastMessageCount: s.lastMessageCount ?? 0,
			...(s.digestBoundary !== undefined ? { digestBoundary: s.digestBoundary } : {}),
			...(s.digestSummaryId !== undefined ? { digestSummaryId: s.digestSummaryId } : {}),
			...(s.sync !== undefined ? { sync: structuredClone(s.sync) } : {}),
		};
	}

	/** Update one session and persist atomically (temp file + rename). The
	 *  write is queued; callers don't need to await it. */
	set(key: string, session: AgySession): void {
		this.cache[key] = session;
		this.pendingOps.push({ kind: "set", key, session });
		this.schedulePersist();
	}

	/** Drop one session binding and persist the deletion. Used when the driver
	 *  recreated the conversation under us: the record's watermark claims
	 *  history the new conversation never saw, and its bound id keeps the
	 *  sysPrompt gate closed. Dropping resets the watermark to 0 AND re-arms
	 *  the gate on the next turn (issue #1). The write is queued. */
	delete(key: string): void {
		delete this.cache[key];
		this.pendingOps.push({ kind: "delete", key });
		this.schedulePersist();
	}

	private async persist(): Promise<void> {
		// Snapshot the exact prefix of queued mutations before awaiting I/O so
		// later set()/delete()/clear()/clearNow() calls stay ordered in pendingOps.
		const count = this.pendingOps.length;
		if (count === 0) return;
		const batch = this.pendingOps.slice(0, count);

		await withFileLock(
			this.path,
			async (assertHeld) => {
				let lastClear = -1;
				let hasClearNow = false;
				for (let i = batch.length - 1; i >= 0; i--) {
					const kind = batch[i]!.kind;
					if (kind === "clearNow") hasClearNow = true;
					if (lastClear < 0 && (kind === "clear" || kind === "clearNow")) {
						lastClear = i;
					}
				}
				let merged: StoreMap;
				let replayStart: number;
				if (lastClear >= 0) {
					merged = {};
					replayStart = lastClear + 1;
				} else {
					let disk: StoreMap = {};
					try {
						const raw = await fs.promises.readFile(this.path, "utf8");
						try {
							disk = narrowStoreMap(JSON.parse(raw));
						} catch {
							/* corrupt JSON - start from empty */
						}
					} catch (err) {
						if ((err as NodeJS.ErrnoException).code !== "ENOENT") throw err;
					}
					merged = { ...disk };
					replayStart = 0;
				}
				for (let i = replayStart; i < batch.length; i++) {
					const op = batch[i]!;
					if (op.kind === "set") merged[op.key] = op.session;
					else if (op.kind === "delete") delete merged[op.key];
				}

				const tmp = `${this.path}.${process.pid}.${randomBytes(4).toString("hex")}.tmp`;
				try {
					await fs.promises.writeFile(tmp, JSON.stringify(merged, null, 2) + "\n", {
						mode: 0o600,
					});
					assertHeld();
					await fs.promises.rename(tmp, this.path);
				} catch (err) {
					await fs.promises.unlink(tmp).catch(() => {});
					throw err;
				}
				// Acknowledge the committed batch immediately after rename so any
				// subsequent unlock hiccup never replays an already-durable write.
				this.pendingOps.splice(0, count);
				if (hasClearNow) {
					// clearNow() does not wipe cache eagerly so a failed clear leaves
					// in-memory state intact. Once durable, rebuild cache from the
					// committed post-clear state plus any ops queued during lock/rename.
					const nextCache: StoreMap = { ...merged };
					for (const op of this.pendingOps) {
						if (op.kind === "set") nextCache[op.key] = op.session;
						else if (op.kind === "delete") delete nextCache[op.key];
						else if (op.kind === "clear") {
							for (const k of Object.keys(nextCache)) delete nextCache[k];
						}
					}
					this.cache = nextCache;
					for (const op of batch) {
						if (op.kind === "clearNow") op.resolve();
					}
				}
			},
			this.lockOpts,
		);
	}

	/** Wipe all session bindings (forces fresh agy conversations on every
	 *  active session). Queues an asynchronous clear. */
	clear(): void {
		this.cache = {};
		this.pendingOps.push({ kind: "clear" });
		this.schedulePersist();
	}

	/** Wipe all session bindings at this point in the mutation queue. Writes
	 *  queued after clearNow() are preserved on disk and in memory; a failed
	 *  clear rejects without leaving a destructive retry queued for a later
	 *  flush(), while preserving ordinary pending writes. */
	clearNow(): Promise<void> {
		return new Promise<void>((resolve, reject) => {
			this.pendingOps.push({ kind: "clearNow", resolve, reject });
			this.schedulePersist();
		});
	}

	/** Resolve once every queued write (set, delete, clear) has reached disk,
	 *  including writes queued while waiting. Rejects if writes could not reach
	 *  disk. */
	async flush(): Promise<void> {
		if (this.activePersist) {
			await this.activePersist;
		} else if (this.pendingOps.length > 0) {
			this.schedulePersist();
			await this.activePersist;
		}
		while (this.activePersist) {
			await this.activePersist;
		}
		if (this.pendingOps.length > 0) {
			throw this.lastError ?? new Error(`Failed to persist session store: ${this.path}`);
		}
	}

	/** Count persisted session bindings (for status display). */
	get size(): number {
		return Object.keys(this.cache).length;
	}
}
