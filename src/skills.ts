// pi Agent Skills catalog + activate_skill bridge.
//
// The MCP bridge exposes ONE `activate_skill` tool to agy whose JSON-schema
// enum is the catalog; the description carries each skill's one-liner so agy
// can tell when a skill applies. Calling it returns the full SKILL.md plus the
// bundled resource dir. Nothing is appended to the prompt: agy sees the
// catalog in tools/list on every spawn, including after pi compaction.

import fs from "node:fs";
import path from "node:path";
import { parseFrontmatter } from "@earendil-works/pi-coding-agent";

export const ACTIVATE_SKILL_TOOL_NAME = "activate_skill";

export interface SkillLite {
	name: string;
	description: string;
	/** Absolute path to SKILL.md. */
	filePath: string;
	/** Absolute directory containing SKILL.md and bundled resources. */
	dir: string;
}

export interface PiCommandLike {
	name: string;
	description?: string;
	source: string;
	sourceInfo?: { path?: string };
}

/** Read and verify a SKILL.md in a single pass, matching Pi's loadSkillFromFile:
 *  fails closed on read errors, malformed YAML, missing/empty description
 *  (e.g. missing closing `---`), or `disable-model-invocation: true`. */
export function readModelInvocableSkill(
	filePath: string,
): { ok: true; body: string } | { ok: false; error: string } {
	let raw: string;
	try {
		raw = fs.readFileSync(filePath, "utf8");
	} catch (err) {
		return { ok: false, error: `failed to read skill: ${err instanceof Error ? err.message : String(err)}` };
	}
	try {
		const { frontmatter } = parseFrontmatter<Record<string, unknown>>(raw);
		if (!frontmatter || typeof frontmatter !== "object") {
			return { ok: false, error: "invalid skill frontmatter" };
		}
		const desc = typeof frontmatter.description === "string" ? frontmatter.description.trim() : "";
		if (!desc) {
			return { ok: false, error: "skill is missing a description" };
		}
		if (frontmatter["disable-model-invocation"] === true) {
			return { ok: false, error: "skill disables model invocation" };
		}
		return { ok: true, body: raw };
	} catch (err) {
		return { ok: false, error: `invalid skill frontmatter: ${err instanceof Error ? err.message : String(err)}` };
	}
}

/** Extract model-invocable skills from Pi's `pi.getCommands()` output so the
 *  bridge mirrors Pi's exact loaded skill set (`--no-skills`, `--skill`,
 *  settings filters, package skills, `disable-model-invocation: true`, and
 *  project trust). */
export function skillsFromCommands(commands: readonly PiCommandLike[]): SkillLite[] {
	const seen = new Set<string>();
	const out: SkillLite[] = [];
	for (const cmd of commands) {
		if (cmd.source !== "skill") continue;
		const filePath = cmd.sourceInfo?.path;
		if (typeof filePath !== "string" || !filePath) continue;
		const name = cmd.name.startsWith("skill:") ? cmd.name.slice("skill:".length) : cmd.name;
		if (!name || seen.has(name)) continue;
		seen.add(name);
		if (!readModelInvocableSkill(filePath).ok) continue;
		out.push({
			name,
			description: (cmd.description ?? "").replace(/\s+/g, " ").trim().slice(0, 160) || name,
			filePath,
			dir: path.dirname(filePath),
		});
	}
	return out;
}

export function findSkillByName(skills: SkillLite[], name: string): SkillLite | undefined {
	return skills.find((s) => s.name === name);
}

/** One-liner list for the tool description: `- name: description`. */
export function catalogSummary(skills: SkillLite[]): string {
	return skills.map((s) => `- ${s.name}: ${s.description}`).join("\n");
}

/** JSON schema for the activate_skill bridge tool. */
export function activateSkillSchema(skills: SkillLite[]): Record<string, unknown> {
	return {
		type: "object",
		properties: {
			name: {
				type: "string",
				enum: skills.map((s) => s.name),
				description: "Skill name from the catalog in this tool's description.",
			},
		},
		required: ["name"],
	};
}
