// Skills catalog tests: verify pi.getCommands() projection and single-read
// activation validation against Pi's frontmatter and model-invocation rules.

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, test } from "vitest";
import {
	activateSkillSchema,
	catalogSummary,
	findSkillByName,
	readModelInvocableSkill,
	skillsFromCommands,
} from "../src/skills.js";

const dirs: string[] = [];
function tmp(): string {
	const d = fs.mkdtempSync(path.join(os.tmpdir(), "agy-skills-"));
	dirs.push(d);
	return d;
}
afterEach(() => {
	while (dirs.length) fs.rmSync(dirs.pop()!, { recursive: true, force: true });
});

const SKILL_MD = (name: string, desc = `Use ${name}.`) =>
	`---\nname: ${name}\ndescription: ${desc}\n---\n\n# ${name}\n\nbody\n`;

test("skills: skillsFromCommands mirrors model-invocable skills and fails closed on invalid or manual-only files", () => {
	const dir = tmp();
	const write = (rel: string, content: string): string => {
		const full = path.join(dir, rel);
		fs.mkdirSync(path.dirname(full), { recursive: true });
		fs.writeFileSync(full, content);
		return full;
	};

	const normalPath = write("merge/SKILL.md", SKILL_MD("merge", "Merge branch"));
	const bomPath = write("bommed/SKILL.md", `\uFEFF${SKILL_MD("bommed", "Survives BOM")}`);
	const manualPath = write(
		"manual-only/SKILL.md",
		"---\nname: manual-only\ndisable-model-invocation: true\ndescription: Manual slash command only\n---\n\nbody\n",
	);
	const shadowPublicPath = write("manual-only-dup/SKILL.md", SKILL_MD("manual-only", "Public duplicate"));
	const invalidYamlPath = write(
		"broken-yaml/SKILL.md",
		"---\nname: broken-yaml\ndisable-model-invocation: true\ndescription: [unterminated\n---\n\nbody\n",
	);
	const unclosedPath = write(
		"unclosed-fm/SKILL.md",
		"---\nname: unclosed-fm\ndisable-model-invocation: true\ndescription: missing closing fence\n\nbody\n",
	);
	const noDescPath = write("no-desc/SKILL.md", "---\nname: no-desc\n---\n\nbody\n");

	const result = skillsFromCommands([
		{ name: "review", description: "prompt", source: "prompt", sourceInfo: { path: "/p/review.md" } },
		{ name: "agy", description: "ext", source: "extension", sourceInfo: { path: "/e/index.ts" } },
		{ name: "skill:manual-only", description: "Manual slash command only", source: "skill", sourceInfo: { path: manualPath } },
		{ name: "skill:manual-only", description: "Public duplicate", source: "skill", sourceInfo: { path: shadowPublicPath } },
		{ name: "skill:broken-yaml", description: "Invalid YAML must fail closed", source: "skill", sourceInfo: { path: invalidYamlPath } },
		{ name: "skill:unclosed-fm", description: "Missing closing --- must fail closed", source: "skill", sourceInfo: { path: unclosedPath } },
		{ name: "skill:no-desc", description: "Missing description must fail closed", source: "skill", sourceInfo: { path: noDescPath } },
		{ name: "skill:deleted", description: "Missing file must fail closed", source: "skill", sourceInfo: { path: path.join(dir, "gone", "SKILL.md") } },
		{ name: "skill:merge", description: "Merge branch", source: "skill", sourceInfo: { path: normalPath } },
		{ name: "skill:merge", description: "Duplicate merge", source: "skill", sourceInfo: { path: normalPath } },
		{ name: "skill:bommed", description: "Survives BOM", source: "skill", sourceInfo: { path: bomPath } },
	]);

	assert.deepEqual(result, [
		{
			name: "merge",
			description: "Merge branch",
			filePath: normalPath,
			dir: path.dirname(normalPath),
		},
		{
			name: "bommed",
			description: "Survives BOM",
			filePath: bomPath,
			dir: path.dirname(bomPath),
		},
	]);
	assert.deepEqual(skillsFromCommands([]), []);
	assert.equal(findSkillByName(result, "merge")?.filePath, normalPath);
	assert.equal(findSkillByName(result, "manual-only"), undefined);
	assert.equal(catalogSummary(result), "- merge: Merge branch\n- bommed: Survives BOM");
	assert.deepEqual((activateSkillSchema(result) as { properties: { name: { enum: string[] } } }).properties.name.enum, [
		"merge",
		"bommed",
	]);
});

test("skills: readModelInvocableSkill validates exact bytes on activation read", () => {
	const dir = tmp();
	const normalPath = path.join(dir, "merge", "SKILL.md");
	fs.mkdirSync(path.dirname(normalPath), { recursive: true });
	const body = SKILL_MD("merge", "Merge branch");
	fs.writeFileSync(normalPath, body);

	const unclosedPath = path.join(dir, "unclosed", "SKILL.md");
	fs.mkdirSync(path.dirname(unclosedPath), { recursive: true });
	fs.writeFileSync(unclosedPath, "---\nname: unclosed\ndescription: missing closing fence\n\nbody\n");

	const okRead = readModelInvocableSkill(normalPath);
	assert.deepEqual(okRead, { ok: true, body });
	assert.equal(readModelInvocableSkill(unclosedPath).ok, false);
	assert.equal(readModelInvocableSkill(path.join(dir, "missing", "SKILL.md")).ok, false);

	// Flipping a visible skill to manual-only rejects on activation read.
	fs.writeFileSync(
		normalPath,
		"---\nname: merge\ndisable-model-invocation: true\ndescription: now manual only\n---\n\nbody\n",
	);
	assert.equal(readModelInvocableSkill(normalPath).ok, false);
});
