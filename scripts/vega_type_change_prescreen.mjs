#!/usr/bin/env node

// Extracts only conservative, single-operation TYPE_CHANGED candidates from
// Vega/Vega-Lite major-version schema boundaries. It persists provenance and
// raw component pairs but does not invoke an oracle, load a model, or assign a
// compatibility label.

import { execFileSync, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";

const SOURCE = "vega/schema";
const SOURCE_URL = "https://github.com/vega/schema";
const SOURCE_KIND = "public-version-history";
const SOURCE_LICENSE = "BSD-3-Clause";
const METADATA_KEYS = new Set(["description", "title", "example", "examples", "default", "$comment"]);
const PAIRS = [
  ["vega", "v2.6.5.json", "v3.0.0.json", "documented Vega 2 to Vega 3 format migration"],
  ["vega", "v3.3.1.json", "v4.0.0.json", "Vega major-version boundary"],
  ["vega", "v4.4.0.json", "v5.0.0.json", "Vega major-version boundary"],
  ["vega", "v5.33.0.json", "v6.0.0.json", "Vega major-version boundary"],
  ["vega-lite", "v1.3.1.json", "v2.0.0.json", "Vega-Lite major-version boundary"],
  ["vega-lite", "v2.7.0.json", "v3.0.0.json", "Vega-Lite major-version boundary"],
  ["vega-lite", "v3.4.0.json", "v4.0.0.json", "Vega-Lite major-version boundary"],
  ["vega-lite", "v4.17.0.json", "v5.0.0.json", "Vega-Lite major-version boundary"],
  ["vega-lite", "v5.23.0.json", "v6.0.0.json", "Vega-Lite major-version boundary"],
];

function parseArguments(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    if (!argv[index]?.startsWith("--") || argv[index + 1] === undefined) {
      throw new Error("usage: vega_type_change_prescreen.mjs --repo <vega/schema clone> --report <report.json> --manifest <manifest.json>");
    }
    values.set(argv[index], argv[index + 1]);
  }
  for (const flag of ["--repo", "--report", "--manifest"]) {
    if (!values.has(flag)) throw new Error(`missing ${flag}`);
  }
  return {
    repository: resolve(values.get("--repo")),
    report: resolve(values.get("--report")),
    manifest: resolve(values.get("--manifest")),
  };
}

function git(repository, ...args) {
  return execFileSync("git", ["-C", repository, ...args], { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 }).trim();
}

function sha256(contents) {
  return createHash("sha256").update(contents).digest("hex");
}

function equal(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function operations(base, candidate, path = [], output = []) {
  if (equal(base, candidate)) return output;
  if (base === undefined) {
    output.push({ operation: "add", path, candidate });
    return output;
  }
  if (candidate === undefined) {
    output.push({ operation: "remove", path, base });
    return output;
  }
  if (
    Array.isArray(base) || Array.isArray(candidate) || base === null || candidate === null ||
    typeof base !== "object" || typeof candidate !== "object"
  ) {
    output.push({ operation: "change", path, base, candidate });
    return output;
  }
  for (const key of [...new Set([...Object.keys(base), ...Object.keys(candidate)])].sort()) {
    operations(base[key], candidate[key], [...path, key], output);
  }
  return output;
}

function isMetadata(operation) {
  return operation.path.some((segment) => METADATA_KEYS.has(segment) || segment.startsWith("x-"));
}

function classify(rawOperations) {
  const semantic = rawOperations.filter((operation) => !isMetadata(operation));
  if (semantic.length === 0) return { decision: "METADATA_ONLY", semantic };
  if (semantic.length === 1 && semantic[0].path.at(-1) === "type" && semantic[0].operation === "change") {
    return { decision: "TYPE_CHANGED", semantic };
  }
  if (semantic.length === 1) return { decision: "UNMAPPED_SINGLE_CHANGE", semantic };
  return { decision: "COMPOUND", semantic };
}

function numstat(basePath, candidatePath) {
  const result = spawnSync("git", ["diff", "--no-index", "--numstat", "--", basePath, candidatePath], { encoding: "utf8" });
  if (![0, 1].includes(result.status)) throw new Error(`git diff failed: ${result.stderr}`);
  const tokens = result.stdout.trim().split(/\s+/);
  return { added: Number(tokens[0] || 0), removed: Number(tokens[1] || 0) };
}

function componentMap(document) {
  const definitions = document.definitions ?? document.$defs;
  if (definitions && Object.keys(definitions).length > 0) return definitions;
  return { "__root__": document };
}

function latestFileCommit(repository, snapshotCommit, sourcePath) {
  const commit = git(
    repository,
    "log",
    "-1",
    "--format=%H",
    snapshotCommit,
    "--",
    sourcePath,
  );
  if (!commit) throw new Error(`no file-history commit for ${sourcePath}`);
  return commit;
}

function inspectPair(configuration, library, baseFile, candidateFile, rationale, commit) {
  const baseRelativePath = `${library}/${baseFile}`;
  const candidateRelativePath = `${library}/${candidateFile}`;
  const basePath = resolve(configuration.repository, baseRelativePath);
  const candidatePath = resolve(configuration.repository, candidateRelativePath);
  if (!existsSync(basePath) || !existsSync(candidatePath)) throw new Error(`missing source pair ${baseRelativePath} -> ${candidateRelativePath}`);
  const baseBytes = readFileSync(basePath);
  const candidateBytes = readFileSync(candidatePath);
  const baseDocument = JSON.parse(baseBytes);
  const candidateDocument = JSON.parse(candidateBytes);
  const baseFileCommit = latestFileCommit(configuration.repository, commit, baseRelativePath);
  const candidateFileCommit = latestFileCommit(configuration.repository, commit, candidateRelativePath);
  const baseBlobSha1 = git(configuration.repository, "rev-parse", `${baseFileCommit}:${baseRelativePath}`);
  const candidateBlobSha1 = git(configuration.repository, "rev-parse", `${candidateFileCommit}:${candidateRelativePath}`);
  const snapshotBaseBlobSha1 = git(configuration.repository, "rev-parse", `${commit}:${baseRelativePath}`);
  const snapshotCandidateBlobSha1 = git(configuration.repository, "rev-parse", `${commit}:${candidateRelativePath}`);
  if (baseBlobSha1 !== snapshotBaseBlobSha1 || candidateBlobSha1 !== snapshotCandidateBlobSha1) {
    throw new Error(`versioned schema changed after its recorded file commit: ${baseRelativePath} -> ${candidateRelativePath}`);
  }
  const baseComponents = componentMap(baseDocument);
  const candidateComponents = componentMap(candidateDocument);
  const commonComponents = [...new Set([...Object.keys(baseComponents), ...Object.keys(candidateComponents)])].sort();
  const classifications = [];
  const wholeAdded = [];
  const wholeRemoved = [];
  const candidates = [];
  for (const component of commonComponents) {
    if (baseComponents[component] === undefined) {
      wholeAdded.push(component);
      continue;
    }
    if (candidateComponents[component] === undefined) {
      wholeRemoved.push(component);
      continue;
    }
    if (equal(baseComponents[component], candidateComponents[component])) continue;
    const rawOperations = operations(baseComponents[component], candidateComponents[component]);
    const classification = classify(rawOperations);
    const row = {
      component,
      decision: classification.decision,
      raw_operation_count: rawOperations.length,
      semantic_operation_count: classification.semantic.length,
      semantic_paths: classification.semantic.map((operation) => operation.path.join("/")),
    };
    classifications.push(row);
    if (classification.decision === "TYPE_CHANGED") {
      const change = classification.semantic[0];
      candidates.push({
        component,
        semantic_path: change.path.join("/"),
        base_type: change.base,
        candidate_type: change.candidate,
        base_schema: baseComponents[component],
        candidate_schema: candidateComponents[component],
      });
    }
  }
  const counts = {};
  for (const row of classifications) counts[row.decision] = (counts[row.decision] ?? 0) + 1;
  return {
    library,
    pair_id: `${library}:${baseFile}->${candidateFile}`,
    base_revision: baseFile.replace(/\.json$/, ""),
    candidate_revision: candidateFile.replace(/\.json$/, ""),
    base_source_path: baseRelativePath,
    candidate_source_path: candidateRelativePath,
    source_snapshot_commit: commit,
    base_file_commit: baseFileCommit,
    candidate_file_commit: candidateFileCommit,
    base_blob_sha1: baseBlobSha1,
    candidate_blob_sha1: candidateBlobSha1,
    base_sha256: sha256(baseBytes),
    candidate_sha256: sha256(candidateBytes),
    raw_diff_lines: numstat(basePath, candidatePath),
    selection_basis: rationale,
    changed_component_count: classifications.length,
    whole_component_added_count: wholeAdded.length,
    whole_component_removed_count: wholeRemoved.length,
    classification_counts: counts,
    type_changed_candidates: candidates,
  };
}

function preservedFile(pair, side) {
  const prefix = side === "base" ? "base" : "candidate";
  return {
    revision: pair[`${prefix}_revision`],
    commit: pair[`${prefix}_file_commit`],
    repository_snapshot_commit: pair.source_snapshot_commit,
    blob_sha1: pair[`${prefix}_blob_sha1`],
    source_path: pair[`${prefix}_source_path`],
    audit_copy_path: `source-repository/${pair[`${prefix}_source_path`]}`,
  };
}

function main() {
  const configuration = parseArguments(process.argv.slice(2));
  if (existsSync(configuration.report) || existsSync(configuration.manifest)) throw new Error("refusing to overwrite an existing pre-screen artifact");
  if (git(configuration.repository, "remote", "get-url", "origin") !== `${SOURCE_URL}.git`) throw new Error("unexpected source repository origin");
  if (git(configuration.repository, "rev-parse", "--is-shallow-repository") !== "false") {
    throw new Error("source repository is shallow; full history is required for per-file commit provenance");
  }
  const commit = git(configuration.repository, "rev-parse", "HEAD");
  const pairReports = PAIRS.map((pair) => inspectPair(configuration, ...pair, commit));
  const extracted = pairReports.flatMap((pair) => pair.type_changed_candidates.map((candidate) => ({ pair, candidate })));
  const recordIds = extracted.map(({ pair, candidate }) => `vega.schema.${pair.library}.${pair.base_revision}-to-${pair.candidate_revision}.${candidate.component}`);
  const mutationCount = extracted.length;
  const transitions = extracted.map(({ pair, candidate }, index) => ({
    record_id: recordIds[index],
    source: SOURCE,
    source_kind: SOURCE_KIND,
    source_url: SOURCE_URL,
    base_revision: pair.base_revision,
    candidate_revision: pair.candidate_revision,
    base_commit: pair.base_file_commit,
    candidate_commit: pair.candidate_file_commit,
    base_blob_sha1: pair.base_blob_sha1,
    candidate_blob_sha1: pair.candidate_blob_sha1,
    source_license: SOURCE_LICENSE,
    source_path: `${pair.base_source_path}->${pair.candidate_source_path}#/definitions/${candidate.component}`,
    family_id: `vega.schema.${pair.library}.${candidate.component}`,
    contract_id: `vega.schema.${pair.library}.${candidate.component}`,
    old_version: pair.base_revision,
    new_version: pair.candidate_revision,
    policy_pack: "baseline",
    mutation_id: "TYPE_CHANGED",
    external_use: "traceability-only",
    base_schema: candidate.base_schema,
    candidate_schema: candidate.candidate_schema,
  }));
  const uniqueSources = new Map();
  for (const pair of pairReports) {
    for (const source of [preservedFile(pair, "base"), preservedFile(pair, "candidate")]) uniqueSources.set(`${source.revision}:${source.source_path}`, source);
  }
  const manifest = {
    format_version: "dcg-external-transition-manifest-v3",
    summary: {
      total_transitions: transitions.length,
      mutation_id_counts: { TYPE_CHANGED: mutationCount },
      limitations: [
        "Pre-screen TYPE_CHANGED candidates only; no compatibility labels or model inference are present.",
        "Candidates require no-oracle V6 overlap projection before any external evaluation extraction decision.",
      ],
      preserved_source_files: [...uniqueSources.values()].sort((left, right) => `${left.source_path}:${left.revision}`.localeCompare(`${right.source_path}:${right.revision}`)),
      traceability_only: {
        status: "traceability-only",
        accuracy_evidence: false,
        clean_record_count: transitions.length,
        clean_record_ids: recordIds,
        rejected_record_count: 0,
        rejection_skew: { TYPE_CHANGED: { selected: mutationCount, clean: mutationCount, rejected: 0, rejection_rate: 0 } },
        overlap_characterization: "Candidate-only manifest; the separate no-oracle V6 projection determines overlap status.",
        preflight_report_path: "type-change-v6-overlap-projection-v1.json",
        original_preflight_report_path: "not-applicable",
        limitations: ["These candidates are not accuracy evidence and carry no oracle labels."],
      },
    },
    transitions,
  };
  const report = {
    format_version: "dcg-vega-type-change-prescreen-v1",
    artifact_kind: "pre-screen-only",
    source_provenance: {
      source: SOURCE,
      source_kind: SOURCE_KIND,
      source_url: SOURCE_URL,
      source_license: SOURCE_LICENSE,
      license_path: "LICENSE",
      repository_snapshot_commit: commit,
      repository_path: relative(process.cwd(), configuration.repository),
      versioning_documentation: "README.md",
      migration_guide_url: "https://vega.github.io/vega/docs/porting-guide/",
      revision_semantics: "base_revision and candidate_revision name immutable versioned schema files, not Git refs; per-side commit fields record the latest commit affecting each file at the pinned repository snapshot",
    },
    current_external_type_changed_coverage: { stripe: 0, kubernetes: 0 },
    selection: {
      objective: "single-operation TYPE_CHANGED candidates across major schema-format boundaries",
      pair_count: pairReports.length,
      candidate_count: mutationCount,
      exclusions: ["whole component add/remove", "METADATA_ONLY", "COMPOUND", "UNMAPPED_SINGLE_CHANGE"],
    },
    pairs: pairReports.map(({ type_changed_candidates, ...pair }) => ({
      ...pair,
      type_changed_candidate_count: type_changed_candidates.length,
      type_changed_candidates: type_changed_candidates.map(({ base_schema, candidate_schema, ...candidate }) => candidate),
    })),
    candidate_manifest_path: relative(dirname(configuration.report), configuration.manifest),
    oracle_invoked: false,
    model_inference_invoked: false,
    labels_produced: false,
  };
  mkdirSync(dirname(configuration.report), { recursive: true });
  mkdirSync(dirname(configuration.manifest), { recursive: true });
  writeFileSync(configuration.report, JSON.stringify(report, null, 2) + "\n");
  writeFileSync(configuration.manifest, JSON.stringify(manifest, null, 2) + "\n");
  process.stdout.write(`${configuration.report}\n${configuration.manifest}\n`);
}

try {
  main();
} catch (error) {
  process.stderr.write(`Vega TYPE_CHANGED pre-screen failed: ${error.message}\n`);
  process.exitCode = 1;
}
