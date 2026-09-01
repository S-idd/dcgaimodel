#!/usr/bin/env node

// Builds a persisted, no-oracle Kubernetes OpenAPI pre-screen report. This is
// deliberately not an external-transition manifest: it retains no raw schema
// pairs and neither launches Java nor performs model inference. A short-lived
// manifest is created only to invoke the existing V9 overlap validator, whose
// report is embedded in the final audit file.

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";

const SPEC_PATH = "api/openapi-spec/swagger.json";
const SOURCE_URL = "https://github.com/kubernetes/kubernetes";
const SOURCE = "kubernetes/kubernetes";
const SOURCE_LICENSE = "Apache-2.0";
const FORMAT_VERSION = "dcg-kubernetes-openapi-prescreen-v1";
const SOURCE_KIND = "public-version-history";
const METADATA_KEYS = new Set(["description", "title", "example", "examples", "externalDocs"]);
const CONSTRAINT_KEYS = new Set([
  "minimum",
  "maximum",
  "exclusiveMinimum",
  "exclusiveMaximum",
  "multipleOf",
  "minLength",
  "maxLength",
  "pattern",
  "minItems",
  "maxItems",
  "uniqueItems",
  "minProperties",
  "maxProperties",
  "additionalProperties",
  "nullable",
  "format",
]);

const releaseNoteDrivenPairs = new Map([
  ["v1.21.0->v1.22.0", "Kubernetes documents removed deprecated API versions in v1.22."],
  ["v1.24.0->v1.25.0", "Kubernetes documents removed deprecated API versions in v1.25."],
  ["v1.28.0->v1.29.0", "Kubernetes documents removal of flowcontrol.apiserver.k8s.io/v1beta2 in v1.29."],
  ["v1.31.0->v1.32.0", "Kubernetes documents removal of flowcontrol.apiserver.k8s.io/v1beta3 in v1.32."],
]);
const discoveredSchemaPair = "v1.27.0->v1.28.0";
const versions = Array.from({ length: 12 }, (_, index) => `v1.${21 + index}.0`);

function parseArguments(argv) {
  const argumentsByName = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index];
    const value = argv[index + 1];
    if (!flag?.startsWith("--") || value === undefined) {
      throw new Error("usage: kubernetes_prescreen.mjs --repo <git-repository> --output <audit.json> --v9-dataset <dataset.json> --jar <jar> --policy-packs <packs.json> [--preflight-command <binary>]");
    }
    argumentsByName.set(flag, value);
  }
  const required = ["--repo", "--output", "--v9-dataset", "--jar", "--policy-packs"];
  for (const flag of required) {
    if (!argumentsByName.has(flag)) {
      throw new Error(`missing ${flag}`);
    }
  }
  return {
    repository: resolve(argumentsByName.get("--repo")),
    output: resolve(argumentsByName.get("--output")),
    v9Dataset: resolve(argumentsByName.get("--v9-dataset")),
    jar: resolve(argumentsByName.get("--jar")),
    policyPacks: resolve(argumentsByName.get("--policy-packs")),
    preflightCommand: resolve(argumentsByName.get("--preflight-command") ?? "target/release/dcgaimodel"),
  };
}

function git(repository, argumentsList) {
  return execFileSync("git", ["-C", repository, ...argumentsList], {
    encoding: "utf8",
    // The Kubernetes Swagger documents are several megabytes each; Node's
    // one-megabyte child-process default would truncate a legitimate source.
    maxBuffer: 64 * 1024 * 1024,
  }).trim();
}

function jsonEqual(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function collectOperations(base, candidate, path = [], operations = []) {
  if (jsonEqual(base, candidate)) {
    return operations;
  }
  if (base === undefined) {
    operations.push({ operation: "add", path, candidate });
    return operations;
  }
  if (candidate === undefined) {
    operations.push({ operation: "remove", path, base });
    return operations;
  }
  if (
    Array.isArray(base) ||
    Array.isArray(candidate) ||
    base === null ||
    candidate === null ||
    typeof base !== "object" ||
    typeof candidate !== "object"
  ) {
    operations.push({ operation: "change", path, base, candidate });
    return operations;
  }
  for (const key of [...new Set([...Object.keys(base), ...Object.keys(candidate)])].sort()) {
    collectOperations(base[key], candidate[key], [...path, key], operations);
  }
  return operations;
}

function isMetadata(operation) {
  return operation.path.some((segment) => segment.startsWith("x-kubernetes-")) ||
    operation.path.some((segment) => METADATA_KEYS.has(segment));
}

function classifyOperations(allOperations) {
  const semanticOperations = allOperations.filter((operation) => !isMetadata(operation));
  if (semanticOperations.length === 0) {
    return { decision: "METADATA_ONLY", semanticOperations: [] };
  }

  const kinds = semanticOperations.map((operation) => {
    const finalSegment = operation.path.at(-1);
    const propertiesIndex = operation.path.lastIndexOf("properties");
    if (propertiesIndex >= 0 && operation.path.length === propertiesIndex + 2 && operation.operation === "add") {
      return "FIELD_ADDED";
    }
    if (propertiesIndex >= 0 && operation.path.length === propertiesIndex + 2 && operation.operation === "remove") {
      return "FIELD_REMOVED";
    }
    if (finalSegment === "type" && operation.operation === "change") {
      return "TYPE_CHANGED";
    }
    if (finalSegment === "enum") {
      return "ENUM_CHANGED";
    }
    if (CONSTRAINT_KEYS.has(finalSegment) || finalSegment === "required") {
      return "CONSTRAINT_OR_RESTRICTION_CHANGED";
    }
    return "UNMAPPED";
  });
  const uniqueKinds = [...new Set(kinds)];
  if (semanticOperations.length === 1 && uniqueKinds.length === 1 && uniqueKinds[0] !== "UNMAPPED") {
    return { decision: uniqueKinds[0], semanticOperations };
  }
  if (semanticOperations.length === 1) {
    return { decision: "UNMAPPED_SINGLE_CHANGE", semanticOperations };
  }
  return { decision: "COMPOUND", semanticOperations, mutationKinds: uniqueKinds };
}

function readSpec(repository, tag) {
  return JSON.parse(git(repository, ["show", `${tag}:${SPEC_PATH}`]));
}

function countBy(rows, key) {
  return rows.reduce((counts, row) => {
    counts[row[key]] = (counts[row[key]] ?? 0) + 1;
    return counts;
  }, {});
}

function inspectPair(repository, baseRevision, candidateRevision) {
  const baseDocument = readSpec(repository, baseRevision);
  const candidateDocument = readSpec(repository, candidateRevision);
  const baseDefinitions = baseDocument.definitions ?? {};
  const candidateDefinitions = candidateDocument.definitions ?? {};
  const componentNames = [...new Set([...Object.keys(baseDefinitions), ...Object.keys(candidateDefinitions)])].sort();
  const changedComponents = [];
  const wholeComponentAdded = [];
  const wholeComponentRemoved = [];

  for (const component of componentNames) {
    if (baseDefinitions[component] === undefined) {
      wholeComponentAdded.push(component);
      continue;
    }
    if (candidateDefinitions[component] === undefined) {
      wholeComponentRemoved.push(component);
      continue;
    }
    if (jsonEqual(baseDefinitions[component], candidateDefinitions[component])) {
      continue;
    }
    const operations = collectOperations(baseDefinitions[component], candidateDefinitions[component]);
    const classification = classifyOperations(operations);
    changedComponents.push({
      component,
      decision: classification.decision,
      mutation_kinds: classification.mutationKinds ?? [classification.decision],
      raw_operation_count: operations.length,
      semantic_paths: classification.semanticOperations.map((operation) => operation.path.join("/")),
    });
  }

  const diffTokens = git(repository, ["diff", "--numstat", baseRevision, candidateRevision, "--", SPEC_PATH]).split(/\s+/);
  const pairId = `${baseRevision}->${candidateRevision}`;
  return {
    pair_id: pairId,
    base_revision: baseRevision,
    candidate_revision: candidateRevision,
    base_commit: git(repository, ["rev-parse", `${baseRevision}^{}`]),
    candidate_commit: git(repository, ["rev-parse", `${candidateRevision}^{}`]),
    base_blob_sha1: git(repository, ["rev-parse", `${baseRevision}:${SPEC_PATH}`]),
    candidate_blob_sha1: git(repository, ["rev-parse", `${candidateRevision}:${SPEC_PATH}`]),
    raw_diff_lines: { added: Number(diffTokens[0]), removed: Number(diffTokens[1]) },
    changed_component_count: changedComponents.length,
    whole_component_added_count: wholeComponentAdded.length,
    whole_component_removed_count: wholeComponentRemoved.length,
    classification_counts: countBy(changedComponents, "decision"),
    changed_components: changedComponents,
  };
}

function buildTemporaryPreflightManifest(selectedPairs, repository) {
  const eligible = new Set(["FIELD_REMOVED", "TYPE_CHANGED", "CONSTRAINT_OR_RESTRICTION_CHANGED"]);
  const transitions = [];
  for (const pair of selectedPairs) {
    const baseDefinitions = readSpec(repository, pair.base_revision).definitions ?? {};
    const candidateDefinitions = readSpec(repository, pair.candidate_revision).definitions ?? {};
    for (const component of pair.changed_components.filter((item) => eligible.has(item.decision))) {
      transitions.push({
        record_id: `kubernetes.prescreen.${pair.base_revision}-to-${pair.candidate_revision}.${component.component}`,
        source: SOURCE,
        source_kind: SOURCE_KIND,
        source_url: SOURCE_URL,
        base_revision: pair.base_revision,
        candidate_revision: pair.candidate_revision,
        base_commit: pair.base_commit,
        candidate_commit: pair.candidate_commit,
        base_blob_sha1: pair.base_blob_sha1,
        candidate_blob_sha1: pair.candidate_blob_sha1,
        source_license: SOURCE_LICENSE,
        source_path: `${SPEC_PATH}#/definitions/${component.component}`,
        family_id: `kubernetes.openapi.${component.component}`,
        contract_id: `kubernetes.openapi.${component.component}`,
        old_version: pair.base_revision,
        new_version: pair.candidate_revision,
        policy_pack: "baseline",
        mutation_id: component.decision,
        base_schema: baseDefinitions[component.component],
        candidate_schema: candidateDefinitions[component.component],
      });
    }
  }
  const mutationIdCounts = {
    FIELD_REMOVED: transitions.filter((transition) => transition.mutation_id === "FIELD_REMOVED").length,
    TYPE_CHANGED: transitions.filter((transition) => transition.mutation_id === "TYPE_CHANGED").length,
    CONSTRAINT_OR_RESTRICTION_CHANGED: transitions.filter((transition) => transition.mutation_id === "CONSTRAINT_OR_RESTRICTION_CHANGED").length,
  };
  const sourceFiles = [...new Set(transitions.flatMap((transition) => [transition.base_revision, transition.candidate_revision]))]
    .sort()
    .map((revision) => ({
      revision,
      commit: git(repository, ["rev-parse", `${revision}^{}`]),
      blob_sha1: git(repository, ["rev-parse", `${revision}:${SPEC_PATH}`]),
      source_path: SPEC_PATH,
      audit_copy_path: `temporary-pre-screen/${revision}-swagger.json`,
    }));
  return {
    format_version: "dcg-external-transition-manifest-v3",
    summary: {
      total_transitions: transitions.length,
      mutation_id_counts: mutationIdCounts,
      limitations: [
        "Temporary Kubernetes pre-screen input only; no oracle labels or model inference.",
        "Restricted to classified single-mutation removal, type, and constraint candidates.",
        "This temporary input is not an extracted external-evaluation manifest.",
      ],
      preserved_source_files: sourceFiles,
    },
    transitions,
  };
}

function runNoOracleProjection(configuration, selectedPairs) {
  const temporaryDirectory = mkdtempSync(join(tmpdir(), "dcg-kubernetes-preflight-"));
  const manifestPath = join(temporaryDirectory, "projection-input.json");
  const outputPath = join(temporaryDirectory, "preflight-report.json");
  try {
    const manifest = buildTemporaryPreflightManifest(selectedPairs, configuration.repository);
    writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
    execFileSync(configuration.preflightCommand, [
      "preflight-external",
      "--manifest", manifestPath,
      "--v9-dataset", configuration.v9Dataset,
      "--jar", configuration.jar,
      "--policy-packs", configuration.policyPacks,
      "--output", outputPath,
    ], { encoding: "utf8" });
    const preflight = JSON.parse(readFileSync(outputPath, "utf8"));
    if (preflight.oracle_invoked || preflight.model_inference_invoked) {
      throw new Error("no-oracle projection unexpectedly invoked an oracle or model inference");
    }
    return {
      input_mutation_distribution: manifest.summary.mutation_id_counts,
      preflight,
      exact_feature_overlap_count: preflight.records.filter((record) => record.exact_model_feature_overlap).length,
      near_structural_overlap_count: preflight.records.filter((record) => record.nearest_policy_free_structural_distance <= 1).length,
    };
  } finally {
    rmSync(temporaryDirectory, { recursive: true, force: true });
  }
}

function main() {
  const configuration = parseArguments(process.argv.slice(2));
  if (existsSync(configuration.output)) {
    throw new Error(`refusing to overwrite existing audit artifact: ${configuration.output}`);
  }
  for (const input of [configuration.repository, configuration.v9Dataset, configuration.jar, configuration.policyPacks, configuration.preflightCommand]) {
    if (!existsSync(input)) {
      throw new Error(`required input does not exist: ${input}`);
    }
  }
  if (git(configuration.repository, ["remote", "get-url", "origin"]) !== SOURCE_URL + ".git") {
    throw new Error("repository origin must be https://github.com/kubernetes/kubernetes.git");
  }

  const scannedPairs = versions.slice(0, -1).map((baseRevision, index) => inspectPair(configuration.repository, baseRevision, versions[index + 1]));
  const selectedPairs = scannedPairs.filter((pair) => releaseNoteDrivenPairs.has(pair.pair_id) || pair.pair_id === discoveredSchemaPair);
  const projection = runNoOracleProjection(configuration, selectedPairs);
  const report = {
    format_version: FORMAT_VERSION,
    artifact_kind: "pre-screen-only",
    source_provenance: {
      source: SOURCE,
      source_kind: SOURCE_KIND,
      source_url: SOURCE_URL,
      source_license: SOURCE_LICENSE,
      source_path_pattern: `${SPEC_PATH}#/definitions/<component-name>`,
      license_path: "LICENSES/LICENSE",
      removal_notes: "https://kubernetes.io/docs/reference/using-api/deprecation-guide/",
      no_raw_schema_pairs_persisted: true,
    },
    method: {
      scan: "all adjacent Kubernetes minor-version bumps from v1.21.0 through v1.32.0",
      single_mutation_rules: ["FIELD_ADDED", "FIELD_REMOVED", "TYPE_CHANGED", "ENUM_CHANGED", "CONSTRAINT_OR_RESTRICTION_CHANGED"],
      exclusions: ["whole component adds/removes", "METADATA_ONLY", "COMPOUND", "UNMAPPED_SINGLE_CHANGE"],
      projection: "existing preflight-external command with max structural distance 1; it hashes the supplied JAR but never launches Java or runs model inference",
    },
    selected_pairs: selectedPairs.map((pair) => ({
      ...pair,
      selection_basis: releaseNoteDrivenPairs.get(pair.pair_id) ?? "Discovered schema-level FIELD_REMOVED candidate; not presented as a release-note-confirmed API removal.",
    })),
    all_scanned_pair_summaries: scannedPairs.map((pair) => ({
      pair_id: pair.pair_id,
      raw_diff_lines: pair.raw_diff_lines,
      changed_component_count: pair.changed_component_count,
      whole_component_added_count: pair.whole_component_added_count,
      whole_component_removed_count: pair.whole_component_removed_count,
      classification_counts: pair.classification_counts,
    })),
    projection,
    oracle_invoked: false,
    model_inference_invoked: false,
    labels_produced: false,
  };
  mkdirSync(dirname(configuration.output), { recursive: true });
  writeFileSync(configuration.output, JSON.stringify(report, null, 2) + "\n");
  process.stdout.write(`${configuration.output}\n`);
}

try {
  main();
} catch (error) {
  process.stderr.write(`kubernetes pre-screen failed: ${error.message}\n`);
  process.exitCode = 1;
}
