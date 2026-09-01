#!/usr/bin/env node

// Builds one persisted, human-auditable trace per Vega TYPE_CHANGED candidate
// from the candidate manifest and no-oracle V6 feature diagnostics.

import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from "node:fs";
import { dirname, relative, resolve } from "node:path";

function parseArguments(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    if (!argv[index]?.startsWith("--") || argv[index + 1] === undefined) {
      throw new Error(
        "usage: build_vega_type_change_traces.mjs --manifest <manifest.json> --diagnostics-dir <dir> --v9-dataset <dataset.json> --policy-packs <packs.json> --output-dir <dir>",
      );
    }
    values.set(argv[index], argv[index + 1]);
  }
  for (const flag of [
    "--manifest",
    "--diagnostics-dir",
    "--v9-dataset",
    "--policy-packs",
    "--output-dir",
  ]) {
    if (!values.has(flag)) throw new Error(`missing ${flag}`);
  }
  return Object.fromEntries(
    [...values].map(([key, value]) => [key, resolve(value)]),
  );
}

function readJson(path) {
  return JSON.parse(readFileSync(path));
}

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function fragmentAt(schema, semanticPath) {
  const segments = semanticPath.split("/");
  const parentSegments = segments.slice(0, -1);
  return parentSegments.reduce((value, segment) => value?.[segment], schema);
}

function fileStem(recordId) {
  if (recordId.includes("ConditionalPredicate")) return "conditional-predicate";
  if (recordId.includes("ConditionalSelection")) return "conditional-selection";
  return recordId.replace(/[^a-zA-Z0-9]+/g, "-").toLowerCase();
}

function main() {
  const args = parseArguments(process.argv.slice(2));
  const manifest = readJson(args["--manifest"]);
  const diagnosticPaths = readdirSync(args["--diagnostics-dir"])
    .filter((name) => /-full-feature-diagnostic-v\d+\.json$/.test(name))
    .sort()
    .map((name) => resolve(args["--diagnostics-dir"], name));
  const diagnostics = new Map(
    diagnosticPaths.map((path) => [readJson(path).record_id, path]),
  );
  if (diagnostics.size !== manifest.transitions.length) {
    throw new Error(
      `expected ${manifest.transitions.length} full diagnostics, found ${diagnostics.size}`,
    );
  }
  mkdirSync(args["--output-dir"], { recursive: true });
  for (const transition of manifest.transitions) {
    const diagnosticPath = diagnostics.get(transition.record_id);
    if (!diagnosticPath) throw new Error(`missing diagnostic ${transition.record_id}`);
    const diagnostic = readJson(diagnosticPath);
    if (
      diagnostic.oracle_invoked ||
      diagnostic.model_inference_invoked ||
      diagnostic.labels_produced
    ) {
      throw new Error(`forbidden execution marker in ${transition.record_id}`);
    }
    const semanticPath = "properties/value/type";
    const baseFragment = fragmentAt(transition.base_schema, semanticPath);
    const candidateFragment = fragmentAt(transition.candidate_schema, semanticPath);
    const output = resolve(
      args["--output-dir"],
      `${fileStem(transition.record_id)}-raw-trace-v2.json`,
    );
    if (existsSync(output)) throw new Error(`refusing to overwrite ${output}`);
    const trace = {
      format_version: "dcg-vega-type-change-raw-trace-v2",
      record_id: transition.record_id,
      mutation_id: transition.mutation_id,
      provenance: {
        source: transition.source,
        source_kind: transition.source_kind,
        source_url: transition.source_url,
        source_license: transition.source_license,
        source_path: transition.source_path,
        family_id: transition.family_id,
        contract_id: transition.contract_id,
        base_revision: transition.base_revision,
        candidate_revision: transition.candidate_revision,
        base_commit: transition.base_commit,
        candidate_commit: transition.candidate_commit,
        base_blob_sha1: transition.base_blob_sha1,
        candidate_blob_sha1: transition.candidate_blob_sha1,
      },
      raw_schema_change: {
        semantic_path: semanticPath,
        before: baseFragment,
        after: candidateFragment,
      },
      v6_feature_trace: {
        feature_count: diagnostic.full_v6_feature_count,
        structural_coordinate_count: diagnostic.structural_coordinate_count,
        nonzero_structural_coordinates: diagnostic.structural_coordinates.filter(
          (coordinate) => coordinate.value !== 0,
        ),
        all_nonzero_features: diagnostic.nonzero_v6_features,
      },
      overlap_projection: {
        exact_full_v6_feature_match_count:
          diagnostic.exact_full_v6_feature_match_count,
        nearest_v9_structural_distance:
          diagnostic.nearest_v9_structural_distance,
        nearest_v9_record_count: diagnostic.nearest_v9_record_count,
        nearest_v9_records: diagnostic.nearest_v9_records,
      },
      evidence: {
        manifest_path: relative(dirname(output), args["--manifest"]),
        manifest_sha256: sha256(args["--manifest"]),
        diagnostic_path: relative(dirname(output), diagnosticPath),
        diagnostic_sha256: sha256(diagnosticPath),
        v9_dataset_path: relative(dirname(output), args["--v9-dataset"]),
        v9_dataset_sha256: sha256(args["--v9-dataset"]),
        policy_packs_path: relative(dirname(output), args["--policy-packs"]),
        policy_packs_sha256: sha256(args["--policy-packs"]),
      },
      scope: {
        use: "traceability-only",
        accuracy_evidence: false,
        saturation_evidence: false,
        sparse_candidate_count: manifest.transitions.length,
      },
      oracle_invoked: false,
      model_inference_invoked: false,
      labels_produced: false,
      calibration_changed: false,
    };
    writeFileSync(output, `${JSON.stringify(trace, null, 2)}\n`);
    process.stdout.write(`${output}\n`);
  }
}

try {
  main();
} catch (error) {
  process.stderr.write(`Vega TYPE_CHANGED trace build failed: ${error.message}\n`);
  process.exitCode = 1;
}
