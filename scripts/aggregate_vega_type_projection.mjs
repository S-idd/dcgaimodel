#!/usr/bin/env node

// Aggregates persisted no-oracle feature diagnostics for the conservative
// Vega TYPE_CHANGED candidate set. This script does not invoke an oracle,
// load a model, assign labels, or alter calibration.

import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from "node:fs";
import { dirname, relative, resolve } from "node:path";

function argumentsByName(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    if (!argv[index]?.startsWith("--") || argv[index + 1] === undefined) {
      throw new Error(
        "usage: aggregate_vega_type_projection.mjs --manifest <manifest.json> --diagnostics-dir <dir> --v9-dataset <dataset.json> --policy-packs <packs.json> --output <report.json>",
      );
    }
    values.set(argv[index], argv[index + 1]);
  }
  for (const flag of [
    "--manifest",
    "--diagnostics-dir",
    "--v9-dataset",
    "--policy-packs",
    "--output",
  ]) {
    if (!values.has(flag)) throw new Error(`missing ${flag}`);
  }
  return Object.fromEntries(
    [...values].map(([key, value]) => [key, resolve(value)]),
  );
}

function bytes(path) {
  return readFileSync(path);
}

function sha256(path) {
  return createHash("sha256").update(bytes(path)).digest("hex");
}

function main() {
  const args = argumentsByName(process.argv.slice(2));
  const output = args["--output"];
  if (existsSync(output)) throw new Error(`refusing to overwrite ${output}`);
  const manifest = JSON.parse(bytes(args["--manifest"]));
  const diagnosticPaths = readdirSync(args["--diagnostics-dir"])
    .filter((name) => /-feature-diagnostic-v\d+\.json$/.test(name))
    .sort()
    .map((name) => resolve(args["--diagnostics-dir"], name));
  const diagnostics = diagnosticPaths.map((path) => ({
    path,
    value: JSON.parse(bytes(path)),
  }));
  if (diagnostics.length !== manifest.transitions.length) {
    throw new Error(
      `expected ${manifest.transitions.length} diagnostics, found ${diagnostics.length}`,
    );
  }
  const candidateIds = new Set(
    manifest.transitions.map((transition) => transition.record_id),
  );
  for (const { value } of diagnostics) {
    if (!candidateIds.has(value.record_id)) {
      throw new Error(`unexpected diagnostic ${value.record_id}`);
    }
    if (
      value.oracle_invoked ||
      value.model_inference_invoked ||
      value.labels_produced
    ) {
      throw new Error(`forbidden execution marker in ${value.record_id}`);
    }
  }
  const threshold = 1;
  const records = diagnostics.map(({ path, value }) => ({
    record_id: value.record_id,
    mutation_id: value.mutation_id,
    nearest_v9_structural_distance: value.nearest_v9_structural_distance,
    exact_full_v6_feature_match_count: value.exact_full_v6_feature_match_count,
    nearest_v9_record_count: value.nearest_v9_record_count,
    accepted: value.nearest_v9_structural_distance > threshold,
    decision:
      value.nearest_v9_structural_distance > threshold
        ? "ACCEPT_NO_NEAR_STRUCTURAL_OVERLAP"
        : "REJECT_NEAR_STRUCTURAL_OVERLAP",
    nearest_mutation_variants: [
      ...new Set(
        value.nearest_v9_records.map((record) => record.mutation_variant),
      ),
    ].sort(),
    nearest_example_differences:
      value.nearest_v9_records[0]?.differing_structural_coordinates ?? [],
    diagnostic_path: relative(dirname(output), path),
    diagnostic_sha256: sha256(path),
  }));
  const accepted = records.filter((record) => record.accepted).length;
  const report = {
    format_version: "dcg-vega-type-change-overlap-projection-v2",
    artifact_kind: "no-oracle-v6-overlap-projection",
    source: "vega/schema",
    source_kind: "public-version-history",
    source_url: "https://github.com/vega/schema",
    source_license: "BSD-3-Clause",
    manifest_path: relative(dirname(output), args["--manifest"]),
    manifest_sha256: sha256(args["--manifest"]),
    v9_dataset_path: relative(dirname(output), args["--v9-dataset"]),
    v9_dataset_sha256: sha256(args["--v9-dataset"]),
    policy_packs_path: relative(dirname(output), args["--policy-packs"]),
    policy_packs_sha256: sha256(args["--policy-packs"]),
    rejection_threshold: {
      metric: "policy-free V6 structural coordinate distance",
      reject_when_distance_lte: threshold,
    },
    summary: {
      candidate_count: records.length,
      accepted_count: accepted,
      rejected_count: records.length - accepted,
      type_changed_accepted_count: accepted,
      type_changed_rejected_count: records.length - accepted,
      outcome:
        accepted === 0
          ? "NO_NON_OVERLAPPING_TYPE_CHANGED_CANDIDATES"
          : "NON_OVERLAPPING_TYPE_CHANGED_CANDIDATES_FOUND",
    },
    records,
    oracle_invoked: false,
    model_inference_invoked: false,
    labels_produced: false,
    calibration_changed: false,
  };
  mkdirSync(dirname(output), { recursive: true });
  writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${output}\n`);
}

try {
  main();
} catch (error) {
  process.stderr.write(
    `Vega TYPE_CHANGED projection aggregation failed: ${error.message}\n`,
  );
  process.exitCode = 1;
}
