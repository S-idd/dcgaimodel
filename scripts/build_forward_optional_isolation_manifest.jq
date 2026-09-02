def variants:
  [
    {name: "string", schema: {type: "string"}},
    {name: "integer", schema: {type: "integer"}},
    {name: "boolean", schema: {type: "boolean"}},
    {name: "array", schema: {type: "array", items: {type: "string"}}}
  ];

# Source fragments contain document-local references that cannot resolve once
# extracted from their parent repository document. Replace each reference node
# with the permissive empty schema in both base and candidate. This projection
# is label-free and identical on both sides, so the only pairwise mutation is
# still the generated optional-field addition.
def erase_document_local_refs:
  walk(
    if type == "object" and has("$ref")
    then {type: "object", additionalProperties: true}
    else .
    end
  );

($policies[0].packs | keys) as $packs |
($stripe[0].transitions
  | map(select(
      .family_id == "stripe.openapi.card"
      or .family_id == "stripe.openapi.payment_intent_next_action"
    ))
  | unique_by(.family_id)
  | map({
      key: ("forward-optional." + .family_id),
      source: "stripe/openapi-generated-forward-optional",
      source_url: "https://github.com/stripe/openapi",
      revision: .base_revision,
      commit: .base_commit,
      blob: .base_blob_sha1,
      license: .source_license,
      path: .source_path,
      schema: .base_schema
    })) as $stripe_families |
((($vega[0].definitions // $vega[0]["$defs"]) | to_entries)
  | map(select(.key == "Scale" or .key == "StyleConfigIndex"))
  | map({
      key: ("forward-optional.vega.schema.vega-lite.v6.4.1." + .key),
      source: "vega/schema-generated-forward-optional",
      source_url: "https://github.com/vega/schema",
      revision: "vega-lite/v6.4.1.json",
      commit: "8feea15c3a84d4261887f5c7e0cbb74909f1c8a2",
      blob: "f0d44514bee7f6c239ce4116f7b81206427d2108",
      license: "BSD-3-Clause",
      path: ("vega-lite/v6.4.1.json#/definitions/" + .key),
      schema: .value
    })) as $vega_families |
($stripe_families + $vega_families) as $families |
[
  $families[] as $family |
  ["open", "closed"][] as $profile |
  variants[] as $variant |
  select(
    $variant.name != "array"
    or $family.key == "forward-optional.stripe.openapi.card"
    or $family.key == "forward-optional.vega.schema.vega-lite.v6.4.1.Scale"
  ) |
  $packs[] as $pack |
  (($family.schema | erase_document_local_refs) + {additionalProperties: ($profile == "open")}) as $base |
  ($base | .properties[("dcg_generated_optional_" + $variant.name)] = $variant.schema) as $candidate |
  {
    record_id: ($family.key + "." + $profile + "." + $variant.name + "." + $pack),
    source: $family.source,
    source_kind: "public-schema-repository",
    source_url: $family.source_url,
    base_revision: $family.revision,
    candidate_revision: "generated-optional-addition-v1",
    base_commit: $family.commit,
    candidate_commit: ("generated-from-" + $family.commit),
    base_blob_sha1: $family.blob,
    candidate_blob_sha1: "generated-candidate-not-a-git-blob",
    source_license: $family.license,
    source_path: $family.path,
    family_id: $family.key,
    contract_id: ($family.key + "." + $profile),
    old_version: $family.revision,
    new_version: "generated-optional-addition-v1",
    policy_pack: $pack,
    mutation_id: "OPTIONAL_FIELD_ADDED",
    mutation_variant: ("optional-" + $variant.name + "-field"),
    external_use: "traceability-only",
    base_schema: $base,
    candidate_schema: $candidate
  }
] as $transitions |
{
  format_version: "dcg-external-transition-manifest-v3",
  summary: {
    total_transitions: ($transitions | length),
    mutation_id_counts: {OPTIONAL_FIELD_ADDED: ($transitions | length)},
    limitations: [
      "The base schemas are independently sourced from preserved Stripe and Vega repositories; candidate transitions are generated optional-field additions and are not represented as public version-history events.",
      "Extracted document-local $ref nodes are replaced by a lint-valid open-object placeholder in both base and candidate so each projected family is standalone. The projection is identical on both sides and is not an additional pairwise mutation.",
      "source_kind records the public repository provenance of each base schema only. Every generated transition remains traceability-only and is never external accuracy evidence.",
      "This manifest exists solely to reuse preflight_external_manifest as the V9 source/family/pair/exact-feature/near-structural contamination gate before corpus JAR invocation."
    ],
    preserved_source_files: ($stripe[0].summary.preserved_source_files + [
      {
        revision: "vega-lite/v6.4.1.json",
        commit: "8feea15c3a84d4261887f5c7e0cbb74909f1c8a2",
        blob_sha1: "f0d44514bee7f6c239ce4116f7b81206427d2108",
        source_path: "vega-lite/v6.4.1.json",
        audit_copy_path: "../../external/vega-schema-type-change-v1/source-repository/vega-lite/v6.4.1.json"
      }
    ]),
    traceability_only: {
      status: "traceability-only",
      accuracy_evidence: false,
      clean_record_count: ($transitions | length),
      clean_record_ids: ($transitions | map(.record_id)),
      rejected_record_count: 0,
      rejection_skew: {
        OPTIONAL_FIELD_ADDED: {
          selected: ($transitions | length),
          clean: ($transitions | length),
          rejected: 0,
          rejection_rate: 0
        }
      },
      overlap_characterization: "Projected clean set only; every retained pair must pass the existing V9 preflight again before generation.",
      preflight_report_path: "v9-preflight-passed-v2.json",
      original_preflight_report_path: "source-profile-prescreen-v1.json",
      limitations: [
        "The manifest supplies contamination-gate inputs, not oracle labels or model scores.",
        "Generated candidates must not be evaluated by the frozen-external-evidence runner as public transition accuracy evidence."
      ]
    }
  },
  transitions: $transitions
}
