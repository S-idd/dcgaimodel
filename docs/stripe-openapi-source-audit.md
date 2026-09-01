# Stripe OpenAPI source audit — pre-oracle

Status: **No DCG JAR invocation. No model inference. No compatibility labels.**

## Provenance

- source: `stripe/openapi`
- source_kind: `public-version-history`
- source_url: `https://github.com/stripe/openapi`
- license: `MIT` (verified from the repository LICENSE)
- source_path pattern: `openapi/spec3.json#/components/schemas/<component-name>`
- policy pack for later oracle evaluation: `baseline`

## Tag-gap investigation

`v2365 → v2366` has identical `openapi/spec3.json` content (SHA-256 `3653ad45bbec54fcbe461c541c908355b715018bdf455a0e11b27bedb2cbdee5`), so it yields no transitions.

Across the 77 adjacent numeric release-tag intervals from `v2365` through `v2442`, only `v2441 → v2442` changes `openapi/spec3.json` (`+2024/-396` raw-file lines). Every fifth interval was identical: `2365→2366`, `2370→2371`, `2375→2376`, `2380→2381`, `2385→2386`, `2390→2391`, `2395→2396`, `2400→2401`, `2405→2406`, `2410→2411`, `2415→2416`, `2420→2421`, `2425→2426`, `2430→2431`, `2435→2436`, and `2440→2441`.

Conclusion: for this clone and the public `spec3.json` path, long no-op stretches are the norm; Stripe tags also cover preview and SDK-only updates. It is not one public-spec change per tag.

## Selected coherent pairs

Both selected pairs are adjacent release tags. They avoid using the large `v2441→v2442` update as the sole source transition set:

| Pair | Raw spec diff | Shared changed components | Added / removed components |
| --- | ---: | ---: | ---: |
| `v2323→v2324` | +1221/-400 | 28 | 9 / 0 |
| `v2348→v2349` | +2641/-137 | 55 | 10 / 1 |

`v2441→v2442` is retained only as gap evidence: it changes 134 shared components, 16 additions, and 2 removals; 117 changed components are title-only, so it is not a good primary transition set.

## v2323 → v2324 component decisions

- base commit: `6e116eb9909c323744c940244c9e2d59ca529fc9`
- candidate commit: `f4ac6d9a7f40730f7f50c8c1d76c27f80178e4f5`
- base source blob: `c5d6078dd0b1392623a0d0c7a579f828ccb3a1f3`
- candidate source blob: `634a4b329a8e6f0d1dd13373d9f92458d0e6ee6d`

| Component | Canonical diff | Semantic paths | Decision |
| --- | ---: | --- | --- |
| `balance_transaction` | +2/-1 | `/properties/type/enum` | include: ENUM_CHANGED |
| `checkout_session_payment_method_options` | +9/-1 | `/properties/sunbit`, `/properties/wechat_pay` | exclude: COMPOUND (2 independent semantic paths) |
| `dispute` | +1/-0 | `/properties/enhanced_eligibility_types/items/enum` | include: ENUM_CHANGED |
| `dispute_enhanced_eligibility` | +4/-0 | `/properties/mastercard_compliance` | include: FIELD_ADDED |
| `dispute_enhanced_evidence` | +4/-0 | `/properties/mastercard_compliance` | include: FIELD_ADDED |
| `financial_connections.account` | +4/-0 | `/properties/status_details` | include: FIELD_ADDED |
| `invoice_payment_method_options_us_bank_account_linked_account_options` | +2/-1 | — | exclude: METADATA_ONLY (documentation or vendor-extension only) |
| `invoices_payment_settings` | +1/-0 | `/properties/payment_method_types/items/enum` | include: ENUM_CHANGED |
| `linked_account_options_common` | +2/-1 | — | exclude: METADATA_ONLY (documentation or vendor-extension only) |
| `payment_intent_payment_method_options` | +11/-0 | `/properties/sunbit` | include: FIELD_ADDED |
| `payment_method_bizum` | +8/-1 | `/properties/buyer_id` | include: FIELD_ADDED |
| `payment_method_blik` | +8/-1 | `/properties/buyer_id` | include: FIELD_ADDED |
| `payment_method_details_bizum` | +6/-0 | `/properties/buyer_id` | include: FIELD_ADDED |
| `payment_method_details_card` | +6/-0 | `/properties/transaction_link_id` | include: FIELD_ADDED |
| `payment_method_details_crypto` | +2/-0 | `/properties/network/enum`, `/properties/token_currency/enum` | exclude: COMPOUND (2 independent semantic paths) |
| `payment_method_details_payment_record_bizum` | +6/-0 | `/properties/buyer_id` | include: FIELD_ADDED |
| `payment_method_details_pix` | +6/-0 | `/properties/fingerprint` | include: FIELD_ADDED |
| `payment_method_options_satispay` | +9/-0 | `/properties/setup_future_usage` | include: FIELD_ADDED |
| `payment_method_options_wechat_pay` | +1/-1 | — | exclude: METADATA_ONLY (documentation or vendor-extension only) |
| `payment_method_pix` | +8/-1 | `/properties/fingerprint` | include: FIELD_ADDED |
| `payments_primitives_payment_records_resource_payment_method_card_details` | +0/-27 | `/properties/iin`, `/properties/issuer`, `/properties/stored_credential_usage` | exclude: COMPOUND (3 independent semantic paths) |
| `payments_primitives_payment_records_resource_payment_method_details` | +1/-1 | `/properties/sunbit/$ref` | exclude: UNMAPPED_SINGLE_CHANGE (single path outside conservative supported shapes) |
| `setup_attempt_payment_method_details` | +4/-0 | `/properties/satispay` | include: FIELD_ADDED |
| `setup_attempt_payment_method_details_pix` | +8/-1 | `/properties/fingerprint` | include: FIELD_ADDED |
| `subscriptions_resource_payment_settings` | +1/-0 | `/properties/payment_method_types/items/enum` | include: ENUM_CHANGED |
| `subscriptions_resource_subscription_invoice_settings` | +21/-0 | `/properties/custom_fields`, `/properties/footer` | exclude: COMPOUND (2 independent semantic paths) |
| `tax_product_registrations_resource_country_options_europe` | +2/-1 | — | exclude: METADATA_ONLY (documentation or vendor-extension only) |
| `verification_session_redaction` | +2/-1 | `/properties/status/enum` | include: ENUM_CHANGED |

Selection: **19** of 28 changed shared components are conservatively eligible for a future external manifest. The remainder are excluded as metadata-only, compound, or unmapped semantic changes.

Whole-component exclusions: added in candidate — `bank_connections_resource_account_status_details`, `bank_connections_resource_account_status_details_api_resource_active_status_details`, `checkout_sunbit_payment_method_options`, `checkout_wechat_pay_payment_method_options`, `dispute_enhanced_eligibility_mastercard_compliance`, `dispute_enhanced_evidence_mastercard_compliance`, `payment_flows_private_payment_methods_satispay_setup_attempt_details`, `payment_method_details_payment_record_sunbit`, `payment_method_options_sunbit`; removed in candidate — none. They have no two-sided component pair.

## v2348 → v2349 component decisions

- base commit: `d014b5753db579bd2001bb3153f367cbf6f6be92`
- candidate commit: `af5309cae53e5f666f9686dfed306d6d3b5fdc67`
- base source blob: `634a4b329a8e6f0d1dd13373d9f92458d0e6ee6d`
- candidate source blob: `92c4d0de7cafefbb253ab4b31bb970b4cb89b4a3`

| Component | Canonical diff | Semantic paths | Decision |
| --- | ---: | --- | --- |
| `account_capability_future_requirements` | +1/-1 | — | exclude: METADATA_ONLY (documentation or vendor-extension only) |
| `account_capability_requirements` | +1/-1 | — | exclude: METADATA_ONLY (documentation or vendor-extension only) |
| `account_future_requirements` | +1/-1 | — | exclude: METADATA_ONLY (documentation or vendor-extension only) |
| `account_requirements` | +2/-2 | — | exclude: METADATA_ONLY (documentation or vendor-extension only) |
| `balance_transaction` | +1/-1 | — | exclude: METADATA_ONLY (documentation or vendor-extension only) |
| `bank_connections_resource_link_account_session_filters` | +9/-0 | `/properties/require_payment_method_support` | include: FIELD_ADDED |
| `billing_portal.configuration` | +1/-1 | — | exclude: METADATA_ONLY (documentation or vendor-extension only) |
| `card` | +0/-5 | `/properties/iin` | include: FIELD_REMOVED |
| `checkout_payco_payment_method_options` | +7/-0 | `/properties/setup_future_usage` | include: FIELD_ADDED |
| `checkout_samsung_pay_payment_method_options` | +7/-0 | `/properties/setup_future_usage` | include: FIELD_ADDED |
| `connect_embedded_disputes_list_features` | +6/-1 | `/properties/smart_disputes_management`, `/required` | exclude: COMPOUND (2 independent semantic paths) |
| `connect_embedded_payment_disputes_features` | +6/-1 | `/properties/smart_disputes_management`, `/required` | exclude: COMPOUND (2 independent semantic paths) |
| `connect_embedded_payments_features` | +6/-1 | `/properties/smart_disputes_management`, `/required` | exclude: COMPOUND (2 independent semantic paths) |
| `dispute_payment_method_details_card` | +7/-1 | `/properties/network`, `/required` | exclude: COMPOUND (2 independent semantic paths) |
| `external_account_requirements` | +1/-1 | — | exclude: METADATA_ONLY (documentation or vendor-extension only) |
| `financial_connections.session` | +13/-1 | `/properties/bank_account_token`, `/properties/limits`, `/properties/manual_entry` | exclude: COMPOUND (3 independent semantic paths) |
| `funding_instructions_bank_transfer_financial_address` | +1/-0 | `/properties/supported_networks/items/enum` | include: ENUM_CHANGED |
| `invoice_setting_quote_setting` | +21/-0 | `/properties/custom_fields`, `/properties/footer` | exclude: COMPOUND (2 independent semantic paths) |
| `invoice_setting_subscription_schedule_phase_setting` | +21/-0 | `/properties/custom_fields`, `/properties/footer` | exclude: COMPOUND (2 independent semantic paths) |
| `invoice_setting_subscription_schedule_setting` | +21/-0 | `/properties/custom_fields`, `/properties/footer` | exclude: COMPOUND (2 independent semantic paths) |
| `invoices_payment_settings` | +2/-0 | `/properties/payment_method_types/items/enum` | include: ENUM_CHANGED |
| `invoices_resource_invoice_tax_id` | +2/-1 | `/properties/type/enum` | include: ENUM_CHANGED |
| `issuing_authorization_request` | +1/-0 | `/properties/reason/enum` | include: ENUM_CHANGED |
| `issuing_card_shipping` | +7/-0 | `/properties/business_name`, `/properties/carrier/enum` | exclude: COMPOUND (2 independent semantic paths) |
| `issuing_transaction_fleet_fuel_price_data` | +1/-1 | — | exclude: METADATA_ONLY (documentation or vendor-extension only) |
| `legal_entity_company` | +8/-0 | `/properties/administrative_address`, `/properties/principal_place_of_business` | exclude: COMPOUND (2 independent semantic paths) |
| `notification_event_data` | +2/-0 | `/properties/object/additionalProperties`, `/properties/previous_attributes/additionalProperties` | exclude: COMPOUND (2 independent semantic paths) |
| `payment_flows_private_payment_methods_payco_payment_method_options` | +7/-0 | `/properties/setup_future_usage` | include: FIELD_ADDED |
| `payment_flows_private_payment_methods_samsung_pay_payment_method_options` | +7/-0 | `/properties/setup_future_usage` | include: FIELD_ADDED |
| `payment_intent` | +107/-0 | `/properties/allowed_payment_method_types` | include: FIELD_ADDED |
| `payment_intent_next_action` | +1/-0 | `/properties/use_stripe_sdk/additionalProperties` | include: CONSTRAINT_OR_RESTRICTION_CHANGED |
| `payment_method_details` | +1/-1 | `/properties/alipay/$ref` | exclude: UNMAPPED_SINGLE_CHANGE (single path outside conservative supported shapes) |
| `payment_method_details_fpx` | +4/-1 | `/properties/bank/enum` | include: ENUM_CHANGED |
| `payment_method_details_passthrough_card` | +1/-1 | — | exclude: METADATA_ONLY (documentation or vendor-extension only) |
| `payment_method_fpx` | +4/-1 | `/properties/bank/enum` | include: ENUM_CHANGED |
| `payment_pages_checkout_session_tax_id` | +2/-1 | `/properties/type/enum` | include: ENUM_CHANGED |
| `payments_primitives_payment_records_resource_payment_method_amazon_pay_details_resource_funding_resource_funding_card` | +1/-1 | — | exclude: METADATA_ONLY (documentation or vendor-extension only) |
| `payments_primitives_payment_records_resource_payment_method_card_details_resource_three_d_secure` | +1/-0 | `/properties/result/enum` | include: ENUM_CHANGED |
| `payments_primitives_payment_records_resource_payment_method_details` | +3/-3 | `/properties/alipay/$ref`, `/properties/au_becs_debit/$ref`, `/properties/bacs_debit/$ref` | exclude: COMPOUND (3 independent semantic paths) |
| `person_future_requirements` | +1/-1 | — | exclude: METADATA_ONLY (documentation or vendor-extension only) |
| `person_requirements` | +1/-1 | — | exclude: METADATA_ONLY (documentation or vendor-extension only) |
| `refund` | +54/-0 | `/properties/customer`, `/properties/customer_account`, `/properties/payment_method` | exclude: COMPOUND (3 independent semantic paths) |
| `setup_intent` | +107/-0 | `/properties/allowed_payment_method_types` | include: FIELD_ADDED |
| `setup_intent_next_action` | +1/-0 | `/properties/use_stripe_sdk/additionalProperties` | include: CONSTRAINT_OR_RESTRICTION_CHANGED |
| `subscription_schedule_phase_configuration` | +4/-0 | `/properties/trial` | include: FIELD_ADDED |
| `subscriptions_resource_payment_settings` | +2/-0 | `/properties/payment_method_types/items/enum` | include: ENUM_CHANGED |
| `tax_id` | +2/-1 | `/properties/type/enum` | include: ENUM_CHANGED |
| `tax_product_registrations_resource_country_options_united_states` | +10/-0 | `/properties/mass_transit_parking_tax`, `/properties/parking_tax`, `/properties/type/enum` | exclude: COMPOUND (3 independent semantic paths) |
| `tax_product_resource_customer_details_resource_tax_id` | +2/-1 | `/properties/type/enum` | include: ENUM_CHANGED |
| `tax_product_resource_line_item_tax_rate_details` | +2/-0 | `/properties/tax_type/enum` | include: ENUM_CHANGED |
| `tax_product_resource_tax_rate_details` | +2/-0 | `/properties/tax_type/enum` | include: ENUM_CHANGED |
| `tax_rate` | +2/-0 | `/properties/tax_type/enum` | include: ENUM_CHANGED |
| `three_d_secure_details` | +1/-0 | `/properties/result/enum` | include: ENUM_CHANGED |
| `three_d_secure_details_charge` | +1/-0 | `/properties/result/enum` | include: ENUM_CHANGED |
| `topup` | +40/-0 | `/properties/initiated_by`, `/properties/payment_method`, `/properties/payment_method_options` | exclude: COMPOUND (3 independent semantic paths) |

Selection: **27** of 55 changed shared components are conservatively eligible for a future external manifest. The remainder are excluded as metadata-only, compound, or unmapped semantic changes.

Whole-component exclusions: added in candidate — `bank_connections_resource_link_account_session_limits`, `bank_connections_resource_link_account_session_manual_entry`, `payment_method_details_alipay`, `payment_method_details_payment_record_alipay`, `payment_method_details_payment_record_au_becs_debit`, `payment_method_details_payment_record_bacs_debit`, `tax_product_registrations_resource_country_options_us_mass_transit_parking_tax`, `tax_product_registrations_resource_country_options_us_parking_tax`, `topup_resource_payment_method_options`, `topup_resource_us_bank_account`; removed in candidate — `payment_flows_private_payment_methods_alipay_details`. They have no two-sided component pair.

## Next action

Generate the V3 external manifest from only rows marked **include**, preserve both exact source files per pair, then perform pinned-JAR labeling as a separate next step. Do not assign compatibility labels from this audit.
