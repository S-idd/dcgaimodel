# Training and leakage prevention

The intended pipeline is:

```text
DCG fixtures → prepared dataset artifact → seeded family-aware split
             → fit scaler on train only → transform all partitions
             → train + epoch validation loss → validate thresholds → final test → persist
```

`DatasetSplitConfig` provides ratios and a seed. `PreparedDcgDataset::split_by_group` uses those values to assign complete family/split groups consistently, so variants from one contract never straddle partitions. The present default model construction and batch ordering are deterministic; its seed is nevertheless retained as artifact metadata for the split and future randomized training options.

Never fit the scaler on the complete dataset, validation set, test set, or an inference request. `DcgPipelineConfig::run` fits it exclusively on the raw training partition and transforms validation/test data with the retained state. It records training and forward-only validation loss per epoch; it never uses validation or test data for gradient updates.

Fixture data is self-contained and illustrative, not production evidence.
