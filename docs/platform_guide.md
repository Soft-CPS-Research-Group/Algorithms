# Run a job on the platform

The worker runs the same configuration and job entry point used locally. Mount the job volume at `/data`, set `OPEVA_BASE_DIR=/data`, and pass `--config /data/config.yaml --job_id <id>`. Dataset paths in the configuration must exist on the worker. Choose an image built for its architecture and record the image revision.

From the Algorithms checkout, `python scripts/manage_remote_experiment.py --help` lists the operations. Set `OPEVA_SERVER` to the configured orchestrator.

1. Validate the YAML locally with `algorithms-job --config <config> --validate`. A dry run also checks that the dataset opens and the agent binds to its observation and action spaces.
2. Use `prepare` to generate a payload for the selected worker, image and submitter. Run `preflight` for the worker before submitting.
3. Submit the reviewed payload with `submit --confirm-submit`. Keep the returned job ID for monitoring and collection.
4. Use `watch` for status and `scripts/collect_remote_results.py` for the completed exports. Check the effective settings, final episode and seed before comparing KPIs across jobs.

For an input or topology error, reproduce the dry run on the same dataset and image. For a failed job, inspect `progress/progress.json`, the job log and finalization status. For an export error, check the requested artifact type and whether the selected agent supports deployment. See the [runtime guide](runtime/README.md) and [exports](runtime/exports.md).
