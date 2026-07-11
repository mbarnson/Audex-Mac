# Licensing And Local Artifacts

Audex-Mac source code is MIT licensed under the repository's
[MIT license](../../LICENSE). That license covers this project's code, not every
dependency or model it can load.

NVIDIA's [Audex-2B](https://huggingface.co/nvidia/Nemotron-Labs-Audex-2B) and
[Audex-30B-A3B](https://huggingface.co/nvidia/Nemotron-Labs-Audex-30B-A3B)
model weights and inference components retain NVIDIA's license.

Review the license shipped with the selected model before downloading or using
it. Audex-Mac displays the model and license information at the download prompt.

Keep downloaded weights, Hugging Face snapshots, generated environments, cache
state, conversations, run logs, SQLite catalogs, and every generated or recorded
audio file out of Git. The repository's ignore rules cover the normal local paths.

Large binary artifacts, especially model shards and WAV files, do not belong in
the repository. Summarize reproducible findings in the engineering docs and keep
the raw evidence under `.audex/`.

Install the repository's local pre-commit hook with:

```sh
scripts/install-hooks.sh
```
