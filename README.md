Render a Codex rollout/session file as a transcript-style HTML page.

The parser recognizes several historical Codex rollout shapes, but the format is not guaranteed to be future-proof across versions/models.

To replace embedded base64 image data URLs with `[image data]`:

```sh
python3 rollout_to_html.py /path/to/rollout.jsonl --exclude-image-data -o transcript.html
```

The option applies to every rendered conversation, thinking, and tool/code section.
You can also use the short option `-e`.
Use `--conversation-only` (or `-c`) to include only user prompts and final Codex replies.
Surrounding text, image paths, and ordinary image URLs are preserved. Without the
option, image data is included as before. The original rollout file is unchanged.

