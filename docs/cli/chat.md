# Streaming Chat Client

The repository does not currently provide an interactive `chat` binary. It
includes a minimal streaming client for one prompt:

```bash
python3 scripts/chat.py "What's your name?"
```

If the prompt argument is omitted, the script asks for one line of input. Useful
options are:

```bash
python3 scripts/chat.py --max-tokens 200 "Explain ownership in Rust."
python3 scripts/chat.py --url http://server:8000/v1/chat/completions "Hello"
```

The script requires `curl`, requests SSE streaming, removes the JSON envelope,
and prints each text delta immediately.
