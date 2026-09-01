# Ring Sentinel — project instructions

- **Never read `.env`.** It holds `GROQ_API_KEY`. Don't open it with Read, don't `cat`/`grep` its contents in Bash. To check whether a key is set, test non-emptiness only (e.g. `[ -s .env ] && grep -q '^GROQ_API_KEY=.\+' .env`) without printing the value, or check `os.environ` from inside a Python process. If the app isn't picking up the key, ask the user to confirm it's set rather than reading the file yourself.
