# Keeps the Ollama server log small. Under concurrent benchmark requests its runner
# writes several lines per request (slots, prompt cache, sampler settings, access
# log): 13 GB in under two hours, enough to exceed the /home quota. Those lines are
# dropped unless they report a problem; the server's own messages are kept, up to
# limit_mb megabytes (default 200), after which later lines are dropped too.
# Usage: ollama serve 2>&1 | awk -v limit_mb=200 -f ollama-log-filter.awk > ollama.log

BEGIN { limit = (limit_mb > 0 ? limit_mb : 200) * 1024 * 1024 }

# Failed requests stay in the log; successful ones are dropped.
/^\[GIN\] / && /\| 200 \|/ { next }
/^(slot |srv |\t)/ && !/[Ee]rr|[Ff]ail|[Ww]arn|[Pp]anic/ { next }

written < limit { print; fflush(); written += length($0) + 1; next }
!truncated {
    print "... Ollama log reached " limit / 1024 / 1024 " MB; later lines are dropped"
    fflush()
    truncated = 1
}
