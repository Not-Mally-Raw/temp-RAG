import threading

class LLMTimeoutError(Exception):
    pass


def call_with_timeout(llm, prompt, timeout=15):
    result = {}
    error = {}

    def target():
        try:
            result["value"] = llm.call(prompt)
        except Exception as e:
            error["err"] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise LLMTimeoutError("LLM call timed out")

    if "err" in error:
        raise error["err"]

    return result.get("value")
