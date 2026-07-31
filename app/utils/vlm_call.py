import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

from app.core.logger import logger


def generate_with_retry(vlm, image, prompt, max_new_tokens=200,
                          retries=2, timeout=180, backoff=2.0):
    """
    Calls vlm.generate() with a timeout + retries.

    Timeout note: a blocking generate() call can't actually be
    cancelled mid-run -- this returns control to the caller after
    `timeout` seconds and treats it as a failure, but the abandoned
    thread keeps running in the background until it finishes on its
    own. Fine for occasional slow calls; not a hard kill switch.
    """
    last_exc = None

    for attempt in range(1, retries + 2):
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    vlm.generate, image=image, prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )
                return future.result(timeout=timeout)

        except FutureTimeoutError:
            last_exc = TimeoutError(f"generate() timed out after {timeout}s")
            logger.warning(f"VLM call timeout (attempt {attempt}/{retries + 1})")

        except Exception as e:
            last_exc = e
            logger.warning(f"VLM call failed (attempt {attempt}/{retries + 1}): {e}")

        if attempt <= retries:
            time.sleep(backoff * attempt)

    raise last_exc


def run_parallel(fn, items, max_workers=2):
    """
    Runs fn(item) for each item concurrently. Order of results matches
    order of items. If fn raises for an item, the exception object
    (not the result) is placed at that index -- one bad region doesn't
    lose the rest.
    """
    results = [None] * len(items)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {ex.submit(fn, item): i for i, item in enumerate(items)}

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = e

    return results
