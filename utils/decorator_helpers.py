import functools
import multiprocessing
import time
from googleapiclient.errors import HttpError

def try_n_times(n, wait=1, backoff='linear'):
    """
    Decorator to retry a function up to n times if it raises an exception.
    Args:
        n (int): Maximum number of attempts.
        wait (int): Time to wait between attempts in seconds.
        backoff (str): Backoff strategy. Options are 'linear' and 'exponential'.
    Returns:
        A decorator function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_wait = wait
            for attempt in range(n):
                try:
                    return func(*args, **kwargs)
                except KeyboardInterrupt:
                    print('Keyboard interrupt')
                    break
                except Exception as e:
                    if attempt < n - 1:
                        print(f"Attempt {attempt + 1} failed with {e}. Retrying after {current_wait} seconds...")
                        time.sleep(current_wait)
                        if backoff == 'linear':
                            current_wait += wait
                        elif backoff == 'exponential':
                            current_wait *= 2
                    else:
                        print(f"Attempt {attempt + 1} failed with {e}. No more retries left.")
                        raise Exception("Max retries exceeded after {} attempts.".format(n))
        return wrapper
    return decorator


def time_limit_process(seconds):
    """
    Decorator that runs the decorated function in a separate process
    and raises TimeoutError if it does not complete within `seconds`.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # This is the target function we will run in a child process
            def target_func(queue, *a, **kw):
                try:
                    result = func(*a, **kw)
                    queue.put(result)
                except Exception as e:
                    queue.put(e)

            # Create a queue to get return value or exception from the child
            queue = multiprocessing.Queue()
            # Create the child process
            process = multiprocessing.Process(target=target_func, args=(queue, *args), kwargs=kwargs)
            # Start the child process
            process.start()
            # Wait for the process to finish or time out
            process.join(seconds)

            if process.is_alive():
                # Time’s up; kill the process
                process.terminate()
                process.join()
                raise TimeoutError(f"Function call timed out after {seconds} seconds.")

            # If not alive, retrieve the result from the queue
            if not queue.empty():
                result = queue.get()
                # If the worker raised an exception, re-raise it here
                if isinstance(result, Exception):
                    raise result
                return result
            else:
                return None  # If no output was put in the queue
        return wrapper
    return decorator


def _is_youtube_quota_exceeded(error: HttpError) -> bool:
    """True when YouTube Data API daily search quota is exhausted (403/429)."""
    error_text = str(error).lower()
    if any(
        marker in error_text
        for marker in ("quotaexceeded", "ratelimitexceeded", "quota exceeded")
    ):
        return True
    status = getattr(getattr(error, "resp", None), "status", None)
    return status in (403, 429) and "quota" in error_text


def cycle_api_keys_decorator(api_keys):
    """
    Returns a decorator that cycles through `api_keys` on YouTube quota errors,
    remembering the last used key index across multiple calls to the wrapped function.
    """
    # This index will persist between calls to the wrapped function
    current_key_index = 0

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal current_key_index

            # We will try up to len(api_keys) times before concluding all are exhausted
            attempts = 0

            while attempts < len(api_keys):
                key = api_keys[current_key_index]
                try:
                    return func(*args, **kwargs, developer_key=key)
                except HttpError as e:
                    if _is_youtube_quota_exceeded(e):
                        exhausted_key_num = current_key_index + 1
                        current_key_index = (current_key_index + 1) % len(api_keys)
                        next_key_num = current_key_index + 1
                        print(
                            f"Key {exhausted_key_num} exhausted, cycling to key {next_key_num}..."
                        )
                        attempts += 1
                    else:
                        # If it's some other error, re-raise
                        raise

            # If we exhaust all keys and never succeed, raise an exception
            raise RuntimeError("All API keys have been exhausted.")
        return wrapper
    return decorator
