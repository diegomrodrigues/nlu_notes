from functools import wraps
import time
import traceback

def retry_on_error(max_retries=3):
    """Decorator to retry operations on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"❌ Final attempt failed for {func.__name__}:")
                        print(f"   Error type: {type(e).__name__}")
                        print(f"   Error message: {str(e)}")
                        print("   Traceback:")
                        print("   " + "\n   ".join(traceback.format_exc().split("\n")))
                        raise Exception(f"Failed after {max_retries} attempts: {type(e).__name__}: {str(e)}") from e
                    print(f"⚠️ Attempt {attempt + 1}/{max_retries} failed for {func.__name__}:")
                    print(f"   Error type: {type(e).__name__}")
                    print(f"   Error message: {str(e)}")
                    retry_delay = 2 ** attempt
                    print(f"   Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            return None
        return wrapper
    return decorator
