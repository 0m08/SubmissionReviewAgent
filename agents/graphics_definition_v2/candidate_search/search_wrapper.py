"""
Pool-independent search wrapper.

Externally, candidate search calls: run_pool_search(query, search_fn, **kwargs)

Only the internals of each search_fn are pool-specific. Parallelization across queries for one pool lives in run_pool_search_queries.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed


def run_pool_search(query, search_fn, **kwargs):
    """
    Invoke one pool search function with a query.

    :param query: Search query string
    :param search_fn: Pool search callable; must accept query as the first argument
    :param kwargs: Pool-specific keyword args forwarded to search_fn
    :return: Whatever search_fn returns (typically a list of hits), or [] on failure
    """
    query = (query or "").strip()
    if not query:
        return []

    try:
        return search_fn(query, **kwargs)
    except Exception as exc:
        print(f'❌ Pool search failed for query "{query[:80]}": {exc}')
        return []


def run_pool_search_queries(queries, search_fn, max_workers=None, **kwargs):
    """
    Run the same pool search_fn across many queries in parallel via run_pool_search.

    :param queries: Search query strings (empty strings are skipped)
    :param search_fn: Pool search callable
    :param max_workers: Thread pool size; defaults to number of valid queries
    :param kwargs: Forwarded to each run_pool_search / search_fn call
    :return: List of (query, results) pairs for completed queries (order not guaranteed)
    """
    valid_queries = [q.strip() for q in queries if q and str(q).strip()]
    if not valid_queries:
        return []

    workers = max_workers if max_workers is not None else len(valid_queries)
    workers = max(1, min(workers, len(valid_queries)))

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_pool_search, query, search_fn, **kwargs): query
            for query in valid_queries
        }
        for future in as_completed(futures):
            query = futures[future]
            try:
                hit_list = future.result()
            except Exception as exc:
                print(f'❌ Pool search future failed for query "{query[:80]}": {exc}')
                hit_list = []
            results.append((query, hit_list if hit_list is not None else []))

    return results
