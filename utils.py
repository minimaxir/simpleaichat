import httpx
from typing import List, Union

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


def wikipedia_search(query: str, n: int = 1) -> Union[str, List[str]]:
    SEARCH_PARAMS = {
        "action": "query",
        "list": "search",
        "format": "json",
        "srlimit": n,
        "srsearch": query,
        "srprop": "",
    }

    r_search = httpx.get(WIKIPEDIA_API_URL, params=SEARCH_PARAMS)
    results = [x["title"] for x in r_search.json()["query"]["search"]]

    return results[0] if n == 1 else results


def wikipedia_lookup(query: str) -> str:
    LOOKUP_PARAMS = {
        "action": "query",
        "prop": "extracts",
        "exsentences": "2",
        "exlimit": "1",
        "explaintext": "1",
        "formatversion": "2",
        "format": "json",
        "titles": query,
    }

    r_lookup = httpx.get(WIKIPEDIA_API_URL, params=LOOKUP_PARAMS)
    return r_lookup.json()["query"]["pages"][0]["extract"]


def wikipedia_search_lookup(query: str) -> str:
    return wikipedia_lookup(wikipedia_search(query, 1))
