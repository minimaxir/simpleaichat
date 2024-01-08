import os
from typing import List, Union

import httpx
from pydantic import Field

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


def wikipedia_search(query: str, n: int = 1) -> Union[str, List[str]]:
    SEARCH_PARAMS = {
        "action": "query",
        "list": "search",
        "format": "json",
        "srlimit": n,
        "srsearch": query,
        "srwhat": "text",
        "srprop": "",
    }

    r_search = httpx.get(WIKIPEDIA_API_URL, params=SEARCH_PARAMS)
    results = [x["title"] for x in r_search.json()["query"]["search"]]

    return results[0] if n == 1 else results


def wikipedia_lookup(query: str, sentences: int = 1) -> str:
    LOOKUP_PARAMS = {
        "action": "query",
        "prop": "extracts",
        "exsentences": sentences,
        "exlimit": "1",
        "explaintext": "1",
        "formatversion": "2",
        "format": "json",
        "titles": query,
    }

    r_lookup = httpx.get(WIKIPEDIA_API_URL, params=LOOKUP_PARAMS)
    return r_lookup.json()["query"]["pages"][0]["extract"]


def wikipedia_search_lookup(query: str, sentences: int = 1) -> str:
    return wikipedia_lookup(wikipedia_search(query, 1), sentences)


async def wikipedia_search_async(query: str, n: int = 1) -> Union[str, List[str]]:
    SEARCH_PARAMS = {
        "action": "query",
        "list": "search",
        "format": "json",
        "srlimit": n,
        "srsearch": query,
        "srwhat": "text",
        "srprop": "",
    }

    async with httpx.AsyncClient(proxies=os.getenv("https_proxy")) as client:
        r_search = await client.get(WIKIPEDIA_API_URL, params=SEARCH_PARAMS)
    results = [x["title"] for x in r_search.json()["query"]["search"]]

    return results[0] if n == 1 else results


async def wikipedia_lookup_async(query: str, sentences: int = 1) -> str:
    LOOKUP_PARAMS = {
        "action": "query",
        "prop": "extracts",
        "exsentences": sentences,
        "exlimit": "1",
        "explaintext": "1",
        "formatversion": "2",
        "format": "json",
        "titles": query,
    }

    async with httpx.AsyncClient(proxies=os.getenv("https_proxy")) as client:
        r_lookup = await client.get(WIKIPEDIA_API_URL, params=LOOKUP_PARAMS)
    return r_lookup.json()["query"]["pages"][0]["extract"]


async def wikipedia_search_lookup_async(query: str, sentences: int = 1) -> str:
    return await wikipedia_lookup_async(
        await wikipedia_search_async(query, 1), sentences
    )


def fd(description: str, **kwargs):
    return Field(description=description, **kwargs)


# https://stackoverflow.com/a/58938747
def remove_a_key(d, remove_key):
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key:
                del d[key]
            else:
                remove_a_key(d[key], remove_key)
