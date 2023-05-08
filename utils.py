import httpx


def wikipedia_lookup_closest(query):
    API_URL = "https://en.wikipedia.org/w/api.php"
    SEARCH_PARAMS = {
        "action": "query",
        "list": "search",
        "format": "json",
        "srlimit": "1",
        "srprop": "",
    }

    LOOKUP_PARAMS = {
        "action": "query",
        "prop": "extracts",
        "exsentences": "3",
        "exlimit": "1",
        "explaintext": "1",
        "formatversion": "2",
        "format": "json",
    }

    search_params = dict(SEARCH_PARAMS, srsearch=query)
    r_search = httpx.get(API_URL, params=search_params)
    lookup_query = r_search.json()["query"]["search"][0]["title"]

    lookup_params = dict(LOOKUP_PARAMS, titles=lookup_query)
    r_lookup = httpx.get(API_URL, params=lookup_params)
    return r_lookup.json()["query"]["pages"][0]["extract"]
