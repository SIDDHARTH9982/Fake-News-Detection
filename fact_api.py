import requests

def google_fact_check(query):
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        res = requests.get(url).json()

        related = res.get("RelatedTopics", [])

        if len(related) > 0:
            return 0.2
        else:
            return 0.7

    except:
        return 0.5