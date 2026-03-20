import os
import requests

API_KEY = os.getenv("FACT_API_KEY")

def google_fact_check(query):
    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

        params = {
            "query": query,
            "key": API_KEY
        }

        res = requests.get(url, params=params).json()

        claims = res.get("claims", [])

        if len(claims) > 0:
            return 0.2   # likely real
        else:
            return 0.7   # suspicious

    except:
        return 0.5