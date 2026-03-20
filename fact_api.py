import requests
import os

API_KEY = os.getenv("FACT_API_KEY")

def google_fact_check(text):
    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": text,
            "key": API_KEY
        }

        res = requests.get(url, params=params)
        data = res.json()

        if "claims" in data:
            return 0.1  # likely real
        else:
            return 0.9  # suspicious

    except:
        return 0.5

import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def gemini_fact_check(text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
        Analyze this news and decide if it is FAKE or REAL.

        News:
        {text}

        Answer only: FAKE or REAL
        """

        response = model.generate_content(prompt)
        result = response.text.strip().upper()

        if "FAKE" in result:
            return 0.9
        else:
            return 0.1

    except:
        return 0.5