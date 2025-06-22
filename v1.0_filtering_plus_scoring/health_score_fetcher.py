
import os, requests
from functools import lru_cache

USDA_API_KEY = os.getenv("USDA_API_KEY")

@lru_cache(maxsize=50)
def get_health_score(food_name: str) -> int:
    url = f"https://api.nal.usda.gov/fdc/v1/foods/search?api_key={USDA_API_KEY}&query={food_name}&pageSize=1"
    resp = requests.get(url)
    if resp.status_code != 200:
        return 3  # default
    foods = resp.json().get("foods", [])
    if not foods:
        return 3

    nutrients = {n["nutrientName"]: n["value"] for n in foods[0].get("foodNutrients", [])}
    score = 0
    if nutrients.get("Dietary fiber", 0) >= 2: score += 1
    if nutrients.get("Vitamin C", 0) >= 20: score += 1
    if nutrients.get("Iron", 0) >= 1: score += 1
    if nutrients.get("Vitamin A", 0) >= 500: score += 1
    if nutrients.get("Calories", 0) <= 100: score += 1
    return min(score, 5)
