"""Static taxonomy: 30 folder classes -> 6 material groups (from the dataset README).
The 30 -> recyclability target map lives in config.yaml (a policy choice), not here.
"""

MATERIAL_GROUP = {
    "aerosol_cans": "metal", "aluminum_food_cans": "metal", "aluminum_soda_cans": "metal",
    "steel_food_cans": "metal",
    "cardboard_boxes": "paper", "cardboard_packaging": "paper", "magazines": "paper",
    "newspaper": "paper", "office_paper": "paper", "paper_cups": "paper",
    "glass_beverage_bottles": "glass", "glass_cosmetic_containers": "glass", "glass_food_jars": "glass",
    "clothing": "textile", "shoes": "textile",
    "coffee_grounds": "organic", "eggshells": "organic", "food_waste": "organic", "tea_bags": "organic",
    "disposable_plastic_cutlery": "plastic", "plastic_cup_lids": "plastic",
    "plastic_detergent_bottles": "plastic", "plastic_food_containers": "plastic",
    "plastic_shopping_bags": "plastic", "plastic_soda_bottles": "plastic", "plastic_straws": "plastic",
    "plastic_trash_bags": "plastic", "plastic_water_bottles": "plastic",
    "styrofoam_cups": "plastic", "styrofoam_food_containers": "plastic",
}

FINE_CLASSES = sorted(MATERIAL_GROUP)  # 30, alphabetical -> stable integer ids
