import joblib
import numpy as np
import pandas as pd
from colorama import Fore,Style, init #for colors

# Initializing colorama
init(autoreset=True)

model = joblib.load("water.pkl")
scaler = joblib.load("water_scaler.pkl")

#guideline ranges
guidelines = {
    "ph": (6.5, 8.5),
    "Hardness": (0, 500),
    "Solids": (0, 50000),
    "Chloramines": (0, 4),
    "Sulfate": (0, 500),
    "Conductivity": (0, 400),
    "Organic_carbon": (0, 10),
    "Trihalomethanes": (0, 80),
    "Turbidity": (0, 5)
}

# Suggestions
suggestions = {
    "ph": "Adjust pH using neutralizers (lime for low pH, acid dosing for high pH).",
    "Hardness": "Use water softening methods like ion-exchange or reverse osmosis.",
    "Solids": "Reduce solids using filtration or reverse osmosis.",
    "Chloramines": "Remove with activated carbon filters or UV treatment.",
    "Sulfate": "Apply reverse osmosis or distillation to lower sulfate levels.",
    "Conductivity": "Indicates dissolved salts — treat with reverse osmosis if high.",
    "Organic_carbon": "Reduce with activated carbon filters or advanced oxidation.",
    "Trihalomethanes": "Improve disinfection process, use activated carbon filtration.",
    "Turbidity": "Use sedimentation, sand filters, or membrane filtration."
}

# Taking inputs from user
print(Fore.CYAN + Style.BRIGHT +"\n HELLO WELCOME TO HYDROSAFE : AN ML BASED WATER POTABILITY PREDICTOR " +Style.RESET_ALL)
print("\n💧 Enter water quality parameters:")
features = []
params = list(guidelines.keys())
user_inputs = {}

for param in params:
    try:
        val = float(input(f"   → {param}: "))
        if val < 0:  # negative values have no physical sense, replacing with 0
            print(f"⚠ Invalid negative value for {param}, using 0 instead.")
            val = 0
        user_inputs[param] = val
        features.append(val)
    except ValueError:
        print(f"❌ Invalid input for {param}, using default 0.")
        user_inputs[param] = 0
        features.append(0)

# Convert input to DataFrame (for scaler)
features_data = pd.DataFrame([features], columns=params)
features_scaled = scaler.transform(features_data)

# ML prediction
prediction = model.predict(features_scaled)[0]
ml_result = "SAFE " if prediction == 1 else "UNSAFE "
print("\n🤖  ML Model Prediction:", ml_result)

# Guideline validation
print("\n📊  Checking Parameters Against WHO Guidelines:")
unsafe_flags = []
for param, (low, high) in guidelines.items():
    value = user_inputs[param]
    if low <= value <= high:
        print(f"   ✔ {param}: {value} (within {low}-{high})")
    else:
        print(f"   ⚠ {param}: {value} (outside {low}-{high})")
        unsafe_flags.append(param)

# Final combined verdict
print("\n🔎 Final Verdict:")
if prediction == 1 and not unsafe_flags:
    print(Fore.GREEN + Style.BRIGHT+"\n✅  WATER IS SAFE TO DRINK (Potable).\n"+Style.RESET_ALL)
elif prediction == 1 and unsafe_flags:
    print("⚠ ML predicted SAFE, but guideline check flagged issues.")
    print(Fore.MAGENTA + Style.BRIGHT + "   Problematic parameters: " + ", ".join(unsafe_flags) + Style.RESET_ALL)
    print(Fore.RED + Style.BRIGHT+"\n❌  Final Verdict: WATER IS UNSAFE TO DRINK.\n"+Style.RESET_ALL)
else:
    print(Fore.RED + Style.BRIGHT+"\n❌ WATER IS UNSAFE TO DRINK.\n"+Style.RESET_ALL)
    if unsafe_flags:
        print(Fore.MAGENTA + Style.BRIGHT + "   Problematic parameters: " + ", ".join(unsafe_flags) + Style.RESET_ALL)

# Suggestions if any problems found
if unsafe_flags:
    print(Fore.YELLOW + Style.BRIGHT + "\n💡 Suggestions for Improvement:" + Style.RESET_ALL)
    for param in unsafe_flags:
        print(Fore.YELLOW + f"   → {param}: {suggestions[param]}" + Style.RESET_ALL)
