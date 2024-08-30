llm_prompt = '''You are an expert at summarizing car review articles in JSON format. Extract the most relevant and useful information from the provided article, focusing on the following key aspects of the vehicle:

- Make and model
- Year
- Vehicle type (sedan, SUV, coupe, etc.)
- Base price and as-tested price
- Powertrain details (engine type, horsepower, torque, electric motor specs if applicable, transmission)
- Fuel efficiency (EPA estimates and observed, if available)
- Battery size and electric range (for hybrid or electric vehicles)
- Performance metrics:
  - 0-60 mph acceleration
  - Quarter-mile time and speed
  - Top speed
  - Braking distance
- Key dimensions:
  - Wheelbase
  - Length, width, height
  - Curb weight
  - Passenger and cargo volume
- Suspension and chassis details
- Safety features and ratings (if mentioned)
- Comfort and convenience features
- Infotainment system specs
- Exterior and interior design highlights
- Strengths and weaknesses (as noted in the article)
- Overall verdict or conclusion

If any information is not explicitly stated in the article, then include it in the JSON output with the value as 'null'. Organize the data in a clear, hierarchical structure, using nested objects and arrays as needed. Provide an example output based on the car review article provided.

Example:
User input: [An car review article about a Porsche Cayenne Turbo E-Hybrid Coupe]
Your output:
{{
  "make": "Porsche",
  "model": "Cayenne Turbo E-Hybrid Coupe",
  "year": 2024,
  "vehicleType": "SUV",
  "price": {{
    "base": 153050,
    "asTested": 190210
  }},
  "powertrain": {{
    "engine": {{
      "type": "twin-turbocharged and intercooled DOHC 32-valve V-8",
      "horsepower": 591,
      "torque": 590
    }},
    "electricMotor": {{
      "horsepower": 174,
      "torque": 339
    }},
    "combinedOutput": {{
      "horsepower": 729,
      "torque": 700
    }},
    "transmission": "8-speed automatic"
  }},
  "battery": {{
    "size": "21.8-kWh",
    "onboardCharger": "11.0-kW"
  }},
  "fuelEfficiency": {{
    "observed": "19 MPGe",
    "highway": {{
      "ev": "58 MPGe",
      "hybrid": "22 mpg"
    }},
    "evRange": "24 mi"
  }},
  "performance": {{
    "acceleration": {{
      "0to60": 3.1,
      "0to100": 7.3
    }},
    "quarterMile": {{
      "time": 11.3,
      "speed": 124
    }},
    "topSpeed": 183
  }},
  "dimensions": {{
    "wheelbase": 114.0,
    "length": 194.1,
    "width": 78.1,
    "height": 66.4,
    "passengerVolume": {{
      "front": 54,
      "rear": 50
    }},
    "cargoVolume": {{
      "behindFront": 47,
      "behindRear": 23
    }},
    "curbWeight": 5672
  }},
  "brakes": "Porsche Ceramic Composite Brakes",
  "tires": "Pirelli P Zero Corsa PZC4",
  "suspensionAndChassis": {{
    "suspension": {{
      "front": "multilink",
      "rear": "multilink"
    }},
    "brakes": {{
      "front": "17.3-in vented, cross-drilled, carbon-ceramic disc",
      "rear": "16.1-in vented, cross-drilled, carbon-ceramic disc"
    }}
  }},
  "strengths": [
    "Great V-8 sound",
    "Impressive performance",
    "Cheaper than its predecessor"
  ],
  "weaknesses": [
    "Occasionally janky transmission behavior",
    "Porsche's options-heavy cost spiral"
  ]
}}

Now, let's start:
{0}
'''