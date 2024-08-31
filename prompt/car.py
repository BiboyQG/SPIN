def get_file_name(json_object):
    return f"{json_object['make']}_{json_object['model']}_{json_object['year']}"


llm_prompt = """You are an expert at summarizing car review articles in JSON format. Extract the most relevant and useful information from the provided article, focusing on the following key aspects of the vehicle:

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
    "observed": "28 mpg",
      "epa": {{
        "combined": 30,
        "city": 28,
        "highway": 32
      }}
  }},
  "performance": {{
    "acceleration": {{
      "0to60": 3.1,
      "0to100": 7.3,
      "0to130": 0,
      "0to150": 0
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
  "brakes": {{
    "front": "17.3-in vented, cross-drilled, carbon-ceramic disc",
    "rear": "16.1-in vented, cross-drilled, carbon-ceramic disc"
  }},
  "tires": {{
    "front": "Pirelli P Zero Corsa PZC4",
    "rear": "Pirelli P Zero Corsa PZC4"
  }},
  "suspensionAndChassis": {{
    "suspension": {{
      "front": "multilink",
      "rear": "multilink"
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
  ],
  "overallVerdict": "The 2024 Porsche Cayenne Turbo E-Hybrid Coupe is a powerful and luxurious SUV that offers a great driving experience. However, it may not be the best choice for those who prioritize fuel efficiency or affordability."
}}

Now, let's start:
{0}
"""

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_data",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "make": {
                    "type": "string"
                },
                "model": {
                    "type": "string"
                },
                "year": {
                    "type": "number"
                },
                "vehicleType": {
                    "type": "string"
                },
                "price": {
                    "type": "object",
                    "properties": {
                        "base": {
                            "type": "number"
                        },
                        "asTested": {
                            "type": "number"
                        }
                    },
                    "additionalProperties": False,
                    "required": ["base", "asTested"]
                },
                "powertrain": {
                    "type": "object",
                    "properties": {
                        "engine": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string"
                                },
                                "horsepower": {
                                    "type": "number"
                                },
                                "torque": {
                                    "type": "number"
                                }
                            },
                            "required": ["type", "horsepower", "torque"],
                            "additionalProperties": False
                        },
                        "transmission": {
                            "type": "string"
                        }
                    },
                    "additionalProperties": False,
                    "required": ["engine", "transmission"]
                },
                "battery": {
                    "type": "object",
                    "properties": {
                        "size": {
                            "type": "string"
                        },
                        "onboardCharger": {
                            "type": "string"
                        }
                    },
                    "additionalProperties": False,
                    "required": ["size", "onboardCharger"]
                },
                "fuelEfficiency": {
                    "type": "object",
                    "properties": {
                        "observed": {
                            "type": "string"
                        },
                        "epa": {
                            "type": "object",
                            "properties": {
                                "combined": {
                                    "type": "number"
                                },
                                "city": {
                                    "type": "number"
                                },
                                "highway": {
                                    "type": "number"
                                }
                            },
                            "required": ["combined", "city", "highway"],
                            "additionalProperties": False
                        }
                    },
                    "additionalProperties": False,
                    "required": ["observed", "epa"]
                },
                "performance": {
                    "type": "object",
                    "properties": {
                        "acceleration": {
                            "type": "object",
                            "properties": {
                                "0to60": {
                                    "type": "number"
                                },
                                "0to100": {
                                    "type": "number"
                                },
                                "0to130": {
                                    "type": "number"
                                },
                                "0to150": {
                                    "type": "number"
                                }
                            },
                            "required": ["0to60", "0to100", "0to130", "0to150"],
                            "additionalProperties": False
                        },
                        "quarterMile": {
                            "type": "object",
                            "properties": {
                                "time": {
                                    "type": "number"
                                },
                                "speed": {
                                    "type": "number"
                                }
                            },
                            "required": ["time", "speed"],
                            "additionalProperties": False
                        },
                        "topSpeed": {
                            "type": "number"
                        }
                    },
                    "additionalProperties": False,
                    "required": ["acceleration", "quarterMile", "topSpeed"]
                },
                "dimensions": {
                    "type": "object",
                    "properties": {
                        "wheelbase": {
                            "type": "number"
                        },
                        "length": {
                            "type": "number"
                        },
                        "width": {
                            "type": "number"
                        },
                        "height": {
                            "type": "number"
                        },
                        "passengerVolume": {
                            "type": "object",
                            "properties": {
                                "front": {
                                    "type": "number"
                                },
                                "rear": {
                                    "type": "number"
                                }
                            },
                            "required": ["front", "rear"],
                            "additionalProperties": False
                        },
                        "cargoVolume": {
                            "type": "object",
                            "properties": {
                                "behindFront": {
                                    "type": "number"
                                },
                                "behindRear": {
                                    "type": "number"
                                }
                            },
                            "required": ["behindFront", "behindRear"],
                            "additionalProperties": False
                        },
                        "curbWeight": {
                            "type": "number"
                        }
                    },
                    "additionalProperties": False,
                    "required": ["wheelbase", "length", "width", "height", "passengerVolume", "cargoVolume", "curbWeight"]
                },
                "brakes": {
                    "type": "object",
                    "properties": {
                        "front": {
                            "type": "string"
                        },
                        "rear": {
                            "type": "string"
                        }
                    },
                    "additionalProperties": False,
                    "required": ["front", "rear"]
                },
                "tires": {
                    "type": "object",
                    "properties": {
                        "front": {
                            "type": "string"
                        },
                        "rear": {
                            "type": "string"
                        }
                    },
                    "additionalProperties": False,
                    "required": ["front", "rear"]
                },
                "suspensionAndChassis": {
                    "type": "object",
                    "properties": {
                        "suspension": {
                            "type": "object",
                            "properties": {
                                "front": {
                                    "type": "string"
                                },
                                "rear": {
                                    "type": "string"
                                }
                            },
                            "required": ["front", "rear"],
                            "additionalProperties": False
                        }
                    },
                    "additionalProperties": False,
                    "required": ["suspension"]
                },
                "strengths": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "weaknesses": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "overallVerdict": {
                    "type": "string"
                }
            },
            "required": ["make", "model", "year", "vehicleType", "price", "powertrain", "battery", "fuelEfficiency", "performance", "dimensions", "brakes", "tires", "suspensionAndChassis", "strengths", "weaknesses", "overallVerdict"],
            "additionalProperties": False
        }
    }
}