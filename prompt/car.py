from pydantic import BaseModel, Field
from typing import List
import json

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
                    "type": "string",
                    "description": "The type of vehicle, e.g. 'sedan', 'SUV', 'coupe', etc."
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
                        "electricMotor": {
                            "type": "object",
                            "properties": {
                                "horsepower": {
                                    "type": "number",
                                    "description": "Horsepower of the electric motor. If the car is not electric, the value should be 0."
                                },
                                "torque": {
                                    "type": "number",
                                    "description": "Torque of the electric motor. If the car is not electric, the value should be 0."
                                }
                            },
                            "required": ["horsepower", "torque"],
                            "additionalProperties": False
                        },
                        "combinedOutput": {
                            "type": "object",
                            "properties": {
                                "horsepower": {
                                    "type": "number"
                                },
                                "torque": {
                                    "type": "number"
                                }
                            },
                            "required": ["horsepower", "torque"],
                            "additionalProperties": False
                        },
                        "transmission": {
                            "type": "string",
                            "description": "Transmission of the car, such as 'direct-drive', 'CVT', 'single-speed', 'multi-speed', 'dual-motor', 'in-wheel motor', etc."
                        }
                    },
                    "additionalProperties": False,
                    "required": ["engine", "electricMotor", "combinedOutput", "transmission"]
                },
                "battery": {
                    "type": "object",
                    "properties": {
                        "size": {
                            "type": "string",
                            "description": "Size of the battery in kWh. The unit is always kWh and should be included. Eg. '100.0 kWh', '107.5 kWh', '0 kWh' (for non-hybrid cars)."
                        },
                        "onboardCharger": {
                            "type": "string",
                            "description": "Onboard charger of the car. The unit is always kW and should be included. Eg. '11.0 kW', '10.5 kW', '0 kW' (for non-hybrid cars)."
                        }
                    },
                    "additionalProperties": False,
                    "required": ["size", "onboardCharger"]
                },
                "fuelEfficiency": {
                    "type": "object",
                    "properties": {
                        "observed": {
                            "type": "string",
                            "description": "Observed fuel efficiency of the car. The value should be a range or a specific value according to the article with the unit 'MPGe' included. Eg. '79-81 MPGe', '17 MPGe', '0 MPGe' (if not applicable)."
                        },
                        "epa": {
                            "type": "object",
                            "properties": {
                                "combined": {
                                    "type": "string"
                                },
                                "city": {
                                    "type": "string"
                                },
                                "highway": {
                                    "type": "string"
                                }
                            },
                            "required": ["combined", "city", "highway"],
                            "additionalProperties": False,
                            "description": "Fuel efficiency of the car according to EPA of city, highway and combined. For each of them, the value should be a range or a specific value according to the article with the unit 'MPGe' included. Eg. '79-81 MPGe', '17 MPGe', '0 MPGe' (if not applicable)."
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
                                    "type": "string"
                                },
                                "0to100": {
                                    "type": "string"
                                },
                                "0to130": {
                                    "type": "string"
                                },
                                "0to150": {
                                    "type": "string"
                                }
                            },
                            "required": ["0to60", "0to100", "0to130", "0to150"],
                            "additionalProperties": False,
                            "description": "Acceleration of the car. For each of them, the value should be a range or a specific value according to the article with the unit 's' excluded. Eg. '3.1-3.5', '7.3', 'N/A' (if not applicable)."
                        },
                        "quarterMile": {
                            "type": "object",
                            "properties": {
                                "time": {
                                    "type": "string"
                                },
                                "speed": {
                                    "type": "number"
                                }
                            },
                            "required": ["time", "speed"],
                            "additionalProperties": False,
                            "description": "Quarter mile of the car. The value for time should be a range or a specific value according to the article with the unit 's' excluded. Eg. '11.3', '15.1', 'N/A' (if not applicable). The value for speed should be a specific value according to the article with the unit 'mph' included. Eg. 112, 125, 0 (if not applicable)."
                        },
                        "topSpeed": {
                            "type": "number",
                            "description": "Top speed of the car. The value should be a specific value according to the article. Eg. 112, 125, 0 (if not applicable)."
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
                            "additionalProperties": False,
                            "description": "Cargo volume of the car. The value for behindFront and behindRear should be a range or a specific value according to the article without the unit. Eg. '71-74', '36', 'N/A' (if not applicable)."
                        },
                        "curbWeight": {
                            "type": "string",
                            "description": "Curb weight of the car. The value should be a range or a specific value according to the article with the unit 'lbs' included. Eg. '5700-6100 lbs', '5200 lbs', 'N/A' (if not applicable)."
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

class Price(BaseModel):
    base: int
    as_tested: float = Field(..., alias="asTested")


class Engine(BaseModel):
    type: str
    horsepower: int
    torque: int


class ElectricMotor(BaseModel):
    horsepower: int = Field(description="Horsepower of the electric motor. If the car is not electric, the value should be 0.")
    torque: int = Field(description="Torque of the electric motor. If the car is not electric, the value should be 0.")


class CombinedOutput(BaseModel):
    horsepower: int
    torque: int


class Powertrain(BaseModel):
    engine: Engine
    electric_motor: ElectricMotor = Field(None, alias="electricMotor")
    combined_output: CombinedOutput = Field(None, alias="combinedOutput")
    transmission: str = Field(description="Transmission of the car, such as 'direct-drive', 'CVT', 'single-speed', 'multi-speed', 'dual-motor', 'in-wheel motor', etc.")


class Battery(BaseModel):
    size: str = Field(description="Size of the battery in kWh. The unit is always kWh and should be included. Eg. '100.0 kWh', '107.5 kWh', '0 kWh' (for non-hybrid cars).")
    onboard_charger: str = Field(description="Onboard charger of the car. The unit is always kW and should be included. Eg. '11.0 kW', '10.5 kW', '0 kW' (for non-hybrid cars).", alias="onboardCharger")


class EPAFuelEfficiency(BaseModel):
    combined: str
    city: str
    highway: str


class FuelEfficiency(BaseModel):
    observed: str = Field(
        description="Observed fuel efficiency of the car. The value should be a range or a specific value according to the article with the unit 'MPGe' included. Eg. '79-81 MPGe', '17 MPGe', '0 MPGe' (if not applicable)."
    )
    epa: EPAFuelEfficiency = Field(description="Fuel efficiency of the car according to EPA of city, highway and combined. For each of them, the value should be a range or a specific value according to the article with the unit 'MPGe' included. Eg. '79-81 MPGe', '17 MPGe', '0 MPGe' (if not applicable).")


class Acceleration(BaseModel):
    zero_to_60: str = Field(..., alias="0to60")
    zero_to_100: str = Field(..., alias="0to100")
    zero_to_130: str = Field(..., alias="0to130")
    zero_to_150: str = Field(..., alias="0to150")


class QuarterMile(BaseModel):
    time: str
    speed: int


class Performance(BaseModel):
    acceleration: Acceleration = Field(description="Acceleration of the car. For each of them, the value should be a range or a specific value according to the article with the unit 's' excluded. Eg. '3.1-3.5', '7.3', 'N/A' (if not applicable).")
    quarter_mile: QuarterMile = Field(description="Quarter mile of the car. The value for time should be a range or a specific value according to the article with the unit 's' excluded. Eg. '11.3', '15.1', 'N/A' (if not applicable). The value for speed should be a specific value according to the article with the unit 'mph' included. Eg. 112, 125, 0 (if not applicable).", alias="quarterMile")
    top_speed: int = Field(
        description="Top speed of the car. The value should be a specific value according to the article. Eg. 112, 125, 0 (if not applicable).",
        alias="topSpeed"
    )


class PassengerVolume(BaseModel):
    front: float
    rear: float


class CargoVolume(BaseModel):
    behind_front: float = Field(..., alias="behindFront")
    behind_rear: float = Field(..., alias="behindRear")


class Dimensions(BaseModel):
    wheelbase: float
    length: float
    width: float
    height: float
    passenger_volume: PassengerVolume = Field(..., alias="passengerVolume")
    cargo_volume: CargoVolume = Field(description="Cargo volume of the car. The value for behindFront and behindRear should be a range or a specific value according to the article without the unit. Eg. '71-74', '36', 'N/A' (if not applicable).", alias="cargoVolume")
    curb_weight: str = Field(description="Curb weight of the car. The value should be a range or a specific value according to the article with the unit 'lbs' included. Eg. '5700-6100 lbs', '5200 lbs', 'N/A' (if not applicable).", alias="curbWeight")


class Brakes(BaseModel):
    front: str
    rear: str


class Tires(BaseModel):
    front: str
    rear: str


class Suspension(BaseModel):
    front: str
    rear: str


class SuspensionAndChassis(BaseModel):
    suspension: Suspension


class Car(BaseModel):
    make: str
    model: str
    year: int
    vehicle_type: str = Field(description="The type of vehicle, e.g. 'sedan', 'SUV', 'coupe', etc.", alias="vehicleType")
    price: Price
    powertrain: Powertrain
    battery: Battery = None
    fuel_efficiency: FuelEfficiency = Field(..., alias="fuelEfficiency")
    performance: Performance
    dimensions: Dimensions
    brakes: Brakes
    tires: Tires
    suspension_and_chassis: SuspensionAndChassis = Field(
        ..., alias="suspensionAndChassis"
    )
    strengths: List[str]
    weaknesses: List[str]
    overall_verdict: str = Field(..., alias="overallVerdict")

    class Config:
        allow_population_by_field_name = True
        alias_generator = lambda field_name: "".join(
            word.capitalize() if i > 0 else word
            for i, word in enumerate(field_name.split("_"))
        )

    def json(self, **kwargs):
        return json.loads(super().json(by_alias=True, exclude_none=True, **kwargs))