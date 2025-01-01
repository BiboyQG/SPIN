from pydantic import BaseModel, Field
from typing import List
import json


class Price(BaseModel):
    base: int
    as_tested: float = Field(..., alias="asTested")


class Engine(BaseModel):
    type: str = Field(
        description="Type of the engine of the car according to the article, e.g. 'permanent-magnet synchronous AC', 'turbocharged and intercooled DOHC 16-valve inline-4', 'twin-turbocharged V8', 'N/A' (for non-engine cars)."
    )
    horsepower: str = Field(
        description="Horsepower of the engine, which should be a specific value or two values according to the article with the unit 'hp' included. E.g. '402 or 510 hp', '295 hp', 'N/A' (if not applicable)."
    )
    torque: str = Field(
        description="Torque of the engine, which should be a specific value or two values according to the article with the unit 'lb-ft' included. E.g. '568 or 671 lb-ft', '295 lb-ft', 'N/A' (if not applicable)."
    )


class Powertrain(BaseModel):
    engine: Engine = Field(..., alias="engine")
    transmission: str = Field(
        description="Transmission of the car, such as 'direct-drive', 'CVT', 'single-speed', 'multi-speed', 'dual-motor', 'in-wheel motor', etc."
    )


class Battery(BaseModel):
    size: str = Field(
        description="Size of the battery in kWh. The unit is always kWh and should be included. Eg. '100.0 kWh', '107.5 kWh', '0 kWh' (for non-hybrid cars)."
    )
    onboard_charger: str = Field(
        description="Onboard charger of the car. The unit is always kW and should be included. Eg. '11.0 kW', '10.5 kW', '0 kW' (for non-hybrid cars).",
        alias="onboardCharger",
    )


class EPAFuelEfficiency(BaseModel):
    combined: str
    city: str
    highway: str


class FuelEfficiency(BaseModel):
    observed: str = Field(
        description="Observed fuel efficiency of the car. The value should be a range or a specific value according to the article with the unit 'MPGe' included. Eg. '79-81 MPGe', '17 MPGe', '0 MPGe' (if not applicable)."
    )
    epa: EPAFuelEfficiency = Field(
        description="Fuel efficiency of the car according to EPA of city, highway and combined. For each of them, the value should be a range or a specific value according to the article with the unit 'MPGe' included. Eg. '79-81 MPGe', '17 MPGe', '0 MPGe' (if not applicable)."
    )


class Acceleration(BaseModel):
    zero_to_60: str = Field(..., alias="0to60")
    zero_to_100: str = Field(..., alias="0to100")
    zero_to_130: str = Field(..., alias="0to130")
    zero_to_150: str = Field(..., alias="0to150")


class QuarterMile(BaseModel):
    time: str
    speed: int


class Performance(BaseModel):
    acceleration: Acceleration = Field(
        description="Acceleration of the car. For each of them, the value should be a range or a specific value according to the article with the unit 's' excluded. Eg. '3.1-3.5', '7.3', 'N/A' (if not applicable)."
    )
    quarter_mile: QuarterMile = Field(
        description="Quarter mile of the car. The value for time should be a range or a specific value according to the article with the unit 's' excluded. Eg. '11.3', '15.1', 'N/A' (if not applicable). The value for speed should be a specific value according to the article with the unit 'mph' included. Eg. 112, 125, 0 (if not applicable).",
        alias="quarterMile",
    )
    top_speed: int = Field(
        description="Top speed of the car. The value should be a specific value according to the article. Eg. 112, 125, 0 (if not applicable).",
        alias="topSpeed",
    )


class PassengerVolume(BaseModel):
    front: str
    rear: str


class CargoVolume(BaseModel):
    behind_front: str = Field(..., alias="behindFront")
    behind_rear: str = Field(..., alias="behindRear")


class Dimensions(BaseModel):
    wheelbase: float
    length: float
    width: float
    height: float
    passenger_volume: PassengerVolume = Field(
        description="Passenger volume of the car. The value for front and rear should be a range or a specific value according to the article without the unit. Eg. '53-55', '51', 'N/A' (if not applicable).",
        alias="passengerVolume",
    )
    cargo_volume: CargoVolume = Field(
        description="Cargo volume of the car. The value for behindFront and behindRear should be a range or a specific value according to the article without the unit. Eg. '71-74', '36', 'N/A' (if not applicable).",
        alias="cargoVolume",
    )
    curb_weight: str = Field(
        description="Curb weight of the car. The value should be a range or a specific value according to the article with the unit 'lbs' included. Eg. '5700-6100 lbs', '5200 lbs', 'N/A' (if not applicable).",
        alias="curbWeight",
    )


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
    vehicle_type: str = Field(
        description="The type of vehicle, e.g. 'sedan', 'SUV', 'coupe', etc.",
        alias="vehicleType",
    )
    price: Price
    powertrain: Powertrain
    battery: Battery
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
