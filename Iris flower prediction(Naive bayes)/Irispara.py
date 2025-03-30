# provide user friendly error
# for POST request
from pydantic import BaseModel

#class which describes Iris flowe prediction measurements
class Irispara(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float