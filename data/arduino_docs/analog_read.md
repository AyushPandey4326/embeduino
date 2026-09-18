# analogRead()

[Language Reference](https://docs.arduino.cc/language-reference/) › Analog I/O

## Description

Reads the value from the specified analog pin. Arduino boards contain a multi-channel, 10-bit analog to digital converter (ADC). This means that it will map input voltages between 0 and the operating voltage (5V or 3.3V) into integer values between 0 and 1023. On an Arduino UNO, for example, this yields a resolution between readings of 5 volts / 1024 units or 0.0049 volts (4.9 mV) per unit.

The input range and resolution can be changed using `analogReference()`.

It takes about 100 microseconds (0.0001 s) to read an analog input, so the maximum reading rate is about 10,000 times a second.

## Syntax

```cpp
analogRead(pin)
```

## Parameters

- `pin`: the name of the analog input pin to read from (A0 to A5 on most boards, A0 to A7 on the Mini and Nano, A0 to A15 on the Mega).

## Returns

The analog reading on the pin. Although it is limited to the resolution of the analog to digital converter (0-1023 for 10 bits or 0-4095 for 12 bits). Data type: `int`.

## Example Code

```cpp
int analogPin = A3;   // potentiometer wiper (middle terminal) connected to analog pin A3
                      // outside leads to ground and +5V
int val = 0;          // variable to store the value read

void setup() {
  Serial.begin(9600); // setup serial
}

void loop() {
  val = analogRead(analogPin);  // read the input pin
  Serial.println(val);          // debug value
}
```

## Notes and Warnings

If the analog input pin is not connected to anything, the value returned by `analogRead()` will fluctuate based on a number of factors (e.g. the values of the other analog inputs, how close your hand is to the board, etc.).
