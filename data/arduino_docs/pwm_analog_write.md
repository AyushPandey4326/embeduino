# analogWrite()

[Language Reference](https://docs.arduino.cc/language-reference/) › Analog I/O

## Description

Writes an analog value (PWM wave) to a pin. Can be used to light an LED at varying brightnesses or drive a motor at various speeds. After a call to `analogWrite()`, the pin will generate a steady rectangular wave of the specified duty cycle until the next call to `analogWrite()` (or a call to `digitalWrite()` or `pinMode()`) on the same pin.

## Syntax

```cpp
analogWrite(pin, value)
```

## Parameters

- `pin`: the Arduino pin to write to. Allowed data types: `int`.
- `value`: the duty cycle: between 0 (always off) and 255 (always on). Allowed data types: `int`.

## Returns

Nothing

## Example Code

Sets the output to the LED proportional to the value read from the potentiometer.

```cpp
int ledPin = 9;      // LED connected to digital pin 9
int analogPin = A3;  // potentiometer connected to analog pin A3
int val = 0;         // variable to store the read value

void setup() {
  pinMode(ledPin, OUTPUT);  // sets the pin as output
}

void loop() {
  val = analogRead(analogPin);        // read the input pin
  analogWrite(ledPin, val / 4);       // analogRead 0-1023; analogWrite 0-255
}
```

## Notes and Warnings

- On most Arduino boards (those with the ATmega168 or ATmega328), this function works on pins 3, 5, 6, 9, 10, and 11. On the Arduino Mega, it works on pins 2–13 and 44–46. Older Arduino boards with an ATmega8 only support `analogWrite()` on pins 9, 10, and 11.
- The PWM frequency is approximately 490 Hz on most pins; pins 5 and 6 are about 980 Hz on Uno.
- You do not need to call `pinMode()` before calling `analogWrite()`. The `analogWrite` function has nothing to do with the `analogRead` function or the `analogReference` function.
