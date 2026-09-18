# pinMode()

[Language Reference](https://docs.arduino.cc/language-reference/) › Digital I/O

## Description

Configures the specified pin to behave either as an input or an output. See the Digital Pins page for details on the functionality of the pins.

As of Arduino 1.0.1, it is possible to enable the internal pull-up resistors with the mode `INPUT_PULLUP`. Additionally, the `INPUT` mode explicitly disables the internal pull-ups.

## Syntax

```cpp
pinMode(pin, mode)
```

## Parameters

- `pin`: the Arduino pin number to set the mode of.
- `mode`: `INPUT`, `OUTPUT`, or `INPUT_PULLUP`. See the Digital Pins page for more information.

## Returns

Nothing

## Example Code

The code makes the digital pin 13 `OUTPUT` and toggles it `HIGH` and `LOW`:

```cpp
void setup() {
  pinMode(13, OUTPUT);    // sets the digital pin 13 as output
}

void loop() {
  digitalWrite(13, HIGH); // sets the digital pin 13 on
  delay(1000);            // waits for a second
  digitalWrite(13, LOW);  // sets the digital pin 13 off
  delay(1000);            // waits for a second
}
```

## Notes and Warnings

The `analogWrite()` function has nothing to do with `pinMode()` or `analogRead()`. You do not need to call `pinMode()` before calling `analogWrite()`.

Pins configured as `OUTPUT` with `pinMode()` are said to be in a low-impedance state. This means that they can provide a substantial amount of current to other circuits. Atmega pins can source (provide positive current) or sink (provide negative current) up to 40 mA (milliamps) of current to other devices/circuits. This is enough current to brightly light up an LED (don't forget the series resistor), or run many sensors, for example, but not enough current to run most relays, solenoids, or motors.

Short circuits on Arduino pins, or attempting to run high current devices from them, can damage or destroy the output transistors in the pin, or damage the entire Atmega chip. Often this will result in a "dead" pin in the microcontroller but the remaining chip will still function adequately. For this reason it is a good idea to connect `OUTPUT` pins to other devices with 470Ω or 1kΩ resistors, unless maximum current draw from the pins is required for a particular application.
