# digitalWrite()

[Language Reference](https://docs.arduino.cc/language-reference/) › Digital I/O

## Description

Write a `HIGH` or a `LOW` value to a digital pin.

If the pin has been configured as an `OUTPUT` with `pinMode()`, its voltage will be set to the corresponding value: 5V (or 3.3V on 3.3V boards) for `HIGH`, 0V (ground) for `LOW`.

If the pin is configured as an `INPUT`, `digitalWrite()` will enable (`HIGH`) or disable (`LOW`) the internal pull-up on the input pin. It is recommended to set `pinMode()` to `INPUT_PULLUP` to enable the internal pull-up resistor.

**NOTE:** If you do not set the `pinMode()` to `OUTPUT`, and connect an LED to a pin, when calling `digitalWrite(HIGH)`, the LED may appear dim. Without explicitly setting `pinMode()`, `digitalWrite()` will have enabled the internal pull-up resistor, which acts like a large current-limiting resistor.

## Syntax

```cpp
digitalWrite(pin, value)
```

## Parameters

- `pin`: the Arduino pin number.
- `value`: `HIGH` or `LOW`.

## Returns

Nothing

## Example Code

Sets pin 13 to `HIGH`, makes a one-second-long delay, and sets the pin back to `LOW`.

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

The PWM outputs generated on pins 3, 5, 6, 9, 10, and 11 by `analogWrite()` can be turned off by calling `digitalWrite(pin, LOW)`.
