# millis()

[Language Reference](https://docs.arduino.cc/language-reference/) › Time

## Description

Returns the number of milliseconds passed since the Arduino board began running the current program. This number will overflow (go back to zero), after approximately 50 days.

## Syntax

```cpp
time = millis()
```

## Parameters

None

## Returns

Number of milliseconds since the program started (`unsigned long`).

## Example Code

This example code prints on the serial port the number of milliseconds passed since the Arduino board started running the code itself.

```cpp
unsigned long time;

void setup() {
  Serial.begin(9600);
}

void loop() {
  Serial.print("Time: ");
  time = millis();
  Serial.println(time); // prints time since program started
  delay(1000);          // wait a second so as not to send massive amounts of data
}
```

## Notes and Warnings

- Please note that the return value for `millis()` is an `unsigned long`. Logic errors are easy if you try to do arithmetic with other data types such as `int`.
- Always use `unsigned long` for time comparisons, e.g. `if (millis() - previousMillis >= interval)`.
- Do not use `delay()` when you need non-blocking timing; prefer `millis()`-based intervals.
