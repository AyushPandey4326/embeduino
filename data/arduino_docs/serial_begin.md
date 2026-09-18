# Serial.begin()

[Language Reference](https://docs.arduino.cc/language-reference/) › Communication › Serial

## Description

Sets the data rate in bits per second (baud) for serial data transmission. For communicating with Serial Monitor, make sure to use one of the baud rates listed in the menu at the bottom right corner of its screen. You can, however, specify other rates - for example, to communicate over pins 0 and 1 with a component that requires a particular baud rate.

An optional second argument configures the data, parity, and stop bits. The default is 8 data bits, no parity, one stop bit.

## Syntax

```cpp
Serial.begin(speed)
Serial.begin(speed, config)
```

## Parameters

- `speed`: the baud rate. Allowed data types: `long`. Common rates: 9600, 19200, 38400, 57600, 115200.
- `config`: sets data, parity, and stop bits. Valid values are listed below; default is `SERIAL_8N1`.

Valid `config` values include: `SERIAL_5N1`, `SERIAL_6N1`, `SERIAL_7N1`, `SERIAL_8N1` (default), `SERIAL_5N2`, `SERIAL_6N2`, `SERIAL_7N2`, `SERIAL_8N2`, `SERIAL_5E1`, `SERIAL_6E1`, `SERIAL_7E1`, `SERIAL_8E1`, `SERIAL_5E2`, `SERIAL_6E2`, `SERIAL_7E2`, `SERIAL_8E2`, `SERIAL_5O1`, `SERIAL_6O1`, `SERIAL_7O1`, `SERIAL_8O1`, `SERIAL_5O2`, `SERIAL_6O2`, `SERIAL_7O2`, `SERIAL_8O2`.

## Returns

Nothing

## Example Code

```cpp
void setup() {
  Serial.begin(9600); // opens serial port, sets data rate to 9600 bps
}

void loop() {}
```

Arduino Mega example with multiple ports:

```cpp
void setup() {
  Serial.begin(9600);
  Serial1.begin(9600);
  Serial2.begin(9600);
  Serial3.begin(9600);
}

void loop() {}
```

## Notes and Warnings

For USB CDC serial ports (e.g. native USB on Leonardo, Zero, Due), calling `Serial.begin()` is optional because USB CDC initializes regardless. For UART ports, `Serial.begin()` is required before use.
