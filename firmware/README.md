# Firmware

Board: Arduino Nano 33 BLE Sense Rev2 (BMI270 + BMM150 IMU). Serial: 115200 baud.
The Arduino IDE requires each sketch to sit in a folder with the same name as its `.ino` file.

| Sketch | Purpose | Libraries |
|---|---|---|
| `acquisition/acquisition.ino` | Streams `time,force,ax,ay,az` over serial; shows force on the LCD | `HX711_ADC`, `Arduino_BMI270_BMM150`, `Wire`, `hd44780` |
| `calibration/calibration.ino` | Load-cell calibration (run this first) | `HX711_ADC` |

HX711 pins: DOUT = 10, SCK = 11.
