#include "Arduino_BMI270_BMM150.h"
#include <HX711_ADC.h>
#include <Wire.h>
#include <hd44780.h>
#include <hd44780ioClass/hd44780_I2Cexp.h>

// ----- HX711_ADC -----
const int HX711_dout = 10;
const int HX711_sck  = 11;
HX711_ADC LoadCell(HX711_dout, HX711_sck);

// ----- LCD I2C -----
hd44780_I2Cexp lcd;
const int LCD_COLS = 16;
const int LCD_ROWS = 2;

// ----- Sampling -----
int fs = 104;  // RECOMENDADO (melhor para HX711)
int period = 0;
unsigned long currentTime = 0, previousTime = 0;

// ----- LCD timing -----
unsigned long lcdTimer = 0;
int lcdPeriod = 500; // 2 Hz

// ----- IMU -----
float x = 0, y = 0, z = 0;

// ----- Force -----
float force = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // Sampling period
  period = int(1000.0 / fs);

  // ----- IMU -----
  if (!IMU.begin()) {
    Serial.println("IMU failed");
    while (1);
  }

  // ----- LCD -----
  int status = lcd.begin(LCD_COLS, LCD_ROWS);
  if (status) {
    Serial.print("LCD failed, status=");
    Serial.println(status);
    while (1);
  }

  lcd.setCursor(0, 0);
  lcd.print("Hand Grip Test");

  // ----- HX711 -----
  LoadCell.begin();
  LoadCell.start(2000); // estabilização
  LoadCell.setCalFactor(47.2132); // ajustar com calibração real

  // Header para Python
  Serial.println("time,force,ax,ay,az");
}

void loop() {

  currentTime = millis();

  // ----- HX711 update contínuo -----
  if (LoadCell.update()) {
    force = LoadCell.getData();
  }

  // ----- Aquisição sincronizada -----
  if (currentTime - previousTime >= period) {
    previousTime = currentTime;

    // IMU
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(x, y, z);
    }

    // SERIAL OUTPUT
    Serial.print(currentTime);
    Serial.print(",");
    Serial.print(force, 3);
    Serial.print(",");
    Serial.print(x, 4);
    Serial.print(",");
    Serial.print(y, 4);
    Serial.print(",");
    Serial.println(z, 4);
  }

  // ----- LCD update (mais lento) -----
  if (currentTime - lcdTimer >= lcdPeriod) {
    lcdTimer = currentTime;

    // limpar linha
    lcd.setCursor(0, 1);
    lcd.print("                ");

    // escrever novo valor
    lcd.setCursor(0, 1);
    lcd.print("W:");
    lcd.print(force / 1000.0, 2);
    lcd.print("kg");
  }
}