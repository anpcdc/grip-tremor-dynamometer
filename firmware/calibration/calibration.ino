#include <HX711_ADC.h>

// Pins
const int HX711_dout = 10;
const int HX711_sck  = 11;

// HX711 object
HX711_ADC LoadCell(HX711_dout, HX711_sck);

unsigned long t = 0;

void setup() {
  Serial.begin(115200);
  delay(10);
  Serial.println();
  Serial.println("HX711 Load Cell Calibration");
  Serial.println("Using Nano 33 BLE Sense Rev2");

  LoadCell.begin();
  // LoadCell.setReverseOutput(); // uncomment if readings are negative
  unsigned long stabilizingTime = 2000; // 2 sec
  boolean doTare = true;
  LoadCell.start(stabilizingTime, doTare);

  if (LoadCell.getTareTimeoutFlag() || LoadCell.getSignalTimeoutFlag()) {
    Serial.println("Error: check HX711 wiring and pins!");
    while (1);
  }

  // Initial calibration factor (will be overwritten)
  LoadCell.setCalFactor(1.0);

  // Wait for first stable reading
  while (!LoadCell.update());
  
  // Start calibration
  calibrate();
}

void loop() {
  static boolean newDataReady = 0;
  const int serialPrintInterval = 5; // ms

  if (LoadCell.update()) newDataReady = true;

  if (newDataReady && millis() > t + serialPrintInterval) {
    float weight = LoadCell.getData();
    Serial.print("Load cell reading: ");
    Serial.print(weight);
    Serial.println(" g");
    Serial.print("______Time: ");
    Serial.print(millis());
    Serial.print(" s");
    newDataReady = 0;
    t = millis();
  }

  // Commands from Serial Monitor
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 't') LoadCell.tareNoDelay();        // tare
    else if (cmd == 'r') calibrate();             // recalibrate
  }
}

void calibrate() {
  Serial.println("\n*** CALIBRATION START ***");
  Serial.println("Step 1: Place the load cell on a level stable surface with nothing on it.");
  Serial.println("Send 't' from Serial Monitor to tare the scale.");

  boolean tareDone = false;
  while (!tareDone) {
    LoadCell.update();
    if (Serial.available() > 0) {
      char inByte = Serial.read();
      if (inByte == 't') LoadCell.tareNoDelay();
    }
    if (LoadCell.getTareStatus()) {
      Serial.println("Tare complete.");
      tareDone = true;
    }
  }

  Serial.println("\nStep 2: Place a known weight on the load cell.");
  Serial.println("Enter the weight in grams via Serial Monitor (e.g., 14000 for 14 kg):");

  float knownWeight = 0;
  boolean weightEntered = false;
  while (!weightEntered) {
    LoadCell.update();
    if (Serial.available() > 0) {
      knownWeight = Serial.parseFloat();
      if (knownWeight > 0) {
        Serial.print("Known weight entered: ");
        Serial.print(knownWeight);
        Serial.println(" g");
        weightEntered = true;
      }
    }
  }

  LoadCell.refreshDataSet(); // ensure dataset is updated
  float newCalFactor = LoadCell.getNewCalibration(knownWeight);

  Serial.print("\nCalibration complete!\n");
  Serial.print("Use this calibration factor in your final code:\n");
  Serial.print("LoadCell.setCalFactor(");
  Serial.print(newCalFactor, 4);
  Serial.println(");");
  Serial.println("*** END CALIBRATION ***\n");
}