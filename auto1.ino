#define BLYNK_TEMPLATE_ID "TMPL3Cp1vNmJ-"
#define BLYNK_TEMPLATE_NAME "Soil Moisture"
#define BLYNK_AUTH_TOKEN "aFU-jlExINjrg4hzsvO2y-CPWCjHUr4D"

#include <ESP8266WiFi.h>
#include <BlynkSimpleEsp8266.h>
#include <DHT.h>

char auth[] = BLYNK_AUTH_TOKEN;
char ssid[] = "Killer";
char pass[] = "abhi7733";

#define DHTPIN 4
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

#define MOISTURE_SENSOR A0

int moistureThreshold = 700; // adjust if needed
bool manualPumpControl = false;
bool pumpState = false;

BlynkTimer timer;

// V4 = Manual switch
BLYNK_WRITE(V4) {
  int value = param.asInt();
  manualPumpControl = (value == 1);
  Serial.write(manualPumpControl ? '1' : '0');  // Send signal to Arduino (1 = ON, 0 = OFF)
  pumpState = manualPumpControl;

  Serial.print("Manual Control - Pump ");
  Serial.println(manualPumpControl ? "ON" : "OFF");
}

void autoWatering() {
  int moisture = analogRead(MOISTURE_SENSOR);
  float humidity = dht.readHumidity();
  float temp = dht.readTemperature();

  Serial.print("Moisture: ");
  Serial.println(moisture);
  Serial.print("Temp: ");
  Serial.println(temp);
  Serial.print("Humidity: ");
  Serial.println(humidity);

  if (!isnan(temp) && !isnan(humidity)) {
    Blynk.virtualWrite(V0, moisture);
    Blynk.virtualWrite(V2, temp);
    Blynk.virtualWrite(V3, humidity);
  }

  if (!manualPumpControl) {
    if (moisture > moistureThreshold) {
      Serial.write('1');  // Send signal to Arduino (1 = ON)
      if (!pumpState) {
        pumpState = true;
        Blynk.virtualWrite(V4, 1);
        Serial.println("Auto Mode - Pump ON");
      }
    } else {
      Serial.write('0');  // Send signal to Arduino (0 = OFF)
      if (pumpState) {
        pumpState = false;
        Blynk.virtualWrite(V4, 0);
        Serial.println("Auto Mode - Pump OFF");
      }
    }
  }
}

void setup() {
  Serial.begin(115200);  // Ensure the baud rate matches the Arduino's
  Blynk.begin(auth, ssid, pass);
  dht.begin();

  timer.setInterval(3000L, autoWatering);
}

void loop() {
  Blynk.run();
  timer.run();
}

