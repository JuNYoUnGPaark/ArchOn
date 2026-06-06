const int emgPin = A0;

void setup() {
  Serial.begin(115200);
}

void loop() {
  int rawValue = analogRead(emgPin);
  Serial.println(rawValue);
  delay(50);
}
