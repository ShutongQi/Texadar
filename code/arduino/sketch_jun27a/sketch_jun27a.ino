void setup() {
  // initialize the serial communication:
  Serial.begin(500000);
}

void loop() {
  // send the value of analog input 0:
  int inWord = analogRead(A0);
  
  Serial.write(0xFF);
  Serial.write(highByte(inWord));
  Serial.write(lowByte(inWord));

//  Serial.println(analogRead(A0));

  //delay(2);
}
