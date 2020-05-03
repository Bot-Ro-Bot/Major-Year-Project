/*
  Author : Rabin Nepal (rabin47nepal@gmail.com)
  Co-Authors: Rhimesh Lwagun, Sanjay Rijal, Upendra Subedi
  Date   : 2020/03/30
*/

/*
  This code intends to achieve a constant and simulataneous sampling of 6 channel EMG signal from signal conditioner
  at a sampling frequency of about 250Hz with the anti-aliasing filter tuned at 100 Hz (an inverse chebyshev of 1st order)
*/

/*NOTE: On ATmega based boards (UNO, Nano, Mini, Mega), it takes about 100 microseconds (0.0001 s) to read an analog input,
  so the maximum reading rate is about 10,000 times a second.
*/

#include<TimerOne.h>

//sampling period in microseconds (250 Hz)
#define samplePeriod 4000
#define smoothRate 0.97
#define normalization 1024

//pins allocated to different channels
#define channel1 A0   //muscle name
#define channel2 A1   //muscle name
#define channel3 A2   //muscle name
#define channel4 A3   //muscle name
#define channel5 A4   //muscle name
#define channel6 A5   //muscle name

//pin for led so verify that the code interrupt is running
#define LED 13

//variables to store all the raw analog data
uint16_t channel[7];  //channel 0 is the time stamp of the signal for phase correction and test signal for wireless communication for testing purposes
uint16_t channelOld[7];  //

//variable to see the time consumed by the adc to read all 6 channels
long int looptime = 0;

//function of read data of all channels
void readData(void) {
  channel[1] = analogRead(channel1);
  channel[2] = analogRead(channel2);
  channel[3] = analogRead(channel3);
  channel[4] = analogRead(channel4);
  channel[5] = analogRead(channel5);
  channel[6] = analogRead(channel6);
  digitalWrite(LED, 1 ^ digitalRead(LED));
  //AVR wala code heroman ley lekhnu hunxa hai yedi chahiyo bhane
}


//function to send data to computer via bluetooth (master device)
void sendData() {
  for (int i = 0; i < 7; i++) {
    Serial.write(channel[i]);
    Serial.write(','); //each value separated by comma 
  }  
  Serial.write('#'); //end character sent after each chunk of data sent
}

//normalize the data with respect to the maximum amplitude seen at a time frame
void normalizeData() {
  for (int i = 1; i < 7; i++) {
    channel[i] = channel[i] / normalization;
  }
}

//use exponential time average filter to smoothen the data
void smoothData() {
  for (int i = 1; i < 7; i++) {
    channel[i] = channel[i] * smoothRate + (1 - smoothRate) * channelOld[i];
    channelOld[i] = channel[i];
  }}

void setup() {
  pinMode(channel1, INPUT);
  pinMode(channel2, INPUT);
  pinMode(channel3, INPUT);
  pinMode(channel4, INPUT);
  pinMode(channel5, INPUT);
  pinMode(channel6, INPUT);
  pinMode(LED, OUTPUT);
  Timer1.initialize(samplePeriod);
  Timer1.attachInterrupt(readData);
  Serial.begin(115200);
}


void loop() {
  //looptime = micros();
  //  readData();
  //Serial.println(micros() - looptime);
  noInterrupts();
  //smoothData();
  sendData();
  interrupts();
}
