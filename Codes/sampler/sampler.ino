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

//pins allocated to different channels
#define channel1 A0   //muscle name
#define channel2 A1   //muscle name
#define channel3 A2   //muscle name
#define channel4 A3   //muscle name
#define channel5 A4   //muscle name
#define channel6 A5   //muscle name

//variables to store all the raw analog data
uint16_t channel[7];

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

  //AVR wala code heroman ley lekhnu hunxa hai yedi chahiyo bhanes
}

void setup() {
  pinMode(channel1, INPUT);
  pinMode(channel2, INPUT);
  pinMode(channel3, INPUT);
  pinMode(channel4, INPUT);
  pinMode(channel5, INPUT);
  pinMode(channel6, INPUT);
  //  Timer1.initialize(samplePeriod);
  //  Timer1.attachInterrupt(readData);
  Serial.begin(115200);
}


void loop() {
  looptime = micros();
  //  readData();
  Serial.println(".");
  Serial.println(micros() - looptime);
}
