#include <SoftwareSerial.h>
int enA = 10;
int in1 = 9;
int in2 = 8;
// motor two
int enB = 5;
int in3 = 7;
int in4 = 6;
int Rx = 12;
int Tx = 13;
SoftwareSerial blueToothSerial(Rx, Tx);


void setup(void){
  Serial.begin(9600);
  blueToothSerial.begin(9600);
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
}

void backward()
{
  // turn on motor A
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  analogWrite(enA, 255);
  // turn on motor B
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  // set speed to 200 out of possible range 0~255
  analogWrite(enB, 255);
}

void forward()
{
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  analogWrite(enA, 255);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
  analogWrite(enB, 255);
}

void leftward()
{
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  analogWrite(enA, 175);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
  analogWrite(enB, 255);
}

void rightward()
{
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  analogWrite(enA, 255);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
  analogWrite(enB, 175);
}


void loop()
{
  char rev;
  if(blueToothSerial.available())
  {   
      rev = blueToothSerial.read();
      Serial.println(rev);
      blueToothSerial.write(rev);
      if(rev =='b')
      {
        backward();
        delay(1000);
      }
       if(rev =='f')
      {
        forward();
        delay(1000);
      }
      if(rev =='l')
      {
        leftward();
        delay(1000);
      }
       if(rev =='r')
      {
        rightward();
        delay(1000);
      }
  }
}

