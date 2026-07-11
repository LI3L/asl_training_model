// ASL_Car_Controller.ino
// Receives ASL letter commands from the ESP32-S3 camera and drives the car.
//
// Communication options:
//   Option 1 (Wire):  ESP32 Serial1 TX (GPIO2) → Mega RX2 (pin 17)
//   Option 2 (USB):   Python bridge reads ESP32 USB log → sends letter to Mega USB
//
// Letter → Action mapping:
//   F = Forward    B = Backward    L = Left    R = Right    S = Stop
//
// The car auto-stops after 3 seconds of no valid command (safety timeout).

#include <Servo.h>

// ===== Motor 1 encoder =====
const int enc1A = 2;   // Interrupt pin
const int enc1B = 3;
volatile long encoderCount1 = 0;

// ===== Motor 2 encoder =====
const int enc2A = 18;  // Interrupt pin
const int enc2B = 19;
volatile long encoderCount2 = 0;

// ===== L298N motor driver =====
const int ENA = 5;
const int IN1 = 6;
const int IN2 = 7;

const int ENB = 12;
const int IN3 = 11;
const int IN4 = 10;

// ===== Servo steering =====
Servo steeringServo;
const int servoPin = 9;

// --- Steering angles ---
const int angleStraight = 50;
const int angleRight    = 30;
const int angleLeft     = 70;

// --- Driving parameters ---
const int DRIVE_SPEED   = 200;  // PWM speed for forward/backward (0-255)
const int TURN_SPEED    = 150;  // PWM speed while turning

// --- Timeout ---
// If no valid command is received for this many milliseconds, stop the car.
const unsigned long TIMEOUT_MS = 3000;

// --- State tracking ---
char currentCommand  = 'S';             // Current active command (S = stopped)
unsigned long lastCommandTime = 0;      // millis() when last valid command arrived
int currentSteeringAngle = angleStraight;

// Target steering angle for smooth transitions
int targetSteeringAngle = angleStraight;

void setup() {
  // USB Serial for debug output (and for receiving commands via USB bridge)
  Serial.begin(9600);

  // Serial2 (pins 16 TX2, 17 RX2) for receiving commands from ESP32 wire
  Serial2.begin(9600);

  // Servo — start straight immediately
  steeringServo.attach(servoPin);
  steeringServo.write(angleStraight);

  // Motor driver pins
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // Encoder pins
  pinMode(enc1A, INPUT_PULLUP);
  pinMode(enc1B, INPUT_PULLUP);
  pinMode(enc2A, INPUT_PULLUP);
  pinMode(enc2B, INPUT_PULLUP);

  // Encoder interrupts
  attachInterrupt(digitalPinToInterrupt(enc1A), encoder1ISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(enc2A), encoder2ISR, CHANGE);

  // Stop motors at startup
  stopMotors();

  Serial.println("ASL Car Controller Ready!");
  Serial.println("Commands: F=Forward B=Back L=Left R=Right S=Stop");
  Serial.println("Waiting for commands from ESP32...");

  lastCommandTime = millis();
}

void loop() {
  // --- 1. Read commands from both serial sources ---
  char cmd = 0;

  // Check Serial2 first (wire from ESP32)
  if (Serial2.available()) {
    String line = Serial2.readStringUntil('\n');
    line.trim();
    if (line.length() == 1) {
      cmd = line.charAt(0);
    }
  }

  // Also check USB Serial (from Python bridge — option 2)
  if (cmd == 0 && Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 1) {
      cmd = line.charAt(0);
    }
  }

  // --- 2. Process the received command ---
  if (cmd != 0) {
    // Only accept valid command letters
    if (cmd == 'F' || cmd == 'B' || cmd == 'L' || cmd == 'R' || cmd == 'S') {
      if (cmd != currentCommand) {
        Serial.print(">> New command: ");
        Serial.println(cmd);
      }
      currentCommand = cmd;
      lastCommandTime = millis();
    }
    // Ignore unrecognized letters silently
  }

  // --- 3. Check for timeout (3 seconds of no commands → auto-stop) ---
  if (currentCommand != 'S' && (millis() - lastCommandTime > TIMEOUT_MS)) {
    Serial.println(">> TIMEOUT: No command for 3s — stopping.");
    currentCommand = 'S';
  }

  // --- 4. Execute the current command ---
  switch (currentCommand) {
    case 'F':
      driveForwardSynced(DRIVE_SPEED);
      targetSteeringAngle = angleStraight;
      break;

    case 'B':
      driveBackward(DRIVE_SPEED);
      targetSteeringAngle = angleStraight;
      break;

    case 'L':
      driveForwardSynced(TURN_SPEED);
      targetSteeringAngle = angleLeft;
      break;

    case 'R':
      driveForwardSynced(TURN_SPEED);
      targetSteeringAngle = angleRight;
      break;

    case 'S':
    default:
      stopMotors();
      targetSteeringAngle = angleStraight;
      break;
  }

  // --- 5. Smoothly transition the steering angle ---
  smoothSteer();

  delay(20);  // ~50 Hz control loop
}

// ===== Driving functions =====

// Forward with encoder-based P-controller synchronization (straight tracking)
void driveForwardSynced(int targetSpeed) {
  long count1 = abs(encoderCount1);
  long count2 = abs(encoderCount2);
  long error = count1 - count2;

  float Kp = 0.5;
  int adjustment = error * Kp;

  int speed1 = constrain(targetSpeed - adjustment, 80, 255);
  int speed2 = constrain(targetSpeed + adjustment, 80, 255);

  forwardMotor1(speed1);
  forwardMotor2(speed2);
}

void driveBackward(int speed) {
  backwardMotor1(speed);
  backwardMotor2(speed);
}

void stopMotors() {
  stopMotor1();
  stopMotor2();
  // Reset encoder counts when stopping so sync starts fresh on next move
  encoderCount1 = 0;
  encoderCount2 = 0;
}

// Gradually move steering towards the target angle (smooth transitions)
void smoothSteer() {
  if (currentSteeringAngle != targetSteeringAngle) {
    // Move 2 degrees per loop iteration (2° × 50 Hz = 100°/s max steering rate)
    if (currentSteeringAngle < targetSteeringAngle) {
      currentSteeringAngle = min(currentSteeringAngle + 2, targetSteeringAngle);
    } else {
      currentSteeringAngle = max(currentSteeringAngle - 2, targetSteeringAngle);
    }
    steeringServo.write(currentSteeringAngle);
  }
}

// ===== Basic motor control =====

void forwardMotor1(int speed) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  analogWrite(ENA, speed);
}

void backwardMotor1(int speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, speed);
}

void stopMotor1() {
  analogWrite(ENA, 0);
}

void forwardMotor2(int speed) {
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  analogWrite(ENB, speed);
}

void backwardMotor2(int speed) {
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, speed);
}

void stopMotor2() {
  analogWrite(ENB, 0);
}

// ===== Encoder interrupts =====

void encoder1ISR() {
  int stateA = digitalRead(enc1A);
  int stateB = digitalRead(enc1B);
  if (stateA == stateB) encoderCount1++;
  else encoderCount1--;
}

void encoder2ISR() {
  int stateA = digitalRead(enc2A);
  int stateB = digitalRead(enc2B);
  if (stateA == stateB) encoderCount2++;
  else encoderCount2--;
}
