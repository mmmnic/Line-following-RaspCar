import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# define para
motorFrequency = 150
servoFrequency = 50
servoCenterDutyCycle = 8.5
servoMaxDutyCycle = 11
servoMaxAngle = 45
servoRatio = (servoMaxDutyCycle - servoCenterDutyCycle)/servoMaxAngle
# define pin
SDA = 2
SCL = 3
BUT1 = 4
BIN1 = 5
BIN2 = 6
LED1 = 9
STBY = 10
LED2 = 11
PWMB = 13
PWMA = 17
EXPIN1 = 16  # haven't config
BUT2 = 19
EXPIN2 = 20  # haven't config
S = 21
AIN1 = 22
BZ = 26
AIN2 = 27

# Setup pin
GPIO.setup(SDA, GPIO.OUT)
GPIO.setup(SCL, GPIO.OUT)
GPIO.setup(BIN1, GPIO.OUT)
GPIO.setup(BIN2, GPIO.OUT)
GPIO.setup(LED1, GPIO.OUT)
GPIO.setup(STBY, GPIO.OUT)
GPIO.setup(LED2, GPIO.OUT)
GPIO.setup(PWMB, GPIO.OUT)
GPIO.setup(PWMA, GPIO.OUT)
GPIO.setup(S, GPIO.OUT)
GPIO.setup(AIN1, GPIO.OUT)
GPIO.setup(BZ, GPIO.OUT)
GPIO.setup(AIN2, GPIO.OUT)
GPIO.setup(BUT1, GPIO.IN)
GPIO.setup(BUT2, GPIO.IN)

# Setup for motor and servo
motor1 = GPIO.PWM(PWMA, motorFrequency)
motor2 = GPIO.PWM(PWMB, motorFrequency)
servo  = GPIO.PWM(S, servoFrequency)
motor1.start(0)
motor2.start(0)
servo.start(servoCenterDutyCycle)
GPIO.output(STBY, 1)

def setSpeed(speedA, speedB):
    # Control derection
    if (speedA >=0):
        GPIO.output(AIN1, 1)
        GPIO.output(AIN2, 0)
    else:
        speedA = -speedA
        GPIO.output(AIN1, 0)
        GPIO.output(AIN2, 1)
    if (speedB >=0):
        GPIO.output(BIN1, 1)
        GPIO.output(BIN2, 0)
    else:
        speedB = -speedB
        GPIO.output(BIN1, 0)
        GPIO.output(BIN2, 1)
    # Change duty cycle for changing speed
    motor1.ChangeDutyCycle(speedA)
    motor2.ChangeDutyCycle(speedB)
    return;

def turnServo(degree):
    # Convert degree into duty cycle for turning
    turnDutyCycle = servoCenterDutyCycle + degree*servoRatio
    servo.ChangeDutyCycle(turnDutyCycle)
    return;

def turnLED1(status):
    if (status == 1):
        GPIO.output(LED1, 0)
    else:
        GPIO.output(LED1, 1)
    return;

def turnLED2(status):
    if (status == 1):
        GPIO.output(LED2, 0)
    else:
        GPIO.output(LED2, 1)
    return;

def turnBuzzer(status):
    if (status == 1):
        GPIO.output(BZ, 0)
    else:
        GPIO.output(BZ, 1)
    return;

def isBUT1():
    if (GPIO.input(BUT1)):
        return 0
    return 1
    
def isBUT2():
    if (GPIO.input(BUT2)):
        return 0
    return 1
