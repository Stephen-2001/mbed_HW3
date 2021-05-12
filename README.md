## mbed_HW3

### How to set up our program

### What are the results
##### Compile commend
  `sudo mbed compile --source . --source ~/ee2405new/mbed-os-build2/ -m B_L4S5I_IOT01A -t GCC_ARM --profile tflite.json -f`
##### After compile, open the screen 
  `sudo screen /dev/ttyACM0`
##### When we type /1/run on the screen, RPC call the gesture mode and LED1 will turn on

###### When the mbed sense the gesture, the angle-threshold will +5 degree. If it arrives 180 degree, it will turn back to 30 degree.

##### Press the userbutton to determine the threshold angle and send the event to broker
