#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "mbed.h"
#include "mbed_rpc.h"
#include "stm32l475e_iot01_accelero.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <cmath>

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

InterruptIn button(USER_BUTTON);
BufferedSerial pc(USBTX, USBRX);

void gesture_RPC(Arguments *in, Reply *out);
void angle_detect_RPC(Arguments *in, Reply *out);
void gesture_UI();
void angle_detect();
int PredictGesture(float* output);
void setting_mode();

#define pi 3.1415926

RPCFunction rpc1(&gesture_RPC, "1");							// /gesture_RPC/run
RPCFunction rpc2(&angle_detect_RPC, "2");		// /angle_detect_RPC/run
DigitalOut myled1(LED1);	// gesture_UI mode
DigitalOut myled2(LED2);	// angle_detect mode
DigitalOut myled3(LED3);	// initialize

Thread t_angle_detect;
Thread t_gesture_UI;

bool setting = false;
int idR[32] = {0};
int indexR = 0;
int Angle = 30;

void setting_mode(){
	setting = true;
}

void gesture_RPC(Arguments *in, Reply *out){
	myled1 = 1;
	t_gesture_UI.start(&gesture_UI);
}

void gesture_UI() {
  // Whether we should clear the buffer next time we fetch data
  bool should_clear_buffer = false;
  bool got_data = false;

  // The gesture index of the prediction
  int gesture_index;

  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return ;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE(), 1);

  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return ;
  }

  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    return ;
  }

  error_reporter->Report("Set up successful...\n");

  while (true) {

    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);

    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }

    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);

    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;

    // Produce an output
    if (gesture_index < label_num) {
			if (Angle == 180)
				Angle = 30;
			else 
				Angle += 5;
			printf("Angle-threshold: %d degree\n", Angle);
			button.rise(&setting_mode);
			if (setting == true) break;
      //error_reporter->Report(config.output_message[gesture_index]);
    }
  }
}

void angle_detect_RPC(Arguments *in, Reply *out) {
	myled2 = 1;
	t_angle_detect.start(&angle_detect);
}

void angle_detect() {
	double a = 0, b = 0, c = 0;
	double theta = 0;
  double cos = 0;
	myled3 = 1;
	int16_t ref_pDataXYZ[3] = {0};
	int16_t pDataXYZ[3] = {0};
	ThisThread::sleep_for(3s);
	BSP_ACCELERO_AccGetXYZ(ref_pDataXYZ);
	ThisThread::sleep_for(1s);
	myled3 = 0;
	while (1) {
		BSP_ACCELERO_AccGetXYZ(pDataXYZ);
		a = pow(ref_pDataXYZ[0],2) + pow(ref_pDataXYZ[1],2) + pow(ref_pDataXYZ[2],2);
		b = pow(pDataXYZ[0],2) + pow(pDataXYZ[1],2) + pow(pDataXYZ[2],2);
		c = pow((pDataXYZ[0]-ref_pDataXYZ[0]),2) + pow((pDataXYZ[1]-ref_pDataXYZ[1]),2) + pow((pDataXYZ[2]-ref_pDataXYZ[2]),2);
		cos = (a*a+b*b-c*c) / (2*a*b);
    theta = acos((pow(a,2) + pow(b,2) - pow(c,2)) / (2 * a * b));
		printf("cos = %f, a = %f, b = %f, c = %f\n", cos, a, b, c);
		printf("Angle: %f\n", theta*180/pi);
		printf("Accelerometer values: (%d, %d, %d)\n", pDataXYZ[0], pDataXYZ[1], pDataXYZ[2]);
		ThisThread::sleep_for(500ms);
	}
	
}

int main() {
	BSP_ACCELERO_Init();
	char buf[256], outbuf[256];

	FILE *devin = fdopen(&pc, "r");
	FILE *devout = fdopen(&pc, "w");

	while (1) {
		memset(buf, 0, 256);      // clear buffer
		for(int i=0; i<255; i++) {
			char recv = fgetc(devin);
			if (recv == '\r' || recv == '\n') {
				printf("\r\n");
				break;
			}
			buf[i] = fputc(recv, devout);
		}
		RPC::call(buf, outbuf);
		printf("%s\r\n", outbuf);
	}
}
// Return the result of the last prediction
int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}

