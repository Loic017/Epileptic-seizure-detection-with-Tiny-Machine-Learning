/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Models
#include "absence_model.h"
#include "tonic-clonic_model.h"
#include "general_model.h"

// Sample(s)
#include "sample.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace
{
  // Absence Model
  const tflite::Model *model_absence = nullptr;
  tflite::MicroInterpreter *interpreter_absence = nullptr;
  TfLiteTensor *input_absence = nullptr;
  TfLiteTensor *output_absence = nullptr;

  // Tonic-Clonic Model
  const tflite::Model *model_tonicclonic = nullptr;
  tflite::MicroInterpreter *interpreter_tonicclonic = nullptr;
  TfLiteTensor *input_tonicclonic = nullptr;
  TfLiteTensor *output_tonicclonic = nullptr;

  // General Model
  const tflite::Model *model_general = nullptr;
  tflite::MicroInterpreter *interpreter_general = nullptr;
  TfLiteTensor *input_general = nullptr;
  TfLiteTensor *output_general = nullptr;

  const int kTensorArenaSize_absence = 30000;
  uint8_t tensor_arena_absence[kTensorArenaSize_absence];

  const int kTensorArenaSize_tonicclonic = 30000;
  alignas(16) uint8_t tensor_arena_tonicclonic[kTensorArenaSize_tonicclonic];

  const int kTensorArenaSize_general = 30000;
  uint8_t tensor_arena_general[kTensorArenaSize_general];
} // namespace

// The name of this function is important for Arduino compatibility.
void setup()
{

  tflite::InitializeTarget();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model_absence = tflite::GetModel(absence_model);
  model_tonicclonic = tflite::GetModel(tonicclonic_model);
  model_general = tflite::GetModel(general_model);

  // This pulls in all the operation implementations we need.
  static tflite::AllOpsResolver resolver_absence;
  static tflite::AllOpsResolver resolver_tonicclonic;
  static tflite::AllOpsResolver resolver_general;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter_absence(
      model_absence, resolver_absence, tensor_arena_absence, kTensorArenaSize_absence);
  interpreter_absence = &static_interpreter_absence;

  static tflite::MicroInterpreter static_interpreter_tonicclonic(
      model_tonicclonic, resolver_tonicclonic, tensor_arena_tonicclonic, kTensorArenaSize_tonicclonic);
  interpreter_tonicclonic = &static_interpreter_tonicclonic;

  static tflite::MicroInterpreter static_interpreter_general(
      model_general, resolver_general, tensor_arena_general, kTensorArenaSize_general);
  interpreter_general = &static_interpreter_general;

  // Allocate memory from the tensor_arena for the model's tensors.
  interpreter_absence->AllocateTensors();
  interpreter_tonicclonic->AllocateTensors();
  interpreter_general->AllocateTensors();

  // Obtain pointers to the model's input and output tensors.
  input_absence = interpreter_absence->input(0);
  input_tonicclonic = interpreter_tonicclonic->input(0);
  input_general = interpreter_general->input(0);
}

// The name of this function is important for Arduino compatibility.
void loop()
{
  Serial.println("Start Loop _");
  Serial.println(" ");

  long int start = millis();

  // Copy the sample data to the input tensor
  for (int i = 0; i < 1 * 1 * 40 * 26; i++)
  {
    input_absence->data.uint8[i] = input_sample[i];
    input_tonicclonic->data.uint8[i] = input_sample[i];
    input_general->data.uint8[i] = input_sample[i];
  }

  // Serial.println(input_absence->type);

  // Run inference, and report any error
  TfLiteStatus invoke_status_absence = interpreter_absence->Invoke();
  TfLiteStatus invoke_status_tonicclonic = interpreter_tonicclonic->Invoke();
  TfLiteStatus invoke_status_general = interpreter_general->Invoke();

  // Collect the results
  TfLiteTensor *output_absence = interpreter_absence->output(0);
  uint8_t *output_absence_data = output_absence->data.uint8;
  TfLiteTensor *output_tonicclonic = interpreter_tonicclonic->output(0);
  uint8_t *output_tonicclonic_data = output_tonicclonic->data.uint8;
  TfLiteTensor *output_general = interpreter_general->output(0);
  uint8_t *output_general_data = output_general->data.uint8;

  long int end = millis();

  // Print the results
  float output_absence_seizure = exp((output_absence->data.uint8[1] - output_absence->params.zero_point) * output_absence->params.scale) /
                                 ((exp((output_absence->data.uint8[1] - output_absence->params.zero_point) * output_absence->params.scale) + exp((output_absence->data.uint8[0] - output_absence->params.zero_point) * output_absence->params.scale)));
  float output_tonicclonic_seizure = exp((output_tonicclonic->data.uint8[1] - output_tonicclonic->params.zero_point) * output_tonicclonic->params.scale) /
                                     ((exp((output_tonicclonic->data.uint8[1] - output_tonicclonic->params.zero_point) * output_tonicclonic->params.scale) + exp((output_tonicclonic->data.uint8[0] - output_tonicclonic->params.zero_point) * output_tonicclonic->params.scale)));
  float output_general_seizure = exp((output_general->data.uint8[1] - output_general->params.zero_point) * output_general->params.scale) /
                                 ((exp((output_general->data.uint8[1] - output_general->params.zero_point) * output_general->params.scale) + exp((output_general->data.uint8[0] - output_general->params.zero_point) * output_general->params.scale)));

  float absence_threshold_maybe = 0.5;
  float absence_threshold_likely = 0.7;
  Serial.print(output_absence_seizure);
  Serial.print(" // ");
  if (output_absence_seizure > absence_threshold_maybe and output_absence_seizure < absence_threshold_likely)
  {
    Serial.println("Absence Seizure: Maybe");
  }
  else if (output_absence_seizure > absence_threshold_likely)
  {
    Serial.println("Absence Seizure: Likely");
  }
  else
  {
    Serial.println("Absence Seizure: Unlikely");
  }

  float tonicclonic_threshold_maybe = 0.5;
  float tonicclonic_threshold_likely = 0.7;
  Serial.print(output_tonicclonic_seizure);
  Serial.print(" // ");
  if (output_tonicclonic_seizure > tonicclonic_threshold_maybe and output_tonicclonic_seizure < tonicclonic_threshold_likely)
  {
    Serial.println("Tonic-Clonic Seizure: Maybe");
  }
  else if (output_tonicclonic_seizure > tonicclonic_threshold_likely)
  {
    Serial.println("Tonic-Clonic Seizure: Likely");
  }
  else
  {
    Serial.println("Tonic-Clonic Seizure: Unlikely");
  }

  float general_threshold_maybe = 0.5;
  float general_threshold_likely = 0.7;
  Serial.print(output_general_seizure);
  Serial.print(" // ");
  if (output_general_seizure > general_threshold_maybe and output_general_seizure < general_threshold_likely)
  {
    Serial.println("General Seizure: Maybe");
  }
  else if (output_general_seizure > general_threshold_likely)
  {
    Serial.println("General Seizure: Likely");
  }
  else
  {
    Serial.println("General Seizure: Unlikely");
  }

  // Print the time taken
  Serial.print("Time taken: ");
  Serial.print(end - start);
  Serial.println(" milliseconds");

  Serial.println("___________________________________");
  Serial.println("");
  delay(4000);
}
