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
#include "combined_model.h"

// Sample(s)
#include "sample.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace
{
  // Combined Model
  const tflite::Model *model_combined = nullptr;
  tflite::MicroInterpreter *interpreter_combined = nullptr;
  TfLiteTensor *input_combined = nullptr;
  TfLiteTensor *output_combined = nullptr;

  constexpr int kTensorArenaSize_combined = 70000;
  alignas(16) uint8_t tensor_arena_combined[kTensorArenaSize_combined];

} // namespace

// The name of this function is important for Arduino compatibility.
void setup()
{

  tflite::InitializeTarget();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model_combined = tflite::GetModel(combined_model);

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver_combined;
  // op_resolver->AddSoftmax(tflite::Register_SOFTMAX_INT8_INT16());

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter_combined(
      model_combined, resolver_combined, tensor_arena_combined, kTensorArenaSize_combined);
  interpreter_combined = &static_interpreter_combined;

  // Allocate memory from the tensor_arena for the model's tensors.
  interpreter_combined->AllocateTensors();

  // Obtain pointers to the model's input and output tensors.
  input_combined = interpreter_combined->input(0);
}

// The name of this function is important for Arduino compatibility.
void loop()
{
  Serial.println("Start Void Loop");

  long int start = millis();

  // Fill input buffer with input sample
  for (int i = 0; i < 1 * 1 * 40 * 26; i++)
  {
    input_combined->data.uint8[i] = input_sample[i];
  }

  // Run inference, and report any error
  TfLiteStatus invoke_status_combined = interpreter_combined->Invoke();

  // Collect the results
  TfLiteTensor *output_combined = interpreter_combined->output(0);

  float background_dequantise = (output_combined->data.uint8[0] - output_combined->params.zero_point) * output_combined->params.scale;
  float absence_dequantise = (output_combined->data.uint8[1] - output_combined->params.zero_point) * output_combined->params.scale;
  float tonicclonic_dequantise = (output_combined->data.uint8[2] - output_combined->params.zero_point) * output_combined->params.scale;
  float general_dequantise = (output_combined->data.uint8[3] - output_combined->params.zero_point) * output_combined->params.scale;

  float background_probability = exp(background_dequantise) / (exp(background_dequantise) + exp(absence_dequantise) + exp(tonicclonic_dequantise) + exp(general_dequantise));
  float absence_probability = exp(absence_dequantise) / (exp(background_dequantise) + exp(absence_dequantise) + exp(tonicclonic_dequantise) + exp(general_dequantise));
  float tonicclonic_probability = exp(tonicclonic_dequantise) / (exp(background_dequantise) + exp(absence_dequantise) + exp(tonicclonic_dequantise) + exp(general_dequantise));
  float general_probability = exp(general_dequantise) / (exp(background_dequantise) + exp(absence_dequantise) + exp(tonicclonic_dequantise) + exp(general_dequantise));

  long int end = millis();

  // Print the results
  float background_threshold_maybe = 0.5;
  float background_threshold_likely = 0.7;
  Serial.print(background_probability);
  Serial.print(" // ");
  if (background_probability > background_threshold_maybe and background_probability < background_threshold_likely)
  {
    Serial.println("Background: Maybe");
  }
  else if (background_probability > background_threshold_likely)
  {
    Serial.println("Background: Likely");
  }
  else
  {
    Serial.println("Background: Unlikely");
  }

  float absence_threshold_maybe = 0.5;
  float absence_threshold_likely = 0.7;
  Serial.print(absence_probability);
  Serial.print(" // ");
  if (absence_probability > absence_threshold_maybe and absence_probability < absence_threshold_likely)
  {
    Serial.println("Absence Seizure: Maybe");
  }
  else if (absence_probability > absence_threshold_likely)
  {
    Serial.println("Absence Seizure: Likely");
  }
  else
  {
    Serial.println("Absence Seizure: Unlikely");
  }

  float tonicclonic_threshold_maybe = 0.5;
  float tonicclonic_threshold_likely = 0.7;
  Serial.print(tonicclonic_probability);
  Serial.print(" // ");
  if (tonicclonic_probability > tonicclonic_threshold_maybe and tonicclonic_probability < tonicclonic_threshold_likely)
  {
    Serial.println("Tonic-Clonic Seizure: Maybe");
  }
  else if (tonicclonic_probability > tonicclonic_threshold_likely)
  {
    Serial.println("Tonic-Clonic Seizure: Likely");
  }
  else
  {
    Serial.println("Tonic-Clonic Seizure: Unlikely");
  }

  float general_threshold_maybe = 0.5;
  float general_threshold_likely = 0.7;
  Serial.print(general_probability);
  Serial.print(" // ");
  if (general_probability > general_threshold_maybe and general_probability < general_threshold_likely)
  {
    Serial.println("General Seizure: Maybe");
  }
  else if (general_probability > general_threshold_likely)
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
