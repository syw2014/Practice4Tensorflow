/*************************************************************************
# File Name: demo-inferece.cc
# Method: 
# Author: Jerry Shi
# Mail: jerryshi0110@gmail.com
# Created Time: 2020年07月19日 18:10:16
 ************************************************************************/
// A demo to show how to use tf2.x C++ api for online inference 

#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/public/session_options.h"


// These are all comon classes it's hady to reference with no namespace.
using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::tstring;


// Load saved model with LoadSavedModel api, then can run with bundle.session->run()
// *Note*:  In TF2.x saved_model was a folder with asserts/,variables/, *.pb file, not a single
// file end with `.pb`
Status LoadGraph(tensorflow::SessionOptions &session_options, 
                        tensorflow::RunOptions& run_options,
                      const std::string& export_dir, 
                      tensorflow::SavedModelBundle &bundle) {
    Status load_status = tensorflow::LoadSavedModel(session_options, run_options,
                                    export_dir,{tensorflow::kSavedModelTagServe}, &bundle);
    if(!load_status.ok()) {
      std::cout << "Load saved model failed: " << load_status << std::endl;
    } else {
      std::cout << "Load saved model succeed from " << export_dir << std::endl;
    }
    return Status::OK();
}

int main(int argc, char* argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model export_dir  
  tensorflow::string graph = "./data/model_pb";

  // *Note*: the name of input and output was found based on `saved_model_cli`, which
  // can found the detail infos of model.In TF2.x can not set name for input and output
  // so you should found the real names with commond 
  // `saved_model_clis show --dir ./saved_path --tag_set serve --signature_def serving_default`
  // here results:
  // The given SavedModel SignatureDef contains the following input(s):
  // inputs['input_1'] tensor_info:
  //    dtype: DT_INT32
  //    shape: (-1, 1)
  //    name: serving_default_input_1:0
  //  The given SavedModel SignatureDef contains the following output(s):
  // outputs['output_1'] tensor_info:
  //    dtype: DT_FLOAT
  //    shape: (-1, 1)
  //    name: StatefulPartitionedCall:0
  // Method name is: tensorflow/serving/predict
  // More info about `saved_model_cli` can be found in 
  // https://www.tensorflow.org/guide/saved_model#load_a_savedmodel_in_c

  tensorflow::string input_layer = "serving_default_input_1";
  tensorflow::string output_layer = "StatefulPartitionedCall";
  
  tensorflow::string root_dir = "";
  std::vector<Flag> flag_list = {
      Flag("graph", &graph, "graph to be executed"),
      Flag("input_layer", &input_layer, "name of input layer"),
      Flag("output_layer", &output_layer, "name of output layer"),
      Flag("root_dir", &root_dir,
           "interpret image and graph file names relative to this directory"),
  };
  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // First we load and initialize the model.
  tensorflow::string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  // load graph with tf2.0
  tensorflow::SavedModelBundle bundle;
  tensorflow::RunOptions run_options;
  tensorflow::SessionOptions session_options;
  Status load_graph_status = LoadGraph(session_options, run_options, graph_path, bundle);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  // define input tensor and assign data
  // The shape of input can be found with `saved_model_cli`
  // in this example, input shape=[batch, dimension]
  int dims = 1;
  // Define tensor, with <data_type, tensor_shape>
  Tensor input(tensorflow::DT_INT32, tensorflow::TensorShape({1, dims}));
  // Get data map from tensor, the second was indics of tensor dimension
  auto input_map = input.tensor<int, 2>();
  // Assign data into tensor
  input_map(0) = 20;

  // Actually run the image through the model.
  // Define output tensor, and get prdict tensor
  std::vector<Tensor> outputs;
  Status run_status = bundle.session->Run({{input_layer, input}},
                                   {output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  // Print detail results in output tensor
  // Get detail data which is the reverse process of set data into tensor
  // As we only have on data so we use outputs[0], and the type of out put was float
  // only have 1-dim
  auto tensor_map = outputs[0].tensor<float, 2>();
  LOG(INFO) << "Prediction dim: " << outputs[0].shape().dim_size(1);
  LOG(INFO) << "Prediction data: " << tensor_map(0, 0);

  return 0;
}


