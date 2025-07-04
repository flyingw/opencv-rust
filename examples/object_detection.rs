use std::env;

use opencv::core::{CommandLineParser, Point, Point2f, Rect, Rect2f, Rect2i, Size, StsBadArg, StsNotImplemented, StsError, TickMeter};
//use opencv::objdetect::{FaceRecognizerSF, FaceRecognizerSF_DisType};
use opencv::prelude::*;
use opencv::{core, highgui, imgcodecs, imgproc, objdetect, videoio, Error, Result};
use opencv::core::{CV_8U, min_max_loc, Vector};
use opencv::imgproc::{FONT_HERSHEY_SIMPLEX};
use opencv::dnn::{Net,DNN_BACKEND_OPENCV};
use opencv::boxed_ref::{BoxedRef};
use std::cmp::max;
use std::fs::File;
use std::io::{BufReader,BufRead};
use std::collections::{VecDeque, BTreeMap};
use std::thread;

pub struct QueueFPS<T>{
  pub q: VecDeque<T>,
}

impl <T> QueueFPS<T> {
    let mut counter: u32;
    let tm: TickMeter = TickMeter::default();
    //mutex

    pub fn new() -> Self {
        counter = 0;
        QueueFPS {q = VecDeque::new()};
    }

    fn push(&self, entry: &T) -> () {
        //std::lock_guard<std::mutex> lock(mutex);
        self.q.push_back(empty);

        counter += 1;
        if (counter == 1)
        {
            // Start counting from a second frame (warmup).
            tm.reset();
            tm.start();
        }
    }
  
    pub fn get(&self) -> T{
        //std::lock_guard<std::mutex> lock(mutex);
        let entry: T= self.q.front();
        seld.q.pop();
        entry
    }

    pub fn getFPS(&self) -> f32 {
        tm.stop();
        let fps: f64  = counter / tm.getTimeSec();
        tm.start();
        fps as f32
    }

    pub fn clear(&self) -> () {
        //std::lock_guard<std::mutex> lock(mutex);
        while (!self.q.empty())
            self.q.pop();
    }

}

//void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
fn drawPred(label: &str, left: i32, mut top: i32, width: i32, height: i32, frame: &mut Mat) -> Result<(), opencv::Error> {
    // Draw bounding box
    let rect = Rect2i::new(top, left, width, height);

    imgproc::rectangle_def(frame, rect, (0., 255., 0.).into());

    let mut baseLine: i32 = 0;
    let labelSize: core::Size = imgproc::get_text_size(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &mut baseLine)?;

    top = max(top, labelSize.height);

    let p1 = Point::new(left, top - labelSize.height);
    let p2 = Point::new(left + labelSize.width, top + baseLine);

    imgproc::rectangle_points_def(frame, p1, p2, core::Scalar::all(255.));

    imgproc::put_text_def(frame, label, Point::new(left, top), FONT_HERSHEY_SIMPLEX, 0.5, core::Scalar::default())
}


//fn preprocess(const Mat& frame, Net& net, Size inpSize, float scale, const Scalar& mean, bool swapRB);
fn preprocess(frame: &mut Mat, net: &mut Net, mut inpSize: core::Size, scale: f64, mean: core::Scalar, swapRB: bool) -> Result<(), opencv::Error> {
    let blob: &mut Mat = &mut Mat::default();
    // Create a 4D blob from a frame.
    if inpSize.width <= 0 { inpSize.width = frame.cols();}
    if inpSize.height <= 0 { inpSize.height = frame.rows();}
    
    opencv::dnn::blob_from_image_to(frame, blob, 1.0, inpSize, core::Scalar::default(), swapRB, false, CV_8U);

    //blobFromImage(frame, blob, 1.0, inpSize, Scalar(), swapRB, false, CV_8U);

    // Run a model.
    net.set_input(blob, "", scale, mean);
    
    let mut l = net.get_layer(0)?;

    if l.output_name_to_index("im_info")? != -1  // Faster-RCNN or R-FCN
    {
        let mut frame_out = Mat::default();
        imgproc::resize_def(frame, &mut frame_out, inpSize)?;
        *frame = frame_out;

        let x = &[inpSize.height as f32, inpSize.width as f32, 1.6];
        let imInfo: BoxedRef<Mat> = Mat::new_rows_cols_with_data(1, 3, x)?;
        
        net.set_input(&imInfo, "im_info",  scale, mean);
    }

    Ok(())
}


//void postprocess(Mat& frame, const std::vector<Mat>& out, Net& net, int backend);
fn postprocess(frame: &mut Mat, outs: &Vec<Mat>, net: &Net, backend: i32, confThreshold: f32, classes: &mut Vec<String>, nmsThreshold: f32) -> Result<(),Error> {
    let outLayers: Vec<i32> = net.get_unconnected_out_layers()?.into();
    let outLayerType: String = net.get_layer(outLayers[0])?.typ();

    let mut classIds: Vec<usize> = Vec::new();
    let mut confidences: Vec<f32> = Vec::new();
    let mut boxes: Vec<Rect> = Vec::new();
    if outLayerType == "DetectionOutput" {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]

        for out in outs.into_iter() {
          let data: &[f32] = out.data_typed::<f32>()?;

            let mut i: usize = 0;
            while i< out.total() {
                let confidence: f32 = data[i + 2];
                if confidence > confThreshold
                {
                    let mut left:i32   = data[i + 3] as i32;
                    let mut top:i32    = data[i + 4] as i32;
                    let mut right:i32  = data[i + 5] as i32;
                    let mut bottom: i32 = data[i + 6] as i32;
                    let mut width: i32  = right - left + 1;
                    let mut height: i32 = bottom - top + 1;
                    if width <= 2 || height <= 2
                    {
                        left   = data[i + 3] as i32 * frame.cols();
                        top    = data[i + 4] as i32 * frame.rows();
                        right  = data[i + 5] as i32 * frame.cols();
                        bottom = data[i + 6] as i32 * frame.rows();
                        width  = right - left + 1;
                        height = bottom - top + 1;
                    }
                    classIds.push(data[i + 1] as usize - 1);  // Skip 0th background class id.
                    boxes.push(Rect::new(left, top, width, height));
                    confidences.push(confidence);
                }
              i+=7;
            }
        }
    }
    else if outLayerType == "Region" {

        for out in outs.into_iter() {
          // Network produces output blob with a shape NxC where N is a number of
          // detected objects and C is a number of classes + 4 where the first 4
          // numbers are [center_x, center_y, width, height]
          let data: &[i32] = out.data_typed::<i32>()?;

          for row in 0..out.rows() {
            let m: &[i32] = out.at_row(row)?;
            let mat: BoxedRef<Mat> = Mat::from_slice(m)?;
            let range = core::Range::new(5, out.cols())?;
            let scores:BoxedRef<Mat> = mat.col_range(&range)?;

            let mut classIdPoint: Point = Point::default();
            let mut confidence = 0. as f64;
            let mut min = 0.;
            min_max_loc(&scores, Some(&mut min), Some(&mut confidence), Some(&mut Point::new(0,0)), Some(&mut classIdPoint), &Mat::default());

            if confidence > confThreshold.into() {
              let centerX: i32 = (data[0] * frame.cols()) as i32;
              let centerY: i32 = (data[1] * frame.rows()) as i32;
              let width: i32=  (data[2] * frame.cols()) as i32;
              let height: i32 = (data[3] * frame.rows()) as i32;
              let left: i32 = centerX - width / 2;
              let top: i32 = centerY - height / 2;

              classIds.push(classIdPoint.x.try_into().unwrap());
              confidences.push(confidence as f32);
              boxes.push(Rect::new(left, top, width, height));
            }
          }
        }

    }  else {
      return Err(Error::new(StsNotImplemented, "Unknown output layer type: ".to_owned() + &outLayerType));
    }

    // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    // or NMS is required if number of outputs > 1
    if outLayers.len() > 1 || (outLayerType == "Region" && backend != DNN_BACKEND_OPENCV) {
        let mut class2indices: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
  
        for class in 0..classIds.len() {
          if confidences[class] >= confThreshold {
            let key: usize = classIds[class];
            class2indices.entry(key.try_into().unwrap()).and_modify(|v| v.push(class));

          }
        }

        let mut nmsBoxes : Vec<Rect> = Vec::new();
        let mut nmsConfidences: Vec<f32> = Vec::new();
        let mut nmsClassIds: Vec<usize> = Vec::new();


        for (k,v) in class2indices {
          let mut localBoxes: Vec<Rect> = Vec::new();
          let mut localConfidences: Vec<f32> = Vec::new();
          let classIndices: Vec<usize> = v;

          for i in classIndices.into_iter() {
            localBoxes.push(boxes[i]);
            localConfidences.push(confidences[i]);
          }

          let nmsIndices: Vec<i32> = Vec::new();
          //nms_boxes_f64_def
  
          opencv::dnn::nms_boxes_def(
              &Vector::from_slice(&localBoxes[..]), 
              &Vector::from_slice(&localConfidences[..]), 
              confThreshold as f32,
              nmsThreshold, 
              &mut Vector::from_slice(&nmsIndices[..]));

          for idx in nmsIndices.into_iter() {
            nmsBoxes.push(localBoxes[idx as usize]);
            nmsConfidences.push(localConfidences[idx as usize]);
            nmsClassIds.push(k.try_into().unwrap());
          }
        }
        boxes = nmsBoxes;
        classIds = nmsClassIds;
        confidences = nmsConfidences;
    }

    let mut idx = 0;
    while idx < boxes.len() {
      idx+=1;
      let box0: Rect = boxes[idx];
      let conf = confidences[idx];
      let classId = classIds[idx];

      let mut label: String = format!("{:.2}", conf);
      if classes.len() > 0 {
        label = classes[classId].clone() + ": " + &label;
      }

      drawPred(label.as_str(), box0.x, box0.y, box0.width, box0.height,frame);
    }

    Ok(())
}

fn main() -> Result<()> {
  let args = env::args().collect::<Vec<_>>();
  let args = args.iter().map(|arg| arg.as_str()).collect::<Vec<_>>();
  let parser = CommandLineParser::new(
    &args,
    concat!(
      "{help  h           |            | Print this message}",
      "{ @alias      | | An alias name of model to extract preprocessing parameters from models.yml file. }",
      "{ zoo         | models.yml | An optional path to file with preprocessing parameters }",
      "{ device      |  0 | camera device number. }",
      "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }",
      "{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }",
      "{ classes     | | Optional path to a text file with names of classes to label detected objects. }",
      "{ thr         | .5 | Confidence threshold. }",
      "{ nms         | .4 | Non-maximum suppression threshold. }",
      "{ backend     |  0 | Choose one of computation backends: 
                         0: automatically (by default),
                         1: Halide language (http://halide-lang.org/), 
                         2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), 
                         3: OpenCV implementation, 
                         4: VKCOM, 
                         5: CUDA }",
      "{ target      | 0 | Choose one of target computation devices:
                         0: CPU target (by default), 
                         1: OpenCL, 
                         2: OpenCL fp16 (half-float precision), 
                         3: VPU, 
                         4: Vulkan, 
                         6: CUDA, 
                         7: CUDA fp16 (half-float preprocess) }",
      "{ async       | 0 | Number of asynchronous forwards at the same time.
                        Choose 0 for synchronous mode }"),
  )?;

  if parser.has("help")? {
    parser.print_message()?;
    return Ok(());
  }
  
  //Use this script to run object detection deep learning networks using OpenCV

  let modelName = parser.get_str_def("@alias")?;
  let zooFile = parser.get_str_def("zoo")?;
  let mut confThreshold = parser.get_f64_def("thr")?;
  let nmsThreshold = parser.get_f64_def("nms")?  as f32;
  let scale = parser.get_f64_def("scale")? as f32;
  //: "104, 117, 123",
  let mean: core::Scalar = parser.get_scalar("mean", true)?;
  //let mean: core::Scalar = core::Scalar::new(mean);
  let swapRB = parser.get_bool_def("rgb")?;
  let inpWidth = parser.get_i32_def("width")?;
  let inpHeight = parser.get_i32_def("height")?;
  let asyncNumReq = parser.get_i32_def("async")? as usize;
  let mut classes: Vec<String> = Vec::new();

  type Callback = Box<(dyn FnMut(i32)+ Send + Sync + 'static)>;

  let cb = |pos: i32| -> () { 
    confThreshold = (pos as f64) * 0.01;
  };

  if parser.has("model")? {
    let modelPath = core::find_file_def(&parser.get_str_def("model")?)?.as_str();
    let configPath = core::find_file_def(&parser.get_str_def("config")?)?.as_str();

    // Open file with classes names.
    if parser.has("classes")? {
        let file = parser.get_str_def("classes")?;
        
        let css = match File::open(file) {
          Ok(ifs) => {
            let reader = BufReader::new(ifs);
            for line in reader.lines().map_while(Result::ok) {
              classes.push(line.to_string());
            };
            Ok(())
          },
          Err(e) => Err(Error::new(StsError, "File ".to_owned() + &file + " not found")),
        };
        let _ = css?;
    }

    // Load a model.
    let mut net: Net = opencv::dnn::read_net(modelPath, configPath, parser.get_str_def("framework")?.as_str())?;
    let backend = parser.get_i32_def("backend")?;
    net.set_preferable_backend(backend);
    net.set_preferable_target(parser.get_i32_def("target")?);
    let outNames: core::Vector<String> = net.get_unconnected_out_layers_names()?;


    // Create a window
    const kWinName: &str = "Deep learning object detection in OpenCV";
    highgui::named_window(kWinName, highgui::WINDOW_NORMAL);
    let mut initialConf: Option<&mut i32> = Some (&mut ((confThreshold as i32) * 100));

    highgui::create_trackbar("Confidence threshold, %", kWinName, initialConf, 99, Some(Box::new(cb)));

    // Open a video file or an image file or a camera stream.
    let mut cap = videoio::VideoCapture::default()?;

    if parser.has("input")? {
        let file = parser.get_str_def("input")?;
        let x = core::find_file_or_keep_def(&file)?;
        cap.open_file_def(&x);
    } else {
        cap.open_def(parser.get_i32_def("device")?);
    }

    // ===================================
    // no threads version
    // if (asyncNumReq){
    //   Err(Error::new(StsNotImplemented, "Asynchronous forward is supported only with Inference Engine backend."));
    // }

    // // Process frames
    // let mut frame: Mat = Mat::default();
    // let blob: Mat  = Mat::default();

    // while (highgui::wait_key_ex(1)? < 0)
    // {
    //     if !cap.read(&mut frame)? {
    //         println!("Can't grab frame! Stop");
    //         break;
    //     }
    //     if frame.size()?.width == 0 {
    //         highgui::wait_key_def();
    //         break;
    //     }

    //     preprocess(&mut frame, net, Size::new(inpWidth, inpHeight), scale, mean, swapRB);

    //     let outs: Vec<Mat> = Vec::new();
    //     net.forward(outs, outNames);

    //     postprocess(&frame, &outs, net, backend);

    //     // Put efficiency information.
    //     let layersTimes:Vec<f64> = Vec::new();
    //     let freq: f64 = get_tick_frequency() / 1000;
    //     let t: f64 = net.getPerfProfile(layersTimes) / freq;
    //     let label = format!("Inference time: %.2f ms", t);
    //     imgproc::put_text_def(&mut frame, &label, Point::new(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, core::Scalar(0, 255, 0));

    //     highgui::imshow(kWinName, &frame);
    // }

    //========================
    // threads version? probably should be default
    let process: bool = true;
    let framesQueue: QueueFPS<Mat> = QueueFPS::new();

    // Frames capturing thread
    let framesThread = thread::spawn(move || loop {
        let frame: Mat;
        while process {
            if !cap.read(&mut frame)? {
              break;
            }
            if frame.size()?.width != 0 {
              framesQueue.push(frame.clone());
            } else {
              break;
            }
        }
    });

    let processedFramesQueue: QueueFPS<Mat> = QueueFPS::new();
    let predictionsQueue: QueueFPS<Vec<Mat>> = QueueFPS::new();

    // Frames processing thread
    let processingThread = thread::spawn(move || loop {
        let futureOutputs: Vec<core::AsyncArray> = Vec::new();
        let blob: Mat;
        while process
        {
            // Get a next frame
            let frame: Mat = Mat::default();
            {
                if !framesQueue.empty() {
                    frame = framesQueue.get();
                    if asyncNumReq != 0 {
                        if futureOutputs.len() == asyncNumReq {
                          frame = Mat::default();
                        }
                    }
                    else {
                      framesQueue.clear();  // Skip the rest of frames
                    }
                }
            }

            // Process the frame
            if !frame.empty() {
                preprocess(&mut frame, &mut net, Size::new(inpWidth, inpHeight), scale.into(), mean, swapRB);
                processedFramesQueue.push(frame);

                if asyncNumReq != 0
                {
                    futureOutputs.push(net.forward_async_def()?);
                }
                else
                {
                    let outs: core::Vector<Mat> = Vec::new().into();
                    net.forward(&mut outs, &outNames);
                    predictionsQueue.push(outs);
                }
            }

            while !futureOutputs.is_empty() &&
                   futureOutputs.into().front().wait_for(std::time::Duration::from_secs(0)) {
                let async_out: core::AsyncArray = futureOutputs.into().front();
                futureOutputs.pop();
                let out: Mat;
                async_out.get(out);
                predictionsQueue.push(out);
            }
        }
    });

    // Postprocessing and rendering loop
    while highgui::wait_key_ex(1)? < 0 {
        if predictionsQueue.empty() {
          continue;
        }

        let outs:Vec<Mat> = predictionsQueue.get();
        let frame: Mat = processedFramesQueue.get();

        postprocess(frame, outs, &net, backend, confThreshold as f32, &mut classes, nmsThreshold);

        if predictionsQueue.counter > 1
        {
            let label = format!("Camera: {:.2} FPS", framesQueue.getFPS());
            imgproc::put_text_def(frame, &label, Point::new(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, core::Scalar::new(0., 0., 255., 0.));

            let label = format!("Network: {:.2} FPS", predictionsQueue.getFPS());
            imgproc::put_text_def(frame, &label, Point::new(0, 30), FONT_HERSHEY_SIMPLEX, 0.5, core::Scalar::new(0., 0., 255., 0.));

            let label = format!("Skipped frames: {:?}", framesQueue.counter - predictionsQueue.counter);
            imgproc::put_text_def(frame, &label, Point::new(0, 45), FONT_HERSHEY_SIMPLEX, 0.5, core::Scalar::new(0., 0., 255., 0.));
        }
        highgui::imshow(kWinName, frame);
    }

    process = false;
    framesThread.join();
    processingThread.join();

    Ok(())
  } else {
    Err(Error::new(StsError, "Cannot find a model"))
  }
}