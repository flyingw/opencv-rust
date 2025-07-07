use std::env;

use opencv::core::{CommandLineParser, Point, Rect, Rect2i, Size, StsNotImplemented, StsError, TickMeter};
//use opencv::objdetect::{FaceRecognizerSF, FaceRecognizerSF_DisType};
use opencv::prelude::*;
use opencv::{core, dnn, highgui, imgproc, videoio, Error, Result};
use opencv::core::{CV_8U, min_max_loc, Vector};
use opencv::imgproc::{FONT_HERSHEY_SIMPLEX};
use opencv::dnn::{Net,DNN_BACKEND_OPENCV};
use opencv::boxed_ref::{BoxedRef};
use std::cmp::max;
use std::fs::File;
use std::io::{BufReader,BufRead};
use std::collections::{VecDeque, BTreeMap};
use std::thread;
use std::cell::Cell;
use std::sync::{Arc,Mutex};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

pub struct QueueFPS<T>{
  pub q: Cell<VecDeque<T>>,
  pub counter: Cell<u32>,
  pub tm: Cell<TickMeter>,
}

impl <T: Clone> QueueFPS<T> {

  pub fn new() -> Self {
    QueueFPS {
      q: Cell::new(VecDeque::new()),
      counter: Cell::new(0),
      tm: Cell::new(TickMeter::default().unwrap()),
    }
  }

  fn push(&mut self, entry: &T) -> () {
      let q = self.q.get_mut();
      q.push_front(entry.clone());
      let tm = self.tm.get_mut();
      let count = self.counter.get();
      self.counter.set(count + 1);
      if self.counter.get() == 1
      {
          // Start counting from a second frame (warmup).
          let _ = tm.reset();
          let _ = tm.start();
      }
  }

  pub fn is_empty(&mut self) -> bool {
    self.q.get_mut().is_empty()
  }
  
  pub fn get(&mut self) -> T {
      self.q.get_mut().pop_front().unwrap()
  }

  pub fn get_fps(&mut self) -> f32 {
      let tm = self.tm.get_mut();
      let _ = tm.stop();
      let count = self.counter.get();
      let fps: f64  = count  as f64 / tm.get_time_sec().unwrap();
      let _ = tm.start();
      fps as f32
  }

  pub fn clear(&mut self) -> () {
    self.q.get_mut().clear();
  }

}

//void draw_pred(int class_id, float conf, int left, int top, int right, int bottom, Mat& frame);
fn draw_pred(label: &str, left: i32, mut top: i32, width: i32, height: i32, frame: &mut Mat) -> Result<(), opencv::Error> {
    // Draw bounding box
    let rect = Rect2i::new(top, left, width, height);

    let _ = imgproc::rectangle_def(frame, rect, (0., 255., 0.).into());

    let mut base_line: i32 = 0;
    let label_size: core::Size = imgproc::get_text_size(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &mut base_line)?;

    top = max(top, label_size.height);

    let p1 = Point::new(left, top - label_size.height);
    let p2 = Point::new(left + label_size.width, top + base_line);

    let _ = imgproc::rectangle_points_def(frame, p1, p2, core::Scalar::all(255.));

    imgproc::put_text_def(frame, label, Point::new(left, top), FONT_HERSHEY_SIMPLEX, 0.5, core::Scalar::default())
}


//fn preprocess(const Mat& frame, Net& net, Size inp_size, float scale, const Scalar& mean, bool swap_rb);
fn preprocess(frame: &mut Mat, net: &mut Net, mut inp_size: core::Size, scale: f64, mean: core::Scalar, swap_rb: bool) -> Result<(), opencv::Error> {
    let blob: &mut Mat = &mut Mat::default();
    // Create a 4D blob from a frame.
    if inp_size.width <= 0 { inp_size.width = frame.cols();}
    if inp_size.height <= 0 { inp_size.height = frame.rows();}
    
    let _ = opencv::dnn::blob_from_image_to(frame, blob, 1.0, inp_size, core::Scalar::default(), swap_rb, false, CV_8U);

    //blobFromImage(frame, blob, 1.0, inp_size, Scalar(), swap_rb, false, CV_8U);

    // Run a model.
    let _ = net.set_input(blob, "", scale, mean);
    
    let mut l = net.get_layer(0)?;

    if l.output_name_to_index("im_info")? != -1  // Faster-RCNN or R-FCN
    {
        let mut frame_out = Mat::default();
        imgproc::resize_def(frame, &mut frame_out, inp_size)?;
        *frame = frame_out;

        let x = &[inp_size.height as f32, inp_size.width as f32, 1.6];
        let im_info: BoxedRef<Mat> = Mat::new_rows_cols_with_data(1, 3, x)?;
        
        let _ = net.set_input(&im_info, "im_info",  scale, mean);
    }

    Ok(())
}


//void postprocess(Mat& frame, const std::vector<Mat>& out, Net& net, int backend);
fn postprocess(frame: &mut Mat,
                outs: &Vec<Mat>,
                net: &Net,
                backend: i32,
                conf_threshold: f32, 
                classes: &mut Vec<String>,
                nms_threshold: f32) -> Result<(),Error> {
    let out_layers: Vec<i32> = net.get_unconnected_out_layers()?.into();
    let out_layer_type: String = net.get_layer(out_layers[0])?.typ();

    let mut class_ids: Vec<usize> = Vec::new();
    let mut confidences: Vec<f32> = Vec::new();
    let mut boxes: Vec<Rect> = Vec::new();
    if out_layer_type == "DetectionOutput" {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, class_id, confidence, left, top, right, bottom]

        for out in outs.into_iter() {
          let data: &[f32] = out.data_typed::<f32>()?;

            let mut i: usize = 0;
            while i< out.total() {
                let confidence: f32 = data[i + 2];
                if confidence > conf_threshold
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
                    class_ids.push(data[i + 1] as usize - 1);  // Skip 0th background class id.
                    boxes.push(Rect::new(left, top, width, height));
                    confidences.push(confidence);
                }
              i+=7;
            }
        }
    }
    else if out_layer_type == "Region" {

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

            let mut class_id_point: Point = Point::default();
            let mut confidence = 0. as f64;
            let mut min = 0.;
            let _ = min_max_loc(&scores, Some(&mut min), Some(&mut confidence), Some(&mut Point::new(0,0)), Some(&mut class_id_point), &Mat::default());

            if confidence > conf_threshold.into() {
              let center_x: i32 = (data[0] * frame.cols()) as i32;
              let center_y: i32 = (data[1] * frame.rows()) as i32;
              let width: i32=  (data[2] * frame.cols()) as i32;
              let height: i32 = (data[3] * frame.rows()) as i32;
              let left: i32 = center_x - width / 2;
              let top: i32 = center_y - height / 2;

              class_ids.push(class_id_point.x.try_into().unwrap());
              confidences.push(confidence as f32);
              boxes.push(Rect::new(left, top, width, height));
            }
          }
        }

    }  else {
      return Err(Error::new(StsNotImplemented, "Unknown output layer type: ".to_owned() + &out_layer_type));
    }

    // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    // or NMS is required if number of outputs > 1
    if out_layers.len() > 1 || (out_layer_type == "Region" && backend != DNN_BACKEND_OPENCV) {
        let mut class2indices: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
  
        for class in 0..class_ids.len() {
          if confidences[class] >= conf_threshold {
            let key: usize = class_ids[class];
            class2indices.entry(key.try_into().unwrap()).and_modify(|v| v.push(class));

          }
        }

        let mut nms_boxes : Vec<Rect> = Vec::new();
        let mut nms_confidences: Vec<f32> = Vec::new();
        let mut nmsclass_ids: Vec<usize> = Vec::new();


        for (k,v) in class2indices {
          let mut local_boxes: Vec<Rect> = Vec::new();
          let mut local_confidences: Vec<f32> = Vec::new();
          let class_indices: Vec<usize> = v;

          for i in class_indices.into_iter() {
            local_boxes.push(boxes[i]);
            local_confidences.push(confidences[i]);
          }

          let nms_indices: Vec<i32> = Vec::new();
          //nms_boxes_f64_def
  
          let _ = opencv::dnn::nms_boxes_def(
              &Vector::from_slice(&local_boxes[..]), 
              &Vector::from_slice(&local_confidences[..]), 
              conf_threshold as f32,
              nms_threshold, 
              &mut Vector::from_slice(&nms_indices[..]));

          for idx in nms_indices.into_iter() {
            nms_boxes.push(local_boxes[idx as usize]);
            nms_confidences.push(local_confidences[idx as usize]);
            nmsclass_ids.push(k.try_into().unwrap());
          }
        }
        boxes = nms_boxes;
        class_ids = nmsclass_ids;
        confidences = nms_confidences;
    }

    let mut idx = 0;
    while idx < boxes.len() {
      idx+=1;
      let box0: Rect = boxes[idx];
      let conf = confidences[idx];
      let class_id = class_ids[idx];

      let mut label: String = format!("{:.2}", conf);
      if classes.len() > 0 {
        label = classes[class_id].clone() + ": " + &label;
      }

      let _ = draw_pred(label.as_str(), box0.x, box0.y, box0.width, box0.height,frame);
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

  let mut conf_threshold = Cell::new(parser.get_f64_def("thr")?);

  let nms_threshold = parser.get_f64_def("nms")?  as f32;
  let scale = parser.get_f64_def("scale")? as f32;
  //: "104, 117, 123",
  //let mean: core::Scalar = parser.get_scalar("mean", true)?;
  let _mean = parser.get_str_def("mean")?;
  let mean: core::Scalar = core::Scalar::new(0., 0., 0., 0.);
  let swap_rb = parser.get_bool_def("rgb")?;
  let inp_width = parser.get_i32_def("width")?;
  let inp_height = parser.get_i32_def("height")?;
  let async_num_req = parser.get_i32_def("async")? as usize;
  let mut classes: Vec<String> = Vec::new();

  //type Callback = Box<(dyn FnMut(i32)+ Send + Sync + 'static)>;
  // let cb = |pos: i32| -> () { 
  //   conf_threshold = (pos as f64) * 0.01;
  // };

  if parser.has("model")? {
    let model = parser.get_str_def("model")?;
    let config = parser.get_str_def("config")?;
    let framework = parser.get_str_def("framework")?;

    // Open file with classes names.
    if parser.has("classes")? {
        let file = parser.get_str_def("classes")?;
        
        let css = match File::open(&file) {
          Ok(ifs) => {
            let reader = BufReader::new(ifs);
            for line in reader.lines().map_while(Result::ok) {
              classes.push(line.to_string());
            };
            Ok(())
          },
          Err(_e) => Err(Error::new(StsError, "File ".to_owned() + &file + " not found")),
        };
        let _ = css?;
    }

    // Load a model.
    let mut net: Net = dnn::read_net(&model, &config, &framework)?;
    let backend = parser.get_i32_def("backend")?;
    let _ = net.set_preferable_backend(backend);
    let _ = net.set_preferable_target(parser.get_i32_def("target")?);
    let out_names: core::Vector<String> = net.get_unconnected_out_layers_names()?;

    // Create a window
    const K_WIN_NAME: &str = "Deep learning object detection in OpenCV";
    let _ = highgui::named_window(K_WIN_NAME, highgui::WINDOW_NORMAL);
    let mut th100: i32 = (conf_threshold as i32) * 100;
    let initial_conf: Option<&mut i32> = Some(&mut th100);

    let _ = highgui::create_trackbar("Confidence threshold, %", 
      K_WIN_NAME, initial_conf, 99, Some(Box::new({
        move |pos| {
          conf_threshold.set((pos as f64) * 0.01);
        }
      })));

    // Open a video file or an image file or a camera stream.
    let mut cap = videoio::VideoCapture::default()?;

    if parser.has("input")? {
        let file = parser.get_str_def("input")?;
        let x = core::find_file_or_keep_def(&file)?;
        let _ = cap.open_file_def(&x);
    } else {
        let _ = cap.open_def(parser.get_i32_def("device")?);
    }

    let process: Arc<AtomicBool> = Arc::new(AtomicBool::new(true));

    let fq: QueueFPS<Mat> = QueueFPS::new();
    let frames_queue = Arc::new(Mutex::new(fq));

    // Frames capturing thread
    let frames_queue1 = frames_queue.clone();
    let process_frames = Arc::clone(&process);
    let frames_thread = thread::spawn(move || loop {
        let mut frames_queue1 = frames_queue1.lock().unwrap();
        let mut frame: Mat = Mat::default();

        while process_frames.load(Ordering::Relaxed) {
            match cap.read(&mut frame) {
              Ok(false) => break,
              Ok(_) => {
                match frame.size() {
                  Ok(s) if s.width != 0 =>
                    frames_queue1.push(&frame),
                  _ => break,
                }
              },
              Err(_) => break,
            }
        }
    });

    let pfq: QueueFPS<Mat> = QueueFPS::new();
    let processedframes_queue = Arc::new(Mutex::new(pfq));
    let pdq: QueueFPS<core::Vector<Mat>> = QueueFPS::new();
    let predictions_queue = Arc::new(Mutex::new(pdq));

    // Frames processing thread
    let frames_queue2 = frames_queue.clone();
    let processedframes_queue2 = processedframes_queue.clone();
    let predictions_queue2 = predictions_queue.clone();
    let process_p = process.clone();

    let net_a = Arc::new(Mutex::new(net));
    let net_b = net_a.clone();
    let processing_thread = thread::spawn(move || loop {
        let mut frames_queue2 = frames_queue2.lock().unwrap();
        let mut future_outputs: VecDeque<core::AsyncArray> = VecDeque::new();
        let blob: Mat = Mat::default();
        while process_p.load(Ordering::Relaxed){
            // Get a next frame
            let mut frame: Mat = Mat::default();
            {
                if !frames_queue2.is_empty() {
                    frame = frames_queue2.get();
                    if async_num_req != 0 {
                        if future_outputs.len() == async_num_req {
                          frame = Mat::default();
                        }
                    }
                    else {
                      frames_queue2.clear();  // Skip the rest of frames
                    }
                }
            }

            // Process the frame
            let mut predictions_queue2 = predictions_queue2.lock().unwrap();
            if !frame.empty() {
                let mut ne = net_b.lock().unwrap();
                let _ = preprocess(&mut frame, &mut ne, Size::new(inp_width, inp_height), scale.into(), mean, swap_rb);
                processedframes_queue2.lock().unwrap().push(&frame);

                if async_num_req != 0
                {
                    future_outputs.push_back(ne.forward_async_def().unwrap());
                }
                else
                {
                    let mut outs: core::Vector<Mat> = Vec::new().into();
                    let _ = ne.forward(&mut outs, &out_names);
                    predictions_queue2.push(&outs);
                }
            }

            while !future_outputs.is_empty() &&
                   future_outputs.front().unwrap().wait_for(0).unwrap() {
                let async_out: core::AsyncArray = future_outputs.pop_front().unwrap();

                let mut out: Mat = Mat::default();
                let _ = async_out.get(&mut out);
                predictions_queue2.push(&core::Vector::from(vec![out]));
            }
        }
    });

    // Postprocessing and rendering loop
    
    let net_aa = net_a.clone();
    while highgui::wait_key_ex(1)? < 0 {
        let mut predictions_queue = predictions_queue.lock().unwrap();
        let mut frames_queue = frames_queue.lock().unwrap();
        let mut processedframes_queue = processedframes_queue.lock().unwrap();
         if predictions_queue.is_empty() {
          continue;
        }

        let outs:Vec<Mat> = predictions_queue.get().into();
        let mut frame: Mat = processedframes_queue.get();
        let ne = net_aa.lock().unwrap();
        let _ = postprocess(&mut frame, &outs, &ne, backend, conf_threshold as f32, &mut classes, nms_threshold);

        if predictions_queue.counter.get() > 1
        {
            let label = format!("Camera: {:.2} FPS", frames_queue.get_fps());
            let _ = imgproc::put_text_def(&mut frame, &label, Point::new(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, core::Scalar::new(0., 0., 255., 0.));

            let label = format!("Network: {:.2} FPS", predictions_queue.get_fps());
            let _ = imgproc::put_text_def(&mut frame, &label, Point::new(0, 30), FONT_HERSHEY_SIMPLEX, 0.5, core::Scalar::new(0., 0., 255., 0.));

            let label = format!("Skipped frames: {:?}", frames_queue.counter.get() - predictions_queue.counter.get());
            let _ = imgproc::put_text_def(&mut frame, &label, Point::new(0, 45), FONT_HERSHEY_SIMPLEX, 0.5, core::Scalar::new(0., 0., 255., 0.));
        }
        let _ = highgui::imshow(K_WIN_NAME, &frame);
    }

    process.store(false, Ordering::Relaxed);
    let _ = frames_thread.join();
    let _ = processing_thread.join();

    Ok(())
  } else {
    Err(Error::new(StsError, "Cannot find a model"))
  }
}