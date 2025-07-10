use std::env;

use opencv::core::{CommandLineParser, Point, Rect, Rect2i, Size, StsNotImplemented, StsError, TickMeter};
use opencv::prelude::*;
use opencv::{core, dnn, highgui, imgproc, videoio, Error, Result};
use opencv::core::{CV_8U, Vector};
use opencv::imgproc::{FONT_HERSHEY_SIMPLEX};
use opencv::dnn::{Net,DNN_BACKEND_OPENCV};
use opencv::boxed_ref::{BoxedRef};
//use videoio::{CAP_FFMPEG,CAP_GSTREAMER};
use std::cmp::max;
use std::collections::{VecDeque, BTreeMap};
use std::thread;
use std::sync::{Arc,RwLock,Mutex};
use std::convert::identity;
use std::time::Duration;

pub struct QueueFPS<T>{
  pub q: VecDeque<T>,
  pub counter: u32,
}

impl <T> QueueFPS<T> {

  pub fn new() -> Self {
    QueueFPS {
      q: VecDeque::<T>::new(),
      counter: 0,
    }
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
fn preprocess(blob: &mut Mat, 
              frame: &mut Mat,
              net: &mut Net,
              mut inp_size: core::Size, 
              scale: f64, 
              mean: core::Scalar, 
              swap_rb: bool) -> Result<(), opencv::Error> {
    // Create a 4D blob from a frame.
    if inp_size.width <= 0 { inp_size.width = frame.cols();}
    if inp_size.height <= 0 { inp_size.height = frame.rows();}
    
    let _ = opencv::dnn::blob_from_image_to(frame, blob, 1.0, inp_size, core::Scalar::default(), swap_rb, false, CV_8U);

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

fn postprocess(frame: &mut Mat,
                outs: &core::Vector<Mat>,
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
          let data: &[f32] = out.data_typed::<f32>()?;

          for row in 0..out.rows() {
            let m: &[f32] = out.at_row::<f32>(row)?;
            let oc: usize = out.cols() as usize;
            let scores: BoxedRef<Mat> = Mat::from_slice(&m[5..oc])?;

            let mut class_id_point: Point = Point::default();
            let mut confidence = 0. as f64;
            let mut min = 0.;
            let _ = core::min_max_loc(&scores, Some(&mut min), Some(&mut confidence), Some(&mut Point::new(0,0)), Some(&mut class_id_point), &Mat::default());

            if confidence > conf_threshold.into() {
              let center_x: i32 = (data[0] as i32 * frame.cols()) as i32;
              let center_y: i32 = (data[1] as i32 * frame.rows()) as i32;
              let width: i32=  (data[2] as i32 * frame.cols()) as i32;
              let height: i32 = (data[3] as i32 * frame.rows()) as i32;
              let left: i32 = center_x - width / 2;
              let top: i32 = center_y - height / 2;

              println!("confidence {0}", confidence);
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

    println!("boxes? {:?}", boxes.len());

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

      println!("draw box...{0}", label);
      let _ = draw_pred(label.as_str(), box0.x, box0.y, box0.width, box0.height,frame);
    }

    Ok(())
}

const KEYS: &str = concat!(
    "{ help  h     |     | Print this message}",
    "{ device      | 0   | camera device number. }",
    "{ input i     |     | Path to input image or video file. Skip this argument to capture frames from a camera. }",
    "{ scale       | 1.0 | Scale factor used to resize input video frames}",
    "{ rgb         | 1   | Indicate that model works with RGB input images instead BGR ones. }",
    "{ framework f |     | Optional name of an origin framework of the model. Detect it automatically if it does not set. }",
    "{ classes     |     | Optional path to a text file with names of classes to label detected objects. }",

    "{ thr         | .5  | Confidence threshold. }",
    "{ nms         | .4  | Non-maximum suppression threshold. }",
    "{ width       | 608 | Preprocess input image by resizing to a specific width. }",
    "{ height      | 608 | Preprocess input image by resizing to a specific height. }",
    "{ config c    | yolov4.cfg | Path to the model configuration file. }",
    "{ backend     |  0  | Choose one of computation backends: 
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
    "{model | yolov4.weights | Model?}",
);

fn main() -> Result<()> {
  let args = env::args().collect::<Vec<_>>();
  let args = args.iter().map(|arg| arg.as_str()).collect::<Vec<_>>();
  let parser = CommandLineParser::new(&args, KEYS);
  let parser = parser.unwrap();

  if parser.has("help")? {
    parser.print_message()?;
    return Ok(());
  }
  let conf_threshold = parser.get_f64_def("thr")?;
  let nms_threshold = parser.get_f64_def("nms")?  as f32;
  let scale = parser.get_f64_def("scale")? as f32;
  let mean: core::Scalar = core::Scalar::new(0., 0., 0., 0.);
  let swap_rb = parser.get_bool_def("rgb")?;
  let inp_width = parser.get_i32_def("width")?;
  let inp_height = parser.get_i32_def("height")?;
  let mut classes: Vec<String> = Vec::new();

  if parser.has("model")? {
    let model = parser.get_str_def("model")?;
    let config = parser.get_str_def("config")?;
    let framework = parser.get_str_def("framework")?;

    let mut net: Net = dnn::read_net(&model, &config, &framework)?;
    let backend = parser.get_i32_def("backend")?;
    let _ = net.set_preferable_backend(backend);
    let _ = net.set_preferable_target(parser.get_i32_def("target")?);
    let out_names: core::Vector<String> = net.get_unconnected_out_layers_names()?;
    let mut cap = videoio::VideoCapture::default()?;//videoio::VideoCapture::new(0, CAP_GSTREAMER)?;
    if parser.has("input")? {
        let input = parser.get_str_def("input")?; //core::find_file_def(&file)?;
        let pipe = format!("filesrc location={input} ! decodebin ! videoconvert ! videoscale ! appsink");
        cap.open_file_def(&pipe)
    } else {
        cap.open_def(parser.get_i32_def("device")?)
    }?;

    if !cap.is_opened().unwrap() {
      return Err(Error::new(StsError, "cap is not opened"));
    }

    highgui::named_window_def("obj detection")?;

    let fq = Arc::new(RwLock::new(QueueFPS::<Mat>::new()));
    let fqtm = TickMeter::default().map(|t| Arc::new(Mutex::new(t)))?;
    
    let ffq = fq.clone();
    let f_fqtm = fqtm.clone();
    let frames = thread::spawn(move || loop {
        let _ = ffq.write()
          .map_err(|_| Error::new(StsError, "can't write to frames queue"))
          .and_then(|mut q| {
            let mut frame: Mat = Mat::default();
           
            cap.read(&mut frame)
                .ok()
                .filter(|x| identity(*x))
                .filter(|_| frame.size().ok().is_some_and(|s| s.width != 0))
                .map(|_| {
                  q.q.push_back(frame);
                  q.counter = q.counter + 1;
                  if q.counter > 1 {
                    let _ = f_fqtm.lock().and_then(|mut tm| {
                      let _ = tm.reset();
                      let _ = tm.start();
                      Ok(())
                    });
                  }
                  drop(q);
                  thread::sleep(Duration::from_millis(10));
                })
                .ok_or(Error::new(StsError, "can't read frame"))
        });
      }
    );

    let pfq = Arc::new(RwLock::new(QueueFPS::<Mat>::new()));
    let pdq = Arc::new(RwLock::new(QueueFPS::<core::Vector<Mat>>::new()));
    let pdqtm:Arc<Mutex<TickMeter>> = TickMeter::default().map(|t| Arc::new(Mutex::new(t)))?;

    let net_a = Arc::new(Mutex::new(net));
    let net_b = net_a.clone();

    let fpq = fq.clone();
    let p_pfq = pfq.clone();
    let p_pdq = pdq.clone();
    let p_pdqtm = pdqtm.clone();

    let processing = thread::spawn(move || loop {
        let mut blob: &mut Mat = &mut Mat::default();
        let p = fpq.write()
          .map_err(|_| Error::new(StsError, "can't access frames queue"))
          .and_then(|mut q| {
            match q.q.pop_front() {
              Some(mut frame) if !frame.empty() => {
                drop(q);
                let mut ne = net_b.lock().unwrap();
                let _ = preprocess(&mut blob, &mut frame, &mut ne, Size::new(inp_width, inp_height), scale.into(), mean, swap_rb);

                let _ = p_pfq.write()
                  .and_then(|mut q| {
                    q.q.push_back(frame);
                    q.counter = q.counter + 1;
                    if q.counter > 1 {
                      let _ = p_pdqtm.lock().and_then(|mut tm| {
                        let _ = tm.reset();
                        let _ = tm.start();
                        Ok(())
                      });
                    }
                    Ok(())
                  });

                p_pdq
                  .write()
                  .map_err(|_| Error::new(StsError, "can't access prediction queue"))
                  .and_then(|mut q| {
                    let mut outs: core::Vector<Mat> = Vec::new().into();
                    let _ = ne.forward(&mut outs, &out_names);
                    q.q.push_back(outs);
                    q.counter = q.counter + 1;
                    Ok(())
                  })
              },
              _ => Err(Error::new(StsError, "no frames")),
            }
          });

        match p {
          Ok(_) => thread::sleep(Duration::from_millis(20)),
          Err(_) => break,
        }
    });

    let net_c = net_a.clone();

    while highgui::wait_key(1)? < 0 {
        // println!("key?");
        let _ = pdq.write().and_then(|mut pq| {
          match pq.q.pop_front() {
            Some(outs) => {
              let _ = pfq.write()
                .and_then(|mut pdq| {
                  match pdq.q.pop_front() {
                    Some(mut frame) => {
                      let ne = net_c.lock().unwrap();
                      let _ = postprocess(&mut frame, &outs, &ne, backend, conf_threshold as f32, &mut classes, nms_threshold);

                      if pdq.counter > 1 {
                          let _ = fq.read().and_then(|fq| {
                            let mut fqtm = fqtm.lock().unwrap();
                            let _ = fqtm.stop();
                            

                            let label = format!("Camera: {:.2} FPS", fq.counter as f64 / fqtm.get_time_sec().unwrap());
                            let _ = fqtm.start();
                            let _ = imgproc::put_text_def(&mut frame, &label, Point::new(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, core::Scalar::new(0., 0., 255., 0.));

                            let mut pdqtm = pdqtm.lock().unwrap();
                            let _ = pdqtm.stop();
                            let label = format!("Network: {:.2} FPS", pdq.counter as f64 / pdqtm.get_time_sec().unwrap());
                            let _ = pdqtm.start();
                            let _ = imgproc::put_text_def(&mut frame, &label, Point::new(0, 30), FONT_HERSHEY_SIMPLEX, 0.5, core::Scalar::new(0., 0., 255., 0.));

                            let label = format!("Skipped frames: {:?}", fq.counter - pdq.counter);
                            let _ = imgproc::put_text_def(&mut frame, &label, Point::new(0, 45), FONT_HERSHEY_SIMPLEX, 0.5, core::Scalar::new(0., 0., 255., 0.));
                            Ok(())
                          }).map_err(|_| Error::new(StsError, "no frames queue"));
                      }
                      let _ = highgui::imshow("obj detection", &frame);
                      Ok(())
                    },
                    None => Ok(()),
                  }
                });
            },
            None => (),
          };
          Ok(())
        });
    }
    println!("kjks");
    let _ = frames.join();
    let _ = processing.join();
    Ok(())
  } else {
    Err(Error::new(StsError, "Cannot find a model"))
  }
}
