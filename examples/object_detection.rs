use std::env;

use opencv::core::{CommandLineParser, Point, Rect, Size, StsNotImplemented, StsError, TickMeter};
use opencv::prelude::*;
use opencv::{core, dnn, highgui, imgproc, videoio, Error, Result};
use opencv::core::{CV_8U, Vector};
// use opencv::imgproc::{FONT_HERSHEY_SIMPLEX,COLOR_BGR2RGB};
use opencv::dnn::{Net,DNN_BACKEND_OPENCV};
use opencv::boxed_ref::{BoxedRef};
//use videoio::{CAP_FFMPEG,CAP_GSTREAMER};
use std::collections::{VecDeque, BTreeMap};
use std::thread;
use std::sync::{Arc,RwLock,Mutex};
use std::borrow::Cow;
use std::sync::mpsc;
use std::ops::DerefMut;

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

    // try resize here
    //let _ = imgproc::cvt_color_def(blob, frame, COLOR_BGR2RGB);
    //let _ = imgproc::resize_def(blob, frame, core::Size::new(408, 408));

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
                _classes: &mut Vec<String>,
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
                        left   = (data[i + 3] * frame.cols() as f32) as i32;
                        top    = (data[i + 4] * frame.rows() as f32) as i32;
                        right  = (data[i + 5] * frame.cols() as f32) as i32;
                        bottom = (data[i + 6] * frame.rows() as f32) as i32;
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
              let c: f32= 100.; // magic number,// all the data is kind of 0.00
              let center_x: i32 = (data[0] * frame.cols() as f32 * c) as i32;
              let center_y: i32 = (data[1] * frame.rows() as f32 * c) as i32;
              let width: i32 = (data[2] * frame.cols() as f32 * c) as i32;
              let height: i32 = (data[3] * frame.rows() as f32 * c) as i32;
              let left: i32 = center_x - width / 2;
              let top: i32 = center_y - height / 2;
              let bo = Rect::new(left, top, width, height);
              class_ids.push(class_id_point.x.try_into().unwrap());
              confidences.push(confidence as f32);
              boxes.push(bo);
            }
          }
        }

    }  else {
      return Err(Error::new(StsNotImplemented, "Unknown output layer type: ".to_owned() + &out_layer_type));
    }

    // NMS is used inside Region layer only on DNN_BACKEND_OPENCV 
    //for another backends we need NMS in sample or NMS is required if number of outputs > 1
    if out_layers.len() > 1 || (out_layer_type == "Region" && backend != DNN_BACKEND_OPENCV) {
        let init: BTreeMap<i32, Vec<usize>> = BTreeMap::from_iter(class_ids.iter().enumerate().map(|(_,c)|(*c as i32, Vec::<usize>::new())));
        let class_it = class_ids.iter().enumerate().map(|(pos,class)| (*class as i32, pos));
        let class2indices: BTreeMap<i32, Vec<usize>> = class_it.fold(
          init, 
            |mut acc: BTreeMap<i32, Vec<usize>>, (k,class)| {
              acc.entry(k).and_modify(|v| v.push(class));
              acc
          });

        let mut nms_boxes : Vec<Rect> = Vec::new();
        let mut nms_confidences: Vec<f32> = Vec::new();
        let mut nmsclass_ids: Vec<usize> = Vec::new();

        for (k, class_indices) in class2indices {
          let mut nms_indices: Vector<i32> = Vector::new();
          let (local_boxes, local_confidences) = boxes.iter()
            .zip(confidences.iter())
            .enumerate()
            .filter(|(pos,_)| class_indices.contains(pos))
            .map(|(_,(x,y))| (*x, *y))
            .unzip();
          
          let _ = opencv::dnn::nms_boxes_def(
               &local_boxes, 
               &local_confidences, 
               conf_threshold as f32,
               nms_threshold, 
               &mut nms_indices);

          for idx in nms_indices.into_iter() {
            nms_boxes.push(local_boxes.get(idx as usize)?);
            nms_confidences.push(local_confidences.get(idx as usize)?);
            nmsclass_ids.push(k as usize);
          }
        }

        boxes = nms_boxes;
        class_ids = nmsclass_ids;
        confidences = nms_confidences;
    }

    boxes.into_iter().for_each(|box0| {
      let _ = imgproc::rectangle_def(frame, box0, (0., 255., 0.).into());
    });

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

    let net_a = Arc::new(Mutex::new(net));
    let net_b = net_a.clone();
    let net_c = net_a.clone();

    type Marc<'a> = Cow<'a, Arc<RwLock<Mat>>>;
    let qc = VecDeque::<Marc>::new();
    let aqc = Arc::new(RwLock::new(qc));
    let raqc = aqc.clone();
    let paqc = aqc.clone();

    #[derive(Debug)]
    enum Msg {
      Ack,
      Input,
      Forward,
    }

    let (tx,rx) = mpsc::channel::<Msg>();

    let pdq = Arc::new(RwLock::new(VecDeque::<core::Vector<Mat>>::new()));
    let p_pdq = pdq.clone();

    let tx = tx.clone();
    let _ = thread::spawn(move || loop {
      let mut frame: Mat = Mat::default();
      let _ = cap.read(&mut frame).ok()
        .map(|_| {
          let _ = raqc.write()
            .and_then(|mut q| {
              let farc = Arc::new(RwLock::new(frame));
              let cowf = Cow::Owned(farc);

              q.push_back(cowf);

              let _ = tx.send(Msg::Ack);
              Ok(())
            });
        });

    });

    let (tx0, rx0) = mpsc::channel::<Msg>();
    let tx01 = tx0.clone();

    let _ = thread::spawn(move || loop {
      match rx.recv() {
        Ok(Msg::Ack) => {
          let _ = paqc.write().and_then(|mut q| {
            match q.back_mut() {
              Some(lock) => {
                let _ = lock.write().and_then(|mut frame| {
                  let mut ne = net_b.lock().unwrap();
                  let mut blob: &mut Mat = &mut Mat::default();
                  let _ = preprocess(&mut blob, &mut frame, &mut ne, Size::new(inp_width, inp_height), scale.into(), mean, swap_rb);
                  let _ = tx01.send(Msg::Input);
                  Ok(())
                });
              },
              None => (),
            }
            Ok(())
          });
        },
        Ok(_) => (),
        Err(_)  => (), 
      }
    });

    let (tx1, rx1) = mpsc::channel::<Msg>(); 
    let tx1 = tx1.clone();

    let rx01 = Arc::new(Mutex::new(rx0));
    let rx02 = rx01.clone();

    let _ = thread::spawn(move || loop {
      match rx02.lock().unwrap().recv() {
        Ok(Msg::Input) => {
          let mut ne = net_c.lock().unwrap();
          let mut outs: core::Vector<Mat> = Vec::new().into();
          let _ = ne.forward(&mut outs, &out_names);
          
          let _ = p_pdq.write()
            .map_err(|_| Error::new(StsError, "can't access prediction queue"))
            .and_then(|mut q| Ok(q.push_back(outs)) );

          let _ = tx1.send(Msg::Forward);
        },
        Ok(_) => (),
        Err(_) => (),
      }

    });

    loop {
      let key = highgui::wait_key(1)?;
      match rx1.try_recv() {
        Ok(Msg::Forward) => {
          let _ = aqc.write().and_then(|mut q| {
            match q.pop_front() {
              Some(mut cow) => {
                let mut guard = cow.to_mut().write().unwrap();
                let f1: &mut Mat = guard.deref_mut();

                let outs: core::Vector<Mat> = match pdq.write() {
                  Ok(mut q) => q.pop_front().unwrap_or(Vec::new().into()),
                  Err(_) => Vec::new().into(),
                };
                let ne = net_a.lock().unwrap();
                let _ = postprocess(f1, &outs, &ne, backend, conf_threshold as f32, &mut classes, nms_threshold);
                let _ = highgui::imshow("obj detection", f1);
              },
              None => (),
            }
            Ok(())
          });
        },
        Ok(_) => (),
        _ => (),
      }

      if key > 0 {
        break;
      }
    }
    Ok(())
  } else {
    Err(Error::new(StsError, "Cannot find a model"))
  }
}
