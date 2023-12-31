#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/Image.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <ackermann_msgs/AckermannDrive.h>
#include "torch/torch.h"
#include "torch/script.h"
#include "car/CarObject.h"
#include "car/CarObjects.h"

using namespace std;
using namespace cv;
torch::Device device(torch::kCUDA);
class LaneFollower{
    private:
    ros::NodeHandle n;
    image_transport::ImageTransport image_transport;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    ros::Publisher drive_pub;
    torch::jit::script::Module semanticSegmentationModule;
    torch::jit::script::Module laneFollowModule;
    torch::jit::script::Module speedControllerModule;
    ros::Publisher object_detection_pub;
    int camera_height,camera_width;
    int desiredX = 256,desiredY=256;
    int outputX = 1280, outputY = 720;
    bool fastMode;
    public:
    LaneFollower():image_transport(n) {
        n = ros::NodeHandle("~");
         n.getParam("/camera_height",camera_height);
         n.getParam("/camera_width",camera_width);
         std::string drive_topic, camera_topic, semanticSegmentationPath, laneFollowPath, imageOutput, object_detection_topic,speedControllerPath;
         n.getParam("/in_fast_mode",fastMode);
         n.getParam("/camera_topic",camera_topic);
        image_sub_ = image_transport.subscribe(camera_topic, 1, &LaneFollower::image_callback, this);
         n.getParam("/nav_drive_topic", drive_topic);
        drive_pub = n.advertise<ackermann_msgs::AckermannDriveStamped>(drive_topic, 10);
        n.getParam("/semantic_segmentation_path",semanticSegmentationPath);
        n.getParam("/lane_follow_path",laneFollowPath);
        n.getParam("/speed_controller_path",speedControllerPath);
        n.getParam("/image_output",imageOutput);
        image_pub_ = image_transport.advertise(imageOutput, 1);
        n.getParam("/object_detection_topic",object_detection_topic);
        object_detection_pub = n.advertise<car::CarObjects>(object_detection_topic,10);
        // Load pytorch model
        try {
          semanticSegmentationModule = torch::jit::load(semanticSegmentationPath);
          semanticSegmentationModule.to(device);
          semanticSegmentationModule.eval();
          laneFollowModule = torch::jit::load(laneFollowPath);
          laneFollowModule.to(device);
          laneFollowModule.eval();
          speedControllerModule = torch::jit::load(speedControllerPath);
          speedControllerModule.to(device);
          speedControllerModule.eval();
          }
          catch (const c10::Error& e) {
            ROS_ERROR("Error Loading Model: %s", e.what());
          }

    }
    ~LaneFollower()
  {
  }
    
  void image_callback(const sensor_msgs::ImageConstPtr& msg){
      cv_bridge::CvImagePtr cv_ptr;
      Mat outputMat;
      double steeringAngle = 0;
      double carSpeed = 0;//1,75
      car::CarObjects objectsDetected;
      bool foundCar = false;
      long minX= 256;
      long maxX=0;
      long minY = 256;
      long maxY = 0;
    try
    {
      float time1 = std::clock();
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGRA8);
      Mat imageInRGB;
      cvtColor(cv_ptr->image,imageInRGB,COLOR_RGBA2RGB);

      //PREPROCESSING STEP
      Mat resolutionReduced;
      cv::resize(imageInRGB,resolutionReduced,Size(desiredX,desiredY));
      //DATA FORMAT FOR INPUT TO NETWORK
      std::vector<torch::jit::IValue> inputs;
      torch::Tensor tensor_image = torch::from_blob(resolutionReduced.data, {1,resolutionReduced.rows, resolutionReduced.cols, resolutionReduced.channels() }, at::kByte).to(device).toType(c10::kFloat);
      tensor_image = tensor_image.permute({ 0,3,1,2 });
      inputs.push_back(tensor_image);
      float time2 = std::clock();
      //PREDICTION FROM SEMANTIC SEGMENTATION NETWORK
      at::Tensor output = semanticSegmentationModule.forward(inputs).toTensor().argmax(1);
      at::Tensor cpuTensor = output.to("cpu");
      float time3 = std::clock();
      //2 = car
      //3 = yellow line
      auto tensor_accessor = cpuTensor.accessor<long,3>();
      for(int i =0; i<tensor_accessor.size(0);i++){
        for(int j= 0;j<tensor_accessor.size(1);j++){
          for(int k =0;k<tensor_accessor.size(2);k++){
            long pixelValue = tensor_accessor[i][j][k];
            if(pixelValue ==2){
              foundCar = true;
              if(j<minY){
                minY=j;
              }
              if(j>maxY){
                maxY=j;
              }
              if(k<minX){
                minX=k;
              }
              if(k>maxX){
                maxX=k;
              }
            }
          }
        }
      }
      //Steering Angle
      std::vector<torch::jit::IValue> inputsToLaneFollow;
      inputsToLaneFollow.push_back(output.toType(c10::kFloat).unsqueeze(0));
      at::Tensor outputSteeringAngle = laneFollowModule.forward(inputsToLaneFollow).toTensor();
      steeringAngle = outputSteeringAngle[0].item().toDouble();
      //Speed
      std::vector<torch::jit::IValue> inputsToSpeedControl;
      inputsToSpeedControl.push_back(output.toType(c10::kFloat).unsqueeze(0));
      at::Tensor outputSpeed = speedControllerModule.forward(inputsToSpeedControl).toTensor();
      carSpeed = outputSpeed[0].item().toDouble();
      float time4 = std::clock();
      ROS_INFO("Time Between Cycles 2-1: %s",std::to_string(time2-time1).c_str());
      ROS_INFO("Time Between Cycles 3-2: %s",std::to_string(time3-time2).c_str());
      ROS_INFO("Time Between Cycles 4-3: %s",std::to_string(time4-time3).c_str());
      //PREPARE FOR CAMERA OUTPUT
      //DISABLE WHEN RUNNING EXPERIMENTS
      at::Tensor prediction = output.permute({1,2,0});
      prediction=prediction.mul(80).clamp(0,255).to(torch::kU8).to(torch::kCPU); // add .mul(80) for visualization
      Mat predictionMat(cv::Size(desiredX, desiredY), CV_8UC1, prediction.data_ptr<uchar>());
      cv::resize(predictionMat,outputMat,Size(outputX,outputY));
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    ackermann_msgs::AckermannDriveStamped drive_st_msg;
    ackermann_msgs::AckermannDrive drive_msg;
    if(fastMode){
        if(carSpeed>1.25){
          carSpeed=1.25;
        }
        if(carSpeed<0){
          carSpeed = 0;
        }
        if(carSpeed<=1.0){
          steeringAngle = steeringAngle / 0.75 * 0.9;
        }
        drive_msg.steering_angle = steeringAngle*-0.75;
        carSpeed = carSpeed+0.75;
        drive_msg.speed = carSpeed;
    }else{
        drive_msg.steering_angle = steeringAngle*-1 * 0.8; // was * 0.8 for normal speed
        drive_msg.speed = carSpeed;
    }
    //TEMP SLOW DOWN
    drive_msg.speed = 0.8;
    ROS_INFO("Steering Angle %s",std::to_string(steeringAngle).c_str());
    ROS_INFO("Speed %s",std::to_string(carSpeed).c_str());
    drive_st_msg.drive = drive_msg;
    drive_pub.publish(drive_st_msg);

    //PUBLISH OBJECT DETECTION DATA
    //TEST DATA
    car::CarObject testObject;
    if(foundCar){
      testObject.classID = 2;
      testObject.minX = minX;
      testObject.maxX = maxX;
      testObject.minY = minY;
      testObject.maxY = maxY;
      objectsDetected.objects.push_back(testObject);
    }
    object_detection_pub.publish(objectsDetected);

    //Upload Mat Image as ROS TOPIC 
    cv_bridge::CvImage out_msg;
    out_msg.header   = msg->header; 
    out_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC1; 
    out_msg.image    = outputMat;

    // Output modified video stream
    image_pub_.publish(out_msg.toImageMsg());
  }
};
int main(int argc, char ** argv) {
    ros::init(argc, argv, "lane_follower");
    LaneFollower rw;
    ros::spin();
    return 0;
}