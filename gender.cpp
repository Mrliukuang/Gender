#include <string>
#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <Magick++.h>
#include <caffe/blob.hpp>
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace caffe;

class GenderPredictor {
public:
    shared_ptr<Net<float> > net;
    int nInputChannels;
    Size inputSize;
    Mat inputMean;
    int nImages;  // # of faces to classify
    int deviceID; // -1 for CPU mode, others for GPU ID

    vector<vector<float> > boys;  // boys feature
    vector<vector<float> > girls;  // girls feature

    GenderPredictor();
    void initModel(const string& prototxt, const string& binary, const string& meanfile);
    void loadMeanFile(const string& meanfile);
    Mat cropImage(const Mat& im, const int xmin, const int ymin, const int xmax, const int ymax);
    pair<int,vector<float> > predict(const vector<Mat>& images);
    void loadFeatures(const string&, const string&);
    Mat decodeImage(const char* inbuff,const int insize);
    Mat loadImageFromBinary(const string& file);
    ~GenderPredictor();

private:
    void warpInputLayer(vector<Mat>* inputChannels);
    void preprocess(const vector<Mat>& images, vector<Mat>* inputChannels);
    vector<vector<float> > loadFeature(const string& filename);
    float getDistance(const vector<float>& a, const vector<float>& b);
    vector<float> getMatch(const vector<float>& v, const vector<vector<float> >& group);
};

GenderPredictor::GenderPredictor():nImages(1),deviceID(-1) {}

GenderPredictor::~GenderPredictor() {}

void GenderPredictor::initModel(const string& prototxt, const string& binary, const string& meanfile) {
    /* Set mode GPU/CPU */
    if (deviceID == -1) {
        cout << "Seting mode CPU" << endl;
        Caffe::set_mode(Caffe::CPU);
    } else {
        cout << "Seting mode GPU" << endl;
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(deviceID);
    }

    /* Load the model */
    cout << "Loading model..." << endl;
    net.reset(new Net<float>(prototxt, TEST));
    net->CopyTrainedLayersFrom(binary);

    Blob<float>* inputLayer = net->input_blobs()[0];
    nInputChannels = inputLayer->channels();
    CHECK(nInputChannels == 3 || nInputChannels == 1)
        << "Input layer should have 1 or 3 channels.";

    inputSize = Size(inputLayer->width(), inputLayer->height());
    cout << "Model loaded." << endl;

    /* Load mean file */
    cout << "Loading mean file..." << endl;
    loadMeanFile(meanfile);
    cout << "Mean file loaded." << endl;
}

void GenderPredictor::loadMeanFile(const string& meanfile) {
    BlobProto blobProto;
    ReadProtoFromBinaryFileOrDie(meanfile.c_str(), &blobProto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> meanBlob;
    meanBlob.FromProto(blobProto);
    CHECK_EQ(meanBlob.channels(), nInputChannels)
        << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    vector<Mat> channels;
    float* data = meanBlob.mutable_cpu_data();
    for (int i = 0; i < nInputChannels; ++i) {
        // extract individual channel
        Mat channel(meanBlob.height(), meanBlob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += meanBlob.height()*meanBlob.width();
    }

    /* Merge the separate channels into a single image. */
    Mat meanMat;
    merge(channels, meanMat);

    /* Compute the global mean pixel value and create a mean image filled with this value. */
    Scalar channelMean = cv::mean(meanMat);
    inputMean = Mat(inputSize, meanMat.type(), channelMean);
    cout << inputMean.size() << endl;
}

void GenderPredictor::warpInputLayer(vector<Mat>* inputChannels) {
    Blob<float>* inputLayer = net->input_blobs()[0];
    int width = inputLayer->width();
    int height = inputLayer->height();
    float* data = inputLayer->mutable_cpu_data();
    for (int i = 0; i < nImages*inputLayer->channels(); ++i) {
        Mat channel(height, width, CV_32FC1, data);
        inputChannels->push_back(channel);
        data += width * height;
    }
}

void GenderPredictor::preprocess(const vector<Mat>& images, vector<Mat>* inputChannels) {
    /* Convert the input image to the input image format of the network. */
    for (int i = 0; i < nImages; ++i) {
        Mat im = images[i];
        Mat m;
        /* Convert color */
        if (im.channels() == 3 && nInputChannels == 1)
          cv::cvtColor(im, m, cv::COLOR_BGR2GRAY);
        else if (im.channels() == 4 && nInputChannels == 1)
          cv::cvtColor(im, m, cv::COLOR_BGRA2GRAY);
        else if (im.channels() == 4 && nInputChannels == 3)
          cv::cvtColor(im, m, cv::COLOR_BGRA2BGR);
        else if (im.channels() == 1 && nInputChannels == 3)
          cv::cvtColor(im, m, cv::COLOR_GRAY2BGR);
        else
          m = im;

        /* Resize */
        if (m.size() != inputSize) {
            cout << "resizing..." << endl;
            resize(m, m, inputSize);
        }

        /* Convert to float */
        cout << "converting to float..." << endl;
        m.convertTo(m, CV_32FC3);

        /* Zero mean */
        cout << "normalzing..." << endl;
        subtract(m, inputMean, m);

        /* This operation will write the separate BGR planes directly to the
          * input layer of the network because it is wrapped by the cv::Mat
          * objects in input_channels. */
        cout << "spliting..." << endl;
        //split(m, *inputChannels);
        vector<Mat> channels;
        split(m, channels);
        // NOTE: cannot use assignment operator of cv::Mat to copy data to input layer
        for (int j = 0; j < channels.size(); j++)
            channels[j].copyTo((*inputChannels)[i*nInputChannels+j]);
    }

    CHECK(reinterpret_cast<float*>(inputChannels->at(0).data)
            == net->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}

pair<int,vector<float> > GenderPredictor::predict(const vector<Mat>& images) {
    // return <gender, match_score>, for nImages = 1
    Blob<float>* inputLayer = net->input_blobs()[0];
    inputLayer->Reshape(nImages, nInputChannels, inputSize.height, inputSize.width);

    /* Forward dimension change to all layers. */
    net->Reshape();

    vector<Mat> inputChannels;
    warpInputLayer(&inputChannels);

    preprocess(images, &inputChannels);

    // For old version caffe in faster-RCNN
	net->ForwardPrefilled();
	// For new version caffe
  	//net->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* outputLayer = net->output_blobs()[0];
    const float* begin = outputLayer->cpu_data();
   //     const float* end = begin + outputLayer->channels();
    const float* end = begin + nImages*outputLayer->channels();

    vector<float> outputs = vector<float>(begin, end);
    // for (int i = 0; i < outputs.size(); ++i) {
    //     cout << outputs[i] << endl;
    // }
    int gender = (outputs[0] > outputs[1]) ? 0 : 1; // 0=Male, 1=Female

    //获取指定层layer_name的特征
    string layerName = "pool5/7x7_s1";
    const float* be1 = net->blob_by_name(layerName)->cpu_data();
    //const float* en2 = be1 + net->blob_by_name(layerName)->channels();
    const float* en2 = be1 + nImages*net->blob_by_name(layerName)->channels();
    //将特征转为为vect
    vector<float> feature(std::vector<float>(be1,en2));

    vector<float> match_score;
    if (gender==0) {
        // Male
        match_score = getMatch(feature,  girls);
    } else {
        // Female
        match_score = getMatch(feature, boys);
    }
    // 调整match_score:把最大的score调整到90+,其余score同等放大
    vector<float>::iterator max_elem = std::max_element(match_score.begin(), match_score.end());
    float max_score = *max_elem;
    int delta = 90-int(max_score*10)*10;
    if (delta > 0) {
        for (size_t i = 0; i < match_score.size(); i++) {
            match_score[i] = match_score[i]*100.0 + delta;
        }
    }
    // And finally the match_score is in range [0,100]
    return make_pair(gender, match_score);
}

Mat GenderPredictor::loadImageFromBinary(const string& imPath) {
    //Begin read file
    FILE *fp;
    int nFileLen;
    fp = fopen(imPath.c_str(), "rb");
    if(fp == NULL){
      printf("open failed\n");
    }
    fseek(fp, 0, SEEK_END);
    nFileLen = ftell(fp);
    fclose(fp);

    //Read binary flow
    char* valueContent;
    valueContent = (char*)calloc(nFileLen, sizeof(char));
    fp = fopen(imPath.c_str(), "rb");
    fread(valueContent,nFileLen,sizeof(char),fp);
    fclose(fp);

    //Reformat image and getFeature
    Mat im = decodeImage(valueContent, nFileLen);
    // IplImage * im_p = Mem2CvImg(valueContent, nFileLen);
    // cv::Mat img = cv::Mat(im_p);
    free(valueContent);
    CHECK(!im.empty()) << "Unable to decode image " << imPath;
	cout << im.size() << endl;
    return im;
}

Mat GenderPredictor::cropImage(const Mat& im, const int xmin, const int ymin, const int xmax, const int ymax) {
    // Crop a 1.5x face region
    int H = im.rows;
    int W = im.cols;

    int w = xmax+1-xmin;
    int h = ymax+1-ymin;

    if (xmin+w > W || ymin+h > H || xmin > xmax || ymin > ymax) {
        cout << "Invalid coordinates" << endl;
        return Mat();
    }

    int xmin2 = std::max(0.0, xmin-0.25*w);
    int ymin2 = std::max(0.0, ymin-0.25*h);
    int xmax2 = std::min(W-1.0, xmax+0.25*w);
    int ymax2 = std::min(H-1.0, ymax+0.25*h);

    Rect rect(xmin2, ymin2, xmax2-xmin2+1, ymax2-ymin2+1);
    return im(rect);
}

vector<vector<float> > GenderPredictor::loadFeature(const string& filename) {
    vector<vector<float> > features;

    ifstream f;
	// For old version gcc, need to add c_str()
    f.open(filename.c_str());
    if (!f.is_open()) {
        cout << filename << " loading error!";
        return vector<vector<float> >();
	}

    for (size_t i = 0; i < 5; i++) {
        vector<float> feature;
        for (size_t j = 0; j < 1024; ++j) {
            float a;
            f >> a;
            feature.push_back(a);
        }
        features.push_back(feature);
    }
    f.close();

    return features;
}

void GenderPredictor::loadFeatures(const string& boysPath, const string& girlsPath) {
    boys = loadFeature(boysPath);
    girls = loadFeature(girlsPath);
}

float GenderPredictor::getDistance(const vector<float>& a, const vector<float>& b) {
    float d = 0;
    for (int i = 0; i < a.size(); ++i)
        d += (a[i]-b[i])*(a[i]-b[i]);
    return d;
}

vector<float> GenderPredictor::getMatch(const vector<float>& v, const vector<vector<float> >& group) {
    vector<float> match_score(group.size());
    for (int i = 0; i < group.size(); ++i) {
        match_score[i] = 1/(getDistance(v, group[i])+1e-6);
    }

    float sum = std::accumulate(match_score.begin(), match_score.end(), 0.0);
    for (int i = 0; i < match_score.size(); ++i) {
        match_score[i] /= sum;
    }

    //vector<float>::iterator max_elem = max_element(match_score.begin(), match_score.end());
    return match_score;
}

// Convert image binary flow to Mat
Mat GenderPredictor::decodeImage(const char* inbuff,const int insize) {
    Magick::Image image;
    Magick::Blob blob(inbuff, insize);
    image.read(blob);
    image.magick("JPEG");
    Magick::Blob blob2;
    image.write(&blob2);
    char* data = (char*)blob2.data();
    int len = blob2.length();
    vector<uchar> buff(data, data+len);
    cv::Mat im = imdecode(cv::Mat(buff),CV_LOAD_IMAGE_COLOR);
    if(im.empty()) {
        std::cout<<"img error"<<std::endl;
    }
    return im;
}

// Split string into vector
vector<string> split(const string& s, const char sep) {
    // make sure there is no extra spaces front and end.
    vector<string> splited;
    size_t left = 0, right = 0;
    while (right != string::npos) {
        right = s.find(sep, left);
        splited.push_back(s.substr(left, right - left));
        left = right + 1;
    }
    return splited;
}

// Split coordinates (xmin,ymin, xmax, ymax) into vector
vector<int> splitXYs(const string& s, const char sep) {
    vector<int> splited;
    size_t left = 0, right = 0;
    while (right != string::npos) {
        right = s.find(sep, left);
        string sub = s.substr(left, right - left);
        splited.push_back(atoi(sub.c_str()));
        left = right + 1;
    }
    return splited;
}

int main() {
    // Model 1
    //    string prototxt = "./models/deploy_gender.prototxt";
    //    string binary = "./models/gender_net.caffemodel";
    //    string meanfile = "./models/mean.binaryproto";

    // Model 2
    string prototxt = "./models/deep_detect/deploy.prototxt";
    string binary = "./models/deep_detect/model_iter_30000.caffemodel";
    string meanfile = "./models/deep_detect/mean.binaryproto";

    // Define Gender Predictor
    GenderPredictor predictor;
    // Set GPU ID, default using CPU
    predictor.deviceID = 0;
    // Set the number of predicting images, default is 1
    predictor.nImages = 1;
    // Load Model
    predictor.initModel(prototxt, binary, meanfile);
    // Load features of main characters
    string boysPath = "./boys.txt";
    string girlsPath = "./girls.txt";
    predictor.loadFeatures(boysPath, girlsPath);


    /* Read the original image from binary */
    //   Mat im = imread("./mm.jpg");

    // TEST
    string imPath = "/search/data/user/liukuang/dataset/VOC/VOCdevkit2007/VOC2007/JPEGImages/";

    ifstream file("xys.txt");
    string line;
    //string line = "facescrub_img_65762 65 46 163 143";
    while (getline(file, line)) {
        cout << line << endl;
        vector<string> splited = split(line, ' ');

        string imName = splited[0]+".jpg";
        int xmin = atoi(splited[1].c_str());
        int ymin = atoi(splited[2].c_str());
        int xmax = atoi(splited[3].c_str());
        int ymax = atoi(splited[4].c_str());

        try {
            Mat im = predictor.loadImageFromBinary(imPath+imName);

            Mat face = predictor.cropImage(im, xmin, ymin, xmax, ymax);
            CHECK(!face.empty()) << "Invalid coordinates." << face;

            // Predict, support predict multiple images in parallel.
            vector<Mat> faces;
            faces.push_back(face);
            pair<int, vector<float> > rlt = predictor.predict(faces);
            if (rlt.first == 0) // Male
                rectangle(im, Point(xmin, ymin), Point(xmax, ymax), Scalar(255, 0, 0), 2);
            else    // Female
                rectangle(im, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);

            imwrite("./pics/"+imName, im);
        } catch(exception& e) {
            cout << "Exception!" << endl;
            continue;
        }
    }


    // // Read image from binary flow
    // string file = "/search/data/user/liukuang/workspace/HappySong/mm.jpg";
    // FILE *fp;
    // int nFileLen;
    // fp = fopen(file.c_str(), "rb");
    // if(fp == NULL){
    //     printf("open failed\n");
    // }
    // fseek(fp, 0, SEEK_END);
    // nFileLen = ftell(fp);
    // fclose(fp);
    //
    // // Read binary flow
    // char* valueContent;
    // valueContent = (char*)calloc(nFileLen, sizeof(char));
    // fp = fopen(file.c_str(), "rb");
    // fread(valueContent,nFileLen,sizeof(char),fp);
    // fclose(fp);
    //
    // Mat im = predictor.decodeImage(valueContent, nFileLen);
    // CHECK(!im.empty()) << "Unable to decode the image." << im;
    // cout << im.size() << endl;
    // free(valueContent);
    //
    // // Get the face coordinates
    // string coordinates = "100,100,200,200";  // xmin, ymin, xmax, ymax
    // // int xmin = 100;
    // // int ymin = 100;
    // // int xmax = 200;
    // // int ymax = 200;
    // CHECK(!coordinates.empty()) << "Invalid coordinates." << coordinates;
    //
    // vector<int> xys = splitXYs(coordinates, ",");
    // for (size_t i = 0; i < xys.size(); i++) {
    //     cout << xys[i] << " ";
    // }
    //
    // // Mat face = predictor.cropImage(im, xmin, ymin, xmax, ymax);
    // Mat face = predictor.cropImage(im, xys[0], xys[1], xys[2], xys[3]);
    // CHECK(!face.empty()) << "Invalid coordinates." << face;
    //
    // // Predict, support predict multiple images in parallel.
    // vector<Mat> faces;
    // faces.push_back(face);
    // pair<int, vector<float> > rlt = predictor.predict(faces);
    //
    // cout << rlt.first << endl;
    // vector<float> match_score = rlt.second;
    // for (size_t i = 0; i < match_score.size(); i++) {
    //     cout << match_score[i] << " ";
    // }
}
