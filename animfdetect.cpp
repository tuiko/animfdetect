#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#pragma comment(lib, "comctl32.lib")

// VC10�p�@�K�X����������

#ifdef _DEBUG
#pragma comment(lib, "opencv_core248d.lib")
#pragma comment(lib, "opencv_highgui248d.lib")
#pragma comment(lib, "opencv_imgproc248d.lib")
#pragma comment(lib, "opencv_objdetect248d.lib")

#pragma comment(lib, "libpngd.lib")
#pragma comment(lib, "zlibd.lib")
#pragma comment(lib, "libtiffd.lib")
#pragma comment(lib, "libjpegd.lib")
#pragma comment(lib, "libjasperd.lib")
#pragma comment(lib, "IlmImfd.lib")
#else
#pragma comment(lib, "opencv_core248.lib")
#pragma comment(lib, "opencv_highgui248.lib")
#pragma comment(lib, "opencv_imgproc248.lib")
#pragma comment(lib, "opencv_objdetect248.lib")

#pragma comment(lib, "libpng.lib")
#pragma comment(lib, "zlib.lib")
#pragma comment(lib, "libtiff.lib")
#pragma comment(lib, "libjpeg.lib")
#pragma comment(lib, "libjasper.lib")
#pragma comment(lib, "IlmImf.lib")
#endif

char window_name[]="animfdetect";
cv::CascadeClassifier cascade;
cv::Mat gray, smallImg, img, showimg;
int trackbar1 = 0;
double scaleFactor = 1.05;
int minNeighbors = 1;
double scale = 4;

void facedetect()
{
	scaleFactor = 1.5 - (trackbar1/255.0*0.5f-0.01);
	std::vector<cv::Rect> faces;
	cascade.detectMultiScale(smallImg, faces,
		scaleFactor, minNeighbors,
		CV_HAAR_SCALE_IMAGE);

	// ���ʂ̕`��
	std::vector<cv::Rect>::const_iterator r = faces.begin();
	for(; r != faces.end(); ++r) {
		cv::rectangle( showimg, r->tl()*scale,r->br()*scale, cv::Scalar(80,80,255), 2, CV_AA );
	}

}

// �C�x���g�n���h��
void onChange1(int val, void* ptr)
{	
	showimg = cv::Mat(img.clone());
	facedetect();

	cv::imshow(window_name, showimg);
}

int main(int argc, char *argv[])
{
	const char *imagename = argc > 1 ? argv[1] : "37467262_p0.jpg";
	img = cv::imread(imagename, 1);
	if(img.empty()) return -1; 

	showimg = cv::Mat(img.clone());

	smallImg = cv::Mat(cv::saturate_cast<int>(img.rows/scale), cv::saturate_cast<int>(img.cols/scale), CV_8UC1);
	// �O���[�X�P�[���摜�ɕϊ�
	cv::cvtColor(img, gray, CV_BGR2GRAY);
	// �������ԒZ�k�̂��߂ɉ摜���k��
	cv::resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
	cv::equalizeHist(smallImg, smallImg);

	// ���ފ�̓ǂݍ���
	std::string cascadeName = "lbpcascade_animeface.xml";
	if(!cascade.load(cascadeName)) return -1;
	
	std::vector<cv::Rect> faces;
	
	/// �}���`�X�P�[���i��j�T��xo
	// �摜�C�o�͋�`�C�k���X�P�[���C�Œ��`���C�i�t���O�j�C�ŏ���`
	cascade.detectMultiScale(smallImg, faces,
		1.05, 1,
		CV_HAAR_SCALE_IMAGE);

	// ���ʂ̕`��
	std::vector<cv::Rect>::const_iterator r = faces.begin();
	for(; r != faces.end(); ++r) {
		cv::rectangle( showimg, r->tl()*scale,r->br()*scale, cv::Scalar(80,80,255), 2, CV_AA );
	}

	cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);

	cv::createTrackbar("scaleFactor", window_name, &trackbar1, 255, onChange1, 0);
	cv::createTrackbar("minNeighbors", window_name, &minNeighbors, 10, onChange1, 0);

	cv::imshow( window_name, showimg );
	cv::waitKey(0);
}