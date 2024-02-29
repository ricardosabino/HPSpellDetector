using OpenCvSharp;
using OpenCvSharp.ML;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using static OpenCvSharp.LineIterator;

namespace HP
{
    class ImageProcessor
    {
        const int DEQUE_BUFFER_SIZE = 40;
        const int TRACE_THICKNESS = 4;
        const int BGS_HISTORY_FRAMES = 200;
        const int TRAINER_IMAGE_WIN_SIZE = 64;


        const int MIN_0_TRACE_AREA = 7600;  //for M
        const int MIN_1_TRACE_AREA = 30000; //for 0
        const int MIN_2_TRACE_AREA = 12500; //for '4'
        const int MIN_3_TRACE_AREA = 23000; //for ~
        const int CROPPED_IMG_MARGIN = 10;   //pixels
        const int MAX_TRACE_SPEED = 150;    //pixels/second (30p/0.2sec)

        // Training Start
        const string GESTURE_TRAINER_IMAGE = "spellChars_64wide_grayscale.png";	//YOUR SPELL/GESTURE TRAINING IMAGE (STORED IN THE PROJECT FOLDER)
        const int NO_OF_IMAGES_PER_ELEMENT = 20;
        // Training End


        private int _frameWidth;
        private int _frameHeight;

        Mat cameraFrame;
        Mat _fgMaskMOG2;
        Mat _wandMoveTracingFrame;

        SimpleBlobDetector.Params _params;

        SimpleBlobDetector _blobDetector;
        BackgroundSubtractor _pMOG2;

        KeyPoint[] _blobKeypoints;

        LinkedList<KeyPoint> _tracePoints;

        Point _traceUpperCorner;
        Point _traceLowerCorner;

        DateTime _lastKeypointTime;
        HOGDescriptor _hog;

        public bool IsInit
        {
            get;
            private set;
        }

        /// <summary>
        /// Initialize all parameters for wand detection and spell recognition.
        /// </summary>
        /// <param name="width">The width of the frame.</param>
        /// <param name="height">Th height of the frame.</param>
        public void Init(int width, int height)
        {
            _frameWidth = width;
            _frameHeight = height;

            _wandMoveTracingFrame = Mat.Zeros(_frameHeight, _frameWidth, (MatType)MatType.CV_8U);
            cameraFrame = new Mat(_frameHeight, _frameWidth, MatType.CV_8U);
            _fgMaskMOG2 = new Mat();

            _pMOG2 = BackgroundSubtractorMOG2.Create(BGS_HISTORY_FRAMES);

            _params = new SimpleBlobDetector.Params();
            _tracePoints = new LinkedList<KeyPoint>();

            // Change thresholds
            _params.MinThreshold = 150;
            _params.MaxThreshold = 255;

            // Fliter by color
            _params.FilterByColor = true;
            _params.BlobColor = 255;

            // Filter by Area.
            _params.FilterByArea = true;
            _params.MinArea = 0.02f;
            _params.MaxArea = 20;

            // Filter by Circularity
            _params.FilterByCircularity = true;
            _params.MinCircularity = 0.2f;

            // Filter by Convexity
            _params.FilterByConvexity = true;
            _params.MinConvexity = 0.5f;

            // Filter by Inertia
            _params.FilterByInertia = false;
            //params.minInertiaRatio = 0.01;

            _blobDetector = SimpleBlobDetector.Create(_params);

            _hog = new HOGDescriptor(
                new Size(64, 64), //winSize  50 x 50
                new Size(32, 32), //blocksize 32 x 32
                new Size(16, 16), //blockStride, 16 x 16
                new Size(16, 16), //cellSize, 16 x 16
                9, //nbins,
                1, //derivAper,
                -1, //winSigma,
                HistogramNormType.L2Hys, //histogramNormType,
                0.2, //L2HysThresh,
                false,//gammal correction,
                64//nlevels=64
                );
            _hog.SignedGradient = true;

            IsInit = true;
        }

        /// <summary>
        /// Check for spurious keypoints and draw a trace by joining the valid blob keypoints
        /// </summary>
        /// <param name="frameData"></param>
        /// <returns>Mat frame containing the trace</returns>
        public Mat GetWandTrace(byte[] frameData)
        {
            WandDetect(frameData, ref _blobKeypoints);

            //Add keypoints to deque. For now, take only the first found keypoint
            if (_blobKeypoints.Length != 0)
            {
                var currentKeypointTime = DateTime.Now;

                if (_tracePoints.Count != 0)
                {
                    var elapsed = currentKeypointTime - _lastKeypointTime;
                    Point pt1 = new Point(_tracePoints.ElementAt(_tracePoints.Count - 1).Pt.X, _tracePoints.ElementAt(_tracePoints.Count - 1).Pt.Y);
                    Point pt2 = new Point(_blobKeypoints[0].Pt.X, _blobKeypoints[0].Pt.Y);

                    if (Distance(pt1, pt2) / elapsed.Milliseconds >= MAX_TRACE_SPEED)
                    {
                        return _wandMoveTracingFrame;
                    }

                    if (_tracePoints.Count >= DEQUE_BUFFER_SIZE)
                        _tracePoints.RemoveFirst();

                    _tracePoints.AddLast(_blobKeypoints[0]);
                    //Point pt2(tracePoints[tracePoints.size() - 1].pt.x, tracePoints[tracePoints.size() - 1].pt.y);
                    Cv2.Line(_wandMoveTracingFrame, pt1, pt2, new Scalar(255), TRACE_THICKNESS);
                }
                else
                {
                    _lastKeypointTime = currentKeypointTime;
                    _tracePoints.AddLast(_blobKeypoints[0]);
                }
            }

            return _wandMoveTracingFrame;
        }

        /// <summary>
        /// Perform background elimination and blob detection.
        /// </summary>
        /// <param name="frameData"></param>
        /// <param name="keypoints">Detected blobs</param>
        private void WandDetect(byte[] frameData, ref KeyPoint[] keypoints /* Storage for blobs */ )
        {
            // Create a BGRA Mat
            Mat bgraMat = new Mat(_frameHeight, _frameWidth, MatType.CV_8UC4, frameData);

            // Convert the BGRA Mat to grayscale
            Cv2.CvtColor(bgraMat, cameraFrame, ColorConversionCodes.BGRA2GRAY);

            bgraMat.Dispose();

            if (cameraFrame.Empty())
                return;

            //Background Elimination
            _pMOG2.Apply(cameraFrame, _fgMaskMOG2);
            Mat bgSubtractedFrame = new Mat();
            cameraFrame.CopyTo(bgSubtractedFrame, _fgMaskMOG2);

            // Detect blobs
            var points = _blobDetector.Detect(bgSubtractedFrame);
            if (keypoints == null || keypoints.Length != points.Length)
            {
                keypoints = new KeyPoint[points.Length];
            }
            points.CopyTo(keypoints, 0);
            bgSubtractedFrame.Dispose();
        }

        /// <summary>
        /// Calculate the distance between 2 points.
        /// </summary>
        /// <param name="p">Point 1</param>
        /// <param name="q">Point 2</param>
        /// <returns>Returns the distance between 2 points.</returns>
        private double Distance(Point p, Point q)
        {
            Point diff = p - q;
            return Math.Sqrt(diff.X * diff.X + diff.Y * diff.Y);
        }

        /// <summary>
        /// Checks if the wand is visible.
        /// </summary>
        /// <returns>Returns true if the wand is visible or false otherwise.</returns>
        private bool WandVisible()
        {
            return _blobKeypoints.Length != 0;
        }

        /// <summary>
        /// Check if the trace qualifies for a possible spell.
        ///		Conditions for qualification:
        ///		a) It isn't being currently drawn.
        ///		   i.e., 5 seconds have passed since the last detected keypoint
        ///		b) It is made of at least 35 keypoints
        ///		c) Area covered by the trace is sufficiently large
        /// </summary>
        /// <returns>Returns true if it's a valid trace otherwise returns false.</returns>
        public bool CheckTraceValidity()
        {
            if (_blobKeypoints.Length == 0)
            {
                var currentKeypointTime = DateTime.Now;
                var elapsed = currentKeypointTime - _lastKeypointTime;

                if (elapsed.Seconds < 5.0)
                {
                    return false;
                }

                if (_tracePoints.Count > DEQUE_BUFFER_SIZE - 5)
                {
                    _traceUpperCorner = new Point(_frameWidth, _frameHeight);
                    _traceLowerCorner = new Point(0, 0);

                    //Draw a trace by connecting all the keypoints stored in the deque
                    //Also update lower and upper bounds of the trace
                    for (int i = 1; i < _tracePoints.Count; i++)
                    {
                        if (_tracePoints.ElementAt(i).Size == -99.0)
                            continue;
                        Point pt1 = new Point(_tracePoints.ElementAt(i - 1).Pt.X, _tracePoints.ElementAt(i - 1).Pt.Y);
                        Point pt2 = new Point(_tracePoints.ElementAt(i).Pt.X, _tracePoints.ElementAt(i).Pt.Y);

                        //Min x,y = traceUpperCorner points
                        //Max x,y = traceLowerCorner points
                        if (pt1.X < _traceUpperCorner.X)
                            _traceUpperCorner.X = pt1.X;
                        if (pt1.X > _traceLowerCorner.X)
                            _traceLowerCorner.X = pt1.X;
                        if (pt1.Y < _traceUpperCorner.Y)
                            _traceUpperCorner.Y = pt1.Y;
                        if (pt1.Y > _traceLowerCorner.Y)
                            _traceLowerCorner.Y = pt1.Y;
                    }

                    long traceArea = (_traceLowerCorner.X - _traceUpperCorner.X) * (_traceLowerCorner.Y - _traceUpperCorner.Y);
                    Debug.WriteLine("Trace area: " + traceArea);

                    if (traceArea > MIN_0_TRACE_AREA)
                        return true;
                }
                //It's been over five seconds since the last keypoint and trace isn't valid
                EraseTrace();
            }
            return false;
        }

        /// <summary>
        /// Erase information about last trace.
        /// </summary>
        public void EraseTrace()
        {
            //Erase existing trace
            for (int i = 0; i < _frameHeight; i++)
            {
                for (int j = 0; j < _frameWidth; j++)
                {
                    _wandMoveTracingFrame.At<byte>(i, j) = 0;
                }
            }
            
            _tracePoints.Clear();
        }

        /// <summary>
        /// Crop and resize the trace to 64x64 pixels.
        /// </summary>
        /// <returns>The resized frame.</returns>
        private Mat CropSaveTrace()
        {
            if (_traceUpperCorner.X > CROPPED_IMG_MARGIN)
                _traceUpperCorner.X -= CROPPED_IMG_MARGIN;
            else
                _traceUpperCorner.X = 0;

            if (_traceUpperCorner.Y > CROPPED_IMG_MARGIN)
                _traceUpperCorner.Y -= CROPPED_IMG_MARGIN;
            else
                _traceUpperCorner.Y = 0;

            if (_traceLowerCorner.X < _frameWidth - CROPPED_IMG_MARGIN)
                _traceLowerCorner.X += CROPPED_IMG_MARGIN;
            else
                _traceLowerCorner.X = _frameWidth;

            if (_traceLowerCorner.Y < _frameHeight - CROPPED_IMG_MARGIN)
                _traceLowerCorner.Y += CROPPED_IMG_MARGIN;
            else
                _traceLowerCorner.Y = _frameHeight;


            int traceWidth = _traceLowerCorner.X - _traceUpperCorner.X;
            int traceHeight = _traceLowerCorner.Y - _traceUpperCorner.Y;

            Mat resizedCroppedTrace = new Mat();
            Size _size;


            if (traceHeight > traceWidth)
            {
                _size.Height = TRAINER_IMAGE_WIN_SIZE;
                _size.Width = traceWidth * TRAINER_IMAGE_WIN_SIZE / traceHeight; //Since traceHeight & traceWidth are always gonna be > TRAINER_IMAGE_WIN_SIZE
            }
            else
            {
                _size.Width = TRAINER_IMAGE_WIN_SIZE;
                _size.Height = traceHeight * TRAINER_IMAGE_WIN_SIZE / traceWidth;
            }

            var tm = _wandMoveTracingFrame.SubMat(new Rect(_traceUpperCorner, new Size(_traceLowerCorner.X - _traceUpperCorner.X, _traceLowerCorner.Y - _traceUpperCorner.Y))).Clone();
            Cv2.Resize(tm, resizedCroppedTrace, _size);
            Mat finalTraceCell = Mat.Zeros(TRAINER_IMAGE_WIN_SIZE, TRAINER_IMAGE_WIN_SIZE, (MatType) MatType.CV_8U);
            for (int i = 0; i < resizedCroppedTrace.Rows; i++)
            {
                for (int j = 0; j < resizedCroppedTrace.Cols; j++)
                {
                    finalTraceCell.At<byte>(i, j) = resizedCroppedTrace.At<byte>(i, j);
                }
            }
            Debug.WriteLine("Done copying to 64 x 64");
            Debug.WriteLine("SAVE: " + Cv2.ImWrite("spellTraceCell.png", finalTraceCell));

            return finalTraceCell;
        }


        /// <summary>
        /// Perform handwriting recognition algorithm provided by OpenCV.
        ///	   to recognize the spells from traces.
        ///	   See here for detailed explanation:
        ///	   https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/
        /// </summary>
        /// <returns>
        /// Returns index of the recognized spell:
        ///     0: 'M'          - Arresto Momentum
        ///     1: 'Phi'        - Alohomora
        ///     2: '4'          - Locomotor
        ///     3: sideways 'S' - Mimblewimble
        /// </returns>
        /* rn: Index of the spell recognized:-								*/
        /*		0:	'M'														*/
        /*		1:	<blinds spell symbol>									*/
        /*		2:	'4'														*/
        /*		3:	<lights spell sumbol>									*/
        public int RecognizeSpell()
        {
            Mat deskewedTrace = Deskew(CropSaveTrace());

            float[] descriptors = _hog.Compute(deskewedTrace);

            Mat descriptorMatrix = new Mat(1, descriptors.Length, MatType.CV_32FC1);
            ConvertVectorToMatrix(descriptors, descriptorMatrix);

            SVM svm = SVM.Load("spellsModel.yml");

            float prediction = svm.Predict(descriptorMatrix);
            Debug.WriteLine(prediction);
            return (int)prediction;
        }

        /// <summary>
        /// Converts a float vector / array to OpenCV Matrix
        /// </summary>
        /// <param name="inHOG">The float vector to convert.</param>
        /// <param name="outMat">The matrix output.</param>
        private void ConvertVectorToMatrix(float[] inHOG, Mat outMat)
        {
            for (int i = 0; i < inHOG.Length; i++)
            {
                outMat.At<float>(0, i) = inHOG[i];
            }
        }

        private Mat Deskew(Mat img)
        {
            int SZ = 20;
            var affineFlags = InterpolationFlags.WarpInverseMap | InterpolationFlags.Linear;

            Moments m = new Moments(img);
            if (Math.Abs(m.Mu02) < 1e-2)
            {
                //No deskewing needed
                return img.Clone();
            }

            //Calculate skew based on central moments
            float skew = (float)m.Mu11 / (float)m.Mu02;

            //Calculate affine transform to correct skewness
            Mat warpMat = new Mat(2, 3, MatType.CV_32FC1, new float[] { 1, skew, -0.5f * SZ * skew, 0, 1, 0 });
            Mat imgOut = Mat.Zeros(img.Rows, img.Cols, img.Type());
            Cv2.WarpAffine(img, imgOut, warpMat, imgOut.Size(), affineFlags);

            return imgOut;

        }

        void CreateDeskewedTrain(List<Mat> deskewedTrainCells, List<Mat> trainCells)
        {
            for (int i = 0; i < trainCells.Count; i++)
            {
                Mat deskewedImg = Deskew(trainCells[i]);
                deskewedTrainCells.Add(deskewedImg);
            }
        }

        void CreateTrainHOG(List<float[]> trainHOG, List<Mat> deskewedtrainCells)
        {
            for (int y = 0; y < deskewedtrainCells.Count; y++)
            {
                var descriptors = _hog.Compute(deskewedtrainCells[y]);
                trainHOG.Add(descriptors);
            }
        }

        private void LoadTrainLabel(String pathName, List<Mat> trainCells, List<int> trainLabels)
        {

            Mat img = Cv2.ImRead(pathName, ImreadModes.Grayscale);

            int ImgCount = 0;
            for (int i = 0; i < img.Rows; i += TRAINER_IMAGE_WIN_SIZE)
            {
                for (int j = 0; j < img.Cols; j += TRAINER_IMAGE_WIN_SIZE)
                {
                    Mat digitImg = (img.ColRange(j, j + TRAINER_IMAGE_WIN_SIZE).RowRange(i, i + TRAINER_IMAGE_WIN_SIZE)).Clone();
                    trainCells.Add(digitImg);
                    ImgCount++;
                }
            }

            Debug.WriteLine("Image Count : %d", ImgCount);
            float digitClassNumber = 0;

            for (int z = 0; z < ImgCount; z++)
            {
                if (z % NO_OF_IMAGES_PER_ELEMENT == 0 && z != 0)
                {
                    digitClassNumber = digitClassNumber + 1;
                }
                trainLabels.Add((int)Math.Round(digitClassNumber));
            }
        }

        private void ConvertVectorToMatrix(List<float[]> inHOG, Mat outMat)
        {
            int descriptor_size = inHOG[0].Length;

            for (int i = 0; i < inHOG.Count; i++)
            {
                for (int j = 0; j < descriptor_size; j++)
                {
                    outMat.At<float>(i, j) = inHOG[i][j];
                }
            }
        }


        private void SpellRecognitionTrainer()
        {
            List<Mat> trainCells = new List<Mat>();
            List<int> trainLabels = new List<int>();
            LoadTrainLabel(GESTURE_TRAINER_IMAGE, trainCells, trainLabels);

            List<Mat> deskewedTrainCells = new List<Mat>();
            CreateDeskewedTrain(deskewedTrainCells, trainCells);

            List<float[]> trainHOG = new List<float[]>();
            CreateTrainHOG(trainHOG, deskewedTrainCells);

            int descriptor_size = trainHOG[0].Count();
            Debug.WriteLine("Descriptor Size: %d", descriptor_size);

            Mat trainMat = new Mat(trainHOG.Count(), descriptor_size, MatType.CV_32FC1);

            ConvertVectorToMatrix(trainHOG, trainMat);

            SVMtrain(trainMat, trainLabels);
        }

        /// <summary>
        /// Debug information about SVM.
        /// </summary>
        /// <param name="svm">SVM to debug.</param>
        private void GetSVMParams(SVM svm)
        {
            Debug.WriteLine("Kernel type     : %s", svm.KernelType);
            Debug.WriteLine("Type            : %s", svm.Type);
            Debug.WriteLine("C               : %s", svm.C);
            Debug.WriteLine("Degree          : %s", svm.Degree);
            Debug.WriteLine("Nu              : %s", svm.Nu);
            Debug.WriteLine("Gamma           : %s", svm.Gamma);
        }

        /// <summary>
        /// Trains the SVM and saves trained model.
        /// </summary>
        /// <param name="trainMat">The training data.</param>
        /// <param name="trainLabels">The training labels.</param>
        private void SVMtrain(Mat trainMat, List<int> trainLabels)
        {
            SVM svm = SVM.Create();
            svm.Gamma = 0.50625;
            svm.C = 12.5;
            svm.KernelType = SVM.KernelTypes.Rbf;
            svm.Type = SVM.Types.CSvc;
            
            //TODO: TYPE MISSING FROM OPENCVSHARP (DOWNLOADED SOURCE AND WORKING ON SUPPORT FOR THIS)
            //TrainData td = TrainData.Create(trainMat, ROW_SAMPLE, trainLabels);
            //svm.Train(td);


            //	svm.trainAuto(td);
            svm.Save("model4.yml");
            GetSVMParams(svm);
        }
    }
}