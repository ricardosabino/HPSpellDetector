using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;
using Windows.UI.Xaml.Media.Imaging;
using System.Threading.Tasks;
using Windows.UI.Popups;
using Windows.UI.Core;
using Windows.Media.Capture;
using Windows.Media.Capture.Frames;
using Windows.Media.MediaProperties;
using Windows.Graphics.Imaging;
using Windows.Graphics.DirectX.Direct3D11;
using OpenCvSharp;
using static System.Net.Mime.MediaTypeNames;
using System.Threading;
using System.Diagnostics;
using Windows.UI.Xaml.Shapes;

namespace HP
{
    /// <summary>
    /// HP spell detector Main Page.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        private int _preferredIndex;
        private MediaCapture _mc;
        private MediaFrameReader _reader;

        private SoftwareBitmap _latestFrame;
        private IDirect3DSurface _latestSurface;

        private ImageProcessor _imgProc = new ImageProcessor();
        private SemaphoreSlim _semaphore = new SemaphoreSlim(1, 1);

        private List<Rectangle> _spellPics;

        public MainPage()
        {
            this.InitializeComponent();

            _spellPics = new List<Rectangle>() { spell0Rect, spell1Rect, spell2Rect, spell3Rect };

            _ = this.Start();
        }

        private async Task Start()
        {
            await ResetMediaCapture();
        }

        private void ReleaseMediaCapture()
        {
            if (_reader != null)
            {
                _reader.FrameArrived -= ReaderFrameArrived;
                _reader.Dispose();
                _reader = null;
            }

            if (_mc != null)
            {
                _mc.Dispose();
                _mc = null;
            }
        }

        private async Task ResetMediaCapture()
        {
            this.ReleaseMediaCapture();

            var list = await MediaFrameSourceGroup.FindAllAsync();

            var selectors = list.Select(group => new
            {
                sourceGroup = group,
                sourceInfo = group.SourceInfos.FirstOrDefault(SelectSource)
            }).Where(t => t.sourceInfo != null).ToList();

            if (!selectors.Any())
            {
                var array = list.ToArray();
                var message = $"No valid camera source.\nFound {array.Length} sensor groups";
                foreach (var item in array)
                {
                    message += $"\n   {item.DisplayName}";
                }
                await new MessageDialog(message).ShowAsync();
                return;
            }

            var selector = selectors[this._preferredIndex % selectors.Count];

            var settings = new MediaCaptureInitializationSettings
            {
                SourceGroup = selector.sourceGroup,
                SharingMode = MediaCaptureSharingMode.SharedReadOnly,
                MemoryPreference = MediaCaptureMemoryPreference.Cpu
            };

            _mc = new MediaCapture();
            await _mc.InitializeAsync(settings);

            var source = _mc.FrameSources[selector.sourceInfo.Id];

            //_reader = await _mc.CreateFrameReaderAsync(source, MediaEncodingSubtypes.Rgb32);
            _reader = await _mc.CreateFrameReaderAsync(source);

            _reader.FrameArrived += ReaderFrameArrived;
            var status = await _reader.StartAsync();

            if (status != MediaFrameReaderStartStatus.Success)
            {
                var message = $"StartAsync Error: {status}";
                await new MessageDialog(message).ShowAsync();
            }
        }

        /// <summary>
        /// Media frame ready event handler.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="args"></param>
        private async void ReaderFrameArrived(MediaFrameReader sender, MediaFrameArrivedEventArgs args)
        {
            await _semaphore.WaitAsync();
            try
            {

                using (var frameRef = sender.TryAcquireLatestFrame())
                {
                        if (_mc == null)
                            return;

                        _latestFrame?.Dispose();
                        _latestFrame = null;
                        _latestSurface?.Dispose();
                        _latestSurface = null;


                        if (frameRef?.SourceKind == MediaFrameSourceKind.Infrared)
                        {
                            _latestFrame = SoftwareBitmap.Convert(frameRef?.VideoMediaFrame?.SoftwareBitmap, BitmapPixelFormat.Bgra8);
                        }
                        else
                        {
                            _latestFrame = frameRef?.VideoMediaFrame?.SoftwareBitmap;
                        }

                        _latestSurface = _latestFrame != null ? null : frameRef?.VideoMediaFrame?.Direct3DSurface;
                }

                await ScheduleRenderFrame();
            }
            finally
            {
                _semaphore.Release();
            }

        }

        /*/
        private readonly object _bitmapLock = new object();
        private async Task ScheduleRenderFrame()
        {
            await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () =>
            {
            lock (_bitmapLock)
            {

                if (_latestFrame != null)
                {
                    // Render the software bitmap to the image control
                    var bitmap = new WriteableBitmap(_latestFrame.PixelWidth, _latestFrame.PixelHeight);
                    _latestFrame.CopyToBuffer(bitmap.PixelBuffer);
                    imageControl.Source = bitmap;

                    var bitmapBuffer = bitmap.PixelBuffer;
                    var dataReader = Windows.Storage.Streams.DataReader.FromBuffer(bitmapBuffer);
                            // Create buffer
                            byte[] trace = new byte[bitmapBuffer.Length];
                            dataReader.ReadBytes(trace);
                        dataReader.Dispose();
                            // Now 'pixels' contains the byte array of the image

                            var numPixels = _latestFrame.PixelWidth * _latestFrame.PixelHeight;
                            //var trace = new byte[numPixels];
                            //var l = bitmap.PixelBuffer.ToArray();
                            //Buffer.BlockCopy(l, 0, trace, 0, numPixels);

                            if (!_imgProc.IsInit)
                            {
                                _imgProc.Init(_latestFrame.PixelWidth, _latestFrame.PixelHeight);
                            }
                            var mat = _imgProc.GetWandTrace(trace, numPixels);
                            //_imgProc.GetWandTrace(trace, numPixels);

                        //Cv2.NamedWindow("main", WindowFlags.OpenGL | WindowFlags.FreeRatio);
                        //Cv2.ImShow("main", mat);
                        //byte[] imageArray = new byte[mat.Cols * mat.Rows * mat.Channels()];
                        byte[] imageArray;
                        mat.GetArray(out imageArray);
                        
                        SoftwareBitmap softwareBitmap = SoftwareBitmap.CreateCopyFromBuffer(
                            imageArray.AsBuffer(),
                            BitmapPixelFormat.Gray8,
                            _latestFrame.PixelWidth,
                            _latestFrame.PixelHeight,
                            BitmapAlphaMode.Ignore
                        );

                        var source = new SoftwareBitmapSource();
                        if (softwareBitmap.BitmapPixelFormat != BitmapPixelFormat.Bgra8 || 
                            softwareBitmap.BitmapAlphaMode == BitmapAlphaMode.Straight)
                        {
                            softwareBitmap = SoftwareBitmap.Convert(softwareBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Ignore);
                        }
                        source.SetBitmapAsync(softwareBitmap);
                        processedImageControl.Source = source;
                        //softwareBitmap.Dispose();
                        //var source = new SoftwareBitmapSource();
                        //await source.SetBitmapAsync(softwareBitmap);
                        //imageControl.Source = source;

                        // var source = new SoftwareBitmapSource();
                        // await source.SetBitmapAsync(outputBitmap);

                    }
                    else if (_latestSurface != null)
                    {
                        // Render the direct3d surface to the swap chain panel
                        //swapChainPanelControl.SwapChainPanel = _latestSurface;
                    }
                }
            });
        }
        /*/

        private async Task ScheduleRenderFrame()
        {
            await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, async () =>
            {
                await _semaphore.WaitAsync();
                try
                {

                    if (_latestFrame != null)
                    {
                        // Render the software bitmap to the image control
                        var bitmap = new WriteableBitmap(_latestFrame.PixelWidth, _latestFrame.PixelHeight);
                        _latestFrame.CopyToBuffer(bitmap.PixelBuffer);
                        imageControl.Source = bitmap;

                        var bitmapBuffer = bitmap.PixelBuffer;
                        var dataReader = Windows.Storage.Streams.DataReader.FromBuffer(bitmapBuffer);
                        // Create buffer
                        byte[] pixels = new byte[bitmapBuffer.Length];
                        dataReader.ReadBytes(pixels);
                        dataReader.Dispose();
                        // Now 'pixels' contains the byte array of the image

                        if (!_imgProc.IsInit)
                        {
                            _imgProc.Init(_latestFrame.PixelWidth, _latestFrame.PixelHeight);
                        }
                        var mat = _imgProc.GetWandTrace(pixels);


                        // ----------------------------
                        if (_imgProc.CheckTraceValidity())
                        {
                            Debug.WriteLine("Trace valid for spell rcognition");

                            switch (_imgProc.RecognizeSpell())
                            {
                                case 0:
                                    Debug.WriteLine("*** 0: Arresto Momentum ***");
                                    tbDetected.Text = "Arresto Momentum";
                                    ShowSpellPic(0);
                                    break;
                                case 1:
                                    Debug.WriteLine("*** 1 Alohomora ***");
                                    tbDetected.Text = "Alohomora";
                                    ShowSpellPic(1);
                                    break;
                                case 2:
                                    Debug.WriteLine("*** 2: Locomotor ***");
                                    tbDetected.Text = "Locomotor";
                                    ShowSpellPic(2);
                                    break;
                                case 3:
                                    Debug.WriteLine("*** 3: Mimblewimble ***");
                                    tbDetected.Text = "Mimblewimble";
                                    ShowSpellPic(3);
                                    break;
                                default:
                                    Debug.WriteLine("That's not a spell");
                                    tbDetected.Text = "Unknown";
                                    ShowSpellPic(-1);
                                    break;
                            }
                            _imgProc.EraseTrace();
                        }
                        // ----------------------------

                        mat.GetArray(out byte[] imageArray);

                        SoftwareBitmap softwareBitmap = SoftwareBitmap.CreateCopyFromBuffer(
                            imageArray.AsBuffer(),
                            BitmapPixelFormat.Gray8,
                            _latestFrame.PixelWidth,
                            _latestFrame.PixelHeight,
                            BitmapAlphaMode.Ignore
                        );

                        var source = new SoftwareBitmapSource();
                        if (softwareBitmap.BitmapPixelFormat != BitmapPixelFormat.Bgra8 ||
                            softwareBitmap.BitmapAlphaMode == BitmapAlphaMode.Straight)
                        {
                            softwareBitmap = SoftwareBitmap.Convert(softwareBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Ignore);
                        }
                        await source.SetBitmapAsync(softwareBitmap);
                        processedImageControl.Source = source;
                    }
                    else if (_latestSurface != null)
                    {
                        // TODO: FIX
                        // Render the direct3d surface to the swap chain panel
                        //swapChainPanelControl.SwapChainPanel = _latestSurface;
                    }
                }
                finally
                {
                    _semaphore.Release();
                }
            });
        }

        /// <summary>
        /// Highlight the spell that was detected.
        /// </summary>
        /// <param name="index">Index of the spell to highlight</param>
        private void ShowSpellPic(int index)
        {
            for(int i = 0; i < _spellPics.Count; i++)
            {
                _spellPics[i].Opacity = 0.7;
            }

            if(index >= 0 && index < _spellPics.Count)
            {
                _spellPics[index].Opacity = 0;
            }
        }
        /**/

        /// <summary>
        /// Select the media source for image capture.
        /// </summary>
        /// <param name="sourceInfo"></param>
        /// <returns>Returns true if there's an infrared source</returns>
        private bool SelectSource(MediaFrameSourceInfo sourceInfo)
        {
            //TODO: CHECK IF WE WANT TO HANDLE THE CASE WHEN THERE'S NO INFRARED
            //return sourceInfo.SourceKind == MediaFrameSourceKind.Color || sourceInfo.SourceKind == MediaFrameSourceKind.Infrared;
            return sourceInfo.SourceKind == MediaFrameSourceKind.Infrared;
        }

    }
}
