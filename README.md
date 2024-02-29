# HPSpellDetector

This project detects spells drawn with a wand from Universal Florida Harry Potter world (or any other IR reflector.)
It is a port / adaptation of the following [project](https://github.com/sanni-t/techMagicApp) in C# making use of OpenCVSharp to run on the Xbox One using Kinect. Currently only 4 spells are detected: Arresto Momentum, Alohomora, Locomotor and Mimblewimble. Support for more spells coming in the future.

### Requirements
- Visual Studio 2022 Community Edition
- Kinect for Xbox One (or infrared source)

### Note
This project was only tested on Xbox One + Kinect.

### Next Steps
Support to train the SVM to allow adding more spells which requires adding support for some non-supported types on OpenCVSharp.