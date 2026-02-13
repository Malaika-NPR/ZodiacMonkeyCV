ZodiacMonkeyCV (Still in progress) 🐒

A real-time face and gesture recognition project built with OpenCV and MediaPipe that maps facial expressions and hand poses to animated monkey reaction images.

I built this as a playful computer vision project inspired by my Chinese zodiac animal, the Monkey. It detects expressions from a webcam feed and dynamically displays corresponding monkey reaction faces.


What It Does

	- Uses MediaPipe FaceMesh for facial landmark detection

	- Uses MediaPipe Hands and Pose for gesture recognition


Detects expressions:

	- Excited

	- Thinking

	- Smirk

	- Surprised

	- Proud

	- Peace sign

	- Sincere

Smooths predictions across frames to reduce flicker

		- Displays live webcam feed alongside a matching monkey image

		- The system combines facial geometry, eye openness, mouth distance, and finger counting to classify expressions.


Running the Project

 Install dependencies:

	pip install opencv-python mediapipe numpy

  Run:

	 python main.py
